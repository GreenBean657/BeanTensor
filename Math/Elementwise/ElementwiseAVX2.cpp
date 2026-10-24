/**
 * @file ElementwiseAVX2.cpp
 * @author GreenBean
 * @brief AVX2+F16C kernels for binary elementwise arithmetic.
 * Compiled with -mavx2 -mf16c; execution is gated at runtime via
 * elementwise_avx2_supported(). Every kernel throws
 * ErrorHandling::CPUIllegalInstruction when the CPU lacks the feature, so a
 * direct call is a valid feature probe (tests rely on this).
 * Bit-exactness contract: kernel output equals the scalar applies in
 * ElementwiseCPU.cpp (apply_native/apply_half/apply_packed/apply_int) on every
 * input, including NaN/Inf/subnormal sentinels. fp16 narrows through the F16C
 * hardware converter with a vectorized replay of the scalar overflow clamp
 * (finite |x| > 65504 -> +/-65504, which hardware RNE would turn into Inf);
 * bf16 narrows through the integer RNE-bias trick with the scalar's NaN
 * quieting and fp32-subnormal flush-to-zero. Phase 2 adds the tensor-scalar
 * splat kernels (scalar pre-narrowed by the caller, broadcast into a vector
 * register, same tail semantics) and the per-nibble Int4x2 wrap kernels
 * (unsigned-nibble byte-lane arithmetic masked to 4 bits reproduces the
 * scalar signed wrap bit-for-bit).
 * @version 0.3
 * @date 2026-10-23
 */
#include <immintrin.h>

#include <type_traits>

#include "Math/Elementwise/Elementwise.h"
#include "Math/Elementwise/ElementwiseKernels.h"
#include "Helpers/Hardware.h"

#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Math::Elementwise {
    namespace {
        using BeanTensor::Tensors::ElementwiseOp;

        // --------------------------- vector op selectors ---------------------------

        /**
         * Applies the fp_binop NaN contract to a computed lane vector: where x is NaN
         * the quieted x wins, otherwise where y is NaN the quieted y wins. Explicit
         * selects, so the choice never depends on how the compiler orders the
         * instruction's operands. Non-NaN lanes keep r.
         * @brief fp32 NaN-contract select.
         */
        [[nodiscard]] __m256 f32_nan_select(const __m256 x, const __m256 y, const __m256 r) {
            const auto quiet = _mm256_castsi256_ps(_mm256_set1_epi32(0x00400000));
            const auto x_nan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
            const auto y_nan = _mm256_cmp_ps(y, y, _CMP_UNORD_Q);
            const auto pick = _mm256_blendv_ps(_mm256_or_ps(y, quiet), _mm256_or_ps(x, quiet), x_nan);
            return _mm256_blendv_ps(r, pick, _mm256_or_ps(x_nan, y_nan));
        }

        /**
         * fp64 twin of f32_nan_select (quiet bit 51).
         * @brief fp64 NaN-contract select.
         */
        [[nodiscard]] __m256d f64_nan_select(const __m256d x, const __m256d y, const __m256d r) {
            const auto quiet = _mm256_castsi256_pd(_mm256_set1_epi64x(static_cast<int64_t>(uint64_t{1} << 51)));
            const auto x_nan = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);
            const auto y_nan = _mm256_cmp_pd(y, y, _CMP_UNORD_Q);
            const auto pick = _mm256_blendv_pd(_mm256_or_pd(y, quiet), _mm256_or_pd(x, quiet), x_nan);
            return _mm256_blendv_pd(r, pick, _mm256_or_pd(x_nan, y_nan));
        }

        /**
         * Compile-time map from the elementwise op selector to the fp32 vector instruction,
         * under the fp_binop NaN contract.
         * @brief fp32 vector op.
         */
        template<ElementwiseOp Op>
        [[nodiscard]] __m256 f32_vec_op(const __m256 x, const __m256 y) {
            if constexpr (Op == ElementwiseOp::Add) {
                return f32_nan_select(x, y, _mm256_add_ps(x, y));
            } else if constexpr (Op == ElementwiseOp::Sub) {
                return f32_nan_select(x, y, _mm256_sub_ps(x, y));
            } else if constexpr (Op == ElementwiseOp::Mul) {
                return f32_nan_select(x, y, _mm256_mul_ps(x, y));
            } else {
                return f32_nan_select(x, y, _mm256_div_ps(x, y));
            }
        }

        /**
         * Compile-time map from the elementwise op selector to the fp64 vector instruction,
         * under the fp_binop NaN contract.
         * @brief fp64 vector op.
         */
        template<ElementwiseOp Op>
        [[nodiscard]] __m256d f64_vec_op(const __m256d x, const __m256d y) {
            if constexpr (Op == ElementwiseOp::Add) {
                return f64_nan_select(x, y, _mm256_add_pd(x, y));
            } else if constexpr (Op == ElementwiseOp::Sub) {
                return f64_nan_select(x, y, _mm256_sub_pd(x, y));
            } else if constexpr (Op == ElementwiseOp::Mul) {
                return f64_nan_select(x, y, _mm256_mul_pd(x, y));
            } else {
                return f64_nan_select(x, y, _mm256_div_pd(x, y));
            }
        }

        // --------------------------- fp16 narrow helpers ---------------------------

        /**
         * The scalar narrow (unaccel_fp32_to_fp16_untracked) clamps any finite |x| >
         * FP16_MAX to +/-65504, while hardware RNE conversion of 65520 would tie-to-even
         * into Inf. Lanes with 65504 < |x| < Inf are replaced with the sign-preserving
         * 65504 bit pattern BEFORE the hardware convert; NaN and Inf lanes pass through
         * untouched (the scalar handles them before its clamp branch, and the hardware
         * converter matches the scalar's NaN quieting/Inf output exactly).
         * @brief Vectorized replay of the scalar fp16 overflow clamp.
         */
        [[nodiscard]] __m256 f16_clamp_finite_overflow(const __m256 r) {
            const auto abs = _mm256_and_ps(r, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
            const auto too_big = _mm256_cmp_ps(abs, _mm256_set1_ps(DataTypes::detail::FP16_MAX), _CMP_GT_OQ);
            const auto finite = _mm256_cmp_ps(abs, _mm256_set1_ps(__builtin_inff()), _CMP_LT_OQ);
            const auto clamp_lanes = _mm256_and_ps(too_big, finite);
            // 0x477FE000 == 65504.0f, OR'd with the source sign bit.
            const auto sign = _mm256_and_si256(_mm256_castps_si256(r), _mm256_set1_epi32(static_cast<int32_t>(0x80000000u)));
            const auto clamped = _mm256_castsi256_ps(_mm256_or_si256(sign, _mm256_set1_epi32(0x477FE000)));
            return _mm256_blendv_ps(r, clamped, clamp_lanes);
        }

        /**
         * Hardware VCVTPS2PH with RNE + no exceptions, matching the scalar narrow
         * everywhere except the finite-overflow clamp applied beforehand.
         * @brief Narrow 8 fp32 lanes to 8 fp16 lanes (bit pattern vector).
         */
        [[nodiscard]] __m128i f16_narrow8(const __m256 w) {
            return _mm256_cvtps_ph(f16_clamp_finite_overflow(w), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        }

        /**
         * Scalar widen/apply/narrow tail lane — byte-for-byte the scalar apply_half.
         * @brief Scalar fp16 lane op (one 16-bit lane by bit pattern).
         */
        template<ElementwiseOp Op>
        [[nodiscard]] uint16_t f16_tail_op(const uint16_t x, const uint16_t y) {
            const auto wx = Intrinsics::detail::unaccel_fp16_to_fp32_untracked(
                Intrinsics::detail::bit_cast_memcpy<float16_t>(x));
            const auto wy = Intrinsics::detail::unaccel_fp16_to_fp32_untracked(
                Intrinsics::detail::bit_cast_memcpy<float16_t>(y));
            const auto w = Math::Elementwise::fp_binop<float32_t, Op>(wx, wy);
            return Intrinsics::detail::bit_cast_memcpy<uint16_t>(
                Intrinsics::detail::unaccel_fp32_to_fp16_untracked(w));
        }

        // --------------------------- bf16 narrow helpers ---------------------------

        /**
         * Integer emulation of the scalar narrow (unaccel_fp32_to_bf16_untracked):
         * NaN/Inf truncate with the NaN quiet bit set, fp32-subnormal/zero flushes
         * to sign-preserving zero, normals round-to-nearest-even via the bias trick.
         * @brief Narrow 8 fp32 lanes to 8 bf16 lane results (low halves of 8 dwords).
         */
        [[nodiscard]] __m256i bf16_narrow8_lo(const __m256 w) {
            const auto bits = _mm256_castps_si256(w);
            const auto exp = _mm256_and_si256(bits, _mm256_set1_epi32(0x7F800000));
            const auto mant = _mm256_and_si256(bits, _mm256_set1_epi32(0x007FFFFF));
            const auto is_special = _mm256_cmpeq_epi32(exp, _mm256_set1_epi32(0x7F800000));
            const auto is_nan = _mm256_and_si256(is_special, _mm256_cmpgt_epi32(mant, _mm256_setzero_si256()));
            const auto is_sub = _mm256_cmpeq_epi32(exp, _mm256_setzero_si256());
            const auto special = _mm256_or_si256(_mm256_srli_epi32(bits, 16),
                                                 _mm256_and_si256(is_nan, _mm256_set1_epi32(0x0040)));
            const auto sub = _mm256_srli_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(static_cast<int32_t>(0x80000000u))), 16);
            const auto lsb = _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1));
            const auto rounded = _mm256_srli_epi32(
                _mm256_add_epi32(_mm256_add_epi32(bits, _mm256_set1_epi32(0x7FFF)), lsb), 16);
            return _mm256_blendv_epi8(_mm256_blendv_epi8(rounded, sub, is_sub), special, is_special);
        }

        /**
         * Results are <= 0xFFFF (in the low halves of the dwords), so the signed
         * saturation of packus never triggers; the 128-bit pack keeps lane order.
         * @brief Compact 8 bf16 lane results (8 dwords) into 8 packed 16-bit lanes.
         */
        [[nodiscard]] __m128i bf16_pack8(const __m256i r) {
            return _mm_packus_epi32(_mm256_castsi256_si128(r), _mm256_extracti128_si256(r, 1));
        }

        /**
         * Widen 8 bf16 lanes to fp32 (zero-extend + 16-bit left shift — the exact
         * bit pattern of unaccel_bf16_to_fp32).
         * @brief Widen 8 bf16 lanes to 8 fp32 lanes.
         */
        [[nodiscard]] __m256 bf16_widen8(const uint16_t* const p) {
            const auto lanes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
            return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(lanes), 16));
        }

        /**
         * Scalar widen/apply/narrow tail lane — byte-for-byte the scalar apply_half.
         * @brief Scalar bf16 lane op (one 16-bit lane by bit pattern).
         */
        template<ElementwiseOp Op>
        [[nodiscard]] uint16_t bf16_tail_op(const uint16_t x, const uint16_t y) {
            const auto wx = Intrinsics::detail::unaccel_bf16_to_fp32(
                Intrinsics::detail::bit_cast_memcpy<bfloat16_t>(x));
            const auto wy = Intrinsics::detail::unaccel_bf16_to_fp32(
                Intrinsics::detail::bit_cast_memcpy<bfloat16_t>(y));
            const auto w = Math::Elementwise::fp_binop<float32_t, Op>(wx, wy);
            return Intrinsics::detail::bit_cast_memcpy<uint16_t>(
                Intrinsics::detail::unaccel_fp32_to_bf16_untracked(w));
        }

        // --------------------------- integer helpers ---------------------------

        /**
         * Compile-time map from the elementwise op selector to the epi8 vector instruction.
         * @brief 8-bit lane vector add/sub.
         */
        template<ElementwiseOp Op>
        [[nodiscard]] __m256i i8_vec_addsub(const __m256i x, const __m256i y) {
            if constexpr (Op == ElementwiseOp::Add) {
                return _mm256_add_epi8(x, y);
            } else {
                return _mm256_sub_epi8(x, y);
            }
        }

        /**
         * No epi8 multiply exists: widen both operands to 16-bit lanes, take the low
         * product byte. Zero-extension makes the low byte sign-agnostic, so one kernel
         * serves int8 and uint8 (wrap semantics).
         * @brief 8-bit lane vector mul (via 16-bit widening).
         */
        [[nodiscard]] __m256i i8_vec_mul(const __m256i x, const __m256i y) {
            const auto xw = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(x));
            const auto yw = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(y));
            const auto xw_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(x, 1));
            const auto yw_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(y, 1));
            const auto lo = _mm256_and_si256(_mm256_mullo_epi16(xw, yw), _mm256_set1_epi16(0x00FF));
            const auto hi = _mm256_and_si256(_mm256_mullo_epi16(xw_hi, yw_hi), _mm256_set1_epi16(0x00FF));
            // After the AND, values are <= 0xFF: packus saturation never triggers.
            return _mm256_permute4x64_epi64(_mm256_packus_epi16(lo, hi), 0xD8);
        }

        /**
         * Compile-time map from the elementwise op selector to the epi16 vector instruction.
         * @brief 16-bit lane vector add/sub/mul.
         */
        template<ElementwiseOp Op>
        [[nodiscard]] __m256i i16_vec_op(const __m256i x, const __m256i y) {
            if constexpr (Op == ElementwiseOp::Add) {
                return _mm256_add_epi16(x, y);
            } else if constexpr (Op == ElementwiseOp::Sub) {
                return _mm256_sub_epi16(x, y);
            } else {
                return _mm256_mullo_epi16(x, y);
            }
        }

        /**
         * Compile-time map from the elementwise op selector to the epi32 vector instruction.
         * @brief 32-bit lane vector add/sub/mul.
         */
        template<ElementwiseOp Op>
        [[nodiscard]] __m256i i32_vec_op(const __m256i x, const __m256i y) {
            if constexpr (Op == ElementwiseOp::Add) {
                return _mm256_add_epi32(x, y);
            } else if constexpr (Op == ElementwiseOp::Sub) {
                return _mm256_sub_epi32(x, y);
            } else {
                return _mm256_mullo_epi32(x, y);
            }
        }

        /**
         * Compile-time map from the elementwise op selector to the epi64 vector instruction.
         * @brief 64-bit lane vector add/sub.
         */
        template<ElementwiseOp Op>
        [[nodiscard]] __m256i i64_vec_addsub(const __m256i x, const __m256i y) {
            if constexpr (Op == ElementwiseOp::Add) {
                return _mm256_add_epi64(x, y);
            } else {
                return _mm256_sub_epi64(x, y);
            }
        }

        /**
         * AVX2 has no mullo_epi64: compose the low 64-bit product from 32x32->64
         * partial products (vpmuludq on the even dwords and shifted odd dwords).
         * Only the low 64 bits are kept, which is sign-agnostic — one kernel serves
         * int64 and uint64 (wrap semantics).
         * @brief 64-bit lane vector mul (low qword).
         */
        [[nodiscard]] __m256i i64_vec_mul(const __m256i x, const __m256i y) {
            const auto p0 = _mm256_mul_epu32(x, y);
            const auto p1 = _mm256_mul_epu32(_mm256_srli_epi64(x, 32), y);
            const auto p2 = _mm256_mul_epu32(x, _mm256_srli_epi64(y, 32));
            // (p1 + p2) may carry past bit 64; the shift discards everything but the
            // low 32 bits, which is exactly what the low qword of the product needs.
            return _mm256_add_epi64(p0, _mm256_slli_epi64(_mm256_add_epi64(p1, p2), 32));
        }

        /**
         * Scalar wrap tail — byte-for-byte the unsigned-twin arithmetic of apply_int.
         * @brief Scalar integer lane op (add/sub/mul, wrap semantics).
         */
        template<typename T, ElementwiseOp Op>
        [[nodiscard]] T int_tail_op(const T a, const T b) {
            using U = std::make_unsigned_t<T>;
            if constexpr (Op == ElementwiseOp::Add) {
                return static_cast<T>(static_cast<U>(static_cast<U>(a) + static_cast<U>(b)));
            } else if constexpr (Op == ElementwiseOp::Sub) {
                return static_cast<T>(static_cast<U>(static_cast<U>(a) - static_cast<U>(b)));
            } else {
                return static_cast<T>(static_cast<U>(static_cast<U>(a) * static_cast<U>(b)));
            }
        }

        // --------------------------- range kernel bodies ---------------------------

        /**
         * 8 lanes per step; the tail runs scalar. IEEE single-rounding op,
         * bit-identical to the scalar apply_native.
         * @brief AVX2 range kernel body (fp32).
         */
        template<ElementwiseOp Op>
        void f32_range_avx2(float32_t* out, const float32_t* a, const float32_t* b, const size_t begin, const size_t end) {
            auto i = begin;
            for (; i + 8 <= end; i += 8) {
                _mm256_storeu_ps(out + i, f32_vec_op<Op>(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
            }
            for (; i < end; ++i) {
                out[i] = Math::Elementwise::fp_binop<float32_t, Op>(a[i], b[i]);
            }
        }

        /**
         * 4 lanes per step; the tail runs scalar. IEEE single-rounding op,
         * bit-identical to the scalar apply_native.
         * @brief AVX2 range kernel body (fp64).
         */
        template<ElementwiseOp Op>
        void f64_range_avx2(float64_t* out, const float64_t* a, const float64_t* b, const size_t begin, const size_t end) {
            auto i = begin;
            for (; i + 4 <= end; i += 4) {
                _mm256_storeu_pd(out + i, f64_vec_op<Op>(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i)));
            }
            for (; i < end; ++i) {
                out[i] = Math::Elementwise::fp_binop<float64_t, Op>(a[i], b[i]);
            }
        }

        /**
         * 8 lanes per step; the tail runs scalar (widen/apply/narrow, the exact
         * scalar apply_half pathway).
         * @brief AVX2 range kernel body (fp16 flat lanes).
         */
        template<ElementwiseOp Op>
        void f16_range_avx2(uint16_t* out, const uint16_t* a, const uint16_t* b, const size_t begin, const size_t end) {
            auto i = begin;
            for (; i + 8 <= end; i += 8) {
                const auto wa = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i)));
                const auto wb = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i)));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(out + i), f16_narrow8(f32_vec_op<Op>(wa, wb)));
            }
            for (; i < end; ++i) {
                out[i] = f16_tail_op<Op>(a[i], b[i]);
            }
        }

        /**
         * 8 lanes per step; the tail runs scalar (widen/apply/narrow, the exact
         * scalar apply_half pathway).
         * @brief AVX2 range kernel body (bf16 flat lanes).
         */
        template<ElementwiseOp Op>
        void bf16_range_avx2(uint16_t* out, const uint16_t* a, const uint16_t* b, const size_t begin, const size_t end) {
            auto i = begin;
            for (; i + 8 <= end; i += 8) {
                const auto wa = bf16_widen8(a + i);
                const auto wb = bf16_widen8(b + i);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(out + i), bf16_pack8(bf16_narrow8_lo(f32_vec_op<Op>(wa, wb))));
            }
            for (; i < end; ++i) {
                out[i] = bf16_tail_op<Op>(a[i], b[i]);
            }
        }

        /**
         * 32 lanes per step; the tail runs scalar (wrap arithmetic, the exact
         * unsigned-twin arithmetic of apply_int).
         * @brief AVX2 range kernel body (8-bit lanes).
         */
        template<ElementwiseOp Op>
        void i8_range_avx2(int8_t* out, const int8_t* a, const int8_t* b, const size_t begin, const size_t end) {
            auto i = begin;
            for (; i + 32 <= end; i += 32) {
                const auto x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
                const auto y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
                __m256i r;
                if constexpr (Op == ElementwiseOp::Mul) {
                    r = i8_vec_mul(x, y);
                } else {
                    r = i8_vec_addsub<Op>(x, y);
                }
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), r);
            }
            for (; i < end; ++i) {
                out[i] = int_tail_op<int8_t, Op>(a[i], b[i]);
            }
        }

        /**
         * 16 lanes per step; the tail runs scalar (wrap arithmetic, the exact
         * unsigned-twin arithmetic of apply_int).
         * @brief AVX2 range kernel body (16-bit lanes).
         */
        template<ElementwiseOp Op>
        void i16_range_avx2(int16_t* out, const int16_t* a, const int16_t* b, const size_t begin, const size_t end) {
            auto i = begin;
            for (; i + 16 <= end; i += 16) {
                const auto x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
                const auto y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), i16_vec_op<Op>(x, y));
            }
            for (; i < end; ++i) {
                out[i] = int_tail_op<int16_t, Op>(a[i], b[i]);
            }
        }

        /**
         * 8 lanes per step; the tail runs scalar (wrap arithmetic, the exact
         * unsigned-twin arithmetic of apply_int).
         * @brief AVX2 range kernel body (32-bit lanes).
         */
        template<ElementwiseOp Op>
        void i32_range_avx2(int32_t* out, const int32_t* a, const int32_t* b, const size_t begin, const size_t end) {
            auto i = begin;
            for (; i + 8 <= end; i += 8) {
                const auto x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
                const auto y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), i32_vec_op<Op>(x, y));
            }
            for (; i < end; ++i) {
                out[i] = int_tail_op<int32_t, Op>(a[i], b[i]);
            }
        }

        /**
         * 4 lanes per step; the tail runs scalar (wrap arithmetic, the exact
         * unsigned-twin arithmetic of apply_int).
         * @brief AVX2 range kernel body (64-bit lanes).
         */
        template<ElementwiseOp Op>
        void i64_range_avx2(int64_t* out, const int64_t* a, const int64_t* b, const size_t begin, const size_t end) {
            auto i = begin;
            for (; i + 4 <= end; i += 4) {
                const auto x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
                const auto y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
                __m256i r;
                if constexpr (Op == ElementwiseOp::Mul) {
                    r = i64_vec_mul(x, y);
                } else {
                    r = i64_vec_addsub<Op>(x, y);
                }
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), r);
            }
            for (; i < end; ++i) {
                out[i] = int_tail_op<int64_t, Op>(a[i], b[i]);
            }
        }

        /**
         * Every public kernel refuses to execute on a CPU lacking AVX2/F16C —
         * this both gates dispatch and gives direct callers (tests) a probe.
         * @brief Feature gate check for the AVX2 kernels.
         */
        void require_avx2() {
            if (!elementwise_avx2_supported()) {
                throw ErrorHandling::CPUIllegalInstruction("AVX2+F16C");
            }
        }

        // --------------------------- splat kernel bodies ---------------------------
        // Same loop shapes as the binary bodies above; the scalar operand is a
        // broadcast vector built once. The scalar arrives ALREADY narrowed into
        // the element type — the bit-exactness contract is against the scalar
        // apply with that narrowed scalar, not against narrowing inside the kernel.

        /**
         * 8 lanes per step; the tail runs scalar (IEEE single-rounding op against s,
         * the exact scalar apply_native pathway).
         * @brief AVX2 splat kernel body (fp32).
         */
        template<ElementwiseOp Op>
        void f32_splat_avx2(float32_t* out, const float32_t* a, const float32_t s, const size_t begin, const size_t end) {
            const auto vs = _mm256_set1_ps(s);
            auto i = begin;
            for (; i + 8 <= end; i += 8) {
                _mm256_storeu_ps(out + i, f32_vec_op<Op>(_mm256_loadu_ps(a + i), vs));
            }
            for (; i < end; ++i) {
                out[i] = Math::Elementwise::fp_binop<float32_t, Op>(a[i], s);
            }
        }

        /**
         * 4 lanes per step; the tail runs scalar (IEEE single-rounding op against s,
         * the exact scalar apply_native pathway).
         * @brief AVX2 splat kernel body (fp64).
         */
        template<ElementwiseOp Op>
        void f64_splat_avx2(float64_t* out, const float64_t* a, const float64_t s, const size_t begin, const size_t end) {
            const auto vs = _mm256_set1_pd(s);
            auto i = begin;
            for (; i + 4 <= end; i += 4) {
                _mm256_storeu_pd(out + i, f64_vec_op<Op>(_mm256_loadu_pd(a + i), vs));
            }
            for (; i < end; ++i) {
                out[i] = Math::Elementwise::fp_binop<float64_t, Op>(a[i], s);
            }
        }

        /**
         * 8 lanes per step; the tail runs scalar (f16_tail_op against the scalar's
         * lane bits — byte-for-byte the scalar apply_half pathway).
         * @brief AVX2 splat kernel body (fp16 flat lanes).
         */
        template<ElementwiseOp Op>
        void f16_splat_avx2(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
            const auto ws = _mm256_set1_ps(
                Intrinsics::detail::unaccel_fp16_to_fp32_untracked(
                    Intrinsics::detail::bit_cast_memcpy<float16_t>(s)));
            auto i = begin;
            for (; i + 8 <= end; i += 8) {
                const auto wa = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i)));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(out + i), f16_narrow8(f32_vec_op<Op>(wa, ws)));
            }
            for (; i < end; ++i) {
                out[i] = f16_tail_op<Op>(a[i], s);
            }
        }

        /**
         * 8 lanes per step; the tail runs scalar (bf16_tail_op against the scalar's
         * lane bits — byte-for-byte the scalar apply_half pathway).
         * @brief AVX2 splat kernel body (bf16 flat lanes).
         */
        template<ElementwiseOp Op>
        void bf16_splat_avx2(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
            const auto ws = _mm256_set1_ps(
                Intrinsics::detail::unaccel_bf16_to_fp32(
                    Intrinsics::detail::bit_cast_memcpy<bfloat16_t>(s)));
            auto i = begin;
            for (; i + 8 <= end; i += 8) {
                const auto wa = bf16_widen8(a + i);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(out + i), bf16_pack8(bf16_narrow8_lo(f32_vec_op<Op>(wa, ws))));
            }
            for (; i < end; ++i) {
                out[i] = bf16_tail_op<Op>(a[i], s);
            }
        }

        /**
         * 32 lanes per step; the tail runs scalar (wrap arithmetic against s,
         * byte-for-byte the unsigned-twin arithmetic of apply_int).
         * @brief AVX2 splat kernel body (8-bit lanes).
         */
        template<ElementwiseOp Op>
        void i8_splat_avx2(int8_t* out, const int8_t* a, const int8_t s, const size_t begin, const size_t end) {
            const auto vs = _mm256_set1_epi8(s);
            auto i = begin;
            for (; i + 32 <= end; i += 32) {
                const auto x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
                __m256i r;
                if constexpr (Op == ElementwiseOp::Mul) {
                    r = i8_vec_mul(x, vs);
                } else {
                    r = i8_vec_addsub<Op>(x, vs);
                }
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), r);
            }
            for (; i < end; ++i) {
                out[i] = int_tail_op<int8_t, Op>(a[i], s);
            }
        }

        /**
         * 16 lanes per step; the tail runs scalar (wrap arithmetic against s,
         * byte-for-byte the unsigned-twin arithmetic of apply_int).
         * @brief AVX2 splat kernel body (16-bit lanes).
         */
        template<ElementwiseOp Op>
        void i16_splat_avx2(int16_t* out, const int16_t* a, const int16_t s, const size_t begin, const size_t end) {
            const auto vs = _mm256_set1_epi16(s);
            auto i = begin;
            for (; i + 16 <= end; i += 16) {
                const auto x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), i16_vec_op<Op>(x, vs));
            }
            for (; i < end; ++i) {
                out[i] = int_tail_op<int16_t, Op>(a[i], s);
            }
        }

        /**
         * 8 lanes per step; the tail runs scalar (wrap arithmetic against s,
         * byte-for-byte the unsigned-twin arithmetic of apply_int).
         * @brief AVX2 splat kernel body (32-bit lanes).
         */
        template<ElementwiseOp Op>
        void i32_splat_avx2(int32_t* out, const int32_t* a, const int32_t s, const size_t begin, const size_t end) {
            const auto vs = _mm256_set1_epi32(s);
            auto i = begin;
            for (; i + 8 <= end; i += 8) {
                const auto x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), i32_vec_op<Op>(x, vs));
            }
            for (; i < end; ++i) {
                out[i] = int_tail_op<int32_t, Op>(a[i], s);
            }
        }

        /**
         * 4 lanes per step; the tail runs scalar (wrap arithmetic against s,
         * byte-for-byte the unsigned-twin arithmetic of apply_int).
         * @brief AVX2 splat kernel body (64-bit lanes).
         */
        template<ElementwiseOp Op>
        void i64_splat_avx2(int64_t* out, const int64_t* a, const int64_t s, const size_t begin, const size_t end) {
            const auto vs = _mm256_set1_epi64x(s);
            auto i = begin;
            for (; i + 4 <= end; i += 4) {
                const auto x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
                __m256i r;
                if constexpr (Op == ElementwiseOp::Mul) {
                    r = i64_vec_mul(x, vs);
                } else {
                    r = i64_vec_addsub<Op>(x, vs);
                }
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), r);
            }
            for (; i < end; ++i) {
                out[i] = int_tail_op<int64_t, Op>(a[i], s);
            }
        }

        // --------------------------- int4x2 helpers ---------------------------

        /**
         * The scalar apply_int4x2 sign-extends each nibble and keeps the low 4 bits
         * of the int32 result; the low 4 bits of (x op y) depend only on the low 4
         * bits of the operands, so unsigned-nibble arithmetic masked to 4 bits
         * reproduces the signed wrap bit-for-bit (e.g. -8 + -1 = -9 -> 0x7, and
         * 8 + 15 = 23 -> 0x7).
         * @brief Scalar int4x2 byte op (per-nibble wrap, bit-exact with apply_int4x2).
         */
        template<ElementwiseOp Op>
        [[nodiscard]] uint8_t i4x2_tail_op(const uint8_t a, const uint8_t b) {
            const auto alo = static_cast<uint32_t>(a & 0x0Fu);
            const auto ahi = static_cast<uint32_t>(a >> 4);
            const auto blo = static_cast<uint32_t>(b & 0x0Fu);
            const auto bhi = static_cast<uint32_t>(b >> 4);
            uint32_t rlo;
            uint32_t rhi;
            if constexpr (Op == ElementwiseOp::Add) {
                rlo = alo + blo;
                rhi = ahi + bhi;
            } else if constexpr (Op == ElementwiseOp::Sub) {
                rlo = alo - blo;
                rhi = ahi - bhi;
            } else {
                rlo = alo * blo;
                rhi = ahi * bhi;
            }
            return static_cast<uint8_t>((rlo & 0x0Fu) | ((rhi & 0x0Fu) << 4));
        }

        /**
         * Per-nibble vector op: unpack both operands' nibbles to zero-extended byte
         * lanes (vpandb 0x0F / vpsrlw 4 + vpandb — the 16-bit shift crosses byte
         * boundaries only in the bits the 0x0F mask removes, so each byte lane ends
         * up with its own nibble), apply the byte-lane op, mask 0x0F, and repack via
         * vpsllw 4 + vpor (the 16-bit left shift moves each nibble into its byte's
         * high half without leaving it — nibble-pair boundaries ARE 16-bit lanes).
         * @brief int4x2 vector op (one 256-bit block of 32 packed bytes).
         */
        template<ElementwiseOp Op>
        [[nodiscard]] __m256i i4x2_vec_op(const __m256i x, const __m256i y) {
            const auto low_mask = _mm256_set1_epi8(0x0F);
            const auto xlo = _mm256_and_si256(x, low_mask);
            const auto ylo = _mm256_and_si256(y, low_mask);
            const auto xhi = _mm256_and_si256(_mm256_srli_epi16(x, 4), low_mask);
            const auto yhi = _mm256_and_si256(_mm256_srli_epi16(y, 4), low_mask);
            __m256i rlo;
            __m256i rhi;
            if constexpr (Op == ElementwiseOp::Mul) {
                // No epi8 multiply: 16-bit widening like i8_vec_mul. Nibbles are
                // zero-extended, so the low byte holds the product; masking 0x0F
                // then keeps (a*b) mod 16, and 256 == 0 (mod 16) makes the low
                // byte of the product sufficient.
                rlo = _mm256_and_si256(i8_vec_mul(xlo, ylo), low_mask);
                rhi = _mm256_and_si256(i8_vec_mul(xhi, yhi), low_mask);
            } else {
                rlo = _mm256_and_si256(i8_vec_addsub<Op>(xlo, ylo), low_mask);
                rhi = _mm256_and_si256(i8_vec_addsub<Op>(xhi, yhi), low_mask);
            }
            return _mm256_or_si256(rlo, _mm256_slli_epi16(rhi, 4));
        }

        /**
         * 32 packed bytes per step; the tail runs scalar (per-nibble wrap, the exact
         * unsigned-nibble identity of apply_int4x2).
         * @brief AVX2 range kernel body (int4x2).
         */
        template<ElementwiseOp Op>
        void i4x2_range_avx2(uint8_t* out, const uint8_t* a, const uint8_t* b, const size_t begin, const size_t end) {
            auto i = begin;
            for (; i + 32 <= end; i += 32) {
                const auto x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
                const auto y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), i4x2_vec_op<Op>(x, y));
            }
            for (; i < end; ++i) {
                out[i] = i4x2_tail_op<Op>(a[i], b[i]);
            }
        }

        // --------------------------- left-splat kernel bodies ---------------------------
        // Broadcast rows with operand A on inner stride 0 compute s (op) a[i] — the
        // scalar on the LEFT, the exact operand order of the scalar apply (matters
        // for sub/div, and selects the same NaN payload for commutative fp ops).

        /**
         * 8 lanes per step; the tail runs scalar with the scalar on the left.
         * @brief AVX2 left-splat kernel body (fp32).
         */
        template<ElementwiseOp Op>
        void f32_splat_lhs_avx2(float32_t* out, const float32_t* a, const float32_t s, const size_t begin, const size_t end) {
            const auto vs = _mm256_set1_ps(s);
            auto i = begin;
            for (; i + 8 <= end; i += 8) {
                _mm256_storeu_ps(out + i, f32_vec_op<Op>(vs, _mm256_loadu_ps(a + i)));
            }
            for (; i < end; ++i) {
                out[i] = Math::Elementwise::fp_binop<float32_t, Op>(s, a[i]);
            }
        }

        /**
         * 4 lanes per step; the tail runs scalar with the scalar on the left.
         * @brief AVX2 left-splat kernel body (fp64).
         */
        template<ElementwiseOp Op>
        void f64_splat_lhs_avx2(float64_t* out, const float64_t* a, const float64_t s, const size_t begin, const size_t end) {
            const auto vs = _mm256_set1_pd(s);
            auto i = begin;
            for (; i + 4 <= end; i += 4) {
                _mm256_storeu_pd(out + i, f64_vec_op<Op>(vs, _mm256_loadu_pd(a + i)));
            }
            for (; i < end; ++i) {
                out[i] = Math::Elementwise::fp_binop<float64_t, Op>(s, a[i]);
            }
        }

        /**
         * 8 lanes per step; the tail runs scalar with the scalar on the left
         * (byte-for-byte the scalar apply_half pathway).
         * @brief AVX2 left-splat kernel body (fp16 flat lanes).
         */
        template<ElementwiseOp Op>
        void f16_splat_lhs_avx2(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
            const auto ws = _mm256_set1_ps(
                Intrinsics::detail::unaccel_fp16_to_fp32_untracked(
                    Intrinsics::detail::bit_cast_memcpy<float16_t>(s)));
            auto i = begin;
            for (; i + 8 <= end; i += 8) {
                const auto wa = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i)));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(out + i), f16_narrow8(f32_vec_op<Op>(ws, wa)));
            }
            for (; i < end; ++i) {
                out[i] = f16_tail_op<Op>(s, a[i]);
            }
        }

        /**
         * 8 lanes per step; the tail runs scalar with the scalar on the left
         * (byte-for-byte the scalar apply_half pathway).
         * @brief AVX2 left-splat kernel body (bf16 flat lanes).
         */
        template<ElementwiseOp Op>
        void bf16_splat_lhs_avx2(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
            const auto ws = _mm256_set1_ps(
                Intrinsics::detail::unaccel_bf16_to_fp32(
                    Intrinsics::detail::bit_cast_memcpy<bfloat16_t>(s)));
            auto i = begin;
            for (; i + 8 <= end; i += 8) {
                const auto wa = bf16_widen8(a + i);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(out + i), bf16_pack8(bf16_narrow8_lo(f32_vec_op<Op>(ws, wa))));
            }
            for (; i < end; ++i) {
                out[i] = bf16_tail_op<Op>(s, a[i]);
            }
        }

        /**
         * 32/16/8/4 lanes per step; the tail runs scalar with the scalar on the left
         * (byte-for-byte the unsigned-twin arithmetic of apply_int). Sub only — wrap
         * add/mul are bitwise commutative, so the plain splats already match.
         * @brief AVX2 left-splat kernel body (8/16/32/64-bit lanes, sub).
         */
        template<typename T>
        void int_splat_lhs_sub_avx2(T* out, const T* a, const T s, const size_t begin, const size_t end) {
            auto i = begin;
            if constexpr (sizeof(T) == 1) {
                const auto vs = _mm256_set1_epi8(s);
                for (; i + 32 <= end; i += 32) {
                    const auto x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), _mm256_sub_epi8(vs, x));
                }
            } else if constexpr (sizeof(T) == 2) {
                const auto vs = _mm256_set1_epi16(s);
                for (; i + 16 <= end; i += 16) {
                    const auto x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), _mm256_sub_epi16(vs, x));
                }
            } else if constexpr (sizeof(T) == 4) {
                const auto vs = _mm256_set1_epi32(s);
                for (; i + 8 <= end; i += 8) {
                    const auto x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), _mm256_sub_epi32(vs, x));
                }
            } else {
                const auto vs = _mm256_set1_epi64x(s);
                for (; i + 4 <= end; i += 4) {
                    const auto x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), _mm256_sub_epi64(vs, x));
                }
            }
            for (; i < end; ++i) {
                out[i] = int_tail_op<T, ElementwiseOp::Sub>(s, a[i]);
            }
        }
    }

    /**
     * Cached through IncludedCPUFeatures(); callers gate every AVX2 kernel on this.
     * @brief AVX2 && F16C verification, for elementwise kernels.
     * @return T/F - AVX2 && F16C is supported by the CPU.
     */
    bool elementwise_avx2_supported() {
        const auto& f = IncludedCPUFeatures();
        return f.has_feature(CPUFeatureSetType_AVX2) && f.has_feature(CPUFeatureSetType_F16C);
    }

    /**
     * Computes out[i] = a[i] + b[i] over [begin, end), 8 lanes per step.
     * @brief AVX2 range add (fp32).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void add_f32_range_avx2(float32_t* out, const float32_t* a, const float32_t* b, const size_t begin, const size_t end) {
        require_avx2();
        f32_range_avx2<ElementwiseOp::Add>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = a[i] - b[i] over [begin, end), 8 lanes per step.
     * @brief AVX2 range sub (fp32).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_f32_range_avx2(float32_t* out, const float32_t* a, const float32_t* b, const size_t begin, const size_t end) {
        require_avx2();
        f32_range_avx2<ElementwiseOp::Sub>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = a[i] * b[i] over [begin, end), 8 lanes per step.
     * @brief AVX2 range mul (fp32).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void mul_f32_range_avx2(float32_t* out, const float32_t* a, const float32_t* b, const size_t begin, const size_t end) {
        require_avx2();
        f32_range_avx2<ElementwiseOp::Mul>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = a[i] / b[i] over [begin, end), 8 lanes per step. IEEE div.
     * @brief AVX2 range div (fp32).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void div_f32_range_avx2(float32_t* out, const float32_t* a, const float32_t* b, const size_t begin, const size_t end) {
        require_avx2();
        f32_range_avx2<ElementwiseOp::Div>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = a[i] + b[i] over [begin, end), 4 lanes per step.
     * @brief AVX2 range add (fp64).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void add_f64_range_avx2(float64_t* out, const float64_t* a, const float64_t* b, const size_t begin, const size_t end) {
        require_avx2();
        f64_range_avx2<ElementwiseOp::Add>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = a[i] - b[i] over [begin, end), 4 lanes per step.
     * @brief AVX2 range sub (fp64).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_f64_range_avx2(float64_t* out, const float64_t* a, const float64_t* b, const size_t begin, const size_t end) {
        require_avx2();
        f64_range_avx2<ElementwiseOp::Sub>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = a[i] * b[i] over [begin, end), 4 lanes per step.
     * @brief AVX2 range mul (fp64).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void mul_f64_range_avx2(float64_t* out, const float64_t* a, const float64_t* b, const size_t begin, const size_t end) {
        require_avx2();
        f64_range_avx2<ElementwiseOp::Mul>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = a[i] / b[i] over [begin, end), 4 lanes per step. IEEE div.
     * @brief AVX2 range div (fp64).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void div_f64_range_avx2(float64_t* out, const float64_t* a, const float64_t* b, const size_t begin, const size_t end) {
        require_avx2();
        f64_range_avx2<ElementwiseOp::Div>(out, a, b, begin, end);
    }

    /**
     * Per-lane widen/add/narrow over [begin, end), 8 lanes per step; the tail runs
     * through the exact scalar widen/apply/narrow pathway.
     * @brief AVX2 range add (fp16 flat lanes).
     * @param out Output lane buffer.
     * @param a Left operand lane buffer.
     * @param b Right operand lane buffer.
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void add_f16_range_avx2(uint16_t* out, const uint16_t* a, const uint16_t* b, const size_t begin, const size_t end) {
        require_avx2();
        f16_range_avx2<ElementwiseOp::Add>(out, a, b, begin, end);
    }

    /**
     * Per-lane widen/sub/narrow over [begin, end), 8 lanes per step; the tail runs
     * through the exact scalar widen/apply/narrow pathway.
     * @brief AVX2 range sub (fp16 flat lanes).
     * @param out Output lane buffer.
     * @param a Left operand lane buffer.
     * @param b Right operand lane buffer.
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void sub_f16_range_avx2(uint16_t* out, const uint16_t* a, const uint16_t* b, const size_t begin, const size_t end) {
        require_avx2();
        f16_range_avx2<ElementwiseOp::Sub>(out, a, b, begin, end);
    }

    /**
     * Per-lane widen/mul/narrow over [begin, end), 8 lanes per step; the tail runs
     * through the exact scalar widen/apply/narrow pathway.
     * @brief AVX2 range mul (fp16 flat lanes).
     * @param out Output lane buffer.
     * @param a Left operand lane buffer.
     * @param b Right operand lane buffer.
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void mul_f16_range_avx2(uint16_t* out, const uint16_t* a, const uint16_t* b, const size_t begin, const size_t end) {
        require_avx2();
        f16_range_avx2<ElementwiseOp::Mul>(out, a, b, begin, end);
    }

    /**
     * Per-lane widen/div/narrow over [begin, end), 8 lanes per step; the tail runs
     * through the exact scalar widen/apply/narrow pathway.
     * @brief AVX2 range div (fp16 flat lanes).
     * @param out Output lane buffer.
     * @param a Left operand lane buffer.
     * @param b Right operand lane buffer.
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void div_f16_range_avx2(uint16_t* out, const uint16_t* a, const uint16_t* b, const size_t begin, const size_t end) {
        require_avx2();
        f16_range_avx2<ElementwiseOp::Div>(out, a, b, begin, end);
    }

    /**
     * Per-lane widen/add/narrow over [begin, end), 8 lanes per step; the tail runs
     * through the exact scalar widen/apply/narrow pathway.
     * @brief AVX2 range add (bf16 flat lanes).
     * @param out Output lane buffer.
     * @param a Left operand lane buffer.
     * @param b Right operand lane buffer.
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void add_bf16_range_avx2(uint16_t* out, const uint16_t* a, const uint16_t* b, const size_t begin, const size_t end) {
        require_avx2();
        bf16_range_avx2<ElementwiseOp::Add>(out, a, b, begin, end);
    }

    /**
     * Per-lane widen/sub/narrow over [begin, end), 8 lanes per step; the tail runs
     * through the exact scalar widen/apply/narrow pathway.
     * @brief AVX2 range sub (bf16 flat lanes).
     * @param out Output lane buffer.
     * @param a Left operand lane buffer.
     * @param b Right operand lane buffer.
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void sub_bf16_range_avx2(uint16_t* out, const uint16_t* a, const uint16_t* b, const size_t begin, const size_t end) {
        require_avx2();
        bf16_range_avx2<ElementwiseOp::Sub>(out, a, b, begin, end);
    }

    /**
     * Per-lane widen/mul/narrow over [begin, end), 8 lanes per step; the tail runs
     * through the exact scalar widen/apply/narrow pathway.
     * @brief AVX2 range mul (bf16 flat lanes).
     * @param out Output lane buffer.
     * @param a Left operand lane buffer.
     * @param b Right operand lane buffer.
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void mul_bf16_range_avx2(uint16_t* out, const uint16_t* a, const uint16_t* b, const size_t begin, const size_t end) {
        require_avx2();
        bf16_range_avx2<ElementwiseOp::Mul>(out, a, b, begin, end);
    }

    /**
     * Per-lane widen/div/narrow over [begin, end), 8 lanes per step; the tail runs
     * through the exact scalar widen/apply/narrow pathway.
     * @brief AVX2 range div (bf16 flat lanes).
     * @param out Output lane buffer.
     * @param a Left operand lane buffer.
     * @param b Right operand lane buffer.
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void div_bf16_range_avx2(uint16_t* out, const uint16_t* a, const uint16_t* b, const size_t begin, const size_t end) {
        require_avx2();
        bf16_range_avx2<ElementwiseOp::Div>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = (a[i] + b[i]) mod 2^8 over [begin, end), 32 lanes per step.
     * @brief AVX2 range add (8-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void add_i8_range_avx2(int8_t* out, const int8_t* a, const int8_t* b, const size_t begin, const size_t end) {
        require_avx2();
        i8_range_avx2<ElementwiseOp::Add>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = (a[i] - b[i]) mod 2^8 over [begin, end), 32 lanes per step.
     * @brief AVX2 range sub (8-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_i8_range_avx2(int8_t* out, const int8_t* a, const int8_t* b, const size_t begin, const size_t end) {
        require_avx2();
        i8_range_avx2<ElementwiseOp::Sub>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = (a[i] * b[i]) mod 2^8 over [begin, end), 32 lanes per step.
     * @brief AVX2 range mul (8-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void mul_i8_range_avx2(int8_t* out, const int8_t* a, const int8_t* b, const size_t begin, const size_t end) {
        require_avx2();
        i8_range_avx2<ElementwiseOp::Mul>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = (a[i] + b[i]) mod 2^16 over [begin, end), 16 lanes per step.
     * @brief AVX2 range add (16-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void add_i16_range_avx2(int16_t* out, const int16_t* a, const int16_t* b, const size_t begin, const size_t end) {
        require_avx2();
        i16_range_avx2<ElementwiseOp::Add>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = (a[i] - b[i]) mod 2^16 over [begin, end), 16 lanes per step.
     * @brief AVX2 range sub (16-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_i16_range_avx2(int16_t* out, const int16_t* a, const int16_t* b, const size_t begin, const size_t end) {
        require_avx2();
        i16_range_avx2<ElementwiseOp::Sub>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = (a[i] * b[i]) mod 2^16 over [begin, end), 16 lanes per step.
     * @brief AVX2 range mul (16-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void mul_i16_range_avx2(int16_t* out, const int16_t* a, const int16_t* b, const size_t begin, const size_t end) {
        require_avx2();
        i16_range_avx2<ElementwiseOp::Mul>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = (a[i] + b[i]) mod 2^32 over [begin, end), 8 lanes per step.
     * @brief AVX2 range add (32-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void add_i32_range_avx2(int32_t* out, const int32_t* a, const int32_t* b, const size_t begin, const size_t end) {
        require_avx2();
        i32_range_avx2<ElementwiseOp::Add>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = (a[i] - b[i]) mod 2^32 over [begin, end), 8 lanes per step.
     * @brief AVX2 range sub (32-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_i32_range_avx2(int32_t* out, const int32_t* a, const int32_t* b, const size_t begin, const size_t end) {
        require_avx2();
        i32_range_avx2<ElementwiseOp::Sub>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = (a[i] * b[i]) mod 2^32 over [begin, end), 8 lanes per step.
     * @brief AVX2 range mul (32-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void mul_i32_range_avx2(int32_t* out, const int32_t* a, const int32_t* b, const size_t begin, const size_t end) {
        require_avx2();
        i32_range_avx2<ElementwiseOp::Mul>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = (a[i] + b[i]) mod 2^64 over [begin, end), 4 lanes per step.
     * @brief AVX2 range add (64-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void add_i64_range_avx2(int64_t* out, const int64_t* a, const int64_t* b, const size_t begin, const size_t end) {
        require_avx2();
        i64_range_avx2<ElementwiseOp::Add>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = (a[i] - b[i]) mod 2^64 over [begin, end), 4 lanes per step.
     * @brief AVX2 range sub (64-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_i64_range_avx2(int64_t* out, const int64_t* a, const int64_t* b, const size_t begin, const size_t end) {
        require_avx2();
        i64_range_avx2<ElementwiseOp::Sub>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = (a[i] * b[i]) mod 2^64 over [begin, end), 4 lanes per step.
     * @brief AVX2 range mul (64-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void mul_i64_range_avx2(int64_t* out, const int64_t* a, const int64_t* b, const size_t begin, const size_t end) {
        require_avx2();
        i64_range_avx2<ElementwiseOp::Mul>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = a[i] + s over [begin, end), 8 lanes per step.
     * @brief AVX2 range splat add (fp32).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into fp32).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void add_f32_splat_avx2(float32_t* out, const float32_t* a, const float32_t s, const size_t begin, const size_t end) {
        require_avx2();
        f32_splat_avx2<ElementwiseOp::Add>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = a[i] - s over [begin, end), 8 lanes per step.
     * @brief AVX2 range splat sub (fp32).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into fp32).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_f32_splat_avx2(float32_t* out, const float32_t* a, const float32_t s, const size_t begin, const size_t end) {
        require_avx2();
        f32_splat_avx2<ElementwiseOp::Sub>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = a[i] * s over [begin, end), 8 lanes per step.
     * @brief AVX2 range splat mul (fp32).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into fp32).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void mul_f32_splat_avx2(float32_t* out, const float32_t* a, const float32_t s, const size_t begin, const size_t end) {
        require_avx2();
        f32_splat_avx2<ElementwiseOp::Mul>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = a[i] / s over [begin, end), 8 lanes per step. IEEE div.
     * @brief AVX2 range splat div (fp32).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into fp32).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void div_f32_splat_avx2(float32_t* out, const float32_t* a, const float32_t s, const size_t begin, const size_t end) {
        require_avx2();
        f32_splat_avx2<ElementwiseOp::Div>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = a[i] + s over [begin, end), 4 lanes per step.
     * @brief AVX2 range splat add (fp64).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into fp64).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void add_f64_splat_avx2(float64_t* out, const float64_t* a, const float64_t s, const size_t begin, const size_t end) {
        require_avx2();
        f64_splat_avx2<ElementwiseOp::Add>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = a[i] - s over [begin, end), 4 lanes per step.
     * @brief AVX2 range splat sub (fp64).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into fp64).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_f64_splat_avx2(float64_t* out, const float64_t* a, const float64_t s, const size_t begin, const size_t end) {
        require_avx2();
        f64_splat_avx2<ElementwiseOp::Sub>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = a[i] * s over [begin, end), 4 lanes per step.
     * @brief AVX2 range splat mul (fp64).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into fp64).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void mul_f64_splat_avx2(float64_t* out, const float64_t* a, const float64_t s, const size_t begin, const size_t end) {
        require_avx2();
        f64_splat_avx2<ElementwiseOp::Mul>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = a[i] / s over [begin, end), 4 lanes per step. IEEE div.
     * @brief AVX2 range splat div (fp64).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into fp64).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void div_f64_splat_avx2(float64_t* out, const float64_t* a, const float64_t s, const size_t begin, const size_t end) {
        require_avx2();
        f64_splat_avx2<ElementwiseOp::Div>(out, a, s, begin, end);
    }

    /**
     * Per-lane widen/add/narrow against the splat scalar over [begin, end), 8 lanes
     * per step; the tail runs through the exact scalar widen/apply/narrow pathway.
     * @brief AVX2 range splat add (fp16 flat lanes).
     * @param out Output lane buffer.
     * @param a Left operand lane buffer.
     * @param s Scalar lane bit pattern (already narrowed into fp16).
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void add_f16_splat_avx2(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
        require_avx2();
        f16_splat_avx2<ElementwiseOp::Add>(out, a, s, begin, end);
    }

    /**
     * Per-lane widen/sub/narrow against the splat scalar over [begin, end), 8 lanes
     * per step; the tail runs through the exact scalar widen/apply/narrow pathway.
     * @brief AVX2 range splat sub (fp16 flat lanes).
     * @param out Output lane buffer.
     * @param a Left operand lane buffer.
     * @param s Scalar lane bit pattern (already narrowed into fp16).
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void sub_f16_splat_avx2(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
        require_avx2();
        f16_splat_avx2<ElementwiseOp::Sub>(out, a, s, begin, end);
    }

    /**
     * Per-lane widen/mul/narrow against the splat scalar over [begin, end), 8 lanes
     * per step; the tail runs through the exact scalar widen/apply/narrow pathway.
     * @brief AVX2 range splat mul (fp16 flat lanes).
     * @param out Output lane buffer.
     * @param a Left operand lane buffer.
     * @param s Scalar lane bit pattern (already narrowed into fp16).
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void mul_f16_splat_avx2(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
        require_avx2();
        f16_splat_avx2<ElementwiseOp::Mul>(out, a, s, begin, end);
    }

    /**
     * Per-lane widen/div/narrow against the splat scalar over [begin, end), 8 lanes
     * per step; the tail runs through the exact scalar widen/apply/narrow pathway.
     * @brief AVX2 range splat div (fp16 flat lanes).
     * @param out Output lane buffer.
     * @param a Left operand lane buffer.
     * @param s Scalar lane bit pattern (already narrowed into fp16).
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void div_f16_splat_avx2(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
        require_avx2();
        f16_splat_avx2<ElementwiseOp::Div>(out, a, s, begin, end);
    }

    /**
     * Per-lane widen/add/narrow against the splat scalar over [begin, end), 8 lanes
     * per step; the tail runs through the exact scalar widen/apply/narrow pathway.
     * @brief AVX2 range splat add (bf16 flat lanes).
     * @param out Output lane buffer.
     * @param a Left operand lane buffer.
     * @param s Scalar lane bit pattern (already narrowed into bf16).
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void add_bf16_splat_avx2(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
        require_avx2();
        bf16_splat_avx2<ElementwiseOp::Add>(out, a, s, begin, end);
    }

    /**
     * Per-lane widen/sub/narrow against the splat scalar over [begin, end), 8 lanes
     * per step; the tail runs through the exact scalar widen/apply/narrow pathway.
     * @brief AVX2 range splat sub (bf16 flat lanes).
     * @param out Output lane buffer.
     * @param a Left operand lane buffer.
     * @param s Scalar lane bit pattern (already narrowed into bf16).
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void sub_bf16_splat_avx2(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
        require_avx2();
        bf16_splat_avx2<ElementwiseOp::Sub>(out, a, s, begin, end);
    }

    /**
     * Per-lane widen/mul/narrow against the splat scalar over [begin, end), 8 lanes
     * per step; the tail runs through the exact scalar widen/apply/narrow pathway.
     * @brief AVX2 range splat mul (bf16 flat lanes).
     * @param out Output lane buffer.
     * @param a Left operand lane buffer.
     * @param s Scalar lane bit pattern (already narrowed into bf16).
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void mul_bf16_splat_avx2(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
        require_avx2();
        bf16_splat_avx2<ElementwiseOp::Mul>(out, a, s, begin, end);
    }

    /**
     * Per-lane widen/div/narrow against the splat scalar over [begin, end), 8 lanes
     * per step; the tail runs through the exact scalar widen/apply/narrow pathway.
     * @brief AVX2 range splat div (bf16 flat lanes).
     * @param out Output lane buffer.
     * @param a Left operand lane buffer.
     * @param s Scalar lane bit pattern (already narrowed into bf16).
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void div_bf16_splat_avx2(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
        require_avx2();
        bf16_splat_avx2<ElementwiseOp::Div>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = (a[i] + s) mod 2^8 over [begin, end), 32 lanes per step.
     * @brief AVX2 range splat add (8-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into the 8-bit dtype).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void add_i8_splat_avx2(int8_t* out, const int8_t* a, const int8_t s, const size_t begin, const size_t end) {
        require_avx2();
        i8_splat_avx2<ElementwiseOp::Add>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = (a[i] - s) mod 2^8 over [begin, end), 32 lanes per step.
     * @brief AVX2 range splat sub (8-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into the 8-bit dtype).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_i8_splat_avx2(int8_t* out, const int8_t* a, const int8_t s, const size_t begin, const size_t end) {
        require_avx2();
        i8_splat_avx2<ElementwiseOp::Sub>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = (a[i] * s) mod 2^8 over [begin, end), 32 lanes per step.
     * @brief AVX2 range splat mul (8-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into the 8-bit dtype).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void mul_i8_splat_avx2(int8_t* out, const int8_t* a, const int8_t s, const size_t begin, const size_t end) {
        require_avx2();
        i8_splat_avx2<ElementwiseOp::Mul>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = (a[i] + s) mod 2^16 over [begin, end), 16 lanes per step.
     * @brief AVX2 range splat add (16-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into the 16-bit dtype).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void add_i16_splat_avx2(int16_t* out, const int16_t* a, const int16_t s, const size_t begin, const size_t end) {
        require_avx2();
        i16_splat_avx2<ElementwiseOp::Add>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = (a[i] - s) mod 2^16 over [begin, end), 16 lanes per step.
     * @brief AVX2 range splat sub (16-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into the 16-bit dtype).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_i16_splat_avx2(int16_t* out, const int16_t* a, const int16_t s, const size_t begin, const size_t end) {
        require_avx2();
        i16_splat_avx2<ElementwiseOp::Sub>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = (a[i] * s) mod 2^16 over [begin, end), 16 lanes per step.
     * @brief AVX2 range splat mul (16-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into the 16-bit dtype).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void mul_i16_splat_avx2(int16_t* out, const int16_t* a, const int16_t s, const size_t begin, const size_t end) {
        require_avx2();
        i16_splat_avx2<ElementwiseOp::Mul>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = (a[i] + s) mod 2^32 over [begin, end), 8 lanes per step.
     * @brief AVX2 range splat add (32-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into the 32-bit dtype).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void add_i32_splat_avx2(int32_t* out, const int32_t* a, const int32_t s, const size_t begin, const size_t end) {
        require_avx2();
        i32_splat_avx2<ElementwiseOp::Add>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = (a[i] - s) mod 2^32 over [begin, end), 8 lanes per step.
     * @brief AVX2 range splat sub (32-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into the 32-bit dtype).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_i32_splat_avx2(int32_t* out, const int32_t* a, const int32_t s, const size_t begin, const size_t end) {
        require_avx2();
        i32_splat_avx2<ElementwiseOp::Sub>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = (a[i] * s) mod 2^32 over [begin, end), 8 lanes per step.
     * @brief AVX2 range splat mul (32-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into the 32-bit dtype).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void mul_i32_splat_avx2(int32_t* out, const int32_t* a, const int32_t s, const size_t begin, const size_t end) {
        require_avx2();
        i32_splat_avx2<ElementwiseOp::Mul>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = (a[i] + s) mod 2^64 over [begin, end), 4 lanes per step.
     * @brief AVX2 range splat add (64-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into the 64-bit dtype).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void add_i64_splat_avx2(int64_t* out, const int64_t* a, const int64_t s, const size_t begin, const size_t end) {
        require_avx2();
        i64_splat_avx2<ElementwiseOp::Add>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = (a[i] - s) mod 2^64 over [begin, end), 4 lanes per step.
     * @brief AVX2 range splat sub (64-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into the 64-bit dtype).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_i64_splat_avx2(int64_t* out, const int64_t* a, const int64_t s, const size_t begin, const size_t end) {
        require_avx2();
        i64_splat_avx2<ElementwiseOp::Sub>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = (a[i] * s) mod 2^64 over [begin, end), 4 lanes per step.
     * @brief AVX2 range splat mul (64-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into the 64-bit dtype).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void mul_i64_splat_avx2(int64_t* out, const int64_t* a, const int64_t s, const size_t begin, const size_t end) {
        require_avx2();
        i64_splat_avx2<ElementwiseOp::Mul>(out, a, s, begin, end);
    }

    /**
     * Computes the per-nibble wrap add over [begin, end), 32 packed bytes per step.
     * @brief AVX2 range add (int4x2).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void add_i4x2_range_avx2(uint8_t* out, const uint8_t* a, const uint8_t* b, const size_t begin, const size_t end) {
        require_avx2();
        i4x2_range_avx2<ElementwiseOp::Add>(out, a, b, begin, end);
    }

    /**
     * Computes the per-nibble wrap sub over [begin, end), 32 packed bytes per step.
     * @brief AVX2 range sub (int4x2).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_i4x2_range_avx2(uint8_t* out, const uint8_t* a, const uint8_t* b, const size_t begin, const size_t end) {
        require_avx2();
        i4x2_range_avx2<ElementwiseOp::Sub>(out, a, b, begin, end);
    }

    /**
     * Computes the per-nibble wrap mul over [begin, end), 32 packed bytes per step.
     * @brief AVX2 range mul (int4x2).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void mul_i4x2_range_avx2(uint8_t* out, const uint8_t* a, const uint8_t* b, const size_t begin, const size_t end) {
        require_avx2();
        i4x2_range_avx2<ElementwiseOp::Mul>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = s + a[i] over [begin, end), 8 lanes per step — scalar on the left.
     * @brief AVX2 left splat add (fp32).
     * @param out Output buffer.
     * @param a Input buffer.
     * @param s Scalar operand (already narrowed into fp32).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void add_f32_splat_lhs_avx2(float32_t* out, const float32_t* a, const float32_t s, const size_t begin, const size_t end) {
        require_avx2();
        f32_splat_lhs_avx2<ElementwiseOp::Add>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = s - a[i] over [begin, end), 8 lanes per step — scalar on the left.
     * @brief AVX2 left splat sub (fp32).
     * @param out Output buffer.
     * @param a Input buffer.
     * @param s Scalar operand (already narrowed into fp32).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_f32_splat_lhs_avx2(float32_t* out, const float32_t* a, const float32_t s, const size_t begin, const size_t end) {
        require_avx2();
        f32_splat_lhs_avx2<ElementwiseOp::Sub>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = s * a[i] over [begin, end), 8 lanes per step — scalar on the left.
     * @brief AVX2 left splat mul (fp32).
     * @param out Output buffer.
     * @param a Input buffer.
     * @param s Scalar operand (already narrowed into fp32).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void mul_f32_splat_lhs_avx2(float32_t* out, const float32_t* a, const float32_t s, const size_t begin, const size_t end) {
        require_avx2();
        f32_splat_lhs_avx2<ElementwiseOp::Mul>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = s / a[i] over [begin, end), 8 lanes per step — scalar on the left. IEEE div.
     * @brief AVX2 left splat div (fp32).
     * @param out Output buffer.
     * @param a Input buffer.
     * @param s Scalar operand (already narrowed into fp32).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void div_f32_splat_lhs_avx2(float32_t* out, const float32_t* a, const float32_t s, const size_t begin, const size_t end) {
        require_avx2();
        f32_splat_lhs_avx2<ElementwiseOp::Div>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = s + a[i] over [begin, end), 4 lanes per step — scalar on the left.
     * @brief AVX2 left splat add (fp64).
     * @param out Output buffer.
     * @param a Input buffer.
     * @param s Scalar operand (already narrowed into fp64).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void add_f64_splat_lhs_avx2(float64_t* out, const float64_t* a, const float64_t s, const size_t begin, const size_t end) {
        require_avx2();
        f64_splat_lhs_avx2<ElementwiseOp::Add>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = s - a[i] over [begin, end), 4 lanes per step — scalar on the left.
     * @brief AVX2 left splat sub (fp64).
     * @param out Output buffer.
     * @param a Input buffer.
     * @param s Scalar operand (already narrowed into fp64).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_f64_splat_lhs_avx2(float64_t* out, const float64_t* a, const float64_t s, const size_t begin, const size_t end) {
        require_avx2();
        f64_splat_lhs_avx2<ElementwiseOp::Sub>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = s * a[i] over [begin, end), 4 lanes per step — scalar on the left.
     * @brief AVX2 left splat mul (fp64).
     * @param out Output buffer.
     * @param a Input buffer.
     * @param s Scalar operand (already narrowed into fp64).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void mul_f64_splat_lhs_avx2(float64_t* out, const float64_t* a, const float64_t s, const size_t begin, const size_t end) {
        require_avx2();
        f64_splat_lhs_avx2<ElementwiseOp::Mul>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = s / a[i] over [begin, end), 4 lanes per step — scalar on the left. IEEE div.
     * @brief AVX2 left splat div (fp64).
     * @param out Output buffer.
     * @param a Input buffer.
     * @param s Scalar operand (already narrowed into fp64).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void div_f64_splat_lhs_avx2(float64_t* out, const float64_t* a, const float64_t s, const size_t begin, const size_t end) {
        require_avx2();
        f64_splat_lhs_avx2<ElementwiseOp::Div>(out, a, s, begin, end);
    }

    /**
     * Per-lane widen/add/narrow with the scalar on the LEFT over [begin, end), 8
     * lanes per step; the tail runs through the exact scalar widen/apply/narrow pathway.
     * @brief AVX2 left splat add (fp16 flat lanes).
     * @param out Output lane buffer.
     * @param a Input lane buffer.
     * @param s Scalar lane bit pattern (already narrowed into fp16).
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void add_f16_splat_lhs_avx2(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
        require_avx2();
        f16_splat_lhs_avx2<ElementwiseOp::Add>(out, a, s, begin, end);
    }

    /**
     * Per-lane widen/sub/narrow with the scalar on the LEFT over [begin, end), 8
     * lanes per step; the tail runs through the exact scalar widen/apply/narrow pathway.
     * @brief AVX2 left splat sub (fp16 flat lanes).
     * @param out Output lane buffer.
     * @param a Input lane buffer.
     * @param s Scalar lane bit pattern (already narrowed into fp16).
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void sub_f16_splat_lhs_avx2(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
        require_avx2();
        f16_splat_lhs_avx2<ElementwiseOp::Sub>(out, a, s, begin, end);
    }

    /**
     * Per-lane widen/mul/narrow with the scalar on the LEFT over [begin, end), 8
     * lanes per step; the tail runs through the exact scalar widen/apply/narrow pathway.
     * @brief AVX2 left splat mul (fp16 flat lanes).
     * @param out Output lane buffer.
     * @param a Input lane buffer.
     * @param s Scalar lane bit pattern (already narrowed into fp16).
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void mul_f16_splat_lhs_avx2(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
        require_avx2();
        f16_splat_lhs_avx2<ElementwiseOp::Mul>(out, a, s, begin, end);
    }

    /**
     * Per-lane widen/div/narrow with the scalar on the LEFT over [begin, end), 8
     * lanes per step; the tail runs through the exact scalar widen/apply/narrow pathway.
     * @brief AVX2 left splat div (fp16 flat lanes).
     * @param out Output lane buffer.
     * @param a Input lane buffer.
     * @param s Scalar lane bit pattern (already narrowed into fp16).
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void div_f16_splat_lhs_avx2(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
        require_avx2();
        f16_splat_lhs_avx2<ElementwiseOp::Div>(out, a, s, begin, end);
    }

    /**
     * Per-lane widen/add/narrow with the scalar on the LEFT over [begin, end), 8
     * lanes per step; the tail runs through the exact scalar widen/apply/narrow pathway.
     * @brief AVX2 left splat add (bf16 flat lanes).
     * @param out Output lane buffer.
     * @param a Input lane buffer.
     * @param s Scalar lane bit pattern (already narrowed into bf16).
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void add_bf16_splat_lhs_avx2(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
        require_avx2();
        bf16_splat_lhs_avx2<ElementwiseOp::Add>(out, a, s, begin, end);
    }

    /**
     * Per-lane widen/sub/narrow with the scalar on the LEFT over [begin, end), 8
     * lanes per step; the tail runs through the exact scalar widen/apply/narrow pathway.
     * @brief AVX2 left splat sub (bf16 flat lanes).
     * @param out Output lane buffer.
     * @param a Input lane buffer.
     * @param s Scalar lane bit pattern (already narrowed into bf16).
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void sub_bf16_splat_lhs_avx2(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
        require_avx2();
        bf16_splat_lhs_avx2<ElementwiseOp::Sub>(out, a, s, begin, end);
    }

    /**
     * Per-lane widen/mul/narrow with the scalar on the LEFT over [begin, end), 8
     * lanes per step; the tail runs through the exact scalar widen/apply/narrow pathway.
     * @brief AVX2 left splat mul (bf16 flat lanes).
     * @param out Output lane buffer.
     * @param a Input lane buffer.
     * @param s Scalar lane bit pattern (already narrowed into bf16).
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void mul_bf16_splat_lhs_avx2(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
        require_avx2();
        bf16_splat_lhs_avx2<ElementwiseOp::Mul>(out, a, s, begin, end);
    }

    /**
     * Per-lane widen/div/narrow with the scalar on the LEFT over [begin, end), 8
     * lanes per step; the tail runs through the exact scalar widen/apply/narrow pathway.
     * @brief AVX2 left splat div (bf16 flat lanes).
     * @param out Output lane buffer.
     * @param a Input lane buffer.
     * @param s Scalar lane bit pattern (already narrowed into bf16).
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void div_bf16_splat_lhs_avx2(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
        require_avx2();
        bf16_splat_lhs_avx2<ElementwiseOp::Div>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = (s - a[i]) mod 2^8 over [begin, end), 32 lanes per step.
     * @brief AVX2 left splat sub (8-bit lanes).
     * @param out Output buffer.
     * @param a Input buffer.
     * @param s Scalar operand (already narrowed into the 8-bit dtype).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_i8_splat_lhs_avx2(int8_t* out, const int8_t* a, const int8_t s, const size_t begin, const size_t end) {
        require_avx2();
        int_splat_lhs_sub_avx2<int8_t>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = (s - a[i]) mod 2^16 over [begin, end), 16 lanes per step.
     * @brief AVX2 left splat sub (16-bit lanes).
     * @param out Output buffer.
     * @param a Input buffer.
     * @param s Scalar operand (already narrowed into the 16-bit dtype).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_i16_splat_lhs_avx2(int16_t* out, const int16_t* a, const int16_t s, const size_t begin, const size_t end) {
        require_avx2();
        int_splat_lhs_sub_avx2<int16_t>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = (s - a[i]) mod 2^32 over [begin, end), 8 lanes per step.
     * @brief AVX2 left splat sub (32-bit lanes).
     * @param out Output buffer.
     * @param a Input buffer.
     * @param s Scalar operand (already narrowed into the 32-bit dtype).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_i32_splat_lhs_avx2(int32_t* out, const int32_t* a, const int32_t s, const size_t begin, const size_t end) {
        require_avx2();
        int_splat_lhs_sub_avx2<int32_t>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = (s - a[i]) mod 2^64 over [begin, end), 4 lanes per step.
     * @brief AVX2 left splat sub (64-bit lanes).
     * @param out Output buffer.
     * @param a Input buffer.
     * @param s Scalar operand (already narrowed into the 64-bit dtype).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_i64_splat_lhs_avx2(int64_t* out, const int64_t* a, const int64_t s, const size_t begin, const size_t end) {
        require_avx2();
        int_splat_lhs_sub_avx2<int64_t>(out, a, s, begin, end);
    }
}
