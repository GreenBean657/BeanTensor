/**
 * @file ElementwiseAVX512.cpp
 * @author GreenBean
 * @brief AVX512 kernels for binary elementwise arithmetic.
 * Compiled with -mavx512f -mavx512bw -mavx512bf16 -mavx512dq -mf16c;
 * execution is gated at runtime via elementwise_avx512_supported(). Every
 * kernel throws ErrorHandling::CPUIllegalInstruction when the CPU lacks the
 * feature, so a direct call is a valid feature probe (tests rely on this).
 * Bit-exactness contract: kernel output equals the scalar applies in
 * ElementwiseCPU.cpp (apply_native/apply_half/apply_packed/apply_int) on every
 * input, including NaN/Inf/subnormal sentinels. fp16 narrows through the
 * hardware converter with a vectorized replay of the scalar overflow clamp
 * (finite |x| > 65504 -> +/-65504, which hardware RNE would turn into Inf);
 * bf16 narrows through _mm512_cvtneps_pbh, whose NaN quieting, fp32-subnormal
 * flush-to-zero and RNE match the scalar narrow bit-for-bit (same contract as
 * the Intrinsics AVX512 conversion kernels). Phase 2 adds the tensor-scalar
 * splat kernels (scalar pre-narrowed by the caller, broadcast into a vector
 * register, masked tails like the binary bodies) and the per-nibble Int4x2
 * wrap kernels (unsigned-nibble byte-lane arithmetic masked to 4 bits
 * reproduces the scalar signed wrap bit-for-bit).
 * @version 0.2
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
         * Compile-time map from the elementwise op selector to the fp32 vector instruction.
         * @brief fp32 vector op.
         */
        template<ElementwiseOp Op>
        [[nodiscard]] __m512 f32_vec_op(const __m512 x, const __m512 y) {
            if constexpr (Op == ElementwiseOp::Add) {
                return _mm512_add_ps(x, y);
            } else if constexpr (Op == ElementwiseOp::Sub) {
                return _mm512_sub_ps(x, y);
            } else if constexpr (Op == ElementwiseOp::Mul) {
                return _mm512_mul_ps(x, y);
            } else {
                return _mm512_div_ps(x, y);
            }
        }

        /**
         * Compile-time map from the elementwise op selector to the fp64 vector instruction.
         * @brief fp64 vector op.
         */
        template<ElementwiseOp Op>
        [[nodiscard]] __m512d f64_vec_op(const __m512d x, const __m512d y) {
            if constexpr (Op == ElementwiseOp::Add) {
                return _mm512_add_pd(x, y);
            } else if constexpr (Op == ElementwiseOp::Sub) {
                return _mm512_sub_pd(x, y);
            } else if constexpr (Op == ElementwiseOp::Mul) {
                return _mm512_mul_pd(x, y);
            } else {
                return _mm512_div_pd(x, y);
            }
        }

        /**
         * Compile-time map from the elementwise op selector to the epi8 vector instruction.
         * @brief 8-bit lane vector add/sub.
         */
        template<ElementwiseOp Op>
        [[nodiscard]] __m512i i8_vec_addsub(const __m512i x, const __m512i y) {
            if constexpr (Op == ElementwiseOp::Add) {
                return _mm512_add_epi8(x, y);
            } else {
                return _mm512_sub_epi8(x, y);
            }
        }

        /**
         * No epi8 multiply exists: widen both operands to 16-bit lanes, take the low
         * product byte (vpmovwb truncation — no saturation). Zero-extension makes the
         * low byte sign-agnostic, so one kernel serves int8 and uint8 (wrap semantics).
         * @brief 8-bit lane vector mul (via 16-bit widening).
         */
        [[nodiscard]] __m512i i8_vec_mul(const __m512i x, const __m512i y) {
            const auto xw = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(x));
            const auto yw = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(y));
            const auto xw_hi = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(x, 1));
            const auto yw_hi = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(y, 1));
            const auto lo = _mm512_and_si512(_mm512_mullo_epi16(xw, yw), _mm512_set1_epi16(0x00FF));
            const auto hi = _mm512_and_si512(_mm512_mullo_epi16(xw_hi, yw_hi), _mm512_set1_epi16(0x00FF));
            const auto pairs = _mm512_packus_epi16(lo, hi);
            // packus is lane-local (4x128): qwords are [lo0 lo1 hi0 hi1 lo2 lo3 hi2 hi3];
            // restore the linear order [lo0 lo1 lo2 lo3 hi0 hi1 hi2 hi3].
            return _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), pairs);
        }

        /**
         * Compile-time map from the elementwise op selector to the epi16 vector instruction.
         * @brief 16-bit lane vector add/sub/mul.
         */
        template<ElementwiseOp Op>
        [[nodiscard]] __m512i i16_vec_op(const __m512i x, const __m512i y) {
            if constexpr (Op == ElementwiseOp::Add) {
                return _mm512_add_epi16(x, y);
            } else if constexpr (Op == ElementwiseOp::Sub) {
                return _mm512_sub_epi16(x, y);
            } else {
                return _mm512_mullo_epi16(x, y);
            }
        }

        /**
         * Compile-time map from the elementwise op selector to the epi32 vector instruction.
         * @brief 32-bit lane vector add/sub/mul.
         */
        template<ElementwiseOp Op>
        [[nodiscard]] __m512i i32_vec_op(const __m512i x, const __m512i y) {
            if constexpr (Op == ElementwiseOp::Add) {
                return _mm512_add_epi32(x, y);
            } else if constexpr (Op == ElementwiseOp::Sub) {
                return _mm512_sub_epi32(x, y);
            } else {
                return _mm512_mullo_epi32(x, y);
            }
        }

        /**
         * Compile-time map from the elementwise op selector to the epi64 vector instruction.
         * @brief 64-bit lane vector add/sub/mul (mullo under -mavx512dq).
         */
        template<ElementwiseOp Op>
        [[nodiscard]] __m512i i64_vec_op(const __m512i x, const __m512i y) {
            if constexpr (Op == ElementwiseOp::Add) {
                return _mm512_add_epi64(x, y);
            } else if constexpr (Op == ElementwiseOp::Sub) {
                return _mm512_sub_epi64(x, y);
            } else {
                return _mm512_mullo_epi64(x, y);
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
        [[nodiscard]] __m512 f16_clamp_finite_overflow(const __m512 r) {
            const auto abs = _mm512_and_ps(r, _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF)));
            const auto too_big = _mm512_cmp_ps_mask(abs, _mm512_set1_ps(DataTypes::detail::FP16_MAX), _CMP_GT_OQ);
            const auto finite = _mm512_cmp_ps_mask(abs, _mm512_set1_ps(__builtin_inff()), _CMP_LT_OQ);
            const auto clamp_lanes = static_cast<__mmask16>(too_big & finite);
            // 0x477FE000 == 65504.0f, OR'd with the source sign bit.
            const auto sign = _mm512_and_si512(_mm512_castps_si512(r), _mm512_set1_epi32(static_cast<int32_t>(0x80000000u)));
            const auto clamped = _mm512_castsi512_ps(_mm512_or_si512(sign, _mm512_set1_epi32(0x477FE000)));
            return _mm512_mask_blend_ps(clamp_lanes, r, clamped);
        }

        /**
         * Hardware VCVTPS2PH (zmm) with RNE + no exceptions, matching the scalar
         * narrow everywhere except the finite-overflow clamp applied beforehand.
         * @brief Narrow 16 fp32 lanes to 16 fp16 lanes (bit pattern vector).
         */
        [[nodiscard]] __m256i f16_narrow16(const __m512 w) {
            const auto fixed = f16_clamp_finite_overflow(w);
#pragma GCC diagnostic push // Snapshot GCC properties.
#pragma GCC diagnostic ignored "-Wsign-conversion" // Modify flag: the avx512fintrin.h _mm512_cvtps_ph macro passes a bare int -1 where __mmask16 belongs (same header quirk as the Intrinsics AVX512 conversion file).
            return _mm512_cvtps_ph(
                fixed,
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC
                );
#pragma GCC diagnostic pop // Revert to pre-recorded GCC state (via push).
        }

        // --------------------------- scalar tails (bit-exact with the scalar applies) ---------------------------

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
            float32_t w;
            if constexpr (Op == ElementwiseOp::Add) {
                w = wx + wy;
            } else if constexpr (Op == ElementwiseOp::Sub) {
                w = wx - wy;
            } else if constexpr (Op == ElementwiseOp::Mul) {
                w = wx * wy;
            } else {
                w = wx / wy;
            }
            return Intrinsics::detail::bit_cast_memcpy<uint16_t>(
                Intrinsics::detail::unaccel_fp32_to_fp16_untracked(w));
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
            float32_t w;
            if constexpr (Op == ElementwiseOp::Add) {
                w = wx + wy;
            } else if constexpr (Op == ElementwiseOp::Sub) {
                w = wx - wy;
            } else if constexpr (Op == ElementwiseOp::Mul) {
                w = wx * wy;
            } else {
                w = wx / wy;
            }
            return Intrinsics::detail::bit_cast_memcpy<uint16_t>(
                Intrinsics::detail::unaccel_fp32_to_bf16_untracked(w));
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
         * 16 lanes per step; the tail runs under a lane mask. IEEE single-rounding op,
         * bit-identical to the scalar apply_native.
         * @brief AVX512 range kernel body (fp32).
         */
        template<ElementwiseOp Op>
        void f32_range_avx512(float32_t* out, const float32_t* a, const float32_t* b, const size_t begin, const size_t end) {
            auto i = begin;
            for (; i + 16 <= end; i += 16) {
                _mm512_storeu_ps(out + i, f32_vec_op<Op>(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
            }
            if (i < end) {
                const auto mask = static_cast<__mmask16>((1u << (end - i)) - 1u);
                _mm512_mask_storeu_ps(out + i, mask,
                                      f32_vec_op<Op>(_mm512_maskz_loadu_ps(mask, a + i), _mm512_maskz_loadu_ps(mask, b + i)));
            }
        }

        /**
         * 8 lanes per step; the tail runs under a lane mask. IEEE single-rounding op,
         * bit-identical to the scalar apply_native.
         * @brief AVX512 range kernel body (fp64).
         */
        template<ElementwiseOp Op>
        void f64_range_avx512(float64_t* out, const float64_t* a, const float64_t* b, const size_t begin, const size_t end) {
            auto i = begin;
            for (; i + 8 <= end; i += 8) {
                _mm512_storeu_pd(out + i, f64_vec_op<Op>(_mm512_loadu_pd(a + i), _mm512_loadu_pd(b + i)));
            }
            if (i < end) {
                const auto mask = static_cast<__mmask8>((1u << (end - i)) - 1u);
                _mm512_mask_storeu_pd(out + i, mask,
                                      f64_vec_op<Op>(_mm512_maskz_loadu_pd(mask, a + i), _mm512_maskz_loadu_pd(mask, b + i)));
            }
        }

        /**
         * 16 lanes per step. The tail loads/stores through 512-bit masked 16-bit
         * lane ops (AVX512BW — no AVX512VL needed) and computes the low 16 lanes.
         * @brief AVX512 range kernel body (fp16 flat lanes).
         */
        template<ElementwiseOp Op>
        void f16_range_avx512(uint16_t* out, const uint16_t* a, const uint16_t* b, const size_t begin, const size_t end) {
            auto i = begin;
            for (; i + 16 <= end; i += 16) {
                const auto wa = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i)));
                const auto wb = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i)));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), f16_narrow16(f32_vec_op<Op>(wa, wb)));
            }
            if (i < end) {
                const auto mask = static_cast<__mmask32>((1u << (end - i)) - 1u);
                const auto va = _mm512_maskz_loadu_epi16(mask, a + i);
                const auto vb = _mm512_maskz_loadu_epi16(mask, b + i);
                const auto wa = _mm512_cvtph_ps(_mm512_castsi512_si256(va));
                const auto wb = _mm512_cvtph_ps(_mm512_castsi512_si256(vb));
                const auto r = f16_narrow16(f32_vec_op<Op>(wa, wb));
                _mm512_mask_storeu_epi16(out + i, mask, _mm512_castsi256_si512(r));
            }
        }

        /**
         * 16 lanes per step; the tail runs under a 512-bit masked 16-bit lane
         * load/store (AVX512BW). Narrow is the hardware cvtneps2bf16, which matches
         * the scalar NaN quieting / fp32-subnormal flush / RNE bit-for-bit.
         * @brief AVX512 range kernel body (bf16 flat lanes).
         */
        template<ElementwiseOp Op>
        void bf16_range_avx512(uint16_t* out, const uint16_t* a, const uint16_t* b, const size_t begin, const size_t end) {
            auto i = begin;
            for (; i + 16 <= end; i += 16) {
                const auto wa = _mm512_castsi512_ps(_mm512_slli_epi32(
                    _mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i))), 16));
                const auto wb = _mm512_castsi512_ps(_mm512_slli_epi32(
                    _mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i))), 16));
                const auto r = reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(f32_vec_op<Op>(wa, wb)));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), r);
            }
            if (i < end) {
                const auto mask = static_cast<__mmask32>((1u << (end - i)) - 1u);
                const auto va = _mm512_maskz_loadu_epi16(mask, a + i);
                const auto vb = _mm512_maskz_loadu_epi16(mask, b + i);
                const auto wa = _mm512_castsi512_ps(_mm512_slli_epi32(
                    _mm512_cvtepu16_epi32(_mm512_castsi512_si256(va)), 16));
                const auto wb = _mm512_castsi512_ps(_mm512_slli_epi32(
                    _mm512_cvtepu16_epi32(_mm512_castsi512_si256(vb)), 16));
                const auto r = reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(f32_vec_op<Op>(wa, wb)));
                _mm512_mask_storeu_epi16(out + i, mask, _mm512_castsi256_si512(r));
            }
        }

        /**
         * 64 lanes per step; the tail runs under a lane mask. Wrap arithmetic,
         * bit-identical to the unsigned-twin arithmetic of apply_int.
         * @brief AVX512 range kernel body (8-bit lanes).
         */
        template<ElementwiseOp Op>
        void i8_range_avx512(int8_t* out, const int8_t* a, const int8_t* b, const size_t begin, const size_t end) {
            auto i = begin;
            for (; i + 64 <= end; i += 64) {
                const auto x = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a + i));
                const auto y = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b + i));
                __m512i r;
                if constexpr (Op == ElementwiseOp::Mul) {
                    r = i8_vec_mul(x, y);
                } else {
                    r = i8_vec_addsub<Op>(x, y);
                }
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(out + i), r);
            }
            if (i < end) {
                const auto mask = static_cast<__mmask64>((1ull << (end - i)) - 1ull);
                const auto x = _mm512_maskz_loadu_epi8(mask, a + i);
                const auto y = _mm512_maskz_loadu_epi8(mask, b + i);
                __m512i r;
                if constexpr (Op == ElementwiseOp::Mul) {
                    r = i8_vec_mul(x, y);
                } else {
                    r = i8_vec_addsub<Op>(x, y);
                }
                _mm512_mask_storeu_epi8(out + i, mask, r);
            }
        }

        /**
         * 32 lanes per step; the tail runs under a lane mask. Wrap arithmetic,
         * bit-identical to the unsigned-twin arithmetic of apply_int.
         * @brief AVX512 range kernel body (16-bit lanes).
         */
        template<ElementwiseOp Op>
        void i16_range_avx512(int16_t* out, const int16_t* a, const int16_t* b, const size_t begin, const size_t end) {
            auto i = begin;
            for (; i + 32 <= end; i += 32) {
                const auto x = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a + i));
                const auto y = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b + i));
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(out + i), i16_vec_op<Op>(x, y));
            }
            if (i < end) {
                const auto mask = static_cast<__mmask32>((1u << (end - i)) - 1u);
                const auto x = _mm512_maskz_loadu_epi16(mask, a + i);
                const auto y = _mm512_maskz_loadu_epi16(mask, b + i);
                _mm512_mask_storeu_epi16(out + i, mask, i16_vec_op<Op>(x, y));
            }
        }

        /**
         * 16 lanes per step; the tail runs under a lane mask. Wrap arithmetic,
         * bit-identical to the unsigned-twin arithmetic of apply_int.
         * @brief AVX512 range kernel body (32-bit lanes).
         */
        template<ElementwiseOp Op>
        void i32_range_avx512(int32_t* out, const int32_t* a, const int32_t* b, const size_t begin, const size_t end) {
            auto i = begin;
            for (; i + 16 <= end; i += 16) {
                const auto x = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a + i));
                const auto y = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b + i));
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(out + i), i32_vec_op<Op>(x, y));
            }
            if (i < end) {
                const auto mask = static_cast<__mmask16>((1u << (end - i)) - 1u);
                const auto x = _mm512_maskz_loadu_epi32(mask, a + i);
                const auto y = _mm512_maskz_loadu_epi32(mask, b + i);
                _mm512_mask_storeu_epi32(out + i, mask, i32_vec_op<Op>(x, y));
            }
        }

        /**
         * 8 lanes per step; the tail runs under a lane mask. Wrap arithmetic,
         * bit-identical to the unsigned-twin arithmetic of apply_int.
         * @brief AVX512 range kernel body (64-bit lanes).
         */
        template<ElementwiseOp Op>
        void i64_range_avx512(int64_t* out, const int64_t* a, const int64_t* b, const size_t begin, const size_t end) {
            auto i = begin;
            for (; i + 8 <= end; i += 8) {
                const auto x = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a + i));
                const auto y = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b + i));
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(out + i), i64_vec_op<Op>(x, y));
            }
            if (i < end) {
                const auto mask = static_cast<__mmask8>((1u << (end - i)) - 1u);
                const auto x = _mm512_maskz_loadu_epi64(mask, a + i);
                const auto y = _mm512_maskz_loadu_epi64(mask, b + i);
                _mm512_mask_storeu_epi64(out + i, mask, i64_vec_op<Op>(x, y));
            }
        }

        /**
         * Every public kernel refuses to execute on a CPU lacking the AVX512 flag
         * union — this both gates dispatch and gives direct callers (tests) a probe.
         * @brief Feature gate check for the AVX512 kernels.
         */
        void require_avx512() {
            if (!elementwise_avx512_supported()) {
                throw ErrorHandling::CPUIllegalInstruction("AVX512F+AVX512BW+AVX512BF16+AVX512DQ+F16C");
            }
        }

        // --------------------------- splat kernel bodies ---------------------------
        // Same loop shapes as the binary bodies above (masked tails included); the
        // scalar operand is a broadcast vector built once. The scalar arrives
        // ALREADY narrowed into the element type — the bit-exactness contract is
        // against the scalar apply with that narrowed scalar.

        /**
         * 16 lanes per step; the tail runs under a lane mask. IEEE single-rounding
         * op against s, bit-identical to the scalar apply_native.
         * @brief AVX512 splat kernel body (fp32).
         */
        template<ElementwiseOp Op>
        void f32_splat_avx512(float32_t* out, const float32_t* a, const float32_t s, const size_t begin, const size_t end) {
            const auto vs = _mm512_set1_ps(s);
            auto i = begin;
            for (; i + 16 <= end; i += 16) {
                _mm512_storeu_ps(out + i, f32_vec_op<Op>(_mm512_loadu_ps(a + i), vs));
            }
            if (i < end) {
                const auto mask = static_cast<__mmask16>((1u << (end - i)) - 1u);
                _mm512_mask_storeu_ps(out + i, mask,
                                      f32_vec_op<Op>(_mm512_maskz_loadu_ps(mask, a + i), vs));
            }
        }

        /**
         * 8 lanes per step; the tail runs under a lane mask. IEEE single-rounding
         * op against s, bit-identical to the scalar apply_native.
         * @brief AVX512 splat kernel body (fp64).
         */
        template<ElementwiseOp Op>
        void f64_splat_avx512(float64_t* out, const float64_t* a, const float64_t s, const size_t begin, const size_t end) {
            const auto vs = _mm512_set1_pd(s);
            auto i = begin;
            for (; i + 8 <= end; i += 8) {
                _mm512_storeu_pd(out + i, f64_vec_op<Op>(_mm512_loadu_pd(a + i), vs));
            }
            if (i < end) {
                const auto mask = static_cast<__mmask8>((1u << (end - i)) - 1u);
                _mm512_mask_storeu_pd(out + i, mask,
                                      f64_vec_op<Op>(_mm512_maskz_loadu_pd(mask, a + i), vs));
            }
        }

        /**
         * 16 lanes per step; the tail loads/stores through 512-bit masked 16-bit
         * lane ops (AVX512BW) and computes the low 16 lanes against the splat scalar.
         * @brief AVX512 splat kernel body (fp16 flat lanes).
         */
        template<ElementwiseOp Op>
        void f16_splat_avx512(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
            const auto ws = _mm512_set1_ps(
                Intrinsics::detail::unaccel_fp16_to_fp32_untracked(
                    Intrinsics::detail::bit_cast_memcpy<float16_t>(s)));
            auto i = begin;
            for (; i + 16 <= end; i += 16) {
                const auto wa = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i)));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), f16_narrow16(f32_vec_op<Op>(wa, ws)));
            }
            if (i < end) {
                const auto mask = static_cast<__mmask32>((1u << (end - i)) - 1u);
                const auto va = _mm512_maskz_loadu_epi16(mask, a + i);
                const auto wa = _mm512_cvtph_ps(_mm512_castsi512_si256(va));
                const auto r = f16_narrow16(f32_vec_op<Op>(wa, ws));
                _mm512_mask_storeu_epi16(out + i, mask, _mm512_castsi256_si512(r));
            }
        }

        /**
         * 16 lanes per step; the tail runs under a 512-bit masked 16-bit lane
         * load/store (AVX512BW). Narrow is hardware cvtneps2bf16 against the splat
         * widened scalar, matching the scalar NaN quieting / fp32-subnormal flush / RNE.
         * @brief AVX512 splat kernel body (bf16 flat lanes).
         */
        template<ElementwiseOp Op>
        void bf16_splat_avx512(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
            const auto ws = _mm512_set1_ps(
                Intrinsics::detail::unaccel_bf16_to_fp32(
                    Intrinsics::detail::bit_cast_memcpy<bfloat16_t>(s)));
            auto i = begin;
            for (; i + 16 <= end; i += 16) {
                const auto wa = _mm512_castsi512_ps(_mm512_slli_epi32(
                    _mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i))), 16));
                const auto r = reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(f32_vec_op<Op>(wa, ws)));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), r);
            }
            if (i < end) {
                const auto mask = static_cast<__mmask32>((1u << (end - i)) - 1u);
                const auto va = _mm512_maskz_loadu_epi16(mask, a + i);
                const auto wa = _mm512_castsi512_ps(_mm512_slli_epi32(
                    _mm512_cvtepu16_epi32(_mm512_castsi512_si256(va)), 16));
                const auto r = reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(f32_vec_op<Op>(wa, ws)));
                _mm512_mask_storeu_epi16(out + i, mask, _mm512_castsi256_si512(r));
            }
        }

        /**
         * 64 lanes per step; the tail runs under a lane mask. Wrap arithmetic
         * against s, bit-identical to the unsigned-twin arithmetic of apply_int.
         * @brief AVX512 splat kernel body (8-bit lanes).
         */
        template<ElementwiseOp Op>
        void i8_splat_avx512(int8_t* out, const int8_t* a, const int8_t s, const size_t begin, const size_t end) {
            const auto vs = _mm512_set1_epi8(s);
            auto i = begin;
            for (; i + 64 <= end; i += 64) {
                const auto x = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a + i));
                __m512i r;
                if constexpr (Op == ElementwiseOp::Mul) {
                    r = i8_vec_mul(x, vs);
                } else {
                    r = i8_vec_addsub<Op>(x, vs);
                }
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(out + i), r);
            }
            if (i < end) {
                const auto mask = static_cast<__mmask64>((1ull << (end - i)) - 1ull);
                const auto x = _mm512_maskz_loadu_epi8(mask, a + i);
                __m512i r;
                if constexpr (Op == ElementwiseOp::Mul) {
                    r = i8_vec_mul(x, vs);
                } else {
                    r = i8_vec_addsub<Op>(x, vs);
                }
                _mm512_mask_storeu_epi8(out + i, mask, r);
            }
        }

        /**
         * 32 lanes per step; the tail runs under a lane mask. Wrap arithmetic
         * against s, bit-identical to the unsigned-twin arithmetic of apply_int.
         * @brief AVX512 splat kernel body (16-bit lanes).
         */
        template<ElementwiseOp Op>
        void i16_splat_avx512(int16_t* out, const int16_t* a, const int16_t s, const size_t begin, const size_t end) {
            const auto vs = _mm512_set1_epi16(s);
            auto i = begin;
            for (; i + 32 <= end; i += 32) {
                const auto x = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a + i));
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(out + i), i16_vec_op<Op>(x, vs));
            }
            if (i < end) {
                const auto mask = static_cast<__mmask32>((1u << (end - i)) - 1u);
                const auto x = _mm512_maskz_loadu_epi16(mask, a + i);
                _mm512_mask_storeu_epi16(out + i, mask, i16_vec_op<Op>(x, vs));
            }
        }

        /**
         * 16 lanes per step; the tail runs under a lane mask. Wrap arithmetic
         * against s, bit-identical to the unsigned-twin arithmetic of apply_int.
         * @brief AVX512 splat kernel body (32-bit lanes).
         */
        template<ElementwiseOp Op>
        void i32_splat_avx512(int32_t* out, const int32_t* a, const int32_t s, const size_t begin, const size_t end) {
            const auto vs = _mm512_set1_epi32(s);
            auto i = begin;
            for (; i + 16 <= end; i += 16) {
                const auto x = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a + i));
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(out + i), i32_vec_op<Op>(x, vs));
            }
            if (i < end) {
                const auto mask = static_cast<__mmask16>((1u << (end - i)) - 1u);
                const auto x = _mm512_maskz_loadu_epi32(mask, a + i);
                _mm512_mask_storeu_epi32(out + i, mask, i32_vec_op<Op>(x, vs));
            }
        }

        /**
         * 8 lanes per step; the tail runs under a lane mask. Wrap arithmetic
         * against s, bit-identical to the unsigned-twin arithmetic of apply_int.
         * @brief AVX512 splat kernel body (64-bit lanes).
         */
        template<ElementwiseOp Op>
        void i64_splat_avx512(int64_t* out, const int64_t* a, const int64_t s, const size_t begin, const size_t end) {
            const auto vs = _mm512_set1_epi64(s);
            auto i = begin;
            for (; i + 8 <= end; i += 8) {
                const auto x = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a + i));
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(out + i), i64_vec_op<Op>(x, vs));
            }
            if (i < end) {
                const auto mask = static_cast<__mmask8>((1u << (end - i)) - 1u);
                const auto x = _mm512_maskz_loadu_epi64(mask, a + i);
                _mm512_mask_storeu_epi64(out + i, mask, i64_vec_op<Op>(x, vs));
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
         * @brief int4x2 vector op (one 512-bit block of 64 packed bytes).
         */
        template<ElementwiseOp Op>
        [[nodiscard]] __m512i i4x2_vec_op(const __m512i x, const __m512i y) {
            const auto low_mask = _mm512_set1_epi8(0x0F);
            const auto xlo = _mm512_and_si512(x, low_mask);
            const auto ylo = _mm512_and_si512(y, low_mask);
            const auto xhi = _mm512_and_si512(_mm512_srli_epi16(x, 4), low_mask);
            const auto yhi = _mm512_and_si512(_mm512_srli_epi16(y, 4), low_mask);
            __m512i rlo;
            __m512i rhi;
            if constexpr (Op == ElementwiseOp::Mul) {
                // No epi8 multiply: 16-bit widening like i8_vec_mul. Nibbles are
                // zero-extended, so the low byte holds the product; masking 0x0F
                // then keeps (a*b) mod 16, and 256 == 0 (mod 16) makes the low
                // byte of the product sufficient.
                rlo = _mm512_and_si512(i8_vec_mul(xlo, ylo), low_mask);
                rhi = _mm512_and_si512(i8_vec_mul(xhi, yhi), low_mask);
            } else {
                rlo = _mm512_and_si512(i8_vec_addsub<Op>(xlo, ylo), low_mask);
                rhi = _mm512_and_si512(i8_vec_addsub<Op>(xhi, yhi), low_mask);
            }
            return _mm512_or_si512(rlo, _mm512_slli_epi16(rhi, 4));
        }

        /**
         * 64 packed bytes per step; the tail runs under a byte-lane mask (the masked
         * lanes compute garbage that the masked store discards). Per-nibble wrap,
         * the exact unsigned-nibble identity of apply_int4x2.
         * @brief AVX512 range kernel body (int4x2).
         */
        template<ElementwiseOp Op>
        void i4x2_range_avx512(uint8_t* out, const uint8_t* a, const uint8_t* b, const size_t begin, const size_t end) {
            auto i = begin;
            for (; i + 64 <= end; i += 64) {
                const auto x = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a + i));
                const auto y = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b + i));
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(out + i), i4x2_vec_op<Op>(x, y));
            }
            if (i < end) {
                const auto mask = static_cast<__mmask64>((1ull << (end - i)) - 1ull);
                const auto x = _mm512_maskz_loadu_epi8(mask, a + i);
                const auto y = _mm512_maskz_loadu_epi8(mask, b + i);
                _mm512_mask_storeu_epi8(out + i, mask, i4x2_vec_op<Op>(x, y));
            }
        }

        // --------------------------- left-splat kernel bodies ---------------------------
        // Broadcast rows with operand A on inner stride 0 compute s (op) a[i] — the
        // scalar on the LEFT, the exact operand order of the scalar apply (matters
        // for sub/div, and selects the same NaN payload for commutative fp ops).

        /**
         * 16 lanes per step; the tail runs under a lane mask with the scalar on the left.
         * @brief AVX512 left-splat kernel body (fp32).
         */
        template<ElementwiseOp Op>
        void f32_splat_lhs_avx512(float32_t* out, const float32_t* a, const float32_t s, const size_t begin, const size_t end) {
            const auto vs = _mm512_set1_ps(s);
            auto i = begin;
            for (; i + 16 <= end; i += 16) {
                _mm512_storeu_ps(out + i, f32_vec_op<Op>(vs, _mm512_loadu_ps(a + i)));
            }
            if (i < end) {
                const auto mask = static_cast<__mmask16>((1u << (end - i)) - 1u);
                _mm512_mask_storeu_ps(out + i, mask,
                                      f32_vec_op<Op>(vs, _mm512_maskz_loadu_ps(mask, a + i)));
            }
        }

        /**
         * 8 lanes per step; the tail runs under a lane mask with the scalar on the left.
         * @brief AVX512 left-splat kernel body (fp64).
         */
        template<ElementwiseOp Op>
        void f64_splat_lhs_avx512(float64_t* out, const float64_t* a, const float64_t s, const size_t begin, const size_t end) {
            const auto vs = _mm512_set1_pd(s);
            auto i = begin;
            for (; i + 8 <= end; i += 8) {
                _mm512_storeu_pd(out + i, f64_vec_op<Op>(vs, _mm512_loadu_pd(a + i)));
            }
            if (i < end) {
                const auto mask = static_cast<__mmask8>((1u << (end - i)) - 1u);
                _mm512_mask_storeu_pd(out + i, mask,
                                      f64_vec_op<Op>(vs, _mm512_maskz_loadu_pd(mask, a + i)));
            }
        }

        /**
         * 16 lanes per step; the tail loads/stores through 512-bit masked 16-bit
         * lane ops with the scalar on the left.
         * @brief AVX512 left-splat kernel body (fp16 flat lanes).
         */
        template<ElementwiseOp Op>
        void f16_splat_lhs_avx512(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
            const auto ws = _mm512_set1_ps(
                Intrinsics::detail::unaccel_fp16_to_fp32_untracked(
                    Intrinsics::detail::bit_cast_memcpy<float16_t>(s)));
            auto i = begin;
            for (; i + 16 <= end; i += 16) {
                const auto wa = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i)));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), f16_narrow16(f32_vec_op<Op>(ws, wa)));
            }
            if (i < end) {
                const auto mask = static_cast<__mmask32>((1u << (end - i)) - 1u);
                const auto va = _mm512_maskz_loadu_epi16(mask, a + i);
                const auto wa = _mm512_cvtph_ps(_mm512_castsi512_si256(va));
                const auto r = f16_narrow16(f32_vec_op<Op>(ws, wa));
                _mm512_mask_storeu_epi16(out + i, mask, _mm512_castsi256_si512(r));
            }
        }

        /**
         * 16 lanes per step; the tail runs under a 512-bit masked 16-bit lane
         * load/store with the scalar on the left. Narrow is hardware cvtneps2bf16.
         * @brief AVX512 left-splat kernel body (bf16 flat lanes).
         */
        template<ElementwiseOp Op>
        void bf16_splat_lhs_avx512(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
            const auto ws = _mm512_set1_ps(
                Intrinsics::detail::unaccel_bf16_to_fp32(
                    Intrinsics::detail::bit_cast_memcpy<bfloat16_t>(s)));
            auto i = begin;
            for (; i + 16 <= end; i += 16) {
                const auto wa = _mm512_castsi512_ps(_mm512_slli_epi32(
                    _mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i))), 16));
                const auto r = reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(f32_vec_op<Op>(ws, wa)));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), r);
            }
            if (i < end) {
                const auto mask = static_cast<__mmask32>((1u << (end - i)) - 1u);
                const auto va = _mm512_maskz_loadu_epi16(mask, a + i);
                const auto wa = _mm512_castsi512_ps(_mm512_slli_epi32(
                    _mm512_cvtepu16_epi32(_mm512_castsi512_si256(va)), 16));
                const auto r = reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(f32_vec_op<Op>(ws, wa)));
                _mm512_mask_storeu_epi16(out + i, mask, _mm512_castsi256_si512(r));
            }
        }

        /**
         * 64/32/16/8 lanes per step; the tail runs under a lane mask with the scalar
         * on the left. Sub only — wrap add/mul are bitwise commutative, so the plain
         * splats already match.
         * @brief AVX512 left-splat kernel body (8/16/32/64-bit lanes, sub).
         */
        template<typename T>
        void int_splat_lhs_sub_avx512(T* out, const T* a, const T s, const size_t begin, const size_t end) {
            auto i = begin;
            if constexpr (sizeof(T) == 1) {
                const auto vs = _mm512_set1_epi8(s);
                for (; i + 64 <= end; i += 64) {
                    const auto x = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a + i));
                    _mm512_storeu_si512(reinterpret_cast<__m512i*>(out + i), _mm512_sub_epi8(vs, x));
                }
                if (i < end) {
                    const auto mask = static_cast<__mmask64>((1ull << (end - i)) - 1ull);
                    const auto x = _mm512_maskz_loadu_epi8(mask, a + i);
                    _mm512_mask_storeu_epi8(out + i, mask, _mm512_sub_epi8(vs, x));
                }
            } else if constexpr (sizeof(T) == 2) {
                const auto vs = _mm512_set1_epi16(s);
                for (; i + 32 <= end; i += 32) {
                    const auto x = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a + i));
                    _mm512_storeu_si512(reinterpret_cast<__m512i*>(out + i), _mm512_sub_epi16(vs, x));
                }
                if (i < end) {
                    const auto mask = static_cast<__mmask32>((1u << (end - i)) - 1u);
                    const auto x = _mm512_maskz_loadu_epi16(mask, a + i);
                    _mm512_mask_storeu_epi16(out + i, mask, _mm512_sub_epi16(vs, x));
                }
            } else if constexpr (sizeof(T) == 4) {
                const auto vs = _mm512_set1_epi32(s);
                for (; i + 16 <= end; i += 16) {
                    const auto x = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a + i));
                    _mm512_storeu_si512(reinterpret_cast<__m512i*>(out + i), _mm512_sub_epi32(vs, x));
                }
                if (i < end) {
                    const auto mask = static_cast<__mmask16>((1u << (end - i)) - 1u);
                    const auto x = _mm512_maskz_loadu_epi32(mask, a + i);
                    _mm512_mask_storeu_epi32(out + i, mask, _mm512_sub_epi32(vs, x));
                }
            } else {
                const auto vs = _mm512_set1_epi64(s);
                for (; i + 8 <= end; i += 8) {
                    const auto x = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a + i));
                    _mm512_storeu_si512(reinterpret_cast<__m512i*>(out + i), _mm512_sub_epi64(vs, x));
                }
                if (i < end) {
                    const auto mask = static_cast<__mmask8>((1u << (end - i)) - 1u);
                    const auto x = _mm512_maskz_loadu_epi64(mask, a + i);
                    _mm512_mask_storeu_epi64(out + i, mask, _mm512_sub_epi64(vs, x));
                }
            }
        }
    }

    /**
     * Cached through IncludedCPUFeatures(); callers gate every AVX512 kernel on this.
     * @brief AVX512 flag-union verification, for elementwise kernels.
     * @return T/F - AVX512F/BW/BF16/DQ + F16C are supported by the CPU.
     */
    bool elementwise_avx512_supported() {
        const auto& f = IncludedCPUFeatures();
        return f.has_feature(CPUFeatureSetType_AVX512F) && f.has_feature(CPUFeatureSetType_AVX512BW)
            && f.has_feature(CPUFeatureSetType_AVX512BF16) && f.has_feature(CPUFeatureSetType_AVX512DQ)
            && f.has_feature(CPUFeatureSetType_F16C);
    }

    /**
     * Computes out[i] = a[i] + b[i] over [begin, end), 16 lanes per step.
     * @brief AVX512 range add (fp32).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void add_f32_range_avx512(float32_t* out, const float32_t* a, const float32_t* b, const size_t begin, const size_t end) {
        require_avx512();
        f32_range_avx512<ElementwiseOp::Add>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = a[i] - b[i] over [begin, end), 16 lanes per step.
     * @brief AVX512 range sub (fp32).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_f32_range_avx512(float32_t* out, const float32_t* a, const float32_t* b, const size_t begin, const size_t end) {
        require_avx512();
        f32_range_avx512<ElementwiseOp::Sub>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = a[i] * b[i] over [begin, end), 16 lanes per step.
     * @brief AVX512 range mul (fp32).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void mul_f32_range_avx512(float32_t* out, const float32_t* a, const float32_t* b, const size_t begin, const size_t end) {
        require_avx512();
        f32_range_avx512<ElementwiseOp::Mul>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = a[i] / b[i] over [begin, end), 16 lanes per step. IEEE div.
     * @brief AVX512 range div (fp32).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void div_f32_range_avx512(float32_t* out, const float32_t* a, const float32_t* b, const size_t begin, const size_t end) {
        require_avx512();
        f32_range_avx512<ElementwiseOp::Div>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = a[i] + b[i] over [begin, end), 8 lanes per step.
     * @brief AVX512 range add (fp64).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void add_f64_range_avx512(float64_t* out, const float64_t* a, const float64_t* b, const size_t begin, const size_t end) {
        require_avx512();
        f64_range_avx512<ElementwiseOp::Add>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = a[i] - b[i] over [begin, end), 8 lanes per step.
     * @brief AVX512 range sub (fp64).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_f64_range_avx512(float64_t* out, const float64_t* a, const float64_t* b, const size_t begin, const size_t end) {
        require_avx512();
        f64_range_avx512<ElementwiseOp::Sub>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = a[i] * b[i] over [begin, end), 8 lanes per step.
     * @brief AVX512 range mul (fp64).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void mul_f64_range_avx512(float64_t* out, const float64_t* a, const float64_t* b, const size_t begin, const size_t end) {
        require_avx512();
        f64_range_avx512<ElementwiseOp::Mul>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = a[i] / b[i] over [begin, end), 8 lanes per step. IEEE div.
     * @brief AVX512 range div (fp64).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void div_f64_range_avx512(float64_t* out, const float64_t* a, const float64_t* b, const size_t begin, const size_t end) {
        require_avx512();
        f64_range_avx512<ElementwiseOp::Div>(out, a, b, begin, end);
    }

    /**
     * Per-lane widen/add/narrow over [begin, end), 16 lanes per step; the tail runs
     * under a masked 16-bit lane load/store.
     * @brief AVX512 range add (fp16 flat lanes).
     * @param out Output lane buffer.
     * @param a Left operand lane buffer.
     * @param b Right operand lane buffer.
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void add_f16_range_avx512(uint16_t* out, const uint16_t* a, const uint16_t* b, const size_t begin, const size_t end) {
        require_avx512();
        f16_range_avx512<ElementwiseOp::Add>(out, a, b, begin, end);
    }

    /**
     * Per-lane widen/sub/narrow over [begin, end), 16 lanes per step; the tail runs
     * under a masked 16-bit lane load/store.
     * @brief AVX512 range sub (fp16 flat lanes).
     * @param out Output lane buffer.
     * @param a Left operand lane buffer.
     * @param b Right operand lane buffer.
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void sub_f16_range_avx512(uint16_t* out, const uint16_t* a, const uint16_t* b, const size_t begin, const size_t end) {
        require_avx512();
        f16_range_avx512<ElementwiseOp::Sub>(out, a, b, begin, end);
    }

    /**
     * Per-lane widen/mul/narrow over [begin, end), 16 lanes per step; the tail runs
     * under a masked 16-bit lane load/store.
     * @brief AVX512 range mul (fp16 flat lanes).
     * @param out Output lane buffer.
     * @param a Left operand lane buffer.
     * @param b Right operand lane buffer.
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void mul_f16_range_avx512(uint16_t* out, const uint16_t* a, const uint16_t* b, const size_t begin, const size_t end) {
        require_avx512();
        f16_range_avx512<ElementwiseOp::Mul>(out, a, b, begin, end);
    }

    /**
     * Per-lane widen/div/narrow over [begin, end), 16 lanes per step; the tail runs
     * under a masked 16-bit lane load/store.
     * @brief AVX512 range div (fp16 flat lanes).
     * @param out Output lane buffer.
     * @param a Left operand lane buffer.
     * @param b Right operand lane buffer.
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void div_f16_range_avx512(uint16_t* out, const uint16_t* a, const uint16_t* b, const size_t begin, const size_t end) {
        require_avx512();
        f16_range_avx512<ElementwiseOp::Div>(out, a, b, begin, end);
    }

    /**
     * Per-lane widen/add/narrow over [begin, end), 16 lanes per step; the tail runs
     * under a masked 16-bit lane load/store. Narrow is hardware cvtneps2bf16.
     * @brief AVX512 range add (bf16 flat lanes).
     * @param out Output lane buffer.
     * @param a Left operand lane buffer.
     * @param b Right operand lane buffer.
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void add_bf16_range_avx512(uint16_t* out, const uint16_t* a, const uint16_t* b, const size_t begin, const size_t end) {
        require_avx512();
        bf16_range_avx512<ElementwiseOp::Add>(out, a, b, begin, end);
    }

    /**
     * Per-lane widen/sub/narrow over [begin, end), 16 lanes per step; the tail runs
     * under a masked 16-bit lane load/store. Narrow is hardware cvtneps2bf16.
     * @brief AVX512 range sub (bf16 flat lanes).
     * @param out Output lane buffer.
     * @param a Left operand lane buffer.
     * @param b Right operand lane buffer.
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void sub_bf16_range_avx512(uint16_t* out, const uint16_t* a, const uint16_t* b, const size_t begin, const size_t end) {
        require_avx512();
        bf16_range_avx512<ElementwiseOp::Sub>(out, a, b, begin, end);
    }

    /**
     * Per-lane widen/mul/narrow over [begin, end), 16 lanes per step; the tail runs
     * under a masked 16-bit lane load/store. Narrow is hardware cvtneps2bf16.
     * @brief AVX512 range mul (bf16 flat lanes).
     * @param out Output lane buffer.
     * @param a Left operand lane buffer.
     * @param b Right operand lane buffer.
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void mul_bf16_range_avx512(uint16_t* out, const uint16_t* a, const uint16_t* b, const size_t begin, const size_t end) {
        require_avx512();
        bf16_range_avx512<ElementwiseOp::Mul>(out, a, b, begin, end);
    }

    /**
     * Per-lane widen/div/narrow over [begin, end), 16 lanes per step; the tail runs
     * under a masked 16-bit lane load/store. Narrow is hardware cvtneps2bf16.
     * @brief AVX512 range div (bf16 flat lanes).
     * @param out Output lane buffer.
     * @param a Left operand lane buffer.
     * @param b Right operand lane buffer.
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void div_bf16_range_avx512(uint16_t* out, const uint16_t* a, const uint16_t* b, const size_t begin, const size_t end) {
        require_avx512();
        bf16_range_avx512<ElementwiseOp::Div>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = (a[i] + b[i]) mod 2^8 over [begin, end), 64 lanes per step.
     * @brief AVX512 range add (8-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void add_i8_range_avx512(int8_t* out, const int8_t* a, const int8_t* b, const size_t begin, const size_t end) {
        require_avx512();
        i8_range_avx512<ElementwiseOp::Add>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = (a[i] - b[i]) mod 2^8 over [begin, end), 64 lanes per step.
     * @brief AVX512 range sub (8-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_i8_range_avx512(int8_t* out, const int8_t* a, const int8_t* b, const size_t begin, const size_t end) {
        require_avx512();
        i8_range_avx512<ElementwiseOp::Sub>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = (a[i] * b[i]) mod 2^8 over [begin, end), 64 lanes per step.
     * @brief AVX512 range mul (8-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void mul_i8_range_avx512(int8_t* out, const int8_t* a, const int8_t* b, const size_t begin, const size_t end) {
        require_avx512();
        i8_range_avx512<ElementwiseOp::Mul>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = (a[i] + b[i]) mod 2^16 over [begin, end), 32 lanes per step.
     * @brief AVX512 range add (16-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void add_i16_range_avx512(int16_t* out, const int16_t* a, const int16_t* b, const size_t begin, const size_t end) {
        require_avx512();
        i16_range_avx512<ElementwiseOp::Add>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = (a[i] - b[i]) mod 2^16 over [begin, end), 32 lanes per step.
     * @brief AVX512 range sub (16-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_i16_range_avx512(int16_t* out, const int16_t* a, const int16_t* b, const size_t begin, const size_t end) {
        require_avx512();
        i16_range_avx512<ElementwiseOp::Sub>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = (a[i] * b[i]) mod 2^16 over [begin, end), 32 lanes per step.
     * @brief AVX512 range mul (16-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void mul_i16_range_avx512(int16_t* out, const int16_t* a, const int16_t* b, const size_t begin, const size_t end) {
        require_avx512();
        i16_range_avx512<ElementwiseOp::Mul>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = (a[i] + b[i]) mod 2^32 over [begin, end), 16 lanes per step.
     * @brief AVX512 range add (32-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void add_i32_range_avx512(int32_t* out, const int32_t* a, const int32_t* b, const size_t begin, const size_t end) {
        require_avx512();
        i32_range_avx512<ElementwiseOp::Add>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = (a[i] - b[i]) mod 2^32 over [begin, end), 16 lanes per step.
     * @brief AVX512 range sub (32-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_i32_range_avx512(int32_t* out, const int32_t* a, const int32_t* b, const size_t begin, const size_t end) {
        require_avx512();
        i32_range_avx512<ElementwiseOp::Sub>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = (a[i] * b[i]) mod 2^32 over [begin, end), 16 lanes per step.
     * @brief AVX512 range mul (32-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void mul_i32_range_avx512(int32_t* out, const int32_t* a, const int32_t* b, const size_t begin, const size_t end) {
        require_avx512();
        i32_range_avx512<ElementwiseOp::Mul>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = (a[i] + b[i]) mod 2^64 over [begin, end), 8 lanes per step.
     * @brief AVX512 range add (64-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void add_i64_range_avx512(int64_t* out, const int64_t* a, const int64_t* b, const size_t begin, const size_t end) {
        require_avx512();
        i64_range_avx512<ElementwiseOp::Add>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = (a[i] - b[i]) mod 2^64 over [begin, end), 8 lanes per step.
     * @brief AVX512 range sub (64-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_i64_range_avx512(int64_t* out, const int64_t* a, const int64_t* b, const size_t begin, const size_t end) {
        require_avx512();
        i64_range_avx512<ElementwiseOp::Sub>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = (a[i] * b[i]) mod 2^64 over [begin, end), 8 lanes per step.
     * @brief AVX512 range mul (64-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void mul_i64_range_avx512(int64_t* out, const int64_t* a, const int64_t* b, const size_t begin, const size_t end) {
        require_avx512();
        i64_range_avx512<ElementwiseOp::Mul>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = a[i] + s over [begin, end), 16 lanes per step.
     * @brief AVX512 range splat add (fp32).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into fp32).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void add_f32_splat_avx512(float32_t* out, const float32_t* a, const float32_t s, const size_t begin, const size_t end) {
        require_avx512();
        f32_splat_avx512<ElementwiseOp::Add>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = a[i] - s over [begin, end), 16 lanes per step.
     * @brief AVX512 range splat sub (fp32).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into fp32).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_f32_splat_avx512(float32_t* out, const float32_t* a, const float32_t s, const size_t begin, const size_t end) {
        require_avx512();
        f32_splat_avx512<ElementwiseOp::Sub>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = a[i] * s over [begin, end), 16 lanes per step.
     * @brief AVX512 range splat mul (fp32).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into fp32).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void mul_f32_splat_avx512(float32_t* out, const float32_t* a, const float32_t s, const size_t begin, const size_t end) {
        require_avx512();
        f32_splat_avx512<ElementwiseOp::Mul>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = a[i] / s over [begin, end), 16 lanes per step. IEEE div.
     * @brief AVX512 range splat div (fp32).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into fp32).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void div_f32_splat_avx512(float32_t* out, const float32_t* a, const float32_t s, const size_t begin, const size_t end) {
        require_avx512();
        f32_splat_avx512<ElementwiseOp::Div>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = a[i] + s over [begin, end), 8 lanes per step.
     * @brief AVX512 range splat add (fp64).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into fp64).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void add_f64_splat_avx512(float64_t* out, const float64_t* a, const float64_t s, const size_t begin, const size_t end) {
        require_avx512();
        f64_splat_avx512<ElementwiseOp::Add>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = a[i] - s over [begin, end), 8 lanes per step.
     * @brief AVX512 range splat sub (fp64).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into fp64).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_f64_splat_avx512(float64_t* out, const float64_t* a, const float64_t s, const size_t begin, const size_t end) {
        require_avx512();
        f64_splat_avx512<ElementwiseOp::Sub>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = a[i] * s over [begin, end), 8 lanes per step.
     * @brief AVX512 range splat mul (fp64).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into fp64).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void mul_f64_splat_avx512(float64_t* out, const float64_t* a, const float64_t s, const size_t begin, const size_t end) {
        require_avx512();
        f64_splat_avx512<ElementwiseOp::Mul>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = a[i] / s over [begin, end), 8 lanes per step. IEEE div.
     * @brief AVX512 range splat div (fp64).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into fp64).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void div_f64_splat_avx512(float64_t* out, const float64_t* a, const float64_t s, const size_t begin, const size_t end) {
        require_avx512();
        f64_splat_avx512<ElementwiseOp::Div>(out, a, s, begin, end);
    }

    /**
     * Per-lane widen/add/narrow against the splat scalar over [begin, end), 16 lanes
     * per step; the tail runs under a masked 16-bit lane load/store.
     * @brief AVX512 range splat add (fp16 flat lanes).
     * @param out Output lane buffer.
     * @param a Left operand lane buffer.
     * @param s Scalar lane bit pattern (already narrowed into fp16).
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void add_f16_splat_avx512(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
        require_avx512();
        f16_splat_avx512<ElementwiseOp::Add>(out, a, s, begin, end);
    }

    /**
     * Per-lane widen/sub/narrow against the splat scalar over [begin, end), 16 lanes
     * per step; the tail runs under a masked 16-bit lane load/store.
     * @brief AVX512 range splat sub (fp16 flat lanes).
     * @param out Output lane buffer.
     * @param a Left operand lane buffer.
     * @param s Scalar lane bit pattern (already narrowed into fp16).
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void sub_f16_splat_avx512(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
        require_avx512();
        f16_splat_avx512<ElementwiseOp::Sub>(out, a, s, begin, end);
    }

    /**
     * Per-lane widen/mul/narrow against the splat scalar over [begin, end), 16 lanes
     * per step; the tail runs under a masked 16-bit lane load/store.
     * @brief AVX512 range splat mul (fp16 flat lanes).
     * @param out Output lane buffer.
     * @param a Left operand lane buffer.
     * @param s Scalar lane bit pattern (already narrowed into fp16).
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void mul_f16_splat_avx512(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
        require_avx512();
        f16_splat_avx512<ElementwiseOp::Mul>(out, a, s, begin, end);
    }

    /**
     * Per-lane widen/div/narrow against the splat scalar over [begin, end), 16 lanes
     * per step; the tail runs under a masked 16-bit lane load/store.
     * @brief AVX512 range splat div (fp16 flat lanes).
     * @param out Output lane buffer.
     * @param a Left operand lane buffer.
     * @param s Scalar lane bit pattern (already narrowed into fp16).
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void div_f16_splat_avx512(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
        require_avx512();
        f16_splat_avx512<ElementwiseOp::Div>(out, a, s, begin, end);
    }

    /**
     * Per-lane widen/add/narrow against the splat scalar over [begin, end), 16 lanes
     * per step; the tail runs under a masked 16-bit lane load/store. Narrow is
     * hardware cvtneps2bf16.
     * @brief AVX512 range splat add (bf16 flat lanes).
     * @param out Output lane buffer.
     * @param a Left operand lane buffer.
     * @param s Scalar lane bit pattern (already narrowed into bf16).
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void add_bf16_splat_avx512(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
        require_avx512();
        bf16_splat_avx512<ElementwiseOp::Add>(out, a, s, begin, end);
    }

    /**
     * Per-lane widen/sub/narrow against the splat scalar over [begin, end), 16 lanes
     * per step; the tail runs under a masked 16-bit lane load/store. Narrow is
     * hardware cvtneps2bf16.
     * @brief AVX512 range splat sub (bf16 flat lanes).
     * @param out Output lane buffer.
     * @param a Left operand lane buffer.
     * @param s Scalar lane bit pattern (already narrowed into bf16).
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void sub_bf16_splat_avx512(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
        require_avx512();
        bf16_splat_avx512<ElementwiseOp::Sub>(out, a, s, begin, end);
    }

    /**
     * Per-lane widen/mul/narrow against the splat scalar over [begin, end), 16 lanes
     * per step; the tail runs under a masked 16-bit lane load/store. Narrow is
     * hardware cvtneps2bf16.
     * @brief AVX512 range splat mul (bf16 flat lanes).
     * @param out Output lane buffer.
     * @param a Left operand lane buffer.
     * @param s Scalar lane bit pattern (already narrowed into bf16).
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void mul_bf16_splat_avx512(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
        require_avx512();
        bf16_splat_avx512<ElementwiseOp::Mul>(out, a, s, begin, end);
    }

    /**
     * Per-lane widen/div/narrow against the splat scalar over [begin, end), 16 lanes
     * per step; the tail runs under a masked 16-bit lane load/store. Narrow is
     * hardware cvtneps2bf16.
     * @brief AVX512 range splat div (bf16 flat lanes).
     * @param out Output lane buffer.
     * @param a Left operand lane buffer.
     * @param s Scalar lane bit pattern (already narrowed into bf16).
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void div_bf16_splat_avx512(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
        require_avx512();
        bf16_splat_avx512<ElementwiseOp::Div>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = (a[i] + s) mod 2^8 over [begin, end), 64 lanes per step.
     * @brief AVX512 range splat add (8-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into the 8-bit dtype).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void add_i8_splat_avx512(int8_t* out, const int8_t* a, const int8_t s, const size_t begin, const size_t end) {
        require_avx512();
        i8_splat_avx512<ElementwiseOp::Add>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = (a[i] - s) mod 2^8 over [begin, end), 64 lanes per step.
     * @brief AVX512 range splat sub (8-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into the 8-bit dtype).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_i8_splat_avx512(int8_t* out, const int8_t* a, const int8_t s, const size_t begin, const size_t end) {
        require_avx512();
        i8_splat_avx512<ElementwiseOp::Sub>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = (a[i] * s) mod 2^8 over [begin, end), 64 lanes per step.
     * @brief AVX512 range splat mul (8-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into the 8-bit dtype).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void mul_i8_splat_avx512(int8_t* out, const int8_t* a, const int8_t s, const size_t begin, const size_t end) {
        require_avx512();
        i8_splat_avx512<ElementwiseOp::Mul>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = (a[i] + s) mod 2^16 over [begin, end), 32 lanes per step.
     * @brief AVX512 range splat add (16-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into the 16-bit dtype).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void add_i16_splat_avx512(int16_t* out, const int16_t* a, const int16_t s, const size_t begin, const size_t end) {
        require_avx512();
        i16_splat_avx512<ElementwiseOp::Add>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = (a[i] - s) mod 2^16 over [begin, end), 32 lanes per step.
     * @brief AVX512 range splat sub (16-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into the 16-bit dtype).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_i16_splat_avx512(int16_t* out, const int16_t* a, const int16_t s, const size_t begin, const size_t end) {
        require_avx512();
        i16_splat_avx512<ElementwiseOp::Sub>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = (a[i] * s) mod 2^16 over [begin, end), 32 lanes per step.
     * @brief AVX512 range splat mul (16-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into the 16-bit dtype).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void mul_i16_splat_avx512(int16_t* out, const int16_t* a, const int16_t s, const size_t begin, const size_t end) {
        require_avx512();
        i16_splat_avx512<ElementwiseOp::Mul>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = (a[i] + s) mod 2^32 over [begin, end), 16 lanes per step.
     * @brief AVX512 range splat add (32-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into the 32-bit dtype).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void add_i32_splat_avx512(int32_t* out, const int32_t* a, const int32_t s, const size_t begin, const size_t end) {
        require_avx512();
        i32_splat_avx512<ElementwiseOp::Add>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = (a[i] - s) mod 2^32 over [begin, end), 16 lanes per step.
     * @brief AVX512 range splat sub (32-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into the 32-bit dtype).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_i32_splat_avx512(int32_t* out, const int32_t* a, const int32_t s, const size_t begin, const size_t end) {
        require_avx512();
        i32_splat_avx512<ElementwiseOp::Sub>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = (a[i] * s) mod 2^32 over [begin, end), 16 lanes per step.
     * @brief AVX512 range splat mul (32-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into the 32-bit dtype).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void mul_i32_splat_avx512(int32_t* out, const int32_t* a, const int32_t s, const size_t begin, const size_t end) {
        require_avx512();
        i32_splat_avx512<ElementwiseOp::Mul>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = (a[i] + s) mod 2^64 over [begin, end), 8 lanes per step.
     * @brief AVX512 range splat add (64-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into the 64-bit dtype).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void add_i64_splat_avx512(int64_t* out, const int64_t* a, const int64_t s, const size_t begin, const size_t end) {
        require_avx512();
        i64_splat_avx512<ElementwiseOp::Add>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = (a[i] - s) mod 2^64 over [begin, end), 8 lanes per step.
     * @brief AVX512 range splat sub (64-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into the 64-bit dtype).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_i64_splat_avx512(int64_t* out, const int64_t* a, const int64_t s, const size_t begin, const size_t end) {
        require_avx512();
        i64_splat_avx512<ElementwiseOp::Sub>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = (a[i] * s) mod 2^64 over [begin, end), 8 lanes per step.
     * @brief AVX512 range splat mul (64-bit lanes).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param s Scalar operand (already narrowed into the 64-bit dtype).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void mul_i64_splat_avx512(int64_t* out, const int64_t* a, const int64_t s, const size_t begin, const size_t end) {
        require_avx512();
        i64_splat_avx512<ElementwiseOp::Mul>(out, a, s, begin, end);
    }

    /**
     * Computes the per-nibble wrap add over [begin, end), 64 packed bytes per step.
     * @brief AVX512 range add (int4x2).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void add_i4x2_range_avx512(uint8_t* out, const uint8_t* a, const uint8_t* b, const size_t begin, const size_t end) {
        require_avx512();
        i4x2_range_avx512<ElementwiseOp::Add>(out, a, b, begin, end);
    }

    /**
     * Computes the per-nibble wrap sub over [begin, end), 64 packed bytes per step.
     * @brief AVX512 range sub (int4x2).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_i4x2_range_avx512(uint8_t* out, const uint8_t* a, const uint8_t* b, const size_t begin, const size_t end) {
        require_avx512();
        i4x2_range_avx512<ElementwiseOp::Sub>(out, a, b, begin, end);
    }

    /**
     * Computes the per-nibble wrap mul over [begin, end), 64 packed bytes per step.
     * @brief AVX512 range mul (int4x2).
     * @param out Output buffer.
     * @param a Left operand buffer.
     * @param b Right operand buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void mul_i4x2_range_avx512(uint8_t* out, const uint8_t* a, const uint8_t* b, const size_t begin, const size_t end) {
        require_avx512();
        i4x2_range_avx512<ElementwiseOp::Mul>(out, a, b, begin, end);
    }

    /**
     * Computes out[i] = s + a[i] over [begin, end), 16 lanes per step — scalar on the left.
     * @brief AVX512 left splat add (fp32).
     * @param out Output buffer.
     * @param a Input buffer.
     * @param s Scalar operand (already narrowed into fp32).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void add_f32_splat_lhs_avx512(float32_t* out, const float32_t* a, const float32_t s, const size_t begin, const size_t end) {
        require_avx512();
        f32_splat_lhs_avx512<ElementwiseOp::Add>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = s - a[i] over [begin, end), 16 lanes per step — scalar on the left.
     * @brief AVX512 left splat sub (fp32).
     * @param out Output buffer.
     * @param a Input buffer.
     * @param s Scalar operand (already narrowed into fp32).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_f32_splat_lhs_avx512(float32_t* out, const float32_t* a, const float32_t s, const size_t begin, const size_t end) {
        require_avx512();
        f32_splat_lhs_avx512<ElementwiseOp::Sub>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = s * a[i] over [begin, end), 16 lanes per step — scalar on the left.
     * @brief AVX512 left splat mul (fp32).
     * @param out Output buffer.
     * @param a Input buffer.
     * @param s Scalar operand (already narrowed into fp32).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void mul_f32_splat_lhs_avx512(float32_t* out, const float32_t* a, const float32_t s, const size_t begin, const size_t end) {
        require_avx512();
        f32_splat_lhs_avx512<ElementwiseOp::Mul>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = s / a[i] over [begin, end), 16 lanes per step — scalar on the left. IEEE div.
     * @brief AVX512 left splat div (fp32).
     * @param out Output buffer.
     * @param a Input buffer.
     * @param s Scalar operand (already narrowed into fp32).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void div_f32_splat_lhs_avx512(float32_t* out, const float32_t* a, const float32_t s, const size_t begin, const size_t end) {
        require_avx512();
        f32_splat_lhs_avx512<ElementwiseOp::Div>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = s + a[i] over [begin, end), 8 lanes per step — scalar on the left.
     * @brief AVX512 left splat add (fp64).
     * @param out Output buffer.
     * @param a Input buffer.
     * @param s Scalar operand (already narrowed into fp64).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void add_f64_splat_lhs_avx512(float64_t* out, const float64_t* a, const float64_t s, const size_t begin, const size_t end) {
        require_avx512();
        f64_splat_lhs_avx512<ElementwiseOp::Add>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = s - a[i] over [begin, end), 8 lanes per step — scalar on the left.
     * @brief AVX512 left splat sub (fp64).
     * @param out Output buffer.
     * @param a Input buffer.
     * @param s Scalar operand (already narrowed into fp64).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_f64_splat_lhs_avx512(float64_t* out, const float64_t* a, const float64_t s, const size_t begin, const size_t end) {
        require_avx512();
        f64_splat_lhs_avx512<ElementwiseOp::Sub>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = s * a[i] over [begin, end), 8 lanes per step — scalar on the left.
     * @brief AVX512 left splat mul (fp64).
     * @param out Output buffer.
     * @param a Input buffer.
     * @param s Scalar operand (already narrowed into fp64).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void mul_f64_splat_lhs_avx512(float64_t* out, const float64_t* a, const float64_t s, const size_t begin, const size_t end) {
        require_avx512();
        f64_splat_lhs_avx512<ElementwiseOp::Mul>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = s / a[i] over [begin, end), 8 lanes per step — scalar on the left. IEEE div.
     * @brief AVX512 left splat div (fp64).
     * @param out Output buffer.
     * @param a Input buffer.
     * @param s Scalar operand (already narrowed into fp64).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void div_f64_splat_lhs_avx512(float64_t* out, const float64_t* a, const float64_t s, const size_t begin, const size_t end) {
        require_avx512();
        f64_splat_lhs_avx512<ElementwiseOp::Div>(out, a, s, begin, end);
    }

    /**
     * Per-lane widen/add/narrow with the scalar on the LEFT over [begin, end), 16
     * lanes per step; the tail runs under a masked 16-bit lane load/store.
     * @brief AVX512 left splat add (fp16 flat lanes).
     * @param out Output lane buffer.
     * @param a Input lane buffer.
     * @param s Scalar lane bit pattern (already narrowed into fp16).
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void add_f16_splat_lhs_avx512(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
        require_avx512();
        f16_splat_lhs_avx512<ElementwiseOp::Add>(out, a, s, begin, end);
    }

    /**
     * Per-lane widen/sub/narrow with the scalar on the LEFT over [begin, end), 16
     * lanes per step; the tail runs under a masked 16-bit lane load/store.
     * @brief AVX512 left splat sub (fp16 flat lanes).
     * @param out Output lane buffer.
     * @param a Input lane buffer.
     * @param s Scalar lane bit pattern (already narrowed into fp16).
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void sub_f16_splat_lhs_avx512(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
        require_avx512();
        f16_splat_lhs_avx512<ElementwiseOp::Sub>(out, a, s, begin, end);
    }

    /**
     * Per-lane widen/mul/narrow with the scalar on the LEFT over [begin, end), 16
     * lanes per step; the tail runs under a masked 16-bit lane load/store.
     * @brief AVX512 left splat mul (fp16 flat lanes).
     * @param out Output lane buffer.
     * @param a Input lane buffer.
     * @param s Scalar lane bit pattern (already narrowed into fp16).
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void mul_f16_splat_lhs_avx512(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
        require_avx512();
        f16_splat_lhs_avx512<ElementwiseOp::Mul>(out, a, s, begin, end);
    }

    /**
     * Per-lane widen/div/narrow with the scalar on the LEFT over [begin, end), 16
     * lanes per step; the tail runs under a masked 16-bit lane load/store.
     * @brief AVX512 left splat div (fp16 flat lanes).
     * @param out Output lane buffer.
     * @param a Input lane buffer.
     * @param s Scalar lane bit pattern (already narrowed into fp16).
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void div_f16_splat_lhs_avx512(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
        require_avx512();
        f16_splat_lhs_avx512<ElementwiseOp::Div>(out, a, s, begin, end);
    }

    /**
     * Per-lane widen/add/narrow with the scalar on the LEFT over [begin, end), 16
     * lanes per step; the tail runs under a masked 16-bit lane load/store. Narrow is
     * hardware cvtneps2bf16.
     * @brief AVX512 left splat add (bf16 flat lanes).
     * @param out Output lane buffer.
     * @param a Input lane buffer.
     * @param s Scalar lane bit pattern (already narrowed into bf16).
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void add_bf16_splat_lhs_avx512(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
        require_avx512();
        bf16_splat_lhs_avx512<ElementwiseOp::Add>(out, a, s, begin, end);
    }

    /**
     * Per-lane widen/sub/narrow with the scalar on the LEFT over [begin, end), 16
     * lanes per step; the tail runs under a masked 16-bit lane load/store. Narrow is
     * hardware cvtneps2bf16.
     * @brief AVX512 left splat sub (bf16 flat lanes).
     * @param out Output lane buffer.
     * @param a Input lane buffer.
     * @param s Scalar lane bit pattern (already narrowed into bf16).
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void sub_bf16_splat_lhs_avx512(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
        require_avx512();
        bf16_splat_lhs_avx512<ElementwiseOp::Sub>(out, a, s, begin, end);
    }

    /**
     * Per-lane widen/mul/narrow with the scalar on the LEFT over [begin, end), 16
     * lanes per step; the tail runs under a masked 16-bit lane load/store. Narrow is
     * hardware cvtneps2bf16.
     * @brief AVX512 left splat mul (bf16 flat lanes).
     * @param out Output lane buffer.
     * @param a Input lane buffer.
     * @param s Scalar lane bit pattern (already narrowed into bf16).
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void mul_bf16_splat_lhs_avx512(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
        require_avx512();
        bf16_splat_lhs_avx512<ElementwiseOp::Mul>(out, a, s, begin, end);
    }

    /**
     * Per-lane widen/div/narrow with the scalar on the LEFT over [begin, end), 16
     * lanes per step; the tail runs under a masked 16-bit lane load/store. Narrow is
     * hardware cvtneps2bf16.
     * @brief AVX512 left splat div (bf16 flat lanes).
     * @param out Output lane buffer.
     * @param a Input lane buffer.
     * @param s Scalar lane bit pattern (already narrowed into bf16).
     * @param begin First lane computed.
     * @param end One past the last lane computed.
     */
    void div_bf16_splat_lhs_avx512(uint16_t* out, const uint16_t* a, const uint16_t s, const size_t begin, const size_t end) {
        require_avx512();
        bf16_splat_lhs_avx512<ElementwiseOp::Div>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = (s - a[i]) mod 2^8 over [begin, end), 64 lanes per step.
     * @brief AVX512 left splat sub (8-bit lanes).
     * @param out Output buffer.
     * @param a Input buffer.
     * @param s Scalar operand (already narrowed into the 8-bit dtype).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_i8_splat_lhs_avx512(int8_t* out, const int8_t* a, const int8_t s, const size_t begin, const size_t end) {
        require_avx512();
        int_splat_lhs_sub_avx512<int8_t>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = (s - a[i]) mod 2^16 over [begin, end), 32 lanes per step.
     * @brief AVX512 left splat sub (16-bit lanes).
     * @param out Output buffer.
     * @param a Input buffer.
     * @param s Scalar operand (already narrowed into the 16-bit dtype).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_i16_splat_lhs_avx512(int16_t* out, const int16_t* a, const int16_t s, const size_t begin, const size_t end) {
        require_avx512();
        int_splat_lhs_sub_avx512<int16_t>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = (s - a[i]) mod 2^32 over [begin, end), 16 lanes per step.
     * @brief AVX512 left splat sub (32-bit lanes).
     * @param out Output buffer.
     * @param a Input buffer.
     * @param s Scalar operand (already narrowed into the 32-bit dtype).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_i32_splat_lhs_avx512(int32_t* out, const int32_t* a, const int32_t s, const size_t begin, const size_t end) {
        require_avx512();
        int_splat_lhs_sub_avx512<int32_t>(out, a, s, begin, end);
    }

    /**
     * Computes out[i] = (s - a[i]) mod 2^64 over [begin, end), 8 lanes per step.
     * @brief AVX512 left splat sub (64-bit lanes).
     * @param out Output buffer.
     * @param a Input buffer.
     * @param s Scalar operand (already narrowed into the 64-bit dtype).
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void sub_i64_splat_lhs_avx512(int64_t* out, const int64_t* a, const int64_t s, const size_t begin, const size_t end) {
        require_avx512();
        int_splat_lhs_sub_avx512<int64_t>(out, a, s, begin, end);
    }
}
