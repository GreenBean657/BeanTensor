/**
 * @file ElementwiseCPU.cpp
 * @author GreenBean
 * @brief CPU backend for elementwise arithmetic ops with broadcasting.
 * Hosts the Tensor::add/sub/mul/div/pow/exp/log method definitions
 * (mirrors how Tensor::bias_add lives in BiasAddCPU.cpp). The binary
 * contiguous path (same-shape, both operands dense) dispatches to the
 * runtime-gated AVX512/AVX2 range kernels in ElementwiseKernels.h;
 * row-decomposable broadcast plans chunk by output row onto the same
 * contiguous kernels (binary rows) and the tensor-scalar splat kernels
 * (inner-dim-broadcast rows); tensor-scalar overloads dispatch to the splat
 * kernels with the narrow_scalar PERMISSIVE conversion applied once on the
 * host. Pow, integer div, non-row-decomposable plans and Int4x2 splat/broadcast
 * rows keep the scalar path.
 *
 * Autograd: the tensor-tensor and tensor-scalar wrappers attach the matching
 * ElementwiseG node to the output whenever an input has requires_grad=true
 * (Math/Autograd/ElementwiseG), so elementwise chains join the computation
 * graph; the tensor-scalar ops persist the scalar in Node::_scalar.
 *
 * Scheduling: pool-chunked launches route through a local twin of
 * linearalgebra_shared::launch_elementwise that submits chunks to the calling
 * thread's CCD group (submit_to_group) whenever the output buffer fits one
 * CCD's LLC — cross-CCD writes of cache-resident buffers cost ~8x. Outputs
 * larger than the LLC keep the unrestricted whole-pool submit.
 * @version 0.7
 * @date 2026-10-25
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include "Math/Autograd/ElementwiseG/ElementwiseG.h"
#include "Math/Autograd/SigmoidG/SigmoidG.h"
#include "Math/Autograd/TanhG/TanhG.h"
#include "Math/Elementwise/Elementwise.h"
#include "Math/Elementwise/ElementwiseKernels.h"
#include "Math/Elementwise/ElementwiseShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/Hardware.h"
#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Tensors::detail {
    using namespace elementwise_shared;
    using namespace linearalgebra_shared;

    namespace {
        // --------------------------- broadcast planning ---------------------------

        /**
         * Right-aligns the operands' dims; each pair must be equal or one of them 1
         * (NumPy rules), the output dim is the max. Stride 0 marks a broadcast dim.
         * @brief Build the broadcast plan and output shape for a tensor-tensor op.
         * @param a_shape Left operand shape.
         * @param a_dims Left operand element strides.
         * @param a_ndim Left operand rank.
         * @param b_shape Right operand shape.
         * @param b_dims Right operand element strides.
         * @param b_ndim Right operand rank.
         * @param out_shape Filled with the broadcast output shape.
         * @return The broadcast execution plan.
         * @throws ErrorHandling::MismatchedShapes A dim pair is unequal and neither is 1.
         */
        [[nodiscard]] ElementwiseBcast compute_bcast_plan(
                const size_t* a_shape, const size_t* a_dims, const size_t a_ndim,
                const size_t* b_shape, const size_t* b_dims, const size_t b_ndim,
                std::vector<size_t>& out_shape) {
            ElementwiseBcast plan{};
            plan.out_ndim = std::max(a_ndim, b_ndim);
            plan.exact = a_ndim == b_ndim && std::equal(a_shape, a_shape + a_ndim, b_shape);
            out_shape.assign(plan.out_ndim, size_t{1});
            plan.out_numel = 1;
            for (auto d = size_t{0}; d < plan.out_ndim; ++d) {
                // d + a_ndim >= out_ndim <=> d >= out_ndim - a_ndim, without underflow.
                const auto a_dim = d + a_ndim >= plan.out_ndim ? a_shape[d - (plan.out_ndim - a_ndim)] : size_t{1};
                const auto b_dim = d + b_ndim >= plan.out_ndim ? b_shape[d - (plan.out_ndim - b_ndim)] : size_t{1};
                if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
                    throw ErrorHandling::MismatchedShapes("elementwise operands are not broadcast-compatible");
                }
                plan.out_shape[d] = std::max(a_dim, b_dim);
                plan.a_strides[d] = a_dim == 1 ? size_t{0} : a_dims[d - (plan.out_ndim - a_ndim)];
                plan.b_strides[d] = b_dim == 1 ? size_t{0} : b_dims[d - (plan.out_ndim - b_ndim)];
                out_shape[d] = plan.out_shape[d];
                plan.out_numel *= plan.out_shape[d];
            }
            plan.out_strides[plan.out_ndim - 1] = 1;
            for (auto d = plan.out_ndim - 1; d-- > 0;) {
                plan.out_strides[d] = plan.out_strides[d + 1] * plan.out_shape[d + 1];
            }
            return plan;
        }

        // --------------------------- arithmetic applies ---------------------------

        /**
         * Single-rounding native op under the shared NaN contract (fp_binop); Div is
         * IEEE (x/0 -> inf/nan), Pow is std::pow.
         * @brief Scalar binary apply (fp32/fp64).
         */
        template<typename T, ElementwiseOp Op>
        [[nodiscard]] T apply_native(const T a, const T b) {
            if constexpr (Op == ElementwiseOp::Pow) {
                return std::pow(a, b);
            } else {
                return Math::Elementwise::fp_binop<T, Op>(a, b);
            }
        }

        /**
         * Widen to fp32, apply, narrow RNE (unaccel pathway) — the same
         * conversion-pathway semantics as the bias_add/relu half kernels.
         * @brief Scalar binary apply (fp16/bf16, fp32 compute).
         */
        template<typename H, ElementwiseOp Op>
        [[nodiscard]] H apply_half(const H& a, const H& b) {
            const auto wa = widen_lane(a);
            const auto wb = widen_lane(b);
            if constexpr (Op == ElementwiseOp::Pow) {
                return narrow_lane<H>(std::pow(wa, wb));
            } else {
                return narrow_lane<H>(Math::Elementwise::fp_binop<float32_t, Op>(wa, wb));
            }
        }

        /**
         * Per-lane widen/apply/narrow, repack on store.
         * @brief Scalar binary apply (fp16x2/bf16x2, per-lane fp32 compute).
         */
        template<typename P, ElementwiseOp Op>
        [[nodiscard]] P apply_packed(const P& a, const P& b) {
            using H = packed_lane_t<P>;
            const auto a_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(a);
            const auto b_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(b);
            const auto lo = apply_half<H, Op>(lane_lo<H>(a_bits), lane_lo<H>(b_bits));
            const auto hi = apply_half<H, Op>(lane_hi<H>(a_bits), lane_hi<H>(b_bits));
            return pack_lanes<P>(Intrinsics::detail::bit_cast_memcpy<uint16_t>(lo),
                                 Intrinsics::detail::bit_cast_memcpy<uint16_t>(hi));
        }

        /**
         * Modular (wrap) add/sub/mul through the unsigned twin — the same wrap
         * contract as the integer bias_add/sum kernels. Div truncates toward zero;
         * division by zero yields 0 and min/-1 yields min (wrap), never UB.
         * Pow on integers is rejected before dispatch and never reaches here.
         * @brief Scalar binary apply (all integer dtypes).
         */
        template<typename T, ElementwiseOp Op>
        [[nodiscard]] T apply_int(const T a, const T b) {
            using U = std::make_unsigned_t<T>;
            if constexpr (Op == ElementwiseOp::Add) {
                return static_cast<T>(static_cast<U>(static_cast<U>(a) + static_cast<U>(b)));
            } else if constexpr (Op == ElementwiseOp::Sub) {
                return static_cast<T>(static_cast<U>(static_cast<U>(a) - static_cast<U>(b)));
            } else if constexpr (Op == ElementwiseOp::Mul) {
                return static_cast<T>(static_cast<U>(static_cast<U>(a) * static_cast<U>(b)));
            } else if constexpr (Op == ElementwiseOp::Div) {
                if (b == T{0}) {
                    return T{0};
                }
                if constexpr (std::is_signed_v<T>) {
                    // min/-1 overflows; define it as the wrap result (min). The
                    // min pattern is built through the unsigned twin because
                    // nvcc rejects std::numeric_limits in device code, and the
                    // GPU twins mirror this branch.
                    constexpr auto min_pattern = static_cast<T>(U{1} << (sizeof(T) * 8u - 1u));
                    if (a == min_pattern && b == T{-1}) {
                        return a;
                    }
                }
                return static_cast<T>(a / b);
            } else {
                return T{0}; // Pow: unreachable — integer pow is rejected at the Tensor method.
            }
        }

        /**
         * Per-nibble apply, repack on store. Add/Sub/Mul wrap per nibble (the
         * pack keeps the low 4 bits); Div truncates per nibble with the same
         * div-by-zero -> 0 rule (min/-1 = 8 wraps to -8 through the pack).
         * @brief Scalar binary apply (int4x2, per-nibble).
         */
        template<ElementwiseOp Op>
        [[nodiscard]] int4x2_t apply_int4x2(const int4x2_t a, const int4x2_t b) {
            const auto alo = int4_lo(a);
            const auto ahi = int4_hi(a);
            const auto blo = int4_lo(b);
            const auto bhi = int4_hi(b);
            if constexpr (Op == ElementwiseOp::Add) {
                return pack_int4(alo + blo, ahi + bhi);
            } else if constexpr (Op == ElementwiseOp::Sub) {
                return pack_int4(alo - blo, ahi - bhi);
            } else if constexpr (Op == ElementwiseOp::Mul) {
                return pack_int4(alo * blo, ahi * bhi);
            } else if constexpr (Op == ElementwiseOp::Div) {
                return pack_int4(blo == 0 ? 0 : alo / blo, bhi == 0 ? 0 : ahi / bhi);
            } else {
                return a; // Pow: unreachable — integer pow is rejected at the Tensor method.
            }
        }

        /**
         * @brief Category dispatch for the binary applies.
         */
        template<typename T, ElementwiseOp Op>
        [[nodiscard]] T apply_any(const T& a, const T& b) {
            if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                return apply_native<T, Op>(a, b);
            } else if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
                return apply_half<T, Op>(a, b);
            } else if constexpr (DataTypes::requires_unpack_v<T>) {
                return apply_packed<T, Op>(a, b);
            } else if constexpr (std::is_same_v<T, int4x2_t>) {
                return apply_int4x2<Op>(a, b);
            } else {
                return apply_int<T, Op>(a, b);
            }
        }

        // --------------------------- unary applies ---------------------------

        /**
         * std::exp/std::log in the storage width; IEEE edges (log of a negative
         * is NaN, log(0) is -inf) come from the library.
         * @brief Scalar unary apply (fp32/fp64).
         */
        template<typename T, ElementwiseUnaryOp Op>
        [[nodiscard]] T apply_unary_native(const T v) {
            if constexpr (Op == ElementwiseUnaryOp::Exp) {
                return std::exp(v);
            } else if constexpr (Op == ElementwiseUnaryOp::Log) {
                return std::log(v);
            } else if constexpr (Op == ElementwiseUnaryOp::Sigmoid) {
                return T{1.0} / (T{1.0} + std::exp(-v));
            } else {
                // Stable tanh: 1 - 2/(exp(2x)+1) avoids overflow at |x| ~= 100.
                return T{1.0} - T{2.0} / (std::exp(T{2.0} * v) + T{1.0});
            }
        }

        /**
         * Widen to fp32, apply, narrow RNE.
         * @brief Scalar unary apply (fp16/bf16, fp32 compute).
         */
        template<typename H, ElementwiseUnaryOp Op>
        [[nodiscard]] H apply_unary_half(const H& v) {
            const auto wide = widen_lane(v);
            if constexpr (Op == ElementwiseUnaryOp::Exp) {
                return narrow_lane<H>(std::exp(wide));
            } else if constexpr (Op == ElementwiseUnaryOp::Log) {
                return narrow_lane<H>(std::log(wide));
            } else if constexpr (Op == ElementwiseUnaryOp::Sigmoid) {
                return narrow_lane<H>(float32_t{1.0f} / (float32_t{1.0f} + std::exp(-wide)));
            } else {
                return narrow_lane<H>(float32_t{1.0f} - float32_t{2.0f} / (std::exp(float32_t{2.0f} * wide) + float32_t{1.0f}));
            }
        }

        /**
         * Per-lane widen/apply/narrow, repack on store.
         * @brief Scalar unary apply (fp16x2/bf16x2, per-lane fp32 compute).
         */
        template<typename P, ElementwiseUnaryOp Op>
        [[nodiscard]] P apply_unary_packed(const P& v) {
            using H = packed_lane_t<P>;
            const auto bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(v);
            const auto lo = apply_unary_half<H, Op>(lane_lo<H>(bits));
            const auto hi = apply_unary_half<H, Op>(lane_hi<H>(bits));
            return pack_lanes<P>(Intrinsics::detail::bit_cast_memcpy<uint16_t>(lo),
                                 Intrinsics::detail::bit_cast_memcpy<uint16_t>(hi));
        }

        /**
         * @brief Category dispatch for the unary applies.
         */
        template<typename T, ElementwiseUnaryOp Op>
        [[nodiscard]] T apply_any_unary(const T& v) {
            if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                return apply_unary_native<T, Op>(v);
            } else if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
                return apply_unary_half<T, Op>(v);
            } else {
                return apply_unary_packed<T, Op>(v);
            }
        }

        // --------------------------- range kernels ---------------------------

        /**
         * @brief Scalar range binary op with broadcast offsets (any dtype, arithmetic ops).
         * @param out Output buffer.
         * @param a Left operand buffer.
         * @param b Right operand buffer.
         * @param plan Broadcast execution plan.
         * @param begin First element computed.
         * @param end One past the last element computed.
         */
        template<typename T, ElementwiseOp Op>
        void binary_range(T* out, const T* a, const T* b, const ElementwiseBcast& plan, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                const auto [ao, bo] = bcast_offsets(plan, i);
                out[i] = apply_any<T, Op>(a[ao], b[bo]);
            }
        }

        /**
         * No plan needed: a is contiguous and the output shares its shape, so
         * both flat indices are i. The scalar rides along by const reference.
         * @brief Scalar range tensor-scalar op (any dtype, arithmetic ops).
         * @param out Output buffer.
         * @param a Input buffer.
         * @param scalar Scalar already narrowed into T.
         * @param begin First element computed.
         * @param end One past the last element computed.
         */
        template<typename T, ElementwiseOp Op>
        void scalar_range(T* out, const T* a, const T& scalar, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                out[i] = apply_any<T, Op>(a[i], scalar);
            }
        }

        /**
         * @brief Scalar range unary op (float kinds).
         * @param out Output buffer.
         * @param in Input buffer.
         * @param begin First element computed.
         * @param end One past the last element computed.
         */
        template<typename T, ElementwiseUnaryOp Op>
        void unary_range(T* out, const T* in, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                out[i] = apply_any_unary<T, Op>(in[i]);
            }
        }

        // --------------------------- contiguous SIMD dispatch ---------------------------

        /**
         * Function pointer type shared by every contiguous binary range kernel
         * (scalar fallback, AVX2, AVX512) for a native element type.
         * @brief Contiguous binary range function pointer type.
         */
        template<typename T>
        using ContigRangeFn = void(*)(T*, const T*, const T*, size_t, size_t);

        /**
         * Function pointer type for the flat 16-bit lane kernels — fp16/bf16 buffers
         * processed as independent 16-bit lanes (one per element, two per packed element).
         * @brief Flat 16-bit lane range function pointer type.
         */
        using HalfLaneRangeFn = void(*)(uint16_t*, const uint16_t*, const uint16_t*, size_t, size_t);

        /**
         * Function pointer type for the tensor-scalar splat kernels over a native
         * element type — the scalar rides by value, already narrowed into T.
         * @brief Contiguous splat range function pointer type.
         */
        template<typename T>
        using SplatRangeFn = void(*)(T*, const T*, T, size_t, size_t);

        /**
         * Function pointer type for the flat 16-bit lane splat kernels (fp16/bf16
         * buffers as lane arrays; the scalar is one lane bit pattern — for packed
         * dtypes the caller narrows it into both lanes, so any lane broadcasts fine).
         * @brief Flat 16-bit lane splat function pointer type.
         */
        using HalfLaneSplatFn = void(*)(uint16_t*, const uint16_t*, uint16_t, size_t, size_t);

        /**
         * Function pointer type for the per-nibble Int4x2 kernels over flat bytes.
         * @brief int4x2 range function pointer type.
         */
        using I4x2RangeFn = void(*)(uint8_t*, const uint8_t*, const uint8_t*, size_t, size_t);

        /**
         * Unsigned integers ride the signed width's kernel: +,-,* bit patterns are
         * sign-agnostic under wrap semantics, so the kernels are declared once per width.
         * @brief Canonical (signed-width) type a dtype's SIMD kernel is declared with.
         */
        template<typename T>
        struct canonical_simd {
            using type = T;
        };
        template<>
        struct canonical_simd<uint8_t> {
            using type = int8_t;
        };
        template<>
        struct canonical_simd<uint16_t> {
            using type = int16_t;
        };
        template<>
        struct canonical_simd<uint32_t> {
            using type = int32_t;
        };
        template<>
        struct canonical_simd<uint64_t> {
            using type = int64_t;
        };
        template<typename T>
        using canonical_simd_t = typename canonical_simd<T>::type;

        /**
         * Compile-time (T, Op) -> AVX512 kernel map for the canonical native types.
         * Resolves nullptr for ops with no kernel (integer div: no x86 integer
         * division; Pow: phase 1 scalar-only) — callers fall through to AVX2/scalar.
         * @brief Resolve the AVX512 contiguous kernel for a native element type.
         * @return The kernel, or nullptr when this (T, Op) stays scalar.
         */
        template<typename T, ElementwiseOp Op>
        [[nodiscard]] ContigRangeFn<T> native_avx512_kernel() {
            if constexpr (std::is_same_v<T, float32_t>) {
                if constexpr (Op == ElementwiseOp::Add) { return &Math::Elementwise::add_f32_range_avx512; }
                if constexpr (Op == ElementwiseOp::Sub) { return &Math::Elementwise::sub_f32_range_avx512; }
                if constexpr (Op == ElementwiseOp::Mul) { return &Math::Elementwise::mul_f32_range_avx512; }
                if constexpr (Op == ElementwiseOp::Div) { return &Math::Elementwise::div_f32_range_avx512; }
            } else if constexpr (std::is_same_v<T, float64_t>) {
                if constexpr (Op == ElementwiseOp::Add) { return &Math::Elementwise::add_f64_range_avx512; }
                if constexpr (Op == ElementwiseOp::Sub) { return &Math::Elementwise::sub_f64_range_avx512; }
                if constexpr (Op == ElementwiseOp::Mul) { return &Math::Elementwise::mul_f64_range_avx512; }
                if constexpr (Op == ElementwiseOp::Div) { return &Math::Elementwise::div_f64_range_avx512; }
            } else if constexpr (std::is_same_v<T, int8_t>) {
                if constexpr (Op == ElementwiseOp::Add) { return &Math::Elementwise::add_i8_range_avx512; }
                if constexpr (Op == ElementwiseOp::Sub) { return &Math::Elementwise::sub_i8_range_avx512; }
                if constexpr (Op == ElementwiseOp::Mul) { return &Math::Elementwise::mul_i8_range_avx512; }
            } else if constexpr (std::is_same_v<T, int16_t>) {
                if constexpr (Op == ElementwiseOp::Add) { return &Math::Elementwise::add_i16_range_avx512; }
                if constexpr (Op == ElementwiseOp::Sub) { return &Math::Elementwise::sub_i16_range_avx512; }
                if constexpr (Op == ElementwiseOp::Mul) { return &Math::Elementwise::mul_i16_range_avx512; }
            } else if constexpr (std::is_same_v<T, int32_t>) {
                if constexpr (Op == ElementwiseOp::Add) { return &Math::Elementwise::add_i32_range_avx512; }
                if constexpr (Op == ElementwiseOp::Sub) { return &Math::Elementwise::sub_i32_range_avx512; }
                if constexpr (Op == ElementwiseOp::Mul) { return &Math::Elementwise::mul_i32_range_avx512; }
            } else if constexpr (std::is_same_v<T, int64_t>) {
                if constexpr (Op == ElementwiseOp::Add) { return &Math::Elementwise::add_i64_range_avx512; }
                if constexpr (Op == ElementwiseOp::Sub) { return &Math::Elementwise::sub_i64_range_avx512; }
                if constexpr (Op == ElementwiseOp::Mul) { return &Math::Elementwise::mul_i64_range_avx512; }
            }
            return nullptr;
        }

        /**
         * Compile-time (T, Op) -> AVX2 kernel map for the canonical native types.
         * Resolves nullptr for ops with no kernel (integer div, Pow) — callers fall
         * through to the scalar apply.
         * @brief Resolve the AVX2 contiguous kernel for a native element type.
         * @return The kernel, or nullptr when this (T, Op) stays scalar.
         */
        template<typename T, ElementwiseOp Op>
        [[nodiscard]] ContigRangeFn<T> native_avx2_kernel() {
            if constexpr (std::is_same_v<T, float32_t>) {
                if constexpr (Op == ElementwiseOp::Add) { return &Math::Elementwise::add_f32_range_avx2; }
                if constexpr (Op == ElementwiseOp::Sub) { return &Math::Elementwise::sub_f32_range_avx2; }
                if constexpr (Op == ElementwiseOp::Mul) { return &Math::Elementwise::mul_f32_range_avx2; }
                if constexpr (Op == ElementwiseOp::Div) { return &Math::Elementwise::div_f32_range_avx2; }
            } else if constexpr (std::is_same_v<T, float64_t>) {
                if constexpr (Op == ElementwiseOp::Add) { return &Math::Elementwise::add_f64_range_avx2; }
                if constexpr (Op == ElementwiseOp::Sub) { return &Math::Elementwise::sub_f64_range_avx2; }
                if constexpr (Op == ElementwiseOp::Mul) { return &Math::Elementwise::mul_f64_range_avx2; }
                if constexpr (Op == ElementwiseOp::Div) { return &Math::Elementwise::div_f64_range_avx2; }
            } else if constexpr (std::is_same_v<T, int8_t>) {
                if constexpr (Op == ElementwiseOp::Add) { return &Math::Elementwise::add_i8_range_avx2; }
                if constexpr (Op == ElementwiseOp::Sub) { return &Math::Elementwise::sub_i8_range_avx2; }
                if constexpr (Op == ElementwiseOp::Mul) { return &Math::Elementwise::mul_i8_range_avx2; }
            } else if constexpr (std::is_same_v<T, int16_t>) {
                if constexpr (Op == ElementwiseOp::Add) { return &Math::Elementwise::add_i16_range_avx2; }
                if constexpr (Op == ElementwiseOp::Sub) { return &Math::Elementwise::sub_i16_range_avx2; }
                if constexpr (Op == ElementwiseOp::Mul) { return &Math::Elementwise::mul_i16_range_avx2; }
            } else if constexpr (std::is_same_v<T, int32_t>) {
                if constexpr (Op == ElementwiseOp::Add) { return &Math::Elementwise::add_i32_range_avx2; }
                if constexpr (Op == ElementwiseOp::Sub) { return &Math::Elementwise::sub_i32_range_avx2; }
                if constexpr (Op == ElementwiseOp::Mul) { return &Math::Elementwise::mul_i32_range_avx2; }
            } else if constexpr (std::is_same_v<T, int64_t>) {
                if constexpr (Op == ElementwiseOp::Add) { return &Math::Elementwise::add_i64_range_avx2; }
                if constexpr (Op == ElementwiseOp::Sub) { return &Math::Elementwise::sub_i64_range_avx2; }
                if constexpr (Op == ElementwiseOp::Mul) { return &Math::Elementwise::mul_i64_range_avx2; }
            }
            return nullptr;
        }

        /**
         * Compile-time Op -> fp16 flat-lane kernel map (add/sub/mul/div all exist).
         * @brief Resolve the fp16 flat-lane kernel for an ISA.
         * @return The kernel, or nullptr for an op with no half kernel.
         */
        template<ElementwiseOp Op, bool Avx512>
        [[nodiscard]] HalfLaneRangeFn f16_lane_kernel() {
            if constexpr (Op == ElementwiseOp::Add) {
                return Avx512 ? &Math::Elementwise::add_f16_range_avx512 : &Math::Elementwise::add_f16_range_avx2;
            }
            if constexpr (Op == ElementwiseOp::Sub) {
                return Avx512 ? &Math::Elementwise::sub_f16_range_avx512 : &Math::Elementwise::sub_f16_range_avx2;
            }
            if constexpr (Op == ElementwiseOp::Mul) {
                return Avx512 ? &Math::Elementwise::mul_f16_range_avx512 : &Math::Elementwise::mul_f16_range_avx2;
            }
            if constexpr (Op == ElementwiseOp::Div) {
                return Avx512 ? &Math::Elementwise::div_f16_range_avx512 : &Math::Elementwise::div_f16_range_avx2;
            }
            return nullptr;
        }

        /**
         * Compile-time Op -> bf16 flat-lane kernel map (add/sub/mul/div all exist).
         * @brief Resolve the bf16 flat-lane kernel for an ISA.
         * @return The kernel, or nullptr for an op with no half kernel.
         */
        template<ElementwiseOp Op, bool Avx512>
        [[nodiscard]] HalfLaneRangeFn bf16_lane_kernel() {
            if constexpr (Op == ElementwiseOp::Add) {
                return Avx512 ? &Math::Elementwise::add_bf16_range_avx512 : &Math::Elementwise::add_bf16_range_avx2;
            }
            if constexpr (Op == ElementwiseOp::Sub) {
                return Avx512 ? &Math::Elementwise::sub_bf16_range_avx512 : &Math::Elementwise::sub_bf16_range_avx2;
            }
            if constexpr (Op == ElementwiseOp::Mul) {
                return Avx512 ? &Math::Elementwise::mul_bf16_range_avx512 : &Math::Elementwise::mul_bf16_range_avx2;
            }
            if constexpr (Op == ElementwiseOp::Div) {
                return Avx512 ? &Math::Elementwise::div_bf16_range_avx512 : &Math::Elementwise::div_bf16_range_avx2;
            }
            return nullptr;
        }

        /**
         * Compile-time (T, Op) -> AVX512 SPLAT kernel map for the canonical native
         * types. Resolves nullptr for ops with no splat kernel (integer div: no x86
         * integer division; Pow: scalar-only) — callers fall through to AVX2/scalar.
         * @brief Resolve the AVX512 contiguous splat kernel for a native element type.
         * @return The kernel, or nullptr when this (T, Op) stays scalar.
         */
        template<typename T, ElementwiseOp Op>
        [[nodiscard]] SplatRangeFn<T> native_avx512_splat_kernel() {
            if constexpr (std::is_same_v<T, float32_t>) {
                if constexpr (Op == ElementwiseOp::Add) { return &Math::Elementwise::add_f32_splat_avx512; }
                if constexpr (Op == ElementwiseOp::Sub) { return &Math::Elementwise::sub_f32_splat_avx512; }
                if constexpr (Op == ElementwiseOp::Mul) { return &Math::Elementwise::mul_f32_splat_avx512; }
                if constexpr (Op == ElementwiseOp::Div) { return &Math::Elementwise::div_f32_splat_avx512; }
            } else if constexpr (std::is_same_v<T, float64_t>) {
                if constexpr (Op == ElementwiseOp::Add) { return &Math::Elementwise::add_f64_splat_avx512; }
                if constexpr (Op == ElementwiseOp::Sub) { return &Math::Elementwise::sub_f64_splat_avx512; }
                if constexpr (Op == ElementwiseOp::Mul) { return &Math::Elementwise::mul_f64_splat_avx512; }
                if constexpr (Op == ElementwiseOp::Div) { return &Math::Elementwise::div_f64_splat_avx512; }
            } else if constexpr (std::is_same_v<T, int8_t>) {
                if constexpr (Op == ElementwiseOp::Add) { return &Math::Elementwise::add_i8_splat_avx512; }
                if constexpr (Op == ElementwiseOp::Sub) { return &Math::Elementwise::sub_i8_splat_avx512; }
                if constexpr (Op == ElementwiseOp::Mul) { return &Math::Elementwise::mul_i8_splat_avx512; }
            } else if constexpr (std::is_same_v<T, int16_t>) {
                if constexpr (Op == ElementwiseOp::Add) { return &Math::Elementwise::add_i16_splat_avx512; }
                if constexpr (Op == ElementwiseOp::Sub) { return &Math::Elementwise::sub_i16_splat_avx512; }
                if constexpr (Op == ElementwiseOp::Mul) { return &Math::Elementwise::mul_i16_splat_avx512; }
            } else if constexpr (std::is_same_v<T, int32_t>) {
                if constexpr (Op == ElementwiseOp::Add) { return &Math::Elementwise::add_i32_splat_avx512; }
                if constexpr (Op == ElementwiseOp::Sub) { return &Math::Elementwise::sub_i32_splat_avx512; }
                if constexpr (Op == ElementwiseOp::Mul) { return &Math::Elementwise::mul_i32_splat_avx512; }
            } else if constexpr (std::is_same_v<T, int64_t>) {
                if constexpr (Op == ElementwiseOp::Add) { return &Math::Elementwise::add_i64_splat_avx512; }
                if constexpr (Op == ElementwiseOp::Sub) { return &Math::Elementwise::sub_i64_splat_avx512; }
                if constexpr (Op == ElementwiseOp::Mul) { return &Math::Elementwise::mul_i64_splat_avx512; }
            }
            return nullptr;
        }

        /**
         * Compile-time (T, Op) -> AVX2 SPLAT kernel map for the canonical native types.
         * Resolves nullptr for ops with no splat kernel (integer div, Pow) — callers
         * fall through to the scalar apply.
         * @brief Resolve the AVX2 contiguous splat kernel for a native element type.
         * @return The kernel, or nullptr when this (T, Op) stays scalar.
         */
        template<typename T, ElementwiseOp Op>
        [[nodiscard]] SplatRangeFn<T> native_avx2_splat_kernel() {
            if constexpr (std::is_same_v<T, float32_t>) {
                if constexpr (Op == ElementwiseOp::Add) { return &Math::Elementwise::add_f32_splat_avx2; }
                if constexpr (Op == ElementwiseOp::Sub) { return &Math::Elementwise::sub_f32_splat_avx2; }
                if constexpr (Op == ElementwiseOp::Mul) { return &Math::Elementwise::mul_f32_splat_avx2; }
                if constexpr (Op == ElementwiseOp::Div) { return &Math::Elementwise::div_f32_splat_avx2; }
            } else if constexpr (std::is_same_v<T, float64_t>) {
                if constexpr (Op == ElementwiseOp::Add) { return &Math::Elementwise::add_f64_splat_avx2; }
                if constexpr (Op == ElementwiseOp::Sub) { return &Math::Elementwise::sub_f64_splat_avx2; }
                if constexpr (Op == ElementwiseOp::Mul) { return &Math::Elementwise::mul_f64_splat_avx2; }
                if constexpr (Op == ElementwiseOp::Div) { return &Math::Elementwise::div_f64_splat_avx2; }
            } else if constexpr (std::is_same_v<T, int8_t>) {
                if constexpr (Op == ElementwiseOp::Add) { return &Math::Elementwise::add_i8_splat_avx2; }
                if constexpr (Op == ElementwiseOp::Sub) { return &Math::Elementwise::sub_i8_splat_avx2; }
                if constexpr (Op == ElementwiseOp::Mul) { return &Math::Elementwise::mul_i8_splat_avx2; }
            } else if constexpr (std::is_same_v<T, int16_t>) {
                if constexpr (Op == ElementwiseOp::Add) { return &Math::Elementwise::add_i16_splat_avx2; }
                if constexpr (Op == ElementwiseOp::Sub) { return &Math::Elementwise::sub_i16_splat_avx2; }
                if constexpr (Op == ElementwiseOp::Mul) { return &Math::Elementwise::mul_i16_splat_avx2; }
            } else if constexpr (std::is_same_v<T, int32_t>) {
                if constexpr (Op == ElementwiseOp::Add) { return &Math::Elementwise::add_i32_splat_avx2; }
                if constexpr (Op == ElementwiseOp::Sub) { return &Math::Elementwise::sub_i32_splat_avx2; }
                if constexpr (Op == ElementwiseOp::Mul) { return &Math::Elementwise::mul_i32_splat_avx2; }
            } else if constexpr (std::is_same_v<T, int64_t>) {
                if constexpr (Op == ElementwiseOp::Add) { return &Math::Elementwise::add_i64_splat_avx2; }
                if constexpr (Op == ElementwiseOp::Sub) { return &Math::Elementwise::sub_i64_splat_avx2; }
                if constexpr (Op == ElementwiseOp::Mul) { return &Math::Elementwise::mul_i64_splat_avx2; }
            }
            return nullptr;
        }

        /**
         * Compile-time Op -> fp16 flat-lane SPLAT kernel map (add/sub/mul/div all exist).
         * @brief Resolve the fp16 flat-lane splat kernel for an ISA.
         * @return The kernel, or nullptr for an op with no half splat kernel.
         */
        template<ElementwiseOp Op, bool Avx512>
        [[nodiscard]] HalfLaneSplatFn f16_lane_splat_kernel() {
            if constexpr (Op == ElementwiseOp::Add) {
                return Avx512 ? &Math::Elementwise::add_f16_splat_avx512 : &Math::Elementwise::add_f16_splat_avx2;
            }
            if constexpr (Op == ElementwiseOp::Sub) {
                return Avx512 ? &Math::Elementwise::sub_f16_splat_avx512 : &Math::Elementwise::sub_f16_splat_avx2;
            }
            if constexpr (Op == ElementwiseOp::Mul) {
                return Avx512 ? &Math::Elementwise::mul_f16_splat_avx512 : &Math::Elementwise::mul_f16_splat_avx2;
            }
            if constexpr (Op == ElementwiseOp::Div) {
                return Avx512 ? &Math::Elementwise::div_f16_splat_avx512 : &Math::Elementwise::div_f16_splat_avx2;
            }
            return nullptr;
        }

        /**
         * Compile-time Op -> bf16 flat-lane SPLAT kernel map (add/sub/mul/div all exist).
         * @brief Resolve the bf16 flat-lane splat kernel for an ISA.
         * @return The kernel, or nullptr for an op with no half splat kernel.
         */
        template<ElementwiseOp Op, bool Avx512>
        [[nodiscard]] HalfLaneSplatFn bf16_lane_splat_kernel() {
            if constexpr (Op == ElementwiseOp::Add) {
                return Avx512 ? &Math::Elementwise::add_bf16_splat_avx512 : &Math::Elementwise::add_bf16_splat_avx2;
            }
            if constexpr (Op == ElementwiseOp::Sub) {
                return Avx512 ? &Math::Elementwise::sub_bf16_splat_avx512 : &Math::Elementwise::sub_bf16_splat_avx2;
            }
            if constexpr (Op == ElementwiseOp::Mul) {
                return Avx512 ? &Math::Elementwise::mul_bf16_splat_avx512 : &Math::Elementwise::mul_bf16_splat_avx2;
            }
            if constexpr (Op == ElementwiseOp::Div) {
                return Avx512 ? &Math::Elementwise::div_bf16_splat_avx512 : &Math::Elementwise::div_bf16_splat_avx2;
            }
            return nullptr;
        }

        /**
         * Compile-time Op -> int4x2 kernel map (wrap add/sub/mul only — div has the
         * scalar per-nibble div-by-zero rule and no x86 integer division).
         * @brief Resolve the int4x2 kernel for an ISA.
         * @return The kernel, or nullptr for an op with no int4x2 kernel.
         */
        template<ElementwiseOp Op, bool Avx512>
        [[nodiscard]] I4x2RangeFn i4x2_kernel() {
            if constexpr (Op == ElementwiseOp::Add) {
                return Avx512 ? &Math::Elementwise::add_i4x2_range_avx512 : &Math::Elementwise::add_i4x2_range_avx2;
            }
            if constexpr (Op == ElementwiseOp::Sub) {
                return Avx512 ? &Math::Elementwise::sub_i4x2_range_avx512 : &Math::Elementwise::sub_i4x2_range_avx2;
            }
            if constexpr (Op == ElementwiseOp::Mul) {
                return Avx512 ? &Math::Elementwise::mul_i4x2_range_avx512 : &Math::Elementwise::mul_i4x2_range_avx2;
            }
            return nullptr;
        }

        /**
         * Compile-time (T, Op) -> AVX512 LEFT-splat kernel map: the scalar on the
         * LEFT (s op a[i]), matching the scalar apply's operand order for broadcast
         * rows with operand A on inner stride 0. fp kinds cover add/sub/mul/div;
         * integers cover only sub (wrap add/mul are bitwise commutative, so the
         * plain splats already match). Resolves nullptr otherwise.
         * @brief Resolve the AVX512 left-splat kernel for a native element type.
         * @return The kernel, or nullptr when this (T, Op) stays scalar.
         */
        template<typename T, ElementwiseOp Op>
        [[nodiscard]] SplatRangeFn<T> native_avx512_lhs_splat_kernel() {
            if constexpr (std::is_same_v<T, float32_t>) {
                if constexpr (Op == ElementwiseOp::Add) { return &Math::Elementwise::add_f32_splat_lhs_avx512; }
                if constexpr (Op == ElementwiseOp::Sub) { return &Math::Elementwise::sub_f32_splat_lhs_avx512; }
                if constexpr (Op == ElementwiseOp::Mul) { return &Math::Elementwise::mul_f32_splat_lhs_avx512; }
                if constexpr (Op == ElementwiseOp::Div) { return &Math::Elementwise::div_f32_splat_lhs_avx512; }
            } else if constexpr (std::is_same_v<T, float64_t>) {
                if constexpr (Op == ElementwiseOp::Add) { return &Math::Elementwise::add_f64_splat_lhs_avx512; }
                if constexpr (Op == ElementwiseOp::Sub) { return &Math::Elementwise::sub_f64_splat_lhs_avx512; }
                if constexpr (Op == ElementwiseOp::Mul) { return &Math::Elementwise::mul_f64_splat_lhs_avx512; }
                if constexpr (Op == ElementwiseOp::Div) { return &Math::Elementwise::div_f64_splat_lhs_avx512; }
            } else if constexpr (std::is_same_v<T, int8_t>) {
                if constexpr (Op == ElementwiseOp::Sub) { return &Math::Elementwise::sub_i8_splat_lhs_avx512; }
            } else if constexpr (std::is_same_v<T, int16_t>) {
                if constexpr (Op == ElementwiseOp::Sub) { return &Math::Elementwise::sub_i16_splat_lhs_avx512; }
            } else if constexpr (std::is_same_v<T, int32_t>) {
                if constexpr (Op == ElementwiseOp::Sub) { return &Math::Elementwise::sub_i32_splat_lhs_avx512; }
            } else if constexpr (std::is_same_v<T, int64_t>) {
                if constexpr (Op == ElementwiseOp::Sub) { return &Math::Elementwise::sub_i64_splat_lhs_avx512; }
            }
            return nullptr;
        }

        /**
         * Compile-time (T, Op) -> AVX2 LEFT-splat kernel map (scalar on the left).
         * Same coverage as the AVX512 map.
         * @brief Resolve the AVX2 left-splat kernel for a native element type.
         * @return The kernel, or nullptr when this (T, Op) stays scalar.
         */
        template<typename T, ElementwiseOp Op>
        [[nodiscard]] SplatRangeFn<T> native_avx2_lhs_splat_kernel() {
            if constexpr (std::is_same_v<T, float32_t>) {
                if constexpr (Op == ElementwiseOp::Add) { return &Math::Elementwise::add_f32_splat_lhs_avx2; }
                if constexpr (Op == ElementwiseOp::Sub) { return &Math::Elementwise::sub_f32_splat_lhs_avx2; }
                if constexpr (Op == ElementwiseOp::Mul) { return &Math::Elementwise::mul_f32_splat_lhs_avx2; }
                if constexpr (Op == ElementwiseOp::Div) { return &Math::Elementwise::div_f32_splat_lhs_avx2; }
            } else if constexpr (std::is_same_v<T, float64_t>) {
                if constexpr (Op == ElementwiseOp::Add) { return &Math::Elementwise::add_f64_splat_lhs_avx2; }
                if constexpr (Op == ElementwiseOp::Sub) { return &Math::Elementwise::sub_f64_splat_lhs_avx2; }
                if constexpr (Op == ElementwiseOp::Mul) { return &Math::Elementwise::mul_f64_splat_lhs_avx2; }
                if constexpr (Op == ElementwiseOp::Div) { return &Math::Elementwise::div_f64_splat_lhs_avx2; }
            } else if constexpr (std::is_same_v<T, int8_t>) {
                if constexpr (Op == ElementwiseOp::Sub) { return &Math::Elementwise::sub_i8_splat_lhs_avx2; }
            } else if constexpr (std::is_same_v<T, int16_t>) {
                if constexpr (Op == ElementwiseOp::Sub) { return &Math::Elementwise::sub_i16_splat_lhs_avx2; }
            } else if constexpr (std::is_same_v<T, int32_t>) {
                if constexpr (Op == ElementwiseOp::Sub) { return &Math::Elementwise::sub_i32_splat_lhs_avx2; }
            } else if constexpr (std::is_same_v<T, int64_t>) {
                if constexpr (Op == ElementwiseOp::Sub) { return &Math::Elementwise::sub_i64_splat_lhs_avx2; }
            }
            return nullptr;
        }

        /**
         * Compile-time Op -> fp16 flat-lane LEFT-splat kernel map (scalar on the left).
         * @brief Resolve the fp16 left-splat kernel for an ISA.
         * @return The kernel, or nullptr for an op with no left-splat kernel.
         */
        template<ElementwiseOp Op, bool Avx512>
        [[nodiscard]] HalfLaneSplatFn f16_lane_lhs_splat_kernel() {
            if constexpr (Op == ElementwiseOp::Add) {
                return Avx512 ? &Math::Elementwise::add_f16_splat_lhs_avx512 : &Math::Elementwise::add_f16_splat_lhs_avx2;
            }
            if constexpr (Op == ElementwiseOp::Sub) {
                return Avx512 ? &Math::Elementwise::sub_f16_splat_lhs_avx512 : &Math::Elementwise::sub_f16_splat_lhs_avx2;
            }
            if constexpr (Op == ElementwiseOp::Mul) {
                return Avx512 ? &Math::Elementwise::mul_f16_splat_lhs_avx512 : &Math::Elementwise::mul_f16_splat_lhs_avx2;
            }
            if constexpr (Op == ElementwiseOp::Div) {
                return Avx512 ? &Math::Elementwise::div_f16_splat_lhs_avx512 : &Math::Elementwise::div_f16_splat_lhs_avx2;
            }
            return nullptr;
        }

        /**
         * Compile-time Op -> bf16 flat-lane LEFT-splat kernel map (scalar on the left).
         * @brief Resolve the bf16 left-splat kernel for an ISA.
         * @return The kernel, or nullptr for an op with no left-splat kernel.
         */
        template<ElementwiseOp Op, bool Avx512>
        [[nodiscard]] HalfLaneSplatFn bf16_lane_lhs_splat_kernel() {
            if constexpr (Op == ElementwiseOp::Add) {
                return Avx512 ? &Math::Elementwise::add_bf16_splat_lhs_avx512 : &Math::Elementwise::add_bf16_splat_lhs_avx2;
            }
            if constexpr (Op == ElementwiseOp::Sub) {
                return Avx512 ? &Math::Elementwise::sub_bf16_splat_lhs_avx512 : &Math::Elementwise::sub_bf16_splat_lhs_avx2;
            }
            if constexpr (Op == ElementwiseOp::Mul) {
                return Avx512 ? &Math::Elementwise::mul_bf16_splat_lhs_avx512 : &Math::Elementwise::mul_bf16_splat_lhs_avx2;
            }
            if constexpr (Op == ElementwiseOp::Div) {
                return Avx512 ? &Math::Elementwise::div_bf16_splat_lhs_avx512 : &Math::Elementwise::div_bf16_splat_lhs_avx2;
            }
            return nullptr;
        }

        /**
         * Both operands dense, same shape, contiguous: the flat index i addresses
         * all three buffers directly, so the AVX512/AVX2 contiguous kernels apply
         * (gated on the probes answered by the kernel TUs — this TU's own
         * IncludedCPUFeatures() copy is blind without the per-file -m flags).
         * Half/packed dtypes are processed as flat 16-bit lane arrays: one lane per
         * element for Float16/BFloat16, two independent lanes per packed element for
         * Float16x2/BFloat16x2 — the same per-lane widen/apply/narrow semantics as
         * the scalar apply, so bulk half processing is bit-identical.
         * Everything without a kernel (broadcast plans never reach here; integer
         * div, Pow, Int4x2) and machines without the features fall through to the
         * scalar apply loop, which is exactly binary_range's exact-plan behavior.
         * @brief Contiguous binary range op with runtime SIMD dispatch.
         * @param out Output buffer.
         * @param a Left operand buffer.
         * @param b Right operand buffer.
         * @param begin First element computed.
         * @param end One past the last element computed.
         */
        template<typename T, ElementwiseOp Op>
        void contig_range(T* out, const T* a, const T* b, const size_t begin, const size_t end) {
            if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t> ||
                          std::is_same_v<T, float16x2_t> || std::is_same_v<T, bfloat16x2_t>) {
                constexpr auto lanes_per_elem = DataTypes::requires_unpack_v<T> ? size_t{2} : size_t{1};
                auto* lane_out = reinterpret_cast<uint16_t*>(out);
                const auto* lane_a = reinterpret_cast<const uint16_t*>(a);
                const auto* lane_b = reinterpret_cast<const uint16_t*>(b);
                const auto lane_begin = begin * lanes_per_elem;
                const auto lane_end = end * lanes_per_elem;
                if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, float16x2_t>) {
                    if (Math::Elementwise::elementwise_avx512_supported()) {
                        if (const auto kernel = f16_lane_kernel<Op, true>(); kernel != nullptr) {
                            kernel(lane_out, lane_a, lane_b, lane_begin, lane_end);
                            return;
                        }
                    }
                    if (Math::Elementwise::elementwise_avx2_supported()) {
                        if (const auto kernel = f16_lane_kernel<Op, false>(); kernel != nullptr) {
                            kernel(lane_out, lane_a, lane_b, lane_begin, lane_end);
                            return;
                        }
                    }
                } else {
                    if (Math::Elementwise::elementwise_avx512_supported()) {
                        if (const auto kernel = bf16_lane_kernel<Op, true>(); kernel != nullptr) {
                            kernel(lane_out, lane_a, lane_b, lane_begin, lane_end);
                            return;
                        }
                    }
                    if (Math::Elementwise::elementwise_avx2_supported()) {
                        if (const auto kernel = bf16_lane_kernel<Op, false>(); kernel != nullptr) {
                            kernel(lane_out, lane_a, lane_b, lane_begin, lane_end);
                            return;
                        }
                    }
                }
            } else if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t> ||
                                 std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> ||
                                 std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t> ||
                                 std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
                                 std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
                using C = canonical_simd_t<T>;
                if (Math::Elementwise::elementwise_avx512_supported()) {
                    if (const auto kernel = native_avx512_kernel<C, Op>(); kernel != nullptr) {
                        kernel(reinterpret_cast<C*>(out), reinterpret_cast<const C*>(a), reinterpret_cast<const C*>(b), begin, end);
                        return;
                    }
                }
                if (Math::Elementwise::elementwise_avx2_supported()) {
                    if (const auto kernel = native_avx2_kernel<C, Op>(); kernel != nullptr) {
                        kernel(reinterpret_cast<C*>(out), reinterpret_cast<const C*>(a), reinterpret_cast<const C*>(b), begin, end);
                        return;
                    }
                }
            } else if constexpr (std::is_same_v<T, int4x2_t>) {
                // Per-nibble wrap kernels over flat bytes (add/sub/mul only — div
                // keeps the scalar per-nibble div-by-zero rule).
                auto* byte_out = reinterpret_cast<uint8_t*>(out);
                const auto* byte_a = reinterpret_cast<const uint8_t*>(a);
                const auto* byte_b = reinterpret_cast<const uint8_t*>(b);
                if (Math::Elementwise::elementwise_avx512_supported()) {
                    if (const auto kernel = i4x2_kernel<Op, true>(); kernel != nullptr) {
                        kernel(byte_out, byte_a, byte_b, begin, end);
                        return;
                    }
                }
                if (Math::Elementwise::elementwise_avx2_supported()) {
                    if (const auto kernel = i4x2_kernel<Op, false>(); kernel != nullptr) {
                        kernel(byte_out, byte_a, byte_b, begin, end);
                        return;
                    }
                }
            } else {
                // Any future dtype: scalar only.
            }
            for (auto i = begin; i < end; ++i) {
                out[i] = apply_any<T, Op>(a[i], b[i]);
            }
        }

        /**
         * Tensor-scalar twin of contig_range: the scalar was narrowed into T by the
         * caller (narrow_scalar's PERMISSIVE contract — applied exactly once, host
         * side) and rides in by const reference. Half dtypes process flat 16-bit
         * lanes exactly like contig_range. Packed dtypes splat only when the caller
         * guarantees uniform lanes (UniformPackedScalar — the tensor-scalar API
         * duplicates the narrowed scalar into both lanes): the broadcast-row
         * callers (bcast_rows_range) pass a full packed element whose lanes may
         * DIFFER, and a one-lane splat cannot broadcast an alternating
         * (lo, hi, lo, hi, ...) pattern — the per-lane scalar apply is the exact
         * semantics there. Int4x2 stays on the scalar apply for the same reason
         * (its per-nibble splat kernels are not wired). Everything without a
         * kernel (integer div, Pow) and machines without the features falls
         * through to the scalar apply loop, which is exactly scalar_range's behavior.
         * @brief Contiguous tensor-scalar range op with runtime SIMD dispatch.
         * @tparam UniformPackedScalar True when the caller guarantees a packed T's
         * scalar has identical lanes (the tensor-scalar API's narrow_scalar).
         * @param out Output buffer.
         * @param a Input buffer.
         * @param scalar Scalar already narrowed into T (by reference).
         * @param begin First element computed.
         * @param end One past the last element computed.
         */
        template<typename T, ElementwiseOp Op, bool UniformPackedScalar = false>
        void contig_splat_range(T* out, const T* a, const T& scalar, const size_t begin, const size_t end) {
            if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t> ||
                          (UniformPackedScalar && DataTypes::requires_unpack_v<T>)) {
                constexpr auto lanes_per_elem = DataTypes::requires_unpack_v<T> ? size_t{2} : size_t{1};
                auto* lane_out = reinterpret_cast<uint16_t*>(out);
                const auto* lane_a = reinterpret_cast<const uint16_t*>(a);
                const auto lane_scalar = Intrinsics::detail::bit_cast_memcpy<uint16_t>(
                    *reinterpret_cast<const uint16_t*>(&scalar));
                const auto lane_begin = begin * lanes_per_elem;
                const auto lane_end = end * lanes_per_elem;
                if constexpr (std::is_same_v<T, float16_t> || (UniformPackedScalar && std::is_same_v<T, float16x2_t>)) {
                    if (Math::Elementwise::elementwise_avx512_supported()) {
                        if (const auto kernel = f16_lane_splat_kernel<Op, true>(); kernel != nullptr) {
                            kernel(lane_out, lane_a, lane_scalar, lane_begin, lane_end);
                            return;
                        }
                    }
                    if (Math::Elementwise::elementwise_avx2_supported()) {
                        if (const auto kernel = f16_lane_splat_kernel<Op, false>(); kernel != nullptr) {
                            kernel(lane_out, lane_a, lane_scalar, lane_begin, lane_end);
                            return;
                        }
                    }
                } else {
                    if (Math::Elementwise::elementwise_avx512_supported()) {
                        if (const auto kernel = bf16_lane_splat_kernel<Op, true>(); kernel != nullptr) {
                            kernel(lane_out, lane_a, lane_scalar, lane_begin, lane_end);
                            return;
                        }
                    }
                    if (Math::Elementwise::elementwise_avx2_supported()) {
                        if (const auto kernel = bf16_lane_splat_kernel<Op, false>(); kernel != nullptr) {
                            kernel(lane_out, lane_a, lane_scalar, lane_begin, lane_end);
                            return;
                        }
                    }
                }
            } else if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t> ||
                                 std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> ||
                                 std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t> ||
                                 std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
                                 std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
                using C = canonical_simd_t<T>;
                const auto c_scalar = *reinterpret_cast<const C*>(&scalar);
                if (Math::Elementwise::elementwise_avx512_supported()) {
                    if (const auto kernel = native_avx512_splat_kernel<C, Op>(); kernel != nullptr) {
                        kernel(reinterpret_cast<C*>(out), reinterpret_cast<const C*>(a), c_scalar, begin, end);
                        return;
                    }
                }
                if (Math::Elementwise::elementwise_avx2_supported()) {
                    if (const auto kernel = native_avx2_splat_kernel<C, Op>(); kernel != nullptr) {
                        kernel(reinterpret_cast<C*>(out), reinterpret_cast<const C*>(a), c_scalar, begin, end);
                        return;
                    }
                }
            } else {
                // Int4x2, non-uniform-lane packed scalars (broadcast rows), and any
                // future dtype: scalar only — per-nibble splat kernels are not wired.
            }
            for (auto i = begin; i < end; ++i) {
                out[i] = apply_any<T, Op>(a[i], scalar);
            }
        }

        /**
         * Scalar-on-the-LEFT twin of contig_splat_range: out[i] = scalar (op) a[i].
         * Used by broadcast rows whose A operand has inner stride 0, where the
         * scalar apply's operand order must be replayed exactly (sub/div differ;
         * commutative fp ops select the same NaN payload). Same gating and lane
         * treatment as contig_splat_range; the fallback is the scalar apply with
         * the scalar in the left slot.
         * @brief Contiguous left-splat range op with runtime SIMD dispatch.
         * @param out Output buffer.
         * @param a Input buffer.
         * @param scalar Scalar already narrowed into T (by reference).
         * @param begin First element computed.
         * @param end One past the last element computed.
         */
        template<typename T, ElementwiseOp Op>
        void contig_lhs_splat_range(T* out, const T* a, const T& scalar, const size_t begin, const size_t end) {
            if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
                auto* lane_out = reinterpret_cast<uint16_t*>(out);
                const auto* lane_a = reinterpret_cast<const uint16_t*>(a);
                const auto lane_scalar = Intrinsics::detail::bit_cast_memcpy<uint16_t>(
                    *reinterpret_cast<const uint16_t*>(&scalar));
                if constexpr (std::is_same_v<T, float16_t>) {
                    if (Math::Elementwise::elementwise_avx512_supported()) {
                        if (const auto kernel = f16_lane_lhs_splat_kernel<Op, true>(); kernel != nullptr) {
                            kernel(lane_out, lane_a, lane_scalar, begin, end);
                            return;
                        }
                    }
                    if (Math::Elementwise::elementwise_avx2_supported()) {
                        if (const auto kernel = f16_lane_lhs_splat_kernel<Op, false>(); kernel != nullptr) {
                            kernel(lane_out, lane_a, lane_scalar, begin, end);
                            return;
                        }
                    }
                } else {
                    if (Math::Elementwise::elementwise_avx512_supported()) {
                        if (const auto kernel = bf16_lane_lhs_splat_kernel<Op, true>(); kernel != nullptr) {
                            kernel(lane_out, lane_a, lane_scalar, begin, end);
                            return;
                        }
                    }
                    if (Math::Elementwise::elementwise_avx2_supported()) {
                        if (const auto kernel = bf16_lane_lhs_splat_kernel<Op, false>(); kernel != nullptr) {
                            kernel(lane_out, lane_a, lane_scalar, begin, end);
                            return;
                        }
                    }
                }
            } else if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t> ||
                                 std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> ||
                                 std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t> ||
                                 std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
                                 std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
                using C = canonical_simd_t<T>;
                const auto c_scalar = *reinterpret_cast<const C*>(&scalar);
                if (Math::Elementwise::elementwise_avx512_supported()) {
                    if (const auto kernel = native_avx512_lhs_splat_kernel<C, Op>(); kernel != nullptr) {
                        kernel(reinterpret_cast<C*>(out), reinterpret_cast<const C*>(a), c_scalar, begin, end);
                        return;
                    }
                }
                if (Math::Elementwise::elementwise_avx2_supported()) {
                    if (const auto kernel = native_avx2_lhs_splat_kernel<C, Op>(); kernel != nullptr) {
                        kernel(reinterpret_cast<C*>(out), reinterpret_cast<const C*>(a), c_scalar, begin, end);
                        return;
                    }
                }
            } else {
                // Int4x2 (and any future dtype): scalar only.
            }
            for (auto i = begin; i < end; ++i) {
                out[i] = apply_any<T, Op>(scalar, a[i]);
            }
        }

        /**
         * Row-decomposition fast path for broadcast plans whose innermost output
         * dim is contiguous (out_strides[last] == 1 — always true for the contiguous
         * outputs this backend receives, verified by the caller) and whose operands'
         * inner-dim strides are each 0 or 1 (contiguous operands guarantee this;
         * anything else falls back to binary_range). Each output row maps to one
         * operand-element pair per operand; walking the OUTER dims (all but the
         * last) linearizes the row index into base offsets via the plan strides.
         * Row classes: (1,1) -> the contiguous binary kernel on the row; (0,1) ->
         * the splat kernel with operand A's row value; (1,0) -> splat with B's
         * value; (0,0) -> degenerate, one scalar apply fills the row.
         * @brief Broadcast range op via per-row kernel decomposition.
         * @param out Output buffer.
         * @param a Left operand buffer.
         * @param b Right operand buffer.
         * @param plan Broadcast execution plan.
         * @param row_begin First output row computed.
         * @param row_end One past the last output row computed.
         */
        template<typename T, ElementwiseOp Op>
        void bcast_rows_range(T* out, const T* a, const T* b, const ElementwiseBcast& plan, const size_t row_begin, const size_t row_end) {
            const auto nd = plan.out_ndim;
            const auto inner = plan.out_shape[nd - 1];
            const auto a_inner = plan.a_strides[nd - 1];
            const auto b_inner = plan.b_strides[nd - 1];
            for (auto row = row_begin; row < row_end; ++row) {
                // Linearize the outer (all but the innermost) output coordinates
                // into each operand's element base offset. out_strides[d] is a
                // multiple of inner for d < nd-1, so the outer index decomposes
                // against out_strides[d] / inner exactly like bcast_offsets does
                // for the full flat index.
                auto a_base = size_t{0};
                auto b_base = size_t{0};
                auto rem = row;
                for (auto d = size_t{0}; d + 1 < nd; ++d) {
                    const auto outer_stride = plan.out_strides[d] / inner;
                    const auto coord = rem / outer_stride;
                    rem -= coord * outer_stride;
                    a_base += coord * plan.a_strides[d];
                    b_base += coord * plan.b_strides[d];
                }
                auto* row_out = out + row * inner;
                if (a_inner == 1 && b_inner == 1) {
                    contig_range<T, Op>(row_out, a + a_base, b + b_base, 0, inner);
                } else if (a_inner == 0 && b_inner == 1) {
                    // A broadcast on the inner dim: out[i] = A (op) b[i] with A's
                    // row value on the LEFT — the left-splat kernels replay the
                    // scalar apply's operand order (plain splat would compute
                    // b[i] (op) A, which differs for sub/div and selects a
                    // different NaN payload for commutative fp ops).
                    contig_lhs_splat_range<T, Op>(row_out, b + b_base, a[a_base], 0, inner);
                } else if (a_inner == 1 && b_inner == 0) {
                    contig_splat_range<T, Op>(row_out, a + a_base, b[b_base], 0, inner);
                } else {
                    const auto v = apply_any<T, Op>(a[a_base], b[b_base]);
                    std::fill_n(row_out, inner, v);
                }
            }
        }

        /**
         * True when the plan can run the row-decomposition path: innermost output
         * dim contiguous and both operands' inner-dim strides in {0, 1}. Anything
         * else keeps the fully scalar binary_range (defensive — contiguous
         * operands never produce such a plan).
         * @brief Is this broadcast plan row-decomposable?
         */
        [[nodiscard]] inline bool row_decomposable(const ElementwiseBcast& plan) {
            const auto last = plan.out_ndim - 1;
            return plan.out_strides[last] == 1 &&
                   (plan.a_strides[last] == 0 || plan.a_strides[last] == 1) &&
                   (plan.b_strides[last] == 0 || plan.b_strides[last] == 1);
        }

        // --------------------------- launch plumbing ---------------------------

        /**
         * Raw device pointers plus view offsets, pre-extracted by the friend
         * entry points (these helpers are not friends of Tensor). numel is the
         * flat output element count.
         * @brief Bundle of pre-extracted tensor buffers.
         */
        struct RawPtrs {
            const void* a;
            size_t a_off;
            const void* b;
            size_t b_off;
            void* out;
            size_t out_off;
            size_t numel;
        };

        /**
         * Local twin of linearalgebra_shared::launch_elementwise with CCD-aware
         * scheduling: when the output span being written fits one CCD's LLC the
         * chunks are submitted to the calling thread's CCD group (submit_to_group)
         * so the cache-resident destination is written locally; otherwise they go
         * to the whole pool exactly as before. The chunk count still assumes all
         * t_count threads — group workers simply claim multiple chunks from the
         * queue, so restricting the submit never starves the decomposition.
         * @brief Threaded elementwise launcher with CCD-local routing.
         * @param worker Callable taking (begin, end); copied into each task.
         * @param numel Total element count to cover.
         * @param out_bytes Bytes of the output buffer actually written by the workers.
         * @param a First tensor the workers read.
         * @param b Second tensor the workers read.
         * @param out Tensor the workers write.
         */
        template<typename Worker>
        void launch_elementwise_ccd(Worker&& worker, const size_t numel, const size_t out_bytes,
                                    const Tensor& a, const Tensor& b, const Tensor& out) {
            const auto t_count = static_cast<size_t>(IncludedCPUFeatures().threads);
            if (t_count <= 1 || numel <= t_count * 1024) {
                worker(0, numel);
                return;
            }
            auto& threadMgr = Threading::GetCPUThreadPool();
            std::vector<std::shared_future<void>> futures;
            const auto chunk = (numel + t_count - 1) / t_count;
            const bool keep_local = KeepWorkLocal(out_bytes);
            const auto group = keep_local ? CcdGroupOfCurrentCpu() : uint32_t{0};
            for (auto t = size_t{0}; t < t_count; ++t) {
                const auto begin = t * chunk;
                if (begin >= numel) {
                    break;
                }
                // worker is captured BY VALUE: these futures outlive this call
                // (they land on the tensors), so a reference would dangle.
                if (keep_local) {
                    futures.push_back(threadMgr.submit_to_group([worker, begin, end = std::min(begin + chunk, numel)] { worker(begin, end); }, group, Threading::TaskPriority::Normal).share());
                } else {
                    futures.push_back(threadMgr.submit([worker, begin, end = std::min(begin + chunk, numel)] { worker(begin, end); }, Threading::TaskPriority::Normal).share());
                }
            }
            a.give_futures(futures);
            b.give_futures(futures);
            out.give_futures(futures);
        }

        /**
         * Exact plan (identical shapes): both operands are dense and contiguous,
         * so the flat index addresses all three buffers and the contiguous SIMD
         * dispatch applies. Non-exact plans with a row-decomposable shape (inner
         * output stride 1, operand inner strides in {0, 1} — the only shapes
         * contiguous operands produce) chunk by OUTPUT ROW across the pool: each
         * row classifies onto the contiguous binary kernel (both dense), the
         * splat kernel (one operand broadcast on the inner dim), or a scalar
         * fill (both broadcast). Anything else keeps the scalar binary_range.
         * @brief Launch a broadcast binary op for element type T on the thread pool.
         */
        template<typename T, ElementwiseOp Op>
        void launch_arith(const RawPtrs& raw, const ElementwiseBcast& plan, const Tensor& a, const Tensor& b, const Tensor& out) {
            auto* out_ptr = static_cast<T*>(raw.out) + raw.out_off;
            const auto* a_ptr = static_cast<const T*>(raw.a) + raw.a_off;
            const auto* b_ptr = static_cast<const T*>(raw.b) + raw.b_off;
            if (plan.exact) {
                /**
                 * Forwards the captured elementwise pointers to the contiguous range kernel.
                 * @brief Elementwise contiguous binary range worker (SIMD or scalar).
                 * @param x First element computed.
                 * @param y One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise_ccd([=](const size_t x, const size_t y) { contig_range<T, Op>(out_ptr, a_ptr, b_ptr, x, y); }, raw.numel, raw.numel * sizeof(T), a, b, out);
            } else if (row_decomposable(plan)) {
                /**
                 * Forwards the captured elementwise pointers to the row-decomposition kernel.
                 * @brief Elementwise broadcast row-decomposition worker (SIMD or scalar).
                 * @param x First output row computed.
                 * @param y One past the last output row computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise_ccd([=](const size_t x, const size_t y) { bcast_rows_range<T, Op>(out_ptr, a_ptr, b_ptr, plan, x, y); }, raw.numel / plan.out_shape[plan.out_ndim - 1], raw.numel * sizeof(T), a, b, out);
            } else {
                /**
                 * Forwards the captured elementwise pointers to the range kernel.
                 * @brief Elementwise binary range worker.
                 * @param x First element computed.
                 * @param y One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise_ccd([=](const size_t x, const size_t y) { binary_range<T, Op>(out_ptr, a_ptr, b_ptr, plan, x, y); }, raw.numel, raw.numel * sizeof(T), a, b, out);
            }
        }

        /**
         * The scalar is narrowed on the host at capture time (narrow_scalar's
         * PERMISSIVE contract, exactly once) and rides into the worker BY VALUE
         * (the futures outlive this call — a pointer to a stack local would
         * dangle). The worker runs the same contiguous SIMD dispatch as the
         * binary path, with the narrowed scalar as the broadcast operand.
         * @brief Launch a tensor-scalar op for element type T on the thread pool.
         */
        template<typename T, ElementwiseOp Op>
        void launch_scalar(const RawPtrs& raw, const float64_t scalar, const Tensor& a, const Tensor& out) {
            auto* out_ptr = static_cast<T*>(raw.out) + raw.out_off;
            const auto* a_ptr = static_cast<const T*>(raw.a) + raw.a_off;
            /**
             * Forwards the captured elementwise pointers to the range kernel.
             * @brief Elementwise tensor-scalar range worker.
             * @param x First element computed.
             * @param y One past the last element computed.
             * @note Lambda subcall. Self-execution prohibited.
             */
            launch_elementwise_ccd([out_ptr, a_ptr, narrowed = narrow_scalar<T>(scalar)](const size_t x, const size_t y) { contig_splat_range<T, Op, true>(out_ptr, a_ptr, narrowed, x, y); }, raw.numel, raw.numel * sizeof(T), a, a, out);
        }

        /**
         * @brief Launch a unary op for element type T on the thread pool.
         */
        template<typename T, ElementwiseUnaryOp Op>
        void launch_unary(const RawPtrs& raw, const Tensor& a, const Tensor& out) {
            auto* out_ptr = static_cast<T*>(raw.out) + raw.out_off;
            const auto* in_ptr = static_cast<const T*>(raw.a) + raw.a_off;
            /**
             * Forwards the captured elementwise pointers to the range kernel.
             * @brief Elementwise unary range worker.
             * @param x First element computed.
             * @param y One past the last element computed.
             * @note Lambda subcall. Self-execution prohibited.
             */
            launch_elementwise_ccd([=](const size_t x, const size_t y) { unary_range<T, Op>(out_ptr, in_ptr, x, y); }, raw.numel, raw.numel * sizeof(T), a, a, out);
        }

        // --------------------------- dtype dispatch ---------------------------

        /**
         * @brief Dispatch a broadcast binary arithmetic op on the input dtype.
         */
        template<ElementwiseOp Op>
        void dispatch_arith(const DataTypes::DType& dtype, const RawPtrs& raw, const ElementwiseBcast& plan, const Tensor& a, const Tensor& b, const Tensor& out) {
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: launch_arith<float32_t, Op>(raw, plan, a, b, out); break;
                case DType::Float64: launch_arith<float64_t, Op>(raw, plan, a, b, out); break;
                case DType::Float16: launch_arith<float16_t, Op>(raw, plan, a, b, out); break;
                case DType::BFloat16: launch_arith<bfloat16_t, Op>(raw, plan, a, b, out); break;
                case DType::Float16x2: launch_arith<float16x2_t, Op>(raw, plan, a, b, out); break;
                case DType::BFloat16x2: launch_arith<bfloat16x2_t, Op>(raw, plan, a, b, out); break;
                case DType::Int8: launch_arith<int8_t, Op>(raw, plan, a, b, out); break;
                case DType::UInt8: launch_arith<uint8_t, Op>(raw, plan, a, b, out); break;
                case DType::Int16: launch_arith<int16_t, Op>(raw, plan, a, b, out); break;
                case DType::UInt16: launch_arith<uint16_t, Op>(raw, plan, a, b, out); break;
                case DType::Int32: launch_arith<int32_t, Op>(raw, plan, a, b, out); break;
                case DType::UInt32: launch_arith<uint32_t, Op>(raw, plan, a, b, out); break;
                case DType::Int64: launch_arith<int64_t, Op>(raw, plan, a, b, out); break;
                case DType::UInt64: launch_arith<uint64_t, Op>(raw, plan, a, b, out); break;
                case DType::Int4x2: launch_arith<int4x2_t, Op>(raw, plan, a, b, out); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
        }

        /**
         * @brief Dispatch a tensor-scalar arithmetic op on the input dtype.
         */
        template<ElementwiseOp Op>
        void dispatch_scalar(const DataTypes::DType& dtype, const RawPtrs& raw, const float64_t scalar, const Tensor& a, const Tensor& out) {
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: launch_scalar<float32_t, Op>(raw, scalar, a, out); break;
                case DType::Float64: launch_scalar<float64_t, Op>(raw, scalar, a, out); break;
                case DType::Float16: launch_scalar<float16_t, Op>(raw, scalar, a, out); break;
                case DType::BFloat16: launch_scalar<bfloat16_t, Op>(raw, scalar, a, out); break;
                case DType::Float16x2: launch_scalar<float16x2_t, Op>(raw, scalar, a, out); break;
                case DType::BFloat16x2: launch_scalar<bfloat16x2_t, Op>(raw, scalar, a, out); break;
                case DType::Int8: launch_scalar<int8_t, Op>(raw, scalar, a, out); break;
                case DType::UInt8: launch_scalar<uint8_t, Op>(raw, scalar, a, out); break;
                case DType::Int16: launch_scalar<int16_t, Op>(raw, scalar, a, out); break;
                case DType::UInt16: launch_scalar<uint16_t, Op>(raw, scalar, a, out); break;
                case DType::Int32: launch_scalar<int32_t, Op>(raw, scalar, a, out); break;
                case DType::UInt32: launch_scalar<uint32_t, Op>(raw, scalar, a, out); break;
                case DType::Int64: launch_scalar<int64_t, Op>(raw, scalar, a, out); break;
                case DType::UInt64: launch_scalar<uint64_t, Op>(raw, scalar, a, out); break;
                case DType::Int4x2: launch_scalar<int4x2_t, Op>(raw, scalar, a, out); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
        }

        /**
         * Float kinds only — integer exp/log is rejected at the Tensor method,
         * so those dtypes fall into the default arm.
         * @brief Dispatch a unary op on the input dtype.
         */
        template<ElementwiseUnaryOp Op>
        void dispatch_unary(const DataTypes::DType& dtype, const RawPtrs& raw, const Tensor& a, const Tensor& out) {
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: launch_unary<float32_t, Op>(raw, a, out); break;
                case DType::Float64: launch_unary<float64_t, Op>(raw, a, out); break;
                case DType::Float16: launch_unary<float16_t, Op>(raw, a, out); break;
                case DType::BFloat16: launch_unary<bfloat16_t, Op>(raw, a, out); break;
                case DType::Float16x2: launch_unary<float16x2_t, Op>(raw, a, out); break;
                case DType::BFloat16x2: launch_unary<bfloat16x2_t, Op>(raw, a, out); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
        }
    }

    /**
     * Work is chunked across the CPU thread pool and the futures land on all
     * three tensors.
     * @brief CPU tensor-tensor elementwise: out = a (op) b with broadcasting.
     * @param a Left operand, CPU resident, contiguous.
     * @param b Right operand, CPU resident, contiguous.
     * @param out Output tensor, broadcast shape, a's dtype.
     * @param op Operation selector.
     * @param plan Broadcast execution plan built by the Tensor method.
     */
    void elementwise_cpu(const Tensor& a, const Tensor& b, const Tensor& out, const ElementwiseOp op, const ElementwiseBcast& plan) {
        if (a.device != Device::CPU || b.device != Device::CPU || out.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(a.device));
        }
        const RawPtrs raw{.a = a.data, .a_off = a.offset, .b = b.data, .b_off = b.offset, .out = out.data, .out_off = out.offset, .numel = plan.out_numel};
        switch (op) {
            case ElementwiseOp::Add: dispatch_arith<ElementwiseOp::Add>(a.dtype, raw, plan, a, b, out); break;
            case ElementwiseOp::Sub: dispatch_arith<ElementwiseOp::Sub>(a.dtype, raw, plan, a, b, out); break;
            case ElementwiseOp::Mul: dispatch_arith<ElementwiseOp::Mul>(a.dtype, raw, plan, a, b, out); break;
            case ElementwiseOp::Div: dispatch_arith<ElementwiseOp::Div>(a.dtype, raw, plan, a, b, out); break;
            case ElementwiseOp::Pow: dispatch_arith<ElementwiseOp::Pow>(a.dtype, raw, plan, a, b, out); break;
            [[unlikely]] default: throw ErrorHandling::UnReachableCode();
        }
    }

    /**
     * The scalar is narrowed into the input dtype per the PERMISSIVE convert
     * contract (see narrow_scalar) before dispatch. Futures land on a and out.
     * @brief CPU tensor-scalar elementwise: out[i] = a[i] (op) scalar.
     * @param a Input tensor, CPU resident, contiguous.
     * @param scalar Scalar operand (float64_t, narrowed into a's dtype).
     * @param out Output tensor, same shape as a, a's dtype.
     * @param op Operation selector.
     */
    void elementwise_scalar_cpu(const Tensor& a, const float64_t& scalar, const Tensor& out, const ElementwiseOp op) {
        if (a.device != Device::CPU || out.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(a.device));
        }
        const RawPtrs raw{.a = a.data, .a_off = a.offset, .b = nullptr, .b_off = 0, .out = out.data, .out_off = out.offset, .numel = a.numel};
        switch (op) {
            case ElementwiseOp::Add: dispatch_scalar<ElementwiseOp::Add>(a.dtype, raw, scalar, a, out); break;
            case ElementwiseOp::Sub: dispatch_scalar<ElementwiseOp::Sub>(a.dtype, raw, scalar, a, out); break;
            case ElementwiseOp::Mul: dispatch_scalar<ElementwiseOp::Mul>(a.dtype, raw, scalar, a, out); break;
            case ElementwiseOp::Div: dispatch_scalar<ElementwiseOp::Div>(a.dtype, raw, scalar, a, out); break;
            case ElementwiseOp::Pow: dispatch_scalar<ElementwiseOp::Pow>(a.dtype, raw, scalar, a, out); break;
            [[unlikely]] default: throw ErrorHandling::UnReachableCode();
        }
    }

    /**
     * @brief CPU unary elementwise: out[i] = fn(a[i]) (exp/log, float kinds only).
     * @param a Input tensor, CPU resident, contiguous.
     * @param out Output tensor, same shape and dtype as a.
     * @param op Operation selector.
     */
    void elementwise_unary_cpu(const Tensor& a, const Tensor& out, const ElementwiseUnaryOp op) {
        if (a.device != Device::CPU || out.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(a.device));
        }
        const RawPtrs raw{.a = a.data, .a_off = a.offset, .b = nullptr, .b_off = 0, .out = out.data, .out_off = out.offset, .numel = a.numel};
        switch (op) {
            case ElementwiseUnaryOp::Exp: dispatch_unary<ElementwiseUnaryOp::Exp>(a.dtype, raw, a, out); break;
            case ElementwiseUnaryOp::Log: dispatch_unary<ElementwiseUnaryOp::Log>(a.dtype, raw, a, out); break;
            case ElementwiseUnaryOp::Sigmoid: dispatch_unary<ElementwiseUnaryOp::Sigmoid>(a.dtype, raw, a, out); break;
            case ElementwiseUnaryOp::Tanh: dispatch_unary<ElementwiseUnaryOp::Tanh>(a.dtype, raw, a, out); break;
            [[unlikely]] default: throw ErrorHandling::UnReachableCode();
        }
    }
}

namespace BeanTensor::Tensors {

    /**
     * Validates dtype/device/contiguity, rejects integer pow, syncs both
     * operands, builds the broadcast plan, then routes to the CPU or GPU
     * backend.
     * @brief Tensor-tensor elementwise op with broadcasting.
     * @param o Right operand, same dtype and device.
     * @param op Operation selector.
     * @return New tensor with the broadcast shape and the input dtype.
     */
    Tensor Tensor::elementwise_binary(const Tensor& o, const ElementwiseOp op) const {
        if (this->dtype != o.dtype) {
            throw ErrorHandling::MismatchedDTypes();
        }
        if (this->device != o.device) {
            throw ErrorHandling::MismatchedDevices();
        }
        if (!this->is_contiguous || !o.is_contiguous) {
            throw ErrorHandling::NotImplemented();
        }
        if (op == ElementwiseOp::Pow && !detail::elementwise_shared::is_float_kind(this->dtype)) {
            throw ErrorHandling::NotImplemented();
        }
        sync();
        o.sync();
        auto out_shape = std::vector<size_t>{};
        const auto plan = detail::compute_bcast_plan(this->shape, this->dims, this->ndim, o.shape, o.dims, o.ndim, out_shape);
        auto out = Tensor(out_shape, this->dtype, this->device, false);
        if (this->device == Device::CPU) {
            detail::elementwise_cpu(*this, o, out, op, plan);
        } else if (this->device == Device::GPU) {
            detail::elementwise_gpu(*this, o, out, op, plan);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        switch (op) {
            case ElementwiseOp::Add: out.wire_node(detail::add_g_node, *this, &o); break;
            case ElementwiseOp::Sub: out.wire_node(detail::sub_g_node, *this, &o); break;
            case ElementwiseOp::Mul: out.wire_node(detail::mul_g_node, *this, &o); break;
            case ElementwiseOp::Div: out.wire_node(detail::div_g_node, *this, &o); break;
            case ElementwiseOp::Pow: out.wire_node(detail::pow_g_node, *this, &o); break;
            [[unlikely]] default: throw ErrorHandling::UnReachableCode();
        }
        return out;
    }

    /**
     * Validates contiguity, rejects integer pow, syncs this tensor, then routes
     * to the CPU or GPU backend.
     * @brief Tensor-scalar elementwise op.
     * @param scalar Scalar operand (float64_t, narrowed into this tensor's dtype).
     * @param op Operation selector.
     * @return New tensor shaped like this.
     */
    Tensor Tensor::elementwise_scalar(const float64_t& scalar, const ElementwiseOp op) const {
        if (!this->is_contiguous) {
            throw ErrorHandling::NotImplemented();
        }
        if (op == ElementwiseOp::Pow && !detail::elementwise_shared::is_float_kind(this->dtype)) {
            throw ErrorHandling::NotImplemented();
        }
        sync();
        auto out = Tensor(std::vector(this->shape, this->shape + this->ndim), this->dtype, this->device, false);
        if (this->device == Device::CPU) {
            detail::elementwise_scalar_cpu(*this, scalar, out, op);
        } else if (this->device == Device::GPU) {
            detail::elementwise_scalar_gpu(*this, scalar, out, op);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        switch (op) {
            case ElementwiseOp::Add: out.wire_node(detail::add_scalar_g_node, *this); break;
            case ElementwiseOp::Sub: out.wire_node(detail::sub_scalar_g_node, *this); break;
            case ElementwiseOp::Mul: out.wire_node(detail::mul_scalar_g_node, *this); break;
            case ElementwiseOp::Div: out.wire_node(detail::div_scalar_g_node, *this); break;
            case ElementwiseOp::Pow: out.wire_node(detail::pow_scalar_g_node, *this); break;
            [[unlikely]] default: throw ErrorHandling::UnReachableCode();
        }
        if (out.node != nullptr) {
            // The tensor-scalar backward trampolines read the operand from the node.
            out.node->_scalar = scalar;
        }
        return out;
    }

    /**
     * Validates contiguity and float-kind dtype, syncs this tensor, then routes
     * to the CPU or GPU backend.
     * @brief Unary elementwise op (exp/log).
     * @param op Operation selector.
     * @return New tensor shaped like this.
     */
    Tensor Tensor::elementwise_unary(const ElementwiseUnaryOp op) const {
        if (!this->is_contiguous) {
            throw ErrorHandling::NotImplemented();
        }
        if (!detail::elementwise_shared::is_float_kind(this->dtype)) {
            throw ErrorHandling::NotImplemented();
        }
        sync();
        auto out = Tensor(std::vector(this->shape, this->shape + this->ndim), this->dtype, this->device, false);
        if (this->device == Device::CPU) {
            detail::elementwise_unary_cpu(*this, out, op);
        } else if (this->device == Device::GPU) {
            detail::elementwise_unary_gpu(*this, out, op);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        switch (op) {
            case ElementwiseUnaryOp::Exp: out.wire_node(detail::exp_g_node, *this); break;
            case ElementwiseUnaryOp::Log: out.wire_node(detail::log_g_node, *this); break;
            case ElementwiseUnaryOp::Sigmoid: out.wire_node(detail::sigmoid_g_node, *this); break;
            case ElementwiseUnaryOp::Tanh: out.wire_node(detail::tanh_g_node, *this); break;
            [[unlikely]] default: throw ErrorHandling::UnReachableCode();
        }
        return out;
    }

    Tensor Tensor::add(const Tensor& o) const { return elementwise_binary(o, ElementwiseOp::Add); }
    Tensor Tensor::add(const float64_t& scalar) const { return elementwise_scalar(scalar, ElementwiseOp::Add); }
    Tensor Tensor::sub(const Tensor& o) const { return elementwise_binary(o, ElementwiseOp::Sub); }
    Tensor Tensor::sub(const float64_t& scalar) const { return elementwise_scalar(scalar, ElementwiseOp::Sub); }
    Tensor Tensor::mul(const Tensor& o) const { return elementwise_binary(o, ElementwiseOp::Mul); }
    Tensor Tensor::mul(const float64_t& scalar) const { return elementwise_scalar(scalar, ElementwiseOp::Mul); }
    Tensor Tensor::div(const Tensor& o) const { return elementwise_binary(o, ElementwiseOp::Div); }
    Tensor Tensor::div(const float64_t& scalar) const { return elementwise_scalar(scalar, ElementwiseOp::Div); }
    Tensor Tensor::pow(const Tensor& o) const { return elementwise_binary(o, ElementwiseOp::Pow); }
    Tensor Tensor::pow(const float64_t& scalar) const { return elementwise_scalar(scalar, ElementwiseOp::Pow); }
    Tensor Tensor::exp() const { return elementwise_unary(ElementwiseUnaryOp::Exp); }
    Tensor Tensor::log() const { return elementwise_unary(ElementwiseUnaryOp::Log); }
    Tensor Tensor::sigmoid() const { return elementwise_unary(ElementwiseUnaryOp::Sigmoid); }
    Tensor Tensor::tanh() const { return elementwise_unary(ElementwiseUnaryOp::Tanh); }
}
