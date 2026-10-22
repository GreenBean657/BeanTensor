/**
 * @file ElementwiseCPU.cpp
 * @author GreenBean
 * @brief CPU backend for elementwise arithmetic/comparison ops with broadcasting.
 * Hosts the Tensor::add/sub/mul/div/pow/exp/log/eq/ne method definitions
 * (mirrors how Tensor::bias_add lives in BiasAddCPU.cpp).
 * @version 0.1
 * @date 2026-10-22
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "Math/Elementwise/Elementwise.h"
#include "Math/Elementwise/ElementwiseShared.h"
#include "Tensor/Tensor.h"

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
         * @param a_shape/a_dims/a_ndim Left operand shape, element strides, rank.
         * @param b_shape/b_dims/b_ndim Right operand shape, element strides, rank.
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
         * Single-rounding native op; Div is IEEE (x/0 -> inf/nan), Pow is std::pow.
         * @brief Scalar binary apply (fp32/fp64).
         */
        template<typename T, ElementwiseOp Op>
        [[nodiscard]] T apply_native(const T a, const T b) {
            if constexpr (Op == ElementwiseOp::Add) {
                return a + b;
            } else if constexpr (Op == ElementwiseOp::Sub) {
                return a - b;
            } else if constexpr (Op == ElementwiseOp::Mul) {
                return a * b;
            } else if constexpr (Op == ElementwiseOp::Div) {
                return a / b;
            } else {
                return std::pow(a, b);
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
            if constexpr (Op == ElementwiseOp::Add) {
                return narrow_lane<H>(wa + wb);
            } else if constexpr (Op == ElementwiseOp::Sub) {
                return narrow_lane<H>(wa - wb);
            } else if constexpr (Op == ElementwiseOp::Mul) {
                return narrow_lane<H>(wa * wb);
            } else if constexpr (Op == ElementwiseOp::Div) {
                return narrow_lane<H>(wa / wb);
            } else {
                return narrow_lane<H>(std::pow(wa, wb));
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
            } else {
                return std::log(v);
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
            } else {
                return narrow_lane<H>(std::log(wide));
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
         * Writes 1/0 per element; NaN never compares equal and -0 equals +0
         * (see values_equal).
         * @brief Scalar range comparison with broadcast offsets (any dtype, Eq/Ne).
         * @param out UInt8 output buffer.
         * @param a Left operand buffer.
         * @param b Right operand buffer.
         * @param plan Broadcast execution plan.
         * @param begin First element computed.
         * @param end One past the last element computed.
         */
        template<typename T, ElementwiseOp Op>
        void compare_range(uint8_t* out, const T* a, const T* b, const ElementwiseBcast& plan, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                const auto [ao, bo] = bcast_offsets(plan, i);
                const auto equal = values_equal(a[ao], b[bo]);
                if constexpr (Op == ElementwiseOp::Eq) {
                    out[i] = equal ? uint8_t{1} : uint8_t{0};
                } else {
                    out[i] = equal ? uint8_t{0} : uint8_t{1};
                }
            }
        }

        /**
         * No plan needed: a is contiguous and the output shares its shape, so
         * both flat indices are i. The scalar rides along by value.
         * @brief Scalar range tensor-scalar op (any dtype, arithmetic ops).
         * @param out Output buffer.
         * @param a Input buffer.
         * @param scalar Scalar already narrowed into T.
         * @param begin First element computed.
         * @param end One past the last element computed.
         */
        template<typename T, ElementwiseOp Op>
        void scalar_range(T* out, const T* a, const T scalar, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                out[i] = apply_any<T, Op>(a[i], scalar);
            }
        }

        /**
         * @brief Scalar range tensor-scalar comparison (any dtype, Eq/Ne, UInt8 output).
         * @param out UInt8 output buffer.
         * @param a Input buffer.
         * @param scalar Scalar already narrowed into T.
         * @param begin First element computed.
         * @param end One past the last element computed.
         */
        template<typename T, ElementwiseOp Op>
        void compare_scalar_range(uint8_t* out, const T* a, const T scalar, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                const auto equal = values_equal(a[i], scalar);
                if constexpr (Op == ElementwiseOp::Eq) {
                    out[i] = equal ? uint8_t{1} : uint8_t{0};
                } else {
                    out[i] = equal ? uint8_t{0} : uint8_t{1};
                }
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
         * @brief Launch a broadcast binary op for element type T on the thread pool.
         */
        template<typename T, ElementwiseOp Op>
        void launch_arith(const RawPtrs& raw, const ElementwiseBcast& plan, const Tensor& a, const Tensor& b, const Tensor& out) {
            auto* out_ptr = static_cast<T*>(raw.out) + raw.out_off;
            const auto* a_ptr = static_cast<const T*>(raw.a) + raw.a_off;
            const auto* b_ptr = static_cast<const T*>(raw.b) + raw.b_off;
            /**
             * Forwards the captured elementwise pointers to the range kernel.
             * @brief Elementwise binary range worker.
             * @param x First element computed.
             * @param y One past the last element computed.
             * @note Lambda subcall. Self-execution prohibited.
             */
            launch_elementwise([=](const size_t x, const size_t y) { binary_range<T, Op>(out_ptr, a_ptr, b_ptr, plan, x, y); }, raw.numel, a, b, out);
        }

        /**
         * @brief Launch a broadcast comparison (UInt8 output) for element type T on the thread pool.
         */
        template<typename T, ElementwiseOp Op>
        void launch_compare(const RawPtrs& raw, const ElementwiseBcast& plan, const Tensor& a, const Tensor& b, const Tensor& out) {
            auto* out_ptr = static_cast<uint8_t*>(raw.out) + raw.out_off;
            const auto* a_ptr = static_cast<const T*>(raw.a) + raw.a_off;
            const auto* b_ptr = static_cast<const T*>(raw.b) + raw.b_off;
            /**
             * Forwards the captured elementwise pointers to the range kernel.
             * @brief Elementwise comparison range worker.
             * @param x First element computed.
             * @param y One past the last element computed.
             * @note Lambda subcall. Self-execution prohibited.
             */
            launch_elementwise([=](const size_t x, const size_t y) { compare_range<T, Op>(out_ptr, a_ptr, b_ptr, plan, x, y); }, raw.numel, a, b, out);
        }

        /**
         * The scalar is narrowed on the host at capture time and rides into the
         * worker BY VALUE (the futures outlive this call — a pointer to a stack
         * local would dangle).
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
            launch_elementwise([out_ptr, a_ptr, narrowed = narrow_scalar<T>(scalar)](const size_t x, const size_t y) { scalar_range<T, Op>(out_ptr, a_ptr, narrowed, x, y); }, raw.numel, a, a, out);
        }

        /**
         * @brief Launch a tensor-scalar comparison (UInt8 output) for element type T on the thread pool.
         */
        template<typename T, ElementwiseOp Op>
        void launch_compare_scalar(const RawPtrs& raw, const float64_t scalar, const Tensor& a, const Tensor& out) {
            auto* out_ptr = static_cast<uint8_t*>(raw.out) + raw.out_off;
            const auto* a_ptr = static_cast<const T*>(raw.a) + raw.a_off;
            /**
             * Forwards the captured elementwise pointers to the range kernel.
             * @brief Elementwise tensor-scalar comparison range worker.
             * @param x First element computed.
             * @param y One past the last element computed.
             * @note Lambda subcall. Self-execution prohibited.
             */
            launch_elementwise([out_ptr, a_ptr, narrowed = narrow_scalar<T>(scalar)](const size_t x, const size_t y) { compare_scalar_range<T, Op>(out_ptr, a_ptr, narrowed, x, y); }, raw.numel, a, a, out);
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
            launch_elementwise([=](const size_t x, const size_t y) { unary_range<T, Op>(out_ptr, in_ptr, x, y); }, raw.numel, a, a, out);
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
         * @brief Dispatch a broadcast comparison (UInt8 output) on the input dtype.
         */
        template<ElementwiseOp Op>
        void dispatch_compare(const DataTypes::DType& dtype, const RawPtrs& raw, const ElementwiseBcast& plan, const Tensor& a, const Tensor& b, const Tensor& out) {
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: launch_compare<float32_t, Op>(raw, plan, a, b, out); break;
                case DType::Float64: launch_compare<float64_t, Op>(raw, plan, a, b, out); break;
                case DType::Float16: launch_compare<float16_t, Op>(raw, plan, a, b, out); break;
                case DType::BFloat16: launch_compare<bfloat16_t, Op>(raw, plan, a, b, out); break;
                case DType::Float16x2: launch_compare<float16x2_t, Op>(raw, plan, a, b, out); break;
                case DType::BFloat16x2: launch_compare<bfloat16x2_t, Op>(raw, plan, a, b, out); break;
                case DType::Int8: launch_compare<int8_t, Op>(raw, plan, a, b, out); break;
                case DType::UInt8: launch_compare<uint8_t, Op>(raw, plan, a, b, out); break;
                case DType::Int16: launch_compare<int16_t, Op>(raw, plan, a, b, out); break;
                case DType::UInt16: launch_compare<uint16_t, Op>(raw, plan, a, b, out); break;
                case DType::Int32: launch_compare<int32_t, Op>(raw, plan, a, b, out); break;
                case DType::UInt32: launch_compare<uint32_t, Op>(raw, plan, a, b, out); break;
                case DType::Int64: launch_compare<int64_t, Op>(raw, plan, a, b, out); break;
                case DType::UInt64: launch_compare<uint64_t, Op>(raw, plan, a, b, out); break;
                case DType::Int4x2: launch_compare<int4x2_t, Op>(raw, plan, a, b, out); break;
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
         * @brief Dispatch a tensor-scalar comparison (UInt8 output) on the input dtype.
         */
        template<ElementwiseOp Op>
        void dispatch_compare_scalar(const DataTypes::DType& dtype, const RawPtrs& raw, const float64_t scalar, const Tensor& a, const Tensor& out) {
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: launch_compare_scalar<float32_t, Op>(raw, scalar, a, out); break;
                case DType::Float64: launch_compare_scalar<float64_t, Op>(raw, scalar, a, out); break;
                case DType::Float16: launch_compare_scalar<float16_t, Op>(raw, scalar, a, out); break;
                case DType::BFloat16: launch_compare_scalar<bfloat16_t, Op>(raw, scalar, a, out); break;
                case DType::Float16x2: launch_compare_scalar<float16x2_t, Op>(raw, scalar, a, out); break;
                case DType::BFloat16x2: launch_compare_scalar<bfloat16x2_t, Op>(raw, scalar, a, out); break;
                case DType::Int8: launch_compare_scalar<int8_t, Op>(raw, scalar, a, out); break;
                case DType::UInt8: launch_compare_scalar<uint8_t, Op>(raw, scalar, a, out); break;
                case DType::Int16: launch_compare_scalar<int16_t, Op>(raw, scalar, a, out); break;
                case DType::UInt16: launch_compare_scalar<uint16_t, Op>(raw, scalar, a, out); break;
                case DType::Int32: launch_compare_scalar<int32_t, Op>(raw, scalar, a, out); break;
                case DType::UInt32: launch_compare_scalar<uint32_t, Op>(raw, scalar, a, out); break;
                case DType::Int64: launch_compare_scalar<int64_t, Op>(raw, scalar, a, out); break;
                case DType::UInt64: launch_compare_scalar<uint64_t, Op>(raw, scalar, a, out); break;
                case DType::Int4x2: launch_compare_scalar<int4x2_t, Op>(raw, scalar, a, out); break;
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
     * Routes Eq/Ne to the comparison dispatch and everything else to the
     * arithmetic dispatch. Work is chunked across the CPU thread pool and the
     * futures land on all three tensors.
     * @brief CPU tensor-tensor elementwise: out = a (op) b with broadcasting.
     * @param a Left operand, CPU resident, contiguous.
     * @param b Right operand, CPU resident, contiguous.
     * @param out Output tensor, broadcast shape; UInt8 dtype for Eq/Ne, a's dtype otherwise.
     * @param op Operation selector.
     * @param plan Broadcast execution plan built by the Tensor method.
     */
    void elementwise_cpu(const Tensor& a, const Tensor& b, Tensor& out, const ElementwiseOp op, const ElementwiseBcast& plan) {
        if (a.device != Device::CPU || b.device != Device::CPU || out.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(a.device));
        }
        const RawPtrs raw{a.data, a.offset, b.data, b.offset, out.data, out.offset, plan.out_numel};
        switch (op) {
            case ElementwiseOp::Add: dispatch_arith<ElementwiseOp::Add>(a.dtype, raw, plan, a, b, out); break;
            case ElementwiseOp::Sub: dispatch_arith<ElementwiseOp::Sub>(a.dtype, raw, plan, a, b, out); break;
            case ElementwiseOp::Mul: dispatch_arith<ElementwiseOp::Mul>(a.dtype, raw, plan, a, b, out); break;
            case ElementwiseOp::Div: dispatch_arith<ElementwiseOp::Div>(a.dtype, raw, plan, a, b, out); break;
            case ElementwiseOp::Pow: dispatch_arith<ElementwiseOp::Pow>(a.dtype, raw, plan, a, b, out); break;
            case ElementwiseOp::Eq: dispatch_compare<ElementwiseOp::Eq>(a.dtype, raw, plan, a, b, out); break;
            case ElementwiseOp::Ne: dispatch_compare<ElementwiseOp::Ne>(a.dtype, raw, plan, a, b, out); break;
            [[unlikely]] default: throw ErrorHandling::UnReachableCode();
        }
    }

    /**
     * The scalar is narrowed into the input dtype per the PERMISSIVE convert
     * contract (see narrow_scalar) before dispatch. Futures land on a and out.
     * @brief CPU tensor-scalar elementwise: out[i] = a[i] (op) scalar.
     * @param a Input tensor, CPU resident, contiguous.
     * @param scalar Scalar operand (float64_t, narrowed into a's dtype).
     * @param out Output tensor, same shape as a; UInt8 dtype for Eq/Ne, a's dtype otherwise.
     * @param op Operation selector.
     */
    void elementwise_scalar_cpu(const Tensor& a, const float64_t& scalar, Tensor& out, const ElementwiseOp op) {
        if (a.device != Device::CPU || out.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(a.device));
        }
        const RawPtrs raw{a.data, a.offset, nullptr, 0, out.data, out.offset, a.numel};
        switch (op) {
            case ElementwiseOp::Add: dispatch_scalar<ElementwiseOp::Add>(a.dtype, raw, scalar, a, out); break;
            case ElementwiseOp::Sub: dispatch_scalar<ElementwiseOp::Sub>(a.dtype, raw, scalar, a, out); break;
            case ElementwiseOp::Mul: dispatch_scalar<ElementwiseOp::Mul>(a.dtype, raw, scalar, a, out); break;
            case ElementwiseOp::Div: dispatch_scalar<ElementwiseOp::Div>(a.dtype, raw, scalar, a, out); break;
            case ElementwiseOp::Pow: dispatch_scalar<ElementwiseOp::Pow>(a.dtype, raw, scalar, a, out); break;
            case ElementwiseOp::Eq: dispatch_compare_scalar<ElementwiseOp::Eq>(a.dtype, raw, scalar, a, out); break;
            case ElementwiseOp::Ne: dispatch_compare_scalar<ElementwiseOp::Ne>(a.dtype, raw, scalar, a, out); break;
            [[unlikely]] default: throw ErrorHandling::UnReachableCode();
        }
    }

    /**
     * @brief CPU unary elementwise: out[i] = fn(a[i]) (exp/log, float kinds only).
     * @param a Input tensor, CPU resident, contiguous.
     * @param out Output tensor, same shape and dtype as a.
     * @param op Operation selector.
     */
    void elementwise_unary_cpu(const Tensor& a, Tensor& out, const ElementwiseUnaryOp op) {
        if (a.device != Device::CPU || out.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(a.device));
        }
        const RawPtrs raw{a.data, a.offset, nullptr, 0, out.data, out.offset, a.numel};
        switch (op) {
            case ElementwiseUnaryOp::Exp: dispatch_unary<ElementwiseUnaryOp::Exp>(a.dtype, raw, a, out); break;
            case ElementwiseUnaryOp::Log: dispatch_unary<ElementwiseUnaryOp::Log>(a.dtype, raw, a, out); break;
            [[unlikely]] default: throw ErrorHandling::UnReachableCode();
        }
    }
}

namespace BeanTensor::Tensors {

    /**
     * Validates dtype/device/contiguity, rejects integer pow, syncs both
     * operands, builds the broadcast plan, then routes to the CPU or GPU
     * backend. Comparisons produce a UInt8 tensor of the broadcast shape.
     * @brief Tensor-tensor elementwise op with broadcasting.
     * @param o Right operand, same dtype and device.
     * @param op Operation selector.
     * @return New tensor with the broadcast shape.
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
        const auto out_dtype = op == ElementwiseOp::Eq || op == ElementwiseOp::Ne ? DataTypes::DType::UInt8 : this->dtype;
        auto out = Tensor(out_shape, out_dtype, this->device, false);
        if (this->device == Device::CPU) {
            detail::elementwise_cpu(*this, o, out, op, plan);
        } else if (this->device == Device::GPU) {
            detail::elementwise_gpu(*this, o, out, op, plan);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        return out;
    }

    /**
     * Validates contiguity, rejects integer pow, syncs this tensor, then routes
     * to the CPU or GPU backend. Comparisons produce a UInt8 tensor of this
     * tensor's shape.
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
        const auto out_dtype = op == ElementwiseOp::Eq || op == ElementwiseOp::Ne ? DataTypes::DType::UInt8 : this->dtype;
        auto out = Tensor(std::vector<size_t>(this->shape, this->shape + this->ndim), out_dtype, this->device, false);
        if (this->device == Device::CPU) {
            detail::elementwise_scalar_cpu(*this, scalar, out, op);
        } else if (this->device == Device::GPU) {
            detail::elementwise_scalar_gpu(*this, scalar, out, op);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
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
        auto out = Tensor(std::vector<size_t>(this->shape, this->shape + this->ndim), this->dtype, this->device, false);
        if (this->device == Device::CPU) {
            detail::elementwise_unary_cpu(*this, out, op);
        } else if (this->device == Device::GPU) {
            detail::elementwise_unary_gpu(*this, out, op);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
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
    Tensor Tensor::eq(const Tensor& o) const { return elementwise_binary(o, ElementwiseOp::Eq); }
    Tensor Tensor::eq(const float64_t& scalar) const { return elementwise_scalar(scalar, ElementwiseOp::Eq); }
    Tensor Tensor::ne(const Tensor& o) const { return elementwise_binary(o, ElementwiseOp::Ne); }
    Tensor Tensor::ne(const float64_t& scalar) const { return elementwise_scalar(scalar, ElementwiseOp::Ne); }
}
