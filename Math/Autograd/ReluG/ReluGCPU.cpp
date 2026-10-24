/**
 * @file ReluGCPU.cpp
 * @author GreenBean
 * @brief CPU backend for relu backward (relu_g), plus the relu backward
 * trampoline (relu_g_node).
 * @version 0.2
 * @date 2026-10-23
 */
#include <type_traits>
#include <vector>

#include "Math/Autograd/AutogradShared.h"
#include "Math/Autograd/GradDType.h"
#include "Math/Autograd/ReluG/ReluG.h"
#include "Math/Autograd/ReluG/ReluGKernels.h"
#include "Tensor/Tensor.h"

#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Tensors::detail {
    using namespace autograd_shared;

    namespace {
        // --------------------------- relu_g kernels ---------------------------

        /**
         * dx[i] = x[i] > 0 ? dy[i] : 0 - a select, not a multiply: NaN in x
         * fails the comparison and zeroes the lane; NaN in a selected dy lane
         * propagates. Bit-identical to the AVX mask/blend kernels.
         * @brief Scalar range relu backward (fp32/fp64).
         * @param dx Output gradient buffer (written, not accumulated).
         * @param x Forward input of the relu.
         * @param dy Upstream gradient.
         * @param begin First element processed.
         * @param end One past the last element processed.
         */
        template<typename T>
        void relu_g_range_scalar(T* dx, const T* x, const T* dy, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                dx[i] = x[i] > T{0} ? dy[i] : T{0};
            }
        }

        /**
         * Half-dtype relu backward: the comparison widens x to fp32 (NaN fails
         * it); the selected lane is stored as-is - no arithmetic, no rounding.
         * @brief Scalar range relu backward (fp16/bf16).
         * @param dx Output gradient buffer (written, not accumulated).
         * @param x Forward input of the relu.
         * @param dy Upstream gradient.
         * @param begin First element processed.
         * @param end One past the last element processed.
         */
        template<typename H>
        void relu_g_range_half(H* dx, const H* x, const H* dy, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                dx[i] = widen_lane(x[i]) > 0.0f ? dy[i] : zero_bits<H>();
            }
        }

        /**
         * Packed-dtype relu backward: per-lane widened compare, per-lane select.
         * @brief Scalar range relu backward (fp16x2/bf16x2).
         * @param dx Output gradient buffer (written, not accumulated).
         * @param x Forward input of the relu.
         * @param dy Upstream gradient.
         * @param begin First element processed.
         * @param end One past the last element processed.
         */
        template<typename P>
        void relu_g_range_packed(P* dx, const P* x, const P* dy, const size_t begin, const size_t end) {
            using H = packed_lane_t<P>;
            for (auto i = begin; i < end; ++i) {
                const auto x_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(x[i]);
                const auto d_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(dy[i]);
                const auto lo = widen_lane(lane_lo<H>(x_bits)) > 0.0f
                    ? static_cast<uint16_t>(d_bits & 0xFFFFu) : uint16_t{0};
                const auto hi = widen_lane(lane_hi<H>(x_bits)) > 0.0f
                    ? static_cast<uint16_t>(d_bits >> 16) : uint16_t{0};
                dx[i] = pack_lanes<P>(lo, hi);
            }
        }

        /**
         * Integer relu backward: signed dtypes select on x > 0; unsigned dtypes
         * copy dy through (matches forward relu's unsigned passthrough).
         * @brief Scalar range relu backward (all integer dtypes).
         * @param dx Output gradient buffer (written, not accumulated).
         * @param x Forward input of the relu.
         * @param dy Upstream gradient.
         * @param begin First element processed.
         * @param end One past the last element processed.
         */
        template<typename T>
        void relu_g_range_int(T* dx, const T* x, const T* dy, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                if constexpr (std::is_signed_v<T>) {
                    dx[i] = x[i] > T{0} ? dy[i] : T{0};
                } else {
                    dx[i] = dy[i];
                }
            }
        }

        /**
         * Int4x2 relu backward: per-nibble select on x > 0, repack on store.
         * @brief Scalar range relu backward (int4x2, per-nibble).
         * @param dx Output gradient buffer (written, not accumulated).
         * @param x Forward input of the relu.
         * @param dy Upstream gradient.
         * @param begin First storage element processed.
         * @param end One past the last storage element processed.
         */
        void relu_g_range_int4x2(int4x2_t* dx, const int4x2_t* x, const int4x2_t* dy, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                const auto lo = int4_lo(x[i]) > 0 ? int4_lo(dy[i]) : 0;
                const auto hi = int4_hi(x[i]) > 0 ? int4_hi(dy[i]) : 0;
                dx[i] = pack_int4(lo, hi);
            }
        }

        /**
         * Fixed-arity function pointer type of a relu_g range kernel (scalar or AVX).
         * @brief Range kernel signature for relu backward: (dx, x, dy, begin, end).
         */
        template<typename T>
        using ReluGScalarFn = void(*)(T*, const T*, const T*, size_t, size_t);
    }

    /**
     * dx = x > 0 ? dy : 0 over x.numel elements. NaN in x zeroes the lane; NaN in a
     * selected dy lane propagates. dx is written, not accumulated.
     * Work is queued on the CPU pool and returns immediately; futures land on x, dy and dx.
     * @brief CPU relu backward.
     * @param dy Upstream gradient.
     * @param x Forward input of the relu.
     * @param dx Output gradient (written).
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if any tensor is not on the CPU.
     * @throws ErrorHandling::NotImplemented for an unsupported dtype.
     */
    void relu_g_cpu(const Tensor& dy, const Tensor& x, const Tensor& dx) {
        if (dy.device != Device::CPU || x.device != Device::CPU || dx.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(x.device));
        }
        using DType = DataTypes::DType;
        const auto numel = x.numel;
        switch (x.dtype) {
            case DType::Float32: {
                auto* dx_ptr = static_cast<float32_t*>(dx.data) + dx.offset;
                const auto* x_ptr = static_cast<const float32_t*>(x.data) + x.offset;
                const auto* dy_ptr = static_cast<const float32_t*>(dy.data) + dy.offset;
                const ReluGScalarFn<float32_t> kernel = Math::Autograd::relu_g_avx512_available()
                    ? Math::Autograd::relu_g_f32_range_avx512
                    : Math::Autograd::relu_g_avx2_available() ? Math::Autograd::relu_g_f32_range_avx2 : relu_g_range_scalar<float32_t>;
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value (the futures outlive this call): dx_ptr, x_ptr, dy_ptr and the dispatched kernel.
                 * Calls kernel(dx_ptr, x_ptr, dy_ptr, b, e), where the pointers already include their tensors' offsets.
                 * @brief Worker for relu_g_cpu, float32_t path (scalar or AVX kernel).
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([=](const size_t b, const size_t e) { kernel(dx_ptr, x_ptr, dy_ptr, b, e); }, numel, x, dy, dx);
                break;
            }
            case DType::Float64: {
                auto* dx_ptr = static_cast<float64_t*>(dx.data) + dx.offset;
                const auto* x_ptr = static_cast<const float64_t*>(x.data) + x.offset;
                const auto* dy_ptr = static_cast<const float64_t*>(dy.data) + dy.offset;
                const ReluGScalarFn<float64_t> kernel = Math::Autograd::relu_g_avx512_available()
                    ? Math::Autograd::relu_g_f64_range_avx512
                    : Math::Autograd::relu_g_avx2_available() ? Math::Autograd::relu_g_f64_range_avx2 : relu_g_range_scalar<float64_t>;
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value (the futures outlive this call): dx_ptr, x_ptr, dy_ptr and the dispatched kernel.
                 * Calls kernel(dx_ptr, x_ptr, dy_ptr, b, e), where the pointers already include their tensors' offsets.
                 * @brief Worker for relu_g_cpu, float64_t path (scalar or AVX kernel).
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([=](const size_t b, const size_t e) { kernel(dx_ptr, x_ptr, dy_ptr, b, e); }, numel, x, dy, dx);
                break;
            }
            case DType::Float16: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (dx_ptr, x_ptr, dy_ptr, offsets already applied); no default capture.
                 * Writes dx = x > 0 ? dy : 0 on [b, e). Lanes widen to fp32, select by x > 0, and narrow RNE.
                 * @brief Worker for relu_g_cpu, the float16_t path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([dx_ptr = static_cast<float16_t*>(dx.data) + dx.offset, x_ptr = static_cast<const float16_t*>(x.data) + x.offset, dy_ptr = static_cast<const float16_t*>(dy.data) + dy.offset](const size_t b, const size_t e) { relu_g_range_half<float16_t>(dx_ptr, x_ptr, dy_ptr, b, e); }, numel, x, dy, dx);
                break;
            }
            case DType::BFloat16: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (dx_ptr, x_ptr, dy_ptr, offsets already applied); no default capture.
                 * Writes dx = x > 0 ? dy : 0 on [b, e). Lanes widen to fp32, select by x > 0, and narrow RNE.
                 * @brief Worker for relu_g_cpu, the bfloat16_t path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([dx_ptr = static_cast<bfloat16_t*>(dx.data) + dx.offset, x_ptr = static_cast<const bfloat16_t*>(x.data) + x.offset, dy_ptr = static_cast<const bfloat16_t*>(dy.data) + dy.offset](const size_t b, const size_t e) { relu_g_range_half<bfloat16_t>(dx_ptr, x_ptr, dy_ptr, b, e); }, numel, x, dy, dx);
                break;
            }
            case DType::Float16x2: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (dx_ptr, x_ptr, dy_ptr, offsets already applied); no default capture.
                 * Writes dx = x > 0 ? dy : 0 on [b, e). Each element holds two lanes; both are processed per element.
                 * @brief Worker for relu_g_cpu, the float16x2_t packed path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([dx_ptr = static_cast<float16x2_t*>(dx.data) + dx.offset, x_ptr = static_cast<const float16x2_t*>(x.data) + x.offset, dy_ptr = static_cast<const float16x2_t*>(dy.data) + dy.offset](const size_t b, const size_t e) { relu_g_range_packed<float16x2_t>(dx_ptr, x_ptr, dy_ptr, b, e); }, numel, x, dy, dx);
                break;
            }
            case DType::BFloat16x2: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (dx_ptr, x_ptr, dy_ptr, offsets already applied); no default capture.
                 * Writes dx = x > 0 ? dy : 0 on [b, e). Each element holds two lanes; both are processed per element.
                 * @brief Worker for relu_g_cpu, the bfloat16x2_t packed path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([dx_ptr = static_cast<bfloat16x2_t*>(dx.data) + dx.offset, x_ptr = static_cast<const bfloat16x2_t*>(x.data) + x.offset, dy_ptr = static_cast<const bfloat16x2_t*>(dy.data) + dy.offset](const size_t b, const size_t e) { relu_g_range_packed<bfloat16x2_t>(dx_ptr, x_ptr, dy_ptr, b, e); }, numel, x, dy, dx);
                break;
            }
            case DType::Int8: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (dx_ptr, x_ptr, dy_ptr, offsets already applied); no default capture.
                 * Writes dx = x > 0 ? dy : 0 on [b, e). Integer arithmetic with the dtype's wrap/clamp semantics.
                 * @brief Worker for relu_g_cpu, the int8_t integer path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([dx_ptr = static_cast<int8_t*>(dx.data) + dx.offset, x_ptr = static_cast<const int8_t*>(x.data) + x.offset, dy_ptr = static_cast<const int8_t*>(dy.data) + dy.offset](const size_t b, const size_t e) { relu_g_range_int<int8_t>(dx_ptr, x_ptr, dy_ptr, b, e); }, numel, x, dy, dx);
                break;
            }
            case DType::UInt8: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (dx_ptr, x_ptr, dy_ptr, offsets already applied); no default capture.
                 * Writes dx = x > 0 ? dy : 0 on [b, e). Integer arithmetic with the dtype's wrap/clamp semantics.
                 * @brief Worker for relu_g_cpu, the uint8_t integer path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([dx_ptr = static_cast<uint8_t*>(dx.data) + dx.offset, x_ptr = static_cast<const uint8_t*>(x.data) + x.offset, dy_ptr = static_cast<const uint8_t*>(dy.data) + dy.offset](const size_t b, const size_t e) { relu_g_range_int<uint8_t>(dx_ptr, x_ptr, dy_ptr, b, e); }, numel, x, dy, dx);
                break;
            }
            case DType::Int16: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (dx_ptr, x_ptr, dy_ptr, offsets already applied); no default capture.
                 * Writes dx = x > 0 ? dy : 0 on [b, e). Integer arithmetic with the dtype's wrap/clamp semantics.
                 * @brief Worker for relu_g_cpu, the int16_t integer path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([dx_ptr = static_cast<int16_t*>(dx.data) + dx.offset, x_ptr = static_cast<const int16_t*>(x.data) + x.offset, dy_ptr = static_cast<const int16_t*>(dy.data) + dy.offset](const size_t b, const size_t e) { relu_g_range_int<int16_t>(dx_ptr, x_ptr, dy_ptr, b, e); }, numel, x, dy, dx);
                break;
            }
            case DType::UInt16: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (dx_ptr, x_ptr, dy_ptr, offsets already applied); no default capture.
                 * Writes dx = x > 0 ? dy : 0 on [b, e). Integer arithmetic with the dtype's wrap/clamp semantics.
                 * @brief Worker for relu_g_cpu, the uint16_t integer path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([dx_ptr = static_cast<uint16_t*>(dx.data) + dx.offset, x_ptr = static_cast<const uint16_t*>(x.data) + x.offset, dy_ptr = static_cast<const uint16_t*>(dy.data) + dy.offset](const size_t b, const size_t e) { relu_g_range_int<uint16_t>(dx_ptr, x_ptr, dy_ptr, b, e); }, numel, x, dy, dx);
                break;
            }
            case DType::Int32: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (dx_ptr, x_ptr, dy_ptr, offsets already applied); no default capture.
                 * Writes dx = x > 0 ? dy : 0 on [b, e). Integer arithmetic with the dtype's wrap/clamp semantics.
                 * @brief Worker for relu_g_cpu, the int32_t integer path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([dx_ptr = static_cast<int32_t*>(dx.data) + dx.offset, x_ptr = static_cast<const int32_t*>(x.data) + x.offset, dy_ptr = static_cast<const int32_t*>(dy.data) + dy.offset](const size_t b, const size_t e) { relu_g_range_int<int32_t>(dx_ptr, x_ptr, dy_ptr, b, e); }, numel, x, dy, dx);
                break;
            }
            case DType::UInt32: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (dx_ptr, x_ptr, dy_ptr, offsets already applied); no default capture.
                 * Writes dx = x > 0 ? dy : 0 on [b, e). Integer arithmetic with the dtype's wrap/clamp semantics.
                 * @brief Worker for relu_g_cpu, the uint32_t integer path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([dx_ptr = static_cast<uint32_t*>(dx.data) + dx.offset, x_ptr = static_cast<const uint32_t*>(x.data) + x.offset, dy_ptr = static_cast<const uint32_t*>(dy.data) + dy.offset](const size_t b, const size_t e) { relu_g_range_int<uint32_t>(dx_ptr, x_ptr, dy_ptr, b, e); }, numel, x, dy, dx);
                break;
            }
            case DType::Int64: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (dx_ptr, x_ptr, dy_ptr, offsets already applied); no default capture.
                 * Writes dx = x > 0 ? dy : 0 on [b, e). Integer arithmetic with the dtype's wrap/clamp semantics.
                 * @brief Worker for relu_g_cpu, the int64_t integer path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([dx_ptr = static_cast<int64_t*>(dx.data) + dx.offset, x_ptr = static_cast<const int64_t*>(x.data) + x.offset, dy_ptr = static_cast<const int64_t*>(dy.data) + dy.offset](const size_t b, const size_t e) { relu_g_range_int<int64_t>(dx_ptr, x_ptr, dy_ptr, b, e); }, numel, x, dy, dx);
                break;
            }
            case DType::UInt64: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (dx_ptr, x_ptr, dy_ptr, offsets already applied); no default capture.
                 * Writes dx = x > 0 ? dy : 0 on [b, e). Integer arithmetic with the dtype's wrap/clamp semantics.
                 * @brief Worker for relu_g_cpu, the uint64_t integer path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([dx_ptr = static_cast<uint64_t*>(dx.data) + dx.offset, x_ptr = static_cast<const uint64_t*>(x.data) + x.offset, dy_ptr = static_cast<const uint64_t*>(dy.data) + dy.offset](const size_t b, const size_t e) { relu_g_range_int<uint64_t>(dx_ptr, x_ptr, dy_ptr, b, e); }, numel, x, dy, dx);
                break;
            }
            case DType::Int4x2: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (dx_ptr, x_ptr, dy_ptr, offsets already applied); no default capture.
                 * Writes dx = x > 0 ? dy : 0 on [b, e). Each element holds two nibbles; both are processed per element.
                 * @brief Worker for relu_g_cpu, the int4x2_t path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([dx_ptr = static_cast<int4x2_t*>(dx.data) + dx.offset, x_ptr = static_cast<const int4x2_t*>(x.data) + x.offset, dy_ptr = static_cast<const int4x2_t*>(dy.data) + dy.offset](const size_t b, const size_t e) { relu_g_range_int4x2(dx_ptr, x_ptr, dy_ptr, b, e); }, numel, x, dy, dx);
                break;
            }
            [[unlikely]] default: {
                throw ErrorHandling::NotImplemented();
            }
        }
    }

    /**
     * Skips work when x does not require grad; otherwise computes dx into a fresh
     * grad_dtype tensor and hands it to x.accumulate_grad. output_grad already
     * carries grad_dtype; relu_g requires dy/x/dx in one dtype, so x is
     * boundary-cast into grad_dtype (GradDType.h, the single conversion rule)
     * when it differs — the x > 0 mask is then evaluated on the converted
     * values (sign-exact for float<->float; integer grad dtypes see the
     * truncated/clamped x, inherent to requesting an integer gradient chain).
     * Syncs output_grad and x first.
     * @brief Backward trampoline for relu.
     * @param output_grad Gradient of this node's output tensor, in grad_dtype.
     * @param inputs Input tensors of the forward op.
     * @param n_inputs Number of entries in inputs.
     * @param grad_dtype The calling backward()'s assigned gradient dtype.
     * @throws ErrorHandling::UnReachableCode if n_inputs is not 1 or the device is neither CPU nor GPU.
     */
    void relu_g_node(Node*, const Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        if (n_inputs != 1) [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        auto& x = *inputs[0];
        if (!x.requires_grad) {
            return;
        }
        output_grad->sync();
        x.sync();
        auto dx = Tensor(std::vector(x.shape, x.shape + x.ndim), grad_dtype, x.device, false);
        /**
         * Captures output_grad, dx and x by reference (all outlive the call).
         * Dispatches relu_g on x's device with dy = *output_grad, x = x_op, dx = dx.
         * @brief Device dispatch for relu_g_node with the (possibly converted) operand.
         * @param x_op Forward input in grad_dtype.
         */
        auto run = [&](const Tensor& x_op) {
            if (x.device == Device::CPU) {
                relu_g_cpu(*output_grad, x_op, dx);
            } else if (x.device == Device::GPU) {
                relu_g_gpu(*output_grad, x_op, dx);
            } else [[unlikely]] {
                throw ErrorHandling::UnReachableCode();
            }
        };
        if (x.dtype == grad_dtype) {
            run(x);
        } else {
            const auto xg = grad_converted_copy(x, grad_dtype);
            run(xg);
        }
        if (x.node != nullptr) {
            grad_probe_hit(dx.dtype); // test hook: an intermediate received a delta
        }
        x.accumulate_grad(std::move(dx));
    }
}
