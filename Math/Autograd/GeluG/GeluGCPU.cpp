/**
 * @file GeluGCPU.cpp
 * @author GreenBean
 * @brief CPU backend for GELU backward (gelu_g), plus the gelu_g backward trampoline.
 * @version 0.1
 * @date 2026-11-10
 */
#include <type_traits>
#include <vector>

#include "Helpers/Errors.h"
#include "Math/Autograd/AutogradShared.h"
#include "Math/Autograd/GradDType.h"
#include "Math/Autograd/GeluG/GeluG.h"
#include "Math/Autograd/GeluG/GeluGKernels.h"
#include "Math/Activations/Gelu/GeluKernels.h"
#include "Tensor/Tensor.h"

#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Tensors::detail {
    using namespace autograd_shared;
    using namespace BeanTensor::Math::Activations::detail;

    namespace {
        // --------------------------- gelu_g kernels ---------------------------

        /**
         * dx[i] = dy[i] * gelu'(x[i]) using std::erf in the storage width.  The
         * piecewise branches match gelu_exact_lane so forward/backward agree at
         * the non-finite limits.
         * @brief Scalar range GELU backward (fp32/fp64).
         * @param dx Output gradient buffer (written, not accumulated).
         * @param x Forward input of the GELU.
         * @param dy Upstream gradient.
         * @param begin First element processed.
         * @param end One past the last element processed.
         */
        template<typename T>
        void gelu_g_range_scalar(T* dx, const T* x, const T* dy, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                dx[i] = dy[i] * gelu_deriv_exact_lane(x[i]);
            }
        }

        /**
         * Half-dtype GELU backward: widen x and dy to fp32, multiply dy by the
         * derivative, narrow RNE.
         * @brief Scalar range GELU backward (fp16/bf16).
         * @param dx Output gradient buffer (written, not accumulated).
         * @param x Forward input of the GELU.
         * @param dy Upstream gradient.
         * @param begin First element processed.
         * @param end One past the last element processed.
         */
        template<typename H>
        void gelu_g_range_half(H* dx, const H* x, const H* dy, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                const auto xw = widen_lane(x[i]);
                const auto dyw = widen_lane(dy[i]);
                dx[i] = narrow_lane<H>(dyw * gelu_deriv_exact_lane(xw));
            }
        }

        /**
         * Packed-dtype GELU backward: per-lane widen, derivative, multiply, narrow,
         * repack on store.
         * @brief Scalar range GELU backward (fp16x2/bf16x2).
         * @param dx Output gradient buffer (written, not accumulated).
         * @param x Forward input of the GELU.
         * @param dy Upstream gradient.
         * @param begin First element processed.
         * @param end One past the last element processed.
         */
        template<typename P>
        void gelu_g_range_packed(P* dx, const P* x, const P* dy, const size_t begin, const size_t end) {
            using H = packed_lane_t<P>;
            for (auto i = begin; i < end; ++i) {
                const auto x_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(x[i]);
                const auto d_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(dy[i]);
                const auto x_lo = widen_lane(lane_lo<H>(x_bits));
                const auto x_hi = widen_lane(lane_hi<H>(x_bits));
                const auto dy_lo = widen_lane(lane_lo<H>(d_bits));
                const auto dy_hi = widen_lane(lane_hi<H>(d_bits));
                const auto lo = narrow_lane<H>(dy_lo * gelu_deriv_exact_lane(x_lo));
                const auto hi = narrow_lane<H>(dy_hi * gelu_deriv_exact_lane(x_hi));
                dx[i] = pack_lanes<P>(Intrinsics::detail::bit_cast_memcpy<uint16_t>(lo),
                                      Intrinsics::detail::bit_cast_memcpy<uint16_t>(hi));
            }
        }

        /**
         * Fixed-arity function pointer type of a gelu_g range kernel (scalar or AVX).
         * @brief Range kernel signature for GELU backward: (dx, x, dy, begin, end).
         */
        template<typename T>
        using GeluGScalarFn = void(*)(T*, const T*, const T*, size_t, size_t);

        /**
         * Local twin of autograd_shared::launch_elementwise with CCD-aware
         * scheduling: gelu_g writes the output gradient buffer, so the gate is
         * the bytes written (numel * element size).
         * @brief Threaded GELU-backward launcher with CCD-local routing.
         * @param worker Callable taking (begin, end); copied into each task.
         * @param numel Total element count to cover.
         * @param out_bytes Bytes of the output gradient buffer actually written by the workers.
         * @param a First tensor the workers touch (receives the futures).
         * @param b Second tensor the workers touch (receives the futures).
         * @param out Output tensor (receives the futures).
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
    }

    /**
     * dx = dy * gelu'(x) over x.numel elements.  dx is written, not accumulated.
     * Work is queued on the CPU pool and returns immediately; futures land on x, dy and dx.
     * @brief CPU GELU backward.
     * @param dy Upstream gradient.
     * @param x Forward input of the GELU.
     * @param dx Output gradient (written).
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if any tensor is not on the CPU.
     * @throws ErrorHandling::NotImplemented for an unsupported dtype.
     */
    void gelu_g_cpu(const Tensor& dy, const Tensor& x, const Tensor& dx) {
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
                const GeluGScalarFn<float32_t> kernel = Math::Autograd::gelu_g_avx512_available()
                    ? Math::Autograd::gelu_g_f32_range_avx512
                    : Math::Autograd::gelu_g_avx2_available() ? Math::Autograd::gelu_g_f32_range_avx2 : gelu_g_range_scalar<float32_t>;
                launch_elementwise_ccd([=](const size_t b, const size_t e) { kernel(dx_ptr, x_ptr, dy_ptr, b, e); }, numel, numel * sizeof(float32_t), x, dy, dx);
                break;
            }
            case DType::Float64: {
                auto* dx_ptr = static_cast<float64_t*>(dx.data) + dx.offset;
                const auto* x_ptr = static_cast<const float64_t*>(x.data) + x.offset;
                const auto* dy_ptr = static_cast<const float64_t*>(dy.data) + dy.offset;
                const GeluGScalarFn<float64_t> kernel = Math::Autograd::gelu_g_avx512_available()
                    ? Math::Autograd::gelu_g_f64_range_avx512
                    : Math::Autograd::gelu_g_avx2_available() ? Math::Autograd::gelu_g_f64_range_avx2 : gelu_g_range_scalar<float64_t>;
                launch_elementwise_ccd([=](const size_t b, const size_t e) { kernel(dx_ptr, x_ptr, dy_ptr, b, e); }, numel, numel * sizeof(float64_t), x, dy, dx);
                break;
            }
            case DType::Float16: {
                launch_elementwise_ccd([dx_ptr = static_cast<float16_t*>(dx.data) + dx.offset, x_ptr = static_cast<const float16_t*>(x.data) + x.offset, dy_ptr = static_cast<const float16_t*>(dy.data) + dy.offset](const size_t b, const size_t e) { gelu_g_range_half<float16_t>(dx_ptr, x_ptr, dy_ptr, b, e); }, numel, numel * sizeof(float16_t), x, dy, dx);
                break;
            }
            case DType::BFloat16: {
                launch_elementwise_ccd([dx_ptr = static_cast<bfloat16_t*>(dx.data) + dx.offset, x_ptr = static_cast<const bfloat16_t*>(x.data) + x.offset, dy_ptr = static_cast<const bfloat16_t*>(dy.data) + dy.offset](const size_t b, const size_t e) { gelu_g_range_half<bfloat16_t>(dx_ptr, x_ptr, dy_ptr, b, e); }, numel, numel * sizeof(bfloat16_t), x, dy, dx);
                break;
            }
            case DType::Float16x2: {
                launch_elementwise_ccd([dx_ptr = static_cast<float16x2_t*>(dx.data) + dx.offset, x_ptr = static_cast<const float16x2_t*>(x.data) + x.offset, dy_ptr = static_cast<const float16x2_t*>(dy.data) + dy.offset](const size_t b, const size_t e) { gelu_g_range_packed<float16x2_t>(dx_ptr, x_ptr, dy_ptr, b, e); }, numel, numel * sizeof(float16x2_t), x, dy, dx);
                break;
            }
            case DType::BFloat16x2: {
                launch_elementwise_ccd([dx_ptr = static_cast<bfloat16x2_t*>(dx.data) + dx.offset, x_ptr = static_cast<const bfloat16x2_t*>(x.data) + x.offset, dy_ptr = static_cast<const bfloat16x2_t*>(dy.data) + dy.offset](const size_t b, const size_t e) { gelu_g_range_packed<bfloat16x2_t>(dx_ptr, x_ptr, dy_ptr, b, e); }, numel, numel * sizeof(bfloat16x2_t), x, dy, dx);
                break;
            }
            [[unlikely]] default: {
                throw ErrorHandling::NotImplemented();
            }
        }
    }

    /**
     * Skips work when x does not require grad; otherwise computes dx into a fresh
     * grad_dtype tensor and hands it to x.accumulate_grad.  output_grad already
     * carries grad_dtype; gelu_g requires dy/x/dx in one dtype, so x is
     * boundary-cast into grad_dtype (GradDType.h, the single conversion rule)
     * when it differs.  Syncs output_grad and x first.
     * @brief Backward trampoline for GELU.
     * @param output_grad Gradient of this node's output tensor, in grad_dtype.
     * @param inputs Input tensors of the forward op.
     * @param n_inputs Number of entries in inputs.
     * @param grad_dtype The calling backward()'s assigned gradient dtype.
     * @throws ErrorHandling::UnReachableCode if n_inputs is not 1 or the device is neither CPU nor GPU.
     */
    void gelu_g_node(Node*, Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
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
         * Dispatches gelu_g on x's device with dy = *output_grad, x = x_op, dx = dx.
         * @param x_op Forward input in grad_dtype.
         */
        auto run = [&](const Tensor& x_op) {
            if (x.device == Device::CPU) {
                gelu_g_cpu(*output_grad, x_op, dx);
            } else if (x.device == Device::GPU) {
                gelu_g_gpu(*output_grad, x_op, dx);
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
            grad_probe_hit(dx.dtype);
        }
        x.accumulate_grad(std::move(dx));
    }
}
