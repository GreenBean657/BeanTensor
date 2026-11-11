/**
 * @file GeluGCUDA.cu
 * @author GreenBean
 * @brief CUDA backend for the gelu_g op (GELU backward dx = dy * gelu'(x)).
 * Divergent twin of Math/Autograd/GeluG/GeluGHIP.cpp — the two backends are
 * written separately per project rule (no HIP-only unification), but must stay
 * behaviorally identical: same fp32/fp64 compute widths, same RNE narrowing,
 * same NaN/Inf contract.
 * @version 0.1
 * @date 2026-11-10
 */
#if defined(USE_CUDA)
#include <cstdint>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "Math/Autograd/GeluG/GeluG.h"
#include "Math/Autograd/AutogradCudaShared.h"
#include "Math/Activations/Gelu/GeluKernels.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    using namespace autograd_gpu;
    using namespace BeanTensor::Math::Activations::detail;

    namespace {
        /**
         * dx[i] = dy[i] * gelu'(x[i]).  float16/bfloat16 compare and compute in
         * fp32; packed dtypes per lane.  Launched as grid_for(numel) blocks of
         * BT_GPU_THREADS threads, grid-stride over numel.
         * @brief GELU backward kernel (float kinds).
         * @param x Forward input of the GELU, numel elements.
         * @param dy Upstream gradient, numel elements.
         * @param dx Output gradient, numel elements.
         * @param numel Element count (storage elements, so packed dtypes count pairs).
         */
        template<typename T>
        __global__ void gelu_g_kernel(const T* __restrict__ x, const T* __restrict__ dy, T* __restrict__ dx, const size_t numel) {
            const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t stride = blockDim.x * gridDim.x;
            for (size_t i = tid; i < numel; i += stride) {
                if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                    dx[i] = dy[i] * gelu_deriv_exact_lane(x[i]);
                } else if constexpr (std::is_same_v<T, float16_t>) {
                    const auto xw = __half2float(x[i]);
                    const auto dyw = __half2float(dy[i]);
                    dx[i] = __float2half_rn(dyw * gelu_deriv_exact_lane(xw));
                } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                    const auto xw = __bfloat162float(x[i]);
                    const auto dyw = __bfloat162float(dy[i]);
                    dx[i] = dev_narrow_bf16(dyw * gelu_deriv_exact_lane(xw));
                } else if constexpr (std::is_same_v<T, float16x2_t>) {
                    const auto xv = x[i];
                    const auto dyv = dy[i];
                    const auto lo = __half2float(__low2half(dyv)) * gelu_deriv_exact_lane(__half2float(__low2half(xv)));
                    const auto hi = __half2float(__high2half(dyv)) * gelu_deriv_exact_lane(__half2float(__high2half(xv)));
                    dx[i] = __halves2half2(__float2half_rn(lo), __float2half_rn(hi));
                } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                    const auto xv = x[i];
                    const auto dyv = dy[i];
                    const auto lo = __bfloat162float(__low2bfloat16(dyv)) * gelu_deriv_exact_lane(__bfloat162float(__low2bfloat16(xv)));
                    const auto hi = __bfloat162float(__high2bfloat16(dyv)) * gelu_deriv_exact_lane(__bfloat162float(__high2bfloat16(xv)));
                    dx[i] = __halves2bfloat162(dev_narrow_bf16(lo), dev_narrow_bf16(hi));
                }
            }
        }

        /**
         * Skips the launch when numel is 0. The launch is asynchronous; the caller syncs.
         * @brief Host-side gelu_g launch.
         * @param x Forward input of the GELU.
         * @param dy Upstream gradient.
         * @param dx Output gradient.
         * @param numel Element count.
         */
        template<typename T>
        __host__ void run_gelu_g(const T* x, const T* dy, T* dx, const size_t numel) {
            if (numel > 0) {
                gelu_g_kernel<T><<<grid_for(numel), BT_GPU_THREADS>>>(x, dy, dx, numel);
            }
        }
    }

    /**
     * Queues the launch on the CPU thread pool (High priority) and returns; the shared
     * future is given to all three tensors, and errors surface at sync().
     * @brief GPU GELU backward dx = dy * gelu'(x) (float kinds).
     * @param dy Upstream gradient tensor.
     * @param x Forward input tensor of the GELU.
     * @param dx Output gradient tensor.
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if any tensor is not on the GPU.
     * @throws ErrorHandling::NotImplemented (from the worker, at sync) for an unsupported dtype.
     */
    __host__ void gelu_g_gpu(const Tensor& dy, const Tensor& x, const Tensor& dx) {
        if (dy.device != Device::GPU || x.device != Device::GPU || dx.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(x.device));
        }
        const auto* const dy_data = dy.data;
        const auto* const x_data = x.data;
        auto* const dx_data = dx.data;
        const auto dy_offset = dy.offset;
        const auto x_offset = x.offset;
        const auto dx_offset = dx.offset;
        const auto dtype = x.dtype;
        const auto numel = x.numel;
        /**
         * Captures raw pointers, offsets, dtype and numel by value, so the lambda stays valid after the enclosing call returns.
         * Dispatches on dtype to run_gelu_g, which launches the GELU backward kernel on the default stream, then calls cudaStreamSynchronize(nullptr) so the future completes only once dx is written.
         * An unsupported dtype throws ErrorHandling::NotImplemented.
         * Runs on a CPU thread pool thread (High priority) via the async pool; the host call has already returned. Any exception thrown here (including a failed CUDA_CHECK_ERROR) is captured in the shared future and rethrown at sync().
         * @brief Async pool worker that launches the gelu_g kernel for the tensor dtype and syncs the stream.
         */
        const auto worker = [=] {
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: run_gelu_g(static_cast<const float32_t*>(x_data) + x_offset, static_cast<const float32_t*>(dy_data) + dy_offset, static_cast<float32_t*>(dx_data) + dx_offset, numel); break;
                case DType::Float64: run_gelu_g(static_cast<const float64_t*>(x_data) + x_offset, static_cast<const float64_t*>(dy_data) + dy_offset, static_cast<float64_t*>(dx_data) + dx_offset, numel); break;
                case DType::Float16: run_gelu_g(static_cast<const float16_t*>(x_data) + x_offset, static_cast<const float16_t*>(dy_data) + dy_offset, static_cast<float16_t*>(dx_data) + dx_offset, numel); break;
                case DType::BFloat16: run_gelu_g(static_cast<const bfloat16_t*>(x_data) + x_offset, static_cast<const bfloat16_t*>(dy_data) + dy_offset, static_cast<bfloat16_t*>(dx_data) + dx_offset, numel); break;
                case DType::Float16x2: run_gelu_g(static_cast<const float16x2_t*>(x_data) + x_offset, static_cast<const float16x2_t*>(dy_data) + dy_offset, static_cast<float16x2_t*>(dx_data) + dx_offset, numel); break;
                case DType::BFloat16x2: run_gelu_g(static_cast<const bfloat16x2_t*>(x_data) + x_offset, static_cast<const bfloat16x2_t*>(dy_data) + dy_offset, static_cast<bfloat16x2_t*>(dx_data) + dx_offset, numel); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
            CUDA_CHECK_ERROR(cudaStreamSynchronize(nullptr));
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        dy.give_future(future);
        x.give_future(future);
        dx.give_future(future);
    }
}
#endif
