/**
 * @file GradScalerHIP.cpp
 * @author GreenBean
 * @brief HIP/ROCm backend for GradScaler::unscale_tensor_: in-place scalar
 * division of float-kind gradients. Divergent twin of GradScalerCUDA.cu.
 * @version 0.1
 * @date 2026-11-14
 */
#if defined(USE_HIP)
#include <cstddef>
#include <type_traits>

#include <hip/hip_runtime.h>

#include "Math/Optimization/GradScaler/GradScaler.h"
#include "Math/Optimization/OptimizationHipShared.h"
#include "Math/Optimization/OptimizationShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Optimization::detail {
    using namespace Tensors;
    using namespace Tensors::detail::optimization_gpu;

    namespace {
        /**
         * In-place division by the inverse scale. Float kinds compute in their
         * natural accumulator (fp32 for fp16/bf16/fp32 and per-lane for packed
         * x2, fp64 for fp64) and narrow RNE on store, matching the CPU backend.
         * @brief Elementwise divide-in-place kernel.
         * @param ptr Buffer to update in place.
         * @param inv_scale 1/scale in the storage-matching precision.
         * @param numel Element count.
         */
        template<typename T>
        __global__ void div_scalar_inplace_kernel(T* __restrict__ ptr, const float64_t inv_scale, const size_t numel) {
            const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t stride = blockDim.x * gridDim.x;
            for (size_t i = tid; i < numel; i += stride) {
                if constexpr (std::is_same_v<T, float32_t>) {
                    ptr[i] *= static_cast<float32_t>(inv_scale);
                } else if constexpr (std::is_same_v<T, float64_t>) {
                    ptr[i] *= inv_scale;
                } else if constexpr (std::is_same_v<T, float16_t>) {
                    ptr[i] = __float2half_rn(__half2float(ptr[i]) * static_cast<float32_t>(inv_scale));
                } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                    ptr[i] = dev_narrow_bf16(__bfloat162float(ptr[i]) * static_cast<float32_t>(inv_scale));
                } else if constexpr (std::is_same_v<T, float16x2_t>) {
                    const auto lo = __float2half_rn(__half2float(__low2half(ptr[i])) * static_cast<float32_t>(inv_scale));
                    const auto hi = __float2half_rn(__half2float(__high2half(ptr[i])) * static_cast<float32_t>(inv_scale));
                    ptr[i] = __halves2half2(lo, hi);
                } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                    const auto lo = dev_narrow_bf16(__bfloat162float(__low2bfloat16(ptr[i])) * static_cast<float32_t>(inv_scale));
                    const auto hi = dev_narrow_bf16(__bfloat162float(__high2bfloat16(ptr[i])) * static_cast<float32_t>(inv_scale));
                    ptr[i] = __halves2bfloat162(lo, hi);
                } else [[unlikely]] {
                    // Integer kinds are rejected before dispatch.
                }
            }
        }

        template<typename T>
        __host__ void run_div_scalar_inplace(T* ptr, const float64_t inv_scale, const size_t numel) {
            if (numel > 0) {
                div_scalar_inplace_kernel<T><<<grid_for(numel), BT_GPU_THREADS>>>(ptr, inv_scale, numel);
            }
        }
    }

    /**
     * GPU in-place division of a contiguous float-kind tensor by a scalar.
     * Mirrors the CPU backend's precision rules.
     * @brief HIP backend for GradScaler::unscale_tensor_.
     * @param grad Gradient tensor (updated in place).
     * @param scale Current scale factor.
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource On a CPU tensor.
     * @throws ErrorHandling::NotImplemented Non-contiguous or non-float dtype.
     */
    __host__ void gradscaler_div_scalar_inplace_gpu(Tensors::Tensor& grad, const float64_t scale) {
        if (grad.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(grad.device));
        }
        if (!grad.is_contiguous) {
            throw ErrorHandling::NotImplemented();
        }
        auto* const data = grad.data;
        const auto offset = grad.offset;
        const auto dtype = grad.dtype;
        const auto numel = grad.numel;
        /**
         * Captures raw pointers by value; the future lands on grad so teardown
         * drains the kernel before freeing the buffer.
         * @brief Pool worker for the GPU divide-in-place.
         * @note Lambda subcall. Self-execution prohibited.
         */
        const auto worker = [=] {
            using DType = DataTypes::DType;
            const auto inv = float64_t{1.0} / scale;
            switch (dtype) {
                case DType::Float32: run_div_scalar_inplace(static_cast<float32_t*>(data) + offset, inv, numel); break;
                case DType::Float64: run_div_scalar_inplace(static_cast<float64_t*>(data) + offset, inv, numel); break;
                case DType::Float16: run_div_scalar_inplace(static_cast<float16_t*>(data) + offset, inv, numel); break;
                case DType::BFloat16: run_div_scalar_inplace(static_cast<bfloat16_t*>(data) + offset, inv, numel); break;
                case DType::Float16x2: run_div_scalar_inplace(static_cast<float16x2_t*>(data) + offset, inv, numel); break;
                case DType::BFloat16x2: run_div_scalar_inplace(static_cast<bfloat16x2_t*>(data) + offset, inv, numel); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
            HIP_CHECK_ERROR(hipStreamSynchronize(nullptr));
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        grad.give_future(future);
    }
}
#endif
