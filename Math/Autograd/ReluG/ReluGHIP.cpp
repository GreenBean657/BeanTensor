/**
 * @file ReluGHIP.cpp
 * @author GreenBean
 * @brief HIP backend for the relu_g op (relu backward dx = x > 0 ? dy : 0).
 * Divergent twin of Math/Autograd/ReluG/ReluGCUDA.cu — the two backends are
 * written separately per project rule (no HIP-only unification), but must stay
 * behaviorally identical: same fp32/fp64 compute widths, same RNE narrowing,
 * same NaN handling.
 * @version 0.1
 * @date 2026-10-20
 */
#if defined(USE_HIP)
#include <cstdint>
#include <type_traits>

#include <hip/hip_runtime.h>

#include "Math/Autograd/ReluG/ReluG.h"
#include "Math/Autograd/AutogradHipShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    using namespace autograd_gpu;

    namespace {
        /**
         * dx[i] = x[i] > 0 ? dy[i] : 0 - a select, not a multiply: NaN in x
         * fails the comparison and zeroes the lane, a selected dy is stored
         * bit-exact (no rounding). float16's compare widened; packed per lane;
         * unsigned integers copy dy through; int4x2 per nibble. Launched as
         * grid_for(numel) blocks of BT_GPU_THREADS threads, grid-stride over numel.
         * @brief relu backward kernel (all dtypes).
         * @param x Forward input of the relu, numel elements.
         * @param dy Upstream gradient, numel elements.
         * @param dx Output gradient, numel elements.
         * @param numel Element count (storage elements, so packed dtypes count pairs).
         */
        template<typename T>
        __global__ void relu_g_kernel(const T* __restrict__ x, const T* __restrict__ dy, T* __restrict__ dx, const size_t numel) {
            const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t stride = blockDim.x * gridDim.x;
            for (size_t i = tid; i < numel; i += stride) {
                if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t> || std::is_signed_v<T>) {
                    dx[i] = x[i] > T{0} ? dy[i] : T{0};
                } else if constexpr (std::is_same_v<T, float16_t>) {
                    dx[i] = __half2float(x[i]) > 0.0f ? dy[i] : __float2half_rn(0.0f);
                } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                    dx[i] = __bfloat162float(x[i]) > 0.0f ? dy[i] : dev_narrow_bf16(0.0f);
                } else if constexpr (std::is_same_v<T, float16x2_t>) {
                    const auto xv = x[i];
                    const auto dv = dy[i];
                    const auto zero = __float2half_rn(0.0f);
                    dx[i] = __halves2half2(__half2float(__low2half(xv)) > 0.0f ? __low2half(dv) : zero,
                                           __half2float(__high2half(xv)) > 0.0f ? __high2half(dv) : zero);
                } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                    const auto xv = x[i];
                    const auto dv = dy[i];
                    const auto zero = dev_narrow_bf16(0.0f);
                    dx[i] = __halves2bfloat162(__bfloat162float(__low2bfloat16(xv)) > 0.0f ? __low2bfloat16(dv) : zero,
                                               __bfloat162float(__high2bfloat16(xv)) > 0.0f ? __high2bfloat16(dv) : zero);
                } else if constexpr (std::is_same_v<T, int4x2_t>) {
                    dx[i] = dev_pack_int4(dev_int4_lo(x[i]) > 0 ? dev_int4_lo(dy[i]) : 0,
                                          dev_int4_hi(x[i]) > 0 ? dev_int4_hi(dy[i]) : 0);
                } else {
                    dx[i] = dy[i];
                }
            }
        }

        /**
         * Skips the launch when numel is 0. The launch is asynchronous; the caller syncs.
         * @brief Host-side relu_g launch.
         * @param x Forward input of the relu.
         * @param dy Upstream gradient.
         * @param dx Output gradient.
         * @param numel Element count.
         */
        template<typename T>
        __host__ void run_relu_g(const T* x, const T* dy, T* dx, const size_t numel) {
            if (numel > 0) {
                relu_g_kernel<T><<<grid_for(numel), BT_GPU_THREADS>>>(x, dy, dx, numel);
            }
        }
    }

    /**
     * Queues the launch on the CPU thread pool (High priority) and returns; the shared
     * future is given to all three tensors, and errors surface at sync(). dtype and
     * numel come from x. Matches the CPU relu_g (select, NaN in x zeroes the lane).
     * @brief GPU relu backward dx = x > 0 ? dy : 0 (all dtypes).
     * @param dy Upstream gradient tensor.
     * @param x Forward input tensor of the relu.
     * @param dx Output gradient tensor.
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if any tensor is not on the GPU.
     * @throws ErrorHandling::NotImplemented (from the worker, at sync) for an unsupported dtype.
     */
    __host__ void relu_g_gpu(const Tensor& dy, const Tensor& x, const Tensor& dx) {
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
         * Dispatches on dtype to run_relu_g, which launches the relu backward kernel on the default stream, then calls hipStreamSynchronize(nullptr) so the future completes only once dx is written.
         * An unsupported dtype throws ErrorHandling::NotImplemented.
         * Runs on a CPU thread pool thread (High priority) via the async pool; the host call has already returned. Any exception thrown here (including a failed HIP_CHECK_ERROR) is captured in the shared future and rethrown at sync().
         * @brief Async pool worker that launches the relu_g kernel for the tensor dtype and syncs the stream.
         */
        const auto worker = [=] {
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: run_relu_g(static_cast<const float32_t*>(x_data) + x_offset, static_cast<const float32_t*>(dy_data) + dy_offset, static_cast<float32_t*>(dx_data) + dx_offset, numel); break;
                case DType::Float64: run_relu_g(static_cast<const float64_t*>(x_data) + x_offset, static_cast<const float64_t*>(dy_data) + dy_offset, static_cast<float64_t*>(dx_data) + dx_offset, numel); break;
                case DType::Float16: run_relu_g(static_cast<const float16_t*>(x_data) + x_offset, static_cast<const float16_t*>(dy_data) + dy_offset, static_cast<float16_t*>(dx_data) + dx_offset, numel); break;
                case DType::BFloat16: run_relu_g(static_cast<const bfloat16_t*>(x_data) + x_offset, static_cast<const bfloat16_t*>(dy_data) + dy_offset, static_cast<bfloat16_t*>(dx_data) + dx_offset, numel); break;
                case DType::Float16x2: run_relu_g(static_cast<const float16x2_t*>(x_data) + x_offset, static_cast<const float16x2_t*>(dy_data) + dy_offset, static_cast<float16x2_t*>(dx_data) + dx_offset, numel); break;
                case DType::BFloat16x2: run_relu_g(static_cast<const bfloat16x2_t*>(x_data) + x_offset, static_cast<const bfloat16x2_t*>(dy_data) + dy_offset, static_cast<bfloat16x2_t*>(dx_data) + dx_offset, numel); break;
                case DType::Int8: run_relu_g(static_cast<const int8_t*>(x_data) + x_offset, static_cast<const int8_t*>(dy_data) + dy_offset, static_cast<int8_t*>(dx_data) + dx_offset, numel); break;
                case DType::UInt8: run_relu_g(static_cast<const uint8_t*>(x_data) + x_offset, static_cast<const uint8_t*>(dy_data) + dy_offset, static_cast<uint8_t*>(dx_data) + dx_offset, numel); break;
                case DType::Int16: run_relu_g(static_cast<const int16_t*>(x_data) + x_offset, static_cast<const int16_t*>(dy_data) + dy_offset, static_cast<int16_t*>(dx_data) + dx_offset, numel); break;
                case DType::UInt16: run_relu_g(static_cast<const uint16_t*>(x_data) + x_offset, static_cast<const uint16_t*>(dy_data) + dy_offset, static_cast<uint16_t*>(dx_data) + dx_offset, numel); break;
                case DType::Int32: run_relu_g(static_cast<const int32_t*>(x_data) + x_offset, static_cast<const int32_t*>(dy_data) + dy_offset, static_cast<int32_t*>(dx_data) + dx_offset, numel); break;
                case DType::UInt32: run_relu_g(static_cast<const uint32_t*>(x_data) + x_offset, static_cast<const uint32_t*>(dy_data) + dy_offset, static_cast<uint32_t*>(dx_data) + dx_offset, numel); break;
                case DType::Int64: run_relu_g(static_cast<const int64_t*>(x_data) + x_offset, static_cast<const int64_t*>(dy_data) + dy_offset, static_cast<int64_t*>(dx_data) + dx_offset, numel); break;
                case DType::UInt64: run_relu_g(static_cast<const uint64_t*>(x_data) + x_offset, static_cast<const uint64_t*>(dy_data) + dy_offset, static_cast<uint64_t*>(dx_data) + dx_offset, numel); break;
                case DType::Int4x2: run_relu_g(static_cast<const int4x2_t*>(x_data) + x_offset, static_cast<const int4x2_t*>(dy_data) + dy_offset, static_cast<int4x2_t*>(dx_data) + dx_offset, numel); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
            HIP_CHECK_ERROR(hipStreamSynchronize(nullptr));
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        dy.give_future(future);
        x.give_future(future);
        dx.give_future(future);
    }
}
#endif
