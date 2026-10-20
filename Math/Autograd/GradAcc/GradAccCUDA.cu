/**
 * @file GradAccCUDA.cu
 * @author GreenBean
 * @brief CUDA backend for the grad_acc op (gradient accumulation g += d).
 * Divergent twin of Math/Autograd/GradAcc/GradAccHIP.cpp — the two backends are
 * written separately per project rule (no HIP-only unification), but must stay
 * behaviorally identical: same fp32/fp64 compute widths, same RNE narrowing,
 * same wrap integer adds.
 * @version 0.1
 * @date 2026-10-20
 */
#if defined(USE_CUDA)
#include <cstdint>
#include <limits>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "Math/Autograd/GradAcc/GradAcc.h"
#include "Math/Autograd/AutogradCudaShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    using namespace autograd_gpu;

    namespace {
        /**
         * Elementwise w[i] += d[i]. Floats single-rounding add in storage width;
         * halfs widen to fp32, add, narrow RNE; packed dtypes per lane; integers
         * wrap through the unsigned twin; int4x2 wraps per nibble. Matches the
         * CPU grad_acc kernels. Launched as grid_for(numel) blocks of BT_GPU_THREADS
         * threads, grid-stride over numel.
         * @brief Gradient accumulation kernel (all dtypes).
         * @param w Accumulator (gradient) data, updated in place, numel elements.
         * @param d Delta data added into w, numel elements.
         * @param numel Element count (storage elements, so packed dtypes count pairs).
         */
        template<typename T>
        __global__ void grad_acc_kernel(T* __restrict__ w, const T* __restrict__ d, const size_t numel) {
            const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t stride = blockDim.x * gridDim.x;
            for (size_t i = tid; i < numel; i += stride) {
                if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                    w[i] = w[i] + d[i];
                } else if constexpr (std::is_same_v<T, float16_t>) {
                    w[i] = __float2half_rn(__half2float(w[i]) + __half2float(d[i]));
                } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                    w[i] = dev_narrow_bf16(__bfloat162float(w[i]) + __bfloat162float(d[i]));
                } else if constexpr (std::is_same_v<T, float16x2_t>) {
                    const auto wv = w[i];
                    const auto dv = d[i];
                    const auto r_lo = __float2half_rn(__half2float(__low2half(wv)) + __half2float(__low2half(dv)));
                    const auto r_hi = __float2half_rn(__half2float(__high2half(wv)) + __half2float(__high2half(dv)));
                    w[i] = __halves2half2(r_lo, r_hi);
                } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                    const auto wv = w[i];
                    const auto dv = d[i];
                    const auto r_lo = dev_narrow_bf16(__bfloat162float(__low2bfloat16(wv)) + __bfloat162float(__low2bfloat16(dv)));
                    const auto r_hi = dev_narrow_bf16(__bfloat162float(__high2bfloat16(wv)) + __bfloat162float(__high2bfloat16(dv)));
                    w[i] = __halves2bfloat162(r_lo, r_hi);
                } else if constexpr (std::is_same_v<T, int4x2_t>) {
                    w[i] = dev_pack_int4(dev_int4_lo(w[i]) + dev_int4_lo(d[i]),
                                         dev_int4_hi(w[i]) + dev_int4_hi(d[i]));
                } else {
                    using U = std::make_unsigned_t<T>;
                    w[i] = static_cast<T>(static_cast<U>(static_cast<U>(w[i]) + static_cast<U>(d[i])));
                }
            }
        }

        /**
         * Skips the launch when numel is 0. The launch is asynchronous; the caller syncs.
         * @brief Host-side grad_acc launch.
         * @param w Accumulator (gradient) data, updated in place.
         * @param d Delta data added into w.
         * @param numel Element count.
         */
        template<typename T>
        __host__ void run_grad_acc(T* w, const T* d, const size_t numel) {
            if (numel > 0) {
                grad_acc_kernel<T><<<grid_for(numel), BT_GPU_THREADS>>>(w, d, numel);
            }
        }
    }

    /**
     * Queues the launch on the CPU thread pool (High priority) and returns; the shared
     * future is given to both tensors, and errors surface at sync(). dtype and numel
     * come from g; d is read with the same dtype and element count. Matches the CPU grad_acc.
     * @brief GPU gradient accumulation g += d (all dtypes).
     * @param g Accumulator gradient tensor, updated in place.
     * @param d Delta tensor added into g.
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if g or d is not on the GPU.
     * @throws ErrorHandling::NotImplemented (from the worker, at sync) for an unsupported dtype.
     */
    __host__ void grad_acc_gpu(Tensor& g, const Tensor& d) {
        if (g.device != Device::GPU || d.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(g.device));
        }
        auto* const g_data = g.data;
        const auto* const d_data = d.data;
        const auto g_offset = g.offset;
        const auto d_offset = d.offset;
        const auto dtype = g.dtype;
        const auto numel = g.numel;
        // Raw pointers captured by value: the future lands on both tensors, so
        // buffer teardown always drains this worker first.
        /**
         * Captures raw pointers, offsets, dtype and numel by value, so the lambda stays valid after the enclosing call returns.
         * Dispatches on dtype to run_grad_acc, which launches the accumulate kernel on the default stream, then calls cudaStreamSynchronize(nullptr) so the future completes only once the buffer writes are done.
         * An unsupported dtype throws ErrorHandling::NotImplemented.
         * Runs on a CPU thread pool thread (High priority) via the async pool; the host call has already returned. Any exception thrown here (including a failed CUDA_CHECK_ERROR) is captured in the shared future and rethrown at sync().
         * @brief Async pool worker that launches the grad_acc kernel for the tensor dtype and syncs the stream.
         */
        const auto worker = [=] {
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: run_grad_acc(static_cast<float32_t*>(g_data) + g_offset, static_cast<const float32_t*>(d_data) + d_offset, numel); break;
                case DType::Float64: run_grad_acc(static_cast<float64_t*>(g_data) + g_offset, static_cast<const float64_t*>(d_data) + d_offset, numel); break;
                case DType::Float16: run_grad_acc(static_cast<float16_t*>(g_data) + g_offset, static_cast<const float16_t*>(d_data) + d_offset, numel); break;
                case DType::BFloat16: run_grad_acc(static_cast<bfloat16_t*>(g_data) + g_offset, static_cast<const bfloat16_t*>(d_data) + d_offset, numel); break;
                case DType::Float16x2: run_grad_acc(static_cast<float16x2_t*>(g_data) + g_offset, static_cast<const float16x2_t*>(d_data) + d_offset, numel); break;
                case DType::BFloat16x2: run_grad_acc(static_cast<bfloat16x2_t*>(g_data) + g_offset, static_cast<const bfloat16x2_t*>(d_data) + d_offset, numel); break;
                case DType::Int8: run_grad_acc(static_cast<int8_t*>(g_data) + g_offset, static_cast<const int8_t*>(d_data) + d_offset, numel); break;
                case DType::UInt8: run_grad_acc(static_cast<uint8_t*>(g_data) + g_offset, static_cast<const uint8_t*>(d_data) + d_offset, numel); break;
                case DType::Int16: run_grad_acc(static_cast<int16_t*>(g_data) + g_offset, static_cast<const int16_t*>(d_data) + d_offset, numel); break;
                case DType::UInt16: run_grad_acc(static_cast<uint16_t*>(g_data) + g_offset, static_cast<const uint16_t*>(d_data) + d_offset, numel); break;
                case DType::Int32: run_grad_acc(static_cast<int32_t*>(g_data) + g_offset, static_cast<const int32_t*>(d_data) + d_offset, numel); break;
                case DType::UInt32: run_grad_acc(static_cast<uint32_t*>(g_data) + g_offset, static_cast<const uint32_t*>(d_data) + d_offset, numel); break;
                case DType::Int64: run_grad_acc(static_cast<int64_t*>(g_data) + g_offset, static_cast<const int64_t*>(d_data) + d_offset, numel); break;
                case DType::UInt64: run_grad_acc(static_cast<uint64_t*>(g_data) + g_offset, static_cast<const uint64_t*>(d_data) + d_offset, numel); break;
                case DType::Int4x2: run_grad_acc(static_cast<int4x2_t*>(g_data) + g_offset, static_cast<const int4x2_t*>(d_data) + d_offset, numel); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
            // The kernel is in-flight when the launch returns; sync so the
            // future only completes once the buffer writes are done.
            CUDA_CHECK_ERROR(cudaStreamSynchronize(nullptr));
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        g.give_future(future);
        d.give_future(future);
    }
}
#endif
