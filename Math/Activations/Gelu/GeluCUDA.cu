/**
 * @file GeluCUDA.cu
 * @author GreenBean
 * @brief CUDA backend for GELU.
 * Divergent twin of Math/Activations/Gelu/GeluHIP.cpp — the two backends are
 * written separately per project rule (no HIP-only unification), but must stay
 * behaviorally identical: same fp32/fp64 compute widths, same RNE narrowing,
 * same NaN/Inf contract.
 * @version 0.1
 * @date 2026-11-10
 */
#if defined(USE_CUDA)
#include <cstdint>
#include <memory>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "Math/Activations/Gelu/Gelu.h"
#include "Math/Activations/Gelu/GeluKernels.h"
#include "Math/Activations/ActivationsCudaShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    using namespace activations_gpu;
    using namespace BeanTensor::Math::Activations::detail;

    namespace {
        // --------------------------- gelu kernel ---------------------------

        /**
         * Elementwise GELU: out[i] = 0.5*in[i]*(1+erf(in[i]/sqrt(2))).  Half and
         * packed dtypes widen to fp32, compute, and narrow RNE.  The piecewise
         * non-finite branches are applied in the fp32 compute width.
         * @brief Elementwise GELU kernel (float kinds).
         * @param in Input tensor data, numel elements.
         * @param out Output tensor data, numel elements.
         * @param numel Total element count of in/out.
         */
        template<typename T>
        __global__ void gelu_kernel(const T* __restrict__ in, T* __restrict__ out, const size_t numel) {
            const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t stride = blockDim.x * gridDim.x;
            for (size_t i = tid; i < numel; i += stride) {
                if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                    out[i] = gelu_exact_lane(in[i]);
                } else if constexpr (std::is_same_v<T, float16_t>) {
                    const auto wide = __half2float(in[i]);
                    out[i] = __float2half_rn(gelu_exact_lane(wide));
                } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                    const auto wide = __bfloat162float(in[i]);
                    out[i] = dev_narrow_bf16(gelu_exact_lane(wide));
                } else if constexpr (std::is_same_v<T, float16x2_t>) {
                    const auto v = in[i];
                    const auto lo = __half2float(__low2half(v));
                    const auto hi = __half2float(__high2half(v));
                    out[i] = __halves2half2(__float2half_rn(gelu_exact_lane(lo)),
                                            __float2half_rn(gelu_exact_lane(hi)));
                } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                    const auto v = in[i];
                    const auto lo = __bfloat162float(__low2bfloat16(v));
                    const auto hi = __bfloat162float(__high2bfloat16(v));
                    out[i] = __halves2bfloat162(dev_narrow_bf16(gelu_exact_lane(lo)),
                                                dev_narrow_bf16(gelu_exact_lane(hi)));
                }
            }
        }

        /**
         * Skips the launch for empty tensors; the kernel is asynchronous, the caller syncs.
         * @brief Host-side GELU launch.
         * @param in Input tensor data, numel elements.
         * @param out Output tensor data, numel elements.
         * @param numel Total element count of in/out.
         */
        template<typename T>
        __host__ void run_gelu(const T* in, T* out, const size_t numel) {
            if (numel > 0) {
                gelu_kernel<T><<<grid_for(numel), BT_GPU_THREADS>>>(in, out, numel);
            }
        }
    }

    /**
     * Queues one pool worker that dispatches on dtype and launches the GELU kernel.
     * The future lands on both tensors so teardown drains it.
     * @brief GPU GELU: out[i] = 0.5*in[i]*(1+erf(in[i]/sqrt(2))).
     * @param t Input tensor, GPU resident, contiguous.
     * @param out Output tensor, same shape and dtype as t.
     * @note Throws DirectFunctionCallOnWrongDataSource when any tensor is not on the GPU.
     */
    __host__ void gelu_gpu(const Tensor& t, const Tensor& out) {
        if (t.device != Device::GPU || out.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(t.device));
        }
        const auto* const t_data = t.data;
        auto* const out_data = out.data;
        const auto t_offset = t.offset;
        const auto out_offset = out.offset;
        const auto dtype = t.dtype;
        const auto numel = t.numel;
        /**
         * Dispatches on dtype to run_gelu, then syncs the default stream so the future only
         * completes once the buffer writes are done.
         * @brief Pool worker for the GELU launch.
         * @note Lambda subcall. Self-execution prohibited.
         */
        const auto worker = [=] {
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: run_gelu(static_cast<const float32_t*>(t_data) + t_offset, static_cast<float32_t*>(out_data) + out_offset, numel); break;
                case DType::Float64: run_gelu(static_cast<const float64_t*>(t_data) + t_offset, static_cast<float64_t*>(out_data) + out_offset, numel); break;
                case DType::Float16: run_gelu(static_cast<const float16_t*>(t_data) + t_offset, static_cast<float16_t*>(out_data) + out_offset, numel); break;
                case DType::BFloat16: run_gelu(static_cast<const bfloat16_t*>(t_data) + t_offset, static_cast<bfloat16_t*>(out_data) + out_offset, numel); break;
                case DType::Float16x2: run_gelu(static_cast<const float16x2_t*>(t_data) + t_offset, static_cast<float16x2_t*>(out_data) + out_offset, numel); break;
                case DType::BFloat16x2: run_gelu(static_cast<const bfloat16x2_t*>(t_data) + t_offset, static_cast<bfloat16x2_t*>(out_data) + out_offset, numel); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
            CUDA_CHECK_ERROR(cudaStreamSynchronize(nullptr));
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        t.give_future(future);
        out.give_future(future);
    }
}
#endif
