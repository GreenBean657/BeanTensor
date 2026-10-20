/**
 * @file ReluHIP.cpp
 * @author GreenBean
 * @brief HIP/ROCm backend for relu.
 * Divergent twin of Math/ComplexOps/Relu/ReluCUDA.cu — the two backends are
 * written separately per project rule (no HIP-only unification), but must stay
 * behaviorally identical: same fp32/fp64 compute widths, same RNE narrowing,
 * same NaN propagation.
 * @version 0.1
 * @date 2026-10-20
 */
#if defined(USE_HIP)
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>

#include <hip/hip_runtime.h>

#include "Math/ComplexOps/Relu/Relu.h"
#include "Math/ComplexOps/ComplexOpsHipShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    using namespace complexops_gpu;
    namespace {
        // --------------------------- relu kernel ---------------------------

        /**
         * Elementwise relu: out[i] = in[i] < 0 ? 0 : in[i]. NaN propagates (the
         * comparison is false for NaN), -0.0 survives; half kinds widen to fp32,
         * relu, and narrow RNE; packed dtypes relu per lane; unsigned integers
         * copy through.
         * @brief Elementwise relu kernel (all dtypes).
         * @param in Input tensor data, numel elements.
         * @param out Output tensor data, numel elements.
         * @param numel Total element count of in/out.
         */
        template<typename T>
        __global__ void relu_kernel(const T* __restrict__ in, T* __restrict__ out, const size_t numel) {
            const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t stride = blockDim.x * gridDim.x;
            for (size_t i = tid; i < numel; i += stride) {
                if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                    out[i] = in[i] < T{0} ? T{0} : in[i];
                } else if constexpr (std::is_same_v<T, float16_t>) {
                    const auto wide = __half2float(in[i]);
                    out[i] = __float2half_rn(wide < 0.0f ? 0.0f : wide);
                } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                    const auto wide = __bfloat162float(in[i]);
                    out[i] = dev_narrow_bf16(wide < 0.0f ? 0.0f : wide);
                } else if constexpr (std::is_same_v<T, float16x2_t>) {
                    const auto v = in[i];
                    const auto lo = __half2float(__low2half(v));
                    const auto hi = __half2float(__high2half(v));
                    out[i] = __halves2half2(__float2half_rn(lo < 0.0f ? 0.0f : lo),
                                            __float2half_rn(hi < 0.0f ? 0.0f : hi));
                } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                    const auto v = in[i];
                    const auto lo = __bfloat162float(__low2bfloat16(v));
                    const auto hi = __bfloat162float(__high2bfloat16(v));
                    out[i] = __halves2bfloat162(dev_narrow_bf16(lo < 0.0f ? 0.0f : lo),
                                                dev_narrow_bf16(hi < 0.0f ? 0.0f : hi));
                } else if constexpr (std::is_same_v<T, int4x2_t>) {
                    const auto lo = dev_int4_lo(in[i]);
                    const auto hi = dev_int4_hi(in[i]);
                    out[i] = dev_pack_int4(lo < 0 ? 0 : lo, hi < 0 ? 0 : hi);
                } else if constexpr (std::is_signed_v<T>) {
                    out[i] = in[i] < T{0} ? T{0} : in[i];
                } else {
                    out[i] = in[i];
                }
            }
        }

        /**
         * Skips the launch for empty tensors; the kernel is asynchronous, the caller syncs.
         * @brief Host-side relu launch.
         * @param in Input tensor data, numel elements.
         * @param out Output tensor data, numel elements.
         * @param numel Total element count of in/out.
         */
        template<typename T>
        __host__ void run_relu(const T* in, T* out, const size_t numel) {
            if (numel > 0) {
                relu_kernel<T><<<grid_for(numel), BT_GPU_THREADS>>>(in, out, numel);
            }
        }
    }

    /**
     * Queues one pool worker that dispatches on dtype and launches the relu kernel. The future lands
     * on both tensors so teardown drains it.
     * @brief GPU relu: out[i] = t[i] < 0 ? 0 : t[i].
     * @param t Input tensor, GPU resident, contiguous.
     * @param out Output tensor, same shape and dtype as t.
     * @note Throws DirectFunctionCallOnWrongDataSource when any tensor is not on the GPU.
     */
    __host__ void relu_gpu(const Tensor& t, Tensor& out) {
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
         * Dispatches on dtype to run_relu, then syncs the default stream so the future only
         * completes once the buffer writes are done.
         * @brief Pool worker for the relu launch.
         * @note Lambda subcall. Self-execution prohibited.
         */
        const auto worker = [=] {
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: run_relu(static_cast<const float32_t*>(t_data) + t_offset, static_cast<float32_t*>(out_data) + out_offset, numel); break;
                case DType::Float64: run_relu(static_cast<const float64_t*>(t_data) + t_offset, static_cast<float64_t*>(out_data) + out_offset, numel); break;
                case DType::Float16: run_relu(static_cast<const float16_t*>(t_data) + t_offset, static_cast<float16_t*>(out_data) + out_offset, numel); break;
                case DType::BFloat16: run_relu(static_cast<const bfloat16_t*>(t_data) + t_offset, static_cast<bfloat16_t*>(out_data) + out_offset, numel); break;
                case DType::Float16x2: run_relu(static_cast<const float16x2_t*>(t_data) + t_offset, static_cast<float16x2_t*>(out_data) + out_offset, numel); break;
                case DType::BFloat16x2: run_relu(static_cast<const bfloat16x2_t*>(t_data) + t_offset, static_cast<bfloat16x2_t*>(out_data) + out_offset, numel); break;
                case DType::Int8: run_relu(static_cast<const int8_t*>(t_data) + t_offset, static_cast<int8_t*>(out_data) + out_offset, numel); break;
                case DType::UInt8: run_relu(static_cast<const uint8_t*>(t_data) + t_offset, static_cast<uint8_t*>(out_data) + out_offset, numel); break;
                case DType::Int16: run_relu(static_cast<const int16_t*>(t_data) + t_offset, static_cast<int16_t*>(out_data) + out_offset, numel); break;
                case DType::UInt16: run_relu(static_cast<const uint16_t*>(t_data) + t_offset, static_cast<uint16_t*>(out_data) + out_offset, numel); break;
                case DType::Int32: run_relu(static_cast<const int32_t*>(t_data) + t_offset, static_cast<int32_t*>(out_data) + out_offset, numel); break;
                case DType::UInt32: run_relu(static_cast<const uint32_t*>(t_data) + t_offset, static_cast<uint32_t*>(out_data) + out_offset, numel); break;
                case DType::Int64: run_relu(static_cast<const int64_t*>(t_data) + t_offset, static_cast<int64_t*>(out_data) + out_offset, numel); break;
                case DType::UInt64: run_relu(static_cast<const uint64_t*>(t_data) + t_offset, static_cast<uint64_t*>(out_data) + out_offset, numel); break;
                case DType::Int4x2: run_relu(static_cast<const int4x2_t*>(t_data) + t_offset, static_cast<int4x2_t*>(out_data) + out_offset, numel); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
            HIP_CHECK_ERROR(hipStreamSynchronize(nullptr));
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        t.give_future(future);
        out.give_future(future);
    }
}
#endif
