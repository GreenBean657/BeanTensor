/**
 * @file BiasAddHIP.cpp
 * @author GreenBean
 * @brief HIP/ROCm backend for bias_add.
 * Divergent twin of Math/LinearAlgebra/BiasAdd/BiasAddCUDA.cu — the two backends are
 * written separately per project rule (no HIP-only unification), but must stay
 * behaviorally identical: same fp32/fp64 compute widths, same RNE narrowing,
 * same NaN propagation, same wrap integer adds.
 * @version 0.1
 * @date 2026-10-20
 */
#if defined(USE_HIP)
#include <cstdint>
#include <memory>
#include <type_traits>

#include <hip/hip_runtime.h>

#include "Math/LinearAlgebra/BiasAdd/BiasAdd.h"
#include "Math/LinearAlgebra/LinearAlgebraHipShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    using namespace linearalgebra_gpu;
    namespace {
        // --------------------------- bias_add kernel ---------------------------

        /**
         * Elementwise bias add: out[i] = in[i] + bias[i % cols]. Float kinds
         * compute in their storage width's natural accumulator (fp32 for
         * fp16/bf16/fp32, fp64 for fp64) and narrow RNE on store, exactly
         * matching the CPU kernels. Integers add modularly (wrap) through the
         * unsigned twin; int4x2 wraps per nibble.
         * @brief Elementwise bias_add kernel (all dtypes).
         * @param in Input tensor data, numel elements.
         * @param bias Bias vector, cols elements.
         * @param out Output tensor data, numel elements.
         * @param cols Last-dimension size; bias index is i % cols.
         * @param numel Total element count of in/out.
         */
        template<typename T>
        __global__ void bias_add_kernel(const T* __restrict__ in, const T* __restrict__ bias, T* __restrict__ out,
                                        const size_t cols, const size_t numel) {
            const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t stride = blockDim.x * gridDim.x;
            for (size_t i = tid; i < numel; i += stride) {
                const auto col = i % cols;
                if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                    out[i] = in[i] + bias[col];
                } else if constexpr (std::is_same_v<T, float16_t>) {
                    out[i] = __float2half_rn(__half2float(in[i]) + __half2float(bias[col]));
                } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                    out[i] = dev_narrow_bf16(__bfloat162float(in[i]) + __bfloat162float(bias[col]));
                } else if constexpr (std::is_same_v<T, float16x2_t>) {
                    const auto iv = in[i];
                    const auto bv = bias[col];
                    const auto r_lo = __float2half_rn(__half2float(__low2half(iv)) + __half2float(__low2half(bv)));
                    const auto r_hi = __float2half_rn(__half2float(__high2half(iv)) + __half2float(__high2half(bv)));
                    out[i] = __halves2half2(r_lo, r_hi);
                } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                    const auto iv = in[i];
                    const auto bv = bias[col];
                    const auto r_lo = dev_narrow_bf16(__bfloat162float(__low2bfloat16(iv)) + __bfloat162float(__low2bfloat16(bv)));
                    const auto r_hi = dev_narrow_bf16(__bfloat162float(__high2bfloat16(iv)) + __bfloat162float(__high2bfloat16(bv)));
                    out[i] = __halves2bfloat162(r_lo, r_hi);
                } else if constexpr (std::is_same_v<T, int4x2_t>) {
                    out[i] = dev_pack_int4(dev_int4_lo(in[i]) + dev_int4_lo(bias[col]),
                                           dev_int4_hi(in[i]) + dev_int4_hi(bias[col]));
                } else {
                    using U = std::make_unsigned_t<T>;
                    out[i] = static_cast<T>(static_cast<U>(static_cast<U>(in[i]) + static_cast<U>(bias[col])));
                }
            }
        }

        /**
         * Skips the launch for empty tensors; the kernel is asynchronous, the caller syncs.
         * @brief Host-side bias_add launch.
         * @param in Input tensor data, numel elements.
         * @param bias Bias vector, cols elements.
         * @param out Output tensor data, numel elements.
         * @param cols Last-dimension size; bias index is i % cols.
         * @param numel Total element count of in/out.
         */
        template<typename T>
        __host__ void run_bias_add(const T* in, const T* bias, T* out, const size_t cols, const size_t numel) {
            if (numel > 0) {
                bias_add_kernel<T><<<grid_for(numel), BT_GPU_THREADS>>>(in, bias, out, cols, numel);
            }
        }
    }

    /**
     * Queues one pool worker that dispatches on dtype and launches the bias_add kernel. The future lands
     * on all three tensors so teardown drains it.
     * @brief GPU bias_add: out[i] = t[i] + bias[i % cols].
     * @param t Input tensor, GPU resident, contiguous.
     * @param bias Rank-1 bias tensor matching the last dimension of t.
     * @param out Output tensor, same shape and dtype as t.
     * @note Throws DirectFunctionCallOnWrongDataSource when any tensor is not on the GPU.
     */
    __host__ void bias_add_gpu(const Tensor& t, const Tensor& bias, const Tensor& out) {
        if (t.device != Device::GPU || bias.device != Device::GPU || out.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(t.device));
        }
        const auto* const t_data = t.data;
        const auto* const b_data = bias.data;
        auto* const out_data = out.data;
        const auto t_offset = t.offset;
        const auto b_offset = bias.offset;
        const auto out_offset = out.offset;
        const auto dtype = t.dtype;
        const auto cols = t.shape[t.ndim - 1];
        const auto numel = t.numel;
        // Raw pointers captured by value: the future lands on all three tensors,
        // so buffer teardown always drains this worker first.
        /**
         * Dispatches on dtype to run_bias_add, then syncs the default stream so the future only
         * completes once the buffer writes are done.
         * @brief Pool worker for the bias_add launch.
         * @note Lambda subcall. Self-execution prohibited.
         */
        const auto worker = [=] {
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: run_bias_add(static_cast<const float32_t*>(t_data) + t_offset, static_cast<const float32_t*>(b_data) + b_offset, static_cast<float32_t*>(out_data) + out_offset, cols, numel); break;
                case DType::Float64: run_bias_add(static_cast<const float64_t*>(t_data) + t_offset, static_cast<const float64_t*>(b_data) + b_offset, static_cast<float64_t*>(out_data) + out_offset, cols, numel); break;
                case DType::Float16: run_bias_add(static_cast<const float16_t*>(t_data) + t_offset, static_cast<const float16_t*>(b_data) + b_offset, static_cast<float16_t*>(out_data) + out_offset, cols, numel); break;
                case DType::BFloat16: run_bias_add(static_cast<const bfloat16_t*>(t_data) + t_offset, static_cast<const bfloat16_t*>(b_data) + b_offset, static_cast<bfloat16_t*>(out_data) + out_offset, cols, numel); break;
                case DType::Float16x2: run_bias_add(static_cast<const float16x2_t*>(t_data) + t_offset, static_cast<const float16x2_t*>(b_data) + b_offset, static_cast<float16x2_t*>(out_data) + out_offset, cols, numel); break;
                case DType::BFloat16x2: run_bias_add(static_cast<const bfloat16x2_t*>(t_data) + t_offset, static_cast<const bfloat16x2_t*>(b_data) + b_offset, static_cast<bfloat16x2_t*>(out_data) + out_offset, cols, numel); break;
                case DType::Int8: run_bias_add(static_cast<const int8_t*>(t_data) + t_offset, static_cast<const int8_t*>(b_data) + b_offset, static_cast<int8_t*>(out_data) + out_offset, cols, numel); break;
                case DType::UInt8: run_bias_add(static_cast<const uint8_t*>(t_data) + t_offset, static_cast<const uint8_t*>(b_data) + b_offset, static_cast<uint8_t*>(out_data) + out_offset, cols, numel); break;
                case DType::Int16: run_bias_add(static_cast<const int16_t*>(t_data) + t_offset, static_cast<const int16_t*>(b_data) + b_offset, static_cast<int16_t*>(out_data) + out_offset, cols, numel); break;
                case DType::UInt16: run_bias_add(static_cast<const uint16_t*>(t_data) + t_offset, static_cast<const uint16_t*>(b_data) + b_offset, static_cast<uint16_t*>(out_data) + out_offset, cols, numel); break;
                case DType::Int32: run_bias_add(static_cast<const int32_t*>(t_data) + t_offset, static_cast<const int32_t*>(b_data) + b_offset, static_cast<int32_t*>(out_data) + out_offset, cols, numel); break;
                case DType::UInt32: run_bias_add(static_cast<const uint32_t*>(t_data) + t_offset, static_cast<const uint32_t*>(b_data) + b_offset, static_cast<uint32_t*>(out_data) + out_offset, cols, numel); break;
                case DType::Int64: run_bias_add(static_cast<const int64_t*>(t_data) + t_offset, static_cast<const int64_t*>(b_data) + b_offset, static_cast<int64_t*>(out_data) + out_offset, cols, numel); break;
                case DType::UInt64: run_bias_add(static_cast<const uint64_t*>(t_data) + t_offset, static_cast<const uint64_t*>(b_data) + b_offset, static_cast<uint64_t*>(out_data) + out_offset, cols, numel); break;
                case DType::Int4x2: run_bias_add(static_cast<const int4x2_t*>(t_data) + t_offset, static_cast<const int4x2_t*>(b_data) + b_offset, static_cast<int4x2_t*>(out_data) + out_offset, cols, numel); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
            // The kernel is in-flight when the launch returns; sync so the
            // future only completes once the buffer writes are done.
            HIP_CHECK_ERROR(hipStreamSynchronize(nullptr));
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        t.give_future(future);
        bias.give_future(future);
        out.give_future(future);
    }
}
#endif
