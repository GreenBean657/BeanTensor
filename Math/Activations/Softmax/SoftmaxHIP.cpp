/**
 * @file SoftmaxHIP.cpp
 * @author GreenBean
 * @brief HIP/ROCm backend for standalone rank-2 softmax.
 * Divergent twin of Math/Activations/Softmax/SoftmaxCUDA.cu — the two backends
 * are written separately per project rule (no HIP-only unification), but must
 * stay behaviorally identical: same fp32/fp64 compute widths, same RNE
 * narrowing, same NaN propagation.
 * @version 0.1
 * @date 2026-10-28
 */
#if defined(USE_HIP)
#include <cstdint>
#include <limits>
#include <type_traits>

#include <hip/hip_runtime.h>

#include "Math/Activations/Softmax/Softmax.h"
#include "Math/Activations/Softmax/SoftmaxHipShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    using namespace softmax_gpu_helpers;

    namespace {
        template<typename T, typename Acc>
        __global__ void softmax_rows_kernel(const T* __restrict__ in, T* __restrict__ out,
                                            const size_t rows, const size_t cols_store, const size_t cols_eff) {
            const auto r = blockIdx.x;
            if (r >= rows) {
                return;
            }
            const auto* row = in + r * cols_store;
            auto* out_row = out + r * cols_store;
            __shared__ Acc shared[BT_GPU_THREADS];
            __shared__ Acc row_max;
            __shared__ Acc row_sum;
            const auto tid = threadIdx.x;
            auto m = acc_lowest<Acc>();
            for (auto j = tid; j < cols_eff; j += BT_GPU_THREADS) {
                if (
                    const auto x = sm_widen<T, Acc>(row, j);
                    x > m
                ) {
                    m = x;
                }
            }
            shared[tid] = m;
            __syncthreads();
            for (auto s = BT_GPU_THREADS / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared[tid] = shared[tid] > shared[tid + s] ? shared[tid] : shared[tid + s];
                }
                __syncthreads();
            }
            if (tid == 0) {
                row_max = shared[0];
            }
            __syncthreads();
            m = row_max;
            auto sum = Acc{0};
            for (auto j = tid; j < cols_eff; j += BT_GPU_THREADS) {
                sum += acc_exp(sm_widen<T, Acc>(row, j) - m);
            }
            shared[tid] = sum;
            __syncthreads();
            for (auto s = BT_GPU_THREADS / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared[tid] = shared[tid] + shared[tid + s];
                }
                __syncthreads();
            }
            if (tid == 0) {
                row_sum = shared[0];
            }
            __syncthreads();
            const auto total = row_sum;
            for (auto e = tid; e < cols_store; e += BT_GPU_THREADS) {
                if constexpr (std::is_same_v<T, float16x2_t>) {
                    out_row[e] = __halves2half2(
                        __float2half_rn(acc_exp(sm_widen<T, Acc>(row, 2 * e) - m) / total),
                        __float2half_rn(acc_exp(sm_widen<T, Acc>(row, 2 * e + 1) - m) / total));
                } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                    out_row[e] = __halves2bfloat162(
                        dev_narrow_bf16(acc_exp(sm_widen<T, Acc>(row, 2 * e) - m) / total),
                        dev_narrow_bf16(acc_exp(sm_widen<T, Acc>(row, 2 * e + 1) - m) / total));
                } else if constexpr (std::is_same_v<T, int4x2_t>) {
                    const auto lo = static_cast<int32_t>(acc_exp(sm_widen<T, Acc>(row, 2 * e) - m) / total);
                    const auto hi = static_cast<int32_t>(acc_exp(sm_widen<T, Acc>(row, 2 * e + 1) - m) / total);
                    out_row[e] = dev_pack_int4(
                        lo <= -8 ? -8 : lo >= 7 ? 7 : lo,
                        hi <= -8 ? -8 : hi >= 7 ? 7 : hi);
                } else {
                    out_row[e] = sm_narrow<T, Acc>(acc_exp(sm_widen<T, Acc>(row, e) - m) / total);
                }
            }
        }

        template<typename T, typename Acc>
        __host__ void run_softmax(const T* in, T* out, const size_t rows, const size_t cols_store, const size_t cols_eff) {
            if (rows > 0) {
                softmax_rows_kernel<T, Acc><<<rows, BT_GPU_THREADS>>>(in, out, rows, cols_store, cols_eff);
            }
        }
    }

    __host__ void softmax_gpu(const Tensor& t, const Tensor& out) {
        if (t.device != Device::GPU || out.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(t.device));
        }
        const auto* const t_data = t.data;
        auto* const out_data = out.data;
        const auto t_offset = t.offset;
        const auto out_offset = out.offset;
        const auto dtype = t.dtype;
        const auto rows = t.shape[0];
        const auto cols_store = t.shape[1];
        const auto packed = dtype == DataTypes::DType::Float16x2 || dtype == DataTypes::DType::BFloat16x2 || dtype == DataTypes::DType::Int4x2;
        const auto cols_eff = cols_store * (packed ? size_t{2} : size_t{1});
        const auto worker = [=] {
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: run_softmax<float32_t, float32_t>(static_cast<const float32_t*>(t_data) + t_offset, static_cast<float32_t*>(out_data) + out_offset, rows, cols_store, cols_eff); break;
                case DType::Float64: run_softmax<float64_t, float64_t>(static_cast<const float64_t*>(t_data) + t_offset, static_cast<float64_t*>(out_data) + out_offset, rows, cols_store, cols_eff); break;
                case DType::Float16: run_softmax<float16_t, float32_t>(static_cast<const float16_t*>(t_data) + t_offset, static_cast<float16_t*>(out_data) + out_offset, rows, cols_store, cols_eff); break;
                case DType::BFloat16: run_softmax<bfloat16_t, float32_t>(static_cast<const bfloat16_t*>(t_data) + t_offset, static_cast<bfloat16_t*>(out_data) + out_offset, rows, cols_store, cols_eff); break;
                case DType::Float16x2: run_softmax<float16x2_t, float32_t>(static_cast<const float16x2_t*>(t_data) + t_offset, static_cast<float16x2_t*>(out_data) + out_offset, rows, cols_store, cols_eff); break;
                case DType::BFloat16x2: run_softmax<bfloat16x2_t, float32_t>(static_cast<const bfloat16x2_t*>(t_data) + t_offset, static_cast<bfloat16x2_t*>(out_data) + out_offset, rows, cols_store, cols_eff); break;
                case DType::Int8: run_softmax<int8_t, float64_t>(static_cast<const int8_t*>(t_data) + t_offset, static_cast<int8_t*>(out_data) + out_offset, rows, cols_store, cols_eff); break;
                case DType::UInt8: run_softmax<uint8_t, float64_t>(static_cast<const uint8_t*>(t_data) + t_offset, static_cast<uint8_t*>(out_data) + out_offset, rows, cols_store, cols_eff); break;
                case DType::Int16: run_softmax<int16_t, float64_t>(static_cast<const int16_t*>(t_data) + t_offset, static_cast<int16_t*>(out_data) + out_offset, rows, cols_store, cols_eff); break;
                case DType::UInt16: run_softmax<uint16_t, float64_t>(static_cast<const uint16_t*>(t_data) + t_offset, static_cast<uint16_t*>(out_data) + out_offset, rows, cols_store, cols_eff); break;
                case DType::Int32: run_softmax<int32_t, float64_t>(static_cast<const int32_t*>(t_data) + t_offset, static_cast<int32_t*>(out_data) + out_offset, rows, cols_store, cols_eff); break;
                case DType::UInt32: run_softmax<uint32_t, float64_t>(static_cast<const uint32_t*>(t_data) + t_offset, static_cast<uint32_t*>(out_data) + out_offset, rows, cols_store, cols_eff); break;
                case DType::Int64: run_softmax<int64_t, float64_t>(static_cast<const int64_t*>(t_data) + t_offset, static_cast<int64_t*>(out_data) + out_offset, rows, cols_store, cols_eff); break;
                case DType::UInt64: run_softmax<uint64_t, float64_t>(static_cast<const uint64_t*>(t_data) + t_offset, static_cast<uint64_t*>(out_data) + out_offset, rows, cols_store, cols_eff); break;
                case DType::Int4x2: run_softmax<int4x2_t, float64_t>(static_cast<const int4x2_t*>(t_data) + t_offset, static_cast<int4x2_t*>(out_data) + out_offset, rows, cols_store, cols_eff); break;
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
