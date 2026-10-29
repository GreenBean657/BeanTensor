/**
 * @file SoftmaxGHIP.cpp
 * @author GreenBean
 * @brief HIP/ROCm backend for standalone softmax backward.
 * Divergent twin of Math/Autograd/SoftmaxG/SoftmaxGCUDA.cu — the two backends
 * are written separately per project rule (no HIP-only unification), but must
 * stay behaviorally identical.
 * @version 0.1
 * @date 2026-10-28
 */
#if defined(USE_HIP)
#include <cstdint>
#include <limits>
#include <type_traits>

#include <hip/hip_runtime.h>

#include "Math/Activations/Softmax/SoftmaxHipShared.h"
#include "Math/Autograd/SoftmaxG/SoftmaxG.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    using namespace softmax_gpu_helpers;

    namespace {
        /**
         * Per-row softmax backward kernel: recompute y from x, reduce dy*y over
         * the row, then write dx = y * (dy - rowsum). One block per row.
         * @brief HIP per-row softmax backward kernel.
         */
        template<typename T, typename Acc>
        __global__ void softmax_g_rows_kernel(const T* __restrict__ x, const T* __restrict__ dy, T* __restrict__ dx,
                                              const size_t rows, const size_t cols_store, const size_t cols_eff) {
            const auto r = blockIdx.x;
            if (r >= rows) {
                return;
            }
            const auto* x_row = x + r * cols_store;
            const auto* dy_row = dy + r * cols_store;
            auto* dx_row = dx + r * cols_store;
            __shared__ Acc shared[BT_GPU_THREADS];
            __shared__ Acc row_max;
            __shared__ Acc row_sum;
            const auto tid = threadIdx.x;
            auto m = acc_lowest<Acc>();
            for (auto j = tid; j < cols_eff; j += BT_GPU_THREADS) {
                if (
                    const auto xv = sm_widen<T, Acc>(x_row, j);
                    xv > m
                ) {
                    m = xv;
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
                sum += acc_exp(sm_widen<T, Acc>(x_row, j) - m);
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
            auto dy_y = Acc{0};
            for (auto j = tid; j < cols_eff; j += BT_GPU_THREADS) {
                const auto yj = acc_exp(sm_widen<T, Acc>(x_row, j) - m) / total;
                dy_y += sm_widen<T, Acc>(dy_row, j) * yj;
            }
            shared[tid] = dy_y;
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
            const auto rsum = row_sum;
            for (auto e = tid; e < cols_store; e += BT_GPU_THREADS) {
                if constexpr (std::is_same_v<T, float16x2_t>) {
                    dx_row[e] = __halves2half2(
                        __float2half_rn((acc_exp(sm_widen<T, Acc>(x_row, 2 * e) - m) / total) * (sm_widen<T, Acc>(dy_row, 2 * e) - rsum)),
                        __float2half_rn((acc_exp(sm_widen<T, Acc>(x_row, 2 * e + 1) - m) / total) * (sm_widen<T, Acc>(dy_row, 2 * e + 1) - rsum)));
                } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                    dx_row[e] = __halves2bfloat162(
                        dev_narrow_bf16((acc_exp(sm_widen<T, Acc>(x_row, 2 * e) - m) / total) * (sm_widen<T, Acc>(dy_row, 2 * e) - rsum)),
                        dev_narrow_bf16((acc_exp(sm_widen<T, Acc>(x_row, 2 * e + 1) - m) / total) * (sm_widen<T, Acc>(dy_row, 2 * e + 1) - rsum)));
                } else if constexpr (std::is_same_v<T, int4x2_t>) {
                    const auto lo = static_cast<int32_t>((acc_exp(sm_widen<T, Acc>(x_row, 2 * e) - m) / total) * (sm_widen<T, Acc>(dy_row, 2 * e) - rsum));
                    const auto hi = static_cast<int32_t>((acc_exp(sm_widen<T, Acc>(x_row, 2 * e + 1) - m) / total) * (sm_widen<T, Acc>(dy_row, 2 * e + 1) - rsum));
                    dx_row[e] = dev_pack_int4(
                        lo <= -8 ? -8 : lo >= 7 ? 7 : lo,
                        hi <= -8 ? -8 : hi >= 7 ? 7 : hi);
                } else {
                    dx_row[e] = sm_narrow<T, Acc>((acc_exp(sm_widen<T, Acc>(x_row, e) - m) / total) * (sm_widen<T, Acc>(dy_row, e) - rsum));
                }
            }
        }

        template<typename T, typename Acc>
        __host__ void run_softmax_g(const T* x, const T* dy, T* dx, const size_t rows, const size_t cols_store, const size_t cols_eff) {
            if (rows > 0) {
                softmax_g_rows_kernel<T, Acc><<<rows, BT_GPU_THREADS>>>(x, dy, dx, rows, cols_store, cols_eff);
            }
        }
    }

    __host__ void softmax_g_gpu(const Tensor& dy, const Tensor& x, const Tensor& dx) {
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
        const auto rows = x.shape[0];
        const auto cols_store = x.shape[1];
        const auto packed = dtype == DataTypes::DType::Float16x2 || dtype == DataTypes::DType::BFloat16x2 || dtype == DataTypes::DType::Int4x2;
        const auto cols_eff = cols_store * (packed ? size_t{2} : size_t{1});
        const auto worker = [=] {
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: run_softmax_g<float32_t, float32_t>(static_cast<const float32_t*>(x_data) + x_offset, static_cast<const float32_t*>(dy_data) + dy_offset, static_cast<float32_t*>(dx_data) + dx_offset, rows, cols_store, cols_eff); break;
                case DType::Float64: run_softmax_g<float64_t, float64_t>(static_cast<const float64_t*>(x_data) + x_offset, static_cast<const float64_t*>(dy_data) + dy_offset, static_cast<float64_t*>(dx_data) + dx_offset, rows, cols_store, cols_eff); break;
                case DType::Float16: run_softmax_g<float16_t, float32_t>(static_cast<const float16_t*>(x_data) + x_offset, static_cast<const float16_t*>(dy_data) + dy_offset, static_cast<float16_t*>(dx_data) + dx_offset, rows, cols_store, cols_eff); break;
                case DType::BFloat16: run_softmax_g<bfloat16_t, float32_t>(static_cast<const bfloat16_t*>(x_data) + x_offset, static_cast<const bfloat16_t*>(dy_data) + dy_offset, static_cast<bfloat16_t*>(dx_data) + dx_offset, rows, cols_store, cols_eff); break;
                case DType::Float16x2: run_softmax_g<float16x2_t, float32_t>(static_cast<const float16x2_t*>(x_data) + x_offset, static_cast<const float16x2_t*>(dy_data) + dy_offset, static_cast<float16x2_t*>(dx_data) + dx_offset, rows, cols_store, cols_eff); break;
                case DType::BFloat16x2: run_softmax_g<bfloat16x2_t, float32_t>(static_cast<const bfloat16x2_t*>(x_data) + x_offset, static_cast<const bfloat16x2_t*>(dy_data) + dy_offset, static_cast<bfloat16x2_t*>(dx_data) + dx_offset, rows, cols_store, cols_eff); break;
                case DType::Int8: run_softmax_g<int8_t, float64_t>(static_cast<const int8_t*>(x_data) + x_offset, static_cast<const int8_t*>(dy_data) + dy_offset, static_cast<int8_t*>(dx_data) + dx_offset, rows, cols_store, cols_eff); break;
                case DType::UInt8: run_softmax_g<uint8_t, float64_t>(static_cast<const uint8_t*>(x_data) + x_offset, static_cast<const uint8_t*>(dy_data) + dy_offset, static_cast<uint8_t*>(dx_data) + dx_offset, rows, cols_store, cols_eff); break;
                case DType::Int16: run_softmax_g<int16_t, float64_t>(static_cast<const int16_t*>(x_data) + x_offset, static_cast<const int16_t*>(dy_data) + dy_offset, static_cast<int16_t*>(dx_data) + dx_offset, rows, cols_store, cols_eff); break;
                case DType::UInt16: run_softmax_g<uint16_t, float64_t>(static_cast<const uint16_t*>(x_data) + x_offset, static_cast<const uint16_t*>(dy_data) + dy_offset, static_cast<uint16_t*>(dx_data) + dx_offset, rows, cols_store, cols_eff); break;
                case DType::Int32: run_softmax_g<int32_t, float64_t>(static_cast<const int32_t*>(x_data) + x_offset, static_cast<const int32_t*>(dy_data) + dy_offset, static_cast<int32_t*>(dx_data) + dx_offset, rows, cols_store, cols_eff); break;
                case DType::UInt32: run_softmax_g<uint32_t, float64_t>(static_cast<const uint32_t*>(x_data) + x_offset, static_cast<const uint32_t*>(dy_data) + dy_offset, static_cast<uint32_t*>(dx_data) + dx_offset, rows, cols_store, cols_eff); break;
                case DType::Int64: run_softmax_g<int64_t, float64_t>(static_cast<const int64_t*>(x_data) + x_offset, static_cast<const int64_t*>(dy_data) + dy_offset, static_cast<int64_t*>(dx_data) + dx_offset, rows, cols_store, cols_eff); break;
                case DType::UInt64: run_softmax_g<uint64_t, float64_t>(static_cast<const uint64_t*>(x_data) + x_offset, static_cast<const uint64_t*>(dy_data) + dy_offset, static_cast<uint64_t*>(dx_data) + dx_offset, rows, cols_store, cols_eff); break;
                case DType::Int4x2: run_softmax_g<int4x2_t, float64_t>(static_cast<const int4x2_t*>(x_data) + x_offset, static_cast<const int4x2_t*>(dy_data) + dy_offset, static_cast<int4x2_t*>(dx_data) + dx_offset, rows, cols_store, cols_eff); break;
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
