/**
 * @file BceWithLogitsHIP.cpp
 * @author GreenBean
 * @brief HIP/ROCm backend for the bce_with_logits op (stable logit-form BCE).
 * Divergent twin of Math/Loss/BceWithLogits/BceWithLogitsCUDA.cu — the two
 * backends are written separately per project rule, but must stay
 * behaviorally identical.
 * @version 0.1
 * @date 2026-11-10
 */
#if defined(USE_HIP)
#include <cstdint>
#include <limits>
#include <type_traits>

#include <hip/hip_runtime.h>

#include "Math/Loss/BceWithLogits/BceWithLogits.h"
#include "Math/Loss/LossHipShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    using namespace loss_gpu;
    namespace {
        template<typename T, typename Acc>
        __device__ __forceinline__ Acc bce_logits_widen(const T* base, const size_t j) {
            if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                return base[j];
            } else if constexpr (std::is_same_v<T, float16_t>) {
                return __half2float(base[j]);
            } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                return __bfloat162float(base[j]);
            } else if constexpr (std::is_same_v<T, float16x2_t>) {
                const auto v = base[j >> 1];
                return (j & 1) != 0 ? __half2float(__high2half(v)) : __half2float(__low2half(v));
            } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                const auto v = base[j >> 1];
                return (j & 1) != 0 ? __bfloat162float(__high2bfloat16(v)) : __bfloat162float(__low2bfloat16(v));
            } else {
                return Acc{0};
            }
        }

        template<typename Acc>
        __device__ __forceinline__ Acc bce_logits_lane_loss(const Acc x, const Acc y) {
            const auto abs_x = fabs(x);
            const auto max_term = x > Acc{0} ? x : Acc{0};
            return max_term - x * y + log1p(exp(-abs_x));
        }

        template<typename T, typename Acc>
        __global__ void bce_logits_partial_kernel(const T* __restrict__ x, const T* __restrict__ t,
                                                  float64_t* __restrict__ partials,
                                                  const size_t numel, const size_t lane_factor) {
            const auto n_logical = numel * lane_factor;
            const auto tid = threadIdx.x;
            const auto stride = blockDim.x * gridDim.x;
            auto acc = Acc{0};
            for (auto j = tid + blockIdx.x * blockDim.x; j < n_logical; j += stride) {
                const auto logit = bce_logits_widen<T, Acc>(x, j);
                const auto y = bce_logits_widen<T, Acc>(t, j);
                acc += bce_logits_lane_loss(logit, y);
            }
            __shared__ Acc shared[BT_GPU_THREADS];
            shared[tid] = acc;
            __syncthreads();
            for (auto s = BT_GPU_THREADS / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared[tid] = shared[tid] + shared[tid + s];
                }
                __syncthreads();
            }
            if (tid == 0) {
                partials[blockIdx.x] = static_cast<float64_t>(shared[0]);
            }
        }

        template<typename T>
        __global__ void bce_logits_final_kernel(const float64_t* __restrict__ partials, const size_t nblocks,
                                                const float64_t n_logical, T* __restrict__ out) {
            __shared__ float64_t shared[BT_GPU_THREADS];
            const auto tid = threadIdx.x;
            auto acc = float64_t{0.0};
            for (auto i = tid; i < nblocks; i += BT_GPU_THREADS) {
                acc += partials[i];
            }
            shared[tid] = acc;
            __syncthreads();
            for (auto s = BT_GPU_THREADS / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared[tid] = shared[tid] + shared[tid + s];
                }
                __syncthreads();
            }
            if (tid == 0) {
                const auto mean = shared[0] / n_logical;
                if constexpr (std::is_same_v<T, float32_t>) {
                    out[0] = static_cast<float32_t>(mean);
                } else if constexpr (std::is_same_v<T, float64_t>) {
                    out[0] = mean;
                } else if constexpr (std::is_same_v<T, float16_t>) {
                    out[0] = __float2half_rn(static_cast<float32_t>(mean));
                } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                    out[0] = dev_narrow_bf16(static_cast<float32_t>(mean));
                } else if constexpr (std::is_same_v<T, float16x2_t>) {
                    const auto lane = __float2half_rn(static_cast<float32_t>(mean));
                    out[0] = __halves2half2(lane, lane);
                } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                    const auto lane = dev_narrow_bf16(static_cast<float32_t>(mean));
                    out[0] = __halves2bfloat162(lane, lane);
                }
            }
        }

        struct BceLogitsPartialsGuard {
            float64_t* ptr = nullptr;
            ~BceLogitsPartialsGuard() {
                if (ptr != nullptr) {
                    static_cast<void>(hipFree(ptr));
                }
            }
        };

        template<typename T, typename Acc>
        __host__ void run_bce_logits(const T* x, const T* t, const size_t numel, const size_t lane_factor, T* out) {
            if (numel == 0) {
                throw ErrorHandling::IndexOutOfRange();
            }
            const auto blocks = grid_for(numel);
            const auto n_logical = static_cast<float64_t>(numel) * static_cast<float64_t>(lane_factor);
            float64_t* partials = nullptr;
            HIP_CHECK_ERROR(hipMalloc(&partials, blocks * sizeof(float64_t)));
            BceLogitsPartialsGuard guard{partials};
            bce_logits_partial_kernel<T, Acc><<<blocks, BT_GPU_THREADS>>>(x, t, partials, numel, lane_factor);
            bce_logits_final_kernel<T><<<1, BT_GPU_THREADS>>>(partials, blocks, n_logical, out);
            HIP_CHECK_ERROR(hipGetLastError());
            guard.ptr = nullptr;
            HIP_CHECK_ERROR(hipFree(partials));
        }
    }

    __host__ void bce_with_logits_gpu(const Tensor& t, const Tensor& target, const Tensor& out) {
        if (t.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(t.device));
        }
        if (target.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(target.device));
        }
        if (out.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(out.device));
        }
        const auto* const x_data = t.data;
        const auto* const t_data = target.data;
        auto* const out_data = out.data;
        const auto x_offset = t.offset;
        const auto t_offset = target.offset;
        const auto dtype = t.dtype;
        const auto numel = t.numel;
        const auto packed_kind = dtype == DataTypes::DType::Float16x2 || dtype == DataTypes::DType::BFloat16x2;
        const auto lane_factor = packed_kind ? size_t{2} : size_t{1};
        const auto worker = [=] {
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: run_bce_logits<float32_t, float32_t>(static_cast<const float32_t*>(x_data) + x_offset, static_cast<const float32_t*>(t_data) + t_offset, numel, lane_factor, static_cast<float32_t*>(out_data)); break;
                case DType::Float64: run_bce_logits<float64_t, float64_t>(static_cast<const float64_t*>(x_data) + x_offset, static_cast<const float64_t*>(t_data) + t_offset, numel, lane_factor, static_cast<float64_t*>(out_data)); break;
                case DType::Float16: run_bce_logits<float16_t, float32_t>(static_cast<const float16_t*>(x_data) + x_offset, static_cast<const float16_t*>(t_data) + t_offset, numel, lane_factor, static_cast<float16_t*>(out_data)); break;
                case DType::BFloat16: run_bce_logits<bfloat16_t, float32_t>(static_cast<const bfloat16_t*>(x_data) + x_offset, static_cast<const bfloat16_t*>(t_data) + t_offset, numel, lane_factor, static_cast<bfloat16_t*>(out_data)); break;
                case DType::Float16x2: run_bce_logits<float16x2_t, float32_t>(static_cast<const float16x2_t*>(x_data) + x_offset, static_cast<const float16x2_t*>(t_data) + t_offset, numel, lane_factor, static_cast<float16x2_t*>(out_data)); break;
                case DType::BFloat16x2: run_bce_logits<bfloat16x2_t, float32_t>(static_cast<const bfloat16x2_t*>(x_data) + x_offset, static_cast<const bfloat16x2_t*>(t_data) + t_offset, numel, lane_factor, static_cast<bfloat16x2_t*>(out_data)); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        t.give_future(future);
        target.give_future(future);
        out.give_future(future);
    }
}
#endif
