/**
 * @file BceWithLogitsGHIP.cpp
 * @author GreenBean
 * @brief HIP/ROCm backend for the bce_with_logits_g op (stable sigmoid backward).
 * Divergent twin of Math/Loss/BceWithLogitsG/BceWithLogitsGCUDA.cu.
 * @version 0.1
 * @date 2026-11-10
 */
#if defined(USE_HIP)
#include <cstdint>
#include <limits>
#include <type_traits>

#include <hip/hip_runtime.h>

#include "Math/Loss/BceWithLogitsG/BceWithLogitsG.h"
#include "Math/Loss/LossHipShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    using namespace loss_gpu;

    namespace {
        template<typename T, typename Acc>
        __device__ __forceinline__ Acc bce_logits_g_dloss(const T* g) {
            if constexpr (std::is_same_v<T, float16x2_t>) {
                return __half2float(__low2half(g[0]));
            } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                return __bfloat162float(__low2bfloat16(g[0]));
            } else if constexpr (std::is_same_v<T, float16_t>) {
                return __half2float(g[0]);
            } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                return __bfloat162float(g[0]);
            } else if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                return g[0];
            } else {
                return Acc{0};
            }
        }

        template<typename Acc>
        __device__ __forceinline__ Acc stable_sigmoid(const Acc x) {
            if (x >= Acc{0}) {
                const auto z = exp(-x);
                return Acc{1} / (Acc{1} + z);
            }
            const auto z = exp(x);
            return z / (Acc{1} + z);
        }

        template<typename Acc>
        __device__ __forceinline__ Acc bce_logits_g_lane(const Acc x, const Acc y) {
            return stable_sigmoid(x) - y;
        }

        template<typename T, typename Acc>
        __global__ void bce_logits_g_kernel(const T* __restrict__ x, const T* __restrict__ t, const T* __restrict__ dloss,
                                            T* __restrict__ out, const size_t n_logical, const size_t numel) {
            const auto scale = (bce_logits_g_dloss<T, Acc>(dloss)) / static_cast<Acc>(n_logical);
            const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
            const auto stride = blockDim.x * gridDim.x;
            for (auto i = tid; i < numel; i += stride) {
                if constexpr (std::is_same_v<T, float16x2_t>) {
                    const auto xv = x[i];
                    const auto tv = t[i];
                    const auto g_lo = scale * bce_logits_g_lane(__half2float(__low2half(xv)), __half2float(__low2half(tv)));
                    const auto g_hi = scale * bce_logits_g_lane(__half2float(__high2half(xv)), __half2float(__high2half(tv)));
                    out[i] = __halves2half2(__float2half_rn(g_lo), __float2half_rn(g_hi));
                } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                    const auto xv = x[i];
                    const auto tv = t[i];
                    const auto g_lo = scale * bce_logits_g_lane(__bfloat162float(__low2bfloat16(xv)), __bfloat162float(__low2bfloat16(tv)));
                    const auto g_hi = scale * bce_logits_g_lane(__bfloat162float(__high2bfloat16(xv)), __bfloat162float(__high2bfloat16(tv)));
                    out[i] = __halves2bfloat162(dev_narrow_bf16(g_lo), dev_narrow_bf16(g_hi));
                } else if constexpr (std::is_same_v<T, float16_t>) {
                    out[i] = __float2half_rn(scale * bce_logits_g_lane(__half2float(x[i]), __half2float(t[i])));
                } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                    out[i] = dev_narrow_bf16(scale * bce_logits_g_lane(__bfloat162float(x[i]), __bfloat162float(t[i])));
                } else if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                    out[i] = scale * bce_logits_g_lane(static_cast<Acc>(x[i]), static_cast<Acc>(t[i]));
                }
            }
        }

        template<typename T, typename Acc>
        __host__ void run_bce_logits_g(const T* x, const T* t, const T* dloss, T* out,
                                       const size_t n_logical, const size_t numel) {
            if (numel > 0) {
                bce_logits_g_kernel<T, Acc><<<grid_for(numel), BT_GPU_THREADS>>>(x, t, dloss, out, n_logical, numel);
            }
        }
    }

    __host__ void bce_with_logits_g_gpu(const Tensor& dloss_grad, const Tensor& logits, const Tensor& targets, const Tensor& dlogits) {
        if (dloss_grad.device != Device::GPU || logits.device != Device::GPU || targets.device != Device::GPU || dlogits.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(logits.device));
        }
        const auto* const g_data = dloss_grad.data;
        const auto* const x_data = logits.data;
        const auto* const t_data = targets.data;
        auto* const out_data = dlogits.data;
        const auto g_offset = dloss_grad.offset;
        const auto x_offset = logits.offset;
        const auto t_offset = targets.offset;
        const auto out_offset = dlogits.offset;
        const auto dtype = logits.dtype;
        const auto numel = logits.numel;
        const auto packed_kind = dtype == DataTypes::DType::Float16x2 || dtype == DataTypes::DType::BFloat16x2;
        const auto lane_factor = packed_kind ? size_t{2} : size_t{1};
        const auto n_logical = numel * lane_factor;
        const auto worker = [=] {
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: run_bce_logits_g<float32_t, float32_t>(static_cast<const float32_t*>(x_data) + x_offset, static_cast<const float32_t*>(t_data) + t_offset, static_cast<const float32_t*>(g_data) + g_offset, static_cast<float32_t*>(out_data) + out_offset, n_logical, numel); break;
                case DType::Float64: run_bce_logits_g<float64_t, float64_t>(static_cast<const float64_t*>(x_data) + x_offset, static_cast<const float64_t*>(t_data) + t_offset, static_cast<const float64_t*>(g_data) + g_offset, static_cast<float64_t*>(out_data) + out_offset, n_logical, numel); break;
                case DType::Float16: run_bce_logits_g<float16_t, float32_t>(static_cast<const float16_t*>(x_data) + x_offset, static_cast<const float16_t*>(t_data) + t_offset, static_cast<const float16_t*>(g_data) + g_offset, static_cast<float16_t*>(out_data) + out_offset, n_logical, numel); break;
                case DType::BFloat16: run_bce_logits_g<bfloat16_t, float32_t>(static_cast<const bfloat16_t*>(x_data) + x_offset, static_cast<const bfloat16_t*>(t_data) + t_offset, static_cast<const bfloat16_t*>(g_data) + g_offset, static_cast<bfloat16_t*>(out_data) + out_offset, n_logical, numel); break;
                case DType::Float16x2: run_bce_logits_g<float16x2_t, float32_t>(static_cast<const float16x2_t*>(x_data) + x_offset, static_cast<const float16x2_t*>(t_data) + t_offset, static_cast<const float16x2_t*>(g_data) + g_offset, static_cast<float16x2_t*>(out_data) + out_offset, n_logical, numel); break;
                case DType::BFloat16x2: run_bce_logits_g<bfloat16x2_t, float32_t>(static_cast<const bfloat16x2_t*>(x_data) + x_offset, static_cast<const bfloat16x2_t*>(t_data) + t_offset, static_cast<const bfloat16x2_t*>(g_data) + g_offset, static_cast<bfloat16x2_t*>(out_data) + out_offset, n_logical, numel); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
            HIP_CHECK_ERROR(hipStreamSynchronize(nullptr));
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        dloss_grad.give_future(future);
        logits.give_future(future);
        targets.give_future(future);
        dlogits.give_future(future);
    }
}
#endif
