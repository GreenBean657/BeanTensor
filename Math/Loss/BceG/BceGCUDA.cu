/**
 * @file BceGCUDA.cu
 * @author GreenBean
 * @brief CUDA backend for the bce_g op (bce backward, probability form).
 * Divergent twin of Math/Loss/BceG/BceGHIP.cpp.
 * @version 0.1
 * @date 2026-11-10
 */
#if defined(USE_CUDA)
#include <cstdint>
#include <limits>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "Math/Loss/BceG/BceG.h"
#include "Math/Loss/LossCudaShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    using namespace loss_gpu;

    namespace {
        template<typename T, typename Acc>
        __device__ __forceinline__ Acc bce_g_dloss(const T* g) {
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
        __device__ __forceinline__ Acc bce_g_clamp_p(const Acc p, const Acc eps) {
            if (p < eps) {
                return eps;
            }
            if (p > Acc{1} - eps) {
                return Acc{1} - eps;
            }
            return p;
        }

        template<typename Acc>
        __device__ __forceinline__ Acc bce_g_lane(const Acc p, const Acc y) {
            const auto eps = [] {
                if constexpr (std::is_same_v<Acc, float64_t>) {
                    return float64_t{1e-15};
                } else {
                    return Acc{1e-7f};
                }
            }();
            const auto pc = bce_g_clamp_p(p, eps);
            return ((Acc{1} - y) / (Acc{1} - pc)) - (y / pc);
        }

        template<typename T, typename Acc>
        __global__ void bce_g_kernel(const T* __restrict__ p, const T* __restrict__ t, const T* __restrict__ dloss,
                                     T* __restrict__ out, const size_t n_logical, const size_t numel) {
            const auto scale = (bce_g_dloss<T, Acc>(dloss)) / static_cast<Acc>(n_logical);
            const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
            const auto stride = blockDim.x * gridDim.x;
            for (auto i = tid; i < numel; i += stride) {
                if constexpr (std::is_same_v<T, float16x2_t>) {
                    const auto pv = p[i];
                    const auto tv = t[i];
                    const auto g_lo = scale * bce_g_lane(__half2float(__low2half(pv)), __half2float(__low2half(tv)));
                    const auto g_hi = scale * bce_g_lane(__half2float(__high2half(pv)), __half2float(__high2half(tv)));
                    out[i] = __halves2half2(__float2half_rn(g_lo), __float2half_rn(g_hi));
                } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                    const auto pv = p[i];
                    const auto tv = t[i];
                    const auto g_lo = scale * bce_g_lane(__bfloat162float(__low2bfloat16(pv)), __bfloat162float(__low2bfloat16(tv)));
                    const auto g_hi = scale * bce_g_lane(__bfloat162float(__high2bfloat16(pv)), __bfloat162float(__high2bfloat16(tv)));
                    out[i] = __halves2bfloat162(dev_narrow_bf16(g_lo), dev_narrow_bf16(g_hi));
                } else if constexpr (std::is_same_v<T, float16_t>) {
                    out[i] = __float2half_rn(scale * bce_g_lane(__half2float(p[i]), __half2float(t[i])));
                } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                    out[i] = dev_narrow_bf16(scale * bce_g_lane(__bfloat162float(p[i]), __bfloat162float(t[i])));
                } else if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                    out[i] = scale * bce_g_lane(static_cast<Acc>(p[i]), static_cast<Acc>(t[i]));
                }
            }
        }

        template<typename T, typename Acc>
        __host__ void run_bce_g(const T* p, const T* t, const T* dloss, T* out,
                                const size_t n_logical, const size_t numel) {
            if (numel > 0) {
                bce_g_kernel<T, Acc><<<grid_for(numel), BT_GPU_THREADS>>>(p, t, dloss, out, n_logical, numel);
            }
        }
    }

    __host__ void bce_g_gpu(const Tensor& dloss_grad, const Tensor& pred, const Tensor& targets, const Tensor& dpred) {
        if (dloss_grad.device != Device::GPU || pred.device != Device::GPU || targets.device != Device::GPU || dpred.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(pred.device));
        }
        const auto* const g_data = dloss_grad.data;
        const auto* const p_data = pred.data;
        const auto* const t_data = targets.data;
        auto* const out_data = dpred.data;
        const auto g_offset = dloss_grad.offset;
        const auto p_offset = pred.offset;
        const auto t_offset = targets.offset;
        const auto out_offset = dpred.offset;
        const auto dtype = pred.dtype;
        const auto numel = pred.numel;
        const auto packed_kind = dtype == DataTypes::DType::Float16x2 || dtype == DataTypes::DType::BFloat16x2;
        const auto lane_factor = packed_kind ? size_t{2} : size_t{1};
        const auto n_logical = numel * lane_factor;
        const auto worker = [=] {
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: run_bce_g<float32_t, float32_t>(static_cast<const float32_t*>(p_data) + p_offset, static_cast<const float32_t*>(t_data) + t_offset, static_cast<const float32_t*>(g_data) + g_offset, static_cast<float32_t*>(out_data) + out_offset, n_logical, numel); break;
                case DType::Float64: run_bce_g<float64_t, float64_t>(static_cast<const float64_t*>(p_data) + p_offset, static_cast<const float64_t*>(t_data) + t_offset, static_cast<const float64_t*>(g_data) + g_offset, static_cast<float64_t*>(out_data) + out_offset, n_logical, numel); break;
                case DType::Float16: run_bce_g<float16_t, float32_t>(static_cast<const float16_t*>(p_data) + p_offset, static_cast<const float16_t*>(t_data) + t_offset, static_cast<const float16_t*>(g_data) + g_offset, static_cast<float16_t*>(out_data) + out_offset, n_logical, numel); break;
                case DType::BFloat16: run_bce_g<bfloat16_t, float32_t>(static_cast<const bfloat16_t*>(p_data) + p_offset, static_cast<const bfloat16_t*>(t_data) + t_offset, static_cast<const bfloat16_t*>(g_data) + g_offset, static_cast<bfloat16_t*>(out_data) + out_offset, n_logical, numel); break;
                case DType::Float16x2: run_bce_g<float16x2_t, float32_t>(static_cast<const float16x2_t*>(p_data) + p_offset, static_cast<const float16x2_t*>(t_data) + t_offset, static_cast<const float16x2_t*>(g_data) + g_offset, static_cast<float16x2_t*>(out_data) + out_offset, n_logical, numel); break;
                case DType::BFloat16x2: run_bce_g<bfloat16x2_t, float32_t>(static_cast<const bfloat16x2_t*>(p_data) + p_offset, static_cast<const bfloat16x2_t*>(t_data) + t_offset, static_cast<const bfloat16x2_t*>(g_data) + g_offset, static_cast<bfloat16x2_t*>(out_data) + out_offset, n_logical, numel); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
            CUDA_CHECK_ERROR(cudaStreamSynchronize(nullptr));
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        dloss_grad.give_future(future);
        pred.give_future(future);
        targets.give_future(future);
        dpred.give_future(future);
    }
}
#endif
