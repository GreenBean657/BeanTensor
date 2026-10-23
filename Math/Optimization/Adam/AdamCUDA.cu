/**
 * @file AdamCUDA.cu
 * @author GreenBean
 * @brief CUDA backend for the adam_update op.
 * Divergent twin of Math/Optimization/Adam/AdamHIP.cpp — the two backends are
 * written separately per project rule (no HIP-only unification), but must stay
 * behaviorally identical: same fp32/fp64 compute widths, same RNE narrowing.
 * @version 0.1
 * @date 2026-10-23
 */
#if defined(USE_CUDA)
#include <cstdint>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "Math/Optimization/Adam/Adam.h"
#include "Math/Optimization/OptimizationCudaShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    using namespace optimization_gpu;

    namespace {
        /**
         * m = fma(b1, m, (1-b1)*g); v = fma(b2, v, (1-b2)*g*g);
         * w = fma(-lr, (m/bc1) / (sqrt(v/bc2) + eps), w). Float kinds compute
         * in their natural accumulator (fp32 for fp16/bf16/fp32 and per-lane
         * for packed x2, fp64 for fp64) and narrow RNE on store, exactly
         * matching the CPU kernels' std::fma/std::sqrt widths.
         * @brief Elementwise Adam kernel (float kinds).
         * @param w Weights, updated in place.
         * @param m First moments, updated in place.
         * @param v Second moments, updated in place.
         * @param g Gradients, read only.
         * @param lr Learning rate.
         * @param beta1 Beta 1.
         * @param beta2 Beta 2.
         * @param beta1_corr 1 - beta1^t (bias correction).
         * @param beta2_corr 1 - beta2^t (bias correction).
         * @param epsilon Epsilon.
         * @param numel Element count.
         */
        template<typename T>
        __global__ void adam_kernel(T* __restrict__ w, T* __restrict__ m, T* __restrict__ v, const T* __restrict__ g,
                                    const float64_t lr, const float64_t beta1, const float64_t beta2,
                                    const float64_t beta1_corr, const float64_t beta2_corr, const float64_t epsilon,
                                    const size_t numel) {
            const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t stride = blockDim.x * gridDim.x;
            const auto neg_lr32 = static_cast<float32_t>(-lr);
            const auto b1_32 = static_cast<float32_t>(beta1);
            const auto b2_32 = static_cast<float32_t>(beta2);
            const auto bc1_32 = static_cast<float32_t>(beta1_corr);
            const auto bc2_32 = static_cast<float32_t>(beta2_corr);
            const auto eps32 = static_cast<float32_t>(epsilon);
            const auto omb1_32 = 1.0f - b1_32;
            const auto omb2_32 = 1.0f - b2_32;
            const auto omb1_64 = 1.0 - beta1;
            const auto omb2_64 = 1.0 - beta2;
            for (size_t i = tid; i < numel; i += stride) {
                if constexpr (std::is_same_v<T, float32_t>) {
                    const auto mm = __fmaf_rn(b1_32, m[i], omb1_32 * g[i]);
                    const auto vv = __fmaf_rn(b2_32, v[i], omb2_32 * g[i] * g[i]);
                    m[i] = mm;
                    v[i] = vv;
                    const auto step = (mm / bc1_32) / (sqrtf(vv / bc2_32) + eps32);
                    w[i] = __fmaf_rn(neg_lr32, step, w[i]);
                } else if constexpr (std::is_same_v<T, float64_t>) {
                    const auto mm = __fma_rn(beta1, m[i], omb1_64 * g[i]);
                    const auto vv = __fma_rn(beta2, v[i], omb2_64 * g[i] * g[i]);
                    m[i] = mm;
                    v[i] = vv;
                    const auto step = (mm / beta1_corr) / (sqrt(vv / beta2_corr) + epsilon);
                    w[i] = __fma_rn(-lr, step, w[i]);
                } else if constexpr (std::is_same_v<T, float16_t>) {
                    const auto gm = __half2float(g[i]);
                    const auto mm = __fmaf_rn(b1_32, __half2float(m[i]), omb1_32 * gm);
                    const auto vv = __fmaf_rn(b2_32, __half2float(v[i]), omb2_32 * gm * gm);
                    m[i] = __float2half_rn(mm);
                    v[i] = __float2half_rn(vv);
                    const auto step = (mm / bc1_32) / (sqrtf(vv / bc2_32) + eps32);
                    w[i] = __float2half_rn(__fmaf_rn(neg_lr32, step, __half2float(w[i])));
                } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                    const auto gm = __bfloat162float(g[i]);
                    const auto mm = __fmaf_rn(b1_32, __bfloat162float(m[i]), omb1_32 * gm);
                    const auto vv = __fmaf_rn(b2_32, __bfloat162float(v[i]), omb2_32 * gm * gm);
                    m[i] = dev_narrow_bf16(mm);
                    v[i] = dev_narrow_bf16(vv);
                    const auto step = (mm / bc1_32) / (sqrtf(vv / bc2_32) + eps32);
                    w[i] = dev_narrow_bf16(__fmaf_rn(neg_lr32, step, __bfloat162float(w[i])));
                } else if constexpr (std::is_same_v<T, float16x2_t>) {
                    const auto gv = g[i];
                    const auto mv = m[i];
                    const auto vv0 = v[i];
                    const auto wv = w[i];
                    const auto gm_lo = __half2float(__low2half(gv));
                    const auto gm_hi = __half2float(__high2half(gv));
                    const auto mm_lo = __fmaf_rn(b1_32, __half2float(__low2half(mv)), omb1_32 * gm_lo);
                    const auto mm_hi = __fmaf_rn(b1_32, __half2float(__high2half(mv)), omb1_32 * gm_hi);
                    const auto vv_lo = __fmaf_rn(b2_32, __half2float(__low2half(vv0)), omb2_32 * gm_lo * gm_lo);
                    const auto vv_hi = __fmaf_rn(b2_32, __half2float(__high2half(vv0)), omb2_32 * gm_hi * gm_hi);
                    m[i] = __halves2half2(__float2half_rn(mm_lo), __float2half_rn(mm_hi));
                    v[i] = __halves2half2(__float2half_rn(vv_lo), __float2half_rn(vv_hi));
                    const auto step_lo = (mm_lo / bc1_32) / (sqrtf(vv_lo / bc2_32) + eps32);
                    const auto step_hi = (mm_hi / bc1_32) / (sqrtf(vv_hi / bc2_32) + eps32);
                    const auto new_lo = __float2half_rn(__fmaf_rn(neg_lr32, step_lo, __half2float(__low2half(wv))));
                    const auto new_hi = __float2half_rn(__fmaf_rn(neg_lr32, step_hi, __half2float(__high2half(wv))));
                    w[i] = __halves2half2(new_lo, new_hi);
                } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                    const auto gv = g[i];
                    const auto mv = m[i];
                    const auto vv0 = v[i];
                    const auto wv = w[i];
                    const auto gm_lo = __bfloat162float(__low2bfloat16(gv));
                    const auto gm_hi = __bfloat162float(__high2bfloat16(gv));
                    const auto mm_lo = __fmaf_rn(b1_32, __bfloat162float(__low2bfloat16(mv)), omb1_32 * gm_lo);
                    const auto mm_hi = __fmaf_rn(b1_32, __bfloat162float(__high2bfloat16(mv)), omb1_32 * gm_hi);
                    const auto vv_lo = __fmaf_rn(b2_32, __bfloat162float(__low2bfloat16(vv0)), omb2_32 * gm_lo * gm_lo);
                    const auto vv_hi = __fmaf_rn(b2_32, __bfloat162float(__high2bfloat16(vv0)), omb2_32 * gm_hi * gm_hi);
                    m[i] = __halves2bfloat162(dev_narrow_bf16(mm_lo), dev_narrow_bf16(mm_hi));
                    v[i] = __halves2bfloat162(dev_narrow_bf16(vv_lo), dev_narrow_bf16(vv_hi));
                    const auto step_lo = (mm_lo / bc1_32) / (sqrtf(vv_lo / bc2_32) + eps32);
                    const auto step_hi = (mm_hi / bc1_32) / (sqrtf(vv_hi / bc2_32) + eps32);
                    const auto new_lo = dev_narrow_bf16(__fmaf_rn(neg_lr32, step_lo, __bfloat162float(__low2bfloat16(wv))));
                    const auto new_hi = dev_narrow_bf16(__fmaf_rn(neg_lr32, step_hi, __bfloat162float(__high2bfloat16(wv))));
                    w[i] = __halves2bfloat162(new_lo, new_hi);
                } else [[unlikely]] {
                    // Integer kinds are rejected by Tensor::adam_update before dispatch.
                }
            }
        }

        /**
         * @brief Host-side Adam launch.
         * @param w Weights, updated in place.
         * @param m First moments, updated in place.
         * @param v Second moments, updated in place.
         * @param g Gradients, read only.
         * @param lr Learning rate.
         * @param beta1 Beta 1.
         * @param beta2 Beta 2.
         * @param beta1_corr 1 - beta1^t (bias correction).
         * @param beta2_corr 1 - beta2^t (bias correction).
         * @param epsilon Epsilon.
         * @param numel Element count.
         */
        template<typename T>
        __host__ void run_adam(T* w, T* m, T* v, const T* g,
                               const float64_t lr, const float64_t beta1, const float64_t beta2,
                               const float64_t beta1_corr, const float64_t beta2_corr, const float64_t epsilon,
                               const size_t numel) {
            if (numel > 0) {
                adam_kernel<T><<<grid_for(numel), BT_GPU_THREADS>>>(w, m, v, g, lr, beta1, beta2, beta1_corr, beta2_corr, epsilon, numel);
            }
        }
    }

    /**
     * Runs on the CPU thread pool at high priority; the future lands on all
     * four tensors so teardown drains the worker.
     * @brief GPU in-place Adam moment step.
     * @param w Weights, updated in place.
     * @param grad Gradients, same shape/dtype/device as w.
     * @param m First moments, same shape/dtype/device as w (updated in place).
     * @param v Second moments, same shape/dtype/device as w (updated in place).
     * @param lr Learning rate.
     * @param beta1 Beta 1.
     * @param beta2 Beta 2.
     * @param beta1_corr 1 - beta1^t (bias correction).
     * @param beta2_corr 1 - beta2^t (bias correction).
     * @param epsilon Epsilon.
     */
    __host__ void adam_update_gpu(const Tensor& w, const Tensor& grad, const Tensor& m, const Tensor& v,
                                  const float64_t lr, const float64_t beta1, const float64_t beta2,
                                  const float64_t beta1_corr, const float64_t beta2_corr, const float64_t epsilon) {
        if (w.device != Device::GPU || grad.device != Device::GPU || m.device != Device::GPU || v.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(w.device));
        }
        auto* const w_data = w.data;
        auto* const m_data = m.data;
        auto* const v_data = v.data;
        const auto* const g_data = grad.data;
        const auto w_offset = w.offset;
        const auto m_offset = m.offset;
        const auto v_offset = v.offset;
        const auto g_offset = grad.offset;
        const auto dtype = w.dtype;
        const auto numel = w.numel;
        // Raw pointers captured by value: the future lands on the tensors, so
        // buffer teardown always drains this worker first.
        /**
         * Raw pointers are captured by value: the future lands on the tensors, so buffer teardown
         * always drains this worker first. Dispatches on dtype to the typed host launcher.
         * @brief Pool worker for the GPU Adam step.
         * @note Lambda subcall. Self-execution prohibited.
         */
        const auto worker = [=] {
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: run_adam(static_cast<float32_t*>(w_data) + w_offset, static_cast<float32_t*>(m_data) + m_offset, static_cast<float32_t*>(v_data) + v_offset, static_cast<const float32_t*>(g_data) + g_offset, lr, beta1, beta2, beta1_corr, beta2_corr, epsilon, numel); break;
                case DType::Float64: run_adam(static_cast<float64_t*>(w_data) + w_offset, static_cast<float64_t*>(m_data) + m_offset, static_cast<float64_t*>(v_data) + v_offset, static_cast<const float64_t*>(g_data) + g_offset, lr, beta1, beta2, beta1_corr, beta2_corr, epsilon, numel); break;
                case DType::Float16: run_adam(static_cast<float16_t*>(w_data) + w_offset, static_cast<float16_t*>(m_data) + m_offset, static_cast<float16_t*>(v_data) + v_offset, static_cast<const float16_t*>(g_data) + g_offset, lr, beta1, beta2, beta1_corr, beta2_corr, epsilon, numel); break;
                case DType::BFloat16: run_adam(static_cast<bfloat16_t*>(w_data) + w_offset, static_cast<bfloat16_t*>(m_data) + m_offset, static_cast<bfloat16_t*>(v_data) + v_offset, static_cast<const bfloat16_t*>(g_data) + g_offset, lr, beta1, beta2, beta1_corr, beta2_corr, epsilon, numel); break;
                case DType::Float16x2: run_adam(static_cast<float16x2_t*>(w_data) + w_offset, static_cast<float16x2_t*>(m_data) + m_offset, static_cast<float16x2_t*>(v_data) + v_offset, static_cast<const float16x2_t*>(g_data) + g_offset, lr, beta1, beta2, beta1_corr, beta2_corr, epsilon, numel); break;
                case DType::BFloat16x2: run_adam(static_cast<bfloat16x2_t*>(w_data) + w_offset, static_cast<bfloat16x2_t*>(m_data) + m_offset, static_cast<bfloat16x2_t*>(v_data) + v_offset, static_cast<const bfloat16x2_t*>(g_data) + g_offset, lr, beta1, beta2, beta1_corr, beta2_corr, epsilon, numel); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
            // The kernel is in-flight when the launch returns; sync so the
            // future only completes once the buffer writes are done.
            CUDA_CHECK_ERROR(cudaStreamSynchronize(nullptr));
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        w.give_future(future);
        grad.give_future(future);
        m.give_future(future);
        v.give_future(future);
    }
}
#endif
