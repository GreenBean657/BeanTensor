/**
 * @file SgdUpdateHIP.cpp
 * @author GreenBean
 * @brief HIP backend for the sgd_update op.
 * Divergent twin of Math/Optimization/SgdUpdate/SgdUpdateCUDA.cu — the two backends are
 * written separately per project rule (no HIP-only unification), but must stay
 * behaviorally identical: same fp32/fp64 compute widths, same RNE narrowing.
 * @version 0.1.1
 * @date 2026-10-20
 */
#if defined(USE_HIP)
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>

#include <hip/hip_runtime.h>

#include "Math/Optimization/SgdUpdate/SgdUpdate.h"
#include "Math/Optimization/OptimizationHipShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    using namespace optimization_gpu;

    namespace {
        /**
         * w = fma(-lr, g, w). Float kinds compute in their natural accumulator (fp32 for fp16/bf16/fp32, fp64 for fp64)
         * and narrow RNE on store, exactly matching the CPU kernels. Integers compute in fp64, clamp to the dtype range, and
         * narrow round-toward-zero; a NaN update leaves the element unchanged (no integer NaN to store).
         * @note lo/hi carry the integer dtype bounds and are ignored for float kinds.
         * @brief Elementwise SGD kernel (all dtypes).
         * @param w Weights, updated in place.
         * @param g Gradients, read only.
         * @param lr Learning rate (not negated).
         * @param lo Lowest representable integer value.
         * @param hi Highest representable integer value.
         * @param numel Element count.
         */
        template<typename T>
        __global__ void sgd_kernel(T* __restrict__ w, const T* __restrict__ g,
                                   const float64_t lr, const float64_t lo, const float64_t hi, const size_t numel) {
            const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t stride = blockDim.x * gridDim.x;
            const auto neg_lr32 = static_cast<float32_t>(-lr);
            for (size_t i = tid; i < numel; i += stride) {
                if constexpr (std::is_same_v<T, float32_t>) {
                    w[i] = __fmaf_rn(neg_lr32, g[i], w[i]);
                } else if constexpr (std::is_same_v<T, float64_t>) {
                    w[i] = fma(-lr, g[i], w[i]);
                } else if constexpr (std::is_same_v<T, float16_t>) {
                    w[i] = __float2half_rn(__fmaf_rn(neg_lr32, __half2float(g[i]), __half2float(w[i])));
                } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                    w[i] = dev_narrow_bf16(__fmaf_rn(neg_lr32, __bfloat162float(g[i]), __bfloat162float(w[i])));
                } else if constexpr (std::is_same_v<T, float16x2_t>) {
                    const auto gv = g[i];
                    const auto wv = w[i];
                    const auto r_lo = __float2half_rn(__fmaf_rn(neg_lr32, __half2float(__low2half(gv)), __half2float(__low2half(wv))));
                    const auto r_hi = __float2half_rn(__fmaf_rn(neg_lr32, __half2float(__high2half(gv)), __half2float(__high2half(wv))));
                    w[i] = __halves2half2(r_lo, r_hi);
                } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                    const auto gv = g[i];
                    const auto wv = w[i];
                    const auto r_lo = dev_narrow_bf16(__fmaf_rn(neg_lr32, __bfloat162float(__low2bfloat16(gv)), __bfloat162float(__low2bfloat16(wv))));
                    const auto r_hi = dev_narrow_bf16(__fmaf_rn(neg_lr32, __bfloat162float(__high2bfloat16(gv)), __bfloat162float(__high2bfloat16(wv))));
                    w[i] = __halves2bfloat162(r_lo, r_hi);
                } else if constexpr (std::is_same_v<T, int4x2_t>) {
                    // Per-lane fp64 compute, clamp to [-8, 7] (lo/hi args),
                    // truncate round-toward-zero; a NaN update keeps the lane.
                    /**
                     * Per-lane fp64 compute, clamp to [lo, hi], truncate round-toward-zero; a NaN update keeps the lane.
                     * @brief Update one int4 lane.
                     * @param w_lane Current weight lane.
                     * @param g_lane Gradient lane.
                     * @return Updated lane; w_lane itself when the update is NaN.
                     * @note Lambda subcall. Self-execution prohibited.
                     */
                    const auto lane_step = [&] __device__ (const int8_t w_lane, const int8_t g_lane) {
                        // Do not strip __device__ flag, the lambda calls fma which causes the IDE to warn it's invalid calling methodology. It *RUNS*, but you are relying on the compiler to not break it.
                        const auto upd = fma(-lr, g_lane, w_lane);
                        if (isnan(upd)) {
                            return w_lane;
                        }
                        if (upd <= lo) {
                            return static_cast<int8_t>(lo);
                        }
                        if (upd >= hi) {
                            return static_cast<int8_t>(hi);
                        }
                        return static_cast<int8_t>(upd);
                    };
                    w[i] = dev_i4_pack(lane_step(dev_i4_lo(w[i]), dev_i4_lo(g[i])),
                                       lane_step(dev_i4_hi(w[i]), dev_i4_hi(g[i])));
                } else {
                    const auto upd = fma(-lr, static_cast<float64_t>(g[i]), static_cast<float64_t>(w[i]));
                    if (isnan(upd)) {
                        continue;
                    }
                    if (upd <= lo) {
                        w[i] = static_cast<T>(lo);
                    } else if (upd >= hi) {
                        w[i] = static_cast<T>(hi);
                    } else {
                        w[i] = static_cast<T>(upd);
                    }
                }
            }
        }

        /**
         * Integer bounds are computed on the host, device code cannot use numeric_limits.
         * @brief Host-side SGD launch.
         * @param w Weights, updated in place.
         * @param g Gradients, read only.
         * @param lr Learning rate (not negated).
         * @param numel Element count.
         */
        template<typename T>
        __host__ void run_sgd(T* w, const T* g, const float64_t lr, const size_t numel) {
            auto lo = 0.0;
            auto hi = 0.0;
            if constexpr (std::is_same_v<T, int4x2_t>) {
                lo = -8.0;
                hi = 7.0;
            } else if constexpr (std::is_integral_v<T>) {
                lo = static_cast<float64_t>(std::numeric_limits<T>::lowest());
                hi = static_cast<float64_t>(std::numeric_limits<T>::max());
            }
            if (numel > 0) {
                sgd_kernel<T><<<grid_for(numel), BT_GPU_THREADS>>>(w, g, lr, lo, hi, numel);
            }
        }
    }

    /**
     * Runs on the CPU thread pool at high priority; the future lands on both tensors so teardown drains the worker.
     * @brief GPU in-place SGD step, w -= lr * grad.
     * @param w Weights, updated in place.
     * @param grad Gradients, same shape and dtype as w.
     * @param lr Learning rate.
     */
    __host__ void sgd_update_gpu(const Tensor& w, const Tensor& grad, const float64_t lr) {
        if (w.device != Device::GPU || grad.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(w.device));
        }
        auto* const w_data = w.data;
        const auto* const g_data = grad.data;
        const auto w_offset = w.offset;
        const auto g_offset = grad.offset;
        const auto dtype = w.dtype;
        const auto numel = w.numel;
        // Raw pointers captured by value: the future lands on both tensors, so
        // buffer teardown always drains this worker first.
        /**
         * Raw pointers are captured by value: the future lands on the tensors, so buffer teardown
         * always drains this worker first. Dispatches on dtype to the typed host launcher.
         * @brief Pool worker for the GPU SGD step.
         * @note Lambda subcall. Self-execution prohibited.
         */
        const auto worker = [=] {
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: run_sgd(static_cast<float32_t*>(w_data) + w_offset, static_cast<const float32_t*>(g_data) + g_offset, lr, numel); break;
                case DType::Float64: run_sgd(static_cast<float64_t*>(w_data) + w_offset, static_cast<const float64_t*>(g_data) + g_offset, lr, numel); break;
                case DType::Float16: run_sgd(static_cast<float16_t*>(w_data) + w_offset, static_cast<const float16_t*>(g_data) + g_offset, lr, numel); break;
                case DType::BFloat16: run_sgd(static_cast<bfloat16_t*>(w_data) + w_offset, static_cast<const bfloat16_t*>(g_data) + g_offset, lr, numel); break;
                case DType::Float16x2: run_sgd(static_cast<float16x2_t*>(w_data) + w_offset, static_cast<const float16x2_t*>(g_data) + g_offset, lr, numel); break;
                case DType::BFloat16x2: run_sgd(static_cast<bfloat16x2_t*>(w_data) + w_offset, static_cast<const bfloat16x2_t*>(g_data) + g_offset, lr, numel); break;
                case DType::Int8: run_sgd(static_cast<int8_t*>(w_data) + w_offset, static_cast<const int8_t*>(g_data) + g_offset, lr, numel); break;
                case DType::UInt8: run_sgd(static_cast<uint8_t*>(w_data) + w_offset, static_cast<const uint8_t*>(g_data) + g_offset, lr, numel); break;
                case DType::Int16: run_sgd(static_cast<int16_t*>(w_data) + w_offset, static_cast<const int16_t*>(g_data) + g_offset, lr, numel); break;
                case DType::UInt16: run_sgd(static_cast<uint16_t*>(w_data) + w_offset, static_cast<const uint16_t*>(g_data) + g_offset, lr, numel); break;
                case DType::Int32: run_sgd(static_cast<int32_t*>(w_data) + w_offset, static_cast<const int32_t*>(g_data) + g_offset, lr, numel); break;
                case DType::UInt32: run_sgd(static_cast<uint32_t*>(w_data) + w_offset, static_cast<const uint32_t*>(g_data) + g_offset, lr, numel); break;
                case DType::Int64: run_sgd(static_cast<int64_t*>(w_data) + w_offset, static_cast<const int64_t*>(g_data) + g_offset, lr, numel); break;
                case DType::UInt64: run_sgd(static_cast<uint64_t*>(w_data) + w_offset, static_cast<const uint64_t*>(g_data) + g_offset, lr, numel); break;
                case DType::Int4x2: run_sgd(static_cast<int4x2_t*>(w_data) + w_offset, static_cast<const int4x2_t*>(g_data) + g_offset, lr, numel); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
            // The kernel is in-flight when the launch returns; sync so the
            // future only completes once the buffer writes are done.
            HIP_CHECK_ERROR(hipStreamSynchronize(nullptr));
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        w.give_future(future);
        grad.give_future(future);
    }
}
#endif
