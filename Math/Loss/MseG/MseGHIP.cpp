/**
 * @file MseGHIP.cpp
 * @author GreenBean
 * @brief HIP/ROCm backend for the mse_g op (mse backward).
 * Divergent twin of Math/Loss/MseG/MseGCUDA.cu — the two backends are
 * written separately per project rule (no HIP-only unification), but must
 * stay behaviorally identical: same fp32/fp64 compute widths, same RNE
 * narrowing, same NaN handling.
 * @version 0.1
 * @date 2026-10-23
 */
#if defined(USE_HIP)
#include <cstdint>
#include <limits>
#include <type_traits>

#include <hip/hip_runtime.h>

#include "Math/Loss/LossHipShared.h"
#include "Math/Loss/MseG/MseG.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    using namespace loss_gpu;

    namespace {
        /**
         * Packed dtypes read the low lane (the forward store duplicates the
         * mean into both lanes); int4x2 reads the low nibble.
         * @brief Widen the scalar dloss gradient element to the accumulator.
         * @param g Pointer to the scalar upstream gradient (element 0 is read).
         * @return The upstream gradient widened to Acc.
         */
        template<typename T, typename Acc>
        __device__ __forceinline__ Acc mse_g_dloss(const T* g) {
            if constexpr (std::is_same_v<T, float16x2_t>) {
                return __half2float(__low2half(g[0]));
            } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                return __bfloat162float(__low2bfloat16(g[0]));
            } else if constexpr (std::is_same_v<T, int4x2_t>) {
                return static_cast<float64_t>(dev_i4_lo(g[0]));
            } else if constexpr (std::is_same_v<T, float16_t>) {
                return __half2float(g[0]);
            } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                return __bfloat162float(g[0]);
            } else if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                return g[0];
            } else {
                return static_cast<float64_t>(g[0]);
            }
        }

        /**
         * NaN becomes 0 (no integer NaN), then the value is clamped to [lo, hi].
         * The truncation toward zero happens in the caller's integer cast.
         * @brief Integer narrowing of one gradient value (NaN -> 0, clamp).
         * @param v Gradient value in fp64.
         * @param lo Lowest representable integer value.
         * @param hi Highest representable integer value.
         * @return Clamped value, ready for the integer cast.
         */
        __device__ __forceinline__ float64_t mse_g_clamp(const float64_t v, const float64_t lo, const float64_t hi) {
            auto x = v;
            if (isnan(x)) {
                x = 0.0;
            }
            if (x <= lo) {
                x = lo;
            } else if (x >= hi) {
                x = hi;
            }
            return x;
        }

        /**
         * Per-nibble gradient narrowed to the signed nibble range: NaN -> 0,
         * clamp to [-8, 7], truncate toward zero.
         * @brief Narrow one int4 mse_g lane.
         * @param v Gradient value in fp64.
         * @return The lane value in [-8, 7].
         */
        __device__ __forceinline__ int8_t mse_g_i4_lane(const float64_t v) {
            return static_cast<int8_t>(static_cast<int32_t>(mse_g_clamp(v, -8.0, 7.0)));
        }

        /**
         * Narrow one gradient value into a non-packed operand element: floats
         * pass (fp16 RNE, bf16 RNE with the CPU-matching flush), integers
         * clamp/truncate (NaN -> 0).
         * @brief Store one mse_g gradient element.
         * @param v Gradient value in the accumulator width.
         * @param lo Lowest integer value of T (ignored for float T).
         * @param hi Highest integer value of T (ignored for float T).
         * @return v converted to T.
         */
        template<typename T, typename Acc>
        __device__ __forceinline__ T mse_g_narrow(const Acc v, const float64_t lo, const float64_t hi) {
            if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                return v;
            } else if constexpr (std::is_same_v<T, float16_t>) {
                return __float2half_rn(static_cast<float32_t>(v));
            } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                return dev_narrow_bf16(static_cast<float32_t>(v));
            } else {
                return static_cast<T>(mse_g_clamp(static_cast<float64_t>(v), lo, hi));
            }
        }

        /**
         * dpred_i = 2 * (p_i - t_i) / n_logical * dloss, grid-stride over the
         * storage elements. Packed dtypes write both lanes per element;
         * int4x2 clamps each nibble to [-8, 7]. lo/hi carry the integer dtype
         * bounds and are ignored for float kinds. The scale is formed per
         * thread exactly like the CPU driver: (2 * dloss) / n_logical in Acc.
         * @brief mse backward elementwise kernel (all dtypes).
         * @param p Forward predictions.
         * @param t Forward targets.
         * @param dloss Scalar upstream gradient.
         * @param out Output gradient, same shape/dtype as p.
         * @param lo Lowest integer value of T (unused for float T).
         * @param hi Highest integer value of T (unused for float T).
         * @param n_logical Logical element count (2 * numel for packed/int4x2).
         * @param numel Storage element count.
         */
        template<typename T, typename Acc>
        __global__ void mse_g_kernel(const T* __restrict__ p, const T* __restrict__ t, const T* __restrict__ dloss,
                                     T* __restrict__ out, const float64_t lo, const float64_t hi,
                                     const size_t n_logical, const size_t numel) {
            const auto scale = (Acc{2} * mse_g_dloss<T, Acc>(dloss)) / static_cast<Acc>(n_logical);
            const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
            const auto stride = blockDim.x * gridDim.x;
            for (auto i = tid; i < numel; i += stride) {
                if constexpr (std::is_same_v<T, float16x2_t>) {
                    const auto pv = p[i];
                    const auto tv = t[i];
                    const auto g_lo = scale * (__half2float(__low2half(pv)) - __half2float(__low2half(tv)));
                    const auto g_hi = scale * (__half2float(__high2half(pv)) - __half2float(__high2half(tv)));
                    out[i] = __halves2half2(__float2half_rn(g_lo), __float2half_rn(g_hi));
                } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                    const auto pv = p[i];
                    const auto tv = t[i];
                    const auto g_lo = scale * (__bfloat162float(__low2bfloat16(pv)) - __bfloat162float(__low2bfloat16(tv)));
                    const auto g_hi = scale * (__bfloat162float(__high2bfloat16(pv)) - __bfloat162float(__high2bfloat16(tv)));
                    out[i] = __halves2bfloat162(dev_narrow_bf16(g_lo), dev_narrow_bf16(g_hi));
                } else if constexpr (std::is_same_v<T, int4x2_t>) {
                    const auto g_lo = scale * (static_cast<float64_t>(dev_i4_lo(p[i])) - static_cast<float64_t>(dev_i4_lo(t[i])));
                    const auto g_hi = scale * (static_cast<float64_t>(dev_i4_hi(p[i])) - static_cast<float64_t>(dev_i4_hi(t[i])));
                    out[i] = dev_i4_pack(mse_g_i4_lane(g_lo), mse_g_i4_lane(g_hi));
                } else if constexpr (std::is_same_v<T, float16_t>) {
                    out[i] = __float2half_rn(scale * (__half2float(p[i]) - __half2float(t[i])));
                } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                    out[i] = dev_narrow_bf16(scale * (__bfloat162float(p[i]) - __bfloat162float(t[i])));
                } else {
                    out[i] = mse_g_narrow<T, Acc>(scale * (static_cast<Acc>(p[i]) - static_cast<Acc>(t[i])), lo, hi);
                }
            }
        }

        /**
         * Integer bounds are computed on the host, device code cannot use numeric_limits.
         * @brief Host-side mse_g launch.
         * @param p Forward predictions, already offset.
         * @param t Forward targets, already offset.
         * @param dloss Scalar upstream gradient, already offset.
         * @param out Output gradient, already offset.
         * @param n_logical Logical element count (2 * numel for packed/int4x2).
         * @param numel Storage element count.
         */
        template<typename T, typename Acc>
        __host__ void run_mse_g(const T* p, const T* t, const T* dloss, T* out,
                                const size_t n_logical, const size_t numel) {
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
                mse_g_kernel<T, Acc><<<grid_for(numel), BT_GPU_THREADS>>>(p, t, dloss, out, lo, hi, n_logical, numel);
            }
        }
    }

    /**
     * Queues the elementwise kernel on the CPU thread pool (High priority)
     * and returns; the shared future is given to all four tensors, and errors
     * surface at sync(). The worker syncs the stream before completing so the
     * future only releases once dpred's writes have landed.
     * @brief GPU mse backward (all operand dtypes).
     * @param dloss_grad Scalar upstream gradient of the loss.
     * @param pred Forward predictions tensor.
     * @param targets Forward targets tensor.
     * @param dpred Output gradient tensor, same shape as pred.
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if any tensor is not on the GPU.
     * @throws ErrorHandling::NotImplemented (from the worker, at sync) for an unsupported pred dtype.
     */
    __host__ void mse_g_gpu(const Tensor& dloss_grad, const Tensor& pred, const Tensor& targets, const Tensor& dpred) {
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
        const auto lane_factor = (packed_kind || dtype == DataTypes::DType::Int4x2) ? size_t{2} : size_t{1};
        const auto n_logical = numel * lane_factor;
        /**
         * Raw pointers are captured by value: the future lands on the tensors,
         * so buffer teardown always drains this worker first. Dispatches on the
         * operand dtype and launches the grid-stride kernel.
         * @brief Pool worker for the GPU mse backward launch.
         * @note Lambda subcall. Self-execution prohibited.
         */
        const auto worker = [=] {
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: run_mse_g<float32_t, float32_t>(static_cast<const float32_t*>(p_data) + p_offset, static_cast<const float32_t*>(t_data) + t_offset, static_cast<const float32_t*>(g_data) + g_offset, static_cast<float32_t*>(out_data) + out_offset, n_logical, numel); break;
                case DType::Float64: run_mse_g<float64_t, float64_t>(static_cast<const float64_t*>(p_data) + p_offset, static_cast<const float64_t*>(t_data) + t_offset, static_cast<const float64_t*>(g_data) + g_offset, static_cast<float64_t*>(out_data) + out_offset, n_logical, numel); break;
                case DType::Float16: run_mse_g<float16_t, float32_t>(static_cast<const float16_t*>(p_data) + p_offset, static_cast<const float16_t*>(t_data) + t_offset, static_cast<const float16_t*>(g_data) + g_offset, static_cast<float16_t*>(out_data) + out_offset, n_logical, numel); break;
                case DType::BFloat16: run_mse_g<bfloat16_t, float32_t>(static_cast<const bfloat16_t*>(p_data) + p_offset, static_cast<const bfloat16_t*>(t_data) + t_offset, static_cast<const bfloat16_t*>(g_data) + g_offset, static_cast<bfloat16_t*>(out_data) + out_offset, n_logical, numel); break;
                case DType::Float16x2: run_mse_g<float16x2_t, float32_t>(static_cast<const float16x2_t*>(p_data) + p_offset, static_cast<const float16x2_t*>(t_data) + t_offset, static_cast<const float16x2_t*>(g_data) + g_offset, static_cast<float16x2_t*>(out_data) + out_offset, n_logical, numel); break;
                case DType::BFloat16x2: run_mse_g<bfloat16x2_t, float32_t>(static_cast<const bfloat16x2_t*>(p_data) + p_offset, static_cast<const bfloat16x2_t*>(t_data) + t_offset, static_cast<const bfloat16x2_t*>(g_data) + g_offset, static_cast<bfloat16x2_t*>(out_data) + out_offset, n_logical, numel); break;
                case DType::Int8: run_mse_g<int8_t, float64_t>(static_cast<const int8_t*>(p_data) + p_offset, static_cast<const int8_t*>(t_data) + t_offset, static_cast<const int8_t*>(g_data) + g_offset, static_cast<int8_t*>(out_data) + out_offset, n_logical, numel); break;
                case DType::UInt8: run_mse_g<uint8_t, float64_t>(static_cast<const uint8_t*>(p_data) + p_offset, static_cast<const uint8_t*>(t_data) + t_offset, static_cast<const uint8_t*>(g_data) + g_offset, static_cast<uint8_t*>(out_data) + out_offset, n_logical, numel); break;
                case DType::Int16: run_mse_g<int16_t, float64_t>(static_cast<const int16_t*>(p_data) + p_offset, static_cast<const int16_t*>(t_data) + t_offset, static_cast<const int16_t*>(g_data) + g_offset, static_cast<int16_t*>(out_data) + out_offset, n_logical, numel); break;
                case DType::UInt16: run_mse_g<uint16_t, float64_t>(static_cast<const uint16_t*>(p_data) + p_offset, static_cast<const uint16_t*>(t_data) + t_offset, static_cast<const uint16_t*>(g_data) + g_offset, static_cast<uint16_t*>(out_data) + out_offset, n_logical, numel); break;
                case DType::Int32: run_mse_g<int32_t, float64_t>(static_cast<const int32_t*>(p_data) + p_offset, static_cast<const int32_t*>(t_data) + t_offset, static_cast<const int32_t*>(g_data) + g_offset, static_cast<int32_t*>(out_data) + out_offset, n_logical, numel); break;
                case DType::UInt32: run_mse_g<uint32_t, float64_t>(static_cast<const uint32_t*>(p_data) + p_offset, static_cast<const uint32_t*>(t_data) + t_offset, static_cast<const uint32_t*>(g_data) + g_offset, static_cast<uint32_t*>(out_data) + out_offset, n_logical, numel); break;
                case DType::Int64: run_mse_g<int64_t, float64_t>(static_cast<const int64_t*>(p_data) + p_offset, static_cast<const int64_t*>(t_data) + t_offset, static_cast<const int64_t*>(g_data) + g_offset, static_cast<int64_t*>(out_data) + out_offset, n_logical, numel); break;
                case DType::UInt64: run_mse_g<uint64_t, float64_t>(static_cast<const uint64_t*>(p_data) + p_offset, static_cast<const uint64_t*>(t_data) + t_offset, static_cast<const uint64_t*>(g_data) + g_offset, static_cast<uint64_t*>(out_data) + out_offset, n_logical, numel); break;
                case DType::Int4x2: run_mse_g<int4x2_t, float64_t>(static_cast<const int4x2_t*>(p_data) + p_offset, static_cast<const int4x2_t*>(t_data) + t_offset, static_cast<const int4x2_t*>(g_data) + g_offset, static_cast<int4x2_t*>(out_data) + out_offset, n_logical, numel); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
            // The kernel is in-flight when the launch returns; sync so the
            // future only completes once the buffer writes are done.
            HIP_CHECK_ERROR(hipStreamSynchronize(nullptr));
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
