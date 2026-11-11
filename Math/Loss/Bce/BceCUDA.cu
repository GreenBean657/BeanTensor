/**
 * @file BceCUDA.cu
 * @author GreenBean
 * @brief CUDA backend for the bce op (binary cross-entropy, probability form).
 * Divergent twin of Math/Loss/Bce/BceHIP.cpp — the two backends are
 * written separately per project rule, but must stay behaviorally identical:
 * same fp32/fp64 compute widths, same RNE narrowing, same clamp epsilon
 * contract.
 * @version 0.1
 * @date 2026-11-10
 */
#if defined(USE_CUDA)
#include <cstdint>
#include <limits>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "Math/Loss/Bce/Bce.h"
#include "Math/Loss/LossCudaShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    using namespace loss_gpu;
    namespace {
        // --------------------------- bce kernels ---------------------------

        /**
         * j is the LOGICAL element index: for packed dtypes it indexes lanes
         * within base[j >> 1]; for plain dtypes it indexes base[j] directly.
         * @brief Widen logical element j of an operand to the accumulator.
         * @param base Operand base pointer (storage elements).
         * @param j Logical element index (2 per storage element for packed dtypes).
         * @return Element j widened to Acc.
         */
        template<typename T, typename Acc>
        __device__ __forceinline__ Acc bce_widen(const T* base, const size_t j) {
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

        /**
         * @brief Clamp a probability to [eps, 1 - eps] in device code.
         * @param p Raw probability value.
         * @param eps Lower/upper epsilon.
         * @return Clamped probability.
         */
        template<typename Acc>
        __device__ __forceinline__ Acc bce_clamp_p(const Acc p, const Acc eps) {
            if (p < eps) {
                return eps;
            }
            if (p > Acc{1} - eps) {
                return Acc{1} - eps;
            }
            return p;
        }

        /**
         * @brief Device BCE contribution for one lane.
         * @param p_clamped Clamped probability.
         * @param y Target probability.
         * @return BCE loss contribution.
         */
        template<typename Acc>
        __device__ __forceinline__ Acc bce_lane_loss(const Acc p_clamped, const Acc y) {
            return -(y * log(p_clamped)) - ((Acc{1} - y) * log(Acc{1} - p_clamped));
        }

        /**
         * Grid-stride over the logical elements: each thread accumulates BCE
         * in Acc, then a shared-memory tree reduces the block into partials.
         * @brief bce per-block partial kernel (all dtypes).
         * @param p Predictions, storage elements.
         * @param t Targets, storage elements.
         * @param partials Per-block fp64 partial output, gridDim.x elements.
         * @param numel Storage element count.
         * @param lane_factor Logical elements per storage element (2 for packed, else 1).
         * @param eps Clamping epsilon in Acc.
         */
        template<typename T, typename Acc>
        __global__ void bce_partial_kernel(const T* __restrict__ p, const T* __restrict__ t,
                                           float64_t* __restrict__ partials,
                                           const size_t numel, const size_t lane_factor, const Acc eps) {
            const auto n_logical = numel * lane_factor;
            const auto tid = threadIdx.x;
            const auto stride = blockDim.x * gridDim.x;
            auto acc = Acc{0};
            for (auto j = tid + blockIdx.x * blockDim.x; j < n_logical; j += stride) {
                const auto pc = bce_clamp_p(bce_widen<T, Acc>(p, j), eps);
                const auto y = bce_widen<T, Acc>(t, j);
                acc += bce_lane_loss(pc, y);
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

        /**
         * Single block reduces the per-block fp64 partials to the mean and
         * stores it narrowed into out[0].
         * @brief bce finalize kernel.
         * @param partials Per-block fp64 partials, nblocks elements.
         * @param nblocks Number of valid partials.
         * @param n_logical Logical element count the mean divides by.
         * @param out Single-element output tensor data.
         */
        template<typename T>
        __global__ void bce_final_kernel(const float64_t* __restrict__ partials, const size_t nblocks,
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

        /**
         * @brief RAII guard for the BCE partials scratch buffer.
         */
        struct BcePartialsGuard {
            float64_t* ptr = nullptr;
            ~BcePartialsGuard() {
                if (ptr != nullptr) {
                    static_cast<void>(cudaFree(ptr));
                }
            }
        };

        /**
         * Host-side two-stage bce.
         * @param p Predictions, already offset to the first element.
         * @param t Targets, already offset to the first element.
         * @param numel Storage element count.
         * @param lane_factor Logical elements per storage element.
         * @param out Single-element output tensor data.
         * @param eps Clamping epsilon.
         * @throws ErrorHandling::IndexOutOfRange when numel is zero.
         */
        template<typename T, typename Acc>
        __host__ void run_bce(const T* p, const T* t, const size_t numel, const size_t lane_factor, T* out, const Acc eps) {
            if (numel == 0) {
                throw ErrorHandling::IndexOutOfRange();
            }
            const auto blocks = grid_for(numel);
            const auto n_logical = static_cast<float64_t>(numel) * static_cast<float64_t>(lane_factor);
            float64_t* partials = nullptr;
            CUDA_CHECK_ERROR(cudaMalloc(&partials, blocks * sizeof(float64_t)));
            BcePartialsGuard guard{partials};
            bce_partial_kernel<T, Acc><<<blocks, BT_GPU_THREADS>>>(p, t, partials, numel, lane_factor, eps);
            bce_final_kernel<T><<<1, BT_GPU_THREADS>>>(partials, blocks, n_logical, out);
            CUDA_CHECK_ERROR(cudaGetLastError());
            guard.ptr = nullptr;
            CUDA_CHECK_ERROR(cudaFree(partials));
        }
    }

    /**
     * Queues one pool worker that dispatches on dtype and runs the two-stage
     * reduction. The future lands on all three tensors so teardown drains it.
     * @brief GPU binary cross-entropy (probability form).
     * @param t Predictions tensor, GPU resident, contiguous.
     * @param target Targets tensor, same shape/dtype/device.
     * @param out Single-element output tensor in the operand dtype.
     * @note Throws DirectFunctionCallOnWrongDataSource when any tensor is not on the GPU.
     */
    __host__ void bce_gpu(const Tensor& t, const Tensor& target, const Tensor& out) {
        if (t.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(t.device));
        }
        if (target.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(target.device));
        }
        if (out.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(out.device));
        }
        const auto* const p_data = t.data;
        const auto* const t_data = target.data;
        auto* const out_data = out.data;
        const auto p_offset = t.offset;
        const auto t_offset = target.offset;
        const auto dtype = t.dtype;
        const auto numel = t.numel;
        const auto packed_kind = dtype == DataTypes::DType::Float16x2 || dtype == DataTypes::DType::BFloat16x2;
        const auto lane_factor = packed_kind ? size_t{2} : size_t{1};
        const auto worker = [=] {
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: run_bce<float32_t, float32_t>(static_cast<const float32_t*>(p_data) + p_offset, static_cast<const float32_t*>(t_data) + t_offset, numel, lane_factor, static_cast<float32_t*>(out_data), float32_t{1e-7f}); break;
                case DType::Float64: run_bce<float64_t, float64_t>(static_cast<const float64_t*>(p_data) + p_offset, static_cast<const float64_t*>(t_data) + t_offset, numel, lane_factor, static_cast<float64_t*>(out_data), float64_t{1e-15}); break;
                case DType::Float16: run_bce<float16_t, float32_t>(static_cast<const float16_t*>(p_data) + p_offset, static_cast<const float16_t*>(t_data) + t_offset, numel, lane_factor, static_cast<float16_t*>(out_data), float32_t{1e-7f}); break;
                case DType::BFloat16: run_bce<bfloat16_t, float32_t>(static_cast<const bfloat16_t*>(p_data) + p_offset, static_cast<const bfloat16_t*>(t_data) + t_offset, numel, lane_factor, static_cast<bfloat16_t*>(out_data), float32_t{1e-7f}); break;
                case DType::Float16x2: run_bce<float16x2_t, float32_t>(static_cast<const float16x2_t*>(p_data) + p_offset, static_cast<const float16x2_t*>(t_data) + t_offset, numel, lane_factor, static_cast<float16x2_t*>(out_data), float32_t{1e-7f}); break;
                case DType::BFloat16x2: run_bce<bfloat16x2_t, float32_t>(static_cast<const bfloat16x2_t*>(p_data) + p_offset, static_cast<const bfloat16x2_t*>(t_data) + t_offset, numel, lane_factor, static_cast<bfloat16x2_t*>(out_data), float32_t{1e-7f}); break;
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
