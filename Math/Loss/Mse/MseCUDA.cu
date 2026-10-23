/**
 * @file MseCUDA.cu
 * @author GreenBean
 * @brief CUDA backend for the mse op (mean squared error loss).
 * Divergent twin of Math/Loss/Mse/MseHIP.cpp — the two backends are
 * written separately per project rule (no HIP-only unification), but must stay
 * behaviorally identical: same fp32/fp64 compute widths, same RNE narrowing,
 * same accumulator taxonomy as the CPU backend.
 * @version 0.1
 * @date 2026-10-23
 */
#if defined(USE_CUDA)
#include <cstdint>
#include <limits>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "Math/Loss/LossCudaShared.h"
#include "Math/Loss/Mse/Mse.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    using namespace loss_gpu;
    namespace {
        // --------------------------- mse kernels ---------------------------

        /**
         * fp16/bf16 and the lanes of packed x2 widen to fp32; fp64 and the
         * integer dtypes (int4x2 unpacks nibbles) widen to fp64. j is the
         * LOGICAL element index: for packed dtypes and int4x2 it indexes
         * lanes/nibbles within base[j >> 1]; for plain dtypes it indexes
         * base[j] directly.
         * @brief Widen logical element j of an operand to the accumulator.
         * @param base Operand base pointer (storage elements).
         * @param j Logical element index (2 per storage element for packed dtypes).
         * @return Element j widened to Acc.
         */
        template<typename T, typename Acc>
        __device__ __forceinline__ Acc mse_widen(const T* base, const size_t j) {
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
            } else if constexpr (std::is_same_v<T, int4x2_t>) {
                return static_cast<float64_t>((j & 1) != 0 ? dev_i4_hi(base[j >> 1]) : dev_i4_lo(base[j >> 1]));
            } else {
                return static_cast<float64_t>(base[j]);
            }
        }

        /**
         * Grid-stride over the logical elements: each thread accumulates
         * (p_j - t_j)^2 in Acc, then a shared-memory tree reduces the block
         * into partials[blockIdx.x] (fp64 upcast, exact from fp32). Mirrors
         * the CPU's per-chunk fp32/fp64 partials.
         * @brief mse per-block partial kernel (all dtypes).
         * @param p Predictions, storage elements.
         * @param t Targets, storage elements.
         * @param partials Per-block fp64 partial output, gridDim.x elements.
         * @param numel Storage element count.
         * @param lane_factor Logical elements per storage element (2 for packed/int4x2, else 1).
         */
        template<typename T, typename Acc>
        __global__ void mse_partial_kernel(const T* __restrict__ p, const T* __restrict__ t,
                                           float64_t* __restrict__ partials,
                                           const size_t numel, const size_t lane_factor) {
            const auto n_logical = numel * lane_factor;
            const auto tid = threadIdx.x;
            const auto stride = blockDim.x * gridDim.x;
            auto acc = Acc{0};
            for (auto j = tid + blockIdx.x * blockDim.x; j < n_logical; j += stride) {
                const auto d = mse_widen<T, Acc>(p, j) - mse_widen<T, Acc>(t, j);
                acc += d * d;
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
         * Single block reduces the per-block fp64 partials to the mean
         * (total / n_logical) and stores it narrowed into out[0]: floats RNE
         * (bf16 with the CPU-matching flush), packed dtypes duplicate the
         * scalar mean into both lanes (the loss is one scalar), integers
         * clamp to the dtype range and truncate (a NaN mean stores 0),
         * int4x2 clamps to [-8, 7]. lo/hi carry the integer dtype bounds and
         * are ignored for float kinds — the CPU store_loss_mean contract.
         * @brief mse finalize kernel.
         * @param partials Per-block fp64 partials, nblocks elements.
         * @param nblocks Number of valid partials.
         * @param n_logical Logical element count the mean divides by.
         * @param lo Integer dtype lower bound; ignored for float kinds.
         * @param hi Integer dtype upper bound; ignored for float kinds.
         * @param out Single-element output tensor data.
         */
        template<typename T>
        __global__ void mse_final_kernel(const float64_t* __restrict__ partials, const size_t nblocks,
                                         const float64_t n_logical, const float64_t lo, const float64_t hi,
                                         T* __restrict__ out) {
            __shared__ float64_t shared[BT_GPU_THREADS];
            const auto tid = threadIdx.x;
            auto acc = 0.0;
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
                } else if constexpr (std::is_same_v<T, int4x2_t>) {
                    auto v = mean;
                    if (isnan(v)) {
                        v = 0.0;
                    }
                    if (v <= -8.0) {
                        v = -8.0;
                    } else if (v >= 7.0) {
                        v = 7.0;
                    }
                    const auto n = static_cast<int32_t>(v);
                    out[0] = dev_i4_pack(static_cast<int8_t>(n), static_cast<int8_t>(n));
                } else {
                    auto v = mean;
                    if (isnan(v)) {
                        v = 0.0;
                    }
                    if (v <= lo) {
                        v = lo;
                    } else if (v >= hi) {
                        v = hi;
                    }
                    out[0] = static_cast<T>(v);
                }
            }
        }

        /**
         * Scratch is worker-local cudaMalloc/cudaFree. The partials buffer is
         * freed after the finalize kernel; cudaFree syncs the default stream,
         * so the result is complete in out before the worker returns.
         * Integer bounds are computed on the host (nvcc rejects numeric_limits
         * in device code).
         * @brief Host-side two-stage mse.
         * @param p Predictions, already offset to the first element.
         * @param t Targets, already offset to the first element.
         * @param numel Storage element count.
         * @param lane_factor Logical elements per storage element.
         * @param out Single-element output tensor data.
         */
        template<typename T, typename Acc>
        __host__ void run_mse(const T* p, const T* t, const size_t numel, const size_t lane_factor, T* out) {
            // Integer bounds are computed on the host (nvcc rejects
            // numeric_limits in device code).
            auto lo = 0.0;
            auto hi = 0.0;
            if constexpr (std::is_integral_v<T>) {
                lo = static_cast<float64_t>(std::numeric_limits<T>::lowest());
                hi = static_cast<float64_t>(std::numeric_limits<T>::max());
            }
            const auto blocks = grid_for(numel);
            const auto n_logical = static_cast<float64_t>(numel) * static_cast<float64_t>(lane_factor);
            float64_t* partials = nullptr;
            CUDA_CHECK_ERROR(cudaMalloc(&partials, blocks * sizeof(float64_t)));
            mse_partial_kernel<T, Acc><<<blocks, BT_GPU_THREADS>>>(p, t, partials, numel, lane_factor);
            mse_final_kernel<T><<<1, BT_GPU_THREADS>>>(partials, blocks, n_logical, lo, hi, out);
            CUDA_CHECK_ERROR(cudaGetLastError());
            CUDA_CHECK_ERROR(cudaFree(partials));
        }
    }

    /**
     * Queues one pool worker that dispatches on dtype and runs the two-stage
     * reduction (per-block partials, then a single-block finalize). The
     * future lands on all three tensors so teardown drains it.
     * @brief GPU mean squared error: mean of (pred - target)^2 over all elements.
     * @param t Predictions tensor, GPU resident, contiguous.
     * @param target Targets tensor, same shape/dtype/device.
     * @param out Single-element output tensor in the operand dtype.
     * @note Throws DirectFunctionCallOnWrongDataSource when any tensor is not on the GPU.
     */
    __host__ void mse_gpu(const Tensor& t, const Tensor& target, const Tensor& out) {
        if (t.device != Device::GPU || target.device != Device::GPU || out.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(t.device));
        }
        const auto* const p_data = t.data;
        const auto* const t_data = target.data;
        auto* const out_data = out.data;
        const auto p_offset = t.offset;
        const auto t_offset = target.offset;
        const auto dtype = t.dtype;
        const auto numel = t.numel;
        const auto packed_kind = dtype == DataTypes::DType::Float16x2 || dtype == DataTypes::DType::BFloat16x2;
        const auto lane_factor = (packed_kind || dtype == DataTypes::DType::Int4x2) ? size_t{2} : size_t{1};
        /**
         * Dispatches on dtype to run_mse. The runner's cudaFree already synced
         * the stream, so no extra sync is needed.
         * @brief Pool worker for the mse launch.
         * @note Lambda subcall. Self-execution prohibited.
         */
        const auto worker = [=] {
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: run_mse<float32_t, float32_t>(static_cast<const float32_t*>(p_data) + p_offset, static_cast<const float32_t*>(t_data) + t_offset, numel, lane_factor, static_cast<float32_t*>(out_data)); break;
                case DType::Float64: run_mse<float64_t, float64_t>(static_cast<const float64_t*>(p_data) + p_offset, static_cast<const float64_t*>(t_data) + t_offset, numel, lane_factor, static_cast<float64_t*>(out_data)); break;
                case DType::Float16: run_mse<float16_t, float32_t>(static_cast<const float16_t*>(p_data) + p_offset, static_cast<const float16_t*>(t_data) + t_offset, numel, lane_factor, static_cast<float16_t*>(out_data)); break;
                case DType::BFloat16: run_mse<bfloat16_t, float32_t>(static_cast<const bfloat16_t*>(p_data) + p_offset, static_cast<const bfloat16_t*>(t_data) + t_offset, numel, lane_factor, static_cast<bfloat16_t*>(out_data)); break;
                case DType::Float16x2: run_mse<float16x2_t, float32_t>(static_cast<const float16x2_t*>(p_data) + p_offset, static_cast<const float16x2_t*>(t_data) + t_offset, numel, lane_factor, static_cast<float16x2_t*>(out_data)); break;
                case DType::BFloat16x2: run_mse<bfloat16x2_t, float32_t>(static_cast<const bfloat16x2_t*>(p_data) + p_offset, static_cast<const bfloat16x2_t*>(t_data) + t_offset, numel, lane_factor, static_cast<bfloat16x2_t*>(out_data)); break;
                case DType::Int8: run_mse<int8_t, float64_t>(static_cast<const int8_t*>(p_data) + p_offset, static_cast<const int8_t*>(t_data) + t_offset, numel, lane_factor, static_cast<int8_t*>(out_data)); break;
                case DType::UInt8: run_mse<uint8_t, float64_t>(static_cast<const uint8_t*>(p_data) + p_offset, static_cast<const uint8_t*>(t_data) + t_offset, numel, lane_factor, static_cast<uint8_t*>(out_data)); break;
                case DType::Int16: run_mse<int16_t, float64_t>(static_cast<const int16_t*>(p_data) + p_offset, static_cast<const int16_t*>(t_data) + t_offset, numel, lane_factor, static_cast<int16_t*>(out_data)); break;
                case DType::UInt16: run_mse<uint16_t, float64_t>(static_cast<const uint16_t*>(p_data) + p_offset, static_cast<const uint16_t*>(t_data) + t_offset, numel, lane_factor, static_cast<uint16_t*>(out_data)); break;
                case DType::Int32: run_mse<int32_t, float64_t>(static_cast<const int32_t*>(p_data) + p_offset, static_cast<const int32_t*>(t_data) + t_offset, numel, lane_factor, static_cast<int32_t*>(out_data)); break;
                case DType::UInt32: run_mse<uint32_t, float64_t>(static_cast<const uint32_t*>(p_data) + p_offset, static_cast<const uint32_t*>(t_data) + t_offset, numel, lane_factor, static_cast<uint32_t*>(out_data)); break;
                case DType::Int64: run_mse<int64_t, float64_t>(static_cast<const int64_t*>(p_data) + p_offset, static_cast<const int64_t*>(t_data) + t_offset, numel, lane_factor, static_cast<int64_t*>(out_data)); break;
                case DType::UInt64: run_mse<uint64_t, float64_t>(static_cast<const uint64_t*>(p_data) + p_offset, static_cast<const uint64_t*>(t_data) + t_offset, numel, lane_factor, static_cast<uint64_t*>(out_data)); break;
                case DType::Int4x2: run_mse<int4x2_t, float64_t>(static_cast<const int4x2_t*>(p_data) + p_offset, static_cast<const int4x2_t*>(t_data) + t_offset, numel, lane_factor, static_cast<int4x2_t*>(out_data)); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
            // run_mse's cudaFree already synced the stream.
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        t.give_future(future);
        target.give_future(future);
        out.give_future(future);
    }
}
#endif
