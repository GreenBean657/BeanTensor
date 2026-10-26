/**
 * @file MeanCUDA.cu
 * @author GreenBean
 * @brief CUDA backend for the mean op.
 * Divergent twin of Math/Reductions/Mean/MeanHIP.cpp — the two backends are
 * written separately per project rule (no HIP-only unification), but must stay
 * behaviorally identical: same fp32/fp64 compute widths, same RNE narrowing,
 * same fp64 integer accumulation with truncate-toward-zero + clamp narrowing.
 * @version 0.2
 * @date 2026-10-25
 */
#if defined(USE_CUDA)
#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "Math/Reductions/Mean/Mean.h"
#include "Math/Reductions/ReductionsCudaShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"
#include "Helpers/KernelTune.h"

namespace BeanTensor::Tensors::detail {
    using namespace reductions_gpu;

    namespace {
        /**
         * @brief Widen one element to its mean accumulator (fp32 passthrough).
         * @param v Element.
         * @return Accumulator value.
         */
        __device__ __forceinline__ float32_t mean_widen(const float32_t v) { return v; }
        /**
         * @brief Widen one element to its mean accumulator (fp64 passthrough).
         * @param v Element.
         * @return Accumulator value.
         */
        __device__ __forceinline__ float64_t mean_widen(const float64_t v) { return v; }
        /**
         * @brief Widen one element to its mean accumulator (fp16 -> fp32).
         * @param v Element.
         * @return Accumulator value.
         */
        __device__ __forceinline__ float32_t mean_widen(const float16_t v) { return __half2float(v); }
        /**
         * @brief Widen one element to its mean accumulator (bf16 -> fp32).
         * @param v Element.
         * @return Accumulator value.
         */
        __device__ __forceinline__ float32_t mean_widen(const bfloat16_t v) { return __bfloat162float(v); }
        /**
         * @brief Widen one packed element to its mean accumulator lanes (fp16x2 -> float2).
         * @param v Packed element.
         * @return Accumulator pair {lo, hi}.
         */
        __device__ __forceinline__ float2 mean_widen(const float16x2_t v) {
            return make_float2(__half2float(__low2half(v)), __half2float(__high2half(v)));
        }
        /**
         * @brief Widen one packed element to its mean accumulator lanes (bf16x2 -> float2).
         * @param v Packed element.
         * @return Accumulator pair {lo, hi}.
         */
        __device__ __forceinline__ float2 mean_widen(const bfloat16x2_t& v) {
            return make_float2(__bfloat162float(__low2bfloat16(v)), __bfloat162float(__high2bfloat16(v)));
        }
        /**
         * Signed lanes, unlike the raw-nibble sum accumulate: a wrapped sum divided by numel
         * would be meaningless. Every nibble value is exact in fp32.
         * @brief Widen one packed element to its mean accumulator lanes (int4x2 -> float2, signed).
         * @param v Packed element.
         * @return Accumulator pair {lo, hi} of signed lane values.
         */
        __device__ __forceinline__ float2 mean_widen(const int4x2_t v) {
            return make_float2(dev_i4_lo(v), dev_i4_hi(v));
        }
        /**
         * The mean accumulates integers in fp64 (no modular wrap, unlike sum). Values past 2^53
         * in magnitude can lose low bits (documented).
         * @brief Widen one integer element to fp64.
         * @param v Element.
         * @return Accumulator value.
         */
        template<typename T> requires std::is_integral_v<T>
        __device__ __forceinline__ float64_t mean_widen(const T v) {
            return static_cast<float64_t>(v);
        }

        /**
         * @brief Accumulator add (fp32).
         * @param a Left accumulator.
         * @param b Right accumulator.
         * @return a + b.
         */
        __device__ __forceinline__ float32_t mean_add(const float32_t a, const float32_t b) { return a + b; }
        /**
         * @brief Accumulator add (fp64).
         * @param a Left accumulator.
         * @param b Right accumulator.
         * @return a + b.
         */
        __device__ __forceinline__ float64_t mean_add(const float64_t a, const float64_t b) { return a + b; }
        /**
         * @brief Accumulator add (per-lane fp32 pair).
         * @param a Left accumulator.
         * @param b Right accumulator.
         * @return Lane-wise a + b.
         */
        __device__ __forceinline__ float2 mean_add(const float2 a, const float2 b) { return make_float2(a.x + b.x, a.y + b.y); }

        /**
         * Specialized per accumulator type; the primary template is declared only.
         * @brief Additive identity of a scalar accumulator type.
         */
        template<typename Acc> struct mean_zero_of;
        template<> struct mean_zero_of<float32_t> { static constexpr float32_t value = 0.0f; };
        template<> struct mean_zero_of<float64_t> { static constexpr float64_t value = 0.0; };

        /**
         * Dynamic shared memory is allocated untyped; each kernel casts to its Acc.
         * @brief Typed pointer to the block's dynamic shared memory.
         * @tparam Acc Accumulator type for this reduction.
         * @return Pointer to the launch-supplied shared memory.
         */
        template<typename Acc>
        __device__ __forceinline__ Acc* dyn_shared() {
            extern __shared__ uint8_t s_base[];
            return reinterpret_cast<Acc*>(s_base);
        }

        /**
         * The tree reduction requires a power-of-two block size in [1, 1024].
         * @brief Validate the runtime block size before launch.
         * @param bt Block size returned by KernelTune.
         * @return The same block size if valid.
         * @throws ErrorHandling::InvalidKernelBlockSize if out of range or not a power of two.
         */
        [[nodiscard]] __host__ size_t validate_block_size(const size_t bt) {
            if (bt < 1 || bt > 1024 || (bt & (bt - 1)) != 0) {
                throw ErrorHandling::InvalidKernelBlockSize("block size must be a power of two in [1, 1024], got " + std::to_string(bt));
            }
            return bt;
        }

        /**
         * Clamp to the dtype range, then truncate toward zero — the same narrowing rule as the
         * GradDType float->int boundary cast. The uint64/int64 maxima round UP to 2^64/2^63 in
         * fp64, so the >= guard must fire before the static_cast (UB at exactly 2^64/2^63).
         * A NaN mean stores 0 (there is no integer NaN); unreachable here, inputs are integers.
         * The bounds are computed on the host (nvcc rejects numeric_limits in device code) and
         * are T's lowest()/max(); they are widened to fp64 only for the comparison.
         * @brief Narrow an fp64 mean to an integer dtype.
         * @param v FP64 mean.
         * @param lo T's lowest().
         * @param hi T's max().
         * @return The mean narrowed into T's range.
         */
        template<typename T> requires std::is_integral_v<T>
        __device__ __forceinline__ T dev_narrow_mean_int(const float64_t v, const T lo, const T hi) {
            const auto lo_f = static_cast<float64_t>(lo);
            const auto hi_f = static_cast<float64_t>(hi);
            if (isnan(v)) {
                return T{0};
            }
            if (v <= lo_f) {
                return lo;
            }
            if (v >= hi_f) {
                return hi;
            }
            return static_cast<T>(v);
        }

        /**
         * Divides the final accumulator by numel, then narrows: fp16/bf16 via RNE matching the
         * unaccel pathway, packed lanes independently, ints truncate toward zero + clamp, int4x2
         * per lane into [-8, 7].
         * @brief Store the narrowed mean into out[0].
         * @param out Output element.
         * @param total Final accumulator.
         * @param numel Element count the accumulator summed over.
         * @param lo T's lowest() (T{} for non-integer T, unused).
         * @param hi T's max() (T{} for non-integer T, unused).
         */
        template<typename T, typename Acc>
        __device__ __forceinline__ void store_mean(T* out, const Acc total, const size_t numel, const T lo, const T hi) {
            if constexpr (std::is_same_v<T, float32_t>) {
                out[0] = total / static_cast<float32_t>(numel);
            } else if constexpr (std::is_same_v<T, float64_t>) {
                out[0] = total / static_cast<float64_t>(numel);
            } else if constexpr (std::is_same_v<T, float16_t>) {
                out[0] = __float2half_rn(total / static_cast<float32_t>(numel));
            } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                out[0] = dev_narrow_bf16(total / static_cast<float32_t>(numel));
            } else if constexpr (std::is_same_v<T, float16x2_t>) {
                const auto n = static_cast<float32_t>(numel);
                out[0] = __halves2half2(__float2half_rn(total.x / n), __float2half_rn(total.y / n));
            } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                const auto n = static_cast<float32_t>(numel);
                out[0] = __halves2bfloat162(dev_narrow_bf16(total.x / n), dev_narrow_bf16(total.y / n));
            } else if constexpr (std::is_same_v<T, int4x2_t>) {
                const auto n = static_cast<float32_t>(numel);
                const auto mean_lo = static_cast<int8_t>(fminf(fmaxf(total.x / n, -8.0f), 7.0f));
                const auto mean_hi = static_cast<int8_t>(fminf(fmaxf(total.y / n, -8.0f), 7.0f));
                out[0] = dev_i4_pack(mean_lo, mean_hi);
            } else {
                out[0] = dev_narrow_mean_int<T>(total / static_cast<float64_t>(numel), lo, hi);
            }
        }

        /**
         * Grid-stride per-thread partials, shared-memory tree reduction per block, one partial per block.
         * @note Stage 1 of the two-stage mean.
         * @brief Mean partial kernel (all dtypes).
         * @param data Input elements.
         * @param partials One accumulator per block, written by thread 0.
         * @param numel Element count.
         */
        template<typename T, typename Acc>
        __global__ void mean_partial_kernel(const T* __restrict__ data, Acc* __restrict__ partials, const size_t numel) {
            auto* const s_shared = dyn_shared<Acc>();
            const auto tid = threadIdx.x;
            const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t stride = blockDim.x * gridDim.x;
            Acc acc;
            if constexpr (std::is_same_v<Acc, float2>) {
                acc = make_float2(0.0f, 0.0f);
            } else {
                acc = mean_zero_of<Acc>::value;
            }
            for (size_t i = gid; i < numel; i += stride) {
                acc = mean_add(acc, mean_widen(data[i]));
            }
            s_shared[tid] = acc;
            __syncthreads();
            for (auto s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    s_shared[tid] = mean_add(s_shared[tid], s_shared[tid + s]);
                }
                __syncthreads();
            }
            if (tid == 0) {
                partials[blockIdx.x] = s_shared[0];
            }
        }

        /**
         * Single block reduces the per-block partials, divides by numel, and writes the narrowed
         * mean to out[0].
         * @note Stage 2 of the two-stage mean.
         * @brief Mean finalize kernel.
         * @param partials Per-block accumulators from mean_partial_kernel.
         * @param count Number of partials.
         * @param out Output element.
         * @param numel Element count the partials summed over.
         * @param lo T's lowest() (T{} for non-integer T, unused).
         * @param hi T's max() (T{} for non-integer T, unused).
         */
        template<typename T, typename Acc>
        __global__ void mean_final_kernel(const Acc* __restrict__ partials, const size_t count, T* __restrict__ out, const size_t numel, const T lo, const T hi) {
            auto* const s_shared = dyn_shared<Acc>();
            const auto tid = threadIdx.x;
            Acc acc;
            if constexpr (std::is_same_v<Acc, float2>) {
                acc = make_float2(0.0f, 0.0f);
            } else {
                acc = mean_zero_of<Acc>::value;
            }
            for (auto i = tid; i < count; i += blockDim.x) {
                acc = mean_add(acc, partials[i]);
            }
            s_shared[tid] = acc;
            __syncthreads();
            for (auto s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    s_shared[tid] = mean_add(s_shared[tid], s_shared[tid + s]);
                }
                __syncthreads();
            }
            if (tid == 0) {
                store_mean<T>(out, s_shared[0], numel, lo, hi);
            }
        }

        /**
         * Scratch is worker-local alloc/free; the free syncs the default stream, so the result is
         * complete in out before the worker returns. numel == 0 never reaches here
         * (Tensor::mean throws first). Integer bounds are computed on the host (nvcc rejects
         * numeric_limits in device code).
         * @brief Host-side two-stage mean.
         * @param data Input elements.
         * @param numel Element count.
         * @param out Output element.
         */
        template<typename T, typename Acc>
        __host__ void run_mean(const T* data, const size_t numel, const size_t bt, T* out) {
            // Integer bounds are computed on the host (nvcc rejects
            // numeric_limits in device code); T{} for non-integer T, unused.
            auto lo = T{};
            auto hi = T{};
            if constexpr (std::is_integral_v<T>) {
                lo = std::numeric_limits<T>::lowest();
                hi = std::numeric_limits<T>::max();
            }
            const auto blocks = grid_for(numel, bt);
            Acc* partials = nullptr;
            CUDA_CHECK_ERROR(cudaMalloc(&partials, blocks * sizeof(Acc)));
            // Largest Acc (float2) is 8 B; 1024 * 8 B = 8 KB, under the 48 KB default.
            const auto shared_bytes = bt * sizeof(Acc);
            mean_partial_kernel<T, Acc><<<blocks, bt, shared_bytes>>>(data, partials, numel);
            mean_final_kernel<T, Acc><<<1, bt, shared_bytes>>>(partials, blocks, out, numel, lo, hi);
            CUDA_CHECK_ERROR(cudaGetLastError());
            CUDA_CHECK_ERROR(cudaFree(partials));
        }
    }

    /**
     * Runs on the CPU thread pool at high priority; the future lands on both tensors.
     * @brief GPU full reduction, averages every element into the scalar output tensor.
     * @param t Input tensor.
     * @param out Single element output tensor, same dtype as t.
     */
    __host__ void mean_gpu(const Tensor& t, const Tensor& out) {
        if (t.device != Device::GPU || out.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(t.device));
        }
        const auto* const t_data = t.data;
        auto* const out_data = out.data;
        const auto t_offset = t.offset;
        const auto dtype = t.dtype;
        const auto numel = t.numel;
        /**
         * Raw pointers are captured by value: the future lands on the tensors, so buffer teardown
         * always drains this worker first. Dispatches on dtype to the typed host launcher.
         * @brief Pool worker for the GPU mean reduction.
         * @note Lambda subcall. Self-execution prohibited.
         */
        const auto worker = [=] {
            const auto bt = validate_block_size(BeanTensor::KernelTune::block_size_for(BeanTensor::KernelTune::GpuFamily::Reduction, numel));
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: run_mean<float32_t, float32_t>(static_cast<const float32_t*>(t_data) + t_offset, numel, bt, static_cast<float32_t*>(out_data)); break;
                case DType::Float64: run_mean<float64_t, float64_t>(static_cast<const float64_t*>(t_data) + t_offset, numel, bt, static_cast<float64_t*>(out_data)); break;
                case DType::Float16: run_mean<float16_t, float32_t>(static_cast<const float16_t*>(t_data) + t_offset, numel, bt, static_cast<float16_t*>(out_data)); break;
                case DType::BFloat16: run_mean<bfloat16_t, float32_t>(static_cast<const bfloat16_t*>(t_data) + t_offset, numel, bt, static_cast<bfloat16_t*>(out_data)); break;
                case DType::Float16x2: run_mean<float16x2_t, float2>(static_cast<const float16x2_t*>(t_data) + t_offset, numel, bt, static_cast<float16x2_t*>(out_data)); break;
                case DType::BFloat16x2: run_mean<bfloat16x2_t, float2>(static_cast<const bfloat16x2_t*>(t_data) + t_offset, numel, bt, static_cast<bfloat16x2_t*>(out_data)); break;
                case DType::Int8: run_mean<int8_t, float64_t>(static_cast<const int8_t*>(t_data) + t_offset, numel, bt, static_cast<int8_t*>(out_data)); break;
                case DType::UInt8: run_mean<uint8_t, float64_t>(static_cast<const uint8_t*>(t_data) + t_offset, numel, bt, static_cast<uint8_t*>(out_data)); break;
                case DType::Int16: run_mean<int16_t, float64_t>(static_cast<const int16_t*>(t_data) + t_offset, numel, bt, static_cast<int16_t*>(out_data)); break;
                case DType::UInt16: run_mean<uint16_t, float64_t>(static_cast<const uint16_t*>(t_data) + t_offset, numel, bt, static_cast<uint16_t*>(out_data)); break;
                case DType::Int32: run_mean<int32_t, float64_t>(static_cast<const int32_t*>(t_data) + t_offset, numel, bt, static_cast<int32_t*>(out_data)); break;
                case DType::UInt32: run_mean<uint32_t, float64_t>(static_cast<const uint32_t*>(t_data) + t_offset, numel, bt, static_cast<uint32_t*>(out_data)); break;
                case DType::Int64: run_mean<int64_t, float64_t>(static_cast<const int64_t*>(t_data) + t_offset, numel, bt, static_cast<int64_t*>(out_data)); break;
                case DType::UInt64: run_mean<uint64_t, float64_t>(static_cast<const uint64_t*>(t_data) + t_offset, numel, bt, static_cast<uint64_t*>(out_data)); break;
                case DType::Int4x2: run_mean<int4x2_t, float2>(static_cast<const int4x2_t*>(t_data) + t_offset, numel, bt, static_cast<int4x2_t*>(out_data)); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
            // run_mean's cudaFree already synced the default stream.
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        t.give_future(future);
        out.give_future(future);
    }
}
#endif
