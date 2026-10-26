/**
 * @file MinCUDA.cu
 * @author GreenBean
 * @brief CUDA backend for the min op.
 * Divergent twin of Math/Reductions/Min/MinHIP.cpp — the two backends are
 * written separately per project rule (no HIP-only unification), but must stay
 * behaviorally identical: same NaN-propagating combine (canonical quiet NaN),
 * same -0.0 tie canonicalization, same per-lane rules for the packed dtypes.
 * @version 0.3
 * @date 2026-10-26
 */
#if defined(USE_CUDA)
#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "Math/Reductions/Min/Min.h"
#include "Math/Reductions/ReductionsCudaShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"
#include "Helpers/KernelTune.h"

namespace BeanTensor::Tensors::detail {
    using namespace reductions_gpu;

    namespace {
        /**
         * @brief Canonical quiet NaN of an accumulator float type (empty payload).
         * @return The canonical NaN.
         */
        __device__ __forceinline__ float32_t dev_canon_nan(const float32_t) {
            return __builtin_bit_cast(float32_t, UINT32_C(0x7FC00000));
        }
        /**
         * @brief Canonical quiet NaN of an accumulator float type (empty payload).
         * @return The canonical NaN.
         */
        __device__ __forceinline__ float64_t dev_canon_nan(const float64_t) {
            return __builtin_bit_cast(float64_t, UINT64_C(0x7FF8000000000000));
        }
        /**
         * Canonical fp16 quiet NaN is 0x7E00; __float2half_rn cannot be trusted to
         * produce it on every stack (some narrow NaN to 0x7FFF), so stores
         * canonicalize NaN lanes explicitly, mirroring the CPU/HIP backends.
         * @brief Canonical quiet NaN of an fp16 lane.
         * @return The canonical half NaN (0x7E00).
         */
        __device__ __forceinline__ float16_t dev_canon_nan(const float16_t) {
            return __builtin_bit_cast(float16_t, uint16_t{0x7E00});
        }

        /**
         * @brief -0.0 of an accumulator float type, by bit pattern.
         * @return -0.0.
         */
        __device__ __forceinline__ float32_t dev_neg_zero(const float32_t) {
            return __builtin_bit_cast(float32_t, UINT32_C(0x80000000));
        }
        /**
         * @brief -0.0 of an accumulator float type, by bit pattern.
         * @return -0.0.
         */
        __device__ __forceinline__ float64_t dev_neg_zero(const float64_t) {
            return __builtin_bit_cast(float64_t, UINT64_C(0x8000000000000000));
        }

        /**
         * @brief Is an accumulator float exactly +/-0.0? (bit test, no float ==)
         * @param v Value to test.
         * @return T/F - the magnitude bits are all zero.
         */
        __device__ __forceinline__ bool dev_is_zero(const float32_t v) {
            return (__builtin_bit_cast(uint32_t, v) & UINT32_C(0x7FFFFFFF)) == 0u;
        }
        /**
         * @brief Is an accumulator float exactly +/-0.0? (bit test, no float ==)
         * @param v Value to test.
         * @return T/F - the magnitude bits are all zero.
         */
        __device__ __forceinline__ bool dev_is_zero(const float64_t v) {
            return (__builtin_bit_cast(uint64_t, v) & UINT64_C(0x7FFFFFFFFFFFFFFF)) == 0u;
        }

        /**
         * NaN-propagating min combine; a +0.0/-0.0 tie canonicalizes to -0.0 so every
         * block/tree shape agrees bit-for-bit (same combine as the CPU backend).
         * @brief Combine two scalar float accumulators, min semantics.
         * @param a Left accumulator.
         * @param b Right accumulator.
         * @return The winning accumulator (any NaN propagates as the canonical quiet NaN).
         */
        template<typename T>
        __device__ __forceinline__ T dev_min_combine(const T a, const T b) {
            if (isnan(a) || isnan(b)) {
                return dev_canon_nan(a);
            }
            if (b < a) {
                return b;
            }
            if (a < b) {
                return a;
            }
            if (dev_is_zero(a)) {
                return dev_neg_zero(a);
            }
            return a;
        }
        /**
         * Per-lane independent combine; the per-lane winners may come from different elements.
         * @brief Combine two per-lane fp32 accumulators, min semantics.
         * @param a Left accumulator pair.
         * @param b Right accumulator pair.
         * @return Lane-wise winning pair.
         */
        __device__ __forceinline__ float2 dev_min_combine(const float2 a, const float2 b) {
            return make_float2(dev_min_combine(a.x, b.x), dev_min_combine(a.y, b.y));
        }
        /**
         * @brief Combine two int64 accumulators, min semantics.
         * @param a Left accumulator.
         * @param b Right accumulator.
         * @return The winning accumulator.
         */
        __device__ __forceinline__ int64_t dev_min_combine(const int64_t a, const int64_t b) {
            return b < a ? b : a;
        }
        /**
         * @brief Combine two uint64 accumulators, min semantics.
         * @param a Left accumulator.
         * @param b Right accumulator.
         * @return The winning accumulator.
         */
        __device__ __forceinline__ uint64_t dev_min_combine(const uint64_t a, const uint64_t b) {
            return b < a ? b : a;
        }
        /**
         * @brief Combine two int4x2 lane-pair accumulators, per-lane signed min.
         * @param a Left accumulator pair.
         * @param b Right accumulator pair.
         * @return Lane-wise winning pair.
         */
        __device__ __forceinline__ int2 dev_min_combine(const int2 a, const int2 b) {
            return make_int2(b.x < a.x ? b.x : a.x, b.y < a.y ? b.y : a.y);
        }

        /**
         * The accumulator always holds the widened twin of an actual input element, so the
         * narrowed non-NaN result is bit-for-bit one of the inputs; +inf is the min identity.
         * @brief Min identity of an accumulator type.
         * @return The identity accumulator.
         */
        template<typename Acc>
        __device__ __forceinline__ Acc min_identity() {
            if constexpr (std::is_same_v<Acc, float32_t>) {
                return __builtin_bit_cast(float32_t, UINT32_C(0x7F800000)); // +inf
            } else if constexpr (std::is_same_v<Acc, float64_t>) {
                return __builtin_bit_cast(float64_t, UINT64_C(0x7FF0000000000000)); // +inf
            } else if constexpr (std::is_same_v<Acc, float2>) {
                return make_float2(__builtin_bit_cast(float32_t, UINT32_C(0x7F800000)), __builtin_bit_cast(float32_t, UINT32_C(0x7F800000)));
            } else if constexpr (std::is_same_v<Acc, int64_t>) {
                return INT64_C(0x7FFFFFFFFFFFFFFF);
            } else if constexpr (std::is_same_v<Acc, uint64_t>) {
                return UINT64_MAX;
            } else {
                static_assert(std::is_same_v<Acc, int2>, "min_identity: unknown accumulator type");
                return make_int2(7, 7);
            }
        }

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
         * @brief Element to accumulator (fp32 passthrough).
         * @param v Element.
         * @return Accumulator value.
         */
        __device__ __forceinline__ float32_t to_acc(const float32_t v) { return v; }
        /**
         * @brief Element to accumulator (fp64 passthrough).
         * @param v Element.
         * @return Accumulator value.
         */
        __device__ __forceinline__ float64_t to_acc(const float64_t v) { return v; }
        /**
         * @brief Element to accumulator (fp16 -> fp32, exact).
         * @param v Element.
         * @return Accumulator value.
         */
        __device__ __forceinline__ float32_t to_acc(const float16_t v) { return __half2float(v); }
        /**
         * @brief Element to accumulator (bf16 -> fp32, exact).
         * @param v Element.
         * @return Accumulator value.
         */
        __device__ __forceinline__ float32_t to_acc(const bfloat16_t v) { return __bfloat162float(v); }
        /**
         * @brief Element to accumulator lanes (fp16x2 -> float2, exact per lane).
         * @param v Packed element.
         * @return Accumulator pair {lo, hi}.
         */
        __device__ __forceinline__ float2 to_acc(const float16x2_t& v) {
            return make_float2(__half2float(__low2half(v)), __half2float(__high2half(v)));
        }
        /**
         * @brief Element to accumulator lanes (bf16x2 -> float2, exact per lane).
         * @param v Packed element.
         * @return Accumulator pair {lo, hi}.
         */
        __device__ __forceinline__ float2 to_acc(const bfloat16x2_t& v) {
            return make_float2(__bfloat162float(__low2bfloat16(v)), __bfloat162float(__high2bfloat16(v)));
        }
        /**
         * @brief Element to accumulator lanes (int4x2 -> signed lane pair).
         * @param v Packed element.
         * @return Accumulator pair {lo, hi}.
         */
        __device__ __forceinline__ int2 to_acc(const int4x2_t v) {
            return make_int2(dev_i4_lo(v), dev_i4_hi(v));
        }
        /**
         * @brief Element to accumulator (signed integers -> int64, exact).
         * @param v Element.
         * @return Accumulator value.
         */
        template<typename T> requires (std::is_integral_v<T> && std::is_signed_v<T>)
        __device__ __forceinline__ int64_t to_acc(const T v) { return static_cast<int64_t>(v); }
        /**
         * @brief Element to accumulator (unsigned integers -> uint64, exact).
         * @param v Element.
         * @return Accumulator value.
         */
        template<typename T> requires (std::is_integral_v<T> && std::is_unsigned_v<T>)
        __device__ __forceinline__ uint64_t to_acc(const T v) { return static_cast<uint64_t>(v); }

        /**
         * Grid-stride per-thread best, shared-memory tree combine per block, one partial per block.
         * @note Stage 1 of the two-stage min.
         * @brief Min partial kernel (all dtypes).
         * @param data Input elements.
         * @param partials One accumulator per block, written by thread 0.
         * @param numel Element count.
         */
        template<typename T, typename Acc>
        __global__ void min_partial_kernel(const T* __restrict__ data, Acc* __restrict__ partials, const size_t numel) {
            auto* const s_shared = dyn_shared<Acc>();
            const auto tid = threadIdx.x;
            const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t stride = blockDim.x * gridDim.x;
            auto acc = min_identity<Acc>();
            for (size_t i = gid; i < numel; i += stride) {
                acc = dev_min_combine(acc, to_acc(data[i]));
            }
            s_shared[tid] = acc;
            __syncthreads();
            for (auto s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    s_shared[tid] = dev_min_combine(s_shared[tid], s_shared[tid + s]);
                }
                __syncthreads();
            }
            if (tid == 0) {
                partials[blockIdx.x] = s_shared[0];
            }
        }

        /**
         * The accumulator holds the widened winning element, so narrowing is exact for the
         * non-NaN result (fp16/bf16 RNE, per-lane for packed, truncation for integers); a NaN
         * accumulator narrows to the canonical lane NaN, matching the CPU backend.
         * @brief Store the winning value into out[0].
         * @param out Output element.
         * @param total Final accumulator.
         */
        template<typename T, typename Acc>
        __device__ __forceinline__ void store_min(T* out, const Acc total) {
            if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                out[0] = total;
            } else if constexpr (std::is_same_v<T, float16_t>) {
                out[0] = isnan(total) ? dev_canon_nan(float16_t{}) : __float2half_rn(total);
            } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                out[0] = dev_narrow_bf16(total);
            } else if constexpr (std::is_same_v<T, float16x2_t>) {
                out[0] = __halves2half2(
                    isnan(total.x) ? dev_canon_nan(float16_t{}) : __float2half_rn(total.x),
                    isnan(total.y) ? dev_canon_nan(float16_t{}) : __float2half_rn(total.y));
            } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                out[0] = __halves2bfloat162(dev_narrow_bf16(total.x), dev_narrow_bf16(total.y));
            } else if constexpr (std::is_same_v<T, int4x2_t>) {
                out[0] = dev_i4_pack(static_cast<int8_t>(total.x), static_cast<int8_t>(total.y));
            } else if constexpr (std::is_integral_v<T>) {
                out[0] = static_cast<T>(total);
            }
        }

        /**
         * Single block reduces the per-block partials and writes the winning value to out[0].
         * @note Stage 2 of the two-stage min.
         * @brief Min finalize kernel.
         * @param partials Per-block accumulators from min_partial_kernel.
         * @param count Number of partials.
         * @param out Output element.
         */
        template<typename T, typename Acc>
        __global__ void min_final_kernel(const Acc* __restrict__ partials, const size_t count, T* __restrict__ out) {
            auto* const s_shared = dyn_shared<Acc>();
            const auto tid = threadIdx.x;
            auto acc = min_identity<Acc>();
            for (auto i = tid; i < count; i += blockDim.x) {
                acc = dev_min_combine(acc, partials[i]);
            }
            s_shared[tid] = acc;
            __syncthreads();
            for (auto s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    s_shared[tid] = dev_min_combine(s_shared[tid], s_shared[tid + s]);
                }
                __syncthreads();
            }
            if (tid == 0) {
                store_min<T>(out, s_shared[0]);
            }
        }

        /**
         * Scratch is worker-local alloc/free; the free syncs the default stream, so the result is complete in out before the worker returns.
         * @brief Host-side two-stage min.
         * @param data Input elements.
         * @param numel Element count.
         * @param out Output element.
         */
        template<typename T, typename Acc>
        __host__ void run_min(const T* data, const size_t numel, const size_t bt, T* out) {
            const auto blocks = grid_for(numel, bt);
            Acc* partials = nullptr;
            CUDA_CHECK_ERROR(cudaMalloc(&partials, blocks * sizeof(Acc)));
            // Largest Acc (float2/int2) is 8 B; 1024 * 8 B = 8 KB, under the 48 KB default.
            const auto shared_bytes = bt * sizeof(Acc);
            min_partial_kernel<T, Acc><<<blocks, bt, shared_bytes>>>(data, partials, numel);
            min_final_kernel<T, Acc><<<1, bt, shared_bytes>>>(partials, blocks, out);
            CUDA_CHECK_ERROR(cudaGetLastError());
            CUDA_CHECK_ERROR(cudaFree(partials));
        }
    }

    /**
     * Runs on the CPU thread pool at high priority; the future lands on both tensors.
     * @brief GPU full reduction, smallest element written into the scalar output tensor.
     * @param t Input tensor.
     * @param out Single element output tensor, same dtype as t.
     */
    __host__ void min_gpu(const Tensor& t, const Tensor& out) {
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
         * @brief Pool worker for the GPU min reduction.
         * @note Lambda subcall. Self-execution prohibited.
         */
        const auto worker = [=] {
            const auto bt = validate_block_size(BeanTensor::KernelTune::block_size_for(BeanTensor::KernelTune::GpuFamily::Reduction, numel));
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: run_min<float32_t, float32_t>(static_cast<const float32_t*>(t_data) + t_offset, numel, bt, static_cast<float32_t*>(out_data)); break;
                case DType::Float64: run_min<float64_t, float64_t>(static_cast<const float64_t*>(t_data) + t_offset, numel, bt, static_cast<float64_t*>(out_data)); break;
                case DType::Float16: run_min<float16_t, float32_t>(static_cast<const float16_t*>(t_data) + t_offset, numel, bt, static_cast<float16_t*>(out_data)); break;
                case DType::BFloat16: run_min<bfloat16_t, float32_t>(static_cast<const bfloat16_t*>(t_data) + t_offset, numel, bt, static_cast<bfloat16_t*>(out_data)); break;
                case DType::Float16x2: run_min<float16x2_t, float2>(static_cast<const float16x2_t*>(t_data) + t_offset, numel, bt, static_cast<float16x2_t*>(out_data)); break;
                case DType::BFloat16x2: run_min<bfloat16x2_t, float2>(static_cast<const bfloat16x2_t*>(t_data) + t_offset, numel, bt, static_cast<bfloat16x2_t*>(out_data)); break;
                case DType::Int8: run_min<int8_t, int64_t>(static_cast<const int8_t*>(t_data) + t_offset, numel, bt, static_cast<int8_t*>(out_data)); break;
                case DType::UInt8: run_min<uint8_t, uint64_t>(static_cast<const uint8_t*>(t_data) + t_offset, numel, bt, static_cast<uint8_t*>(out_data)); break;
                case DType::Int16: run_min<int16_t, int64_t>(static_cast<const int16_t*>(t_data) + t_offset, numel, bt, static_cast<int16_t*>(out_data)); break;
                case DType::UInt16: run_min<uint16_t, uint64_t>(static_cast<const uint16_t*>(t_data) + t_offset, numel, bt, static_cast<uint16_t*>(out_data)); break;
                case DType::Int32: run_min<int32_t, int64_t>(static_cast<const int32_t*>(t_data) + t_offset, numel, bt, static_cast<int32_t*>(out_data)); break;
                case DType::UInt32: run_min<uint32_t, uint64_t>(static_cast<const uint32_t*>(t_data) + t_offset, numel, bt, static_cast<uint32_t*>(out_data)); break;
                case DType::Int64: run_min<int64_t, int64_t>(static_cast<const int64_t*>(t_data) + t_offset, numel, bt, static_cast<int64_t*>(out_data)); break;
                case DType::UInt64: run_min<uint64_t, uint64_t>(static_cast<const uint64_t*>(t_data) + t_offset, numel, bt, static_cast<uint64_t*>(out_data)); break;
                case DType::Int4x2: run_min<int4x2_t, int2>(static_cast<const int4x2_t*>(t_data) + t_offset, numel, bt, static_cast<int4x2_t*>(out_data)); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
            // run_min's cudaFree already synced the default stream.
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        t.give_future(future);
        out.give_future(future);
    }
}
#endif
