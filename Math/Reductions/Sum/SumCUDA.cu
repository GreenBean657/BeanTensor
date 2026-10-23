/**
 * @file SumCUDA.cu
 * @author GreenBean
 * @brief CUDA backend for the sum op.
 * Divergent twin of Math/Reductions/Sum/SumHIP.cpp — the two backends are
 * written separately per project rule (no HIP-only unification), but must stay
 * behaviorally identical: same fp32/fp64 compute widths, same RNE narrowing,
 * same uint64 wrap integer sums.
 * @version 0.2
 * @date 2026-10-20
 */
#if defined(USE_CUDA)
#include <cstdint>
#include <memory>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "Math/Reductions/Sum/Sum.h"
#include "Math/Reductions/ReductionsCudaShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    using namespace reductions_gpu;

    namespace {
        /**
         * @brief Widen one element to its accumulator (fp32 passthrough).
         * @param v Element.
         * @return Accumulator value.
         */
        __device__ __forceinline__ float32_t widen_acc(const float32_t v) { return v; }
        /**
         * @brief Widen one element to its accumulator (fp64 passthrough).
         * @param v Element.
         * @return Accumulator value.
         */
        __device__ __forceinline__ float64_t widen_acc(const float64_t v) { return v; }
        /**
         * @brief Widen one element to its accumulator (fp16 -> fp32).
         * @param v Element.
         * @return Accumulator value.
         */
        __device__ __forceinline__ float32_t widen_acc(const float16_t v) { return __half2float(v); }
        /**
         * @brief Widen one element to its accumulator (bf16 -> fp32).
         * @param v Element.
         * @return Accumulator value.
         */
        __device__ __forceinline__ float32_t widen_acc(const bfloat16_t v) { return __bfloat162float(v); }
        /**
         * @brief Widen one packed element to its accumulator lanes (fp16x2 -> float2).
         * @param v Packed element.
         * @return Accumulator pair {lo, hi}.
         */
        __device__ __forceinline__ float2 widen_acc(const float16x2_t& v) {
            return make_float2(__half2float(__low2half(v)), __half2float(__high2half(v)));
        }
        /**
         * @brief Widen one packed element to its accumulator lanes (bf16x2 -> float2).
         * @param v Packed element.
         * @return Accumulator pair {lo, hi}.
         */
        __device__ __forceinline__ float2 widen_acc(const bfloat16x2_t& v) {
            return make_float2(__bfloat162float(__low2bfloat16(v)), __bfloat162float(__high2bfloat16(v)));
        }
        /**
         * Raw nibbles accumulate, a nibble IS its two's-complement residue.
         * @brief Widen one packed element to its accumulator lanes (int4x2 -> uint2).
         * @param v Packed element.
         * @return Accumulator pair {lo, hi} of raw nibbles.
         */
        __device__ __forceinline__ uint2 widen_acc(const int4x2_t v) {
            const auto bits = __builtin_bit_cast(uint8_t, v);
            return make_uint2(bits & 0x0Fu, bits >> 4);
        }
        /**
         * Modular accumulate keeps every backend/chunking bit-identical.
         * @brief Widen one integer element to its unsigned twin in uint64.
         * @param v Element.
         * @return Accumulator value.
         */
        template<typename T> requires std::is_integral_v<T>
        __device__ __forceinline__ uint64_t widen_acc(const T v) {
            return static_cast<uint64_t>(static_cast<std::make_unsigned_t<T>>(v));
        }

        /**
         * @brief Accumulator add (fp32).
         * @param a Left accumulator.
         * @param b Right accumulator.
         * @return a + b.
         */
        __device__ __forceinline__ float32_t acc_add(const float32_t a, const float32_t b) { return a + b; }
        /**
         * @brief Accumulator add (fp64).
         * @param a Left accumulator.
         * @param b Right accumulator.
         * @return a + b.
         */
        __device__ __forceinline__ float64_t acc_add(const float64_t a, const float64_t b) { return a + b; }
        /**
         * @brief Accumulator add (uint64, modular).
         * @param a Left accumulator.
         * @param b Right accumulator.
         * @return a + b, wrapping.
         */
        __device__ __forceinline__ uint64_t acc_add(const uint64_t a, const uint64_t b) { return a + b; }
        /**
         * @brief Accumulator add (per-lane fp32 pair).
         * @param a Left accumulator.
         * @param b Right accumulator.
         * @return Lane-wise a + b.
         */
        __device__ __forceinline__ float2 acc_add(const float2 a, const float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
        /**
         * @brief Accumulator add (per-lane uint32 pair, raw nibble sums).
         * @param a Left accumulator.
         * @param b Right accumulator.
         * @return Lane-wise a + b.
         */
        __device__ __forceinline__ uint2 acc_add(const uint2 a, const uint2 b) { return make_uint2(a.x + b.x, a.y + b.y); }

        /**
         * Specialized per accumulator type; the primary template is declared only.
         * @brief Additive identity of a scalar accumulator type.
         */
        template<typename Acc> struct acc_zero_of;
        template<> struct acc_zero_of<float32_t> { static constexpr float32_t value = 0.0f; };
        template<> struct acc_zero_of<float64_t> { static constexpr float64_t value = 0.0; };
        template<> struct acc_zero_of<uint64_t> { static constexpr uint64_t value = 0; };

        /**
         * Grid-stride per-thread partials, shared-memory tree reduction per block, one partial per block.
         * @note Stage 1 of the two-stage sum.
         * @brief Sum partial kernel (all dtypes).
         * @param data Input elements.
         * @param partials One accumulator per block, written by thread 0.
         * @param numel Element count.
         */
        template<typename T, typename Acc>
        __global__ void sum_partial_kernel(const T* __restrict__ data, Acc* __restrict__ partials, const size_t numel) {
            __shared__ Acc shared[BT_GPU_THREADS];
            const auto tid = threadIdx.x;
            const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t stride = blockDim.x * gridDim.x;
            auto acc = Acc{};
            if constexpr (std::is_same_v<Acc, float2>) {
                acc = make_float2(0.0f, 0.0f);
            } else if constexpr (std::is_same_v<Acc, uint2>) {
                acc = make_uint2(0u, 0u);
            } else {
                acc = acc_zero_of<Acc>::value;
            }
            for (size_t i = gid; i < numel; i += stride) {
                acc = acc_add(acc, widen_acc(data[i]));
            }
            shared[tid] = acc;
            __syncthreads();
            for (auto s = BT_GPU_THREADS / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared[tid] = acc_add(shared[tid], shared[tid + s]);
                }
                __syncthreads();
            }
            if (tid == 0) {
                partials[blockIdx.x] = shared[0];
            }
        }

        /**
         * Fp16/bf16 narrow via RNE matching the unaccel pathway, packed lanes independently, ints by truncation (wrap).
         * @brief Store the narrowed sum into out[0].
         * @param out Output element.
         * @param total Final accumulator.
         */
        template<typename T, typename Acc>
        __device__ __forceinline__ void store_sum(T* out, const Acc total) {
            if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t> || std::is_integral_v<T>) {
                out[0] = static_cast<T>(total);
            } else if constexpr (std::is_same_v<T, float16_t>) {
                out[0] = __float2half_rn(total);
            } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                out[0] = dev_narrow_bf16(total);
            } else if constexpr (std::is_same_v<T, float16x2_t>) {
                out[0] = __halves2half2(__float2half_rn(total.x), __float2half_rn(total.y));
            } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                out[0] = __halves2bfloat162(dev_narrow_bf16(total.x), dev_narrow_bf16(total.y));
            } else if constexpr (std::is_same_v<T, int4x2_t>) {
                out[0] = dev_i4_pack(static_cast<int8_t>(total.x & 0x0Fu), static_cast<int8_t>(total.y & 0x0Fu));
            }
        }

        /**
         * Single block reduces the per-block partials and writes the narrowed result to out[0].
         * @note Stage 2 of the two-stage sum.
         * @brief Sum finalize kernel.
         * @param partials Per-block accumulators from sum_partial_kernel.
         * @param count Number of partials.
         * @param out Output element.
         */
        template<typename T, typename Acc>
        __global__ void sum_final_kernel(const Acc* __restrict__ partials, const size_t count, T* __restrict__ out) {
            __shared__ Acc shared[BT_GPU_THREADS];
            const auto tid = threadIdx.x;
            auto acc = Acc{};
            if constexpr (std::is_same_v<Acc, float2>) {
                acc = make_float2(0.0f, 0.0f);
            } else if constexpr (std::is_same_v<Acc, uint2>) {
                acc = make_uint2(0u, 0u);
            } else {
                acc = acc_zero_of<Acc>::value;
            }
            for (auto i = tid; i < count; i += BT_GPU_THREADS) {
                acc = acc_add(acc, partials[i]);
            }
            shared[tid] = acc;
            __syncthreads();
            for (auto s = BT_GPU_THREADS / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared[tid] = acc_add(shared[tid], shared[tid + s]);
                }
                __syncthreads();
            }
            if (tid == 0) {
                store_sum<T>(out, shared[0]);
            }
        }

        /**
         * Scratch is worker-local alloc/free; the free syncs the default stream, so the result is complete in out before the worker returns.
         * @brief Host-side two-stage sum.
         * @param data Input elements.
         * @param numel Element count.
         * @param out Output element.
         */
        template<typename T, typename Acc>
        __host__ void run_sum(const T* data, const size_t numel, T* out) {
            const auto blocks = grid_for(numel);
            Acc* partials = nullptr;
            CUDA_CHECK_ERROR(cudaMalloc(&partials, blocks * sizeof(Acc)));
            sum_partial_kernel<T, Acc><<<blocks, BT_GPU_THREADS>>>(data, partials, numel);
            sum_final_kernel<T, Acc><<<1, BT_GPU_THREADS>>>(partials, blocks, out);
            CUDA_CHECK_ERROR(cudaGetLastError());
            CUDA_CHECK_ERROR(cudaFree(partials));
        }
    }

    /**
     * Runs on the CPU thread pool at high priority; the future lands on both tensors.
     * @brief GPU full reduction, sums every element into the scalar output tensor.
     * @param t Input tensor.
     * @param out Single element output tensor, same dtype as t.
     */
    __host__ void sum_gpu(const Tensor& t, const Tensor& out) {
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
         * @brief Pool worker for the GPU sum reduction.
         * @note Lambda subcall. Self-execution prohibited.
         */
        const auto worker = [=] {
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: run_sum<float32_t, float32_t>(static_cast<const float32_t*>(t_data) + t_offset, numel, static_cast<float32_t*>(out_data)); break;
                case DType::Float64: run_sum<float64_t, float64_t>(static_cast<const float64_t*>(t_data) + t_offset, numel, static_cast<float64_t*>(out_data)); break;
                case DType::Float16: run_sum<float16_t, float32_t>(static_cast<const float16_t*>(t_data) + t_offset, numel, static_cast<float16_t*>(out_data)); break;
                case DType::BFloat16: run_sum<bfloat16_t, float32_t>(static_cast<const bfloat16_t*>(t_data) + t_offset, numel, static_cast<bfloat16_t*>(out_data)); break;
                case DType::Float16x2: run_sum<float16x2_t, float2>(static_cast<const float16x2_t*>(t_data) + t_offset, numel, static_cast<float16x2_t*>(out_data)); break;
                case DType::BFloat16x2: run_sum<bfloat16x2_t, float2>(static_cast<const bfloat16x2_t*>(t_data) + t_offset, numel, static_cast<bfloat16x2_t*>(out_data)); break;
                case DType::Int8: run_sum<int8_t, uint64_t>(static_cast<const int8_t*>(t_data) + t_offset, numel, static_cast<int8_t*>(out_data)); break;
                case DType::UInt8: run_sum<uint8_t, uint64_t>(static_cast<const uint8_t*>(t_data) + t_offset, numel, static_cast<uint8_t*>(out_data)); break;
                case DType::Int16: run_sum<int16_t, uint64_t>(static_cast<const int16_t*>(t_data) + t_offset, numel, static_cast<int16_t*>(out_data)); break;
                case DType::UInt16: run_sum<uint16_t, uint64_t>(static_cast<const uint16_t*>(t_data) + t_offset, numel, static_cast<uint16_t*>(out_data)); break;
                case DType::Int32: run_sum<int32_t, uint64_t>(static_cast<const int32_t*>(t_data) + t_offset, numel, static_cast<int32_t*>(out_data)); break;
                case DType::UInt32: run_sum<uint32_t, uint64_t>(static_cast<const uint32_t*>(t_data) + t_offset, numel, static_cast<uint32_t*>(out_data)); break;
                case DType::Int64: run_sum<int64_t, uint64_t>(static_cast<const int64_t*>(t_data) + t_offset, numel, static_cast<int64_t*>(out_data)); break;
                case DType::UInt64: run_sum<uint64_t, uint64_t>(static_cast<const uint64_t*>(t_data) + t_offset, numel, static_cast<uint64_t*>(out_data)); break;
                case DType::Int4x2: run_sum<int4x2_t, uint2>(static_cast<const int4x2_t*>(t_data) + t_offset, numel, static_cast<int4x2_t*>(out_data)); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
            // run_sum's cudaFree already synced the default stream.
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        t.give_future(future);
        out.give_future(future);
    }
}
#endif
