/**
 * @file ArgmaxCUDA.cu
 * @author GreenBean
 * @brief CUDA backend for the argmax op.
 * Divergent twin of Math/Reductions/Argmax/ArgmaxHIP.cpp — the two backends are
 * written separately per project rule (no HIP-only unification), but must stay
 * behaviorally identical: same NaN-wins argmax rule.
 * @version 0.3
 * @date 2026-10-22
 */
#if defined(USE_CUDA)
#include <cstdint>
#include <memory>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "Math/Reductions/Argmax/Argmax.h"
#include "Math/Reductions/ReductionsCudaShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    using namespace reductions_gpu;

    namespace {
        /**
         * idx == UINT64_MAX means "empty" (no element seen yet).
         * @brief (key, flat index) pair for argmax reductions.
         */
        template<typename K>
        struct ValIdx {
            K key;
            uint64_t idx;
        };

        /**
         * PyTorch-compatible NaN rule: NaN beats every non-NaN value; the first NaN wins against later NaNs (strict comparison).
         * @brief Is the candidate scalar key better than the incumbent?
         * @param cand Candidate key.
         * @param best Incumbent key.
         * @return T/F - the candidate should replace the incumbent.
         */
        __device__ __forceinline__ bool dev_key_better(const float64_t cand, const float64_t best) {
            if (isnan(cand)) {
                return !isnan(best);
            }
            return !isnan(best) && cand > best;
        }

        /**
         * Floats use the NaN-wins rule.
         * @brief Does the candidate strictly beat the incumbent? (fp64 key)
         * @param c Candidate key.
         * @param b Incumbent key.
         * @return T/F - the candidate is strictly better.
         */
        __device__ __forceinline__ bool vi_better(const float64_t c, const float64_t b) { return dev_key_better(c, b); }
        /**
         * @brief Does the candidate strictly beat the incumbent? (int64 key)
         * @param c Candidate key.
         * @param b Incumbent key.
         * @return T/F - the candidate is strictly better.
         */
        __device__ __forceinline__ bool vi_better(const int64_t c, const int64_t b) { return c > b; }
        /**
         * @brief Does the candidate strictly beat the incumbent? (uint64 key)
         * @param c Candidate key.
         * @param b Incumbent key.
         * @return T/F - the candidate is strictly better.
         */
        __device__ __forceinline__ bool vi_better(const uint64_t c, const uint64_t b) { return c > b; }
        /**
         * Compares lexicographically (lo, then hi), NaN-wins per lane.
         * @brief Does the candidate strictly beat the incumbent? (packed float pair key)
         * @param c Candidate key.
         * @param b Incumbent key.
         * @return T/F - the candidate is strictly better.
         */
        __device__ __forceinline__ bool vi_better(const float2 c, const float2 b) {
            if (dev_key_better(c.x, b.x)) {
                return true;
            }
            if (dev_key_better(b.x, c.x)) {
                return false;
            }
            return dev_key_better(c.y, b.y);
        }
        /**
         * Compares lexicographically (lo, then hi).
         * @brief Does the candidate strictly beat the incumbent? (int4x2 lane pair key)
         * @param c Candidate key.
         * @param b Incumbent key.
         * @return T/F - the candidate is strictly better.
         */
        __device__ __forceinline__ bool vi_better(const int2 c, const int2 b) {
            if (c.x != b.x) {
                return c.x > b.x;
            }
            return c.y > b.y;
        }

        /**
         * @brief Key equality without float ==: neither strictly beats the other.
         * @param a Left key.
         * @param b Right key.
         * @return T/F - the keys tie.
         */
        template<typename K>
        __device__ __forceinline__ bool vi_eq(const K& a, const K& b) {
            return !vi_better(a, b) && !vi_better(b, a);
        }

        /**
         * Equal keys keep the lower flat index (matches the CPU chunk-order merge).
         * @brief Combine two (key, idx) partials, the better key wins.
         * @param a Left partial.
         * @param b Right partial.
         * @return The winning partial.
         */
        template<typename K>
        __device__ __forceinline__ ValIdx<K> vi_combine(const ValIdx<K>& a, const ValIdx<K>& b) {
            if (b.idx == UINT64_MAX) {
                return a;
            }
            if (a.idx == UINT64_MAX) {
                return b;
            }
            if (vi_better(b.key, a.key) || (vi_eq(a.key, b.key) && b.idx < a.idx)) {
                return b;
            }
            return a;
        }

        /**
         * Floats widen to fp64 (exact), packed dtypes to a lane pair, ints to int64/uint64.
         * @brief Element to argmax key (fp32).
         * @param v Element.
         * @return Argmax key.
         */
        __device__ __forceinline__ float64_t to_key(const float32_t v) { return v; }
        /**
         * @brief Element to argmax key (fp64).
         * @param v Element.
         * @return Argmax key.
         */
        __device__ __forceinline__ float64_t to_key(const float64_t v) { return v; }
        /**
         * @brief Element to argmax key (fp16, widened exactly).
         * @param v Element.
         * @return Argmax key.
         */
        __device__ __forceinline__ float64_t to_key(const float16_t v) { return __half2float(v); }
        /**
         * @brief Element to argmax key (bf16, widened exactly).
         * @param v Element.
         * @return Argmax key.
         */
        __device__ __forceinline__ float64_t to_key(const bfloat16_t v) { return __bfloat162float(v); }
        /**
         * @brief Element to argmax key (fp16x2 -> lane pair).
         * @param v Element.
         * @return Argmax key {lo, hi}.
         */
        __device__ __forceinline__ float2 to_key(const float16x2_t& v) {
            return make_float2(__half2float(__low2half(v)), __half2float(__high2half(v)));
        }
        /**
         * @brief Element to argmax key (bf16x2 -> lane pair).
         * @param v Element.
         * @return Argmax key {lo, hi}.
         */
        __device__ __forceinline__ float2 to_key(const bfloat16x2_t& v) {
            return make_float2(__bfloat162float(__low2bfloat16(v)), __bfloat162float(__high2bfloat16(v)));
        }
        /**
         * @brief Element to argmax key (int4x2 -> signed lane pair).
         * @param v Element.
         * @return Argmax key {lo, hi}.
         */
        __device__ __forceinline__ int2 to_key(const int4x2_t v) {
            return make_int2(dev_i4_lo(v), dev_i4_hi(v));
        }
        /**
         * @brief Element to argmax key (signed integers -> int64).
         * @param v Element.
         * @return Argmax key.
         */
        template<typename T> requires (std::is_integral_v<T> && std::is_signed_v<T>)
        __device__ __forceinline__ int64_t to_key(const T v) { return static_cast<int64_t>(v); }
        /**
         * @brief Element to argmax key (unsigned integers -> uint64).
         * @param v Element.
         * @return Argmax key.
         */
        template<typename T> requires (std::is_integral_v<T> && std::is_unsigned_v<T>)
        __device__ __forceinline__ uint64_t to_key(const T v) { return static_cast<uint64_t>(v); }

        /**
         * Paired with the empty sentinel, never compared against.
         * @brief Zero key of an argmax key type.
         * @return Zero-initialized key.
         */
        template<typename K>
        __device__ __forceinline__ K key_zero() {
            if constexpr (std::is_same_v<K, float2>) {
                return make_float2(0.0f, 0.0f);
            } else if constexpr (std::is_same_v<K, int2>) {
                return make_int2(0, 0);
            } else {
                return K{0};
            }
        }

        /**
         * Grid-stride per-thread best, shared-memory tree combine per block, one ValIdx partial per block.
         * @note Stage 1 of the two-stage argmax.
         * @brief Argmax partial kernel (all dtypes).
         * @param data Input elements.
         * @param partials One ValIdx per block, written by thread 0.
         * @param numel Element count.
         */
        template<typename T, typename K>
        __global__ void argmax_partial_kernel(const T* __restrict__ data, ValIdx<K>* __restrict__ partials, const size_t numel) {
            __shared__ ValIdx<K> shared[BT_GPU_THREADS];
            const auto tid = threadIdx.x;
            const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t stride = blockDim.x * gridDim.x;
            auto acc = ValIdx<K>{key_zero<K>(), UINT64_MAX};
            for (size_t i = gid; i < numel; i += stride) {
                acc = vi_combine(acc, ValIdx<K>{to_key(data[i]), i});
            }
            shared[tid] = acc;
            __syncthreads();
            for (auto s = BT_GPU_THREADS / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared[tid] = vi_combine(shared[tid], shared[tid + s]);
                }
                __syncthreads();
            }
            if (tid == 0) {
                partials[blockIdx.x] = shared[0];
            }
        }

        /**
         * Single block combines the per-block partials and writes the winning flat index to result.
         * @note Stage 2 of the two-stage argmax.
         * @brief Argmax finalize kernel.
         * @param partials Per-block ValIdx from argmax_partial_kernel.
         * @param count Number of partials.
         * @param result Device output, flat index of the winner.
         */
        template<typename K>
        __global__ void argmax_final_kernel(const ValIdx<K>* __restrict__ partials, const size_t count, uint64_t* __restrict__ result) {
            __shared__ ValIdx<K> shared[BT_GPU_THREADS];
            const auto tid = threadIdx.x;
            auto acc = ValIdx<K>{key_zero<K>(), UINT64_MAX};
            for (auto i = tid; i < count; i += BT_GPU_THREADS) {
                acc = vi_combine(acc, partials[i]);
            }
            shared[tid] = acc;
            __syncthreads();
            for (auto s = BT_GPU_THREADS / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared[tid] = vi_combine(shared[tid], shared[tid + s]);
                }
                __syncthreads();
            }
            if (tid == 0) {
                result[0] = shared[0].idx;
            }
        }

        /**
         * The D2H memcpy syncs the stream, so *out_host is final before the worker returns.
         * numel == 0 never reaches here (Tensor::argmax throws first).
         * @brief Host-side two-stage argmax.
         * @param data Input elements.
         * @param numel Element count.
         * @param out_host Host output, flat index of the winner.
         */
        template<typename T, typename K>
        __host__ void run_argmax(const T* data, const size_t numel, uint64_t* out_host) {
            const auto blocks = grid_for(numel);
            ValIdx<K>* partials = nullptr;
            uint64_t* dev_result = nullptr;
            CUDA_CHECK_ERROR(cudaMalloc(&partials, blocks * sizeof(ValIdx<K>)));
            CUDA_CHECK_ERROR(cudaMalloc(&dev_result, sizeof(uint64_t)));
            argmax_partial_kernel<T, K><<<blocks, BT_GPU_THREADS>>>(data, partials, numel);
            argmax_final_kernel<K><<<1, BT_GPU_THREADS>>>(partials, blocks, dev_result);
            CUDA_CHECK_ERROR(cudaGetLastError());
            CUDA_CHECK_ERROR(cudaMemcpy(out_host, dev_result, sizeof(uint64_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK_ERROR(cudaFree(partials));
            CUDA_CHECK_ERROR(cudaFree(dev_result));
        }

        // --------------------------- axis argmax kernel ---------------------------

        /**
         * One thread per output position (grid-stride); each scans the reduced axis and writes
         * the winning axis index. vi_combine keeps the lower index on ties, matching the CPU
         * first-max-wins rule; NaN-wins rides on vi_better.
         * @brief Axis argmax kernel (all dtypes).
         * @param data Input elements.
         * @param out Output int64 indices, out_numel elements.
         * @param out_numel Output element count.
         * @param axis_len Length of the reduced axis.
         * @param inner Stride of the reduced axis (product of the trailing dims).
         */
        template<typename T, typename K>
        __global__ void argmax_axis_kernel(const T* __restrict__ data, int64_t* __restrict__ out,
                                           const size_t out_numel, const size_t axis_len, const size_t inner) {
            const size_t stride = blockDim.x * gridDim.x;
            for (size_t o = threadIdx.x + blockIdx.x * blockDim.x; o < out_numel; o += stride) {
                const auto base = o / inner * (axis_len * inner) + o % inner;
                auto best = ValIdx<K>{key_zero<K>(), UINT64_MAX};
                for (auto i = size_t{0}; i < axis_len; ++i) {
                    best = vi_combine(best, ValIdx<K>{to_key(data[base + i * inner]), i});
                }
                out[o] = static_cast<int64_t>(best.idx);
            }
        }

        /**
         * Skips the launch for empty outputs; the kernel is asynchronous, the caller syncs.
         * @brief Host-side axis argmax launch.
         * @param data Input elements.
         * @param out Output int64 indices.
         * @param out_numel Output element count.
         * @param axis_len Length of the reduced axis.
         * @param inner Stride of the reduced axis.
         */
        template<typename T, typename K>
        __host__ void run_argmax_axis(const T* data, int64_t* out, const size_t out_numel, const size_t axis_len, const size_t inner) {
            if (out_numel > 0) {
                argmax_axis_kernel<T, K><<<grid_for(out_numel), BT_GPU_THREADS>>>(data, out, out_numel, axis_len, inner);
            }
        }
    }

    /**
     * Blocks until the reduction lands, legal as this runs on the caller's thread, never in a pool worker.
     * @brief GPU flat index of the largest element.
     * @param t Input tensor.
     * @param result Flat index of the winner.
     */
    __host__ void argmax_gpu(const Tensor& t, uint64_t& result) {
        if (t.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(t.device));
        }
        const auto* const t_data = t.data;
        const auto t_offset = t.offset;
        const auto dtype = t.dtype;
        const auto numel = t.numel;
        const auto res = std::make_shared<uint64_t>(0);
        /**
         * Raw pointers are captured by value: the future lands on the tensors, so buffer teardown
         * always drains this worker first. Dispatches on dtype to the typed host launcher.
         * @brief Pool worker for the GPU argmax reduction.
         * @note Lambda subcall. Self-execution prohibited.
         */
        const auto worker = [=] {
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: run_argmax<float32_t, float64_t>(static_cast<const float32_t*>(t_data) + t_offset, numel, res.get()); break;
                case DType::Float64: run_argmax<float64_t, float64_t>(static_cast<const float64_t*>(t_data) + t_offset, numel, res.get()); break;
                case DType::Float16: run_argmax<float16_t, float64_t>(static_cast<const float16_t*>(t_data) + t_offset, numel, res.get()); break;
                case DType::BFloat16: run_argmax<bfloat16_t, float64_t>(static_cast<const bfloat16_t*>(t_data) + t_offset, numel, res.get()); break;
                case DType::Float16x2: run_argmax<float16x2_t, float2>(static_cast<const float16x2_t*>(t_data) + t_offset, numel, res.get()); break;
                case DType::BFloat16x2: run_argmax<bfloat16x2_t, float2>(static_cast<const bfloat16x2_t*>(t_data) + t_offset, numel, res.get()); break;
                case DType::Int8: run_argmax<int8_t, int64_t>(static_cast<const int8_t*>(t_data) + t_offset, numel, res.get()); break;
                case DType::UInt8: run_argmax<uint8_t, uint64_t>(static_cast<const uint8_t*>(t_data) + t_offset, numel, res.get()); break;
                case DType::Int16: run_argmax<int16_t, int64_t>(static_cast<const int16_t*>(t_data) + t_offset, numel, res.get()); break;
                case DType::UInt16: run_argmax<uint16_t, uint64_t>(static_cast<const uint16_t*>(t_data) + t_offset, numel, res.get()); break;
                case DType::Int32: run_argmax<int32_t, int64_t>(static_cast<const int32_t*>(t_data) + t_offset, numel, res.get()); break;
                case DType::UInt32: run_argmax<uint32_t, uint64_t>(static_cast<const uint32_t*>(t_data) + t_offset, numel, res.get()); break;
                case DType::Int64: run_argmax<int64_t, int64_t>(static_cast<const int64_t*>(t_data) + t_offset, numel, res.get()); break;
                case DType::UInt64: run_argmax<uint64_t, uint64_t>(static_cast<const uint64_t*>(t_data) + t_offset, numel, res.get()); break;
                case DType::Int4x2: run_argmax<int4x2_t, int2>(static_cast<const int4x2_t*>(t_data) + t_offset, numel, res.get()); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
            // run_argmax's D2H memcpy already synced the stream.
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        t.give_future(future);
        // argmax returns the index by value: block until the reduction lands.
        // Legal here — this runs on the caller's thread, never in a pool worker.
        future.get();
        result = *res;
    }

    /**
     * Queues one pool worker that dispatches on dtype and launches the axis argmax kernel. The
     * future lands on both tensors so teardown drains it.
     * @brief GPU argmax along one axis, written as int64 indices into the output tensor.
     * @param t Input tensor, GPU resident, contiguous.
     * @param out Output tensor (input shape minus the axis), Int64 dtype, GPU resident.
     * @param axis Axis to reduce.
     * @note Throws DirectFunctionCallOnWrongDataSource when any tensor is not on the GPU.
     */
    __host__ void argmax_axis_gpu(const Tensor& t, const Tensor& out, const size_t axis) {
        if (t.device != Device::GPU || out.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(t.device));
        }
        const auto* const t_data = t.data;
        auto* const out_data = out.data;
        const auto t_offset = t.offset;
        const auto dtype = t.dtype;
        const auto axis_len = t.shape[axis];
        const auto inner = t.dims[axis];
        const auto out_numel = out.numel;
        /**
         * Raw pointers are captured by value: the future lands on the tensors, so buffer teardown
         * always drains this worker first. Dispatches on dtype to the typed host launcher.
         * @brief Pool worker for the GPU axis argmax.
         * @note Lambda subcall. Self-execution prohibited.
         */
        const auto worker = [=] {
            auto* const out_ptr = static_cast<int64_t*>(out_data);
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: run_argmax_axis<float32_t, float64_t>(static_cast<const float32_t*>(t_data) + t_offset, out_ptr, out_numel, axis_len, inner); break;
                case DType::Float64: run_argmax_axis<float64_t, float64_t>(static_cast<const float64_t*>(t_data) + t_offset, out_ptr, out_numel, axis_len, inner); break;
                case DType::Float16: run_argmax_axis<float16_t, float64_t>(static_cast<const float16_t*>(t_data) + t_offset, out_ptr, out_numel, axis_len, inner); break;
                case DType::BFloat16: run_argmax_axis<bfloat16_t, float64_t>(static_cast<const bfloat16_t*>(t_data) + t_offset, out_ptr, out_numel, axis_len, inner); break;
                case DType::Float16x2: run_argmax_axis<float16x2_t, float2>(static_cast<const float16x2_t*>(t_data) + t_offset, out_ptr, out_numel, axis_len, inner); break;
                case DType::BFloat16x2: run_argmax_axis<bfloat16x2_t, float2>(static_cast<const bfloat16x2_t*>(t_data) + t_offset, out_ptr, out_numel, axis_len, inner); break;
                case DType::Int8: run_argmax_axis<int8_t, int64_t>(static_cast<const int8_t*>(t_data) + t_offset, out_ptr, out_numel, axis_len, inner); break;
                case DType::UInt8: run_argmax_axis<uint8_t, uint64_t>(static_cast<const uint8_t*>(t_data) + t_offset, out_ptr, out_numel, axis_len, inner); break;
                case DType::Int16: run_argmax_axis<int16_t, int64_t>(static_cast<const int16_t*>(t_data) + t_offset, out_ptr, out_numel, axis_len, inner); break;
                case DType::UInt16: run_argmax_axis<uint16_t, uint64_t>(static_cast<const uint16_t*>(t_data) + t_offset, out_ptr, out_numel, axis_len, inner); break;
                case DType::Int32: run_argmax_axis<int32_t, int64_t>(static_cast<const int32_t*>(t_data) + t_offset, out_ptr, out_numel, axis_len, inner); break;
                case DType::UInt32: run_argmax_axis<uint32_t, uint64_t>(static_cast<const uint32_t*>(t_data) + t_offset, out_ptr, out_numel, axis_len, inner); break;
                case DType::Int64: run_argmax_axis<int64_t, int64_t>(static_cast<const int64_t*>(t_data) + t_offset, out_ptr, out_numel, axis_len, inner); break;
                case DType::UInt64: run_argmax_axis<uint64_t, uint64_t>(static_cast<const uint64_t*>(t_data) + t_offset, out_ptr, out_numel, axis_len, inner); break;
                case DType::Int4x2: run_argmax_axis<int4x2_t, int2>(static_cast<const int4x2_t*>(t_data) + t_offset, out_ptr, out_numel, axis_len, inner); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
            CUDA_CHECK_ERROR(cudaStreamSynchronize(nullptr));
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        t.give_future(future);
        out.give_future(future);
    }
}
#endif
