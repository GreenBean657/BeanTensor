/**
 * @file BasicOpsCUDA.cu
 * @author GreenBean
 * @brief CUDA backend for basic tensor ops: sgd_update, sum, argmax.
 * Divergent twin of Math/BasicOps/BasicOpsHIP.cpp — the two backends are
 * written separately per project rule (no HIP-only unification), but must stay
 * behaviorally identical: same fp32/fp64 compute widths, same RNE narrowing,
 * same NaN-wins argmax rule, same uint64 wrap integer sums.
 * @version 0.1
 * @date 2026-10-18
 */
#if defined(USE_CUDA)
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "Math/BasicOps/BasicOps.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    namespace {
        /**
         * @brief Threads per block for every kernel here (also the shared-memory tree width).
         */
        constexpr size_t BT_GPU_THREADS = 256;
        /**
         * @brief Upper bound on the grid size; grid-stride loops cover the remainder.
         */
        constexpr size_t BT_GPU_MAX_BLOCKS = 1024;

        /**
         * Clamped to [1, BT_GPU_MAX_BLOCKS].
         * @brief Block count for a grid-stride kernel.
         * @param numel Element count to cover.
         * @return Number of blocks to launch.
         */
        [[nodiscard]] __host__ size_t grid_for(const size_t numel) {
            const auto blocks = (numel + BT_GPU_THREADS - 1) / BT_GPU_THREADS;
            return blocks < 1 ? 1 : (blocks > BT_GPU_MAX_BLOCKS ? BT_GPU_MAX_BLOCKS : blocks);
        }

        /**
         * FP32-subnormal inputs flush to signed zero, everything else rounds RNE (mirrors the CPU unaccel pathway).
         * @brief FP32 -> BF16 narrowing with CPU-matching flush.
         * @param f FP32 value.
         * @return The value narrowed to bf16.
         */
        __device__ __forceinline__ bfloat16_t dev_narrow_bf16(const float32_t f) {
            const auto bits = __builtin_bit_cast(uint32_t, f);
            if ((bits & 0x7F800000u) == 0u) {
                return __float2bfloat16_rn(__builtin_bit_cast(float32_t, bits & 0x80000000u));
            }
            return __float2bfloat16_rn(f);
        }

        // --------------------------- sgd_update kernel ---------------------------

        /**
         * @brief Signed low lane of an int4x2 byte (bits[3:0], sign-extended).
         * @param v Packed int4x2 byte.
         * @return Lane value in [-8, 7].
         */
        __device__ __forceinline__ int8_t dev_i4_lo(const int4x2_t v) {
            const auto n = static_cast<int8_t>(__builtin_bit_cast(uint8_t, v) & 0x0Fu);
            return n > 7 ? static_cast<int8_t>(n - 16) : n;
        }

        /**
         * @brief Signed high lane of an int4x2 byte (bits[7:4], sign-extended).
         * @param v Packed int4x2 byte.
         * @return Lane value in [-8, 7].
         */
        __device__ __forceinline__ int8_t dev_i4_hi(const int4x2_t v) {
            const auto n = static_cast<int8_t>(__builtin_bit_cast(uint8_t, v) >> 4);
            return n > 7 ? static_cast<int8_t>(n - 16) : n;
        }

        /**
         * Each lane keeps only its low 4 bits, so out-of-range values wrap mod 16.
         * @brief Pack two signed lanes into an int4x2 byte.
         * @param lo Low lane value.
         * @param hi High lane value.
         * @return The packed int4x2 byte.
         */
        __device__ __forceinline__ int4x2_t dev_i4_pack(const int8_t lo, const int8_t hi) {
            const auto lo_bits = static_cast<uint8_t>(lo) & 0x0Fu;
            const auto hi_bits = static_cast<uint8_t>(hi) & 0x0Fu;
            return __builtin_bit_cast(int4x2_t, static_cast<uint8_t>(lo_bits | static_cast<uint8_t>(hi_bits << 4)));
        }

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
                    w[i] = __fma_rn(-lr, g[i], w[i]);
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
                    const auto lane_step = [&](const int8_t w_lane, const int8_t g_lane) {
                        const auto upd = __fma_rn(-lr, static_cast<float64_t>(g_lane), static_cast<float64_t>(w_lane));
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
                    const auto upd = __fma_rn(-lr, static_cast<float64_t>(g[i]), static_cast<float64_t>(w[i]));
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

        // --------------------------- sum kernels ---------------------------

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
        __device__ __forceinline__ float2 widen_acc(const float16x2_t v) {
            return make_float2(__half2float(__low2half(v)), __half2float(__high2half(v)));
        }
        /**
         * @brief Widen one packed element to its accumulator lanes (bf16x2 -> float2).
         * @param v Packed element.
         * @return Accumulator pair {lo, hi}.
         */
        __device__ __forceinline__ float2 widen_acc(const bfloat16x2_t v) {
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
            return make_uint2(static_cast<uint32_t>(bits & 0x0Fu), static_cast<uint32_t>(bits >> 4));
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
            Acc acc;
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
            Acc acc;
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

        // --------------------------- argmax kernels ---------------------------

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
            if (dev_key_better(static_cast<float64_t>(c.x), static_cast<float64_t>(b.x))) {
                return true;
            }
            if (dev_key_better(static_cast<float64_t>(b.x), static_cast<float64_t>(c.x))) {
                return false;
            }
            return dev_key_better(static_cast<float64_t>(c.y), static_cast<float64_t>(b.y));
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
        __device__ __forceinline__ float64_t to_key(const float32_t v) { return static_cast<float64_t>(v); }
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
        __device__ __forceinline__ float64_t to_key(const float16_t v) { return static_cast<float64_t>(__half2float(v)); }
        /**
         * @brief Element to argmax key (bf16, widened exactly).
         * @param v Element.
         * @return Argmax key.
         */
        __device__ __forceinline__ float64_t to_key(const bfloat16_t v) { return static_cast<float64_t>(__bfloat162float(v)); }
        /**
         * @brief Element to argmax key (fp16x2 -> lane pair).
         * @param v Element.
         * @return Argmax key {lo, hi}.
         */
        __device__ __forceinline__ float2 to_key(const float16x2_t v) {
            return make_float2(__half2float(__low2half(v)), __half2float(__high2half(v)));
        }
        /**
         * @brief Element to argmax key (bf16x2 -> lane pair).
         * @param v Element.
         * @return Argmax key {lo, hi}.
         */
        __device__ __forceinline__ float2 to_key(const bfloat16x2_t v) {
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
    }

    /**
     * Runs on the CPU thread pool at high priority; the future lands on both tensors so teardown drains the worker.
     * @brief GPU in-place SGD step, w -= lr * grad.
     * @param w Weights, updated in place.
     * @param grad Gradients, same shape and dtype as w.
     * @param lr Learning rate.
     */
    __host__ void sgd_update_gpu(Tensor& w, const Tensor& grad, const float64_t lr) {
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
            CUDA_CHECK_ERROR(cudaStreamSynchronize(nullptr));
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        w.give_future(future);
        grad.give_future(future);
    }

    /**
     * Runs on the CPU thread pool at high priority; the future lands on both tensors.
     * @brief GPU full reduction, sums every element into the scalar output tensor.
     * @param t Input tensor.
     * @param out Single element output tensor, same dtype as t.
     */
    __host__ void sum_gpu(const Tensor& t, Tensor& out) {
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
        auto res = std::make_shared<uint64_t>(0);
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
}
#endif
