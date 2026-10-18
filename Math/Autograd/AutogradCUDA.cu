/**
 * @file AutogradCUDA.cu
 * @author GreenBean
 * @brief CUDA backend for the autograd backward ops: grad accumulation,
 * last-two-dims transpose (GEMM_G operand materialization), relu_g,
 * bias_add_g, softmax_ce_g. GEMM_G itself is orchestrated in AutogradCPU.cpp
 * (transpose + the existing matmul_gpu path + grad accumulation).
 * Divergent twin of Math/Autograd/AutogradHIP.cpp - the two backends are
 * written separately per project rule (no HIP-only unification), but must stay
 * behaviorally identical: same fp32/fp64 compute widths, same RNE narrowing,
 * same NaN handling, same wrap integer adds.
 * @version 0.1
 * @date 2026-10-18
 */
#if defined(USE_CUDA)
#include <cstdint>
#include <limits>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "Math/Autograd/Autograd.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    namespace {
        /**
         * Also the shared-memory reduction width of the softmax_ce_g kernel (power of two required).
         * @brief Threads per block for every autograd kernel.
         */
        constexpr size_t BT_GPU_THREADS = 256;
        /**
         * Grid-stride loops cover any remainder, so large inputs never need more blocks than this.
         * @brief Upper bound on the block count of one grid-stride launch.
         */
        constexpr size_t BT_GPU_MAX_BLOCKS = 1024;
        /**
         * Bounds the partials scratch of bias_add_g to at most this many row chunks per column.
         * @brief Upper bound on the row-chunk count of the bias_add_g column sum.
         */
        constexpr size_t BT_GPU_COLSUM_MAX_CHUNKS = 256;
        /**
         * The chunk count is ceil(rows / this), clamped to [1, BT_GPU_COLSUM_MAX_CHUNKS].
         * @brief Target rows summed serially by one thread in the bias_add_g stage 1 kernel.
         */
        constexpr size_t BT_GPU_COLSUM_ROWS_PER_CHUNK = 32;

        /**
         * Clamped to [1, BT_GPU_MAX_BLOCKS]; the grid-stride loop covers any remainder.
         * @brief Block count for a grid-stride kernel over numel elements.
         * @param numel Element count the kernel walks.
         * @return Block count of BT_GPU_THREADS wide blocks.
         */
        [[nodiscard]] __host__ size_t grid_for(const size_t numel) {
            const auto blocks = (numel + BT_GPU_THREADS - 1) / BT_GPU_THREADS;
            return blocks < 1 ? 1 : (blocks > BT_GPU_MAX_BLOCKS ? BT_GPU_MAX_BLOCKS : blocks);
        }

        /**
         * Narrow FP32 to BF16 mirroring the CPU unaccel pathway: FP32-subnormal
         * inputs flush to signed zero, everything else rounds RNE.
         * @brief FP32 -> BF16 narrowing with CPU-matching flush.
         * @param f FP32 value to narrow.
         * @return BF16 value, RNE, subnormal inputs flushed to signed zero.
         */
        __device__ __forceinline__ bfloat16_t dev_narrow_bf16(const float32_t f) {
            const auto bits = __builtin_bit_cast(uint32_t, f);
            if ((bits & 0x7F800000u) == 0u) {
                return __float2bfloat16_rn(__builtin_bit_cast(float32_t, bits & 0x80000000u));
            }
            return __float2bfloat16_rn(f);
        }

        // --------------------------- int4x2 helpers ---------------------------

        /**
         * Low 4 bits, sign-extended from bit 3.
         * @brief Sign-extended low nibble of a packed int4 pair.
         * @param v Packed int4 pair.
         * @return Low nibble in [-8, 7].
         */
        __device__ __forceinline__ int32_t dev_int4_lo(const int4x2_t v) {
            const auto n = static_cast<int32_t>(static_cast<uint8_t>(v) & 0x0Fu);
            return n >= 8 ? n - 16 : n;
        }

        /**
         * High 4 bits, sign-extended from bit 3.
         * @brief Sign-extended high nibble of a packed int4 pair.
         * @param v Packed int4 pair.
         * @return High nibble in [-8, 7].
         */
        __device__ __forceinline__ int32_t dev_int4_hi(const int4x2_t v) {
            const auto n = static_cast<int32_t>(static_cast<uint8_t>(v) >> 4);
            return n >= 8 ? n - 16 : n;
        }

        /**
         * Only the low 4 bits of each input survive, so out-of-range sums wrap.
         * @brief Pack two signed values into an int4 pair (modular, low 4 bits each).
         * @param lo Value for the low nibble.
         * @param hi Value for the high nibble.
         * @return Packed int4 pair.
         */
        __device__ __forceinline__ int4x2_t dev_pack_int4(const int32_t lo, const int32_t hi) {
            return static_cast<int4x2_t>(static_cast<uint8_t>((lo & 0xF) | ((hi & 0xF) << 4)));
        }

        // --------------------------- grad_acc kernel ---------------------------

        /**
         * Elementwise w[i] += d[i]. Floats single-rounding add in storage width;
         * halfs widen to fp32, add, narrow RNE; packed dtypes per lane; integers
         * wrap through the unsigned twin; int4x2 wraps per nibble. Matches the
         * CPU grad_acc kernels. Launched as grid_for(numel) blocks of BT_GPU_THREADS
         * threads, grid-stride over numel.
         * @brief Gradient accumulation kernel (all dtypes).
         * @param w Accumulator (gradient) data, updated in place, numel elements.
         * @param d Delta data added into w, numel elements.
         * @param numel Element count (storage elements, so packed dtypes count pairs).
         */
        template<typename T>
        __global__ void grad_acc_kernel(T* __restrict__ w, const T* __restrict__ d, const size_t numel) {
            const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t stride = blockDim.x * gridDim.x;
            for (size_t i = tid; i < numel; i += stride) {
                if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                    w[i] = w[i] + d[i];
                } else if constexpr (std::is_same_v<T, float16_t>) {
                    w[i] = __float2half_rn(__half2float(w[i]) + __half2float(d[i]));
                } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                    w[i] = dev_narrow_bf16(__bfloat162float(w[i]) + __bfloat162float(d[i]));
                } else if constexpr (std::is_same_v<T, float16x2_t>) {
                    const auto wv = w[i];
                    const auto dv = d[i];
                    const auto r_lo = __float2half_rn(__half2float(__low2half(wv)) + __half2float(__low2half(dv)));
                    const auto r_hi = __float2half_rn(__half2float(__high2half(wv)) + __half2float(__high2half(dv)));
                    w[i] = __halves2half2(r_lo, r_hi);
                } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                    const auto wv = w[i];
                    const auto dv = d[i];
                    const auto r_lo = dev_narrow_bf16(__bfloat162float(__low2bfloat16(wv)) + __bfloat162float(__low2bfloat16(dv)));
                    const auto r_hi = dev_narrow_bf16(__bfloat162float(__high2bfloat16(wv)) + __bfloat162float(__high2bfloat16(dv)));
                    w[i] = __halves2bfloat162(r_lo, r_hi);
                } else if constexpr (std::is_same_v<T, int4x2_t>) {
                    w[i] = dev_pack_int4(dev_int4_lo(w[i]) + dev_int4_lo(d[i]),
                                         dev_int4_hi(w[i]) + dev_int4_hi(d[i]));
                } else {
                    using U = std::make_unsigned_t<T>;
                    w[i] = static_cast<T>(static_cast<U>(static_cast<U>(w[i]) + static_cast<U>(d[i])));
                }
            }
        }

        /**
         * Skips the launch when numel is 0. The launch is asynchronous; the caller syncs.
         * @brief Host-side grad_acc launch.
         * @param w Accumulator (gradient) data, updated in place.
         * @param d Delta data added into w.
         * @param numel Element count.
         */
        template<typename T>
        __host__ void run_grad_acc(T* w, const T* d, const size_t numel) {
            if (numel > 0) {
                grad_acc_kernel<T><<<grid_for(numel), BT_GPU_THREADS>>>(w, d, numel);
            }
        }

        // --------------------------- relu_g kernel ---------------------------

        /**
         * dx[i] = x[i] > 0 ? dy[i] : 0 - a select, not a multiply: NaN in x
         * fails the comparison and zeroes the lane, a selected dy is stored
         * bit-exact (no rounding). Halfs compare widened; packed per lane;
         * unsigned integers copy dy through; int4x2 per nibble. Launched as
         * grid_for(numel) blocks of BT_GPU_THREADS threads, grid-stride over numel.
         * @brief relu backward kernel (all dtypes).
         * @param x Forward input of the relu, numel elements.
         * @param dy Upstream gradient, numel elements.
         * @param dx Output gradient, numel elements.
         * @param numel Element count (storage elements, so packed dtypes count pairs).
         */
        template<typename T>
        __global__ void relu_g_kernel(const T* __restrict__ x, const T* __restrict__ dy, T* __restrict__ dx, const size_t numel) {
            const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t stride = blockDim.x * gridDim.x;
            for (size_t i = tid; i < numel; i += stride) {
                if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                    dx[i] = x[i] > T{0} ? dy[i] : T{0};
                } else if constexpr (std::is_same_v<T, float16_t>) {
                    dx[i] = __half2float(x[i]) > 0.0f ? dy[i] : __float2half_rn(0.0f);
                } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                    dx[i] = __bfloat162float(x[i]) > 0.0f ? dy[i] : dev_narrow_bf16(0.0f);
                } else if constexpr (std::is_same_v<T, float16x2_t>) {
                    const auto xv = x[i];
                    const auto dv = dy[i];
                    const auto zero = __float2half_rn(0.0f);
                    dx[i] = __halves2half2(__half2float(__low2half(xv)) > 0.0f ? __low2half(dv) : zero,
                                           __half2float(__high2half(xv)) > 0.0f ? __high2half(dv) : zero);
                } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                    const auto xv = x[i];
                    const auto dv = dy[i];
                    const auto zero = dev_narrow_bf16(0.0f);
                    dx[i] = __halves2bfloat162(__bfloat162float(__low2bfloat16(xv)) > 0.0f ? __low2bfloat16(dv) : zero,
                                               __bfloat162float(__high2bfloat16(xv)) > 0.0f ? __high2bfloat16(dv) : zero);
                } else if constexpr (std::is_same_v<T, int4x2_t>) {
                    dx[i] = dev_pack_int4(dev_int4_lo(x[i]) > 0 ? dev_int4_lo(dy[i]) : 0,
                                          dev_int4_hi(x[i]) > 0 ? dev_int4_hi(dy[i]) : 0);
                } else if constexpr (std::is_signed_v<T>) {
                    dx[i] = x[i] > T{0} ? dy[i] : T{0};
                } else {
                    dx[i] = dy[i];
                }
            }
        }

        /**
         * Skips the launch when numel is 0. The launch is asynchronous; the caller syncs.
         * @brief Host-side relu_g launch.
         * @param x Forward input of the relu.
         * @param dy Upstream gradient.
         * @param dx Output gradient.
         * @param numel Element count.
         */
        template<typename T>
        __host__ void run_relu_g(const T* x, const T* dy, T* dx, const size_t numel) {
            if (numel > 0) {
                relu_g_kernel<T><<<grid_for(numel), BT_GPU_THREADS>>>(x, dy, dx, numel);
            }
        }

        // --------------------------- transpose kernel ---------------------------

        /**
         * Dtype-agnostic last-two-dims transpose: element bits are moved as an
         * unsigned word of the dtype's width (E), so no value is interpreted.
         * Launched as grid_for(numel) blocks of BT_GPU_THREADS threads, grid-stride
         * over the flat input index.
         * @brief Batched transpose kernel (out[b, j, i] = in[b, i, j]).
         * @param in Input data, batch slabs of rows x cols, numel elements.
         * @param out Output data, batch slabs of cols x rows, numel elements.
         * @param rows Second-to-last dimension of in.
         * @param cols Last dimension of in.
         * @param numel Total element count (batch * rows * cols).
         */
        template<typename E>
        __global__ void transpose_kernel(const E* __restrict__ in, E* __restrict__ out,
                                         const size_t rows, const size_t cols, const size_t numel) {
            const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t stride = blockDim.x * gridDim.x;
            const auto slab = rows * cols;
            for (size_t idx = tid; idx < numel; idx += stride) {
                const auto batch = idx / slab;
                const auto rem = idx - batch * slab;
                const auto i = rem / cols;
                const auto j = rem - i * cols;
                out[batch * slab + j * rows + i] = in[idx];
            }
        }

        /**
         * Skips the launch when numel is 0. The launch is asynchronous; the caller syncs.
         * @brief Host-side transpose launch (E = unsigned word of the element width).
         * @param in Input data (type-erased).
         * @param out Output data (type-erased).
         * @param rows Second-to-last dimension of in.
         * @param cols Last dimension of in.
         * @param numel Total element count.
         */
        template<typename E>
        __host__ void run_transpose(const void* in, void* out, const size_t rows, const size_t cols, const size_t numel) {
            if (numel > 0) {
                transpose_kernel<E><<<grid_for(numel), BT_GPU_THREADS>>>(static_cast<const E*>(in), static_cast<E*>(out), rows, cols, numel);
            }
        }

        // --------------------------- bias_add_g kernels ---------------------------

        /**
         * Packed dtypes expose their two lanes as consecutive effective columns; int4x2 its two
         * nibbles; integers reinterpret into the unsigned twin (wrap add).
         * @brief Widen dy[r, c_eff] to the column-sum accumulator.
         * @param dy Gradient data, rows x cols storage elements.
         * @param r Row index.
         * @param cols Storage columns per row (packed elements count as one).
         * @param ce Effective column index (2 per storage element for packed dtypes).
         * @return The element widened to Acc.
         */
        template<typename T, typename Acc>
        __device__ __forceinline__ Acc colsum_elem(const T* dy, const size_t r, const size_t cols, const size_t ce) {
            if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                return dy[r * cols + ce];
            } else if constexpr (std::is_same_v<T, float16_t>) {
                return __half2float(dy[r * cols + ce]);
            } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                return __bfloat162float(dy[r * cols + ce]);
            } else if constexpr (std::is_same_v<T, float16x2_t>) {
                const auto v = dy[r * cols + (ce >> 1)];
                return (ce & 1) != 0 ? __half2float(__high2half(v)) : __half2float(__low2half(v));
            } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                const auto v = dy[r * cols + (ce >> 1)];
                return (ce & 1) != 0 ? __bfloat162float(__high2bfloat16(v)) : __bfloat162float(__low2bfloat16(v));
            } else if constexpr (std::is_same_v<T, int4x2_t>) {
                const auto v = dy[r * cols + (ce >> 1)];
                return (ce & 1) != 0 ? dev_int4_hi(v) : dev_int4_lo(v);
            } else {
                return static_cast<Acc>(dy[r * cols + ce]);
            }
        }

        /**
         * Stage 1: one thread per (row chunk, effective column) sums its
         * column over the chunk's rows in order, into partials[chunk, c_eff].
         * Adjacent threads read adjacent columns (coalesced). Launched as
         * grid_for(chunks * cols_eff) blocks of BT_GPU_THREADS threads, grid-stride.
         * @brief bias_add_g column-sum partial kernel (all dtypes).
         * @param dy Gradient data, rows x cols storage elements.
         * @param partials Output partial sums, chunks x cols_eff accumulators.
         * @param rows Total row count.
         * @param cols Storage columns per row.
         * @param cols_eff Effective columns per row (2 * cols for packed dtypes).
         * @param chunks Number of row chunks.
         * @param rows_per_chunk Rows summed per chunk (the last chunk may be shorter).
         */
        template<typename T, typename Acc>
        __global__ void bias_add_g_colsum_kernel(const T* __restrict__ dy, Acc* __restrict__ partials,
                                                 const size_t rows, const size_t cols, const size_t cols_eff,
                                                 const size_t chunks, const size_t rows_per_chunk) {
            const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t stride = blockDim.x * gridDim.x;
            const auto total = chunks * cols_eff;
            for (size_t idx = tid; idx < total; idx += stride) {
                const auto chunk = idx / cols_eff;
                const auto ce = idx - chunk * cols_eff;
                const auto r_begin = chunk * rows_per_chunk;
                const auto r_end = r_begin + rows_per_chunk < rows ? r_begin + rows_per_chunk : rows;
                auto acc = Acc{0};
                for (auto r = r_begin; r < r_end; ++r) {
                    acc = static_cast<Acc>(acc + colsum_elem<T, Acc>(dy, r, cols, ce));
                }
                partials[idx] = acc;
            }
        }

        /**
         * Stage 2: one thread per storage column merges the chunk partials in
         * chunk order and narrows into dbias: floats in the accumulator width
         * (halfs RNE), packed per lane, integers wrap, int4x2 wraps per nibble.
         * Launched as grid_for(cols) blocks of BT_GPU_THREADS threads, grid-stride.
         * @brief bias_add_g merge/store kernel (all dtypes).
         * @param partials Partial sums from the column-sum kernel, chunks x cols_eff.
         * @param dbias Output bias gradient, cols storage elements.
         * @param cols Storage columns per row.
         * @param cols_eff Effective columns per row (2 * cols for packed dtypes).
         * @param chunks Number of row chunks to merge.
         */
        template<typename T, typename Acc>
        __global__ void bias_add_g_store_kernel(const Acc* __restrict__ partials, T* __restrict__ dbias,
                                                const size_t cols, const size_t cols_eff, const size_t chunks) {
            const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t stride = blockDim.x * gridDim.x;
            for (size_t c = tid; c < cols; c += stride) {
                if constexpr (std::is_same_v<T, float16x2_t> || std::is_same_v<T, bfloat16x2_t> || std::is_same_v<T, int4x2_t>) {
                    auto s_lo = Acc{0};
                    auto s_hi = Acc{0};
                    for (size_t k = 0; k < chunks; ++k) {
                        s_lo = s_lo + partials[k * cols_eff + 2 * c];
                        s_hi = s_hi + partials[k * cols_eff + 2 * c + 1];
                    }
                    if constexpr (std::is_same_v<T, float16x2_t>) {
                        dbias[c] = __halves2half2(__float2half_rn(s_lo), __float2half_rn(s_hi));
                    } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                        dbias[c] = __halves2bfloat162(dev_narrow_bf16(s_lo), dev_narrow_bf16(s_hi));
                    } else {
                        dbias[c] = dev_pack_int4(s_lo, s_hi);
                    }
                } else {
                    auto s = Acc{0};
                    for (size_t k = 0; k < chunks; ++k) {
                        s = static_cast<Acc>(s + partials[k * cols_eff + c]);
                    }
                    if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                        dbias[c] = s;
                    } else if constexpr (std::is_same_v<T, float16_t>) {
                        dbias[c] = __float2half_rn(s);
                    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                        dbias[c] = dev_narrow_bf16(s);
                    } else {
                        dbias[c] = static_cast<T>(s);
                    }
                }
            }
        }

        /**
         * The partials scratch is worker-local cudaMalloc/cudaFree; cudaFree syncs, so
         * dbias is complete before the worker returns. Returns without work when cols is 0.
         * The chunk count is ceil(rows / BT_GPU_COLSUM_ROWS_PER_CHUNK) clamped to
         * [1, BT_GPU_COLSUM_MAX_CHUNKS]. Both kernels are launched, then cudaGetLastError is checked.
         * @brief Host-side two-stage bias_add_g.
         * @param dy Gradient data, rows x cols storage elements.
         * @param dbias Output bias gradient, cols storage elements.
         * @param rows Total row count.
         * @param cols Storage columns per row.
         * @param cols_eff Effective columns per row (2 * cols for packed dtypes).
         */
        template<typename T, typename Acc>
        __host__ void run_bias_add_g(const T* dy, T* dbias, const size_t rows, const size_t cols, const size_t cols_eff) {
            if (cols == 0) {
                return;
            }
            auto chunks = (rows + BT_GPU_COLSUM_ROWS_PER_CHUNK - 1) / BT_GPU_COLSUM_ROWS_PER_CHUNK;
            chunks = chunks < 1 ? 1 : (chunks > BT_GPU_COLSUM_MAX_CHUNKS ? BT_GPU_COLSUM_MAX_CHUNKS : chunks);
            const auto rows_per_chunk = (rows + chunks - 1) / chunks;
            Acc* partials = nullptr;
            CUDA_CHECK_ERROR(cudaMalloc(&partials, chunks * cols_eff * sizeof(Acc)));
            bias_add_g_colsum_kernel<T, Acc><<<grid_for(chunks * cols_eff), BT_GPU_THREADS>>>(dy, partials, rows, cols, cols_eff, chunks, rows_per_chunk);
            bias_add_g_store_kernel<T, Acc><<<grid_for(cols), BT_GPU_THREADS>>>(partials, dbias, cols, cols_eff, chunks);
            CUDA_CHECK_ERROR(cudaGetLastError());
            CUDA_CHECK_ERROR(cudaFree(partials));
        }

        // --------------------------- softmax_ce_g kernels ---------------------------

        /**
         * Lowest finite value of an accumulator type (nvcc rejects numeric_limits in device code).
         * @brief Lowest finite value of an accumulator type.
         * @return -FLT_MAX for fp32 accumulators, -DBL_MAX otherwise.
         */
        template<typename Acc>
        __device__ __forceinline__ Acc acc_lowest() {
            if constexpr (std::is_same_v<Acc, float32_t>) {
                return -3.4028234663852886e+38f;
            } else {
                return -1.7976931348623157e+308;
            }
        }

        /**
         * @brief exp() in the accumulator width (expf for fp32).
         * @param v Exponent.
         * @return expf(v).
         */
        __device__ __forceinline__ float32_t acc_exp(const float32_t v) { return expf(v); }
        /**
         * @brief exp() in the accumulator width (exp for fp64).
         * @param v Exponent.
         * @return exp(v).
         */
        __device__ __forceinline__ float64_t acc_exp(const float64_t v) { return exp(v); }

        /**
         * fp16/bf16 and packed x2 (lanes unpack as consecutive classes) widen to fp32,
         * fp64 and the integer dtypes (int4x2 unpacks nibbles as consecutive classes) to fp64.
         * @brief Widen class j of a logits row to the accumulator.
         * @param row Logits row base pointer (storage elements).
         * @param j Effective class index (2 per storage element for packed dtypes).
         * @return Class j widened to Acc.
         */
        template<typename T, typename Acc>
        __device__ __forceinline__ Acc sce_widen(const T* row, const size_t j) {
            if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                return row[j];
            } else if constexpr (std::is_same_v<T, float16_t>) {
                return __half2float(row[j]);
            } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                return __bfloat162float(row[j]);
            } else if constexpr (std::is_same_v<T, float16x2_t>) {
                const auto v = row[j >> 1];
                return (j & 1) != 0 ? __half2float(__high2half(v)) : __half2float(__low2half(v));
            } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                const auto v = row[j >> 1];
                return (j & 1) != 0 ? __bfloat162float(__high2bfloat16(v)) : __bfloat162float(__low2bfloat16(v));
            } else if constexpr (std::is_same_v<T, int4x2_t>) {
                return static_cast<float64_t>((j & 1) != 0 ? dev_int4_hi(row[j >> 1]) : dev_int4_lo(row[j >> 1]));
            } else {
                return static_cast<float64_t>(row[j]);
            }
        }

        /**
         * Int4x2 targets use the low nibble (matches the packed->unpacked first-lane conversion contract).
         * @brief Read target class index r.
         * @param targets Target data, one class index per row.
         * @param r Row index.
         * @return Class index as int64_t (may be out of range; the caller validates).
         */
        template<typename I>
        __device__ __forceinline__ int64_t sce_target(const I* targets, const size_t r) {
            if constexpr (std::is_same_v<I, int4x2_t>) {
                return dev_int4_lo(targets[r]);
            } else {
                return static_cast<int64_t>(targets[r]);
            }
        }

        /**
         * Packed dtypes read the low lane (forward duplicates the mean into
         * both lanes); int4x2 reads the low nibble.
         * @brief Widen the scalar dloss gradient element to the accumulator.
         * @param g Pointer to the scalar upstream gradient (element 0 is read).
         * @return The upstream gradient widened to Acc.
         */
        template<typename T, typename Acc>
        __device__ __forceinline__ Acc sce_dloss(const T* g) {
            if constexpr (std::is_same_v<T, float16x2_t>) {
                return __half2float(__low2half(g[0]));
            } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                return __bfloat162float(__low2bfloat16(g[0]));
            } else if constexpr (std::is_same_v<T, int4x2_t>) {
                return static_cast<float64_t>(dev_int4_lo(g[0]));
            } else {
                return sce_widen<T, Acc>(g, 0);
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
        __device__ __forceinline__ float64_t sce_clamp(float64_t v, const float64_t lo, const float64_t hi) {
            if (isnan(v)) {
                v = 0.0;
            }
            if (v <= lo) {
                v = lo;
            } else if (v >= hi) {
                v = hi;
            }
            return v;
        }

        /**
         * Narrow one class gradient into a non-packed logits element: floats
         * RNE (bf16 with the CPU-matching flush), integers clamp/truncate.
         * @brief Store one softmax_ce_g gradient element.
         * @param v Gradient value in the accumulator width.
         * @param lo Lowest integer value of T (ignored for float T).
         * @param hi Highest integer value of T (ignored for float T).
         * @return v converted to T.
         */
        template<typename T, typename Acc>
        __device__ __forceinline__ T sce_narrow(const Acc v, const float64_t lo, const float64_t hi) {
            if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                return v;
            } else if constexpr (std::is_same_v<T, float16_t>) {
                return __float2half_rn(v);
            } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                return dev_narrow_bf16(v);
            } else {
                return static_cast<T>(sce_clamp(static_cast<float64_t>(v), lo, hi));
            }
        }

        /**
         * Row softmax_ce backward, one block per row: block-reduce the widened
         * max and sum(exp(x - m)) exactly like the forward kernel, then every
         * thread writes dx_j = (exp(x_j - m) / s - 1{j == target}) * dloss / rows
         * for its classes. Packed dtypes write both lanes of an element per
         * thread; int4x2 clamps each nibble to [-8, 7]. A NaN logit poisons s
         * and thus the whole row. An out-of-range target sets *err (the host
         * throws IndexOutOfRange after sync - device code cannot throw).
         * Launched as one block of BT_GPU_THREADS threads per row (grid = rows).
         * @brief softmax_ce backward per-row kernel (all dtypes).
         * @param logits Forward logits, rows x cols_store storage elements.
         * @param targets Target class index per row, rows elements.
         * @param dloss Scalar upstream gradient (one element).
         * @param dlogits Output gradient, rows x cols_store storage elements.
         * @param err Device error flag, set to 1 when a target is out of range.
         * @param rows Row (batch) count.
         * @param cols_store Storage columns per row (packed elements count as one).
         * @param cols_eff Class count per row (2 * cols_store for packed dtypes).
         * @param lo Lowest integer value of T (unused for float T).
         * @param hi Highest integer value of T (unused for float T).
         */
        template<typename T, typename I, typename Acc>
        __global__ void softmax_ce_g_rows_kernel(const T* __restrict__ logits, const I* __restrict__ targets,
                                                 const T* __restrict__ dloss, T* __restrict__ dlogits,
                                                 uint32_t* __restrict__ err, const size_t rows,
                                                 const size_t cols_store, const size_t cols_eff,
                                                 const float64_t lo, const float64_t hi) {
            const auto r = blockIdx.x;
            if (r >= rows) {
                return;
            }
            const auto* row = logits + r * cols_store;
            auto* out_row = dlogits + r * cols_store;
            __shared__ Acc shared[BT_GPU_THREADS];
            __shared__ Acc row_max;
            const auto tid = threadIdx.x;
            auto m = acc_lowest<Acc>();
            for (auto j = tid; j < cols_eff; j += BT_GPU_THREADS) {
                if (
                    const auto x = sce_widen<T, Acc>(row, j);
                    x > m
                ) {
                    m = x;
                }
            }
            shared[tid] = m;
            __syncthreads();
            for (auto s = BT_GPU_THREADS / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared[tid] = shared[tid] > shared[tid + s] ? shared[tid] : shared[tid + s];
                }
                __syncthreads();
            }
            if (tid == 0) {
                row_max = shared[0];
            }
            __syncthreads();
            m = row_max;
            auto sum = Acc{0};
            for (auto j = tid; j < cols_eff; j += BT_GPU_THREADS) {
                sum += acc_exp(sce_widen<T, Acc>(row, j) - m);
            }
            shared[tid] = sum;
            __syncthreads();
            for (auto s = BT_GPU_THREADS / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared[tid] = shared[tid] + shared[tid + s];
                }
                __syncthreads();
            }
            const auto total = shared[0];
            auto target = sce_target(targets, r);
            if (target < 0 || static_cast<uint64_t>(target) >= cols_eff) {
                if (tid == 0) {
                    *err = 1u;
                }
                target = -1; // no one-hot; the host throws after sync
            }
            const auto scaled = sce_dloss<T, Acc>(dloss) / static_cast<Acc>(rows);
            /**
             * Captures the enclosing kernel state (row, m, total, target, scaled) by reference and is called per element by each thread.
             * Computes the softmax probability exp(logit - m) / total, subtracts 1 at the target column, and scales by dloss / rows.
             * A target of -1 (invalid, flagged for the host) yields no one-hot subtraction.
             * @brief Device-side gradient of the softmax cross-entropy loss for one logical class column.
             * @param j Logical (unpacked) class column index within the row.
             * @return Gradient value in the accumulator type Acc.
             */
            auto grad_at = [&](const size_t j) -> Acc {
                const auto p = acc_exp(sce_widen<T, Acc>(row, j) - m) / total;
                return (p - (static_cast<int64_t>(j) == target ? Acc{1} : Acc{0})) * scaled;
            };
            for (auto e = tid; e < cols_store; e += BT_GPU_THREADS) {
                if constexpr (std::is_same_v<T, float16x2_t>) {
                    out_row[e] = __halves2half2(__float2half_rn(grad_at(2 * e)), __float2half_rn(grad_at(2 * e + 1)));
                } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                    out_row[e] = __halves2bfloat162(dev_narrow_bf16(grad_at(2 * e)), dev_narrow_bf16(grad_at(2 * e + 1)));
                } else if constexpr (std::is_same_v<T, int4x2_t>) {
                    out_row[e] = dev_pack_int4(static_cast<int32_t>(sce_clamp(grad_at(2 * e), -8.0, 7.0)),
                                               static_cast<int32_t>(sce_clamp(grad_at(2 * e + 1), -8.0, 7.0)));
                } else {
                    out_row[e] = sce_narrow<T, Acc>(grad_at(e), lo, hi);
                }
            }
        }

        /**
         * The error flag is worker-local cudaMalloc/cudaFree; the D2H flag copy syncs the
         * default stream, so dlogits is complete before the worker returns. Integer
         * bounds are computed on the host (nvcc rejects numeric_limits in device code).
         * @brief Host-side softmax_ce_g launch.
         * @param logits Forward logits.
         * @param targets Target class index per row.
         * @param dloss Scalar upstream gradient.
         * @param dlogits Output gradient.
         * @param rows Row (batch) count.
         * @param cols_store Storage columns per row.
         * @param cols_eff Class count per row (2 * cols_store for packed dtypes).
         * @throws ErrorHandling::IndexOutOfRange when any target is outside [0, cols_eff).
         */
        template<typename T, typename I, typename Acc>
        __host__ void run_softmax_ce_g(const T* logits, const I* targets, const T* dloss, T* dlogits,
                              const size_t rows, const size_t cols_store, const size_t cols_eff) {
            // Integer bounds are computed on the host (nvcc rejects
            // numeric_limits in device code).
            auto lo = 0.0;
            auto hi = 0.0;
            if constexpr (std::is_integral_v<T>) {
                lo = static_cast<float64_t>(std::numeric_limits<T>::lowest());
                hi = static_cast<float64_t>(std::numeric_limits<T>::max());
            }
            uint32_t* err = nullptr;
            CUDA_CHECK_ERROR(cudaMalloc(&err, sizeof(uint32_t)));
            CUDA_CHECK_ERROR(cudaMemset(err, 0, sizeof(uint32_t)));
            softmax_ce_g_rows_kernel<T, I, Acc><<<rows, BT_GPU_THREADS>>>(logits, targets, dloss, dlogits, err, rows, cols_store, cols_eff, lo, hi);
            CUDA_CHECK_ERROR(cudaGetLastError());
            auto err_host = uint32_t{0};
            CUDA_CHECK_ERROR(cudaMemcpy(&err_host, err, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK_ERROR(cudaFree(err));
            if (err_host != 0u) {
                throw ErrorHandling::IndexOutOfRange();
            }
        }

        /**
         * Integer kinds only - validated by Tensor::softmax_ce. Any other target dtype throws NotImplemented.
         * @brief Dispatch the softmax_ce_g runner over the target dtype.
         * @param logits Forward logits.
         * @param targets Target data (type-erased).
         * @param t_dtype Dtype of targets.
         * @param dloss Scalar upstream gradient.
         * @param dlogits Output gradient.
         * @param rows Row (batch) count.
         * @param cols_store Storage columns per row.
         * @param cols_eff Class count per row (2 * cols_store for packed dtypes).
         * @throws ErrorHandling::NotImplemented for a non-integer target dtype.
         */
        template<typename T, typename Acc>
        __host__ void run_sce_g_targets(const T* logits, const void* targets, const DataTypes::DType t_dtype, const T* dloss, T* dlogits,
                               const size_t rows, const size_t cols_store, const size_t cols_eff) {
            using DType = DataTypes::DType;
            switch (t_dtype) {
                case DType::Int8: run_softmax_ce_g<T, int8_t, Acc>(logits, static_cast<const int8_t*>(targets), dloss, dlogits, rows, cols_store, cols_eff); break;
                case DType::UInt8: run_softmax_ce_g<T, uint8_t, Acc>(logits, static_cast<const uint8_t*>(targets), dloss, dlogits, rows, cols_store, cols_eff); break;
                case DType::Int16: run_softmax_ce_g<T, int16_t, Acc>(logits, static_cast<const int16_t*>(targets), dloss, dlogits, rows, cols_store, cols_eff); break;
                case DType::UInt16: run_softmax_ce_g<T, uint16_t, Acc>(logits, static_cast<const uint16_t*>(targets), dloss, dlogits, rows, cols_store, cols_eff); break;
                case DType::Int32: run_softmax_ce_g<T, int32_t, Acc>(logits, static_cast<const int32_t*>(targets), dloss, dlogits, rows, cols_store, cols_eff); break;
                case DType::UInt32: run_softmax_ce_g<T, uint32_t, Acc>(logits, static_cast<const uint32_t*>(targets), dloss, dlogits, rows, cols_store, cols_eff); break;
                case DType::Int64: run_softmax_ce_g<T, int64_t, Acc>(logits, static_cast<const int64_t*>(targets), dloss, dlogits, rows, cols_store, cols_eff); break;
                case DType::UInt64: run_softmax_ce_g<T, uint64_t, Acc>(logits, static_cast<const uint64_t*>(targets), dloss, dlogits, rows, cols_store, cols_eff); break;
                case DType::Int4x2: run_softmax_ce_g<T, int4x2_t, Acc>(logits, static_cast<const int4x2_t*>(targets), dloss, dlogits, rows, cols_store, cols_eff); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
        }
    }

    /**
     * Queues the launch on the CPU thread pool (High priority) and returns; the shared
     * future is given to both tensors, and errors surface at sync(). dtype and numel
     * come from g; d is read with the same dtype and element count. Matches the CPU grad_acc.
     * @brief GPU gradient accumulation g += d (all dtypes).
     * @param g Accumulator gradient tensor, updated in place.
     * @param d Delta tensor added into g.
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if g or d is not on the GPU.
     * @throws ErrorHandling::NotImplemented (from the worker, at sync) for an unsupported dtype.
     */
    __host__ void grad_acc_gpu(Tensor& g, const Tensor& d) {
        if (g.device != Device::GPU || d.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(g.device));
        }
        auto* const g_data = g.data;
        const auto* const d_data = d.data;
        const auto g_offset = g.offset;
        const auto d_offset = d.offset;
        const auto dtype = g.dtype;
        const auto numel = g.numel;
        // Raw pointers captured by value: the future lands on both tensors, so
        // buffer teardown always drains this worker first.
        /**
         * Captures raw pointers, offsets, dtype and numel by value, so the lambda stays valid after the enclosing call returns.
         * Dispatches on dtype to run_grad_acc, which launches the accumulate kernel on the default stream, then calls cudaStreamSynchronize(nullptr) so the future completes only once the buffer writes are done.
         * An unsupported dtype throws ErrorHandling::NotImplemented.
         * Runs on a CPU thread pool thread (High priority) via the async pool; the host call has already returned. Any exception thrown here (including a failed CUDA_CHECK_ERROR) is captured in the shared future and rethrown at sync().
         * @brief Async pool worker that launches the grad_acc kernel for the tensor dtype and syncs the stream.
         */
        const auto worker = [=] {
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: run_grad_acc(static_cast<float32_t*>(g_data) + g_offset, static_cast<const float32_t*>(d_data) + d_offset, numel); break;
                case DType::Float64: run_grad_acc(static_cast<float64_t*>(g_data) + g_offset, static_cast<const float64_t*>(d_data) + d_offset, numel); break;
                case DType::Float16: run_grad_acc(static_cast<float16_t*>(g_data) + g_offset, static_cast<const float16_t*>(d_data) + d_offset, numel); break;
                case DType::BFloat16: run_grad_acc(static_cast<bfloat16_t*>(g_data) + g_offset, static_cast<const bfloat16_t*>(d_data) + d_offset, numel); break;
                case DType::Float16x2: run_grad_acc(static_cast<float16x2_t*>(g_data) + g_offset, static_cast<const float16x2_t*>(d_data) + d_offset, numel); break;
                case DType::BFloat16x2: run_grad_acc(static_cast<bfloat16x2_t*>(g_data) + g_offset, static_cast<const bfloat16x2_t*>(d_data) + d_offset, numel); break;
                case DType::Int8: run_grad_acc(static_cast<int8_t*>(g_data) + g_offset, static_cast<const int8_t*>(d_data) + d_offset, numel); break;
                case DType::UInt8: run_grad_acc(static_cast<uint8_t*>(g_data) + g_offset, static_cast<const uint8_t*>(d_data) + d_offset, numel); break;
                case DType::Int16: run_grad_acc(static_cast<int16_t*>(g_data) + g_offset, static_cast<const int16_t*>(d_data) + d_offset, numel); break;
                case DType::UInt16: run_grad_acc(static_cast<uint16_t*>(g_data) + g_offset, static_cast<const uint16_t*>(d_data) + d_offset, numel); break;
                case DType::Int32: run_grad_acc(static_cast<int32_t*>(g_data) + g_offset, static_cast<const int32_t*>(d_data) + d_offset, numel); break;
                case DType::UInt32: run_grad_acc(static_cast<uint32_t*>(g_data) + g_offset, static_cast<const uint32_t*>(d_data) + d_offset, numel); break;
                case DType::Int64: run_grad_acc(static_cast<int64_t*>(g_data) + g_offset, static_cast<const int64_t*>(d_data) + d_offset, numel); break;
                case DType::UInt64: run_grad_acc(static_cast<uint64_t*>(g_data) + g_offset, static_cast<const uint64_t*>(d_data) + d_offset, numel); break;
                case DType::Int4x2: run_grad_acc(static_cast<int4x2_t*>(g_data) + g_offset, static_cast<const int4x2_t*>(d_data) + d_offset, numel); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
            // The kernel is in-flight when the launch returns; sync so the
            // future only completes once the buffer writes are done.
            CUDA_CHECK_ERROR(cudaStreamSynchronize(nullptr));
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        g.give_future(future);
        d.give_future(future);
    }

    /**
     * Queues the launch on the CPU thread pool (High priority) and returns; the shared
     * future is given to all three tensors, and errors surface at sync(). dtype and
     * numel come from x. Matches the CPU relu_g (select, NaN in x zeroes the lane).
     * @brief GPU relu backward dx = x > 0 ? dy : 0 (all dtypes).
     * @param dy Upstream gradient tensor.
     * @param x Forward input tensor of the relu.
     * @param dx Output gradient tensor.
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if any tensor is not on the GPU.
     * @throws ErrorHandling::NotImplemented (from the worker, at sync) for an unsupported dtype.
     */
    __host__ void relu_g_gpu(const Tensor& dy, const Tensor& x, Tensor& dx) {
        if (dy.device != Device::GPU || x.device != Device::GPU || dx.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(x.device));
        }
        const auto* const dy_data = dy.data;
        const auto* const x_data = x.data;
        auto* const dx_data = dx.data;
        const auto dy_offset = dy.offset;
        const auto x_offset = x.offset;
        const auto dx_offset = dx.offset;
        const auto dtype = x.dtype;
        const auto numel = x.numel;
        /**
         * Captures raw pointers, offsets, dtype and numel by value, so the lambda stays valid after the enclosing call returns.
         * Dispatches on dtype to run_relu_g, which launches the relu backward kernel on the default stream, then calls cudaStreamSynchronize(nullptr) so the future completes only once dx is written.
         * An unsupported dtype throws ErrorHandling::NotImplemented.
         * Runs on a CPU thread pool thread (High priority) via the async pool; the host call has already returned. Any exception thrown here (including a failed CUDA_CHECK_ERROR) is captured in the shared future and rethrown at sync().
         * @brief Async pool worker that launches the relu_g kernel for the tensor dtype and syncs the stream.
         */
        const auto worker = [=] {
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: run_relu_g(static_cast<const float32_t*>(x_data) + x_offset, static_cast<const float32_t*>(dy_data) + dy_offset, static_cast<float32_t*>(dx_data) + dx_offset, numel); break;
                case DType::Float64: run_relu_g(static_cast<const float64_t*>(x_data) + x_offset, static_cast<const float64_t*>(dy_data) + dy_offset, static_cast<float64_t*>(dx_data) + dx_offset, numel); break;
                case DType::Float16: run_relu_g(static_cast<const float16_t*>(x_data) + x_offset, static_cast<const float16_t*>(dy_data) + dy_offset, static_cast<float16_t*>(dx_data) + dx_offset, numel); break;
                case DType::BFloat16: run_relu_g(static_cast<const bfloat16_t*>(x_data) + x_offset, static_cast<const bfloat16_t*>(dy_data) + dy_offset, static_cast<bfloat16_t*>(dx_data) + dx_offset, numel); break;
                case DType::Float16x2: run_relu_g(static_cast<const float16x2_t*>(x_data) + x_offset, static_cast<const float16x2_t*>(dy_data) + dy_offset, static_cast<float16x2_t*>(dx_data) + dx_offset, numel); break;
                case DType::BFloat16x2: run_relu_g(static_cast<const bfloat16x2_t*>(x_data) + x_offset, static_cast<const bfloat16x2_t*>(dy_data) + dy_offset, static_cast<bfloat16x2_t*>(dx_data) + dx_offset, numel); break;
                case DType::Int8: run_relu_g(static_cast<const int8_t*>(x_data) + x_offset, static_cast<const int8_t*>(dy_data) + dy_offset, static_cast<int8_t*>(dx_data) + dx_offset, numel); break;
                case DType::UInt8: run_relu_g(static_cast<const uint8_t*>(x_data) + x_offset, static_cast<const uint8_t*>(dy_data) + dy_offset, static_cast<uint8_t*>(dx_data) + dx_offset, numel); break;
                case DType::Int16: run_relu_g(static_cast<const int16_t*>(x_data) + x_offset, static_cast<const int16_t*>(dy_data) + dy_offset, static_cast<int16_t*>(dx_data) + dx_offset, numel); break;
                case DType::UInt16: run_relu_g(static_cast<const uint16_t*>(x_data) + x_offset, static_cast<const uint16_t*>(dy_data) + dy_offset, static_cast<uint16_t*>(dx_data) + dx_offset, numel); break;
                case DType::Int32: run_relu_g(static_cast<const int32_t*>(x_data) + x_offset, static_cast<const int32_t*>(dy_data) + dy_offset, static_cast<int32_t*>(dx_data) + dx_offset, numel); break;
                case DType::UInt32: run_relu_g(static_cast<const uint32_t*>(x_data) + x_offset, static_cast<const uint32_t*>(dy_data) + dy_offset, static_cast<uint32_t*>(dx_data) + dx_offset, numel); break;
                case DType::Int64: run_relu_g(static_cast<const int64_t*>(x_data) + x_offset, static_cast<const int64_t*>(dy_data) + dy_offset, static_cast<int64_t*>(dx_data) + dx_offset, numel); break;
                case DType::UInt64: run_relu_g(static_cast<const uint64_t*>(x_data) + x_offset, static_cast<const uint64_t*>(dy_data) + dy_offset, static_cast<uint64_t*>(dx_data) + dx_offset, numel); break;
                case DType::Int4x2: run_relu_g(static_cast<const int4x2_t*>(x_data) + x_offset, static_cast<const int4x2_t*>(dy_data) + dy_offset, static_cast<int4x2_t*>(dx_data) + dx_offset, numel); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
            CUDA_CHECK_ERROR(cudaStreamSynchronize(nullptr));
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        dy.give_future(future);
        x.give_future(future);
        dx.give_future(future);
    }

    /**
     * Dtype-agnostic: bits are moved as an unsigned word of the element byte width
     * (1, 2, 4 or 8), so no value is interpreted. Batched over all leading dimensions.
     * Queues the launch on the CPU thread pool (High priority) and returns; the shared
     * future is given to both tensors, and errors surface at sync().
     * @brief GPU transpose of the last two dimensions (out[..., j, i] = t[..., i, j]).
     * @param t Input tensor.
     * @param out Output tensor with the last two dimensions swapped.
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if t or out is not on the GPU.
     * @throws ErrorHandling::UnReachableCode (from the worker, at sync) for an element width other than 1, 2, 4 or 8 bytes.
     */
    __host__ void transpose_last2_gpu(const Tensor& t, Tensor& out) {
        if (t.device != Device::GPU || out.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(t.device));
        }
        const auto eb = DataTypes::dtype_bytes(t.dtype);
        const auto* const in_data = static_cast<const uint8_t*>(t.data) + t.offset * eb;
        auto* const out_data = static_cast<uint8_t*>(out.data) + out.offset * eb;
        const auto rows = t.shape[t.ndim - 2];
        const auto cols = t.shape[t.ndim - 1];
        const auto numel = t.numel;
        /**
         * Captures raw pointers, element byte width, rows, cols and numel by value, so the lambda stays valid after the enclosing call returns.
         * Dispatches on the element byte width (1, 2, 4 or 8) to run_transpose over the matching unsigned word type, then calls cudaStreamSynchronize(nullptr) so the future completes only once out is written.
         * Any other byte width throws ErrorHandling::UnReachableCode.
         * Runs on a CPU thread pool thread (High priority) via the async pool; the host call has already returned. Any exception thrown here (including a failed CUDA_CHECK_ERROR) is captured in the shared future and rethrown at sync().
         * @brief Async pool worker that launches the transpose kernel for the element width and syncs the stream.
         */
        const auto worker = [=] {
            switch (eb) {
                case 1: run_transpose<uint8_t>(in_data, out_data, rows, cols, numel); break;
                case 2: run_transpose<uint16_t>(in_data, out_data, rows, cols, numel); break;
                case 4: run_transpose<uint32_t>(in_data, out_data, rows, cols, numel); break;
                case 8: run_transpose<uint64_t>(in_data, out_data, rows, cols, numel); break;
                [[unlikely]] default: throw ErrorHandling::UnReachableCode();
            }
            CUDA_CHECK_ERROR(cudaStreamSynchronize(nullptr));
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        t.give_future(future);
        out.give_future(future);
    }

    /**
     * Column sum of dy over all leading dimensions (cols = last dim, rows = numel / cols).
     * Accumulators: fp32 for fp16/bf16 (scalar and packed), native width for fp32/fp64,
     * the unsigned twin (wrap) for integers, int32 per nibble for int4x2. Queues the
     * launch on the CPU thread pool (High priority) and returns; the shared future is
     * given to both tensors, and errors surface at sync().
     * @brief GPU bias_add backward dbias = sum over rows of dy (all dtypes).
     * @param dy Upstream gradient tensor.
     * @param dbias Output bias gradient tensor, one element per last-dimension column.
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if dy or dbias is not on the GPU.
     * @throws ErrorHandling::NotImplemented (from the worker, at sync) for an unsupported dtype.
     */
    __host__ void bias_add_g_gpu(const Tensor& dy, Tensor& dbias) {
        if (dy.device != Device::GPU || dbias.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(dy.device));
        }
        const auto* const dy_data = dy.data;
        auto* const db_data = dbias.data;
        const auto dy_offset = dy.offset;
        const auto db_offset = dbias.offset;
        const auto dtype = dy.dtype;
        const auto cols = dy.shape[dy.ndim - 1];
        const auto rows = cols == 0 ? size_t{0} : dy.numel / cols;
        /**
         * Captures raw pointers, offsets, dtype, rows and cols by value, so the lambda stays valid after the enclosing call returns.
         * Dispatches on dtype to run_bias_add_g, which launches the column-sum kernel and frees its scratch before returning; that cudaFree already syncs the stream, so the worker issues no explicit sync.
         * An unsupported dtype throws ErrorHandling::NotImplemented.
         * Runs on a CPU thread pool thread (High priority) via the async pool; the host call has already returned. Any exception thrown here (including a failed CUDA_CHECK_ERROR) is captured in the shared future and rethrown at sync().
         * @brief Async pool worker that launches the bias_add_g kernel for the tensor dtype.
         */
        const auto worker = [=] {
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: run_bias_add_g<float32_t, float32_t>(static_cast<const float32_t*>(dy_data) + dy_offset, static_cast<float32_t*>(db_data) + db_offset, rows, cols, cols); break;
                case DType::Float64: run_bias_add_g<float64_t, float64_t>(static_cast<const float64_t*>(dy_data) + dy_offset, static_cast<float64_t*>(db_data) + db_offset, rows, cols, cols); break;
                case DType::Float16: run_bias_add_g<float16_t, float32_t>(static_cast<const float16_t*>(dy_data) + dy_offset, static_cast<float16_t*>(db_data) + db_offset, rows, cols, cols); break;
                case DType::BFloat16: run_bias_add_g<bfloat16_t, float32_t>(static_cast<const bfloat16_t*>(dy_data) + dy_offset, static_cast<bfloat16_t*>(db_data) + db_offset, rows, cols, cols); break;
                case DType::Float16x2: run_bias_add_g<float16x2_t, float32_t>(static_cast<const float16x2_t*>(dy_data) + dy_offset, static_cast<float16x2_t*>(db_data) + db_offset, rows, cols, 2 * cols); break;
                case DType::BFloat16x2: run_bias_add_g<bfloat16x2_t, float32_t>(static_cast<const bfloat16x2_t*>(dy_data) + dy_offset, static_cast<bfloat16x2_t*>(db_data) + db_offset, rows, cols, 2 * cols); break;
                case DType::Int8: run_bias_add_g<int8_t, uint8_t>(static_cast<const int8_t*>(dy_data) + dy_offset, static_cast<int8_t*>(db_data) + db_offset, rows, cols, cols); break;
                case DType::UInt8: run_bias_add_g<uint8_t, uint8_t>(static_cast<const uint8_t*>(dy_data) + dy_offset, static_cast<uint8_t*>(db_data) + db_offset, rows, cols, cols); break;
                case DType::Int16: run_bias_add_g<int16_t, uint16_t>(static_cast<const int16_t*>(dy_data) + dy_offset, static_cast<int16_t*>(db_data) + db_offset, rows, cols, cols); break;
                case DType::UInt16: run_bias_add_g<uint16_t, uint16_t>(static_cast<const uint16_t*>(dy_data) + dy_offset, static_cast<uint16_t*>(db_data) + db_offset, rows, cols, cols); break;
                case DType::Int32: run_bias_add_g<int32_t, uint32_t>(static_cast<const int32_t*>(dy_data) + dy_offset, static_cast<int32_t*>(db_data) + db_offset, rows, cols, cols); break;
                case DType::UInt32: run_bias_add_g<uint32_t, uint32_t>(static_cast<const uint32_t*>(dy_data) + dy_offset, static_cast<uint32_t*>(db_data) + db_offset, rows, cols, cols); break;
                case DType::Int64: run_bias_add_g<int64_t, uint64_t>(static_cast<const int64_t*>(dy_data) + dy_offset, static_cast<int64_t*>(db_data) + db_offset, rows, cols, cols); break;
                case DType::UInt64: run_bias_add_g<uint64_t, uint64_t>(static_cast<const uint64_t*>(dy_data) + dy_offset, static_cast<uint64_t*>(db_data) + db_offset, rows, cols, cols); break;
                case DType::Int4x2: run_bias_add_g<int4x2_t, int32_t>(static_cast<const int4x2_t*>(dy_data) + dy_offset, static_cast<int4x2_t*>(db_data) + db_offset, rows, cols, 2 * cols); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
            // run_bias_add_g's cudaFree already synced the stream.
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        dy.give_future(future);
        dbias.give_future(future);
    }

    /**
     * logits is a 2D [rows, cols_store] tensor (packed dtypes carry two classes per
     * element); targets holds one integer class index per row. Accumulators: fp32 for
     * fp32/fp16/bf16 (scalar and packed), fp64 for fp64 and integer dtypes. Queues the
     * launch on the CPU thread pool (High priority) and returns; the shared future is
     * given to all four tensors, and errors surface at sync().
     * @brief GPU softmax cross-entropy backward (all logits dtypes, integer targets).
     * @param dloss_grad Scalar upstream gradient of the loss.
     * @param logits Forward logits tensor.
     * @param targets Integer target class index per row.
     * @param dlogits Output gradient tensor, same shape as logits.
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if any tensor is not on the GPU.
     * @throws ErrorHandling::NotImplemented (from the worker, at sync) for an unsupported logits or targets dtype.
     * @throws ErrorHandling::IndexOutOfRange (from the worker, at sync) when a target is outside [0, cols_eff).
     */
    __host__ void softmax_ce_g_gpu(const Tensor& dloss_grad, const Tensor& logits, const Tensor& targets, Tensor& dlogits) {
        if (dloss_grad.device != Device::GPU || logits.device != Device::GPU || targets.device != Device::GPU || dlogits.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(logits.device));
        }
        const auto* const g_data = dloss_grad.data;
        const auto* const l_data = logits.data;
        const auto* const targets_data = targets.data;
        auto* const out_data = dlogits.data;
        const auto g_offset = dloss_grad.offset;
        const auto l_offset = logits.offset;
        const auto targets_offset = targets.offset;
        const auto out_offset = dlogits.offset;
        const auto dtype = logits.dtype;
        const auto t_dtype = targets.dtype;
        const auto rows = logits.shape[0];
        const auto cols_store = logits.shape[1];
        const auto packed = dtype == DataTypes::DType::Float16x2 || dtype == DataTypes::DType::BFloat16x2 || dtype == DataTypes::DType::Int4x2;
        const auto cols_eff = cols_store * (packed ? size_t{2} : size_t{1});
        /**
         * Captures raw pointers, offsets, dtypes, rows, cols_store and cols_eff by value, so the lambda stays valid after the enclosing call returns.
         * Computes the targets base pointer, then dispatches on the logits dtype to run_sce_g_targets, which dispatches on the target dtype and runs run_softmax_ce_g. That runner copies the error flag back and frees it, which syncs the stream, so the worker issues no explicit sync.
         * An unsupported logits or targets dtype throws ErrorHandling::NotImplemented; a target outside [0, cols_eff) throws ErrorHandling::IndexOutOfRange.
         * Runs on a CPU thread pool thread (High priority) via the async pool; the host call has already returned. Any exception thrown here (including a failed CUDA_CHECK_ERROR) is captured in the shared future and rethrown at sync().
         * @brief Async pool worker that launches the softmax_ce_g kernel for the logits and target dtypes.
         */
        const auto worker = [=] {
            using DType = DataTypes::DType;
            const auto* targets_base = static_cast<const uint8_t*>(targets_data) + targets_offset * DataTypes::dtype_bytes(t_dtype);
            switch (dtype) {
                case DType::Float32: run_sce_g_targets<float32_t, float32_t>(static_cast<const float32_t*>(l_data) + l_offset, targets_base, t_dtype, static_cast<const float32_t*>(g_data) + g_offset, static_cast<float32_t*>(out_data) + out_offset, rows, cols_store, cols_eff); break;
                case DType::Float64: run_sce_g_targets<float64_t, float64_t>(static_cast<const float64_t*>(l_data) + l_offset, targets_base, t_dtype, static_cast<const float64_t*>(g_data) + g_offset, static_cast<float64_t*>(out_data) + out_offset, rows, cols_store, cols_eff); break;
                case DType::Float16: run_sce_g_targets<float16_t, float32_t>(static_cast<const float16_t*>(l_data) + l_offset, targets_base, t_dtype, static_cast<const float16_t*>(g_data) + g_offset, static_cast<float16_t*>(out_data) + out_offset, rows, cols_store, cols_eff); break;
                case DType::BFloat16: run_sce_g_targets<bfloat16_t, float32_t>(static_cast<const bfloat16_t*>(l_data) + l_offset, targets_base, t_dtype, static_cast<const bfloat16_t*>(g_data) + g_offset, static_cast<bfloat16_t*>(out_data) + out_offset, rows, cols_store, cols_eff); break;
                case DType::Float16x2: run_sce_g_targets<float16x2_t, float32_t>(static_cast<const float16x2_t*>(l_data) + l_offset, targets_base, t_dtype, static_cast<const float16x2_t*>(g_data) + g_offset, static_cast<float16x2_t*>(out_data) + out_offset, rows, cols_store, cols_eff); break;
                case DType::BFloat16x2: run_sce_g_targets<bfloat16x2_t, float32_t>(static_cast<const bfloat16x2_t*>(l_data) + l_offset, targets_base, t_dtype, static_cast<const bfloat16x2_t*>(g_data) + g_offset, static_cast<bfloat16x2_t*>(out_data) + out_offset, rows, cols_store, cols_eff); break;
                case DType::Int8: run_sce_g_targets<int8_t, float64_t>(static_cast<const int8_t*>(l_data) + l_offset, targets_base, t_dtype, static_cast<const int8_t*>(g_data) + g_offset, static_cast<int8_t*>(out_data) + out_offset, rows, cols_store, cols_eff); break;
                case DType::UInt8: run_sce_g_targets<uint8_t, float64_t>(static_cast<const uint8_t*>(l_data) + l_offset, targets_base, t_dtype, static_cast<const uint8_t*>(g_data) + g_offset, static_cast<uint8_t*>(out_data) + out_offset, rows, cols_store, cols_eff); break;
                case DType::Int16: run_sce_g_targets<int16_t, float64_t>(static_cast<const int16_t*>(l_data) + l_offset, targets_base, t_dtype, static_cast<const int16_t*>(g_data) + g_offset, static_cast<int16_t*>(out_data) + out_offset, rows, cols_store, cols_eff); break;
                case DType::UInt16: run_sce_g_targets<uint16_t, float64_t>(static_cast<const uint16_t*>(l_data) + l_offset, targets_base, t_dtype, static_cast<const uint16_t*>(g_data) + g_offset, static_cast<uint16_t*>(out_data) + out_offset, rows, cols_store, cols_eff); break;
                case DType::Int32: run_sce_g_targets<int32_t, float64_t>(static_cast<const int32_t*>(l_data) + l_offset, targets_base, t_dtype, static_cast<const int32_t*>(g_data) + g_offset, static_cast<int32_t*>(out_data) + out_offset, rows, cols_store, cols_eff); break;
                case DType::UInt32: run_sce_g_targets<uint32_t, float64_t>(static_cast<const uint32_t*>(l_data) + l_offset, targets_base, t_dtype, static_cast<const uint32_t*>(g_data) + g_offset, static_cast<uint32_t*>(out_data) + out_offset, rows, cols_store, cols_eff); break;
                case DType::Int64: run_sce_g_targets<int64_t, float64_t>(static_cast<const int64_t*>(l_data) + l_offset, targets_base, t_dtype, static_cast<const int64_t*>(g_data) + g_offset, static_cast<int64_t*>(out_data) + out_offset, rows, cols_store, cols_eff); break;
                case DType::UInt64: run_sce_g_targets<uint64_t, float64_t>(static_cast<const uint64_t*>(l_data) + l_offset, targets_base, t_dtype, static_cast<const uint64_t*>(g_data) + g_offset, static_cast<uint64_t*>(out_data) + out_offset, rows, cols_store, cols_eff); break;
                case DType::Int4x2: run_sce_g_targets<int4x2_t, float64_t>(static_cast<const int4x2_t*>(l_data) + l_offset, targets_base, t_dtype, static_cast<const int4x2_t*>(g_data) + g_offset, static_cast<int4x2_t*>(out_data) + out_offset, rows, cols_store, cols_eff); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
            // run_softmax_ce_g's D2H memcpy and cudaFree already synced the stream.
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        dloss_grad.give_future(future);
        logits.give_future(future);
        targets.give_future(future);
        dlogits.give_future(future);
    }
}
#endif
