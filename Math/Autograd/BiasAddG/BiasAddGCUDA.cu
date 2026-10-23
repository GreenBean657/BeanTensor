/**
 * @file BiasAddGCUDA.cu
 * @author GreenBean
 * @brief CUDA backend for the bias_add_g op (bias_add backward, dbias = column
 * sum of dy).
 * Divergent twin of Math/Autograd/BiasAddG/BiasAddGHIP.cpp — the two backends
 * are written separately per project rule (no HIP-only unification), but must
 * stay behaviorally identical: same fp32/fp64 compute widths, same RNE
 * narrowing, same wrap integer adds.
 * @version 0.1
 * @date 2026-10-20
 */
#if defined(USE_CUDA)
#include <cstdint>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "Math/Autograd/BiasAddG/BiasAddG.h"
#include "Math/Autograd/AutogradCudaShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    using namespace autograd_gpu;

    namespace {
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
            chunks = chunks < 1 ? 1 : chunks > BT_GPU_COLSUM_MAX_CHUNKS ? BT_GPU_COLSUM_MAX_CHUNKS : chunks;
            const auto rows_per_chunk = (rows + chunks - 1) / chunks;
            Acc* partials = nullptr;
            CUDA_CHECK_ERROR(cudaMalloc(&partials, chunks * cols_eff * sizeof(Acc)));
            bias_add_g_colsum_kernel<T, Acc><<<grid_for(chunks * cols_eff), BT_GPU_THREADS>>>(dy, partials, rows, cols, cols_eff, chunks, rows_per_chunk);
            bias_add_g_store_kernel<T, Acc><<<grid_for(cols), BT_GPU_THREADS>>>(partials, dbias, cols, cols_eff, chunks);
            CUDA_CHECK_ERROR(cudaGetLastError());
            CUDA_CHECK_ERROR(cudaFree(partials));
        }
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
    __host__ void bias_add_g_gpu(const Tensor& dy, const Tensor& dbias) {
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
}
#endif
