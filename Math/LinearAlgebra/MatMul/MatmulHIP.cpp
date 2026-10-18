/**
 * @file MatmulHIP.cpp
 * @author GreenBean
 * @brief HIP backend for matmul.
 * @version 0.1
 * @date 2026-10-18
 */
#include <cstdint>
#include <limits>
#include <type_traits>

#include <rocblas/rocblas.h>

#include "Math/LinearAlgebra/MatMul/Matmul.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    namespace {
        static_assert(std::is_same_v<rocblas_int, int32_t>, "rocBLAS is built with 32-bit rocblas_int; if this fails, the gemm dim casts need revisiting");
        static_assert(std::is_same_v<rocblas_stride, int64_t>, "rocBLAS batch strides are 64-bit; if this fails, the stride casts need revisiting");

        /**
         * Process-wide rocBLAS handle, created lazily on first GEMM. Never
         * destroyed (process-lifetime resource, like the CPU thread pool) and
         * always used on the default stream, so concurrent pool tasks serialize
         * on the stream itself.
         * @brief Get the shared rocBLAS handle.
         * @return Initialized rocBLAS handle.
         */
        __host__ rocblas_handle get_rocblas_handle() {
            /**
             * Creates the rocBLAS handle on first use.
             * @note Lambda subcall. Self-execution prohibited.
             * @brief Create the rocBLAS handle.
             */
            static rocblas_handle handle = [] {
                rocblas_handle h = nullptr;
                ROCBLAS_CHECK_ERROR(rocblas_create_handle(&h));
                return h;
            }();
            return handle;
        }

        /**
         * Device-side lane extraction for the lane-plane deinterleave.
         * @brief Low fp16 lane of a packed fp16x2 element.
         * @param v Packed fp16x2 element.
         * @return The lane value.
         */
        __device__ __forceinline__ float16_t dev_lane_lo(const float16x2_t v) { return __low2half(v); }
        /**
         * Device-side lane extraction for the lane-plane deinterleave.
         * @brief High fp16 lane of a packed fp16x2 element.
         * @param v Packed fp16x2 element.
         * @return The lane value.
         */
        __device__ __forceinline__ float16_t dev_lane_hi(const float16x2_t v) { return __high2half(v); }
        /**
         * Device-side lane extraction for the lane-plane deinterleave.
         * @brief Low bf16 lane of a packed bfloat16x2 element.
         * @param v Packed bfloat16x2 element.
         * @return The lane value.
         */
        __device__ __forceinline__ bfloat16_t dev_lane_lo(const bfloat16x2_t v) { return __low2bfloat16(v); }
        /**
         * Device-side lane extraction for the lane-plane deinterleave.
         * @brief High bf16 lane of a packed bfloat16x2 element.
         * @param v Packed bfloat16x2 element.
         * @return The lane value.
         */
        __device__ __forceinline__ bfloat16_t dev_lane_hi(const bfloat16x2_t v) { return __high2bfloat16(v); }
        /**
         * Device-side lane packing for the lane-plane interleave.
         * @brief Pack two fp16 lanes into one fp16x2 element.
         * @param lo Low lane.
         * @param hi High lane.
         * @return The packed element.
         */
        __device__ __forceinline__ float16x2_t dev_pack_lanes(const float16_t lo, const float16_t hi) { return __halves2half2(lo, hi); }
        /**
         * Device-side lane packing for the lane-plane interleave.
         * @brief Pack two bf16 lanes into one bfloat16x2 element.
         * @param lo Low lane.
         * @param hi High lane.
         * @return The packed element.
         */
        __device__ __forceinline__ bfloat16x2_t dev_pack_lanes(const bfloat16_t lo, const bfloat16_t hi) { return __halves2bfloat162(lo, hi); }

        /**
         * Split packed elements into per-lane planes laid out as extra batches:
         * batch bi's lo lane lands at plane 2*bi, its hi lane at plane 2*bi+1, each plane holding per_batch lanes.
         * That uniform plane stride is what lets the pair ride a single strided-batched GEMM.
         * @brief Packed -> per-lane plane deinterleave kernel.
         * @param src Packed source (numel packed elements).
         * @param dst Lane planes output (2 x numel lanes).
         * @param numel Total packed elements across all batches.
         * @param per_batch Packed elements per batch (m*k or k*n).
         */
        template<typename P, typename H>
        __global__ void deinterleave_kernel(const P* __restrict__ src, H* __restrict__ dst, const size_t numel, const size_t per_batch) {
            const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t stride = blockDim.x * gridDim.x;
            for (size_t e = tid; e < numel; e += stride) {
                const auto bi = e / per_batch;
                const auto r = e - bi * per_batch;
                const auto v = src[e];
                dst[(2 * bi) * per_batch + r] = dev_lane_lo(v);
                dst[(2 * bi + 1) * per_batch + r] = dev_lane_hi(v);
            }
        }

        /**
         * Inverse of deinterleave_kernel: reads planes 2*bi and 2*bi+1 and packs them back into one element per lane pair.
         * @brief Per-lane plane -> packed interleave kernel.
         * @param src Lane planes input (2 x numel lanes).
         * @param dst Packed output (numel packed elements).
         * @param numel Total packed elements across all batches.
         * @param per_batch Packed elements per batch (m*k or k*n).
         */
        template<typename P, typename H>
        __global__ void interleave_kernel(const H* __restrict__ src, P* __restrict__ dst, const size_t numel, const size_t per_batch) {
            const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t stride = blockDim.x * gridDim.x;
            for (size_t e = tid; e < numel; e += stride) {
                const auto bi = e / per_batch;
                const auto r = e - bi * per_batch;
                dst[e] = dev_pack_lanes(src[(2 * bi) * per_batch + r], src[(2 * bi + 1) * per_batch + r]);
            }
        }

        /**
         * rocBLAS i8 GEMM widens the output to int32, but matmul's contract is
         * an int8 output, so the strided-batched result lands in scratch and
         * this kernel narrows it. static_cast<int8_t> is modular (C++20), the
         * same low-8-bits truncation the CPU integer kernels perform on store.
         * @brief Narrow int32 GEMM accumulators to int8 (mod 2^8).
         * @param src Int32 accumulators (numel elements, device).
         * @param dst Int8 output (numel elements, device).
         * @param numel Total elements across all batches.
         */
        __global__ void narrow_i32_to_i8_kernel(const int32_t* __restrict__ src, int8_t* __restrict__ dst, const size_t numel) {
            const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t stride = blockDim.x * gridDim.x;
            for (size_t e = tid; e < numel; e += stride) {
                dst[e] = static_cast<int8_t>(src[e]);
            }
        }

        /**
         * Integer GEMM for the dtypes rocBLAS cannot do (everything but int8):
         * exact modular (two's-complement wrap) arithmetic. Each product is
         * formed in the widened unsigned twin of T (uint32_t for <=32-bit T,
         * uint64_t for 64-bit T) and accumulated with wrapping adds; the store
         * truncates back to T. Because truncate(sum mod 2^|AccT|) = sum mod 2^|T|,
         * this is bit-identical to the CPU scalar/VNNI reference for every
         * element. One 16×16 C tile per block, batch from blockIdx.z;
         * out-of-range rows/cols zero-fill their tile loads and skip stores.
         * @brief Tiled shared-memory integer GEMM (wrap semantics).
         * @param a Row-major (B×)m×k left operand (device).
         * @param b Row-major (B×)k×n right operand (device).
         * @param c Row-major (B×)m×n output (device).
         * @param m Rows of a/c.
         * @param k Inner dimension.
         * @param n Columns of b/c.
         */
        template<typename T>
        __global__ void gemm_int_fallback_kernel(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c,
                                                 const size_t m, const size_t k, const size_t n) {
            using UT = std::make_unsigned_t<T>;
            using AccT = std::conditional_t<sizeof(T) <= sizeof(uint32_t), uint32_t, uint64_t>;
            constexpr size_t tile = 16;
            __shared__ T a_tile[tile][tile];
            __shared__ T b_tile[tile][tile];
            const auto row = blockIdx.y * tile + threadIdx.y;
            const auto col = blockIdx.x * tile + threadIdx.x;
            const auto* a_base = a + blockIdx.z * m * k;
            const auto* b_base = b + blockIdx.z * k * n;
            auto* c_base = c + blockIdx.z * m * n;
            auto acc = AccT{0};
            for (auto kt = size_t{0}; kt * tile < k; ++kt) {
                const auto a_kk = kt * tile + threadIdx.x;
                a_tile[threadIdx.y][threadIdx.x] = row < m && a_kk < k ? a_base[row * k + a_kk] : T{0};
                const auto b_kk = kt * tile + threadIdx.y;
                b_tile[threadIdx.y][threadIdx.x] = b_kk < k && col < n ? b_base[b_kk * n + col] : T{0};
                __syncthreads();
                for (auto e = size_t{0}; e < tile; ++e) {
                    acc += static_cast<AccT>(static_cast<UT>(a_tile[threadIdx.y][e])) *
                           static_cast<AccT>(static_cast<UT>(b_tile[e][threadIdx.x]));
                }
                __syncthreads();
            }
            if (row < m && col < n) {
                c_base[row * n + col] = static_cast<T>(acc);
            }
        }

        /**
         * Float32/Float64 ride the typed strided-batched GEMM. Float16/BFloat16 ride gemm_ex with FP32 compute,
         * matching the CPU conversion-pathway kernels. Row-major C = A*B runs as the column-major C^T = B^T * A^T,
         * so the A and B buffers swap roles and no transpose is needed.
         * @brief Strided-batched GEMM for the native float dtypes via rocBLAS.
         * @param a_data Row-major (batch x)m x k left operand (device).
         * @param b_data Row-major (batch x)k x n right operand (device).
         * @param c_data Row-major (batch x)m x n output (device).
         * @param batch Batch count (1 for rank-2).
         * @param m Rows of a/c.
         * @param k Inner dimension.
         * @param n Columns of b/c.
         * @param dtype Element dtype of all three buffers.
         */
        __host__ void launch_gemm_rocblas(const void* a_data, const void* b_data, void* c_data,
                                 const size_t batch, const size_t m, const size_t k, const size_t n, const DataTypes::DType dtype) {
            const auto handle = get_rocblas_handle();
            const auto blas_m = static_cast<int32_t>(n);
            const auto blas_n = static_cast<int32_t>(m);
            const auto blas_k = static_cast<int32_t>(k);
            const auto blas_lda = static_cast<int32_t>(n);
            const auto blas_ldb = static_cast<int32_t>(k);
            const auto blas_ldc = static_cast<int32_t>(n);
            const auto stride_a = static_cast<int64_t>(k * n); // blas_A aliases the B buffer
            const auto stride_b = static_cast<int64_t>(m * k); // blas_B aliases the A buffer
            const auto stride_c = static_cast<int64_t>(m * n);
            const auto batch_count = static_cast<int32_t>(batch);
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: {
                    constexpr float32_t alpha = 1.0f;
                    constexpr float32_t beta = 0.0f;
                    ROCBLAS_CHECK_ERROR(rocblas_sgemm_strided_batched(handle, rocblas_operation_none, rocblas_operation_none,
                                                                      blas_m, blas_n, blas_k, &alpha,
                                                                      static_cast<const float32_t*>(b_data), blas_lda, stride_a,
                                                                      static_cast<const float32_t*>(a_data), blas_ldb, stride_b,
                                                                      &beta, static_cast<float32_t*>(c_data), blas_ldc, stride_c,
                                                                      batch_count));
                    break;
                }
                case DType::Float64: {
                    constexpr float64_t alpha = 1.0;
                    constexpr float64_t beta = 0.0;
                    ROCBLAS_CHECK_ERROR(rocblas_dgemm_strided_batched(handle, rocblas_operation_none, rocblas_operation_none,
                                                                      blas_m, blas_n, blas_k, &alpha,
                                                                      static_cast<const float64_t*>(b_data), blas_lda, stride_a,
                                                                      static_cast<const float64_t*>(a_data), blas_ldb, stride_b,
                                                                      &beta, static_cast<float64_t*>(c_data), blas_ldc, stride_c,
                                                                      batch_count));
                    break;
                }
                case DType::Float16:
                case DType::BFloat16: {
                    // gemm_ex with FP32 compute, matching the CPU conversion-pathway
                    // kernels and the cuBLAS GemmStridedBatchedEx COMPUTE_32F path.
                    constexpr float32_t alpha = 1.0f;
                    constexpr float32_t beta = 0.0f;
                    const auto lane_type = dtype == DType::Float16 ? rocblas_datatype_f16_r : rocblas_datatype_bf16_r;
                    ROCBLAS_CHECK_ERROR(rocblas_gemm_strided_batched_ex(handle, rocblas_operation_none, rocblas_operation_none,
                                                                        blas_m, blas_n, blas_k, &alpha,
                                                                        b_data, lane_type, blas_lda, stride_a,
                                                                        a_data, lane_type, blas_ldb, stride_b,
                                                                        &beta,
                                                                        c_data, lane_type, blas_ldc, stride_c,
                                                                        c_data, lane_type, blas_ldc, stride_c,
                                                                        batch_count,
                                                                        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
                    break;
                }
                [[unlikely]] default: {
                    throw ErrorHandling::NotImplemented();
                }
            }
        }

        /**
         * gemm_ex with 8-bit inputs and 32-bit output under 32-bit integer compute, into worker-local int32 scratch,
         * then a narrowing kernel realizes the contract's mod-2^8 store. Same row-major-via-column-major aliasing as the float path.
         * @brief Int8 strided-batched GEMM via rocBLAS i32 compute.
         * @param a_data Row-major (batch x)m x k left operand (device).
         * @param b_data Row-major (batch x)k x n right operand (device).
         * @param c_data Row-major (batch x)m x n output (device).
         * @param batch Batch count (1 for rank-2).
         * @param m Rows of a/c.
         * @param k Inner dimension.
         * @param n Columns of b/c.
         */
        __host__ void launch_gemm_i8_rocblas(const void* a_data, const void* b_data, void* c_data,
                                    const size_t batch, const size_t m, const size_t k, const size_t n) {
            const auto c_elems = batch * m * n;
            if (c_elems == 0) {
                return;
            }
            int32_t* scratch = nullptr;
            HIP_CHECK_ERROR(hipMalloc(&scratch, c_elems * sizeof(int32_t)));
            // compute_type i32 -> alpha/beta are int32 scalars.
            constexpr int32_t alpha = 1;
            constexpr int32_t beta = 0;
            ROCBLAS_CHECK_ERROR(rocblas_gemm_strided_batched_ex(get_rocblas_handle(), rocblas_operation_none, rocblas_operation_none,
                                                                static_cast<int32_t>(n), static_cast<int32_t>(m), static_cast<int32_t>(k), &alpha,
                                                                b_data, rocblas_datatype_i8_r, static_cast<int32_t>(n), static_cast<int64_t>(k * n),
                                                                a_data, rocblas_datatype_i8_r, static_cast<int32_t>(k), static_cast<int64_t>(m * k),
                                                                &beta,
                                                                scratch, rocblas_datatype_i32_r, static_cast<int32_t>(n), static_cast<int64_t>(m * n),
                                                                scratch, rocblas_datatype_i32_r, static_cast<int32_t>(n), static_cast<int64_t>(m * n),
                                                                static_cast<int32_t>(batch),
                                                                rocblas_datatype_i32_r, rocblas_gemm_algo_standard, 0, 0));
            constexpr size_t threads = 256;
            narrow_i32_to_i8_kernel<<<(c_elems + threads - 1) / threads, threads>>>(scratch, static_cast<int8_t*>(c_data), c_elems);
            HIP_CHECK_ERROR(hipGetLastError());
            HIP_CHECK_ERROR(hipFree(scratch));
        }

        /**
         * Grid maps x -> column tiles, y -> row tiles, z -> batch. Empty products return without launching.
         * @brief Fallback integer GEMM launcher (non-int8 dtypes).
         * @param a_data Row-major (batch x)m x k left operand (device).
         * @param b_data Row-major (batch x)k x n right operand (device).
         * @param c_data Row-major (batch x)m x n output (device).
         * @param batch Batch count (1 for rank-2).
         * @param m Rows of a/c.
         * @param k Inner dimension.
         * @param n Columns of b/c.
         */
        template<typename T>
        __host__ void launch_gemm_int_fallback(const void* a_data, const void* b_data, void* c_data,
                                      const size_t batch, const size_t m, const size_t k, const size_t n) {
            if (batch == 0 || m == 0 || n == 0) {
                return;
            }
            constexpr size_t tile = 16;
            const dim3 block(static_cast<uint32_t>(tile), static_cast<uint32_t>(tile), uint32_t{1});
            const dim3 grid(static_cast<uint32_t>((n + tile - 1) / tile),
                            static_cast<uint32_t>((m + tile - 1) / tile),
                            static_cast<uint32_t>(batch));
            gemm_int_fallback_kernel<T><<<grid, block>>>(static_cast<const T*>(a_data), static_cast<const T*>(b_data), static_cast<T*>(c_data), m, k, n);
            HIP_CHECK_ERROR(hipGetLastError());
        }

        /**
         * @brief Sign-extended lo nibble (bits [3:0]) of a packed int4x2 byte, as int8_t.
         * @param v The packed byte.
         * @return The sign-extended lo nibble in [-8, 7].
         */
        __device__ __forceinline__ int8_t dev_nib_lo(const int4x2_t v) {
            const auto nib = static_cast<int8_t>(*reinterpret_cast<const uint8_t*>(&v) & 0x0Fu);
            return nib > 7 ? static_cast<int8_t>(nib - 16) : nib;
        }
        /**
         * @brief Sign-extended hi nibble (bits [7:4]) of a packed int4x2 byte, as int8_t.
         * @param v The packed byte.
         * @return The sign-extended hi nibble in [-8, 7].
         */
        __device__ __forceinline__ int8_t dev_nib_hi(const int4x2_t v) {
            const auto nib = static_cast<int8_t>(*reinterpret_cast<const uint8_t*>(&v) >> 4);
            return nib > 7 ? static_cast<int8_t>(nib - 16) : nib;
        }

        /**
         * Per-lane modular arithmetic: two independent 4-bit lanes per element, each accumulated in uint32_t
         * and wrapped to the nibble on store. int4x2_t cannot ride gemm_int_fallback_kernel, as an enum class has no arithmetic operators.
         * One 16x16 C tile per block, batch from blockIdx.z.
         * @brief Tiled shared-memory int4x2 GEMM (per-lane wrap semantics).
         * @param a Row-major (batch x)m x k left operand (device).
         * @param b Row-major (batch x)k x n right operand (device).
         * @param c Row-major (batch x)m x n output (device).
         * @param m Rows of a/c.
         * @param k Inner dimension.
         * @param n Columns of b/c.
         */
        __global__ void gemm_int4x2_fallback_kernel(const int4x2_t* __restrict__ a, const int4x2_t* __restrict__ b, int4x2_t* __restrict__ c,
                                                    const size_t m, const size_t k, const size_t n) {
            constexpr size_t tile = 16;
            __shared__ int4x2_t a_tile[tile][tile];
            __shared__ int4x2_t b_tile[tile][tile];
            const auto row = blockIdx.y * tile + threadIdx.y;
            const auto col = blockIdx.x * tile + threadIdx.x;
            const auto* a_base = a + blockIdx.z * m * k;
            const auto* b_base = b + blockIdx.z * k * n;
            auto* c_base = c + blockIdx.z * m * n;
            auto acc_lo = uint32_t{0};
            auto acc_hi = uint32_t{0};
            for (auto kt = size_t{0}; kt * tile < k; ++kt) {
                const auto a_kk = kt * tile + threadIdx.x;
                a_tile[threadIdx.y][threadIdx.x] = row < m && a_kk < k ? a_base[row * k + a_kk] : int4x2_t{0};
                const auto b_kk = kt * tile + threadIdx.y;
                b_tile[threadIdx.y][threadIdx.x] = b_kk < k && col < n ? b_base[b_kk * n + col] : int4x2_t{0};
                __syncthreads();
                for (auto e = size_t{0}; e < tile; ++e) {
                    const auto av = a_tile[threadIdx.y][e];
                    const auto bv = b_tile[e][threadIdx.x];
                    acc_lo += static_cast<uint32_t>(static_cast<int32_t>(dev_nib_lo(av)) * static_cast<int32_t>(dev_nib_lo(bv)));
                    acc_hi += static_cast<uint32_t>(static_cast<int32_t>(dev_nib_hi(av)) * static_cast<int32_t>(dev_nib_hi(bv)));
                }
                __syncthreads();
            }
            if (row < m && col < n) {
                const auto packed = static_cast<uint8_t>((acc_lo & 0x0Fu) | ((acc_hi & 0x0Fu) << 4));
                int4x2_t out_v{};
                *reinterpret_cast<uint8_t*>(&out_v) = packed;
                c_base[row * n + col] = out_v;
            }
        }

        /**
         * Same tiling and limits as launch_gemm_int_fallback.
         * @brief Fallback int4x2 GEMM launcher.
         * @param a_data Row-major (batch x)m x k left operand (device).
         * @param b_data Row-major (batch x)k x n right operand (device).
         * @param c_data Row-major (batch x)m x n output (device).
         * @param batch Batch count (1 for rank-2).
         * @param m Rows of a/c.
         * @param k Inner dimension.
         * @param n Columns of b/c.
         */
        __host__ void launch_gemm_int4x2_fallback(const void* a_data, const void* b_data, void* c_data,
                                         const size_t batch, const size_t m, const size_t k, const size_t n) {
            if (batch == 0 || m == 0 || n == 0) {
                return;
            }
            constexpr size_t tile = 16;
            const dim3 block(static_cast<uint32_t>(tile), static_cast<uint32_t>(tile), uint32_t{1});
            const dim3 grid(static_cast<uint32_t>((n + tile - 1) / tile),
                            static_cast<uint32_t>((m + tile - 1) / tile),
                            static_cast<uint32_t>(batch));
            gemm_int4x2_fallback_kernel<<<grid, block>>>(static_cast<const int4x2_t*>(a_data), static_cast<const int4x2_t*>(b_data), static_cast<int4x2_t*>(c_data), m, k, n);
            HIP_CHECK_ERROR(hipGetLastError());
        }

        /**
         * Deinterleave both operands into per-lane planes, run one strided-batched FP16/BF16 gemm_ex (FP32 compute)
         * with batchCount = 2*batch over the planes, then interleave the result lanes back into the packed output.
         * Scratch is worker-local malloc/free, never the tensor convert_buffer, whose swap semantics belong to convert_dtype.
         * @brief Packed (fp16x2/bf16x2) strided-batched GEMM via lane planes.
         * @param a_data Row-major (batch x)m x k left operand (device).
         * @param b_data Row-major (batch x)k x n right operand (device).
         * @param c_data Row-major (batch x)m x n output (device).
         * @param batch Batch count (1 for rank-2).
         * @param m Rows of a/c.
         * @param k Inner dimension.
         * @param n Columns of b/c.
         */
        template<typename P, typename H>
        __host__ void launch_gemm_packed_rocblas(const void* a_data, const void* b_data, void* c_data,
                                        const size_t batch, const size_t m, const size_t k, const size_t n) {
            const auto a_elems = batch * m * k;
            const auto b_elems = batch * k * n;
            const auto c_elems = batch * m * n;
            H* scratch = nullptr;
            HIP_CHECK_ERROR(hipMalloc(&scratch, 2 * (a_elems + b_elems + c_elems) * sizeof(H)));
            auto* a_lanes = scratch;
            auto* b_lanes = scratch + 2 * a_elems;
            auto* c_lanes = b_lanes + 2 * b_elems;

            constexpr size_t threads = 256;
            /**
             * Ceil-divides numel by the block size to get the 1D grid extent.
             * @note Lambda subcall. Self-execution prohibited.
             * @brief Grid size for a 1D launch over numel elements.
             * @param numel Element count.
             * @return Block count.
             */
            const auto grid_for = [](const size_t numel) { return (numel + threads - 1) / threads; };
            if (a_elems > 0) {
                deinterleave_kernel<P, H><<<grid_for(a_elems), threads>>>(static_cast<const P*>(a_data), a_lanes, a_elems, m * k);
            }
            if (b_elems > 0) {
                deinterleave_kernel<P, H><<<grid_for(b_elems), threads>>>(static_cast<const P*>(b_data), b_lanes, b_elems, k * n);
            }

            constexpr float32_t alpha = 1.0f;
            constexpr float32_t beta = 0.0f;
            constexpr auto lane_type = std::is_same_v<H, float16_t> ? rocblas_datatype_f16_r : rocblas_datatype_bf16_r;
            ROCBLAS_CHECK_ERROR(rocblas_gemm_strided_batched_ex(get_rocblas_handle(), rocblas_operation_none, rocblas_operation_none,
                                                                static_cast<int32_t>(n), static_cast<int32_t>(m), static_cast<int32_t>(k), &alpha,
                                                                b_lanes, lane_type, static_cast<int32_t>(n), static_cast<int64_t>(k * n),
                                                                a_lanes, lane_type, static_cast<int32_t>(k), static_cast<int64_t>(m * k),
                                                                &beta,
                                                                c_lanes, lane_type, static_cast<int32_t>(n), static_cast<int64_t>(m * n),
                                                                c_lanes, lane_type, static_cast<int32_t>(n), static_cast<int64_t>(m * n),
                                                                static_cast<int32_t>(2 * batch),
                                                                rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));

            if (c_elems > 0) {
                interleave_kernel<P, H><<<grid_for(c_elems), threads>>>(c_lanes, static_cast<P*>(c_data), c_elems, m * n);
            }
            HIP_CHECK_ERROR(hipGetLastError());
            // hipFree implicitly synchronizes the default stream, so the
            // deinterleave/GEMM/interleave chain above has finished before the
            // scratch allocation is released.
            HIP_CHECK_ERROR(hipFree(scratch));
        }

        /**
         * Floats ride rocBLAS directly, packed dtypes ride the lane-plane path, Int8 rides i32 compute plus narrowing,
         * and the remaining integers use the tiled fallback kernel.
         * @brief Dtype support probe for the HIP matmul path.
         * @param dtype Dtype to probe.
         * @return T/F - a HIP matmul path exists for the dtype.
         */
        __host__ bool dtype_has_gpu_gemm(const DataTypes::DType dtype) {
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32:
                case DType::Float64:
                case DType::Float16:
                case DType::BFloat16:
                case DType::Float16x2:
                case DType::BFloat16x2:
                case DType::Int8:
                case DType::UInt8:
                case DType::Int16:
                case DType::UInt16:
                case DType::Int32:
                case DType::UInt32:
                case DType::Int64:
                case DType::UInt64:
                case DType::Int4x2:
                    return true;
                default:
                    return false;
            }
        }

        /**
         * Everything except Int8, which rides rocBLAS i32 compute.
         * @brief Dtype probe for the tiled integer fallback kernels.
         * @param dtype Dtype to probe.
         * @return T/F - the dtype is served by the tiled fallback kernels.
         */
        __host__ bool dtype_uses_int_fallback(const DataTypes::DType dtype) {
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::UInt8:
                case DType::Int16:
                case DType::UInt16:
                case DType::Int32:
                case DType::UInt32:
                case DType::Int64:
                case DType::UInt64:
                case DType::Int4x2:
                    return true;
                default:
                    return false;
            }
        }
    }

    /**
     * Validates device placement, dtype support, and rocBLAS 32-bit indexing limits, then queues the dtype's launcher onto the thread pool.
     * Errors materialize at sync().
     * @brief HIP backend entry point for matmul.
     * @param a Left operand, m x k or batch x m x k.
     * @param b Right operand, k x n or batch x k x n.
     * @param out Preallocated output, m x n or batch x m x n.
     */
    __host__ void matmul_gpu(const Tensor& a, const Tensor& b, Tensor& out) {
        if (a.device != Device::GPU || b.device != Device::GPU || out.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(a.device));
        }
        if (!dtype_has_gpu_gemm(a.dtype)) [[unlikely]] {
            throw ErrorHandling::NotImplemented();
        }
        const auto packed = a.dtype == DataTypes::DType::Float16x2 || a.dtype == DataTypes::DType::BFloat16x2;
        const auto batch = a.ndim == 3 ? a.shape[0] : size_t{1};
        const auto m = a.shape[a.ndim == 3 ? 1 : 0];
        const auto k = a.shape[a.ndim == 3 ? 2 : 1];
        const auto n = b.shape[b.ndim == 3 ? 2 : 1];
        const auto int32_max = static_cast<size_t>(std::numeric_limits<int32_t>::max());
        if (m > int32_max || k > int32_max || n > int32_max) {
            throw ErrorHandling::DimensionTooLarge("m/k/n exceed rocBLAS 32-bit indexing limits");
        }
        if ((packed ? 2 * batch : batch) > int32_max) {
            throw ErrorHandling::DimensionTooLarge("batch count exceeds rocBLAS 32-bit indexing limits");
        }
        // The fallback kernel puts the batch index on grid.z, whose launch
        // limit is 65535 on every supported GPU.
        constexpr size_t grid_z_max = 65535;
        if (dtype_uses_int_fallback(a.dtype) && batch > grid_z_max) {
            throw ErrorHandling::DimensionTooLarge("batch count exceeds the GPU grid.z launch limit (65535)");
        }
        auto a_data = a.data;
        auto b_data = b.data;
        auto c_data = out.data;
        const auto dtype = a.dtype;
        /**
         * Dispatches one queued matmul onto the dtype's HIP launcher. Raw pointers and dims are captured by value.
         * @note Lambda subcall. Self-execution prohibited.
         * @brief GPU matmul task body.
         */
        // Raw pointers + dims captured by value: the future lands on all three
        // tensors, so buffer teardown/view re-homing always drains this first.
        const auto worker = [a_data, b_data, c_data, batch, m, k, n, dtype] {
            switch (dtype) {
                case DataTypes::DType::Float16x2: {
                    launch_gemm_packed_rocblas<float16x2_t, float16_t>(a_data, b_data, c_data, batch, m, k, n);
                    break;
                }
                case DataTypes::DType::BFloat16x2: {
                    launch_gemm_packed_rocblas<bfloat16x2_t, bfloat16_t>(a_data, b_data, c_data, batch, m, k, n);
                    break;
                }
                case DataTypes::DType::Int8: {
                    launch_gemm_i8_rocblas(a_data, b_data, c_data, batch, m, k, n);
                    break;
                }
                case DataTypes::DType::UInt8: {
                    launch_gemm_int_fallback<uint8_t>(a_data, b_data, c_data, batch, m, k, n);
                    break;
                }
                case DataTypes::DType::Int16: {
                    launch_gemm_int_fallback<int16_t>(a_data, b_data, c_data, batch, m, k, n);
                    break;
                }
                case DataTypes::DType::UInt16: {
                    launch_gemm_int_fallback<uint16_t>(a_data, b_data, c_data, batch, m, k, n);
                    break;
                }
                case DataTypes::DType::Int32: {
                    launch_gemm_int_fallback<int32_t>(a_data, b_data, c_data, batch, m, k, n);
                    break;
                }
                case DataTypes::DType::UInt32: {
                    launch_gemm_int_fallback<uint32_t>(a_data, b_data, c_data, batch, m, k, n);
                    break;
                }
                case DataTypes::DType::Int64: {
                    launch_gemm_int_fallback<int64_t>(a_data, b_data, c_data, batch, m, k, n);
                    break;
                }
                case DataTypes::DType::UInt64: {
                    launch_gemm_int_fallback<uint64_t>(a_data, b_data, c_data, batch, m, k, n);
                    break;
                }
                case DataTypes::DType::Int4x2: {
                    launch_gemm_int4x2_fallback(a_data, b_data, c_data, batch, m, k, n);
                    break;
                }
                default: {
                    launch_gemm_rocblas(a_data, b_data, c_data, batch, m, k, n, dtype);
                    break;
                }
            }
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        a.give_future(future);
        b.give_future(future);
        out.give_future(future);
    }
}
