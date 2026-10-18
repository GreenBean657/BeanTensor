/**
 * @file ComplexOpsCUDA.cu
 * @author GreenBean
 * @brief CUDA backend for complex tensor ops: bias_add, relu, softmax_ce.
 * Divergent twin of Math/ComplexOps/ComplexOpsHIP.cpp — the two backends are
 * written separately per project rule (no HIP-only unification), but must stay
 * behaviorally identical: same fp32/fp64 compute widths, same RNE narrowing,
 * same NaN propagation, same wrap integer adds.
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

#include "Math/ComplexOps/ComplexOps.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    namespace {
        /**
         * Also the shared-memory reduction width of the softmax_ce kernels (power of two required).
         * @brief Threads per block for every complex-op kernel.
         */
        constexpr size_t BT_GPU_THREADS = 256;
        /**
         * Grid-stride loops cover any remainder, so large inputs never need more blocks than this.
         * @brief Upper bound on the block count of one launch.
         */
        constexpr size_t BT_GPU_MAX_BLOCKS = 1024;

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

        // --------------------------- bias_add kernel ---------------------------

        /**
         * Elementwise bias add: out[i] = in[i] + bias[i % cols]. Float kinds
         * compute in their storage width's natural accumulator (fp32 for
         * fp16/bf16/fp32, fp64 for fp64) and narrow RNE on store, exactly
         * matching the CPU kernels. Integers add modularly (wrap) through the
         * unsigned twin; int4x2 wraps per nibble.
         * @brief Elementwise bias_add kernel (all dtypes).
         * @param in Input tensor data, numel elements.
         * @param bias Bias vector, cols elements.
         * @param out Output tensor data, numel elements.
         * @param cols Last-dimension size; bias index is i % cols.
         * @param numel Total element count of in/out.
         */
        template<typename T>
        __global__ void bias_add_kernel(const T* __restrict__ in, const T* __restrict__ bias, T* __restrict__ out,
                                        const size_t cols, const size_t numel) {
            const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t stride = blockDim.x * gridDim.x;
            for (size_t i = tid; i < numel; i += stride) {
                const auto col = i % cols;
                if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                    out[i] = in[i] + bias[col];
                } else if constexpr (std::is_same_v<T, float16_t>) {
                    out[i] = __float2half_rn(__half2float(in[i]) + __half2float(bias[col]));
                } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                    out[i] = dev_narrow_bf16(__bfloat162float(in[i]) + __bfloat162float(bias[col]));
                } else if constexpr (std::is_same_v<T, float16x2_t>) {
                    const auto iv = in[i];
                    const auto bv = bias[col];
                    const auto r_lo = __float2half_rn(__half2float(__low2half(iv)) + __half2float(__low2half(bv)));
                    const auto r_hi = __float2half_rn(__half2float(__high2half(iv)) + __half2float(__high2half(bv)));
                    out[i] = __halves2half2(r_lo, r_hi);
                } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                    const auto iv = in[i];
                    const auto bv = bias[col];
                    const auto r_lo = dev_narrow_bf16(__bfloat162float(__low2bfloat16(iv)) + __bfloat162float(__low2bfloat16(bv)));
                    const auto r_hi = dev_narrow_bf16(__bfloat162float(__high2bfloat16(iv)) + __bfloat162float(__high2bfloat16(bv)));
                    out[i] = __halves2bfloat162(r_lo, r_hi);
                } else if constexpr (std::is_same_v<T, int4x2_t>) {
                    out[i] = dev_pack_int4(dev_int4_lo(in[i]) + dev_int4_lo(bias[col]),
                                           dev_int4_hi(in[i]) + dev_int4_hi(bias[col]));
                } else {
                    using U = std::make_unsigned_t<T>;
                    out[i] = static_cast<T>(static_cast<U>(static_cast<U>(in[i]) + static_cast<U>(bias[col])));
                }
            }
        }

        /**
         * Skips the launch for empty tensors; the kernel is asynchronous, the caller syncs.
         * @brief Host-side bias_add launch.
         * @param in Input tensor data, numel elements.
         * @param bias Bias vector, cols elements.
         * @param out Output tensor data, numel elements.
         * @param cols Last-dimension size; bias index is i % cols.
         * @param numel Total element count of in/out.
         */
        template<typename T>
        __host__ void run_bias_add(const T* in, const T* bias, T* out, const size_t cols, const size_t numel) {
            if (numel > 0) {
                bias_add_kernel<T><<<grid_for(numel), BT_GPU_THREADS>>>(in, bias, out, cols, numel);
            }
        }

        // --------------------------- relu kernel ---------------------------

        /**
         * Elementwise relu: out[i] = in[i] < 0 ? 0 : in[i]. NaN propagates (the
         * comparison is false for NaN), -0.0 survives; half kinds widen to fp32,
         * relu, and narrow RNE; packed dtypes relu per lane; unsigned integers
         * copy through.
         * @brief Elementwise relu kernel (all dtypes).
         * @param in Input tensor data, numel elements.
         * @param out Output tensor data, numel elements.
         * @param numel Total element count of in/out.
         */
        template<typename T>
        __global__ void relu_kernel(const T* __restrict__ in, T* __restrict__ out, const size_t numel) {
            const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t stride = blockDim.x * gridDim.x;
            for (size_t i = tid; i < numel; i += stride) {
                if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                    out[i] = in[i] < T{0} ? T{0} : in[i];
                } else if constexpr (std::is_same_v<T, float16_t>) {
                    const auto wide = __half2float(in[i]);
                    out[i] = __float2half_rn(wide < 0.0f ? 0.0f : wide);
                } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                    const auto wide = __bfloat162float(in[i]);
                    out[i] = dev_narrow_bf16(wide < 0.0f ? 0.0f : wide);
                } else if constexpr (std::is_same_v<T, float16x2_t>) {
                    const auto v = in[i];
                    const auto lo = __half2float(__low2half(v));
                    const auto hi = __half2float(__high2half(v));
                    out[i] = __halves2half2(__float2half_rn(lo < 0.0f ? 0.0f : lo),
                                            __float2half_rn(hi < 0.0f ? 0.0f : hi));
                } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                    const auto v = in[i];
                    const auto lo = __bfloat162float(__low2bfloat16(v));
                    const auto hi = __bfloat162float(__high2bfloat16(v));
                    out[i] = __halves2bfloat162(dev_narrow_bf16(lo < 0.0f ? 0.0f : lo),
                                                dev_narrow_bf16(hi < 0.0f ? 0.0f : hi));
                } else if constexpr (std::is_same_v<T, int4x2_t>) {
                    const auto lo = dev_int4_lo(in[i]);
                    const auto hi = dev_int4_hi(in[i]);
                    out[i] = dev_pack_int4(lo < 0 ? 0 : lo, hi < 0 ? 0 : hi);
                } else if constexpr (std::is_signed_v<T>) {
                    out[i] = in[i] < T{0} ? T{0} : in[i];
                } else {
                    out[i] = in[i];
                }
            }
        }

        /**
         * Skips the launch for empty tensors; the kernel is asynchronous, the caller syncs.
         * @brief Host-side relu launch.
         * @param in Input tensor data, numel elements.
         * @param out Output tensor data, numel elements.
         * @param numel Total element count of in/out.
         */
        template<typename T>
        __host__ void run_relu(const T* in, T* out, const size_t numel) {
            if (numel > 0) {
                relu_kernel<T><<<grid_for(numel), BT_GPU_THREADS>>>(in, out, numel);
            }
        }

        // --------------------------- softmax_ce kernels ---------------------------

        /**
         * Literal constants, since nvcc rejects numeric_limits in device code.
         * @brief Lowest finite value of an accumulator type.
         * @return -FLT_MAX for fp32, -DBL_MAX for fp64.
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
         * Overload pair; fp32 maps to expf, fp64 to exp.
         * @brief exp() in the accumulator width.
         * @param v Exponent.
         * @return e^v in the accumulator width.
         */
        __device__ __forceinline__ float32_t acc_exp(const float32_t v) { return expf(v); }
        __device__ __forceinline__ float64_t acc_exp(const float64_t v) { return exp(v); }

        /**
         * Overload pair; fp32 maps to logf, fp64 to log.
         * @brief log() in the accumulator width.
         * @param v Argument.
         * @return Natural log of v in the accumulator width.
         */
        __device__ __forceinline__ float32_t acc_log(const float32_t v) { return logf(v); }
        __device__ __forceinline__ float64_t acc_log(const float64_t v) { return log(v); }

        /**
         * fp16/bf16 and packed x2 (lanes unpack as consecutive classes) widen to fp32.
         * fp64 and the integer dtypes (int4x2 unpacks nibbles as consecutive classes) widen to fp64.
         * @brief Widen class j of a logits row to the accumulator.
         * @param row First element of the logits row.
         * @param j Effective class index within the row.
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
         * @param targets Target class indices, one per row.
         * @param r Row index.
         * @return Signed class index; the caller bounds-checks it.
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
         * Numerically-stable row softmax cross-entropy, one block per row:
         * block-reduce the widened max, block-reduce sum(exp(x - m)), then
         * thread 0 writes loss = m + log(sum) - x_target to row_losses in fp64.
         * A NaN logit poisons the sum (max ignores NaN), so the row loss is NaN.
         * An out-of-range target sets *err and stores a 0 placeholder — the host
         * throws IndexOutOfRange after sync (device code cannot throw).
         * @brief softmax_ce per-row kernel (all dtypes).
         * @param logits Row-major logits, rows x cols_store.
         * @param targets Target class indices, one per row.
         * @param row_losses Per-row fp64 loss output, rows elements.
         * @param err Device flag set to 1 when any target is out of range.
         * @param rows Row count; launched with one block per row.
         * @param cols_store Stored element count per row.
         * @param cols_eff Effective class count per row (doubled for packed dtypes).
         */
        template<typename T, typename I, typename Acc>
        __global__ void softmax_ce_rows_kernel(const T* __restrict__ logits, const I* __restrict__ targets,
                                               float64_t* __restrict__ row_losses, uint32_t* __restrict__ err,
                                               const size_t rows, const size_t cols_store, const size_t cols_eff) {
            const auto r = blockIdx.x;
            if (r >= rows) {
                return;
            }
            const auto* row = logits + r * cols_store;
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
            if (tid == 0) {
                const auto t = sce_target(targets, r);
                if (t < 0 || static_cast<uint64_t>(t) >= cols_eff) {
                    *err = 1u;
                    row_losses[r] = 0.0;
                    return;
                }
                const auto loss = m + acc_log(shared[0]) - sce_widen<T, Acc>(row, static_cast<size_t>(t));
                row_losses[r] = static_cast<float64_t>(loss);
            }
        }

        /**
         * Single block reduces the per-row fp64 losses to the mean and stores it
         * narrowed into out[0]: floats RNE (bf16 with the CPU-matching flush),
         * packed dtypes duplicate into both lanes, integers clamp to the dtype
         * range and truncate — a NaN mean stores 0 (there is no integer NaN).
         * lo/hi carry the integer dtype bounds and are ignored for float kinds.
         * @brief softmax_ce finalize kernel.
         * @param row_losses Per-row fp64 losses, rows elements.
         * @param rows Row count to average over.
         * @param lo Integer dtype lower bound; ignored for float kinds.
         * @param hi Integer dtype upper bound; ignored for float kinds.
         * @param out Single-element output tensor data.
         */
        template<typename T>
        __global__ void softmax_ce_final_kernel(const float64_t* __restrict__ row_losses, const size_t rows,
                                                const float64_t lo, const float64_t hi, T* __restrict__ out) {
            __shared__ float64_t shared[BT_GPU_THREADS];
            const auto tid = threadIdx.x;
            auto acc = 0.0;
            for (auto i = tid; i < rows; i += BT_GPU_THREADS) {
                acc += row_losses[i];
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
                const auto mean = shared[0] / static_cast<float64_t>(rows);
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
                    out[0] = dev_pack_int4(n, n);
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
         * Scratch is worker-local cudaMalloc/cudaFree. The D2H flag copy and the frees sync the default
         * stream, so the result is complete in out before the worker returns.
         * @brief Host-side two-stage softmax_ce.
         * @param logits Row-major logits, rows x cols_store.
         * @param targets Target class indices, one per row.
         * @param rows Row count.
         * @param cols_store Stored element count per row.
         * @param cols_eff Effective class count per row (doubled for packed dtypes).
         * @param out Single-element output tensor data.
         */
        template<typename T, typename I, typename Acc>
        __host__ void run_softmax_ce(const T* logits, const I* targets, const size_t rows,
                            const size_t cols_store, const size_t cols_eff, T* out) {
            // Integer bounds are computed on the host (nvcc rejects
            // numeric_limits in device code).
            auto lo = 0.0;
            auto hi = 0.0;
            if constexpr (std::is_integral_v<T>) {
                lo = static_cast<float64_t>(std::numeric_limits<T>::lowest());
                hi = static_cast<float64_t>(std::numeric_limits<T>::max());
            }
            float64_t* row_losses = nullptr;
            uint32_t* err = nullptr;
            CUDA_CHECK_ERROR(cudaMalloc(&row_losses, rows * sizeof(float64_t)));
            CUDA_CHECK_ERROR(cudaMalloc(&err, sizeof(uint32_t)));
            CUDA_CHECK_ERROR(cudaMemset(err, 0, sizeof(uint32_t)));
            softmax_ce_rows_kernel<T, I, Acc><<<rows, BT_GPU_THREADS>>>(logits, targets, row_losses, err, rows, cols_store, cols_eff);
            softmax_ce_final_kernel<T><<<1, BT_GPU_THREADS>>>(row_losses, rows, lo, hi, out);
            CUDA_CHECK_ERROR(cudaGetLastError());
            auto err_host = uint32_t{0};
            CUDA_CHECK_ERROR(cudaMemcpy(&err_host, err, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK_ERROR(cudaFree(row_losses));
            CUDA_CHECK_ERROR(cudaFree(err));
            if (err_host != 0u) {
                throw ErrorHandling::IndexOutOfRange();
            }
        }

        /**
         * Integer target kinds only; validated by Tensor::softmax_ce before it reaches here.
         * @brief Dispatch the softmax_ce runner over the target dtype.
         * @param logits Row-major logits, rows x cols_store.
         * @param targets Raw target buffer, already offset to the first element.
         * @param t_dtype Target dtype selecting the run_softmax_ce instantiation.
         * @param rows Row count.
         * @param cols_store Stored element count per row.
         * @param cols_eff Effective class count per row (doubled for packed dtypes).
         * @param out Single-element output tensor data.
         */
        template<typename T, typename Acc>
        __host__ void run_sce_targets(const T* logits, const void* targets, const DataTypes::DType t_dtype,
                             const size_t rows, const size_t cols_store, const size_t cols_eff, T* out) {
            using DType = DataTypes::DType;
            switch (t_dtype) {
                case DType::Int8: run_softmax_ce<T, int8_t, Acc>(logits, static_cast<const int8_t*>(targets), rows, cols_store, cols_eff, out); break;
                case DType::UInt8: run_softmax_ce<T, uint8_t, Acc>(logits, static_cast<const uint8_t*>(targets), rows, cols_store, cols_eff, out); break;
                case DType::Int16: run_softmax_ce<T, int16_t, Acc>(logits, static_cast<const int16_t*>(targets), rows, cols_store, cols_eff, out); break;
                case DType::UInt16: run_softmax_ce<T, uint16_t, Acc>(logits, static_cast<const uint16_t*>(targets), rows, cols_store, cols_eff, out); break;
                case DType::Int32: run_softmax_ce<T, int32_t, Acc>(logits, static_cast<const int32_t*>(targets), rows, cols_store, cols_eff, out); break;
                case DType::UInt32: run_softmax_ce<T, uint32_t, Acc>(logits, static_cast<const uint32_t*>(targets), rows, cols_store, cols_eff, out); break;
                case DType::Int64: run_softmax_ce<T, int64_t, Acc>(logits, static_cast<const int64_t*>(targets), rows, cols_store, cols_eff, out); break;
                case DType::UInt64: run_softmax_ce<T, uint64_t, Acc>(logits, static_cast<const uint64_t*>(targets), rows, cols_store, cols_eff, out); break;
                case DType::Int4x2: run_softmax_ce<T, int4x2_t, Acc>(logits, static_cast<const int4x2_t*>(targets), rows, cols_store, cols_eff, out); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
        }
    }

    /**
     * Queues one pool worker that dispatches on dtype and launches the bias_add kernel. The future lands
     * on all three tensors so teardown drains it.
     * @brief GPU bias_add: out[i] = t[i] + bias[i % cols].
     * @param t Input tensor, GPU resident, contiguous.
     * @param bias Rank-1 bias tensor matching the last dimension of t.
     * @param out Output tensor, same shape and dtype as t.
     * @note Throws DirectFunctionCallOnWrongDataSource when any tensor is not on the GPU.
     */
    __host__ void bias_add_gpu(const Tensor& t, const Tensor& bias, Tensor& out) {
        if (t.device != Device::GPU || bias.device != Device::GPU || out.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(t.device));
        }
        const auto* const t_data = t.data;
        const auto* const b_data = bias.data;
        auto* const out_data = out.data;
        const auto t_offset = t.offset;
        const auto b_offset = bias.offset;
        const auto out_offset = out.offset;
        const auto dtype = t.dtype;
        const auto cols = t.shape[t.ndim - 1];
        const auto numel = t.numel;
        // Raw pointers captured by value: the future lands on all three tensors,
        // so buffer teardown always drains this worker first.
        /**
         * Dispatches on dtype to run_bias_add, then syncs the default stream so the future only
         * completes once the buffer writes are done.
         * @brief Pool worker for the bias_add launch.
         * @note Lambda subcall. Self-execution prohibited.
         */
        const auto worker = [=] {
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: run_bias_add(static_cast<const float32_t*>(t_data) + t_offset, static_cast<const float32_t*>(b_data) + b_offset, static_cast<float32_t*>(out_data) + out_offset, cols, numel); break;
                case DType::Float64: run_bias_add(static_cast<const float64_t*>(t_data) + t_offset, static_cast<const float64_t*>(b_data) + b_offset, static_cast<float64_t*>(out_data) + out_offset, cols, numel); break;
                case DType::Float16: run_bias_add(static_cast<const float16_t*>(t_data) + t_offset, static_cast<const float16_t*>(b_data) + b_offset, static_cast<float16_t*>(out_data) + out_offset, cols, numel); break;
                case DType::BFloat16: run_bias_add(static_cast<const bfloat16_t*>(t_data) + t_offset, static_cast<const bfloat16_t*>(b_data) + b_offset, static_cast<bfloat16_t*>(out_data) + out_offset, cols, numel); break;
                case DType::Float16x2: run_bias_add(static_cast<const float16x2_t*>(t_data) + t_offset, static_cast<const float16x2_t*>(b_data) + b_offset, static_cast<float16x2_t*>(out_data) + out_offset, cols, numel); break;
                case DType::BFloat16x2: run_bias_add(static_cast<const bfloat16x2_t*>(t_data) + t_offset, static_cast<const bfloat16x2_t*>(b_data) + b_offset, static_cast<bfloat16x2_t*>(out_data) + out_offset, cols, numel); break;
                case DType::Int8: run_bias_add(static_cast<const int8_t*>(t_data) + t_offset, static_cast<const int8_t*>(b_data) + b_offset, static_cast<int8_t*>(out_data) + out_offset, cols, numel); break;
                case DType::UInt8: run_bias_add(static_cast<const uint8_t*>(t_data) + t_offset, static_cast<const uint8_t*>(b_data) + b_offset, static_cast<uint8_t*>(out_data) + out_offset, cols, numel); break;
                case DType::Int16: run_bias_add(static_cast<const int16_t*>(t_data) + t_offset, static_cast<const int16_t*>(b_data) + b_offset, static_cast<int16_t*>(out_data) + out_offset, cols, numel); break;
                case DType::UInt16: run_bias_add(static_cast<const uint16_t*>(t_data) + t_offset, static_cast<const uint16_t*>(b_data) + b_offset, static_cast<uint16_t*>(out_data) + out_offset, cols, numel); break;
                case DType::Int32: run_bias_add(static_cast<const int32_t*>(t_data) + t_offset, static_cast<const int32_t*>(b_data) + b_offset, static_cast<int32_t*>(out_data) + out_offset, cols, numel); break;
                case DType::UInt32: run_bias_add(static_cast<const uint32_t*>(t_data) + t_offset, static_cast<const uint32_t*>(b_data) + b_offset, static_cast<uint32_t*>(out_data) + out_offset, cols, numel); break;
                case DType::Int64: run_bias_add(static_cast<const int64_t*>(t_data) + t_offset, static_cast<const int64_t*>(b_data) + b_offset, static_cast<int64_t*>(out_data) + out_offset, cols, numel); break;
                case DType::UInt64: run_bias_add(static_cast<const uint64_t*>(t_data) + t_offset, static_cast<const uint64_t*>(b_data) + b_offset, static_cast<uint64_t*>(out_data) + out_offset, cols, numel); break;
                case DType::Int4x2: run_bias_add(static_cast<const int4x2_t*>(t_data) + t_offset, static_cast<const int4x2_t*>(b_data) + b_offset, static_cast<int4x2_t*>(out_data) + out_offset, cols, numel); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
            // The kernel is in-flight when the launch returns; sync so the
            // future only completes once the buffer writes are done.
            CUDA_CHECK_ERROR(cudaStreamSynchronize(nullptr));
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        t.give_future(future);
        bias.give_future(future);
        out.give_future(future);
    }

    /**
     * Queues one pool worker that dispatches on dtype and launches the relu kernel. The future lands
     * on both tensors so teardown drains it.
     * @brief GPU relu: out[i] = t[i] < 0 ? 0 : t[i].
     * @param t Input tensor, GPU resident, contiguous.
     * @param out Output tensor, same shape and dtype as t.
     * @note Throws DirectFunctionCallOnWrongDataSource when any tensor is not on the GPU.
     */
    __host__ void relu_gpu(const Tensor& t, Tensor& out) {
        if (t.device != Device::GPU || out.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(t.device));
        }
        const auto* const t_data = t.data;
        auto* const out_data = out.data;
        const auto t_offset = t.offset;
        const auto out_offset = out.offset;
        const auto dtype = t.dtype;
        const auto numel = t.numel;
        /**
         * Dispatches on dtype to run_relu, then syncs the default stream so the future only
         * completes once the buffer writes are done.
         * @brief Pool worker for the relu launch.
         * @note Lambda subcall. Self-execution prohibited.
         */
        const auto worker = [=] {
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: run_relu(static_cast<const float32_t*>(t_data) + t_offset, static_cast<float32_t*>(out_data) + out_offset, numel); break;
                case DType::Float64: run_relu(static_cast<const float64_t*>(t_data) + t_offset, static_cast<float64_t*>(out_data) + out_offset, numel); break;
                case DType::Float16: run_relu(static_cast<const float16_t*>(t_data) + t_offset, static_cast<float16_t*>(out_data) + out_offset, numel); break;
                case DType::BFloat16: run_relu(static_cast<const bfloat16_t*>(t_data) + t_offset, static_cast<bfloat16_t*>(out_data) + out_offset, numel); break;
                case DType::Float16x2: run_relu(static_cast<const float16x2_t*>(t_data) + t_offset, static_cast<float16x2_t*>(out_data) + out_offset, numel); break;
                case DType::BFloat16x2: run_relu(static_cast<const bfloat16x2_t*>(t_data) + t_offset, static_cast<bfloat16x2_t*>(out_data) + out_offset, numel); break;
                case DType::Int8: run_relu(static_cast<const int8_t*>(t_data) + t_offset, static_cast<int8_t*>(out_data) + out_offset, numel); break;
                case DType::UInt8: run_relu(static_cast<const uint8_t*>(t_data) + t_offset, static_cast<uint8_t*>(out_data) + out_offset, numel); break;
                case DType::Int16: run_relu(static_cast<const int16_t*>(t_data) + t_offset, static_cast<int16_t*>(out_data) + out_offset, numel); break;
                case DType::UInt16: run_relu(static_cast<const uint16_t*>(t_data) + t_offset, static_cast<uint16_t*>(out_data) + out_offset, numel); break;
                case DType::Int32: run_relu(static_cast<const int32_t*>(t_data) + t_offset, static_cast<int32_t*>(out_data) + out_offset, numel); break;
                case DType::UInt32: run_relu(static_cast<const uint32_t*>(t_data) + t_offset, static_cast<uint32_t*>(out_data) + out_offset, numel); break;
                case DType::Int64: run_relu(static_cast<const int64_t*>(t_data) + t_offset, static_cast<int64_t*>(out_data) + out_offset, numel); break;
                case DType::UInt64: run_relu(static_cast<const uint64_t*>(t_data) + t_offset, static_cast<uint64_t*>(out_data) + out_offset, numel); break;
                case DType::Int4x2: run_relu(static_cast<const int4x2_t*>(t_data) + t_offset, static_cast<int4x2_t*>(out_data) + out_offset, numel); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
            CUDA_CHECK_ERROR(cudaStreamSynchronize(nullptr));
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        t.give_future(future);
        out.give_future(future);
    }

    /**
     * Queues one pool worker that dispatches on logits dtype, then target dtype, and runs the two-stage
     * reduction. The future lands on all three tensors so teardown drains it.
     * @brief GPU softmax cross-entropy: mean over rows of logsumexp(row) - row[target].
     * @param t Rank-2 logits tensor (rows x classes), GPU resident, contiguous.
     * @param targets Rank-1 integer class indices, one per row.
     * @param out Single-element output tensor in the logits dtype.
     * @note Throws DirectFunctionCallOnWrongDataSource when any tensor is not on the GPU.
     */
    __host__ void softmax_ce_gpu(const Tensor& t, const Tensor& targets, Tensor& out) {
        if (t.device != Device::GPU || targets.device != Device::GPU || out.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(t.device));
        }
        const auto* const t_data = t.data;
        const auto* const targets_data = targets.data;
        auto* const out_data = out.data;
        const auto t_offset = t.offset;
        const auto targets_offset = targets.offset;
        const auto dtype = t.dtype;
        const auto t_dtype = targets.dtype;
        const auto rows = t.shape[0];
        const auto cols_store = t.shape[1];
        const auto packed = dtype == DataTypes::DType::Float16x2 || dtype == DataTypes::DType::BFloat16x2 || dtype == DataTypes::DType::Int4x2;
        const auto cols_eff = cols_store * (packed ? size_t{2} : size_t{1});
        /**
         * Dispatches on logits dtype to run_sce_targets, which dispatches on target dtype. The runner's
         * D2H copy already synced the stream, so no extra sync is needed.
         * @brief Pool worker for the softmax_ce launch.
         * @note Lambda subcall. Self-execution prohibited.
         */
        const auto worker = [=] {
            using DType = DataTypes::DType;
            const auto* targets_base = static_cast<const uint8_t*>(targets_data) + targets_offset * DataTypes::dtype_bytes(t_dtype);
            switch (dtype) {
                case DType::Float32: run_sce_targets<float32_t, float32_t>(static_cast<const float32_t*>(t_data) + t_offset, targets_base, t_dtype, rows, cols_store, cols_eff, static_cast<float32_t*>(out_data)); break;
                case DType::Float64: run_sce_targets<float64_t, float64_t>(static_cast<const float64_t*>(t_data) + t_offset, targets_base, t_dtype, rows, cols_store, cols_eff, static_cast<float64_t*>(out_data)); break;
                case DType::Float16: run_sce_targets<float16_t, float32_t>(static_cast<const float16_t*>(t_data) + t_offset, targets_base, t_dtype, rows, cols_store, cols_eff, static_cast<float16_t*>(out_data)); break;
                case DType::BFloat16: run_sce_targets<bfloat16_t, float32_t>(static_cast<const bfloat16_t*>(t_data) + t_offset, targets_base, t_dtype, rows, cols_store, cols_eff, static_cast<bfloat16_t*>(out_data)); break;
                case DType::Float16x2: run_sce_targets<float16x2_t, float32_t>(static_cast<const float16x2_t*>(t_data) + t_offset, targets_base, t_dtype, rows, cols_store, cols_eff, static_cast<float16x2_t*>(out_data)); break;
                case DType::BFloat16x2: run_sce_targets<bfloat16x2_t, float32_t>(static_cast<const bfloat16x2_t*>(t_data) + t_offset, targets_base, t_dtype, rows, cols_store, cols_eff, static_cast<bfloat16x2_t*>(out_data)); break;
                case DType::Int8: run_sce_targets<int8_t, float64_t>(static_cast<const int8_t*>(t_data) + t_offset, targets_base, t_dtype, rows, cols_store, cols_eff, static_cast<int8_t*>(out_data)); break;
                case DType::UInt8: run_sce_targets<uint8_t, float64_t>(static_cast<const uint8_t*>(t_data) + t_offset, targets_base, t_dtype, rows, cols_store, cols_eff, static_cast<uint8_t*>(out_data)); break;
                case DType::Int16: run_sce_targets<int16_t, float64_t>(static_cast<const int16_t*>(t_data) + t_offset, targets_base, t_dtype, rows, cols_store, cols_eff, static_cast<int16_t*>(out_data)); break;
                case DType::UInt16: run_sce_targets<uint16_t, float64_t>(static_cast<const uint16_t*>(t_data) + t_offset, targets_base, t_dtype, rows, cols_store, cols_eff, static_cast<uint16_t*>(out_data)); break;
                case DType::Int32: run_sce_targets<int32_t, float64_t>(static_cast<const int32_t*>(t_data) + t_offset, targets_base, t_dtype, rows, cols_store, cols_eff, static_cast<int32_t*>(out_data)); break;
                case DType::UInt32: run_sce_targets<uint32_t, float64_t>(static_cast<const uint32_t*>(t_data) + t_offset, targets_base, t_dtype, rows, cols_store, cols_eff, static_cast<uint32_t*>(out_data)); break;
                case DType::Int64: run_sce_targets<int64_t, float64_t>(static_cast<const int64_t*>(t_data) + t_offset, targets_base, t_dtype, rows, cols_store, cols_eff, static_cast<int64_t*>(out_data)); break;
                case DType::UInt64: run_sce_targets<uint64_t, float64_t>(static_cast<const uint64_t*>(t_data) + t_offset, targets_base, t_dtype, rows, cols_store, cols_eff, static_cast<uint64_t*>(out_data)); break;
                case DType::Int4x2: run_sce_targets<int4x2_t, float64_t>(static_cast<const int4x2_t*>(t_data) + t_offset, targets_base, t_dtype, rows, cols_store, cols_eff, static_cast<int4x2_t*>(out_data)); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
            // run_softmax_ce's D2H memcpy and cudaFree already synced the stream.
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        t.give_future(future);
        targets.give_future(future);
        out.give_future(future);
    }
}
#endif
