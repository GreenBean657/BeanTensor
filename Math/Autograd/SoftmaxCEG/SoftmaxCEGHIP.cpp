/**
 * @file SoftmaxCEGHIP.cpp
 * @author GreenBean
 * @brief HIP backend for the softmax_ce_g op (softmax cross-entropy backward).
 * Divergent twin of Math/Autograd/SoftmaxCEG/SoftmaxCEGCUDA.cu — the two
 * backends are written separately per project rule (no HIP-only unification),
 * but must stay behaviorally identical: same fp32/fp64 compute widths, same
 * RNE narrowing, same NaN handling.
 * @version 0.3
 * @date 2026-10-26
 */
#if defined(USE_HIP)
#include <cstdint>
#include <limits>
#include <type_traits>

#include <hip/hip_runtime.h>

#include "Math/Autograd/SoftmaxCEG/SoftmaxCEG.h"
#include "Math/Autograd/AutogradHipShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    using namespace autograd_gpu;

    namespace {
        /**
         * Lowest finite value of an accumulator type (device code cannot rely on numeric_limits).
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
         * The uint64/int64 maxima round UP to 2^64/2^63 in fp64, so the >=
         * guard returns the true integer max before the static_cast (UB at
         * exactly 2^64/2^63). The bounds are passed as T so the guard can
         * return the exact limit value.
         * @brief Store one softmax_ce_g gradient element.
         * @param v Gradient value in the accumulator width.
         * @param lo Lowest integer value of T (ignored for float T).
         * @param hi Highest integer value of T (ignored for float T).
         * @return v converted to T.
         */
        template<typename T, typename Acc>
        __device__ __forceinline__ T sce_narrow(const Acc v, const T lo, const T hi) {
            if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                return v;
            } else if constexpr (std::is_same_v<T, float16_t>) {
                return __float2half_rn(v);
            } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                return dev_narrow_bf16(v);
            } else {
                const auto lo_f = static_cast<float64_t>(lo);
                const auto hi_f = static_cast<float64_t>(hi);
                const auto cv = sce_clamp(static_cast<float64_t>(v), lo_f, hi_f);
                if (cv <= lo_f) {
                    return lo;
                }
                if (cv >= hi_f) {
                    return hi;
                }
                return static_cast<T>(cv);
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
         * @param lo Lowest integer value of T (value of type T; unused for float T).
         * @param hi Highest integer value of T (value of type T; unused for float T).
         */
        template<typename T, typename I, typename Acc>
        __global__ void softmax_ce_g_rows_kernel(const T* __restrict__ logits, const I* __restrict__ targets,
                                                 const T* __restrict__ dloss, T* __restrict__ dlogits,
                                                 uint32_t* __restrict__ err, const size_t rows,
                                                 const size_t cols_store, const size_t cols_eff,
                                                 const T lo, const T hi) {
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
         * Worker-local hipMalloc buffer that frees on scope exit even when an
         * exception is in flight. The explicit hipFree below still checks for
         * errors on the success path.
         * @brief RAII guard for the softmax_ce_g error flag scratch buffer.
         */
        struct SoftmaxCEGErrGuard {
            uint32_t* ptr = nullptr;
            ~SoftmaxCEGErrGuard() {
                if (ptr != nullptr) {
                    static_cast<void>(hipFree(ptr));
                }
            }
        };

        /**
         * The error flag is worker-local hipMalloc/hipFree; the D2H flag copy syncs the
         * default stream, so dlogits is complete before the worker returns. Integer
         * bounds are computed on the host (device code cannot rely on numeric_limits).
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
            // Integer bounds are computed on the host (device code cannot rely
            // on numeric_limits); T{} for non-integer T, unused.
            auto lo = T{};
            auto hi = T{};
            if constexpr (std::is_integral_v<T>) {
                lo = std::numeric_limits<T>::lowest();
                hi = std::numeric_limits<T>::max();
            }
            uint32_t* err = nullptr;
            HIP_CHECK_ERROR(hipMalloc(&err, sizeof(uint32_t)));
            SoftmaxCEGErrGuard guard{err};
            HIP_CHECK_ERROR(hipMemset(err, 0, sizeof(uint32_t)));
            softmax_ce_g_rows_kernel<T, I, Acc><<<rows, BT_GPU_THREADS>>>(logits, targets, dloss, dlogits, err, rows, cols_store, cols_eff, lo, hi);
            HIP_CHECK_ERROR(hipGetLastError());
            auto err_host = uint32_t{0};
            HIP_CHECK_ERROR(hipMemcpy(&err_host, err, sizeof(uint32_t), hipMemcpyDeviceToHost));
            guard.ptr = nullptr;
            HIP_CHECK_ERROR(hipFree(err));
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
    __host__ void softmax_ce_g_gpu(const Tensor& dloss_grad, const Tensor& logits, const Tensor& targets, const Tensor& dlogits) {
        if (dloss_grad.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(dloss_grad.device));
        }
        if (logits.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(logits.device));
        }
        if (targets.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(targets.device));
        }
        if (dlogits.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(dlogits.device));
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
         * Runs on a CPU thread pool thread (High priority) via the async pool; the host call has already returned. Any exception thrown here (including a failed HIP_CHECK_ERROR) is captured in the shared future and rethrown at sync().
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
            // run_softmax_ce_g's D2H memcpy and hipFree already synced the stream.
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
