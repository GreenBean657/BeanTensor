/**
 * @file BiasAddGCPU.cpp
 * @author GreenBean
 * @brief CPU backend for bias_add backward (column sums of dy), plus the
 * bias_add backward trampoline (bias_add_g_node).
 * @version 0.1
 * @date 2026-10-20
 */
#include <algorithm>
#include <cstdint>
#include <future>
#include <type_traits>
#include <vector>

#include "Math/Autograd/AutogradShared.h"
#include "Math/Autograd/BiasAddG/BiasAddG.h"
#include "Math/Autograd/BiasAddG/BiasAddGKernels.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Tensors::detail {
    using namespace autograd_shared;

    namespace {
        // --------------------------- bias_add_g kernels ---------------------------

        /**
         * Per-column sums of dy over a row range, in the storage width.
         * @brief Scalar column-sum partial (fp32/fp64).
         * @param partial Per-slot column accumulator, cols elements; added to, not zeroed.
         * @param dy Upstream gradient, row-major rows x cols.
         * @param cols Last-dimension size.
         * @param rbegin First row summed.
         * @param rend One past the last row summed.
         */
        template<typename T>
        void bias_add_g_colsum_scalar(T* partial, const T* dy, const size_t cols, const size_t rbegin, const size_t rend) {
            for (auto r = rbegin; r < rend; ++r) {
                for (auto c = size_t{0}; c < cols; ++c) {
                    partial[c] = partial[c] + dy[r * cols + c];
                }
            }
        }

        /**
         * Half-dtype column sums: widen each element to fp32, accumulate in
         * fp32; narrowed RNE once at store time.
         * @brief Scalar column-sum partial (fp16/bf16, fp32 accumulate).
         * @param partial Per-slot fp32 column accumulator, cols elements; added to, not zeroed.
         * @param dy Upstream gradient, row-major rows x cols.
         * @param cols Last-dimension size.
         * @param rbegin First row summed.
         * @param rend One past the last row summed.
         */
        template<typename H>
        void bias_add_g_colsum_half(float32_t* partial, const H* dy, const size_t cols, const size_t rbegin, const size_t rend) {
            for (auto r = rbegin; r < rend; ++r) {
                for (auto c = size_t{0}; c < cols; ++c) {
                    partial[c] = partial[c] + widen_lane(dy[r * cols + c]);
                }
            }
        }

        /**
         * Packed-dtype column sums: each element's two lanes are consecutive
         * columns of a 2*cols-wide fp32 partial.
         * @brief Scalar column-sum partial (fp16x2/bf16x2, per-lane fp32 accumulate).
         * @param partial Per-slot fp32 column accumulator, 2*cols elements; added to, not zeroed.
         * @param dy Upstream gradient, row-major rows x cols packed elements.
         * @param cols Last-dimension size in packed elements.
         * @param rbegin First row summed.
         * @param rend One past the last row summed.
         */
        template<typename P>
        void bias_add_g_colsum_packed(float32_t* partial, const P* dy, const size_t cols, const size_t rbegin, const size_t rend) {
            using H = packed_lane_t<P>;
            for (auto r = rbegin; r < rend; ++r) {
                for (auto c = size_t{0}; c < cols; ++c) {
                    const auto bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(dy[r * cols + c]);
                    partial[2 * c] = partial[2 * c] + widen_lane(lane_lo<H>(bits));
                    partial[2 * c + 1] = partial[2 * c + 1] + widen_lane(lane_hi<H>(bits));
                }
            }
        }

        /**
         * Integer column sums: modular (wrap) accumulation in the unsigned
         * twin, bit-identical on every backend and chunking.
         * @brief Scalar column-sum partial (all integer dtypes, wrap).
         * @param partial Per-slot column accumulator, cols elements; added to, not zeroed.
         * @param dy Upstream gradient, row-major rows x cols.
         * @param cols Last-dimension size.
         * @param rbegin First row summed.
         * @param rend One past the last row summed.
         */
        template<typename T>
        void bias_add_g_colsum_int(T* partial, const T* dy, const size_t cols, const size_t rbegin, const size_t rend) {
            using U = std::make_unsigned_t<T>;
            auto* acc = reinterpret_cast<U*>(partial);
            const auto* in = reinterpret_cast<const U*>(dy);
            for (auto r = rbegin; r < rend; ++r) {
                for (auto c = size_t{0}; c < cols; ++c) {
                    acc[c] = static_cast<U>(acc[c] + in[r * cols + c]);
                }
            }
        }

        /**
         * Int4x2 column sums: per-nibble int32 partials (2*cols wide); the
         * modular wrap lands at store time via pack_int4's low-4-bit mask,
         * equivalent to per-add wrap.
         * @brief Scalar column-sum partial (int4x2, per-nibble).
         * @param partial Per-slot int32 nibble accumulator, 2*cols elements; added to, not zeroed.
         * @param dy Upstream gradient, row-major rows x cols packed elements.
         * @param cols Last-dimension size in packed elements.
         * @param rbegin First row summed.
         * @param rend One past the last row summed.
         */
        void bias_add_g_colsum_int4x2(int32_t* partial, const int4x2_t* dy, const size_t cols, const size_t rbegin, const size_t rend) {
            for (auto r = rbegin; r < rend; ++r) {
                for (auto c = size_t{0}; c < cols; ++c) {
                    partial[2 * c] += int4_lo(dy[r * cols + c]);
                    partial[2 * c + 1] += int4_hi(dy[r * cols + c]);
                }
            }
        }

        /**
         * Merge the per-slot partials and narrow into dbias: floats in the
         * accumulator width (halfs RNE via the unaccel pathway), packed dtypes
         * per lane, integers wrap, int4x2 wraps per nibble. Float merge order
         * is chunking-defined - not guaranteed bit-identical across tiers.
         * @brief Merge column-sum partials into the bias gradient.
         * @param dbias Output bias gradient (overwritten); packed dtypes receive cols_eff / 2 elements.
         * @param partials Per-slot partials, slots x cols_eff row-major.
         * @param slots Number of partial slots to merge.
         * @param cols_eff Effective column count (2 * cols for packed dtypes).
         */
        template<typename T, typename Acc>
        void bias_add_g_store(T* dbias, const std::vector<Acc>& partials, const size_t slots, const size_t cols_eff) {
            for (auto c = size_t{0}; c < cols_eff; ++c) {
                if constexpr (std::is_same_v<T, float16x2_t> || std::is_same_v<T, bfloat16x2_t>) {
                    if ((c & 1) != 0) {
                        continue;
                    }
                    using H = packed_lane_t<T>;
                    auto s_lo = Acc{0};
                    auto s_hi = Acc{0};
                    for (auto slot = size_t{0}; slot < slots; ++slot) {
                        s_lo = s_lo + partials[slot * cols_eff + c];
                        s_hi = s_hi + partials[slot * cols_eff + c + 1];
                    }
                    const auto lo = narrow_lane<H>(s_lo);
                    const auto hi = narrow_lane<H>(s_hi);
                    dbias[c >> 1] = pack_lanes<T>(Intrinsics::detail::bit_cast_memcpy<uint16_t>(lo),
                                                  Intrinsics::detail::bit_cast_memcpy<uint16_t>(hi));
                } else if constexpr (std::is_integral_v<T>) {
                    // Wrap-merge through the unsigned twin - signed overflow is UB.
                    using U = std::make_unsigned_t<T>;
                    auto s = U{0};
                    for (auto slot = size_t{0}; slot < slots; ++slot) {
                        s = static_cast<U>(s + static_cast<U>(partials[slot * cols_eff + c]));
                    }
                    dbias[c] = static_cast<T>(s);
                } else {
                    auto s = Acc{0};
                    for (auto slot = size_t{0}; slot < slots; ++slot) {
                        s = s + partials[slot * cols_eff + c];
                    }
                    if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                        dbias[c] = s;
                    } else {
                        dbias[c] = narrow_lane<T>(s);
                    }
                }
            }
        }

        /**
         * Int4x2 store: merge int32 nibble partials and wrap-pack each pair.
         * @brief Merge int4x2 column-sum partials into the bias gradient.
         * @param dbias Output bias gradient (overwritten), cols packed elements.
         * @param partials Per-slot int32 nibble partials, slots x (2 * cols) row-major.
         * @param slots Number of partial slots to merge.
         * @param cols Last-dimension size in packed elements.
         */
        void bias_add_g_store_int4x2(int4x2_t* dbias, const std::vector<int32_t>& partials, const size_t slots, const size_t cols) {
            for (auto c = size_t{0}; c < cols; ++c) {
                auto lo = int32_t{0};
                auto hi = int32_t{0};
                for (auto slot = size_t{0}; slot < slots; ++slot) {
                    lo += partials[slot * 2 * cols + 2 * c];
                    hi += partials[slot * 2 * cols + 2 * c + 1];
                }
                dbias[c] = pack_int4(lo, hi);
            }
        }

        /**
         * Chunk a row-indexed worker (slot, begin, end) across the pool,
         * returning the futures for the caller to drain on its own thread
         * (this runs in a node fn under backward(), never inside a pool
         * worker, so waiting there is legal). Small inputs run inline and return no futures.
         * @brief Threaded row launcher with per-slot partials. Returns the futures for the caller to drain.
         * @param rows Total row count.
         * @param range_fn Callable (slot, begin, end) over a row range; copied into each task.
         * @param t First tensor the workers touch (receives the futures).
         * @param extra Second tensor the workers touch (receives the futures).
         * @return Futures of the submitted chunks; empty if the work ran inline.
         */
        template<typename RangeFn>
        std::vector<std::shared_future<void>> launch_rows_partial(const size_t rows, RangeFn&& range_fn, const Tensor& t, const Tensor& extra) {
            const auto t_count = static_cast<size_t>(IncludedCPUFeatures().threads);
            if (t_count <= 1 || rows <= t_count * 8) {
                range_fn(0, 0, rows);
                return {};
            }
            auto& threadMgr = Threading::GetCPUThreadPool();
            std::vector<std::shared_future<void>> futures;
            const auto chunk = (rows + t_count - 1) / t_count;
            auto slot = size_t{0};
            for (auto ti = size_t{0}; ti < t_count; ++ti) {
                const auto begin = ti * chunk;
                if (begin >= rows) {
                    break;
                }
                // range_fn is captured by value; its own captures reference the
                // caller's partials, which stay alive until drain() returns.
                futures.push_back(threadMgr.submit([range_fn, slot, begin, end = std::min(begin + chunk, rows)] { range_fn(slot, begin, end); }, Threading::TaskPriority::Normal).share());
                ++slot;
            }
            t.give_futures(futures);
            extra.give_futures(futures);
            return futures;
        }

        /**
         * Fixed-arity function pointer type of a bias_add_g column-sum kernel (scalar or AVX).
         * @brief Column-sum kernel signature: (partial, dy, cols, rbegin, rend).
         */
        template<typename T>
        using ColsumScalarFn = void(*)(T*, const T*, size_t, size_t, size_t);
    }

    /**
     * dbias = per-column sum of dy over all leading rows (last dimension = cols).
     * Per-thread-slot partials are merged at the end, so float results depend on the
     * chunking; halfs accumulate in fp32 and narrow RNE once, integers wrap. dbias is
     * overwritten. A zero-width last dimension returns without touching dbias. Blocks
     * until the workers finish (drains on the caller thread).
     * @brief CPU bias_add backward, column-sum of dy.
     * @param dy Upstream gradient, any rank >= 1.
     * @param dbias Output bias gradient, last-dimension sized (written).
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if either tensor is not on the CPU.
     * @throws ErrorHandling::NotImplemented for an unsupported dtype.
     */
    void bias_add_g_cpu(const Tensor& dy, Tensor& dbias) {
        if (dy.device != Device::CPU || dbias.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(dy.device));
        }
        using DType = DataTypes::DType;
        const auto cols = dy.shape[dy.ndim - 1];
        if (cols == 0) {
            return;
        }
        const auto rows = dy.numel / cols;
        const auto slots = static_cast<size_t>(IncludedCPUFeatures().threads);
        switch (dy.dtype) {
            case DType::Float32: {
                const auto* dy_ptr = static_cast<const float32_t*>(dy.data) + dy.offset;
                std::vector<float32_t> partials(slots * cols, 0.0f);
                const ColsumScalarFn<float32_t> kernel = Math::Autograd::bias_add_g_avx512_available()
                    ? Math::Autograd::bias_add_g_f32_colsum_avx512
                    : (Math::Autograd::bias_add_g_avx2_available() ? Math::Autograd::bias_add_g_f32_colsum_avx2 : bias_add_g_colsum_scalar<float32_t>);
                /**
                 * Runs on a pool worker (or inline, slot 0, for small inputs) over the row range [b, e). Captures by reference (partials and dy_ptr outlive it: the caller drains before returning).
                 * Adds the column sums of those rows of dy into its own slot of partials (partials + slot * cols), so chunks never race; the partials are merged afterwards. Uses the dispatched kernel.
                 * @brief Worker for bias_add_g_cpu, float32_t path (scalar or AVX kernel).
                 * @param slot Index of the per-thread partial buffer this chunk accumulates into.
                 * @param b First row index of the range (inclusive).
                 * @param e One past the last row index of the range.
                 */
                drain(launch_rows_partial(rows, [&](const size_t slot, const size_t b, const size_t e) { kernel(partials.data() + slot * cols, dy_ptr, cols, b, e); }, dy, dbias));
                bias_add_g_store<float32_t, float32_t>(static_cast<float32_t*>(dbias.data) + dbias.offset, partials, slots, cols);
                break;
            }
            case DType::Float64: {
                const auto* dy_ptr = static_cast<const float64_t*>(dy.data) + dy.offset;
                std::vector<float64_t> partials(slots * cols, 0.0);
                const ColsumScalarFn<float64_t> kernel = Math::Autograd::bias_add_g_avx512_available()
                    ? Math::Autograd::bias_add_g_f64_colsum_avx512
                    : (Math::Autograd::bias_add_g_avx2_available() ? Math::Autograd::bias_add_g_f64_colsum_avx2 : bias_add_g_colsum_scalar<float64_t>);
                /**
                 * Runs on a pool worker (or inline, slot 0, for small inputs) over the row range [b, e). Captures by reference (partials and dy_ptr outlive it: the caller drains before returning).
                 * Adds the column sums of those rows of dy into its own slot of partials (partials + slot * cols), so chunks never race; the partials are merged afterwards. Uses the dispatched kernel.
                 * @brief Worker for bias_add_g_cpu, float64_t path (scalar or AVX kernel).
                 * @param slot Index of the per-thread partial buffer this chunk accumulates into.
                 * @param b First row index of the range (inclusive).
                 * @param e One past the last row index of the range.
                 */
                drain(launch_rows_partial(rows, [&](const size_t slot, const size_t b, const size_t e) { kernel(partials.data() + slot * cols, dy_ptr, cols, b, e); }, dy, dbias));
                bias_add_g_store<float64_t, float64_t>(static_cast<float64_t*>(dbias.data) + dbias.offset, partials, slots, cols);
                break;
            }
            case DType::Float16: {
                const auto* dy_ptr = static_cast<const float16_t*>(dy.data) + dy.offset;
                std::vector<float32_t> partials(slots * cols, 0.0f);
                /**
                 * Runs on a pool worker (or inline, slot 0, for small inputs) over the row range [b, e). Captures by reference (partials and dy_ptr outlive it: the caller drains before returning).
                 * Adds the column sums of those rows of dy into its own slot of partials (partials + slot * cols), so chunks never race; the partials are merged afterwards. Accumulates in fp32.
                 * @brief Worker for bias_add_g_cpu, float16_t path.
                 * @param slot Index of the per-thread partial buffer this chunk accumulates into.
                 * @param b First row index of the range (inclusive).
                 * @param e One past the last row index of the range.
                 */
                drain(launch_rows_partial(rows, [&](const size_t slot, const size_t b, const size_t e) { bias_add_g_colsum_half<float16_t>(partials.data() + slot * cols, dy_ptr, cols, b, e); }, dy, dbias));
                bias_add_g_store<float16_t, float32_t>(static_cast<float16_t*>(dbias.data) + dbias.offset, partials, slots, cols);
                break;
            }
            case DType::BFloat16: {
                const auto* dy_ptr = static_cast<const bfloat16_t*>(dy.data) + dy.offset;
                std::vector<float32_t> partials(slots * cols, 0.0f);
                /**
                 * Runs on a pool worker (or inline, slot 0, for small inputs) over the row range [b, e). Captures by reference (partials and dy_ptr outlive it: the caller drains before returning).
                 * Adds the column sums of those rows of dy into its own slot of partials (partials + slot * cols), so chunks never race; the partials are merged afterwards. Accumulates in fp32.
                 * @brief Worker for bias_add_g_cpu, bfloat16_t path.
                 * @param slot Index of the per-thread partial buffer this chunk accumulates into.
                 * @param b First row index of the range (inclusive).
                 * @param e One past the last row index of the range.
                 */
                drain(launch_rows_partial(rows, [&](const size_t slot, const size_t b, const size_t e) { bias_add_g_colsum_half<bfloat16_t>(partials.data() + slot * cols, dy_ptr, cols, b, e); }, dy, dbias));
                bias_add_g_store<bfloat16_t, float32_t>(static_cast<bfloat16_t*>(dbias.data) + dbias.offset, partials, slots, cols);
                break;
            }
            case DType::Float16x2: {
                const auto* dy_ptr = static_cast<const float16x2_t*>(dy.data) + dy.offset;
                std::vector<float32_t> partials(slots * 2 * cols, 0.0f);
                /**
                 * Runs on a pool worker (or inline, slot 0, for small inputs) over the row range [b, e). Captures by reference (partials and dy_ptr outlive it: the caller drains before returning).
                 * Adds the column sums of those rows of dy into its own slot of partials (partials + slot * 2 * cols), so chunks never race; the partials are merged afterwards. Each element carries two columns, so each slot holds 2 * cols fp32 sums.
                 * @brief Worker for bias_add_g_cpu, float16x2_t packed path.
                 * @param slot Index of the per-thread partial buffer this chunk accumulates into.
                 * @param b First row index of the range (inclusive).
                 * @param e One past the last row index of the range.
                 */
                drain(launch_rows_partial(rows, [&](const size_t slot, const size_t b, const size_t e) { bias_add_g_colsum_packed<float16x2_t>(partials.data() + slot * 2 * cols, dy_ptr, cols, b, e); }, dy, dbias));
                bias_add_g_store<float16x2_t, float32_t>(static_cast<float16x2_t*>(dbias.data) + dbias.offset, partials, slots, 2 * cols);
                break;
            }
            case DType::BFloat16x2: {
                const auto* dy_ptr = static_cast<const bfloat16x2_t*>(dy.data) + dy.offset;
                std::vector<float32_t> partials(slots * 2 * cols, 0.0f);
                /**
                 * Runs on a pool worker (or inline, slot 0, for small inputs) over the row range [b, e). Captures by reference (partials and dy_ptr outlive it: the caller drains before returning).
                 * Adds the column sums of those rows of dy into its own slot of partials (partials + slot * 2 * cols), so chunks never race; the partials are merged afterwards. Each element carries two columns, so each slot holds 2 * cols fp32 sums.
                 * @brief Worker for bias_add_g_cpu, bfloat16x2_t packed path.
                 * @param slot Index of the per-thread partial buffer this chunk accumulates into.
                 * @param b First row index of the range (inclusive).
                 * @param e One past the last row index of the range.
                 */
                drain(launch_rows_partial(rows, [&](const size_t slot, const size_t b, const size_t e) { bias_add_g_colsum_packed<bfloat16x2_t>(partials.data() + slot * 2 * cols, dy_ptr, cols, b, e); }, dy, dbias));
                bias_add_g_store<bfloat16x2_t, float32_t>(static_cast<bfloat16x2_t*>(dbias.data) + dbias.offset, partials, slots, 2 * cols);
                break;
            }
            case DType::Int4x2: {
                const auto* dy_ptr = static_cast<const int4x2_t*>(dy.data) + dy.offset;
                std::vector<int32_t> partials(slots * 2 * cols, 0);
                /**
                 * Runs on a pool worker (or inline, slot 0, for small inputs) over the row range [b, e). Captures by reference (partials and dy_ptr outlive it: the caller drains before returning).
                 * Adds the column sums of those rows of dy into its own slot of partials (partials + slot * 2 * cols), so chunks never race; the partials are merged afterwards. Each element carries two nibbles, so each slot holds 2 * cols int32 sums.
                 * @brief Worker for bias_add_g_cpu, int4x2_t path.
                 * @param slot Index of the per-thread partial buffer this chunk accumulates into.
                 * @param b First row index of the range (inclusive).
                 * @param e One past the last row index of the range.
                 */
                drain(launch_rows_partial(rows, [&](const size_t slot, const size_t b, const size_t e) { bias_add_g_colsum_int4x2(partials.data() + slot * 2 * cols, dy_ptr, cols, b, e); }, dy, dbias));
                bias_add_g_store_int4x2(static_cast<int4x2_t*>(dbias.data) + dbias.offset, partials, slots, cols);
                break;
            }
            default: {
                /**
                 * Captures by reference: slots, cols, rows, dy and dbias from the enclosing function. Generic lambda; T is deduced from dy_ptr.
                 * Allocates zeroed per-slot partials, drains the row-partial launch on the caller thread, then merges the slots into dbias with wrapping integer sums.
                 * @brief Integer-dtype bias_add backward driver for element type T.
                 * @param T Integer element type of dy (int8_t ... uint64_t).
                 * @param dy_ptr Pointer to the first element of dy, offset already applied.
                 */
                auto run_int = [&]<typename T>(const T* dy_ptr) {
                    std::vector<T> partials(slots * cols, T{0});
                    /**
                     * Runs on a pool worker (or inline, slot 0, for small inputs) over the row range [b, e). Captures by reference (partials and dy_ptr outlive it: the caller drains before returning).
                     * Adds the column sums of those rows of dy into its own slot of partials (partials + slot * cols), so chunks never race; the partials are merged afterwards. Integer sums wrap in T.
                     * @brief Worker for bias_add_g_cpu, integer path of type T.
                     * @param slot Index of the per-thread partial buffer this chunk accumulates into.
                     * @param b First row index of the range (inclusive).
                     * @param e One past the last row index of the range.
                     */
                    drain(launch_rows_partial(rows, [&](const size_t slot, const size_t b, const size_t e) { bias_add_g_colsum_int<T>(partials.data() + slot * cols, dy_ptr, cols, b, e); }, dy, dbias));
                    bias_add_g_store<T, T>(static_cast<T*>(dbias.data) + dbias.offset, partials, slots, cols);
                };
                switch (dy.dtype) {
                    case DType::Int8: run_int(static_cast<const int8_t*>(dy.data) + dy.offset); break;
                    case DType::UInt8: run_int(static_cast<const uint8_t*>(dy.data) + dy.offset); break;
                    case DType::Int16: run_int(static_cast<const int16_t*>(dy.data) + dy.offset); break;
                    case DType::UInt16: run_int(static_cast<const uint16_t*>(dy.data) + dy.offset); break;
                    case DType::Int32: run_int(static_cast<const int32_t*>(dy.data) + dy.offset); break;
                    case DType::UInt32: run_int(static_cast<const uint32_t*>(dy.data) + dy.offset); break;
                    case DType::Int64: run_int(static_cast<const int64_t*>(dy.data) + dy.offset); break;
                    case DType::UInt64: run_int(static_cast<const uint64_t*>(dy.data) + dy.offset); break;
                    [[unlikely]] default: throw ErrorHandling::NotImplemented();
                }
                break;
            }
        }
    }

    /**
     * inputs are (x, bias). dx is a copy of output_grad (the forward add is bias-only per
     * element); dbias is the column-sum of output_grad. Each is computed only if its tensor
     * requires grad, then handed to accumulate_grad. Syncs the tensors it reads.
     * @brief Backward trampoline for bias_add.
     * @param output_grad Gradient of this node's output tensor.
     * @param inputs Input tensors of the forward op.
     * @param n_inputs Number of entries in inputs.
     * @throws ErrorHandling::UnReachableCode if n_inputs is not 2 or the device is neither CPU nor GPU.
     */
    void bias_add_g_node(Tensor* output_grad, Tensor** inputs, const size_t n_inputs) {
        if (n_inputs != 2) [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        auto& x = *inputs[0];
        auto& bias = *inputs[1];
        output_grad->sync();
        if (x.requires_grad) {
            x.sync();
            // dx = dy, exactly (the forward add per element is bias-only).
            auto dx = output_grad->clone(true);
            x.accumulate_grad(std::move(dx));
        }
        if (bias.requires_grad) {
            bias.sync();
            auto dbias = Tensor(std::vector<size_t>(bias.shape, bias.shape + bias.ndim), bias.dtype, bias.device, false);
            if (bias.device == Device::CPU) {
                bias_add_g_cpu(*output_grad, dbias);
            } else if (bias.device == Device::GPU) {
                bias_add_g_gpu(*output_grad, dbias);
            } else [[unlikely]] {
                throw ErrorHandling::UnReachableCode();
            }
            bias.accumulate_grad(std::move(dbias));
        }
    }
}
