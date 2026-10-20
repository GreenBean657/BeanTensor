/**
 * @file SoftmaxCEGCPU.cpp
 * @author GreenBean
 * @brief CPU backend for softmax cross-entropy backward, plus the softmax_ce
 * backward trampoline (softmax_ce_g_node). Scalar exp()/log() dominate, so
 * there are no SIMD kernels (same reason as forward softmax_ce).
 * @version 0.1
 * @date 2026-10-20
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <future>
#include <limits>
#include <type_traits>
#include <vector>

#include "Math/Autograd/AutogradShared.h"
#include "Math/Autograd/SoftmaxCEG/SoftmaxCEG.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Tensors::detail {
    using namespace autograd_shared;

    namespace {
        // --------------------------- softmax_ce_g kernels ---------------------------

        /**
         * @brief Widen class j of a row to fp32 (fp32/fp16/bf16/packed x2 rows;
         * packed dtypes unpack each element's lanes as consecutive classes).
         * @param row Logits row.
         * @param j Class index (lane-granular for packed dtypes).
         * @return Class j widened to fp32.
         */
        template<typename T>
        [[nodiscard]] float32_t widen_class_f32(const T* row, const size_t j) {
            if constexpr (std::is_same_v<T, float32_t>) {
                return row[j];
            } else if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
                return widen_lane(row[j]);
            } else {
                using H = packed_lane_t<T>;
                const auto bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(row[j >> 1]);
                return widen_lane((j & 1) != 0 ? lane_hi<H>(bits) : lane_lo<H>(bits));
            }
        }

        /**
         * @brief Widen class j of a row to fp64 (fp64/integer/int4x2 rows;
         * int4x2 unpacks each element's nibbles as consecutive classes).
         * @param row Logits row.
         * @param j Class index (nibble-granular for int4x2).
         * @return Class j widened to fp64.
         */
        template<typename T>
        [[nodiscard]] float64_t widen_class_f64(const T* row, const size_t j) {
            if constexpr (std::is_same_v<T, float64_t>) {
                return row[j];
            } else if constexpr (std::is_same_v<T, int4x2_t>) {
                return static_cast<float64_t>((j & 1) != 0 ? int4_hi(row[j >> 1]) : int4_lo(row[j >> 1]));
            } else {
                return static_cast<float64_t>(row[j]);
            }
        }

        /**
         * @brief Widen the scalar dloss gradient element to the accumulator
         * width. Packed grad dtypes read the low lane (the forward store_mean
         * duplicates the mean into both lanes); int4x2 reads the low nibble.
         * @param g Pointer to the single dloss gradient element, of dtype T.
         * @return The dloss value in the accumulator type Acc.
         */
        template<typename T, typename Acc>
        [[nodiscard]] Acc widen_dloss(const void* g) {
            if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                return *static_cast<const T*>(g);
            } else if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
                return widen_lane(*static_cast<const T*>(g));
            } else if constexpr (std::is_same_v<T, float16x2_t> || std::is_same_v<T, bfloat16x2_t>) {
                using H = packed_lane_t<T>;
                const auto bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(*static_cast<const T*>(g));
                return widen_lane(lane_lo<H>(bits));
            } else if constexpr (std::is_same_v<T, int4x2_t>) {
                return static_cast<float64_t>(int4_lo(*static_cast<const int4x2_t*>(g)));
            } else {
                return static_cast<float64_t>(*static_cast<const T*>(g));
            }
        }

        /**
         * Narrow one accumulated gradient value into the logits dtype: floats
         * RNE (halfs via the unaccel pathway), integers clamp to the dtype
         * range and truncate - a NaN gradient stores 0 (no integer NaN),
         * matching store_mean's contract.
         * @brief Store one softmax_ce_g gradient element.
         * @param v_raw Accumulated gradient value.
         * @return The value narrowed to T.
         */
        template<typename T, typename Acc>
        [[nodiscard]] T narrow_grad(const Acc v_raw) {
            if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                return v_raw;
            } else if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
                return narrow_lane<T>(v_raw);
            } else {
                auto v = static_cast<float64_t>(v_raw);
                if (std::isnan(v)) {
                    v = 0.0;
                }
                const auto lo = static_cast<float64_t>(std::numeric_limits<T>::lowest());
                const auto hi = static_cast<float64_t>(std::numeric_limits<T>::max());
                if (v <= lo) {
                    v = lo;
                } else if (v >= hi) {
                    v = hi;
                }
                return static_cast<T>(v);
            }
        }

        /**
         * Row softmax gradient: dx_j = (exp(x_j - m) / s - (j == target)) *
         * dloss_scaled, with m/s recomputed exactly like the forward pass (max
         * ignores NaN; a NaN logit poisons s and thus the whole row).
         * @brief Scalar row softmax_ce backward (non-packed dtypes).
         * @param row Logits row.
         * @param out_row Output gradient row (written).
         * @param cols Number of classes in the row.
         * @param target Target class index, already bounds-checked.
         * @param dloss_scaled Upstream loss gradient divided by the row count.
         */
        template<typename T, typename Acc>
        void softmax_ce_g_row(const T* row, T* out_row, const size_t cols, const size_t target, const Acc dloss_scaled) {
            /**
             * Captures by reference: row (or the logits row) and the accumulator type Acc come from the enclosing function.
             * Selects at compile time between widen_class_f32 (Acc is float32_t) and widen_class_f64 (otherwise), reading class j of the row.
             * @brief Widen logit j of the row to the accumulator type.
             * @param j Class index within the row, in [0, cols).
             * @return Logit j widened to Acc.
             */
            auto widen = [&](const size_t j) -> Acc {
                if constexpr (std::is_same_v<Acc, float32_t>) {
                    return widen_class_f32(row, j);
                } else {
                    return widen_class_f64(row, j);
                }
            };
            auto m = std::numeric_limits<Acc>::lowest();
            for (auto j = size_t{0}; j < cols; ++j) {
                if (
                    const auto x = widen(j);
                    x > m
                ) {
                    m = x;
                }
            }
            auto s = Acc{0};
            for (auto j = size_t{0}; j < cols; ++j) {
                s += std::exp(widen(j) - m);
            }
            for (auto j = size_t{0}; j < cols; ++j) {
                const auto p = std::exp(widen(j) - m) / s;
                const auto g = (p - (j == target ? Acc{1} : Acc{0})) * dloss_scaled;
                out_row[j] = narrow_grad<T, Acc>(g);
            }
        }

        /**
         * Packed/int4x2 row softmax gradient: same math, but each output
         * element packs two consecutive class gradients. int4x2 results are
         * narrowed through int8 (NaN -> 0, clamp, truncate) then clamped to [-8, 7].
         * @brief Scalar row softmax_ce backward (packed x2 / int4x2 dtypes).
         * @param out_row Output gradient row (written), cols_eff / 2 packed elements.
         * @param probs Precomputed softmax probabilities, cols_eff entries.
         * @param cols_eff Effective class count (two per packed element).
         * @param target Target class index, already bounds-checked.
         * @param dloss_scaled Upstream loss gradient divided by the row count.
         */
        template<typename T, typename Acc>
        void softmax_ce_g_row_packed(T* out_row, const Acc* probs, const size_t cols_eff, const size_t target, const Acc dloss_scaled) {
            for (auto e = size_t{0}; e < cols_eff / 2; ++e) {
                const auto j = 2 * e;
                const auto glo = (probs[j] - (j == target ? Acc{1} : Acc{0})) * dloss_scaled;
                const auto ghi = (probs[j + 1] - (j + 1 == target ? Acc{1} : Acc{0})) * dloss_scaled;
                if constexpr (std::is_same_v<T, int4x2_t>) {
                    // Clamp to the signed nibble range [-8, 7] (store_mean's
                    // int4x2 contract), after the int8 NaN->0 / truncate.
                    out_row[e] = pack_int4(std::clamp(static_cast<int32_t>(narrow_grad<int8_t, Acc>(glo)), -8, 7),
                                           std::clamp(static_cast<int32_t>(narrow_grad<int8_t, Acc>(ghi)), -8, 7));
                } else {
                    using H = packed_lane_t<T>;
                    const auto lo = narrow_lane<H>(static_cast<float32_t>(glo));
                    const auto hi = narrow_lane<H>(static_cast<float32_t>(ghi));
                    out_row[e] = pack_lanes<T>(Intrinsics::detail::bit_cast_memcpy<uint16_t>(lo),
                                               Intrinsics::detail::bit_cast_memcpy<uint16_t>(hi));
                }
            }
        }

        /**
         * Packed variant driver: compute widened probs per class, then pack.
         * @brief Scalar row softmax_ce backward (packed x2 / int4x2 rows).
         * @param row Logits row of packed elements.
         * @param out_row Output gradient row (written).
         * @param cols_eff Effective class count (two per packed element).
         * @param target Target class index, already bounds-checked.
         * @param dloss_scaled Upstream loss gradient divided by the row count.
         */
        template<typename T, typename Acc>
        void softmax_ce_g_row_wide(const T* row, T* out_row, const size_t cols_eff, const size_t target, const Acc dloss_scaled) {
            /**
             * Captures by reference: row and the accumulator type Acc come from the enclosing function.
             * Selects at compile time between widen_class_f32 (Acc is float32_t) and widen_class_f64 (otherwise), reading effective class j (two per packed element).
             * @brief Widen effective class j of the packed row to the accumulator type.
             * @param j Effective class index, in [0, cols_eff).
             * @return Class j widened to Acc.
             */
            auto widen = [&](const size_t j) -> Acc {
                if constexpr (std::is_same_v<Acc, float32_t>) {
                    return widen_class_f32(row, j);
                } else {
                    return widen_class_f64(row, j);
                }
            };
            auto m = std::numeric_limits<Acc>::lowest();
            for (auto j = size_t{0}; j < cols_eff; ++j) {
                if (
                    const auto x = widen(j);
                    x > m
                ) {
                    m = x;
                }
            }
            auto s = Acc{0};
            for (auto j = size_t{0}; j < cols_eff; ++j) {
                s += std::exp(widen(j) - m);
            }
            std::vector<Acc> probs(cols_eff);
            for (auto j = size_t{0}; j < cols_eff; ++j) {
                probs[j] = std::exp(widen(j) - m) / s;
            }
            softmax_ce_g_row_packed<T, Acc>(out_row, probs.data(), cols_eff, target, dloss_scaled);
        }

        /**
         * @brief Read target class index r, bounds-checked against the effective
         * class count. Int4x2 targets use the low nibble (matches the
         * packed->unpacked first-lane conversion contract).
         * @param targets Target buffer, one class index per row.
         * @param r Row whose target is read.
         * @param cols Effective class count; valid targets are in [0, cols).
         * @return The target class index.
         * @throws ErrorHandling::IndexOutOfRange if the target is negative or >= cols.
         */
        template<typename I>
        [[nodiscard]] uint64_t target_index(const I* targets, const size_t r, const size_t cols) {
            int64_t t;
            if constexpr (std::is_same_v<I, int4x2_t>) {
                t = int4_lo(targets[r]);
            } else {
                t = static_cast<int64_t>(targets[r]);
            }
            if (t < 0 || static_cast<uint64_t>(t) >= cols) {
                throw ErrorHandling::IndexOutOfRange();
            }
            return static_cast<uint64_t>(t);
        }

        /**
         * Chunk a row-indexed worker (begin, end) across the pool; futures land
         * on the touched tensors and are returned for the caller to drain.
         * Small inputs run inline and return no futures.
         * @brief Threaded row launcher without partials. Returns the futures for the caller to drain.
         * @param rows Total row count.
         * @param range_fn Callable (begin, end) over a row range; copied into each task.
         * @param t First tensor the workers touch (receives the futures).
         * @param extra Second tensor the workers touch (receives the futures).
         * @param out Output tensor (receives the futures).
         * @return Futures of the submitted chunks; empty if the work ran inline.
         */
        template<typename RangeFn>
        std::vector<std::shared_future<void>> launch_rows(const size_t rows, RangeFn&& range_fn, const Tensor& t, const Tensor& extra, const Tensor& out) {
            const auto t_count = static_cast<size_t>(IncludedCPUFeatures().threads);
            if (t_count <= 1 || rows <= t_count * 8) {
                range_fn(0, rows);
                return {};
            }
            auto& threadMgr = Threading::GetCPUThreadPool();
            std::vector<std::shared_future<void>> futures;
            const auto chunk = (rows + t_count - 1) / t_count;
            for (auto ti = size_t{0}; ti < t_count; ++ti) {
                const auto begin = ti * chunk;
                if (begin >= rows) {
                    break;
                }
                futures.push_back(threadMgr.submit([range_fn, begin, end = std::min(begin + chunk, rows)] { range_fn(begin, end); }, Threading::TaskPriority::Normal).share());
            }
            t.give_futures(futures);
            extra.give_futures(futures);
            out.give_futures(futures);
            return futures;
        }
    }

    /**
     * dlogits = (softmax(logits) - onehot(target)) * dloss / rows, row by row, with the
     * softmax recomputed as in the forward pass (a NaN logit poisons its whole row).
     * logits must be 2-D (rows x cols); packed dtypes carry two classes per element.
     * Accumulation is fp32 for fp32/fp16/bf16/packed halfs and fp64 for fp64, integers
     * and int4x2; integer outputs clamp and map NaN to 0. dloss_grad is a single element
     * (packed dtypes read the low lane, int4x2 the low nibble). Blocks until the workers finish.
     * @brief CPU softmax cross-entropy backward.
     * @param dloss_grad Scalar upstream gradient of the loss.
     * @param logits Forward logits, rows x cols.
     * @param targets Per-row target class indices (any integer dtype or int4x2).
     * @param dlogits Output gradient, same shape/dtype as logits (written).
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if any tensor is not on the CPU.
     * @throws ErrorHandling::IndexOutOfRange if a target is outside [0, effective classes).
     * @throws ErrorHandling::UnReachableCode for a non-integer targets dtype.
     * @throws ErrorHandling::NotImplemented for an unsupported logits dtype.
     */
    void softmax_ce_g_cpu(const Tensor& dloss_grad, const Tensor& logits, const Tensor& targets, Tensor& dlogits) {
        if (dloss_grad.device != Device::CPU || logits.device != Device::CPU || targets.device != Device::CPU || dlogits.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(logits.device));
        }
        using DType = DataTypes::DType;
        const auto rows = logits.shape[0];
        const auto cols_store = logits.shape[1];
        const auto packed = logits.dtype == DType::Float16x2 || logits.dtype == DType::BFloat16x2 || logits.dtype == DType::Int4x2;
        const auto cols_eff = cols_store * (packed ? size_t{2} : size_t{1});
        const auto tgt_dtype = targets.dtype;
        const auto* const tgt_data = targets.data;
        const auto tgt_offset = targets.offset;
        /**
         * Captures by reference: rows, cols_eff, cols_store, tgt_dtype, tgt_data, tgt_offset, dy-side tensors from the enclosing function. Generic lambda with explicit template arguments, invoked through operator()<T, Acc> from the logits dtype switch.
         * Widens the scalar upstream gradient, divides it by the row count, then drains a launch_rows over the rows; a (b, e) range is a block of logits rows.
         * @brief Logits-dtype driver for softmax_ce_g_cpu.
         * @param T Storage element type of logits and dlogits.
         * @param Acc Accumulator type (float32_t for fp32/halfs, float64_t for fp64/integers/int4x2).
         * @param logits_ptr First logits element, offset applied.
         * @param dlogits_ptr First dlogits element, offset applied (written).
         * @param g_data Raw data pointer of the scalar dloss gradient tensor.
         */
        auto run_logits = [&]<typename T, typename Acc>(const T* logits_ptr, T* dlogits_ptr, const void* g_data) {
            const auto dloss = widen_dloss<T, Acc>(g_data);
            const auto dloss_scaled = dloss / static_cast<Acc>(rows);
            /**
             * Runs on a pool worker (or inline for small inputs) over the row range [b, e). Captures by reference; launch_rows copies it into each task and the caller drains before the captured locals die.
             * Dispatches on the targets dtype (run_targets) and writes the gradient for each row of the range.
             * @brief Row-range worker for softmax_ce_g_cpu.
             * @param b First row index of the range (inclusive).
             * @param e One past the last row index of the range.
             */
            drain(launch_rows(rows, [&](const size_t b, const size_t e) {
                /**
                 * Captures by reference: b, e, T, Acc, pointers and sizes of the enclosing worker. Generic lambda; I is deduced from t_ptr.
                 * For each row r in [b, e) it reads the bounds-checked target (target_index, throws IndexOutOfRange) and calls softmax_ce_g_row_wide for packed/int4x2 logits or softmax_ce_g_row otherwise.
                 * @brief Per-targets-dtype row loop for the softmax_ce backward.
                 * @param I Integer (or int4x2_t) element type of the targets.
                 * @param t_ptr First target element, offset applied.
                 */
                auto run_targets = [&]<typename I>(const I* t_ptr) {
                    for (auto r = b; r < e; ++r) {
                        const auto idx = target_index(t_ptr, r, cols_eff);
                        if constexpr (std::is_same_v<T, float16x2_t> || std::is_same_v<T, bfloat16x2_t> || std::is_same_v<T, int4x2_t>) {
                            softmax_ce_g_row_wide<T, Acc>(logits_ptr + r * cols_store, dlogits_ptr + r * cols_store, cols_eff, idx, dloss_scaled);
                        } else {
                            softmax_ce_g_row<T, Acc>(logits_ptr + r * cols_store, dlogits_ptr + r * cols_store, cols_eff, idx, dloss_scaled);
                        }
                    }
                };
                switch (tgt_dtype) {
                    case DType::Int8: run_targets(static_cast<const int8_t*>(tgt_data) + tgt_offset); break;
                    case DType::UInt8: run_targets(static_cast<const uint8_t*>(tgt_data) + tgt_offset); break;
                    case DType::Int16: run_targets(static_cast<const int16_t*>(tgt_data) + tgt_offset); break;
                    case DType::UInt16: run_targets(static_cast<const uint16_t*>(tgt_data) + tgt_offset); break;
                    case DType::Int32: run_targets(static_cast<const int32_t*>(tgt_data) + tgt_offset); break;
                    case DType::UInt32: run_targets(static_cast<const uint32_t*>(tgt_data) + tgt_offset); break;
                    case DType::Int64: run_targets(static_cast<const int64_t*>(tgt_data) + tgt_offset); break;
                    case DType::UInt64: run_targets(static_cast<const uint64_t*>(tgt_data) + tgt_offset); break;
                    case DType::Int4x2: run_targets(static_cast<const int4x2_t*>(tgt_data) + tgt_offset); break;
                    [[unlikely]] default: throw ErrorHandling::UnReachableCode();
                }
            }, logits, targets, dlogits));
        };
        switch (logits.dtype) {
            case DType::Float32: run_logits.operator()<float32_t, float32_t>(static_cast<const float32_t*>(logits.data) + logits.offset, static_cast<float32_t*>(dlogits.data) + dlogits.offset, dloss_grad.data); break;
            case DType::Float64: run_logits.operator()<float64_t, float64_t>(static_cast<const float64_t*>(logits.data) + logits.offset, static_cast<float64_t*>(dlogits.data) + dlogits.offset, dloss_grad.data); break;
            case DType::Float16: run_logits.operator()<float16_t, float32_t>(static_cast<const float16_t*>(logits.data) + logits.offset, static_cast<float16_t*>(dlogits.data) + dlogits.offset, dloss_grad.data); break;
            case DType::BFloat16: run_logits.operator()<bfloat16_t, float32_t>(static_cast<const bfloat16_t*>(logits.data) + logits.offset, static_cast<bfloat16_t*>(dlogits.data) + dlogits.offset, dloss_grad.data); break;
            case DType::Float16x2: run_logits.operator()<float16x2_t, float32_t>(static_cast<const float16x2_t*>(logits.data) + logits.offset, static_cast<float16x2_t*>(dlogits.data) + dlogits.offset, dloss_grad.data); break;
            case DType::BFloat16x2: run_logits.operator()<bfloat16x2_t, float32_t>(static_cast<const bfloat16x2_t*>(logits.data) + logits.offset, static_cast<bfloat16x2_t*>(dlogits.data) + dlogits.offset, dloss_grad.data); break;
            case DType::Int8: run_logits.operator()<int8_t, float64_t>(static_cast<const int8_t*>(logits.data) + logits.offset, static_cast<int8_t*>(dlogits.data) + dlogits.offset, dloss_grad.data); break;
            case DType::UInt8: run_logits.operator()<uint8_t, float64_t>(static_cast<const uint8_t*>(logits.data) + logits.offset, static_cast<uint8_t*>(dlogits.data) + dlogits.offset, dloss_grad.data); break;
            case DType::Int16: run_logits.operator()<int16_t, float64_t>(static_cast<const int16_t*>(logits.data) + logits.offset, static_cast<int16_t*>(dlogits.data) + dlogits.offset, dloss_grad.data); break;
            case DType::UInt16: run_logits.operator()<uint16_t, float64_t>(static_cast<const uint16_t*>(logits.data) + logits.offset, static_cast<uint16_t*>(dlogits.data) + dlogits.offset, dloss_grad.data); break;
            case DType::Int32: run_logits.operator()<int32_t, float64_t>(static_cast<const int32_t*>(logits.data) + logits.offset, static_cast<int32_t*>(dlogits.data) + dlogits.offset, dloss_grad.data); break;
            case DType::UInt32: run_logits.operator()<uint32_t, float64_t>(static_cast<const uint32_t*>(logits.data) + logits.offset, static_cast<uint32_t*>(dlogits.data) + dlogits.offset, dloss_grad.data); break;
            case DType::Int64: run_logits.operator()<int64_t, float64_t>(static_cast<const int64_t*>(logits.data) + logits.offset, static_cast<int64_t*>(dlogits.data) + dlogits.offset, dloss_grad.data); break;
            case DType::UInt64: run_logits.operator()<uint64_t, float64_t>(static_cast<const uint64_t*>(logits.data) + logits.offset, static_cast<uint64_t*>(dlogits.data) + dlogits.offset, dloss_grad.data); break;
            case DType::Int4x2: run_logits.operator()<int4x2_t, float64_t>(static_cast<const int4x2_t*>(logits.data) + logits.offset, static_cast<int4x2_t*>(dlogits.data) + dlogits.offset, dloss_grad.data); break;
            [[unlikely]] default: throw ErrorHandling::NotImplemented();
        }
    }

    /**
     * inputs are (logits, targets). Only logits receives a gradient (targets are
     * class indices); skipped when logits does not require grad. Syncs the three tensors it reads.
     * @brief Backward trampoline for softmax_ce.
     * @param output_grad Gradient of this node's output tensor.
     * @param inputs Input tensors of the forward op.
     * @param n_inputs Number of entries in inputs.
     * @throws ErrorHandling::UnReachableCode if n_inputs is not 2 or the device is neither CPU nor GPU.
     */
    void softmax_ce_g_node(Tensor* output_grad, Tensor** inputs, const size_t n_inputs) {
        if (n_inputs != 2) [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        auto& logits = *inputs[0];
        auto& targets = *inputs[1];
        if (!logits.requires_grad) {
            return;
        }
        output_grad->sync();
        logits.sync();
        targets.sync();
        auto dlogits = Tensor(std::vector<size_t>(logits.shape, logits.shape + logits.ndim), logits.dtype, logits.device, false);
        if (logits.device == Device::CPU) {
            softmax_ce_g_cpu(*output_grad, logits, targets, dlogits);
        } else if (logits.device == Device::GPU) {
            softmax_ce_g_gpu(*output_grad, logits, targets, dlogits);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        logits.accumulate_grad(std::move(dlogits));
    }
}
