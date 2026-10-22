/**
 * @file SoftmaxCECPU.cpp
 * @author GreenBean
 * @brief CPU backend for softmax_ce.
 * Hosts the Tensor::softmax_ce method definition (mirrors how Tensor::matmul
 * lives in MatmulCPU.cpp).
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

#include "Math/Activations/ActivationsShared.h"
#include "Math/Activations/SoftmaxCE/SoftmaxCE.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"
#include "Helpers/Errors.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Tensors::detail {
    using namespace activations_shared;

    namespace {
        // --------------------------- softmax_ce kernels ---------------------------

        /**
         * Covers fp32/fp16/bf16/packed x2 rows. Packed dtypes unpack each element's lanes as consecutive classes.
         * @brief Widen class j of a row to fp32.
         * @param row First element of the logits row.
         * @param j Effective class index within the row.
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
         * Covers fp64/integer/int4x2 rows. Int4x2 unpacks each element's nibbles as consecutive classes.
         * @brief Widen class j of a row to fp64.
         * @param row First element of the logits row.
         * @param j Effective class index within the row.
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
         * Numerically stable: loss = m + log(sum(exp(x - m))) - x_target, computed in the dtype's
         * accumulator width (fp32 for fp32/fp16/bf16/packed, fp64 for fp64/integers).
         * A NaN logit makes the row's loss NaN (max ignores NaN, exp(NaN - m) poisons the sum).
         * @brief Scalar row softmax cross-entropy.
         * @param row First element of the logits row.
         * @param cols Effective class count of the row.
         * @param target Class index of the row's target, already bounds-checked.
         * @return The row loss, widened to fp64.
         */
        template<typename T, typename Acc>
        [[nodiscard]] float64_t softmax_ce_row(const T* row, const size_t cols, const size_t target) {
            /**
             * Selects the fp32 or fp64 class widening to match the accumulator width.
             * @brief Widen class j of the captured row to Acc.
             * @param j Effective class index within the row.
             * @return Class j widened to Acc.
             * @note Lambda subcall. Self-execution prohibited.
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
            const auto loss = m + std::log(s) - widen(target);
            return static_cast<float64_t>(loss);
        }

        /**
         * Int4x2 targets use the low nibble (matches the packed->unpacked first-lane conversion contract).
         * Throws ErrorHandling::IndexOutOfRange for a negative index or one at or past cols.
         * @brief Read target class index r, bounds-checked against the effective class count.
         * @param targets Target class indices, one per row.
         * @param r Row index.
         * @param cols Effective class count of the row.
         * @return The class index, validated to [0, cols).
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

        // --------------------------- launchers ---------------------------

        /**
         * Chunks a row-indexed worker across the pool. Runs in the Tensor method, never inside a pool
         * worker, so the caller waiting on the returned futures is legal.
         * @brief Threaded row launcher. Returns the futures for the caller to drain.
         * @param rows Total row count to cover.
         * @param range_fn Callable taking (slot, begin, end); copied into each task.
         * @param t Logits tensor the workers read.
         * @param targets Targets tensor the workers read.
         * @return Futures of every submitted chunk; empty when the work ran inline.
         */
        template<typename RangeFn>
        std::vector<std::shared_future<void>> launch_rows(const size_t rows, RangeFn&& range_fn, const Tensor& t, const Tensor& targets) {
            const auto t_count = static_cast<size_t>(IncludedCPUFeatures().threads);
            if (t_count <= 1 || rows <= t_count * 8) {
                range_fn(0, 0, rows);
                return {};
            }
            auto& threadMgr = Threading::GetCPUThreadPool();
            std::vector<std::shared_future<void>> futures;
            const auto chunk = (rows + t_count - 1) / t_count;
            auto slot = size_t{0};
            for (auto t = size_t{0}; t < t_count; ++t) {
                const auto begin = t * chunk;
                if (begin >= rows) {
                    break;
                }
                // range_fn is captured by value; its own captures reference the
                // caller's partials, which stay alive until drain() returns.
                futures.push_back(threadMgr.submit([range_fn, slot, begin, end = std::min(begin + chunk, rows)] { range_fn(slot, begin, end); }, Threading::TaskPriority::Normal).share());
                ++slot;
            }
            t.give_futures(futures);
            targets.give_futures(futures);
            return futures;
        }

        /**
         * Rethrows the first task error through future::get.
         * @brief Drain futures on the caller thread.
         * @param futures Futures to wait on.
         */
        void drain(const std::vector<std::shared_future<void>>& futures) {
            for (const auto& f : futures) {
                f.get();
            }
        }

        /**
         * Row losses accumulate into partials[slot] in fp64. Not a friend of Tensor: the target
         * dtype/base pointer/offset arrive precomputed.
         * @brief Threaded softmax_ce partial over a row range.
         * @param logits Row-major logits, already offset to the first element.
         * @param tgt_dtype Dtype of the targets buffer; selects the index reader.
         * @param tgt_data Base pointer of the targets buffer.
         * @param tgt_offset Element offset into the targets buffer.
         * @param cols_store Stored element count per logits row.
         * @param cols_eff Effective class count per row (doubled for packed dtypes).
         * @param slot Index of this chunk's entry in partials.
         * @param begin First row computed.
         * @param end One past the last row computed.
         * @param partials Per-chunk fp64 loss sums; only partials[slot] is written.
         */
        template<typename T, typename Acc>
        void softmax_ce_rows(const T* logits, const DataTypes::DType tgt_dtype, const void* tgt_data, const size_t tgt_offset,
                             const size_t cols_store, const size_t cols_eff,
                             const size_t slot, const size_t begin, const size_t end, std::vector<float64_t>& partials) {
            auto acc = 0.0;
            /**
             * Reads each row's target index as I and accumulates that row's loss into acc.
             * @brief Accumulate the row losses for one target dtype.
             * @param t_ptr Targets buffer typed as I, already offset.
             * @note Lambda subcall. Self-execution prohibited.
             */
            auto run = [&]<typename I>(const I* t_ptr) {
                for (auto r = begin; r < end; ++r) {
                    const auto idx = target_index(t_ptr, r, cols_eff);
                    acc += softmax_ce_row<T, Acc>(logits + r * cols_store, cols_eff, idx);
                }
            };
            using DType = DataTypes::DType;
            switch (tgt_dtype) {
                case DType::Int8: run(static_cast<const int8_t*>(tgt_data) + tgt_offset); break;
                case DType::UInt8: run(static_cast<const uint8_t*>(tgt_data) + tgt_offset); break;
                case DType::Int16: run(static_cast<const int16_t*>(tgt_data) + tgt_offset); break;
                case DType::UInt16: run(static_cast<const uint16_t*>(tgt_data) + tgt_offset); break;
                case DType::Int32: run(static_cast<const int32_t*>(tgt_data) + tgt_offset); break;
                case DType::UInt32: run(static_cast<const uint32_t*>(tgt_data) + tgt_offset); break;
                case DType::Int64: run(static_cast<const int64_t*>(tgt_data) + tgt_offset); break;
                case DType::UInt64: run(static_cast<const uint64_t*>(tgt_data) + tgt_offset); break;
                case DType::Int4x2: run(static_cast<const int4x2_t*>(tgt_data) + tgt_offset); break;
                [[unlikely]] default: throw ErrorHandling::UnReachableCode();
            }
            partials[slot] = acc;
        }

        /**
         * Floats narrow RNE (halfs via the unaccel pathway). Packed dtypes duplicate into both lanes
         * (matches the unpacked->packed conversion contract). Integers clamp to the dtype range and
         * truncate; a NaN loss stores 0 (there is no integer NaN to store).
         * Not a friend of Tensor: the output dtype and base pointer arrive precomputed.
         * @brief Store the narrowed softmax_ce mean loss.
         * @param out_dtype Dtype of the output element.
         * @param out_data Base pointer of the output element.
         * @param mean Mean loss in fp64.
         */
        void store_mean(const DataTypes::DType out_dtype, void* const out_data, const float64_t mean) {
            using DType = DataTypes::DType;
            switch (out_dtype) {
                case DType::Float32: *static_cast<float32_t*>(out_data) = static_cast<float32_t>(mean); break;
                case DType::Float64: *static_cast<float64_t*>(out_data) = mean; break;
                case DType::Float16: *static_cast<float16_t*>(out_data) = narrow_lane<float16_t>(static_cast<float32_t>(mean)); break;
                case DType::BFloat16: *static_cast<bfloat16_t*>(out_data) = narrow_lane<bfloat16_t>(static_cast<float32_t>(mean)); break;
                case DType::Float16x2: {
                    const auto lane = Intrinsics::detail::bit_cast_memcpy<uint16_t>(narrow_lane<float16_t>(static_cast<float32_t>(mean)));
                    *static_cast<float16x2_t*>(out_data) = pack_lanes<float16x2_t>(lane, lane);
                    break;
                }
                case DType::BFloat16x2: {
                    const auto lane = Intrinsics::detail::bit_cast_memcpy<uint16_t>(narrow_lane<bfloat16_t>(static_cast<float32_t>(mean)));
                    *static_cast<bfloat16x2_t*>(out_data) = pack_lanes<bfloat16x2_t>(lane, lane);
                    break;
                }
                case DType::Int4x2: {
                    auto v = mean;
                    if (std::isnan(v)) {
                        v = 0.0;
                    }
                    if (v <= -8.0) {
                        v = -8.0;
                    } else if (v >= 7.0) {
                        v = 7.0;
                    }
                    const auto n = static_cast<int32_t>(v);
                    *static_cast<int4x2_t*>(out_data) = pack_int4(n, n);
                    break;
                }
                default: {
                    /**
                     * Maps NaN to 0, clamps the mean to I's range, truncates and stores.
                     * @brief Store the mean loss as integer type I.
                     * @param p Output element.
                     * @note Lambda subcall. Self-execution prohibited.
                     */
                    auto store_int = [&]<typename I>(I* p) {
                        auto v = mean;
                        if (std::isnan(v)) {
                            v = 0.0;
                        }
                        const auto lo = static_cast<float64_t>(std::numeric_limits<I>::lowest());
                        const auto hi = static_cast<float64_t>(std::numeric_limits<I>::max());
                        if (v <= lo) {
                            v = lo;
                        } else if (v >= hi) {
                            v = hi;
                        }
                        *p = static_cast<I>(v);
                    };
                    switch (out_dtype) {
                        case DType::Int8: store_int(static_cast<int8_t*>(out_data)); break;
                        case DType::UInt8: store_int(static_cast<uint8_t*>(out_data)); break;
                        case DType::Int16: store_int(static_cast<int16_t*>(out_data)); break;
                        case DType::UInt16: store_int(static_cast<uint16_t*>(out_data)); break;
                        case DType::Int32: store_int(static_cast<int32_t*>(out_data)); break;
                        case DType::UInt32: store_int(static_cast<uint32_t*>(out_data)); break;
                        case DType::Int64: store_int(static_cast<int64_t*>(out_data)); break;
                        case DType::UInt64: store_int(static_cast<uint64_t*>(out_data)); break;
                        [[unlikely]] default: throw ErrorHandling::UnReachableCode();
                    }
                    break;
                }
            }
        }
    }

    /**
     * Rows are chunked across the pool into fp64 partial sums, drained on the caller thread, then
     * averaged and narrowed into out. Throws ErrorHandling::IndexOutOfRange for a bad target.
     * @brief CPU softmax cross-entropy: mean over rows of logsumexp(row) - row[target].
     * @param t Rank-2 logits tensor (rows x classes), CPU resident, contiguous.
     * @param targets Rank-1 integer class indices, one per row.
     * @param out Single-element output tensor in the logits dtype.
     */
    void softmax_ce_cpu(const Tensor& t, const Tensor& targets, Tensor& out) {
        if (t.device != Device::CPU || targets.device != Device::CPU || out.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(t.device));
        }
        using DType = DataTypes::DType;
        const auto rows = t.shape[0];
        const auto cols_store = t.shape[1];
        const auto packed = t.dtype == DType::Float16x2 || t.dtype == DType::BFloat16x2 || t.dtype == DType::Int4x2;
        const auto cols_eff = cols_store * (packed ? size_t{2} : size_t{1});
        // softmax_ce_rows is not a Tensor friend: hand it the precomputed triple.
        const auto tgt_dtype = targets.dtype;
        const auto* const tgt_data = targets.data;
        const auto tgt_offset = targets.offset;
        std::vector<float64_t> partials(IncludedCPUFeatures().threads, 0.0);
        switch (t.dtype) {
            case DType::Float32: {
                const auto* logits = static_cast<const float32_t*>(t.data) + t.offset;
                drain(launch_rows(rows, [&](const size_t slot, const size_t b, const size_t e) { softmax_ce_rows<float32_t, float32_t>(logits, tgt_dtype, tgt_data, tgt_offset, cols_store, cols_eff, slot, b, e, partials); }, t, targets));
                break;
            }
            case DType::Float64: {
                const auto* logits = static_cast<const float64_t*>(t.data) + t.offset;
                drain(launch_rows(rows, [&](const size_t slot, const size_t b, const size_t e) { softmax_ce_rows<float64_t, float64_t>(logits, tgt_dtype, tgt_data, tgt_offset, cols_store, cols_eff, slot, b, e, partials); }, t, targets));
                break;
            }
            case DType::Float16: {
                const auto* logits = static_cast<const float16_t*>(t.data) + t.offset;
                drain(launch_rows(rows, [&](const size_t slot, const size_t b, const size_t e) { softmax_ce_rows<float16_t, float32_t>(logits, tgt_dtype, tgt_data, tgt_offset, cols_store, cols_eff, slot, b, e, partials); }, t, targets));
                break;
            }
            case DType::BFloat16: {
                const auto* logits = static_cast<const bfloat16_t*>(t.data) + t.offset;
                drain(launch_rows(rows, [&](const size_t slot, const size_t b, const size_t e) { softmax_ce_rows<bfloat16_t, float32_t>(logits, tgt_dtype, tgt_data, tgt_offset, cols_store, cols_eff, slot, b, e, partials); }, t, targets));
                break;
            }
            case DType::Float16x2: {
                const auto* logits = static_cast<const float16x2_t*>(t.data) + t.offset;
                drain(launch_rows(rows, [&](const size_t slot, const size_t b, const size_t e) { softmax_ce_rows<float16x2_t, float32_t>(logits, tgt_dtype, tgt_data, tgt_offset, cols_store, cols_eff, slot, b, e, partials); }, t, targets));
                break;
            }
            case DType::BFloat16x2: {
                const auto* logits = static_cast<const bfloat16x2_t*>(t.data) + t.offset;
                drain(launch_rows(rows, [&](const size_t slot, const size_t b, const size_t e) { softmax_ce_rows<bfloat16x2_t, float32_t>(logits, tgt_dtype, tgt_data, tgt_offset, cols_store, cols_eff, slot, b, e, partials); }, t, targets));
                break;
            }
            case DType::Int8: case DType::UInt8: case DType::Int16: case DType::UInt16:
            case DType::Int32: case DType::UInt32: case DType::Int64: case DType::UInt64:
            case DType::Int4x2: {
                /**
                 * Launches the row workers for integer logits type T and drains them.
                 * @brief Run softmax_ce rows for one integer logits dtype.
                 * @param logits Logits buffer typed as T, already offset.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                auto launch_int = [&]<typename T>(const T* logits) {
                    drain(launch_rows(rows, [&](const size_t slot, const size_t b, const size_t e) { softmax_ce_rows<T, float64_t>(logits, tgt_dtype, tgt_data, tgt_offset, cols_store, cols_eff, slot, b, e, partials); }, t, targets));
                };
                switch (t.dtype) {
                    case DType::Int8: launch_int(static_cast<const int8_t*>(t.data) + t.offset); break;
                    case DType::UInt8: launch_int(static_cast<const uint8_t*>(t.data) + t.offset); break;
                    case DType::Int16: launch_int(static_cast<const int16_t*>(t.data) + t.offset); break;
                    case DType::UInt16: launch_int(static_cast<const uint16_t*>(t.data) + t.offset); break;
                    case DType::Int32: launch_int(static_cast<const int32_t*>(t.data) + t.offset); break;
                    case DType::UInt32: launch_int(static_cast<const uint32_t*>(t.data) + t.offset); break;
                    case DType::Int64: launch_int(static_cast<const int64_t*>(t.data) + t.offset); break;
                    case DType::UInt64: launch_int(static_cast<const uint64_t*>(t.data) + t.offset); break;
                    case DType::Int4x2: launch_int(static_cast<const int4x2_t*>(t.data) + t.offset); break;
                    [[unlikely]] default: throw ErrorHandling::UnReachableCode();
                }
                break;
            }
            [[unlikely]] default: {
                throw ErrorHandling::NotImplemented();
            }
        }
        auto total = 0.0;
        for (const auto p : partials) {
            total += p;
        }
        store_mean(out.dtype, out.data, total / static_cast<float64_t>(rows));
    }
}

namespace BeanTensor::Tensors {

    /**
     * Validates shapes, devices and target dtype (integer kinds only), syncs both operands, then routes to the
     * backend. Only the logits join the autograd graph, as targets never receive a gradient.
     * @brief Mean softmax cross-entropy over the rows of rank-2 logits.
     * @param targets Rank-1 integer class indices, one per row, same device.
     * @return Single-element tensor in the logits dtype.
     */
    Tensor Tensor::softmax_ce(const Tensor& targets) const {
        if (this->ndim != 2) {
            throw ErrorHandling::MismatchedShapes("softmax_ce requires rank-2 logits (rows x classes)");
        }
        if (targets.ndim != 1 || targets.shape[0] != this->shape[0]) {
            throw ErrorHandling::MismatchedShapes("softmax_ce requires rank-1 targets with one class index per row");
        }
        if (this->shape[0] == 0 || this->shape[1] == 0) {
            throw ErrorHandling::MismatchedShapes("softmax_ce requires non-empty logits");
        }
        if (this->device != targets.device) {
            throw ErrorHandling::MismatchedDevices();
        }
        if (!this->is_contiguous || !targets.is_contiguous) {
            throw ErrorHandling::NotImplemented();
        }
        {
            using DType = DataTypes::DType;
            switch (targets.dtype) {
                case DType::Int8: case DType::UInt8: case DType::Int16: case DType::UInt16:
                case DType::Int32: case DType::UInt32: case DType::Int64: case DType::UInt64:
                case DType::Int4x2: {
                    break;
                }
                default: {
                    throw ErrorHandling::MismatchedDTypes();
                }
            }
        }
        sync();
        targets.sync();
        auto out = Tensor(std::vector<size_t>{1}, this->dtype, this->device, false);
        if (this->device == Device::CPU) {
            detail::softmax_ce_cpu(*this, targets, out);
        } else if (this->device == Device::GPU) {
            detail::softmax_ce_gpu(*this, targets, out);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        // Targets are class indices and never receive a gradient: only the
        // logits decide whether the output joins the graph.
        if (this->requires_grad) {
            out.wire_node(detail::softmax_ce_g_node, *this, &targets);
        }
        return out;
    }
}
