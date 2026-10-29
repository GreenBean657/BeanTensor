/**
 * @file SoftmaxShared.h
 * @author GreenBean
 * @brief Helpers shared across the standalone softmax CPU backends (scalar and
 * AVX2/AVX512): half/packed/int4 lane widening and packing, the numerically
 * stable per-row softmax kernel, and the threaded row launcher.
 * @version 0.1
 * @date 2026-10-28
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <future>
#include <limits>
#include <type_traits>
#include <vector>

#include "Helpers/CPUThreading.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/Native/DTypeConversions.h"
#include "Tensor/Tensor.h"

#include "Math/Activations/ActivationsShared.h"

namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail::softmax_shared {
        using namespace activations_shared;

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
                return (j & 1) != 0 ? int4_hi(row[j >> 1]) : int4_lo(row[j >> 1]);
            } else {
                return static_cast<float64_t>(row[j]);
            }
        }

        /**
         * @brief Store one accumulated probability value into the output dtype.
         * Floats narrow RNE; integers clamp/truncate and map NaN to 0.
         * @param v_raw Accumulated probability value.
         * @return The value narrowed to T.
         */
        template<typename T, typename Acc>
        [[nodiscard]] T narrow_prob(const Acc v_raw) {
            if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                return v_raw;
            } else if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
                return narrow_lane<T>(static_cast<float32_t>(v_raw));
            } else {
                auto v = static_cast<float64_t>(v_raw);
                if (__builtin_isnan(v)) {
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
         * Numerically stable per-row softmax: max-subtract, exp, sum, normalize.
         * A NaN logit poisons the whole row (max ignores NaN, exp(NaN-m) = NaN).
         * @brief Scalar row softmax (non-packed dtypes).
         * @param row Logits row of storage elements.
         * @param out_row Output probability row (written).
         * @param cols Effective class count per row.
         */
        template<typename T, typename Acc>
        void softmax_row(const T* row, T* out_row, const size_t cols) {
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
                out_row[j] = narrow_prob<T, Acc>(std::exp(widen(j) - m) / s);
            }
        }

        /**
         * @brief Scalar row softmax for packed x2 / int4x2 rows.
         * @param row Logits row of packed storage elements.
         * @param out_row Output probability row (written).
         * @param cols_eff Effective class count (two per packed element).
         */
        template<typename T, typename Acc>
        void softmax_row_packed(const T* row, T* out_row, const size_t cols_eff) {
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
            for (auto e = size_t{0}; e < cols_eff / 2; ++e) {
                const auto j = 2 * e;
                if constexpr (std::is_same_v<T, int4x2_t>) {
                    out_row[e] = pack_int4(
                        std::clamp(static_cast<int32_t>(narrow_prob<int8_t, Acc>(std::exp(widen(j) - m) / s)), -8, 7),
                        std::clamp(static_cast<int32_t>(narrow_prob<int8_t, Acc>(std::exp(widen(j + 1) - m) / s)), -8, 7));
                } else {
                    using H = packed_lane_t<T>;
                    const auto lo = narrow_lane<H>(static_cast<float32_t>(std::exp(widen(j) - m) / s));
                    const auto hi = narrow_lane<H>(static_cast<float32_t>(std::exp(widen(j + 1) - m) / s));
                    out_row[e] = pack_lanes<T>(Intrinsics::detail::bit_cast_memcpy<uint16_t>(lo),
                                               Intrinsics::detail::bit_cast_memcpy<uint16_t>(hi));
                }
            }
        }

        /**
         * Chunk a row-indexed worker (begin, end) across the pool; futures land
         * on the touched tensors and are returned for the caller to drain.
         * Small inputs run inline and return no futures.
         * CCD-aware: each worker writes its slice of the output span, so the gate
         * is the bytes written. When that fits one CCD's LLC the chunks go to the
         * calling thread's CCD group so the cache-resident destination is written
         * locally; otherwise they go to the whole pool as before.
         * @brief Threaded row launcher without partials.
         * @param rows Total row count.
         * @param range_fn Callable (begin, end) over a row range; copied into each task.
         * @param out_bytes Bytes of the output span the workers write across all chunks.
         * @param t Input tensor (receives the futures).
         * @param out Output tensor (receives the futures).
         * @return Futures of the submitted chunks; empty if the work ran inline.
         */
        template<typename RangeFn>
        std::vector<std::shared_future<void>> launch_rows(const size_t rows, RangeFn&& range_fn, const size_t out_bytes, const Tensor& t, const Tensor& out) {
            const auto t_count = static_cast<size_t>(IncludedCPUFeatures().threads);
            if (t_count <= 1 || rows <= t_count * 8) {
                range_fn(0, rows);
                return {};
            }
            auto& threadMgr = Threading::GetCPUThreadPool();
            std::vector<std::shared_future<void>> futures;
            const auto chunk = (rows + t_count - 1) / t_count;
            const auto keep_local = KeepWorkLocal(out_bytes);
            const auto group = keep_local ? CcdGroupOfCurrentCpu() : uint32_t{0};
            for (auto ti = size_t{0}; ti < t_count; ++ti) {
                const auto begin = ti * chunk;
                if (begin >= rows) {
                    break;
                }
                if (keep_local) {
                    futures.push_back(threadMgr.submit_to_group([range_fn, begin, end = std::min(begin + chunk, rows)] { range_fn(begin, end); }, group, Threading::TaskPriority::Normal).share());
                } else {
                    futures.push_back(threadMgr.submit([range_fn, begin, end = std::min(begin + chunk, rows)] { range_fn(begin, end); }, Threading::TaskPriority::Normal).share());
                }
            }
            t.give_futures(futures);
            out.give_futures(futures);
            return futures;
        }

        /**
         * Drains EVERY future before rethrowing the first worker error.
         * @brief Drain futures on the caller thread, rethrowing the first error after all join.
         * @param futures Futures to wait on.
         */
        inline void drain(const std::vector<std::shared_future<void>>& futures) {
            auto first_error = std::exception_ptr{};
            for (const auto& f : futures) {
                try {
                    f.get();
                }
                catch (...) {
                    if (!first_error) {
                        first_error = std::current_exception();
                    }
                }
            }
            if (first_error) {
                std::rethrow_exception(first_error);
            }
        }
    }
}
