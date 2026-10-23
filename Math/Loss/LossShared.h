/**
 * @file LossShared.h
 * @author GreenBean
 * @brief Shared threaded reduction launchers, the mean-narrowing store, and
 * function-pointer typedefs used by the loss ops (mse forward and backward).
 * The scalar/packed lane helpers (widen_lane, narrow_lane, lane_lo/hi,
 * pack_lanes, the i4 lane helpers, packed_lane_t, drain) are NOT duplicated
 * here: LossShared re-exports the canonical copies from
 * Math/Optimization/OptimizationShared.h, so a loss narrowing can never drift
 * from the project's single conversion pathway. Header-only.
 * @version 0.1.1
 * @date 2026-10-23
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <future>
#include <limits>
#include <type_traits>
#include <vector>

#include "Math/Optimization/OptimizationShared.h"
#include "Tensor/Tensor.h"

namespace BeanTensor::Tensors::detail::loss_shared {
    // The canonical lane helpers live in optimization_shared (included
    // above); re-export them here so the loss TUs' `using namespace
    // loss_shared;` keeps resolving every name it used when LossShared
    // mirrored them locally.
    using optimization_shared::widen_lane;
    using optimization_shared::narrow_lane;
    using optimization_shared::packed_lane;
    template<typename P>
    using packed_lane_t = typename optimization_shared::packed_lane<P>::type;
    using optimization_shared::lane_lo;
    using optimization_shared::lane_hi;
    using optimization_shared::pack_lanes;
    using optimization_shared::i4_lane_lo;
    using optimization_shared::i4_lane_hi;
    using optimization_shared::i4_pack;
    using optimization_shared::drain;

    /**
     * Floats narrow RNE (halfs through the unaccel pathway). Packed dtypes
     * duplicate the scalar mean into both lanes (the loss is a single scalar;
     * this matches softmax_ce's store_mean contract). Integers clamp to the
     * dtype range and truncate toward zero; a NaN mean stores 0 (there is no
     * integer NaN). int4x2 clamps to the signed nibble range [-8, 7].
     * @brief Store a narrowed mean loss into a single output element.
     * @param out_dtype Dtype of the output element.
     * @param out_data Base pointer of the output element.
     * @param mean Mean loss in fp64 (already divided by the element count).
     */
    inline void store_loss_mean(const DataTypes::DType out_dtype, void* const out_data, const float64_t mean) {
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
                if (__builtin_isnan(v)) {
                    v = 0.0;
                }
                if (v <= -8.0) {
                    v = -8.0;
                } else if (v >= 7.0) {
                    v = 7.0;
                }
                const auto n = static_cast<int32_t>(v);
                *static_cast<int4x2_t*>(out_data) = i4_pack(static_cast<int8_t>(n), static_cast<int8_t>(n));
                break;
            }
            default: {
                /**
                 * Maps NaN to 0, clamps the mean to I's range, truncates toward zero and stores.
                 * @brief Store the mean loss as integer type I.
                 * @param p Output element.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                auto store_int = [&]<typename I>(I* p) {
                    auto v = mean;
                    if (__builtin_isnan(v)) {
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

    /**
     * Per-chunk partials are merged on the CALLER thread (runs in the Tensor
     * method, never inside a pool worker, so waiting is legal). Deterministic
     * merge: partials combine in chunk order. Small inputs run inline. Both
     * read tensors receive the futures so teardown drains the workers.
     * @brief Threaded reduction launcher, chunks a range worker across the pool.
     * @param numel Element count to cover.
     * @param range_fn Callable taking (slot, begin, end); slot indexes the caller's partials.
     * @param t First tensor receiving the futures.
     * @param extra Second tensor receiving the futures.
     * @return Futures for the caller to drain.
     */
    template<typename RangeFn>
    std::vector<std::shared_future<void>> launch_loss_reduce(const size_t numel, RangeFn&& range_fn, const Tensor& t, const Tensor& extra) {
        const auto t_count = static_cast<size_t>(IncludedCPUFeatures().threads);
        if (t_count <= 1 || numel <= t_count * 1024) {
            range_fn(0, 0, numel);
            return {};
        }
        auto& threadMgr = Threading::GetCPUThreadPool();
        std::vector<std::shared_future<void>> futures;
        const auto chunk = (numel + t_count - 1) / t_count;
        auto slot = size_t{0};
        for (auto tid = size_t{0}; tid < t_count; ++tid) {
            const auto begin = tid * chunk;
            if (begin >= numel) {
                break;
            }
            // range_fn is captured by value; its own captures reference the
            // caller's partials, which stay alive until drain() returns.
            futures.push_back(threadMgr.submit([range_fn, slot, begin, end = std::min(begin + chunk, numel)] { range_fn(slot, begin, end); }, Threading::TaskPriority::Normal).share());
            ++slot;
        }
        t.give_futures(futures);
        extra.give_futures(futures);
        return futures;
    }

    /**
     * Same launcher shape as launch_loss_reduce, but for the three-tensor
     * elementwise backward (pred, target, dpred all receive the futures).
     * @brief Threaded elementwise launcher, chunks a range worker across the pool.
     * @param numel Element count to cover.
     * @param range_fn Callable taking (begin, end).
     * @param a First tensor receiving the futures.
     * @param b Second tensor receiving the futures.
     * @param out Third tensor receiving the futures.
     * @return Futures for the caller to drain.
     */
    template<typename RangeFn>
    std::vector<std::shared_future<void>> launch_loss_rows(const size_t numel, RangeFn&& range_fn, const Tensor& a, const Tensor& b, const Tensor& out) {
        const auto t_count = static_cast<size_t>(IncludedCPUFeatures().threads);
        if (t_count <= 1 || numel <= t_count * 1024) {
            range_fn(0, numel);
            return {};
        }
        auto& threadMgr = Threading::GetCPUThreadPool();
        std::vector<std::shared_future<void>> futures;
        const auto chunk = (numel + t_count - 1) / t_count;
        for (auto ti = size_t{0}; ti < t_count; ++ti) {
            const auto begin = ti * chunk;
            if (begin >= numel) {
                break;
            }
            futures.push_back(threadMgr.submit([range_fn, begin, end = std::min(begin + chunk, numel)] { range_fn(begin, end); }, Threading::TaskPriority::Normal).share());
        }
        a.give_futures(futures);
        b.give_futures(futures);
        out.give_futures(futures);
        return futures;
    }
}
