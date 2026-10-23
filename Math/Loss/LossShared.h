/**
 * @file LossShared.h
 * @author GreenBean
 * @brief Shared scalar/packed lane helpers, threaded reduction launcher, the
 * mean-narrowing store, and function-pointer typedefs used by the loss ops
 * (mse forward and backward). Header-only: everything here is a template or
 * inline. The helpers mirror OptimizationShared.h's contracts (unaccel
 * conversion pathway, RNE narrowing, PERMISSIVE-clamp-adjacent semantics) so
 * a loss narrowing never invents new conversion behavior.
 * @version 0.1
 * @date 2026-10-23
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <future>
#include <limits>
#include <type_traits>
#include <vector>

#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Tensors::detail::loss_shared {
    /**
     * Routes through the unaccel conversion pathway; conversion flags are not tracked.
     * @brief Widen one fp16/bf16 lane to fp32.
     * @param v Half-precision lane.
     * @return The lane value as fp32.
     */
    template<typename H> requires (std::is_same_v<H, float16_t> || std::is_same_v<H, bfloat16_t>)
    [[nodiscard]] float32_t widen_lane(const H& v) {
        if constexpr (std::is_same_v<H, float16_t>) {
            return Intrinsics::detail::unaccel_fp16_to_fp32_untracked(v);
        } else {
            return Intrinsics::detail::unaccel_bf16_to_fp32(v);
        }
    }

    /**
     * Routes through the unaccel conversion pathway; untracked, PERMISSIVE clamp method.
     * @brief Narrow one fp32 accumulator to fp16/bf16 (RNE).
     * @param v FP32 accumulator.
     * @return The value rounded to the half-precision lane type.
     */
    template<typename H> requires (std::is_same_v<H, float16_t> || std::is_same_v<H, bfloat16_t>)
    [[nodiscard]] H narrow_lane(const float32_t v) {
        if constexpr (std::is_same_v<H, float16_t>) {
            return Intrinsics::detail::unaccel_fp32_to_fp16_untracked(v);
        } else {
            return Intrinsics::detail::unaccel_fp32_to_bf16_untracked(v);
        }
    }

    /**
     * Specialized per packed dtype; the primary template is declared only.
     * @brief Lane type of a packed dtype (float16x2_t -> float16_t, bfloat16x2_t -> bfloat16_t).
     */
    template<typename P> struct packed_lane;
    template<> struct packed_lane<float16x2_t> { using type = float16_t; };
    template<> struct packed_lane<bfloat16x2_t> { using type = bfloat16_t; };
    template<typename P> using packed_lane_t = packed_lane<P>::type;

    /**
     * Reads bits[15:0] of the packed 32-bit unit.
     * @brief Extract the low lane of a packed value, via its bit pattern.
     * @param bits Raw 32-bit packed unit.
     * @return The low lane as the half-precision lane type.
     */
    template<typename H>
    [[nodiscard]] H lane_lo(const uint32_t bits) {
        return Intrinsics::detail::bit_cast_memcpy<H>(static_cast<uint16_t>(bits & 0xFFFFu));
    }

    /**
     * Reads bits[31:16] of the packed 32-bit unit.
     * @brief Extract the high lane of a packed value, via its bit pattern.
     * @param bits Raw 32-bit packed unit.
     * @return The high lane as the half-precision lane type.
     */
    template<typename H>
    [[nodiscard]] H lane_hi(const uint32_t bits) {
        return Intrinsics::detail::bit_cast_memcpy<H>(static_cast<uint16_t>(bits >> 16));
    }

    /**
     * Inverse of lane_lo/lane_hi.
     * @brief Pack two narrowed lanes back into a packed dtype value.
     * @param lo Raw bits of the low lane.
     * @param hi Raw bits of the high lane.
     * @return The packed value with lo in bits[15:0] and hi in bits[31:16].
     */
    template<typename P>
    [[nodiscard]] P pack_lanes(const uint16_t lo, const uint16_t hi) {
        return Intrinsics::detail::bit_cast_memcpy<P>(static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16));
    }

    /**
     * @brief Signed low lane of an int4x2 byte (bits[3:0], sign-extended).
     * @param v Packed int4x2 byte.
     * @return Lane value in [-8, 7].
     */
    [[nodiscard]] inline int8_t i4_lane_lo(const int4x2_t v) {
        const auto n = static_cast<int8_t>(Intrinsics::detail::bit_cast_memcpy<uint8_t>(v) & 0x0Fu);
        return n > 7 ? static_cast<int8_t>(n - 16) : n;
    }

    /**
     * @brief Signed high lane of an int4x2 byte (bits[7:4], sign-extended).
     * @param v Packed int4x2 byte.
     * @return Lane value in [-8, 7].
     */
    [[nodiscard]] inline int8_t i4_lane_hi(const int4x2_t v) {
        const auto n = static_cast<int8_t>(Intrinsics::detail::bit_cast_memcpy<uint8_t>(v) >> 4);
        return n > 7 ? static_cast<int8_t>(n - 16) : n;
    }

    /**
     * Each lane keeps only its low 4 bits, so out-of-range values wrap mod 16.
     * @brief Pack two signed lanes into an int4x2 byte.
     * @param lo Low lane value.
     * @param hi High lane value.
     * @return The packed int4x2 byte.
     */
    [[nodiscard]] inline int4x2_t i4_pack(const int8_t lo, const int8_t hi) {
        const auto lo_bits = static_cast<uint8_t>(lo) & 0x0Fu;
        const auto hi_bits = static_cast<uint8_t>(hi) & 0x0Fu;
        return Intrinsics::detail::bit_cast_memcpy<int4x2_t>(static_cast<uint8_t>(lo_bits | static_cast<uint8_t>(hi_bits << 4)));
    }

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

    /**
     * @brief Drain futures on the caller thread.
     * @param futures Futures to wait on; rethrows the first stored worker error.
     */
    inline void drain(const std::vector<std::shared_future<void>>& futures) {
        for (const auto& f : futures) {
            f.get();
        }
    }
}
