/**
 * @file OptimizationShared.h
 * @author GreenBean
 * @brief Shared scalar/packed lane helpers, threaded reduction launcher, and
 * function-pointer typedefs used by the optimization op (sgd_update).
 * Header-only: everything here is a template or inline.
 * @version 0.2
 * @date 2026-10-25
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <future>
#include <type_traits>
#include <vector>

#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Tensors::detail::optimization_shared {
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
     * Per-chunk partials are merged on the CALLER thread (runs in the Tensor method, never inside a pool worker,
     * so waiting is legal). Deterministic merge: partials combine in chunk order. Small inputs run inline.
     * @brief Threaded reduction launcher, chunks a reduction range worker across the pool.
     * @param numel Element count to cover.
     * @param range_fn Callable taking (slot, begin, end); slot indexes the caller's partials.
     * @param t Tensor receiving the futures.
     * @return Futures for the caller to drain.
     */
    template<typename RangeFn>
    std::vector<std::shared_future<void>> launch_reduce(const size_t numel, RangeFn&& range_fn, const Tensor& t) {
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
        return futures;
    }

    /**
     * Drains EVERY future before rethrowing the first stored worker error: a
     * chunk's range_fn may reference the caller's stack (partials), so the
     * caller's frame must outlive ALL chunks even when one threw — rethrowing
     * on the first get() would let the rest run against a destroyed frame.
     * @brief Drain futures on the caller thread, rethrowing the first error after all join.
     * @param futures Futures to wait on; rethrows the first stored worker error.
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

    /**
     * @brief Function pointer type shared by the scalar and SIMD SGD range kernels.
     */
    template<typename T>
    using SgdScalarFn = void(*)(T*, const T*, T, size_t, size_t);
}
