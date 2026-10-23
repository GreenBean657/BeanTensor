/**
 * @file CompareShared.h
 * @author GreenBean
 * @brief Shared scalar/packed lane helpers and the threaded elementwise launcher
 * used by the comparison ops (equal). Header-only: everything here is a
 * template or inline.
 * @version 0.1
 * @date 2026-10-22
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <future>
#include <type_traits>
#include <vector>

#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Tensors::detail::compare_shared {
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
     * Ordered-comparison equality without float ==: a NaN on either side is never equal,
     * -0.0 equals 0.0 (both ordered compares hold).
     * @brief Float equality key (IEEE == semantics, NaN excluded).
     * @param a Left key.
     * @param b Right key.
     * @return T/F - the keys are equal.
     */
    [[nodiscard]] inline bool float_eq(const float64_t a, const float64_t b) {
        return !(a < b) && !(b < a) && !__builtin_isnan(a) && !__builtin_isnan(b);
    }

    // --------------------------- launchers ---------------------------

    /**
     * Chunks an elementwise range worker across the CPU thread pool. Futures land on every tensor the
     * workers touch so teardown drains them (they hold raw buffer pointers). Small inputs run inline.
     * Mirrors linearalgebra_shared::launch_elementwise.
     * @brief Threaded elementwise launcher.
     * @param worker Callable taking (begin, end); copied into each task.
     * @param numel Total element count to cover.
     * @param a First tensor the workers read.
     * @param b Second tensor the workers read.
     * @param out Tensor the workers write.
     */
    template<typename Worker>
    void launch_compare(Worker&& worker, const size_t numel, const Tensor& a, const Tensor& b, const Tensor& out) {
        const auto t_count = static_cast<size_t>(IncludedCPUFeatures().threads);
        if (t_count <= 1 || numel <= t_count * 1024) {
            worker(0, numel);
            return;
        }
        auto& threadMgr = Threading::GetCPUThreadPool();
        std::vector<std::shared_future<void>> futures;
        const auto chunk = (numel + t_count - 1) / t_count;
        for (auto t = size_t{0}; t < t_count; ++t) {
            const auto begin = t * chunk;
            if (begin >= numel) {
                break;
            }
            // worker is captured BY VALUE: these futures outlive this call
            // (they land on the tensors), so a reference would dangle.
            futures.push_back(threadMgr.submit([worker, begin, end = std::min(begin + chunk, numel)] { worker(begin, end); }, Threading::TaskPriority::Normal).share());
        }
        a.give_futures(futures);
        b.give_futures(futures);
        out.give_futures(futures);
    }
}
