/**
 * @file ComplexOpsShared.h
 * @author GreenBean
 * @brief Helpers shared across the complex tensor ops (bias_add, relu, softmax_ce):
 * half/packed/int4 lane widening and packing, plus the threaded elementwise launcher.
 * @version 0.1
 * @date 2026-10-20
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <future>
#include <vector>

#include "Helpers/CPUThreading.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/Native/DTypeConversions.h"
#include "Tensor/Tensor.h"

namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail::complexops_shared {
        /**
         * Dispatches at compile time: fp16 and bf16 each use their own unaccel conversion.
         * @brief Widen one fp16/bf16 lane to fp32 (unaccel pathway, untracked).
         * @param v Half-width lane value.
         * @return The lane widened to fp32.
         */
        template<typename H>
        [[nodiscard]] float32_t widen_lane(const H& v) {
            if constexpr (std::is_same_v<H, float16_t>) {
                return Intrinsics::detail::unaccel_fp16_to_fp32_untracked(v);
            } else {
                return Intrinsics::detail::unaccel_bf16_to_fp32(v);
            }
        }

        /**
         * Dispatches at compile time: fp16 and bf16 each use their own unaccel conversion.
         * @brief Narrow one fp32 accumulator to fp16/bf16 (unaccel pathway, untracked/PERMISSIVE).
         * @param v FP32 accumulator value.
         * @return The value narrowed to H, RNE.
         */
        template<typename H>
        [[nodiscard]] H narrow_lane(const float32_t v) {
            if constexpr (std::is_same_v<H, float16_t>) {
                return Intrinsics::detail::unaccel_fp32_to_fp16_untracked(v);
            } else {
                return Intrinsics::detail::unaccel_fp32_to_bf16_untracked(v);
            }
        }

        /**
         * Only float16x2_t and bfloat16x2_t are specialized; any other P fails to compile.
         * @brief Lane type of a packed dtype (float16x2_t -> float16_t, bfloat16x2_t -> bfloat16_t).
         */
        template<typename P> struct packed_lane;
        template<> struct packed_lane<float16x2_t> { using type = float16_t; };
        template<> struct packed_lane<bfloat16x2_t> { using type = bfloat16_t; };
        template<typename P> using packed_lane_t = typename packed_lane<P>::type;

        /**
         * Reinterprets the low 16 bits through memcpy, never a pointer cast.
         * @brief Low lane of a packed value, via its bit pattern.
         * @param bits Packed 32-bit pattern, (lo, hi).
         * @return The low 16 bits as lane type H.
         */
        template<typename H>
        [[nodiscard]] H lane_lo(const uint32_t bits) {
            return Intrinsics::detail::bit_cast_memcpy<H>(static_cast<uint16_t>(bits & 0xFFFFu));
        }

        /**
         * Reinterprets the high 16 bits through memcpy, never a pointer cast.
         * @brief High lane of a packed value, via its bit pattern.
         * @param bits Packed 32-bit pattern, (lo, hi).
         * @return The high 16 bits as lane type H.
         */
        template<typename H>
        [[nodiscard]] H lane_hi(const uint32_t bits) {
            return Intrinsics::detail::bit_cast_memcpy<H>(static_cast<uint16_t>(bits >> 16));
        }

        /**
         * @brief Pack two narrowed lanes back into a packed dtype value.
         * @param lo Bit pattern of the low lane.
         * @param hi Bit pattern of the high lane.
         * @return Packed value P with lo in the low half, hi in the high half.
         */
        template<typename P>
        [[nodiscard]] P pack_lanes(const uint16_t lo, const uint16_t hi) {
            return Intrinsics::detail::bit_cast_memcpy<P>(static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16));
        }

        // --------------------------- int4x2 helpers ---------------------------

        /**
         * Low 4 bits, sign-extended from bit 3.
         * @brief Sign-extended low nibble of a packed int4 pair.
         * @param v Packed int4 pair.
         * @return Low nibble in [-8, 7].
         */
        [[nodiscard]] inline int32_t int4_lo(const int4x2_t v) {
            const auto n = static_cast<int32_t>(Intrinsics::detail::bit_cast_memcpy<uint8_t>(v) & 0x0Fu);
            return n >= 8 ? n - 16 : n;
        }

        /**
         * High 4 bits, sign-extended from bit 3.
         * @brief Sign-extended high nibble of a packed int4 pair.
         * @param v Packed int4 pair.
         * @return High nibble in [-8, 7].
         */
        [[nodiscard]] inline int32_t int4_hi(const int4x2_t v) {
            const auto n = static_cast<int32_t>(Intrinsics::detail::bit_cast_memcpy<uint8_t>(v) >> 4);
            return n >= 8 ? n - 16 : n;
        }

        /**
         * Only the low 4 bits of each input survive, so out-of-range sums wrap.
         * @brief Pack two signed values into an int4 pair (modular, low 4 bits each).
         * @param lo Value for the low nibble.
         * @param hi Value for the high nibble.
         * @return Packed int4 pair.
         */
        [[nodiscard]] inline int4x2_t pack_int4(const int32_t lo, const int32_t hi) {
            return Intrinsics::detail::bit_cast_memcpy<int4x2_t>(
                static_cast<uint8_t>((lo & 0xF) | ((hi & 0xF) << 4)));
        }

        // --------------------------- launchers ---------------------------

        /**
         * Chunks an elementwise range worker across the CPU thread pool. Futures land on every tensor the
         * workers touch so teardown drains them (they hold raw buffer pointers). Small inputs run inline.
         * @brief Threaded elementwise launcher.
         * @param worker Callable taking (begin, end); copied into each task.
         * @param numel Total element count to cover.
         * @param a First tensor the workers read.
         * @param b Second tensor the workers read.
         * @param out Tensor the workers write.
         */
        template<typename Worker>
        void launch_elementwise(Worker&& worker, const size_t numel, const Tensor& a, const Tensor& b, const Tensor& out) {
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
}
