/**
 * @file IndexingGShared.h
 * @author GreenBean
 * @brief Shared helpers for the indexing/view backward ops (index_select,
 * gather, cat, slice): little-endian index reads (forward already bounds-checked
 * the index tensor; backward re-reads it as a graph constant) and per-element
 * gradient accumulation mirroring the grad_acc dtype contracts. Header-only:
 * everything here is a template or inline.
 * @version 0.1
 * @date 2026-11-14
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "Math/Autograd/AutogradShared.h"
#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors::detail::indexing_g_shared {
    /**
     * Reads width bytes little-endian; the forward pass already bounds-checked
     * every index (and signed dtypes rejected negatives there), so the
     * zero-extended unsigned read preserves every legal value. All supported
     * targets are little-endian.
     * @brief Read one index element without a typed pointer.
     * @param p First byte of the index element.
     * @param width Element width in bytes (1, 2, 4 or 8).
     * @return The index value, zero-extended to uint64_t.
     */
    [[nodiscard]] inline uint64_t read_index_le(const uint8_t* p, const size_t width) {
        auto v = uint64_t{0};
        for (auto b = size_t{0}; b < width; ++b) {
            v |= static_cast<uint64_t>(p[b]) << (8 * b);
        }
        return v;
    }

    /**
     * Accumulates one gradient element, w += d, under the grad_acc dtype
     * contracts (Math/Autograd/GradAcc): fp32/fp64 add in storage width, halfs
     * widen to fp32 and narrow RNE, packed dtypes compute per lane, integers
     * wrap through the unsigned twin, int4x2 wraps per nibble. Duplicate-index
     * scatter calls this in ascending pick order per destination element, so
     * the accumulation order is deterministic and identical on every backend.
     * @brief Per-element gradient accumulation (all dtypes).
     * @param w Accumulator element, updated in place.
     * @param d Delta element added into w.
     */
    template<typename T>
    void acc_element(T* w, const T d) {
        using namespace autograd_shared;
        if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
            *w = *w + d;
        } else if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
            *w = narrow_lane<T>(widen_lane(*w) + widen_lane(d));
        } else if constexpr (std::is_same_v<T, float16x2_t> || std::is_same_v<T, bfloat16x2_t>) {
            using H = packed_lane_t<T>;
            const auto w_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(*w);
            const auto d_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(d);
            const auto lo = narrow_lane<H>(widen_lane(lane_lo<H>(w_bits)) + widen_lane(lane_lo<H>(d_bits)));
            const auto hi = narrow_lane<H>(widen_lane(lane_hi<H>(w_bits)) + widen_lane(lane_hi<H>(d_bits)));
            *w = pack_lanes<T>(Intrinsics::detail::bit_cast_memcpy<uint16_t>(lo),
                               Intrinsics::detail::bit_cast_memcpy<uint16_t>(hi));
        } else if constexpr (std::is_same_v<T, int4x2_t>) {
            *w = pack_int4(int4_lo(*w) + int4_lo(d), int4_hi(*w) + int4_hi(d));
        } else {
            using U = std::make_unsigned_t<T>;
            *w = static_cast<T>(static_cast<U>(static_cast<U>(*w) + static_cast<U>(d)));
        }
    }
}
