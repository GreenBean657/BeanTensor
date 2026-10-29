/**
 * @file DropoutShared.h
 * @brief CPU-side helpers shared by the dropout mask implementation: float-kind
 * probe and the scalar threshold-scale kernels built from existing lane
 * primitives (widen/narrow/pack). No new intrinsics are introduced.
 * @version 0.1
 * @date 2026-10-28
 */
#pragma once

#include <cmath>
#include <cstdint>
#include <type_traits>

#include "Math/Activations/ActivationsShared.h"
#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors::detail::dropout_shared {

    using BeanTensor::Tensors::detail::activations_shared::widen_lane;
    using BeanTensor::Tensors::detail::activations_shared::narrow_lane;
    using BeanTensor::Tensors::detail::activations_shared::lane_lo;
    using BeanTensor::Tensors::detail::activations_shared::lane_hi;
    using BeanTensor::Tensors::detail::activations_shared::pack_lanes;
    template<typename P>
    using packed_lane_t = BeanTensor::Tensors::detail::activations_shared::packed_lane_t<P>;

    /**
     * @brief Is this dtype supported by train_mask? Float16/BFloat16/Float32/Float64.
     * @param dt Dtype to check.
     * @return True for the four native/half float kinds.
     */
    [[nodiscard]] inline bool is_dropout_float_kind(const DataTypes::DType& dt) {
        switch (dt) {
            case DataTypes::DType::Float16:
            case DataTypes::DType::BFloat16:
            case DataTypes::DType::Float32:
            case DataTypes::DType::Float64:
                return true;
            default:
                return false;
        }
    }

    /**
     * @brief Native float threshold-scale: out = (rand >= p) ? scale : 0.
     */
    template<typename T>
    [[nodiscard]] T mask_scalar_native(const T rand, const float64_t p, const float64_t scale) {
        return rand >= static_cast<T>(p) ? static_cast<T>(scale) : T{0};
    }

    /**
     * @brief Half-kind threshold-scale in widened fp32, then narrow RNE.
     */
    template<typename H>
    [[nodiscard]] H mask_scalar_half(const H rand, const float64_t p, const float64_t scale) {
        const auto wide = widen_lane(rand);
        const auto out = wide >= static_cast<float32_t>(p) ? static_cast<float32_t>(scale) : 0.0f;
        return narrow_lane<H>(out);
    }

    /**
     * @brief Packed half-kind threshold-scale per lane, repack on store.
     */
    template<typename P>
    [[nodiscard]] P mask_scalar_packed(const P rand, const float64_t p, const float64_t scale) {
        using H = packed_lane_t<P>;
        const auto bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(rand);
        const auto lo = widen_lane(lane_lo<H>(bits));
        const auto hi = widen_lane(lane_hi<H>(bits));
        const auto lo_out = lo >= static_cast<float32_t>(p) ? static_cast<float32_t>(scale) : 0.0f;
        const auto hi_out = hi >= static_cast<float32_t>(p) ? static_cast<float32_t>(scale) : 0.0f;
        return pack_lanes<P>(Intrinsics::detail::bit_cast_memcpy<uint16_t>(narrow_lane<H>(lo_out)),
                             Intrinsics::detail::bit_cast_memcpy<uint16_t>(narrow_lane<H>(hi_out)));
    }

}
