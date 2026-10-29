/**
 * @file ElementwiseShared.h
 * @author GreenBean
 * @brief Host-side helpers shared across the elementwise backends: float-kind probe,
 * scalar narrowing into a tensor dtype (the PERMISSIVE convert contract), and
 * broadcast offset decomposition. Reuses the half/packed/int4 lane helpers and
 * the threaded launcher from LinearAlgebraShared.h.
 * @version 0.2
 * @date 2026-10-28
 */
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

#include "Math/Elementwise/Elementwise.h"
#include "Math/LinearAlgebra/LinearAlgebraShared.h"
#include "Tensor/Tensor.h"

namespace BeanTensor::Tensors::detail::elementwise_shared {
    // ElementwiseBcast's fixed array extents must match the tensor rank cap.
    static_assert(BT_TENSOR_MAX_DIMS == 4, "ElementwiseBcast is sized for BT_TENSOR_MAX_DIMS == 4");

    /**
     * @brief Is this dtype any float kind (native, half, or packed-half)?
     * @param dt Dtype to check.
     * @return True for Float16/Float16x2/BFloat16/BFloat16x2/Float32/Float64.
     */
    [[nodiscard]] inline bool is_float_kind(const DataTypes::DType& dt) {
        switch (dt) {
            case DataTypes::DType::Float16:
            case DataTypes::DType::Float16x2:
            case DataTypes::DType::BFloat16:
            case DataTypes::DType::BFloat16x2:
            case DataTypes::DType::Float32:
            case DataTypes::DType::Float64:
                return true;
            default:
                return false;
        }
    }

    /**
     * Mirrors the PERMISSIVE scalar_convert contract documented in
     * Math/Autograd/GradDType.h: float narrows RNE (halfs narrow through an fp32
     * intermediate — double rounding), float -> int truncates toward zero and
     * clamps to the target's [lowest, max], NaN -> 0 for integer kinds; packed
     * dtypes duplicate the narrowed value into both lanes; int4x2 clamps to
     * [-8, 7] per nibble (both nibbles).
     * @brief Narrow a float64_t scalar into an element of dtype T.
     * @tparam T Target element type.
     * @param v Scalar value.
     * @return The scalar as T under the PERMISSIVE conversion contract.
     */
    template<typename T>
    [[nodiscard]] T narrow_scalar(const float64_t v) {
        if constexpr (std::is_same_v<T, float32_t>) {
            return static_cast<float32_t>(v);
        } else if constexpr (std::is_same_v<T, float64_t>) {
            return v;
        } else if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
            return linearalgebra_shared::narrow_lane<T>(static_cast<float32_t>(v));
        } else if constexpr (DataTypes::requires_unpack_v<T>) {
            using H = linearalgebra_shared::packed_lane_t<T>;
            const auto lane = linearalgebra_shared::narrow_lane<H>(static_cast<float32_t>(v));
            const auto bits = Intrinsics::detail::bit_cast_memcpy<uint16_t>(lane);
            return linearalgebra_shared::pack_lanes<T>(bits, bits);
        } else if constexpr (std::is_same_v<T, int4x2_t>) {
            if (__builtin_isnan(v)) {
                return linearalgebra_shared::pack_int4(0, 0);
            }
            const auto t = std::trunc(v);
            const auto clamped = t > 7.0 ? int32_t{7} : t < -8.0 ? int32_t{-8} : static_cast<int32_t>(t);
            return linearalgebra_shared::pack_int4(clamped, clamped);
        } else {
            static_assert(std::is_integral_v<T>);
            if (__builtin_isnan(v)) {
                return T{0};
            }
            const auto t = std::trunc(v);
            constexpr auto lo = static_cast<float64_t>(std::numeric_limits<T>::lowest());
            // hi for uint64/int64 rounds UP in fp64 (2^64 / 2^63), so t >= hi
            // catches every out-of-range value before the UB cast.
            if (
                constexpr auto hi = static_cast<float64_t>(std::numeric_limits<T>::max());
                t >= hi
                ) {
                return std::numeric_limits<T>::max();
            }
            if (t <= lo) {
                return std::numeric_limits<T>::lowest();
            }
            return static_cast<T>(t);
        }
    }

    /**
     * Decomposes the flat output index into output coordinates and accumulates
     * each operand's element offset from its broadcast-adjusted strides. The
     * exact fast path (identical shapes) maps both offsets to the flat index.
     * The GPU backends carry a device-side twin of this loop (device code
     * cannot call this host inline); the two must stay behaviorally identical.
     * @brief Element offsets of both operands for output flat index i.
     * @param plan Broadcast execution plan.
     * @param i Flat output index.
     * @return (a_offset, b_offset) element offsets.
     */
    [[nodiscard]] inline std::pair<size_t, size_t> bcast_offsets(const ElementwiseBcast& plan, const size_t i) {
        if (plan.exact) {
            return {i, i};
        }
        auto ao = size_t{0};
        auto bo = size_t{0};
        auto rem = i;
        for (auto d = size_t{0}; d < plan.out_ndim; ++d) {
            const auto coord = rem / plan.out_strides[d];
            rem -= coord * plan.out_strides[d];
            ao += coord * plan.a_strides[d];
            bo += coord * plan.b_strides[d];
        }
        return {ao, bo};
    }
}
