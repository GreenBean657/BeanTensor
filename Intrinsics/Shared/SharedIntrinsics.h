/**
 * @file SharedIntrinsics.h
 * @author GreenBean
 * @brief Conversion helpers shared between the scalar, AVX2 and AVX512 dtype paths.
 * @version 0.1
 * @date 2026-10-15
 */
#pragma once
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

#include "Intrinsics/ConversionMethods.h"
#include "Tensor/DTypes.h"

namespace BeanTensor::Intrinsics::detail {
    /**
     * Convert within standard DTypes
     * @tparam src SRC Type to convert from
     * @tparam dst DST Type to convert to
     * @param src_v SRC Value to convert from
     * @param flag_met Log overflow and other casting errors.
     * @param method Permissive Method for log overflow methods.
     * @return DST Value to convert to
     */
    template<typename src, typename dst> requires
            (
                std::is_floating_point_v<src> &&
                std::is_floating_point_v<dst>
            ) || (
                std::is_integral_v<src> &&
                std::is_integral_v<dst>
            ) || (
                !DataTypes::is_nonstandard_float_kind_v<src> &&
                !DataTypes::is_nonstandard_float_kind_v<dst>
            )
    [[nodiscard]] dst standard2standard(const src& src_v, bool& flag_met, const ConversionClampMethod method = ConversionClampMethod::PERMISSIVE) {
        if constexpr (std::is_integral_v<src> && std::is_integral_v<dst>) {
            if (std::cmp_greater(src_v, std::numeric_limits<dst>::max())) {
                // Overflow
                if (method != ConversionClampMethod::PERMISSIVE) flag_met = true;
                return std::numeric_limits<dst>::max();
            }
            if (std::cmp_less(src_v, std::numeric_limits<dst>::min())) {
                // Underflow
                if (method != ConversionClampMethod::PERMISSIVE) flag_met = true;
                return std::numeric_limits<dst>::min();
            }
            return static_cast<dst>(src_v);
        } else {
            // Compare in the common (wider) type explicitly — identical semantics to the
            // implicit promotion, but clean under -Werror=double-promotion.
            using cmp_t = std::common_type_t<src, dst>;
            if (static_cast<cmp_t>(src_v) > static_cast<cmp_t>(std::numeric_limits<dst>::max())) {
                // Overflow
                if (method != ConversionClampMethod::PERMISSIVE) flag_met = true;
                return std::numeric_limits<dst>::max();
            }
            if (static_cast<cmp_t>(src_v) < static_cast<cmp_t>(std::numeric_limits<dst>::lowest())) {
                // Underflow
                if (method != ConversionClampMethod::PERMISSIVE) flag_met = true;
                return std::numeric_limits<dst>::lowest();
            }
            if (std::isnan(src_v)) {
                //Nan
                if (method != ConversionClampMethod::PERMISSIVE) flag_met = true;
            } else if (std::isinf(src_v)) {
                // Infinity
                if (method == ConversionClampMethod::STRICT) flag_met = true;
            }
            return static_cast<dst>(src_v);
        }
    }
}
