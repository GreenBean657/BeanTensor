#pragma once
#include <cmath>

#include "Intrinsics/ConversionMethods.h"
#include "Tensor/Dtypes.h"

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
            (std::is_floating_point_v<src> &&
            std::is_floating_point_v<dst>) ||
            (std::is_integral_v<src> &&
            std::is_integral_v<dst>)
    [[nodiscard]] dst standard2standard(const src& src_v, bool& flag_met, ConversionClampMethod method = ConversionClampMethod::PERMISSIVE) {
        assert(!Casting::is_nonstandard_float_kind_v<src>);
        assert(!Casting::is_nonstandard_float_kind_v<dst>);
        //Conventional Casting
        auto min = std::numeric_limits<dst>::min();
        auto max = std::numeric_limits<dst>::max();
        if (src_v > max) {
            // Overflow
            if (method != ConversionClampMethod::PERMISSIVE) {
                flag_met = true;
            }
            return static_cast<dst>(max);
        }
        if (src_v < min) {
            // Underflow
            if (method != ConversionClampMethod::PERMISSIVE) {
                flag_met = true;
            }
            return static_cast<dst>(min);
        }
        if (std::isnan(src_v)) {
            //Nan
            if (method != ConversionClampMethod::PERMISSIVE) {
                flag_met = true;
            }
        } else if (std::isinf(src_v)) {
            // Infinity
            if (method == ConversionClampMethod::STRICT) {
                flag_met = true;
            }
        }
        return static_cast<dst>(min);
    }
}
