#pragma once
#include "casting_detail.h"
using namespace BeanTensor;
namespace BeanTensor::Casting {
    template <class T> __always_inline constexpr bool is_float_kind_v =
        std::is_floating_point_v<T>
        || std::is_same_v<T, Casting::bfloat16_t>
        || std::is_same_v<T, Casting::float16_t>;

    inline DType get_highest_precision(const DType& a, const DType& b) {

        return DType::BFloat16;
    }
}
