#pragma once
#include "casting_detail.h"
using namespace BeanTensor;
namespace BeanTensor::Casting {

    /*
    enum class DType {
        BFloat16 = -99,
        Float16 = -100,
        Float32 = -101,
        Float64 = -102,
        Int64 = 11,
        Int32 = 10,
        Int16 = 9,
        Int8 = 8,
        UInt64 = 21,
        UInt32 = 20,
        UInt16 = 19,
        UInt8 = 18
    };
    */
    inline DType get_highest_precision(const DType& a, const DType& b) {

        return DType::BFloat16;
    }


}