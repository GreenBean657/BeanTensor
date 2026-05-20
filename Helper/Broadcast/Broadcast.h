#pragma once
#include "Tensor/tensor.h"
#include <cstddef>

namespace BeanTensor::Helpers {
    inline bool broadcast_compatible(
        const size_t* shapeA,
        const size_t ndims_A,
        const size_t* shapeB,
        const size_t ndims_B,
        size_t* out,
        size_t* out_ndims
        ) {
        *out_ndims = ndims_A > ndims_B ? ndims_A : ndims_B;
        for (int i = BT_TENSOR_MAX_DIMS - 1; i >= 0; i--) {
            size_t a = shapeA[i];
            size_t b = shapeB[i];

            a = a ? a : 1;
            b = b ? b : 1;

            if (a == b || (b == 1)) {
                out[i] = a;
            } else if (a == 1) {
                out[i] = b;
            } else {
                return false;
            }
        }
        return true;
    }
};