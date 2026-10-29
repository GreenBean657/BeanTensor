/**
 * @file DropoutKernels.h
 * @brief SIMD kernel declarations for dropout mask construction.
 * @version 0.1
 * @date 2026-10-28
 */
#pragma once

#include "Tensor/DTypes.h"

namespace BeanTensor::Math::Dropout {
    // Dropout uses scalar range kernels composed from existing lane primitives;
    // no new intrinsics wrappers are added. Future SIMD variants would land here.
}
