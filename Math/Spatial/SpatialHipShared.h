/**
 * @file SpatialHipShared.h
 * @author GreenBean
 * @brief HIP/ROCm-side shared helpers for the split spatial translation units
 * (max_pool2d): grid sizing, BF16 narrowing, and int4x2 pack/unpack helpers.
 * Reuses the activation family's HIP shared header to avoid duplication.
 * @version 0.1
 * @date 2026-10-28
 */
#pragma once

#include "Math/Activations/ActivationsHipShared.h"

namespace BeanTensor::Tensors::detail::spatial_gpu {
    using namespace activations_gpu;
}
