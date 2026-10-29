/**
 * @file SpatialCudaShared.h
 * @author GreenBean
 * @brief CUDA-side shared helpers for the split spatial translation units
 * (max_pool2d): grid sizing, BF16 narrowing, and int4x2 pack/unpack helpers.
 * Reuses the activation family's CUDA shared header to avoid duplication.
 * @version 0.1
 * @date 2026-10-28
 */
#pragma once

#include "Math/Activations/ActivationsCudaShared.h"

namespace BeanTensor::Tensors::detail::spatial_gpu {
    using namespace activations_gpu;
}
