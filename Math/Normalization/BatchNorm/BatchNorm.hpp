/**
 * @file BatchNorm.hpp
 * @author GreenBean
 * @brief CPU only build fall back for batch normalization - throws and does nothing else.
 * @version 0.1
 * @date 2026-10-30
 */
#pragma once

#include "Helpers/Errors.h"
#include "Tensor/DTypes.h"

#if !defined(USE_HIP) && !defined(USE_CUDA)
namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {
        // CPU-only build: no GPU backend exists to link against.
        /**
         * @brief CPU-only build stub for the GPU batch normalization forward. Always throws.
         */
        inline void batch_norm_gpu(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                   Tensor&, Tensor&, bool, float64_t, float64_t) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
    }
}
#endif
