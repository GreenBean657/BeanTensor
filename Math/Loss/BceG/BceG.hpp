/**
 * @file BceG.hpp
 * @author GreenBean
 * @brief CPU-only build fall back for bce backward - throws and does nothing else.
 * @version 0.1
 * @date 2026-11-10
 */
#pragma once

#include "Helpers/Errors.h"
#include "Tensor/DTypes.h"

#if !defined(USE_HIP) && !defined(USE_CUDA)
namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {
        /**
         * @brief CPU-only build stub for the GPU bce backward backend: throws ErrorHandling::GPUTaskOnCPUBuild.
         */
        inline void bce_g_gpu(const Tensor&, const Tensor&, const Tensor&, const Tensor&) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
    }
}
#endif
