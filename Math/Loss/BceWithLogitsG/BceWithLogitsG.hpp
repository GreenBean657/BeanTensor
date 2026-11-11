/**
 * @file BceWithLogitsG.hpp
 * @author GreenBean
 * @brief CPU-only build fall back for bce_with_logits backward - throws and does nothing else.
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
         * @brief CPU-only build stub for the GPU bce_with_logits backward backend: throws ErrorHandling::GPUTaskOnCPUBuild.
         */
        inline void bce_with_logits_g_gpu(const Tensor&, const Tensor&, const Tensor&, const Tensor&) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
    }
}
#endif
