/**
 * @file SoftmaxG.hpp
 * @author GreenBean
 * @brief CPU only build fall back for standalone softmax backward - throws and does nothing else.
 * @version 0.1
 * @date 2026-10-28
 */
#pragma once

#include "Helpers/Errors.h"
#include "Tensor/DTypes.h"

#if !defined(USE_HIP) && !defined(USE_CUDA)
namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {
        /**
         * @brief CPU-only build stub for GPU softmax backward.
         * @note Always throws ErrorHandling::GPUTaskOnCPUBuild.
         */
        inline void softmax_g_gpu(const Tensor&, const Tensor&, const Tensor&) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
    }
}
#endif
