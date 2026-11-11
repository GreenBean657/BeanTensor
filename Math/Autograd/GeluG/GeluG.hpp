/**
 * @file GeluG.hpp
 * @author GreenBean
 * @brief CPU only build fall back for GELU backward - throws and does nothing else.
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
         * @brief CPU-only build stub for the GPU GELU backward backend: throws ErrorHandling::GPUTaskOnCPUBuild, since no GPU backend exists to link against.
         */
        inline void gelu_g_gpu(const Tensor&, const Tensor&, const Tensor&) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
    }
}
#endif
