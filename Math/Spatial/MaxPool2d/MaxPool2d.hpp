/**
 * @file MaxPool2d.hpp
 * @author GreenBean
 * @brief CPU only build fall back for MaxPool2d forward/backward GPU backends.
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
         * @brief CPU-only build stub for GPU MaxPool2d forward.
         * @note Always throws ErrorHandling::GPUTaskOnCPUBuild.
         */
        inline void max_pool2d_gpu(const Tensor&, Tensor&, uint32_t, uint32_t) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }

        /**
         * @brief CPU-only build stub for GPU MaxPool2d backward.
         * @note Always throws ErrorHandling::GPUTaskOnCPUBuild.
         */
        inline void max_pool2d_g_gpu(const Tensor&, const Tensor&, Tensor&, uint32_t, uint32_t) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
    }
}
#endif
