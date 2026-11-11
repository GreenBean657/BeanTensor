/**
 * @file AvgPool2d.hpp
 * @author GreenBean
 * @brief CPU only build fall back for AvgPool2d forward/backward GPU backends.
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
         * @brief CPU-only build stub for GPU AvgPool2d forward.
         * @note Always throws ErrorHandling::GPUTaskOnCPUBuild.
         */
        inline void avg_pool2d_gpu(const Tensor&, Tensor&, uint32_t, uint32_t) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }

        /**
         * @brief CPU-only build stub for GPU AvgPool2d backward.
         * @note Always throws ErrorHandling::GPUTaskOnCPUBuild.
         */
        inline void avg_pool2d_g_gpu(const Tensor&, Tensor&, uint32_t, uint32_t) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
    }
}
#endif
