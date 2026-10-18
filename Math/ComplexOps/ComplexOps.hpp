/**
 * @file ComplexOps.hpp
 * @author GreenBean
 * @brief CPU only build fall back for complex ops - throws and does nothing else.
 * @version 0.1
 * @date 2026-10-18
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
         * @brief CPU-only build stub for GPU bias_add.
         * @note Always throws ErrorHandling::GPUTaskOnCPUBuild.
         */
        inline void bias_add_gpu(const Tensor&, const Tensor&, Tensor&) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
        /**
         * @brief CPU-only build stub for GPU relu.
         * @note Always throws ErrorHandling::GPUTaskOnCPUBuild.
         */
        inline void relu_gpu(const Tensor&, Tensor&) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
        /**
         * @brief CPU-only build stub for GPU softmax_ce.
         * @note Always throws ErrorHandling::GPUTaskOnCPUBuild.
         */
        inline void softmax_ce_gpu(const Tensor&, const Tensor&, Tensor&) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
    }
}
#endif
