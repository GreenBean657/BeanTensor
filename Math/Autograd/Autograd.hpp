/**
 * @file Autograd.hpp
 * @author GreenBean
 * @brief CPU only build fall back for autograd backward ops - throws and does nothing else.
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
        /**
         * @brief CPU-only build stub for the GPU grad accumulation backend: throws ErrorHandling::GPUTaskOnCPUBuild, since no GPU backend exists to link against.
         */
        inline void grad_acc_gpu(Tensor&, const Tensor&) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
        /**
         * @brief CPU-only build stub for the GPU last-two-dims transpose backend: throws ErrorHandling::GPUTaskOnCPUBuild, since no GPU backend exists to link against.
         */
        inline void transpose_last2_gpu(const Tensor&, Tensor&) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
        /**
         * @brief CPU-only build stub for the GPU relu backward backend: throws ErrorHandling::GPUTaskOnCPUBuild, since no GPU backend exists to link against.
         */
        inline void relu_g_gpu(const Tensor&, const Tensor&, Tensor&) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
        /**
         * @brief CPU-only build stub for the GPU bias_add backward backend: throws ErrorHandling::GPUTaskOnCPUBuild, since no GPU backend exists to link against.
         */
        inline void bias_add_g_gpu(const Tensor&, Tensor&) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
        /**
         * @brief CPU-only build stub for the GPU softmax_ce backward backend: throws ErrorHandling::GPUTaskOnCPUBuild, since no GPU backend exists to link against.
         */
        inline void softmax_ce_g_gpu(const Tensor&, const Tensor&, const Tensor&, Tensor&) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
    }
}
#endif
