/**
 * @file SoftmaxCEG.hpp
 * @author GreenBean
 * @brief CPU only build fall back for softmax_ce backward - throws and does nothing else.
 * @version 0.1
 * @date 2026-10-20
 */
#pragma once

#include "Helpers/Errors.h"
#include "Tensor/DTypes.h"

#if !defined(USE_HIP) && !defined(USE_CUDA)
namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {
        /**
         * @brief CPU-only build stub for the GPU softmax_ce backward backend: throws ErrorHandling::GPUTaskOnCPUBuild, since no GPU backend exists to link against.
         */
        inline void softmax_ce_g_gpu(const Tensor&, const Tensor&, const Tensor&, const Tensor&) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
    }
}
#endif
