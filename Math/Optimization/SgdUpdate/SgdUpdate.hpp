/**
 * @file SgdUpdate.hpp
 * @author GreenBean
 * @brief CPU only build fall back for sgd_update - throws and does nothing else.
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
        // CPU-only build: no GPU backend exists to link against.
        /**
         * @brief CPU-only build stub for the GPU SGD step. Always throws.
         */
        inline void sgd_update_gpu(const Tensor&, const Tensor&, float64_t) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
    }
}
#endif
