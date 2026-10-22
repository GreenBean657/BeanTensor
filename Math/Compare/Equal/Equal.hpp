/**
 * @file Equal.hpp
 * @author GreenBean
 * @brief CPU only build fall back for equal - throws and does nothing else.
 * @version 0.1
 * @date 2026-10-22
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
         * @brief CPU-only build stub for the GPU equal op. Always throws.
         */
        inline void equal_gpu(const Tensor&, const Tensor&, Tensor&) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
    }
}
#endif
