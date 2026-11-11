/**
 * @file BceWithLogits.hpp
 * @author GreenBean
 * @brief CPU-only build fall back for bce_with_logits - throws and does nothing else.
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
        // CPU-only build: no GPU backend exists to link against.
        /**
         * @brief CPU-only build stub for the GPU bce_with_logits backend. Always throws.
         */
        inline void bce_with_logits_gpu(const Tensor&, const Tensor&, const Tensor&) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
    }
}
#endif
