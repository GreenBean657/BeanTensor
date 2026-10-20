/**
 * @file Argmax.hpp
 * @author GreenBean
 * @brief CPU only build fall back for argmax - throws and does nothing else.
 * @version 0.1
 * @date 2026-10-20
 */
#pragma once

#include <cstdint>

#include "Helpers/Errors.h"
#include "Tensor/DTypes.h"

#if !defined(USE_HIP) && !defined(USE_CUDA)
namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {
        // CPU-only build: no GPU backend exists to link against.
        /**
         * @brief CPU-only build stub for the GPU argmax reduction. Always throws.
         */
        inline void argmax_gpu(const Tensor&, uint64_t&) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
    }
}
#endif
