/**
 * @file Matmul.hpp
 * @author GreenBean
 * @brief CPU only build fall back for matmul - throws and does nothing else.
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
        // CPU-only build: no GPU GEMM backend exists to link against.
        /**
         * No GPU GEMM backend exists in a CPU-only build.
         * @note Always throws ErrorHandling::GPUTaskOnCPUBuild.
         * @brief CPU-only build stub for matmul_gpu.
         */
        inline void matmul_gpu(const Tensor&, const Tensor&, const Tensor&) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
    }
}
#endif
