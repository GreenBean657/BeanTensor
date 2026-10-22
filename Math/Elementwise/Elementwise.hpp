/**
 * @file Elementwise.hpp
 * @author GreenBean
 * @brief CPU only build fall back for elementwise ops - throws and does nothing else.
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
         * @brief CPU-only build stub for GPU tensor-tensor elementwise.
         * @note Always throws ErrorHandling::GPUTaskOnCPUBuild.
         */
        inline void elementwise_gpu(const Tensor&, const Tensor&, Tensor&, const ElementwiseOp, const ElementwiseBcast&) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
        /**
         * @brief CPU-only build stub for GPU tensor-scalar elementwise.
         * @note Always throws ErrorHandling::GPUTaskOnCPUBuild.
         */
        inline void elementwise_scalar_gpu(const Tensor&, const float64_t&, Tensor&, const ElementwiseOp) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
        /**
         * @brief CPU-only build stub for GPU unary elementwise.
         * @note Always throws ErrorHandling::GPUTaskOnCPUBuild.
         */
        inline void elementwise_unary_gpu(const Tensor&, Tensor&, const ElementwiseUnaryOp) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
    }
}
#endif
