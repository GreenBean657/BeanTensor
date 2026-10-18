/**
 * @file Matmul.h
 * @author GreenBean
 * @brief Entry point for matmul (cpu vs gpu)
 * @version 0.1
 * @date 2026-10-18
 */
#pragma once

#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {

        /**
         * Queues the CPU GEMM onto the thread pool; errors materialize at sync().
         * @brief CPU backend entry point for matmul.
         */
        void matmul_cpu(const Tensor&, const Tensor&, Tensor&);

        /**
         * Queues the GPU GEMM onto the thread pool; errors materialize at sync().
         * @brief GPU backend entry point for matmul (CUDA or HIP build).
         */
        void matmul_gpu(const Tensor&, const Tensor&, Tensor&);
    }
}

#include "Matmul.hpp"
