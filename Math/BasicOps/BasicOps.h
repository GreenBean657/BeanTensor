/**
 * @file BasicOps.h
 * @author GreenBean
 * @brief Entry point for basic tensor ops (cpu vs gpu): sgd_update, sum, argmax.
 * reshape is metadata-only and needs no backend dispatch.
 * @version 0.1
 * @date 2026-10-18
 */
#pragma once

#include <cstdint>

#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {

        /**
         * @brief CPU in-place SGD step, w += -lr * g (weights, gradients, learning rate).
         */
        void sgd_update_cpu(Tensor&, const Tensor&, float64_t);
        /**
         * @brief GPU in-place SGD step, w += -lr * g (weights, gradients, learning rate).
         */
        void sgd_update_gpu(Tensor&, const Tensor&, float64_t);

        /**
         * @brief CPU full reduction, sums every element of the input into the scalar output tensor.
         */
        void sum_cpu(const Tensor&, Tensor&);
        /**
         * @brief GPU full reduction, sums every element of the input into the scalar output tensor.
         */
        void sum_gpu(const Tensor&, Tensor&);

        /**
         * @brief CPU flat index of the largest element of the input, written to the output integer.
         */
        void argmax_cpu(const Tensor&, uint64_t&);
        /**
         * @brief GPU flat index of the largest element of the input, written to the output integer.
         */
        void argmax_gpu(const Tensor&, uint64_t&);
    }
}

#include "BasicOps.hpp"
