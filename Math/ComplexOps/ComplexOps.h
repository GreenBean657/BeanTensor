/**
 * @file ComplexOps.h
 * @author GreenBean
 * @brief Entry point for complex tensor ops (cpu vs gpu): bias_add, relu, softmax_ce.
 * @version 0.1
 * @date 2026-10-18
 */
#pragma once

#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {

        /**
         * @brief CPU bias_add backend: out = input + bias broadcast over the last dimension.
         */
        void bias_add_cpu(const Tensor&, const Tensor&, Tensor&);
        /**
         * @brief GPU bias_add backend: out = input + bias broadcast over the last dimension.
         */
        void bias_add_gpu(const Tensor&, const Tensor&, Tensor&);

        /**
         * @brief CPU relu backend: out = max(input, 0), NaN propagating.
         */
        void relu_cpu(const Tensor&, Tensor&);
        /**
         * @brief GPU relu backend: out = max(input, 0), NaN propagating.
         */
        void relu_gpu(const Tensor&, Tensor&);

        /**
         * @brief CPU softmax cross-entropy backend: mean loss of rank-2 logits against integer class targets.
         */
        void softmax_ce_cpu(const Tensor&, const Tensor&, Tensor&);
        /**
         * @brief GPU softmax cross-entropy backend: mean loss of rank-2 logits against integer class targets.
         */
        void softmax_ce_gpu(const Tensor&, const Tensor&, Tensor&);
    }
}

#include "ComplexOps.hpp"
