/**
 * @file SoftmaxCE.h
 * @author GreenBean
 * @brief Entry point for softmax_ce (cpu vs gpu)
 * @version 0.1
 * @date 2026-10-20
 */
#pragma once

#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {

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

#include "SoftmaxCE.hpp"
