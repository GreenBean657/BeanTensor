/**
 * @file BceWithLogits.h
 * @author GreenBean
 * @brief Entry point for bce_with_logits (cpu vs gpu): binary cross-entropy
 * in numerically stable logit form, mean over all elements, reduced to a
 * shape-{1} tensor in the operand dtype.
 * @version 0.1
 * @date 2026-11-10
 */
#pragma once

#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {

        /**
         * @brief CPU bce_with_logits backend: stable logit-form BCE mean.
         */
        void bce_with_logits_cpu(const Tensor&, const Tensor&, Tensor&);
        /**
         * @brief GPU bce_with_logits backend: stable logit-form BCE mean.
         */
        void bce_with_logits_gpu(const Tensor&, const Tensor&, const Tensor&);
    }
}

#include "BceWithLogits.hpp"
