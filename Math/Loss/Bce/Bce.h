/**
 * @file Bce.h
 * @author GreenBean
 * @brief Entry point for bce (cpu vs gpu): binary cross-entropy in probability
 * form, mean over all elements, reduced to a shape-{1} tensor in the operand
 * dtype.
 * @version 0.1
 * @date 2026-11-10
 */
#pragma once

#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {

        /**
         * @brief CPU bce backend: full-tensor mean of the BCE probability form.
         */
        void bce_cpu(const Tensor&, const Tensor&, Tensor&);
        /**
         * @brief GPU bce backend: full-tensor mean of the BCE probability form.
         */
        void bce_gpu(const Tensor&, const Tensor&, const Tensor&);
    }
}

#include "Bce.hpp"
