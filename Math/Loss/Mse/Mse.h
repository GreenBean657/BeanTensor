/**
 * @file Mse.h
 * @author GreenBean
 * @brief Entry point for mse (cpu vs gpu): mean of (pred - target)^2 over
 * all elements, reduced to a shape-{1} tensor in the operand dtype.
 * @version 0.1
 * @date 2026-10-23
 */
#pragma once

#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {

        /**
         * @brief CPU mse backend: full-tensor mean of squared differences.
         */
        void mse_cpu(const Tensor&, const Tensor&, Tensor&);
        /**
         * @brief GPU mse backend: full-tensor mean of squared differences.
         */
        void mse_gpu(const Tensor&, const Tensor&, const Tensor&);
    }
}

#include "Mse.hpp"
