/**
 * @file Relu.h
 * @author GreenBean
 * @brief Entry point for relu (cpu vs gpu)
 * @version 0.1
 * @date 2026-10-20
 */
#pragma once

#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {

        /**
         * @brief CPU relu backend: out = max(input, 0), NaN propagating.
         */
        void relu_cpu(const Tensor&, const Tensor&);
        /**
         * @brief GPU relu backend: out = max(input, 0), NaN propagating.
         */
        void relu_gpu(const Tensor&, const Tensor&);
    }
}

#include "Relu.hpp"
