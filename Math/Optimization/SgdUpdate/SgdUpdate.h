/**
 * @file SgdUpdate.h
 * @author GreenBean
 * @brief Entry point for sgd_update (cpu vs gpu).
 * @version 0.1
 * @date 2026-10-20
 */
#pragma once

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
    }
}

#include "SgdUpdate.hpp"
