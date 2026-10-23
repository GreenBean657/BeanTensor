/**
 * @file Adam.h
 * @author GreenBean
 * @brief Entry point for adam_update (cpu vs gpu): one Adam moment step,
 * m = b1*m + (1-b1)*g, v = b2*v + (1-b2)*g^2, w -= lr * (m/bc1) /
 * (sqrt(v/bc2) + eps), over the weight/gradient/moment buffers. Moment state
 * (m, v, t) is owned by the optimizer-side map in AdamCPU.cpp, not by Tensor.
 * @version 0.1
 * @date 2026-10-23
 */
#pragma once

#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {

        /**
         * @brief CPU in-place Adam moment step (weights, gradients, first/second moments, hyperparameters).
         */
        void adam_update_cpu(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                             float64_t, float64_t, float64_t, float64_t, float64_t, float64_t);
        /**
         * @brief GPU in-place Adam moment step (weights, gradients, first/second moments, hyperparameters).
         */
        void adam_update_gpu(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                             float64_t, float64_t, float64_t, float64_t, float64_t, float64_t);
    }
}

#include "Adam.hpp"
