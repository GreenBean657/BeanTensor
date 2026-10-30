/**
 * @file BatchNorm.h
 * @author GreenBean
 * @brief Entry point for batch normalization (cpu vs gpu).
 * @version 0.1
 * @date 2026-10-30
 */
#pragma once

#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {

        /**
         * CPU batch normalization forward. Train mode: per-channel biased
         * batch statistics, pinned normalization arithmetic, and the pinned
         * three-step running-stat blend, chained onto the running buffers'
         * own task queues. Eval mode: normalizes with the running buffers and
         * leaves them untouched.
         * @brief CPU batch normalization forward.
         */
        void batch_norm_cpu(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                            Tensor&, Tensor&, bool, float64_t, float64_t);
        /**
         * @brief GPU batch normalization forward (stub in .hpp for CPU-only builds).
         */
        void batch_norm_gpu(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                            Tensor&, Tensor&, bool, float64_t, float64_t);
    }
}

#include "BatchNorm.hpp"
