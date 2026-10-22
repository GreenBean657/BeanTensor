/**
 * @file Mean.h
 * @author GreenBean
 * @brief Entry point for mean (cpu vs gpu).
 * @version 0.1
 * @date 2026-10-22
 */
#pragma once

#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {

        /**
         * @brief CPU full reduction, averages every element of the input into the scalar output tensor.
         */
        void mean_cpu(const Tensor&, Tensor&);
        /**
         * @brief GPU full reduction, averages every element of the input into the scalar output tensor.
         */
        void mean_gpu(const Tensor&, Tensor&);
    }
}

#include "Mean.hpp"
