/**
 * @file Mean.h
 * @author GreenBean
 * @brief Entry point for mean (cpu vs gpu).
 * @version 0.2
 * @date 2026-10-22
 */
#pragma once

namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {

        /**
         * @brief CPU full reduction, averages every element of the input into the scalar output tensor.
         */
        void mean_cpu(const Tensor&, const Tensor&);
        /**
         * @brief GPU full reduction, averages every element of the input into the scalar output tensor.
         */
        void mean_gpu(const Tensor&, const Tensor&);
    }
}

#include "Mean.hpp"
