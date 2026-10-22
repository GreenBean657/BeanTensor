/**
 * @file Sum.h
 * @author GreenBean
 * @brief Entry point for sum (cpu vs gpu).
 * @version 0.1
 * @date 2026-10-20
 */
#pragma once

#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {

        /**
         * @brief CPU full reduction, sums every element of the input into the scalar output tensor.
         */
        void sum_cpu(const Tensor&, Tensor&);
        /**
         * @brief GPU full reduction, sums every element of the input into the scalar output tensor.
         */
        void sum_gpu(const Tensor&, Tensor&);
    }
}

#include "Sum.hpp"
