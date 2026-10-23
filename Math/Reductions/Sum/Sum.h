/**
 * @file Sum.h
 * @author GreenBean
 * @brief Entry point for sum (cpu vs gpu).
 * @version 0.2
 * @date 2026-10-20
 */
#pragma once

namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {

        /**
         * @brief CPU full reduction, sums every element of the input into the scalar output tensor.
         */
        void sum_cpu(const Tensor&, const Tensor&);
        /**
         * @brief GPU full reduction, sums every element of the input into the scalar output tensor.
         */
        void sum_gpu(const Tensor&, const Tensor&);
    }
}

#include "Sum.hpp"
