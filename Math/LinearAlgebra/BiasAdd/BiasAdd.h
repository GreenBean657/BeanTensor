/**
 * @file BiasAdd.h
 * @author GreenBean
 * @brief Entry point for bias_add (cpu vs gpu)
 * @version 0.1
 * @date 2026-10-20
 */
#pragma once

namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {

        /**
         * @brief CPU bias_add backend: out = input + bias broadcast over the last dimension.
         */
        void bias_add_cpu(const Tensor&, const Tensor&, const Tensor&);
        /**
         * @brief GPU bias_add backend: out = input + bias broadcast over the last dimension.
         */
        void bias_add_gpu(const Tensor&, const Tensor&, const Tensor&);
    }
}

#include "BiasAdd.hpp"
