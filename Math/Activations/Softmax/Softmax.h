/**
 * @file Softmax.h
 * @author GreenBean
 * @brief Entry point for standalone rank-2 softmax (cpu vs gpu).
 * @version 0.1
 * @date 2026-10-28
 */
#pragma once

namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {

        /**
         * @brief CPU standalone softmax backend: last-axis softmax over rank-2 input.
         */
        void softmax_cpu(const Tensor&, Tensor&);
        /**
         * @brief GPU standalone softmax backend: last-axis softmax over rank-2 input.
         */
        void softmax_gpu(const Tensor&, const Tensor&);
    }
}

#include "Softmax.hpp"
