/**
 * @file Max.h
 * @author GreenBean
 * @brief Entry point for max (cpu vs gpu).
 * @version 0.1
 * @date 2026-10-25
 */
#pragma once

namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {

        /**
         * @brief CPU full reduction, largest element of the input written into the scalar output tensor.
         */
        void max_cpu(const Tensor&, const Tensor&);
        /**
         * @brief GPU full reduction, largest element of the input written into the scalar output tensor.
         */
        void max_gpu(const Tensor&, const Tensor&);
    }
}

#include "Max.hpp"
