/**
 * @file Min.h
 * @author GreenBean
 * @brief Entry point for min (cpu vs gpu).
 * @version 0.1
 * @date 2026-10-25
 */
#pragma once

namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {

        /**
         * @brief CPU full reduction, smallest element of the input written into the scalar output tensor.
         */
        void min_cpu(const Tensor&, const Tensor&);
        /**
         * @brief GPU full reduction, smallest element of the input written into the scalar output tensor.
         */
        void min_gpu(const Tensor&, const Tensor&);
    }
}

#include "Min.hpp"
