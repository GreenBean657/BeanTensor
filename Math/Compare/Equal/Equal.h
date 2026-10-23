/**
 * @file Equal.h
 * @author GreenBean
 * @brief Entry point for elementwise equal (cpu vs gpu).
 * @version 0.1.1
 * @date 2026-10-22
 */
#pragma once

namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {

        /**
         * @brief CPU elementwise equality, writing a UInt8 (0/1) mask into the output tensor.
         */
        void equal_cpu(const Tensor&, const Tensor&, Tensor&);
        /**
         * @brief GPU elementwise equality, writing a UInt8 (0/1) mask into the output tensor.
         */
        void equal_gpu(const Tensor&, const Tensor&, const Tensor&);
    }
}

#include "Equal.hpp"
