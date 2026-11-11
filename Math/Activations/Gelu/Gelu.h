/**
 * @file Gelu.h
 * @author GreenBean
 * @brief Entry point for GELU (cpu vs gpu).
 * @version 0.1
 * @date 2026-11-10
 */
#pragma once

namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {

        /**
         * @brief CPU GELU backend: out = 0.5*x*(1+erf(x/sqrt(2))).
         */
        void gelu_cpu(const Tensor&, const Tensor&);
        /**
         * @brief GPU GELU backend: out = 0.5*x*(1+erf(x/sqrt(2))).
         */
        void gelu_gpu(const Tensor&, const Tensor&);
    }
}

#include "Gelu.hpp"
