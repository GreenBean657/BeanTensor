/**
 * @file BiasAddG.h
 * @author GreenBean
 * @brief Entry point for bias_add backward (cpu vs gpu): dbias = column sum of dy.
 * @version 0.1
 * @date 2026-10-20
 */
#pragma once

#include <cstddef>

#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {

        /**
         * @brief CPU bias_add backward: dbias = per-column sum of dy over all leading rows.
         */
        void bias_add_g_cpu(const Tensor&, Tensor&);
        /**
         * @brief GPU bias_add backward: dbias = per-column sum of dy over all leading rows.
         */
        void bias_add_g_gpu(const Tensor&, Tensor&);

        /**
         * @brief Backward trampoline for bias_add; matches Node::_backward_fn's signature.
         */
        void bias_add_g_node(Tensor*, Tensor**, size_t);
    }
}

#include "BiasAddG.hpp"
