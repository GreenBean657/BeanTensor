/**
 * @file BiasAddG.h
 * @author GreenBean
 * @brief Entry point for bias_add backward (cpu vs gpu): dbias = column sum of dy.
 * @version 0.2
 * @date 2026-10-23
 */
#pragma once

#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors {
    class Tensor;
    struct Node;
    namespace detail {

        /**
         * @brief CPU bias_add backward: dbias = per-column sum of dy over all leading rows.
         */
        void bias_add_g_cpu(const Tensor&, Tensor&);
        /**
         * @brief GPU bias_add backward: dbias = per-column sum of dy over all leading rows.
         */
        void bias_add_g_gpu(const Tensor&, const Tensor&);

        /**
         * @brief Backward trampoline for bias_add; matches Node::_backward_fn's
         * signature. The leading Node* is the owning node; the trailing DType is the
         * calling backward()'s assigned grad dtype.
         */
        void bias_add_g_node(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
    }
}

#include "BiasAddG.hpp"
