/**
 * @file FlattenG.h
 * @author GreenBean
 * @brief Entry point for flatten backward: dx = reshape(dy, x.shape()).
 * @version 0.1
 * @date 2026-10-28
 */
#pragma once

#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors {
    class Tensor;
    struct Node;
    namespace detail {

        /**
         * @brief Backward trampoline for flatten; matches Node::_backward_fn's
         * signature. The leading Node* is the owning node; the trailing DType is the
         * calling backward()'s assigned gradient dtype.
         */
        void flatten_g_node(Node*, const Tensor*, Tensor**, size_t, DataTypes::DType);

        /**
         * @brief Signature bridge between wire_node and flatten_g_node.
         */
        void flatten_g_bridge(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
    }
}
