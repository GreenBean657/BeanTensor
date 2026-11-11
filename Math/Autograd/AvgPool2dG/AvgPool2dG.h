/**
 * @file AvgPool2dG.h
 * @author GreenBean
 * @brief AvgPool2d backward node: dx[window] += dy / (kernel²).
 * @version 0.1
 * @date 2026-11-10
 */
#pragma once

#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors {
    class Tensor;
    struct Node;
    namespace detail {

        /**
         * @brief Backward trampoline for AvgPool2d; matches Node::_backward_fn's
         * signature. The leading Node* is the owning node; the trailing DType is
         * the calling backward()'s assigned gradient dtype.
         */
        void avg_pool2d_g_node(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);

        /**
         * @brief Signature bridge between wire_node (mutable output_grad) and
         * avg_pool2d_g_node.
         */
        void avg_pool2d_g_bridge(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
    }
}
