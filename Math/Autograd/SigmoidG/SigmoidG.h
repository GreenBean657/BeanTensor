/**
 * @file SigmoidG.h
 * @author GreenBean
 * @brief Backward trampoline for sigmoid: dx = dy * s * (1-s), where
 * s = sigmoid(x) is recomputed from the saved forward input.
 * @version 0.2
 * @date 2026-10-28
 */
#pragma once

#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors {
    class Tensor;
    struct Node;
    namespace detail {

        /**
         * @brief Backward trampoline for sigmoid; matches Node::_backward_fn's
         * signature.
         */
        void sigmoid_g_node(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
    }
}
