/**
 * @file SumG.h
 * @author GreenBean
 * @brief sum backward node: dx = the scalar seed broadcast to x's shape.
 * @version 0.1
 * @date 2026-10-25
 */
#pragma once

#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors {
    class Tensor;
    struct Node;
    namespace detail {

        /**
         * @brief Backward trampoline for sum; matches Node::_backward_fn's
         * signature. The leading Node* is the owning node; the trailing DType is
         * the calling backward()'s assigned grad dtype. Device-neutral: the
         * broadcast is composed entirely from tensor ops that dispatch by device.
         */
        void sum_g_node(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
    }
}
