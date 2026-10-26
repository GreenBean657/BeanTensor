/**
 * @file MeanG.h
 * @author GreenBean
 * @brief mean backward node: dx = (the scalar seed / numel) broadcast to x's shape.
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
         * @brief Backward trampoline for mean; matches Node::_backward_fn's
         * signature. The leading Node* is the owning node; the trailing DType is
         * the calling backward()'s assigned grad dtype. Device-neutral: the
         * scale and broadcast are composed entirely from tensor ops that
         * dispatch by device.
         */
        void mean_g_node(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
    }
}
