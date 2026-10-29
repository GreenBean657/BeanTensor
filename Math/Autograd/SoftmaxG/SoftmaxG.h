/**
 * @file SoftmaxG.h
 * @author GreenBean
 * @brief Backward trampoline for standalone softmax: dx = y * (dy - rowsum(dy*y)),
 * where y is recomputed from the saved forward input.
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
         * @brief CPU standalone softmax backward, row by row.
         */
        void softmax_g_cpu(const Tensor&, const Tensor&, Tensor&);
        /**
         * @brief GPU standalone softmax backward, row by row.
         */
        void softmax_g_gpu(const Tensor&, const Tensor&, const Tensor&);

        /**
         * @brief Backward trampoline for standalone softmax; matches Node::_backward_fn's
         * signature. The leading Node* is the owning node; the trailing DType is the
         * calling backward()'s assigned grad dtype.
         */
        void softmax_g_node(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
        /**
         * @brief Signature bridge between wire_node (mutable output_grad) and
         * softmax_g_node.
         */
        void softmax_g_bridge(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
    }
}

#include "SoftmaxG.hpp"
