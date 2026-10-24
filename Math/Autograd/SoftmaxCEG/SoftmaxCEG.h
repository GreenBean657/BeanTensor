/**
 * @file SoftmaxCEG.h
 * @author GreenBean
 * @brief Entry point for softmax cross-entropy backward (cpu vs gpu):
 * dlogits = (softmax(logits) - onehot(target)) * dloss / rows.
 * @version 0.1.2
 * @date 2026-10-23
 */
#pragma once

#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors {
    class Tensor;
    struct Node;
    namespace detail {

        /**
         * @brief CPU softmax cross-entropy backward, row by row.
         */
        void softmax_ce_g_cpu(const Tensor&, const Tensor&, const Tensor&, Tensor&);
        /**
         * @brief GPU softmax cross-entropy backward, row by row.
         */
        void softmax_ce_g_gpu(const Tensor&, const Tensor&, const Tensor&, const Tensor&);

        /**
         * @brief Backward trampoline for softmax_ce; matches Node::_backward_fn's
         * signature. The leading Node* is the owning node; the trailing DType is the
         * calling backward()'s assigned grad dtype.
         */
        void softmax_ce_g_node(Node*, const Tensor*, Tensor**, size_t, DataTypes::DType);
        /**
         * @brief Signature bridge between wire_node (mutable output_grad) and
         * softmax_ce_g_node (const output_grad).
         */
        void softmax_ce_g_bridge(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
    }
}

#include "SoftmaxCEG.hpp"
