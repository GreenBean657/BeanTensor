/**
 * @file BceG.h
 * @author GreenBean
 * @brief Entry point for bce backward (cpu vs gpu):
 * dpred = ((1-y)/(1-p̂) - y/p̂) * dloss / N, with p̂ the same clamped
 * probability used in the forward pass.
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
         * @brief CPU bce backward, elementwise over the operand range.
         */
        void bce_g_cpu(const Tensor&, const Tensor&, const Tensor&, Tensor&);
        /**
         * @brief GPU bce backward, elementwise over the operand range.
         */
        void bce_g_gpu(const Tensor&, const Tensor&, const Tensor&, const Tensor&);

        /**
         * @brief Backward trampoline for bce; matches Node::_backward_fn's signature.
         */
        void bce_g_node(Node*, const Tensor*, Tensor**, size_t, DataTypes::DType);
        /**
         * @brief Signature bridge between wire_node and bce_g_node.
         */
        void bce_g_bridge(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
    }
}

#include "BceG.hpp"
