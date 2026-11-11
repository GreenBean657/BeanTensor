/**
 * @file BceWithLogitsG.h
 * @author GreenBean
 * @brief Entry point for bce_with_logits backward (cpu vs gpu):
 * dlogits = (σ(x) - y) * dloss / N, using the stable sigmoid branch.
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
         * @brief CPU bce_with_logits backward, elementwise over the operand range.
         */
        void bce_with_logits_g_cpu(const Tensor&, const Tensor&, const Tensor&, Tensor&);
        /**
         * @brief GPU bce_with_logits backward, elementwise over the operand range.
         */
        void bce_with_logits_g_gpu(const Tensor&, const Tensor&, const Tensor&, const Tensor&);

        /**
         * @brief Backward trampoline for bce_with_logits; matches Node::_backward_fn's signature.
         */
        void bce_with_logits_g_node(Node*, const Tensor*, Tensor**, size_t, DataTypes::DType);
        /**
         * @brief Signature bridge between wire_node and bce_with_logits_g_node.
         */
        void bce_with_logits_g_bridge(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
    }
}

#include "BceWithLogitsG.hpp"
