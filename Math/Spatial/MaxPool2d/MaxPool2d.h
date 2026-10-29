/**
 * @file MaxPool2d.h
 * @author GreenBean
 * @brief Entry point for MaxPool2d (cpu vs gpu).
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
         * @brief CPU MaxPool2d backend: NCHW rank-4 -> NCHW rank-4, first-max tie rule.
         */
        void max_pool2d_cpu(const Tensor&, Tensor&, uint32_t, uint32_t);
        /**
         * @brief GPU MaxPool2d backend: NCHW rank-4 -> NCHW rank-4, first-max tie rule.
         */
        void max_pool2d_gpu(const Tensor&, Tensor&, uint32_t, uint32_t);

        /**
         * @brief CPU MaxPool2d backward backend: recomputes window argmax from x and routes dy.
         */
        void max_pool2d_g_cpu(const Tensor&, const Tensor&, Tensor&, uint32_t, uint32_t);
        /**
         * @brief GPU MaxPool2d backward backend: recomputes window argmax from x and routes dy.
         */
        void max_pool2d_g_gpu(const Tensor&, const Tensor&, Tensor&, uint32_t, uint32_t);

        /**
         * @brief Backward trampoline for MaxPool2d; matches Node::_backward_fn's signature.
         */
        void max_pool2d_g_node(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
        /**
         * @brief Signature bridge between wire_node (mutable output_grad) and max_pool2d_g_node.
         */
        void max_pool2d_g_bridge(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
    }
}

#include "MaxPool2d.hpp"
