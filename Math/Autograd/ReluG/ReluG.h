/**
 * @file ReluG.h
 * @author GreenBean
 * @brief Entry point for relu backward (cpu vs gpu): dx = x > 0 ? dy : 0.
 * @version 0.2
 * @date 2026-10-23
 */
#pragma once


#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors {
    class Tensor;
    struct Node;
    namespace detail {

        /**
         * @brief CPU relu backward: dx = x > 0 ? dy : 0 (written, not accumulated).
         */
        void relu_g_cpu(const Tensor&, const Tensor&, const Tensor&);
        /**
         * @brief GPU relu backward: dx = x > 0 ? dy : 0 (written, not accumulated).
         */
        void relu_g_gpu(const Tensor&, const Tensor&, const Tensor&);

        /**
         * @brief Backward trampoline for relu; matches Node::_backward_fn's
         * signature. The leading Node* is the owning node; the trailing DType is the
         * calling backward()'s assigned grad dtype.
         */
        void relu_g_node(Node*, const Tensor*, Tensor**, size_t, DataTypes::DType);
    }
}

#include "ReluG.hpp"
