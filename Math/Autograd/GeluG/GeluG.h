/**
 * @file GeluG.h
 * @author GreenBean
 * @brief Entry point for GELU backward (cpu vs gpu): dx = dy * gelu'(x).
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
         * @brief CPU GELU backward: dx = dy * gelu'(x) (written, not accumulated).
         */
        void gelu_g_cpu(const Tensor&, const Tensor&, const Tensor&);
        /**
         * @brief GPU GELU backward: dx = dy * gelu'(x) (written, not accumulated).
         */
        void gelu_g_gpu(const Tensor&, const Tensor&, const Tensor&);

        /**
         * @brief Backward trampoline for GELU; matches Node::_backward_fn's
         * signature.
         */
        void gelu_g_node(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
    }
}

#include "GeluG.hpp"
