/**
 * @file TanhG.h
 * @author GreenBean
 * @brief Backward trampoline for tanh: dx = dy * (1 - t*t), where
 * t = tanh(x) is recomputed from the saved forward input.
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
         * @brief Backward trampoline for tanh; matches Node::_backward_fn's
         * signature.
         */
        void tanh_g_node(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
    }
}
