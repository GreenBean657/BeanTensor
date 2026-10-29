/**
 * @file Conv2dG.h
 * @author GreenBean
 * @brief Backward trampoline for Conv2d.
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
         * @brief Backward trampoline for Conv2d (2- or 3-input node).
         */
        void conv2d_g_node(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
        /**
         * @brief CUDA implementation of the Conv2d backward trampoline.
         */
        void conv2d_g_node_gpu(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
    }
}
