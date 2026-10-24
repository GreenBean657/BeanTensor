/**
 * @file GemmG.h
 * @author GreenBean
 * @brief Entry point for the GEMM backward trampoline (gemm_g_node). No
 * dedicated cpu/gpu backend: the node reuses matmul plus transpose_last2.
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
         * @brief Backward trampoline for matmul (gemm); matches Node::_backward_fn's
         * signature. The leading Node* is the owning node; the trailing DType is the
         * calling backward()'s assigned grad dtype.
         */
        void gemm_g_node(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
    }
}
