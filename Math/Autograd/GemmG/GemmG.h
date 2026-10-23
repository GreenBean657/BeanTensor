/**
 * @file GemmG.h
 * @author GreenBean
 * @brief Entry point for the GEMM backward trampoline (gemm_g_node). No
 * dedicated cpu/gpu backend: the node reuses matmul plus transpose_last2.
 * @version 0.1
 * @date 2026-10-20
 */
#pragma once

#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {

        /**
         * @brief Backward trampoline for matmul (gemm); matches Node::_backward_fn's
         * signature. The trailing DType is the calling backward()'s assigned grad dtype.
         */
        void gemm_g_node(Tensor*, Tensor**, size_t, DataTypes::DType);
    }
}
