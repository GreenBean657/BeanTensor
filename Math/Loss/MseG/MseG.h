/**
 * @file MseG.h
 * @author GreenBean
 * @brief Entry point for mse backward (cpu vs gpu):
 * dpred = 2 * (pred - target) / numel * dloss.
 * @version 0.1
 * @date 2026-10-23
 */
#pragma once

#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {

        /**
         * @brief CPU mse backward, elementwise over the operand range.
         */
        void mse_g_cpu(const Tensor&, const Tensor&, const Tensor&, Tensor&);
        /**
         * @brief GPU mse backward, elementwise over the operand range.
         */
        void mse_g_gpu(const Tensor&, const Tensor&, const Tensor&, const Tensor&);

        /**
         * @brief Backward trampoline for mse; matches Node::_backward_fn's
         * signature. The trailing DType is the calling backward()'s assigned grad dtype.
         */
        void mse_g_node(const Tensor*, Tensor**, size_t, DataTypes::DType);
        /**
         * @brief Signature bridge between wire_node (mutable output_grad) and
         * mse_g_node (const output_grad).
         */
        void mse_g_bridge(Tensor*, Tensor**, size_t, DataTypes::DType);
    }
}

#include "MseG.hpp"
