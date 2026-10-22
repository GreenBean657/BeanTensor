/**
 * @file SoftmaxCEG.h
 * @author GreenBean
 * @brief Entry point for softmax cross-entropy backward (cpu vs gpu):
 * dlogits = (softmax(logits) - onehot(target)) * dloss / rows.
 * @version 0.1
 * @date 2026-10-20
 */
#pragma once

#include <cstddef>

#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {

        /**
         * @brief CPU softmax cross-entropy backward, row by row.
         */
        void softmax_ce_g_cpu(const Tensor&, const Tensor&, const Tensor&, Tensor&);
        /**
         * @brief GPU softmax cross-entropy backward, row by row.
         */
        void softmax_ce_g_gpu(const Tensor&, const Tensor&, const Tensor&, Tensor&);

        /**
         * @brief Backward trampoline for softmax_ce; matches Node::_backward_fn's
         * signature. The trailing DType is the calling backward()'s assigned grad dtype.
         */
        void softmax_ce_g_node(Tensor*, Tensor**, size_t, DataTypes::DType);
    }
}

#include "SoftmaxCEG.hpp"
