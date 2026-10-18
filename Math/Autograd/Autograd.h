/**
 * @file Autograd.h
 * @author GreenBean
 * @brief Entry point for the autograd backward ops (cpu vs gpu): gemm_g,
 * bias_add_g, relu_g, softmax_ce_g, plus the shared gradient plumbing
 * (grad accumulation, last-two-dims transpose for GEMM backward).
 * The four *_g_node functions match Node::_backward_fn's signature and are
 * what the forward ops store when they wire the computation graph.
 * @version 0.1
 * @date 2026-10-18
 */
#pragma once

#include <cstddef>

#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {

        void gemm_g_node(Tensor*, Tensor**, size_t);
        void bias_add_g_node(Tensor*, Tensor**, size_t);
        void relu_g_node(Tensor*, Tensor**, size_t);
        void softmax_ce_g_node(Tensor*, Tensor**, size_t);

        void grad_acc_cpu(Tensor&, const Tensor&);
        void grad_acc_gpu(Tensor&, const Tensor&);

        void transpose_last2_cpu(const Tensor&, Tensor&);
        void transpose_last2_gpu(const Tensor&, Tensor&);

        void relu_g_cpu(const Tensor&, const Tensor&, Tensor&);
        void relu_g_gpu(const Tensor&, const Tensor&, Tensor&);

        void bias_add_g_cpu(const Tensor&, Tensor&);
        void bias_add_g_gpu(const Tensor&, Tensor&);

        void softmax_ce_g_cpu(const Tensor&, const Tensor&, const Tensor&, Tensor&);
        void softmax_ce_g_gpu(const Tensor&, const Tensor&, const Tensor&, Tensor&);
    }
}

#include "Autograd.hpp"
