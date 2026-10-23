/**
 * @file TransposeLast2.h
 * @author GreenBean
 * @brief Entry point for the last-two-dims transpose (cpu vs gpu), used by
 * the GEMM backward.
 * @version 0.1
 * @date 2026-10-20
 */
#pragma once

namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {

        /**
         * @brief CPU transpose of the last two dimensions: out[..., j, i] = t[..., i, j].
         */
        void transpose_last2_cpu(const Tensor&, const Tensor&);
        /**
         * @brief GPU transpose of the last two dimensions: out[..., j, i] = t[..., i, j].
         */
        void transpose_last2_gpu(const Tensor&, const Tensor&);
    }
}

#include "TransposeLast2.hpp"
