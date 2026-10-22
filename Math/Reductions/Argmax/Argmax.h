/**
 * @file Argmax.h
 * @author GreenBean
 * @brief Entry point for argmax (cpu vs gpu).
 * @version 0.2
 * @date 2026-10-22
 */
#pragma once

#include <cstdint>

#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {

        /**
         * @brief CPU flat index of the largest element of the input, written to the output integer.
         */
        void argmax_cpu(const Tensor&, uint64_t&);
        /**
         * @brief GPU flat index of the largest element of the input, written to the output integer.
         */
        void argmax_gpu(const Tensor&, uint64_t&);
        /**
         * @brief CPU argmax along one axis, written as int64 indices into the output tensor.
         */
        void argmax_axis_cpu(const Tensor&, Tensor&, size_t);
        /**
         * @brief GPU argmax along one axis, written as int64 indices into the output tensor.
         */
        void argmax_axis_gpu(const Tensor&, Tensor&, size_t);
    }
}

#include "Argmax.hpp"
