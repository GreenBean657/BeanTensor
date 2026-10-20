/**
 * @file Argmax.h
 * @author GreenBean
 * @brief Entry point for argmax (cpu vs gpu).
 * @version 0.1
 * @date 2026-10-20
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
    }
}

#include "Argmax.hpp"
