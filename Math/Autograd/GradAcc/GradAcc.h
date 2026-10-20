/**
 * @file GradAcc.h
 * @author GreenBean
 * @brief Entry point for grad accumulation (cpu vs gpu): g += d in place.
 * @version 0.1
 * @date 2026-10-20
 */
#pragma once

#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {

        /**
         * @brief CPU gradient accumulation, g += d.
         */
        void grad_acc_cpu(Tensor&, const Tensor&);
        /**
         * @brief GPU gradient accumulation, g += d.
         */
        void grad_acc_gpu(Tensor&, const Tensor&);
    }
}

#include "GradAcc.hpp"
