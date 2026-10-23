/**
 * @file GradAcc.h
 * @author GreenBean
 * @brief Entry point for grad accumulation (cpu vs gpu): g += d in place.
 * @version 0.1
 * @date 2026-10-20
 */
#pragma once


namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {

        /**
         * @brief CPU gradient accumulation, g += d.
         */
        void grad_acc_cpu(const Tensor&, const Tensor&);
        /**
         * @brief GPU gradient accumulation, g += d.
         */
        void grad_acc_gpu(const Tensor&, const Tensor&);
    }
}

#include "GradAcc.hpp"
