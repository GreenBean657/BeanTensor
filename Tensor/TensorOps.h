#pragma once
#include <cstdint>
/*
 * This file handles HIP/CUDA Passthrough for basic tensor operations.
 * Tensor operations handled here are not for addition and generalized ops
 * But instead are specializations for Tensor Datatype Conversions.
 *
 * Linkage:
 * TensorOps.hip
 * TensorOps.cu
 */


namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {
        void set_amount_gpu(Tensor& t, int32_t amount);
        /**
         * Fill a tensor with random values on the GPU.
         * @param t Tensor to fill with random values.
         * @param min Min value to set.
         * @param max Highest Value to set.
         * @param seed Randomized Seed to use.
         */
        void set_random_gpu(Tensor& t, double min, double max, std::int64_t seed);
        void move_to_gpu(Tensor& t, uint32_t device);
        void move_to_cpu(Tensor& t);
        void move_gpu_to_gpu(Tensor& t, uint32_t new_device);
    }
}

#include "Tensor/TensorOps.hpp"