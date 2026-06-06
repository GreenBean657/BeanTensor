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

        /**
         * Move a tensor to the GPU.
         * @param t Tensor to move.
         * @param device GPUId to move to.
         */
        void move_to_gpu(Tensor& t, uint32_t device);

        /**
         * Move a tensor to the CPU.
         * @param t Tensor to move.
         */
        void move_to_cpu(Tensor& t);

        /**
         * Move a tensor from GPU A > GPU B.
         * @note Requires two devices.
         * @param t Tensor to move.
         * @param new_device New GPUId to move to.
         */
        void move_gpu_to_gpu(Tensor& t, uint32_t new_device);

        /**
         * Convert a tensor on the GPU to a new DType.
         * @note Requires GPU conversion kernels.
         * @param t Tensor to convert.
         * @param new_dtype New DType to convert to.
         * @param use_inf_conversions Whether to use InF and NaN conversions. If false, the program will fire a failure if an unsafe conversion occurs.
         */
        void convert_gpu(Tensor& t, const BeanTensor::Casting::DType& new_dtype, bool use_inf_conversions = false);

        /**
         * @note Calls on base to-string after soft cloning the tensor.
         * @param t Tensor to pull from
         * @return PMR String as requested.
         */
        std::string gpu_tensor_to_string(const Tensor& t);
    }
}

#include "Tensor/TensorOps.hpp"