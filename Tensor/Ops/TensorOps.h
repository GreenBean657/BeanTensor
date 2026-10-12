#pragma once
#include <cstdint>

#include "Intrinsics/ConversionMethods.h"
#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {
        /**
         * Fill a GPU tensor with a fixed value. Packed dtypes fill both lanes.
         * @brief GPU fixed-value fill.
         * @param t Tensor to fill.
         * @param amount Value to fill with.
         */
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
         * Move a tensor to the GPU. Single-GPU: only device 0 is supported.
         * @param t Tensor to move.
         * @param device GPUId to move to (must be 0 — multi-GPU was dropped in V5).
         */
        void move_to_gpu(Tensor& t, uint32_t device);

        /**
         * Move a tensor to the CPU.
         * @param t Tensor to move.
         */
        void move_to_cpu(Tensor& t);

        /**
         * Convert a tensor on the GPU to a new DType.
         * @note Requires GPU conversion kernels. The kernels mirror the CPU
         * dispatcher's per-method clamp/flag semantics; non-PERMISSIVE failures
         * surface as std::range_error at the tensor's sync().
         * @param t Tensor to convert.
         * @param new_dtype New DType to convert to.
         * @param method Active ConversionClampMethod - How should conversion tolerance be handled?
         */
        void convert_gpu(Tensor& t, const DataTypes::DType& new_dtype, ConversionClampMethod method = ConversionClampMethod::PERMISSIVE);
    }
}

#include "TensorOps.hpp"