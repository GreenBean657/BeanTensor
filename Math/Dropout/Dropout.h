/**
 * @file Dropout.h
 * @author GreenBean
 * @brief Public entry point for dropout mask construction.
 * @version 0.2
 * @date 2026-10-29
 */
#pragma once

#include <cstdint>

#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors {
    class Tensor;
}

namespace BeanTensor::Math::Dropout {

    /**
     * Build an inverted dropout mask for the given input shape/dtype/device.
     * The mask is filled from a uniform [0, 1) random stream seeded by @p seed,
     * then each element becomes `1 / (1 - p)` where the random value is >= p
     * and `0` otherwise. Float kinds only (Float16/BFloat16/Float32/Float64);
     * integer dtypes throw ErrorHandling::NotImplemented.
     * @brief Deterministic inverted-dropout threshold mask.
     * @param x Input tensor whose shape, dtype and device define the mask.
     * @param p Drop probability; must satisfy `0 <= p < 1`.
     * @param seed Explicit seed for the random stream.
     * @return Mask tensor with the same shape, dtype and device as @p x.
     * @throws ErrorHandling::NotImplemented Non-float dtype.
     */
    [[nodiscard]] Tensors::Tensor train_mask(const Tensors::Tensor& x, float64_t p, int64_t seed);

    /**
     * @brief CPU backend for train_mask. Internal: call train_mask, which routes on device.
     * @param x CPU-resident input tensor.
     * @param p Drop probability; must satisfy `0 <= p < 1`.
     * @param seed Explicit seed for the random stream.
     * @return Mask tensor with the same shape and dtype as @p x, on CPU.
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource Non-CPU input.
     */
    [[nodiscard]] Tensors::Tensor train_mask_cpu(const Tensors::Tensor& x, float64_t p, int64_t seed);

    /**
     * @brief GPU backend for train_mask. Internal: call train_mask, which routes on device.
     * Defined by the CUDA/HIP translation units only; the dispatcher guards the call
     * site and throws ErrorHandling::GPUTaskOnCPUBuild in CPU-only builds.
     * @param x GPU-resident input tensor.
     * @param p Drop probability; must satisfy `0 <= p < 1`.
     * @param seed Explicit seed for the random stream.
     * @return Mask tensor with the same shape and dtype as @p x, on GPU.
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource Non-GPU input.
     */
    [[nodiscard]] Tensors::Tensor train_mask_gpu(const Tensors::Tensor& x, float64_t p, int64_t seed);

}
