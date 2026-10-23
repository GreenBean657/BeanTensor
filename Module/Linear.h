/**
 * @file Linear.h
 * @author GreenBean
 * @brief Fully connected layer: weight {in_features, out_features} plus
 * optional bias {out_features}, both registered as named parameters
 * ("weight", "bias") on the Module base, and forward = matmul + bias_add
 * following the project's x(m×k)·w(k×n) GEMM convention.
 * @version 0.1
 * @date 2026-10-23
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>

#include "Module/Module.h"
#include "Tensor/DTypes.h"

namespace BeanTensor::Modules {

    /**
     * Affine map out = x·weight + bias. Weight shape is {in_features,
     * out_features} so a rank-2 input {m, in_features} matmuls directly;
     * bias is added along the last dimension via Tensor::bias_add. Both
     * parameters are created requires_grad=true on the construction device,
     * in the construction dtype, and initialized uniformly in
     * [-1/sqrt(in_features), 1/sqrt(in_features)] — the project's Linear
     * convention, built on Tensor::random.
     * @brief Fully connected layer (matmul + bias_add).
     */
    class Linear : public Module {
    public:
        /**
         * Construct and register the parameters, then initialize them.
         * @brief Construct a Linear layer.
         * @param in_features Input feature count (x's last dimension).
         * @param out_features Output feature count.
         * @param dtype Parameter/input dtype (Float32 by default).
         * @param device Device to hold the parameters on (CPU by default).
         * @param bias Whether to register and add a bias parameter.
         * @param seed Init seed for the uniform parameter fill (random by default).
         */
        Linear(size_t in_features, size_t out_features,
               DataTypes::DType dtype = DataTypes::DType::Float32,
               Tensors::Device device = Tensors::Device::CPU,
               bool bias = true,
               int64_t seed = std::random_device()());

        /**
         * x(m×in)·weight(in×out), then + bias(out) when the layer has one.
         * Joins the computation graph when x or the parameters require grad.
         * @brief forward: matmul then bias_add.
         * @param x Input tensor, rank 2 (or rank-3 batched) with last dimension == in_features.
         * @return x·weight (+ bias).
         */
        [[nodiscard]] Tensors::Tensor forward(const Tensors::Tensor& x) override;

        /**
         * @brief Human-readable layer type.
         * @return "Linear".
         */
        [[nodiscard]] std::string name() const override;

        /**
         * @brief Input feature count.
         * @return in_features.
         */
        [[nodiscard]] size_t in_features() const;
        /**
         * @brief Output feature count.
         * @return out_features.
         */
        [[nodiscard]] size_t out_features() const;
        /**
         * @brief Whether the layer carries a bias parameter.
         * @return has_bias.
         */
        [[nodiscard]] bool has_bias() const;

    private:
        size_t in_features_;
        size_t out_features_;
        bool has_bias_;
    };
}
