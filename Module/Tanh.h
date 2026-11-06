/**
 * @file Tanh.h
 * @author GreenBean
 * @brief Parameter-free tanh layer: forward = Tensor::tanh on the input.
 * @version 0.2
 * @date 2026-11-05
 */
#pragma once

#include <string>

#include "Module/Module.h"

namespace BeanTensor::Modules {

    /**
     * Elementwise tanh with no parameters. Joins the computation graph
     * when its input requires grad.
     * @brief Parameter-free tanh layer.
     */
    class Tanh : public Module {
    public:
        /**
         * @brief forward: Tensor::tanh of the input.
         * @return tanh(x), same shape/dtype/device as the input.
         */
        [[nodiscard]] Tensors::Tensor forward(const Tensors::Tensor&) override;

        /**
         * @brief Human-readable layer type.
         * @return "Tanh".
         */
        [[nodiscard]] std::string name() const override;

        /**
         * Exports this layer's constructor facts as a .bt architecture spec.
         * @brief Architecture spec of this layer.
         * @return Spec with this layer's type and hyperparameters.
         */
        [[nodiscard]] ModuleSpec spec() const override;

        /**
         * @brief Construct this layer from a parsed spec.
         * @throws ErrorHandling::CorruptFile a hyperparameter is missing, malformed, or out of range.
         */
        [[nodiscard]] static std::unique_ptr<Module> construct(const ModuleSpec&, DataTypes::DType, Tensors::Device);
    };
}
