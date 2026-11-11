/**
 * @file Gelu.h
 * @author GreenBean
 * @brief Parameter-free GELU layer: forward = Tensor::gelu on the input.
 * @version 0.1
 * @date 2026-11-10
 */
#pragma once

#include <string>

#include "Module/Module.h"

namespace BeanTensor::Modules {

    /**
     * Elementwise GELU with no parameters.  Joins the computation graph
     * when its input requires grad.
     * @brief Parameter-free GELU layer.
     */
    class Gelu : public Module {
    public:
        /**
         * @brief forward: Tensor::gelu of the input.
         * @return gelu(x), same shape/dtype/device as the input.
         */
        [[nodiscard]] Tensors::Tensor forward(const Tensors::Tensor&) override;

        /**
         * @brief Human-readable layer type.
         * @return "Gelu".
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
