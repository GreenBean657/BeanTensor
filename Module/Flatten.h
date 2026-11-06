/**
 * @file Flatten.h
 * @author GreenBean
 * @brief Parameter-free Flatten layer: forward = Tensor::flatten on the input.
 * Joins the computation graph when its input requires grad (FlattenG backward:
 * dx = reshape(dy, x.shape())).
 * @version 0.2
 * @date 2026-11-05
 */
#pragma once

#include <string>

#include "Module/Module.h"

namespace BeanTensor::Modules {

    /**
     * Reshape a contiguous tensor to rank 2 {d0, numel/d0} with no parameters.
     * Joins the computation graph when its input requires grad.
     * @brief Parameter-free Flatten layer.
     */
    class Flatten : public Module {
    public:
        /**
         * @brief forward: Tensor::flatten of the input.
         * @return Rank-2 view of the input, shape {d0, numel/d0}.
         */
        [[nodiscard]] Tensors::Tensor forward(const Tensors::Tensor&) override;

        /**
         * @brief Human-readable layer type.
         * @return "Flatten".
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
