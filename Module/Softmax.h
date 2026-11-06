/**
 * @file Softmax.h
 * @author GreenBean
 * @brief Parameter-free rank-2 softmax layer: forward = Tensor::softmax on the
 * input. Joins the computation graph when the input requires grad.
 * @version 0.2
 * @date 2026-11-05
 */
#pragma once

#include <string>

#include "Module/Module.h"

namespace BeanTensor::Modules {

    /**
     * Last-axis softmax over a rank-2 input (rows x classes) with no parameters.
     * Joins the computation graph when its input requires grad (softmax_G
     * backward: dx = y * (dy - rowsum(dy*y)), with y recomputed from the saved
     * input).
     * @brief Parameter-free rank-2 softmax layer.
     */
    class Softmax : public Module {
    public:
        /**
         * @brief forward: Tensor::softmax of the input.
         * @return Softmax probabilities, same shape/dtype/device as the input.
         */
        [[nodiscard]] Tensors::Tensor forward(const Tensors::Tensor&) override;

        /**
         * @brief Human-readable layer type.
         * @return "Softmax".
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
