/**
 * @file Tanh.h
 * @author GreenBean
 * @brief Parameter-free tanh layer: forward = Tensor::tanh on the input.
 * @version 0.1
 * @date 2026-10-28
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
    };
}
