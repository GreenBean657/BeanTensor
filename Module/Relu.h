/**
 * @file Relu.h
 * @author GreenBean
 * @brief Parameter-free ReLU layer: forward = Tensor::relu on the input. Its
 * graph node points at its input, so a container such as Sequential keeps the
 * preceding stage's output alive in its forward cache.
 * @version 0.1
 * @date 2026-10-23
 */
#pragma once

#include <string>

#include "Module/Module.h"

namespace BeanTensor::Modules {

    /**
     * Elementwise max(x, 0) with no parameters. Joins the computation graph
     * when its input requires grad (relu_G backward: dx = x > 0 ? dy : 0).
     * @brief Parameter-free ReLU layer.
     */
    class Relu : public Module {
    public:
        /**
         * @brief forward: Tensor::relu of the input.
         * @return max(x, 0), same shape/dtype/device as the input.
         */
        [[nodiscard]] Tensors::Tensor forward(const Tensors::Tensor&) override;

        /**
         * @brief Human-readable layer type.
         * @return "Relu".
         */
        [[nodiscard]] std::string name() const override;
    };
}
