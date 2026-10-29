/**
 * @file Sigmoid.h
 * @author GreenBean
 * @brief Parameter-free sigmoid layer: forward = Tensor::sigmoid on the input.
 * @version 0.1
 * @date 2026-10-28
 */
#pragma once

#include <string>

#include "Module/Module.h"

namespace BeanTensor::Modules {

    /**
     * Elementwise sigmoid with no parameters. Joins the computation graph
     * when its input requires grad.
     * @brief Parameter-free sigmoid layer.
     */
    class Sigmoid : public Module {
    public:
        /**
         * @brief forward: Tensor::sigmoid of the input.
         * @return 1/(1+exp(-x)), same shape/dtype/device as the input.
         */
        [[nodiscard]] Tensors::Tensor forward(const Tensors::Tensor&) override;

        /**
         * @brief Human-readable layer type.
         * @return "Sigmoid".
         */
        [[nodiscard]] std::string name() const override;
    };
}
