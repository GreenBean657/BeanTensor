/**
 * @file Softmax.h
 * @author GreenBean
 * @brief Parameter-free rank-2 softmax layer: forward = Tensor::softmax on the
 * input. Joins the computation graph when the input requires grad.
 * @version 0.1
 * @date 2026-10-28
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
    };
}
