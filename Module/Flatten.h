/**
 * @file Flatten.h
 * @author GreenBean
 * @brief Parameter-free Flatten layer: forward = Tensor::flatten on the input.
 * Joins the computation graph when its input requires grad (FlattenG backward:
 * dx = reshape(dy, x.shape())).
 * @version 0.1
 * @date 2026-10-28
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
    };
}
