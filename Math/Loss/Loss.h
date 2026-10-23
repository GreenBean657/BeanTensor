/**
 * @file Loss.h
 * @author GreenBean
 * @brief Entry point for loss ops: mse (forward) and its backward node
 * (mse_g), plus the callable free function BeanTensor::Loss::mse(pred,
 * target) for callers that do not want to go through a Tensor method (e.g.
 * the Module layer). The free function is declared here and defined in
 * Math/Loss/Loss.cpp, so this header only needs a forward declaration of
 * Tensor and stays includable from Tensor.h.
 * @version 0.1
 * @date 2026-10-23
 */
#pragma once

namespace BeanTensor::Tensors {
    class Tensor;
}
namespace BeanTensor::Loss {
    /**
     * Mean squared error loss: mean of (pred - target)^2 over all elements,
     * rank-agnostic. Identical contract to Tensor::mse (see Tensor.h); this
     * free form exists so higher layers (Modules) can call the loss without
     * editing Tensor.h.
     * @brief Mean squared error loss (free function form).
     * @param pred Predictions; the only operand that can receive a gradient.
     * @param target Targets; never receive a gradient.
     * @return Shape-{1} tensor holding the mean loss, in pred's dtype/device.
     */
    [[nodiscard]] Tensors::Tensor mse(const Tensors::Tensor&, const Tensors::Tensor&);
}

#include "Math/Loss/Mse/Mse.h"
#include "Math/Loss/MseG/MseG.h"
