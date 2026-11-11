/**
 * @file Loss.h
 * @author GreenBean
 * @brief Entry point for loss ops: mse (forward) and its backward node
 * (mse_g), plus the callable free function BeanTensor::Loss::mse(pred,
 * target) for callers that do not want to go through a Tensor method (e.g.
 * the Module layer). The free function is declared here and defined in
 * Math/Loss/Loss.cpp, so this header only needs a forward declaration of
 * Tensor and stays includable from Tensor.h.
 * @version 0.2
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
     */
    [[nodiscard]] Tensors::Tensor mse(const Tensors::Tensor&, const Tensors::Tensor&);

    /**
     * Binary cross-entropy loss (probability form): mean of -y*log(p̂) -
     * (1-y)*log(1-p̂) over all elements. Identical contract to Tensor::bce;
     * this free form exists for callers that do not go through a Tensor
     * method.
     * @brief Binary cross-entropy (probability form, free function).
     */
    [[nodiscard]] Tensors::Tensor bce(const Tensors::Tensor&, const Tensors::Tensor&);

    /**
     * Binary cross-entropy with logits (stable logit form): mean of
     * max(x, 0) - x*y + log1p(exp(-|x|)) over all elements. Identical
     * contract to Tensor::bce_with_logits; this free form exists for callers
     * that do not go through a Tensor method.
     * @brief Binary cross-entropy with logits (free function).
     */
    [[nodiscard]] Tensors::Tensor bce_with_logits(const Tensors::Tensor&, const Tensors::Tensor&);
}

#include "Math/Loss/Mse/Mse.h"
#include "Math/Loss/MseG/MseG.h"
#include "Math/Loss/Bce/Bce.h"
#include "Math/Loss/BceG/BceG.h"
#include "Math/Loss/BceWithLogits/BceWithLogits.h"
#include "Math/Loss/BceWithLogitsG/BceWithLogitsG.h"
