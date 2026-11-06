/**
 * @file Dropout.h
 * @author GreenBean
 * @brief Inverted dropout module: train forward scales surviving elements by
 * 1/(1-p) using a seeded random mask; eval forward is an identity passthrough.
 * @version 0.2
 * @date 2026-11-05
 */
#pragma once

#include <cstdint>
#include <string>

#include "Module/Module.h"
#include "Tensor/DTypes.h"

namespace BeanTensor::Modules {

    /**
     * Inverted dropout layer. Train forward is `x.mul(train_mask(x, p, seed))`,
     * implemented through the existing elementwise multiply so the autograd
     * path yields `dx = dy * mask`. The mask is held alive in the module's
     * forward cache. Eval forward is `x.mul(1.0)`, a fresh identity tensor with
     * passthrough gradients and no RNG.
     * @brief Parameter-free inverted dropout module.
     */
    class Dropout final : public Module {
    public:
        /**
         * @brief Construct a Dropout module.
         * @param p Drop probability; must satisfy `0 <= p < 1`.
         * @throws ErrorHandling::InvalidProbability If @p p is outside [0, 1).
         */
        explicit Dropout(float64_t p);

        /**
         * @brief Forward: masked in train mode, identity in eval mode.
         * @param x Input tensor.
         * @return `x * mask` in train mode, `x * 1.0` in eval mode.
         */
        [[nodiscard]] Tensors::Tensor forward(const Tensors::Tensor& x) override;

        /**
         * @brief Human-readable layer type.
         * @return "Dropout".
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

    private:
        float64_t p_;
    };

}
