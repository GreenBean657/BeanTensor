/**
 * @file AvgPool2d.h
 * @author GreenBean
 * @brief AvgPool2d layer: forward = Tensor::avg_pool2d on the input. Its graph
 * node points at its input, so a container such as Sequential keeps the
 * preceding stage's output alive in its forward cache.
 * @version 0.1
 * @date 2026-11-10
 */
#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "Module/Module.h"

namespace BeanTensor::Modules {

    /**
     * 2D average pooling layer with a square kernel and stride. No padding;
     * output spatial size is floor((H - kernel) / stride) + 1. Joins the
     * computation graph when its input requires grad (AvgPool2dG backward:
     * dx[window] += dy / (kernel²)).
     * @brief Square-kernel 2D average pooling module.
     */
    class AvgPool2d : public Module {
    public:
        /**
         * @brief Construct an AvgPool2d layer.
         * @param kernel Window height/width.
         * @param stride Step between adjacent windows; omitted -> kernel.
         */
        AvgPool2d(uint32_t kernel, std::optional<uint32_t> stride = std::nullopt);

        /**
         * @brief forward: Tensor::avg_pool2d of the input.
         * @return Pooled tensor, rank-4 NCHW.
         */
        [[nodiscard]] Tensors::Tensor forward(const Tensors::Tensor&) override;

        /**
         * @brief Human-readable layer type.
         * @return "AvgPool2d".
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
        uint32_t kernel;
        uint32_t stride;
    };
}
