/**
 * @file AvgPool2d.cpp
 * @author GreenBean
 * @brief AvgPool2d layer implementation: forward delegates to Tensor::avg_pool2d.
 * @version 0.1
 * @date 2026-11-10
 */
#include "Module/AvgPool2d.h"

#include "Tensor/Tensor.h"

namespace BeanTensor::Modules {

    AvgPool2d::AvgPool2d(const uint32_t kernel, const std::optional<uint32_t> stride)
        : kernel(kernel), stride(stride.value_or(kernel)) {
    }

    Tensors::Tensor AvgPool2d::forward(const Tensors::Tensor& x) {
        return x.avg_pool2d(this->kernel, this->stride);
    }

    std::string AvgPool2d::name() const {
        return "AvgPool2d";
    }

    ModuleSpec AvgPool2d::spec() const {
        auto out = ModuleSpec{};
        out.type = "AvgPool2d";
        out.hyperparameters = {
            {"kernel", std::to_string(this->kernel)},
            {"stride", std::to_string(this->stride)},
        };
        return out;
    }

    /**
     * Parameter values are overwritten from the file's tensor records after
     * construction; this only establishes shapes and registry.
     * @param spec Architecture spec for this node only.
     * @return The constructed module.
     * @throws ErrorHandling::CorruptFile a hyperparameter is missing, malformed, or out of range.
     */
    std::unique_ptr<Module> AvgPool2d::construct(const ModuleSpec& spec, DataTypes::DType, Tensors::Device) {
        const auto kernel = require_hyper_u64(spec, "kernel");
        const auto stride = require_hyper_u64(spec, "stride");
        if (kernel == 0 || kernel > std::numeric_limits<uint32_t>::max() ||
            stride == 0 || stride > std::numeric_limits<uint32_t>::max()) {
            throw ErrorHandling::CorruptFile("AvgPool2d kernel/stride out of uint32 range or zero");
        }
        return std::make_unique<AvgPool2d>(static_cast<uint32_t>(kernel), std::optional<uint32_t>{static_cast<uint32_t>(stride)});
    }

}
