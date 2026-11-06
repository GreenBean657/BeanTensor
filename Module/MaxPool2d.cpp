/**
 * @file MaxPool2d.cpp
 * @author GreenBean
 * @brief MaxPool2d layer implementation: forward delegates to Tensor::max_pool2d.
 * @version 0.2
 * @date 2026-11-05
 */
#include "Module/MaxPool2d.h"

#include "Tensor/Tensor.h"

namespace BeanTensor::Modules {

    MaxPool2d::MaxPool2d(const uint32_t kernel, const std::optional<uint32_t> stride)
        : kernel(kernel), stride(stride.value_or(kernel)) {
    }

    Tensors::Tensor MaxPool2d::forward(const Tensors::Tensor& x) {
        return x.max_pool2d(this->kernel, this->stride);
    }

    std::string MaxPool2d::name() const {
        return "MaxPool2d";
    }

    ModuleSpec MaxPool2d::spec() const {
        auto out = ModuleSpec{};
        out.type = "MaxPool2d";
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
    std::unique_ptr<Module> MaxPool2d::construct(const ModuleSpec& spec, DataTypes::DType, Tensors::Device) {
        const auto kernel = require_hyper_u64(spec, "kernel");
        const auto stride = require_hyper_u64(spec, "stride");
        if (kernel == 0 || kernel > std::numeric_limits<uint32_t>::max() ||
            stride == 0 || stride > std::numeric_limits<uint32_t>::max()) {
            throw ErrorHandling::CorruptFile("MaxPool2d kernel/stride out of uint32 range or zero");
        }
        return std::make_unique<MaxPool2d>(static_cast<uint32_t>(kernel), std::optional<uint32_t>{static_cast<uint32_t>(stride)});
    }

}
