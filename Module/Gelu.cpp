/**
 * @file Gelu.cpp
 * @author GreenBean
 * @brief Gelu layer implementation: forward delegates to Tensor::gelu.
 * @version 0.1
 * @date 2026-11-10
 */
#include "Module/Gelu.h"

#include "Tensor/Tensor.h"

namespace BeanTensor::Modules {

    Tensors::Tensor Gelu::forward(const Tensors::Tensor& x) {
        return x.gelu();
    }

    std::string Gelu::name() const {
        return "Gelu";
    }

    ModuleSpec Gelu::spec() const {
        auto out = ModuleSpec{};
        out.type = "Gelu";
        return out;
    }

    /**
     * Parameter values are overwritten from the file's tensor records after
     * construction; this only establishes shapes and registry.
     * @param spec Architecture spec for this node only.
     * @return The constructed module.
     * @throws ErrorHandling::CorruptFile a hyperparameter is missing, malformed, or out of range.
     */
    std::unique_ptr<Module> Gelu::construct(const ModuleSpec&, DataTypes::DType, Tensors::Device) {
        return std::make_unique<Gelu>();
    }

}
