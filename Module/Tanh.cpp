/**
 * @file Tanh.cpp
 * @author GreenBean
 * @brief Tanh layer implementation: forward delegates to Tensor::tanh.
 * @version 0.2
 * @date 2026-11-05
 */
#include "Module/Tanh.h"

#include "Tensor/Tensor.h"

namespace BeanTensor::Modules {

    Tensors::Tensor Tanh::forward(const Tensors::Tensor& x) {
        return x.tanh();
    }

    std::string Tanh::name() const {
        return "Tanh";
    }

    ModuleSpec Tanh::spec() const {
        auto out = ModuleSpec{};
        out.type = "Tanh";
        return out;
    }

    /**
     * Parameter values are overwritten from the file's tensor records after
     * construction; this only establishes shapes and registry.
     * @param spec Architecture spec for this node only.
     * @return The constructed module.
     * @throws ErrorHandling::CorruptFile a hyperparameter is missing, malformed, or out of range.
     */
    std::unique_ptr<Module> Tanh::construct(const ModuleSpec&, DataTypes::DType, Tensors::Device) {
        return std::make_unique<Tanh>();
    }

}
