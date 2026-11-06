/**
 * @file Flatten.cpp
 * @author GreenBean
 * @brief Flatten layer implementation: forward delegates to Tensor::flatten.
 * @version 0.2
 * @date 2026-11-05
 */
#include "Module/Flatten.h"

#include "Tensor/Tensor.h"

namespace BeanTensor::Modules {

    Tensors::Tensor Flatten::forward(const Tensors::Tensor& x) {
        return x.flatten();
    }

    std::string Flatten::name() const {
        return "Flatten";
    }

    ModuleSpec Flatten::spec() const {
        auto out = ModuleSpec{};
        out.type = "Flatten";
        return out;
    }

    /**
     * Parameter values are overwritten from the file's tensor records after
     * construction; this only establishes shapes and registry.
     * @param spec Architecture spec for this node only.
     * @return The constructed module.
     * @throws ErrorHandling::CorruptFile a hyperparameter is missing, malformed, or out of range.
     */
    std::unique_ptr<Module> Flatten::construct(const ModuleSpec&, DataTypes::DType, Tensors::Device) {
        return std::make_unique<Flatten>();
    }

}
