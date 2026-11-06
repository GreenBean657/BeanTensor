/**
 * @file Relu.cpp
 * @author GreenBean
 * @brief ReLU layer implementation: forward delegates to Tensor::relu.
 * @version 0.2
 * @date 2026-11-05
 */
#include "Module/Relu.h"

#include "Tensor/Tensor.h"

namespace BeanTensor::Modules {

    Tensors::Tensor Relu::forward(const Tensors::Tensor& x) {
        return x.relu();
    }

    std::string Relu::name() const {
        return "Relu";
    }

    ModuleSpec Relu::spec() const {
        auto out = ModuleSpec{};
        out.type = "Relu";
        return out;
    }

    /**
     * Parameter values are overwritten from the file's tensor records after
     * construction; this only establishes shapes and registry.
     * @param spec Architecture spec for this node only.
     * @return The constructed module.
     * @throws ErrorHandling::CorruptFile a hyperparameter is missing, malformed, or out of range.
     */
    std::unique_ptr<Module> Relu::construct(const ModuleSpec&, DataTypes::DType, Tensors::Device) {
        return std::make_unique<Relu>();
    }

}
