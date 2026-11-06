/**
 * @file Sigmoid.cpp
 * @author GreenBean
 * @brief Sigmoid layer implementation: forward delegates to Tensor::sigmoid.
 * @version 0.2
 * @date 2026-11-05
 */
#include "Module/Sigmoid.h"

#include "Tensor/Tensor.h"

namespace BeanTensor::Modules {

    Tensors::Tensor Sigmoid::forward(const Tensors::Tensor& x) {
        return x.sigmoid();
    }

    std::string Sigmoid::name() const {
        return "Sigmoid";
    }

    ModuleSpec Sigmoid::spec() const {
        auto out = ModuleSpec{};
        out.type = "Sigmoid";
        return out;
    }

    /**
     * Parameter values are overwritten from the file's tensor records after
     * construction; this only establishes shapes and registry.
     * @param spec Architecture spec for this node only.
     * @return The constructed module.
     * @throws ErrorHandling::CorruptFile a hyperparameter is missing, malformed, or out of range.
     */
    std::unique_ptr<Module> Sigmoid::construct(const ModuleSpec&, DataTypes::DType, Tensors::Device) {
        return std::make_unique<Sigmoid>();
    }

}
