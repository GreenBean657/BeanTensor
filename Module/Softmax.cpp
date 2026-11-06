/**
 * @file Softmax.cpp
 * @author GreenBean
 * @brief Softmax layer implementation: forward delegates to Tensor::softmax.
 * @version 0.2
 * @date 2026-11-05
 */
#include "Module/Softmax.h"

#include "Tensor/Tensor.h"

namespace BeanTensor::Modules {

    Tensors::Tensor Softmax::forward(const Tensors::Tensor& x) {
        return x.softmax();
    }

    std::string Softmax::name() const {
        return "Softmax";
    }

    ModuleSpec Softmax::spec() const {
        auto out = ModuleSpec{};
        out.type = "Softmax";
        return out;
    }

    /**
     * Parameter values are overwritten from the file's tensor records after
     * construction; this only establishes shapes and registry.
     * @param spec Architecture spec for this node only.
     * @return The constructed module.
     * @throws ErrorHandling::CorruptFile a hyperparameter is missing, malformed, or out of range.
     */
    std::unique_ptr<Module> Softmax::construct(const ModuleSpec&, DataTypes::DType, Tensors::Device) {
        return std::make_unique<Softmax>();
    }

}
