/**
 * @file Dropout.cpp
 * @brief Dropout module implementation: train multiplies by a seeded threshold
 * mask, eval multiplies by the scalar 1.0 identity passthrough.
 * @version 0.2
 * @date 2026-11-05
 */
#include "Module/Dropout.h"

#include "Helpers/Errors.h"
#include "Math/Dropout/Dropout.h"
#include "Random/Generator.h"
#include "Tensor/Tensor.h"

namespace BeanTensor::Modules {

    Dropout::Dropout(const float64_t p)
        : p_(p) {
        if (p < float64_t{0.0} || p >= float64_t{1.0}) {
            throw ErrorHandling::InvalidProbability("dropout probability must satisfy 0 <= p < 1");
        }
    }

    Tensors::Tensor Dropout::forward(const Tensors::Tensor& x) {
        this->clear_forward_cache();
        if (!this->is_training()) {
            return x.mul(float64_t{1.0});
        }
        const auto seed = static_cast<int64_t>(Random::default_generator().draw_seed());
        auto& held = this->hold_intermediate(Math::Dropout::train_mask(x, this->p_, seed));
        return x.mul(held);
    }

    std::string Dropout::name() const {
        return "Dropout";
    }


    ModuleSpec Dropout::spec() const {
        auto out = ModuleSpec{};
        out.type = "Dropout";
        out.hyperparameters = {{"p", format_hyper_f64(this->p_)}};
        return out;
    }

    /**
     * Parameter values are overwritten from the file's tensor records after
     * construction; this only establishes shapes and registry.
     * @param spec Architecture spec for this node only.
     * @return The constructed module.
     * @throws ErrorHandling::CorruptFile a hyperparameter is missing, malformed, or out of range.
     */
    std::unique_ptr<Module> Dropout::construct(const ModuleSpec& spec, DataTypes::DType, Tensors::Device) {
        return std::make_unique<Dropout>(require_hyper_f64(spec, "p"));
    }

}
