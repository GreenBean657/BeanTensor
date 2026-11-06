/**
 * @file Linear.cpp
 * @author GreenBean
 * @brief Linear layer implementation: He-uniform weight initialization
 * (bound sqrt(6 / in_features)), zero bias, and the matmul + bias_add
 * forward.
 * @version 0.4
 * @date 2026-11-05
 */
#include "Module/Linear.h"

#include <cmath>
#include <utility>

#include "Tensor/Tensor.h"

namespace BeanTensor::Modules {

    Linear::Linear(const size_t in_features, const size_t out_features, const DataTypes::DType dtype,
                   const Tensors::Device device, const bool bias, const std::optional<int64_t> seed)
        : in_features_(in_features), out_features_(out_features), has_bias_(bias) {
        // He (Kaiming) uniform on the fan-in: weight ~ U[-bound, bound],
        // bound = sqrt(6 / in_features); bias is exactly zero. Built on
        // Tensor::random so it behaves identically on CPU and GPU.
        const auto bound = std::sqrt(6.0 / static_cast<float64_t>(in_features));
        auto weight = Tensors::Tensor({in_features, out_features}, dtype, device, true);
        if (seed.has_value()) {
            weight.random(-bound, bound, seed.value());
        } else {
            weight.random(-bound, bound, std::nullopt);
        }
        this->add_parameter("weight", std::move(weight));
        if (bias) {
            auto bias_param = Tensors::Tensor({out_features}, dtype, device, true);
            bias_param.set_zero();
            this->add_parameter("bias", std::move(bias_param));
        }
    }

    Tensors::Tensor Linear::forward(const Tensors::Tensor& x) {
        this->clear_forward_cache();
        if (this->has_bias_) {
            // The matmul intermediate is referenced by bias_add's graph
            // node, so it must outlive this stack frame — hold it in the
            // forward cache.
            const auto& y = this->hold_intermediate(x.matmul(this->parameter("weight")));
            return y.bias_add(this->parameter("bias"));
        }
        return x.matmul(this->parameter("weight"));
    }

    std::string Linear::name() const {
        return "Linear";
    }

    size_t Linear::in_features() const noexcept {
        return this->in_features_;
    }

    size_t Linear::out_features() const noexcept {
        return this->out_features_;
    }

    bool Linear::has_bias() const noexcept {
        return this->has_bias_;
    }

    ModuleSpec Linear::spec() const {
        auto out = ModuleSpec{};
        out.type = "Linear";
        out.hyperparameters = {
            {"in_features", std::to_string(this->in_features_)},
            {"out_features", std::to_string(this->out_features_)},
            {"bias", this->has_bias_ ? "1" : "0"},
        };
        return out;
    }

    /**
     * Parameter values are overwritten from the file's tensor records after
     * construction; this only establishes shapes and registry.
     * @param spec Architecture spec for this node only.
     * @param dtype Parameter dtype for the rebuilt parameters.
     * @param device Device to hold the rebuilt parameters on.
     * @return The constructed module.
     * @throws ErrorHandling::CorruptFile a hyperparameter is missing, malformed, or out of range.
     */
    std::unique_ptr<Module> Linear::construct(const ModuleSpec& spec, DataTypes::DType dtype, Tensors::Device device) {
        return std::make_unique<Linear>(
            static_cast<size_t>(require_hyper_u64(spec, "in_features")),
            static_cast<size_t>(require_hyper_u64(spec, "out_features")),
            dtype,
            device,
            require_hyper_bool(spec, "bias"),
            std::nullopt);
    }

}
