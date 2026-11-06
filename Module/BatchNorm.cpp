/**
 * @file BatchNorm.cpp
 * @author GreenBean
 * @brief BatchNorm module implementation: constructor validation
 * (num_features > 0, 0 < momentum <= 1, eps > 0), gamma = ones / beta = zeros
 * parameter registration on the Module base, address-stable running-stat
 * slot allocation (mean zeros, var ones, requires_grad false), and the
 * train/eval forward through Tensor::batch_norm.
 * @version 0.3
 * @date 2026-11-05
 */
#include "Module/BatchNorm.h"

#include <memory>
#include <string>
#include <utility>

#include "Helpers/Errors.h"
#include "Tensor/Tensor.h"

namespace BeanTensor::Modules {

    BatchNorm::BatchNorm(const size_t num_features, const DataTypes::DType dtype,
                         const Tensors::Device device, const float64_t momentum, const float64_t eps)
        : num_features_(num_features), momentum_(momentum), eps_(eps) {
        if (num_features == 0) {
            throw ErrorHandling::IndexOutOfRange();
        }
        if (
            const auto bad_momentum = !(momentum > float64_t{0} && momentum <= float64_t{1});
            bad_momentum
        ) {
            throw ErrorHandling::InvalidHyperparameter(
                "momentum must satisfy 0 < momentum <= 1, got " + std::to_string(momentum));
        }
        if (
            const auto bad_eps = !(eps > float64_t{0});
            bad_eps
        ) {
            throw ErrorHandling::InvalidHyperparameter(
                "eps must be > 0, got " + std::to_string(eps));
        }
        auto gamma = Tensors::Tensor({num_features}, dtype, device, true);
        gamma.set_one();
        this->add_parameter("gamma", std::move(gamma));
        auto beta = Tensors::Tensor({num_features}, dtype, device, true);
        beta.set_zero();
        this->add_parameter("beta", std::move(beta));
        // The running stats are buffers, not parameters: excluded from
        // named_parameters()/parameter_count() and optimizer state, never
        // gradient-receiving. They live in address-stable heap slots because
        // eval-mode graph nodes hold raw Node::_saved pointers into them, so
        // a module move must not relocate them.
        auto running_mean = Tensors::Tensor({num_features}, dtype, device, false);
        running_mean.set_zero();
        this->running_stats_.push_back(std::make_unique<Tensors::Tensor>(std::move(running_mean)));
        auto running_var = Tensors::Tensor({num_features}, dtype, device, false);
        running_var.set_one();
        this->running_stats_.push_back(std::make_unique<Tensors::Tensor>(std::move(running_var)));
    }

    Tensors::Tensor BatchNorm::forward(const Tensors::Tensor& x) {
        this->clear_forward_cache();
        return x.batch_norm(this->parameter("gamma"), this->parameter("beta"),
                            *this->running_stats_[0], *this->running_stats_[1],
                            this->training, this->momentum_, this->eps_);
    }

    std::string BatchNorm::name() const {
        return "BatchNorm";
    }

    std::pair<const Tensors::Tensor&, const Tensors::Tensor&> BatchNorm::running_stats() const {
        return {*this->running_stats_[0], *this->running_stats_[1]};
    }

    ModuleSpec BatchNorm::spec() const {
        auto out = ModuleSpec{};
        out.type = "BatchNorm";
        out.hyperparameters = {
            {"num_features", std::to_string(this->num_features_)},
            {"momentum", format_hyper_f64(this->momentum_)},
            {"eps", format_hyper_f64(this->eps_)},
        };
        return out;
    }

    std::vector<std::pair<std::string, Tensors::Tensor*>> BatchNorm::state_tensors() {
        return {{"running_mean", this->running_stats_[0].get()}, {"running_var", this->running_stats_[1].get()}};
    }

    std::vector<std::pair<std::string, const Tensors::Tensor*>> BatchNorm::state_tensors() const {
        return {{"running_mean", this->running_stats_[0].get()}, {"running_var", this->running_stats_[1].get()}};
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
    std::unique_ptr<Module> BatchNorm::construct(const ModuleSpec& spec, DataTypes::DType dtype, Tensors::Device device) {
        return std::make_unique<BatchNorm>(
            static_cast<size_t>(require_hyper_u64(spec, "num_features")),
            dtype,
            device,
            require_hyper_f64(spec, "momentum"),
            require_hyper_f64(spec, "eps"));
    }

}
