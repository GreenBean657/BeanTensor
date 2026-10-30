/**
 * @file BatchNorm.cpp
 * @author GreenBean
 * @brief BatchNorm module implementation: constructor validation
 * (num_features > 0, 0 < momentum <= 1, eps > 0), gamma = ones / beta = zeros
 * parameter registration on the Module base, address-stable running-stat
 * slot allocation (mean zeros, var ones, requires_grad false), and the
 * train/eval forward through Tensor::batch_norm.
 * @version 0.2
 * @date 2026-10-30
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
}
