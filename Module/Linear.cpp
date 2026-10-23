/**
 * @file Linear.cpp
 * @author GreenBean
 * @brief Linear layer implementation: parameter construction + uniform
 * [-1/sqrt(in), 1/sqrt(in)] initialization, and the matmul + bias_add
 * forward.
 * @version 0.1
 * @date 2026-10-23
 */
#include "Module/Linear.h"

#include <cmath>
#include <utility>

#include "Tensor/Tensor.h"

namespace BeanTensor::Modules {

    Linear::Linear(const size_t in_features, const size_t out_features, const DataTypes::DType dtype,
                   const Tensors::Device device, const bool bias, const int64_t seed)
        : in_features_(in_features), out_features_(out_features), has_bias_(bias) {
        // Kaiming-uniform-style bound on the fan-in, applied to both
        // parameters; built on Tensor::random so it behaves identically on
        // CPU and GPU.
        const auto bound = 1.0 / std::sqrt(static_cast<float64_t>(in_features));
        auto weight = Tensors::Tensor({in_features, out_features}, dtype, device, true);
        weight.random(-bound, bound, seed);
        this->add_parameter("weight", std::move(weight));
        if (bias) {
            auto bias_param = Tensors::Tensor({out_features}, dtype, device, true);
            // Distinct-but-derived stream: xor 1 flips the low seed bit.
            bias_param.random(-bound, bound, seed ^ 1);
            this->add_parameter("bias", std::move(bias_param));
        }
    }

    Tensors::Tensor Linear::forward(const Tensors::Tensor& x) {
        this->clear_forward_cache();
        if (this->has_bias_) {
            // The matmul intermediate is referenced by bias_add's graph
            // node, so it must outlive this stack frame — hold it in the
            // forward cache.
            auto& y = this->hold_intermediate(x.matmul(this->parameter("weight")));
            return y.bias_add(this->parameter("bias"));
        }
        return x.matmul(this->parameter("weight"));
    }

    std::string Linear::name() const {
        return "Linear";
    }

    size_t Linear::in_features() const {
        return this->in_features_;
    }

    size_t Linear::out_features() const {
        return this->out_features_;
    }

    bool Linear::has_bias() const {
        return this->has_bias_;
    }
}
