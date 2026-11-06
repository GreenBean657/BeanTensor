/**
 * @file Conv2d.cpp
 * @author GreenBean
 * @brief Conv2d layer implementation: He-uniform weight init, zero bias,
 * and Tensor::conv2d forward.
 * @version 0.2
 * @date 2026-11-05
 */
#include "Module/Conv2d.h"

#include <cmath>
#include <utility>

#include "Random/Generator.h"
#include "Tensor/Tensor.h"

namespace BeanTensor::Modules {

    Conv2d::Conv2d(const uint32_t in_ch, const uint32_t out_ch, const uint32_t kernel,
                   const uint32_t stride, const uint32_t padding,
                   const DataTypes::DType dtype, const Tensors::Device device, const bool bias,
                   const std::optional<int64_t> seed)
        : in_ch_(in_ch), out_ch_(out_ch), kernel_(kernel), stride_(stride), padding_(padding), has_bias_(bias) {
        const auto fan_in = static_cast<float64_t>(in_ch) * static_cast<float64_t>(kernel) * static_cast<float64_t>(kernel);
        const auto bound = std::sqrt(6.0 / fan_in);
        auto weight = Tensors::Tensor(std::vector<size_t>{in_ch, kernel, kernel, out_ch}, dtype, device, true);
        if (seed.has_value()) {
            weight.random(-bound, bound, seed.value());
        } else {
            weight.random(-bound, bound, std::nullopt);
        }
        this->add_parameter("weight", std::move(weight));
        if (bias) {
            auto bias_param = Tensors::Tensor(std::vector<size_t>{out_ch}, dtype, device, true);
            bias_param.set_zero();
            this->add_parameter("bias", std::move(bias_param));
        }
    }

    Tensors::Tensor Conv2d::forward(const Tensors::Tensor& x) {
        this->clear_forward_cache();
        if (this->has_bias_) {
            return x.conv2d(this->parameter("weight"), this->parameter("bias"), this->stride_, this->padding_);
        }
        return x.conv2d(this->parameter("weight"), this->stride_, this->padding_);
    }

    std::string Conv2d::name() const {
        return "Conv2d";
    }

    ModuleSpec Conv2d::spec() const {
        auto out = ModuleSpec{};
        out.type = "Conv2d";
        out.hyperparameters = {
            {"in_channels", std::to_string(this->in_ch_)},
            {"out_channels", std::to_string(this->out_ch_)},
            {"kernel", std::to_string(this->kernel_)},
            {"stride", std::to_string(this->stride_)},
            {"padding", std::to_string(this->padding_)},
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
    std::unique_ptr<Module> Conv2d::construct(const ModuleSpec& spec, DataTypes::DType dtype, Tensors::Device device) {
        const auto to_u32 = [](const char* key, const ModuleSpec& s) {
            const auto v = require_hyper_u64(s, key);
            if (v > std::numeric_limits<uint32_t>::max()) {
                throw ErrorHandling::CorruptFile(std::string("hyperparameter '") + key + "' exceeds uint32 range");
            }
            return static_cast<uint32_t>(v);
        };
        return std::make_unique<Conv2d>(
            to_u32("in_channels", spec),
            to_u32("out_channels", spec),
            to_u32("kernel", spec),
            to_u32("stride", spec),
            to_u32("padding", spec),
            dtype,
            device,
            require_hyper_bool(spec, "bias"),
            std::nullopt);
    }

}
