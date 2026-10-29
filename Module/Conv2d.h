/**
 * @file Conv2d.h
 * @author GreenBean
 * @brief 2-D convolution module: weight {C,R,S,K} plus optional bias {K},
 * He fan-in initialization, and forward = Tensor::conv2d.
 * @version 0.1
 * @date 2026-10-28
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "Module/Module.h"
#include "Tensor/DTypes.h"

namespace BeanTensor::Modules {

    /**
     * 2-D convolution layer for NCHW inputs. Weight shape is {in_channels,
     * kernel_size, kernel_size, out_channels}; bias is {out_channels} when
     * enabled. Weight initialization is He (Kaiming) uniform with bound
     * sqrt(6 / (in_channels * kernel_size * kernel_size)), bias is exactly zero.
     * Built on Tensor::conv2d so CPU and GPU share the same forward/backward.
     * @brief 2-D convolution layer.
     */
    class Conv2d final : public Module {
    public:
        /**
         * Construct and register the parameters, then initialize them.
         * @brief Construct a Conv2d layer.
         * @param in_ch Number of input channels.
         * @param out_ch Number of output channels.
         * @param kernel Kernel size (used for both R and S).
         * @param stride Stride along H and W.
         * @param padding Zero padding on all sides.
         * @param dtype Parameter/input dtype.
         * @param device Device to hold the parameters.
         * @param bias Whether to register and add a bias parameter.
         * @param seed Init seed for the uniform weight fill. When omitted,
         *        weight is filled from the next draw of the process-wide
         *        Generator; the bias is set to zero and consumes no draw.
         */
        Conv2d(uint32_t in_ch, uint32_t out_ch, uint32_t kernel,
               uint32_t stride = 1, uint32_t padding = 0,
               DataTypes::DType dtype = DataTypes::DType::Float32,
               Tensors::Device device = Tensors::Device::CPU,
               bool bias = true,
               std::optional<int64_t> seed = std::nullopt);

        /**
         * NCHW conv2d forward.
         * @brief forward: conv2d (+ bias).
         * @param x Input tensor, rank 4 {N,C,H,W}.
         * @return Convolved output {N,K,Hout,Wout}.
         */
        [[nodiscard]] Tensors::Tensor forward(const Tensors::Tensor& x) override;

        /**
         * @brief Human-readable layer type.
         * @return "Conv2d".
         */
        [[nodiscard]] std::string name() const override;

    private:
        uint32_t in_ch_;
        uint32_t out_ch_;
        uint32_t kernel_;
        uint32_t stride_;
        uint32_t padding_;
        bool has_bias_;
    };
}
