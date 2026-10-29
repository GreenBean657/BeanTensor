/**
 * @file Convolution.cpp
 * @author GreenBean
 * @brief Tensor::conv2d method implementation: validation, output-shape math,
 * backend dispatch, and graph wiring.
 * @version 0.1
 * @date 2026-10-28
 */
#include "Math/Convolution/Convolution.h"
#include "Math/Convolution/Conv2dG.h"
#include "Math/Convolution/ConvolutionShared.h"
#include "Tensor/Tensor.h"

namespace BeanTensor::Tensors {

    Tensor Tensor::conv2d(const Tensor& weight, const Tensor& bias, const uint32_t stride, const uint32_t padding) const {
        if (this->ndim != 4) {
            throw ErrorHandling::MismatchedShapes("conv2d input must be rank-4 NCHW, got rank " + std::to_string(this->ndim));
        }
        if (weight.ndim != 4) {
            throw ErrorHandling::MismatchedShapes("conv2d weight must be rank-4 {C,R,S,K}, got rank " + std::to_string(weight.ndim));
        }
        if (bias.ndim != 1 || bias.shape[0] != weight.shape[3]) {
            throw ErrorHandling::MismatchedShapes("conv2d bias must be rank-1 {K}");
        }
        if (this->dtype != weight.dtype || this->dtype != bias.dtype) {
            throw ErrorHandling::MismatchedDTypes();
        }
        if (this->device != weight.device || this->device != bias.device) {
            throw ErrorHandling::MismatchedDevices();
        }
        if (!this->is_contiguous || !weight.is_contiguous || !bias.is_contiguous) {
            throw ErrorHandling::NotImplemented();
        }
        if (this->shape[1] != weight.shape[0]) {
            throw ErrorHandling::MismatchedShapes("input channels " + std::to_string(this->shape[1]) + " do not match weight channels " + std::to_string(weight.shape[0]));
        }
        if (stride == 0) {
            throw ErrorHandling::MismatchedShapes("conv2d stride must be > 0");
        }
        const auto H = this->shape[2];
        const auto W = this->shape[3];
        const auto R = weight.shape[1];
        const auto S = weight.shape[2];
        if (H + 2u * padding < R || W + 2u * padding < S) {
            throw ErrorHandling::MismatchedShapes("padded input smaller than kernel");
        }
        const auto Hout = detail::conv_out_size(H, R, padding, stride);
        const auto Wout = detail::conv_out_size(W, S, padding, stride);
        const auto K = weight.shape[3];
        const auto N = this->shape[0];

        this->sync();
        weight.sync();
        bias.sync();

        const auto out_shape = std::vector{N, K, Hout, Wout};
        auto out = Tensor(out_shape, this->dtype, this->device, false);
        if (this->device == Device::CPU) {
            detail::conv2d_cpu(*this, weight, &bias, out, stride, padding);
        } else if (this->device == Device::GPU) {
#if defined(USE_CUDA) || defined(USE_HIP)
            detail::conv2d_gpu(*this, weight, &bias, out, stride, padding);
#else
            throw ErrorHandling::GPUTaskOnCPUBuild();
#endif
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        out.wire_node(detail::conv2d_g_node, *this, weight, bias);
        if (out.node != nullptr) {
            out.node->_scalar = detail::conv2d_pack_stride_padding(stride, padding);
        }
        return out;
    }

    Tensor Tensor::conv2d(const Tensor& weight, const uint32_t stride, const uint32_t padding) const {
        if (this->ndim != 4) {
            throw ErrorHandling::MismatchedShapes("conv2d input must be rank-4 NCHW, got rank " + std::to_string(this->ndim));
        }
        if (weight.ndim != 4) {
            throw ErrorHandling::MismatchedShapes("conv2d weight must be rank-4 {C,R,S,K}, got rank " + std::to_string(weight.ndim));
        }
        if (this->dtype != weight.dtype) {
            throw ErrorHandling::MismatchedDTypes();
        }
        if (this->device != weight.device) {
            throw ErrorHandling::MismatchedDevices();
        }
        if (!this->is_contiguous || !weight.is_contiguous) {
            throw ErrorHandling::NotImplemented();
        }
        if (this->shape[1] != weight.shape[0]) {
            throw ErrorHandling::MismatchedShapes("input channels " + std::to_string(this->shape[1]) + " do not match weight channels " + std::to_string(weight.shape[0]));
        }
        if (stride == 0) {
            throw ErrorHandling::MismatchedShapes("conv2d stride must be > 0");
        }
        const auto H = this->shape[2];
        const auto W = this->shape[3];
        const auto R = weight.shape[1];
        const auto S = weight.shape[2];
        if (H + 2u * padding < R || W + 2u * padding < S) {
            throw ErrorHandling::MismatchedShapes("padded input smaller than kernel");
        }
        const auto Hout = detail::conv_out_size(H, R, padding, stride);
        const auto Wout = detail::conv_out_size(W, S, padding, stride);
        const auto K = weight.shape[3];
        const auto N = this->shape[0];

        this->sync();
        weight.sync();

        const auto out_shape = std::vector{N, K, Hout, Wout};
        auto out = Tensor(out_shape, this->dtype, this->device, false);
        if (this->device == Device::CPU) {
            detail::conv2d_cpu(*this, weight, nullptr, out, stride, padding);
        } else if (this->device == Device::GPU) {
#if defined(USE_CUDA) || defined(USE_HIP)
            detail::conv2d_gpu(*this, weight, nullptr, out, stride, padding);
#else
            throw ErrorHandling::GPUTaskOnCPUBuild();
#endif
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        out.wire_node(detail::conv2d_g_node, *this, &weight);
        if (out.node != nullptr) {
            out.node->_scalar = detail::conv2d_pack_stride_padding(stride, padding);
        }
        return out;
    }
}
