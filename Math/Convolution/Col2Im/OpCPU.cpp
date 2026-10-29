/**
 * @file OpCPU.cpp
 * @author GreenBean
 * @brief CPU backend for col2im.
 * @version 0.1
 * @date 2026-10-28
 */
#include "Math/Convolution/Col2Im/Op.h"
#include "Math/Convolution/ConvolutionShared.h"

namespace BeanTensor::Tensors::detail {

    void col2im_cpu(const Tensor& col, const Tensor& dx, const uint32_t stride, const uint32_t padding, const size_t R, const size_t S) {
        if (col.device != Device::CPU || dx.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(col.device));
        }
        const auto dx_shape = std::vector<size_t>(dx.shape, dx.shape + dx.ndim);
        const auto x_shape = dx_shape;
        const auto C = x_shape[1];
        const auto K = size_t{1};
        const auto w_shape = std::vector<size_t>{C, R, S, K};
        const auto s = make_conv2d_shape(x_shape, w_shape, stride, padding);
        col2im_dispatch(col, dx, s, stride, padding);
    }
}
