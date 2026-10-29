/**
 * @file OpCPU.cpp
 * @author GreenBean
 * @brief CPU backend for fused Conv2d forward.
 * @version 0.1
 * @date 2026-10-28
 */
#include "Math/Convolution/Conv2d/Op.h"
#include "Math/Convolution/ConvolutionShared.h"
#include "Math/LinearAlgebra/MatMul/Matmul.h"

namespace BeanTensor::Tensors::detail {

    void conv2d_cpu(const Tensor& x, const Tensor& weight, const Tensor* bias, const Tensor& out,
                    const uint32_t stride, const uint32_t padding) {
        if (x.device != Device::CPU || weight.device != Device::CPU || out.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(x.device));
        }
        if (bias != nullptr && bias->device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(bias->device));
        }
        const auto x_shape = std::vector<size_t>(x.shape, x.shape + x.ndim);
        const auto w_shape = std::vector<size_t>(weight.shape, weight.shape + weight.ndim);
        const auto s = make_conv2d_shape(x_shape, w_shape, stride, padding);

        auto col = Tensor(std::vector<size_t>{s.M, s.CRS}, x.dtype, Device::CPU, false);
        im2col_cpu(x, col, stride, padding, s.R, s.S);
        col.sync();

        auto weight2d = const_cast<Tensor&>(weight).reshape(std::vector<size_t>{s.CRS, s.K});
        auto gemm_out = Tensor(std::vector<size_t>{s.M, s.K}, x.dtype, Device::CPU, false);
        matmul_cpu(col, weight2d, gemm_out);
        gemm_out.sync();

        mk_to_nchw_kernel(static_cast<const uint8_t*>(gemm_out.data) + gemm_out.offset * DataTypes::dtype_bytes(x.dtype),
                          static_cast<uint8_t*>(out.data) + out.offset * DataTypes::dtype_bytes(x.dtype),
                          s, static_cast<size_t>(DataTypes::dtype_bytes(x.dtype)));

        if (bias != nullptr) {
            bias_add_nchw_dispatch(*bias, out);
        }
    }
}
