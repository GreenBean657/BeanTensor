/**
 * @file OpCPU.cpp
 * @author GreenBean
 * @brief CPU backend for im2col.
 * @version 0.1
 * @date 2026-10-28
 */
#include "Math/Convolution/Im2Col/Op.h"
#include "Math/Convolution/ConvolutionShared.h"

namespace BeanTensor::Tensors::detail {

    void im2col_cpu(const Tensor& x, const Tensor& col, const uint32_t stride, const uint32_t padding, const size_t R, const size_t S) {
        if (x.device != Device::CPU || col.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(x.device));
        }
        const auto x_shape = std::vector<size_t>(x.shape, x.shape + x.ndim);
        const auto C = x_shape[1];
        const auto K = size_t{1};
        const auto w_shape = std::vector<size_t>{C, R, S, K};
        const auto s = make_conv2d_shape(x_shape, w_shape, stride, padding);
        conv_zero_tensor(col);
        const auto eb = static_cast<size_t>(DataTypes::dtype_bytes(x.dtype));
        const auto* x_ptr = static_cast<const uint8_t*>(x.data) + x.offset * eb;
        auto* col_ptr = static_cast<uint8_t*>(col.data) + col.offset * eb;
        conv_launch_elementwise([=](const size_t b, const size_t e) {
            for (auto idx = b; idx < e; ++idx) {
                const auto m = idx / s.CRS;
                const auto col_idx = idx % s.CRS;
                const auto n = m / (s.Hout * s.Wout);
                const auto rem = m % (s.Hout * s.Wout);
                const auto ho = rem / s.Wout;
                const auto wo = rem % s.Wout;
                const auto c = col_idx / (s.R * s.S);
                const auto r = (col_idx / s.S) % s.R;
                const auto ss = col_idx % s.S;
                const auto ih = static_cast<int64_t>(ho * stride + r) - static_cast<int64_t>(padding);
                const auto iw = static_cast<int64_t>(wo * stride + ss) - static_cast<int64_t>(padding);
                if (
                    ih >= 0 && ih < static_cast<int64_t>(s.H) &&
                    iw >= 0 && iw < static_cast<int64_t>(s.W)
                ) {
                    const auto xi = nchw_offset(s, n, c, static_cast<size_t>(ih), static_cast<size_t>(iw)) * eb;
                    const auto ci = col_offset(s, m, col_idx) * eb;
                    std::memcpy(col_ptr + ci, x_ptr + xi, eb);
                }
            }
        }, s.M * s.CRS, std::vector<const Tensor*>{&x, &col});
    }
}
