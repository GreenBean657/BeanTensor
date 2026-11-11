/**
 * @file avg_pool2d_tests.cpp
 * @author GreenBean
 * @brief Framework-free tests for AvgPool2d: forward values, floor geometry,
 * overlapping-window accumulation, backward equal-share scatter, error cases,
 * the module wrapper, serialization, and a Conv2d->AvgPool2d->Linear smoke.
 * @version 0.1
 * @date 2026-11-10
 */
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

#include "BeanTensor.h"
#include "Module/AvgPool2d.h"
#include "Tensor/Tensor.h"

#if defined(USE_HIP)
#include <hip/hip_runtime.h>
#elif defined(USE_CUDA)
#include <cuda_runtime.h>
#endif

using BeanTensor::Tensors::Tensor;
using BeanTensor::Tensors::Device;
using BeanTensor::float32_t;
using BeanTensor::float64_t;
using BeanTensor::float16_t;
using DType = BeanTensor::DataTypes::DType;
using Sequential = BeanTensor::Modules::Sequential;

static int32_t failures = 0;
#define CHECK(...) do { \
    if (!(__VA_ARGS__)) { ++failures; std::cerr << "FAIL: " << #__VA_ARGS__ << " @ line " << __LINE__ << "\n"; } \
} while (0)

static bool near64(const float64_t a, const float64_t b, const float64_t eps) {
    return std::fabs(a - b) <= eps;
}

static bool eq32(const float32_t a, const float32_t b) {
    return std::fabs(a - b) <= 1e-5f;
}

static bool gpu_available() {
#if defined(USE_HIP)
    int32_t count = 0;
    return hipGetDeviceCount(&count) == hipSuccess && count > 0;
#elif defined(USE_CUDA)
    int32_t count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
#else
    return false;
#endif
}

static float32_t get4(Tensor& t, const std::vector<size_t>& idx) {
    return t.at<float32_t>({idx[0], idx[1], idx[2], idx[3]});
}

static void set4(Tensor& t, const std::vector<size_t>& idx, const float32_t v) {
    t.set<float32_t>({idx[0], idx[1], idx[2], idx[3]}, v);
}

static Tensor ones(const std::vector<size_t>& shape, const DType dt) {
    auto t = Tensor(shape, dt, Device::CPU, false);
    t.set_one();
    return t;
}

static std::vector<float32_t> ref_avg_pool2d_forward(const Tensor& x, const size_t kernel, const size_t stride) {
    const auto N = x.dim(0);
    const auto C = x.dim(1);
    const auto H = x.dim(2);
    const auto W = x.dim(3);
    const auto Hout = (H - kernel) / stride + 1;
    const auto Wout = (W - kernel) / stride + 1;
    auto out = std::vector<float32_t>(N * C * Hout * Wout, 0.0f);
    for (auto n = size_t{0}; n < N; ++n) {
        for (auto c = size_t{0}; c < C; ++c) {
            for (auto oh = size_t{0}; oh < Hout; ++oh) {
                for (auto ow = size_t{0}; ow < Wout; ++ow) {
                    auto total = float64_t{0.0};
                    for (auto kh = size_t{0}; kh < kernel; ++kh) {
                        for (auto kw = size_t{0}; kw < kernel; ++kw) {
                            total += static_cast<float64_t>(get4(const_cast<Tensor&>(x), {n, c, oh * stride + kh, ow * stride + kw}));
                        }
                    }
                    out[((n * C + c) * Hout + oh) * Wout + ow] = static_cast<float32_t>(total / static_cast<float64_t>(kernel * kernel));
                }
            }
        }
    }
    return out;
}

static std::vector<float32_t> ref_avg_pool2d_backward(const std::vector<size_t>& xshape, const Tensor& dy, const size_t kernel, const size_t stride) {
    const auto N = xshape[0];
    const auto C = xshape[1];
    const auto H = xshape[2];
    const auto W = xshape[3];
    const auto Hout = dy.dim(2);
    const auto Wout = dy.dim(3);
    auto dx = std::vector<float32_t>(N * C * H * W, 0.0f);
    const auto scale = float64_t{1.0} / static_cast<float64_t>(kernel * kernel);
    for (auto n = size_t{0}; n < N; ++n) {
        for (auto c = size_t{0}; c < C; ++c) {
            for (auto oh = size_t{0}; oh < Hout; ++oh) {
                for (auto ow = size_t{0}; ow < Wout; ++ow) {
                    const auto inc = static_cast<float64_t>(get4(const_cast<Tensor&>(dy), {n, c, oh, ow})) * scale;
                    for (auto kh = size_t{0}; kh < kernel; ++kh) {
                        for (auto kw = size_t{0}; kw < kernel; ++kw) {
                            dx[((n * C + c) * H + oh * stride + kh) * W + ow * stride + kw] += static_cast<float32_t>(inc);
                        }
                    }
                }
            }
        }
    }
    return dx;
}

static void test_forward_values() {
    auto x = Tensor({1, 1, 4, 4}, DType::Float32, Device::CPU);
    const auto xv = std::vector<float32_t>{
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };
    for (auto i = size_t{0}; i < xv.size(); ++i) {
        x.set<float32_t>({0, 0, i / 4, i % 4}, xv[i]);
    }

    auto y = x.avg_pool2d(2, 2);
    CHECK(y.rank() == 4);
    CHECK(y.dim(0) == 1);
    CHECK(y.dim(1) == 1);
    CHECK(y.dim(2) == 2);
    CHECK(y.dim(3) == 2);
    CHECK(eq32(y.at<float32_t>({0, 0, 0, 0}), 3.5f));
    CHECK(eq32(y.at<float32_t>({0, 0, 0, 1}), 5.5f));
    CHECK(eq32(y.at<float32_t>({0, 0, 1, 0}), 11.5f));
    CHECK(eq32(y.at<float32_t>({0, 0, 1, 1}), 13.5f));
}

static void test_default_stride() {
    auto x = Tensor({1, 1, 4, 4}, DType::Float32, Device::CPU);
    x.set_zero();
    set4(x, {0, 0, 0, 0}, 1.0f);
    set4(x, {0, 0, 0, 1}, 2.0f);
    set4(x, {0, 0, 1, 0}, 3.0f);
    set4(x, {0, 0, 1, 1}, 4.0f);

    auto y_default = x.avg_pool2d(2);
    auto y_explicit = x.avg_pool2d(2, 2);
    CHECK(eq32(y_default.at<float32_t>({0, 0, 0, 0}), y_explicit.at<float32_t>({0, 0, 0, 0})));
    CHECK(eq32(y_default.at<float32_t>({0, 0, 0, 0}), 2.5f));
}

static void test_floor_geometry() {
    auto x = Tensor({1, 1, 5, 5}, DType::Float32, Device::CPU);
    x.set_zero();
    auto y = x.avg_pool2d(2, 2);
    CHECK(y.dim(2) == 2);
    CHECK(y.dim(3) == 2);
}

static void test_rectangular() {
    auto x = Tensor({1, 1, 4, 6}, DType::Float32, Device::CPU);
    x.set_zero();
    for (auto h = size_t{0}; h < 4; ++h) {
        for (auto w = size_t{0}; w < 6; ++w) {
            set4(x, {0, 0, h, w}, static_cast<float32_t>(h * 6 + w + 1));
        }
    }
    auto y = x.avg_pool2d(2, 2);
    CHECK(y.dim(2) == 2);
    CHECK(y.dim(3) == 3);
    CHECK(eq32(y.at<float32_t>({0, 0, 0, 0}), 4.5f));
    CHECK(eq32(y.at<float32_t>({0, 0, 0, 1}), 6.5f));
    CHECK(eq32(y.at<float32_t>({0, 0, 1, 0}), 16.5f));
    CHECK(eq32(y.at<float32_t>({0, 0, 1, 1}), 18.5f));
}

static void test_multi_batch_channel() {
    auto x = Tensor({2, 3, 4, 4}, DType::Float32, Device::CPU);
    x.set_zero();
    for (auto i = size_t{0}; i < 2 * 3 * 4 * 4; ++i) {
        x.set<float32_t>({i / 48, (i % 48) / 16, (i % 16) / 4, i % 4}, static_cast<float32_t>(i + 1));
    }
    auto y = x.avg_pool2d(2, 2);
    CHECK(y.dim(0) == 2);
    CHECK(y.dim(1) == 3);
    CHECK(y.dim(2) == 2);
    CHECK(y.dim(3) == 2);
    const auto ref = ref_avg_pool2d_forward(x, 2, 2);
    for (auto i = size_t{0}; i < ref.size(); ++i) {
        CHECK(eq32(y.at<float32_t>({i / 12, (i % 12) / 4, (i % 4) / 2, i % 2}), ref[i]));
    }
}

static void test_odd_sizes() {
    auto x = Tensor({1, 1, 7, 7}, DType::Float32, Device::CPU);
    x.set_zero();
    for (auto i = size_t{0}; i < 49; ++i) {
        x.set<float32_t>({0, 0, i / 7, i % 7}, static_cast<float32_t>(i + 1));
    }
    auto y = x.avg_pool2d(3, 2);
    CHECK(y.dim(2) == 3);
    CHECK(y.dim(3) == 3);
    const auto ref = ref_avg_pool2d_forward(x, 3, 2);
    for (auto i = size_t{0}; i < ref.size(); ++i) {
        CHECK(eq32(y.at<float32_t>({0, 0, i / 3, i % 3}), ref[i]));
    }
}

static void test_overlapping_stride() {
    auto x = Tensor({1, 1, 3, 3}, DType::Float32, Device::CPU, true);
    const auto xv = std::vector<float32_t>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    for (auto i = size_t{0}; i < xv.size(); ++i) {
        x.set<float32_t>({0, 0, i / 3, i % 3}, xv[i]);
    }
    auto y = x.avg_pool2d(2, 1);
    CHECK(y.dim(2) == 2);
    CHECK(y.dim(3) == 2);
    CHECK(eq32(y.at<float32_t>({0, 0, 0, 0}), 3.0f));
    CHECK(eq32(y.at<float32_t>({0, 0, 0, 1}), 4.0f));
    CHECK(eq32(y.at<float32_t>({0, 0, 1, 0}), 6.0f));
    CHECK(eq32(y.at<float32_t>({0, 0, 1, 1}), 7.0f));

    auto loss = y.sum();
    loss.backward(DType::Float32);
    CHECK(x.has_grad());
    // Center element (1,1) participates in all four windows.
    CHECK(eq32(x.gradient().at<float32_t>({0, 0, 1, 1}), 1.0f));
}

static void test_backward_routing() {
    const auto N = size_t{1};
    const auto C = size_t{2};
    const auto H = size_t{4};
    const auto W = size_t{4};
    const auto kernel = size_t{2};
    const auto stride = size_t{2};

    auto x = Tensor({N, C, H, W}, DType::Float32, Device::CPU, true);
    for (auto i = size_t{0}; i < N * C * H * W; ++i) {
        x.set<float32_t>({0, i / 16, (i % 16) / 4, i % 4}, static_cast<float32_t>(i + 1));
    }

    auto y = x.avg_pool2d(static_cast<uint32_t>(kernel), static_cast<uint32_t>(stride));
    auto dy = Tensor({N, C, y.dim(2), y.dim(3)}, DType::Float32, Device::CPU, false);
    for (auto i = size_t{0}; i < dy.element_count(); ++i) {
        dy.set<float32_t>({0, i / 4, (i % 4) / 2, i % 2}, static_cast<float32_t>(i + 1));
    }
    auto z = y.mul(dy);
    auto loss = z.sum();
    loss.backward(DType::Float32);
    CHECK(x.has_grad());

    const auto dx_ref = ref_avg_pool2d_backward({N, C, H, W}, dy, kernel, stride);
    for (auto i = size_t{0}; i < dx_ref.size(); ++i) {
        const auto got = static_cast<float64_t>(x.gradient().at<float32_t>({0, i / 16, (i % 16) / 4, i % 4}));
        CHECK(near64(got, static_cast<float64_t>(dx_ref[i]), 1e-5));
    }
}

static void test_backward_overlapping() {
    const auto H = size_t{5};
    const auto W = size_t{5};
    const auto kernel = size_t{3};
    const auto stride = size_t{2};

    auto x = Tensor({1, 1, H, W}, DType::Float32, Device::CPU, true);
    for (auto i = size_t{0}; i < H * W; ++i) {
        x.set<float32_t>({0, 0, i / W, i % W}, static_cast<float32_t>(i + 1));
    }

    auto y = x.avg_pool2d(static_cast<uint32_t>(kernel), static_cast<uint32_t>(stride));
    auto dy = Tensor({1, 1, y.dim(2), y.dim(3)}, DType::Float32, Device::CPU, false);
    dy.set_one();
    auto z = y.mul(dy);
    auto loss = z.sum();
    loss.backward(DType::Float32);
    CHECK(x.has_grad());

    const auto dx_ref = ref_avg_pool2d_backward({1, 1, H, W}, dy, kernel, stride);
    for (auto i = size_t{0}; i < dx_ref.size(); ++i) {
        const auto got = static_cast<float64_t>(x.gradient().at<float32_t>({0, 0, i / W, i % W}));
        CHECK(near64(got, static_cast<float64_t>(dx_ref[i]), 1e-5));
    }
}

static void test_non_unit_upstream() {
    auto x = Tensor({1, 1, 4, 4}, DType::Float32, Device::CPU, true);
    for (auto i = size_t{0}; i < 16; ++i) {
        x.set<float32_t>({0, 0, i / 4, i % 4}, static_cast<float32_t>(i + 1));
    }
    auto y = x.avg_pool2d(2, 2);
    auto dy = Tensor({1, 1, 2, 2}, DType::Float32, Device::CPU, false);
    dy.set<float32_t>({0, 0, 0, 0}, 2.0f);
    dy.set<float32_t>({0, 0, 0, 1}, -3.0f);
    dy.set<float32_t>({0, 0, 1, 0}, 4.0f);
    dy.set<float32_t>({0, 0, 1, 1}, -5.0f);
    auto z = y.mul(dy);
    auto loss = z.sum();
    loss.backward(DType::Float32);
    CHECK(x.has_grad());

    const auto dx_ref = ref_avg_pool2d_backward({1, 1, 4, 4}, dy, 2, 2);
    for (auto i = size_t{0}; i < dx_ref.size(); ++i) {
        const auto got = static_cast<float64_t>(x.gradient().at<float32_t>({0, 0, i / 4, i % 4}));
        CHECK(near64(got, static_cast<float64_t>(dx_ref[i]), 1e-5));
    }
}

static void test_fan_out() {
    auto x = Tensor({1, 1, 4, 4}, DType::Float32, Device::CPU, true);
    for (auto i = size_t{0}; i < 16; ++i) {
        x.set<float32_t>({0, 0, i / 4, i % 4}, static_cast<float32_t>(i + 1));
    }
    auto y1 = x.avg_pool2d(2, 2);
    auto y2 = x.avg_pool2d(2, 2);
    auto s1 = y1.sum();
    auto s2 = y2.sum();
    auto loss = s1.add(s2);
    loss.backward(DType::Float32);
    CHECK(x.has_grad());

    // Each output element contributes twice through y1 and y2, so dx is doubled.
    const auto base = ref_avg_pool2d_backward({1, 1, 4, 4}, ones({1, 1, 2, 2}, DType::Float32), 2, 2);
    for (auto i = size_t{0}; i < base.size(); ++i) {
        const auto got = static_cast<float64_t>(x.gradient().at<float32_t>({0, 0, i / 4, i % 4}));
        CHECK(near64(got, static_cast<float64_t>(base[i] * 2.0f), 1e-5));
    }
}

static void test_repeated_backward() {
    auto x = Tensor({1, 1, 4, 4}, DType::Float32, Device::CPU, true);
    for (auto i = size_t{0}; i < 16; ++i) {
        x.set<float32_t>({0, 0, i / 4, i % 4}, static_cast<float32_t>(i + 1));
    }
    auto y = x.avg_pool2d(2, 2);
    auto loss1 = y.sum();
    loss1.backward(DType::Float32);
    auto loss2 = y.sum();
    loss2.backward(DType::Float32);

    const auto base = ref_avg_pool2d_backward({1, 1, 4, 4}, ones({1, 1, 2, 2}, DType::Float32), 2, 2);
    for (auto i = size_t{0}; i < base.size(); ++i) {
        const auto got = static_cast<float64_t>(x.gradient().at<float32_t>({0, 0, i / 4, i % 4}));
        CHECK(near64(got, static_cast<float64_t>(base[i] * 2.0f), 1e-5));
    }
}

static void test_grad_dtype() {
    auto x = Tensor({1, 1, 4, 4}, DType::Float32, Device::CPU, true);
    for (auto i = size_t{0}; i < 16; ++i) {
        x.set<float32_t>({0, 0, i / 4, i % 4}, static_cast<float32_t>(i + 1));
    }
    auto y = x.avg_pool2d(2, 2);
    auto loss = y.sum();
    loss.backward(DType::Float64);
    CHECK(x.has_grad());
    CHECK(x.gradient().get_dtype() == DType::Float64);

    const auto base = ref_avg_pool2d_backward({1, 1, 4, 4}, ones({1, 1, 2, 2}, DType::Float32), 2, 2);
    for (auto i = size_t{0}; i < base.size(); ++i) {
        const auto got = x.gradient().at<float64_t>({0, 0, i / 4, i % 4});
        CHECK(near64(got, static_cast<float64_t>(base[i]), 1e-5));
    }
}

static void test_integer_forward() {
    auto x = Tensor({1, 1, 4, 4}, DType::Int32, Device::CPU);
    for (auto i = size_t{0}; i < 16; ++i) {
        x.set<int32_t>({0, 0, i / 4, i % 4}, static_cast<int32_t>(i + 1));
    }
    auto y = x.avg_pool2d(2, 2);
    CHECK(y.get_dtype() == DType::Int32);
    // Means: 3.5, 5.5, 11.5, 13.5 -> truncate toward zero.
    CHECK(y.at<int32_t>({0, 0, 0, 0}) == 3);
    CHECK(y.at<int32_t>({0, 0, 0, 1}) == 5);
    CHECK(y.at<int32_t>({0, 0, 1, 0}) == 11);
    CHECK(y.at<int32_t>({0, 0, 1, 1}) == 13);
}

static void test_half_forward() {
    auto x = Tensor({1, 1, 2, 2}, DType::Float16, Device::CPU);
    namespace idet = BeanTensor::Intrinsics::detail;
    x.set<float16_t>({0, 0, 0, 0}, idet::unaccel_fp32_to_fp16_untracked(1.0f));
    x.set<float16_t>({0, 0, 0, 1}, idet::unaccel_fp32_to_fp16_untracked(2.0f));
    x.set<float16_t>({0, 0, 1, 0}, idet::unaccel_fp32_to_fp16_untracked(3.0f));
    x.set<float16_t>({0, 0, 1, 1}, idet::unaccel_fp32_to_fp16_untracked(4.0f));
    auto y = x.avg_pool2d(2, 2);
    CHECK(y.get_dtype() == DType::Float16);
    y.sync();
    const auto v = y.at<float16_t>({0, 0, 0, 0});
    const auto vf = idet::unaccel_fp16_to_fp32_untracked(v);
    CHECK(std::fabs(vf - 2.5f) <= 1e-3f);
}

static void test_finite_difference() {
    const auto H = size_t{6};
    const auto W = size_t{6};
    const auto kernel = size_t{3};
    const auto stride = size_t{2};

    auto x = Tensor({1, 1, H, W}, DType::Float64, Device::CPU, true);
    for (auto i = size_t{0}; i < H * W; ++i) {
        x.set<float64_t>({0, 0, i / W, i % W}, static_cast<float64_t>(i + 1) * 0.1 + 0.01);
    }
    auto y = x.avg_pool2d(static_cast<uint32_t>(kernel), static_cast<uint32_t>(stride));
    auto loss = y.sum();
    loss.backward(DType::Float64);
    CHECK(x.has_grad());

    // Analytic gradient for sum of averages is 1/(kernel²) for every input element
    // that falls inside at least one output window. For no padding, corner/edge
    // elements may participate in fewer windows.
    const auto dy_ones = ones({1, 1, y.dim(2), y.dim(3)}, DType::Float32);
    const auto dx_ref = ref_avg_pool2d_backward({1, 1, H, W}, dy_ones, kernel, stride);
    for (auto i = size_t{0}; i < dx_ref.size(); ++i) {
        const auto got = x.gradient().at<float64_t>({0, 0, i / W, i % W});
        CHECK(near64(got, static_cast<float64_t>(dx_ref[i]), 1e-5));
    }
}

static void test_errors() {
    {
        auto x = Tensor({2, 3, 4}, DType::Float32, Device::CPU);
        auto threw = false;
        try {
            auto bad = x.avg_pool2d(2, 2);
            static_cast<void>(bad);
        } catch (const BeanTensor::ErrorHandling::MismatchedShapes&) {
            threw = true;
        }
        CHECK(threw);
    }
    {
        auto x = Tensor({1, 1, 4, 4}, DType::Float32, Device::CPU);
        auto threw = false;
        try {
            auto bad = x.avg_pool2d(5, 1);
            static_cast<void>(bad);
        } catch (const BeanTensor::ErrorHandling::MismatchedShapes&) {
            threw = true;
        }
        CHECK(threw);
    }
    {
        auto x = Tensor({1, 1, 4, 4}, DType::Float32, Device::CPU);
        auto threw = false;
        try {
            auto bad = x.avg_pool2d(2, 0);
            static_cast<void>(bad);
        } catch (const BeanTensor::ErrorHandling::MismatchedShapes&) {
            threw = true;
        }
        CHECK(threw);
    }
    {
        auto x = Tensor({1, 1, 4, 4}, DType::Float32, Device::CPU);
        auto threw = false;
        try {
            auto bad = x.avg_pool2d(0);
            static_cast<void>(bad);
        } catch (const BeanTensor::ErrorHandling::MismatchedShapes&) {
            threw = true;
        }
        CHECK(threw);
    }
}

static void test_module() {
    BeanTensor::Modules::AvgPool2d pool_default(2);
    BeanTensor::Modules::AvgPool2d pool_explicit(2, 2);

    CHECK(pool_default.name() == "AvgPool2d");
    CHECK(pool_explicit.name() == "AvgPool2d");

    auto x = Tensor({1, 1, 4, 4}, DType::Float32, Device::CPU);
    x.set_zero();
    auto y_default = pool_default.forward(x);
    auto y_explicit = pool_explicit.forward(x);
    CHECK(y_default.is_same_shape(y_explicit));
    CHECK(y_default.dim(2) == 2);
    CHECK(y_default.dim(3) == 2);

    auto spec = pool_default.spec();
    CHECK(spec.type == "AvgPool2d");
    CHECK(spec.hyperparameters.size() == 2);
}

static void test_serialization_roundtrip() {
    auto net = Sequential();
    net.append("pool", std::make_unique<BeanTensor::Modules::AvgPool2d>(3, 2));
    auto x = Tensor({1, 1, 8, 8}, DType::Float32, Device::CPU, true);
    x.set_one();
    auto y1 = net.forward(x);
    auto loss1 = y1.sum();
    loss1.backward(DType::Float32);

    std::error_code ec;
    std::filesystem::create_directories("scratch", ec);
    const auto path = std::string{"scratch/avgpool2d_roundtrip_test.bt"};
    BeanTensor::Serialization::save(net.at("pool"), path, false, BeanTensor::Serialization::CheckpointLevel::Weights);
    auto loaded = BeanTensor::Serialization::load(path, Device::CPU);
    auto y2 = loaded->forward(x);
    CHECK(y1.is_same_shape(y2));
    for (auto i = size_t{0}; i < y1.element_count(); ++i) {
        CHECK(eq32(y1.at<float32_t>({0, 0, i / y1.dim(3), i % y1.dim(3)}),
                   y2.at<float32_t>({0, 0, i / y2.dim(3), i % y2.dim(3)})));
    }
}

static void test_serialization_malformed() {
    auto spec = BeanTensor::Modules::ModuleSpec{};
    spec.type = "AvgPool2d";
    spec.hyperparameters = {{"kernel", "0"}, {"stride", "2"}};
    auto threw = false;
    try {
        auto bad = BeanTensor::Modules::AvgPool2d::construct(spec, DType::Float32, Device::CPU);
        static_cast<void>(bad);
    } catch (const BeanTensor::ErrorHandling::CorruptFile&) {
        threw = true;
    }
    CHECK(threw);
}

static void test_conv_avgpool_linear_smoke() {
    auto net = Sequential();
    net.append("conv", std::make_unique<BeanTensor::Modules::Conv2d>(1, 2, 3, 1, 1));
    net.append("pool", std::make_unique<BeanTensor::Modules::AvgPool2d>(2, 2));
    net.append("flat", std::make_unique<BeanTensor::Modules::Flatten>());
    net.append("fc", std::make_unique<BeanTensor::Modules::Linear>(2 * 4 * 4, 3));
    auto x = Tensor({1, 1, 8, 8}, DType::Float32, Device::CPU, true);
    x.set_one();
    auto y = net.forward(x);
    CHECK(y.rank() == 2);
    CHECK(y.dim(0) == 1);
    CHECK(y.dim(1) == 3);

    auto loss = y.sum();
    loss.backward(DType::Float32);

    // All conv and linear parameters should have gradients.
    const auto params = net.named_parameters();
    auto grad_count = size_t{0};
    for (const auto& [_, p] : params) {
        if (p->has_grad()) {
            ++grad_count;
        }
    }
    CHECK(grad_count == params.size());
}

static void test_forward_values_gpu() {
    if (!gpu_available()) {
        return;
    }
    auto x = Tensor({2, 3, 4, 4}, DType::Float32, Device::CPU);
    for (auto i = size_t{0}; i < 2 * 3 * 4 * 4; ++i) {
        x.set<float32_t>({i / 48, (i % 48) / 16, (i % 16) / 4, i % 4}, static_cast<float32_t>(i + 1));
    }
    auto x_g = x.clone(true);
    x_g.to(Device::GPU);
    auto y = x_g.avg_pool2d(2, 2);
    y.sync();
    y.to(Device::CPU);
    y.sync();

    const auto ref = ref_avg_pool2d_forward(x, 2, 2);
    for (auto i = size_t{0}; i < ref.size(); ++i) {
        CHECK(eq32(y.at<float32_t>({i / 12, (i % 12) / 4, (i % 4) / 2, i % 2}), ref[i]));
    }
}

static void test_backward_routing_gpu() {
    if (!gpu_available()) {
        return;
    }
    const auto N = size_t{1};
    const auto C = size_t{2};
    const auto H = size_t{4};
    const auto W = size_t{4};
    const auto kernel = size_t{2};
    const auto stride = size_t{2};

    auto x = Tensor({N, C, H, W}, DType::Float32, Device::CPU, true);
    for (auto i = size_t{0}; i < N * C * H * W; ++i) {
        x.set<float32_t>({0, i / 16, (i % 16) / 4, i % 4}, static_cast<float32_t>(i + 1));
    }
    auto dy = Tensor({N, C, (H - kernel) / stride + 1, (W - kernel) / stride + 1}, DType::Float32, Device::CPU, false);
    for (auto i = size_t{0}; i < dy.element_count(); ++i) {
        dy.set<float32_t>({0, i / 4, (i % 4) / 2, i % 2}, static_cast<float32_t>(i + 1));
    }

    auto x_g = x.clone(true);
    auto dy_g = dy.clone(true);
    x_g.to(Device::GPU);
    dy_g.to(Device::GPU);

    auto y = x_g.avg_pool2d(static_cast<uint32_t>(kernel), static_cast<uint32_t>(stride));
    auto z = y.mul(dy_g);
    auto loss = z.sum();
    loss.backward(DType::Float32);
    x_g.sync();

    auto dx = x_g.gradient().clone(true);
    dx.to(Device::CPU);
    dx.sync();

    const auto dx_ref = ref_avg_pool2d_backward({N, C, H, W}, dy, kernel, stride);
    for (auto i = size_t{0}; i < dx_ref.size(); ++i) {
        const auto got = static_cast<float64_t>(dx.at<float32_t>({0, i / 16, (i % 16) / 4, i % 4}));
        CHECK(near64(got, static_cast<float64_t>(dx_ref[i]), 1e-5));
    }
}

static void test_half_forward_gpu_error() {
    if (!gpu_available()) {
        return;
    }
    auto x = Tensor({1, 1, 2, 2}, DType::Float16, Device::CPU, true);
    x.set_one();
    auto x_g = x.clone(true);
    x_g.to(Device::GPU);

    auto y = x_g.avg_pool2d(2, 2);
    auto threw = false;
    try {
        y.sync();
    } catch (const BeanTensor::ErrorHandling::NotImplemented&) {
        threw = true;
    }
    CHECK(threw);
}

static void test_half_backward_gpu_error() {
    if (!gpu_available()) {
        return;
    }
    auto x = Tensor({1, 1, 2, 2}, DType::Float32, Device::CPU, true);
    x.set_one();
    auto x_g = x.clone(true);
    x_g.to(Device::GPU);

    auto y = x_g.avg_pool2d(2, 2);
    auto threw = false;
    try {
        // A half-precision gradient dtype forces the GPU backward path to see
        // Float16 dy/dx, which is intentionally not implemented.
        y.backward(DType::Float16);
    } catch (const BeanTensor::ErrorHandling::NotImplemented&) {
        threw = true;
    }
    CHECK(threw);
}

int main() {
    test_forward_values();
    test_default_stride();
    test_floor_geometry();
    test_rectangular();
    test_multi_batch_channel();
    test_odd_sizes();
    test_overlapping_stride();
    test_backward_routing();
    test_backward_overlapping();
    test_non_unit_upstream();
    test_fan_out();
    test_repeated_backward();
    test_grad_dtype();
    test_integer_forward();
    test_half_forward();
    test_finite_difference();
    test_errors();
    test_module();
    test_serialization_roundtrip();
    test_serialization_malformed();
    test_conv_avgpool_linear_smoke();
    test_forward_values_gpu();
    test_backward_routing_gpu();
    test_half_forward_gpu_error();
    test_half_backward_gpu_error();

    if (failures == 0) {
        std::cout << "ALL TESTS PASSED\n";
        return 0;
    }
    std::cerr << failures << " test(s) failed\n";
    return 1;
}
