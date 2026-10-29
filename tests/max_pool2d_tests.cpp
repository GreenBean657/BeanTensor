/**
 * @file max_pool2d_tests.cpp
 * @author GreenBean
 * @brief Framework-free tests for MaxPool2d: forward values, floor geometry,
 * first-max tie routing, backward routing, error cases, and the module wrapper.
 * @version 0.2
 * @date 2026-10-28
 */
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include "Module/MaxPool2d.h"
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
using DType = BeanTensor::DataTypes::DType;

static int32_t failures = 0;
#define CHECK(...) do { \
    if (!(__VA_ARGS__)) { ++failures; std::cerr << "FAIL: " << #__VA_ARGS__ << " @ line " << __LINE__ << "\n"; } \
} while (0)

static bool near64(const float64_t a, const float64_t b, const float64_t eps) {
    return std::fabs(a - b) <= eps;
}

static bool eq32(const float32_t a, const float32_t b) {
    return std::fabs(a - b) <= 1e-6f;
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

static float32_t get(Tensor& t, const std::vector<size_t>& idx) {
    return t.at<float32_t>({idx[0], idx[1], idx[2], idx[3]});
}

static void set(Tensor& t, const std::vector<size_t>& idx, const float32_t v) {
    t.set<float32_t>({idx[0], idx[1], idx[2], idx[3]}, v);
}

static void test_forward_values() {
    auto x = Tensor({2, 3, 4, 4}, DType::Float32, Device::CPU);
    const auto xv = std::vector<float32_t>{
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
        -1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f, -9.0f, -10.0f, -11.0f, -12.0f, -13.0f, -14.0f, -15.0f, -16.0f,
        0.5f, 0.25f, 0.125f, 0.0625f, 1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f, 128.0f, 256.0f, 512.0f, 1024.0f, 2048.0f
    };
    for (auto i = size_t{0}; i < xv.size(); ++i) {
        x.set<float32_t>({i / 48, (i % 48) / 16, (i % 16) / 4, i % 4}, xv[i]);
    }

    auto y = x.max_pool2d(2, 2);
    CHECK(y.rank() == 4);
    CHECK(y.dim(0) == 2);
    CHECK(y.dim(1) == 3);
    CHECK(y.dim(2) == 2);
    CHECK(y.dim(3) == 2);

    // Manually verify first batch, first channel windows.
    CHECK(eq32(y.at<float32_t>({0, 0, 0, 0}), 6.0f));
    CHECK(eq32(y.at<float32_t>({0, 0, 0, 1}), 8.0f));
    CHECK(eq32(y.at<float32_t>({0, 0, 1, 0}), 14.0f));
    CHECK(eq32(y.at<float32_t>({0, 0, 1, 1}), 16.0f));
}

static void test_floor_geometry() {
    auto x = Tensor({1, 1, 5, 5}, DType::Float32, Device::CPU);
    x.set_zero();
    auto y = x.max_pool2d(2, 2);
    CHECK(y.dim(2) == 2);
    CHECK(y.dim(3) == 2);
}

static void test_first_max_tie() {
    // Single batch/channel with one 2x2 window containing two equal maxima.
    auto x = Tensor({1, 1, 2, 2}, DType::Float32, Device::CPU, true);
    x.set<float32_t>({0, 0, 0, 0}, 1.0f);
    x.set<float32_t>({0, 0, 0, 1}, 2.0f);
    x.set<float32_t>({0, 0, 1, 0}, 2.0f);
    x.set<float32_t>({0, 0, 1, 1}, 0.0f);

    auto y = x.max_pool2d(2, 2);
    CHECK(eq32(y.at<float32_t>({0, 0, 0, 0}), 2.0f));

    auto loss = y.sum();
    loss.backward(DType::Float32);
    CHECK(x.has_grad());
    // First max in row-major order is (0,1); gradient should land there only.
    CHECK(eq32(x.gradient().at<float32_t>({0, 0, 0, 0}), 0.0f));
    CHECK(eq32(x.gradient().at<float32_t>({0, 0, 0, 1}), 1.0f));
    CHECK(eq32(x.gradient().at<float32_t>({0, 0, 1, 0}), 0.0f));
    CHECK(eq32(x.gradient().at<float32_t>({0, 0, 1, 1}), 0.0f));
}

static void test_backward_routing() {
    const auto N = size_t{1};
    const auto C = size_t{2};
    const auto H = size_t{4};
    const auto W = size_t{4};
    const auto kernel = size_t{2};
    const auto stride = size_t{2};
    const auto Hout = (H - kernel) / stride + 1;
    const auto Wout = (W - kernel) / stride + 1;

    auto x = Tensor({N, C, H, W}, DType::Float32, Device::CPU, true);
    const auto xv = std::vector<float32_t>{
        0.5f, -1.0f, 2.0f, 0.25f,
        -0.5f, 1.5f, 0.0f, 1.0f,
        2.0f, -2.0f, 0.5f, -0.25f,
        0.0f, 1.0f, -1.0f, 2.0f,

        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };
    for (auto i = size_t{0}; i < xv.size(); ++i) {
        x.set<float32_t>({0, i / 16, (i % 16) / 4, i % 4}, xv[i]);
    }

    auto dy = Tensor({N, C, Hout, Wout}, DType::Float32, Device::CPU, false);
    const auto dyv = std::vector<float32_t>{
        1.0f, 2.0f,
        3.0f, 4.0f,

        -1.0f, -2.0f,
        -3.0f, -4.0f
    };
    for (auto i = size_t{0}; i < dyv.size(); ++i) {
        dy.set<float32_t>({0, i / 4, (i % 4) / 2, i % 2}, dyv[i]);
    }

    auto y = x.max_pool2d(static_cast<uint32_t>(kernel), static_cast<uint32_t>(stride));
    auto z = y.mul(dy);
    auto loss = z.sum();
    loss.backward(DType::Float32);
    CHECK(x.has_grad());

    // Naive argmax-routing reference.
    auto dx_ref = std::vector<float32_t>(N * C * H * W, 0.0f);
    for (auto n = size_t{0}; n < N; ++n) {
        for (auto c = size_t{0}; c < C; ++c) {
            for (auto oh = size_t{0}; oh < Hout; ++oh) {
                for (auto ow = size_t{0}; ow < Wout; ++ow) {
                    auto best_h = size_t{0};
                    auto best_w = size_t{0};
                    auto best_v = -std::numeric_limits<float32_t>::infinity();
                    for (auto kh = size_t{0}; kh < kernel; ++kh) {
                        for (auto kw = size_t{0}; kw < kernel; ++kw) {
                            const auto h = oh * stride + kh;
                            const auto w = ow * stride + kw;
                            const auto v = get(x, {n, c, h, w});
                            if (v > best_v) {
                                best_v = v;
                                best_h = h;
                                best_w = w;
                            }
                        }
                    }
                    dx_ref[((n * C + c) * H + best_h) * W + best_w] += get(dy, {n, c, oh, ow});
                }
            }
        }
    }

    for (auto i = size_t{0}; i < dx_ref.size(); ++i) {
        const auto expected = static_cast<float64_t>(dx_ref[i]);
        const auto got = static_cast<float64_t>(x.gradient().at<float32_t>({0, i / 16, (i % 16) / 4, i % 4}));
        CHECK(near64(got, expected, 1e-6));
    }
}

static void test_overlapping_stride() {
    // 2x2 windows with stride 1 overlap; the center element is the max of all four
    // windows and must accumulate the gradients from every window.
    auto x = Tensor({1, 1, 3, 3}, DType::Float32, Device::CPU, true);
    const auto xv = std::vector<float32_t>{1.0f, 2.0f, 3.0f, 4.0f, 9.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    for (auto i = size_t{0}; i < xv.size(); ++i) {
        x.set<float32_t>({0, 0, i / 3, i % 3}, xv[i]);
    }
    auto y = x.max_pool2d(2, 1);
    CHECK(y.dim(2) == 2);
    CHECK(y.dim(3) == 2);

    auto dy = Tensor({1, 1, 2, 2}, DType::Float32, Device::CPU, false);
    dy.set<float32_t>({0, 0, 0, 0}, 1.0f);
    dy.set<float32_t>({0, 0, 0, 1}, 2.0f);
    dy.set<float32_t>({0, 0, 1, 0}, 4.0f);
    dy.set<float32_t>({0, 0, 1, 1}, 8.0f);

    auto z = y.mul(dy);
    auto loss = z.sum();
    loss.backward(DType::Float32);
    CHECK(x.has_grad());
    CHECK(eq32(x.gradient().at<float32_t>({0, 0, 1, 1}), 15.0f));
    for (auto h = size_t{0}; h < 3; ++h) {
        for (auto w = size_t{0}; w < 3; ++w) {
            if (h == 1 && w == 1) {
                continue;
            }
            CHECK(eq32(x.gradient().at<float32_t>({0, 0, h, w}), 0.0f));
        }
    }
}

static void test_pool_size_partitioning() {
    // Large enough to engage launch_elementwise's threaded path on typical boxes
    // (output elements > thread_count * 1024).
    const auto H = size_t{160};
    const auto W = size_t{160};
    const auto kernel = size_t{2};
    const auto stride = size_t{2};
    const auto Hout = (H - kernel) / stride + 1;
    const auto Wout = (W - kernel) / stride + 1;
    auto x = Tensor({1, 1, H, W}, DType::Float32, Device::CPU, true);
    x.set_zero();
    for (auto oh = size_t{0}; oh < Hout; ++oh) {
        for (auto ow = size_t{0}; ow < Wout; ++ow) {
            x.set<float32_t>({0, 0, oh * stride, ow * stride}, static_cast<float32_t>(oh * Wout + ow + 1.0f));
        }
    }
    auto y = x.max_pool2d(static_cast<uint32_t>(kernel), static_cast<uint32_t>(stride));
    auto dy = Tensor({1, 1, Hout, Wout}, DType::Float32, Device::CPU, false);
    dy.set_one();
    auto z = y.mul(dy);
    auto loss = z.sum();
    loss.backward(DType::Float32);
    CHECK(x.has_grad());
    for (auto oh = size_t{0}; oh < Hout; ++oh) {
        for (auto ow = size_t{0}; ow < Wout; ++ow) {
            CHECK(eq32(x.gradient().at<float32_t>({0, 0, oh * stride, ow * stride}), 1.0f));
        }
    }
    // A few non-max positions should stay zero.
    CHECK(eq32(x.gradient().at<float32_t>({0, 0, 0, 1}), 0.0f));
    CHECK(eq32(x.gradient().at<float32_t>({0, 0, 1, 0}), 0.0f));
    CHECK(eq32(x.gradient().at<float32_t>({0, 0, H - 1, W - 1}), 0.0f));
}

static void test_forward_values_gpu() {
    if (!gpu_available()) {
        return;
    }
    auto x = Tensor({2, 3, 4, 4}, DType::Float32, Device::CPU);
    const auto xv = std::vector<float32_t>{
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
        -1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f, -9.0f, -10.0f, -11.0f, -12.0f, -13.0f, -14.0f, -15.0f, -16.0f,
        0.5f, 0.25f, 0.125f, 0.0625f, 1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f, 128.0f, 256.0f, 512.0f, 1024.0f, 2048.0f
    };
    for (auto i = size_t{0}; i < xv.size(); ++i) {
        x.set<float32_t>({i / 48, (i % 48) / 16, (i % 16) / 4, i % 4}, xv[i]);
    }
    auto x_g = x.clone(true);
    x_g.to(Device::GPU);
    auto y = x_g.max_pool2d(2, 2);
    y.sync();
    y.to(Device::CPU);
    y.sync();
    CHECK(y.rank() == 4);
    CHECK(y.dim(0) == 2);
    CHECK(y.dim(1) == 3);
    CHECK(y.dim(2) == 2);
    CHECK(y.dim(3) == 2);
    CHECK(eq32(y.at<float32_t>({0, 0, 0, 0}), 6.0f));
    CHECK(eq32(y.at<float32_t>({0, 0, 0, 1}), 8.0f));
    CHECK(eq32(y.at<float32_t>({0, 0, 1, 0}), 14.0f));
    CHECK(eq32(y.at<float32_t>({0, 0, 1, 1}), 16.0f));
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
    const auto Hout = (H - kernel) / stride + 1;
    const auto Wout = (W - kernel) / stride + 1;

    auto x = Tensor({N, C, H, W}, DType::Float32, Device::CPU, true);
    const auto xv = std::vector<float32_t>{
        0.5f, -1.0f, 2.0f, 0.25f,
        -0.5f, 1.5f, 0.0f, 1.0f,
        2.0f, -2.0f, 0.5f, -0.25f,
        0.0f, 1.0f, -1.0f, 2.0f,

        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };
    for (auto i = size_t{0}; i < xv.size(); ++i) {
        x.set<float32_t>({0, i / 16, (i % 16) / 4, i % 4}, xv[i]);
    }

    auto dy = Tensor({N, C, Hout, Wout}, DType::Float32, Device::CPU, false);
    const auto dyv = std::vector<float32_t>{
        1.0f, 2.0f,
        3.0f, 4.0f,

        -1.0f, -2.0f,
        -3.0f, -4.0f
    };
    for (auto i = size_t{0}; i < dyv.size(); ++i) {
        dy.set<float32_t>({0, i / 4, (i % 4) / 2, i % 2}, dyv[i]);
    }

    auto x_g = x.clone(true);
    auto dy_g = dy.clone(true);
    x_g.to(Device::GPU);
    dy_g.to(Device::GPU);

    auto y = x_g.max_pool2d(static_cast<uint32_t>(kernel), static_cast<uint32_t>(stride));
    auto z = y.mul(dy_g);
    auto loss = z.sum();
    loss.backward(DType::Float32);
    x_g.sync();

    auto dx = x_g.gradient().clone(true);
    dx.to(Device::CPU);
    dx.sync();

    auto dx_ref = std::vector<float32_t>(N * C * H * W, 0.0f);
    for (auto n = size_t{0}; n < N; ++n) {
        for (auto c = size_t{0}; c < C; ++c) {
            for (auto oh = size_t{0}; oh < Hout; ++oh) {
                for (auto ow = size_t{0}; ow < Wout; ++ow) {
                    auto best_h = size_t{0};
                    auto best_w = size_t{0};
                    auto best_v = -std::numeric_limits<float32_t>::infinity();
                    for (auto kh = size_t{0}; kh < kernel; ++kh) {
                        for (auto kw = size_t{0}; kw < kernel; ++kw) {
                            const auto h = oh * stride + kh;
                            const auto w = ow * stride + kw;
                            const auto v = get(x, {n, c, h, w});
                            if (v > best_v) {
                                best_v = v;
                                best_h = h;
                                best_w = w;
                            }
                        }
                    }
                    dx_ref[((n * C + c) * H + best_h) * W + best_w] += get(dy, {n, c, oh, ow});
                }
            }
        }
    }

    for (auto i = size_t{0}; i < dx_ref.size(); ++i) {
        const auto expected = static_cast<float64_t>(dx_ref[i]);
        const auto got = static_cast<float64_t>(dx.at<float32_t>({0, i / 16, (i % 16) / 4, i % 4}));
        CHECK(near64(got, expected, 1e-6));
    }
}

static void test_errors() {
    {
        auto x = Tensor({2, 3, 4}, DType::Float32, Device::CPU);
        auto threw = false;
        try {
            auto bad = x.max_pool2d(2, 2);
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
            auto bad = x.max_pool2d(5, 1);
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
            auto bad = x.max_pool2d(2, 0);
            static_cast<void>(bad);
        } catch (const BeanTensor::ErrorHandling::MismatchedShapes&) {
            threw = true;
        }
        CHECK(threw);
    }
}

static void test_module() {
    BeanTensor::Modules::MaxPool2d pool_default(2);
    BeanTensor::Modules::MaxPool2d pool_explicit(2, 2);

    CHECK(pool_default.name() == "MaxPool2d");
    CHECK(pool_explicit.name() == "MaxPool2d");

    auto x = Tensor({1, 1, 4, 4}, DType::Float32, Device::CPU);
    x.set_zero();
    auto y_default = pool_default.forward(x);
    auto y_explicit = pool_explicit.forward(x);
    CHECK(y_default.is_same_shape(y_explicit));
    CHECK(y_default.dim(2) == 2);
    CHECK(y_default.dim(3) == 2);
}

int main() {
    test_forward_values();
    test_floor_geometry();
    test_first_max_tie();
    test_backward_routing();
    test_overlapping_stride();
    test_pool_size_partitioning();
    test_forward_values_gpu();
    test_backward_routing_gpu();
    test_errors();
    test_module();

    if (failures == 0) {
        std::cout << "ALL TESTS PASSED\n";
        return 0;
    }
    std::cerr << failures << " test(s) failed\n";
    return 1;
}
