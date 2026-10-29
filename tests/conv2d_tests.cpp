/**
 * @file conv2d_tests.cpp
 * @author GreenBean
 * @brief Conv2d tests — framework-free assertion suite.
 * Covers forward values against a naive triple-loop reference (float64
 * accumulation, float32 tolerance), stride/padding/asymmetric/floor geometry,
 * the bias=false overload, shape/dtype/device validation errors, backward
 * dx/dW/db against finite differences, He fan-in initialization determinism
 * plus zero bias, and the Module registry/adam_step contract.
 * @version 0.2
 * @date 2026-10-28
 */
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include "Helpers/Errors.h"
#include "Module/Conv2d.h"
#include "Random/Generator.h"
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

static bool near64(const float32_t a, const float32_t b, const float64_t eps) {
    return std::fabs(static_cast<float64_t>(a) - static_cast<float64_t>(b)) <= eps;
}

static size_t numel_of(const std::vector<size_t>& shape) {
    auto n = size_t{1};
    for (const auto s : shape) {
        n *= s;
    }
    return n;
}

static std::vector<float64_t> ref_conv2d(
    const std::vector<float64_t>& x, const std::vector<size_t>& x_shape,
    const std::vector<float64_t>& w, const std::vector<size_t>& w_shape,
    const std::vector<float64_t>* bias,
    const uint32_t stride, const uint32_t padding) {

    const auto N = x_shape[0];
    const auto C = x_shape[1];
    const auto H = x_shape[2];
    const auto W = x_shape[3];
    const auto R = w_shape[1];
    const auto S = w_shape[2];
    const auto K = w_shape[3];
    const auto Hout = (H + 2u * padding - R) / stride + 1u;
    const auto Wout = (W + 2u * padding - S) / stride + 1u;

    std::vector<float64_t> out(N * K * Hout * Wout, 0.0);
    for (auto n = size_t{0}; n < N; ++n) {
        for (auto k = size_t{0}; k < K; ++k) {
            for (auto ho = size_t{0}; ho < Hout; ++ho) {
                for (auto wo = size_t{0}; wo < Wout; ++wo) {
                    auto acc = float64_t{0.0};
                    for (auto c = size_t{0}; c < C; ++c) {
                        for (auto r = size_t{0}; r < R; ++r) {
                            for (auto s = size_t{0}; s < S; ++s) {
                                const auto ih = static_cast<int64_t>(ho * stride + r) - static_cast<int64_t>(padding);
                                const auto iw = static_cast<int64_t>(wo * stride + s) - static_cast<int64_t>(padding);
                                if (
                                    ih >= 0 && ih < static_cast<int64_t>(H) &&
                                    iw >= 0 && iw < static_cast<int64_t>(W)
                                ) {
                                    const auto xi = ((n * C + c) * H + static_cast<size_t>(ih)) * W + static_cast<size_t>(iw);
                                    const auto wi = ((c * R + r) * S + s) * K + k;
                                    acc += x[xi] * w[wi];
                                }
                            }
                        }
                    }
                    if (bias != nullptr) {
                        acc += (*bias)[k];
                    }
                    const auto oi = ((n * K + k) * Hout + ho) * Wout + wo;
                    out[oi] = acc;
                }
            }
        }
    }
    return out;
}

static std::vector<size_t> tensor_shape(const Tensor& t) {
    std::vector<size_t> shape(t.rank());
    for (size_t i = 0; i < t.rank(); ++i) {
        shape[i] = t.dim(i);
    }
    return shape;
}

static std::vector<float64_t> tensor_to_fp64(Tensor& t) {
    const auto shape = tensor_shape(t);
    const auto n = numel_of(shape);
    std::vector<float64_t> out(n);
    if (shape.size() == 4) {
        for (auto i0 = size_t{0}; i0 < shape[0]; ++i0) {
            for (auto i1 = size_t{0}; i1 < shape[1]; ++i1) {
                for (auto i2 = size_t{0}; i2 < shape[2]; ++i2) {
                    for (auto i3 = size_t{0}; i3 < shape[3]; ++i3) {
                        const auto idx = ((i0 * shape[1] + i1) * shape[2] + i2) * shape[3] + i3;
                        out[idx] = static_cast<float64_t>(t.at<float32_t>({i0, i1, i2, i3}));
                    }
                }
            }
        }
    } else if (shape.size() == 1) {
        for (auto i = size_t{0}; i < n; ++i) {
            out[i] = static_cast<float64_t>(t.at<float32_t>({i}));
        }
    } else {
        for (auto i = size_t{0}; i < n; ++i) {
            out[i] = static_cast<float64_t>(t.at<float32_t>({i}));
        }
    }
    return out;
}

static void set_tensor_fp64(Tensor& t, const std::vector<float64_t>& vals) {
    const auto shape = tensor_shape(t);
    if (shape.size() == 4) {
        for (auto i0 = size_t{0}; i0 < shape[0]; ++i0) {
            for (auto i1 = size_t{0}; i1 < shape[1]; ++i1) {
                for (auto i2 = size_t{0}; i2 < shape[2]; ++i2) {
                    for (auto i3 = size_t{0}; i3 < shape[3]; ++i3) {
                        const auto idx = ((i0 * shape[1] + i1) * shape[2] + i2) * shape[3] + i3;
                        t.set<float32_t>({i0, i1, i2, i3}, static_cast<float32_t>(vals[idx]));
                    }
                }
            }
        }
    } else if (shape.size() == 1) {
        for (auto i = size_t{0}; i < vals.size(); ++i) {
            t.set<float32_t>({i}, static_cast<float32_t>(vals[i]));
        }
    }
    t.sync();
}

static bool check_forward_case(
    const std::vector<size_t>& x_shape,
    const std::vector<size_t>& w_shape,
    const uint32_t stride, const uint32_t padding,
    const bool use_bias) {

    auto x = Tensor(x_shape, DType::Float32, Device::CPU, false);
    x.random(-1.0, 1.0, 1234);
    x.sync();
    auto w = Tensor(w_shape, DType::Float32, Device::CPU, false);
    w.random(-1.0, 1.0, 5678);
    w.sync();

    std::vector<float64_t> b64;
    Tensor* bias_ptr = nullptr;
    std::optional<Tensor> bias_holder;
    if (use_bias) {
        bias_holder.emplace(Tensor(std::vector<size_t>{w_shape[3]}, DType::Float32, Device::CPU, false));
        bias_holder->random(-1.0, 1.0, 9012);
        bias_holder->sync();
        bias_ptr = &*bias_holder;
        b64 = tensor_to_fp64(*bias_ptr);
    }

    const auto x64 = tensor_to_fp64(x);
    const auto w64 = tensor_to_fp64(w);
    const auto ref = ref_conv2d(x64, x_shape, w64, w_shape, use_bias ? &b64 : nullptr, stride, padding);

    Tensor y = use_bias
        ? x.conv2d(w, *bias_ptr, stride, padding)
        : x.conv2d(w, stride, padding);
    y.sync();

    const auto y_shape = tensor_shape(y);
    if (y_shape.size() != 4) {
        return false;
    }
    const auto n = numel_of(y_shape);
    for (auto i = size_t{0}; i < n; ++i) {
        const auto yi = static_cast<float64_t>(y.at<float32_t>({i / (y_shape[1] * y_shape[2] * y_shape[3]),
            (i / (y_shape[2] * y_shape[3])) % y_shape[1],
            (i / y_shape[3]) % y_shape[2],
            i % y_shape[3]}));
        if (!near64(static_cast<float32_t>(yi), static_cast<float32_t>(ref[i]), 1e-4)) {
            return false;
        }
    }
    return true;
}

static bool test_forward_stride1_pad0() {
    return check_forward_case({2, 3, 5, 5}, {3, 3, 3, 4}, 1, 0, true);
}

static bool test_forward_stride2_pad1() {
    return check_forward_case({2, 3, 6, 4}, {3, 3, 3, 4}, 2, 1, true);
}

static bool test_forward_floor_geometry() {
    return check_forward_case({1, 1, 5, 5}, {1, 3, 3, 1}, 2, 0, true);
}

static bool test_forward_no_bias() {
    auto x = Tensor(std::vector<size_t>{2, 3, 5, 5}, DType::Float32, Device::CPU, false);
    x.random(-1.0, 1.0, 1234);
    x.sync();
    auto w = Tensor(std::vector<size_t>{3, 3, 3, 4}, DType::Float32, Device::CPU, false);
    w.random(-1.0, 1.0, 5678);
    w.sync();
    auto bias = Tensor(std::vector<size_t>{4}, DType::Float32, Device::CPU, false);
    bias.set_zero();
    bias.sync();

    auto y_with = x.conv2d(w, bias, 1, 0);
    auto y_without = x.conv2d(w, 1, 0);
    y_with.sync();
    y_without.sync();

    const auto shape = tensor_shape(y_with);
    const auto n = numel_of(shape);
    for (auto i = size_t{0}; i < n; ++i) {
        const std::initializer_list<size_t> idx = {
            i / (shape[1] * shape[2] * shape[3]),
            (i / (shape[2] * shape[3])) % shape[1],
            (i / shape[3]) % shape[2],
            i % shape[3]
        };
        const auto diff = std::fabs(static_cast<float64_t>(y_with.at<float32_t>(idx)) - static_cast<float64_t>(y_without.at<float32_t>(idx)));
        if (diff > 1e-6) {
            return false;
        }
    }
    return true;
}

static bool test_errors() {
    auto x4 = Tensor(std::vector<size_t>{1, 1, 4, 4}, DType::Float32, Device::CPU, false);
    auto x3 = Tensor(std::vector<size_t>{1, 4, 4}, DType::Float32, Device::CPU, false);
    auto w4 = Tensor(std::vector<size_t>{1, 2, 2, 1}, DType::Float32, Device::CPU, false);
    auto w3 = Tensor(std::vector<size_t>{1, 2, 2}, DType::Float32, Device::CPU, false);
    auto w_bad_c = Tensor(std::vector<size_t>{2, 2, 2, 1}, DType::Float32, Device::CPU, false);
    auto w_tall = Tensor(std::vector<size_t>{1, 5, 5, 1}, DType::Float32, Device::CPU, false);
    auto b = Tensor(std::vector<size_t>{1}, DType::Float32, Device::CPU, false);

    auto throws_mismatched = [](auto&& fn) {
        try {
            fn();
            return false;
        } catch (const BeanTensor::ErrorHandling::MismatchedShapes&) {
            return true;
        } catch (...) {
            return false;
        }
    };

    bool ok = true;
    ok &= throws_mismatched([&] { (void)x3.conv2d(w4, b, 1, 0); });
    ok &= throws_mismatched([&] { (void)x4.conv2d(w3, b, 1, 0); });
    ok &= throws_mismatched([&] { (void)x4.conv2d(w_bad_c, b, 1, 0); });
    ok &= throws_mismatched([&] { (void)x4.conv2d(w_tall, b, 1, 0); });
    ok &= throws_mismatched([&] { (void)x4.conv2d(w4, b, 0, 0); });
    return ok;
}

static float64_t compute_loss(Tensor& x, Tensor& w, Tensor& b) {
    auto y = x.conv2d(w, b, 1, 0);
    y.sync();
    auto target = Tensor(tensor_shape(y), y.get_dtype(), y.get_device(), false);
    target.set_zero();
    target.sync();
    auto loss = y.mse(target);
    loss.sync();
    return static_cast<float64_t>(loss.at<float32_t>({0}));
}

static bool test_backward_finite_diff() {
    auto x = Tensor(std::vector<size_t>{1, 1, 3, 3}, DType::Float32, Device::CPU, true);
    x.random(-0.5, 0.5, 1111);
    x.sync();
    auto w = Tensor(std::vector<size_t>{1, 2, 2, 1}, DType::Float32, Device::CPU, true);
    w.random(-0.5, 0.5, 2222);
    w.sync();
    auto b = Tensor(std::vector<size_t>{1}, DType::Float32, Device::CPU, true);
    b.random(-0.5, 0.5, 3333);
    b.sync();

    const auto loss0 = compute_loss(x, w, b);

    auto y = x.conv2d(w, b, 1, 0);
    auto target = Tensor(tensor_shape(y), y.get_dtype(), y.get_device(), false);
    target.set_zero();
    target.sync();
    auto loss = y.mse(target);
    loss.backward(DType::Float32);
    x.sync(); w.sync(); b.sync();

    const auto eps = float64_t{1e-3};

    auto x64 = tensor_to_fp64(x);
    for (auto i = size_t{0}; i < x64.size(); ++i) {
        auto x_perturb = Tensor(std::vector<size_t>{1, 1, 3, 3}, DType::Float32, Device::CPU, true);
        set_tensor_fp64(x_perturb, x64);
        const std::initializer_list<size_t> idx = {0, 0, i / 3, i % 3};
        const auto orig = static_cast<float64_t>(x_perturb.at<float32_t>(idx));
        x_perturb.set<float32_t>(idx, static_cast<float32_t>(orig + eps));
        x_perturb.sync();
        const auto loss_perturb = compute_loss(x_perturb, w, b);
        const auto numeric = (loss_perturb - loss0) / eps;
        const auto analytic = static_cast<float64_t>(x.gradient().at<float32_t>(idx));
        if (!near64(static_cast<float32_t>(numeric), static_cast<float32_t>(analytic), 1e-2)) {
            return false;
        }
    }

    auto w64 = tensor_to_fp64(w);
    for (auto i = size_t{0}; i < w64.size(); ++i) {
        auto w_perturb = Tensor(std::vector<size_t>{1, 2, 2, 1}, DType::Float32, Device::CPU, true);
        set_tensor_fp64(w_perturb, w64);
        const std::initializer_list<size_t> idx ={i / 4, (i / 2) % 2, i % 2, 0};
        const auto orig = static_cast<float64_t>(w_perturb.at<float32_t>(idx));
        w_perturb.set<float32_t>(idx, static_cast<float32_t>(orig + eps));
        w_perturb.sync();
        const auto loss_perturb = compute_loss(x, w_perturb, b);
        const auto numeric = (loss_perturb - loss0) / eps;
        const auto analytic = static_cast<float64_t>(w.gradient().at<float32_t>(idx));
        if (!near64(static_cast<float32_t>(numeric), static_cast<float32_t>(analytic), 1e-2)) {
            return false;
        }
    }

    auto b64 = tensor_to_fp64(b);
    const auto orig_b = b64[0];
    auto b_perturb = Tensor(std::vector<size_t>{1}, DType::Float32, Device::CPU, true);
    set_tensor_fp64(b_perturb, b64);
    b_perturb.set<float32_t>({0}, static_cast<float32_t>(orig_b + eps));
    b_perturb.sync();
    const auto loss_perturb = compute_loss(x, w, b_perturb);
    const auto numeric = (loss_perturb - loss0) / eps;
    const auto analytic = static_cast<float64_t>(b.gradient().at<float32_t>({0}));
    if (!near64(static_cast<float32_t>(numeric), static_cast<float32_t>(analytic), 1e-2)) {
        return false;
    }

    return true;
}

static bool test_init() {
    using BeanTensor::Random::set_seed;
    using BeanTensor::Random::default_generator;
    default_generator().seed(424242);
    auto a = BeanTensor::Modules::Conv2d(3, 4, 3, 1, 0, DType::Float32, Device::CPU, true, std::nullopt);
    default_generator().seed(424242);
    auto b = BeanTensor::Modules::Conv2d(3, 4, 3, 1, 0, DType::Float32, Device::CPU, true, std::nullopt);

    auto& wa = a.parameter("weight");
    auto& wb = b.parameter("weight");
    auto& ba = a.parameter("bias");
    auto& bb = b.parameter("bias");
    wa.sync(); wb.sync(); ba.sync(); bb.sync();

    const auto fan_in = static_cast<float64_t>(3 * 3 * 3);
    const auto bound = std::sqrt(6.0 / fan_in);

    const auto shape = tensor_shape(wa);
    const auto n = numel_of(shape);
    for (auto i = size_t{0}; i < n; ++i) {
        const std::initializer_list<size_t> idx = {
            i / (shape[1] * shape[2] * shape[3]),
            (i / (shape[2] * shape[3])) % shape[1],
            (i / shape[3]) % shape[2],
            i % shape[3]
        };
        const auto va = static_cast<float64_t>(wa.at<float32_t>(idx));
        const auto vb = static_cast<float64_t>(wb.at<float32_t>(idx));
        if (!near64(static_cast<float32_t>(va), static_cast<float32_t>(vb), 1e-7f)) {
            return false;
        }
        if (std::fabs(va) > bound) {
            return false;
        }
    }

    const auto bn = numel_of(tensor_shape(ba));
    for (auto i = size_t{0}; i < bn; ++i) {
        if (!near64(ba.at<float32_t>({i}), 0.0f, 1e-7f)) {
            return false;
        }
        if (!near64(bb.at<float32_t>({i}), 0.0f, 1e-7f)) {
            return false;
        }
    }
    return true;
}

static bool test_module_registry_and_adam() {
    auto conv = BeanTensor::Modules::Conv2d(2, 3, 3, 1, 1, DType::Float32, Device::CPU, true, 7777);
    const auto params = conv.named_parameters();
    if (params.size() != 2) {
        return false;
    }
    if (params[0].first != "weight" || params[1].first != "bias") {
        return false;
    }
    const auto w_shape = tensor_shape(*params[0].second);
    const auto b_shape = tensor_shape(*params[1].second);
    if (w_shape != std::vector<size_t>({2, 3, 3, 3})) {
        return false;
    }
    if (b_shape != std::vector<size_t>({3})) {
        return false;
    }

    auto x = Tensor(std::vector<size_t>{1, 2, 5, 5}, DType::Float32, Device::CPU, false);
    x.random(-0.5, 0.5, 8888);
    auto y = conv.forward(x);
    auto loss = y.mean();
    loss.backward(DType::Float32);
    conv.adam_step(0.01, 0.9, 0.999, 1e-8);
    return true;
}

static bool test_forward_gpu() {
    if (!gpu_available()) {
        return true;
    }
    const auto x_shape = std::vector<size_t>{2, 3, 5, 5};
    const auto w_shape = std::vector<size_t>{3, 3, 3, 4};
    auto x = Tensor(x_shape, DType::Float32, Device::CPU, false);
    x.random(-1.0, 1.0, 1234);
    x.sync();
    auto w = Tensor(w_shape, DType::Float32, Device::CPU, false);
    w.random(-1.0, 1.0, 5678);
    w.sync();
    auto b = Tensor(std::vector<size_t>{w_shape[3]}, DType::Float32, Device::CPU, false);
    b.random(-1.0, 1.0, 9012);
    b.sync();

    const auto x64 = tensor_to_fp64(x);
    const auto w64 = tensor_to_fp64(w);
    const auto b64 = tensor_to_fp64(b);
    const auto ref = ref_conv2d(x64, x_shape, w64, w_shape, &b64, 1, 0);

    auto x_g = x.clone(true);
    auto w_g = w.clone(true);
    auto b_g = b.clone(true);
    x_g.to(Device::GPU);
    w_g.to(Device::GPU);
    b_g.to(Device::GPU);

    auto y = x_g.conv2d(w_g, b_g, 1, 0);
    y.sync();
    y.to(Device::CPU);
    y.sync();

    const auto y_shape = tensor_shape(y);
    if (y_shape.size() != 4) {
        return false;
    }
    const auto n = numel_of(y_shape);
    for (auto i = size_t{0}; i < n; ++i) {
        const auto yi = static_cast<float64_t>(y.at<float32_t>({i / (y_shape[1] * y_shape[2] * y_shape[3]),
            (i / (y_shape[2] * y_shape[3])) % y_shape[1],
            (i / y_shape[3]) % y_shape[2],
            i % y_shape[3]}));
        if (!near64(static_cast<float32_t>(yi), static_cast<float32_t>(ref[i]), 1e-4)) {
            return false;
        }
    }
    return true;
}

static bool test_backward_gpu() {
    if (!gpu_available()) {
        return true;
    }
    auto x = Tensor(std::vector<size_t>{1, 1, 3, 3}, DType::Float32, Device::CPU, true);
    x.random(-0.5, 0.5, 1111);
    x.sync();
    auto w = Tensor(std::vector<size_t>{1, 2, 2, 1}, DType::Float32, Device::CPU, true);
    w.random(-0.5, 0.5, 2222);
    w.sync();
    auto b = Tensor(std::vector<size_t>{1}, DType::Float32, Device::CPU, true);
    b.random(-0.5, 0.5, 3333);
    b.sync();

    auto y_ref = x.conv2d(w, b, 1, 0);
    auto target_ref = Tensor(tensor_shape(y_ref), DType::Float32, Device::CPU, false);
    target_ref.set_zero();
    target_ref.sync();
    auto loss_ref = y_ref.mse(target_ref);
    loss_ref.backward(DType::Float32);
    x.sync(); w.sync(); b.sync();
    const auto dx_ref = tensor_to_fp64(x.gradient());
    const auto dw_ref = tensor_to_fp64(w.gradient());
    const auto db_ref = tensor_to_fp64(b.gradient());

    x.zero_grad();
    w.zero_grad();
    b.zero_grad();

    auto x_g = x.clone(true);
    auto w_g = w.clone(true);
    auto b_g = b.clone(true);
    x_g.to(Device::GPU);
    w_g.to(Device::GPU);
    b_g.to(Device::GPU);

    auto y = x_g.conv2d(w_g, b_g, 1, 0);
    auto target = Tensor(tensor_shape(y), DType::Float32, Device::GPU, false);
    target.set_zero();
    target.sync();
    auto loss = y.mse(target);
    loss.backward(DType::Float32);
    x_g.sync(); w_g.sync(); b_g.sync();

    auto dx = x_g.gradient().clone(true);
    auto dw = w_g.gradient().clone(true);
    auto db = b_g.gradient().clone(true);
    dx.to(Device::CPU); dx.sync();
    dw.to(Device::CPU); dw.sync();
    db.to(Device::CPU); db.sync();

    const auto dx_g = tensor_to_fp64(dx);
    const auto dw_g = tensor_to_fp64(dw);
    const auto db_g = tensor_to_fp64(db);

    for (auto i = size_t{0}; i < dx_ref.size(); ++i) {
        if (!near64(static_cast<float32_t>(dx_g[i]), static_cast<float32_t>(dx_ref[i]), 1e-4)) {
            return false;
        }
    }
    for (auto i = size_t{0}; i < dw_ref.size(); ++i) {
        if (!near64(static_cast<float32_t>(dw_g[i]), static_cast<float32_t>(dw_ref[i]), 1e-4)) {
            return false;
        }
    }
    for (auto i = size_t{0}; i < db_ref.size(); ++i) {
        if (!near64(static_cast<float32_t>(db_g[i]), static_cast<float32_t>(db_ref[i]), 1e-4)) {
            return false;
        }
    }
    return true;
}

int main() {
    CHECK(test_forward_stride1_pad0());
    CHECK(test_forward_stride2_pad1());
    CHECK(test_forward_floor_geometry());
    CHECK(test_forward_no_bias());
    CHECK(test_errors());
    CHECK(test_backward_finite_diff());
    CHECK(test_init());
    CHECK(test_module_registry_and_adam());
    CHECK(test_forward_gpu());
    CHECK(test_backward_gpu());

    if (failures == 0) {
        std::cout << "ALL TESTS PASSED\n";
    } else {
        std::cout << failures << " TEST(S) FAILED\n";
    }
    return failures == 0 ? 0 : 1;
}
