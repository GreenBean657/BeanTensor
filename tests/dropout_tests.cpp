// BeanTensorV5 dropout module tests — framework-free assertion suite.
// Covers: eval identity (exact and autograd passthrough), train p=0 identity,
// seed determinism, output statistics, gradient-vs-mask agreement, invalid
// probability validation, train/eval switching, and the integer-dtype train
// gate (eval stays legal).
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "Helpers/Errors.h"
#include "Math/Dropout/Dropout.h"
#include "Module/Dropout.h"
#include "Module/Module.h"
#include "Random/Generator.h"
#include "Tensor/Tensor.h"

#if defined(USE_HIP)
#include <hip/hip_runtime.h>
#elif defined(USE_CUDA)
#include <cuda_runtime.h>
#endif

using BeanTensor::Modules::Dropout;
using BeanTensor::Modules::Module;
using BeanTensor::Tensors::Tensor;
using BeanTensor::Tensors::Device;
using BeanTensor::float16_t;
using BeanTensor::bfloat16_t;
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

static bool near64(const float64_t a, const float64_t b, const float64_t eps) {
    return std::fabs(a - b) <= eps;
}

static void fill_ones(Tensor& t) {
    const auto numel = t.element_count();
    for (auto i = size_t{0}; i < numel; ++i) {
        switch (t.get_dtype()) {
            case DType::Float16:    t.set<float16_t>({i}, static_cast<float16_t>(0x3C00u)); break;
            case DType::BFloat16:   t.set<bfloat16_t>({i}, static_cast<bfloat16_t>(0x3F80u)); break;
            case DType::Float32:    t.set<float32_t>({i}, float32_t{1.0f}); break;
            case DType::Float64:    t.set<float64_t>({i}, float64_t{1.0}); break;
            case DType::Int32:      t.set<int32_t>({i}, int32_t{1}); break;
            default:                t.set<float32_t>({i}, float32_t{1.0f}); break;
        }
    }
}

static bool f32_exact_eq(const float32_t a, const float32_t b) {
    return std::bit_cast<uint32_t>(a) == std::bit_cast<uint32_t>(b);
}

static void test_eval_identity(const Device dev) {
    auto d = Dropout(float64_t{0.5});
    d.eval();
    auto x = Tensor({4}, DType::Float32, dev, false);
    for (auto i = size_t{0}; i < 4; ++i) {
        x.set<float32_t>({i}, static_cast<float32_t>(0.25 * static_cast<float64_t>(i) - 0.5));
    }
    auto y = d.forward(x);
    CHECK(y.is_same_shape(x));
    CHECK(y.get_dtype() == x.get_dtype());
    CHECK(y.get_device() == x.get_device());
    CHECK(&y != &x);
    for (auto i = size_t{0}; i < 4; ++i) {
        CHECK(f32_exact_eq(y.at<float32_t>({i}), x.at<float32_t>({i})));
    }
    CHECK(d.name() == "Dropout");
    CHECK(d.named_parameters().empty());
}

static void test_eval_grad(const Device dev) {
    auto d = Dropout(float64_t{0.5});
    d.eval();
    auto x = Tensor({6}, DType::Float32, dev, true);
    auto target = Tensor({6}, DType::Float32, dev, false);
    for (auto i = size_t{0}; i < 6; ++i) {
        x.set<float32_t>({i}, static_cast<float32_t>(0.5 * static_cast<float64_t>(i) - 1.5));
        target.set<float32_t>({i}, float32_t{0.0f});
    }
    auto y = d.forward(x);
    auto loss = BeanTensor::Modules::mse_loss(y, target);
    loss.backward(DType::Float32);
    loss.sync();
    for (auto i = size_t{0}; i < 6; ++i) {
        const auto dy = 2.0f * y.at<float32_t>({i}) / 6.0f;
        CHECK(near64(x.gradient().at<float32_t>({i}), static_cast<float64_t>(dy), 1e-6));
    }
}

static void test_train_p_zero(const Device dev) {
    auto d = Dropout(float64_t{0.0});
    auto x = Tensor({4}, DType::Float32, dev, false);
    for (auto i = size_t{0}; i < 4; ++i) {
        x.set<float32_t>({i}, static_cast<float32_t>(0.1 * static_cast<float64_t>(i) - 0.2));
    }
    auto y = d.forward(x);
    for (auto i = size_t{0}; i < 4; ++i) {
        CHECK(f32_exact_eq(y.at<float32_t>({i}), x.at<float32_t>({i})));
    }
}

static void test_train_determinism(const Device dev) {
    BeanTensor::Random::set_seed(123);
    auto x = Tensor({8}, DType::Float32, dev, false);
    for (auto i = size_t{0}; i < 8; ++i) {
        x.set<float32_t>({i}, static_cast<float32_t>(0.125 * static_cast<float64_t>(i)));
    }
    auto d1 = Dropout(float64_t{0.5});
    auto d2 = Dropout(float64_t{0.5});
    BeanTensor::Random::set_seed(123);
    auto y1 = d1.forward(x);
    BeanTensor::Random::set_seed(123);
    auto y2 = d2.forward(x);
    auto eq = y1.equal(y2);
    eq.sync();
    CHECK(eq.at<uint8_t>({0}) == uint8_t{1});
}

static void test_train_statistics(const Device dev) {
    BeanTensor::Random::set_seed(7);
    auto x = Tensor({4096}, DType::Float32, dev, false);
    fill_ones(x);
    auto d = Dropout(float64_t{0.25});
    auto y = d.forward(x);
    y.sync();
    auto zero_count = size_t{0};
    auto scale_sum = float64_t{0.0};
    for (auto i = size_t{0}; i < 4096; ++i) {
        const auto v = y.at<float32_t>({i});
        if (std::bit_cast<uint32_t>(v) == uint32_t{0}) {
            ++zero_count;
        } else {
            scale_sum += static_cast<float64_t>(v);
        }
    }
    const auto zero_frac = static_cast<float64_t>(zero_count) / 4096.0;
    CHECK(near64(zero_frac, 0.25, 0.02));
    const auto scale = 1.0 / (1.0 - 0.25);
    const auto avg_survivor = scale_sum / static_cast<float64_t>(4096 - zero_count);
    CHECK(near64(avg_survivor, scale, 1e-6));

    // Chunked-fill path (>1M elements) — the async-fill race lived here.
    if (dev == Device::CPU) {
        constexpr auto kBig = size_t{2'000'000};
        BeanTensor::Random::set_seed(13);
        auto xbig = Tensor({kBig}, DType::Float32, dev, false);
        fill_ones(xbig);
        auto dbig = Dropout(float64_t{0.25});
        auto ybig = dbig.forward(xbig);
        ybig.sync();
        auto big_zero_count = size_t{0};
        auto big_scale_sum = float64_t{0.0};
        for (auto i = size_t{0}; i < kBig; ++i) {
            const auto v = ybig.at<float32_t>({i});
            if (std::bit_cast<uint32_t>(v) == uint32_t{0}) {
                ++big_zero_count;
            } else {
                big_scale_sum += static_cast<float64_t>(v);
            }
        }
        const auto big_zero_frac = static_cast<float64_t>(big_zero_count) / static_cast<float64_t>(kBig);
        CHECK(near64(big_zero_frac, 0.25, 0.01));
        const auto big_avg_survivor = big_scale_sum / static_cast<float64_t>(kBig - big_zero_count);
        CHECK(near64(big_avg_survivor, scale, 1e-6));
    }
}

static void test_train_grad(const Device dev) {
    // p=0 path: mask is all ones, so dx = dy.
    {
        auto d = Dropout(float64_t{0.0});
        auto x = Tensor({4}, DType::Float32, dev, true);
        auto target = Tensor({4}, DType::Float32, dev, false);
        for (auto i = size_t{0}; i < 4; ++i) {
            x.set<float32_t>({i}, static_cast<float32_t>(0.5 * static_cast<float64_t>(i) - 1.0));
            target.set<float32_t>({i}, float32_t{0.0f});
        }
        auto y = d.forward(x);
        auto loss = BeanTensor::Modules::mse_loss(y, target);
        loss.backward(DType::Float32);
        loss.sync();
        for (auto i = size_t{0}; i < 4; ++i) {
            const auto dy = 2.0f * y.at<float32_t>({i}) / 4.0f;
            CHECK(near64(x.gradient().at<float32_t>({i}), static_cast<float64_t>(dy), 1e-6));
        }
    }
    // Fixed-seed free-function path: build y = x * train_mask(x, p, captured)
    // and verify backward gives dx = dy * mask (the same autograd path the
    // module uses, exercised directly with a captured seed).
    {
        BeanTensor::Random::set_seed(42);
        auto x = Tensor({8}, DType::Float32, dev, true);
        auto target = Tensor({8}, DType::Float32, dev, false);
        for (auto i = size_t{0}; i < 8; ++i) {
            x.set<float32_t>({i}, static_cast<float32_t>(0.25 * static_cast<float64_t>(i) - 1.0));
            target.set<float32_t>({i}, float32_t{0.0f});
        }
        const auto p = float64_t{0.5};
        const auto captured = static_cast<int64_t>(BeanTensor::Random::default_generator().draw_seed());
        auto mask = BeanTensor::Math::Dropout::train_mask(x, p, captured);
        auto y = x.mul(mask);
        auto loss = BeanTensor::Modules::mse_loss(y, target);
        loss.backward(DType::Float32);
        loss.sync();
        auto dy_tensor = y.mul(float64_t{2.0}).div(static_cast<float64_t>(y.element_count()));
        auto expected_dx = dy_tensor.mul(mask);
        expected_dx.sync();
        for (auto i = size_t{0}; i < 8; ++i) {
            const auto got = static_cast<float64_t>(x.gradient().at<float32_t>({i}));
            const auto want = static_cast<float64_t>(expected_dx.at<float32_t>({i}));
            CHECK(near64(got, want, 1e-5));
        }
    }
}

static void test_invalid_probability() {
    auto threw = false;
    try {
        auto d = Dropout(float64_t{-0.1});
        static_cast<void>(d);
    } catch (const BeanTensor::ErrorHandling::InvalidProbability&) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        auto d = Dropout(float64_t{1.0});
        static_cast<void>(d);
    } catch (const BeanTensor::ErrorHandling::InvalidProbability&) {
        threw = true;
    }
    CHECK(threw);
    // Boundary legality.
    auto d0 = Dropout(float64_t{0.0});
    auto d_just_under = Dropout(float64_t{0.9999999999});
    static_cast<void>(d0);
    static_cast<void>(d_just_under);
}

static void test_mid_loop_switch(const Device dev) {
    BeanTensor::Random::set_seed(99);
    auto x = Tensor({8}, DType::Float32, dev, false);
    for (auto i = size_t{0}; i < 8; ++i) {
        x.set<float32_t>({i}, static_cast<float32_t>(0.5 * static_cast<float64_t>(i) - 2.0));
    }
    auto d = Dropout(float64_t{0.5});
    auto train1 = d.forward(x);
    d.eval();
    auto eval1 = d.forward(x);
    d.train();
    auto train2 = d.forward(x);
    // eval is exact identity; train outputs are masked. Both train passes
    // must differ from the identity eval pass (and from x), confirming the
    // mode switch alternates masked/unmasked deterministically.
    for (auto i = size_t{0}; i < 8; ++i) {
        CHECK(f32_exact_eq(eval1.at<float32_t>({i}), x.at<float32_t>({i})));
    }
    auto eq_train1_eval = train1.equal(eval1);
    eq_train1_eval.sync();
    CHECK(eq_train1_eval.at<uint8_t>({0}) == uint8_t{0});
    auto eq_train2_eval = train2.equal(eval1);
    eq_train2_eval.sync();
    CHECK(eq_train2_eval.at<uint8_t>({0}) == uint8_t{0});
}

static void test_dtype_gate(const Device dev) {
    auto x = Tensor({4}, DType::Int32, dev, false);
    for (auto i = size_t{0}; i < 4; ++i) {
        x.set<int32_t>({i}, static_cast<int32_t>(i));
    }
    auto d = Dropout(float64_t{0.5});
    auto threw = false;
    try {
        auto y = d.forward(x);
        static_cast<void>(y);
    } catch (const BeanTensor::ErrorHandling::NotImplemented&) {
        threw = true;
    }
    CHECK(threw);
    d.eval();
    auto y = d.forward(x);
    for (auto i = size_t{0}; i < 4; ++i) {
        CHECK(y.at<int32_t>({i}) == x.at<int32_t>({i}));
    }
}

static void run_all(const Device dev) {
    test_eval_identity(dev);
    test_eval_grad(dev);
    test_train_p_zero(dev);
    test_train_determinism(dev);
    test_train_statistics(dev);
    test_train_grad(dev);
    test_invalid_probability();
    test_mid_loop_switch(dev);
    test_dtype_gate(dev);
}

int main() {
    run_all(Device::CPU);
    if (gpu_available()) {
        run_all(Device::GPU);
    }
    if (failures == 0) {
        std::cout << "ALL TESTS PASSED\n";
        return 0;
    }
    std::cerr << failures << " test(s) failed\n";
    return 1;
}
