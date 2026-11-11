/**
 * @file gelu_tests.cpp
 * @author GreenBean
 * @brief Framework-free tests for the exact erf-based GELU activation op,
 * its polynomial-erf SIMD tiers, its derivative backward node, the parameter-free
 * Modules::Gelu layer, and .bt serialization.  Covers CPU values, extreme-input
 * stability, analytic backward gradients, scalar/SIMD/GPU parity at the
 * activation family's documented tolerance, and the design §1.7 matrix.
 * @version 0.1
 * @date 2026-11-10
 */
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numbers>
#include <vector>

#include "BeanTensor.h"
#include "Intrinsics/Native/DTypeConversions.h"
#include "Math/Activations/Gelu/GeluKernels.h"
#include "Math/Autograd/GeluG/GeluGKernels.h"
#include "Module/Gelu.h"
#include "Module/Sequential.h"
#include "Tensor/Tensor.h"

#if defined(USE_HIP)
#include <hip/hip_runtime.h>
#elif defined(USE_CUDA)
#include <cuda_runtime.h>
#endif

using BeanTensor::Tensors::Tensor;
using BeanTensor::Tensors::Device;
using BeanTensor::float16_t;
using BeanTensor::bfloat16_t;
using BeanTensor::float16x2_t;
using BeanTensor::bfloat16x2_t;
using BeanTensor::float32_t;
using BeanTensor::float64_t;
using DType = BeanTensor::DataTypes::DType;

static int32_t failures = 0;
#define CHECK(...) do { \
    if (!(__VA_ARGS__)) { ++failures; std::cerr << "FAIL: " << #__VA_ARGS__ << " @ line " << __LINE__ << "\n"; } \
} while (0)

static bool near64(const float64_t a, const float64_t b, const float64_t eps) {
    if (std::isnan(a) && std::isnan(b)) { return true; }
    if (std::isinf(a) && std::isinf(b) && std::signbit(a) == std::signbit(b)) { return true; }
    return std::fabs(a - b) <= eps;
}

namespace idet = BeanTensor::Intrinsics::detail;

static float16_t narrow_half(const float32_t v) {
    return idet::unaccel_fp32_to_fp16_untracked(v);
}

static float32_t widen_half(const float16_t v) {
    return idet::unaccel_fp16_to_fp32_untracked(v);
}

static bfloat16_t narrow_bf16(const float32_t v) {
    return idet::unaccel_fp32_to_bf16_untracked(v);
}

static float32_t widen_bf16(const bfloat16_t v) {
    return idet::unaccel_bf16_to_fp32(v);
}

template<typename H>
static H narrow_lane(const float32_t v) {
    if constexpr (std::is_same_v<H, float16_t>) {
        return narrow_half(v);
    } else {
        return narrow_bf16(v);
    }
}

template<typename H>
static float32_t widen_lane(const H v) {
    if constexpr (std::is_same_v<H, float16_t>) {
        return widen_half(v);
    } else {
        return widen_bf16(v);
    }
}

template<typename P>
using packed_lane_t = std::conditional_t<std::is_same_v<P, float16x2_t>, float16_t, bfloat16_t>;

template<typename P>
static P pack_x2(const float32_t lo, const float32_t hi) {
    using H = packed_lane_t<P>;
    const auto lo_bits = idet::bit_cast_memcpy<uint16_t>(narrow_lane<H>(lo));
    const auto hi_bits = idet::bit_cast_memcpy<uint16_t>(narrow_lane<H>(hi));
    return idet::bit_cast_memcpy<P>(static_cast<uint32_t>(lo_bits) | (static_cast<uint32_t>(hi_bits) << 16));
}

template<typename P>
static float32_t lane_x2(const P v, const bool hi) {
    using H = packed_lane_t<P>;
    const auto bits = idet::bit_cast_memcpy<uint32_t>(v);
    const auto lane_bits = static_cast<uint16_t>(hi ? (bits >> 16) : (bits & 0xFFFFu));
    return widen_lane(idet::bit_cast_memcpy<H>(lane_bits));
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

static float64_t ref_gelu(const float64_t x) {
    if (std::isnan(x)) { return x; }
    if (std::isinf(x)) { return x > 0.0 ? x : 0.0; }
    return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}

static float64_t ref_gelu_deriv(const float64_t x) {
    if (std::isnan(x)) { return x; }
    if (std::isinf(x)) { return x > 0.0 ? 1.0 : 0.0; }
    const auto phi = std::exp(-0.5 * x * x) / std::sqrt(2.0 * std::numbers::pi);
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0))) + x * phi;
}

static void test_gelu_values() {
    auto x = Tensor({9}, DType::Float32, Device::CPU);
    const auto inputs = std::vector<float32_t>{
        0.0f,
        1.0f,
        -1.0f,
        2.0f,
        -2.0f,
        5.7f,
        -5.7f,
        6.0f,
        -6.0f
    };
    for (auto i = size_t{0}; i < inputs.size(); ++i) {
        x.set<float32_t>({i}, inputs[i]);
    }
    auto y = x.gelu();
    for (auto i = size_t{0}; i < inputs.size(); ++i) {
        const auto expected = ref_gelu(static_cast<float64_t>(inputs[i]));
        const auto got = static_cast<float64_t>(y.at<float32_t>({i}));
        CHECK(near64(got, expected, 1e-6));
    }
}

static void test_gelu_nonfinite() {
    auto x = Tensor({4}, DType::Float32, Device::CPU);
    x.set<float32_t>({0}, std::numeric_limits<float32_t>::quiet_NaN());
    x.set<float32_t>({1}, std::numeric_limits<float32_t>::infinity());
    x.set<float32_t>({2}, -std::numeric_limits<float32_t>::infinity());
    x.set<float32_t>({3}, -8.0f);

    auto y = x.gelu();
    CHECK(std::isnan(y.at<float32_t>({0})));
    CHECK(std::isinf(y.at<float32_t>({1})) && y.at<float32_t>({1}) > 0.0f);
    CHECK(std::bit_cast<uint32_t>(y.at<float32_t>({2})) == 0u);
    CHECK(std::bit_cast<uint32_t>(y.at<float32_t>({3})) == 0u);
}

static void test_gelu_signed_zero() {
    auto x = Tensor({2}, DType::Float32, Device::CPU);
    x.set<float32_t>({0}, 0.0f);
    x.set<float32_t>({1}, -0.0f);
    auto y = x.gelu();
    CHECK(std::bit_cast<uint32_t>(y.at<float32_t>({0})) == 0u);
    CHECK(std::bit_cast<uint32_t>(y.at<float32_t>({1})) == 0x80000000u);
}

static void test_gelu_integer_rejection() {
    auto x = Tensor({3}, DType::Int32, Device::CPU);
    x.set<int32_t>({0}, 1);
    x.set<int32_t>({1}, 0);
    x.set<int32_t>({2}, -1);
    auto threw = false;
    try {
        auto bad = x.gelu();
        static_cast<void>(bad);
    } catch (const BeanTensor::ErrorHandling::NotImplemented&) {
        threw = true;
    }
    CHECK(threw);
}

static void test_gelu_noncontiguous_rejection() {
    auto x = Tensor({2, 3}, DType::Float32, Device::CPU);
    for (auto i = size_t{0}; i < 6; ++i) {
        x.set<float32_t>({i / 3, i % 3}, static_cast<float32_t>(i));
    }
    auto y = x.slice({{0, 2}, {1, 3}}); // non-contiguous inner view
    auto threw = false;
    try {
        auto bad = y.gelu();
        static_cast<void>(bad);
    } catch (const BeanTensor::ErrorHandling::NotImplemented&) {
        threw = true;
    }
    CHECK(threw);
}

static void test_gelu_all_float_dtypes() {
    {
        const auto dtypes = std::vector<DType>{DType::Float32, DType::Float64, DType::Float16, DType::BFloat16};
        for (const auto dt : dtypes) {
            auto x = Tensor({3}, dt, Device::CPU);
            if (dt == DType::Float16) {
                x.set<float16_t>({0}, narrow_half(1.0f));
                x.set<float16_t>({1}, narrow_half(-1.0f));
                x.set<float16_t>({2}, narrow_half(2.0f));
            } else if (dt == DType::BFloat16) {
                x.set<bfloat16_t>({0}, narrow_bf16(1.0f));
                x.set<bfloat16_t>({1}, narrow_bf16(-1.0f));
                x.set<bfloat16_t>({2}, narrow_bf16(2.0f));
            } else if (dt == DType::Float64) {
                x.set<float64_t>({0}, 1.0);
                x.set<float64_t>({1}, -1.0);
                x.set<float64_t>({2}, 2.0);
            } else {
                x.set<float32_t>({0}, 1.0f);
                x.set<float32_t>({1}, -1.0f);
                x.set<float32_t>({2}, 2.0f);
            }
            auto y = x.gelu();
            for (auto i = size_t{0}; i < 3; ++i) {
                float64_t xi = 0.0;
                float64_t yi = 0.0;
                if (dt == DType::Float16) {
                    xi = static_cast<float64_t>(widen_half(x.at<float16_t>({i})));
                    yi = static_cast<float64_t>(widen_half(y.at<float16_t>({i})));
                } else if (dt == DType::BFloat16) {
                    xi = static_cast<float64_t>(widen_bf16(x.at<bfloat16_t>({i})));
                    yi = static_cast<float64_t>(widen_bf16(y.at<bfloat16_t>({i})));
                } else if (dt == DType::Float64) {
                    xi = x.at<float64_t>({i});
                    yi = y.at<float64_t>({i});
                } else {
                    xi = static_cast<float64_t>(x.at<float32_t>({i}));
                    yi = static_cast<float64_t>(y.at<float32_t>({i}));
                }
                CHECK(near64(yi, ref_gelu(xi), (dt == DType::Float32 || dt == DType::Float64) ? 1e-3 : 5e-3));
            }
        }
    }
    {
        const auto run_packed = []<typename P>(const DType dt) {
            auto x = Tensor({2}, dt, Device::CPU);
            x.set<P>({0}, pack_x2<P>(1.0f, -1.0f));
            x.set<P>({1}, pack_x2<P>(2.0f, 0.5f));
            auto y = x.gelu();
            for (auto i = size_t{0}; i < 4; ++i) {
                const auto xi = lane_x2(x.at<P>({i / 2}), i % 2 == 1);
                const auto expected = ref_gelu(static_cast<float64_t>(xi));
                const auto got = static_cast<float64_t>(lane_x2(y.at<P>({i / 2}), i % 2 == 1));
                CHECK(near64(got, expected, 5e-3));
            }
        };
        run_packed.template operator()<float16x2_t>(DType::Float16x2);
        run_packed.template operator()<bfloat16x2_t>(DType::BFloat16x2);
    }
}

static void test_gelu_odd_sizes() {
    for (const auto n : std::vector<size_t>{1, 7, 15, 33, 257, 1025}) {
        auto x = Tensor({n}, DType::Float32, Device::CPU);
        for (auto i = size_t{0}; i < n; ++i) {
            x.set<float32_t>({i}, static_cast<float32_t>(i % 13) - 6.0f);
        }
        auto y = x.gelu();
        for (auto i = size_t{0}; i < n; ++i) {
            const auto expected = ref_gelu(static_cast<float64_t>(x.at<float32_t>({i})));
            const auto got = static_cast<float64_t>(y.at<float32_t>({i}));
            CHECK(near64(got, expected, 1e-6));
        }
    }
}

static void test_gelu_finite_difference() {
    auto x = Tensor({7}, DType::Float64, Device::CPU, true);
    const auto inputs = std::vector<float64_t>{-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0};
    for (auto i = size_t{0}; i < inputs.size(); ++i) {
        x.set<float64_t>({i}, inputs[i]);
    }
    auto y = x.gelu();
    auto loss = y.sum();
    loss.backward(DType::Float64);
    CHECK(x.has_grad());

    const auto h = float64_t{1e-6};
    for (auto i = size_t{0}; i < inputs.size(); ++i) {
        const auto xi = inputs[i];
        const auto fd = (ref_gelu(xi + h) - ref_gelu(xi - h)) / (2.0 * h);
        const auto got = static_cast<float64_t>(x.gradient().at<float64_t>({i}));
        CHECK(near64(got, fd, 1e-5));
    }
}

static void test_gelu_backward() {
    auto x = Tensor({5}, DType::Float32, Device::CPU, true);
    const auto inputs = std::vector<float32_t>{-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    for (auto i = size_t{0}; i < inputs.size(); ++i) {
        x.set<float32_t>({i}, inputs[i]);
    }
    auto y = x.gelu();
    auto loss = y.sum();
    loss.backward(DType::Float32);
    CHECK(x.has_grad());

    for (auto i = size_t{0}; i < inputs.size(); ++i) {
        const auto expected = ref_gelu_deriv(static_cast<float64_t>(inputs[i]));
        const auto got = static_cast<float64_t>(x.gradient().at<float32_t>({i}));
        CHECK(near64(got, expected, 1e-5));
    }
}

static void test_gelu_backward_nonfinite() {
    using BeanTensor::Math::Activations::detail::gelu_deriv_exact_lane;
    using BeanTensor::Math::Autograd::gelu_g_avx2_available;
    using BeanTensor::Math::Autograd::gelu_g_avx512_available;
    using BeanTensor::Math::Autograd::gelu_g_f32_range_avx2;
    using BeanTensor::Math::Autograd::gelu_g_f64_range_avx2;
    using BeanTensor::Math::Autograd::gelu_g_f32_range_avx512;
    using BeanTensor::Math::Autograd::gelu_g_f64_range_avx512;

    const auto inf32 = std::numeric_limits<float32_t>::infinity();
    const auto nan32 = std::numeric_limits<float32_t>::quiet_NaN();
    const auto inf64 = std::numeric_limits<float64_t>::infinity();
    const auto nan64 = std::numeric_limits<float64_t>::quiet_NaN();

    CHECK(idet::bit_cast_memcpy<uint32_t>(gelu_deriv_exact_lane(inf32)) == idet::bit_cast_memcpy<uint32_t>(float32_t{1.0f}));
    CHECK(idet::bit_cast_memcpy<uint32_t>(gelu_deriv_exact_lane(-inf32)) == idet::bit_cast_memcpy<uint32_t>(float32_t{0.0f}));
    CHECK(std::isnan(gelu_deriv_exact_lane(nan32)));
    CHECK(idet::bit_cast_memcpy<uint64_t>(gelu_deriv_exact_lane(inf64)) == idet::bit_cast_memcpy<uint64_t>(float64_t{1.0}));
    CHECK(idet::bit_cast_memcpy<uint64_t>(gelu_deriv_exact_lane(-inf64)) == idet::bit_cast_memcpy<uint64_t>(float64_t{0.0}));
    CHECK(std::isnan(gelu_deriv_exact_lane(nan64)));

    const auto run_kernel = [](auto kernel, auto x0, auto expected) {
        using T = decltype(x0);
        using Bits = std::conditional_t<std::is_same_v<T, float32_t>, uint32_t, uint64_t>;
        std::vector<T> x(17, T{0});
        std::vector<T> dy(17, T{1});
        std::vector<T> dx(17, T{0});
        x[3] = x0;
        kernel(dx.data(), x.data(), dy.data(), 0, x.size());
        const auto got = dx[3];
        if (std::isnan(expected)) {
            CHECK(std::isnan(got));
        } else {
            CHECK(idet::bit_cast_memcpy<Bits>(got) == idet::bit_cast_memcpy<Bits>(expected));
        }
    };

    if (gelu_g_avx2_available()) {
        run_kernel(gelu_g_f32_range_avx2, inf32, float32_t{1.0f});
        run_kernel(gelu_g_f32_range_avx2, -inf32, float32_t{0.0f});
        run_kernel(gelu_g_f32_range_avx2, nan32, nan32);
        run_kernel(gelu_g_f64_range_avx2, inf64, float64_t{1.0});
        run_kernel(gelu_g_f64_range_avx2, -inf64, float64_t{0.0});
        run_kernel(gelu_g_f64_range_avx2, nan64, nan64);
    }
    if (gelu_g_avx512_available()) {
        run_kernel(gelu_g_f32_range_avx512, inf32, float32_t{1.0f});
        run_kernel(gelu_g_f32_range_avx512, -inf32, float32_t{0.0f});
        run_kernel(gelu_g_f32_range_avx512, nan32, nan32);
        run_kernel(gelu_g_f64_range_avx512, inf64, float64_t{1.0});
        run_kernel(gelu_g_f64_range_avx512, -inf64, float64_t{0.0});
        run_kernel(gelu_g_f64_range_avx512, nan64, nan64);
    }
}

static void test_gelu_backward_non_unit_upstream() {
    auto x = Tensor({4}, DType::Float32, Device::CPU, true);
    const auto inputs = std::vector<float32_t>{-1.5f, -0.5f, 0.5f, 1.5f};
    for (auto i = size_t{0}; i < inputs.size(); ++i) {
        x.set<float32_t>({i}, inputs[i]);
    }
    auto y = x.gelu();
    auto upstream = Tensor({4}, DType::Float32, Device::CPU, false);
    for (auto i = size_t{0}; i < 4; ++i) {
        upstream.set<float32_t>({i}, static_cast<float32_t>(i + 1));
    }
    auto z = y.mul(upstream);
    auto loss = z.sum();
    loss.backward(DType::Float32);
    CHECK(x.has_grad());

    for (auto i = size_t{0}; i < 4; ++i) {
        const auto dy = static_cast<float64_t>(upstream.at<float32_t>({i}));
        const auto expected = dy * ref_gelu_deriv(static_cast<float64_t>(inputs[i]));
        const auto got = static_cast<float64_t>(x.gradient().at<float32_t>({i}));
        CHECK(near64(got, expected, 1e-5));
    }
}

static void test_gelu_backward_chained() {
    auto x = Tensor({3}, DType::Float32, Device::CPU, true);
    x.set<float32_t>({0}, -1.0f);
    x.set<float32_t>({1}, 0.5f);
    x.set<float32_t>({2}, 1.5f);

    auto y = x.gelu();
    auto z = y.gelu();
    auto loss = z.sum();
    loss.backward(DType::Float32);
    CHECK(x.has_grad());

    for (auto i = size_t{0}; i < 3; ++i) {
        const auto xi = static_cast<float64_t>(x.at<float32_t>({i}));
        const auto yi = ref_gelu(xi);
        const auto expected = ref_gelu_deriv(yi) * ref_gelu_deriv(xi);
        const auto got = static_cast<float64_t>(x.gradient().at<float32_t>({i}));
        CHECK(near64(got, expected, 1e-4));
    }
}

static void test_gelu_backward_fanout() {
    auto x = Tensor({3}, DType::Float32, Device::CPU, true);
    x.set<float32_t>({0}, -1.0f);
    x.set<float32_t>({1}, 0.5f);
    x.set<float32_t>({2}, 1.5f);

    auto y1 = x.gelu();
    auto y2 = x.gelu();
    auto s = y1.add(y2);
    auto loss = s.sum();
    loss.backward(DType::Float32);
    CHECK(x.has_grad());

    for (auto i = size_t{0}; i < 3; ++i) {
        const auto expected = 2.0 * ref_gelu_deriv(static_cast<float64_t>(x.at<float32_t>({i})));
        const auto got = static_cast<float64_t>(x.gradient().at<float32_t>({i}));
        CHECK(near64(got, expected, 1e-5));
    }
}

static void test_gelu_backward_non_default_grad_dtype() {
    auto x = Tensor({3}, DType::Float32, Device::CPU, true);
    x.set<float32_t>({0}, -1.0f);
    x.set<float32_t>({1}, 0.5f);
    x.set<float32_t>({2}, 1.5f);

    auto y = x.gelu();
    auto loss = y.sum();
    loss.backward(DType::Float64);
    CHECK(x.has_grad());
    CHECK(x.gradient().get_dtype() == DType::Float64);

    for (auto i = size_t{0}; i < 3; ++i) {
        const auto expected = ref_gelu_deriv(static_cast<float64_t>(x.at<float32_t>({i})));
        const auto got = static_cast<float64_t>(x.gradient().at<float64_t>({i}));
        CHECK(near64(got, expected, 1e-5));
    }
}

static void test_gelu_module() {
    BeanTensor::Modules::Gelu mod;
    CHECK(mod.name() == "Gelu");

    auto x = Tensor({3}, DType::Float32, Device::CPU, true);
    x.set<float32_t>({0}, -1.0f);
    x.set<float32_t>({1}, 0.0f);
    x.set<float32_t>({2}, 1.0f);

    auto y = mod.forward(x);
    for (auto i = size_t{0}; i < 3; ++i) {
        const auto expected = ref_gelu(static_cast<float64_t>(x.at<float32_t>({i})));
        const auto got = static_cast<float64_t>(y.at<float32_t>({i}));
        CHECK(near64(got, expected, 1e-6));
    }

    auto loss = y.sum();
    loss.backward(DType::Float32);
    CHECK(x.has_grad());
}

static void test_gelu_sequential() {
    auto seq = BeanTensor::Modules::Sequential();
    seq.append("gelu1", std::make_unique<BeanTensor::Modules::Gelu>());
    seq.append("gelu2", std::make_unique<BeanTensor::Modules::Gelu>());

    auto x = Tensor({3}, DType::Float32, Device::CPU, true);
    x.set<float32_t>({0}, -1.0f);
    x.set<float32_t>({1}, 0.5f);
    x.set<float32_t>({2}, 1.5f);

    auto y = seq.forward(x);
    auto loss = y.sum();
    loss.backward(DType::Float32);
    CHECK(x.has_grad());
}

static void test_gelu_serialization_roundtrip() {
    auto mod = std::make_unique<BeanTensor::Modules::Gelu>();
    std::error_code ec;
    std::filesystem::create_directories("scratch", ec);
    const auto path = std::string{"scratch/gelu_test.bt"};
    BeanTensor::Serialization::save(*mod, path);
    auto loaded = BeanTensor::Serialization::load(path, Device::CPU);
    CHECK(loaded->name() == "Gelu");

    auto x = Tensor({3}, DType::Float32, Device::CPU);
    x.set<float32_t>({0}, -1.0f);
    x.set<float32_t>({1}, 0.5f);
    x.set<float32_t>({2}, 1.5f);

    auto y1 = mod->forward(x);
    auto y2 = loaded->forward(x);
    for (auto i = size_t{0}; i < 3; ++i) {
        CHECK(std::bit_cast<uint32_t>(y1.at<float32_t>({i})) == std::bit_cast<uint32_t>(y2.at<float32_t>({i})));
    }
}

static void test_gelu_malformed_spec() {
    auto spec = BeanTensor::Modules::ModuleSpec{};
    spec.type = "Gelu";
    spec.hyperparameters.push_back({"extra", "ignored"}); // Gelu ignores all hyperparameters
    auto mod = BeanTensor::Modules::Gelu::construct(spec, DType::Float32, Device::CPU);
    CHECK(mod->name() == "Gelu");
}

template<typename T, typename Bits>
static void check_gelu_kernel_tier(void (*kernel)(T*, const T*, size_t, size_t),
                                   void (*kernel_g)(T*, const T*, const T*, size_t, size_t)) {
    using BeanTensor::Math::Activations::detail::gelu_exact_lane;
    using BeanTensor::Math::Activations::detail::gelu_deriv_exact_lane;
    constexpr size_t n = 1037;
    std::vector<T> x(n);
    std::vector<T> dy(n);
    for (auto i = size_t{0}; i < n; ++i) {
        x[i] = static_cast<T>(std::sin(0.91 * static_cast<float64_t>(i)) * 3.0);
        dy[i] = static_cast<T>(std::cos(0.37 * static_cast<float64_t>(i)) * 1.5);
    }
    x[5] = std::numeric_limits<T>::quiet_NaN();
    x[6] = static_cast<T>(-0.0);
    dy[7] = static_cast<T>(-0.0);
    x[9] = T{0};
    dy[11] = std::numeric_limits<T>::quiet_NaN();

    auto bad = size_t{0};
    for (const auto& [b, e] : std::vector<std::pair<size_t, size_t>>{{0, n}, {3, n - 2}, {17, 30}, {0, 5}}) {
        auto y_simd = std::vector<T>(n, T{7});
        auto y_ref = std::vector<T>(n, T{7});
        auto dx_simd = std::vector<T>(n, T{7});
        auto dx_ref = std::vector<T>(n, T{7});
        kernel(y_simd.data(), x.data(), b, e);
        kernel_g(dx_simd.data(), x.data(), dy.data(), b, e);
        for (auto i = b; i < e; ++i) {
            y_ref[i] = gelu_exact_lane(x[i]);
            dx_ref[i] = dy[i] * gelu_deriv_exact_lane(x[i]);
        }
        for (auto i = size_t{0}; i < n; ++i) {
            if constexpr (std::is_same_v<T, float64_t>) {
                if (idet::bit_cast_memcpy<Bits>(y_simd[i]) != idet::bit_cast_memcpy<Bits>(y_ref[i])) { ++bad; }
                if (idet::bit_cast_memcpy<Bits>(dx_simd[i]) != idet::bit_cast_memcpy<Bits>(dx_ref[i])) { ++bad; }
            } else {
                if (!near64(static_cast<float64_t>(y_simd[i]), static_cast<float64_t>(y_ref[i]), 5e-7)) { ++bad; }
                if (!near64(static_cast<float64_t>(dx_simd[i]), static_cast<float64_t>(dx_ref[i]), 5e-7)) { ++bad; }
            }
        }
    }
    CHECK(bad == 0);
}

static void test_gelu_simd_parity() {
    using namespace BeanTensor::Math::Activations;
    using namespace BeanTensor::Math::Autograd;
    if (gelu_avx2_available() && gelu_g_avx2_available()) {
        check_gelu_kernel_tier<float32_t, uint32_t>(gelu_f32_range_avx2, gelu_g_f32_range_avx2);
        check_gelu_kernel_tier<float64_t, uint64_t>(gelu_f64_range_avx2, gelu_g_f64_range_avx2);
    } else {
        std::cout << "AVX2 not available — skipping AVX2 GELU parity\n";
    }
    if (gelu_avx512_available() && gelu_g_avx512_available()) {
        check_gelu_kernel_tier<float32_t, uint32_t>(gelu_f32_range_avx512, gelu_g_f32_range_avx512);
        check_gelu_kernel_tier<float64_t, uint64_t>(gelu_f64_range_avx512, gelu_g_f64_range_avx512);
    } else {
        std::cout << "AVX512F not available — skipping AVX512 GELU parity\n";
    }
}

static void test_gelu_values_gpu() {
    if (!gpu_available()) {
        return;
    }
    auto x = Tensor({9}, DType::Float32, Device::CPU);
    const auto inputs = std::vector<float32_t>{
        0.0f,
        1.0f,
        -1.0f,
        2.0f,
        -2.0f,
        5.7f,
        -5.7f,
        6.0f,
        -6.0f
    };
    for (auto i = size_t{0}; i < inputs.size(); ++i) {
        x.set<float32_t>({i}, inputs[i]);
    }
    auto x_g = x.clone(true);
    x_g.to(Device::GPU);
    auto y = x_g.gelu();
    y.sync();
    y.to(Device::CPU);
    y.sync();
    for (auto i = size_t{0}; i < inputs.size(); ++i) {
        const auto expected = ref_gelu(static_cast<float64_t>(inputs[i]));
        const auto got = static_cast<float64_t>(y.at<float32_t>({i}));
        CHECK(near64(got, expected, 1e-6));
    }
}

static void test_gelu_backward_gpu() {
    if (!gpu_available()) {
        return;
    }
    auto x = Tensor({5}, DType::Float32, Device::CPU, true);
    const auto inputs = std::vector<float32_t>{-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    for (auto i = size_t{0}; i < inputs.size(); ++i) {
        x.set<float32_t>({i}, inputs[i]);
    }
    auto x_g = x.clone(true);
    x_g.to(Device::GPU);
    auto y = x_g.gelu();
    auto loss = y.sum();
    loss.backward(DType::Float32);
    x_g.sync();
    x_g.to(Device::CPU);
    x_g.sync();

    for (auto i = size_t{0}; i < inputs.size(); ++i) {
        const auto expected = ref_gelu_deriv(static_cast<float64_t>(inputs[i]));
        const auto got = static_cast<float64_t>(x_g.gradient().at<float32_t>({i}));
        CHECK(near64(got, expected, 1e-5));
    }
}

static void test_gelu_parity_cpu_gpu() {
    if (!gpu_available()) {
        return;
    }
    const auto n = size_t{1025};
    auto x = Tensor({n}, DType::Float32, Device::CPU);
    for (auto i = size_t{0}; i < n; ++i) {
        x.set<float32_t>({i}, static_cast<float32_t>(i % 31) - 15.0f);
    }
    auto x_g = x.clone(true);
    x_g.to(Device::GPU);
    auto y_cpu = x.gelu();
    auto y_gpu = x_g.gelu();
    y_gpu.sync();
    y_gpu.to(Device::CPU);
    y_gpu.sync();
    for (auto i = size_t{0}; i < n; ++i) {
        const auto c = static_cast<float64_t>(y_cpu.at<float32_t>({i}));
        const auto g = static_cast<float64_t>(y_gpu.at<float32_t>({i}));
        CHECK(near64(c, g, 1e-5));
    }
}

int main() {
    test_gelu_values();
    test_gelu_nonfinite();
    test_gelu_signed_zero();
    test_gelu_integer_rejection();
    test_gelu_noncontiguous_rejection();
    test_gelu_all_float_dtypes();
    test_gelu_odd_sizes();
    test_gelu_finite_difference();
    test_gelu_backward();
    test_gelu_backward_nonfinite();
    test_gelu_backward_non_unit_upstream();
    test_gelu_backward_chained();
    test_gelu_backward_fanout();
    test_gelu_backward_non_default_grad_dtype();
    test_gelu_module();
    test_gelu_sequential();
    test_gelu_serialization_roundtrip();
    test_gelu_malformed_spec();
    test_gelu_simd_parity();
    test_gelu_values_gpu();
    test_gelu_backward_gpu();
    test_gelu_parity_cpu_gpu();

    if (failures == 0) {
        std::cout << "ALL TESTS PASSED\n";
        return 0;
    }
    std::cerr << failures << " test(s) failed\n";
    return 1;
}
