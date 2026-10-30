/**
 * @file batch_norm_tests.cpp
 * @author GreenBean
 * @brief Batch normalization tests — framework-free assertion suite.
 * Covers: train-mode forward against an in-test fp64 reference (rank 2 and
 * rank 4, fp32; fp64 tightly), the BIASED running-stat blend pinned by the
 * plan's Global Constraints, the batch-of-one train case (biased var 0,
 * invstd 1/sqrt(eps), running-var contribution exactly 0 — never 0/0 NaN),
 * running-stat updates with no-grad inputs, fp16/bf16 dtype-preserving
 * forward smoke, eval-mode forward against running statistics (used, not
 * batch statistics; buffers bit-untouched; ordered after a train forward
 * with no intervening sync), and the validation/error matrix (rank, operand
 * device agreement, channel count, contiguity of the input and every
 * operand, dtype, empty, eps, momentum), plus train-mode backward: analytic
 * dx/dgamma/dbeta against central finite differences on rank 2 and rank 4
 * fp64 inputs, and grad_dtype respected on fp32 inputs, and eval-mode
 * backward: dx analytically gamma * invstd * dy against the closed form with
 * the running statistics as graph constants, and dx/dgamma/dbeta against
 * central finite differences (dgamma = sum(dy * x_hat) with the
 * running-stat x_hat). The eval backward reads the running buffers through
 * Node::_saved — implicit lifetime parents that must outlive backward();
 * that contract is pinned by doc comments, not a runtime test (no code test
 * catches a dangling raw pointer reliably). Module coverage: the BatchNorm
 * module registers gamma/beta only (the running stats are address-stable
 * non-parameter buffers, excluded from named_parameters()/parameter_count()
 * and the optimizer), forwards in train and eval modes per the training
 * flag, keeps the eval backward's saved-tensor pointers valid across
 * module moves, and preserves the eval mode across move assignment
 * (Module::operator=(Module&&) carries the training flag). NaN
 * hyperparameters are rejected at both the Tensor::batch_norm entry point
 * and the module constructor (positive-form checks). GPU coverage (CUDA/HIP builds, Device::GPU tensors): train
 * forward rank 2/rank 4, running-stat update, eval-after-train ordering, and
 * fp16/bf16 smoke at the CPU suite's tolerances; train and eval backward
 * against central finite differences computed from DEVICE READBACK values in
 * fp64 host loops (numeric-vs-analytic on the same device, never CPU goldens
 * — float-reduction order is backend-defined), plus the eval dx closed form.
 * @version 0.8
 * @date 2026-10-30
 */
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "Helpers/Errors.h"
#include "Intrinsics/Native/DTypeConversions.h"
#include "Module/BatchNorm.h"
#include "Tensor/Tensor.h"

#if defined(USE_HIP)
#include <hip/hip_runtime.h>
#elif defined(USE_CUDA)
#include <cuda_runtime.h>
#endif

using BeanTensor::Tensors::Tensor;
using BeanTensor::Tensors::Device;
using BeanTensor::Modules::BatchNorm;
using BeanTensor::bfloat16_t;
using BeanTensor::float16_t;
using BeanTensor::float32_t;
using BeanTensor::float64_t;
using DType = BeanTensor::DataTypes::DType;
namespace idet = BeanTensor::Intrinsics::detail;

static int32_t failures = 0;
#define CHECK(...) do { \
    if (!(__VA_ARGS__)) { ++failures; std::cerr << "FAIL: " << #__VA_ARGS__ << " @ line " << __LINE__ << "\n"; } \
} while (0)

#define CHECK_THROWS(stmt, exc) \
    do { \
        auto thrown = false; \
        try { \
            static_cast<void>(stmt); \
        } catch (const exc&) { \
            thrown = true; \
        } \
        CHECK(thrown); \
    } while (0)

// Relative tolerance compare: |a-b| <= tol * max(1, |a|, |b|). Mixed-precision
// comparisons happen at float64 through explicit casts, so no call site relies
// on an implicit float->double promotion.
static bool rel_near(const float64_t a, const float64_t b, const float64_t tol) {
    const auto scale = std::max({float64_t{1.0}, std::fabs(a), std::fabs(b)});
    return std::fabs(a - b) <= tol * scale;
}

// --- half helpers (host-side, via the unaccel conversion pathway) ---

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

// Dtype-generic element write so the reference-fill code stays dtype-agnostic.
static void set_value(Tensor& t, const std::initializer_list<size_t> idx, const float64_t v) {
    switch (t.get_dtype()) {
        case DType::Float32: t.set<float32_t>(idx, static_cast<float32_t>(v)); break;
        case DType::Float64: t.set<float64_t>(idx, v); break;
        case DType::Float16: t.set<float16_t>(idx, narrow_half(static_cast<float32_t>(v))); break;
        case DType::BFloat16: t.set<bfloat16_t>(idx, narrow_bf16(static_cast<float32_t>(v))); break;
        default: break;
    }
}

// Deterministic exactly-representable fill values (quarters): identical in
// fp64 reference math and fp32/fp16/bf16 storage, so the in-test reference
// sees bit-identical inputs.
static float64_t sample_value(const size_t n, const size_t c, const size_t j) {
    return (static_cast<float64_t>(n) * 1.5) - (static_cast<float64_t>(c) * 0.75)
         + (static_cast<float64_t>(j) * 0.25) + 0.5;
}

static std::vector<float64_t> fill_rank2(Tensor& x, const size_t n_size, const size_t channels) {
    auto vals = std::vector<float64_t>(n_size * channels);
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            const auto v = sample_value(n, c, 0);
            vals[n * channels + c] = v;
            set_value(x, {n, c}, v);
        }
    }
    return vals;
}

static std::vector<float64_t> fill_rank4(Tensor& x, const size_t n_size, const size_t channels,
                                         const size_t height, const size_t width) {
    auto vals = std::vector<float64_t>(n_size * channels * height * width);
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            for (auto h = size_t{0}; h < height; ++h) {
                for (auto w = size_t{0}; w < width; ++w) {
                    const auto j = h * width + w;
                    const auto v = sample_value(n, c, j);
                    vals[(n * channels + c) * height * width + j] = v;
                    set_value(x, {n, c, h, w}, v);
                }
            }
        }
    }
    return vals;
}

// In-test fp64 train-mode reference. Independent two-pass biased variance
// (the kernel uses the sum/sumsq computational form — agreeing well inside
// the tolerances below), invstd = 1/sqrt(var + eps), y = x_hat*gamma + beta.
// Flat layout matches the kernel: i = n*(C*inner) + c*inner + j.
static void train_reference(const std::vector<float64_t>& x,
                            const size_t channels,
                            const size_t inner,
                            const std::vector<float64_t>& gamma,
                            const std::vector<float64_t>& beta,
                            const float64_t eps,
                            std::vector<float64_t>& expected_out,
                            std::vector<float64_t>& batch_mean,
                            std::vector<float64_t>& batch_var) {
    const auto numel = x.size();
    const auto m = static_cast<float64_t>(numel / channels);
    batch_mean.assign(channels, 0.0);
    for (auto i = size_t{0}; i < numel; ++i) {
        batch_mean[(i / inner) % channels] += x[i];
    }
    for (auto c = size_t{0}; c < channels; ++c) {
        batch_mean[c] /= m;
    }
    batch_var.assign(channels, 0.0);
    for (auto i = size_t{0}; i < numel; ++i) {
        const auto c = (i / inner) % channels;
        const auto d = x[i] - batch_mean[c];
        batch_var[c] += d * d;
    }
    for (auto c = size_t{0}; c < channels; ++c) {
        batch_var[c] /= m;
    }
    expected_out.resize(numel);
    for (auto i = size_t{0}; i < numel; ++i) {
        const auto c = (i / inner) % channels;
        const auto invstd = float64_t{1.0} / std::sqrt(batch_var[c] + eps);
        const auto x_hat = (x[i] - batch_mean[c]) * invstd;
        expected_out[i] = x_hat * gamma[c] + beta[c];
    }
}

static void ones_zeros_gamma_beta(Tensor& gamma, Tensor& beta, Tensor& running_mean, Tensor& running_var,
                                  const size_t channels) {
    for (auto c = size_t{0}; c < channels; ++c) {
        set_value(gamma, {c}, 1.0);
        set_value(beta, {c}, 0.0);
        set_value(running_mean, {c}, 0.0);
        set_value(running_var, {c}, 0.0);
    }
}

// In-test fp64 eval-mode reference: normalizes with the RUNNING statistics,
// never batch statistics — y = x_hat * gamma[c] + beta[c] with
// x_hat = (x - running_mean[c]) * invstd and invstd = 1/sqrt(running_var[c] + eps).
// Flat layout matches the kernel: i = n*(C*inner) + c*inner + j.
static void eval_reference(const std::vector<float64_t>& x,
                           const size_t channels,
                           const size_t inner,
                           const std::vector<float64_t>& gamma,
                           const std::vector<float64_t>& beta,
                           const std::vector<float64_t>& running_mean,
                           const std::vector<float64_t>& running_var,
                           const float64_t eps,
                           std::vector<float64_t>& expected_out) {
    const auto numel = x.size();
    expected_out.resize(numel);
    for (auto i = size_t{0}; i < numel; ++i) {
        const auto c = (i / inner) % channels;
        const auto invstd = float64_t{1.0} / std::sqrt(running_var[c] + eps);
        const auto x_hat = (x[i] - running_mean[c]) * invstd;
        expected_out[i] = x_hat * gamma[c] + beta[c];
    }
}

static void train_forward_rank2_matches_reference() {
    const auto n_size = size_t{4};
    const auto channels = size_t{3};
    auto x = Tensor({n_size, channels}, DType::Float32, Device::CPU);
    const auto vals = fill_rank2(x, n_size, channels);
    auto gamma = Tensor({channels}, DType::Float32, Device::CPU);
    auto beta = Tensor({channels}, DType::Float32, Device::CPU);
    auto running_mean = Tensor({channels}, DType::Float32, Device::CPU);
    auto running_var = Tensor({channels}, DType::Float32, Device::CPU);
    ones_zeros_gamma_beta(gamma, beta, running_mean, running_var, channels);
    const auto eps = float64_t{1e-5};
    auto y = x.batch_norm(gamma, beta, running_mean, running_var, true, 0.1, eps);
    CHECK(y.get_dtype() == DType::Float32);
    CHECK(y.is_same_shape(x));
    auto expected = std::vector<float64_t>{};
    auto batch_mean = std::vector<float64_t>{};
    auto batch_var = std::vector<float64_t>{};
    train_reference(vals, channels, 1, {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}, eps, expected, batch_mean, batch_var);
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            const auto got = static_cast<float64_t>(y.at<float32_t>({n, c}));
            CHECK(rel_near(got, expected[n * channels + c], 1e-4));
        }
    }
}

static void train_forward_rank4_matches_reference() {
    const auto n_size = size_t{2};
    const auto channels = size_t{3};
    const auto height = size_t{2};
    const auto width = size_t{2};
    auto x = Tensor({n_size, channels, height, width}, DType::Float32, Device::CPU);
    const auto vals = fill_rank4(x, n_size, channels, height, width);
    auto gamma = Tensor({channels}, DType::Float32, Device::CPU);
    auto beta = Tensor({channels}, DType::Float32, Device::CPU);
    auto running_mean = Tensor({channels}, DType::Float32, Device::CPU);
    auto running_var = Tensor({channels}, DType::Float32, Device::CPU);
    ones_zeros_gamma_beta(gamma, beta, running_mean, running_var, channels);
    const auto eps = float64_t{1e-5};
    auto y = x.batch_norm(gamma, beta, running_mean, running_var, true, 0.1, eps);
    CHECK(y.get_dtype() == DType::Float32);
    CHECK(y.is_same_shape(x));
    auto expected = std::vector<float64_t>{};
    auto batch_mean = std::vector<float64_t>{};
    auto batch_var = std::vector<float64_t>{};
    train_reference(vals, channels, height * width, {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}, eps,
                    expected, batch_mean, batch_var);
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            for (auto h = size_t{0}; h < height; ++h) {
                for (auto w = size_t{0}; w < width; ++w) {
                    const auto j = h * width + w;
                    const auto got = static_cast<float64_t>(y.at<float32_t>({n, c, h, w}));
                    CHECK(rel_near(got, expected[(n * channels + c) * height * width + j], 1e-4));
                }
            }
        }
    }
}

static void train_forward_fp64_matches_tightly() {
    const auto n_size = size_t{3};
    const auto channels = size_t{2};
    auto x = Tensor({n_size, channels}, DType::Float64, Device::CPU);
    const auto vals = fill_rank2(x, n_size, channels);
    auto gamma = Tensor({channels}, DType::Float64, Device::CPU);
    auto beta = Tensor({channels}, DType::Float64, Device::CPU);
    auto running_mean = Tensor({channels}, DType::Float64, Device::CPU);
    auto running_var = Tensor({channels}, DType::Float64, Device::CPU);
    ones_zeros_gamma_beta(gamma, beta, running_mean, running_var, channels);
    const auto eps = float64_t{1e-5};
    auto y = x.batch_norm(gamma, beta, running_mean, running_var, true, 0.1, eps);
    CHECK(y.get_dtype() == DType::Float64);
    auto expected = std::vector<float64_t>{};
    auto batch_mean = std::vector<float64_t>{};
    auto batch_var = std::vector<float64_t>{};
    train_reference(vals, channels, 1, {1.0, 1.0}, {0.0, 0.0}, eps, expected, batch_mean, batch_var);
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            const auto got = y.at<float64_t>({n, c});
            CHECK(rel_near(got, expected[n * channels + c], 1e-10));
        }
    }
}

static void running_stats_update_exact() {
    const auto n_size = size_t{4};
    const auto channels = size_t{3};
    auto x = Tensor({n_size, channels}, DType::Float32, Device::CPU);
    const auto vals = fill_rank2(x, n_size, channels);
    auto gamma = Tensor({channels}, DType::Float32, Device::CPU);
    auto beta = Tensor({channels}, DType::Float32, Device::CPU);
    auto running_mean = Tensor({channels}, DType::Float32, Device::CPU);
    auto running_var = Tensor({channels}, DType::Float32, Device::CPU);
    ones_zeros_gamma_beta(gamma, beta, running_mean, running_var, channels);
    const auto old_mean = std::vector<float64_t>{0.5, -1.0, 2.0};
    const auto old_var = std::vector<float64_t>{1.5, 0.5, 3.0};
    for (auto c = size_t{0}; c < channels; ++c) {
        set_value(running_mean, {c}, old_mean[c]);
        set_value(running_var, {c}, old_var[c]);
    }
    const auto momentum = float64_t{0.25};
    const auto eps = float64_t{1e-5};
    auto y = x.batch_norm(gamma, beta, running_mean, running_var, true, momentum, eps);
    static_cast<void>(y);
    auto expected = std::vector<float64_t>{};
    auto batch_mean = std::vector<float64_t>{};
    auto batch_var = std::vector<float64_t>{};
    train_reference(vals, channels, 1, {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}, eps, expected, batch_mean, batch_var);
    // Pinned blend: new = old * (1 - m) + batch * m, BIASED batch variance.
    for (auto c = size_t{0}; c < channels; ++c) {
        const auto want_mean = (old_mean[c] * (1.0 - momentum)) + (batch_mean[c] * momentum);
        const auto want_var = (old_var[c] * (1.0 - momentum)) + (batch_var[c] * momentum);
        CHECK(rel_near(static_cast<float64_t>(running_mean.at<float32_t>({c})), want_mean, 1e-5));
        CHECK(rel_near(static_cast<float64_t>(running_var.at<float32_t>({c})), want_var, 1e-5));
    }
}

static void batch_of_one_in_train_is_defined() {
    const auto channels = size_t{3};
    auto x = Tensor({1, channels}, DType::Float32, Device::CPU);
    const auto vals = fill_rank2(x, 1, channels);
    auto gamma = Tensor({channels}, DType::Float32, Device::CPU);
    auto beta = Tensor({channels}, DType::Float32, Device::CPU);
    auto running_mean = Tensor({channels}, DType::Float32, Device::CPU);
    auto running_var = Tensor({channels}, DType::Float32, Device::CPU);
    ones_zeros_gamma_beta(gamma, beta, running_mean, running_var, channels);
    const auto eps = float64_t{1e-5};
    auto y = x.batch_norm(gamma, beta, running_mean, running_var, true, 0.1, eps);
    for (auto c = size_t{0}; c < channels; ++c) {
        const auto got = static_cast<float64_t>(y.at<float32_t>({0, c}));
        // var = 0 (biased), invstd = 1/sqrt(0 + eps), out = (x - mean) * gamma / sqrt(0 + eps) + beta = 0.
        const auto invstd = float64_t{1.0} / std::sqrt(0.0 + eps);
        const auto want = ((vals[c] - vals[c]) * 1.0 * invstd) + 0.0;
        CHECK(std::isfinite(got));
        CHECK(rel_near(got, want, 1e-4));
        // Exactly +0.0: the pinned blend contributes 0*m = 0, never 0/0 NaN.
        CHECK(std::bit_cast<uint32_t>(y.at<float32_t>({0, c})) == 0u);
        CHECK(std::bit_cast<uint32_t>(running_var.at<float32_t>({c})) == 0u);
    }
}

static void no_grad_inputs_still_update_running_stats() {
    const auto n_size = size_t{4};
    const auto channels = size_t{3};
    auto x = Tensor({n_size, channels}, DType::Float32, Device::CPU, false);
    const auto vals = fill_rank2(x, n_size, channels);
    auto gamma = Tensor({channels}, DType::Float32, Device::CPU, false);
    auto beta = Tensor({channels}, DType::Float32, Device::CPU, false);
    auto running_mean = Tensor({channels}, DType::Float32, Device::CPU, false);
    auto running_var = Tensor({channels}, DType::Float32, Device::CPU, false);
    ones_zeros_gamma_beta(gamma, beta, running_mean, running_var, channels);
    const auto momentum = float64_t{0.1};
    const auto eps = float64_t{1e-5};
    auto y = x.batch_norm(gamma, beta, running_mean, running_var, true, momentum, eps);
    // No node is wired because every operand has requires_grad=false, so the
    // output has no gradient to access.
    CHECK_THROWS(y.gradient(), BeanTensor::ErrorHandling::NoGradient);
    auto expected = std::vector<float64_t>{};
    auto batch_mean = std::vector<float64_t>{};
    auto batch_var = std::vector<float64_t>{};
    train_reference(vals, channels, 1, {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}, eps, expected, batch_mean, batch_var);
    // Buffers started at zero: new = batch * m — they moved from the initial values.
    CHECK(std::bit_cast<uint32_t>(running_mean.at<float32_t>({0})) != 0u);
    for (auto c = size_t{0}; c < channels; ++c) {
        CHECK(rel_near(static_cast<float64_t>(running_mean.at<float32_t>({c})), batch_mean[c] * momentum, 1e-5));
        CHECK(rel_near(static_cast<float64_t>(running_var.at<float32_t>({c})), batch_var[c] * momentum, 1e-5));
    }
}

static void fp16_and_bf16_forward_smoke() {
    const auto n_size = size_t{4};
    const auto channels = size_t{3};
    const auto eps = float64_t{1e-5};
    {
        auto x = Tensor({n_size, channels}, DType::Float16, Device::CPU);
        const auto vals = fill_rank2(x, n_size, channels);
        auto gamma = Tensor({channels}, DType::Float16, Device::CPU);
        auto beta = Tensor({channels}, DType::Float16, Device::CPU);
        auto running_mean = Tensor({channels}, DType::Float16, Device::CPU);
        auto running_var = Tensor({channels}, DType::Float16, Device::CPU);
        ones_zeros_gamma_beta(gamma, beta, running_mean, running_var, channels);
        auto y = x.batch_norm(gamma, beta, running_mean, running_var, true, 0.1, eps);
        CHECK(y.get_dtype() == DType::Float16);
        auto expected = std::vector<float64_t>{};
        auto batch_mean = std::vector<float64_t>{};
        auto batch_var = std::vector<float64_t>{};
        train_reference(vals, channels, 1, {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}, eps, expected, batch_mean, batch_var);
        for (auto n = size_t{0}; n < n_size; ++n) {
            for (auto c = size_t{0}; c < channels; ++c) {
                const auto got = static_cast<float64_t>(widen_half(y.at<float16_t>({n, c})));
                CHECK(rel_near(got, expected[n * channels + c], 1e-2));
            }
        }
    }
    {
        auto x = Tensor({n_size, channels}, DType::BFloat16, Device::CPU);
        const auto vals = fill_rank2(x, n_size, channels);
        auto gamma = Tensor({channels}, DType::BFloat16, Device::CPU);
        auto beta = Tensor({channels}, DType::BFloat16, Device::CPU);
        auto running_mean = Tensor({channels}, DType::BFloat16, Device::CPU);
        auto running_var = Tensor({channels}, DType::BFloat16, Device::CPU);
        ones_zeros_gamma_beta(gamma, beta, running_mean, running_var, channels);
        auto y = x.batch_norm(gamma, beta, running_mean, running_var, true, 0.1, eps);
        CHECK(y.get_dtype() == DType::BFloat16);
        auto expected = std::vector<float64_t>{};
        auto batch_mean = std::vector<float64_t>{};
        auto batch_var = std::vector<float64_t>{};
        train_reference(vals, channels, 1, {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}, eps, expected, batch_mean, batch_var);
        for (auto n = size_t{0}; n < n_size; ++n) {
            for (auto c = size_t{0}; c < channels; ++c) {
                const auto got = static_cast<float64_t>(widen_bf16(y.at<bfloat16_t>({n, c})));
                CHECK(rel_near(got, expected[n * channels + c], 1e-2));
            }
        }
    }
}

static void eval_forward_uses_running_stats() {
    const auto n_size = size_t{4};
    const auto channels = size_t{3};
    auto x = Tensor({n_size, channels}, DType::Float32, Device::CPU);
    const auto vals = fill_rank2(x, n_size, channels);
    auto gamma = Tensor({channels}, DType::Float32, Device::CPU);
    auto beta = Tensor({channels}, DType::Float32, Device::CPU);
    auto running_mean = Tensor({channels}, DType::Float32, Device::CPU);
    auto running_var = Tensor({channels}, DType::Float32, Device::CPU);
    // Non-trivial running statistics, deliberately far from the batch
    // statistics of the deterministic fill (batch means are 2.75, 2.0, 1.25
    // per channel; see sample_value), so an eval forward that silently used
    // batch statistics would miss the reference by O(1).
    const auto rm_vals = std::vector<float64_t>{2.0, -1.5, 0.25};
    const auto rv_vals = std::vector<float64_t>{0.5, 3.0, 1.25};
    const auto g_vals = std::vector<float64_t>{1.5, -0.5, 2.0};
    const auto b_vals = std::vector<float64_t>{0.25, 1.0, -0.75};
    for (auto c = size_t{0}; c < channels; ++c) {
        set_value(gamma, {c}, g_vals[c]);
        set_value(beta, {c}, b_vals[c]);
        set_value(running_mean, {c}, rm_vals[c]);
        set_value(running_var, {c}, rv_vals[c]);
    }
    const auto eps = float64_t{1e-5};
    auto y = x.batch_norm(gamma, beta, running_mean, running_var, false, 0.1, eps);
    CHECK(y.get_dtype() == DType::Float32);
    CHECK(y.is_same_shape(x));
    auto expected = std::vector<float64_t>{};
    eval_reference(vals, channels, 1, g_vals, b_vals, rm_vals, rv_vals, eps, expected);
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            const auto got = static_cast<float64_t>(y.at<float32_t>({n, c}));
            CHECK(rel_near(got, expected[n * channels + c], 1e-4));
        }
    }
    // Negative control: the batch-statistic normalization must NOT match —
    // proves the output tracks the running statistics, not the batch.
    auto batch_mean = std::vector<float64_t>{};
    auto batch_var = std::vector<float64_t>{};
    auto train_expected = std::vector<float64_t>{};
    train_reference(vals, channels, 1, g_vals, b_vals, eps, train_expected, batch_mean, batch_var);
    const auto got00 = static_cast<float64_t>(y.at<float32_t>({0, 0}));
    CHECK(!rel_near(got00, train_expected[0], 1e-3));
}

static void eval_forward_leaves_running_stats_untouched() {
    const auto n_size = size_t{4};
    const auto channels = size_t{3};
    auto x = Tensor({n_size, channels}, DType::Float32, Device::CPU);
    const auto vals = fill_rank2(x, n_size, channels);
    auto gamma = Tensor({channels}, DType::Float32, Device::CPU);
    auto beta = Tensor({channels}, DType::Float32, Device::CPU);
    auto running_mean = Tensor({channels}, DType::Float32, Device::CPU);
    auto running_var = Tensor({channels}, DType::Float32, Device::CPU);
    const auto rm_vals = std::vector<float64_t>{0.5, -1.0, 2.0};
    const auto rv_vals = std::vector<float64_t>{1.5, 0.5, 3.0};
    for (auto c = size_t{0}; c < channels; ++c) {
        set_value(gamma, {c}, 1.0);
        set_value(beta, {c}, 0.0);
        set_value(running_mean, {c}, rm_vals[c]);
        set_value(running_var, {c}, rv_vals[c]);
    }
    // Bit-exact snapshot of both buffers before the forward: any store the
    // eval path made (blend or otherwise) flips these bits.
    auto mean_bits_before = std::vector<uint32_t>(channels);
    auto var_bits_before = std::vector<uint32_t>(channels);
    for (auto c = size_t{0}; c < channels; ++c) {
        mean_bits_before[c] = std::bit_cast<uint32_t>(running_mean.at<float32_t>({c}));
        var_bits_before[c] = std::bit_cast<uint32_t>(running_var.at<float32_t>({c}));
    }
    const auto eps = float64_t{1e-5};
    auto y = x.batch_norm(gamma, beta, running_mean, running_var, false, 0.1, eps);
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            CHECK(std::isfinite(static_cast<float64_t>(y.at<float32_t>({n, c}))));
        }
    }
    for (auto c = size_t{0}; c < channels; ++c) {
        CHECK(std::bit_cast<uint32_t>(running_mean.at<float32_t>({c})) == mean_bits_before[c]);
        CHECK(std::bit_cast<uint32_t>(running_var.at<float32_t>({c})) == var_bits_before[c]);
    }
}

static void eval_after_train_without_sync() {
    const auto n_size = size_t{4};
    const auto channels = size_t{3};
    auto x = Tensor({n_size, channels}, DType::Float32, Device::CPU);
    const auto vals = fill_rank2(x, n_size, channels);
    auto gamma = Tensor({channels}, DType::Float32, Device::CPU);
    auto beta = Tensor({channels}, DType::Float32, Device::CPU);
    auto running_mean = Tensor({channels}, DType::Float32, Device::CPU);
    auto running_var = Tensor({channels}, DType::Float32, Device::CPU);
    ones_zeros_gamma_beta(gamma, beta, running_mean, running_var, channels);
    const auto old_mean = std::vector<float64_t>{0.5, -1.0, 2.0};
    const auto old_var = std::vector<float64_t>{1.5, 0.5, 3.0};
    for (auto c = size_t{0}; c < channels; ++c) {
        set_value(running_mean, {c}, old_mean[c]);
        set_value(running_var, {c}, old_var[c]);
    }
    const auto momentum = float64_t{0.25};
    const auto eps = float64_t{1e-5};
    auto y_train = x.batch_norm(gamma, beta, running_mean, running_var, true, momentum, eps);
    static_cast<void>(y_train);
    // No sync of x, the output, or the running buffers in between: the eval
    // forward's entry drain of the buffers must order the train-mode buffer
    // mutation before the eval read, so this forward normalizes with the
    // POST-update running statistics.
    auto y_eval = x.batch_norm(gamma, beta, running_mean, running_var, false, momentum, eps);
    // Read back the post-update buffers (at() syncs) and build the eval
    // expectation from the stored values themselves.
    auto post_mean = std::vector<float64_t>(channels);
    auto post_var = std::vector<float64_t>(channels);
    for (auto c = size_t{0}; c < channels; ++c) {
        post_mean[c] = static_cast<float64_t>(running_mean.at<float32_t>({c}));
        post_var[c] = static_cast<float64_t>(running_var.at<float32_t>({c}));
    }
    auto expected = std::vector<float64_t>{};
    eval_reference(vals, channels, 1, {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}, post_mean, post_var, eps, expected);
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            const auto got = static_cast<float64_t>(y_eval.at<float32_t>({n, c}));
            CHECK(rel_near(got, expected[n * channels + c], 1e-4));
        }
    }
    // Negative control: an eval read that raced the train mutation (reading
    // the PRE-update buffers) would produce the old-statistic normalization
    // for element (0, 0), which is exactly 0 there ((x - old_mean) == 0 for
    // channel 0) — the post-update statistic gives a clearly nonzero value.
    const auto got00 = static_cast<float64_t>(y_eval.at<float32_t>({0, 0}));
    CHECK(!rel_near(got00, 0.0, 1e-3));
}

// In-test fp64 train-mode loss for the backward finite differences:
// L = sum_i(out[i] * w[i]) with out from the in-test forward reference, so the
// perturbed losses differentiate the same formulas the analytic node inverts.
static float64_t train_loss_reference(const std::vector<float64_t>& x,
                                      const size_t channels,
                                      const size_t inner,
                                      const std::vector<float64_t>& gamma,
                                      const std::vector<float64_t>& beta,
                                      const std::vector<float64_t>& w,
                                      const float64_t eps) {
    auto out = std::vector<float64_t>{};
    auto batch_mean = std::vector<float64_t>{};
    auto batch_var = std::vector<float64_t>{};
    train_reference(x, channels, inner, gamma, beta, eps, out, batch_mean, batch_var);
    auto loss = float64_t{0.0};
    for (auto i = size_t{0}; i < out.size(); ++i) {
        loss += out[i] * w[i];
    }
    return loss;
}

// In-test fp64 eval-mode loss for the backward finite differences:
// L = sum_i(out[i] * w[i]) with out from the in-test eval forward reference
// (running-stat normalization), so the perturbed losses differentiate the
// same formulas the analytic eval node inverts. The eval forward leaves the
// running buffers untouched, so repeated forward calls in the perturbation
// loop observe identical statistics.
static float64_t eval_loss_reference(const std::vector<float64_t>& x,
                                     const size_t channels,
                                     const size_t inner,
                                     const std::vector<float64_t>& gamma,
                                     const std::vector<float64_t>& beta,
                                     const std::vector<float64_t>& running_mean,
                                     const std::vector<float64_t>& running_var,
                                     const std::vector<float64_t>& w,
                                     const float64_t eps) {
    auto out = std::vector<float64_t>{};
    eval_reference(x, channels, inner, gamma, beta, running_mean, running_var, eps, out);
    auto loss = float64_t{0.0};
    for (auto i = size_t{0}; i < out.size(); ++i) {
        loss += out[i] * w[i];
    }
    return loss;
}

// Deterministic exactly-representable dy weights (halves/quarters in [-1, 1]).
static std::vector<float64_t> dy_weights(const size_t numel) {
    auto w = std::vector<float64_t>(numel);
    for (auto i = size_t{0}; i < numel; ++i) {
        w[i] = (static_cast<float64_t>(i % 5) - 2.0) * 0.5;
    }
    return w;
}

static void train_backward_matches_numeric_gradient_rank2() {
    const auto n_size = size_t{3};
    const auto channels = size_t{4};
    const auto eps = float64_t{1e-5};
    const auto h = float64_t{1e-5};
    auto x = Tensor({n_size, channels}, DType::Float64, Device::CPU, true);
    const auto x_vals = fill_rank2(x, n_size, channels);
    auto gamma = Tensor({channels}, DType::Float64, Device::CPU, true);
    auto beta = Tensor({channels}, DType::Float64, Device::CPU, true);
    auto running_mean = Tensor({channels}, DType::Float64, Device::CPU);
    auto running_var = Tensor({channels}, DType::Float64, Device::CPU);
    auto w = Tensor({n_size, channels}, DType::Float64, Device::CPU);
    const auto g_vals = std::vector<float64_t>{1.5, -0.5, 0.75, 2.0};
    const auto b_vals = std::vector<float64_t>{0.25, -1.0, 0.5, -0.25};
    const auto w_vals = dy_weights(n_size * channels);
    for (auto c = size_t{0}; c < channels; ++c) {
        set_value(gamma, {c}, g_vals[c]);
        set_value(beta, {c}, b_vals[c]);
        set_value(running_mean, {c}, 0.0);
        set_value(running_var, {c}, 0.0);
    }
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            set_value(w, {n, c}, w_vals[n * channels + c]);
        }
    }
    auto out = x.batch_norm(gamma, beta, running_mean, running_var, true, 0.1, eps);
    // The mul result is named: graph parents are raw pointers, so every
    // tensor in the graph must outlive backward().
    auto weighted = out.mul(w);
    auto loss = weighted.sum();
    loss.backward(DType::Float64);
    // dx: central differences of the in-test fp64 reference loss.
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            const auto i = n * channels + c;
            auto xp = x_vals;
            xp[i] += h;
            auto xm = x_vals;
            xm[i] -= h;
            const auto numeric = (train_loss_reference(xp, channels, 1, g_vals, b_vals, w_vals, eps)
                                - train_loss_reference(xm, channels, 1, g_vals, b_vals, w_vals, eps)) / (2.0 * h);
            CHECK(rel_near(x.gradient().at<float64_t>({n, c}), numeric, 1e-6));
        }
    }
    // dgamma and dbeta: per-channel parameter perturbations.
    for (auto c = size_t{0}; c < channels; ++c) {
        auto gp = g_vals;
        gp[c] += h;
        auto gm = g_vals;
        gm[c] -= h;
        const auto numeric_g = (train_loss_reference(x_vals, channels, 1, gp, b_vals, w_vals, eps)
                              - train_loss_reference(x_vals, channels, 1, gm, b_vals, w_vals, eps)) / (2.0 * h);
        CHECK(rel_near(gamma.gradient().at<float64_t>({c}), numeric_g, 1e-6));
        auto bp = b_vals;
        bp[c] += h;
        auto bm = b_vals;
        bm[c] -= h;
        const auto numeric_b = (train_loss_reference(x_vals, channels, 1, g_vals, bp, w_vals, eps)
                              - train_loss_reference(x_vals, channels, 1, g_vals, bm, w_vals, eps)) / (2.0 * h);
        CHECK(rel_near(beta.gradient().at<float64_t>({c}), numeric_b, 1e-6));
    }
}

static void train_backward_matches_numeric_gradient_rank4() {
    const auto n_size = size_t{2};
    const auto channels = size_t{2};
    const auto height = size_t{2};
    const auto width = size_t{2};
    const auto inner = height * width;
    const auto eps = float64_t{1e-5};
    const auto h = float64_t{1e-5};
    auto x = Tensor({n_size, channels, height, width}, DType::Float64, Device::CPU, true);
    const auto x_vals = fill_rank4(x, n_size, channels, height, width);
    auto gamma = Tensor({channels}, DType::Float64, Device::CPU, true);
    auto beta = Tensor({channels}, DType::Float64, Device::CPU, true);
    auto running_mean = Tensor({channels}, DType::Float64, Device::CPU);
    auto running_var = Tensor({channels}, DType::Float64, Device::CPU);
    auto w = Tensor({n_size, channels, height, width}, DType::Float64, Device::CPU);
    const auto g_vals = std::vector<float64_t>{1.5, -0.5};
    const auto b_vals = std::vector<float64_t>{0.25, -1.0};
    const auto w_vals = dy_weights(n_size * channels * height * width);
    for (auto c = size_t{0}; c < channels; ++c) {
        set_value(gamma, {c}, g_vals[c]);
        set_value(beta, {c}, b_vals[c]);
        set_value(running_mean, {c}, 0.0);
        set_value(running_var, {c}, 0.0);
    }
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            for (auto hh = size_t{0}; hh < height; ++hh) {
                for (auto ww = size_t{0}; ww < width; ++ww) {
                    const auto j = hh * width + ww;
                    set_value(w, {n, c, hh, ww}, w_vals[(n * channels + c) * inner + j]);
                }
            }
        }
    }
    auto out = x.batch_norm(gamma, beta, running_mean, running_var, true, 0.1, eps);
    // The mul result is named: graph parents are raw pointers, so every
    // tensor in the graph must outlive backward().
    auto weighted = out.mul(w);
    auto loss = weighted.sum();
    loss.backward(DType::Float64);
    // dx: central differences of the in-test fp64 reference loss.
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            for (auto hh = size_t{0}; hh < height; ++hh) {
                for (auto ww = size_t{0}; ww < width; ++ww) {
                    const auto i = (n * channels + c) * inner + hh * width + ww;
                    auto xp = x_vals;
                    xp[i] += h;
                    auto xm = x_vals;
                    xm[i] -= h;
                    const auto numeric = (train_loss_reference(xp, channels, inner, g_vals, b_vals, w_vals, eps)
                                        - train_loss_reference(xm, channels, inner, g_vals, b_vals, w_vals, eps)) / (2.0 * h);
                    CHECK(rel_near(x.gradient().at<float64_t>({n, c, hh, ww}), numeric, 1e-6));
                }
            }
        }
    }
    // dgamma and dbeta: per-channel parameter perturbations.
    for (auto c = size_t{0}; c < channels; ++c) {
        auto gp = g_vals;
        gp[c] += h;
        auto gm = g_vals;
        gm[c] -= h;
        const auto numeric_g = (train_loss_reference(x_vals, channels, inner, gp, b_vals, w_vals, eps)
                              - train_loss_reference(x_vals, channels, inner, gm, b_vals, w_vals, eps)) / (2.0 * h);
        CHECK(rel_near(gamma.gradient().at<float64_t>({c}), numeric_g, 1e-6));
        auto bp = b_vals;
        bp[c] += h;
        auto bm = b_vals;
        bm[c] -= h;
        const auto numeric_b = (train_loss_reference(x_vals, channels, inner, g_vals, bp, w_vals, eps)
                              - train_loss_reference(x_vals, channels, inner, g_vals, bm, w_vals, eps)) / (2.0 * h);
        CHECK(rel_near(beta.gradient().at<float64_t>({c}), numeric_b, 1e-6));
    }
}

static void train_backward_respects_grad_dtype() {
    const auto n_size = size_t{3};
    const auto channels = size_t{4};
    const auto eps = float64_t{1e-5};
    // fp32 loss quantization dominates finite differences below ~1e-3, so h is
    // wider than in the fp64 tests; the loss is accumulated in fp64 from fp32
    // readbacks of the framework forward itself (the function actually
    // differentiated), never from the fp64 reference.
    const auto h = float32_t{1e-2f};
    auto x = Tensor({n_size, channels}, DType::Float32, Device::CPU, true);
    const auto x_vals = fill_rank2(x, n_size, channels);
    auto gamma = Tensor({channels}, DType::Float32, Device::CPU, true);
    auto beta = Tensor({channels}, DType::Float32, Device::CPU, true);
    auto running_mean = Tensor({channels}, DType::Float32, Device::CPU);
    auto running_var = Tensor({channels}, DType::Float32, Device::CPU);
    auto w = Tensor({n_size, channels}, DType::Float32, Device::CPU);
    const auto g_vals = std::vector<float64_t>{1.5, -0.5, 0.75, 2.0};
    const auto b_vals = std::vector<float64_t>{0.25, -1.0, 0.5, -0.25};
    const auto w_vals = dy_weights(n_size * channels);
    for (auto c = size_t{0}; c < channels; ++c) {
        set_value(gamma, {c}, g_vals[c]);
        set_value(beta, {c}, b_vals[c]);
        set_value(running_mean, {c}, 0.0);
        set_value(running_var, {c}, 0.0);
    }
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            set_value(w, {n, c}, w_vals[n * channels + c]);
        }
    }
    auto out = x.batch_norm(gamma, beta, running_mean, running_var, true, 0.1, eps);
    // The mul result is named: graph parents are raw pointers, so every
    // tensor in the graph must outlive backward().
    auto weighted = out.mul(w);
    auto loss = weighted.sum();
    loss.backward(DType::Float32);
    // The assigned grad dtype pins every leaf grad buffer.
    CHECK(x.gradient().get_dtype() == DType::Float32);
    CHECK(gamma.gradient().get_dtype() == DType::Float32);
    CHECK(beta.gradient().get_dtype() == DType::Float32);
    // fp32 forward loss, read back and accumulated in fp64. Re-running
    // batch_norm re-blends the running buffers, which the loss never reads.
    auto fp32_loss = [&]() {
        auto o = x.batch_norm(gamma, beta, running_mean, running_var, true, 0.1, eps);
        auto total = float64_t{0.0};
        for (auto n = size_t{0}; n < n_size; ++n) {
            for (auto c = size_t{0}; c < channels; ++c) {
                total += static_cast<float64_t>(o.at<float32_t>({n, c})) * w_vals[n * channels + c];
            }
        }
        return total;
    };
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            const auto orig = x.at<float32_t>({n, c});
            x.set<float32_t>({n, c}, orig + h);
            const auto up = fp32_loss();
            x.set<float32_t>({n, c}, orig - h);
            const auto down = fp32_loss();
            x.set<float32_t>({n, c}, orig);
            const auto numeric = (up - down) / (2.0 * static_cast<float64_t>(h));
            CHECK(rel_near(static_cast<float64_t>(x.gradient().at<float32_t>({n, c})), numeric, 1e-3));
        }
    }
    for (auto c = size_t{0}; c < channels; ++c) {
        const auto orig_g = gamma.at<float32_t>({c});
        gamma.set<float32_t>({c}, orig_g + h);
        const auto up_g = fp32_loss();
        gamma.set<float32_t>({c}, orig_g - h);
        const auto down_g = fp32_loss();
        gamma.set<float32_t>({c}, orig_g);
        const auto numeric_g = (up_g - down_g) / (2.0 * static_cast<float64_t>(h));
        CHECK(rel_near(static_cast<float64_t>(gamma.gradient().at<float32_t>({c})), numeric_g, 1e-3));
        const auto orig_b = beta.at<float32_t>({c});
        beta.set<float32_t>({c}, orig_b + h);
        const auto up_b = fp32_loss();
        beta.set<float32_t>({c}, orig_b - h);
        const auto down_b = fp32_loss();
        beta.set<float32_t>({c}, orig_b);
        const auto numeric_b = (up_b - down_b) / (2.0 * static_cast<float64_t>(h));
        CHECK(rel_near(static_cast<float64_t>(beta.gradient().at<float32_t>({c})), numeric_b, 1e-3));
    }
}

// The eval backward's dx is analytically gamma * invstd * dy with the running
// statistics as graph constants — asserted against the closed form built from
// the test's own constants (dy is the constant weight tensor of the
// (out * w).sum() loss), not a finite difference. The running buffers are set
// to known non-trivial values and, per the eval contract, are NOT read back
// for the expectation: dx depends only on gamma, running_var and eps.
static void eval_backward_dx_is_gamma_invstd_dy() {
    const auto n_size = size_t{3};
    const auto channels = size_t{4};
    const auto eps = float64_t{1e-5};
    auto x = Tensor({n_size, channels}, DType::Float64, Device::CPU, true);
    static_cast<void>(fill_rank2(x, n_size, channels));
    auto gamma = Tensor({channels}, DType::Float64, Device::CPU, true);
    auto beta = Tensor({channels}, DType::Float64, Device::CPU, true);
    auto running_mean = Tensor({channels}, DType::Float64, Device::CPU);
    auto running_var = Tensor({channels}, DType::Float64, Device::CPU);
    auto w = Tensor({n_size, channels}, DType::Float64, Device::CPU);
    const auto rm_vals = std::vector<float64_t>{0.5, -1.0, 2.0, 0.25};
    const auto rv_vals = std::vector<float64_t>{1.5, 0.5, 3.0, 1.25};
    const auto g_vals = std::vector<float64_t>{1.5, -0.5, 0.75, 2.0};
    const auto b_vals = std::vector<float64_t>{0.25, -1.0, 0.5, -0.25};
    const auto w_vals = dy_weights(n_size * channels);
    for (auto c = size_t{0}; c < channels; ++c) {
        set_value(gamma, {c}, g_vals[c]);
        set_value(beta, {c}, b_vals[c]);
        set_value(running_mean, {c}, rm_vals[c]);
        set_value(running_var, {c}, rv_vals[c]);
    }
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            set_value(w, {n, c}, w_vals[n * channels + c]);
        }
    }
    auto out = x.batch_norm(gamma, beta, running_mean, running_var, false, 0.1, eps);
    // The mul result is named: graph parents are raw pointers, so every
    // tensor in the graph (the running buffers included) must outlive
    // backward().
    auto weighted = out.mul(w);
    auto loss = weighted.sum();
    loss.backward(DType::Float64);
    // Closed form, same op order the eval node pins: dx[i] =
    // (gamma[c] * invstd[c]) * dy[i] with invstd = 1/sqrt(running_var[c] + eps)
    // and dy = w.
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            const auto invstd = float64_t{1.0} / std::sqrt(rv_vals[c] + eps);
            const auto want = (g_vals[c] * invstd) * w_vals[n * channels + c];
            CHECK(rel_near(x.gradient().at<float64_t>({n, c}), want, 1e-12));
        }
    }
}

static void eval_backward_matches_numeric_gradient() {
    const auto n_size = size_t{3};
    const auto channels = size_t{4};
    const auto eps = float64_t{1e-5};
    const auto h = float64_t{1e-5};
    auto x = Tensor({n_size, channels}, DType::Float64, Device::CPU, true);
    const auto x_vals = fill_rank2(x, n_size, channels);
    auto gamma = Tensor({channels}, DType::Float64, Device::CPU, true);
    auto beta = Tensor({channels}, DType::Float64, Device::CPU, true);
    auto running_mean = Tensor({channels}, DType::Float64, Device::CPU);
    auto running_var = Tensor({channels}, DType::Float64, Device::CPU);
    auto w = Tensor({n_size, channels}, DType::Float64, Device::CPU);
    const auto rm_vals = std::vector<float64_t>{0.5, -1.0, 2.0, 0.25};
    const auto rv_vals = std::vector<float64_t>{1.5, 0.5, 3.0, 1.25};
    const auto g_vals = std::vector<float64_t>{1.5, -0.5, 0.75, 2.0};
    const auto b_vals = std::vector<float64_t>{0.25, -1.0, 0.5, -0.25};
    const auto w_vals = dy_weights(n_size * channels);
    for (auto c = size_t{0}; c < channels; ++c) {
        set_value(gamma, {c}, g_vals[c]);
        set_value(beta, {c}, b_vals[c]);
        set_value(running_mean, {c}, rm_vals[c]);
        set_value(running_var, {c}, rv_vals[c]);
    }
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            set_value(w, {n, c}, w_vals[n * channels + c]);
        }
    }
    auto out = x.batch_norm(gamma, beta, running_mean, running_var, false, 0.1, eps);
    // The mul result is named: graph parents are raw pointers, so every
    // tensor in the graph (the running buffers included) must outlive
    // backward().
    auto weighted = out.mul(w);
    auto loss = weighted.sum();
    loss.backward(DType::Float64);
    // dx: central differences of the in-test fp64 eval reference loss.
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            const auto i = n * channels + c;
            auto xp = x_vals;
            xp[i] += h;
            auto xm = x_vals;
            xm[i] -= h;
            const auto numeric = (eval_loss_reference(xp, channels, 1, g_vals, b_vals, rm_vals, rv_vals, w_vals, eps)
                                - eval_loss_reference(xm, channels, 1, g_vals, b_vals, rm_vals, rv_vals, w_vals, eps)) / (2.0 * h);
            CHECK(rel_near(x.gradient().at<float64_t>({n, c}), numeric, 1e-6));
        }
    }
    // dgamma and dbeta: per-channel parameter perturbations — the analytic
    // dgamma is sum(dy * x_hat) with the running-stat x_hat, dbeta is sum(dy).
    for (auto c = size_t{0}; c < channels; ++c) {
        auto gp = g_vals;
        gp[c] += h;
        auto gm = g_vals;
        gm[c] -= h;
        const auto numeric_g = (eval_loss_reference(x_vals, channels, 1, gp, b_vals, rm_vals, rv_vals, w_vals, eps)
                              - eval_loss_reference(x_vals, channels, 1, gm, b_vals, rm_vals, rv_vals, w_vals, eps)) / (2.0 * h);
        CHECK(rel_near(gamma.gradient().at<float64_t>({c}), numeric_g, 1e-6));
        auto bp = b_vals;
        bp[c] += h;
        auto bm = b_vals;
        bm[c] -= h;
        const auto numeric_b = (eval_loss_reference(x_vals, channels, 1, g_vals, bp, rm_vals, rv_vals, w_vals, eps)
                              - eval_loss_reference(x_vals, channels, 1, g_vals, bm, rm_vals, rv_vals, w_vals, eps)) / (2.0 * h);
        CHECK(rel_near(beta.gradient().at<float64_t>({c}), numeric_b, 1e-6));
    }
}

static void errors() {
    const auto eps = float64_t{1e-5};
    {
        // Rank 3 is not a supported layout.
        auto x = Tensor({2, 3, 4}, DType::Float32, Device::CPU);
        auto gamma = Tensor({3}, DType::Float32, Device::CPU);
        auto beta = Tensor({3}, DType::Float32, Device::CPU);
        auto running_mean = Tensor({3}, DType::Float32, Device::CPU);
        auto running_var = Tensor({3}, DType::Float32, Device::CPU);
        CHECK_THROWS(x.batch_norm(gamma, beta, running_mean, running_var, true, 0.1, eps),
                     BeanTensor::ErrorHandling::MismatchedShapes);
    }
    {
        // Channel-count mismatch between input and gamma.
        auto x = Tensor({4, 3}, DType::Float32, Device::CPU);
        auto gamma = Tensor({2}, DType::Float32, Device::CPU);
        auto beta = Tensor({3}, DType::Float32, Device::CPU);
        auto running_mean = Tensor({3}, DType::Float32, Device::CPU);
        auto running_var = Tensor({3}, DType::Float32, Device::CPU);
        CHECK_THROWS(x.batch_norm(gamma, beta, running_mean, running_var, true, 0.1, eps),
                     BeanTensor::ErrorHandling::MismatchedShapes);
    }
    {
        // Non-contiguous input or operand is rejected (the kernel reads every
        // operand flat through data + offset, so strided views would compute
        // wrong values silently).
        auto base = Tensor({2, 6}, DType::Float32, Device::CPU);
        auto ranges = std::vector<std::pair<size_t, size_t>>{{0, 2}, {1, 4}};
        auto gamma = Tensor({3}, DType::Float32, Device::CPU);
        auto beta = Tensor({3}, DType::Float32, Device::CPU);
        auto running_mean = Tensor({3}, DType::Float32, Device::CPU);
        auto running_var = Tensor({3}, DType::Float32, Device::CPU);
        auto x_view = base.slice(ranges); // inner box: non-contiguous view, shape {2,3}
        CHECK_THROWS(x_view.batch_norm(gamma, beta, running_mean, running_var, true, 0.1, eps),
                     BeanTensor::ErrorHandling::NotImplemented);
        auto x = Tensor({4, 3}, DType::Float32, Device::CPU);
        auto gamma_view = base.slice(ranges); // same non-contiguity through an operand
        CHECK_THROWS(x.batch_norm(gamma_view, beta, running_mean, running_var, true, 0.1, eps),
                     BeanTensor::ErrorHandling::NotImplemented);
    }
    {
        // Integer input dtype is unsupported.
        auto x = Tensor({4, 3}, DType::Int32, Device::CPU);
        auto gamma = Tensor({3}, DType::Int32, Device::CPU);
        auto beta = Tensor({3}, DType::Int32, Device::CPU);
        auto running_mean = Tensor({3}, DType::Int32, Device::CPU);
        auto running_var = Tensor({3}, DType::Int32, Device::CPU);
        CHECK_THROWS(x.batch_norm(gamma, beta, running_mean, running_var, true, 0.1, eps),
                     BeanTensor::ErrorHandling::NotImplemented);
    }
    {
        // Empty input.
        auto x = Tensor({0, 3}, DType::Float32, Device::CPU);
        auto gamma = Tensor({3}, DType::Float32, Device::CPU);
        auto beta = Tensor({3}, DType::Float32, Device::CPU);
        auto running_mean = Tensor({3}, DType::Float32, Device::CPU);
        auto running_var = Tensor({3}, DType::Float32, Device::CPU);
        CHECK_THROWS(x.batch_norm(gamma, beta, running_mean, running_var, true, 0.1, eps),
                     BeanTensor::ErrorHandling::IndexOutOfRange);
    }
    {
        auto x = Tensor({4, 3}, DType::Float32, Device::CPU);
        auto gamma = Tensor({3}, DType::Float32, Device::CPU);
        auto beta = Tensor({3}, DType::Float32, Device::CPU);
        auto running_mean = Tensor({3}, DType::Float32, Device::CPU);
        auto running_var = Tensor({3}, DType::Float32, Device::CPU);
        CHECK_THROWS(x.batch_norm(gamma, beta, running_mean, running_var, true, 0.1, 0.0),
                     BeanTensor::ErrorHandling::InvalidHyperparameter);
        CHECK_THROWS(x.batch_norm(gamma, beta, running_mean, running_var, true, 0.0, eps),
                     BeanTensor::ErrorHandling::InvalidHyperparameter);
        CHECK_THROWS(x.batch_norm(gamma, beta, running_mean, running_var, true, 1.5, eps),
                     BeanTensor::ErrorHandling::InvalidHyperparameter);
        // NaN hyperparameters: the positive-form checks reject them (NaN fails
        // every comparison), at the direct Tensor::batch_norm entry point.
        const auto nan = std::numeric_limits<float64_t>::quiet_NaN();
        CHECK_THROWS(x.batch_norm(gamma, beta, running_mean, running_var, true, 0.1, nan),
                     BeanTensor::ErrorHandling::InvalidHyperparameter);
        CHECK_THROWS(x.batch_norm(gamma, beta, running_mean, running_var, true, nan, eps),
                     BeanTensor::ErrorHandling::InvalidHyperparameter);
        CHECK_THROWS(x.batch_norm(gamma, beta, running_mean, running_var, false, nan, nan),
                     BeanTensor::ErrorHandling::InvalidHyperparameter);
    }
}

// =====================================================================
// BatchNorm module
// =====================================================================

// In-test fp32 train-mode loss for the module gradient checks:
// L = sum_n_c(out[n,c] * w[n,c]) accumulated in fp64 from the framework
// forward's fp32 readbacks (the function actually differentiated). Train-mode
// forwards re-blend the module's running buffers on every call, which the
// loss never reads, so repeated calls inside the perturbation loop are safe.
static float64_t module_train_loss(BatchNorm& module, Tensor& x, Tensor& w,
                                   const size_t n_size, const size_t channels) {
    auto out = module.forward(x);
    auto total = float64_t{0.0};
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            total += static_cast<float64_t>(out.at<float32_t>({n, c}))
                   * static_cast<float64_t>(w.at<float32_t>({n, c}));
        }
    }
    return total;
}

// Const-safe element readback of a rank-1 fp32 tensor. Tensor::at is
// non-const, so the const refs handed out by running_stats() are read
// through the const index_select instead (one-element gather + read).
static float32_t read_rank1_f32(const Tensor& t, const size_t c) {
    auto idx = Tensor({1}, DType::Int32, Device::CPU);
    idx.set<int32_t>({0}, static_cast<int32_t>(c));
    auto selected = t.index_select(0, idx);
    return selected.at<float32_t>({0});
}

// Same const-safe rank-1 readback for fp64 tensors.
static float64_t read_rank1_f64(const Tensor& t, const size_t c) {
    auto idx = Tensor({1}, DType::Int32, Device::CPU);
    idx.set<int32_t>({0}, static_cast<int32_t>(c));
    auto selected = t.index_select(0, idx);
    return selected.at<float64_t>({0});
}

static void module_train_forward_and_param_grads() {
    const auto n_size = size_t{4};
    const auto channels = size_t{3};
    // fp32 loss quantization dominates finite differences below ~1e-3, so h
    // is wider than in the fp64 tests; the loss is accumulated in fp64 from
    // fp32 readbacks of the framework forward itself.
    const auto h = float32_t{1e-2f};
    auto module = BatchNorm(channels);
    auto& gamma = module.parameter("gamma");
    auto& beta = module.parameter("beta");
    // Deterministic exactly-representable parameters (quarters): identical in
    // the fp32 storage and the fp64 accumulation below.
    for (auto c = size_t{0}; c < channels; ++c) {
        gamma.set<float32_t>({c}, static_cast<float32_t>(1.0 + 0.25 * static_cast<float64_t>(c)));
        beta.set<float32_t>({c}, static_cast<float32_t>(-0.5 + 0.25 * static_cast<float64_t>(c)));
    }
    auto x = Tensor({n_size, channels}, DType::Float32, Device::CPU, true);
    static_cast<void>(fill_rank2(x, n_size, channels));
    auto w = Tensor({n_size, channels}, DType::Float32, Device::CPU);
    const auto w_vals = dy_weights(n_size * channels);
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            set_value(w, {n, c}, w_vals[n * channels + c]);
        }
    }
    auto out = module.forward(x);
    // Named locals: graph parents are raw pointers, so every tensor in the
    // graph must outlive backward().
    auto weighted = out.mul(w);
    auto loss = weighted.sum();
    loss.backward(DType::Float32);
    // dgamma and dbeta: central differences of the fp32 module loss.
    for (auto c = size_t{0}; c < channels; ++c) {
        const auto orig_g = gamma.at<float32_t>({c});
        gamma.set<float32_t>({c}, orig_g + h);
        const auto up_g = module_train_loss(module, x, w, n_size, channels);
        gamma.set<float32_t>({c}, orig_g - h);
        const auto down_g = module_train_loss(module, x, w, n_size, channels);
        gamma.set<float32_t>({c}, orig_g);
        const auto numeric_g = (up_g - down_g) / (2.0 * static_cast<float64_t>(h));
        CHECK(rel_near(static_cast<float64_t>(module.grad("gamma").at<float32_t>({c})), numeric_g, 1e-3));
        const auto orig_b = beta.at<float32_t>({c});
        beta.set<float32_t>({c}, orig_b + h);
        const auto up_b = module_train_loss(module, x, w, n_size, channels);
        beta.set<float32_t>({c}, orig_b - h);
        const auto down_b = module_train_loss(module, x, w, n_size, channels);
        beta.set<float32_t>({c}, orig_b);
        const auto numeric_b = (up_b - down_b) / (2.0 * static_cast<float64_t>(h));
        CHECK(rel_near(static_cast<float64_t>(module.grad("beta").at<float32_t>({c})), numeric_b, 1e-3));
    }
    // Registry: exactly gamma and beta — the running stats are buffers, not
    // parameters, and must not appear in any optimizer-facing view.
    const auto params = module.named_parameters();
    CHECK(params.size() == 2);
    CHECK(params[0].first == "gamma");
    CHECK(params[1].first == "beta");
    CHECK(module.parameter_count() == 6);
    CHECK(module.name() == "BatchNorm");
}

static void module_eval_uses_running_stats() {
    const auto n_size = size_t{4};
    const auto channels = size_t{3};
    const auto momentum = float64_t{0.1};
    const auto eps = float64_t{1e-5};
    auto module = BatchNorm(channels);
    auto train_x = Tensor({n_size, channels}, DType::Float32, Device::CPU);
    const auto train_vals = fill_rank2(train_x, n_size, channels);
    auto y_train = module.forward(train_x);
    static_cast<void>(y_train);
    // Post-train buffers, read back through the const-safe accessor path and
    // pinned against the in-test blend: init mean 0 / var 1, BIASED batch
    // variance, momentum blend — the op-level contract at module scope.
    const auto& rm = module.running_stats().first;
    const auto& rv = module.running_stats().second;
    auto expected = std::vector<float64_t>{};
    auto batch_mean = std::vector<float64_t>{};
    auto batch_var = std::vector<float64_t>{};
    train_reference(train_vals, channels, 1, {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}, eps,
                    expected, batch_mean, batch_var);
    auto rm_vals = std::vector<float64_t>(channels);
    auto rv_vals = std::vector<float64_t>(channels);
    auto mean_bits = std::vector<uint32_t>(channels);
    auto var_bits = std::vector<uint32_t>(channels);
    for (auto c = size_t{0}; c < channels; ++c) {
        const auto got_mean = read_rank1_f32(rm, c);
        const auto got_var = read_rank1_f32(rv, c);
        CHECK(rel_near(static_cast<float64_t>(got_mean), batch_mean[c] * momentum, 1e-5));
        CHECK(rel_near(static_cast<float64_t>(got_var),
                       (1.0 * (1.0 - momentum)) + (batch_var[c] * momentum), 1e-5));
        rm_vals[c] = static_cast<float64_t>(got_mean);
        rv_vals[c] = static_cast<float64_t>(got_var);
        mean_bits[c] = std::bit_cast<uint32_t>(got_mean);
        var_bits[c] = std::bit_cast<uint32_t>(got_var);
    }
    // Eval forward: normalizes with the running statistics, touches nothing.
    module.eval();
    auto eval_x = Tensor({n_size, channels}, DType::Float32, Device::CPU);
    const auto eval_vals = fill_rank2(eval_x, n_size, channels);
    auto y_eval = module.forward(eval_x);
    auto eval_expected = std::vector<float64_t>{};
    eval_reference(eval_vals, channels, 1, {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}, rm_vals, rv_vals, eps,
                   eval_expected);
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            CHECK(rel_near(static_cast<float64_t>(y_eval.at<float32_t>({n, c})),
                           eval_expected[n * channels + c], 1e-4));
        }
    }
    // Negative control: the eval_x BATCH statistics must NOT reproduce the
    // output — the running stats after one blend from the zeros/ones init
    // differ from any batch statistic by O(1).
    auto batch_expected = std::vector<float64_t>{};
    train_reference(eval_vals, channels, 1, {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}, eps,
                    batch_expected, batch_mean, batch_var);
    CHECK(!rel_near(static_cast<float64_t>(y_eval.at<float32_t>({0, 0})), batch_expected[0], 1e-3));
    // Buffers bit-untouched by the eval forward.
    for (auto c = size_t{0}; c < channels; ++c) {
        CHECK(std::bit_cast<uint32_t>(read_rank1_f32(rm, c)) == mean_bits[c]);
        CHECK(std::bit_cast<uint32_t>(read_rank1_f32(rv, c)) == var_bits[c]);
    }
}

static void module_train_eval_switch_restores_batch_stats() {
    const auto n_size = size_t{4};
    const auto channels = size_t{3};
    const auto eps = float64_t{1e-5};
    auto module = BatchNorm(channels);
    auto train_x = Tensor({n_size, channels}, DType::Float32, Device::CPU);
    static_cast<void>(fill_rank2(train_x, n_size, channels));
    auto y_train = module.forward(train_x);
    static_cast<void>(y_train);
    module.eval();
    auto eval_x = Tensor({n_size, channels}, DType::Float32, Device::CPU);
    const auto eval_vals = fill_rank2(eval_x, n_size, channels);
    auto y_eval = module.forward(eval_x);
    static_cast<void>(y_eval);
    // Back to train: the forward normalizes with the eval_x BATCH statistics
    // again (train mode never consults the running buffers).
    module.train();
    auto y_train2 = module.forward(eval_x);
    auto expected = std::vector<float64_t>{};
    auto batch_mean = std::vector<float64_t>{};
    auto batch_var = std::vector<float64_t>{};
    train_reference(eval_vals, channels, 1, {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}, eps,
                    expected, batch_mean, batch_var);
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            CHECK(rel_near(static_cast<float64_t>(y_train2.at<float32_t>({n, c})),
                           expected[n * channels + c], 1e-4));
        }
    }
    // Negative control: the train-mode output differs from the eval-mode
    // output on the same input (running stats differ from batch stats).
    CHECK(!rel_near(static_cast<float64_t>(y_train2.at<float32_t>({0, 0})),
                    static_cast<float64_t>(y_eval.at<float32_t>({0, 0})), 1e-3));
}

static void module_move_then_backward_keeps_saved_pointers_valid() {
    const auto n_size = size_t{4};
    const auto channels = size_t{3};
    const auto eps = float64_t{1e-5};
    // The eval backward reads the running stats through raw Node::_saved
    // pointers into the module's slots, so the slots must keep their
    // addresses across a module move and stay valid through backward().
    // fp64 pins the moved-vs-unmoved gradient comparison tightly.
    auto x_moved = Tensor({n_size, channels}, DType::Float64, Device::CPU, true);
    static_cast<void>(fill_rank2(x_moved, n_size, channels));
    auto x_ref = Tensor({n_size, channels}, DType::Float64, Device::CPU, true);
    static_cast<void>(fill_rank2(x_ref, n_size, channels));
    auto w = Tensor({n_size, channels}, DType::Float64, Device::CPU);
    const auto w_vals = dy_weights(n_size * channels);
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            set_value(w, {n, c}, w_vals[n * channels + c]);
        }
    }
    auto module_moved = BatchNorm(channels, DType::Float64, Device::CPU, 0.1, eps);
    module_moved.eval();
    auto out_moved = module_moved.forward(x_moved);
    // Named locals: graph parents are raw pointers (the eval node's _saved
    // pointers target the module's running-stat slots), so the whole graph
    // must outlive backward().
    auto weighted_moved = out_moved.mul(w);
    auto loss_moved = weighted_moved.sum();
    // Move AFTER the forward: the deque-of-unique_ptr slot layout must keep
    // the running-stat heap addresses stable, so the held graph stays valid.
    auto module_moved_to = std::move(module_moved);
    loss_moved.backward(DType::Float64);
    // Unmoved reference run: identical construction, no move.
    auto module_ref = BatchNorm(channels, DType::Float64, Device::CPU, 0.1, eps);
    module_ref.eval();
    auto out_ref = module_ref.forward(x_ref);
    auto weighted_ref = out_ref.mul(w);
    auto loss_ref = weighted_ref.sum();
    loss_ref.backward(DType::Float64);
    // Same grads as the unmoved module: the saved pointers stayed valid.
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            CHECK(rel_near(x_moved.gradient().at<float64_t>({n, c}),
                           x_ref.gradient().at<float64_t>({n, c}), 1e-12));
        }
    }
    for (auto c = size_t{0}; c < channels; ++c) {
        CHECK(rel_near(module_moved_to.grad("gamma").at<float64_t>({c}),
                       module_ref.grad("gamma").at<float64_t>({c}), 1e-12));
        CHECK(rel_near(module_moved_to.grad("beta").at<float64_t>({c}),
                       module_ref.grad("beta").at<float64_t>({c}), 1e-12));
    }
}

// Move-ASSIGNMENT regression: Module::operator=(Module&&) must carry the
// training flag (the defaulted move constructor does), or a move-assigned
// eval module silently reverts to train mode — re-blending the running
// buffers and normalizing with batch statistics on the next forward. Also
// pins the _saved lifetime: a backward through an eval output produced
// BEFORE the move must still read valid running buffers afterwards.
static void module_move_assignment_preserves_eval_mode() {
    const auto n_size = size_t{4};
    const auto channels = size_t{3};
    const auto eps = float64_t{1e-5};
    auto x = Tensor({n_size, channels}, DType::Float64, Device::CPU, true);
    const auto x_vals = fill_rank2(x, n_size, channels);
    auto w = Tensor({n_size, channels}, DType::Float64, Device::CPU);
    const auto w_vals = dy_weights(n_size * channels);
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            set_value(w, {n, c}, w_vals[n * channels + c]);
        }
    }
    auto source = BatchNorm(channels, DType::Float64, Device::CPU, 0.1, eps);
    source.eval();
    auto out = source.forward(x);
    // Named locals: graph parents are raw pointers, so every tensor in the
    // graph must outlive backward().
    auto weighted = out.mul(w);
    auto loss = weighted.sum();
    auto target = BatchNorm(channels, DType::Float64, Device::CPU, 0.1, eps);
    target = std::move(source);
    // (a) Mode preservation: the destination was constructed in train mode;
    // the eval flag must have moved across, not been left at the default.
    CHECK(!target.is_training());
    // (b) Backward through the pre-move eval output: the eval node's _saved
    // pointers target the running-stat slots, whose heap addresses survived
    // the assignment (deque of unique_ptr moved, pointees untouched).
    loss.backward(DType::Float64);
    // Closed form: gamma ones, running stats at the init mean 0 / var 1, so
    // dx[i] = (1 * invstd) * dy[i] with invstd = 1/sqrt(1 + eps), dy = w.
    const auto invstd = float64_t{1.0} / std::sqrt(1.0 + eps);
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            CHECK(rel_near(x.gradient().at<float64_t>({n, c}),
                           invstd * w_vals[n * channels + c], 1e-12));
        }
    }
    // (a) continued: a forward AFTER the move still uses the running
    // statistics (init values, gamma ones / beta zeros) and leaves the
    // buffers bit-untouched. A module silently back in train mode would
    // instead normalize with the batch statistics and blend the buffers.
    auto y = target.forward(x);
    auto expected = std::vector<float64_t>{};
    eval_reference(x_vals, channels, 1, {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0},
                   {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, eps, expected);
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            CHECK(rel_near(y.at<float64_t>({n, c}), expected[n * channels + c], 1e-12));
        }
    }
    // Negative control: the batch-statistic normalization must NOT match.
    auto batch_expected = std::vector<float64_t>{};
    auto batch_mean = std::vector<float64_t>{};
    auto batch_var = std::vector<float64_t>{};
    train_reference(x_vals, channels, 1, {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}, eps,
                    batch_expected, batch_mean, batch_var);
    CHECK(!rel_near(y.at<float64_t>({0, 0}), batch_expected[0], 1e-3));
    // Buffers bit-untouched by the post-move forward: still the init zeros/ones.
    const auto& rm = target.running_stats().first;
    const auto& rv = target.running_stats().second;
    for (auto c = size_t{0}; c < channels; ++c) {
        CHECK(std::bit_cast<uint64_t>(read_rank1_f64(rm, c)) == 0u);
        CHECK(std::bit_cast<uint64_t>(read_rank1_f64(rv, c)) == std::bit_cast<uint64_t>(float64_t{1.0}));
    }
}

static void module_constructor_validation() {
    CHECK_THROWS(BatchNorm(0), BeanTensor::ErrorHandling::IndexOutOfRange);
    CHECK_THROWS((BatchNorm(3, DType::Float32, Device::CPU, 0.0, 1e-5)),
                 BeanTensor::ErrorHandling::InvalidHyperparameter);
    CHECK_THROWS((BatchNorm(3, DType::Float32, Device::CPU, 1.5, 1e-5)),
                 BeanTensor::ErrorHandling::InvalidHyperparameter);
    CHECK_THROWS((BatchNorm(3, DType::Float32, Device::CPU, 0.1, 0.0)),
                 BeanTensor::ErrorHandling::InvalidHyperparameter);
    // NaN hyperparameters: the positive-form checks reject them at the module
    // constructor entry point as well.
    const auto nan = std::numeric_limits<float64_t>::quiet_NaN();
    CHECK_THROWS((BatchNorm(3, DType::Float32, Device::CPU, nan, 1e-5)),
                 BeanTensor::ErrorHandling::InvalidHyperparameter);
    CHECK_THROWS((BatchNorm(3, DType::Float32, Device::CPU, 0.1, nan)),
                 BeanTensor::ErrorHandling::InvalidHyperparameter);
}

static void module_print_includes_batchnorm() {
    auto module = BatchNorm(3);
    const auto s = module.print();
    CHECK(s.find("BatchNorm") != std::string::npos);
    CHECK(s.find("gamma: Float32 {3}") != std::string::npos);
    CHECK(s.find("beta: Float32 {3}") != std::string::npos);
    CHECK(s.find("total: 6 parameters") != std::string::npos);
    // The running stats are buffers, not parameters: no lines for them.
    CHECK(s.find("running") == std::string::npos);
}

// =====================================================================
// GPU backend coverage (CUDA/HIP builds, Device::GPU tensors). The forward
// checks reuse the in-test fp64 references at the CPU suite's tolerances
// (the float-order exemption covers grid-stride vs chunk-merge differences);
// the backward checks compute central finite differences from DEVICE
// READBACK values accumulated in fp64 host loops — numeric-vs-analytic on
// the same device, never GPU grads against CPU-golden tensors.
// =====================================================================
#if defined(USE_CUDA) || defined(USE_HIP)

static bool gpu_available() {
#if defined(USE_HIP)
    int32_t count = 0;
    return hipGetDeviceCount(&count) == hipSuccess && count > 0;
#else
    int32_t count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
#endif
}

static void gpu_train_forward_rank2_matches_reference() {
    const auto n_size = size_t{4};
    const auto channels = size_t{3};
    auto x = Tensor({n_size, channels}, DType::Float32, Device::GPU);
    const auto vals = fill_rank2(x, n_size, channels);
    auto gamma = Tensor({channels}, DType::Float32, Device::GPU);
    auto beta = Tensor({channels}, DType::Float32, Device::GPU);
    auto running_mean = Tensor({channels}, DType::Float32, Device::GPU);
    auto running_var = Tensor({channels}, DType::Float32, Device::GPU);
    ones_zeros_gamma_beta(gamma, beta, running_mean, running_var, channels);
    const auto eps = float64_t{1e-5};
    auto y = x.batch_norm(gamma, beta, running_mean, running_var, true, 0.1, eps);
    CHECK(y.get_dtype() == DType::Float32);
    CHECK(y.get_device() == Device::GPU);
    CHECK(y.is_same_shape(x));
    auto expected = std::vector<float64_t>{};
    auto batch_mean = std::vector<float64_t>{};
    auto batch_var = std::vector<float64_t>{};
    train_reference(vals, channels, 1, {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}, eps, expected, batch_mean, batch_var);
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            const auto got = static_cast<float64_t>(y.at<float32_t>({n, c}));
            CHECK(rel_near(got, expected[n * channels + c], 1e-4));
        }
    }
}

static void gpu_train_forward_rank4_matches_reference() {
    const auto n_size = size_t{2};
    const auto channels = size_t{3};
    const auto height = size_t{2};
    const auto width = size_t{2};
    auto x = Tensor({n_size, channels, height, width}, DType::Float32, Device::GPU);
    const auto vals = fill_rank4(x, n_size, channels, height, width);
    auto gamma = Tensor({channels}, DType::Float32, Device::GPU);
    auto beta = Tensor({channels}, DType::Float32, Device::GPU);
    auto running_mean = Tensor({channels}, DType::Float32, Device::GPU);
    auto running_var = Tensor({channels}, DType::Float32, Device::GPU);
    ones_zeros_gamma_beta(gamma, beta, running_mean, running_var, channels);
    const auto eps = float64_t{1e-5};
    auto y = x.batch_norm(gamma, beta, running_mean, running_var, true, 0.1, eps);
    CHECK(y.get_dtype() == DType::Float32);
    CHECK(y.is_same_shape(x));
    auto expected = std::vector<float64_t>{};
    auto batch_mean = std::vector<float64_t>{};
    auto batch_var = std::vector<float64_t>{};
    train_reference(vals, channels, height * width, {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}, eps,
                    expected, batch_mean, batch_var);
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            for (auto h = size_t{0}; h < height; ++h) {
                for (auto w = size_t{0}; w < width; ++w) {
                    const auto j = h * width + w;
                    const auto got = static_cast<float64_t>(y.at<float32_t>({n, c, h, w}));
                    CHECK(rel_near(got, expected[(n * channels + c) * height * width + j], 1e-4));
                }
            }
        }
    }
}

static void gpu_running_stats_update_matches_reference() {
    const auto n_size = size_t{4};
    const auto channels = size_t{3};
    auto x = Tensor({n_size, channels}, DType::Float32, Device::GPU);
    const auto vals = fill_rank2(x, n_size, channels);
    auto gamma = Tensor({channels}, DType::Float32, Device::GPU);
    auto beta = Tensor({channels}, DType::Float32, Device::GPU);
    auto running_mean = Tensor({channels}, DType::Float32, Device::GPU);
    auto running_var = Tensor({channels}, DType::Float32, Device::GPU);
    ones_zeros_gamma_beta(gamma, beta, running_mean, running_var, channels);
    const auto old_mean = std::vector<float64_t>{0.5, -1.0, 2.0};
    const auto old_var = std::vector<float64_t>{1.5, 0.5, 3.0};
    for (auto c = size_t{0}; c < channels; ++c) {
        set_value(running_mean, {c}, old_mean[c]);
        set_value(running_var, {c}, old_var[c]);
    }
    const auto momentum = float64_t{0.25};
    const auto eps = float64_t{1e-5};
    auto y = x.batch_norm(gamma, beta, running_mean, running_var, true, momentum, eps);
    static_cast<void>(y);
    auto expected = std::vector<float64_t>{};
    auto batch_mean = std::vector<float64_t>{};
    auto batch_var = std::vector<float64_t>{};
    train_reference(vals, channels, 1, {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}, eps, expected, batch_mean, batch_var);
    // Pinned blend: new = old * (1 - m) + batch * m, BIASED batch variance.
    for (auto c = size_t{0}; c < channels; ++c) {
        const auto want_mean = (old_mean[c] * (1.0 - momentum)) + (batch_mean[c] * momentum);
        const auto want_var = (old_var[c] * (1.0 - momentum)) + (batch_var[c] * momentum);
        CHECK(rel_near(static_cast<float64_t>(running_mean.at<float32_t>({c})), want_mean, 1e-5));
        CHECK(rel_near(static_cast<float64_t>(running_var.at<float32_t>({c})), want_var, 1e-5));
    }
}

static void gpu_eval_after_train_without_sync() {
    const auto n_size = size_t{4};
    const auto channels = size_t{3};
    auto x = Tensor({n_size, channels}, DType::Float32, Device::GPU);
    const auto vals = fill_rank2(x, n_size, channels);
    auto gamma = Tensor({channels}, DType::Float32, Device::GPU);
    auto beta = Tensor({channels}, DType::Float32, Device::GPU);
    auto running_mean = Tensor({channels}, DType::Float32, Device::GPU);
    auto running_var = Tensor({channels}, DType::Float32, Device::GPU);
    ones_zeros_gamma_beta(gamma, beta, running_mean, running_var, channels);
    const auto old_mean = std::vector<float64_t>{0.5, -1.0, 2.0};
    const auto old_var = std::vector<float64_t>{1.5, 0.5, 3.0};
    for (auto c = size_t{0}; c < channels; ++c) {
        set_value(running_mean, {c}, old_mean[c]);
        set_value(running_var, {c}, old_var[c]);
    }
    const auto momentum = float64_t{0.25};
    const auto eps = float64_t{1e-5};
    auto y_train = x.batch_norm(gamma, beta, running_mean, running_var, true, momentum, eps);
    static_cast<void>(y_train);
    // No sync of x, the output, or the running buffers in between: the eval
    // forward's entry drain of the buffers must order the train-mode buffer
    // mutation before the eval read, so this forward normalizes with the
    // POST-update running statistics.
    auto y_eval = x.batch_norm(gamma, beta, running_mean, running_var, false, momentum, eps);
    auto post_mean = std::vector<float64_t>(channels);
    auto post_var = std::vector<float64_t>(channels);
    for (auto c = size_t{0}; c < channels; ++c) {
        post_mean[c] = static_cast<float64_t>(running_mean.at<float32_t>({c}));
        post_var[c] = static_cast<float64_t>(running_var.at<float32_t>({c}));
    }
    auto expected = std::vector<float64_t>{};
    eval_reference(vals, channels, 1, {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}, post_mean, post_var, eps, expected);
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            const auto got = static_cast<float64_t>(y_eval.at<float32_t>({n, c}));
            CHECK(rel_near(got, expected[n * channels + c], 1e-4));
        }
    }
    // Negative control: a pre-update eval read would give exactly 0 at (0, 0)
    // ((x - old_mean) == 0 for channel 0); the post-update statistic does not.
    const auto got00 = static_cast<float64_t>(y_eval.at<float32_t>({0, 0}));
    CHECK(!rel_near(got00, 0.0, 1e-3));
}

static void gpu_fp16_bf16_forward_smoke() {
    const auto n_size = size_t{4};
    const auto channels = size_t{3};
    const auto eps = float64_t{1e-5};
    {
        auto x = Tensor({n_size, channels}, DType::Float16, Device::GPU);
        const auto vals = fill_rank2(x, n_size, channels);
        auto gamma = Tensor({channels}, DType::Float16, Device::GPU);
        auto beta = Tensor({channels}, DType::Float16, Device::GPU);
        auto running_mean = Tensor({channels}, DType::Float16, Device::GPU);
        auto running_var = Tensor({channels}, DType::Float16, Device::GPU);
        ones_zeros_gamma_beta(gamma, beta, running_mean, running_var, channels);
        auto y = x.batch_norm(gamma, beta, running_mean, running_var, true, 0.1, eps);
        CHECK(y.get_dtype() == DType::Float16);
        auto expected = std::vector<float64_t>{};
        auto batch_mean = std::vector<float64_t>{};
        auto batch_var = std::vector<float64_t>{};
        train_reference(vals, channels, 1, {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}, eps, expected, batch_mean, batch_var);
        for (auto n = size_t{0}; n < n_size; ++n) {
            for (auto c = size_t{0}; c < channels; ++c) {
                const auto got = static_cast<float64_t>(widen_half(y.at<float16_t>({n, c})));
                CHECK(rel_near(got, expected[n * channels + c], 1e-2));
            }
        }
    }
    {
        auto x = Tensor({n_size, channels}, DType::BFloat16, Device::GPU);
        const auto vals = fill_rank2(x, n_size, channels);
        auto gamma = Tensor({channels}, DType::BFloat16, Device::GPU);
        auto beta = Tensor({channels}, DType::BFloat16, Device::GPU);
        auto running_mean = Tensor({channels}, DType::BFloat16, Device::GPU);
        auto running_var = Tensor({channels}, DType::BFloat16, Device::GPU);
        ones_zeros_gamma_beta(gamma, beta, running_mean, running_var, channels);
        auto y = x.batch_norm(gamma, beta, running_mean, running_var, true, 0.1, eps);
        CHECK(y.get_dtype() == DType::BFloat16);
        auto expected = std::vector<float64_t>{};
        auto batch_mean = std::vector<float64_t>{};
        auto batch_var = std::vector<float64_t>{};
        train_reference(vals, channels, 1, {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}, eps, expected, batch_mean, batch_var);
        for (auto n = size_t{0}; n < n_size; ++n) {
            for (auto c = size_t{0}; c < channels; ++c) {
                const auto got = static_cast<float64_t>(widen_bf16(y.at<bfloat16_t>({n, c})));
                CHECK(rel_near(got, expected[n * channels + c], 1e-2));
            }
        }
    }
}

static void gpu_train_backward_matches_numeric_gradient_rank2() {
    const auto n_size = size_t{3};
    const auto channels = size_t{4};
    const auto eps = float64_t{1e-5};
    const auto h = float64_t{1e-5};
    auto x = Tensor({n_size, channels}, DType::Float64, Device::GPU, true);
    static_cast<void>(fill_rank2(x, n_size, channels));
    auto gamma = Tensor({channels}, DType::Float64, Device::GPU, true);
    auto beta = Tensor({channels}, DType::Float64, Device::GPU, true);
    auto running_mean = Tensor({channels}, DType::Float64, Device::GPU);
    auto running_var = Tensor({channels}, DType::Float64, Device::GPU);
    auto w = Tensor({n_size, channels}, DType::Float64, Device::GPU);
    const auto g_vals = std::vector<float64_t>{1.5, -0.5, 0.75, 2.0};
    const auto b_vals = std::vector<float64_t>{0.25, -1.0, 0.5, -0.25};
    const auto w_vals = dy_weights(n_size * channels);
    for (auto c = size_t{0}; c < channels; ++c) {
        set_value(gamma, {c}, g_vals[c]);
        set_value(beta, {c}, b_vals[c]);
        set_value(running_mean, {c}, 0.0);
        set_value(running_var, {c}, 0.0);
    }
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            set_value(w, {n, c}, w_vals[n * channels + c]);
        }
    }
    auto out = x.batch_norm(gamma, beta, running_mean, running_var, true, 0.1, eps);
    // Named locals: graph parents are raw pointers, so every tensor in the
    // graph must outlive backward().
    auto weighted = out.mul(w);
    auto loss = weighted.sum();
    loss.backward(DType::Float64);
    // fp64 device-readback loss: each perturbed forward runs on the GPU and
    // its elements are read back and accumulated on the host in fp64, so the
    // numeric gradient differentiates the same device function the analytic
    // node inverted. Re-running batch_norm re-blends the running buffers,
    // which the loss never reads.
    auto device_loss = [&]() {
        auto o = x.batch_norm(gamma, beta, running_mean, running_var, true, 0.1, eps);
        auto total = float64_t{0.0};
        for (auto n = size_t{0}; n < n_size; ++n) {
            for (auto c = size_t{0}; c < channels; ++c) {
                total += o.at<float64_t>({n, c}) * w_vals[n * channels + c];
            }
        }
        return total;
    };
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            const auto orig = x.at<float64_t>({n, c});
            x.set<float64_t>({n, c}, orig + h);
            const auto up = device_loss();
            x.set<float64_t>({n, c}, orig - h);
            const auto down = device_loss();
            x.set<float64_t>({n, c}, orig);
            const auto numeric = (up - down) / (2.0 * h);
            CHECK(rel_near(x.gradient().at<float64_t>({n, c}), numeric, 1e-5));
        }
    }
    for (auto c = size_t{0}; c < channels; ++c) {
        const auto orig_g = gamma.at<float64_t>({c});
        gamma.set<float64_t>({c}, orig_g + h);
        const auto up_g = device_loss();
        gamma.set<float64_t>({c}, orig_g - h);
        const auto down_g = device_loss();
        gamma.set<float64_t>({c}, orig_g);
        const auto numeric_g = (up_g - down_g) / (2.0 * h);
        CHECK(rel_near(gamma.gradient().at<float64_t>({c}), numeric_g, 1e-5));
        const auto orig_b = beta.at<float64_t>({c});
        beta.set<float64_t>({c}, orig_b + h);
        const auto up_b = device_loss();
        beta.set<float64_t>({c}, orig_b - h);
        const auto down_b = device_loss();
        beta.set<float64_t>({c}, orig_b);
        const auto numeric_b = (up_b - down_b) / (2.0 * h);
        CHECK(rel_near(beta.gradient().at<float64_t>({c}), numeric_b, 1e-5));
    }
}

static void gpu_train_backward_matches_numeric_gradient_rank4() {
    const auto n_size = size_t{2};
    const auto channels = size_t{2};
    const auto height = size_t{2};
    const auto width = size_t{2};
    const auto inner = height * width;
    const auto eps = float64_t{1e-5};
    const auto h = float64_t{1e-5};
    auto x = Tensor({n_size, channels, height, width}, DType::Float64, Device::GPU, true);
    static_cast<void>(fill_rank4(x, n_size, channels, height, width));
    auto gamma = Tensor({channels}, DType::Float64, Device::GPU, true);
    auto beta = Tensor({channels}, DType::Float64, Device::GPU, true);
    auto running_mean = Tensor({channels}, DType::Float64, Device::GPU);
    auto running_var = Tensor({channels}, DType::Float64, Device::GPU);
    auto w = Tensor({n_size, channels, height, width}, DType::Float64, Device::GPU);
    const auto g_vals = std::vector<float64_t>{1.5, -0.5};
    const auto b_vals = std::vector<float64_t>{0.25, -1.0};
    const auto w_vals = dy_weights(n_size * channels * inner);
    for (auto c = size_t{0}; c < channels; ++c) {
        set_value(gamma, {c}, g_vals[c]);
        set_value(beta, {c}, b_vals[c]);
        set_value(running_mean, {c}, 0.0);
        set_value(running_var, {c}, 0.0);
    }
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            for (auto hh = size_t{0}; hh < height; ++hh) {
                for (auto ww = size_t{0}; ww < width; ++ww) {
                    set_value(w, {n, c, hh, ww}, w_vals[(n * channels + c) * inner + hh * width + ww]);
                }
            }
        }
    }
    auto out = x.batch_norm(gamma, beta, running_mean, running_var, true, 0.1, eps);
    // Named locals: graph parents are raw pointers, so every tensor in the
    // graph must outlive backward().
    auto weighted = out.mul(w);
    auto loss = weighted.sum();
    loss.backward(DType::Float64);
    // fp64 device-readback loss, same construction as the rank-2 check.
    auto device_loss = [&]() {
        auto o = x.batch_norm(gamma, beta, running_mean, running_var, true, 0.1, eps);
        auto total = float64_t{0.0};
        for (auto n = size_t{0}; n < n_size; ++n) {
            for (auto c = size_t{0}; c < channels; ++c) {
                for (auto hh = size_t{0}; hh < height; ++hh) {
                    for (auto ww = size_t{0}; ww < width; ++ww) {
                        total += o.at<float64_t>({n, c, hh, ww}) * w_vals[(n * channels + c) * inner + hh * width + ww];
                    }
                }
            }
        }
        return total;
    };
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            for (auto hh = size_t{0}; hh < height; ++hh) {
                for (auto ww = size_t{0}; ww < width; ++ww) {
                    const auto orig = x.at<float64_t>({n, c, hh, ww});
                    x.set<float64_t>({n, c, hh, ww}, orig + h);
                    const auto up = device_loss();
                    x.set<float64_t>({n, c, hh, ww}, orig - h);
                    const auto down = device_loss();
                    x.set<float64_t>({n, c, hh, ww}, orig);
                    const auto numeric = (up - down) / (2.0 * h);
                    CHECK(rel_near(x.gradient().at<float64_t>({n, c, hh, ww}), numeric, 1e-5));
                }
            }
        }
    }
    for (auto c = size_t{0}; c < channels; ++c) {
        const auto orig_g = gamma.at<float64_t>({c});
        gamma.set<float64_t>({c}, orig_g + h);
        const auto up_g = device_loss();
        gamma.set<float64_t>({c}, orig_g - h);
        const auto down_g = device_loss();
        gamma.set<float64_t>({c}, orig_g);
        const auto numeric_g = (up_g - down_g) / (2.0 * h);
        CHECK(rel_near(gamma.gradient().at<float64_t>({c}), numeric_g, 1e-5));
        const auto orig_b = beta.at<float64_t>({c});
        beta.set<float64_t>({c}, orig_b + h);
        const auto up_b = device_loss();
        beta.set<float64_t>({c}, orig_b - h);
        const auto down_b = device_loss();
        beta.set<float64_t>({c}, orig_b);
        const auto numeric_b = (up_b - down_b) / (2.0 * h);
        CHECK(rel_near(beta.gradient().at<float64_t>({c}), numeric_b, 1e-5));
    }
}

static void gpu_train_backward_respects_grad_dtype() {
    const auto n_size = size_t{3};
    const auto channels = size_t{4};
    const auto eps = float64_t{1e-5};
    // fp32 forward quantization dominates finite differences below ~1e-3, so
    // h is wider than in the fp64 tests; the loss is accumulated in fp64 from
    // fp32 DEVICE READBACKS of the framework forward itself (the function
    // actually differentiated).
    const auto h = float32_t{1e-2f};
    auto x = Tensor({n_size, channels}, DType::Float32, Device::GPU, true);
    static_cast<void>(fill_rank2(x, n_size, channels));
    auto gamma = Tensor({channels}, DType::Float32, Device::GPU, true);
    auto beta = Tensor({channels}, DType::Float32, Device::GPU, true);
    auto running_mean = Tensor({channels}, DType::Float32, Device::GPU);
    auto running_var = Tensor({channels}, DType::Float32, Device::GPU);
    auto w = Tensor({n_size, channels}, DType::Float32, Device::GPU);
    const auto g_vals = std::vector<float64_t>{1.5, -0.5, 0.75, 2.0};
    const auto b_vals = std::vector<float64_t>{0.25, -1.0, 0.5, -0.25};
    const auto w_vals = dy_weights(n_size * channels);
    for (auto c = size_t{0}; c < channels; ++c) {
        set_value(gamma, {c}, g_vals[c]);
        set_value(beta, {c}, b_vals[c]);
        set_value(running_mean, {c}, 0.0);
        set_value(running_var, {c}, 0.0);
    }
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            set_value(w, {n, c}, w_vals[n * channels + c]);
        }
    }
    auto out = x.batch_norm(gamma, beta, running_mean, running_var, true, 0.1, eps);
    // Named locals: graph parents are raw pointers, so every tensor in the
    // graph must outlive backward().
    auto weighted = out.mul(w);
    auto loss = weighted.sum();
    loss.backward(DType::Float32);
    // The assigned grad dtype pins every leaf grad buffer.
    CHECK(x.gradient().get_dtype() == DType::Float32);
    CHECK(gamma.gradient().get_dtype() == DType::Float32);
    CHECK(beta.gradient().get_dtype() == DType::Float32);
    auto fp32_loss = [&]() {
        auto o = x.batch_norm(gamma, beta, running_mean, running_var, true, 0.1, eps);
        auto total = float64_t{0.0};
        for (auto n = size_t{0}; n < n_size; ++n) {
            for (auto c = size_t{0}; c < channels; ++c) {
                total += static_cast<float64_t>(o.at<float32_t>({n, c})) * w_vals[n * channels + c];
            }
        }
        return total;
    };
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            const auto orig = x.at<float32_t>({n, c});
            x.set<float32_t>({n, c}, orig + h);
            const auto up = fp32_loss();
            x.set<float32_t>({n, c}, orig - h);
            const auto down = fp32_loss();
            x.set<float32_t>({n, c}, orig);
            const auto numeric = (up - down) / (2.0 * static_cast<float64_t>(h));
            CHECK(rel_near(static_cast<float64_t>(x.gradient().at<float32_t>({n, c})), numeric, 1e-3));
        }
    }
    for (auto c = size_t{0}; c < channels; ++c) {
        const auto orig_g = gamma.at<float32_t>({c});
        gamma.set<float32_t>({c}, orig_g + h);
        const auto up_g = fp32_loss();
        gamma.set<float32_t>({c}, orig_g - h);
        const auto down_g = fp32_loss();
        gamma.set<float32_t>({c}, orig_g);
        const auto numeric_g = (up_g - down_g) / (2.0 * static_cast<float64_t>(h));
        CHECK(rel_near(static_cast<float64_t>(gamma.gradient().at<float32_t>({c})), numeric_g, 1e-3));
        const auto orig_b = beta.at<float32_t>({c});
        beta.set<float32_t>({c}, orig_b + h);
        const auto up_b = fp32_loss();
        beta.set<float32_t>({c}, orig_b - h);
        const auto down_b = fp32_loss();
        beta.set<float32_t>({c}, orig_b);
        const auto numeric_b = (up_b - down_b) / (2.0 * static_cast<float64_t>(h));
        CHECK(rel_near(static_cast<float64_t>(beta.gradient().at<float32_t>({c})), numeric_b, 1e-3));
    }
}

static void gpu_eval_backward_dx_is_gamma_invstd_dy() {
    const auto n_size = size_t{3};
    const auto channels = size_t{4};
    const auto eps = float64_t{1e-5};
    auto x = Tensor({n_size, channels}, DType::Float64, Device::GPU, true);
    static_cast<void>(fill_rank2(x, n_size, channels));
    auto gamma = Tensor({channels}, DType::Float64, Device::GPU, true);
    auto beta = Tensor({channels}, DType::Float64, Device::GPU, true);
    auto running_mean = Tensor({channels}, DType::Float64, Device::GPU);
    auto running_var = Tensor({channels}, DType::Float64, Device::GPU);
    auto w = Tensor({n_size, channels}, DType::Float64, Device::GPU);
    const auto rm_vals = std::vector<float64_t>{0.5, -1.0, 2.0, 0.25};
    const auto rv_vals = std::vector<float64_t>{1.5, 0.5, 3.0, 1.25};
    const auto g_vals = std::vector<float64_t>{1.5, -0.5, 0.75, 2.0};
    const auto b_vals = std::vector<float64_t>{0.25, -1.0, 0.5, -0.25};
    const auto w_vals = dy_weights(n_size * channels);
    for (auto c = size_t{0}; c < channels; ++c) {
        set_value(gamma, {c}, g_vals[c]);
        set_value(beta, {c}, b_vals[c]);
        set_value(running_mean, {c}, rm_vals[c]);
        set_value(running_var, {c}, rv_vals[c]);
    }
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            set_value(w, {n, c}, w_vals[n * channels + c]);
        }
    }
    auto out = x.batch_norm(gamma, beta, running_mean, running_var, false, 0.1, eps);
    // Named locals: graph parents are raw pointers (the eval node's _saved
    // pointers target the running buffers), so the whole graph must outlive
    // backward().
    auto weighted = out.mul(w);
    auto loss = weighted.sum();
    loss.backward(DType::Float64);
    // Closed form, same op order the eval node pins: dx[i] =
    // (gamma[c] * invstd[c]) * dy[i] with invstd = 1/sqrt(running_var[c] + eps)
    // and dy = w.
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            const auto invstd = float64_t{1.0} / std::sqrt(rv_vals[c] + eps);
            const auto want = (g_vals[c] * invstd) * w_vals[n * channels + c];
            CHECK(rel_near(x.gradient().at<float64_t>({n, c}), want, 1e-9));
        }
    }
}

static void gpu_eval_backward_matches_numeric_gradient() {
    const auto n_size = size_t{3};
    const auto channels = size_t{4};
    const auto eps = float64_t{1e-5};
    const auto h = float64_t{1e-5};
    auto x = Tensor({n_size, channels}, DType::Float64, Device::GPU, true);
    static_cast<void>(fill_rank2(x, n_size, channels));
    auto gamma = Tensor({channels}, DType::Float64, Device::GPU, true);
    auto beta = Tensor({channels}, DType::Float64, Device::GPU, true);
    auto running_mean = Tensor({channels}, DType::Float64, Device::GPU);
    auto running_var = Tensor({channels}, DType::Float64, Device::GPU);
    auto w = Tensor({n_size, channels}, DType::Float64, Device::GPU);
    const auto rm_vals = std::vector<float64_t>{0.5, -1.0, 2.0, 0.25};
    const auto rv_vals = std::vector<float64_t>{1.5, 0.5, 3.0, 1.25};
    const auto g_vals = std::vector<float64_t>{1.5, -0.5, 0.75, 2.0};
    const auto b_vals = std::vector<float64_t>{0.25, -1.0, 0.5, -0.25};
    const auto w_vals = dy_weights(n_size * channels);
    for (auto c = size_t{0}; c < channels; ++c) {
        set_value(gamma, {c}, g_vals[c]);
        set_value(beta, {c}, b_vals[c]);
        set_value(running_mean, {c}, rm_vals[c]);
        set_value(running_var, {c}, rv_vals[c]);
    }
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            set_value(w, {n, c}, w_vals[n * channels + c]);
        }
    }
    auto out = x.batch_norm(gamma, beta, running_mean, running_var, false, 0.1, eps);
    // Named locals: graph parents are raw pointers (the eval node's _saved
    // pointers target the running buffers), so the whole graph must outlive
    // backward().
    auto weighted = out.mul(w);
    auto loss = weighted.sum();
    loss.backward(DType::Float64);
    // fp64 device-readback eval loss: the eval forward leaves the running
    // buffers untouched, so repeated forwards in the perturbation loop
    // observe identical statistics.
    auto device_loss = [&]() {
        auto o = x.batch_norm(gamma, beta, running_mean, running_var, false, 0.1, eps);
        auto total = float64_t{0.0};
        for (auto n = size_t{0}; n < n_size; ++n) {
            for (auto c = size_t{0}; c < channels; ++c) {
                total += o.at<float64_t>({n, c}) * w_vals[n * channels + c];
            }
        }
        return total;
    };
    for (auto n = size_t{0}; n < n_size; ++n) {
        for (auto c = size_t{0}; c < channels; ++c) {
            const auto orig = x.at<float64_t>({n, c});
            x.set<float64_t>({n, c}, orig + h);
            const auto up = device_loss();
            x.set<float64_t>({n, c}, orig - h);
            const auto down = device_loss();
            x.set<float64_t>({n, c}, orig);
            const auto numeric = (up - down) / (2.0 * h);
            CHECK(rel_near(x.gradient().at<float64_t>({n, c}), numeric, 1e-5));
        }
    }
    for (auto c = size_t{0}; c < channels; ++c) {
        const auto orig_g = gamma.at<float64_t>({c});
        gamma.set<float64_t>({c}, orig_g + h);
        const auto up_g = device_loss();
        gamma.set<float64_t>({c}, orig_g - h);
        const auto down_g = device_loss();
        gamma.set<float64_t>({c}, orig_g);
        const auto numeric_g = (up_g - down_g) / (2.0 * h);
        CHECK(rel_near(gamma.gradient().at<float64_t>({c}), numeric_g, 1e-5));
        const auto orig_b = beta.at<float64_t>({c});
        beta.set<float64_t>({c}, orig_b + h);
        const auto up_b = device_loss();
        beta.set<float64_t>({c}, orig_b - h);
        const auto down_b = device_loss();
        beta.set<float64_t>({c}, orig_b);
        const auto numeric_b = (up_b - down_b) / (2.0 * h);
        CHECK(rel_near(beta.gradient().at<float64_t>({c}), numeric_b, 1e-5));
    }
}

#endif

int main() {
    train_forward_rank2_matches_reference();
    train_forward_rank4_matches_reference();
    train_forward_fp64_matches_tightly();
    running_stats_update_exact();
    batch_of_one_in_train_is_defined();
    no_grad_inputs_still_update_running_stats();
    fp16_and_bf16_forward_smoke();
    eval_forward_uses_running_stats();
    eval_forward_leaves_running_stats_untouched();
    eval_after_train_without_sync();
    train_backward_matches_numeric_gradient_rank2();
    train_backward_matches_numeric_gradient_rank4();
    train_backward_respects_grad_dtype();
    eval_backward_dx_is_gamma_invstd_dy();
    eval_backward_matches_numeric_gradient();
    errors();
    module_train_forward_and_param_grads();
    module_eval_uses_running_stats();
    module_train_eval_switch_restores_batch_stats();
    module_move_then_backward_keeps_saved_pointers_valid();
    module_move_assignment_preserves_eval_mode();
    module_constructor_validation();
    module_print_includes_batchnorm();
#if defined(USE_CUDA) || defined(USE_HIP)
    if (gpu_available()) {
        std::cout << "GPU backend detected: running Device::GPU batch_norm tests\n";
        gpu_train_forward_rank2_matches_reference();
        gpu_train_forward_rank4_matches_reference();
        gpu_running_stats_update_matches_reference();
        gpu_eval_after_train_without_sync();
        gpu_fp16_bf16_forward_smoke();
        gpu_train_backward_matches_numeric_gradient_rank2();
        gpu_train_backward_matches_numeric_gradient_rank4();
        gpu_train_backward_respects_grad_dtype();
        gpu_eval_backward_dx_is_gamma_invstd_dy();
        gpu_eval_backward_matches_numeric_gradient();
    }
#endif
    if (failures == 0) {
        std::cout << "ALL BATCHNORM TESTS PASSED\n";
        return 0;
    }
    std::cerr << failures << " FAILURE(S)\n";
    return 1;
}
