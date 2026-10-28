// BeanTensorV5 module tests — framework-free assertion suite.
// Covers: Module registry (named parameter access, named_parameters
// flattening, manual gradient access via Tensor::grad), Linear
// (forward = matmul + bias_add, bias-free variant, init bounds),
// Sequential (chained forward, empty forward throws EmptyModule, layer injection:
// append/insert/replace/remove mutate the registry AND forward),
// mse_loss -> backward -> Module::adam_step training loop (graph tensors
// kept alive across the loop), operator<< tree output, error paths, Relu
// (forward + gradient gating), softmax_cross_entropy (value and logit
// gradient vs an fp64 reference, and a softmax-trained classifier), and —
// when a backend + device exist — forward/backward on the GPU with
// CPU/GPU gradient parity.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "Helpers/Errors.h"
#include "Module/Linear.h"
#include "Module/Module.h"
#include "Module/ModuleStream.h"
#include "Module/Relu.h"
#include "Module/Sequential.h"
#include "Tensor/Tensor.h"

#if defined(USE_HIP)
#include <hip/hip_runtime.h>
#elif defined(USE_CUDA)
#include <cuda_runtime.h>
#endif

using BeanTensor::Modules::Linear;
using BeanTensor::Modules::Module;
using BeanTensor::Modules::Sequential;
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

// Mixed-precision comparisons happen at float64 through explicit casts, so no
// call site relies on an implicit float->double promotion.
template<typename A, typename B>
static bool near64(const A a, const B b, const float64_t eps) {
    return std::fabs(static_cast<float64_t>(a) - static_cast<float64_t>(b)) <= eps;
}

// Deterministically overwrite a {n_in, n_out} Linear weight (column-major
// value function v(in_idx, out_idx)) and optional bias.
static void set_linear_params(Module& lin, const size_t n_in, const size_t n_out,
                              const float64_t scale, const float64_t shift) {
    auto& w = lin.parameter("weight");
    for (auto i = size_t{0}; i < n_in; ++i) {
        for (auto j = size_t{0}; j < n_out; ++j) {
            w.set<float32_t>({i, j}, static_cast<float32_t>(scale * static_cast<float64_t>(i * n_out + j) + shift));
        }
    }
    try {
        auto& b = lin.parameter("bias");
        for (auto j = size_t{0}; j < n_out; ++j) {
            b.set<float32_t>({j}, static_cast<float32_t>(0.1 * static_cast<float64_t>(j) - 0.2));
        }
    } catch (const BeanTensor::ErrorHandling::NameError&) {
        // bias-free layer: nothing to set
    }
}

// fp64 reference of out = x·w + b for a {m,k} x {k,n} weight and optional {n} bias.
static float64_t ref_linear(const std::vector<float64_t>& xv, const size_t k,
                            const std::vector<float64_t>& wv, const std::vector<float64_t>& bv,
                            const size_t row, const size_t col) {
    auto acc = 0.0;
    const auto n = wv.size() / k;
    for (auto t = size_t{0}; t < k; ++t) {
        acc = std::fma(xv[row * k + t], wv[t * n + col], acc);
    }
    if (!bv.empty()) {
        acc += bv[col];
    }
    return acc;
}

// =====================================================================
// Linear forward
// =====================================================================

static void test_linear_forward(const Device dev) {
    const auto m = size_t{2};
    const auto k = size_t{3};
    const auto n = size_t{2};
    auto lin = Linear(k, n, DType::Float32, dev, true, 42);
    set_linear_params(lin, k, n, 0.25, -0.5);
    auto& w = lin.parameter("weight");
    auto& b = lin.parameter("bias");
    auto x = Tensor({m, k}, DType::Float32, dev);
    const auto xv = std::vector<float64_t>{1.0, -2.0, 0.5, 0.25, 3.0, -1.0};
    for (auto i = size_t{0}; i < m * k; ++i) {
        x.set<float32_t>({i / k, i % k}, static_cast<float32_t>(xv[i]));
    }
    auto out = lin.forward(x);
    CHECK(out.is_same_shape(Tensor({m, n})));
    CHECK(out.get_dtype() == DType::Float32);
    auto wv = std::vector<float64_t>(k * n);
    auto bv = std::vector<float64_t>(n);
    for (auto i = size_t{0}; i < k; ++i) {
        for (auto j = size_t{0}; j < n; ++j) {
            wv[i * n + j] = static_cast<float64_t>(w.at<float32_t>({i, j}));
        }
    }
    for (auto j = size_t{0}; j < n; ++j) {
        bv[j] = static_cast<float64_t>(b.at<float32_t>({j}));
    }
    for (auto i = size_t{0}; i < m; ++i) {
        for (auto j = size_t{0}; j < n; ++j) {
            CHECK(near64(out.at<float32_t>({i, j}), ref_linear(xv, k, wv, bv, i, j), 1e-5));
        }
    }
    // Registry metadata.
    CHECK(lin.in_features() == k);
    CHECK(lin.out_features() == n);
    CHECK(lin.has_bias());
    CHECK(lin.name() == "Linear");
    const auto params = lin.named_parameters();
    CHECK(params.size() == 2);
    CHECK(params[0].first == "weight");
    CHECK(params[1].first == "bias");
}

static void test_linear_no_bias(const Device dev) {
    const auto k = size_t{4};
    const auto n = size_t{3};
    auto lin = Linear(k, n, DType::Float32, dev, false, 7);
    set_linear_params(lin, k, n, 0.1, 0.3);
    auto& w = lin.parameter("weight");
    auto x = Tensor({2, k}, DType::Float32, dev);
    const auto xv = std::vector<float64_t>{0.5, -1.0, 2.0, 1.5, -0.25, 0.0, 3.0, -2.0};
    for (auto i = size_t{0}; i < 2 * k; ++i) {
        x.set<float32_t>({i / k, i % k}, static_cast<float32_t>(xv[i]));
    }
    auto out = lin.forward(x);
    auto wv = std::vector<float64_t>(k * n);
    for (auto i = size_t{0}; i < k; ++i) {
        for (auto j = size_t{0}; j < n; ++j) {
            wv[i * n + j] = static_cast<float64_t>(w.at<float32_t>({i, j}));
        }
    }
    const auto bv = std::vector<float64_t>{};
    for (auto i = size_t{0}; i < 2; ++i) {
        for (auto j = size_t{0}; j < n; ++j) {
            CHECK(near64(out.at<float32_t>({i, j}), ref_linear(xv, k, wv, bv, i, j), 1e-5));
        }
    }
    CHECK(!lin.has_bias());
    const auto params = lin.named_parameters();
    CHECK(params.size() == 1);
    auto threw = false;
    try {
        static_cast<void>(lin.parameter("bias"));
    } catch (const BeanTensor::ErrorHandling::NameError&) {
        threw = true;
    }
    CHECK(threw);
}

// Init convention: values inside [-1/sqrt(in), 1/sqrt(in)], fp64 param likewise.
static void test_linear_init_bounds() {
    auto lin = Linear(16, 8, DType::Float32, Device::CPU, true, 123);
    const auto bound = 1.0 / std::sqrt(16.0);
    auto& w = lin.parameter("weight");
    for (auto i = size_t{0}; i < 16; ++i) {
        for (auto j = size_t{0}; j < 8; ++j) {
            const auto v = static_cast<float64_t>(w.at<float32_t>({i, j}));
            CHECK(v >= -bound && v <= bound);
        }
    }
    auto lin64 = Linear(4, 4, DType::Float64, Device::CPU, true, 123);
    const auto bound64 = 1.0 / std::sqrt(4.0);
    auto& w64 = lin64.parameter("weight");
    for (auto i = size_t{0}; i < 4; ++i) {
        for (auto j = size_t{0}; j < 4; ++j) {
            const auto v = w64.at<float64_t>({i, j});
            CHECK(v >= -bound64 && v <= bound64);
        }
    }
}

// =====================================================================
// Sequential forward + injection
// =====================================================================

static void test_sequential_forward(const Device dev) {
    auto net = Sequential();
    net.append("l1", std::make_unique<Linear>(3, 4, DType::Float32, dev, true, 1));
    net.append("l2", std::make_unique<Linear>(4, 2, DType::Float32, dev, true, 2));
    set_linear_params(net[0], 3, 4, 0.125, -0.25);
    set_linear_params(net[1], 4, 2, 0.5, 0.1);
    CHECK(net.size() == 2);
    CHECK(net.name() == "Sequential");
    CHECK(net.index_of("l2") == 1);
    CHECK(net.name_at(0) == "l1");
    CHECK(net.at("l1").name() == "Linear");
    auto x = Tensor({2, 3}, DType::Float32, dev);
    const auto xv = std::vector<float64_t>{1.0, 0.5, -1.5, 2.0, -0.25, 1.0};
    for (auto i = size_t{0}; i < 6; ++i) {
        x.set<float32_t>({i / 3, i % 3}, static_cast<float32_t>(xv[i]));
    }
    // fp64 reference of the full chain.
    auto w1 = std::vector<float64_t>(3 * 4);
    auto b1 = std::vector<float64_t>(4);
    auto w2 = std::vector<float64_t>(4 * 2);
    auto b2 = std::vector<float64_t>(2);
    for (auto i = size_t{0}; i < 3; ++i) {
        for (auto j = size_t{0}; j < 4; ++j) {
            w1[i * 4 + j] = static_cast<float64_t>(net[0].parameter("weight").at<float32_t>({i, j}));
        }
    }
    for (auto j = size_t{0}; j < 4; ++j) {
        b1[j] = static_cast<float64_t>(net[0].parameter("bias").at<float32_t>({j}));
    }
    for (auto i = size_t{0}; i < 4; ++i) {
        for (auto j = size_t{0}; j < 2; ++j) {
            w2[i * 2 + j] = static_cast<float64_t>(net[1].parameter("weight").at<float32_t>({i, j}));
        }
    }
    for (auto j = size_t{0}; j < 2; ++j) {
        b2[j] = static_cast<float64_t>(net[1].parameter("bias").at<float32_t>({j}));
    }
    auto out = net.forward(x);
    CHECK(out.is_same_shape(Tensor({2, 2})));
    for (auto i = size_t{0}; i < 2; ++i) {
        auto mid = std::vector<float64_t>(4);
        for (auto j = size_t{0}; j < 4; ++j) {
            mid[j] = ref_linear(xv, 3, w1, b1, i, j);
        }
        for (auto j = size_t{0}; j < 2; ++j) {
            auto acc = 0.0;
            for (auto t = size_t{0}; t < 4; ++t) {
                acc = std::fma(mid[t], w2[t * 2 + j], acc);
            }
            acc += b2[j];
            CHECK(near64(out.at<float32_t>({i, j}), acc, 1e-4));
        }
    }
    // Qualified parameter flattening: whole subtree, registry order.
    const auto params = net.named_parameters();
    CHECK(params.size() == 4);
    CHECK(params[0].first == "l1.weight");
    CHECK(params[1].first == "l1.bias");
    CHECK(params[2].first == "l2.weight");
    CHECK(params[3].first == "l2.bias");
}

static void test_sequential_empty(const Device dev) {
    auto net = Sequential();
    CHECK(net.empty());
    auto x = Tensor({2, 3}, DType::Float32, dev);
    x.set_zero();
    auto threw = false;
    try {
        auto bad = net.forward(x);
        static_cast<void>(bad);
    } catch (const BeanTensor::ErrorHandling::EmptyModule&) {
        threw = true;
    }
    CHECK(threw);
}

static void test_sequential_injection(const Device dev) {
    // Two identity-scaled layers: l1 doubles, l2 halves — forward is x until
    // injection changes the chain.
    auto net = Sequential();
    net.append("l1", std::make_unique<Linear>(2, 2, DType::Float32, dev, false, 3));
    net.append("l2", std::make_unique<Linear>(2, 2, DType::Float32, dev, false, 4));
    auto& w1 = net[0].parameter("weight");
    auto& w2 = net[1].parameter("weight");
    // l1 = 2*I
    w1.set<float32_t>({0, 0}, 2.0f);
    w1.set<float32_t>({1, 1}, 2.0f);
    w1.set<float32_t>({0, 1}, 0.0f);
    w1.set<float32_t>({1, 0}, 0.0f);
    // l2 = 0.5*I
    w2.set<float32_t>({0, 0}, 0.5f);
    w2.set<float32_t>({1, 1}, 0.5f);
    w2.set<float32_t>({0, 1}, 0.0f);
    w2.set<float32_t>({1, 0}, 0.0f);
    auto x = Tensor({1, 2}, DType::Float32, dev);
    x.set<float32_t>({0, 0}, 3.0f);
    x.set<float32_t>({0, 1}, -4.0f);
    auto out_before = net.forward(x);
    CHECK(near64(out_before.at<float32_t>({0, 0}), 3.0, 1e-5));
    CHECK(near64(out_before.at<float32_t>({0, 1}), -4.0, 1e-5));

    // insert at 0: a pass-through layer in front — registry and forward move.
    net.insert(0, "l0", std::make_unique<Linear>(2, 2, DType::Float32, dev, false, 5));
    auto& w0 = net[0].parameter("weight");
    w0.set<float32_t>({0, 0}, 1.0f);
    w0.set<float32_t>({1, 1}, 1.0f);
    w0.set<float32_t>({0, 1}, 0.0f);
    w0.set<float32_t>({1, 0}, 0.0f);
    CHECK(net.size() == 3);
    CHECK(net.index_of("l0") == 0);
    auto out_insert = net.forward(x);
    CHECK(near64(out_insert.at<float32_t>({0, 0}), 3.0, 1e-5));

    // replace l2 (the halver) with a tripler: same slot/name, new forward.
    auto replacement = std::make_unique<Linear>(2, 2, DType::Float32, dev, false, 6);
    auto& wr = replacement->parameter("weight");
    wr.set<float32_t>({0, 0}, 3.0f);
    wr.set<float32_t>({1, 1}, 3.0f);
    wr.set<float32_t>({0, 1}, 0.0f);
    wr.set<float32_t>({1, 0}, 0.0f);
    auto displaced = net.replace(net.index_of("l2"), std::move(replacement));
    CHECK(displaced != nullptr);
    CHECK(net.size() == 3);
    CHECK(net.name_at(2) == "l2"); // slot keeps its name
    auto out_replace = net.forward(x);
    // x -> I -> 2I -> 3I = 6x.
    CHECK(near64(out_replace.at<float32_t>({0, 0}), 18.0, 1e-4));
    CHECK(near64(out_replace.at<float32_t>({0, 1}), -24.0, 1e-4));

    // remove l0: back to the 2*3 = 6x chain... l1 doubles then l2 triples.
    auto removed = net.remove(0);
    CHECK(removed != nullptr);
    CHECK(net.size() == 2);
    CHECK(net.index_of("l1") == 0);
    auto out_remove = net.forward(x);
    CHECK(near64(out_remove.at<float32_t>({0, 0}), 18.0, 1e-4));

    // Auto-naming on empty names: next index.
    auto net2 = Sequential();
    net2.append("", std::make_unique<Linear>(2, 2, DType::Float32, dev, false, 9));
    net2.append("", std::make_unique<Linear>(2, 2, DType::Float32, dev, false, 10));
    CHECK(net2.name_at(0) == "0");
    CHECK(net2.name_at(1) == "1");
}

static void test_registry_errors() {
    auto net = Sequential();
    net.append("a", std::make_unique<Linear>(2, 2, DType::Float32, Device::CPU, false, 11));
    auto threw = false;
    try {
        net.append("a", std::make_unique<Linear>(2, 2, DType::Float32, Device::CPU, false, 12));
    } catch (const BeanTensor::ErrorHandling::NameError&) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        net.insert(9, "b", std::make_unique<Linear>(2, 2, DType::Float32, Device::CPU, false, 13));
    } catch (const BeanTensor::ErrorHandling::IndexOutOfRange&) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        static_cast<void>(net[5]);
    } catch (const BeanTensor::ErrorHandling::IndexOutOfRange&) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        static_cast<void>(net.at("nope"));
    } catch (const BeanTensor::ErrorHandling::NameError&) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        static_cast<void>(net.index_of("nope"));
    } catch (const BeanTensor::ErrorHandling::NameError&) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        static_cast<void>(net.remove(0));
        static_cast<void>(net.remove(0));
    } catch (const BeanTensor::ErrorHandling::IndexOutOfRange&) {
        threw = true;
    }
    CHECK(threw);
    auto lin = Linear(2, 2, DType::Float32, Device::CPU, false, 14);
    threw = false;
    try {
        static_cast<void>(lin.parameter("nope"));
    } catch (const BeanTensor::ErrorHandling::NameError&) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        static_cast<void>(lin.grad("weight"));
    } catch (const BeanTensor::ErrorHandling::NoGradient&) {
        threw = true;
    }
    CHECK(threw);
}

// =====================================================================
// Backward through a Sequential + training loop
// =====================================================================

static void test_backward_grads(const Device dev) {
    // net: x(2x3) --l1--> h(2x4) --l2--> out(2x1); mse against target.
    auto net = Sequential();
    net.append("l1", std::make_unique<Linear>(3, 4, DType::Float32, dev, true, 21));
    net.append("l2", std::make_unique<Linear>(4, 1, DType::Float32, dev, true, 22));
    set_linear_params(net[0], 3, 4, 0.2, -0.1);
    set_linear_params(net[1], 4, 1, 0.3, 0.05);
    auto x = Tensor({2, 3}, DType::Float32, dev, false);
    auto target = Tensor({2, 1}, DType::Float32, dev, false);
    const auto xv = std::vector<float64_t>{1.0, 0.5, -1.0, 0.25, -0.5, 2.0};
    const auto tv = std::vector<float64_t>{0.5, -1.5};
    for (auto i = size_t{0}; i < 6; ++i) {
        x.set<float32_t>({i / 3, i % 3}, static_cast<float32_t>(xv[i]));
    }
    for (auto i = size_t{0}; i < 2; ++i) {
        target.set<float32_t>({i, 0}, static_cast<float32_t>(tv[i]));
    }
    auto out = net.forward(x);
    auto loss = BeanTensor::Modules::mse_loss(out, target);
    loss.backward(DType::Float32);
    // Every parameter on the path holds a grad now.
    for (const auto& [pname, p] : net.named_parameters()) {
        CHECK(p->has_grad());
        CHECK(p->gradient().get_dtype() == DType::Float32);
        static_cast<void>(pname);
    }
    // fp64 reference: dW2 = h^T·dout, db2 = colsum(dout), dout1 = dout·w2^T,
    // dW1 = x^T·dout1, db1 = colsum(dout1); dout = 2*(out - t)/2 = (out - t).
    auto w1 = std::vector<float64_t>(3 * 4);
    auto b1 = std::vector<float64_t>(4);
    auto w2 = std::vector<float64_t>(4);
    auto b2v = std::vector<float64_t>(1);
    for (auto i = size_t{0}; i < 3; ++i) {
        for (auto j = size_t{0}; j < 4; ++j) {
            w1[i * 4 + j] = static_cast<float64_t>(net[0].parameter("weight").at<float32_t>({i, j}));
        }
    }
    for (auto j = size_t{0}; j < 4; ++j) {
        b1[j] = static_cast<float64_t>(net[0].parameter("bias").at<float32_t>({j}));
        w2[j] = static_cast<float64_t>(net[1].parameter("weight").at<float32_t>({j, 0}));
    }
    b2v[0] = static_cast<float64_t>(net[1].parameter("bias").at<float32_t>({0}));
    auto h = std::vector<float64_t>(8);
    auto dout = std::vector<float64_t>(2);
    for (auto i = size_t{0}; i < 2; ++i) {
        for (auto j = size_t{0}; j < 4; ++j) {
            h[i * 4 + j] = ref_linear(xv, 3, w1, b1, i, j);
        }
        auto o = 0.0;
        for (auto t = size_t{0}; t < 4; ++t) {
            o = std::fma(h[i * 4 + t], w2[t], o);
        }
        o += b2v[0];
        dout[i] = 2.0 * (o - tv[i]) / 2.0;
    }
    // dW2 (4x1), db2 — read back through Module::grad (manual gradient access).
    auto& dw2 = net[1].grad("weight");
    auto& db2 = net[1].grad("bias");
    for (auto j = size_t{0}; j < 4; ++j) {
        const auto want = h[0 * 4 + j] * dout[0] + h[1 * 4 + j] * dout[1];
        CHECK(near64(dw2.at<float32_t>({j, 0}), want, 1e-4));
    }
    CHECK(near64(db2.at<float32_t>({0}), dout[0] + dout[1], 1e-4));
    // dW1 (3x4), db1.
    auto& dw1 = net[0].grad("weight");
    auto& db1 = net[0].grad("bias");
    for (auto i = size_t{0}; i < 3; ++i) {
        for (auto j = size_t{0}; j < 4; ++j) {
            const auto want = xv[i] * dout[0] * w2[j] + xv[3 + i] * dout[1] * w2[j];
            CHECK(near64(dw1.at<float32_t>({i, j}), want, 1e-4));
        }
    }
    for (auto j = size_t{0}; j < 4; ++j) {
        const auto want = dout[0] * w2[j] + dout[1] * w2[j];
        CHECK(near64(db1.at<float32_t>({j}), want, 1e-4));
    }
    // Second backward accumulates into the same leaf grads (doubled). The
    // forward output is bound to a named local: the graph borrows its
    // address, so a temporary would dangle by the time backward() runs.
    auto out2 = net.forward(x);
    auto loss2 = BeanTensor::Modules::mse_loss(out2, target);
    loss2.backward(DType::Float32);
    CHECK(near64(net[1].grad("bias").at<float32_t>({0}), 2.0 * (dout[0] + dout[1]), 1e-3));
}

static void test_training_loop(const Device dev) {
    // Fit y = 2x + 1 through Linear(1,8) -> Linear(8,1) with the module
    // conveniences: mse_loss, backward, Module::adam_step, zero_grad.
    // Graph tensors (x, target) and the net itself stay alive for the whole
    // loop — backward borrows raw pointers into them.
    const auto n = size_t{16};
    auto x = Tensor({n, 1}, DType::Float32, dev, false);
    auto target = Tensor({n, 1}, DType::Float32, dev, false);
    for (auto i = size_t{0}; i < n; ++i) {
        const auto xi = static_cast<float64_t>(i) / static_cast<float64_t>(n - 1);
        x.set<float32_t>({i, 0}, static_cast<float32_t>(xi));
        target.set<float32_t>({i, 0}, static_cast<float32_t>(2.0 * xi + 1.0));
    }
    auto net = Sequential();
    net.append("l1", std::make_unique<Linear>(1, 8, DType::Float32, dev, true, 31));
    net.append("l2", std::make_unique<Linear>(8, 1, DType::Float32, dev, true, 32));
    auto first = float64_t{0.0};
    for (auto step = size_t{0}; step < 400; ++step) {
        auto out = net.forward(x);
        auto loss = BeanTensor::Modules::mse_loss(out, target);
        if (step == 0) {
            first = static_cast<float64_t>(loss.at<float32_t>({0}));
        }
        loss.backward(DType::Float32);
        net.adam_step(0.05, 0.9, 0.999, 1e-8);
        net.zero_grad();
    }
    auto out = net.forward(x);
    auto loss = BeanTensor::Modules::mse_loss(out, target);
    loss.sync();
    const auto fin = static_cast<float64_t>(loss.at<float32_t>({0}));
    CHECK(fin < first);
    CHECK(fin < 1e-3);
    // The fit itself landed.
    for (auto i = size_t{0}; i < n; ++i) {
        const auto xi = static_cast<float64_t>(i) / static_cast<float64_t>(n - 1);
        CHECK(near64(out.at<float32_t>({i, 0}), 2.0 * xi + 1.0, 5e-2));
    }
}

// =====================================================================
// Printing
// =====================================================================

static void test_print() {
    auto net = Sequential();
    net.append("l1", std::make_unique<Linear>(3, 4, DType::Float32, Device::CPU, true, 41));
    net.append("l2", std::make_unique<Linear>(4, 2, DType::Float64, Device::CPU, false, 42));
    auto os = std::ostringstream();
    os << net;
    const auto s = os.str();
    CHECK(s.find("Sequential") != std::string::npos);
    CHECK(s.find("[l1] Linear") != std::string::npos);
    CHECK(s.find("[l2] Linear") != std::string::npos);
    CHECK(s.find("weight: Float32 {3, 4}") != std::string::npos);
    CHECK(s.find("bias: Float32 {4}") != std::string::npos);
    CHECK(s.find("weight: Float64 {4, 2}") != std::string::npos);
    // l2 has no bias line.
    const auto l2_pos = s.find("[l2]");
    CHECK(l2_pos != std::string::npos);
    CHECK(s.find("bias", l2_pos) == std::string::npos);
    // A bare Linear prints its own registry.
    auto lin = Linear(2, 3, DType::Float32, Device::CPU, true, 43);
    auto os2 = std::ostringstream();
    os2 << lin;
    const auto s2 = os2.str();
    CHECK(s2.find("Linear") != std::string::npos);
    CHECK(s2.find("weight: Float32 {2, 3}") != std::string::npos);
}

// =====================================================================
// Relu and softmax_cross_entropy
// =====================================================================

static void test_relu_forward(const Device dev) {
    // Forward keeps positives and zeroes the rest; backward gates the
    // gradient through x > 0 (relu_G: dx = x > 0 ? dy : 0). With
    // mse_loss against zeros, dy = 2*y/numel = y/3 for numel 6, so the
    // expected x-gradient is relu(x)/3.
    auto relu = BeanTensor::Modules::Relu();
    auto x = Tensor({2, 3}, DType::Float32, dev, true);
    auto target = Tensor({2, 3}, DType::Float32, dev, false);
    target.set_zero();
    const auto xv = std::vector<float64_t>{1.0, -2.0, 0.5, -0.25, 3.0, 0.0};
    for (auto i = size_t{0}; i < 6; ++i) {
        x.set<float32_t>({i / 3, i % 3}, static_cast<float32_t>(xv[i]));
    }
    auto y = relu.forward(x);
    auto loss = BeanTensor::Modules::mse_loss(y, target);
    loss.backward(DType::Float32);
    loss.sync();
    for (auto i = size_t{0}; i < 6; ++i) {
        const auto expect_y = xv[i] > 0.0 ? xv[i] : 0.0;
        CHECK(near64(y.at<float32_t>({i / 3, i % 3}), expect_y, 1e-6));
        CHECK(x.has_grad());
        CHECK(near64(x.gradient().at<float32_t>({i / 3, i % 3}), expect_y / 3.0, 1e-6));
    }
    CHECK(relu.name() == "Relu");
    CHECK(relu.named_parameters().empty());
}

static void test_softmax_cross_entropy(const Device dev) {
    // loss = mean over rows of (logsumexp(x_r) - x_r[label_r]); backward
    // fills dlogits = (softmax(x) - onehot(label)) / rows. Checked against
    // an fp64 reference computed here.
    const auto rows = size_t{2};
    const auto classes = size_t{3};
    auto logits = Tensor({rows, classes}, DType::Float32, dev, true);
    auto labels = Tensor({rows}, DType::Int64, dev, false);
    const auto lv = std::vector<float64_t>{0.5, -1.0, 2.0, 1.5, 0.0, -0.5};
    const auto label_of = std::vector<size_t>{2, 0};
    for (auto i = size_t{0}; i < rows * classes; ++i) {
        logits.set<float32_t>({i / classes, i % classes}, static_cast<float32_t>(lv[i]));
    }
    for (auto r = size_t{0}; r < rows; ++r) {
        labels.set<int64_t>({r}, static_cast<int64_t>(label_of[r]));
    }

    auto ref_loss = float64_t{0.0};
    auto ref_grad = std::vector<float64_t>(rows * classes);
    for (auto r = size_t{0}; r < rows; ++r) {
        auto m = lv[r * classes];
        for (auto c = size_t{0}; c < classes; ++c) {
            m = std::max(m, lv[r * classes + c]);
        }
        auto denom = float64_t{0.0};
        for (auto c = size_t{0}; c < classes; ++c) {
            denom += std::exp(lv[r * classes + c] - m);
        }
        ref_loss += (m + std::log(denom) - lv[r * classes + label_of[r]]) / static_cast<float64_t>(rows);
        for (auto c = size_t{0}; c < classes; ++c) {
            const auto p = std::exp(lv[r * classes + c] - m) / denom;
            const auto onehot = c == label_of[r] ? float64_t{1.0} : float64_t{0.0};
            ref_grad[r * classes + c] = (p - onehot) / static_cast<float64_t>(rows);
        }
    }

    auto loss = BeanTensor::Modules::softmax_cross_entropy(logits, labels);
    loss.backward(DType::Float32);
    loss.sync();
    CHECK(near64(loss.at<float32_t>({0}), ref_loss, 1e-5));
    CHECK(logits.has_grad());
    for (auto i = size_t{0}; i < rows * classes; ++i) {
        CHECK(near64(logits.gradient().at<float32_t>({i / classes, i % classes}), ref_grad[i], 1e-5));
    }
}

static void test_softmax_training_loop(const Device dev) {
    // Three well-separated 2-D clusters, four samples each, classified by
    // Linear(2,8) -> Relu -> Linear(8,3) trained with softmax_cross_entropy
    // and Module::adam_step. Graph tensors stay alive across the loop.
    const auto n = size_t{12};
    const auto cx = std::vector<float64_t>{-1.0, 1.0, 0.0};
    const auto cy = std::vector<float64_t>{-1.0, -1.0, 1.0};
    auto x = Tensor({n, 2}, DType::Float32, dev, false);
    auto labels = Tensor({n}, DType::Int64, dev, false);
    for (auto i = size_t{0}; i < n; ++i) {
        const auto c = i / 4;
        const auto jitter_x = 0.1 * static_cast<float64_t>(i % 4) - 0.15;
        const auto jitter_y = 0.1 * static_cast<float64_t>(i % 2) - 0.05;
        x.set<float32_t>({i, 0}, static_cast<float32_t>(cx[c] + jitter_x));
        x.set<float32_t>({i, 1}, static_cast<float32_t>(cy[c] + jitter_y));
        labels.set<int64_t>({i}, static_cast<int64_t>(c));
    }

    auto net = Sequential();
    net.append("fc1", std::make_unique<Linear>(2, 8, DType::Float32, dev, true, 41));
    net.append("relu", std::make_unique<BeanTensor::Modules::Relu>());
    net.append("fc2", std::make_unique<Linear>(8, 3, DType::Float32, dev, true, 42));
    auto first = float64_t{0.0};
    for (auto step = size_t{0}; step < 300; ++step) {
        auto out = net.forward(x);
        auto loss = BeanTensor::Modules::softmax_cross_entropy(out, labels);
        if (step == 0) {
            first = static_cast<float64_t>(loss.at<float32_t>({0}));
        }
        loss.backward(DType::Float32);
        net.adam_step(0.05, 0.9, 0.999, 1e-8);
        net.zero_grad();
    }
    auto out = net.forward(x);
    auto loss = BeanTensor::Modules::softmax_cross_entropy(out, labels);
    loss.sync();
    const auto fin = static_cast<float64_t>(loss.at<float32_t>({0}));
    CHECK(fin < first);
    CHECK(fin < 0.05);
    auto predictions = out.argmax(1);
    for (auto i = size_t{0}; i < n; ++i) {
        CHECK(predictions.at<int64_t>({i}) == static_cast<int64_t>(i / 4));
    }
}

// =====================================================================
// Byte and element counts
// =====================================================================

static void test_byte_counts(const Device dev) {
    // Tensor level: element_count is the product of the shape, byte_count is
    // elements times the dtype's element size.
    auto t64 = Tensor({2, 3}, DType::Float64, dev, false);
    CHECK(t64.element_count() == 6);
    CHECK(t64.byte_count() == 48);
    auto ti = Tensor({5}, DType::Int64, dev, false);
    CHECK(ti.element_count() == 5);
    CHECK(ti.byte_count() == 40);
    // Module level: each parameter counts at its own dtype.
    // Linear(3,4, Float64, bias): weight 12 + bias 4 = 16 params x 8 B = 128 B.
    // Linear(4,2, Float32, no bias): weight 8 params x 4 B = 32 B.
    auto net = Sequential();
    net.append("l1", std::make_unique<Linear>(3, 4, DType::Float64, dev, true, 51));
    net.append("l2", std::make_unique<Linear>(4, 2, DType::Float32, dev, false, 52));
    CHECK(net.parameter_count() == 24);
    CHECK(net.parameter_bytes() == 160);
    CHECK(net[0].parameter_count() == 16);
    CHECK(net[0].parameter_bytes() == 128);
    CHECK(net[1].parameter_count() == 8);
    CHECK(net[1].parameter_bytes() == 32);
    // A container with no children holds no parameters.
    const auto empty = Sequential();
    CHECK(empty.parameter_count() == 0);
    CHECK(empty.parameter_bytes() == 0);
}

static void test_memory_accounting() {
    // Linear(3,4, Float64, bias): 16 parameters x 8 B. A gradient exists only
    // after backward(), Adam state only after adam_step(), and zero_grad()
    // frees the gradient while leaving the Adam moments in place.
    // First, a module that is stepped and then destroyed. Its optimizer state
    // must go with it, so a later module that reuses those addresses starts
    // with none (regression: the Adam map is keyed by parameter address).
    {
        auto gone = Sequential();
        gone.append("l1", std::make_unique<Linear>(3, 4, DType::Float64, Device::CPU, true, 71));
        auto gx = Tensor({2, 3}, DType::Float64, Device::CPU, false);
        gx.set_one();
        auto gt = Tensor({2, 4}, DType::Float64, Device::CPU, false);
        gt.set_zero();
        auto gout = gone.forward(gx);
        auto gloss = BeanTensor::Modules::mse_loss(gout, gt);
        gloss.backward(DType::Float64);
        gone.adam_step(0.01, 0.9, 0.999, 1e-8);
        CHECK(gone.optimizer_state_bytes() == 256);
    }
    auto net = Sequential();
    net.append("l1", std::make_unique<Linear>(3, 4, DType::Float64, Device::CPU, true, 61));
    CHECK(net.parameter_grad_bytes() == 0);
    CHECK(net.optimizer_state_bytes() == 0);
    auto x = Tensor({2, 3}, DType::Float64, Device::CPU, false);
    x.set_one();
    auto target = Tensor({2, 4}, DType::Float64, Device::CPU, false);
    target.set_zero();
    auto out = net.forward(x);
    auto loss = BeanTensor::Modules::mse_loss(out, target);
    loss.backward(DType::Float64);
    CHECK(net.parameter_grad_bytes() == 128);
    CHECK(net.optimizer_state_bytes() == 0);
    net.adam_step(0.01, 0.9, 0.999, 1e-8);
    // First and second moments: 2 x 128 B.
    CHECK(net.optimizer_state_bytes() == 256);
    net.zero_grad();
    CHECK(net.parameter_grad_bytes() == 0);
    CHECK(net.optimizer_state_bytes() == 256);

    // Formatting: thousands separators and binary units.
    CHECK(BeanTensor::Modules::format_count(0) == "0");
    CHECK(BeanTensor::Modules::format_count(999) == "999");
    CHECK(BeanTensor::Modules::format_count(1000) == "1,000");
    CHECK(BeanTensor::Modules::format_count(101770) == "101,770");
    CHECK(BeanTensor::Modules::format_bytes(0) == "0 B");
    CHECK(BeanTensor::Modules::format_bytes(1023) == "1023 B");
    CHECK(BeanTensor::Modules::format_bytes(1536) == "1.5 KB");
    CHECK(BeanTensor::Modules::format_bytes(203540) == "198.8 KB");
    CHECK(BeanTensor::Modules::format_bytes(size_t{1048576}) == "1.0 MB");
    CHECK(BeanTensor::Modules::format_bytes(size_t{3} * size_t{1024} * size_t{1024} * size_t{1024}) == "3.0 GB");
}

static void test_argmax_lanes() {
    // Float32 values convert to Float16x2 with both lanes equal, so each element
    // holds one value twice. Ties go to the first lane: class 2j.
    auto logits = Tensor({2, 3}, DType::Float32, Device::CPU, false);
    const auto v = std::vector<float64_t>{0.1, 0.5, 0.3, 0.9, 0.2, 0.2};
    for (auto i = size_t{0}; i < v.size(); ++i) {
        logits.set<float32_t>({i / 3, i % 3}, static_cast<float32_t>(v[i]));
    }
    logits.convert_dtype(DType::Float16x2);
    logits.sync();
    auto classes = logits.argmax_lanes(1);
    CHECK(classes.get_dtype() == DType::Int64);
    CHECK(classes.at<int64_t>({0}) == 2);
    CHECK(classes.at<int64_t>({1}) == 0);
    // Non-packed input is refused rather than read as lanes.
    auto plain = Tensor({2, 3}, DType::Float32, Device::CPU, false);
    auto threw = false;
    try {
        auto bad = plain.argmax_lanes(1);
        static_cast<void>(bad);
    } catch (const BeanTensor::ErrorHandling::MismatchedDTypes&) {
        threw = true;
    }
    CHECK(threw);
}

int main() {
    test_linear_forward(Device::CPU);
    test_linear_no_bias(Device::CPU);
    test_linear_init_bounds();
    test_sequential_forward(Device::CPU);
    test_sequential_empty(Device::CPU);
    test_sequential_injection(Device::CPU);
    test_registry_errors();
    test_backward_grads(Device::CPU);
    test_training_loop(Device::CPU);
    test_print();
    test_relu_forward(Device::CPU);
    test_softmax_cross_entropy(Device::CPU);
    test_softmax_training_loop(Device::CPU);
    test_byte_counts(Device::CPU);
    test_memory_accounting();
    test_argmax_lanes();
#if defined(USE_HIP) || defined(USE_CUDA)
    if (gpu_available()) {
        test_linear_forward(Device::GPU);
        test_linear_no_bias(Device::GPU);
        test_sequential_forward(Device::GPU);
        test_sequential_empty(Device::GPU);
        test_backward_grads(Device::GPU);
        test_training_loop(Device::GPU);
        test_relu_forward(Device::GPU);
        test_softmax_cross_entropy(Device::GPU);
        test_softmax_training_loop(Device::GPU);
    } else {
        std::cout << "GPU not available — skipping GPU module tests\n";
    }
#endif

    if (failures == 0) {
        std::cout << "ALL MODULE TESTS PASSED\n";
        return 0;
    }
    std::cerr << failures << " FAILURE(S)\n";
    return 1;
}
