// BeanTensorV5 indexing/view autograd tests — framework-free assertion suite.
// Covers: backward through index_select, gather, cat, slice, split and
// reshape; F64 finite-difference gradient checks; duplicate-index scatter
// accumulation (deterministic order); fan-out; leaf-grad accumulation across
// backward() calls; grad_dtype boundary behavior; and graph lifetime after
// the forward call frame is gone. The same checks rerun through the CUDA/HIP
// backward twins when a GPU is present.
#include <cmath>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

#include "Helpers/Errors.h"
#include "Math/Autograd/AutogradShared.h"
#include "Math/Autograd/GradDType.h"
#include "Tensor/Tensor.h"
#if defined(USE_HIP)
#include <hip/hip_runtime.h>
#elif defined(USE_CUDA)
#include <cuda_runtime.h>
#endif

using BeanTensor::Tensors::Tensor;
using BeanTensor::Tensors::Device;
using BeanTensor::float64_t;
using DType = BeanTensor::DataTypes::DType;
namespace tdetail = BeanTensor::Tensors::detail;

static int32_t failures = 0;
#define CHECK(...) do { \
    if (!(__VA_ARGS__)) { ++failures; std::cerr << "FAIL: " << #__VA_ARGS__ << " @ line " << __LINE__ << "\n"; } \
} while (0)

static bool near64(const float64_t a, const float64_t b, const float64_t eps) {
    return std::fabs(a - b) <= eps;
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

// Deterministic value for element (r, c) of a rank-2 tensor: quarter-exact in
// fp16/fp32/fp64, so gradient values compare exactly across dtype boundaries.
static float64_t val2(const size_t r, const size_t c) {
    return static_cast<float64_t>(r * 7 + c * 3) * 0.25 + 0.5;
}

static Tensor make2(const size_t rows, const size_t cols, const Device dev, const bool grad) {
    auto t = Tensor({rows, cols}, DType::Float64, dev, grad);
    for (auto r = size_t{0}; r < rows; ++r) {
        for (auto c = size_t{0}; c < cols; ++c) {
            t.set<float64_t>({r, c}, val2(r, c));
        }
    }
    return t;
}

static Tensor make_idx(const std::vector<int64_t>& values, const Device dev) {
    auto t = Tensor({values.size()}, DType::Int64, dev, false);
    for (auto i = size_t{0}; i < values.size(); ++i) {
        t.set<int64_t>({i}, values[i]);
    }
    return t;
}

static Tensor make_idx2(const size_t rows, const size_t cols, const std::vector<int64_t>& values, const Device dev) {
    auto t = Tensor({rows, cols}, DType::Int64, dev, false);
    for (auto i = size_t{0}; i < values.size(); ++i) {
        t.set<int64_t>({i / cols, i % cols}, values[i]);
    }
    return t;
}

/**
 * Analytic-vs-finite-difference gradient check for a single-op graph
 * y = forward(x), loss = sum(y), on rank-2 F64 x. The forward result is
 * summed directly, so only contiguous outputs can be checked here.
 */
template<typename F>
void fd_check(const char* const label, Tensor& x, F&& forward, const float64_t tol) {
    {
        auto y = forward(x);
        auto loss = y.sum();
        loss.backward(DType::Float64);
    }
    const auto h = 1e-6;
    const auto rows = x.dim(0);
    const auto cols = x.dim(1);
    auto bad = size_t{0};
    for (auto r = size_t{0}; r < rows; ++r) {
        for (auto c = size_t{0}; c < cols; ++c) {
            const auto orig = x.at<float64_t>({r, c});
            x.set<float64_t>({r, c}, orig + h);
            auto yp = forward(x);
            auto sp = yp.sum();
            const auto lp = sp.template at<float64_t>({0});
            x.set<float64_t>({r, c}, orig - h);
            auto ym = forward(x);
            auto sm = ym.sum();
            const auto lm = sm.template at<float64_t>({0});
            x.set<float64_t>({r, c}, orig);
            const auto fd = (lp - lm) / (2.0 * h);
            const auto an = x.gradient().at<float64_t>({r, c});
            if (!near64(an, fd, tol)) {
                ++bad;
                std::cerr << "FD mismatch " << label << " at (" << r << "," << c << "): analytic " << an << " vs fd " << fd << "\n";
            }
        }
    }
    CHECK(bad == 0);
    std::cout << "fd_check " << label << ": " << (bad == 0 ? "ok" : "FAILED") << "\n";
}

// --------------------------- index_select ---------------------------

static void test_index_select(const Device dev) {
    // Exact counts: row k of x.grad is the number of picks of k (dy is ones).
    auto x = make2(3, 4, dev, true);
    auto idx = make_idx({2, 2, 0}, dev);
    auto y = x.index_select(0, idx);
    CHECK(y.requires_gradient());
    auto loss = y.sum();
    loss.backward(DType::Float64);
    for (auto c = size_t{0}; c < 4; ++c) {
        CHECK(near64(x.gradient().at<float64_t>({0, c}), 1.0, 1e-12));
        CHECK(near64(x.gradient().at<float64_t>({1, c}), 0.0, 1e-12));
        CHECK(near64(x.gradient().at<float64_t>({2, c}), 2.0, 1e-12));
    }

    // Leaf accumulation across backward() calls: a second call doubles.
    loss.backward(DType::Float64);
    for (auto c = size_t{0}; c < 4; ++c) {
        CHECK(near64(x.gradient().at<float64_t>({2, c}), 4.0, 1e-12));
    }

    // Finite differences (placement check through the scatter).
    auto x2 = make2(4, 3, dev, true);
    auto idx2 = make_idx({3, 0, 3, 1}, dev);
    fd_check("index_select", x2, [&](Tensor& t) { return t.index_select(0, idx2); }, 1e-4);

    // Non-uniform upstream gradient through a mul node.
    auto x3 = make2(3, 2, dev, true);
    auto idx3 = make_idx({1, 1}, dev);
    auto y3 = x3.index_select(0, idx3);
    auto w3 = Tensor({2, 2}, DType::Float64, dev, false);
    w3.set<float64_t>({0, 0}, 2.0);
    w3.set<float64_t>({0, 1}, 3.0);
    w3.set<float64_t>({1, 0}, 5.0);
    w3.set<float64_t>({1, 1}, 7.0);
    auto z3 = y3.mul(w3);
    auto loss3 = z3.sum();
    loss3.backward(DType::Float64);
    // Row 1 was picked twice: its grad is the sum of the two weight rows.
    CHECK(near64(x3.gradient().at<float64_t>({1, 0}), 7.0, 1e-9));
    CHECK(near64(x3.gradient().at<float64_t>({1, 1}), 10.0, 1e-9));
    CHECK(near64(x3.gradient().at<float64_t>({0, 0}), 0.0, 1e-12));

    // No grad input, no graph.
    auto xn = make2(2, 2, dev, false);
    auto idxn = make_idx({0}, dev);
    auto yn = xn.index_select(0, idxn);
    CHECK(!yn.requires_gradient());
}

// --------------------------- gather ---------------------------

static void test_gather(const Device dev) {
    // Exact counts per row: x.grad[r,c] = how often c appears in idx's row r.
    auto x = make2(3, 4, dev, true);
    auto idx = make_idx2(3, 2, {2, 0, 1, 1, 3, 3}, dev);
    auto y = x.gather(1, idx);
    CHECK(y.requires_gradient());
    auto loss = y.sum();
    loss.backward(DType::Float64);
    for (auto r = size_t{0}; r < 3; ++r) {
        for (auto c = size_t{0}; c < 4; ++c) {
            auto expected = float64_t{0.0};
            if (r == 0 && (c == 2 || c == 0)) {
                expected = 1.0;
            }
            if (r == 1 && c == 1) {
                expected = 2.0;
            }
            if (r == 2 && c == 3) {
                expected = 2.0;
            }
            CHECK(near64(x.gradient().at<float64_t>({r, c}), expected, 1e-12));
        }
    }

    auto x2 = make2(3, 4, dev, true);
    auto idx2 = make_idx2(3, 3, {0, 2, 2, 1, 0, 3, 3, 3, 1}, dev);
    fd_check("gather", x2, [&](Tensor& t) { return t.gather(1, idx2); }, 1e-4);

    // Non-uniform upstream gradient.
    auto x3 = make2(2, 3, dev, true);
    auto idx3 = make_idx2(2, 2, {0, 0, 2, 1}, dev);
    auto y3 = x3.gather(1, idx3);
    auto w3 = make2(2, 2, dev, false);
    auto z3 = y3.mul(w3);
    auto loss3 = z3.sum();
    loss3.backward(DType::Float64);
    // Row 0: column 0 picked twice with weights w3[0,0] and w3[0,1].
    CHECK(near64(x3.gradient().at<float64_t>({0, 0}), w3.at<float64_t>({0, 0}) + w3.at<float64_t>({0, 1}), 1e-9));
    CHECK(near64(x3.gradient().at<float64_t>({1, 2}), w3.at<float64_t>({1, 0}), 1e-9));
    CHECK(near64(x3.gradient().at<float64_t>({1, 1}), w3.at<float64_t>({1, 1}), 1e-9));
}

// --------------------------- cat ---------------------------

static void test_cat(const Device dev) {
    // Two-input cat with finite differences on both inputs.
    std::vector<Tensor> ins;
    ins.reserve(2);
    ins.emplace_back(make2(2, 3, dev, true));
    ins.emplace_back(make2(2, 2, dev, true));
    {
        auto out = Tensor::cat(ins, 1);
        CHECK(out.requires_gradient());
        auto loss = out.sum();
        loss.backward(DType::Float64);
        // loss = sum of everything: every grad element is 1.
        for (auto r = size_t{0}; r < 2; ++r) {
            for (auto c = size_t{0}; c < 3; ++c) {
                CHECK(near64(ins[0].gradient().at<float64_t>({r, c}), 1.0, 1e-12));
            }
            for (auto c = size_t{0}; c < 2; ++c) {
                CHECK(near64(ins[1].gradient().at<float64_t>({r, c}), 1.0, 1e-12));
            }
        }
    }

    // Weighted downstream through the slabs.
    {
        auto& a = ins[0];
        auto& b = ins[1];
        a.zero_grad();
        b.zero_grad();
        auto out = Tensor::cat(ins, 1);
        auto w = make2(2, 5, dev, false);
        auto z = out.mul(w);
        auto loss = z.sum();
        loss.backward(DType::Float64);
        for (auto r = size_t{0}; r < 2; ++r) {
            for (auto c = size_t{0}; c < 3; ++c) {
                CHECK(near64(a.gradient().at<float64_t>({r, c}), w.at<float64_t>({r, c}), 1e-9));
            }
            for (auto c = size_t{0}; c < 2; ++c) {
                CHECK(near64(b.gradient().at<float64_t>({r, c}), w.at<float64_t>({r, 3 + c}), 1e-9));
            }
        }
    }

    // Four inputs exercise the NodeMeta overflow-parent list (capacity 3).
    {
        std::vector<Tensor> four;
        four.reserve(4);
        for (auto i = size_t{0}; i < 4; ++i) {
            four.emplace_back(make2(2, 1, dev, true));
        }
        auto out = Tensor::cat(four, 1);
        auto loss = out.sum();
        loss.backward(DType::Float64);
        for (auto i = size_t{0}; i < 4; ++i) {
            CHECK(near64(four[i].gradient().at<float64_t>({0, 0}), 1.0, 1e-12));
            CHECK(near64(four[i].gradient().at<float64_t>({1, 0}), 1.0, 1e-12));
        }
    }

    // Mixed requires_grad: only the grad-marked input accumulates.
    {
        std::vector<Tensor> mixed;
        mixed.reserve(2);
        mixed.emplace_back(make2(2, 2, dev, true));
        mixed.emplace_back(make2(2, 2, dev, false));
        auto out = Tensor::cat(mixed, 0);
        CHECK(out.requires_gradient());
        auto loss = out.sum();
        loss.backward(DType::Float64);
        CHECK(mixed[0].has_grad());
        CHECK(!mixed[1].has_grad());
    }

    // No grad input at all: no graph.
    {
        std::vector<Tensor> plain;
        plain.reserve(2);
        plain.emplace_back(make2(2, 2, dev, false));
        plain.emplace_back(make2(2, 2, dev, false));
        auto out = Tensor::cat(plain, 0);
        CHECK(!out.requires_gradient());
    }

    // cat of split pieces round-trips: disjoint regions, each gets its ones.
    {
        auto x = make2(4, 6, dev, true);
        auto parts = x.split(0, 2);
        auto out = Tensor::cat(parts, 0);
        auto loss = out.sum();
        loss.backward(DType::Float64);
        for (auto r = size_t{0}; r < 4; ++r) {
            for (auto c = size_t{0}; c < 6; ++c) {
                CHECK(near64(x.gradient().at<float64_t>({r, c}), 1.0, 1e-12));
            }
        }
    }
}

// --------------------------- slice ---------------------------

static void test_slice(const Device dev) {
    // Leading-dim slice (contiguous view): finite differences.
    auto x = make2(4, 3, dev, true);
    fd_check("slice-leading", x, [&](Tensor& t) { return t.slice({{1, 3}, {0, 3}}); }, 1e-4);

    // Inner-dim box (non-contiguous view): seed-ones backward, exact box check.
    {
        auto xb = make2(4, 5, dev, true);
        auto y = xb.slice({{1, 3}, {1, 4}});
        CHECK(y.requires_gradient());
        y.backward(DType::Float64); // seeds ones over the view
        for (auto r = size_t{0}; r < 4; ++r) {
            for (auto c = size_t{0}; c < 5; ++c) {
                const auto in_box = r >= 1 && r < 3 && c >= 1 && c < 4;
                CHECK(near64(xb.gradient().at<float64_t>({r, c}), in_box ? 1.0 : 0.0, 1e-12));
            }
        }
        // (The non-contiguous view still feeds backward() correctly, which is
        // what this block exercises; contiguous-only ops reject it per the
        // documented contract.)
    }

    // Nested views: slice of a slice chains two slice_g nodes.
    {
        auto xn = make2(4, 6, dev, true);
        auto mid = xn.slice({{1, 4}, {0, 6}}); // {3,6}, contiguous
        auto inner = mid.slice({{0, 3}, {2, 5}}); // {3,3}, non-contiguous
        inner.backward(DType::Float64);
        for (auto r = size_t{0}; r < 4; ++r) {
            for (auto c = size_t{0}; c < 6; ++c) {
                const auto in_box = r >= 1 && r < 4 && c >= 2 && c < 5;
                CHECK(near64(xn.gradient().at<float64_t>({r, c}), in_box ? 1.0 : 0.0, 1e-12));
            }
        }
    }

    // Overlapping slices fan out: the overlap accumulates both deltas.
    {
        auto xo = make2(4, 3, dev, true);
        auto y1 = xo.slice({{0, 2}, {0, 3}});
        auto y2 = xo.slice({{1, 3}, {0, 3}});
        auto s1 = y1.sum();
        auto s2 = y2.sum();
        auto loss = s1.add(s2);
        loss.backward(DType::Float64);
        for (auto c = size_t{0}; c < 3; ++c) {
            CHECK(near64(xo.gradient().at<float64_t>({0, c}), 1.0, 1e-12));
            CHECK(near64(xo.gradient().at<float64_t>({1, c}), 2.0, 1e-12));
            CHECK(near64(xo.gradient().at<float64_t>({2, c}), 1.0, 1e-12));
            CHECK(near64(xo.gradient().at<float64_t>({3, c}), 0.0, 1e-12));
        }
    }

    // Empty slice: zero-extent view, backward is a no-op scatter.
    {
        auto xe = make2(3, 3, dev, true);
        auto y = xe.slice({{1, 1}, {0, 3}});
        y.backward(DType::Float64);
        for (auto r = size_t{0}; r < 3; ++r) {
            for (auto c = size_t{0}; c < 3; ++c) {
                CHECK(near64(xe.gradient().at<float64_t>({r, c}), 0.0, 1e-12));
            }
        }
    }
}

// --------------------------- split ---------------------------

static void test_split(const Device dev) {
    // Unweighted: both halves receive ones.
    {
        auto x = make2(4, 6, dev, true);
        auto parts = x.split(0, 2);
        CHECK(parts.size() == 2);
        CHECK(parts[0].requires_gradient() && parts[1].requires_gradient());
        auto s0 = parts[0].sum();
        auto s1 = parts[1].sum();
        auto loss = s0.add(s1);
        loss.backward(DType::Float64);
        for (auto r = size_t{0}; r < 4; ++r) {
            for (auto c = size_t{0}; c < 6; ++c) {
                CHECK(near64(x.gradient().at<float64_t>({r, c}), 1.0, 1e-12));
            }
        }
    }

    // Weighted per chunk: scalar mul wires per chunk, so the halves differ.
    {
        auto x = make2(4, 2, dev, true);
        auto parts = x.split(0, 2);
        auto m0 = parts[0].mul(float64_t{2.0});
        auto m1 = parts[1].mul(float64_t{3.0});
        auto s0 = m0.sum();
        auto s1 = m1.sum();
        auto loss = s0.add(s1);
        loss.backward(DType::Float64);
        for (auto r = size_t{0}; r < 4; ++r) {
            for (auto c = size_t{0}; c < 2; ++c) {
                CHECK(near64(x.gradient().at<float64_t>({r, c}), r < 2 ? 2.0 : 3.0, 1e-12));
            }
        }
    }

    // Inner-axis split yields non-contiguous chunks; seed-ones backward on one
    // chunk must still land in the right box of x.grad.
    {
        auto x = make2(2, 4, dev, true);
        auto parts = x.split(1, 2);
        parts[1].backward(DType::Float64);
        for (auto r = size_t{0}; r < 2; ++r) {
            for (auto c = size_t{0}; c < 4; ++c) {
                CHECK(near64(x.gradient().at<float64_t>({r, c}), c >= 2 ? 1.0 : 0.0, 1e-12));
            }
        }
    }
}

// --------------------------- reshape ---------------------------

static void test_reshape(const Device dev) {
    auto x = make2(2, 6, dev, true);
    fd_check("reshape", x, [&](Tensor& t) { return t.reshape({3, 4}); }, 1e-4);

    // Seed-ones backward through the view: every element receives 1.
    auto x2 = make2(2, 6, dev, true);
    auto y = x2.reshape({3, 4});
    CHECK(y.requires_gradient());
    y.backward(DType::Float64);
    for (auto r = size_t{0}; r < 2; ++r) {
        for (auto c = size_t{0}; c < 6; ++c) {
            CHECK(near64(x2.gradient().at<float64_t>({r, c}), 1.0, 1e-12));
        }
    }
}

// --------------------------- grad dtype boundary ---------------------------

// Chain slice -> relu -> index_select -> sum; the leaf is F64 but the call
// assigns Float16: every intermediate delta must be F16 (probe hook), the
// leaf buffer pins F16, and a later F32 call converts+accumulates into it.
static void test_grad_dtype_chain() {
    tdetail::grad_probe_reset();
    auto x = make2(3, 3, Device::CPU, true);
    auto idx = make_idx({0, 0, 2}, Device::CPU);
    auto s = x.slice({{0, 3}, {0, 3}}); // identity box, contiguous
    auto r = s.relu();
    auto y = r.index_select(0, idx);
    auto loss = y.sum();
    loss.backward(DType::Float16);
    CHECK(x.grad_dtype() == DType::Float16);
    CHECK(tdetail::grad_probe_count(DType::Float16) >= 2);
    // Values are quarter-exact, so F16 holds them exactly.
    using BeanTensor::float16_t;
    using tdetail::autograd_shared::widen_lane;
    for (auto c = size_t{0}; c < 3; ++c) {
        CHECK(near64(static_cast<float64_t>(widen_lane(x.gradient().at<float16_t>({0, c}))), 2.0, 1e-3));
        CHECK(near64(static_cast<float64_t>(widen_lane(x.gradient().at<float16_t>({1, c}))), 0.0, 1e-3));
        CHECK(near64(static_cast<float64_t>(widen_lane(x.gradient().at<float16_t>({2, c}))), 1.0, 1e-3));
    }
    // Second call in F32: the delta converts into the pinned F16 leaf buffer.
    loss.backward(DType::Float32);
    CHECK(x.grad_dtype() == DType::Float16);
    CHECK(near64(static_cast<float64_t>(widen_lane(x.gradient().at<float16_t>({0, 0}))), 4.0, 1e-3));
}

// --------------------------- lifetime after forward returns ---------------------------

// The forward call frame (with its op-local metadata construction) is gone
// before backward(); the graph must hold only node-owned metadata and
// pointers to tensors the caller keeps alive.
static void test_lifetime() {
    auto x = make2(3, 4, Device::CPU, true);
    auto idx = make_idx({2, 0}, Device::CPU);
    auto y = [&x, &idx] { return x.index_select(0, idx); }();
    y.backward(DType::Float64); // seed ones, y is the root
    for (auto c = size_t{0}; c < 4; ++c) {
        CHECK(near64(x.gradient().at<float64_t>({2, c}), 1.0, 1e-12));
        CHECK(near64(x.gradient().at<float64_t>({0, c}), 1.0, 1e-12));
        CHECK(near64(x.gradient().at<float64_t>({1, c}), 0.0, 1e-12));
    }

    auto a = make2(2, 2, Device::CPU, true);
    auto b = make2(2, 2, Device::CPU, true);
    std::vector<Tensor> ins;
    ins.reserve(2);
    ins.push_back(std::move(a));
    ins.push_back(std::move(b));
    auto out = [&ins] { return Tensor::cat(ins, 0); }();
    auto loss = out.sum();
    loss.backward(DType::Float64);
    CHECK(near64(ins[0].gradient().at<float64_t>({0, 0}), 1.0, 1e-12));
    CHECK(near64(ins[1].gradient().at<float64_t>({1, 1}), 1.0, 1e-12));
}

// --------------------------- main ---------------------------

int main() {
    test_index_select(Device::CPU);
    std::cout << "index_select cpu ok\n";
    test_gather(Device::CPU);
    std::cout << "gather cpu ok\n";
    test_cat(Device::CPU);
    std::cout << "cat cpu ok\n";
    test_slice(Device::CPU);
    std::cout << "slice cpu ok\n";
    test_split(Device::CPU);
    std::cout << "split cpu ok\n";
    test_reshape(Device::CPU);
    std::cout << "reshape cpu ok\n";
    test_grad_dtype_chain();
    std::cout << "grad dtype chain ok\n";
    test_lifetime();
    std::cout << "lifetime ok\n";

#if defined(USE_HIP) || defined(USE_CUDA)
    if (gpu_available()) {
        test_index_select(Device::GPU);
        std::cout << "index_select gpu ok\n";
        test_gather(Device::GPU);
        std::cout << "gather gpu ok\n";
        test_cat(Device::GPU);
        std::cout << "cat gpu ok\n";
        test_slice(Device::GPU);
        std::cout << "slice gpu ok\n";
        test_split(Device::GPU);
        std::cout << "split gpu ok\n";
        test_reshape(Device::GPU);
        std::cout << "reshape gpu ok\n";
    } else {
        std::cout << "no GPU present; GPU sections skipped\n";
    }
#endif

    if (failures == 0) {
        std::cout << "ALL TESTS PASSED\n";
        return 0;
    }
    std::cerr << failures << " failures\n";
    return 1;
}
