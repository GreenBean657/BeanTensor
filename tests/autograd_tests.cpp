// BeanTensorV5 autograd tests — framework-free assertion suite.
// Covers: the four backward pathways (GEMM_G incl. batched, bias_add_G,
// relu_G, softmaxCE_G) plus the elementwise backward nodes (add/sub/mul/div/
// pow tensor-tensor and tensor-scalar, exp/log — analytic checks, broadcast
// gradient reduction, per-call grad dtype probes) via the forward ops'
// auto-wired graph and Tensor::backward(); analytic checks on fp32/fp64,
// finite-difference checks (fp64) of a full relu(x·W + b) -> softmax_ce
// chain, threaded/SIMD paths (large tensors), the full-tensor reduction
// backward nodes (sum_G/mean_G: broadcast-seed deltas, non-default grad
// dtype, leaf accumulation), every dtype pathway (fp16/bf16,
// packed x2, all integer dtypes, int4x2), gradient accumulation across
// backward() calls, fan-out DAGs, zero_grad, error paths, per-call grad dtype
// override (backward(DType): fp64/fp16/fp32/int8/int32 on the same graph, with
// the intermediate-delta probe and leaf/intermediate grad dtype checks), and —
// when a backend + device exist — the
// same checks on the GPU plus CPU/GPU gradient parity.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "Helpers/Errors.h"
#include "Intrinsics/Native/DTypeConversions.h"
#include "Math/Autograd/BiasAddG/BiasAddGKernels.h"
#include "Math/Autograd/GradAcc/GradAccKernels.h"
#include "Math/Autograd/GradDType.h"
#include "Math/Autograd/ReluG/ReluGKernels.h"
#include "Tensor/Tensor.h"

#if defined(USE_HIP)
#include <hip/hip_runtime.h>
#elif defined(USE_CUDA)
#include <cuda_runtime.h>
#endif

using BeanTensor::Tensors::Tensor;
using BeanTensor::Tensors::Device;
using BeanTensor::Tensors::ElementwiseOp;
using BeanTensor::float16_t;
using BeanTensor::float16x2_t;
using BeanTensor::bfloat16_t;
using BeanTensor::bfloat16x2_t;
using BeanTensor::float32_t;
using BeanTensor::float64_t;
using BeanTensor::int4x2_t;
using DType = BeanTensor::DataTypes::DType;
namespace idet = BeanTensor::Intrinsics::detail;
namespace tdetail = BeanTensor::Tensors::detail;

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

// Mixed-precision comparisons happen at float64: the casts are explicit so no
// call site relies on an implicit float->double promotion.
template<typename A, typename B>
static bool near64(const A a, const B b, const float64_t eps) {
    return std::fabs(static_cast<float64_t>(a) - static_cast<float64_t>(b)) <= eps;
}

// --- Packed/half-dtype helpers (host-side, via the unaccel conversion pathway) ---

template<typename H>
static H narrow_lane(const float32_t v) {
    if constexpr (std::is_same_v<H, float16_t>) {
        return idet::unaccel_fp32_to_fp16_untracked(v);
    } else {
        return idet::unaccel_fp32_to_bf16_untracked(v);
    }
}

template<typename H>
static float32_t widen_lane(const H& v) {
    if constexpr (std::is_same_v<H, float16_t>) {
        return idet::unaccel_fp16_to_fp32_untracked(v);
    } else {
        return idet::unaccel_bf16_to_fp32(v);
    }
}

template<typename P, typename H>
static P pack_lanes(const float32_t lo, const float32_t hi) {
    const auto lo_bits = static_cast<uint32_t>(idet::bit_cast_memcpy<uint16_t>(narrow_lane<H>(lo)));
    const auto hi_bits = static_cast<uint32_t>(idet::bit_cast_memcpy<uint16_t>(narrow_lane<H>(hi)));
    return idet::bit_cast_memcpy<P>(lo_bits | (hi_bits << 16));
}

template<typename H, typename P>
static float32_t lane_lo_f32(const P& v) {
    const auto bits = idet::bit_cast_memcpy<uint32_t>(v);
    return widen_lane(idet::bit_cast_memcpy<H>(static_cast<uint16_t>(bits & 0xFFFFu)));
}

template<typename H, typename P>
static float32_t lane_hi_f32(const P& v) {
    const auto bits = idet::bit_cast_memcpy<uint32_t>(v);
    return widen_lane(idet::bit_cast_memcpy<H>(static_cast<uint16_t>(bits >> 16)));
}

// --- int4x2 helpers ---

static int32_t int4_lo(const int4x2_t v) {
    const auto n = static_cast<int32_t>(idet::bit_cast_memcpy<uint8_t>(v) & 0x0Fu);
    return n >= 8 ? n - 16 : n;
}

static int32_t int4_hi(const int4x2_t v) {
    const int32_t n = idet::bit_cast_memcpy<uint8_t>(v) >> 4;
    return n >= 8 ? n - 16 : n;
}

static int4x2_t pack_int4(const int32_t lo, const int32_t hi) {
    return idet::bit_cast_memcpy<int4x2_t>(static_cast<uint8_t>((lo & 0xF) | ((hi & 0xF) << 4)));
}

// --- flat element access (rank 1-3) ---

template<typename T>
static void set_flat(Tensor& t, const std::vector<size_t>& shape, const size_t i, const T v) {
    if (shape.size() == 1) {
        t.set<T>({i}, v);
    } else if (shape.size() == 2) {
        t.set<T>({i / shape[1], i % shape[1]}, v);
    } else {
        t.set<T>({i / (shape[1] * shape[2]), i / shape[2] % shape[1], i % shape[2]}, v);
    }
}

template<typename T>
static T get_flat(Tensor& t, const std::vector<size_t>& shape, const size_t i) {
    if (shape.size() == 1) {
        return t.at<T>({i});
    }
    if (shape.size() == 2) {
        return t.at<T>({i / shape[1], i % shape[1]});
    }
    return t.at<T>({i / (shape[1] * shape[2]), i / shape[2] % shape[1], i % shape[2]});
}

static size_t numel_of(const std::vector<size_t>& shape) {
    auto n = size_t{1};
    for (const auto s : shape) {
        n *= s;
    }
    return n;
}

// --- fp64 references ---

/**
 * dlogits = (softmax(row) - onehot(target)) / rows, row-major.
 */
static std::vector<float64_t> ref_sce_grad(const std::vector<float64_t>& logits, const size_t rows, const size_t cols,
                                           const std::vector<size_t>& targets) {
    std::vector<float64_t> out(rows * cols);
    for (auto r = size_t{0}; r < rows; ++r) {
        auto m = std::numeric_limits<float64_t>::lowest();
        for (auto j = size_t{0}; j < cols; ++j) {
            m = std::max(m, logits[r * cols + j]);
        }
        auto s = 0.0;
        for (auto j = size_t{0}; j < cols; ++j) {
            s += std::exp(logits[r * cols + j] - m);
        }
        for (auto j = size_t{0}; j < cols; ++j) {
            const auto p = std::exp(logits[r * cols + j] - m) / s;
            out[r * cols + j] = (p - (j == targets[r] ? 1.0 : 0.0)) / static_cast<float64_t>(rows);
        }
    }
    return out;
}

/**
 * GEMM_G with an all-ones output gradient: dA[i,k] = sum_j B[k,j],
 * dB[k,j] = sum_i A[i,k] (per batch).
 */
static void ref_gemm_grads_ones(const std::vector<float64_t>& a, const std::vector<float64_t>& b,
                                const size_t batch, const size_t m, const size_t k, const size_t n,
                                std::vector<float64_t>& da, std::vector<float64_t>& db) {
    da.assign(batch * m * k, 0.0);
    db.assign(batch * k * n, 0.0);
    for (auto p = size_t{0}; p < batch; ++p) {
        for (auto i = size_t{0}; i < m; ++i) {
            for (auto kk = size_t{0}; kk < k; ++kk) {
                for (auto j = size_t{0}; j < n; ++j) {
                    da[p * m * k + i * k + kk] += b[p * k * n + kk * n + j];
                    db[p * k * n + kk * n + j] += a[p * m * k + i * k + kk];
                }
            }
        }
    }
}

// =====================================================================
// SIMD kernel parity (every tier the CPU supports vs the scalar contract)
// =====================================================================

namespace ag = BeanTensor::Math::Autograd;

template<typename T, typename Bits>
static void check_kernel_tier(void (*acc)(T*, const T*, size_t, size_t),
                              void (*relu)(T*, const T*, const T*, size_t, size_t),
                              void (*colsum)(T*, const T*, size_t, size_t, size_t)) {
    constexpr size_t n = 1037;
    std::vector<T> x(n);
    std::vector<T> d(n);
    for (auto i = size_t{0}; i < n; ++i) {
        x[i] = static_cast<T>(std::sin(0.91 * static_cast<float64_t>(i)) * 3.0);
        d[i] = static_cast<T>(std::cos(0.37 * static_cast<float64_t>(i)) * 1.5);
    }
    x[5] = std::numeric_limits<T>::quiet_NaN();
    x[6] = static_cast<T>(-0.0);
    x[7] = T{0};
    d[9] = static_cast<T>(-0.0);
    x[9] = T{1};
    d[11] = std::numeric_limits<T>::quiet_NaN();
    x[11] = T{2};
    auto bad = size_t{0};
    // Odd [begin, end) ranges exercise the unaligned heads and masked tails.
    for (const auto& [b, e] : std::vector<std::pair<size_t, size_t>>{{0, n}, {3, n - 2}, {17, 30}, {0, 5}}) {
        auto w_simd = d;
        auto w_ref = d;
        acc(w_simd.data(), x.data(), b, e);
        auto dx_simd = std::vector<T>(n, T{7});
        auto dx_ref = std::vector<T>(n, T{7});
        relu(dx_simd.data(), x.data(), d.data(), b, e);
        for (auto i = b; i < e; ++i) {
            w_ref[i] = w_ref[i] + x[i];
            dx_ref[i] = x[i] > T{0} ? d[i] : T{0};
        }
        for (auto i = size_t{0}; i < n; ++i) {
            if (idet::bit_cast_memcpy<Bits>(w_simd[i]) != idet::bit_cast_memcpy<Bits>(w_ref[i])) {
                ++bad;
            }
            if (idet::bit_cast_memcpy<Bits>(dx_simd[i]) != idet::bit_cast_memcpy<Bits>(dx_ref[i])) {
                ++bad;
            }
        }
    }
    // Column sums: 61 rows x 17 cols (vector body + tail per row).
    constexpr size_t cols = 17;
    constexpr size_t rows = 61;
    auto p_simd = std::vector<T>(cols, T{0});
    auto p_ref = std::vector<T>(cols, T{0});
    colsum(p_simd.data(), d.data(), cols, 2, rows);
    for (auto r = size_t{2}; r < rows; ++r) {
        for (auto c = size_t{0}; c < cols; ++c) {
            p_ref[c] = p_ref[c] + d[r * cols + c];
        }
    }
    for (auto c = size_t{0}; c < cols; ++c) {
        if (idet::bit_cast_memcpy<Bits>(p_simd[c]) != idet::bit_cast_memcpy<Bits>(p_ref[c])) {
            ++bad;
        }
    }
    CHECK(bad == 0);
}

static void test_simd_kernel_parity() {
    if (ag::grad_acc_avx2_available() && ag::relu_g_avx2_available() && ag::bias_add_g_avx2_available()) {
        check_kernel_tier<float32_t, uint32_t>(ag::grad_acc_f32_range_avx2, ag::relu_g_f32_range_avx2, ag::bias_add_g_f32_colsum_avx2);
        check_kernel_tier<float64_t, uint64_t>(ag::grad_acc_f64_range_avx2, ag::relu_g_f64_range_avx2, ag::bias_add_g_f64_colsum_avx2);
    } else {
        std::cout << "AVX2 not available — skipping AVX2 autograd kernel parity\n";
    }
    if (ag::grad_acc_avx512_available() && ag::relu_g_avx512_available() && ag::bias_add_g_avx512_available()) {
        check_kernel_tier<float32_t, uint32_t>(ag::grad_acc_f32_range_avx512, ag::relu_g_f32_range_avx512, ag::bias_add_g_f32_colsum_avx512);
        check_kernel_tier<float64_t, uint64_t>(ag::grad_acc_f64_range_avx512, ag::relu_g_f64_range_avx512, ag::bias_add_g_f64_colsum_avx512);
    } else {
        std::cout << "AVX512F not available — skipping AVX512 autograd kernel parity\n";
    }
}

// =====================================================================
// relu_G
// =====================================================================

static void test_relu_g_f32(const Device dev) {
    Tensor x({1, 5}, DType::Float32, dev, true);
    x.set<float32_t>({0, 0}, -2.5f);
    x.set<float32_t>({0, 1}, 3.5f);
    x.set<float32_t>({0, 2}, 0.0f);
    x.set<float32_t>({0, 3}, std::numeric_limits<float32_t>::quiet_NaN());
    x.set<float32_t>({0, 4}, 1.0e-30f);
    auto y = x.relu();
    y.backward(DType::Float32);
    CHECK(x.has_grad());
    auto& g = x.gradient();
    // A select: NaN and 0 inputs get exactly +0, positives pass dy = 1 through.
    CHECK(idet::bit_cast_memcpy<uint32_t>(g.at<float32_t>({0, 0})) == 0u);
    CHECK(idet::bit_cast_memcpy<uint32_t>(g.at<float32_t>({0, 1})) == idet::bit_cast_memcpy<uint32_t>(1.0f));
    CHECK(idet::bit_cast_memcpy<uint32_t>(g.at<float32_t>({0, 2})) == 0u);
    CHECK(idet::bit_cast_memcpy<uint32_t>(g.at<float32_t>({0, 3})) == 0u);
    CHECK(idet::bit_cast_memcpy<uint32_t>(g.at<float32_t>({0, 4})) == idet::bit_cast_memcpy<uint32_t>(1.0f));
    // The root's own gradient is the ones seed.
    CHECK(idet::bit_cast_memcpy<uint32_t>(y.gradient().at<float32_t>({0, 2})) == idet::bit_cast_memcpy<uint32_t>(1.0f));
}

template<typename T>
static void test_relu_g_threaded(const Device dev, const DType dt) {
    // 300 x 1001 = 300300 elements > threads*1024: threaded path, chunk
    // boundaries that land mid-vector for the AVX kernels.
    const auto shape = std::vector<size_t>{300, 1001};
    const auto n = numel_of(shape);
    Tensor x(shape, dt, dev, true);
    if (dev == Device::CPU) {
        for (auto i = size_t{0}; i < n; ++i) {
            set_flat<T>(x, shape, i, static_cast<T>(static_cast<T>(i % 7) - static_cast<T>(3) + static_cast<T>(0.5)));
        }
    } else {
        Tensor xc(shape, dt, Device::CPU, true);
        for (auto i = size_t{0}; i < n; ++i) {
            set_flat<T>(xc, shape, i, static_cast<T>(static_cast<T>(i % 7) - static_cast<T>(3) + static_cast<T>(0.5)));
        }
        x = xc.clone(true);
        x.to(Device::GPU);
    }
    auto y = x.relu();
    y.backward(dt);
    auto g = x.gradient().clone(true);
    if (dev == Device::GPU) {
        g.to(Device::CPU);
    }
    g.sync();
    auto bad = size_t{0};
    for (auto i = size_t{0}; i < n; i += 37) {
        const auto want = static_cast<T>(i % 7) - static_cast<T>(3) + static_cast<T>(0.5) > T{0} ? T{1} : T{0};
        if (std::fabs(get_flat<T>(g, shape, i) - want) > T{0}) {
            ++bad;
        }
    }
    CHECK(bad == 0);
}

// =====================================================================
// bias_add_G
// =====================================================================

static void test_bias_add_g_f32(const Device dev) {
    Tensor x({3, 4}, DType::Float32, dev, true);
    Tensor b({4}, DType::Float32, dev, true);
    x.set_zero();
    b.set_zero();
    auto y = x.bias_add(b);
    y.backward(DType::Float32);
    for (auto c = size_t{0}; c < 4; ++c) {
        CHECK(idet::bit_cast_memcpy<uint32_t>(b.gradient().at<float32_t>({c})) == idet::bit_cast_memcpy<uint32_t>(3.0f));
        for (auto r = size_t{0}; r < 3; ++r) {
            CHECK(idet::bit_cast_memcpy<uint32_t>(x.gradient().at<float32_t>({r, c})) == idet::bit_cast_memcpy<uint32_t>(1.0f));
        }
    }
}

template<typename T>
static void test_bias_add_g_threaded(const Device dev, const DType dt) {
    // 5000 rows split across the pool's per-slot partials (CPU) or chunked
    // partials (GPU); 37 columns exercise the SIMD tails.
    constexpr size_t rows = 5000;
    constexpr size_t cols = 37;
    Tensor x({rows, cols}, dt, dev, true);
    Tensor b({cols}, dt, dev, true);
    x.set_zero();
    b.set_zero();
    auto y = x.bias_add(b);
    y.backward(dt);
    for (auto c = size_t{0}; c < cols; ++c) {
        CHECK(std::fabs(b.gradient().at<T>({c}) - static_cast<T>(rows)) <= T{0});
    }
}

// =====================================================================
// GEMM_G
// =====================================================================

template<typename T>
static void run_gemm_g_case(const Device dev, const DType dt, const size_t batch, const size_t m, const size_t k, const size_t n, const float64_t eps) {
    const auto a_shape = batch == 0 ? std::vector{m, k} : std::vector{batch, m, k};
    const auto b_shape = batch == 0 ? std::vector{k, n} : std::vector{batch, k, n};
    const auto bt = batch == 0 ? size_t{1} : batch;
    std::vector<float64_t> av(bt * m * k);
    std::vector<float64_t> bv(bt * k * n);
    Tensor a(a_shape, dt, dev, true);
    Tensor b(b_shape, dt, dev, true);
    for (auto i = size_t{0}; i < av.size(); ++i) {
        av[i] = static_cast<float64_t>(static_cast<int64_t>(i % 5) - 2) * 0.5;
        set_flat<T>(a, a_shape, i, static_cast<T>(av[i]));
    }
    for (auto i = size_t{0}; i < bv.size(); ++i) {
        bv[i] = static_cast<float64_t>(static_cast<int64_t>(i % 7) - 3) * 0.25;
        set_flat<T>(b, b_shape, i, static_cast<T>(bv[i]));
    }
    auto y = a.matmul(b);
    y.backward(dt);
    std::vector<float64_t> da;
    std::vector<float64_t> db;
    ref_gemm_grads_ones(av, bv, bt, m, k, n, da, db);
    auto bad = size_t{0};
    for (auto i = size_t{0}; i < da.size(); ++i) {
        if (!near64(static_cast<float64_t>(get_flat<T>(a.gradient(), a_shape, i)), da[i], eps)) {
            ++bad;
        }
    }
    for (auto i = size_t{0}; i < db.size(); ++i) {
        if (!near64(static_cast<float64_t>(get_flat<T>(b.gradient(), b_shape, i)), db[i], eps)) {
            ++bad;
        }
    }
    CHECK(bad == 0);
}

static void test_gemm_g(const Device dev) {
    run_gemm_g_case<float64_t>(dev, DType::Float64, 0, 2, 3, 4, 1e-12);
    run_gemm_g_case<float32_t>(dev, DType::Float32, 0, 5, 7, 3, 1e-5);
    // Batched rank-3.
    run_gemm_g_case<float64_t>(dev, DType::Float64, 3, 4, 2, 5, 1e-12);
    run_gemm_g_case<float32_t>(dev, DType::Float32, 2, 3, 6, 2, 1e-5);
    // Large enough for the threaded GEMM units + transpose launcher.
    run_gemm_g_case<float32_t>(dev, DType::Float32, 0, 96, 130, 70, 1e-3);
}

static void test_gemm_g_one_side(const Device dev) {
    // Only A requires grad: B must not receive a gradient.
    Tensor a({2, 2}, DType::Float32, dev, true);
    Tensor b({2, 2}, DType::Float32, dev, false);
    a.set_one();
    b.set_one();
    auto y = a.matmul(b);
    y.backward(DType::Float32);
    CHECK(a.has_grad());
    CHECK(!b.has_grad());
    CHECK(idet::bit_cast_memcpy<uint32_t>(a.gradient().at<float32_t>({1, 1})) == idet::bit_cast_memcpy<uint32_t>(2.0f));
}

// =====================================================================
// softmaxCE_G
// =====================================================================

template<typename T, typename I>
static void run_sce_g_case(const Device dev, const DType dt, const DType it, const float64_t eps) {
    constexpr size_t rows = 3;
    constexpr size_t cols = 4;
    const auto lv = std::vector{1.0, 2.0, 3.0, -1.0, 0.5, -1.0, 2.0, 0.25, -3.0, 0.0, 1.5, 1.5};
    const auto tv = std::vector<size_t>{2, 0, 3};
    Tensor logits({rows, cols}, dt, dev, true);
    Tensor targets({rows}, it, dev, false);
    for (auto i = size_t{0}; i < rows * cols; ++i) {
        logits.set<T>({i / cols, i % cols}, static_cast<T>(lv[i]));
    }
    for (auto r = size_t{0}; r < rows; ++r) {
        targets.set<I>({r}, static_cast<I>(tv[r]));
    }
    auto loss = logits.softmax_ce(targets);
    loss.backward(dt);
    CHECK(!targets.has_grad());
    const auto want = ref_sce_grad(lv, rows, cols, tv);
    auto bad = size_t{0};
    for (auto i = size_t{0}; i < rows * cols; ++i) {
        if (!near64(static_cast<float64_t>(logits.gradient().at<T>({i / cols, i % cols})), want[i], eps)) {
            ++bad;
        }
    }
    CHECK(bad == 0);
}

static void test_softmax_ce_g(const Device dev) {
    run_sce_g_case<float64_t, int32_t>(dev, DType::Float64, DType::Int32, 1e-12);
    run_sce_g_case<float32_t, int64_t>(dev, DType::Float32, DType::Int64, 1e-6);
    run_sce_g_case<float32_t, uint8_t>(dev, DType::Float32, DType::UInt8, 1e-6);
}

static void test_softmax_ce_g_threaded(const Device dev) {
    // 4096 rows > threads*8: the threaded row launcher on CPU.
    constexpr size_t rows = 4096;
    constexpr size_t cols = 10;
    std::vector<float64_t> lv(rows * cols);
    std::vector<size_t> tv(rows);
    Tensor logits({rows, cols}, DType::Float64, Device::CPU, true);
    Tensor targets({rows}, DType::Int32, Device::CPU, false);
    for (auto i = size_t{0}; i < rows * cols; ++i) {
        lv[i] = std::sin(0.37 * static_cast<float64_t>(i));
        logits.set<float64_t>({i / cols, i % cols}, lv[i]);
    }
    for (auto r = size_t{0}; r < rows; ++r) {
        tv[r] = r * 7 % cols;
        targets.set<int32_t>({r}, static_cast<int32_t>(tv[r]));
    }
    if (dev == Device::GPU) {
        logits.to(Device::GPU);
        targets.to(Device::GPU);
    }
    auto loss = logits.softmax_ce(targets);
    loss.backward(DType::Float64);
    const auto want = ref_sce_grad(lv, rows, cols, tv);
    auto bad = size_t{0};
    for (auto i = size_t{0}; i < rows * cols; i += 13) {
        if (!near64(logits.gradient().at<float64_t>({i / cols, i % cols}), want[i], 1e-15)) {
            ++bad;
        }
    }
    CHECK(bad == 0);
}

// Softmax-collapse regression (arXiv:2610.31642): with the target logit past
// the accumulator cliff, the naive (p - 1) target gradient rounds to exactly 0
// and the row gradient stops summing to zero; the dlogits row must still sum
// to zero (target gradient = negative sum of the non-target gradients). The
// pre-fix residue here is ~1.7e-17, the post-fix sum is exactly 0, so the
// discriminating tolerance sits well below the residue.
template<typename T>
static void run_sce_g_zero_sum_case(const Device dev, const DType dt) {
    constexpr size_t rows = 1;
    constexpr size_t cols = 5;
    Tensor logits({rows, cols}, dt, dev, true);
    Tensor targets({rows}, DType::Int32, dev, false);
    const auto put = [&](const size_t j, const float32_t v) {
        if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
            logits.set<T>({0, j}, static_cast<T>(v));
        } else {
            logits.set<T>({0, j}, narrow_lane<T>(v));
        }
    };
    for (auto j = size_t{0}; j < cols; ++j) {
        put(j, j == 0 ? 40.0f : 0.0f);
    }
    targets.set<int32_t>({0}, 0);
    auto loss = logits.softmax_ce(targets);
    loss.backward(dt);
    auto total = float64_t{0.0};
    for (auto j = size_t{0}; j < cols; ++j) {
        if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
            total += static_cast<float64_t>(logits.gradient().at<T>({0, j}));
        } else {
            total += static_cast<float64_t>(widen_lane(logits.gradient().at<T>({0, j})));
        }
    }
    CHECK(std::fabs(total) <= 1e-20);
}

// Packed variant: 4 effective classes in 2 storage elements, target = the lo
// lane of element 0, residue-class logit 20 (still past the fp32-accumulator
// cliff: 3*exp(-20) << 2^-25, so s rounds to 1 and the old target gradient
// collapses to 0 with a ~3*exp(-20) non-target residue). This is the only
// zero-sum case that exercises the packed accumulation rewrite. The packed
// bf16 lanes round independently (RNE), so the post-fix row sum is zero only
// to the storage ulp (~3*exp(-20)*2^-8); the tolerance sits above that bound
// and far below the pre-fix residue.
template<typename P, typename H>
static void run_sce_g_zero_sum_packed_case(const Device dev, const DType dt) {
    constexpr size_t rows = 1;
    constexpr size_t cols_store = 2;
    Tensor logits({rows, cols_store}, dt, dev, true);
    Tensor targets({rows}, DType::Int32, dev, false);
    for (auto e = size_t{0}; e < cols_store; ++e) {
        logits.set<P>({0, e}, pack_lanes<P, H>(e == 0 ? 40.0f : 20.0f, e == 0 ? 20.0f : 0.0f));
    }
    targets.set<int32_t>({0}, 0);
    auto loss = logits.softmax_ce(targets);
    loss.backward(dt);
    auto total = float64_t{0.0};
    for (auto e = size_t{0}; e < cols_store; ++e) {
        const auto g = logits.gradient().at<P>({0, e});
        total += static_cast<float64_t>(lane_lo_f32<H>(g));
        total += static_cast<float64_t>(lane_hi_f32<H>(g));
    }
    CHECK(std::fabs(total) <= 1e-10);
}

// Every dtype in which the collapse is observable in the stored gradient:
// fp32/bf16/bfloat16x2 on the fp32 accumulator (cliff ~16.6) and fp64 on the
// fp64 accumulator (cliff ~36.7). fp16/float16x2 and the integer/int4x2
// logits are storage-limited: at the collapse margins their storage flushes
// the tiny non-target residue to zero anyway, so the stored row is all-zero
// under both the old and the new code - a test there would be vacuous.
static void test_softmax_ce_g_zero_sum(const Device dev) {
    run_sce_g_zero_sum_case<float32_t>(dev, DType::Float32);
    run_sce_g_zero_sum_case<bfloat16_t>(dev, DType::BFloat16);
    run_sce_g_zero_sum_case<float64_t>(dev, DType::Float64);
    run_sce_g_zero_sum_packed_case<bfloat16x2_t, bfloat16_t>(dev, DType::BFloat16x2);
}

// =====================================================================
// Full chain: relu(x·W + b).softmax_ce(t)
// =====================================================================

namespace {
    struct Chain {
        static constexpr size_t n = 4;
        static constexpr size_t d = 3;
        static constexpr size_t h = 5;
        std::vector<float64_t> xv = std::vector<float64_t>(n * d);
        std::vector<float64_t> wv = std::vector<float64_t>(d * h);
        std::vector<float64_t> bv = std::vector<float64_t>(h);
        std::vector<size_t> tv = {0, 4, 2, 1};

        Chain() {
            for (auto i = size_t{0}; i < xv.size(); ++i) {
                xv[i] = std::sin(0.7 * static_cast<float64_t>(i) + 0.3);
            }
            for (auto i = size_t{0}; i < wv.size(); ++i) {
                wv[i] = 0.8 * std::cos(1.3 * static_cast<float64_t>(i));
            }
            for (auto i = size_t{0}; i < bv.size(); ++i) {
                bv[i] = 0.1 * static_cast<float64_t>(i) - 0.2;
            }
        }

        template<typename T>
        void load(Tensor& x, Tensor& w, Tensor& b, Tensor& t) const {
            for (auto i = size_t{0}; i < xv.size(); ++i) {
                x.set<T>({i / d, i % d}, static_cast<T>(xv[i]));
            }
            for (auto i = size_t{0}; i < wv.size(); ++i) {
                w.set<T>({i / h, i % h}, static_cast<T>(wv[i]));
            }
            for (auto i = size_t{0}; i < bv.size(); ++i) {
                b.set<T>({i}, static_cast<T>(bv[i]));
            }
            for (auto i = size_t{0}; i < n; ++i) {
                t.set<int32_t>({i}, static_cast<int32_t>(tv[i]));
            }
        }
    };

    /**
     * Run the chain forward and backward; every intermediate stays alive in
     * this scope, as backward() requires. Returns the loss.
     */
    template<typename T>
    T chain_forward_backward(Tensor& x, Tensor& w, Tensor& b, Tensor& t, const bool do_backward, const DType dt) {
        auto z = x.matmul(w);
        auto zb = z.bias_add(b);
        auto a = zb.relu();
        auto loss = a.softmax_ce(t);
        if (do_backward) {
            loss.backward(dt);
        }
        return loss.at<T>({0});
    }
}

static void test_chain_finite_difference() {
    const auto c = Chain{};
    Tensor x({Chain::n, Chain::d}, DType::Float64, Device::CPU, true);
    Tensor w({Chain::d, Chain::h}, DType::Float64, Device::CPU, true);
    Tensor b({Chain::h}, DType::Float64, Device::CPU, true);
    Tensor t({Chain::n}, DType::Int32, Device::CPU, false);
    c.load<float64_t>(x, w, b, t);
    static_cast<void>(chain_forward_backward<float64_t>(x, w, b, t, true, DType::Float64));

    constexpr auto step = 1e-6;
    constexpr auto tol = 1e-7;
    auto check_param = [&](Tensor& p, const size_t rank1_len, const size_t cols) {
        auto bad = size_t{0};
        for (auto i = size_t{0}; i < rank1_len; ++i) {
            const auto is_vec = cols == 0;
            const auto orig = is_vec ? p.at<float64_t>({i}) : p.at<float64_t>({i / cols, i % cols});
            const auto analytic = is_vec ? p.gradient().at<float64_t>({i}) : p.gradient().at<float64_t>({i / cols, i % cols});
            auto put = [&](const float64_t v) {
                if (is_vec) {
                    p.set<float64_t>({i}, v);
                } else {
                    p.set<float64_t>({i / cols, i % cols}, v);
                }
            };
            put(orig + step);
            const auto up = chain_forward_backward<float64_t>(x, w, b, t, false, DType::Float64);
            put(orig - step);
            const auto down = chain_forward_backward<float64_t>(x, w, b, t, false, DType::Float64);
            put(orig);
            if (!near64(analytic, (up - down) / (2.0 * step), tol)) {
                ++bad;
            }
        }
        CHECK(bad == 0);
    };
    check_param(x, Chain::n * Chain::d, Chain::d);
    check_param(w, Chain::d * Chain::h, Chain::h);
    check_param(b, Chain::h, 0);
}

// =====================================================================
// Accumulation, fan-out, zero_grad, errors
// =====================================================================

static void test_accumulation_and_zero_grad(const Device dev) {
    Tensor x({2, 2}, DType::Float32, dev, true);
    x.set<float32_t>({0, 0}, 1.0f);
    x.set<float32_t>({0, 1}, -1.0f);
    x.set<float32_t>({1, 0}, 2.0f);
    x.set<float32_t>({1, 1}, -2.0f);
    auto y = x.relu();
    y.backward(DType::Float32);
    y.backward(DType::Float32); // leaf grads accumulate across calls
    CHECK(idet::bit_cast_memcpy<uint32_t>(x.gradient().at<float32_t>({0, 0})) == idet::bit_cast_memcpy<uint32_t>(2.0f));
    CHECK(idet::bit_cast_memcpy<uint32_t>(x.gradient().at<float32_t>({0, 1})) == 0u);
    CHECK(idet::bit_cast_memcpy<uint32_t>(x.gradient().at<float32_t>({1, 0})) == idet::bit_cast_memcpy<uint32_t>(2.0f));
    // The intermediate (root) grad holds only the current call's seed.
    CHECK(idet::bit_cast_memcpy<uint32_t>(y.gradient().at<float32_t>({0, 0})) == idet::bit_cast_memcpy<uint32_t>(1.0f));
    x.zero_grad();
    CHECK(!x.has_grad());
    y.backward(DType::Float32);
    CHECK(idet::bit_cast_memcpy<uint32_t>(x.gradient().at<float32_t>({1, 0})) == idet::bit_cast_memcpy<uint32_t>(1.0f));
}

static void test_fan_out(const Device dev) {
    // h feeds both matmul operands: its node fires once, after both
    // contributions (G·hᵀ + hᵀ·G) are accumulated.
    Tensor x({2, 2}, DType::Float64, dev, true);
    x.set<float64_t>({0, 0}, 1.0);
    x.set<float64_t>({0, 1}, -2.0);
    x.set<float64_t>({1, 0}, 3.0);
    x.set<float64_t>({1, 1}, 4.0);
    auto h = x.relu();
    auto y = h.matmul(h);
    y.backward(DType::Float64);
    // h = [[1,0],[3,4]]; G·hᵀ = [[1,7],[1,7]]; hᵀ·G = [[4,4],[4,4]]; dH = [[5,11],[5,11]].
    CHECK(near64(h.gradient().at<float64_t>({0, 0}), 5.0, 0.0));
    CHECK(near64(h.gradient().at<float64_t>({0, 1}), 11.0, 0.0));
    CHECK(near64(x.gradient().at<float64_t>({0, 0}), 5.0, 0.0));
    CHECK(near64(x.gradient().at<float64_t>({0, 1}), 0.0, 0.0));
    CHECK(near64(x.gradient().at<float64_t>({1, 0}), 5.0, 0.0));
    CHECK(near64(x.gradient().at<float64_t>({1, 1}), 11.0, 0.0));
}

static void test_no_gradient_errors() {
    auto threw = false;
    try {
        Tensor t({2}, DType::Float32);
        t.backward(DType::Float32);
    } catch (const BeanTensor::ErrorHandling::NoGradient&) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        Tensor t({2}, DType::Float32, Device::CPU, true);
        static_cast<void>(t.gradient());
    } catch (const BeanTensor::ErrorHandling::NoGradient&) {
        threw = true;
    }
    CHECK(threw);
    // No input requires grad: the output never joins a graph.
    Tensor a({2, 2}, DType::Float32);
    a.set_one();
    auto r = a.relu();
    threw = false;
    try {
        r.backward(DType::Float32);
    } catch (const BeanTensor::ErrorHandling::NoGradient&) {
        threw = true;
    }
    CHECK(threw);
}

// =====================================================================
// sum_G / mean_G (full-tensor reduction backward)
// =====================================================================

static void test_sum_g_basic(const Device dev) {
    // Graph terminated by sum: dx = ones x seed, and the forward value matches.
    Tensor x({2, 3}, DType::Float32, dev, true);
    for (auto i = size_t{0}; i < 6; ++i) {
        x.set<float32_t>({i / 3, i % 3}, static_cast<float32_t>(i + 1));
    }
    auto y = x.sum();
    // Integers sum exactly in fp32 on every reduction order.
    CHECK(near64(y.at<float32_t>({0}), 21.0, 0.0));
    y.backward(DType::Float32);
    CHECK(x.gradient().get_dtype() == DType::Float32);
    for (auto i = size_t{0}; i < 6; ++i) {
        CHECK(idet::bit_cast_memcpy<uint32_t>(x.gradient().at<float32_t>({i / 3, i % 3})) == idet::bit_cast_memcpy<uint32_t>(1.0f));
    }
    // The root's own gradient is the ones seed.
    CHECK(idet::bit_cast_memcpy<uint32_t>(y.gradient().at<float32_t>({0})) == idet::bit_cast_memcpy<uint32_t>(1.0f));
}

static void test_sum_g_seed_and_grad_dtype(const Device dev) {
    // Non-trivial upstream (seed 2.5 through the .mul(c) root) under a
    // non-default assigned grad dtype: dx = 2.5 in Float16 everywhere; the
    // sum node's delta into the mul intermediate carries Float16 (probe).
    Tensor x({2, 2}, DType::Float32, dev, true);
    Tensor c({1}, DType::Float32, dev, false);
    x.set_one();
    c.set<float32_t>({0}, 2.5f);
    auto y = x.sum();
    auto root = y.mul(c);
    if (dev == Device::CPU) {
        tdetail::grad_probe_reset();
    }
    root.backward(DType::Float16);
    CHECK(x.gradient().get_dtype() == DType::Float16);
    CHECK(y.gradient().get_dtype() == DType::Float16);
    // 2.5 is exact in fp16 through the whole chain.
    for (auto i = size_t{0}; i < 4; ++i) {
        CHECK(near64(widen_lane(x.gradient().at<float16_t>({i / 2, i % 2})), 2.5, 0.0));
    }
    if (dev == Device::CPU) {
        // sum_g delivers exactly one intermediate delta (into y).
        CHECK(tdetail::grad_probe_count(DType::Float16) == 1);
    }
}

static void test_sum_g_in_matmul_graph(const Device dev) {
    // z = a.matmul(b), root = z.sum(): with a ones seed, da[i,k] = sum_j b[k,j]
    // and db[k,j] = sum_i a[i,k] — the same reference GEMM_G uses.
    Tensor a({2, 3}, DType::Float32, dev, true);
    Tensor b({3, 2}, DType::Float32, dev, true);
    std::vector<float64_t> av(6);
    std::vector<float64_t> bv(6);
    for (auto i = size_t{0}; i < 6; ++i) {
        av[i] = static_cast<float64_t>(i % 4) - 1.5;
        bv[i] = static_cast<float64_t>(i % 3) - 0.75;
        a.set<float32_t>({i / 3, i % 3}, static_cast<float32_t>(av[i]));
        b.set<float32_t>({i / 2, i % 2}, static_cast<float32_t>(bv[i]));
    }
    auto z = a.matmul(b);
    auto root = z.sum();
    if (dev == Device::CPU) {
        tdetail::grad_probe_reset();
    }
    root.backward(DType::Float32);
    if (dev == Device::CPU) {
        // sum_g delivers exactly one intermediate delta (into z).
        CHECK(tdetail::grad_probe_count(DType::Float32) == 1);
    }
    std::vector<float64_t> da;
    std::vector<float64_t> db;
    ref_gemm_grads_ones(av, bv, 1, 2, 3, 2, da, db);
    auto bad = size_t{0};
    for (auto i = size_t{0}; i < 6; ++i) {
        if (!near64(a.gradient().at<float32_t>({i / 3, i % 3}), da[i], 1e-5)) {
            ++bad;
        }
        if (!near64(b.gradient().at<float32_t>({i / 2, i % 2}), db[i], 1e-5)) {
            ++bad;
        }
    }
    CHECK(bad == 0);
}

static void test_sum_g_accumulation(const Device dev) {
    // Leaf grads accumulate across backward() calls through the same sum.
    Tensor x({2, 2}, DType::Float32, dev, true);
    x.set_one();
    auto y = x.sum();
    y.backward(DType::Float32);
    y.backward(DType::Float32);
    for (auto i = size_t{0}; i < 4; ++i) {
        CHECK(idet::bit_cast_memcpy<uint32_t>(x.gradient().at<float32_t>({i / 2, i % 2})) == idet::bit_cast_memcpy<uint32_t>(2.0f));
    }
    // The intermediate (root) grad holds only this call's seed.
    CHECK(idet::bit_cast_memcpy<uint32_t>(y.gradient().at<float32_t>({0})) == idet::bit_cast_memcpy<uint32_t>(1.0f));
}

static void test_mean_g_basic(const Device dev) {
    // dx = seed/numel broadcast: root = mean(x) * 2 over 4 elements -> 0.5 each.
    Tensor x({2, 2}, DType::Float64, dev, true);
    Tensor c({1}, DType::Float64, dev, false);
    for (auto i = size_t{0}; i < 4; ++i) {
        x.set<float64_t>({i / 2, i % 2}, static_cast<float64_t>(i + 1));
    }
    c.set<float64_t>({0}, 2.0);
    auto y = x.mean();
    CHECK(near64(y.at<float64_t>({0}), 2.5, 1e-12));
    auto root = y.mul(c);
    root.backward(DType::Float64);
    for (auto i = size_t{0}; i < 4; ++i) {
        CHECK(near64(x.gradient().at<float64_t>({i / 2, i % 2}), 0.5, 1e-12));
    }
}

// =====================================================================
// dtype pathways
// =====================================================================

template<typename T>
static void test_int_pathways(const Device dev, const DType dt) {
    // relu_G: signed select on x > 0; unsigned copies dy through.
    {
        Tensor x({1, 2}, dt, dev, true);
        x.set<T>({0, 0}, std::is_signed_v<T> ? static_cast<T>(-2) : T{0});
        x.set<T>({0, 1}, static_cast<T>(3));
        auto y = x.relu();
        y.backward(dt);
        CHECK(x.gradient().at<T>({0, 0}) == (std::is_signed_v<T> ? T{0} : T{1}));
        CHECK(x.gradient().at<T>({0, 1}) == T{1});
    }
    // bias_add_G: column sums of ones, wrap accumulation.
    {
        Tensor x({5, 3}, dt, dev, true);
        Tensor b({3}, dt, dev, true);
        x.set_zero();
        b.set_zero();
        auto y = x.bias_add(b);
        y.backward(dt);
        for (auto c = size_t{0}; c < 3; ++c) {
            CHECK(b.gradient().at<T>({c}) == static_cast<T>(5));
            CHECK(x.gradient().at<T>({1, c}) == T{1});
        }
    }
    // GEMM_G: exact small-integer sums.
    {
        Tensor a({2, 3}, dt, dev, true);
        Tensor b({3, 2}, dt, dev, true);
        std::vector<int64_t> av(6);
        std::vector<int64_t> bv(6);
        for (auto i = size_t{0}; i < 6; ++i) {
            av[i] = static_cast<int64_t>(i % 3) + 1;
            bv[i] = static_cast<int64_t>(i % 2) + 1;
            a.set<T>({i / 3, i % 3}, static_cast<T>(av[i]));
            b.set<T>({i / 2, i % 2}, static_cast<T>(bv[i]));
        }
        auto y = a.matmul(b);
        y.backward(dt);
        for (auto i = size_t{0}; i < 2; ++i) {
            for (auto k = size_t{0}; k < 3; ++k) {
                CHECK(a.gradient().at<T>({i, k}) == static_cast<T>(bv[k * 2] + bv[k * 2 + 1]));
            }
        }
        for (auto k = size_t{0}; k < 3; ++k) {
            for (auto j = size_t{0}; j < 2; ++j) {
                CHECK(b.gradient().at<T>({k, j}) == static_cast<T>(av[k] + av[3 + k]));
            }
        }
    }
    // softmaxCE_G: |grad| < 1/rows truncates to 0 — the pathway runs and
    // stores clamp-truncated values.
    {
        Tensor logits({2, 3}, dt, dev, true);
        Tensor targets({2}, DType::Int32, dev, false);
        logits.set_zero();
        targets.set<int32_t>({0}, 0);
        targets.set<int32_t>({1}, 2);
        auto loss = logits.softmax_ce(targets);
        loss.backward(dt);
        for (auto j = size_t{0}; j < 3; ++j) {
            CHECK(logits.gradient().at<T>({0, j}) == T{0});
            CHECK(logits.gradient().at<T>({1, j}) == T{0});
        }
    }
}

static void test_int4x2_pathways(const Device dev) {
    // relu_G per nibble.
    {
        Tensor x({1, 1}, DType::Int4x2, dev, true);
        x.set<int4x2_t>({0, 0}, pack_int4(-2, 3));
        auto y = x.relu();
        y.backward(DType::Int4x2);
        const auto g = x.gradient().at<int4x2_t>({0, 0});
        CHECK(int4_lo(g) == 0);
        CHECK(int4_hi(g) == 1);
    }
    // bias_add_G: 5 rows -> nibble sums of 5 in both lanes.
    {
        Tensor x({5, 2}, DType::Int4x2, dev, true);
        Tensor b({2}, DType::Int4x2, dev, true);
        x.set_zero();
        b.set_zero();
        auto y = x.bias_add(b);
        y.backward(DType::Int4x2);
        for (auto c = size_t{0}; c < 2; ++c) {
            const auto g = b.gradient().at<int4x2_t>({c});
            CHECK(int4_lo(g) == 5);
            CHECK(int4_hi(g) == 5);
        }
    }
    // GEMM_G lane-plane: lo lanes and hi lanes are independent GEMMs.
    {
        Tensor a({2, 2}, DType::Int4x2, dev, true);
        Tensor b({2, 2}, DType::Int4x2, dev, true);
        const auto alo = std::vector{1, -1, 2, 0};
        const auto ahi = std::vector{0, 1, -1, 1};
        const auto blo = std::vector{1, 2, -1, 1};
        const auto bhi = std::vector{2, -1, 0, 1};
        for (auto i = size_t{0}; i < 4; ++i) {
            a.set<int4x2_t>({i / 2, i % 2}, pack_int4(alo[i], ahi[i]));
            b.set<int4x2_t>({i / 2, i % 2}, pack_int4(blo[i], bhi[i]));
        }
        auto y = a.matmul(b);
        y.backward(DType::Int4x2);
        for (auto i = size_t{0}; i < 2; ++i) {
            for (auto k = size_t{0}; k < 2; ++k) {
                const auto g = a.gradient().at<int4x2_t>({i, k});
                CHECK(int4_lo(g) == blo[k * 2] + blo[k * 2 + 1]);
                CHECK(int4_hi(g) == bhi[k * 2] + bhi[k * 2 + 1]);
                const auto gb = b.gradient().at<int4x2_t>({i, k});
                CHECK(int4_lo(gb) == alo[i] + alo[2 + i]);
                CHECK(int4_hi(gb) == ahi[i] + ahi[2 + i]);
            }
        }
    }
    // softmaxCE_G: nibbles as consecutive classes; |grad| < 1 truncates to 0.
    {
        Tensor logits({2, 2}, DType::Int4x2, dev, true);
        Tensor targets({2}, DType::Int8, dev, false);
        logits.set_zero();
        targets.set<int8_t>({0}, 3);
        targets.set<int8_t>({1}, 0);
        auto loss = logits.softmax_ce(targets);
        loss.backward(DType::Int4x2);
        const auto g = logits.gradient().at<int4x2_t>({0, 1});
        CHECK(int4_lo(g) == 0);
        CHECK(int4_hi(g) == 0);
    }
}

template<typename H>
static void test_half_pathways(const Device dev, const DType dt) {
    // relu_G: selected dy stored bit-exact; masked lanes are +0.
    {
        Tensor x({1, 2}, dt, dev, true);
        x.set<H>({0, 0}, narrow_lane<H>(-1.0f));
        x.set<H>({0, 1}, narrow_lane<H>(2.0f));
        auto y = x.relu();
        y.backward(dt);
        CHECK(idet::bit_cast_memcpy<uint16_t>(x.gradient().at<H>({0, 0})) == 0u);
        CHECK(idet::bit_cast_memcpy<uint16_t>(x.gradient().at<H>({0, 1})) == idet::bit_cast_memcpy<uint16_t>(narrow_lane<H>(1.0f)));
    }
    // bias_add_G: fp32 accumulation, narrowed once.
    {
        Tensor x({4, 3}, dt, dev, true);
        Tensor b({3}, dt, dev, true);
        x.set_zero();
        b.set_zero();
        auto y = x.bias_add(b);
        y.backward(dt);
        for (auto c = size_t{0}; c < 3; ++c) {
            CHECK(idet::bit_cast_memcpy<uint16_t>(b.gradient().at<H>({c})) == idet::bit_cast_memcpy<uint16_t>(narrow_lane<H>(4.0f)));
        }
    }
    // GEMM_G: small exact values.
    {
        Tensor a({2, 3}, dt, dev, true);
        Tensor b({3, 2}, dt, dev, true);
        std::vector<float32_t> av(6);
        std::vector<float32_t> bv(6);
        for (auto i = size_t{0}; i < 6; ++i) {
            av[i] = static_cast<float32_t>(i % 3) - 1.0f;
            bv[i] = static_cast<float32_t>(i % 2) + 0.5f;
            a.set<H>({i / 3, i % 3}, narrow_lane<H>(av[i]));
            b.set<H>({i / 2, i % 2}, narrow_lane<H>(bv[i]));
        }
        auto y = a.matmul(b);
        y.backward(dt);
        for (auto i = size_t{0}; i < 2; ++i) {
            for (auto k = size_t{0}; k < 3; ++k) {
                CHECK(near64(widen_lane(a.gradient().at<H>({i, k})), bv[k * 2] + bv[k * 2 + 1], 0.0));
            }
        }
        for (auto k = size_t{0}; k < 3; ++k) {
            for (auto j = size_t{0}; j < 2; ++j) {
                CHECK(near64(widen_lane(b.gradient().at<H>({k, j})), av[k] + av[3 + k], 0.0));
            }
        }
    }
    // softmaxCE_G against the fp64 reference (bf16 has ~3 significant digits).
    {
        constexpr size_t rows = 2;
        constexpr size_t cols = 3;
        const auto lv = std::vector{1.0, 2.0, 0.5, -1.0, 0.25, 1.5};
        const auto tv = std::vector<size_t>{1, 2};
        Tensor logits({rows, cols}, dt, dev, true);
        Tensor targets({rows}, DType::Int16, dev, false);
        for (auto i = size_t{0}; i < rows * cols; ++i) {
            logits.set<H>({i / cols, i % cols}, narrow_lane<H>(static_cast<float32_t>(lv[i])));
        }
        for (auto r = size_t{0}; r < rows; ++r) {
            targets.set<int16_t>({r}, static_cast<int16_t>(tv[r]));
        }
        auto loss = logits.softmax_ce(targets);
        loss.backward(dt);
        const auto want = ref_sce_grad(lv, rows, cols, tv);
        for (auto i = size_t{0}; i < rows * cols; ++i) {
            CHECK(near64(widen_lane(logits.gradient().at<H>({i / cols, i % cols})), want[i], 5e-3));
        }
    }
}

template<typename P, typename H>
static void test_packed_pathways(const Device dev, const DType dt) {
    // relu_G per lane.
    {
        Tensor x({1, 1}, dt, dev, true);
        x.set<P>({0, 0}, pack_lanes<P, H>(-1.0f, 2.0f));
        auto y = x.relu();
        y.backward(dt);
        const auto g = x.gradient().at<P>({0, 0});
        CHECK(idet::bit_cast_memcpy<uint32_t>(g) == static_cast<uint32_t>(idet::bit_cast_memcpy<uint16_t>(narrow_lane<H>(1.0f))) << 16);
    }
    // bias_add_G: per-lane fp32 column sums.
    {
        Tensor x({4, 3}, dt, dev, true);
        Tensor b({3}, dt, dev, true);
        x.set_zero();
        b.set_zero();
        auto y = x.bias_add(b);
        y.backward(dt);
        for (auto c = size_t{0}; c < 3; ++c) {
            const auto g = b.gradient().at<P>({c});
            CHECK(near64(lane_lo_f32<H>(g), 4.0, 0.0));
            CHECK(near64(lane_hi_f32<H>(g), 4.0, 0.0));
        }
    }
    // GEMM_G lane-plane: hi lanes carry twice the lo-lane values.
    {
        Tensor a({2, 2}, dt, dev, true);
        Tensor b({2, 2}, dt, dev, true);
        const auto av = std::vector{1.0f, -0.5f, 2.0f, 0.25f};
        const auto bv = std::vector{0.5f, 1.0f, -1.0f, 1.5f};
        for (auto i = size_t{0}; i < 4; ++i) {
            a.set<P>({i / 2, i % 2}, pack_lanes<P, H>(av[i], 2.0f * av[i]));
            b.set<P>({i / 2, i % 2}, pack_lanes<P, H>(bv[i], 2.0f * bv[i]));
        }
        auto y = a.matmul(b);
        y.backward(dt);
        for (auto i = size_t{0}; i < 2; ++i) {
            for (auto k = size_t{0}; k < 2; ++k) {
                const auto ga = a.gradient().at<P>({i, k});
                const auto want_a = static_cast<float64_t>(bv[k * 2] + bv[k * 2 + 1]);
                CHECK(near64(lane_lo_f32<H>(ga), want_a, 0.0));
                CHECK(near64(lane_hi_f32<H>(ga), 2.0 * want_a, 0.0));
                const auto gb = b.gradient().at<P>({i, k});
                const auto want_b = static_cast<float64_t>(av[i] + av[2 + i]);
                CHECK(near64(lane_lo_f32<H>(gb), want_b, 0.0));
                CHECK(near64(lane_hi_f32<H>(gb), 2.0 * want_b, 0.0));
            }
        }
    }
    // softmaxCE_G: lanes unpack as consecutive classes (2 elements = 4 classes).
    {
        constexpr size_t rows = 2;
        constexpr size_t cols_store = 2;
        constexpr size_t cols = 4;
        const auto lv = std::vector{1.0, -0.5, 2.0, 0.0, 0.5, 0.5, -1.0, 1.5};
        const auto tv = std::vector<size_t>{3, 0};
        Tensor logits({rows, cols_store}, dt, dev, true);
        Tensor targets({rows}, DType::UInt32, dev, false);
        for (auto r = size_t{0}; r < rows; ++r) {
            for (auto e = size_t{0}; e < cols_store; ++e) {
                logits.set<P>({r, e}, pack_lanes<P, H>(static_cast<float32_t>(lv[r * cols + 2 * e]), static_cast<float32_t>(lv[r * cols + 2 * e + 1])));
            }
            targets.set<uint32_t>({r}, static_cast<uint32_t>(tv[r]));
        }
        auto loss = logits.softmax_ce(targets);
        loss.backward(dt);
        const auto want = ref_sce_grad(lv, rows, cols, tv);
        for (auto r = size_t{0}; r < rows; ++r) {
            for (auto e = size_t{0}; e < cols_store; ++e) {
                const auto g = logits.gradient().at<P>({r, e});
                CHECK(near64(lane_lo_f32<H>(g), want[r * cols + 2 * e], 5e-3));
                CHECK(near64(lane_hi_f32<H>(g), want[r * cols + 2 * e + 1], 5e-3));
            }
        }
    }
}

static void test_all_dtype_pathways(const Device dev) {
    test_int_pathways<int8_t>(dev, DType::Int8);
    test_int_pathways<uint8_t>(dev, DType::UInt8);
    test_int_pathways<int16_t>(dev, DType::Int16);
    test_int_pathways<uint16_t>(dev, DType::UInt16);
    test_int_pathways<int32_t>(dev, DType::Int32);
    test_int_pathways<uint32_t>(dev, DType::UInt32);
    test_int_pathways<int64_t>(dev, DType::Int64);
    test_int_pathways<uint64_t>(dev, DType::UInt64);
    test_int4x2_pathways(dev);
    test_half_pathways<float16_t>(dev, DType::Float16);
    test_half_pathways<bfloat16_t>(dev, DType::BFloat16);
    test_packed_pathways<float16x2_t, float16_t>(dev, DType::Float16x2);
    test_packed_pathways<bfloat16x2_t, bfloat16_t>(dev, DType::BFloat16x2);
}

// elementwise backward tests (defined below, next to the other grad-dtype suites)
static void test_ew_binary_gradients(Device);
static void test_ew_one_side(Device);
static void test_ew_broadcast_reduce(Device);
static void test_ew_scalar_gradients(Device);
static void test_ew_scalar_chain_gradients(Device);
static void test_ew_exp_log_gradients(Device);
static void test_ew_accumulation(Device);

static void run_device_suite(const Device dev) {
    test_relu_g_f32(dev);
    test_relu_g_threaded<float32_t>(dev, DType::Float32);
    test_relu_g_threaded<float64_t>(dev, DType::Float64);
    test_bias_add_g_f32(dev);
    test_bias_add_g_threaded<float32_t>(dev, DType::Float32);
    test_bias_add_g_threaded<float64_t>(dev, DType::Float64);
    test_gemm_g(dev);
    test_gemm_g_one_side(dev);
    test_softmax_ce_g(dev);
    test_softmax_ce_g_threaded(dev);
    test_softmax_ce_g_zero_sum(dev);
    test_accumulation_and_zero_grad(dev);
    test_fan_out(dev);
    test_sum_g_basic(dev);
    test_sum_g_seed_and_grad_dtype(dev);
    test_sum_g_in_matmul_graph(dev);
    test_sum_g_accumulation(dev);
    test_mean_g_basic(dev);
    test_all_dtype_pathways(dev);
    test_ew_binary_gradients(dev);
    test_ew_one_side(dev);
    test_ew_broadcast_reduce(dev);
    test_ew_scalar_gradients(dev);
    test_ew_scalar_chain_gradients(dev);
    test_ew_exp_log_gradients(dev);
    test_ew_accumulation(dev);
}

#if defined(USE_HIP) || defined(USE_CUDA)
static void test_chain_gpu_cpu_parity() {
    const auto c = Chain{};
    Tensor x({Chain::n, Chain::d}, DType::Float64, Device::CPU, true);
    Tensor w({Chain::d, Chain::h}, DType::Float64, Device::CPU, true);
    Tensor b({Chain::h}, DType::Float64, Device::CPU, true);
    Tensor t({Chain::n}, DType::Int32, Device::CPU, false);
    c.load<float64_t>(x, w, b, t);
    auto xg = x.clone(true);
    auto wg = w.clone(true);
    auto bg = b.clone(true);
    auto tg = t.clone(true);
    xg.to(Device::GPU);
    wg.to(Device::GPU);
    bg.to(Device::GPU);
    tg.to(Device::GPU);
    const auto loss_cpu = chain_forward_backward<float64_t>(x, w, b, t, true, DType::Float64);
    const auto loss_gpu = chain_forward_backward<float64_t>(xg, wg, bg, tg, true, DType::Float64);
    CHECK(near64(loss_cpu, loss_gpu, 1e-12));
    auto bad = size_t{0};
    for (auto i = size_t{0}; i < Chain::d * Chain::h; ++i) {
        if (!near64(w.gradient().at<float64_t>({i / Chain::h, i % Chain::h}), wg.gradient().at<float64_t>({i / Chain::h, i % Chain::h}), 1e-12)) {
            ++bad;
        }
    }
    for (auto i = size_t{0}; i < Chain::n * Chain::d; ++i) {
        if (!near64(x.gradient().at<float64_t>({i / Chain::d, i % Chain::d}), xg.gradient().at<float64_t>({i / Chain::d, i % Chain::d}), 1e-12)) {
            ++bad;
        }
    }
    for (auto i = size_t{0}; i < Chain::h; ++i) {
        if (!near64(b.gradient().at<float64_t>({i}), bg.gradient().at<float64_t>({i}), 1e-12)) {
            ++bad;
        }
    }
    CHECK(bad == 0);
}

static void test_grad_follows_to() {
    Tensor x({2}, DType::Float32, Device::CPU, true);
    x.set_one();
    auto y = x.relu();
    y.backward(DType::Float32);
    x.to(Device::GPU);
    CHECK(idet::bit_cast_memcpy<uint32_t>(x.gradient().at<float32_t>({1})) == idet::bit_cast_memcpy<uint32_t>(1.0f));
}
#endif

// =====================================================================
// Per-call grad dtype override: backward(DType)
// =====================================================================

namespace {
    /**
     * matmul -> bias_add -> relu -> softmax_ce on 2x3 x 3x4 CPU tensors of one
     * dtype; every intermediate is a member so it outlives backward(), as the
     * graph's raw parent pointers require. Fixed seeds make two instances
     * value-identical.
     */
    struct GradDtypeChain {
        Tensor x;
        Tensor w;
        Tensor b;
        Tensor t;
        Tensor z;
        Tensor zb;
        Tensor a;
        Tensor loss;
        explicit GradDtypeChain(const DType dt)
            : x({2, 3}, dt, Device::CPU, true),
              w({3, 4}, dt, Device::CPU, true),
              b({4}, dt, Device::CPU, true),
              t({2}, DType::Int32, Device::CPU, false),
              z({1}, dt, Device::CPU, false),
              zb({1}, dt, Device::CPU, false),
              a({1}, dt, Device::CPU, false),
              loss({1}, dt, Device::CPU, false) {
            x.random(-0.5, 0.5, 11);
            w.random(-0.5, 0.5, 22);
            b.random(-0.25, 0.25, 33);
            t.set<int32_t>({0}, 1);
            t.set<int32_t>({1}, 3);
            z = x.matmul(w);
            zb = z.bias_add(b);
            a = zb.relu();
            loss = a.softmax_ce(t);
        }
    };
}

// The SAME graph, backward() called with five different grad dtypes. The probe
// (tdetail::grad_probe_count) records deltas delivered into INTERMEDIATE
// (node-owning) tensors; the intermediate grads' own dtype is checked directly
// via get_dtype(). Per call the intermediates a, zb and z each receive one
// delta (softmax_ce_g -> a, relu_g -> zb, bias_add_g -> z); gemm_g feeds only
// the leaves x/w, so exactly 3 intermediate deliveries land per call.
static void test_grad_dtype_override_same_graph() {
    auto c = GradDtypeChain(DType::Float32);
    for (const auto cases = std::vector{DType::Float64, DType::Float16, DType::Float32, DType::Int8, DType::Int32}; const auto gd : cases) {
        tdetail::grad_probe_reset();
        c.loss.backward(gd);
        CHECK(tdetail::grad_probe_count(gd) == 3);
        for (const auto other : cases) {
            if (other != gd) {
                CHECK(tdetail::grad_probe_count(other) == 0);
            }
        }
        // Intermediate (node-owning) grads hold THIS call's delta in THIS call's dtype.
        CHECK(c.loss.gradient().get_dtype() == gd);
        CHECK(c.a.gradient().get_dtype() == gd);
        CHECK(c.zb.gradient().get_dtype() == gd);
        CHECK(c.z.gradient().get_dtype() == gd);
        CHECK(c.x.has_grad() && c.w.has_grad() && c.b.has_grad());
        std::cout << "backward(" << BeanTensor::DataTypes::dtype_to_string(gd)
                  << ") on fp32 graph: " << tdetail::grad_probe_count(gd)
                  << " intermediate deltas carried the assigned dtype\n";
    }
    // Leaf grads accumulate across calls: the first call (Float64) pinned the
    // buffer dtype; later calls' deltas converted into it.
    CHECK(c.x.gradient().get_dtype() == DType::Float64);
    CHECK(c.w.gradient().get_dtype() == DType::Float64);
    CHECK(c.b.gradient().get_dtype() == DType::Float64);
    auto bad = size_t{0};
    for (auto i = size_t{0}; i < 6; ++i) {
        if (!std::isfinite(get_flat<float64_t>(c.x.gradient(), {2, 3}, i))) {
            ++bad;
        }
    }
    for (auto i = size_t{0}; i < 12; ++i) {
        if (!std::isfinite(get_flat<float64_t>(c.w.gradient(), {3, 4}, i))) {
            ++bad;
        }
    }
    for (auto i = size_t{0}; i < 4; ++i) {
        if (!std::isfinite(c.b.gradient().at<float64_t>({i}))) {
            ++bad;
        }
    }
    CHECK(bad == 0);
}

// fp64 grads on an fp32 graph must match fp32 grads on the identical fp32
// graph (same seeds): both differentiate the same fp32-valued forward, so only
// rounding separates them.
static void test_grad_dtype_override_numeric() {
    auto c32 = GradDtypeChain(DType::Float32);
    auto c64 = GradDtypeChain(DType::Float32);
    c32.loss.backward(DType::Float32);
    c64.loss.backward(DType::Float64);
    CHECK(c32.x.gradient().get_dtype() == DType::Float32);
    CHECK(c64.x.gradient().get_dtype() == DType::Float64);
    auto bad = size_t{0};
    for (auto i = size_t{0}; i < 6; ++i) {
        if (!near64(get_flat<float32_t>(c32.x.gradient(), {2, 3}, i),
                    get_flat<float64_t>(c64.x.gradient(), {2, 3}, i), 1e-5)) {
            ++bad;
        }
    }
    for (auto i = size_t{0}; i < 12; ++i) {
        if (!near64(get_flat<float32_t>(c32.w.gradient(), {3, 4}, i),
                    get_flat<float64_t>(c64.w.gradient(), {3, 4}, i), 1e-5)) {
            ++bad;
        }
    }
    for (auto i = size_t{0}; i < 4; ++i) {
        if (!near64(c32.b.gradient().at<float32_t>({i}),
                    c64.b.gradient().at<float64_t>({i}), 1e-5)) {
            ++bad;
        }
    }
    CHECK(bad == 0);
}

// fp16 grads on an fp32 graph, and fp32 grads on an fp16 graph: leaf grads
// carry the assigned dtype, intermediates carry it (probe), and values stay
// within half-precision reach of the matching-dtype reference.
static void test_grad_dtype_override_half_mixed() {
    // fp16 grad dtype on an fp32 graph.
    {
        auto c = GradDtypeChain(DType::Float32);
        auto ref = GradDtypeChain(DType::Float32);
        tdetail::grad_probe_reset();
        c.loss.backward(DType::Float16);
        ref.loss.backward(DType::Float32);
        CHECK(tdetail::grad_probe_count(DType::Float16) == 3);
        CHECK(c.z.gradient().get_dtype() == DType::Float16);
        CHECK(c.x.gradient().get_dtype() == DType::Float16);
        CHECK(c.w.gradient().get_dtype() == DType::Float16);
        CHECK(c.b.gradient().get_dtype() == DType::Float16);
        auto bad = size_t{0};
        for (auto i = size_t{0}; i < 12; ++i) {
            const auto got = widen_lane(get_flat<float16_t>(c.w.gradient(), {3, 4}, i));
            if (
                const auto want = get_flat<float32_t>(ref.w.gradient(), {3, 4}, i);
                !near64(got, want, 1e-2)
            ) {
                ++bad;
            }
        }
        for (auto i = size_t{0}; i < 4; ++i) {
            const auto got = widen_lane(c.b.gradient().at<float16_t>({i}));
            if (
                const auto want = ref.b.gradient().at<float32_t>({i});
                !near64(got, want, 1e-2)
            ) {
                ++bad;
            }
        }
        CHECK(bad == 0);
    }
    // fp32 grad dtype on an fp16 graph.
    {
        auto c = GradDtypeChain(DType::Float16);
        auto ref = GradDtypeChain(DType::Float16);
        tdetail::grad_probe_reset();
        c.loss.backward(DType::Float32);
        ref.loss.backward(DType::Float16);
        CHECK(tdetail::grad_probe_count(DType::Float32) == 3);
        CHECK(c.z.gradient().get_dtype() == DType::Float32);
        CHECK(c.x.gradient().get_dtype() == DType::Float32);
        CHECK(c.w.gradient().get_dtype() == DType::Float32);
        CHECK(c.b.gradient().get_dtype() == DType::Float32);
        auto bad = size_t{0};
        for (auto i = size_t{0}; i < 12; ++i) {
            const auto got = get_flat<float32_t>(c.w.gradient(), {3, 4}, i);
            if (
                const auto want = widen_lane(get_flat<float16_t>(ref.w.gradient(), {3, 4}, i));
                !near64(got, want, 1e-2)
            ) {
                ++bad;
            }
        }
        for (auto i = size_t{0}; i < 4; ++i) {
            const auto got = c.b.gradient().at<float32_t>({i});
            if (
                const auto want = widen_lane(ref.b.gradient().at<float16_t>({i}));
                !near64(got, want, 1e-2)
            ) {
                ++bad;
            }
        }
        CHECK(bad == 0);
    }
}

// Integer grad dtypes stay assignable: the full chain runs with int8 deltas
// (wrap/clamp, NaN -> 0 through the single conversion rule), and an exact
// truncation check on a minimal relu graph (seed 1 -> int32 1 through the
// positive lane, 0 through the non-positive lane).
static void test_grad_dtype_override_int_sanity() {
    auto c = GradDtypeChain(DType::Float32);
    tdetail::grad_probe_reset();
    c.loss.backward(DType::Int8);
    CHECK(tdetail::grad_probe_count(DType::Int8) == 3);
    CHECK(c.z.gradient().get_dtype() == DType::Int8);
    CHECK(c.x.gradient().get_dtype() == DType::Int8);
    CHECK(c.w.gradient().get_dtype() == DType::Int8);
    CHECK(c.b.gradient().get_dtype() == DType::Int8);
    // Exact integer semantics on a one-node graph.
    Tensor x({1, 2}, DType::Float32, Device::CPU, true);
    x.set<float32_t>({0, 0}, 2.0f);
    x.set<float32_t>({0, 1}, -1.0f);
    auto y = x.relu();
    y.backward(DType::Int32);
    CHECK(x.gradient().get_dtype() == DType::Int32);
    CHECK(x.gradient().at<int32_t>({0, 0}) == 1);
    CHECK(x.gradient().at<int32_t>({0, 1}) == 0);
    CHECK(y.gradient().at<int32_t>({0, 0}) == 1);
}

// sgd_step auto-converts a grad whose dtype differs from the weight dtype.
static void test_sgd_step_mixed_dtype() {
    Tensor w({2}, DType::Float32, Device::CPU, true);
    w.set<float32_t>({0}, 1.5f);
    w.set<float32_t>({1}, 2.5f);
    auto y = w.relu();
    y.backward(DType::Float64);
    CHECK(w.gradient().get_dtype() == DType::Float64);
    w.sgd_step(0.1);
    w.sync();
    CHECK(near64(w.at<float32_t>({0}), 1.4, 1e-6));
    CHECK(near64(w.at<float32_t>({1}), 2.4, 1e-6));
}

// =====================================================================
// elementwise backward: add/sub/mul/div/pow (tensor-tensor and
// tensor-scalar) and exp/log — analytic checks against hand-computed values,
// broadcast gradient reduction, grad-dtype probes
// =====================================================================

/**
 * Root = (x op w).mul(c) with c requires_grad=false: the elementwise node's
 * upstream delta is c's values (here 2 everywhere), not the ones seed, so the
 * local derivatives are exercised against a non-trivial upstream gradient.
 * Analytic: add dx=dy, dw=dy; sub dx=dy, dw=-dy; mul dx=dy*w, dw=dy*x;
 * div dx=dy/w, dw=-dy*x/w^2; pow dx=dy*w*x^(w-1), dw=dy*x^w*ln(x) (x>0).
 */
static float64_t ew_ref_dx(const ElementwiseOp op, const float64_t x, const float64_t w, const float64_t dy) {
    switch (op) {
        case ElementwiseOp::Add: return dy;
        case ElementwiseOp::Sub: return dy;
        case ElementwiseOp::Mul: return dy * w;
        case ElementwiseOp::Div: return dy / w;
        case ElementwiseOp::Pow: return dy * w * std::pow(x, w - 1.0);
        [[unlikely]] default: throw BeanTensor::ErrorHandling::UnReachableCode();
    }
}

static float64_t ew_ref_dw(const ElementwiseOp op, const float64_t x, const float64_t w, const float64_t dy) {
    switch (op) {
        case ElementwiseOp::Add: return dy;
        case ElementwiseOp::Sub: return -dy;
        case ElementwiseOp::Mul: return dy * x;
        case ElementwiseOp::Div: return -dy * x / (w * w);
        case ElementwiseOp::Pow: return dy * std::pow(x, w) * std::log(x);
        [[unlikely]] default: throw BeanTensor::ErrorHandling::UnReachableCode();
    }
}

static Tensor ew_apply(const Tensor& x, const Tensor& w, const ElementwiseOp op) {
    switch (op) {
        case ElementwiseOp::Add: return x.add(w);
        case ElementwiseOp::Sub: return x.sub(w);
        case ElementwiseOp::Mul: return x.mul(w);
        case ElementwiseOp::Div: return x.div(w);
        case ElementwiseOp::Pow: return x.pow(w);
        [[unlikely]] default: throw BeanTensor::ErrorHandling::UnReachableCode();
    }
}

template<typename T>
static void run_ew_binary_case(const Device dev, const DType dt, const ElementwiseOp op,
                               const std::vector<float64_t>& xs, const std::vector<float64_t>& ws,
                               const float64_t eps) {
    constexpr size_t n = 4;
    constexpr auto cval = float64_t{2.0}; // upstream gradient into the op node
    Tensor x({2, 2}, dt, dev, true);
    Tensor w({2, 2}, dt, dev, true);
    Tensor c({2, 2}, dt, dev, false);
    for (auto i = size_t{0}; i < n; ++i) {
        x.set<T>({i / 2, i % 2}, static_cast<T>(xs[i]));
        w.set<T>({i / 2, i % 2}, static_cast<T>(ws[i]));
        c.set<T>({i / 2, i % 2}, static_cast<T>(cval));
    }
    auto p = ew_apply(x, w, op);
    auto root = p.mul(c);
    root.backward(dt);
    auto bad = size_t{0};
    for (auto i = size_t{0}; i < n; ++i) {
        const auto want_dx = ew_ref_dx(op, xs[i], ws[i], cval);
        const auto want_dw = ew_ref_dw(op, xs[i], ws[i], cval);
        if (!near64(x.gradient().at<T>({i / 2, i % 2}), want_dx, eps)) {
            ++bad;
        }
        if (!near64(w.gradient().at<T>({i / 2, i % 2}), want_dw, eps)) {
            ++bad;
        }
    }
    CHECK(bad == 0);
}

static void test_ew_binary_gradients(const Device dev) {
    const auto xs = std::vector{1.0, 2.0, 3.0, 4.0};
    const auto ws = std::vector{5.0, 6.0, 7.0, 8.0};
    for (const auto op : std::vector{ElementwiseOp::Add, ElementwiseOp::Sub, ElementwiseOp::Mul, ElementwiseOp::Div}) {
        run_ew_binary_case<float64_t>(dev, DType::Float64, op, xs, ws, 1e-12);
        run_ew_binary_case<float32_t>(dev, DType::Float32, op, xs, ws, 1e-4);
    }
    // pow: x > 0 (ln(x) in dw), small integer exponents for exact fp32 values.
    const auto ps = std::vector{1.0, 2.0, 3.0, 4.0};
    const auto pw = std::vector{2.0, 2.0, 2.0, 2.0};
    run_ew_binary_case<float64_t>(dev, DType::Float64, ElementwiseOp::Pow, ps, pw, 1e-12);
    run_ew_binary_case<float32_t>(dev, DType::Float32, ElementwiseOp::Pow, ps, pw, 1e-3);
}

// Only one side requires grad: the other receives nothing.
static void test_ew_one_side(const Device dev) {
    Tensor x({2}, DType::Float32, dev, true);
    Tensor w({2}, DType::Float32, dev, false);
    x.set<float32_t>({0}, 3.0f);
    x.set<float32_t>({1}, 4.0f);
    w.set<float32_t>({0}, 5.0f);
    w.set<float32_t>({1}, 6.0f);
    auto y = x.mul(w);
    y.backward(DType::Float32);
    CHECK(x.has_grad());
    CHECK(!w.has_grad());
    CHECK(near64(x.gradient().at<float32_t>({0}), 5.0, 1e-6));
    CHECK(near64(x.gradient().at<float32_t>({1}), 6.0, 1e-6));
}

static void test_ew_broadcast_reduce(const Device dev) {
    // {2,3} x {3}: dw[j] = sum_i dy[i,j]*x[i,j] (dy = 2 via the .mul(c) root).
    {
        Tensor x({2, 3}, DType::Float64, dev, true);
        Tensor w({3}, DType::Float64, dev, true);
        Tensor c({2, 3}, DType::Float64, dev, false);
        const auto xv = std::vector{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        for (auto i = size_t{0}; i < 6; ++i) {
            x.set<float64_t>({i / 3, i % 3}, xv[i]);
            c.set<float64_t>({i / 3, i % 3}, 2.0);
        }
        w.set<float64_t>({0}, 2.0);
        w.set<float64_t>({1}, 3.0);
        w.set<float64_t>({2}, 4.0);
        auto p = x.mul(w);
        auto root = p.mul(c);
        root.backward(DType::Float64);
        const auto want_dw = std::vector{10.0, 14.0, 18.0}; // 2*(x[0,j]+x[1,j])
        const auto want_dx = std::vector{4.0, 6.0, 8.0, 4.0, 6.0, 8.0}; // 2*w[j]
        auto bad = size_t{0};
        for (auto j = size_t{0}; j < 3; ++j) {
            if (!near64(w.gradient().at<float64_t>({j}), want_dw[j], 1e-12)) {
                ++bad;
            }
            for (auto i = size_t{0}; i < 2; ++i) {
                if (!near64(x.gradient().at<float64_t>({i, j}), want_dx[i * 3 + j], 1e-12)) {
                    ++bad;
                }
            }
        }
        CHECK(bad == 0);
    }
    // Rank-3 general broadcast: {2,1,4} x {3,4} -> {2,3,4}, ones upstream.
    {
        Tensor a({2, 1, 4}, DType::Float64, dev, true);
        Tensor b({3, 4}, DType::Float64, dev, true);
        for (auto i = size_t{0}; i < 2; ++i) {
            for (auto k = size_t{0}; k < 4; ++k) {
                a.set<float64_t>({i, 0, k}, static_cast<float64_t>(1 + i * 4 + k));
            }
        }
        for (auto j = size_t{0}; j < 3; ++j) {
            for (auto k = size_t{0}; k < 4; ++k) {
                b.set<float64_t>({j, k}, static_cast<float64_t>(1 + j * 4 + k));
            }
        }
        auto root = a.mul(b);
        root.backward(DType::Float64);
        auto bad = size_t{0};
        for (auto i = size_t{0}; i < 2; ++i) {
            for (auto k = size_t{0}; k < 4; ++k) {
                // da[i,0,k] = sum_j b[j,k] = 3k + 15.
                if (!near64(a.gradient().at<float64_t>({i, 0, k}), 3.0 * static_cast<float64_t>(k) + 15.0, 1e-12)) {
                    ++bad;
                }
            }
        }
        for (auto j = size_t{0}; j < 3; ++j) {
            for (auto k = size_t{0}; k < 4; ++k) {
                // db[j,k] = sum_i a[i,0,k] = 2k + 6.
                if (!near64(b.gradient().at<float64_t>({j, k}), 2.0 * static_cast<float64_t>(k) + 6.0, 1e-12)) {
                    ++bad;
                }
            }
        }
        CHECK(bad == 0);
    }
    // add/sub broadcast with a rank-1 operand: dw[j] = sum_i dy[i,j] (resp. -).
    for (const auto op : std::vector{ElementwiseOp::Add, ElementwiseOp::Sub}) {
        Tensor x({2, 3}, DType::Float64, dev, true);
        Tensor w({3}, DType::Float64, dev, true);
        Tensor c({2, 3}, DType::Float64, dev, false);
        for (auto i = size_t{0}; i < 6; ++i) {
            x.set<float64_t>({i / 3, i % 3}, static_cast<float64_t>(1 + i));
            c.set<float64_t>({i / 3, i % 3}, 2.0);
        }
        x.set_zero();
        w.set_zero();
        auto p = ew_apply(x, w, op);
        auto root = p.mul(c);
        root.backward(DType::Float64);
        const auto sign = op == ElementwiseOp::Add ? 1.0 : -1.0;
        auto bad = size_t{0};
        for (auto j = size_t{0}; j < 3; ++j) {
            // dw[j] = sum_i (+/-)dy[i,j] = (+/-)(2+2).
            if (!near64(w.gradient().at<float64_t>({j}), sign * 4.0, 1e-12)) {
                ++bad;
            }
            for (auto i = size_t{0}; i < 2; ++i) {
                if (!near64(x.gradient().at<float64_t>({i, j}), 2.0, 1e-12)) {
                    ++bad;
                }
            }
        }
        CHECK(bad == 0);
    }
    // Div broadcast reduction (dy = 2 via the .mul(c) root):
    // da = sum over broadcast dims of dy/b, db = sum over broadcast dims of -dy*a/b^2.
    const auto run_div_broadcast = [&]<typename T>(const DType dt, const float64_t eps) {
        {
            // {2,3} / {3}: right operand broadcast, db[j] sums over rows.
            Tensor a({2, 3}, dt, dev, true);
            Tensor b({3}, dt, dev, true);
            Tensor c({2, 3}, dt, dev, false);
            for (auto i = size_t{0}; i < 6; ++i) {
                a.set<T>({i / 3, i % 3}, static_cast<T>(1 + i));
                c.set<T>({i / 3, i % 3}, static_cast<T>(2.0));
            }
            b.set<T>({0}, static_cast<T>(2.0));
            b.set<T>({1}, static_cast<T>(3.0));
            b.set<T>({2}, static_cast<T>(4.0));
            auto p = a.div(b);
            auto root = p.mul(c);
            root.backward(dt);
            const auto want_da = std::vector{1.0, 2.0 / 3.0, 0.5}; // 2 / b[j]
            const auto want_db = std::vector{-2.5, -14.0 / 9.0, -1.125}; // -2 * sum_i a[i,j] / b[j]^2
            auto bad = size_t{0};
            for (auto j = size_t{0}; j < 3; ++j) {
                if (!near64(b.gradient().at<T>({j}), want_db[j], eps)) {
                    ++bad;
                }
                for (auto i = size_t{0}; i < 2; ++i) {
                    if (!near64(a.gradient().at<T>({i, j}), want_da[j], eps)) {
                        ++bad;
                    }
                }
            }
            CHECK(bad == 0);
        }
        {
            // {3} / {2,3}: left operand broadcast, da[j] sums over rows.
            Tensor a({3}, dt, dev, true);
            Tensor b({2, 3}, dt, dev, true);
            Tensor c({2, 3}, dt, dev, false);
            a.set<T>({0}, static_cast<T>(2.0));
            a.set<T>({1}, static_cast<T>(3.0));
            a.set<T>({2}, static_cast<T>(4.0));
            for (auto i = size_t{0}; i < 6; ++i) {
                b.set<T>({i / 3, i % 3}, static_cast<T>(1 + i));
                c.set<T>({i / 3, i % 3}, static_cast<T>(2.0));
            }
            auto p = a.div(b);
            auto root = p.mul(c);
            root.backward(dt);
            const auto want_da = std::vector{2.5, 1.4, 1.0}; // sum_i 2 / b[i,j]
            const auto want_db = std::vector{-4.0, -1.5, -8.0 / 9.0, -0.25, -0.24, -2.0 / 9.0}; // -2 * a[j] / b[i,j]^2
            auto bad = size_t{0};
            for (auto i = size_t{0}; i < 2; ++i) {
                for (auto j = size_t{0}; j < 3; ++j) {
                    if (!near64(b.gradient().at<T>({i, j}), want_db[i * 3 + j], eps)) {
                        ++bad;
                    }
                }
            }
            for (auto j = size_t{0}; j < 3; ++j) {
                if (!near64(a.gradient().at<T>({j}), want_da[j], eps)) {
                    ++bad;
                }
            }
            CHECK(bad == 0);
        }
    };
    run_div_broadcast.operator()<float64_t>(DType::Float64, 1e-12);
    run_div_broadcast.operator()<float32_t>(DType::Float32, 1e-5);
}

static void test_ew_scalar_gradients(const Device dev) {
    // Root = (x op s).mul(c): dx = dy * d(x op s)/dx with dy = 2, s scalar.
    const auto xv = std::vector{1.0, 2.0, 4.0};
    auto run = [&](const ElementwiseOp op, const float64_t s, const std::vector<float64_t>& want) {
        Tensor x({3}, DType::Float64, dev, true);
        Tensor c({3}, DType::Float64, dev, false);
        for (auto i = size_t{0}; i < 3; ++i) {
            x.set<float64_t>({i}, xv[i]);
            c.set<float64_t>({i}, 2.0);
        }
        Tensor t({1}, DType::Float64, dev, false);
        switch (op) {
            case ElementwiseOp::Add: t = x.add(s); break;
            case ElementwiseOp::Sub: t = x.sub(s); break;
            case ElementwiseOp::Mul: t = x.mul(s); break;
            case ElementwiseOp::Div: t = x.div(s); break;
            case ElementwiseOp::Pow: t = x.pow(s); break;
            [[unlikely]] default: throw BeanTensor::ErrorHandling::UnReachableCode();
        }
        auto root = t.mul(c);
        root.backward(DType::Float64);
        auto bad = size_t{0};
        for (auto i = size_t{0}; i < 3; ++i) {
            if (!near64(x.gradient().at<float64_t>({i}), want[i], 1e-12)) {
                ++bad;
            }
        }
        CHECK(bad == 0);
    };
    run(ElementwiseOp::Add, 5.0, std::vector{2.0, 2.0, 2.0});            // dx = dy
    run(ElementwiseOp::Sub, 5.0, std::vector{2.0, 2.0, 2.0});            // dx = dy
    run(ElementwiseOp::Mul, 3.0, std::vector{6.0, 6.0, 6.0});            // dx = dy*s
    run(ElementwiseOp::Div, 2.0, std::vector{1.0, 1.0, 1.0});            // dx = dy/s
    run(ElementwiseOp::Pow, 2.0, std::vector{4.0, 8.0, 16.0});           // dx = dy*s*x^(s-1)
}

static void test_ew_scalar_chain_gradients(const Device dev) {
    // Two tensor-scalar nodes in one graph: each backward must read its own
    // stored scalar, so a lost or swapped scalar changes the result.
    // mul chain: dx = (-4) * 2.5 = -10.  div chain: dx = (1 / 4) * (1 / -0.5) = -0.5.
    {
        Tensor x({3}, DType::Float64, dev, true);
        for (auto i = size_t{0}; i < 3; ++i) {
            x.set<float64_t>({i}, 1.0 + static_cast<float64_t>(i));
        }
        auto y = x.mul(2.5);
        auto z = y.mul(-4.0);
        z.backward(DType::Float64);
        for (auto i = size_t{0}; i < 3; ++i) {
            CHECK(near64(x.gradient().at<float64_t>({i}), -10.0, 1e-12));
        }
    }
    {
        Tensor x({3}, DType::Float64, dev, true);
        for (auto i = size_t{0}; i < 3; ++i) {
            x.set<float64_t>({i}, 1.0 + static_cast<float64_t>(i));
        }
        auto y = x.div(-0.5);
        auto z = y.div(4.0);
        z.backward(DType::Float64);
        for (auto i = size_t{0}; i < 3; ++i) {
            CHECK(near64(x.gradient().at<float64_t>({i}), -0.5, 1e-12));
        }
    }
}

static void test_ew_exp_log_gradients(const Device dev) {
    const auto xv = std::vector{1.0, 2.0, 3.0};
    // exp with a non-trivial upstream: root = exp(x)*c -> dx = 2*exp(x).
    {
        Tensor x({3}, DType::Float64, dev, true);
        Tensor c({3}, DType::Float64, dev, false);
        for (auto i = size_t{0}; i < 3; ++i) {
            x.set<float64_t>({i}, xv[i]);
            c.set<float64_t>({i}, 2.0);
        }
        auto e = x.exp();
        auto root = e.mul(c);
        root.backward(DType::Float64);
        auto bad = size_t{0};
        for (auto i = size_t{0}; i < 3; ++i) {
            if (!near64(x.gradient().at<float64_t>({i}), 2.0 * std::exp(xv[i]), 1e-12)) {
                ++bad;
            }
        }
        CHECK(bad == 0);
    }
    // log: root = log(x)*c -> dx = 2/x.
    {
        Tensor x({3}, DType::Float64, dev, true);
        Tensor c({3}, DType::Float64, dev, false);
        for (auto i = size_t{0}; i < 3; ++i) {
            x.set<float64_t>({i}, xv[i]);
            c.set<float64_t>({i}, 2.0);
        }
        auto l = x.log();
        auto root = l.mul(c);
        root.backward(DType::Float64);
        auto bad = size_t{0};
        for (auto i = size_t{0}; i < 3; ++i) {
            if (!near64(x.gradient().at<float64_t>({i}), 2.0 / xv[i], 1e-12)) {
                ++bad;
            }
        }
        CHECK(bad == 0);
    }
    // exp-then-log: root = log(exp(x)) -> dx = 1 by the chain rule
    // (1/exp(x) * exp(x)). The log node's delta lands in the exp()
    // intermediate — one probe hit per call.
    if (dev == Device::CPU) {
        tdetail::grad_probe_reset();
    }
    {
        Tensor x({3}, DType::Float64, dev, true);
        for (auto i = size_t{0}; i < 3; ++i) {
            x.set<float64_t>({i}, xv[i]);
        }
        auto e = x.exp();
        auto root = e.log();
        root.backward(DType::Float64);
        auto bad = size_t{0};
        for (auto i = size_t{0}; i < 3; ++i) {
            if (!near64(x.gradient().at<float64_t>({i}), 1.0, 1e-12)) {
                ++bad;
            }
        }
        CHECK(bad == 0);
    }
    if (dev == Device::CPU) {
        CHECK(tdetail::grad_probe_count(DType::Float64) == 1);
    }
}

static void test_ew_accumulation(const Device dev) {
    // Leaf grads accumulate across backward() calls on an elementwise graph.
    Tensor x({2}, DType::Float32, dev, true);
    Tensor w({2}, DType::Float32, dev, true);
    x.set<float32_t>({0}, 1.0f);
    x.set<float32_t>({1}, 2.0f);
    w.set<float32_t>({0}, 3.0f);
    w.set<float32_t>({1}, 4.0f);
    auto y = x.mul(w);
    y.backward(DType::Float32);
    y.backward(DType::Float32);
    CHECK(near64(x.gradient().at<float32_t>({0}), 6.0f, 1e-6));
    CHECK(near64(x.gradient().at<float32_t>({1}), 8.0f, 1e-6));
    CHECK(near64(w.gradient().at<float32_t>({0}), 2.0f, 1e-6));
    CHECK(near64(w.gradient().at<float32_t>({1}), 4.0f, 1e-6));
}

static void test_ew_no_grad_error() {
    // No input requires grad: the output never joins a graph.
    Tensor x({2}, DType::Float32);
    Tensor w({2}, DType::Float32);
    x.set_one();
    w.set_one();
    auto threw = false;
    try {
        auto y = x.add(w);
        y.backward(DType::Float32);
    } catch (const BeanTensor::ErrorHandling::NoGradient&) {
        threw = true;
    }
    CHECK(threw);
}

// Per-call grad dtype override on an elementwise chain: z = x.mul(w) feeds
// root = z.add(v). add_g delivers one intermediate delta into z per call
// (mul_g feeds only the leaves), so exactly one probe hit lands per call,
// in the assigned dtype; z's grad is rebuilt per call while the leaf grads
// pin the first call's dtype and accumulate.
static void test_ew_grad_dtype_probe() {
    Tensor x({2, 3}, DType::Float32, Device::CPU, true);
    Tensor w({3}, DType::Float32, Device::CPU, true);
    Tensor v({2, 3}, DType::Float32, Device::CPU, true);
    Tensor z({1}, DType::Float32, Device::CPU, false);
    Tensor root({1}, DType::Float32, Device::CPU, false);
    x.set_zero();
    w.set_zero();
    v.set_zero();
    z = x.mul(w);
    root = z.add(v);

    tdetail::grad_probe_reset();
    root.backward(DType::Float64);
    CHECK(tdetail::grad_probe_count(DType::Float64) == 1);
    CHECK(tdetail::grad_probe_count(DType::Float16) == 0);
    CHECK(z.gradient().get_dtype() == DType::Float64);
    CHECK(x.gradient().get_dtype() == DType::Float64);

    root.backward(DType::Float16);
    CHECK(tdetail::grad_probe_count(DType::Float16) == 1);
    CHECK(tdetail::grad_probe_count(DType::Float64) == 1); // previous call's hit stays
    CHECK(z.gradient().get_dtype() == DType::Float16);
    CHECK(x.gradient().get_dtype() == DType::Float64); // leaf buffer pinned by call 1

    root.backward(DType::Int32);
    CHECK(tdetail::grad_probe_count(DType::Int32) == 1);
    CHECK(z.gradient().get_dtype() == DType::Int32);
    CHECK(x.gradient().get_dtype() == DType::Float64);
    CHECK(v.gradient().get_dtype() == DType::Float64);
}

// Integer assigned grad dtypes on elementwise graphs: mul/div/add/sub compute
// their deltas natively in the assigned dtype (wrap/truncate semantics); pow
// and exp need a transcendental, so they compute in fp32 and boundary-cast
// (truncate) the delta into the assigned dtype at delivery.
static void test_ew_int_grad_dtype() {
    // mul: dy = 1 -> dx = w, dw = x, exact in int32.
    {
        Tensor x({3}, DType::Int32, Device::CPU, true);
        Tensor w({3}, DType::Int32, Device::CPU, true);
        x.set<int32_t>({0}, 1);
        x.set<int32_t>({1}, 2);
        x.set<int32_t>({2}, 3);
        w.set<int32_t>({0}, 4);
        w.set<int32_t>({1}, 5);
        w.set<int32_t>({2}, 6);
        auto y = x.mul(w);
        y.backward(DType::Int32);
        CHECK(x.gradient().at<int32_t>({0}) == 4);
        CHECK(x.gradient().at<int32_t>({1}) == 5);
        CHECK(x.gradient().at<int32_t>({2}) == 6);
        CHECK(w.gradient().at<int32_t>({0}) == 1);
        CHECK(w.gradient().at<int32_t>({2}) == 3);
    }
    // div: dy = 1 -> dx = 1/w (truncating integer div), dw = -x/w^2.
    {
        Tensor x({3}, DType::Int32, Device::CPU, true);
        Tensor w({3}, DType::Int32, Device::CPU, true);
        x.set<int32_t>({0}, 4);
        x.set<int32_t>({1}, 6);
        x.set<int32_t>({2}, 8);
        w.set<int32_t>({0}, 2);
        w.set<int32_t>({1}, 3);
        w.set<int32_t>({2}, 4);
        auto y = x.div(w);
        y.backward(DType::Int32);
        CHECK(x.gradient().at<int32_t>({0}) == 0); // 1/2 truncates to 0
        CHECK(w.gradient().at<int32_t>({0}) == -1); // -4/4
        CHECK(w.gradient().at<int32_t>({1}) == 0);  // -6/9 truncates to 0
        CHECK(w.gradient().at<int32_t>({2}) == 0);  // -8/16 truncates to 0
    }
    // mul under int8, with a broadcast reduction: dw[j] = sum_i x[i,j].
    {
        Tensor x({2, 3}, DType::Int8, Device::CPU, true);
        Tensor w({3}, DType::Int8, Device::CPU, true);
        x.set<int8_t>({0, 0}, 1);
        x.set<int8_t>({0, 1}, 2);
        x.set<int8_t>({0, 2}, 3);
        x.set<int8_t>({1, 0}, 4);
        x.set<int8_t>({1, 1}, 5);
        x.set<int8_t>({1, 2}, 6);
        w.set_zero();
        auto y = x.mul(w);
        y.backward(DType::Int8);
        CHECK(w.gradient().at<int8_t>({0}) == 5); // 1 + 4
        CHECK(w.gradient().at<int8_t>({1}) == 7); // 2 + 5
        CHECK(w.gradient().at<int8_t>({2}) == 9); // 3 + 6
    }
    // pow under int32 assigned dtype: fp32 compute, truncating boundary cast.
    // dx = trunc(w*x^(w-1)) = [2, 4, 8]; dw = trunc(x^w * ln(x)) = [0, 2, 22].
    {
        Tensor x({3}, DType::Float32, Device::CPU, true);
        Tensor w({3}, DType::Float32, Device::CPU, true);
        const auto xv = std::vector{1.0f, 2.0f, 4.0f};
        for (auto i = size_t{0}; i < 3; ++i) {
            x.set<float32_t>({i}, xv[i]);
            w.set<float32_t>({i}, 2.0f);
        }
        auto y = x.pow(w);
        y.backward(DType::Int32);
        CHECK(x.gradient().at<int32_t>({0}) == 2);
        CHECK(x.gradient().at<int32_t>({1}) == 4);
        CHECK(x.gradient().at<int32_t>({2}) == 8);
        CHECK(w.gradient().at<int32_t>({0}) == 0);  // 1^2 * ln(1) = 0
        CHECK(w.gradient().at<int32_t>({1}) == 2);  // trunc(4 * 0.693) = 2
        CHECK(w.gradient().at<int32_t>({2}) == 22); // trunc(16 * 1.386) = 22
    }
    // exp under int32 assigned dtype: dx = trunc(exp(x)) = [1, 2, 7].
    {
        Tensor x({3}, DType::Float32, Device::CPU, true);
        x.set<float32_t>({0}, 0.0f);
        x.set<float32_t>({1}, 1.0f);
        x.set<float32_t>({2}, 2.0f);
        auto y = x.exp();
        y.backward(DType::Int32);
        CHECK(x.gradient().at<int32_t>({0}) == 1);
        CHECK(x.gradient().at<int32_t>({1}) == 2); // trunc(2.718)
        CHECK(x.gradient().at<int32_t>({2}) == 7); // trunc(7.389)
    }
}

int main() {
    test_simd_kernel_parity();
    run_device_suite(Device::CPU);
    test_chain_finite_difference();
    test_no_gradient_errors();
    test_grad_dtype_override_same_graph();
    test_grad_dtype_override_numeric();
    test_grad_dtype_override_half_mixed();
    test_grad_dtype_override_int_sanity();
    test_sgd_step_mixed_dtype();
    test_ew_no_grad_error();
    test_ew_grad_dtype_probe();
    test_ew_int_grad_dtype();
#if defined(USE_HIP) || defined(USE_CUDA)
    if (gpu_available()) {
        run_device_suite(Device::GPU);
        test_chain_gpu_cpu_parity();
        test_grad_follows_to();
    } else {
        std::cout << "GPU not available — skipping GPU autograd tests\n";
    }
#endif

    if (failures == 0) {
        std::cout << "All autograd tests passed\n";
    } else {
        std::cerr << failures << " autograd test(s) failed\n";
    }
    return failures == 0 ? 0 : 1;
}
