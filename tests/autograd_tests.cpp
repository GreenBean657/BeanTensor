// BeanTensorV5 autograd tests — framework-free assertion suite.
// Covers: the four backward pathways (GEMM_G incl. batched, bias_add_G,
// relu_G, softmaxCE_G) via the forward ops' auto-wired graph and
// Tensor::backward(); analytic checks on fp32/fp64, finite-difference checks
// (fp64) of a full relu(x·W + b) -> softmax_ce chain, threaded/SIMD paths
// (large tensors), every dtype pathway (fp16/bf16, packed x2, all integer
// dtypes, int4x2), gradient accumulation across backward() calls, fan-out
// DAGs, zero_grad, error paths, and — when a backend + device exist — the
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
#include "Math/Autograd/AutogradKernels.h"
#include "Tensor/Tensor.h"

#if defined(USE_HIP)
#include <hip/hip_runtime.h>
#elif defined(USE_CUDA)
#include <cuda_runtime.h>
#endif

using BeanTensor::Tensors::Tensor;
using BeanTensor::Tensors::Device;
using BeanTensor::float16_t;
using BeanTensor::float16x2_t;
using BeanTensor::bfloat16_t;
using BeanTensor::bfloat16x2_t;
using BeanTensor::float32_t;
using BeanTensor::float64_t;
using BeanTensor::int4x2_t;
using DType = BeanTensor::DataTypes::DType;
namespace idet = BeanTensor::Intrinsics::detail;

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
    const auto n = static_cast<int32_t>(idet::bit_cast_memcpy<uint8_t>(v) >> 4);
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
        t.set<T>({i / (shape[1] * shape[2]), (i / shape[2]) % shape[1], i % shape[2]}, v);
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
    return t.at<T>({i / (shape[1] * shape[2]), (i / shape[2]) % shape[1], i % shape[2]});
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
    if (ag::autograd_avx2_available()) {
        check_kernel_tier<float32_t, uint32_t>(ag::grad_acc_f32_range_avx2, ag::relu_g_f32_range_avx2, ag::bias_add_g_f32_colsum_avx2);
        check_kernel_tier<float64_t, uint64_t>(ag::grad_acc_f64_range_avx2, ag::relu_g_f64_range_avx2, ag::bias_add_g_f64_colsum_avx2);
    } else {
        std::cout << "AVX2 not available — skipping AVX2 autograd kernel parity\n";
    }
    if (ag::autograd_avx512_available()) {
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
    y.backward();
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
    y.backward();
    auto g = x.gradient().clone(true);
    if (dev == Device::GPU) {
        g.to(Device::CPU);
    }
    g.sync();
    auto bad = size_t{0};
    for (auto i = size_t{0}; i < n; i += 37) {
        const auto want = (static_cast<T>(i % 7) - static_cast<T>(3) + static_cast<T>(0.5)) > T{0} ? T{1} : T{0};
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
    y.backward();
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
    y.backward();
    for (auto c = size_t{0}; c < cols; ++c) {
        CHECK(std::fabs(b.gradient().at<T>({c}) - static_cast<T>(rows)) <= T{0});
    }
}

// =====================================================================
// GEMM_G
// =====================================================================

template<typename T>
static void run_gemm_g_case(const Device dev, const DType dt, const size_t batch, const size_t m, const size_t k, const size_t n, const float64_t eps) {
    const auto a_shape = batch == 0 ? std::vector<size_t>{m, k} : std::vector<size_t>{batch, m, k};
    const auto b_shape = batch == 0 ? std::vector<size_t>{k, n} : std::vector<size_t>{batch, k, n};
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
    y.backward();
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
    y.backward();
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
    const auto lv = std::vector<float64_t>{1.0, 2.0, 3.0, -1.0, 0.5, -1.0, 2.0, 0.25, -3.0, 0.0, 1.5, 1.5};
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
    loss.backward();
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
        tv[r] = (r * 7) % cols;
        targets.set<int32_t>({r}, static_cast<int32_t>(tv[r]));
    }
    if (dev == Device::GPU) {
        logits.to(Device::GPU);
        targets.to(Device::GPU);
    }
    auto loss = logits.softmax_ce(targets);
    loss.backward();
    const auto want = ref_sce_grad(lv, rows, cols, tv);
    auto bad = size_t{0};
    for (auto i = size_t{0}; i < rows * cols; i += 13) {
        if (!near64(logits.gradient().at<float64_t>({i / cols, i % cols}), want[i], 1e-15)) {
            ++bad;
        }
    }
    CHECK(bad == 0);
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
    T chain_forward_backward(Tensor& x, Tensor& w, Tensor& b, Tensor& t, const bool do_backward) {
        auto z = x.matmul(w);
        auto zb = z.bias_add(b);
        auto a = zb.relu();
        auto loss = a.softmax_ce(t);
        if (do_backward) {
            loss.backward();
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
    static_cast<void>(chain_forward_backward<float64_t>(x, w, b, t, true));

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
            const auto up = chain_forward_backward<float64_t>(x, w, b, t, false);
            put(orig - step);
            const auto down = chain_forward_backward<float64_t>(x, w, b, t, false);
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
    y.backward();
    y.backward(); // leaf grads accumulate across calls
    CHECK(idet::bit_cast_memcpy<uint32_t>(x.gradient().at<float32_t>({0, 0})) == idet::bit_cast_memcpy<uint32_t>(2.0f));
    CHECK(idet::bit_cast_memcpy<uint32_t>(x.gradient().at<float32_t>({0, 1})) == 0u);
    CHECK(idet::bit_cast_memcpy<uint32_t>(x.gradient().at<float32_t>({1, 0})) == idet::bit_cast_memcpy<uint32_t>(2.0f));
    // The intermediate (root) grad holds only the current call's seed.
    CHECK(idet::bit_cast_memcpy<uint32_t>(y.gradient().at<float32_t>({0, 0})) == idet::bit_cast_memcpy<uint32_t>(1.0f));
    x.zero_grad();
    CHECK(!x.has_grad());
    y.backward();
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
    y.backward();
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
        t.backward();
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
        r.backward();
    } catch (const BeanTensor::ErrorHandling::NoGradient&) {
        threw = true;
    }
    CHECK(threw);
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
        y.backward();
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
        y.backward();
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
        y.backward();
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
        loss.backward();
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
        y.backward();
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
        y.backward();
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
        const auto alo = std::vector<int32_t>{1, -1, 2, 0};
        const auto ahi = std::vector<int32_t>{0, 1, -1, 1};
        const auto blo = std::vector<int32_t>{1, 2, -1, 1};
        const auto bhi = std::vector<int32_t>{2, -1, 0, 1};
        for (auto i = size_t{0}; i < 4; ++i) {
            a.set<int4x2_t>({i / 2, i % 2}, pack_int4(alo[i], ahi[i]));
            b.set<int4x2_t>({i / 2, i % 2}, pack_int4(blo[i], bhi[i]));
        }
        auto y = a.matmul(b);
        y.backward();
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
        loss.backward();
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
        y.backward();
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
        y.backward();
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
        y.backward();
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
        const auto lv = std::vector<float64_t>{1.0, 2.0, 0.5, -1.0, 0.25, 1.5};
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
        loss.backward();
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
        y.backward();
        const auto g = x.gradient().at<P>({0, 0});
        CHECK(idet::bit_cast_memcpy<uint32_t>(g) == (static_cast<uint32_t>(idet::bit_cast_memcpy<uint16_t>(narrow_lane<H>(1.0f))) << 16));
    }
    // bias_add_G: per-lane fp32 column sums.
    {
        Tensor x({4, 3}, dt, dev, true);
        Tensor b({3}, dt, dev, true);
        x.set_zero();
        b.set_zero();
        auto y = x.bias_add(b);
        y.backward();
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
        const auto av = std::vector<float32_t>{1.0f, -0.5f, 2.0f, 0.25f};
        const auto bv = std::vector<float32_t>{0.5f, 1.0f, -1.0f, 1.5f};
        for (auto i = size_t{0}; i < 4; ++i) {
            a.set<P>({i / 2, i % 2}, pack_lanes<P, H>(av[i], 2.0f * av[i]));
            b.set<P>({i / 2, i % 2}, pack_lanes<P, H>(bv[i], 2.0f * bv[i]));
        }
        auto y = a.matmul(b);
        y.backward();
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
        const auto lv = std::vector<float64_t>{1.0, -0.5, 2.0, 0.0, 0.5, 0.5, -1.0, 1.5};
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
        loss.backward();
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
    test_accumulation_and_zero_grad(dev);
    test_fan_out(dev);
    test_all_dtype_pathways(dev);
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
    const auto loss_cpu = chain_forward_backward<float64_t>(x, w, b, t, true);
    const auto loss_gpu = chain_forward_backward<float64_t>(xg, wg, bg, tg, true);
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
    y.backward();
    x.to(Device::GPU);
    CHECK(idet::bit_cast_memcpy<uint32_t>(x.gradient().at<float32_t>({1})) == idet::bit_cast_memcpy<uint32_t>(1.0f));
}
#endif

int main() {
    test_simd_kernel_parity();
    run_device_suite(Device::CPU);
    test_chain_finite_difference();
    test_no_gradient_errors();
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
