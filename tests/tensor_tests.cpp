/**
 * @file tensor_tests.cpp
 * @author GreenBean
 * @brief Tensor tests - framework-free assertion suite.
 * Covers CPU fills, scalar and runtime-gated AVX512/AVX2 conversions, clamp semantics, NaN/Inf flag consistency,
 * printing, clone, and the GPU paths incl. GPU<->CPU conversion parity when a backend and device exist.
 * @version 0.3
 * @date 2026-10-18
 */
#include <atomic>
#include <bit>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "Helpers/Hardware.h"
#include "Intrinsics/AVX2/DTypeConversions.h"
#include "Intrinsics/AVX512/DTypeConversions.h"
#include "Intrinsics/Dispatcher.h"
#include "Intrinsics/Native/DTypeConversions.h"
#include "Intrinsics/Shared/SharedIntrinsics.h"
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
using BeanTensor::int4x2_t;
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

static void test_fills_cpu() {
    Tensor f32({2, 2}, DType::Float32);
    f32.set_one();
    f32.sync();
    CHECK(f32.contents_to_string() == "[[1, 1],\n [1, 1]]");

    f32.set_zero();
    f32.sync();
    CHECK(f32.contents_to_string() == "[[0, 0],\n [0, 0]]");

    Tensor i32({4}, DType::Int32);
    i32.set_one();
    i32.sync();
    CHECK(i32.contents_to_string() == "[1, 1, 1, 1]");

    Tensor f16({2}, DType::Float16);
    f16.set_one();
    f16.sync();
    CHECK(f16.contents_to_string() == "[1, 1]");

    Tensor bf16({2}, DType::BFloat16);
    bf16.set_one();
    bf16.sync();
    CHECK(bf16.contents_to_string() == "[1, 1]");

    // Packed dtypes: each element prints both lanes.
    Tensor f16x2({2}, DType::Float16x2);
    f16x2.set_one();
    f16x2.sync();
    CHECK(f16x2.contents_to_string() == "[(1, 1), (1, 1)]");

    Tensor bf16x2({2}, DType::BFloat16x2);
    bf16x2.set_one();
    bf16x2.sync();
    CHECK(bf16x2.contents_to_string() == "[(1, 1), (1, 1)]");

    bf16x2.set_zero();
    bf16x2.sync();
    CHECK(bf16x2.contents_to_string() == "[(0, 0), (0, 0)]");
}

static void test_random_cpu() {
    Tensor a({8}, DType::Float32);
    Tensor b({8}, DType::Float32);
    a.random(0.0, 1.0, 42);
    b.random(0.0, 1.0, 42);
    a.sync();
    b.sync();
    CHECK(a.contents_to_string() == b.contents_to_string()); // seeded determinism

    Tensor packed({4}, DType::BFloat16x2);
    packed.random(-1.0, 1.0, 7);
    packed.sync();
    CHECK(packed.contents_to_string().front() == '['); // smoke: no throw, printable

    bool threw = false;
    try {
        Tensor bad({4}, DType::Float16);
        bad.random(0.0, 1e9, 1); // out of fp16 range
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

static void test_scalar_intrinsics() {
    namespace det = BeanTensor::Intrinsics::detail;
    // fp32 -> bf16 of 1.0f is exactly 0x3F80.
    const auto one_bf = det::unaccel_fp32_to_bf16_untracked(1.0f);
    CHECK(std::bit_cast<uint16_t>(one_bf) == 0x3F80);
    CHECK(std::bit_cast<uint32_t>(det::unaccel_bf16_to_fp32(one_bf)) == std::bit_cast<uint32_t>(1.0f));
    CHECK(std::bit_cast<uint32_t>(det::unaccel_fp16_to_fp32_untracked(det::unaccel_fp32_to_fp16_untracked(2.5f))) == std::bit_cast<uint32_t>(2.5f));
}

static void test_standard2standard() {
    namespace det = BeanTensor::Intrinsics::detail;
    bool flag = false;

    CHECK(std::bit_cast<uint32_t>(det::standard2standard<float32_t, float32_t>(2.5f, flag)) == std::bit_cast<uint32_t>(2.5f));
    CHECK(!flag);
    CHECK(std::bit_cast<uint64_t>(det::standard2standard<float32_t, float64_t>(1.5f, flag)) == std::bit_cast<uint64_t>(1.5));
    CHECK(det::standard2standard<int32_t, int64_t>(-7, flag) == -7);
    CHECK(std::bit_cast<uint32_t>(det::standard2standard<float32_t, float32_t>(0.0f, flag)) == std::bit_cast<uint32_t>(0.0f));

    // Overflow clamps; PERMISSIVE never flags.
    CHECK(std::bit_cast<uint32_t>(det::standard2standard<float64_t, float32_t>(1e300, flag, ConversionClampMethod::PERMISSIVE)) == std::bit_cast<uint32_t>(std::numeric_limits<float32_t>::max()));
    CHECK(!flag);
    // STRICT flags finite overflow.
    CHECK(std::bit_cast<uint32_t>(det::standard2standard<float64_t, float32_t>(1e300, flag, ConversionClampMethod::STRICT)) == std::bit_cast<uint32_t>(std::numeric_limits<float32_t>::max()));
    CHECK(flag);
    // Mixed-sign integral range checks.
    flag = false;
    CHECK(det::standard2standard<uint64_t, int8_t>(1000, flag, ConversionClampMethod::PERMISSIVE) == 127);
    CHECK(det::standard2standard<int32_t, uint8_t>(-5, flag, ConversionClampMethod::PERMISSIVE) == 0);
}

// NaN/Inf flag contract across clamp methods and conversion paths:
// NaN flags under STRICT and ALLOW_INF; Inf flags under STRICT only.
// Values pass through regardless (NaN stays NaN, Inf stays Inf).
static void test_nan_inf_flags() {
    namespace det = BeanTensor::Intrinsics::detail;
    using BeanTensor::float16_t;

    constexpr float32_t nan32 = std::numeric_limits<float32_t>::quiet_NaN();
    constexpr float32_t inf32 = std::numeric_limits<float32_t>::infinity();
    const auto nan16 = det::bit_cast_memcpy<float16_t>(static_cast<uint16_t>(0x7E00));
    const auto inf16 = det::bit_cast_memcpy<float16_t>(static_cast<uint16_t>(0x7C00));

    constexpr ConversionClampMethod methods[3] = {
        ConversionClampMethod::STRICT, ConversionClampMethod::ALLOW_INF, ConversionClampMethod::PERMISSIVE};

    for (int32_t m = 0; m < 3; ++m) {
        bool flag = false;
        constexpr bool want_nan[3] = {true, true, false};
        constexpr bool want_inf[3] = {true, false, false};

        flag = false; (void)det::unaccel_fp32_to_fp16_tracked(nan32, flag, methods[m]);
        CHECK(flag == want_nan[m]);
        flag = false; (void)det::unaccel_fp32_to_fp16_tracked(inf32, flag, methods[m]);
        CHECK(flag == want_inf[m]);

        flag = false; (void)det::unaccel_fp32_to_bf16_tracked(nan32, flag, methods[m]);
        CHECK(flag == want_nan[m]);
        flag = false; (void)det::unaccel_fp32_to_bf16_tracked(inf32, flag, methods[m]);
        CHECK(flag == want_inf[m]);

        flag = false; (void)det::unaccel_fp16_to_fp32_tracked(nan16, flag, methods[m]);
        CHECK(flag == want_nan[m]);
        flag = false; (void)det::unaccel_fp16_to_fp32_tracked(inf16, flag, methods[m]);
        CHECK(flag == want_inf[m]);

        flag = false; (void)det::scalar_convert<float32_t, int32_t>(nan32, flag, methods[m]);
        CHECK(flag == want_nan[m]);

        flag = false; (void)det::standard2standard<float32_t, float64_t>(nan32, flag, methods[m]);
        CHECK(flag == want_nan[m]);
    }

    bool flag = false;
    const float16_t n16 = det::unaccel_fp32_to_fp16_tracked(nan32, flag, ConversionClampMethod::PERMISSIVE);
    const auto nb = det::bit_cast_memcpy<uint16_t>(n16);
    CHECK((nb & 0x7C00) == 0x7C00 && (nb & 0x03FF) != 0); // NaN stays NaN (quiet)
    const float16_t i16 = det::unaccel_fp32_to_fp16_tracked(inf32, flag, ConversionClampMethod::PERMISSIVE);
    CHECK(det::bit_cast_memcpy<uint16_t>(i16) == 0x7C00); // Inf stays Inf
}

static void test_avx512_runtime_gate() {
    alignas(64) float32_t src[16];
    alignas(64) BeanTensor::bfloat16_t dst[16]{};
    for (int32_t i = 0; i < 16; ++i) {
        src[i] = static_cast<float32_t>(i) * 0.5f;
    }

    // The Accelerated targets compile these kernels in even when the CPU
    // lacks AVX512BF16, and the runtime gate's answer is only authoritative
    // inside the kernel TU (this TU's compile flags must not decide it).
    // A CPUIllegalInstruction means the gate refused — the CPU lacks
    // AVX512BF16 and dst must be untouched. Reaching the result checks means
    // the gate let the kernels run — the conversion must then be exact. Both
    // outcomes pass; a wrong answer or a missing refusal fails.
    try {
        std::atomic flag{false};
        for (const auto futures = BeanTensor::Intrinsics::detail::avx512_fp32_to_bf16(src, dst, 16, flag, ConversionClampMethod::PERMISSIVE); auto& f : futures) {
            f.get();
        }
        for (int32_t i = 0; i < 16; ++i) {
            CHECK(std::bit_cast<uint32_t>(BeanTensor::Intrinsics::detail::unaccel_bf16_to_fp32(dst[i])) == std::bit_cast<uint32_t>(src[i]));
        }
    } catch (const BeanTensor::ErrorHandling::CPUIllegalInstruction&) {
        for (const auto& d : dst) {
            CHECK(std::bit_cast<uint16_t>(d) == uint16_t{0});
        }
    }
}

// This TU compiles with -mavx2 -mf16c only, so its own IncludedCPUFeatures()
// copy reports every AVX512 feature absent — gating AVX512 test halves on it
// would silently skip them even on AVX512 machines (an AVX512 parity test
// degrading to scalar-only is an error, not a skip). The kernel TUs carry the
// per-file -mavx512* flags, so their internal supported-gate is the only
// authoritative check. Probe through a tiny kernel launch: a thrown
// CPUIllegalInstruction means the CPU genuinely lacks the feature; reaching
// the return means the kernel ran, so the gated parity checks must execute.
static bool probe_avx512_bf16() {
    namespace bd = BeanTensor::Intrinsics::detail;
    constexpr float32_t src[1] = {1.0f};
    BeanTensor::bfloat16_t dst[1]{};
    try {
        for (std::atomic flag{false}; auto& f : bd::avx512_fp32_to_bf16(src, dst, 1, flag, ConversionClampMethod::PERMISSIVE)) {
            f.get();
        }
        return true;
    } catch (const BeanTensor::ErrorHandling::CPUIllegalInstruction&) {
        return false;
    }
}

static bool probe_avx512_f() {
    namespace bd = BeanTensor::Intrinsics::detail;
    constexpr float32_t src[1] = {1.0f};
    BeanTensor::float16_t dst[1]{};
    try {
        for (std::atomic flag{false}; auto& f : bd::avx512_fp32_to_fp16(src, dst, 1, flag, ConversionClampMethod::PERMISSIVE)) {
            f.get();
        }
        return true;
    } catch (const BeanTensor::ErrorHandling::CPUIllegalInstruction&) {
        return false;
    }
}

static bool probe_avx512_dq() {
    namespace bd = BeanTensor::Intrinsics::detail;
    constexpr int64_t src[1] = {1};
    float64_t dst[1]{};
    try {
        for (auto& f : bd::avx512_int64_to_fp64(src, dst, 1)) {
            f.get();
        }
        return true;
    } catch (const BeanTensor::ErrorHandling::CPUIllegalInstruction&) {
        return false;
    }
}

// Scalar fallbacks and AVX512 kernels must be bit-for-bit identical on every
// input — including NaN payloads, Inf, overflow and subnormals.
static void test_scalar_avx512_parity() {
    namespace bd = BeanTensor::Intrinsics::detail;
    using BeanTensor::bfloat16_t;
    using BeanTensor::float16_t;
    using BeanTensor::float32_t;

    std::vector<uint32_t> specials = {
        0x00000000, 0x80000000,             // +-0
        0x00000001, 0x007FFFFF, 0x00800000, // fp32 subnormals / min normal
        0x3F800000, 0xC0200000,             // 1.0, -2.5
        0x477FDFFF, 0x477FE000, 0x477FE001, 0x477FF000, // around FP16_MAX (65504)
        0x33800000, 0x34000000, 0x35555555, 0x387FF000, // fp16 subnormal territory
        0x7F800000, 0xFF800000,             // +-Inf
        0x7F800001, 0x7FC00000, 0x7FFFFFFF, 0xFF800001, // NaNs (low-payload sNaN, qNaN, full payload)
        0x7F7FFFFF, 0xFF7FFFFF,             // +-fp32 max
    };
    const auto seed = std::random_device{}();
    std::mt19937 rng(seed);
    for (int32_t i = 0; i < 1 << 18; ++i) {
        specials.push_back(rng()); // random bit patterns incl. NaN/subnormal
    }

    std::atomic flag{false};
    std::vector<float32_t> src(specials.size());
    for (size_t i = 0; i < specials.size(); ++i) {
        src[i] = std::bit_cast<float32_t>(specials[i]);
    }

    // Odd sizes exercise the vector tails and the thread-chunk remainder —
    // every element must be converted exactly once.
    const std::vector<size_t> sizes = {1, 7, 37, src.size()};

    if (probe_avx512_bf16()) {
        // fp32 -> bf16: scalar vs AVX512, at several sizes
        for (const size_t n : sizes) {
            std::vector<bfloat16_t> a(n), b(n);
            for (size_t i = 0; i < n; ++i) {
                a[i] = bd::unaccel_fp32_to_bf16_untracked(src[i]);
            }
            for (auto& f : bd::avx512_fp32_to_bf16(src.data(), b.data(), n, flag, ConversionClampMethod::PERMISSIVE)) {
                f.get();
            }
            for (size_t i = 0; i < n; ++i) {
                CHECK(bd::bit_cast_memcpy<uint16_t>(a[i]) == bd::bit_cast_memcpy<uint16_t>(b[i]));
            }
        }

        // bf16 -> fp32: exhaustive over all 65536 patterns
        std::vector<bfloat16_t> all16(65536);
        std::vector<float32_t> fa(65536), fb(65536);
        for (uint32_t i = 0; i < 65536; ++i) {
            all16[i] = bd::bit_cast_memcpy<bfloat16_t>(static_cast<uint16_t>(i));
        }
        for (uint32_t i = 0; i < 65536; ++i) {
            fa[i] = bd::unaccel_bf16_to_fp32(all16[i]);
        }
        for (auto& f : bd::avx512_bf16_to_fp32(all16.data(), fb.data(), 65536)) {
            f.get();
        }
        for (uint32_t i = 0; i < 65536; ++i) {
            CHECK(std::bit_cast<uint32_t>(fa[i]) == std::bit_cast<uint32_t>(fb[i]));
        }
    }

    if (probe_avx512_f()) {
        // fp32 -> fp16: scalar vs AVX512, at several sizes
        for (const size_t n : sizes) {
            std::vector<float16_t> a(n), b(n);
            for (size_t i = 0; i < n; ++i) {
                a[i] = bd::unaccel_fp32_to_fp16_untracked(src[i]);
            }
            for (auto& f : bd::avx512_fp32_to_fp16(src.data(), b.data(), n, flag, ConversionClampMethod::PERMISSIVE)) {
                f.get();
            }
            for (size_t i = 0; i < n; ++i) {
                CHECK(bd::bit_cast_memcpy<uint16_t>(a[i]) == bd::bit_cast_memcpy<uint16_t>(b[i]));
            }
        }

        // fp16 -> fp32: exhaustive over all 65536 patterns
        std::vector<float16_t> all16(65536);
        std::vector<float32_t> fa(65536), fb(65536);
        for (uint32_t i = 0; i < 65536; ++i) {
            all16[i] = bd::bit_cast_memcpy<float16_t>(static_cast<uint16_t>(i));
        }
        for (uint32_t i = 0; i < 65536; ++i) {
            fa[i] = bd::unaccel_fp16_to_fp32_untracked(all16[i]);
        }
        for (auto& f : bd::avx512_fp16_to_fp32(all16.data(), fb.data(), 65536, flag, ConversionClampMethod::PERMISSIVE)) {
            f.get();
        }
        for (uint32_t i = 0; i < 65536; ++i) {
            CHECK(std::bit_cast<uint32_t>(fa[i]) == std::bit_cast<uint32_t>(fb[i]));
        }
    }
}

static void test_avx2_runtime_gate() {
    const auto supported = IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX2);
    alignas(32) float32_t src[8];
    alignas(32) BeanTensor::bfloat16_t dst[8];
    for (auto i = 0; i < 8; ++i) {
        src[i] = static_cast<float32_t>(i) * 0.5f;
    }

    if (supported) {
        std::atomic flag{false};
        for (const auto futures = BeanTensor::Intrinsics::detail::avx2_fp32_to_bf16(src, dst, 8, flag, ConversionClampMethod::PERMISSIVE); auto& f : futures) {
            f.get();
        }
        for (auto i = 0; i < 8; ++i) {
            CHECK(std::bit_cast<uint32_t>(BeanTensor::Intrinsics::detail::unaccel_bf16_to_fp32(dst[i])) == std::bit_cast<uint32_t>(src[i]));
        }
    } else {
        // The Accelerated targets compile these kernels in even when the CPU
        // lacks AVX2 — the runtime gate must refuse to execute them.
        auto threw = false;
        try {
            std::atomic flag{false};
            [[maybe_unused]] auto futures = BeanTensor::Intrinsics::detail::avx2_fp32_to_bf16(src, dst, 8, flag, ConversionClampMethod::PERMISSIVE);
        } catch (const BeanTensor::ErrorHandling::CPUIllegalInstruction&) {
            threw = true;
        }
        CHECK(threw);
    }
}

// Scalar fallbacks and AVX2 kernels must be bit-for-bit identical on every
// input — including NaN payloads, Inf, overflow and subnormals.
static void test_scalar_avx2_parity() {
    namespace bd = BeanTensor::Intrinsics::detail;
    using BeanTensor::bfloat16_t;
    using BeanTensor::float16_t;
    using BeanTensor::float32_t;

    auto specials = std::vector<uint32_t>{
        0x00000000, 0x80000000,             // +-0
        0x00000001, 0x007FFFFF, 0x00800000, // fp32 subnormals / min normal
        0x3F800000, 0xC0200000,             // 1.0, -2.5
        0x477FDFFF, 0x477FE000, 0x477FE001, 0x477FF000, // around FP16_MAX (65504)
        0x33800000, 0x34000000, 0x35555555, 0x387FF000, // fp16 subnormal territory
        0x7F800000, 0xFF800000,             // +-Inf
        0x7F800001, 0x7FC00000, 0x7FFFFFFF, 0xFF800001, // NaNs (low-payload sNaN, qNaN, full payload)
        0x7F7FFFFF, 0xFF7FFFFF,             // +-fp32 max
    };
    auto rng = std::mt19937{std::random_device{}()};
    for (auto i = 0; i < 1 << 18; ++i) {
        specials.push_back(rng()); // random bit patterns incl. NaN/subnormal
    }

    std::atomic flag{false};
    auto src = std::vector<float32_t>(specials.size());
    for (auto i = size_t{0}; i < specials.size(); ++i) {
        src[i] = std::bit_cast<float32_t>(specials[i]);
    }

    // Odd sizes exercise the vector tails and the thread-chunk remainder —
    // every element must be converted exactly once.
    const auto sizes = std::vector<size_t>{1, 7, 37, src.size()};

    if (IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX2)) {
        // fp32 -> bf16: scalar vs AVX2, at several sizes
        for (const auto n : sizes) {
            auto a = std::vector<bfloat16_t>(n);
            auto b = std::vector<bfloat16_t>(n);
            for (auto i = size_t{0}; i < n; ++i) {
                a[i] = bd::unaccel_fp32_to_bf16_untracked(src[i]);
            }
            for (auto& f : bd::avx2_fp32_to_bf16(src.data(), b.data(), n, flag, ConversionClampMethod::PERMISSIVE)) {
                f.get();
            }
            for (auto i = size_t{0}; i < n; ++i) {
                CHECK(bd::bit_cast_memcpy<uint16_t>(a[i]) == bd::bit_cast_memcpy<uint16_t>(b[i]));
            }
        }

        // bf16 -> fp32: exhaustive over all 65536 patterns
        auto all16 = std::vector<bfloat16_t>(65536);
        auto fa = std::vector<float32_t>(65536);
        auto fb = std::vector<float32_t>(65536);
        for (auto i = 0u; i < 65536; ++i) {
            all16[i] = bd::bit_cast_memcpy<bfloat16_t>(static_cast<uint16_t>(i));
        }
        for (auto i = 0u; i < 65536; ++i) {
            fa[i] = bd::unaccel_bf16_to_fp32(all16[i]);
        }
        for (auto& f : bd::avx2_bf16_to_fp32(all16.data(), fb.data(), 65536)) {
            f.get();
        }
        for (auto i = 0u; i < 65536; ++i) {
            CHECK(std::bit_cast<uint32_t>(fa[i]) == std::bit_cast<uint32_t>(fb[i]));
        }
    }

    if (IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX2) &&
        IncludedCPUFeatures().has_feature(CPUFeatureSetType_F16C)) {
        // fp32 -> fp16: scalar vs AVX2/F16C, at several sizes
        for (const auto n : sizes) {
            auto a = std::vector<float16_t>(n);
            auto b = std::vector<float16_t>(n);
            for (auto i = size_t{0}; i < n; ++i) {
                a[i] = bd::unaccel_fp32_to_fp16_untracked(src[i]);
            }
            for (auto& f : bd::avx2_fp32_to_fp16(src.data(), b.data(), n, flag, ConversionClampMethod::PERMISSIVE)) {
                f.get();
            }
            for (auto i = size_t{0}; i < n; ++i) {
                CHECK(bd::bit_cast_memcpy<uint16_t>(a[i]) == bd::bit_cast_memcpy<uint16_t>(b[i]));
            }
        }

        // fp16 -> fp32: exhaustive over all 65536 patterns
        auto all16 = std::vector<float16_t>(65536);
        auto fa = std::vector<float32_t>(65536);
        auto fb = std::vector<float32_t>(65536);
        for (auto i = 0u; i < 65536; ++i) {
            all16[i] = bd::bit_cast_memcpy<float16_t>(static_cast<uint16_t>(i));
        }
        for (auto i = 0u; i < 65536; ++i) {
            fa[i] = bd::unaccel_fp16_to_fp32_untracked(all16[i]);
        }
        for (auto& f : bd::avx2_fp16_to_fp32(all16.data(), fb.data(), 65536, flag, ConversionClampMethod::PERMISSIVE)) {
            f.get();
        }
        for (auto i = 0u; i < 65536; ++i) {
            CHECK(std::bit_cast<uint32_t>(fa[i]) == std::bit_cast<uint32_t>(fb[i]));
        }
    }
}

// Vector fp64/fp32/int kernels must be bit-for-bit identical to the scalar
// standard2standard/scalar_convert paths — including clamp targets, NaN
// payloads, and the flag outcome under every ConversionClampMethod.
// vec/scal take (src, dst, n, flag, method) / (value, flag, method) shapes;
// wrap flagless kernels (int->float) in an adapting lambda at the call site.
template<typename SrcT, typename DstT, typename VecF, typename ScalF>
static void check_vector_parity(const std::vector<SrcT>& src, const VecF& vec, const ScalF& scal) {
    using Bits = std::conditional_t<sizeof(DstT) == 8, uint64_t, uint32_t>;
    // Odd sizes exercise the vector tails and the thread-chunk remainder.
    for (const auto sizes = std::vector<size_t>{1, 5, 37, src.size()}; const auto n : sizes) {
        auto a = std::vector<DstT>(n);
        auto b = std::vector<DstT>(n);
        for (auto i = size_t{0}; i < n; ++i) {
            auto ignored = false;
            a[i] = scal(src[i], ignored, ConversionClampMethod::PERMISSIVE);
        }
        for (std::atomic flag{false}; auto& f : vec(src.data(), b.data(), n, flag, ConversionClampMethod::PERMISSIVE)) {
            f.get();
        }
        for (auto i = size_t{0}; i < n; ++i) {
            CHECK(std::bit_cast<Bits>(a[i]) == std::bit_cast<Bits>(b[i]));
        }
    }
}

template<typename SrcT, typename DstT, typename VecF, typename ScalF>
static void check_flag_parity(const std::vector<SrcT>& src, const VecF& vec, const ScalF& scal) {
    constexpr ConversionClampMethod methods[3] = {
        ConversionClampMethod::STRICT, ConversionClampMethod::ALLOW_INF, ConversionClampMethod::PERMISSIVE
    };
    for (const auto m : methods) {
        auto scalar_flag = false;
        for (const auto& v : src) {
            [[maybe_unused]] auto r = scal(v, scalar_flag, m);
        }
        auto b = std::vector<DstT>(src.size());
        std::atomic flag{false};
        for (auto& f : vec(src.data(), b.data(), src.size(), flag, m)) {
            f.get();
        }
        CHECK(flag.load() == scalar_flag);
    }
}

static void test_extended_conversion_parity() {
    namespace bd = BeanTensor::Intrinsics::detail;
    using BeanTensor::float32_t;
    using BeanTensor::float64_t;

    const auto features = IncludedCPUFeatures();

    // FP32 inputs: specials + boundary values for the int32/uint32 clamps + random bit patterns.
    auto s32 = std::vector<uint32_t>{
        0x00000000, 0x80000000,                         // +-0
        0x00000001, 0x007FFFFF, 0x00800000,             // subnormals / min normal
        0x3F800000, 0xC0200000,                         // 1.0, -2.5
        0x7F800000, 0xFF800000,                         // +-Inf
        0x7F800001, 0x7FC00000, 0x7FFFFFFF, 0xFF800001, // NaNs
        0x4EFFFFFF, 0x4F000000,                         // largest fp32 < 2^31, 2^31 (int32 overflow boundary)
        0xCF000000, 0xCF000001,                         // -2^31 (in range), just below (underflow)
        0x4F7FFFFF, 0x4F800000,                         // largest fp32 < 2^32, 2^32 (uint32 overflow boundary)
        0xBF000000, 0x3F000000,                         // -0.5 (uint32 underflow), 0.5
    };
    auto rng32 = std::mt19937{std::random_device{}()};
    for (auto i = 0; i < 1 << 16; ++i) {
        s32.push_back(rng32());
    }
    auto f32src = std::vector<float32_t>(s32.size());
    for (auto i = size_t{0}; i < s32.size(); ++i) {
        f32src[i] = std::bit_cast<float32_t>(s32[i]);
    }

    // FP64 inputs: specials + FP32_MAX boundary ulps + int64 clamp boundaries + random bit patterns.
    auto s64 = std::vector<uint64_t>{
        0x0000000000000000, 0x8000000000000000,                 // +-0
        0x0000000000000001, 0x000FFFFFFFFFFFFF, 0x0010000000000000, // subnormals / min normal
        0x3FF0000000000000, 0xC002800000000000,                 // 1.0, -2.5
        0x47EFFFFFE0000000, 0x47EFFFFFE0000001, 0x47EFFFFFE0000002, // FP32_MAX as double, +ulp ulps (clamp, flag)
        0xC7EFFFFFE0000000, 0xC7EFFFFFE0000001,                 // -FP32_MAX, -ulp
        0x47F0000000000000, 0xC7F0000000000000,                 // +-2^128 (deep overflow)
        0x7FF0000000000000, 0xFFF0000000000000,                 // +-Inf (clamp to +-FP32_MAX, flag — no passthrough)
        0x7FF0000000000001, 0x7FF8000000000000, 0x7FFFFFFFFFFFFFFF, // NaNs (sNaN, qNaN, full payload)
        0x43E0000000000000, 0x43DFFFFFFFFFFFFF,                 // 2^63 (int64 overflow), largest double < 2^63
        0xC3E0000000000000, 0xC3E0000000000001,                 // -2^63 (in range), just below (underflow)
    };
    auto rng64 = std::mt19937_64{std::random_device{}()};
    for (auto i = 0; i < 1 << 16; ++i) {
        s64.push_back(rng64());
    }
    auto f64src = std::vector<float64_t>(s64.size());
    for (auto i = size_t{0}; i < s64.size(); ++i) {
        f64src[i] = std::bit_cast<float64_t>(s64[i]);
    }

    // INT/UINT inputs: clamp and RNE boundaries (2^24/2^53) + random bit patterns.
    auto i32src = std::vector{0, 1, -1, 16777215, 16777216, 16777217, -16777216, -16777217,
                              std::numeric_limits<int32_t>::max(), std::numeric_limits<int32_t>::min()};
    for (auto i = 0; i < 1 << 16; ++i) {
        i32src.push_back(static_cast<int32_t>(rng32()));
    }
    auto u32src = std::vector<uint32_t>{0, 1, 16777216, 16777217, std::numeric_limits<uint32_t>::max()};
    for (auto i = 0; i < 1 << 16; ++i) {
        u32src.push_back(static_cast<uint32_t>(rng32()));
    }
    auto i64src = std::vector<int64_t>{0, 1, -1, 9007199254740991, 9007199254740992, 9007199254740993,
                                       -9007199254740992, -9007199254740993,
                                       std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::min()};
    for (auto i = 0; i < 1 << 16; ++i) {
        i64src.push_back(std::bit_cast<int64_t>(rng64()));
    }

    if (features.has_feature(CPUFeatureSetType_AVX2)) {
        check_vector_parity<float32_t, float64_t>(f32src, bd::avx2_fp32_to_fp64, bd::standard2standard<float32_t, float64_t>);
        check_flag_parity<float32_t, float64_t>(f32src, bd::avx2_fp32_to_fp64, bd::standard2standard<float32_t, float64_t>);
        check_vector_parity<float64_t, float32_t>(f64src, bd::avx2_fp64_to_fp32, bd::standard2standard<float64_t, float32_t>);
        check_flag_parity<float64_t, float32_t>(f64src, bd::avx2_fp64_to_fp32, bd::standard2standard<float64_t, float32_t>);
        check_vector_parity<int32_t, float32_t>(i32src,
            [](const int32_t* s, float32_t* d, const size_t n, std::atomic<bool>&, ConversionClampMethod) {
                return bd::avx2_int32_to_fp32(s, d, n);
            },
            [](const int32_t& v, bool&, ConversionClampMethod) {
                return static_cast<float32_t>(v);
            });
        check_vector_parity<float32_t, int32_t>(f32src, bd::avx2_fp32_to_int32, bd::scalar_convert<float32_t, int32_t>);
        check_flag_parity<float32_t, int32_t>(f32src, bd::avx2_fp32_to_int32, bd::scalar_convert<float32_t, int32_t>);
    }

    if (probe_avx512_f()) {
        check_vector_parity<float32_t, float64_t>(f32src, bd::avx512_fp32_to_fp64, bd::standard2standard<float32_t, float64_t>);
        check_flag_parity<float32_t, float64_t>(f32src, bd::avx512_fp32_to_fp64, bd::standard2standard<float32_t, float64_t>);
        check_vector_parity<float64_t, float32_t>(f64src, bd::avx512_fp64_to_fp32, bd::standard2standard<float64_t, float32_t>);
        check_flag_parity<float64_t, float32_t>(f64src, bd::avx512_fp64_to_fp32, bd::standard2standard<float64_t, float32_t>);
        check_vector_parity<int32_t, float32_t>(i32src,
            [](const int32_t* s, float32_t* d, const size_t n, std::atomic<bool>&, ConversionClampMethod) {
                return bd::avx512_int32_to_fp32(s, d, n);
            },
            [](const int32_t& v, bool&, ConversionClampMethod) {
                return static_cast<float32_t>(v);
            });
        check_vector_parity<float32_t, int32_t>(f32src, bd::avx512_fp32_to_int32, bd::scalar_convert<float32_t, int32_t>);
        check_flag_parity<float32_t, int32_t>(f32src, bd::avx512_fp32_to_int32, bd::scalar_convert<float32_t, int32_t>);
        check_vector_parity<uint32_t, float32_t>(u32src,
            [](const uint32_t* s, float32_t* d, const size_t n, std::atomic<bool>&, ConversionClampMethod) {
                return bd::avx512_uint32_to_fp32(s, d, n);
            },
            [](const uint32_t& v, bool&, ConversionClampMethod) {
                return static_cast<float32_t>(v);
            });
        check_vector_parity<float32_t, uint32_t>(f32src, bd::avx512_fp32_to_uint32, bd::scalar_convert<float32_t, uint32_t>);
        check_flag_parity<float32_t, uint32_t>(f32src, bd::avx512_fp32_to_uint32, bd::scalar_convert<float32_t, uint32_t>);
    }

    if (probe_avx512_dq()) {
        check_vector_parity<int64_t, float64_t>(i64src,
            [](const int64_t* s, float64_t* d, const size_t n, std::atomic<bool>&, ConversionClampMethod) {
                return bd::avx512_int64_to_fp64(s, d, n);
            },
            [](const int64_t& v, bool&, ConversionClampMethod) {
                return static_cast<float64_t>(v);
            });
        check_vector_parity<float64_t, int64_t>(f64src, bd::avx512_fp64_to_int64, bd::scalar_convert<float64_t, int64_t>);
        check_flag_parity<float64_t, int64_t>(f64src, bd::avx512_fp64_to_int64, bd::scalar_convert<float64_t, int64_t>);
    }
}

static void test_convert_dtype_cpu() {
    // fp32 -> bf16 -> fp32 round trip on exactly-representable values.
    Tensor t({4}, DType::Float32);
    t.set_one();
    t.sync();
    t.convert_dtype(DType::BFloat16);
    CHECK(t.contents_to_string() == "[1, 1, 1, 1]");
    t.convert_dtype(DType::Float32);
    CHECK(t.contents_to_string() == "[1, 1, 1, 1]");

    // Packed: fp32 -> fp16x2 duplicates lanes, back takes the first lane.
    t.convert_dtype(DType::Float16x2);
    CHECK(t.contents_to_string() == "[(1, 1), (1, 1), (1, 1), (1, 1)]");
    t.convert_dtype(DType::Float32);
    CHECK(t.contents_to_string() == "[1, 1, 1, 1]");

    // Int <-> float32.
    Tensor i32({3}, DType::Int32);
    i32.set_one();
    i32.sync();
    i32.convert_dtype(DType::Float32);
    CHECK(i32.contents_to_string() == "[1, 1, 1]");
    i32.convert_dtype(DType::Int32);
    CHECK(i32.contents_to_string() == "[1, 1, 1]");

    // STRICT queues fine; the finite-overflow error (fp64 1e300 -> fp32)
    // surfaces at sync(), matching the GPU path.
    Tensor big({2}, DType::Float64);
    big.random(1e300, 1e300, 1);
    big.sync();
    big.convert_dtype(DType::Float32, ConversionClampMethod::STRICT);
    bool threw = false;
    try {
        big.sync();
    } catch (const std::range_error&) {
        threw = true;
    }
    CHECK(threw);
    // The error is non-sticky: a later sync() does not rethrow.
    big.sync();

    // PERMISSIVE clamps to FLT_MAX instead of throwing.
    Tensor big2({2}, DType::Float64);
    big2.random(1e300, 1e300, 1);
    big2.sync();
    big2.convert_dtype(DType::Float32, ConversionClampMethod::PERMISSIVE);
    CHECK(big2.contents_to_string().find("3.40282e+38") != std::string::npos);

    // A STRICT-flagged tensor destroyed without sync() must not throw or
    // corrupt — the destructor drains pending work quietly.
    {
        Tensor doomed({2}, DType::Float64);
        doomed.random(1e300, 1e300, 1);
        doomed.sync();
        doomed.convert_dtype(DType::Float32, ConversionClampMethod::STRICT);
    }
}

// convert_dtype is asynchronous: conversions of independent tensors queue
// back-to-back and overlap on the thread pool; results land at sync().
static void test_convert_dtype_async_cpu() {
    // Reference: same work with a sync between every step.
    Tensor ref({256, 256}, DType::Float32);
    ref.set_one();
    ref.sync();
    ref.convert_dtype(DType::BFloat16);
    ref.sync();
    ref.convert_dtype(DType::Float32);
    ref.sync();

    // Queued: three tensors, each converted there and back with no
    // intermediate syncs — A B C (pause) instead of A (pause) B (pause) C.
    Tensor a({256, 256}, DType::Float32);
    Tensor b({256, 256}, DType::Float32);
    Tensor c({256, 256}, DType::Float32);
    a.set_one();
    b.set_one();
    c.set_one();
    a.convert_dtype(DType::BFloat16);
    b.convert_dtype(DType::BFloat16);
    c.convert_dtype(DType::BFloat16);
    a.convert_dtype(DType::Float32);
    b.convert_dtype(DType::Float32);
    c.convert_dtype(DType::Float32);

    const std::string expected = ref.contents_to_string();
    CHECK(a.contents_to_string() == expected);
    CHECK(b.contents_to_string() == expected);
    CHECK(c.contents_to_string() == expected);

    // A pending conversion is transparent to every other entry point:
    // set_one syncs first, so it sees (and overwrites) the converted dtype.
    Tensor d({4}, DType::Float32);
    d.set_one();
    d.sync();
    d.convert_dtype(DType::BFloat16);
    d.set_zero(); // syncs, then zeroes the bf16 buffer
    CHECK(d.contents_to_string() == "[0, 0, 0, 0]");
}

static void test_clone() {
    Tensor t({2, 2}, DType::Float32);
    t.set_one();
    t.sync();
    auto hard = t.clone(true);
    auto soft = t.clone(false);
    CHECK(hard.contents_to_string() == "[[1, 1],\n [1, 1]]");
    CHECK(soft.contents_to_string() == "[[1, 1],\n [1, 1]]");

    // A soft clone outlives its parent: the parent's destructor hands the
    // child its own copy of the data (notify_child).
    auto parent = std::make_unique<Tensor>(std::vector<size_t>{2}, DType::Float32);
    parent->set_one();
    parent->sync();
    auto orphan = parent->clone(false);
    parent.reset();
    orphan.sync();
    CHECK(orphan.contents_to_string() == "[1, 1]");

    // Move construction re-points the view graph: the moved-from husk has no
    // children to notify, and killing the moved-to parent still hands the
    // soft clone its own buffer.
    auto mover = std::make_unique<Tensor>(std::vector<size_t>{2}, DType::Float32);
    mover->set_one();
    mover->sync();
    auto mchild = mover->clone(false);
    auto relocated = std::make_unique<Tensor>(std::move(*mover));
    mover.reset(); // moved-from husk: must not notify anyone
    CHECK(mchild.contents_to_string() == "[1, 1]"); // still views relocated's buffer
    relocated.reset(); // the real owner dies -> mchild clones the data
    CHECK(mchild.contents_to_string() == "[1, 1]");

    // Move assignment over a viewed parent hands the child its own copy.
    Tensor ma({2}, DType::Float32);
    ma.set_one();
    ma.sync();
    auto ma_child = ma.clone(false);
    ma = Tensor({3}, DType::Float32);
    ma_child.sync();
    CHECK(ma_child.contents_to_string() == "[1, 1]");

    // A view dying before its parent deregisters itself; the parent's
    // destructor must not touch the dead child.
    auto long_parent = std::make_unique<Tensor>(std::vector<size_t>{2}, DType::Float32);
    long_parent->set_one();
    long_parent->sync();
    {
        auto short_lived = long_parent->clone(false);
        short_lived.sync();
    }
    long_parent.reset();

    // Grandchild chain: gp <- v <- g. gp dies -> v clones and g re-points
    // onto v's fresh buffer; v dies -> g clones in turn.
    auto gp = std::make_unique<Tensor>(std::vector<size_t>{2}, DType::Float32);
    gp->set_one();
    gp->sync();
    auto v = std::make_unique<Tensor>(gp->clone(false));
    auto g = v->clone(false);
    gp.reset();
    CHECK(g.contents_to_string() == "[1, 1]");
    v.reset();
    CHECK(g.contents_to_string() == "[1, 1]");
}

static void test_at_cpu() {
    Tensor t({2, 3}, DType::Float32);
    t.set_zero();
    t.sync();
    t.set<float32_t>({1, 2}, 5.0f);
    CHECK(std::bit_cast<uint32_t>(t.at<float32_t>({0, 0})) == std::bit_cast<uint32_t>(0.0f));
    CHECK(std::bit_cast<uint32_t>(t.at<float32_t>({1, 2})) == std::bit_cast<uint32_t>(5.0f));
    CHECK(t.contents_to_string() == "[[0, 0, 0],\n [0, 0, 5]]");

    Tensor i64({4}, DType::Int64);
    i64.set_zero();
    i64.sync();
    i64.set<int64_t>({3}, -7);
    CHECK(i64.at<int64_t>({3}) == -7);

    auto threw = false;
    try {
        [[maybe_unused]] auto v = t.at<float32_t>({2, 0}); // row out of bounds
    } catch (const BeanTensor::ErrorHandling::IndexOutOfRange&) {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try {
        [[maybe_unused]] auto v = t.at<float32_t>({0}); // wrong rank
    } catch (const BeanTensor::ErrorHandling::IndexOutOfRange&) {
        threw = true;
    }
    CHECK(threw);

    // set() bounds-checks identically.
    threw = false;
    try {
        t.set<float32_t>({0, 3}, 1.0f); // column out of bounds
    } catch (const BeanTensor::ErrorHandling::IndexOutOfRange&) {
        threw = true;
    }
    CHECK(threw);
}

static void test_gpu() {
#if defined(USE_HIP) || defined(USE_CUDA)
    if (!gpu_available()) {
        std::cout << "GPU tests skipped (no device)\n";
        return;
    }
    // Move roundtrip.
    Tensor t({2, 2}, DType::Float32);
    t.set_one();
    t.to(Device::GPU);
    t.to(Device::CPU);
    t.sync();
    CHECK(t.contents_to_string() == "[[1, 1],\n [1, 1]]");

    // GPU-side fills, including packed.
    Tensor g({4}, DType::Float32, Device::GPU);
    g.set_zero();
    g.to(Device::CPU);
    g.sync();
    CHECK(g.contents_to_string() == "[0, 0, 0, 0]");

    g.to(Device::GPU);
    g.set_one();
    g.to(Device::CPU);
    g.sync();
    CHECK(g.contents_to_string() == "[1, 1, 1, 1]");

    Tensor gp({2}, DType::BFloat16x2, Device::GPU);
    gp.set_one();
    gp.to(Device::CPU);
    gp.sync();
    CHECK(gp.contents_to_string() == "[(1, 1), (1, 1)]");

    // set_amount_gpu overloads: int64_t exact past 2^32 (no 32-bit truncation),
    // float64_t carries fractional values to float dtypes.
    Tensor gi({4}, DType::Int64, Device::GPU);
    BeanTensor::Tensors::detail::set_amount_gpu(gi, int64_t{5000000000});
    gi.to(Device::CPU);
    gi.sync();
    CHECK(gi.contents_to_string() == "[5000000000, 5000000000, 5000000000, 5000000000]");

    Tensor gu({2}, DType::UInt64, Device::GPU);
    BeanTensor::Tensors::detail::set_amount_gpu(gu, int64_t{4000000000});
    gu.to(Device::CPU);
    gu.sync();
    CHECK(gu.contents_to_string() == "[4000000000, 4000000000]");

    // The full 64-bit range must be settable: INT64_MIN via the int64_t
    // overload, UINT64_MAX via the uint64_t overload.
    Tensor gil({2}, DType::Int64, Device::GPU);
    BeanTensor::Tensors::detail::set_amount_gpu(gil, std::numeric_limits<int64_t>::min());
    gil.to(Device::CPU);
    gil.sync();
    CHECK(gil.contents_to_string() == "[-9223372036854775808, -9223372036854775808]");

    Tensor gum({2}, DType::UInt64, Device::GPU);
    BeanTensor::Tensors::detail::set_amount_gpu(gum, std::numeric_limits<uint64_t>::max());
    gum.to(Device::CPU);
    gum.sync();
    CHECK(gum.contents_to_string() == "[18446744073709551615, 18446744073709551615]");

    Tensor gf({4}, DType::Float32, Device::GPU);
    BeanTensor::Tensors::detail::set_amount_gpu(gf, float64_t{2.5});
    gf.to(Device::CPU);
    gf.sync();
    CHECK(gf.contents_to_string() == "[2.5, 2.5, 2.5, 2.5]");

    Tensor gd({2}, DType::Float64, Device::GPU);
    BeanTensor::Tensors::detail::set_amount_gpu(gd, float64_t{-0.25});
    gd.to(Device::CPU);
    gd.sync();
    CHECK(gd.contents_to_string() == "[-0.25, -0.25]");

    Tensor gpx({2}, DType::Float16x2, Device::GPU);
    BeanTensor::Tensors::detail::set_amount_gpu(gpx, float64_t{2.5});
    gpx.to(Device::CPU);
    gpx.sync();
    CHECK(gpx.contents_to_string() == "[(2.5, 2.5), (2.5, 2.5)]");

    // Wrong device: set_amount_gpu on a CPU tensor throws, not silently returns.
    Tensor cpu_only({2}, DType::Float32);
    bool fill_threw = false;
    try {
        BeanTensor::Tensors::detail::set_amount_gpu(cpu_only, int64_t{1});
    } catch (const BeanTensor::ErrorHandling::DirectFunctionCallOnWrongDataSource&) {
        fill_threw = true;
    }
    CHECK(fill_threw);

    // A soft-cloned GPU child outlives its parent: notify_child clones the
    // buffer device-to-device so the child stays valid.
    auto gpu_parent = std::make_unique<Tensor>(std::vector<size_t>{2}, DType::Float32, Device::GPU);
    gpu_parent->set_one();
    gpu_parent->sync();
    auto gpu_orphan = gpu_parent->clone(false);
    gpu_parent.reset();
    gpu_orphan.to(Device::CPU);
    gpu_orphan.sync();
    CHECK(gpu_orphan.contents_to_string() == "[1, 1]");

    // at()/set() work on GPU tensors: single-element H2D/D2H copies.
    Tensor ea({2, 2}, DType::Float32, Device::GPU);
    ea.set_zero();
    ea.set<float32_t>({1, 1}, 3.5f);
    CHECK(std::bit_cast<uint32_t>(ea.at<float32_t>({0, 0})) == std::bit_cast<uint32_t>(0.0f));
    CHECK(std::bit_cast<uint32_t>(ea.at<float32_t>({1, 1})) == std::bit_cast<uint32_t>(3.5f));
    ea.to(Device::CPU);
    ea.sync();
    CHECK(ea.contents_to_string() == "[[0, 0],\n [0, 3.5]]");

    // STRICT integer overflow surfaces at sync() on the GPU path too.
    Tensor big_i32({4}, DType::Int32, Device::GPU);
    BeanTensor::Tensors::detail::set_amount_gpu(big_i32, int64_t{1000});
    big_i32.convert_dtype(DType::Int8, ConversionClampMethod::STRICT);
    auto int_threw = false;
    try {
        big_i32.sync();
    } catch (const std::range_error&) {
        int_threw = true;
    }
    CHECK(int_threw);

    // PERMISSIVE integer clamps land on the same values on both devices.
    Tensor clamp_cpu({3}, DType::Int32);
    clamp_cpu.set<int32_t>({0}, 1000);
    clamp_cpu.set<int32_t>({1}, -1000);
    clamp_cpu.set<int32_t>({2}, 42);
    auto clamp_gpu = clamp_cpu.clone(true);
    clamp_cpu.convert_dtype(DType::Int8);
    clamp_gpu.to(Device::GPU);
    clamp_gpu.convert_dtype(DType::Int8);
    clamp_gpu.to(Device::CPU);
    clamp_cpu.sync();
    clamp_gpu.sync();
    // Int8 prints as chars; compare values via at(), and devices via string.
    CHECK(clamp_cpu.at<int8_t>({0}) == 127);
    CHECK(clamp_cpu.at<int8_t>({1}) == -128);
    CHECK(clamp_cpu.at<int8_t>({2}) == 42);
    CHECK(clamp_gpu.contents_to_string() == clamp_cpu.contents_to_string());
    CHECK(clamp_gpu.at<int8_t>({0}) == 127);
    CHECK(clamp_gpu.at<int8_t>({1}) == -128);
    CHECK(clamp_gpu.at<int8_t>({2}) == 42);

    // GPU conversion roundtrip (fp32 -> bf16 -> fp32).
    Tensor c({4}, DType::Float32, Device::GPU);
    c.set_one();
    c.convert_dtype(DType::BFloat16);
    c.convert_dtype(DType::Float32);
    c.to(Device::CPU);
    c.sync();
    CHECK(c.contents_to_string() == "[1, 1, 1, 1]");

    // GPU STRICT failure surfaces at sync().
    Tensor huge({4}, DType::Float32);
    huge.random(1e38, 1e38, 1);
    huge.sync();
    huge.to(Device::GPU);
    huge.convert_dtype(DType::Float16, ConversionClampMethod::STRICT);
    bool threw = false;
    try {
        huge.sync();
    } catch (const std::range_error&) {
        threw = true;
    }
    CHECK(threw);

    // GPU PERMISSIVE conversions must be bit-identical to the CPU path
    // (compared as strings after moving back; equal bits print equal strings).
    const auto parity = [](const DType src_dtype, const DType dst_dtype, const float64_t lo, const float64_t hi) {
        Tensor cpu_ref({64}, src_dtype);
        cpu_ref.random(lo, hi, 7);
        cpu_ref.sync();
        Tensor gpu_ver = cpu_ref.clone(true);
        cpu_ref.convert_dtype(dst_dtype, ConversionClampMethod::PERMISSIVE);
        gpu_ver.to(Device::GPU);
        gpu_ver.convert_dtype(dst_dtype, ConversionClampMethod::PERMISSIVE);
        gpu_ver.to(Device::CPU);
        cpu_ref.sync();
        gpu_ver.sync();
        CHECK(cpu_ref.contents_to_string() == gpu_ver.contents_to_string());
    };

    parity(DType::Float32, DType::Float16, -1e38, 1e38);    // overflow clamps to +-65504 on both
    parity(DType::Float32, DType::Float16, -10.0, 10.0);
    parity(DType::Float32, DType::BFloat16, -1e38, 1e38);   // finite overflow rounds to bf16 inf on both
    parity(DType::Float32, DType::BFloat16, -1e-38, 1e-38); // fp32 subnormals flush to signed zero on both
    parity(DType::Float32, DType::Float16x2, -1e38, 1e38);  // packed lanes duplicate
    parity(DType::Float32, DType::Float64, -1e30, 1e30);    // exact widening
    parity(DType::Float64, DType::Float32, -1e300, 1e300);  // clamps to +-FLT_MAX on both
    parity(DType::Float64, DType::Float16, -1e38, 1e38);    // fp64 -> fp32 -> fp16 double clamp
    parity(DType::Float64, DType::BFloat16, -1e300, 1e300);
    parity(DType::Float16, DType::Float32, -100.0, 100.0);  // exact widening
    parity(DType::BFloat16, DType::Float32, -1e38, 1e38);
    parity(DType::BFloat16, DType::Float16, -1e38, 1e38);   // clamps through the fp32 lane
    parity(DType::Float16, DType::BFloat16, -100.0, 100.0);
    parity(DType::Float16x2, DType::Float32, -100.0, 100.0); // packed source takes the first lane
    parity(DType::BFloat16x2, DType::BFloat16, -1e38, 1e38);
    parity(DType::Int32, DType::Int8, -1000, 1000);         // signed narrow clamps
    parity(DType::Int32, DType::UInt8, -100, 1000);         // mixed-sign clamps
    parity(DType::Int64, DType::Int32, -5e9, 5e9);          // wide -> narrow
    parity(DType::UInt64, DType::Int16, 0, 100000);         // unsigned -> signed narrow
    parity(DType::UInt8, DType::Int64, 0, 200);             // exact widening
    parity(DType::Int64, DType::Float32, -1e9, 1e9);        // int -> float
    parity(DType::UInt64, DType::Float64, 0, 1e9);          // int -> float widening
    parity(DType::Float32, DType::Int32, -1e4, 1e4);        // float -> int truncation
    parity(DType::Float64, DType::UInt8, -10.0, 1000.0);    // float -> unsigned clamps
    parity(DType::Int32, DType::Float16, -1000, 1000);      // int -> packed-lane narrowing
    parity(DType::Float16, DType::Int16, -100.0, 100.0);    // packed-lane source -> int

    // ALLOW_INF lets a source Inf through on both devices (STRICT would throw).
    // 3.4e38 is past the bf16 RNE overflow boundary, so every element becomes Inf.
    Tensor inf_src({64}, DType::Float32);
    inf_src.random(3.4e38, 3.4e38, 11);
    inf_src.sync();
    inf_src.convert_dtype(DType::BFloat16); // PERMISSIVE: yields bf16 Inf, unflagged
    inf_src.sync();
    CHECK(inf_src.contents_to_string().find("inf") != std::string::npos);
    Tensor inf_gpu = inf_src.clone(true);
    inf_src.convert_dtype(DType::Float16, ConversionClampMethod::ALLOW_INF);
    inf_gpu.to(Device::GPU);
    inf_gpu.convert_dtype(DType::Float16, ConversionClampMethod::ALLOW_INF);
    inf_gpu.to(Device::CPU);
    bool inf_threw = false;
    try {
        inf_src.sync();
        inf_gpu.sync();
    } catch (const std::range_error&) {
        inf_threw = true;
    }
    CHECK(!inf_threw);
    CHECK(inf_src.contents_to_string() == inf_gpu.contents_to_string());
#else
    std::cout << "GPU tests skipped (CPU-only build)\n";
#endif
}

// Int4x2: packed signed 4-bit lanes (lo = bits[3:0], hi = bits[7:4]).
// Unpacked -> packed duplicates lanes; packed -> unpacked takes the lo lane;
// out-of-range values clamp to [-8, 7] under PERMISSIVE and flag otherwise.
static void test_convert_dtype_int4x2() {
    Tensor t({4}, DType::Int8);
    t.set<int8_t>({0}, 1);
    t.set<int8_t>({1}, -1);
    t.set<int8_t>({2}, 7);
    t.set<int8_t>({3}, -8);
    t.sync();
    t.convert_dtype(DType::Int4x2);
    t.sync();
    CHECK(t.contents_to_string() == "[(1, 1), (-1, -1), (7, 7), (-8, -8)]");
    t.convert_dtype(DType::Int32); // Int8 prints as chars; Int32 prints numerically.
    CHECK(t.contents_to_string() == "[1, -1, 7, -8]");

    // dtype metadata.
    CHECK(BeanTensor::DataTypes::dtype_bytes(DType::Int4x2) == 1);
    CHECK(BeanTensor::DataTypes::dtype_to_string(DType::Int4x2) == "Int4x2");

    // Out-of-range: PERMISSIVE clamps to 7 silently.
    Tensor over({1}, DType::Int8);
    over.set<int8_t>({0}, 100);
    over.sync();
    over.convert_dtype(DType::Int4x2, ConversionClampMethod::PERMISSIVE);
    over.sync();
    CHECK(over.contents_to_string() == "[(7, 7)]");

    // STRICT flags the clamp as a range error at sync().
    Tensor strict({1}, DType::Int8);
    strict.set<int8_t>({0}, 100);
    strict.sync();
    strict.convert_dtype(DType::Int4x2, ConversionClampMethod::STRICT);
    auto threw = false;
    try {
        strict.sync();
    } catch (const std::range_error&) {
        threw = true;
    }
    CHECK(threw);

    // random() covers Int4x2, and out-of-range bounds reject.
    Tensor r({64}, DType::Int4x2);
    r.random(-8.0, 7.0, 5);
    r.sync();
    threw = false;
    try {
        Tensor bad({1}, DType::Int4x2);
        bad.random(-100.0, 100.0, 5);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

static void test_move_defers_pending_error() {
    // A failed pending task drained by a move must rethrow at the
    // destination's next sync() — never silently discarded (noexcept moves).
    Tensor t({4}, DType::Float32);
    std::promise<void> p;
    t.give_future(p.get_future().share());
    p.set_exception(std::make_exception_ptr(BeanTensor::ErrorHandling::UnReachableCode()));
    Tensor moved(std::move(t)); // must not throw
    auto threw = false;
    try {
        moved.sync();
    } catch (const BeanTensor::ErrorHandling::UnReachableCode&) {
        threw = true;
    }
    CHECK(threw);

    // Thrown-once: the poison was consumed by that sync().
    moved.sync();

    // Chained moves carry the deferred error along.
    Tensor t2({4}, DType::Float32);
    std::promise<void> p2;
    t2.give_future(p2.get_future().share());
    p2.set_exception(std::make_exception_ptr(BeanTensor::ErrorHandling::UnReachableCode()));
    Tensor m2(std::move(t2));
    Tensor m3(std::move(m2));
    threw = false;
    try {
        m3.sync();
    } catch (const BeanTensor::ErrorHandling::UnReachableCode&) {
        threw = true;
    }
    CHECK(threw);

    // Move assignment defers the source's error the same way.
    Tensor src({4}, DType::Float32);
    std::promise<void> p3;
    src.give_future(p3.get_future().share());
    p3.set_exception(std::make_exception_ptr(BeanTensor::ErrorHandling::UnReachableCode()));
    Tensor dst({4}, DType::Float32);
    dst = std::move(src);
    threw = false;
    try {
        dst.sync();
    } catch (const BeanTensor::ErrorHandling::UnReachableCode&) {
        threw = true;
    }
    CHECK(threw);

    // Move assignment also preserves the target's OWN pending error.
    Tensor dst2({4}, DType::Float32);
    std::promise<void> p4;
    dst2.give_future(p4.get_future().share());
    p4.set_exception(std::make_exception_ptr(BeanTensor::ErrorHandling::UnReachableCode()));
    Tensor src2({4}, DType::Float32);
    dst2 = std::move(src2);
    threw = false;
    try {
        dst2.sync();
    } catch (const BeanTensor::ErrorHandling::UnReachableCode&) {
        threw = true;
    }
    CHECK(threw);
}

static_assert(alignof(BeanTensor::Tensors::Node) == 8, "Node must stay 8-aligned, or be reviewed.");
static_assert(alignof(Tensor) == 8, "Tensor must stay 8-aligned, or be reviewed.");
// BT_TENSOR_MAX_DIMS 3: 216 + std::vector view_parents (24 bytes). Re-verify with offsetof on change.
static_assert(sizeof(Tensor) == 240, "Tensor size changed; re-verify layout and update the project docs.");
static_assert(std::is_nothrow_move_constructible_v<Tensor>, "Tensor move ctor must stay noexcept, or be reviewed.");
static_assert(std::is_nothrow_move_assignable_v<Tensor>, "Tensor move assignment must stay noexcept, or be reviewed.");

// slice/split/index_select/gather/cat on CPU: view semantics, byte-movement,
// and the documented error paths. Values are exact ints so at() compares bit-exact.
static void test_indexing_cpu() {
    // Values are exact small ints; compare via bit_cast to avoid -Werror=float-equal.
    auto eq = [](const float32_t a, const float32_t b) {
        return std::bit_cast<uint32_t>(a) == std::bit_cast<uint32_t>(b);
    };
    // Two-input cat lists are built with moves: Tensor copies are deleted.
    auto make_pair = [](Tensor x, Tensor y) {
        std::vector<Tensor> v;
        v.reserve(2);
        v.emplace_back(std::move(x));
        v.emplace_back(std::move(y));
        return v;
    };
    // ---- slice: leading-dim slice is contiguous; inner slice is not ----
    Tensor t({2, 3}, DType::Float32);
    for (auto r = size_t{0}; r < 2; ++r) {
        for (auto c = size_t{0}; c < 3; ++c) {
            t.set<float32_t>({r, c}, static_cast<float32_t>(r * 3 + c));
        }
    }
    t.sync();

    // Leading-dim slice -> contiguous view of shape {1,3}.
    auto lead = t.slice({{0, 1}, {0, 3}});
    CHECK(lead.is_same_shape(Tensor({1, 3})));
    CHECK(eq(lead.at<float32_t>({0, 0}), 0.0f));
    CHECK(eq(lead.at<float32_t>({0, 2}), 2.0f));

    // Inner-dim slice -> non-contiguous view, but reads are offset-correct.
    auto inner = t.slice({{0, 2}, {1, 3}});
    CHECK(inner.is_same_shape(Tensor({2, 2})));
    CHECK(eq(inner.at<float32_t>({0, 0}), 1.0f)); // maps to t(0,1)
    CHECK(eq(inner.at<float32_t>({1, 1}), 5.0f)); // maps to t(1,2)

    // slice is a live view: a write through the parent propagates into it.
    t.set<float32_t>({0, 2}, 123.0f);
    CHECK(eq(inner.at<float32_t>({0, 1}), 123.0f));

    // A full-range slice sees the parent's writes too.
    auto whole = t.slice({{0, 2}, {0, 3}});
    CHECK(whole.is_same_shape(Tensor({2, 3})));
    CHECK(eq(whole.at<float32_t>({0, 2}), 123.0f));

    // Degenerate slice: an empty range yields a zero-extent dimension.
    auto empty = t.slice({{0, 2}, {0, 0}});
    CHECK(empty.is_same_shape(Tensor({2, 0})));

    // Error paths: rank mismatch, end past the extent, start past end.
    auto threw = false;
    try { [[maybe_unused]] auto bad = t.slice({{0, 2}}); } catch (const BeanTensor::ErrorHandling::MismatchedShapes&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { [[maybe_unused]] auto bad = t.slice({{0, 3}, {0, 3}}); } catch (const BeanTensor::ErrorHandling::IndexOutOfRange&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { [[maybe_unused]] auto bad = t.slice({{2, 1}, {0, 3}}); } catch (const BeanTensor::ErrorHandling::IndexOutOfRange&) { threw = true; }
    CHECK(threw);

    // ---- split: equal chunks along an axis are slice views that round-trip with cat ----
    Tensor grid({2, 3}, DType::Float32);
    for (auto r = size_t{0}; r < 2; ++r) {
        for (auto c = size_t{0}; c < 3; ++c) {
            grid.set<float32_t>({r, c}, static_cast<float32_t>(r * 3 + c));
        }
    }
    grid.sync();

    auto cols = grid.split(1, 3); // three {2,1} column views
    CHECK(cols.size() == 3);
    for (auto i = size_t{0}; i < 3; ++i) {
        CHECK(cols[i].is_same_shape(Tensor({2, 1})));
        CHECK(eq(cols[i].at<float32_t>({0, 0}), static_cast<float32_t>(i)));
        CHECK(eq(cols[i].at<float32_t>({1, 0}), static_cast<float32_t>(3 + i)));
    }

    // Inner-axis chunks are non-contiguous views: readable as slices, but
    // rejected by cat (which requires contiguous operands, per Tensor.h).
    threw = false;
    try { [[maybe_unused]] auto bad = Tensor::cat(cols, 1); } catch (const BeanTensor::ErrorHandling::NotImplemented&) { threw = true; }
    CHECK(threw);

    // cat(split(axis, n), axis) reproduces the original when the chunks are
    // contiguous — true for a leading-axis split.
    auto rows = grid.split(0, 2); // two {1,3} row views
    CHECK(rows.size() == 2);
    CHECK(rows[0].is_same_shape(Tensor({1, 3})));
    CHECK(rows[1].is_same_shape(Tensor({1, 3})));
    auto rebuilt = Tensor::cat(rows, 0);
    CHECK(rebuilt.is_same_shape(Tensor({2, 3})));
    for (auto r = size_t{0}; r < 2; ++r) {
        for (auto c = size_t{0}; c < 3; ++c) {
            CHECK(eq(rebuilt.at<float32_t>({r, c}), static_cast<float32_t>(r * 3 + c)));
        }
    }

    // split error paths: axis out of bounds, zero sections, non-dividing sections.
    threw = false;
    try { [[maybe_unused]] auto bad = grid.split(2, 1); } catch (const BeanTensor::ErrorHandling::IndexOutOfRange&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { [[maybe_unused]] auto bad = grid.split(1, 0); } catch (const BeanTensor::ErrorHandling::MismatchedShapes&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { [[maybe_unused]] auto bad = grid.split(1, 2); } catch (const BeanTensor::ErrorHandling::MismatchedShapes&) { threw = true; } // 3 % 2 != 0
    CHECK(threw);

    // ---- index_select: pick columns by a rank-1 integer index tensor ----
    Tensor src({2, 3}, DType::Float32);
    for (auto r = size_t{0}; r < 2; ++r) {
        for (auto c = size_t{0}; c < 3; ++c) {
            src.set<float32_t>({r, c}, static_cast<float32_t>(r * 3 + c));
        }
    }
    src.sync();

    Tensor idx({3}, DType::Int64);
    idx.set<int64_t>({0}, 2);
    idx.set<int64_t>({1}, 0);
    idx.set<int64_t>({2}, 2);
    idx.sync();
    auto sel = src.index_select(1, idx); // {2,3} axis-1 -> {2,3}; cols [2,0,2]
    CHECK(sel.is_same_shape(Tensor({2, 3})));
    CHECK(eq(sel.at<float32_t>({0, 0}), 2.0f));
    CHECK(eq(sel.at<float32_t>({0, 1}), 0.0f));
    CHECK(eq(sel.at<float32_t>({0, 2}), 2.0f));
    CHECK(eq(sel.at<float32_t>({1, 0}), 5.0f));
    CHECK(eq(sel.at<float32_t>({1, 2}), 5.0f));

    // Selecting along axis 0 picks whole rows.
    Tensor ridx({1}, DType::Int64);
    ridx.set<int64_t>({0}, 1);
    ridx.sync();
    auto rowsel = src.index_select(0, ridx); // {1,3} = row 1
    CHECK(rowsel.is_same_shape(Tensor({1, 3})));
    CHECK(eq(rowsel.at<float32_t>({0, 0}), 3.0f));
    CHECK(eq(rowsel.at<float32_t>({0, 2}), 5.0f));

    // Duplicate + out-of-order indices are allowed and preserved in order.
    Tensor dup({2}, DType::Int64);
    dup.set<int64_t>({0}, 1);
    dup.set<int64_t>({1}, 1);
    dup.sync();
    auto duped = src.index_select(1, dup); // {2,2}: both columns are col 1
    CHECK(duped.is_same_shape(Tensor({2, 2})));
    CHECK(eq(duped.at<float32_t>({0, 0}), 1.0f));
    CHECK(eq(duped.at<float32_t>({0, 1}), 1.0f));
    CHECK(eq(duped.at<float32_t>({1, 0}), 4.0f));
    CHECK(eq(duped.at<float32_t>({1, 1}), 4.0f));

    // Any plain integer dtype is accepted as the index tensor.
    Tensor idx32({2}, DType::Int32);
    idx32.set<int32_t>({0}, 1);
    idx32.set<int32_t>({1}, 0);
    idx32.sync();
    auto sel32 = src.index_select(1, idx32); // {2,2} cols [1,0]
    CHECK(sel32.is_same_shape(Tensor({2, 2})));
    CHECK(eq(sel32.at<float32_t>({0, 0}), 1.0f));
    CHECK(eq(sel32.at<float32_t>({0, 1}), 0.0f));

    // An empty index tensor yields a zero-extent output along the axis.
    Tensor empty_idx({0}, DType::Int64);
    auto empty_sel = src.index_select(1, empty_idx);
    CHECK(empty_sel.is_same_shape(Tensor({2, 0})));

    // Non-contiguous input is rejected.
    auto nc_view = src.slice({{0, 2}, {1, 3}});
    threw = false;
    try { [[maybe_unused]] auto bad = nc_view.index_select(1, idx); } catch (const BeanTensor::ErrorHandling::NotImplemented&) { threw = true; }
    CHECK(threw);

    // index_select error paths: axis, index rank, index dtype, out-of-bounds, negative.
    threw = false;
    try { [[maybe_unused]] auto bad = src.index_select(5, idx); } catch (const BeanTensor::ErrorHandling::IndexOutOfRange&) { threw = true; }
    CHECK(threw);
    threw = false;
    Tensor notrank1({2, 2}, DType::Int64);
    try { [[maybe_unused]] auto bad = src.index_select(1, notrank1); } catch (const BeanTensor::ErrorHandling::MismatchedShapes&) { threw = true; }
    CHECK(threw);
    threw = false;
    Tensor floatidx({2}, DType::Float32);
    try { [[maybe_unused]] auto bad = src.index_select(1, floatidx); } catch (const BeanTensor::ErrorHandling::MismatchedDTypes&) { threw = true; }
    CHECK(threw);
    threw = false;
    Tensor oobidx({1}, DType::Int64);
    oobidx.set<int64_t>({0}, 3);
    oobidx.sync();
    try { [[maybe_unused]] auto bad = src.index_select(1, oobidx); } catch (const BeanTensor::ErrorHandling::IndexOutOfRange&) { threw = true; }
    CHECK(threw);
    threw = false;
    Tensor negidx({1}, DType::Int64);
    negidx.set<int64_t>({0}, -1);
    negidx.sync();
    try { [[maybe_unused]] auto bad = src.index_select(1, negidx); } catch (const BeanTensor::ErrorHandling::IndexOutOfRange&) { threw = true; }
    CHECK(threw);

    // ---- gather: same-rank integer index tensor ----
    Tensor gidx({2, 2}, DType::Int64);
    gidx.set<int64_t>({0, 0}, 0);
    gidx.set<int64_t>({0, 1}, 2);
    gidx.set<int64_t>({1, 0}, 1);
    gidx.set<int64_t>({1, 1}, 0);
    gidx.sync();
    auto gat = src.gather(1, gidx); // {2,2}
    CHECK(gat.is_same_shape(Tensor({2, 2})));
    CHECK(eq(gat.at<float32_t>({0, 0}), 0.0f));
    CHECK(eq(gat.at<float32_t>({0, 1}), 2.0f));
    CHECK(eq(gat.at<float32_t>({1, 0}), 4.0f));
    CHECK(eq(gat.at<float32_t>({1, 1}), 3.0f));

    // Smaller off-axis extents are allowed (PyTorch rule).
    Tensor narrow({2, 1}, DType::Int64);
    narrow.set<int64_t>({0, 0}, 0);
    narrow.set<int64_t>({1, 0}, 2);
    narrow.sync();
    auto ngat = src.gather(1, narrow); // {2,1}
    CHECK(ngat.is_same_shape(Tensor({2, 1})));
    CHECK(eq(ngat.at<float32_t>({0, 0}), 0.0f));
    CHECK(eq(ngat.at<float32_t>({1, 0}), 5.0f));

    // gather error paths: axis, rank, off-axis extent, index dtype, bounds.
    threw = false;
    try { [[maybe_unused]] auto bad = src.gather(5, gidx); } catch (const BeanTensor::ErrorHandling::IndexOutOfRange&) { threw = true; }
    CHECK(threw);
    threw = false;
    Tensor drank({2}, DType::Int64);
    try { [[maybe_unused]] auto bad = src.gather(1, drank); } catch (const BeanTensor::ErrorHandling::MismatchedShapes&) { threw = true; }
    CHECK(threw);
    threw = false;
    Tensor dwide({3, 3}, DType::Int64);
    try { [[maybe_unused]] auto bad = src.gather(1, dwide); } catch (const BeanTensor::ErrorHandling::MismatchedShapes&) { threw = true; } // off-axis 3 > 2
    CHECK(threw);
    threw = false;
    Tensor gfloat({2, 2}, DType::Float32);
    try { [[maybe_unused]] auto bad = src.gather(1, gfloat); } catch (const BeanTensor::ErrorHandling::MismatchedDTypes&) { threw = true; }
    CHECK(threw);
    threw = false;
    Tensor goob({2, 2}, DType::Int64);
    goob.set<int64_t>({0, 0}, 3);
    goob.sync();
    try { [[maybe_unused]] auto bad = src.gather(1, goob); } catch (const BeanTensor::ErrorHandling::IndexOutOfRange&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { [[maybe_unused]] auto bad = nc_view.gather(1, gidx); } catch (const BeanTensor::ErrorHandling::NotImplemented&) { threw = true; }
    CHECK(threw);

    // ---- cat along axis 0 and axis 1 ----
    Tensor a({2, 2}, DType::Float32);
    Tensor b({2, 2}, DType::Float32);
    for (auto i = size_t{0}; i < 4; ++i) {
        a.set<float32_t>({i / 2, i % 2}, static_cast<float32_t>(i));
        b.set<float32_t>({i / 2, i % 2}, static_cast<float32_t>(10 + i));
    }
    a.sync();
    b.sync();
    auto pair = make_pair(std::move(a), std::move(b));
    auto c0 = Tensor::cat(pair, 0); // {4,2}
    CHECK(c0.is_same_shape(Tensor({4, 2})));
    CHECK(eq(c0.at<float32_t>({2, 0}), 10.0f)); // first element of b
    CHECK(eq(c0.at<float32_t>({3, 1}), 13.0f));
    auto c1 = Tensor::cat(pair, 1); // {2,4}: rows are a's row then b's row
    CHECK(c1.is_same_shape(Tensor({2, 4})));
    CHECK(eq(c1.at<float32_t>({0, 2}), 10.0f));
    CHECK(eq(c1.at<float32_t>({1, 3}), 13.0f));

    // cat error paths: empty list, rank, dtype, off-axis shape, axis, contiguity.
    threw = false;
    try { [[maybe_unused]] auto bad = Tensor::cat(std::vector<Tensor>{}, size_t{0}); } catch (const BeanTensor::ErrorHandling::MismatchedShapes&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { [[maybe_unused]] auto bad = Tensor::cat(make_pair(Tensor({2, 2}, DType::Float32), Tensor({2, 2, 1}, DType::Float32)), 0); } catch (const BeanTensor::ErrorHandling::MismatchedShapes&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { [[maybe_unused]] auto bad = Tensor::cat(make_pair(Tensor({2, 2}, DType::Float32), Tensor({2, 2}, DType::Int32)), 0); } catch (const BeanTensor::ErrorHandling::MismatchedDTypes&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { [[maybe_unused]] auto bad = Tensor::cat(make_pair(Tensor({2, 2}, DType::Float32), Tensor({3, 2}, DType::Float32)), 1); } catch (const BeanTensor::ErrorHandling::MismatchedShapes&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { [[maybe_unused]] auto bad = Tensor::cat(make_pair(Tensor({2, 2}, DType::Float32), Tensor({2, 2}, DType::Float32)), 5); } catch (const BeanTensor::ErrorHandling::IndexOutOfRange&) { threw = true; }
    CHECK(threw);
    threw = false;
    try {
        auto nca = grid.slice({{0, 2}, {1, 3}});
        auto ncb = grid.slice({{0, 2}, {1, 3}});
        [[maybe_unused]] auto bad = Tensor::cat(make_pair(std::move(nca), std::move(ncb)), 0);
    } catch (const BeanTensor::ErrorHandling::NotImplemented&) { threw = true; }
    CHECK(threw);
}

// index_select/gather/cat on GPU must match the CPU results bit-for-bit.
static void test_indexing_gpu() {
#if defined(USE_HIP) || defined(USE_CUDA)
    if (!gpu_available()) {
        std::cout << "indexing GPU tests skipped (no device)\n";
        return;
    }
    // Reference CPU tensors.
    Tensor cpu_src({2, 3}, DType::Float32);
    for (auto r = size_t{0}; r < 2; ++r) {
        for (auto c = size_t{0}; c < 3; ++c) {
            cpu_src.set<float32_t>({r, c}, static_cast<float32_t>(r * 3 + c));
        }
    }
    cpu_src.sync();

    auto gpu_src = cpu_src.clone(true);
    gpu_src.to(Device::GPU);

    // index_select parity.
    Tensor cpu_idx({3}, DType::Int64);
    cpu_idx.set<int64_t>({0}, 2);
    cpu_idx.set<int64_t>({1}, 0);
    cpu_idx.set<int64_t>({2}, 1);
    cpu_idx.sync();
    auto cpu_sel = cpu_src.index_select(1, cpu_idx);
    Tensor gpu_idx = cpu_idx.clone(true);
    gpu_idx.to(Device::GPU);
    auto gpu_sel = gpu_src.index_select(1, gpu_idx);
    gpu_sel.to(Device::CPU);
    gpu_sel.sync();
    cpu_sel.sync();
    CHECK(cpu_sel.contents_to_string() == gpu_sel.contents_to_string());

    // gather parity.
    Tensor cpu_g({2, 2}, DType::Int64);
    cpu_g.set<int64_t>({0, 0}, 0);
    cpu_g.set<int64_t>({0, 1}, 2);
    cpu_g.set<int64_t>({1, 0}, 1);
    cpu_g.set<int64_t>({1, 1}, 0);
    cpu_g.sync();
    auto cpu_gat = cpu_src.gather(1, cpu_g);
    Tensor gpu_g = cpu_g.clone(true);
    gpu_g.to(Device::GPU);
    auto gpu_gat = gpu_src.gather(1, gpu_g);
    gpu_gat.to(Device::CPU);
    gpu_gat.sync();
    CHECK(cpu_gat.contents_to_string() == gpu_gat.contents_to_string());

    // cat parity along axis 0 and axis 1.
    Tensor cpu_a({2, 2}, DType::Float32);
    Tensor cpu_b({2, 2}, DType::Float32);
    for (auto i = size_t{0}; i < 4; ++i) {
        cpu_a.set<float32_t>({i / 2, i % 2}, static_cast<float32_t>(i));
        cpu_b.set<float32_t>({i / 2, i % 2}, static_cast<float32_t>(10 + i));
    }
    cpu_a.sync();
    cpu_b.sync();
    std::vector<Tensor> cpu_pair;
    cpu_pair.emplace_back(std::move(cpu_a));
    cpu_pair.emplace_back(std::move(cpu_b));
    auto cpu_c0 = Tensor::cat(cpu_pair, 0);
    auto cpu_c1 = Tensor::cat(cpu_pair, 1);
    Tensor gpu_a = cpu_pair[0].clone(true);
    gpu_a.to(Device::GPU);
    Tensor gpu_b = cpu_pair[1].clone(true);
    gpu_b.to(Device::GPU);
    std::vector<Tensor> gpu_pair;
    gpu_pair.emplace_back(std::move(gpu_a));
    gpu_pair.emplace_back(std::move(gpu_b));
    auto gpu_c0 = Tensor::cat(gpu_pair, 0);
    auto gpu_c1 = Tensor::cat(gpu_pair, 1);
    gpu_c0.to(Device::CPU);
    gpu_c1.to(Device::CPU);
    gpu_c0.sync();
    gpu_c1.sync();
    cpu_c0.sync();
    cpu_c1.sync();
    CHECK(cpu_c0.contents_to_string() == gpu_c0.contents_to_string());
    CHECK(cpu_c1.contents_to_string() == gpu_c1.contents_to_string());

    // Device mismatch is rejected eagerly (before any backend copy).
    auto gpu_one = cpu_pair[0].clone(true);
    gpu_one.to(Device::GPU);
    std::vector<Tensor> mixed;
    mixed.emplace_back(cpu_pair[0].clone(true));
    mixed.emplace_back(std::move(gpu_one));
    auto threw = false;
    try { [[maybe_unused]] auto bad = Tensor::cat(mixed, 0); } catch (const BeanTensor::ErrorHandling::MismatchedDevices&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { [[maybe_unused]] auto bad = cpu_pair[0].index_select(1, gpu_idx); } catch (const BeanTensor::ErrorHandling::MismatchedDevices&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { [[maybe_unused]] auto bad = cpu_src.gather(1, gpu_g); } catch (const BeanTensor::ErrorHandling::MismatchedDevices&) { threw = true; }
    CHECK(threw);
#else
    std::cout << "indexing GPU tests skipped (CPU-only build)\n";
#endif
}

// GPU random() stages a host fill inside a pool worker. Above threads*1024
// elements that fill is chunked across the pool; the worker must help-wait on
// the chunks, or it parks on one stuck in its own deque and the pool spins
// forever (hung on Linear construction in main.cpp). The staged values must
// also match a CPU fill with the same seed bit for bit.
static void test_random_gpu_chunked() {
#if defined(USE_HIP) || defined(USE_CUDA)
    if (!gpu_available()) {
        std::cout << "GPU chunked random test skipped (no device)\n";
        return;
    }
    constexpr size_t kNumel = size_t{1} << 20;
    constexpr int64_t kSeed = 1234;
    auto host = Tensor({kNumel}, DType::Float32);
    host.random(-1.0, 1.0, kSeed);
    host.sync();
    auto device = Tensor({kNumel}, DType::Float32, Device::GPU);
    device.random(-1.0, 1.0, kSeed);
    device.to(Device::CPU);
    device.sync();
    auto mismatches = size_t{0};
    for (auto i = size_t{0}; i < kNumel; ++i) {
        if (
            const auto same = std::bit_cast<uint32_t>(host.at<float32_t>({i})) == std::bit_cast<uint32_t>(device.at<float32_t>({i}));
            !same
        ) {
            ++mismatches;
        }
    }
    CHECK(mismatches == 0);
#else
    std::cout << "GPU chunked random test skipped (CPU-only build)\n";
#endif
}

int32_t main() {
    std::cout << "TensorTests" << std::endl;
    test_fills_cpu();
    test_random_cpu();
    test_scalar_intrinsics();
    test_standard2standard();
    test_nan_inf_flags();
    test_avx512_runtime_gate();
    test_scalar_avx512_parity();
    test_avx2_runtime_gate();
    test_scalar_avx2_parity();
    test_extended_conversion_parity();
    test_convert_dtype_cpu();
    test_convert_dtype_int4x2();
    test_convert_dtype_async_cpu();
    test_clone();
    test_at_cpu();
    test_move_defers_pending_error();
    test_indexing_cpu();
    test_indexing_gpu();
    test_gpu();
    test_random_gpu_chunked();
    if (failures == 0) {
        std::cout << "ALL TESTS PASSED" << std::endl;
        return 0;
    }
    std::cout << failures << " TEST(S) FAILED" << std::endl;
    return 1;
}
