// BeanTensorV5 tests — framework-free assertion suite.
// Covers: CPU fills (incl. packed dtypes), scalar + runtime-gated AVX512
// conversions, standard2standard clamp semantics, NaN/Inf clamp-flag
// consistency, printing, clone, and (when a backend + device exist) the GPU
// paths incl. GPU<->CPU conversion parity.
#include <atomic>
#include <bit>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "Helpers/Hardware.h"
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
using DType = BeanTensor::DataTypes::DType;

static int failures = 0;
#define CHECK(...) do { \
    if (!(__VA_ARGS__)) { ++failures; std::cerr << "FAIL: " << #__VA_ARGS__ << " @ line " << __LINE__ << "\n"; } \
} while (0)

static bool gpu_available() {
#if defined(USE_HIP)
    int count = 0;
    return hipGetDeviceCount(&count) == hipSuccess && count > 0;
#elif defined(USE_CUDA)
    int count = 0;
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
    CHECK(det::unaccel_bf16_to_fp32(one_bf) == 1.0f);
    CHECK(det::unaccel_fp16_to_fp32_untracked(det::unaccel_fp32_to_fp16_untracked(2.5f)) == 2.5f);
}

static void test_standard2standard() {
    namespace det = BeanTensor::Intrinsics::detail;
    bool flag = false;

    CHECK(det::standard2standard<float32_t, float32_t>(2.5f, flag) == 2.5f);
    CHECK(!flag);
    CHECK(det::standard2standard<float32_t, float64_t>(1.5f, flag) == 1.5);
    CHECK(det::standard2standard<int32_t, int64_t>(-7, flag) == -7);
    CHECK(det::standard2standard<float32_t, float32_t>(0.0f, flag) == 0.0f);

    // Overflow clamps; PERMISSIVE never flags.
    CHECK(det::standard2standard<float64_t, float32_t>(1e300, flag, ConversionClampMethod::PERMISSIVE) == std::numeric_limits<float32_t>::max());
    CHECK(!flag);
    // STRICT flags finite overflow.
    CHECK(det::standard2standard<float64_t, float32_t>(1e300, flag, ConversionClampMethod::STRICT) == std::numeric_limits<float32_t>::max());
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

    const float32_t nan32 = std::numeric_limits<float32_t>::quiet_NaN();
    const float32_t inf32 = std::numeric_limits<float32_t>::infinity();
    const float16_t nan16 = det::bit_cast_memcpy<float16_t>(static_cast<uint16_t>(0x7E00));
    const float16_t inf16 = det::bit_cast_memcpy<float16_t>(static_cast<uint16_t>(0x7C00));

    const ConversionClampMethod methods[3] = {
        ConversionClampMethod::STRICT, ConversionClampMethod::ALLOW_INF, ConversionClampMethod::PERMISSIVE};
    const bool want_nan[3] = {true, true, false};
    const bool want_inf[3] = {true, false, false};

    for (int m = 0; m < 3; ++m) {
        bool flag = false;

        flag = false; det::unaccel_fp32_to_fp16_tracked(nan32, flag, methods[m]);
        CHECK(flag == want_nan[m]);
        flag = false; det::unaccel_fp32_to_fp16_tracked(inf32, flag, methods[m]);
        CHECK(flag == want_inf[m]);

        flag = false; det::unaccel_fp32_to_bf16_tracked(nan32, flag, methods[m]);
        CHECK(flag == want_nan[m]);
        flag = false; det::unaccel_fp32_to_bf16_tracked(inf32, flag, methods[m]);
        CHECK(flag == want_inf[m]);

        flag = false; det::unaccel_fp16_to_fp32_tracked(nan16, flag, methods[m]);
        CHECK(flag == want_nan[m]);
        flag = false; det::unaccel_fp16_to_fp32_tracked(inf16, flag, methods[m]);
        CHECK(flag == want_inf[m]);

        flag = false; det::scalar_convert<float32_t, int32_t>(nan32, flag, methods[m]);
        CHECK(flag == want_nan[m]);

        flag = false; det::standard2standard<float32_t, float64_t>(nan32, flag, methods[m]);
        CHECK(flag == want_nan[m]);
    }

    bool flag = false;
    const float16_t n16 = det::unaccel_fp32_to_fp16_tracked(nan32, flag, ConversionClampMethod::PERMISSIVE);
    const uint16_t nb = det::bit_cast_memcpy<uint16_t>(n16);
    CHECK((nb & 0x7C00) == 0x7C00 && (nb & 0x03FF) != 0); // NaN stays NaN (quiet)
    const float16_t i16 = det::unaccel_fp32_to_fp16_tracked(inf32, flag, ConversionClampMethod::PERMISSIVE);
    CHECK(det::bit_cast_memcpy<uint16_t>(i16) == 0x7C00); // Inf stays Inf
}

static void test_avx512_runtime_gate() {
    const bool supported = IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512BF16);
    alignas(64) float32_t src[16];
    alignas(64) BeanTensor::bfloat16_t dst[16];
    for (int i = 0; i < 16; ++i) src[i] = static_cast<float32_t>(i) * 0.5f;

    if (supported) {
        std::atomic<bool> flag{false};
        auto futures = BeanTensor::Intrinsics::detail::avx512_fp32_to_bf16(src, dst, 16, flag, ConversionClampMethod::PERMISSIVE);
        for (auto& f : futures) f.get();
        for (int i = 0; i < 16; ++i) {
            CHECK(BeanTensor::Intrinsics::detail::unaccel_bf16_to_fp32(dst[i]) == src[i]);
        }
    } else {
        // The Accelerated targets compile these kernels in even when the CPU
        // lacks AVX512BF16 — the runtime gate must refuse to execute them.
        bool threw = false;
        try {
            std::atomic<bool> flag{false};
            [[maybe_unused]] auto futures = BeanTensor::Intrinsics::detail::avx512_fp32_to_bf16(src, dst, 16, flag, ConversionClampMethod::PERMISSIVE);
        } catch (const BeanTensor::ErrorHandling::CPUIllegalInstruction&) {
            threw = true;
        }
        CHECK(threw);
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
    std::mt19937 rng(1234);
    for (int i = 0; i < (1 << 18); ++i) specials.push_back(rng()); // random bit patterns incl. NaN/subnormal

    std::atomic<bool> flag{false};
    std::vector<float32_t> src(specials.size());
    for (size_t i = 0; i < specials.size(); ++i) src[i] = std::bit_cast<float32_t>(specials[i]);

    // Odd sizes exercise the vector tails and the thread-chunk remainder —
    // every element must be converted exactly once.
    const std::vector<size_t> sizes = {1, 7, 37, src.size()};

    if (IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512BF16)) {
        // fp32 -> bf16: scalar vs AVX512, at several sizes
        for (const size_t n : sizes) {
            std::vector<bfloat16_t> a(n), b(n);
            for (size_t i = 0; i < n; ++i) a[i] = bd::unaccel_fp32_to_bf16_untracked(src[i]);
            for (auto& f : bd::avx512_fp32_to_bf16(src.data(), b.data(), n, flag, ConversionClampMethod::PERMISSIVE)) f.get();
            for (size_t i = 0; i < n; ++i)
                CHECK(bd::bit_cast_memcpy<uint16_t>(a[i]) == bd::bit_cast_memcpy<uint16_t>(b[i]));
        }

        // bf16 -> fp32: exhaustive over all 65536 patterns
        std::vector<bfloat16_t> all16(65536);
        std::vector<float32_t> fa(65536), fb(65536);
        for (uint32_t i = 0; i < 65536; ++i) all16[i] = bd::bit_cast_memcpy<bfloat16_t>(static_cast<uint16_t>(i));
        for (uint32_t i = 0; i < 65536; ++i) fa[i] = bd::unaccel_bf16_to_fp32(all16[i]);
        for (auto& f : bd::avx512_bf16_to_fp32(all16.data(), fb.data(), 65536)) f.get();
        for (uint32_t i = 0; i < 65536; ++i)
            CHECK(std::bit_cast<uint32_t>(fa[i]) == std::bit_cast<uint32_t>(fb[i]));
    }

    if (IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512F)) {
        // fp32 -> fp16: scalar vs AVX512, at several sizes
        for (const size_t n : sizes) {
            std::vector<float16_t> a(n), b(n);
            for (size_t i = 0; i < n; ++i) a[i] = bd::unaccel_fp32_to_fp16_untracked(src[i]);
            for (auto& f : bd::avx512_fp32_to_fp16(src.data(), b.data(), n, flag, ConversionClampMethod::PERMISSIVE)) f.get();
            for (size_t i = 0; i < n; ++i)
                CHECK(bd::bit_cast_memcpy<uint16_t>(a[i]) == bd::bit_cast_memcpy<uint16_t>(b[i]));
        }

        // fp16 -> fp32: exhaustive over all 65536 patterns
        std::vector<float16_t> all16(65536);
        std::vector<float32_t> fa(65536), fb(65536);
        for (uint32_t i = 0; i < 65536; ++i) all16[i] = bd::bit_cast_memcpy<float16_t>(static_cast<uint16_t>(i));
        for (uint32_t i = 0; i < 65536; ++i) fa[i] = bd::unaccel_fp16_to_fp32_untracked(all16[i]);
        for (auto& f : bd::avx512_fp16_to_fp32(all16.data(), fb.data(), 65536, flag, ConversionClampMethod::PERMISSIVE)) f.get();
        for (uint32_t i = 0; i < 65536; ++i)
            CHECK(std::bit_cast<uint32_t>(fa[i]) == std::bit_cast<uint32_t>(fb[i]));
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

int main() {
    std::cout << "TensorTests" << std::endl;
    test_fills_cpu();
    test_random_cpu();
    test_scalar_intrinsics();
    test_standard2standard();
    test_nan_inf_flags();
    test_avx512_runtime_gate();
    test_scalar_avx512_parity();
    test_convert_dtype_cpu();
    test_convert_dtype_async_cpu();
    test_clone();
    test_gpu();
    if (failures == 0) {
        std::cout << "ALL TESTS PASSED" << std::endl;
        return 0;
    }
    std::cout << failures << " TEST(S) FAILED" << std::endl;
    return 1;
}
