// BeanTensorV5 random seed tests — framework-free assertion suite.
// Covers: Tensor::random with explicit and omitted (Generator-drawn) seeds,
// seed determinism, negative seeds, and Generator stream identity/reseed.
#include <algorithm>
#include <barrier>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <thread>
#include <vector>

#include "Module/Linear.h"
#include "Random/Generator.h"
#include "Tensor/Tensor.h"

using BeanTensor::Modules::Linear;

using BeanTensor::Random::Generator;
using BeanTensor::Tensors::Device;
using BeanTensor::Tensors::Tensor;
using BeanTensor::float32_t;
using BeanTensor::float64_t;
using DType = BeanTensor::DataTypes::DType;

static int32_t failures = 0;
#define CHECK(...) do { \
    if (!(__VA_ARGS__)) { ++failures; std::cerr << "FAIL: " << #__VA_ARGS__ << " @ line " << __LINE__ << "\n"; } \
} while (0)

static bool near64(const float64_t a, const float64_t b, const float64_t eps) {
    return std::fabs(a - b) <= eps;
}

static std::vector<float32_t> read_flat_float32(Tensor& t) {
    t.sync();
    auto values = std::vector<float32_t>(t.element_count());
    for (auto i = size_t{0}; i < t.element_count(); ++i) {
        values[i] = t.at<float32_t>({i});
    }
    return values;
}

static bool values_equal(const std::vector<float32_t>& a, const std::vector<float32_t>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (auto i = size_t{0}; i < a.size(); ++i) {
        if (!near64(static_cast<float64_t>(a[i]), static_cast<float64_t>(b[i]), 1e-5)) {
            return false;
        }
    }
    return true;
}

static bool values_differ(const std::vector<float32_t>& a, const std::vector<float32_t>& b) {
    if (a.size() != b.size()) {
        return true;
    }
    for (auto i = size_t{0}; i < a.size(); ++i) {
        if (!near64(static_cast<float64_t>(a[i]), static_cast<float64_t>(b[i]), 1e-5)) {
            return true;
        }
    }
    return false;
}

static bool bit_exact_equal(const std::vector<float32_t>& a, const std::vector<float32_t>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(float32_t)) == 0;
}

static bool all_in_range(const std::vector<float32_t>& v, const float64_t min, const float64_t max) {
    for (const auto& x : v) {
        const auto xf = static_cast<float64_t>(x);
        if (xf < min || xf > max) {
            return false;
        }
    }
    return true;
}

// =====================================================================
// Tensor::random optional/explicit seed behavior
// =====================================================================

static void test_same_generator_seed_identical() {
    // Omitted seed draws from the process-wide generator. Reseeding to the
    // same value before each draw should yield identical tensors.
    auto a = Tensor({64 * 64}, DType::Float32);
    auto b = Tensor({64 * 64}, DType::Float32);
    BeanTensor::Random::set_seed(42);
    a.random(0.0, 1.0, std::nullopt);
    BeanTensor::Random::set_seed(42);
    b.random(0.0, 1.0, std::nullopt);
    CHECK(values_equal(read_flat_float32(a), read_flat_float32(b)));
}

static void test_different_generator_seed_different() {
    auto a = Tensor({64 * 64}, DType::Float32);
    auto b = Tensor({64 * 64}, DType::Float32);
    BeanTensor::Random::set_seed(1);
    a.random(0.0, 1.0, std::nullopt);
    BeanTensor::Random::set_seed(2);
    b.random(0.0, 1.0, std::nullopt);
    CHECK(values_differ(read_flat_float32(a), read_flat_float32(b)));
}

static void test_explicit_seed_deterministic() {
    // Explicit int64_t seed call sites remain bit-exact for single-chunk fills.
    auto a = Tensor({64 * 64}, DType::Float32);
    auto b = Tensor({64 * 64}, DType::Float32);
    a.random(0.0, 1.0, 123);
    b.random(0.0, 1.0, 123);
    CHECK(values_equal(read_flat_float32(a), read_flat_float32(b)));
}

static void test_negative_explicit_seed() {
    // Negative seeds are valid int64_t seed values.
    auto t = Tensor({64 * 64}, DType::Float32);
    t.random(0.0, 1.0, -1);
    const auto v = read_flat_float32(t);
    CHECK(all_in_range(v, 0.0, 1.0));
}

// =====================================================================
// Generator stream identity and reseed
// =====================================================================

static void test_generator_same_seed_identical_draws() {
    auto gen_a = Generator(123);
    auto gen_b = Generator(123);
    for (auto i = size_t{0}; i < 16; ++i) {
        CHECK(gen_a.draw_seed() == gen_b.draw_seed());
    }
}

static void test_generator_reseed_resets_stream() {
    auto gen = Generator(123);
    auto first = std::vector<uint64_t>(8);
    for (auto i = size_t{0}; i < first.size(); ++i) {
        first[i] = gen.draw_seed();
    }
    gen.seed(123);
    for (auto i = size_t{0}; i < first.size(); ++i) {
        CHECK(gen.draw_seed() == first[i]);
    }
}

// =====================================================================
// Linear optional seed behavior
// =====================================================================

static void test_linear_omitted_seed_deterministic() {
    // Omitted seed routes through the process-wide generator. Reseeding to
    // the same value before each Linear construction should yield identical
    // weight and bias parameters.
    const auto in_features = size_t{3};
    const auto out_features = size_t{5};
    BeanTensor::Random::set_seed(20261028);
    auto a = Linear(in_features, out_features);
    BeanTensor::Random::set_seed(20261028);
    auto b = Linear(in_features, out_features);
    CHECK(a.parameter("weight").contents_to_string() == b.parameter("weight").contents_to_string());
    CHECK(a.parameter("bias").contents_to_string() == b.parameter("bias").contents_to_string());
}

// =====================================================================
// Concurrency and chunk-grid invariants
// =====================================================================

static void test_concurrent_draws() {
    // N=8 barrier-started threads share one Generator. No thread should lose
    // or duplicate a draw index; completion order is irrelevant.
    constexpr auto n_threads = size_t{8};
    const auto seed = uint64_t{12345};
    auto gen = Generator(seed);

    struct ThreadResult {
        uint64_t drawn_seed = 0;
        Tensor tensor = Tensor({1024}, DType::Float32);
    };
    auto results = std::vector<ThreadResult>(n_threads);
    auto start = std::barrier(n_threads);
    auto threads = std::vector<std::thread>();
    threads.reserve(n_threads);

    for (auto t = size_t{0}; t < n_threads; ++t) {
        threads.emplace_back([&, t]() {
            start.arrive_and_wait();
            const auto drawn = gen.draw_seed();
            results[t].drawn_seed = drawn;
            results[t].tensor.random(0.0, 1.0, static_cast<int64_t>(drawn));
        });
    }
    for (auto& th : threads) {
        th.join();
    }

    auto expected_gen = Generator(seed);
    auto expected_seeds = std::vector<uint64_t>(n_threads);
    for (auto i = size_t{0}; i < n_threads; ++i) {
        expected_seeds[i] = expected_gen.draw_seed();
    }

    auto recorded_seeds = std::vector<uint64_t>(n_threads);
    for (auto i = size_t{0}; i < n_threads; ++i) {
        recorded_seeds[i] = results[i].drawn_seed;
    }
    std::sort(recorded_seeds.begin(), recorded_seeds.end());
    std::sort(expected_seeds.begin(), expected_seeds.end());
    CHECK(recorded_seeds == expected_seeds);

    for (auto i = size_t{0}; i < n_threads; ++i) {
        auto ref = Tensor({1024}, DType::Float32);
        ref.random(0.0, 1.0, static_cast<int64_t>(results[i].drawn_seed));
        CHECK(bit_exact_equal(read_flat_float32(results[i].tensor), read_flat_float32(ref)));
    }
}

static void test_chunk_grid_property() {
    // Pinned to the chunk size used in Tensor.cpp's random fill.
    constexpr auto kRandomChunkElems = size_t{1} << 20;
    const auto seed = uint64_t{54321};

    BeanTensor::Random::set_seed(seed);
    auto a = Tensor({kRandomChunkElems}, DType::Float32);
    a.random(0.0, 1.0, std::nullopt);

    BeanTensor::Random::set_seed(seed);
    auto b = Tensor({4 * kRandomChunkElems}, DType::Float32);
    b.random(0.0, 1.0, std::nullopt);

    const auto a_values = read_flat_float32(a);
    const auto b_values = read_flat_float32(b);
    CHECK(a_values.size() == kRandomChunkElems);
    CHECK(b_values.size() == 4 * kRandomChunkElems);
    CHECK(std::memcmp(a_values.data(), b_values.data(), kRandomChunkElems * sizeof(float32_t)) == 0);
}

// =====================================================================
// GPU parity (no CPU↔GPU equality asserted — backend-internal parity only)
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

static void test_gpu_parity() {
    if (!gpu_available()) {
        std::cout << "SKIP: no GPU device available\n";
        return;
    }

    // Same-seed omitted-seed random() on GPU tensors is identical after CPU move.
    const auto seed = uint64_t{77777};
    BeanTensor::Random::set_seed(seed);
    auto a = Tensor({64 * 64}, DType::Float32, Device::GPU);
    a.random(0.0, 1.0, std::nullopt);
    a.to(Device::CPU);

    BeanTensor::Random::set_seed(seed);
    auto b = Tensor({64 * 64}, DType::Float32, Device::GPU);
    b.random(0.0, 1.0, std::nullopt);
    b.to(Device::CPU);

    CHECK(bit_exact_equal(read_flat_float32(a), read_flat_float32(b)));

    // Two GPU Linears constructed after the same set_seed are identical.
    const auto in_features = size_t{3};
    const auto out_features = size_t{5};
    BeanTensor::Random::set_seed(seed);
    auto lin_a = Linear(in_features, out_features, DType::Float32, Device::GPU);
    BeanTensor::Random::set_seed(seed);
    auto lin_b = Linear(in_features, out_features, DType::Float32, Device::GPU);

    CHECK(lin_a.parameter("weight").contents_to_string() == lin_b.parameter("weight").contents_to_string());
    CHECK(lin_a.parameter("bias").contents_to_string() == lin_b.parameter("bias").contents_to_string());
}
#endif

int main() {
    test_same_generator_seed_identical();
    test_different_generator_seed_different();
    test_explicit_seed_deterministic();
    test_negative_explicit_seed();
    test_generator_same_seed_identical_draws();
    test_generator_reseed_resets_stream();
    test_linear_omitted_seed_deterministic();
    test_concurrent_draws();
    test_chunk_grid_property();
#if defined(USE_CUDA) || defined(USE_HIP)
    test_gpu_parity();
#endif

    if (failures == 0) {
        std::cout << "ALL RANDOM TESTS PASSED\n";
        return 0;
    }
    std::cerr << failures << " FAILURE(S)\n";
    return 1;
}
