/**
 * @file kernel_tune_tests.cpp
 * @author GreenBean
 * @brief KernelTune API tests: default size-class boundaries, override/cache/default
 * precedence, cache file round-trip, mismatched-device fallback, and the CPU-only
 * autotune() NotImplemented guarantee.
 * @version 0.4
 * @date 2026-10-26
 */
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "Helpers/Errors.h"
#include "Helpers/KernelTune.h"
#include "Tensor/DTypes.h"

using BeanTensor::KernelTune::GpuFamily;
using BeanTensor::KernelTune::SizeClass;
using BeanTensor::KernelTune::autotune;
using BeanTensor::KernelTune::block_size_for;
using BeanTensor::KernelTune::cache_path;
using BeanTensor::KernelTune::clear_override;
using BeanTensor::KernelTune::set_override;
using BeanTensor::float32_t;
namespace ktd = BeanTensor::KernelTune::detail;

static int32_t failures = 0;
#define CHECK(...) do { \
    if (!(__VA_ARGS__)) { ++failures; std::cerr << "FAIL: " << #__VA_ARGS__ << " @ line " << __LINE__ << "\n"; } \
} while (0)

static void test_defaults() {
    ktd::reset_cache();
    clear_override();
    // Tiny: <= 65536, default 128.
    CHECK(block_size_for(GpuFamily::Reduction, 1) == 128);
    CHECK(block_size_for(GpuFamily::Reduction, 65536) == 128);
    // Small: 65537 .. 2097152, default 128.
    CHECK(block_size_for(GpuFamily::Reduction, 65537) == 128);
    CHECK(block_size_for(GpuFamily::Reduction, 2097152) == 128);
    // Medium: 2097153 .. 8388608, default 256.
    CHECK(block_size_for(GpuFamily::Reduction, 2097153) == 256);
    CHECK(block_size_for(GpuFamily::Reduction, 8388608) == 256);
    // Large: > 8388608, default 512.
    CHECK(block_size_for(GpuFamily::Reduction, 8388609) == 512);
}

static void test_override() {
    ktd::reset_cache();
    clear_override();
    set_override(GpuFamily::Reduction, SizeClass::Tiny, 512);
    CHECK(block_size_for(GpuFamily::Reduction, 1) == 512);
    set_override(GpuFamily::Reduction, SizeClass::Large, 64);
    CHECK(block_size_for(GpuFamily::Reduction, 100000000) == 64);
    clear_override();
    CHECK(block_size_for(GpuFamily::Reduction, 1) == 128);
}

static void test_override_validation() {
    ktd::reset_cache();
    clear_override();

    auto threw = false;
    try {
        set_override(GpuFamily::Reduction, SizeClass::Tiny, 999);
    }
    catch (const BeanTensor::ErrorHandling::InvalidKernelBlockSize&) {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try {
        set_override(GpuFamily::Reduction, SizeClass::Tiny, 0);
    }
    catch (const BeanTensor::ErrorHandling::InvalidKernelBlockSize&) {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try {
        set_override(GpuFamily::Reduction, SizeClass::Tiny, 1025);
    }
    catch (const BeanTensor::ErrorHandling::InvalidKernelBlockSize&) {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try {
        set_override(GpuFamily::Reduction, SizeClass::Tiny, 384);
    }
    catch (const BeanTensor::ErrorHandling::InvalidKernelBlockSize&) {
        threw = true;
    }
    CHECK(threw);
}

static void test_cache_precedence() {
    ktd::reset_cache();
    clear_override();
    const auto tmp = std::filesystem::temp_directory_path() / "bt_kernel_tune_precedence.conf";
    const auto tmp_str = tmp.string();
    setenv("BEANTENSOR_KERNEL_TUNE", tmp_str.c_str(), 1);
    const std::array<size_t, 4> values = {64, 128, 256, 1024};
    ktd::write_cache_file(cache_path(), "testdev", values);
    ktd::load_cache_for_device("testdev");
    CHECK(block_size_for(GpuFamily::Reduction, 1) == 64);
    CHECK(block_size_for(GpuFamily::Reduction, 65537) == 128);
    CHECK(block_size_for(GpuFamily::Reduction, 2097153) == 256);
    CHECK(block_size_for(GpuFamily::Reduction, 8388609) == 1024);
    // Override beats cache.
    set_override(GpuFamily::Reduction, SizeClass::Small, 256);
    CHECK(block_size_for(GpuFamily::Reduction, 65537) == 256);
    std::error_code ec;
    std::filesystem::remove(tmp, ec);
    unsetenv("BEANTENSOR_KERNEL_TUNE");
}

static void test_mismatched_device() {
    ktd::reset_cache();
    clear_override();
    const auto tmp = std::filesystem::temp_directory_path() / "bt_kernel_tune_mismatch.conf";
    const auto tmp_str = tmp.string();
    setenv("BEANTENSOR_KERNEL_TUNE", tmp_str.c_str(), 1);
    const std::array<size_t, 4> values = {64, 128, 256, 1024};
    ktd::write_cache_file(cache_path(), "wrongdev", values);
    ktd::load_cache_for_device("rightdev");
    CHECK(block_size_for(GpuFamily::Reduction, 1) == 128);
    CHECK(block_size_for(GpuFamily::Reduction, 8388609) == 512);
    std::error_code ec;
    std::filesystem::remove(tmp, ec);
    unsetenv("BEANTENSOR_KERNEL_TUNE");
}

static void test_corrupt_cache_values() {
    ktd::reset_cache();
    clear_override();
    const auto tmp = std::filesystem::temp_directory_path() / "bt_kernel_tune_corrupt.conf";
    const auto tmp_str = tmp.string();
    setenv("BEANTENSOR_KERNEL_TUNE", tmp_str.c_str(), 1);
    // Write a file with out-of-range and non-numeric values: they must be ignored.
    std::ofstream out(cache_path());
    out << "device=corruptdev\n";
    out << "ReductionTiny=99999\n";     // above 1024 cap
    out << "ReductionSmall=abc\n";      // non-numeric
    out << "ReductionMedium=0\n";       // below 1
    out << "ReductionLarge=2048\n";     // above 1024 cap
    out.close();
    ktd::load_cache_for_device("corruptdev");
    CHECK(block_size_for(GpuFamily::Reduction, 1) == 128);       // Tiny default
    CHECK(block_size_for(GpuFamily::Reduction, 65537) == 128);   // Small default
    CHECK(block_size_for(GpuFamily::Reduction, 2097153) == 256); // Medium default
    CHECK(block_size_for(GpuFamily::Reduction, 8388609) == 512); // Large default
    std::error_code ec;
    std::filesystem::remove(tmp, ec);
    unsetenv("BEANTENSOR_KERNEL_TUNE");
}

static void test_non_pow2_cache_values() {
    ktd::reset_cache();
    clear_override();
    const auto tmp = std::filesystem::temp_directory_path() / "bt_kernel_tune_nonpow2.conf";
    const auto tmp_str = tmp.string();
    setenv("BEANTENSOR_KERNEL_TUNE", tmp_str.c_str(), 1);
    std::ofstream out(cache_path());
    out << "device=nonpow2dev\n";
    out << "ReductionTiny=384\n";    // non-pow2, ignored
    out << "ReductionSmall=768\n";   // non-pow2, ignored
    out << "ReductionMedium=512\n";  // pow2, accepted
    out << "ReductionLarge=1024\n";  // pow2, accepted
    out.close();
    ktd::load_cache_for_device("nonpow2dev");
    CHECK(block_size_for(GpuFamily::Reduction, 1) == 128);         // Tiny default
    CHECK(block_size_for(GpuFamily::Reduction, 65537) == 128);     // Small default
    CHECK(block_size_for(GpuFamily::Reduction, 2097153) == 512);   // Medium cached
    CHECK(block_size_for(GpuFamily::Reduction, 8388609) == 1024);  // Large cached
    std::error_code ec;
    std::filesystem::remove(tmp, ec);
    unsetenv("BEANTENSOR_KERNEL_TUNE");
}

static void test_autotune_representatives() {
    constexpr size_t kSlack = 1024 * 1024;
    constexpr size_t kFullBudget = 67108864 * sizeof(float32_t) + kSlack;
    const auto full = ktd::autotune_representatives(kFullBudget);
    CHECK(full[0] == 32768);
    CHECK(full[1] == 1048576);
    CHECK(full[2] == 8388608);
    CHECK(full[3] == 67108864);

    constexpr size_t k40MB = 40 * 1024 * 1024;
    const auto small = ktd::autotune_representatives(k40MB);
    CHECK(small[0] >= 1 && small[0] <= 65536);
    CHECK(small[1] >= 65537 && small[1] <= 2097152);
    CHECK(small[2] >= 2097153 && small[2] <= 8388608);
    CHECK(small[3] >= 8388609 && small[3] <= (k40MB - kSlack) / sizeof(float32_t));
    CHECK(small[3] * sizeof(float32_t) + kSlack <= k40MB);

    auto threw = false;
    try {
        static_cast<void>(ktd::autotune_representatives(1024 * 1024));
    }
    catch (const BeanTensor::ErrorHandling::NotImplemented&) {
        threw = true;
    }
    CHECK(threw);
}

static void test_cache_round_trip() {
    ktd::reset_cache();
    clear_override();

    const auto tmp = std::filesystem::temp_directory_path() / "bt_kernel_tune_test.conf";
    const auto tmp_str = tmp.string();
    setenv("BEANTENSOR_KERNEL_TUNE", tmp_str.c_str(), 1);

    const std::array<size_t, 4> values = {64, 256, 512, 1024};
    ktd::write_cache_file(cache_path(), "rounddev", values);
    ktd::load_cache_for_device("rounddev");
    CHECK(block_size_for(GpuFamily::Reduction, 1) == 64);
    CHECK(block_size_for(GpuFamily::Reduction, 65537) == 256);
    CHECK(block_size_for(GpuFamily::Reduction, 2097153) == 512);
    CHECK(block_size_for(GpuFamily::Reduction, 8388609) == 1024);

    // Reset to defaults and reload from the same file via cache_path().
    ktd::reset_cache();
    CHECK(block_size_for(GpuFamily::Reduction, 1) == 128);
    ktd::load_cache_for_device("rounddev");
    CHECK(block_size_for(GpuFamily::Reduction, 1) == 64);

    std::error_code ec;
    std::filesystem::remove(tmp, ec);
    unsetenv("BEANTENSOR_KERNEL_TUNE");
}

static void test_cpu_autotune_throws() {
#if !defined(USE_HIP) && !defined(USE_CUDA)
    auto threw = false;
    try {
        autotune();
    }
    catch (const BeanTensor::ErrorHandling::NotImplemented&) {
        threw = true;
    }
    CHECK(threw);
#else
    std::cerr << "SKIP: CPU autotune-throw test on GPU build\n";
#endif
}

int32_t main() {
    // Force the KernelTune cache path to a non-existent file so the
    // defaults/boundary assertions below exercise the compiled-default
    // fallback path. A machine that has run autotune() has a real
    // per-device cache at ~/.cache/beantensor/kernel_tune.conf that would
    // legitimately override the compiled defaults; those values are not
    // the defaults this suite verifies.
    setenv("BEANTENSOR_KERNEL_TUNE", "/nonexistent/beantensor_test_no_cache.conf", 1);

    test_defaults();
    test_override();
    test_override_validation();
    test_cache_precedence();
    test_mismatched_device();
    test_corrupt_cache_values();
    test_non_pow2_cache_values();
    test_cache_round_trip();
    test_autotune_representatives();
    test_cpu_autotune_throws();
    if (failures == 0) {
        std::cout << "ALL KERNEL TUNE TESTS PASSED\n";
        return 0;
    }
    std::cerr << failures << " FAILURE(S)\n";
    return 1;
}
