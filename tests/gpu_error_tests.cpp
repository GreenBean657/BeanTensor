/**
 * @file gpu_error_tests.cpp
 * @author GreenBean
 * @brief GPU error taxonomy tests: verify throw_gpu_error maps HIP/CUDA codes
 * to the typed exception family, that the family derives from GPUError, and
 * that messages include api/file:line/backend-string context.
 * @version 0.2
 * @date 2026-10-26
 */
#include <cstdint>
#include <iostream>
#include <string>

#include "Helpers/Errors.h"

namespace eh = BeanTensor::ErrorHandling;

static int32_t failures = 0;
#define CHECK(...) do { \
    if (!(__VA_ARGS__)) { ++failures; std::cerr << "FAIL: " << #__VA_ARGS__ << " @ line " << __LINE__ << "\n"; } \
} while (0)

[[nodiscard]] static bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

template<typename Expected>
static void expect_typed_throw(
    const int32_t code,
    const char* api,
    const std::string& backend_string
) {
    auto threw_expected = false;
    auto threw_gpu_error = false;
    try {
        eh::detail::throw_gpu_error(code, api, __FILE__, __LINE__);
    }
    catch (const Expected& e) {
        threw_expected = true;
        const auto what = std::string{e.what()};
        CHECK(contains(what, api));
        CHECK(contains(what, __FILE__));
        CHECK(contains(what, ":"));
        CHECK(contains(what, backend_string));
    }
    catch (const eh::GPUError& e) {
        threw_gpu_error = true;
        std::cerr << "FAIL: code " << code << " from " << api
                  << " threw GPUError instead of " << typeid(Expected).name()
                  << ": " << e.what() << " @ line " << __LINE__ << "\n";
    }
    catch (...) {
        std::cerr << "FAIL: code " << code << " from " << api
                  << " threw unexpected type @ line " << __LINE__ << "\n";
    }
    CHECK(threw_expected);
    CHECK(!threw_gpu_error);
}

static void test_gpu_error_mapping() {
#if defined(USE_HIP)
    expect_typed_throw<eh::GPUOutOfMemory>(
        static_cast<int32_t>(hipErrorOutOfMemory),
        "hipMalloc",
        hipGetErrorString(hipErrorOutOfMemory));
    expect_typed_throw<eh::GPUIllegalAddress>(
        static_cast<int32_t>(hipErrorIllegalAddress),
        "hipMemcpy",
        hipGetErrorString(hipErrorIllegalAddress));
    expect_typed_throw<eh::GPULaunchFailure>(
        static_cast<int32_t>(hipErrorLaunchFailure),
        "hipLaunchKernel",
        hipGetErrorString(hipErrorLaunchFailure));
    expect_typed_throw<eh::GPUInvalidValue>(
        static_cast<int32_t>(hipErrorInvalidValue),
        "hipMemcpyAsync",
        hipGetErrorString(hipErrorInvalidValue));
    expect_typed_throw<eh::GPUNotInitialized>(
        static_cast<int32_t>(hipErrorNotInitialized),
        "hipGetDevice",
        hipGetErrorString(hipErrorNotInitialized));

    {
        auto threw = false;
        try {
            eh::detail::throw_gpu_error(static_cast<int32_t>(hipErrorInvalidDevice), "unknown_api", __FILE__, __LINE__);
        }
        catch (const eh::GPUError& e) {
            threw = true;
            const auto what = std::string{e.what()};
            CHECK(contains(what, "GPUError:"));
            CHECK(!contains(what, "GPUOutOfMemory"));
            CHECK(contains(what, "unknown_api"));
            CHECK(contains(what, __FILE__));
            CHECK(contains(what, hipGetErrorString(hipErrorInvalidDevice)));
        }
        catch (...) {
            threw = true;
            CHECK(false);
        }
        CHECK(threw);
    }
#elif defined(USE_CUDA)
    expect_typed_throw<eh::GPUOutOfMemory>(
        static_cast<int32_t>(cudaErrorMemoryAllocation),
        "cudaMalloc",
        cudaGetErrorString(cudaErrorMemoryAllocation));
    expect_typed_throw<eh::GPUIllegalAddress>(
        static_cast<int32_t>(cudaErrorIllegalAddress),
        "cudaMemcpy",
        cudaGetErrorString(cudaErrorIllegalAddress));
    expect_typed_throw<eh::GPULaunchFailure>(
        static_cast<int32_t>(cudaErrorLaunchFailure),
        "cudaLaunchKernel",
        cudaGetErrorString(cudaErrorLaunchFailure));
    expect_typed_throw<eh::GPUInvalidValue>(
        static_cast<int32_t>(cudaErrorInvalidValue),
        "cudaMemcpyAsync",
        cudaGetErrorString(cudaErrorInvalidValue));
    expect_typed_throw<eh::GPUNotInitialized>(
        static_cast<int32_t>(cudaErrorInitializationError),
        "cudaGetDevice",
        cudaGetErrorString(cudaErrorInitializationError));

    {
        auto threw = false;
        try {
            eh::detail::throw_gpu_error(static_cast<int32_t>(cudaErrorInvalidDevice), "unknown_api", __FILE__, __LINE__);
        }
        catch (const eh::GPUError& e) {
            threw = true;
            const auto what = std::string{e.what()};
            CHECK(contains(what, "GPUError:"));
            CHECK(!contains(what, "GPUOutOfMemory"));
            CHECK(contains(what, "unknown_api"));
            CHECK(contains(what, __FILE__));
            CHECK(contains(what, cudaGetErrorString(cudaErrorInvalidDevice)));
        }
        catch (...) {
            threw = true;
            CHECK(false);
        }
        CHECK(threw);
    }
#else
    {
        auto threw = false;
        try {
            eh::detail::throw_gpu_error(42, "cpu_api", __FILE__, __LINE__);
        }
        catch (const eh::GPUError& e) {
            threw = true;
            const auto what = std::string{e.what()};
            CHECK(contains(what, "GPUError:"));
            CHECK(contains(what, "cpu_api"));
            CHECK(contains(what, __FILE__));
            CHECK(contains(what, "code 42"));
        }
        catch (...) {
            threw = true;
            CHECK(false);
        }
        CHECK(threw);
    }
#endif
}

static void test_gpu_error_base_catchability() {
#if defined(USE_HIP)
    auto caught_as_base = false;
    try {
        eh::detail::throw_gpu_error(static_cast<int32_t>(hipErrorOutOfMemory), "base_test", __FILE__, __LINE__);
    }
    catch (const eh::GPUError& e) {
        caught_as_base = true;
        CHECK(contains(std::string{e.what()}, "GPUOutOfMemory:"));
        CHECK(contains(std::string{e.what()}, "base_test"));
    }
    catch (...) {
        CHECK(false);
    }
    CHECK(caught_as_base);
#elif defined(USE_CUDA)
    auto caught_as_base = false;
    try {
        eh::detail::throw_gpu_error(static_cast<int32_t>(cudaErrorMemoryAllocation), "base_test", __FILE__, __LINE__);
    }
    catch (const eh::GPUError& e) {
        caught_as_base = true;
        CHECK(contains(std::string{e.what()}, "GPUOutOfMemory:"));
        CHECK(contains(std::string{e.what()}, "base_test"));
    }
    catch (...) {
        CHECK(false);
    }
    CHECK(caught_as_base);
#endif
}

int32_t main() {
    test_gpu_error_mapping();
    test_gpu_error_base_catchability();
    if (failures == 0) {
        std::cout << "ALL GPU ERROR TESTS PASSED\n";
        return 0;
    }
    std::cerr << failures << " FAILURE(S)\n";
    return 1;
}
