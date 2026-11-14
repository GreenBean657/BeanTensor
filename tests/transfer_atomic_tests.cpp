/**
 * @file transfer_atomic_tests.cpp
 * @brief Failure-atomic device-transfer tests.
 *
 * Validates the new Tensor::to() state machine: source is untouched until
 * allocate+copy succeed, views become owned contiguous copies, the gradient
 * rides along, failures are non-sticky, and the classification hook returns
 * the expected flags. Fault injection is compiled in only for test builds
 * (BEANTENSOR_TEST_FAULT_INJECTION).
 *
 * CUDA execution is verified via gpu-test.sh. The HIP code path is built when
 * USE_HIP is enabled but is reported unverified on this box (no ROCm runtime).
 * @version 0.1
 * @date 2026-11-14
 */
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "Tensor/Tensor.h"

namespace eh = BeanTensor::ErrorHandling;
using BeanTensor::DataTypes::DType;
using BeanTensor::Tensors::Device;
using BeanTensor::Tensors::Tensor;
using BeanTensor::Tensors::detail::classify_transfer_failure;
using BeanTensor::float32_t;

static int32_t failures = 0;
#define CHECK(...) do { \
    if (!(__VA_ARGS__)) { \
        ++failures; \
        std::cerr << "FAIL: " << #__VA_ARGS__ << " @ line " << __LINE__ << "\n"; \
    } \
} while (0)

static bool eq32(const float32_t a, const float32_t b) {
    return std::abs(a - b) < 1e-6f;
}

static bool runtime_gpu_present() {
#if defined(USE_CUDA)
    int32_t count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        return false;
    }
    return count > 0;
#elif defined(USE_HIP)
    int32_t count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess) {
        return false;
    }
    return count > 0;
#else
    return false;
#endif
}

static void test_cpu_to_cpu_noop() {
    auto x = Tensor({2, 3}, DType::Float32, Device::CPU, false);
    x.set_one();
    x.to(Device::CPU);
    CHECK(x.get_device() == Device::CPU);
    CHECK(eq32(x.at<float32_t>({1, 2}), 1.0f));
}

#if !defined(USE_CUDA) && !defined(USE_HIP)
static void test_gpu_request_on_cpu_build_throws() {
    auto x = Tensor({2}, DType::Float32, Device::CPU);
    x.set_one();
    auto threw = false;
    try {
        x.to(Device::GPU);
    } catch (const eh::GPUTaskOnCPUBuild&) {
        threw = true;
    }
    CHECK(threw);
    CHECK(x.get_device() == Device::CPU);
    CHECK(eq32(x.at<float32_t>({0}), 1.0f));
}
#endif

static void test_classification() {
    const auto h2d_alloc = classify_transfer_failure(Device::CPU, Device::GPU, false);
    CHECK(h2d_alloc.alloc_failed_pre_change);
    CHECK(!h2d_alloc.copy_failed_with_temp);
    CHECK(h2d_alloc.h2d_fallback_eligible);
    CHECK(!h2d_alloc.fallback_meaningless);

    const auto h2d_copy = classify_transfer_failure(Device::CPU, Device::GPU, true);
    CHECK(!h2d_copy.alloc_failed_pre_change);
    CHECK(h2d_copy.copy_failed_with_temp);
    CHECK(h2d_copy.h2d_fallback_eligible);
    CHECK(!h2d_copy.fallback_meaningless);

    const auto d2h_alloc = classify_transfer_failure(Device::GPU, Device::CPU, false);
    CHECK(d2h_alloc.alloc_failed_pre_change);
    CHECK(!d2h_alloc.copy_failed_with_temp);
    CHECK(!d2h_alloc.h2d_fallback_eligible);
    CHECK(d2h_alloc.fallback_meaningless);

    const auto d2h_copy = classify_transfer_failure(Device::GPU, Device::CPU, true);
    CHECK(!d2h_copy.alloc_failed_pre_change);
    CHECK(d2h_copy.copy_failed_with_temp);
    CHECK(!d2h_copy.h2d_fallback_eligible);
    CHECK(d2h_copy.fallback_meaningless);
}

#if defined(USE_CUDA) || defined(USE_HIP)
static void test_h2d_roundtrip_data_intact() {
    auto x = Tensor({2, 3}, DType::Float32, Device::CPU, false);
    for (size_t i = 0; i < x.dim(0); ++i) {
        for (size_t j = 0; j < x.dim(1); ++j) {
            x.set({i, j}, static_cast<float32_t>(i * 10 + j));
        }
    }
    const auto before = x.at<float32_t>({1, 2});
    x.to(Device::GPU);
    CHECK(x.get_device() == Device::GPU);
    CHECK(eq32(x.at<float32_t>({1, 2}), before));
    x.to(Device::CPU);
    CHECK(x.get_device() == Device::CPU);
    CHECK(eq32(x.at<float32_t>({1, 2}), before));
}

static void test_h2d_alloc_failure_source_unchanged() {
    auto x = Tensor({2, 3}, DType::Float32, Device::CPU, false);
    x.set_one();
    BeanTensor::Tensors::detail::TransferFaultInjection::force_alloc_fail.store(true);
    auto threw = false;
    try {
        x.to(Device::GPU);
    } catch (...) {
        threw = true;
    }
    BeanTensor::Tensors::detail::TransferFaultInjection::reset();
    CHECK(threw);
    CHECK(x.get_device() == Device::CPU);
    CHECK(eq32(x.at<float32_t>({1, 2}), 1.0f));
    // Retry succeeds and the tensor is now retryable.
    x.to(Device::GPU);
    CHECK(x.get_device() == Device::GPU);
    CHECK(eq32(x.at<float32_t>({1, 2}), 1.0f));
}

static void test_h2d_copy_failure_source_unchanged() {
    auto x = Tensor({2, 3}, DType::Float32, Device::CPU, false);
    x.set_one();
    BeanTensor::Tensors::detail::TransferFaultInjection::force_copy_fail.store(true);
    auto threw = false;
    try {
        x.to(Device::GPU);
    } catch (...) {
        threw = true;
    }
    BeanTensor::Tensors::detail::TransferFaultInjection::reset();
    CHECK(threw);
    CHECK(x.get_device() == Device::CPU);
    CHECK(eq32(x.at<float32_t>({1, 2}), 1.0f));
    x.to(Device::GPU);
    CHECK(x.get_device() == Device::GPU);
    CHECK(eq32(x.at<float32_t>({1, 2}), 1.0f));
}

static void test_d2h_alloc_failure_source_unchanged() {
    auto x = Tensor({2, 3}, DType::Float32, Device::CPU, false);
    x.set_one();
    x.to(Device::GPU);
    BeanTensor::Tensors::detail::TransferFaultInjection::force_alloc_fail.store(true);
    auto threw = false;
    try {
        x.to(Device::CPU);
    } catch (...) {
        threw = true;
    }
    BeanTensor::Tensors::detail::TransferFaultInjection::reset();
    CHECK(threw);
    CHECK(x.get_device() == Device::GPU);
    CHECK(eq32(x.at<float32_t>({1, 2}), 1.0f));
    x.to(Device::CPU);
    CHECK(x.get_device() == Device::CPU);
    CHECK(eq32(x.at<float32_t>({1, 2}), 1.0f));
}

static void test_d2h_copy_failure_source_unchanged() {
    auto x = Tensor({2, 3}, DType::Float32, Device::CPU, false);
    x.set_one();
    x.to(Device::GPU);
    BeanTensor::Tensors::detail::TransferFaultInjection::force_copy_fail.store(true);
    auto threw = false;
    try {
        x.to(Device::CPU);
    } catch (...) {
        threw = true;
    }
    BeanTensor::Tensors::detail::TransferFaultInjection::reset();
    CHECK(threw);
    CHECK(x.get_device() == Device::GPU);
    CHECK(eq32(x.at<float32_t>({1, 2}), 1.0f));
    x.to(Device::CPU);
    CHECK(x.get_device() == Device::CPU);
    CHECK(eq32(x.at<float32_t>({1, 2}), 1.0f));
}

static void test_view_transfer_becomes_owned_contiguous() {
    auto base = Tensor({4, 4}, DType::Float32, Device::CPU, false);
    for (size_t i = 0; i < base.dim(0); ++i) {
        for (size_t j = 0; j < base.dim(1); ++j) {
            base.set({i, j}, static_cast<float32_t>(i * 10 + j));
        }
    }
    // A contiguous leading-dimension view: rows 1..2, all columns.
    auto ranges = std::vector<std::pair<size_t, size_t>>{{1, 3}, {0, 4}};
    auto v = base.slice(ranges);
    CHECK(v.get_device() == Device::CPU);
    CHECK(eq32(v.at<float32_t>({0, 0}), 10.0f));
    CHECK(eq32(v.at<float32_t>({1, 1}), 21.0f));

    v.to(Device::GPU);
    CHECK(v.get_device() == Device::GPU);
    CHECK(eq32(v.at<float32_t>({0, 0}), 10.0f));
    CHECK(eq32(v.at<float32_t>({1, 1}), 21.0f));

    v.to(Device::CPU);
    CHECK(v.get_device() == Device::CPU);
    CHECK(eq32(v.at<float32_t>({1, 1}), 21.0f));

    // The original base is unchanged.
    CHECK(eq32(base.at<float32_t>({2, 1}), 21.0f));
}

static void test_view_transfer_alloc_failure_unchanged() {
    auto base = Tensor({4, 4}, DType::Float32, Device::CPU, false);
    for (size_t i = 0; i < base.dim(0); ++i) {
        for (size_t j = 0; j < base.dim(1); ++j) {
            base.set({i, j}, static_cast<float32_t>(i * 10 + j));
        }
    }
    auto ranges = std::vector<std::pair<size_t, size_t>>{{1, 3}, {0, 4}};
    auto v = base.slice(ranges);
    BeanTensor::Tensors::detail::TransferFaultInjection::force_alloc_fail.store(true);
    auto threw = false;
    try {
        v.to(Device::GPU);
    } catch (...) {
        threw = true;
    }
    BeanTensor::Tensors::detail::TransferFaultInjection::reset();
    CHECK(threw);
    CHECK(v.get_device() == Device::CPU);
    CHECK(eq32(v.at<float32_t>({1, 1}), 21.0f));
    CHECK(eq32(base.at<float32_t>({2, 1}), 21.0f));
}

static void test_view_transfer_alloc_failure_parent_dies_first() {
    // Regression for the round-1 P1: a failed transfer must not detach a
    // contiguous view from its parent before the failure point. If it did,
    // destroying the parent before the view would leave the view dangling.
    auto v = [&]() {
        auto base = Tensor({4, 4}, DType::Float32, Device::CPU, false);
        for (size_t i = 0; i < base.dim(0); ++i) {
            for (size_t j = 0; j < base.dim(1); ++j) {
                base.set({i, j}, static_cast<float32_t>(i * 10 + j));
            }
        }
        auto ranges = std::vector<std::pair<size_t, size_t>>{{1, 3}, {0, 4}};
        auto view = base.slice(ranges);
        BeanTensor::Tensors::detail::TransferFaultInjection::force_alloc_fail.store(true);
        auto threw = false;
        try {
            view.to(Device::GPU);
        } catch (...) {
            threw = true;
        }
        BeanTensor::Tensors::detail::TransferFaultInjection::reset();
        CHECK(threw);
        CHECK(view.get_device() == Device::CPU);
        CHECK(eq32(view.at<float32_t>({1, 1}), 21.0f));
        // base is destroyed here; because the view is still registered, the
        // destructor hands it off into its own buffer.
        return view;
    }();
    CHECK(v.get_device() == Device::CPU);
    CHECK(eq32(v.at<float32_t>({1, 1}), 21.0f));
}

static void test_grad_rides_along() {
    auto x = Tensor({2, 3}, DType::Float32, Device::CPU, true);
    x.set_one();
    auto s = x.sum();
    s.backward(DType::Float32);
    CHECK(x.has_grad());
    x.to(Device::GPU);
    CHECK(x.get_device() == Device::GPU);
    CHECK(x.has_grad());
    CHECK(x.gradient().get_device() == Device::GPU);
    CHECK(eq32(x.gradient().at<float32_t>({0, 0}), 1.0f));
}

static void test_grad_transfer_failure_non_sticky() {
    auto x = Tensor({2}, DType::Float32, Device::CPU, true);
    x.set_one();
    auto s = x.sum();
    s.backward(DType::Float32);
    // Move both data and grad to GPU, then back to CPU.
    x.to(Device::GPU);
    x.to(Device::CPU);
    CHECK(x.get_device() == Device::CPU);
    CHECK(x.gradient().get_device() == Device::CPU);

    // Arm the seam so the data allocation succeeds but the gradient allocation
    // fails. The data transfer should commit and the gradient should stay on
    // CPU; a retry then moves the gradient.
    BeanTensor::Tensors::detail::TransferFaultInjection::alloc_fail_after_n.store(2);
    auto threw = false;
    try {
        x.to(Device::GPU);
    } catch (...) {
        threw = true;
    }
    BeanTensor::Tensors::detail::TransferFaultInjection::reset();
    CHECK(threw);
    CHECK(x.get_device() == Device::GPU);
    CHECK(x.gradient().get_device() == Device::CPU);

    x.to(Device::GPU);
    CHECK(x.get_device() == Device::GPU);
    CHECK(x.gradient().get_device() == Device::GPU);
    CHECK(eq32(x.gradient().at<float32_t>({0}), 1.0f));
}
#endif

int main() {
    test_cpu_to_cpu_noop();
#if !defined(USE_CUDA) && !defined(USE_HIP)
    test_gpu_request_on_cpu_build_throws();
#endif
    test_classification();

#if defined(USE_CUDA) || defined(USE_HIP)
    if (runtime_gpu_present()) {
        test_h2d_roundtrip_data_intact();
        test_h2d_alloc_failure_source_unchanged();
        test_h2d_copy_failure_source_unchanged();
        test_d2h_alloc_failure_source_unchanged();
        test_d2h_copy_failure_source_unchanged();
        test_view_transfer_becomes_owned_contiguous();
        test_view_transfer_alloc_failure_unchanged();
        test_view_transfer_alloc_failure_parent_dies_first();
        test_grad_rides_along();
        test_grad_transfer_failure_non_sticky();
    } else {
        std::cout << "SKIP: GPU runtime not present\n";
    }
#endif

    if (failures == 0) {
        std::cout << "ALL TESTS PASSED\n";
        return 0;
    }
    std::cout << failures << " TEST(S) FAILED\n";
    return 1;
}
