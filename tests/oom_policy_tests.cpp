/**
 * @file oom_policy_tests.cpp
 * @brief OOM policy mode tests.
 *
 * Validates the three process-wide OOM policy modes (PreserveLogThrow,
 * H2DFallback, CheckpointAndThrow), the append-only diagnostic log, config
 * validation, and the fake compute-dispatcher boundary test seam. Transfer
 * integration tests run only when a GPU runtime is available.
 *
 * CUDA execution is verified via gpu-test.sh. The HIP code path is built when
 * USE_HIP is enabled but is reported unverified on this box (no ROCm runtime).
 * @version 0.1
 * @date 2026-11-14
 */
#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "Helpers/OOMPolicy.h"
#include "Module/Linear.h"
#include "Serialization/Serialization.h"
#include "Tensor/Tensor.h"

namespace eh = BeanTensor::ErrorHandling;
using BeanTensor::DataTypes::DType;
using BeanTensor::Helpers::OOMPolicy;
using BeanTensor::Helpers::OOMPolicyAction;
using BeanTensor::Helpers::OOMPolicyConfig;
using BeanTensor::Helpers::OOMPolicyMode;
using BeanTensor::Tensors::Device;
using BeanTensor::Tensors::Tensor;
using BeanTensor::Tensors::detail::classify_transfer_failure;
using BeanTensor::float32_t;
using BeanTensor::Modules::Linear;
namespace ser = BeanTensor::Serialization;

static int32_t failures = 0;
#define CHECK(...) do { \
    if (!(__VA_ARGS__)) { \
        ++failures; \
        std::cerr << "FAIL: " << #__VA_ARGS__ << " @ line " << __LINE__ << "\n"; \
    } \
} while (0)

static std::filesystem::path scratch_path(const std::string& name) {
    std::filesystem::create_directories("scratch");
    return std::filesystem::path("scratch") / name;
}

static std::string read_log(const std::string& path) {
    auto f = std::ifstream(path);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
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

static bool eq32(const float32_t a, const float32_t b) {
    return std::abs(a - b) < 1e-6f;
}

static void reset_policy() {
    OOMPolicy::instance().reset_for_tests();
    OOMPolicy::set_compute_fallback_eligible_for_test(false);
#if defined(BEANTENSOR_TEST_FAULT_INJECTION)
    BeanTensor::Tensors::detail::TransferFaultInjection::reset();
#endif
}

static void test_checkpoint_config_requires_hook() {
    reset_policy();
    auto config = OOMPolicyConfig{};
    config.mode = OOMPolicyMode::CheckpointAndThrow;
    config.checkpoint_destination = "scratch/oom_checkpoint.bt";
    auto threw = false;
    try {
        OOMPolicy::instance().configure(config);
    } catch (const eh::InvalidOOMPolicy&) {
        threw = true;
    }
    CHECK(threw);
}

static void test_checkpoint_config_requires_destination() {
    reset_policy();
    auto config = OOMPolicyConfig{};
    config.mode = OOMPolicyMode::CheckpointAndThrow;
    config.checkpoint_hook = [](const std::exception&) { };
    auto threw = false;
    try {
        OOMPolicy::instance().configure(config);
    } catch (const eh::InvalidOOMPolicy&) {
        threw = true;
    }
    CHECK(threw);
}

static void test_preserve_mode_logs_and_rethrows_h2d() {
    reset_policy();
    auto config = OOMPolicyConfig{};
    config.mode = OOMPolicyMode::PreserveLogThrow;
    const auto log_path = scratch_path("oom_preserve.log").string();
    config.log_path = log_path;
    OOMPolicy::instance().configure(config);

    auto classification = classify_transfer_failure(Device::CPU, Device::GPU, false);
    const auto x = Tensor({4}, DType::Float32, Device::CPU);
    const eh::OutOfMemory oom;
    const auto action = OOMPolicy::instance().handle_transfer_failure(classification, oom, x, Device::GPU);
    CHECK(action == OOMPolicyAction::Rethrow);

    const auto log = read_log(log_path);
    CHECK(log.find("OOM transfer CPU -> GPU") != std::string::npos);
    CHECK(log.find("pre=1") != std::string::npos);
    CHECK(log.find("h2d=1") != std::string::npos);
}

static void test_h2d_fallback_swallows_eligible() {
    reset_policy();
    auto config = OOMPolicyConfig{};
    config.mode = OOMPolicyMode::H2DFallback;
    const auto log_path = scratch_path("oom_h2d.log").string();
    config.log_path = log_path;
    OOMPolicy::instance().configure(config);

    auto classification = classify_transfer_failure(Device::CPU, Device::GPU, true);
    const auto x = Tensor({4}, DType::Float32, Device::CPU);
    const eh::OutOfMemory oom;
    const auto action = OOMPolicy::instance().handle_transfer_failure(classification, oom, x, Device::GPU);
    CHECK(action == OOMPolicyAction::Swallow);

    const auto log = read_log(log_path);
    CHECK(log.find("H2D fallback: leaving tensor on CPU") != std::string::npos);
}

static void test_fallback_meaningless_rethrows() {
    reset_policy();
    auto config = OOMPolicyConfig{};
    config.mode = OOMPolicyMode::H2DFallback;
    OOMPolicy::instance().configure(config);

    auto classification = classify_transfer_failure(Device::GPU, Device::CPU, false);
    const auto x = Tensor({4}, DType::Float32, Device::CPU);
    const eh::OutOfMemory oom;
    const auto action = OOMPolicy::instance().handle_transfer_failure(classification, oom, x, Device::CPU);
    CHECK(action == OOMPolicyAction::Rethrow);
}

static void test_checkpoint_mode_calls_hook_and_rethrows() {
    reset_policy();
    auto config = OOMPolicyConfig{};
    config.mode = OOMPolicyMode::CheckpointAndThrow;
    auto hook_called = false;
    std::string captured_what;
    config.checkpoint_hook = [&](const std::exception& e) {
        hook_called = true;
        captured_what = e.what();
    };
    config.checkpoint_destination = "scratch/oom_checkpoint.bt";
    OOMPolicy::instance().configure(config);

    auto classification = classify_transfer_failure(Device::CPU, Device::GPU, false);
    const auto x = Tensor({4}, DType::Float32, Device::CPU);
    const eh::OutOfMemory oom;
    const auto action = OOMPolicy::instance().handle_transfer_failure(classification, oom, x, Device::GPU);
    CHECK(action == OOMPolicyAction::Rethrow);
    CHECK(hook_called);
    CHECK(!captured_what.empty());
}

static void test_compute_fallback_fake_boundary() {
    reset_policy();
    auto config = OOMPolicyConfig{};
    config.mode = OOMPolicyMode::H2DFallback;
    OOMPolicy::instance().configure(config);

    const auto x = Tensor({4}, DType::Float32, Device::CPU);
    const eh::OutOfMemory oom;

    OOMPolicy::set_compute_fallback_eligible_for_test(true);
    auto action = OOMPolicy::instance().handle_compute_failure(oom, x);
    CHECK(action == OOMPolicyAction::Swallow);

    OOMPolicy::set_compute_fallback_eligible_for_test(false);
    action = OOMPolicy::instance().handle_compute_failure(oom, x);
    CHECK(action == OOMPolicyAction::Rethrow);
}

static void test_logging_failure_does_not_mask() {
    reset_policy();
    auto config = OOMPolicyConfig{};
    config.mode = OOMPolicyMode::PreserveLogThrow;
    // A path that cannot be opened (parent directory does not exist and we do
    // not create it) should not prevent the original OOM from being rethrown.
    config.log_path = "/nonexistent_dir_/oom.log";
    OOMPolicy::instance().configure(config);

    auto classification = classify_transfer_failure(Device::CPU, Device::GPU, false);
    const auto x = Tensor({4}, DType::Float32, Device::CPU);
    const eh::OutOfMemory oom;
    const auto action = OOMPolicy::instance().handle_transfer_failure(classification, oom, x, Device::GPU);
    CHECK(action == OOMPolicyAction::Rethrow);
}

static void test_log_appends_multiple_failures() {
    reset_policy();
    auto config = OOMPolicyConfig{};
    config.mode = OOMPolicyMode::PreserveLogThrow;
    const auto log_path = scratch_path("oom_append.log").string();
    config.log_path = log_path;
    OOMPolicy::instance().configure(config);

    const auto x = Tensor({4}, DType::Float32, Device::CPU);
    const eh::OutOfMemory oom;
    auto first = classify_transfer_failure(Device::CPU, Device::GPU, false);
    auto second = classify_transfer_failure(Device::CPU, Device::GPU, true);

    (void)OOMPolicy::instance().handle_transfer_failure(first, oom, x, Device::GPU);
    (void)OOMPolicy::instance().handle_transfer_failure(second, oom, x, Device::GPU);

    const auto log = read_log(log_path);
    auto first_line = log.find("OOM transfer CPU -> GPU");
    auto second_line = log.rfind("OOM transfer CPU -> GPU");
    CHECK(first_line != std::string::npos);
    CHECK(second_line != std::string::npos);
    CHECK(first_line != second_line);
    CHECK(log.find("pre=1") != std::string::npos);
    CHECK(log.find("temp=1") != std::string::npos);
}

static void test_log_concurrent_storm() {
    reset_policy();
    auto config = OOMPolicyConfig{};
    config.mode = OOMPolicyMode::PreserveLogThrow;
    const auto log_path = scratch_path("oom_storm.log").string();
    std::error_code ec;
    std::filesystem::remove(log_path, ec);
    config.log_path = log_path;
    OOMPolicy::instance().configure(config);

    constexpr auto k_threads = size_t{8};
    constexpr auto k_messages = size_t{64};
    const auto x = Tensor({4}, DType::Float32, Device::CPU);
    const eh::OutOfMemory oom;
    auto classification = classify_transfer_failure(Device::CPU, Device::GPU, false);

    auto threads = std::vector<std::thread>{};
    threads.reserve(k_threads);
    auto launched = std::atomic<size_t>{0};
    for (size_t t = 0; t < k_threads; ++t) {
        threads.emplace_back([&]() {
            launched.fetch_add(1);
            for (size_t i = 0; i < k_messages; ++i) {
                (void)OOMPolicy::instance().handle_transfer_failure(classification, oom, x, Device::GPU);
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }

    const auto log = read_log(log_path);
    auto lines = size_t{0};
    auto pos = size_t{0};
    while ((pos = log.find('\n', pos)) != std::string::npos) {
        ++lines;
        ++pos;
    }
    CHECK(lines == k_threads * k_messages);
    CHECK(launched.load() == k_threads);
}

static bool bit_same32(const float32_t a, const float32_t b) {
    auto ua = uint32_t{0};
    auto ub = uint32_t{0};
    std::memcpy(&ua, &a, sizeof ua);
    std::memcpy(&ub, &b, sizeof ub);
    return ua == ub;
}

static bool params_bit_equal(BeanTensor::Modules::Module& a, BeanTensor::Modules::Module& b) {
    const auto pa = a.named_parameters();
    const auto pb = b.named_parameters();
    if (pa.size() != pb.size()) {
        return false;
    }
    for (auto k = size_t{0}; k < pa.size(); ++k) {
        if (pa[k].first != pb[k].first) {
            return false;
        }
        auto* x = pa[k].second;
        auto* y = pb[k].second;
        if (x->rank() != y->rank() || x->get_device() != y->get_device()) {
            return false;
        }
        for (auto d = size_t{0}; d < x->rank(); ++d) {
            if (x->dim(d) != y->dim(d)) {
                return false;
            }
        }
        if (x->rank() == 2) {
            for (auto i = size_t{0}; i < x->dim(0); ++i) {
                for (auto j = size_t{0}; j < x->dim(1); ++j) {
                    if (!bit_same32(x->at<float32_t>({i, j}), y->at<float32_t>({i, j}))) {
                        return false;
                    }
                }
            }
        } else {
            for (auto i = size_t{0}; i < x->element_count(); ++i) {
                if (!bit_same32(x->at<float32_t>({i}), y->at<float32_t>({i}))) {
                    return false;
                }
            }
        }
        const auto ca = x->adam_step_count();
        const auto cb = y->adam_step_count();
        if (ca.has_value() != cb.has_value()) {
            return false;
        }
        if (ca.has_value() && ca.value() != cb.value()) {
            return false;
        }
    }
    return true;
}

static float32_t train_step(BeanTensor::Modules::Module& model, const Tensor& xs, const Tensor& ys) {
    model.zero_grad();
    auto logits = model.forward(xs);
    auto loss = logits.mse(ys);
    loss.backward(DType::Float32);
    const auto loss_value = loss.at<float32_t>({0});
    model.adam_step(0.05, 0.9, 0.999, 1e-8);
    return loss_value;
}

static void test_checkpoint_mode_full_checkpoint_resumes_bit_reproducibly() {
    reset_policy();

    auto model = Linear(size_t{4}, size_t{3}, DType::Float32, Device::CPU);
    model.train();
    auto init_idx = size_t{0};
    for (auto& entry : model.named_parameters()) {
        auto* p = entry.second;
        if (p->rank() == 2) {
            for (auto i = size_t{0}; i < p->dim(0); ++i) {
                for (auto j = size_t{0}; j < p->dim(1); ++j) {
                    p->set<float32_t>({i, j}, 0.05f * static_cast<float32_t>(init_idx) - 0.3f);
                    ++init_idx;
                }
            }
        } else {
            for (auto i = size_t{0}; i < p->element_count(); ++i) {
                p->set<float32_t>({i}, 0.05f * static_cast<float32_t>(init_idx) - 0.3f);
                ++init_idx;
            }
        }
    }

    auto xs = Tensor({1, 4}, DType::Float32, Device::CPU, false);
    auto ys = Tensor({1, 3}, DType::Float32, Device::CPU, false);
    for (auto j = size_t{0}; j < 4; ++j) {
        xs.set<float32_t>({0, j}, 0.25f * static_cast<float32_t>(j) - 0.2f);
    }
    for (auto j = size_t{0}; j < 3; ++j) {
        ys.set<float32_t>({0, j}, 0.4f * static_cast<float32_t>(j) - 0.3f);
    }

    // Two real training steps so parameters, gradients, Adam moments, and the
    // step counter are all non-trivial in the checkpoint.
    train_step(model, xs, ys);
    train_step(model, xs, ys);

    // A decoy stands in for the last usable recovery file: the hook must
    // replace it via temp-file + rename, never write in place.
    const auto dest = scratch_path("oom_full_checkpoint.bt");
    const auto tmp = scratch_path("oom_full_checkpoint.bt.tmp");
    {
        auto decoy = std::ofstream(dest.string());
        decoy << "stale";
    }
    auto hook_called = false;
    auto config = OOMPolicyConfig{};
    config.mode = OOMPolicyMode::CheckpointAndThrow;
    config.checkpoint_destination = dest.string();
    config.checkpoint_hook = [&](const std::exception&) {
        ser::save(model, tmp.string(), true, ser::CheckpointLevel::Full);
        std::filesystem::rename(tmp, dest);
        hook_called = true;
    };
    OOMPolicy::instance().configure(config);

    const auto classification = classify_transfer_failure(Device::CPU, Device::GPU, false);
    const auto x = Tensor({4}, DType::Float32, Device::CPU);
    const eh::OutOfMemory oom;
    const auto action = OOMPolicy::instance().handle_transfer_failure(classification, oom, x, Device::GPU);
    CHECK(action == OOMPolicyAction::Rethrow);
    CHECK(hook_called);
    CHECK(!std::filesystem::exists(tmp));
    CHECK(std::filesystem::exists(dest));

    auto loaded = ser::load(dest.string(), Device::CPU);
    CHECK(params_bit_equal(model, *loaded));

    // Both models take the same next step: losses and resulting parameters
    // must agree bitwise, proving the Full checkpoint resumes bit-reproducibly.
    const auto loss_orig = train_step(model, xs, ys);
    const auto loss_resumed = train_step(*loaded, xs, ys);
    CHECK(bit_same32(loss_orig, loss_resumed));
    CHECK(params_bit_equal(model, *loaded));
}

#if defined(USE_CUDA) || defined(USE_HIP)
static void test_h2d_alloc_failure_fallback_to_cpu() {
    reset_policy();
    auto config = OOMPolicyConfig{};
    config.mode = OOMPolicyMode::H2DFallback;
    config.log_path = scratch_path("oom_h2d_alloc_fallback.log").string();
    OOMPolicy::instance().configure(config);

    auto x = Tensor({2, 3}, DType::Float32, Device::CPU);
    x.set_one();
    BeanTensor::Tensors::detail::TransferFaultInjection::force_alloc_fail.store(true);
    auto threw = false;
    try {
        x.to(Device::GPU);
    } catch (...) {
        threw = true;
    }
    BeanTensor::Tensors::detail::TransferFaultInjection::reset();
    CHECK(!threw);
    CHECK(x.get_device() == Device::CPU);
    CHECK(eq32(x.at<float32_t>({1, 2}), 1.0f));
}

static void test_h2d_copy_failure_fallback_to_cpu() {
    reset_policy();
    auto config = OOMPolicyConfig{};
    config.mode = OOMPolicyMode::H2DFallback;
    OOMPolicy::instance().configure(config);

    auto x = Tensor({2, 3}, DType::Float32, Device::CPU);
    x.set_one();
    BeanTensor::Tensors::detail::TransferFaultInjection::force_copy_fail.store(true);
    auto threw = false;
    try {
        x.to(Device::GPU);
    } catch (...) {
        threw = true;
    }
    BeanTensor::Tensors::detail::TransferFaultInjection::reset();
    CHECK(!threw);
    CHECK(x.get_device() == Device::CPU);
    CHECK(eq32(x.at<float32_t>({1, 2}), 1.0f));
}

static void test_preserve_mode_rethrows_h2d_transfer() {
    reset_policy();
    auto config = OOMPolicyConfig{};
    config.mode = OOMPolicyMode::PreserveLogThrow;
    config.log_path = scratch_path("oom_preserve_transfer.log").string();
    OOMPolicy::instance().configure(config);

    auto x = Tensor({2, 3}, DType::Float32, Device::CPU);
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

    const auto log = read_log(config.log_path);
    CHECK(log.find("OOM transfer CPU -> GPU") != std::string::npos);
}

static void test_checkpoint_mode_rethrows_h2d_transfer() {
    reset_policy();
    auto config = OOMPolicyConfig{};
    config.mode = OOMPolicyMode::CheckpointAndThrow;
    auto hook_called = false;
    config.checkpoint_hook = [&](const std::exception&) { hook_called = true; };
    config.checkpoint_destination = "scratch/oom_checkpoint.bt";
    OOMPolicy::instance().configure(config);

    auto x = Tensor({2, 3}, DType::Float32, Device::CPU);
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
    CHECK(hook_called);
    CHECK(x.get_device() == Device::CPU);
    CHECK(eq32(x.at<float32_t>({1, 2}), 1.0f));
}
#endif

int main() {
    test_checkpoint_config_requires_hook();
    test_checkpoint_config_requires_destination();
    test_preserve_mode_logs_and_rethrows_h2d();
    test_h2d_fallback_swallows_eligible();
    test_fallback_meaningless_rethrows();
    test_checkpoint_mode_calls_hook_and_rethrows();
    test_checkpoint_mode_full_checkpoint_resumes_bit_reproducibly();
    test_compute_fallback_fake_boundary();
    test_logging_failure_does_not_mask();
    test_log_appends_multiple_failures();
    test_log_concurrent_storm();

#if defined(USE_CUDA) || defined(USE_HIP)
    if (runtime_gpu_present()) {
        test_h2d_alloc_failure_fallback_to_cpu();
        test_h2d_copy_failure_fallback_to_cpu();
        test_preserve_mode_rethrows_h2d_transfer();
        test_checkpoint_mode_rethrows_h2d_transfer();
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
