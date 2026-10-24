/**
 * @file grad_dtype_tests.cpp
 * @author GreenBean
 * @brief Gradient-dtype knob tests - framework-free assertion suite.
 * Covers Tensor::set_grad_dtype (pinning before the first backward, widening
 * and narrowing of an existing gradient, the Reject/Allow policy, same-dtype
 * no-op, non-leaf and no-gradient errors), Tensor::grad_dtype, the
 * validate-then-apply contract of Module::set_grad_dtype over a whole subtree,
 * Module::grad_dtype(name), and the DataTypes::is_narrowing_conversion table.
 * @version 0.1
 * @date 2026-10-23
 */
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include "Helpers/Errors.h"
#include "Module/Linear.h"
#include "Module/Module.h"
#include "Tensor/Tensor.h"

using BeanTensor::Tensors::Tensor;
using BeanTensor::Tensors::Device;
using BeanTensor::DataTypes::DType;
using BeanTensor::DataTypes::GradNarrowing;
using BeanTensor::DataTypes::is_narrowing_conversion;
using BeanTensor::float32_t;
using BeanTensor::float64_t;
using BeanTensor::Modules::Linear;

static int32_t passes = 0;
static int32_t failures = 0;

#define CHECK(cond) \
    do { \
        if (cond) { \
            ++passes; \
        } else { \
            ++failures; \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
        } \
    } while (0)

#define CHECK_THROWS(stmt, exc) \
    do { \
        auto thrown = false; \
        try { \
            static_cast<void>(stmt); \
        } catch (const exc&) { \
            thrown = true; \
        } \
        CHECK(thrown); \
    } while (0)

// Two-element Float32 constant c = {2, 2}. For a leaf x (requires_grad) and
// y = x.mul(c), dy/dx = c, so backward(seed = ones) leaves x.grad = {2, 2}.
static Tensor twos() {
    auto c = Tensor({2}, DType::Float32, Device::CPU, false);
    c.set<float32_t>({0}, float32_t{2});
    c.set<float32_t>({1}, float32_t{2});
    return c;
}

// Tolerance compare for the exact values used here (-Werror=float-equal rejects ==).
static bool near(const float64_t v, const float64_t want) {
    return std::abs(v - want) < 1e-12;
}

static void test_pin_before_backward() {
    auto x = Tensor({2}, DType::Float32, Device::CPU, true);
    auto c = twos();
    x.set_grad_dtype(DType::Float64);
    CHECK(x.has_grad());
    CHECK(x.grad_dtype() == DType::Float64);
    auto y = x.mul(c);
    // Seed and deltas arrive as Float32; the pinned Float64 buffer takes them.
    y.backward(DType::Float32);
    CHECK(x.grad_dtype() == DType::Float64);
    CHECK(near(x.gradient().at<float64_t>({0}), 2.0));
    CHECK(near(x.gradient().at<float64_t>({1}), 2.0));
}

static void test_widen_existing_grad() {
    auto x = Tensor({2}, DType::Float32, Device::CPU, true);
    auto c = twos();
    auto y = x.mul(c);
    y.backward(DType::Float32);
    CHECK(x.grad_dtype() == DType::Float32);
    x.set_grad_dtype(DType::Float64);
    CHECK(x.grad_dtype() == DType::Float64);
    CHECK(near(x.gradient().at<float64_t>({0}), 2.0));
    CHECK(near(x.gradient().at<float64_t>({1}), 2.0));
}

static void test_narrowing_rejected_by_default() {
    auto x = Tensor({2}, DType::Float32, Device::CPU, true);
    auto c = twos();
    x.set_grad_dtype(DType::Float64);
    auto y = x.mul(c);
    y.backward(DType::Float32);
    CHECK_THROWS(x.set_grad_dtype(DType::Float32), BeanTensor::ErrorHandling::NarrowingGradConversion);
    CHECK(x.grad_dtype() == DType::Float64);
    CHECK(near(x.gradient().at<float64_t>({0}), 2.0));
    CHECK_THROWS(x.validate_grad_dtype(DType::Float16), BeanTensor::ErrorHandling::NarrowingGradConversion);
    CHECK(x.grad_dtype() == DType::Float64);
}

static void test_narrowing_allowed() {
    auto x = Tensor({2}, DType::Float32, Device::CPU, true);
    auto c = twos();
    x.set_grad_dtype(DType::Float64);
    auto y = x.mul(c);
    y.backward(DType::Float32);
    x.set_grad_dtype(DType::Float32, GradNarrowing::Allow);
    CHECK(x.grad_dtype() == DType::Float32);
    CHECK(near(x.gradient().at<float32_t>({0}), 2.0));
    CHECK(near(x.gradient().at<float32_t>({1}), 2.0));
}

static void test_same_dtype_is_noop() {
    auto x = Tensor({2}, DType::Float32, Device::CPU, true);
    auto c = twos();
    auto y = x.mul(c);
    y.backward(DType::Float32);
    x.set_grad_dtype(DType::Float32);
    CHECK(x.grad_dtype() == DType::Float32);
    CHECK(near(x.gradient().at<float32_t>({0}), 2.0));
    CHECK(near(x.gradient().at<float32_t>({1}), 2.0));
}

static void test_fresh_leaf_pins_narrow_dtype() {
    // No gradient held yet, so no narrowing check applies: a fresh leaf may pin any dtype.
    auto x = Tensor({2}, DType::Float32, Device::CPU, true);
    CHECK(!x.grad_dtype().has_value());
    CHECK(!x.has_grad());
    x.set_grad_dtype(DType::Float16);
    CHECK(x.grad_dtype() == DType::Float16);
    CHECK(x.has_grad());
}

static void test_non_leaf_throws() {
    auto x = Tensor({2}, DType::Float32, Device::CPU, true);
    auto c = twos();
    auto y = x.mul(c);
    CHECK_THROWS(y.set_grad_dtype(DType::Float64), BeanTensor::ErrorHandling::NonLeafGradDType);
    CHECK(!y.grad_dtype().has_value());
}

static void test_no_gradient_throws() {
    auto z = Tensor({2}, DType::Float32, Device::CPU, false);
    CHECK_THROWS(z.set_grad_dtype(DType::Float64), BeanTensor::ErrorHandling::NoGradient);
    CHECK(!z.grad_dtype().has_value());
}

struct NarrowingCase {
    DType src;
    DType dst;
    bool narrowing;
};

static void test_narrowing_table() {
    const auto cases = std::vector<NarrowingCase>{
        // same dtype
        {DType::Float32, DType::Float32, false},
        {DType::Int16, DType::Int16, false},
        // float -> float: exact when significand, max exponent and min subnormal fit
        {DType::Float64, DType::Float32, true},
        {DType::Float32, DType::Float64, false},
        {DType::Float16, DType::Float32, false},
        {DType::BFloat16, DType::Float32, false},
        {DType::Float16, DType::BFloat16, true},
        {DType::BFloat16, DType::Float16, true},
        {DType::Float32, DType::Float16, true},
        {DType::Float16, DType::Float64, false},
        {DType::BFloat16, DType::Float64, false},
        // float -> integer always truncates
        {DType::Float32, DType::Int32, true},
        {DType::Float64, DType::Int64, true},
        {DType::Float16, DType::UInt8, true},
        // integer -> float: exact when the bit width fits the significand
        {DType::Int8, DType::Float16, false},
        {DType::UInt8, DType::Float16, false},
        {DType::Int8, DType::BFloat16, false},
        {DType::Int16, DType::Float16, true},
        {DType::Int16, DType::BFloat16, true},
        {DType::Int16, DType::Float32, false},
        {DType::Int32, DType::Float32, true},
        {DType::Int32, DType::Float64, false},
        {DType::Int64, DType::Float64, true},
        // integer -> integer: destination must hold every source value
        {DType::Int32, DType::Int64, false},
        {DType::Int64, DType::Int32, true},
        {DType::Int8, DType::UInt8, true},
        {DType::UInt8, DType::Int16, false},
        {DType::UInt8, DType::Int8, true},
        {DType::UInt16, DType::Int16, true},
        {DType::UInt32, DType::Int64, false},
        {DType::UInt64, DType::UInt32, true},
        {DType::UInt8, DType::UInt64, false},
        // packed sources drop the high lane
        {DType::Float16x2, DType::Float16, true},
        {DType::Float16x2, DType::BFloat16x2, true},
        {DType::Float16x2, DType::Float16x2, false},
        {DType::Int4x2, DType::Int8, true},
        {DType::Int4x2, DType::Int4x2, false},
        // unpacked into packed: narrowing exactly when narrowing into the lane type
        {DType::Float16, DType::Float16x2, false},
        {DType::BFloat16, DType::BFloat16x2, false},
        {DType::Float32, DType::Float16x2, true},
        {DType::Float32, DType::BFloat16x2, true},
        {DType::Int8, DType::Float16x2, false},
        {DType::Int16, DType::Float16x2, true},
        {DType::Int8, DType::Int4x2, true},
    };
    for (const auto& c : cases) {
        CHECK(is_narrowing_conversion(c.src, c.dst) == c.narrowing);
    }
}

static void test_module_grad_dtype() {
    auto lin = Linear(3, 2, DType::Float32, Device::CPU, true, 42);
    // Only the bias holds a gradient. A refused change must leave the weight
    // (no gradient yet) untouched: validate-all-then-apply, not a partial apply.
    lin.parameter("bias").set_grad_dtype(DType::Float64);
    CHECK(!lin.grad_dtype("weight").has_value());
    CHECK(lin.grad_dtype("bias") == DType::Float64);
    CHECK_THROWS(lin.set_grad_dtype(DType::Float16), BeanTensor::ErrorHandling::NarrowingGradConversion);
    CHECK(!lin.grad_dtype("weight").has_value());
    CHECK(lin.grad_dtype("bias") == DType::Float64);
    lin.set_grad_dtype(DType::Float16, GradNarrowing::Allow);
    CHECK(lin.grad_dtype("weight") == DType::Float16);
    CHECK(lin.grad_dtype("bias") == DType::Float16);
    CHECK_THROWS(static_cast<void>(lin.grad_dtype("nope")), BeanTensor::ErrorHandling::NameError);
}

static void test_module_widen_all_parameters() {
    auto lin = Linear(3, 2, DType::Float32, Device::CPU, true, 7);
    lin.set_grad_dtype(DType::Float64);
    CHECK(lin.grad_dtype("weight") == DType::Float64);
    CHECK(lin.grad_dtype("bias") == DType::Float64);
}

// Only one operand requires grad, and the other stays Float32 while backward
// runs in Float64. The non-grad operand must be converted into the compute
// dtype; borrowing it as-is threw MismatchedDTypes inside the composed products.
static void test_mixed_operand_dtype_backward() {
    auto x = Tensor({2}, DType::Float32, Device::CPU, true);
    x.set<float32_t>({0}, float32_t{2});
    x.set<float32_t>({1}, float32_t{4});
    auto c = twos();
    auto m = x.mul(c);
    m.backward(DType::Float64);
    CHECK(x.grad_dtype() == DType::Float64);
    CHECK(near(x.gradient().at<float64_t>({0}), 2.0));
    CHECK(near(x.gradient().at<float64_t>({1}), 2.0));

    // x is the right operand: d/dx (c / x) = -c / x^2 = -0.5 and -0.125.
    auto y = Tensor({2}, DType::Float32, Device::CPU, true);
    y.set<float32_t>({0}, float32_t{2});
    y.set<float32_t>({1}, float32_t{4});
    auto q = c.div(y);
    q.backward(DType::Float64);
    CHECK(near(y.gradient().at<float64_t>({0}), -0.5));
    CHECK(near(y.gradient().at<float64_t>({1}), -0.125));

    // x is the left operand: d/dx (x / c) = 1 / c = 0.5.
    auto z = Tensor({2}, DType::Float32, Device::CPU, true);
    z.set<float32_t>({0}, float32_t{2});
    z.set<float32_t>({1}, float32_t{4});
    auto r = z.div(c);
    r.backward(DType::Float64);
    CHECK(near(z.gradient().at<float64_t>({0}), 0.5));
    CHECK(near(z.gradient().at<float64_t>({1}), 0.5));
}

int main() {
    test_pin_before_backward();
    test_widen_existing_grad();
    test_narrowing_rejected_by_default();
    test_narrowing_allowed();
    test_same_dtype_is_noop();
    test_fresh_leaf_pins_narrow_dtype();
    test_non_leaf_throws();
    test_no_gradient_throws();
    test_narrowing_table();
    test_module_grad_dtype();
    test_module_widen_all_parameters();
    test_mixed_operand_dtype_backward();

    std::cout << "grad dtype tests: " << passes << " checks passed, " << failures << " failed\n";
    return failures == 0 ? 0 : 1;
}
