/**
 * @file tensor_dtype_guard_tests.cpp
 * @author GreenBean
 * @brief Tensor::set / Tensor::at dtype guard tests - framework-free assertion suite.
 * Covers the DTypeOf mapping (compile-time rejection of unmapped types), exact
 * round trips for every mapped element type, same-size and different-size
 * mismatches (rejected with MismatchedDTypes, nothing written), and the
 * untouched contents after a rejected write.
 * @version 0.1
 * @date 2026-10-23
 */
#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>

#include "Helpers/Errors.h"
#include "Tensor/Tensor.h"

using BeanTensor::Tensors::Tensor;
using BeanTensor::DataTypes::DType;
using BeanTensor::DataTypes::DTypeMapped;
using BeanTensor::DataTypes::dtype_bytes;
using BeanTensor::DataTypes::dtype_of_v;
using BeanTensor::float16_t;
using BeanTensor::float16x2_t;
using BeanTensor::bfloat16_t;
using BeanTensor::bfloat16x2_t;
using BeanTensor::float32_t;
using BeanTensor::float64_t;
using BeanTensor::int4x2_t;
using MismatchedDTypes = BeanTensor::ErrorHandling::MismatchedDTypes;

static size_t failures = 0;
#define CHECK(...) do { \
    if (!(__VA_ARGS__)) { ++failures; std::cerr << "FAIL: " << #__VA_ARGS__ << " @ line " << __LINE__ << "\n"; } \
} while (0)

// Compile-time gate: set<T> and at<T> must not be viable for unmapped types.
template <class T>
concept CanSetElement = requires(Tensor& t, const T& v) {
    t.template set<T>({size_t{0}}, v);
};

template <class T>
concept CanAtElement = requires(Tensor& t) {
    t.template at<T>({size_t{0}});
};

static_assert(DTypeMapped<float16_t>);
static_assert(DTypeMapped<bfloat16_t>);
static_assert(DTypeMapped<float16x2_t>);
static_assert(DTypeMapped<bfloat16x2_t>);
static_assert(DTypeMapped<float32_t>);
static_assert(DTypeMapped<float64_t>);
static_assert(DTypeMapped<int64_t>);
static_assert(DTypeMapped<int32_t>);
static_assert(DTypeMapped<int16_t>);
static_assert(DTypeMapped<int8_t>);
static_assert(DTypeMapped<uint64_t>);
static_assert(DTypeMapped<uint32_t>);
static_assert(DTypeMapped<uint16_t>);
static_assert(DTypeMapped<uint8_t>);
static_assert(DTypeMapped<int4x2_t>);
static_assert(!DTypeMapped<bool>);
static_assert(!CanSetElement<bool>);
static_assert(!CanAtElement<bool>);
static_assert(CanSetElement<float32_t>);
static_assert(CanAtElement<int4x2_t>);
static_assert(dtype_of_v<float64_t> == DType::Float64);
static_assert(dtype_of_v<int32_t> == DType::Int32);
static_assert(dtype_of_v<int4x2_t> == DType::Int4x2);

// Bit-exact equality (no float ==, which -Werror=float-equal rejects).
template <class T>
static bool same_bits(const T& a, const T& b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

// Runs f and reports whether it threw MismatchedDTypes (any other exception is a failure).
template <class F>
static bool throws_mismatch(F&& f) {
    try {
        f();
    } catch (const MismatchedDTypes&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

// Every mapped type: a byte pattern written through set<T> reads back bit-exact through at<T>,
// and the stride is sizeof(T) == dtype_bytes(DType) (checked per type).
template <class T>
static void roundtrip_type(const DType dt) {
    CHECK(dtype_bytes(dtype_of_v<T>) == sizeof(T));
    CHECK(dtype_of_v<T> == dt);

    auto t = Tensor({3}, dt);
    auto a = T{};
    auto b = T{};
    std::memset(&a, 0x5A, sizeof(T));
    std::memset(&b, 0xC3, sizeof(T));
    t.set<T>({0}, a);
    t.set<T>({2}, b);

    const auto ra = t.at<T>({0});
    const auto rb = t.at<T>({2});
    CHECK(std::memcmp(&ra, &a, sizeof(T)) == 0);
    CHECK(std::memcmp(&rb, &b, sizeof(T)) == 0);
}

static void test_roundtrip_every_mapped_type() {
    roundtrip_type<float16_t>(DType::Float16);
    roundtrip_type<float16x2_t>(DType::Float16x2);
    roundtrip_type<bfloat16_t>(DType::BFloat16);
    roundtrip_type<bfloat16x2_t>(DType::BFloat16x2);
    roundtrip_type<float32_t>(DType::Float32);
    roundtrip_type<float64_t>(DType::Float64);
    roundtrip_type<int64_t>(DType::Int64);
    roundtrip_type<int32_t>(DType::Int32);
    roundtrip_type<int16_t>(DType::Int16);
    roundtrip_type<int8_t>(DType::Int8);
    roundtrip_type<uint64_t>(DType::UInt64);
    roundtrip_type<uint32_t>(DType::UInt32);
    roundtrip_type<uint16_t>(DType::UInt16);
    roundtrip_type<uint8_t>(DType::UInt8);
    roundtrip_type<int4x2_t>(DType::Int4x2);
}

// Matching dtype, values that would expose any stride or rounding error.
static void test_matching_values_exact() {
    auto f64 = Tensor({4}, DType::Float64);
    for (size_t i = 0; i < 4; ++i) {
        f64.set<float64_t>({i}, float64_t{0.1} * static_cast<float64_t>(i) + float64_t{1e-12});
    }
    for (size_t i = 0; i < 4; ++i) {
        CHECK(same_bits(f64.at<float64_t>({i}), float64_t{0.1} * static_cast<float64_t>(i) + float64_t{1e-12}));
    }

    auto i32 = Tensor({3}, DType::Int32);
    i32.set<int32_t>({0}, std::numeric_limits<int32_t>::min());
    i32.set<int32_t>({1}, 0);
    i32.set<int32_t>({2}, std::numeric_limits<int32_t>::max());
    CHECK(i32.at<int32_t>({0}) == std::numeric_limits<int32_t>::min());
    CHECK(i32.at<int32_t>({1}) == 0);
    CHECK(i32.at<int32_t>({2}) == std::numeric_limits<int32_t>::max());

    auto i64 = Tensor({2}, DType::Int64);
    i64.set<int64_t>({1}, std::numeric_limits<int64_t>::min());
    CHECK(i64.at<int64_t>({1}) == std::numeric_limits<int64_t>::min());
}

// Same byte size, different type: the case the size-only check would miss.
static void test_same_size_mismatch_rejected() {
    auto i32 = Tensor({2}, DType::Int32);
    i32.set<int32_t>({0}, 41);
    CHECK(throws_mismatch([&] { static_cast<void>(i32.at<float32_t>({0})); }));
    CHECK(throws_mismatch([&] { i32.set<float32_t>({0}, float32_t{1}); }));
    CHECK(i32.at<int32_t>({0}) == 41);

    auto f32 = Tensor({2}, DType::Float32);
    CHECK(throws_mismatch([&] { f32.set<int32_t>({1}, int32_t{7}); }));
    CHECK(throws_mismatch([&] { static_cast<void>(f32.at<int32_t>({1})); }));

    auto f64 = Tensor({2}, DType::Float64);
    CHECK(throws_mismatch([&] { f64.set<int64_t>({0}, int64_t{5}); }));
    CHECK(throws_mismatch([&] { static_cast<void>(f64.at<int64_t>({0})); }));
    CHECK(throws_mismatch([&] { f64.set<uint64_t>({0}, uint64_t{5}); }));

    auto i64 = Tensor({2}, DType::Int64);
    CHECK(throws_mismatch([&] { i64.set<float64_t>({0}, float64_t{1}); }));

    auto u16 = Tensor({2}, DType::UInt16);
    CHECK(throws_mismatch([&] { u16.set<float16_t>({0}, float16_t{}); }));
}

// Different byte size: the stride would be wrong, so these must be refused too.
static void test_different_size_mismatch_rejected() {
    auto f64 = Tensor({4}, DType::Float64);
    for (size_t i = 0; i < 4; ++i) {
        f64.set<float64_t>({i}, float64_t{7});
    }
    CHECK(throws_mismatch([&] { f64.set<float32_t>({1}, float32_t{9}); }));
    CHECK(throws_mismatch([&] { static_cast<void>(f64.at<float32_t>({1})); }));

    auto f32 = Tensor({4}, DType::Float32);
    CHECK(throws_mismatch([&] { static_cast<void>(f32.at<float64_t>({0})); }));

    auto i8 = Tensor({4}, DType::Int8);
    CHECK(throws_mismatch([&] { i8.set<int64_t>({0}, int64_t{3}); }));
    CHECK(throws_mismatch([&] { static_cast<void>(i8.at<int16_t>({0})); }));

    // Rejected writes leave every element untouched.
    for (size_t i = 0; i < 4; ++i) {
        CHECK(same_bits(f64.at<float64_t>({i}), float64_t{7}));
    }
}

// A matching dtype with an out-of-range index still throws IndexOutOfRange:
// the dtype guard does not replace the bounds check.
static void test_bad_index_still_checked() {
    auto f64 = Tensor({2}, DType::Float64);
    auto threw_index = false;
    try {
        f64.set<float64_t>({5}, float64_t{1});
    } catch (const BeanTensor::ErrorHandling::IndexOutOfRange&) {
        threw_index = true;
    }
    CHECK(threw_index);
}

int main() {
    std::cout << "TensorDTypeGuardTests" << std::endl;
    test_roundtrip_every_mapped_type();
    test_matching_values_exact();
    test_same_size_mismatch_rejected();
    test_different_size_mismatch_rejected();
    test_bad_index_still_checked();
    if (failures == 0) {
        std::cout << "ALL DTYPE GUARD TESTS PASSED" << std::endl;
        return 0;
    }
    std::cout << failures << " DTYPE GUARD TEST(S) FAILED" << std::endl;
    return 1;
}
