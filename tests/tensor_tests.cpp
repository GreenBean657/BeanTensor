#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>

#include "Tensor/Tensor.h"

using BeanTensor::Casting::DType;
using BeanTensor::Tensors::Tensor;

namespace {

int g_failures = 0;

void check(const bool condition, const std::string& message) {
    if (!condition) {
        ++g_failures;
        std::cerr << "[FAIL] " << message << '\n';
    }
}

void test_make_tensor_sets_metadata_and_strides() {
    const size_t shape[] = {2, 3, 4};
    Tensor t = BeanTensor::Tensors::make_tensor(shape, 3, DType::Float32, Tensors::Device::CPU, false);

    check(t._owns_data, "make_tensor should mark allocated tensors as owning their memory");
    check(t._ndim == 3, "ndim should match input rank");
    check(t._shape[0] == 2 && t._shape[1] == 3 && t._shape[2] == 4, "shape should be copied");
    check(t._strides[0] == 12 && t._strides[1] == 4 && t._strides[2] == 1, "strides should be row-major");
    check(t._numel == 24, "numel should be product of shape");
    check(t._nbytes == 24 * sizeof(float), "nbytes should match dtype size times numel");

    BeanTensor::Tensors::free_tensor(&t);
}

void test_at_location_maps_multi_index_to_flat_memory() {
    const size_t shape[] = {2, 3};
    Tensor t = BeanTensor::Tensors::make_tensor(shape, 2, DType::Int32, Tensors::Device::CPU, false);
    std::memset(t._data, 0, t._nbytes);

    auto* at_12 = BeanTensor::Tensors::at_location<int32_t>(&t, {1, 2});
    *at_12 = 42;

    auto* flat = static_cast<int32_t*>(t._data);
    check(flat[5] == 42, "at_location should use tensor strides for indexing");

    BeanTensor::Tensors::free_tensor(&t);
}

void test_transpose_returns_non_contiguous_view() {
    constexpr size_t shape[] = {2, 3};
    Tensor t = BeanTensor::Tensors::make_tensor(shape, 2, DType::Float32, Tensors::Device::CPU, false);

    Tensor tr = BeanTensor::Tensors::transpose(t, 0, 1);

    check(tr._data == t._data, "transpose should return a view sharing the same data pointer");
    check(!tr._owns_data, "transpose view must not own parent memory");
    check(!tr._is_contiguous, "transpose view should be marked non-contiguous");
    check(tr._shape[0] == 3 && tr._shape[1] == 2, "transpose should swap selected shape dimensions");
    check(tr._strides[0] == 1 && tr._strides[1] == 3, "transpose should swap selected stride dimensions");

    BeanTensor::Tensors::free_tensor(&t);
}

void test_deep_copy_allocates_independent_storage() {
    const size_t shape[] = {2, 2};
    Tensor src = BeanTensor::Tensors::make_tensor(shape, 2, DType::Int32, Tensors::Device::CPU, false);

    auto* src_data = static_cast<int32_t*>(src._data);
    src_data[0] = 1;
    src_data[1] = 2;
    src_data[2] = 3;
    src_data[3] = 4;

    Tensor copy = BeanTensor::Tensors::deep_copy(src);
    auto* copy_data = static_cast<int32_t*>(copy._data);

    check(copy._owns_data, "deep_copy should produce an owning tensor");
    check(copy_data[0] == 1 && copy_data[3] == 4, "deep_copy should copy bytes from source");

    src_data[0] = 99;
    check(copy_data[0] == 1, "deep_copy result should not alias source storage");

    BeanTensor::Tensors::free_tensor(&src);
    BeanTensor::Tensors::free_tensor(&copy);
}

void test_shape_and_device_helpers() {
    const size_t shape_a[] = {2, 3};
    const size_t shape_b[] = {2, 3};
    const size_t shape_c[] = {3, 2};

    Tensor a = BeanTensor::Tensors::make_tensor(shape_a, 2, DType::Float32, Tensors::Device::CPU, false);
    Tensor b = BeanTensor::Tensors::make_tensor(shape_b, 2, DType::Float32, Tensors::Device::CPU, false);
    Tensor c = BeanTensor::Tensors::make_tensor(shape_c, 2, DType::Float32, Tensors::Device::CPU, false);

    check(BeanTensor::Tensors::is_same_shape(a, b), "is_same_shape should detect identical shapes");
    check(!BeanTensor::Tensors::is_same_shape(a, c), "is_same_shape should reject different shapes");
    check(BeanTensor::Tensors::is_same_device(a, b), "CPU tensors should be considered same device");

    const std::string shape_text = BeanTensor::Tensors::shape_to_string(a);
    check(shape_text == "{2, 3}", "shape_to_string should format dimensions in braces");

    const std::string tensor_text = BeanTensor::Tensors::to_string(a);
    check(tensor_text.find("device[CPU]") != std::string::npos, "to_string should include CPU device marker");

    BeanTensor::Tensors::free_tensor(&a);
    BeanTensor::Tensors::free_tensor(&b);
    BeanTensor::Tensors::free_tensor(&c);
}

} // namespace

int main() {
    test_make_tensor_sets_metadata_and_strides();
    test_at_location_maps_multi_index_to_flat_memory();
    test_transpose_returns_non_contiguous_view();
    test_deep_copy_allocates_independent_storage();
    test_shape_and_device_helpers();

    if (g_failures == 0) {
        std::cout << "All tensor unit tests passed." << '\n';
        return 0;
    }

    std::cerr << g_failures << " test(s) failed." << '\n';
    return 1;
}

