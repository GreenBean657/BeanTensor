#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>

#include "Tensor/Tensor.h"
namespace {

int g_failures = 0;

void check(const bool condition, const std::string& message) {
    if (!condition) {
        ++g_failures;
        std::cerr << "[FAIL] " << message << '\n';
    }
}
    using namespace BeanTensor;
void test_make_tensor_zeroify() {
    const size_t shape[] = {2, 3, 4};
    auto t = BeanTensor::Tensors::Tensor({2, 3, 4}, Casting::DType::Float32, Tensors::Device::CPU, false);
    t.set_zero();
    t.sync();
    for (const auto vec = t.contents_to_flat_vector(); const unsigned long i : vec) {
        if (vec.at(i) == 0.0f) {
            continue;
        }
        g_failures++;
        std::cerr << "[FAIL] Vector isn't all zeros." << std::endl;
        break;
    }}
}


int main() {
    test_make_tensor_zeroify();

    if (g_failures == 0) {
        std::cout << "All tensor unit tests passed." << '\n';
        return 0;
    }

    std::cerr << g_failures << " test(s) failed." << '\n';
    return 1;
}

