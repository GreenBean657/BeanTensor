#include <iostream>
#include <chrono>
#include <random>
#include <cstring>
#include <immintrin.h>

#include "Tensor/Tensor.h"
/*
template<typename Fn>
double run_bench_tops(Fn fn, const int iterations, const size_t N, const size_t ops) {
    for (int i = 0; i < 10; i++) fn();  // warmup

    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) fn();
    const auto t1 = std::chrono::high_resolution_clock::now();

    const double elapsed   = std::chrono::duration<double>(t1 - t0).count();
    const double total_ops = static_cast<double>(N) * iterations * static_cast<double>(ops);
    const double tops      = total_ops / elapsed / 1e12;
    return tops;
}
using namespace BeanTensor;
#include <cstdio>

int main() {
    constexpr size_t N = 2048 * 2048 * 16;
    auto b = BeanTensor::Tensors::Tensor({N}, Casting::DType::Float32,
                                          Tensors::Device::CPU, false);
    b.set_one();
    std::cout << b.contents_to_string() << std::endl;

    const auto tops = run_bench_tops([&]() {
        b.convert_dtype(Casting::DType::BFloat16);
        b.convert_dtype(Casting::DType::Float32);
    }, 100, N, 2);
    std::cout << "Done, FP16:" << std::endl;
    std::cout << b.contents_to_string() << std::endl;
    std::cout << tops << " TFLOPS" << std::endl;
}
*/
using namespace BeanTensor;
int main() {
    auto b = Tensors::Tensor({2, 3}, Casting::DType::Float32, Tensors::Device::CPU, false);
    b.random(0, 2, 42);
    std::cout << "Original on CPU:" << std::endl;
    std::cout << b.contents_to_string() << std::endl;
    b.to(Tensors::Device::GPU);
    b.sync();
    std::cout << "Back to CPU:" << std::endl;
    b.to(Tensors::Device::CPU);
    std::cout << b.contents_to_string() << std::endl;
}