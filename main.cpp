#include <iostream>
#include <chrono>
#include <random>
#include <cstring>
#include <immintrin.h>

#include "Tensor/Dtypes.h"
#include "Tensor/Dtypes.h"
#include "Tensor/tensor.h"

template<typename Fn>
double run_bench_tops(Fn fn, int iterations, size_t N, size_t ops) {
    for (int i = 0; i < 10; i++) fn();  // warmup

    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) fn();
    const auto t1 = std::chrono::high_resolution_clock::now();

    const double elapsed   = std::chrono::duration<double>(t1 - t0).count();
    const double total_ops = static_cast<double>(N) * iterations * static_cast<double>(ops);
    const double tops      = total_ops / elapsed / 1e12;
    return tops;
}

int main() {
    size_t n[2] = {2, 2};
    size_t ndim     = 2;


    const Tensors::Tensor b = Tensors::make_tensor(std::vector<size_t>({2, 2}), Casting::DType::UInt32, Tensors::Device::CPU, false);
    Tensors::set_random(b, 1, 2, 20);
    Tensors::Tensor a = Tensors::make_tensor({2048, 2048, 16}, Casting::DType::Float32, Tensors::Device::CPU, false);
    set_random(a, 0.0, 1.0, 42);
    const auto tops = run_bench_tops([&]() {
        Tensors::convert_dtype(a, Casting::DType::BFloat16);
        Tensors::convert_dtype(a, Casting::DType::Float32);
    }, 1000, 4, 2);
    std::cout << "Done" << std::endl;
    std::cout << tops << " TFLOPS" << std::endl;
}
 
