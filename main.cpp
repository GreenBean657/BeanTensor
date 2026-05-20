#include <iostream>
#include <chrono>
#include <random>
#include <cstring>
#include <immintrin.h>

#include "Tensor/Dtypes.h"
#include "Tensor/Dtypes.h"
#include "Tensor/tensor.h"

template<typename Fn>
double run_bench_tops(Fn fn, int iterations, size_t N) {
    for (int i = 0; i < 10; i++) fn();  // warmup

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) fn();
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed   = std::chrono::duration<double>(t1 - t0).count();
    double total_ops = static_cast<double>(N) * iterations * 2;  // 2 casts per iteration
    double tops      = total_ops / elapsed / 1e12;
    return tops;
}

int main() {
    size_t shape[3] = {1024, 1024, 1024};
    size_t ndim     = 3;

    size_t total = 1;
    for (int i = 0; i < ndim; i++) total *= shape[i];

    Tensors::Tensor b = Tensors::make_tensor(shape, ndim, Casting::DType::Float32, Tensors::Device::CPU, false);
    Tensors::set_zero(b);

    double tops1 = run_bench_tops([&]() {
        Tensors::convert_dtype(b, Casting::DType::BFloat16);
        Tensors::convert_dtype(b, Casting::DType::Float32);
    }, 100, total);
    double tops2 = run_bench_tops([&]() {
    Tensors::convert_dtype(b, Casting::DType::BFloat16);
    Tensors::convert_dtype(b, Casting::DType::Float32);
}, 100, total);
    double tops3 = run_bench_tops([&]() {
    Tensors::convert_dtype(b, Casting::DType::BFloat16);
    Tensors::convert_dtype(b, Casting::DType::Float32);
}, 100, total);


    std::cout << tops1 << "1 TOPS\n";
    std::cout << tops2 << "2 TOPS\n";
    std::cout << tops3 << "3 TOPS\n";
}
 
