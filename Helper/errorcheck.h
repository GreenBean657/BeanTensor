#pragma once
#include <iostream>
#include <hip/hip_runtime.h>
namespace BeanTensor::ErrorHandling {
#ifdef USE_HIP
    inline void checkError(const hipError_t err) {
        if (err != hipSuccess) {
            std::cerr << "HIP Error: " << hipGetErrorString(err) << std::endl;
        }
    }
#elif USE_CUDA
    inline void checkError(const cudaError_t err) {
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        }
    }
#else
    inline void checkError(const void err) {
        assert(false && "No GPU backend defined, but checkError was called");
    }
#endif

}