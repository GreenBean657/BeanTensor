#pragma once
#include <hip/hip_runtime.h>
namespace BeanTensor::ErrorHandling {
    inline void checkError(const hipError_t err) {
        if (err != hipSuccess) {
            std::cerr << "HIP Error: " << hipGetErrorString(err) << std::endl;
        }
    }
}