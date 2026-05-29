#pragma once

#include <stdexcept>

namespace BeanTensor::ErrorHandling {
    class NotImplemented : public std::logic_error
    {
    public:
        NotImplemented() : std::logic_error("Function not yet implemented") { };
    };
    class GPUTaskOnCPUBuild : public std::logic_error {
        public:
        GPUTaskOnCPUBuild() : std::logic_error("GPU task cannot be run on CPU build") { };
    };
    class OutOfMemory : public std::bad_alloc {
        public:
        OutOfMemory() = default;
    };

#if defined(USE_HIP)
#include <hip/hip_runtime.h>
#define HIP_CHECK_ERROR(err) \
    if (err != hipSuccess) { \
        if (err == hipErrorMemoryAllocation) { \
        throw BeanTensor::ErrorHandling::OutOfMemory(); \
        } \
        throw std::runtime_error(std::string("HIP Error: ") + hipGetErrorString(err)); \
    }
#endif

#if defined(USE_CUDA)
#include <cuda_runtime.h>
#define CUDA_CHECK_ERROR(err) { \
    if ((err) != cudaSuccess) { \
    if ((err) == cudaErrorMemoryAllocation) { \
    throw BeanTensor::ErrorHandling::OutOfMemory(); \
    } \
    throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err)); \
    } \
}
#endif

}