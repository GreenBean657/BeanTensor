#pragma once

#include <stdexcept>

#if defined(USE_HIP)
#include <hip/hip_runtime.h>
#endif

#if defined(USE_CUDA)
#include <cuda_runtime.h>
#endif

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
    class CPUIllegalInstruction : public std::logic_error {
    public:
        CPUIllegalInstruction() = delete;
        explicit CPUIllegalInstruction(const std::string& attempt) : std::logic_error("Attempted illegal instruction: " + attempt) { };
    };
    class DirectFunctionCallOnWrongDataSource : public std::logic_error {
    public:
        DirectFunctionCallOnWrongDataSource() = delete;

        explicit DirectFunctionCallOnWrongDataSource(const std::string& true_source = "Unknown", const std::string& assumed_source = "Unknown") : std::logic_error("Illegal access of data on device '" + true_source + "' when the data was reported to be on '" + assumed_source + "'") { };
    };
}

#if defined(USE_HIP)
#define HIP_CHECK_ERROR(err) \
    if (err != hipSuccess) { \
        if (err == hipErrorMemoryAllocation) { \
        throw BeanTensor::ErrorHandling::OutOfMemory(); \
        } \
        throw std::runtime_error(std::string("HIP Error: ") + hipGetErrorString(err)); \
    }
#endif

#if defined(USE_CUDA)
#define CUDA_CHECK_ERROR(err) { \
    if ((err) != cudaSuccess) { \
    if ((err) == cudaErrorMemoryAllocation) { \
    throw BeanTensor::ErrorHandling::OutOfMemory(); \
    } \
    throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err)); \
    } \
}
#endif