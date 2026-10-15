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
        /**
         * @brief Exception thrown when a function is not yet implemented.
         */
        NotImplemented() : std::logic_error("Function not yet implemented") { };
    };
    class GPUTaskOnCPUBuild : public std::logic_error {
    public:
        /**
         * @brief Exception thrown when a GPU task is attempted on a CPU-only build.
         */
        GPUTaskOnCPUBuild() : std::logic_error("GPU task cannot be run on CPU build") { };
    };
    class OutOfMemory : public std::bad_alloc {
    public:
        /**
         * @brief Exception thrown when a memory allocation fails due to insufficient memory.
         */
        OutOfMemory() = default;
    };
    class CPUIllegalInstruction : public std::logic_error {
    public:
        CPUIllegalInstruction() = delete;

        /**
         * @brief Exception thrown when an illegal instruction is attempted on the CPU.
         * @param attempt Illegal instruction attempted. (Such as AVX512BF16.)
         */
        explicit CPUIllegalInstruction(const std::string& attempt) : std::logic_error("Attempted illegal instruction: " + attempt) { };
    };

    class DirectFunctionCallOnWrongDataSource : public std::logic_error {
    public:
        DirectFunctionCallOnWrongDataSource() = delete;
        /**
         * @brief Exception thrown when a function is called on the wrong data source (CPU vs GPU).
         * @param true_source The actual data source of the tensor (such as "CPU" or "GPU").
         * @param assumed_source The data source that the function assumed the tensor was on (such as "CPU" or "GPU").
         */
        explicit DirectFunctionCallOnWrongDataSource(const std::string& true_source = "Unknown", const std::string& assumed_source = "Unknown") : std::logic_error("Illegal access of data on device '" + true_source + "' when the data was reported to be on '" + assumed_source + "'") { };
    };

    class IndexOutOfRange : public std::out_of_range {
    public:
        /**
         * @brief Exception thrown when a tensor index has the wrong rank or is out of bounds.
         */
        IndexOutOfRange() : std::out_of_range("Tensor index out of range") { };
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
