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
    /// Hip-check for errors.
    /// @param err Error to check.
#define HIP_CHECK_ERROR(err) \
    if (err != hipSuccess) { \
        if (err == hipErrorMemoryAllocation) { \
        throw ErrorHandling::OutOfMemory(); \
        } \
        throw std::runtime_error(std::string("HIP Error: ") + hipGetErrorString(err)); \
    }
}
#endif