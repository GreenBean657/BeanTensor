/**
 * @file Errors.h
 * @author GreenBean
 * @brief ErrorHandling exception types and GPU API error-check macros.
 * @version 0.1
 * @date 2026-10-18
 */
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
        NotImplemented() : std::logic_error("Function not yet implemented") { }
    };
    class GPUTaskOnCPUBuild : public std::logic_error {
    public:
        /**
         * @brief Exception thrown when a GPU task is attempted on a CPU-only build.
         */
        GPUTaskOnCPUBuild() : std::logic_error("GPU task cannot be run on CPU build") { }
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
        explicit CPUIllegalInstruction(const std::string& attempt) : std::logic_error("Attempted illegal instruction: " + attempt) { }
    };

    class DirectFunctionCallOnWrongDataSource : public std::logic_error {
    public:
        DirectFunctionCallOnWrongDataSource() = delete;
        /**
         * @brief Exception thrown when a function is called on the wrong data source (CPU vs GPU).
         * @param true_source The actual data source of the tensor (such as "CPU" or "GPU").
         * @param assumed_source The data source that the function assumed the tensor was on (such as "CPU" or "GPU").
         */
        explicit DirectFunctionCallOnWrongDataSource(const std::string& true_source = "Unknown", const std::string& assumed_source = "Unknown") : std::logic_error("Illegal access of data on device '" + true_source + "' when the data was reported to be on '" + assumed_source + "'") { }
    };

    class IndexOutOfRange : public std::out_of_range {
    public:
        /**
         * @brief Exception thrown when a tensor index has the wrong rank or is out of bounds.
         */
        IndexOutOfRange() : std::out_of_range("Tensor index out of range") { }
    };

    class UnReachableCode : public std::logic_error {
    public:
        /**
         * @brief Exception thrown when code that should be unreachable is executed.
         */
        UnReachableCode() : std::logic_error("Unreachable code.") { }
    };

    class GPULibraryError : public std::runtime_error {
    public:
        GPULibraryError() = delete;
        /**
         * @brief Exception thrown when a vendor GPU library call (cuBLAS/rocBLAS) fails.
         * @param message Library name and status text.
         */
        explicit GPULibraryError(const std::string& message) : std::runtime_error(message) { }
    };

    class MismatchedShapes : public std::invalid_argument {
    public:
        MismatchedShapes() = delete;
        /**
         * @brief Exception thrown when two tensors' shapes are incompatible for an operation.
         * @param detail What mismatched (rank, inner dimension, batch extent, ...).
         */
        explicit MismatchedShapes(const std::string& detail) : std::invalid_argument("Mismatched tensor shapes: " + detail) { }
    };

    class MismatchedDTypes : public std::invalid_argument {
    public:
        /**
         * @brief Exception thrown when two tensors' dtypes differ where an operation requires equality.
         */
        MismatchedDTypes() : std::invalid_argument("Mismatched tensor dtypes") { }
    };

    class MismatchedDevices : public std::invalid_argument {
    public:
        /**
         * @brief Exception thrown when two tensors live on different devices where an operation requires equality.
         */
        MismatchedDevices() : std::invalid_argument("Mismatched tensor devices") { }
    };

    class DimensionTooLarge : public std::out_of_range {
    public:
        DimensionTooLarge() = delete;
        /**
         * @brief Exception thrown when a tensor dimension exceeds a backend's indexing limit.
         * @param detail Which limit was exceeded.
         */
        explicit DimensionTooLarge(const std::string& detail) : std::out_of_range("Tensor dimension too large: " + detail) { }
    };

    class NoGradient : public std::logic_error {
    public:
        /**
         * @brief Exception thrown when an operation needs an accumulated gradient
         * (Tensor::grad) but the tensor does not require grad or has none.
         */
        NoGradient() : std::logic_error("Tensor has no accumulated gradient (requires_grad is false or grad is null)") { }
    };
}

#if defined(USE_HIP)
#define HIP_CHECK_ERROR(err) \
    if ((err) != hipSuccess) { \
        if ((err) == hipErrorMemoryAllocation) { \
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

// BLAS status checks. The status types/stringify functions resolve at the use
// site — the including TU must have pulled in <cublas_v2.h> / <rocblas/rocblas.h>.
#if defined(USE_HIP)
#define ROCBLAS_CHECK_ERROR(status) { \
    if ((status) != rocblas_status_success) { \
    throw BeanTensor::ErrorHandling::GPULibraryError(std::string("rocBLAS Error: ") + rocblas_status_to_string(status)); \
    } \
}
#endif

#if defined(USE_CUDA)
#define CUBLAS_CHECK_ERROR(status) { \
    if ((status) != CUBLAS_STATUS_SUCCESS) { \
    throw BeanTensor::ErrorHandling::GPULibraryError(std::string("cuBLAS Error: ") + cublasGetStatusString(status)); \
    } \
}
#endif
