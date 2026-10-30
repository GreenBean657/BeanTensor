/**
 * @file Errors.h
 * @author GreenBean
 * @brief ErrorHandling exception types and GPU API error-check macros.
 * @version 0.10
 * @date 2026-10-30
 */
#pragma once

#include <stdexcept>
#include <string>

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

    class CacheWriteFailed : public std::runtime_error {
    public:
        CacheWriteFailed() = delete;
        /**
         * @brief Exception thrown when the kernel-tuning cache file cannot be written.
         * @param path Path of the cache file that could not be written.
         */
        explicit CacheWriteFailed(const std::string& path) : std::runtime_error("Failed to write kernel-tuning cache file: " + path) { }
    };
    class InvalidKernelBlockSize final : public std::runtime_error {
    public:
        InvalidKernelBlockSize() = delete;
        /**
         * @brief Exception thrown when a GPU kernel is launched with an invalid block size.
         * @param detail Description of the invalid block size.
         */
        explicit InvalidKernelBlockSize(const std::string& detail) : std::runtime_error("InvalidKernelBlockSize: " + detail) { }
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

    class GPUError : public std::runtime_error {
    public:
        GPUError() = delete;
        /**
         * @brief Base/fallback exception for the GPU API error family.
         * Derived types supply their own class-name prefix in @p message; the
         * fallback path in throw_gpu_error prefixes "GPUError: " itself.
         * @param message Fully formatted error message (including any prefix).
         */
        explicit GPUError(const std::string& message) : std::runtime_error(message) { }
    };

    class GPUOutOfMemory final : public GPUError {
    public:
        GPUOutOfMemory() = delete;
        /**
         * @brief Exception thrown when a GPU allocation request cannot be satisfied.
         * @param detail Backend error string and context.
         */
        explicit GPUOutOfMemory(const std::string& detail) : GPUError("GPUOutOfMemory: " + detail) { }
    };

    class GPUIllegalAddress final : public GPUError {
    public:
        GPUIllegalAddress() = delete;
        /**
         * @brief Exception thrown when a GPU kernel accesses an illegal address.
         * @param detail Backend error string and context.
         */
        explicit GPUIllegalAddress(const std::string& detail) : GPUError("GPUIllegalAddress: " + detail) { }
    };

    class GPULaunchFailure final : public GPUError {
    public:
        GPULaunchFailure() = delete;
        /**
         * @brief Exception thrown when a GPU kernel launch fails.
         * @param detail Backend error string and context.
         */
        explicit GPULaunchFailure(const std::string& detail) : GPUError("GPULaunchFailure: " + detail) { }
    };

    class GPUInvalidValue final : public GPUError {
    public:
        GPUInvalidValue() = delete;
        /**
         * @brief Exception thrown when a GPU API call receives an invalid argument.
         * @param detail Backend error string and context.
         */
        explicit GPUInvalidValue(const std::string& detail) : GPUError("GPUInvalidValue: " + detail) { }
    };

    class GPUNotInitialized final : public GPUError {
    public:
        GPUNotInitialized() = delete;
        /**
         * @brief Exception thrown when a GPU API call is made before the runtime is initialized.
         * @param detail Backend error string and context.
         */
        explicit GPUNotInitialized(const std::string& detail) : GPUError("GPUNotInitialized: " + detail) { }
    };

    class ThreadPoolStopped final : public std::runtime_error {
    public:
        ThreadPoolStopped() = delete;
        /**
         * @brief Exception thrown when a task is submitted to a stopped thread pool.
         * @param detail Which submit path was used.
         */
        explicit ThreadPoolStopped(const std::string& detail) : std::runtime_error("ThreadPoolStopped: " + detail) { }
    };

    class ThreadingUnsupported final : public std::runtime_error {
    public:
        ThreadingUnsupported() = delete;
        /**
         * @brief Exception thrown when the build cannot support thread-safe static initialization.
         * @param detail What the caller must change.
         */
        explicit ThreadingUnsupported(const std::string& detail) : std::runtime_error("ThreadingUnsupported: " + detail) { }
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

    class EmptyDataset : public std::invalid_argument {
    public:
        EmptyDataset() = delete;
        /**
         * @brief Exception thrown when a dataset tensor pair contains zero rows.
         * @param detail Which tensor (features, labels, or both) was empty.
         */
        explicit EmptyDataset(const std::string& detail) : std::invalid_argument("Empty dataset: " + detail) { }
    };

    class MismatchedRowCounts : public std::invalid_argument {
    public:
        MismatchedRowCounts() = delete;
        /**
         * @brief Exception thrown when feature and label tensors have different row counts.
         * @param detail Which row counts mismatched.
         */
        explicit MismatchedRowCounts(const std::string& detail) : std::invalid_argument("Mismatched dataset row counts: " + detail) { }
    };

    class InvalidBatchSize : public std::invalid_argument {
    public:
        InvalidBatchSize() = delete;
        /**
         * @brief Exception thrown when a batch size is zero or otherwise invalid.
         * @param detail What value or constraint was violated.
         */
        explicit InvalidBatchSize(const std::string& detail) : std::invalid_argument("Invalid batch size: " + detail) { }
    };

    class InvalidProbability : public std::invalid_argument {
    public:
        InvalidProbability() = delete;
        /**
         * @brief Exception thrown when a dropout (or similar) probability is outside [0, 1).
         * @param detail What value or constraint was violated.
         */
        explicit InvalidProbability(const std::string& detail) : std::invalid_argument("Invalid probability: " + detail) { }
    };

    class InvalidHyperparameter : public std::invalid_argument {
    public:
        InvalidHyperparameter() = delete;
        /**
         * @brief Exception thrown when a hyperparameter is outside its valid range.
         * @param detail What value or constraint was violated.
         */
        explicit InvalidHyperparameter(const std::string& detail) : std::invalid_argument("Invalid hyperparameter: " + detail) { }
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

    class NameError : public std::out_of_range {
    public:
        NameError() = delete;
        /**
         * @brief Exception thrown when a registry name is empty, unknown, or already taken.
         * @param detail What failed (empty name, unknown name, duplicate name, ...).
         */
        explicit NameError(const std::string& detail) : std::out_of_range("Name error: " + detail) { }
    };

    class NoGradient : public std::logic_error {
    public:
        /**
         * @brief Exception thrown when an operation needs an accumulated gradient
         * (Tensor::grad) but the tensor does not require grad or has none.
         */
        NoGradient() : std::logic_error("Tensor has no accumulated gradient (requires_grad is false or grad is null)") { }
    };

    class NonLeafGradDType : public std::logic_error {
    public:
        /**
         * @brief Exception thrown when a gradient dtype is set on a tensor that
         * owns a graph node: intermediate gradients are rebuilt per backward()
         * call, so only leaf gradients keep a pinned dtype.
         */
        NonLeafGradDType() : std::logic_error("Gradient dtype can only be set on a leaf tensor (a tensor with no graph node)") { }
    };

    class NarrowingGradConversion : public std::logic_error {
    public:
        /**
         * @brief Exception thrown when a gradient-dtype change would narrow an
         * existing gradient and the caller did not allow narrowing.
         */
        NarrowingGradConversion() : std::logic_error("Gradient dtype change would narrow the accumulated gradient (pass GradNarrowing::Allow to permit it)") { }
    };

    class EmptyModule : public std::logic_error {
    public:
        /**
         * @brief Exception thrown when a container module with no child modules
         * is asked to run (e.g. Sequential::forward on an empty registry).
         */
        EmptyModule() : std::logic_error("Module has no child modules to run") { }
    };

    namespace detail {
        /**
         * @brief Map a GPU API error code to the typed exception family and throw.
         * @param code Backend error code cast to int32_t.
         * @param api Stringified GPU API expression that failed.
         * @param file Source file where the check occurred.
         * @param line Source line where the check occurred.
         */
        [[noreturn]] inline void throw_gpu_error(const int32_t code, const char* api, const char* file, const int32_t line) {
            const auto detail = std::string(api) + " failed at " + file + ":" + std::to_string(line) + " - " +
#if defined(USE_HIP)
                                hipGetErrorString(static_cast<hipError_t>(code));
#elif defined(USE_CUDA)
                                cudaGetErrorString(static_cast<cudaError_t>(code));
#else
                                "code " + std::to_string(code);
#endif
#if defined(USE_HIP)
            switch (static_cast<hipError_t>(code)) {
                case hipErrorOutOfMemory: throw GPUOutOfMemory(detail);
                case hipErrorIllegalAddress: throw GPUIllegalAddress(detail);
                case hipErrorLaunchFailure: throw GPULaunchFailure(detail);
                case hipErrorInvalidValue: throw GPUInvalidValue(detail);
                case hipErrorNotInitialized: throw GPUNotInitialized(detail);
                default: throw GPUError("GPUError: " + detail);
            }
#elif defined(USE_CUDA)
            switch (static_cast<cudaError_t>(code)) {
                case cudaErrorMemoryAllocation: throw GPUOutOfMemory(detail);
                case cudaErrorIllegalAddress: throw GPUIllegalAddress(detail);
                case cudaErrorLaunchFailure: throw GPULaunchFailure(detail);
                case cudaErrorInvalidValue: throw GPUInvalidValue(detail);
                case cudaErrorInitializationError: throw GPUNotInitialized(detail);
                default: throw GPUError("GPUError: " + detail);
            }
#else
            static_cast<void>(code);
            throw GPUError("GPUError: " + detail);
#endif
        }
    }
}

#if defined(USE_HIP)
#define HIP_CHECK_ERROR(expr) \
    do { \
        const auto bt_gpu_err = (expr); \
        if (bt_gpu_err != hipSuccess) { \
            BeanTensor::ErrorHandling::detail::throw_gpu_error(static_cast<int32_t>(bt_gpu_err), #expr, __FILE__, __LINE__); \
        } \
    } while (0)
#endif

#if defined(USE_CUDA)
#define CUDA_CHECK_ERROR(expr) \
    do { \
        const auto bt_gpu_err = (expr); \
        if (bt_gpu_err != cudaSuccess) { \
            BeanTensor::ErrorHandling::detail::throw_gpu_error(static_cast<int32_t>(bt_gpu_err), #expr, __FILE__, __LINE__); \
        } \
    } while (0)
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
