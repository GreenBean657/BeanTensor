/**
 * @file AvgPool2dCUDA.cu
 * @author GreenBean
 * @brief CUDA backend for AvgPool2d forward/backward.
 * Divergent twin of Math/Spatial/AvgPool2d/AvgPool2dHIP.cpp.
 * @version 0.1
 * @date 2026-11-10
 */
#if defined(USE_CUDA)
#include <climits>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include <cuda_runtime.h>

#include "Math/Spatial/AvgPool2d/AvgPool2d.h"
#include "Math/Spatial/SpatialCudaShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    using namespace spatial_gpu;

    namespace {
        // GPU backward only accumulates into float32/float64 buffers; keep only
        // the overloads that are actually reached.
        __device__ __forceinline__ void dev_atomic_add(float32_t* ptr, float32_t val) { atomicAdd(ptr, val); }
        __device__ __forceinline__ void dev_atomic_add(float64_t* ptr, float64_t val) { atomicAdd(ptr, val); }

        /**
         * Device-safe integer limits. std::numeric_limits members are constexpr
         * __host__ only, so we use compile-time-known literals per type.
         */
        template<typename T>
        __device__ __forceinline__ float64_t integer_min_f64() {
            if constexpr (std::is_same_v<T, int8_t>) {
                return float64_t{-128.0};
            } else if constexpr (std::is_same_v<T, uint8_t>) {
                return float64_t{0.0};
            } else if constexpr (std::is_same_v<T, int16_t>) {
                return float64_t{-32768.0};
            } else if constexpr (std::is_same_v<T, uint16_t>) {
                return float64_t{0.0};
            } else if constexpr (std::is_same_v<T, int32_t>) {
                return float64_t{-2147483648.0};
            } else if constexpr (std::is_same_v<T, uint32_t>) {
                return float64_t{0.0};
            } else if constexpr (std::is_same_v<T, int64_t>) {
                return float64_t{-9223372036854775808.0};
            } else {
                return float64_t{0.0};
            }
        }

        template<typename T>
        __device__ __forceinline__ float64_t integer_max_f64() {
            if constexpr (std::is_same_v<T, int8_t>) {
                return float64_t{127.0};
            } else if constexpr (std::is_same_v<T, uint8_t>) {
                return float64_t{255.0};
            } else if constexpr (std::is_same_v<T, int16_t>) {
                return float64_t{32767.0};
            } else if constexpr (std::is_same_v<T, uint16_t>) {
                return float64_t{65535.0};
            } else if constexpr (std::is_same_v<T, int32_t>) {
                return float64_t{2147483647.0};
            } else if constexpr (std::is_same_v<T, uint32_t>) {
                return float64_t{4294967295.0};
            } else if constexpr (std::is_same_v<T, int64_t>) {
                return float64_t{9223372036854775807.0};
            } else {
                return float64_t{18446744073709551615.0};
            }
        }

        template<typename T>
        __device__ __forceinline__ T integer_min() {
            if constexpr (std::is_same_v<T, int8_t>) {
                return INT8_MIN;
            } else if constexpr (std::is_same_v<T, uint8_t>) {
                return uint8_t{0};
            } else if constexpr (std::is_same_v<T, int16_t>) {
                return INT16_MIN;
            } else if constexpr (std::is_same_v<T, uint16_t>) {
                return uint16_t{0};
            } else if constexpr (std::is_same_v<T, int32_t>) {
                return INT32_MIN;
            } else if constexpr (std::is_same_v<T, uint32_t>) {
                return uint32_t{0};
            } else if constexpr (std::is_same_v<T, int64_t>) {
                return INT64_MIN;
            } else {
                return uint64_t{0};
            }
        }

        template<typename T>
        __device__ __forceinline__ T integer_max() {
            if constexpr (std::is_same_v<T, int8_t>) {
                return INT8_MAX;
            } else if constexpr (std::is_same_v<T, uint8_t>) {
                return uint8_t{255};
            } else if constexpr (std::is_same_v<T, int16_t>) {
                return INT16_MAX;
            } else if constexpr (std::is_same_v<T, uint16_t>) {
                return uint16_t{65535};
            } else if constexpr (std::is_same_v<T, int32_t>) {
                return INT32_MAX;
            } else if constexpr (std::is_same_v<T, uint32_t>) {
                return uint32_t{4294967295U};
            } else if constexpr (std::is_same_v<T, int64_t>) {
                return INT64_MAX;
            } else {
                return uint64_t{18446744073709551615ULL};
            }
        }

        /**
         * Narrow an fp64 mean to an integer dtype on the device. Mirrors the CPU
         * narrow_mean_int helper: NaN -> 0, clamp to range, truncate toward zero.
         * @brief Device-side fp64 mean -> integer narrowing.
         */
        template<typename T>
        __device__ __forceinline__ T narrow_mean_int(const float64_t v) {
            const auto lo = integer_min_f64<T>();
            const auto hi = integer_max_f64<T>();
            if (__builtin_isnan(v)) {
                return T{0};
            }
            if (v <= lo) {
                return integer_min<T>();
            }
            if (v >= hi) {
                return integer_max<T>();
            }
            return static_cast<T>(v);
        }

        // --------------------------- forward kernels ---------------------------

        /**
         * Per-output-position AvgPool2d forward kernel. One thread per output
         * element; grid-stride loop covers any remainder. Computes the arithmetic
         * mean of the kernel×kernel window.
         * @brief CUDA AvgPool2d forward kernel (scalar dtypes).
         */
        template<typename T>
        __global__ void avg_pool2d_kernel(const T* __restrict__ in, T* __restrict__ out,
                                          const size_t N, const size_t C,
                                          const size_t H, const size_t W,
                                          const size_t Hout, const size_t Wout,
                                          const size_t kernel, const size_t stride) {
            const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t count = N * C * Hout * Wout;
            const size_t stride_grid = blockDim.x * gridDim.x;
            for (size_t idx = tid; idx < count; idx += stride_grid) {
                const auto ow = idx % Wout;
                const auto oh = (idx / Wout) % Hout;
                const auto c = (idx / (Wout * Hout)) % C;
                const auto n = idx / (Wout * Hout * C);
                const auto h0 = oh * stride;
                const auto w0 = ow * stride;
                if constexpr (std::is_floating_point_v<T>) {
                    auto total = T{0};
                    for (auto kh = size_t{0}; kh < kernel; ++kh) {
                        for (auto kw = size_t{0}; kw < kernel; ++kw) {
                            total += in[((n * C + c) * H + h0 + kh) * W + w0 + kw];
                        }
                    }
                    out[((n * C + c) * Hout + oh) * Wout + ow] = total / static_cast<T>(kernel * kernel);
                } else {
                    auto total = float64_t{0.0};
                    for (auto kh = size_t{0}; kh < kernel; ++kh) {
                        for (auto kw = size_t{0}; kw < kernel; ++kw) {
                            total += static_cast<float64_t>(in[((n * C + c) * H + h0 + kh) * W + w0 + kw]);
                        }
                    }
                    out[((n * C + c) * Hout + oh) * Wout + ow] = narrow_mean_int<T>(total / static_cast<float64_t>(kernel * kernel));
                }
            }
        }

        /**
         * @brief Host-side AvgPool2d forward launch for one scalar dtype.
         */
        template<typename T>
        __host__ void run_avg_pool2d(const T* in, T* out,
                                     const size_t N, const size_t C,
                                     const size_t H, const size_t W,
                                     const size_t Hout, const size_t Wout,
                                     const size_t kernel, const size_t stride) {
            const auto count = N * C * Hout * Wout;
            if (count > 0) {
                avg_pool2d_kernel<T><<<grid_for(count), BT_GPU_THREADS>>>(in, out, N, C, H, W, Hout, Wout, kernel, stride);
            }
        }

        // --------------------------- backward kernels ---------------------------

        /**
         * Per-output-position AvgPool2d backward kernel. Adds dy/(kernel²) to
         * every element of the source window using atomicAdd.
         * @brief CUDA AvgPool2d backward kernel (scalar float dtypes).
         */
        template<typename T>
        __global__ void avg_pool2d_g_kernel(const T* __restrict__ dy, T* __restrict__ dx,
                                            const size_t N, const size_t C,
                                            const size_t H, const size_t W,
                                            const size_t Hout, const size_t Wout,
                                            const size_t kernel, const size_t stride) {
            const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t count = N * C * Hout * Wout;
            const size_t stride_grid = blockDim.x * gridDim.x;
            const auto scale = T{1} / static_cast<T>(kernel * kernel);
            for (size_t idx = tid; idx < count; idx += stride_grid) {
                const auto ow = idx % Wout;
                const auto oh = (idx / Wout) % Hout;
                const auto c = (idx / (Wout * Hout)) % C;
                const auto n = idx / (Wout * Hout * C);
                const auto h0 = oh * stride;
                const auto w0 = ow * stride;
                const auto inc = dy[((n * C + c) * Hout + oh) * Wout + ow] * scale;
                for (auto kh = size_t{0}; kh < kernel; ++kh) {
                    for (auto kw = size_t{0}; kw < kernel; ++kw) {
                        const auto in_idx = ((n * C + c) * H + h0 + kh) * W + w0 + kw;
                        dev_atomic_add(&dx[in_idx], inc);
                    }
                }
            }
        }

        /**
         * @brief Host-side AvgPool2d backward launch for one scalar float dtype.
         */
        template<typename T>
        __host__ void run_avg_pool2d_g(const T* dy, T* dx,
                                       const size_t N, const size_t C,
                                       const size_t H, const size_t W,
                                       const size_t Hout, const size_t Wout,
                                       const size_t kernel, const size_t stride) {
            const auto count = N * C * Hout * Wout;
            if (count > 0) {
                avg_pool2d_g_kernel<T><<<grid_for(count), BT_GPU_THREADS>>>(dy, dx, N, C, H, W, Hout, Wout, kernel, stride);
            }
        }
    }

    /**
     * @brief GPU AvgPool2d forward.
     */
    __host__ void avg_pool2d_gpu(const Tensor& t, Tensor& out, const uint32_t kernel, const uint32_t stride) {
        if (t.device != Device::GPU || out.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(t.device));
        }
        const auto* const t_data = t.data;
        auto* const out_data = out.data;
        const auto t_offset = t.offset;
        const auto out_offset = out.offset;
        const auto dtype = t.dtype;
        const auto N = t.shape[0];
        const auto C = t.shape[1];
        const auto H = t.shape[2];
        const auto W = t.shape[3];
        const auto Hout = out.shape[2];
        const auto Wout = out.shape[3];
        const auto worker = [=] {
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: run_avg_pool2d(static_cast<const float32_t*>(t_data) + t_offset, static_cast<float32_t*>(out_data) + out_offset, N, C, H, W, Hout, Wout, kernel, stride); break;
                case DType::Float64: run_avg_pool2d(static_cast<const float64_t*>(t_data) + t_offset, static_cast<float64_t*>(out_data) + out_offset, N, C, H, W, Hout, Wout, kernel, stride); break;
                case DType::Int8: run_avg_pool2d(static_cast<const int8_t*>(t_data) + t_offset, static_cast<int8_t*>(out_data) + out_offset, N, C, H, W, Hout, Wout, kernel, stride); break;
                case DType::UInt8: run_avg_pool2d(static_cast<const uint8_t*>(t_data) + t_offset, static_cast<uint8_t*>(out_data) + out_offset, N, C, H, W, Hout, Wout, kernel, stride); break;
                case DType::Int16: run_avg_pool2d(static_cast<const int16_t*>(t_data) + t_offset, static_cast<int16_t*>(out_data) + out_offset, N, C, H, W, Hout, Wout, kernel, stride); break;
                case DType::UInt16: run_avg_pool2d(static_cast<const uint16_t*>(t_data) + t_offset, static_cast<uint16_t*>(out_data) + out_offset, N, C, H, W, Hout, Wout, kernel, stride); break;
                case DType::Int32: run_avg_pool2d(static_cast<const int32_t*>(t_data) + t_offset, static_cast<int32_t*>(out_data) + out_offset, N, C, H, W, Hout, Wout, kernel, stride); break;
                case DType::UInt32: run_avg_pool2d(static_cast<const uint32_t*>(t_data) + t_offset, static_cast<uint32_t*>(out_data) + out_offset, N, C, H, W, Hout, Wout, kernel, stride); break;
                case DType::Int64: run_avg_pool2d(static_cast<const int64_t*>(t_data) + t_offset, static_cast<int64_t*>(out_data) + out_offset, N, C, H, W, Hout, Wout, kernel, stride); break;
                case DType::UInt64: run_avg_pool2d(static_cast<const uint64_t*>(t_data) + t_offset, static_cast<uint64_t*>(out_data) + out_offset, N, C, H, W, Hout, Wout, kernel, stride); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
            CUDA_CHECK_ERROR(cudaStreamSynchronize(nullptr));
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        t.give_future(future);
        out.give_future(future);
    }

    /**
     * @brief GPU AvgPool2d backward.
     */
    __host__ void avg_pool2d_g_gpu(const Tensor& dy, Tensor& dx, const uint32_t kernel, const uint32_t stride) {
        if (dy.device != Device::GPU || dx.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(dy.device));
        }
        const auto* const dy_data = dy.data;
        auto* const dx_data = dx.data;
        const auto dy_offset = dy.offset;
        const auto dx_offset = dx.offset;
        const auto dtype = dy.dtype;
        const auto N = dx.shape[0];
        const auto C = dx.shape[1];
        const auto H = dx.shape[2];
        const auto W = dx.shape[3];
        const auto Hout = dy.shape[2];
        const auto Wout = dy.shape[3];
        const auto dx_numel = dx.numel;
        const auto worker = [=] {
            // dx must be zeroed before the atomic adds.
            CUDA_CHECK_ERROR(cudaMemset(static_cast<uint8_t*>(dx_data) + dx_offset * DataTypes::dtype_bytes(dtype), 0, dx_numel * DataTypes::dtype_bytes(dtype)));
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: run_avg_pool2d_g(static_cast<const float32_t*>(dy_data) + dy_offset, static_cast<float32_t*>(dx_data) + dx_offset, N, C, H, W, Hout, Wout, kernel, stride); break;
                case DType::Float64: run_avg_pool2d_g(static_cast<const float64_t*>(dy_data) + dy_offset, static_cast<float64_t*>(dx_data) + dx_offset, N, C, H, W, Hout, Wout, kernel, stride); break;
                // GPU backward keeps full-precision accumulation only. CUDA 12.9's
                // headers expose atomicAdd for __half, __half2, and __nv_bfloat16
                // at sm_70/sm_75, but this file does not include <cuda_fp16.h> /
                // <cuda_bf16.h> and the half backward path has not been verified;
                // extending it is left for future work. CPU backward still supports half.
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
            CUDA_CHECK_ERROR(cudaStreamSynchronize(nullptr));
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        dy.give_future(future);
        dx.give_future(future);
    }
}
#endif
