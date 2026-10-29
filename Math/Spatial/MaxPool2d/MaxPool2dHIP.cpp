/**
 * @file MaxPool2dHIP.cpp
 * @author GreenBean
 * @brief HIP/ROCm backend for MaxPool2d forward/backward.
 * Divergent twin of Math/Spatial/MaxPool2d/MaxPool2dCUDA.cu.
 * @version 0.1
 * @date 2026-10-28
 */
#if defined(USE_HIP)
#include <climits>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include <hip/hip_runtime.h>

#include "Math/Spatial/MaxPool2d/MaxPool2d.h"
#include "Math/Spatial/SpatialHipShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    using namespace spatial_gpu;

    namespace {
        // Device-safe replacement for std::numeric_limits initial values.
        template<typename T>
        __device__ __forceinline__ T pool_min_value() {
            if constexpr (std::is_floating_point_v<T>) {
                T value{};
                if constexpr (sizeof(T) == 4) {
                    const auto bits = uint32_t{0xFF800000U};
                    std::memcpy(&value, &bits, sizeof(T));
                } else {
                    const auto bits = uint64_t{0xFFF0000000000000ULL};
                    std::memcpy(&value, &bits, sizeof(T));
                }
                return value;
            } else if constexpr (std::is_same_v<T, int8_t>) {
                return INT8_MIN;
            } else if constexpr (std::is_same_v<T, int16_t>) {
                return INT16_MIN;
            } else if constexpr (std::is_same_v<T, int32_t>) {
                return INT32_MIN;
            } else if constexpr (std::is_same_v<T, int64_t>) {
                return INT64_MIN;
            } else {
                return T{0};
            }
        }

        // HIP's atomicAdd has no int64_t/uint64_t overload on x86_64 Linux
        // because int64_t is 'long' while the intrinsic expects 'long long'.
        // These wrappers use the explicitly-typed overloads the runtime provides.
        __device__ __forceinline__ void dev_atomic_add(float32_t* ptr, float32_t val) { atomicAdd(ptr, val); }
        __device__ __forceinline__ void dev_atomic_add(float64_t* ptr, float64_t val) { atomicAdd(ptr, val); }
        __device__ __forceinline__ void dev_atomic_add(int32_t* ptr, int32_t val) { atomicAdd(ptr, val); }
        __device__ __forceinline__ void dev_atomic_add(uint32_t* ptr, uint32_t val) { atomicAdd(ptr, val); }
        // HIP has no signed 64-bit atomicAdd overload; unsigned 64-bit uses
        // 'unsigned long long' while uint64_t is 'unsigned long' on x86_64.
        __device__ __forceinline__ void dev_atomic_add(uint64_t* ptr, uint64_t val) {
            atomicAdd(reinterpret_cast<unsigned long long*>(ptr), static_cast<unsigned long long>(val));
        }

        // --------------------------- forward kernels ---------------------------

        /**
         * Per-output-position MaxPool2d forward kernel. One thread per output
         * element; grid-stride loop covers any remainder. First max wins; NaN
         * beats non-NaN and first NaN wins.
         * @brief HIP MaxPool2d forward kernel (scalar dtypes).
         */
        template<typename T>
        __global__ void max_pool2d_kernel(const T* __restrict__ in, T* __restrict__ out,
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
                auto best_v = pool_min_value<T>();
                auto best_h = size_t{0};
                auto best_w = size_t{0};
                for (auto kh = size_t{0}; kh < kernel; ++kh) {
                    for (auto kw = size_t{0}; kw < kernel; ++kw) {
                        const auto h = h0 + kh;
                        const auto w = w0 + kw;
                        const auto v = in[((n * C + c) * H + h) * W + w];
                        if constexpr (std::is_floating_point_v<T>) {
                            if (__builtin_isnan(v)) {
                                if (!__builtin_isnan(best_v)) {
                                    best_v = v;
                                    best_h = h;
                                    best_w = w;
                                }
                            } else if (__builtin_isnan(best_v) || v > best_v) {
                                best_v = v;
                                best_h = h;
                                best_w = w;
                            }
                        } else {
                            if (v > best_v) {
                                best_v = v;
                                best_h = h;
                                best_w = w;
                            }
                        }
                    }
                }
                out[((n * C + c) * Hout + oh) * Wout + ow] = in[((n * C + c) * H + best_h) * W + best_w];
            }
        }

        /**
         * @brief Host-side MaxPool2d forward launch for one scalar dtype.
         */
        template<typename T>
        __host__ void run_max_pool2d(const T* in, T* out,
                                     const size_t N, const size_t C,
                                     const size_t H, const size_t W,
                                     const size_t Hout, const size_t Wout,
                                     const size_t kernel, const size_t stride) {
            const auto count = N * C * Hout * Wout;
            if (count > 0) {
                hipLaunchKernelGGL(max_pool2d_kernel<T>, dim3(grid_for(count)), dim3(BT_GPU_THREADS), 0, nullptr,
                                   in, out, N, C, H, W, Hout, Wout, kernel, stride);
            }
        }

        // --------------------------- backward kernels ---------------------------

        /**
         * Per-output-position MaxPool2d backward kernel. Recomputes the first-max
         * argmax from x and atomically adds dy there.
         * @brief HIP MaxPool2d backward kernel (scalar dtypes with atomicAdd).
         */
        template<typename T>
        __global__ void max_pool2d_g_kernel(const T* __restrict__ dy, const T* __restrict__ x, T* __restrict__ dx,
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
                auto best_v = pool_min_value<T>();
                auto best_h = size_t{0};
                auto best_w = size_t{0};
                for (auto kh = size_t{0}; kh < kernel; ++kh) {
                    for (auto kw = size_t{0}; kw < kernel; ++kw) {
                        const auto h = h0 + kh;
                        const auto w = w0 + kw;
                        const auto v = x[((n * C + c) * H + h) * W + w];
                        if constexpr (std::is_floating_point_v<T>) {
                            if (__builtin_isnan(v)) {
                                if (!__builtin_isnan(best_v)) {
                                    best_v = v;
                                    best_h = h;
                                    best_w = w;
                                }
                            } else if (__builtin_isnan(best_v) || v > best_v) {
                                best_v = v;
                                best_h = h;
                                best_w = w;
                            }
                        } else {
                            if (v > best_v) {
                                best_v = v;
                                best_h = h;
                                best_w = w;
                            }
                        }
                    }
                }
                const auto out_idx = ((n * C + c) * Hout + oh) * Wout + ow;
                const auto in_idx = ((n * C + c) * H + best_h) * W + best_w;
                dev_atomic_add(&dx[in_idx], dy[out_idx]);
            }
        }

        /**
         * @brief Host-side MaxPool2d backward launch for one scalar dtype.
         */
        template<typename T>
        __host__ void run_max_pool2d_g(const T* dy, const T* x, T* dx,
                                       const size_t N, const size_t C,
                                       const size_t H, const size_t W,
                                       const size_t Hout, const size_t Wout,
                                       const size_t kernel, const size_t stride) {
            const auto count = N * C * Hout * Wout;
            if (count > 0) {
                hipLaunchKernelGGL(max_pool2d_g_kernel<T>, dim3(grid_for(count)), dim3(BT_GPU_THREADS), 0, nullptr,
                                   dy, x, dx, N, C, H, W, Hout, Wout, kernel, stride);
            }
        }
    }

    /**
     * @brief GPU MaxPool2d forward.
     */
    __host__ void max_pool2d_gpu(const Tensor& t, Tensor& out, const uint32_t kernel, const uint32_t stride) {
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
                case DType::Float32: run_max_pool2d(static_cast<const float32_t*>(t_data) + t_offset, static_cast<float32_t*>(out_data) + out_offset, N, C, H, W, Hout, Wout, kernel, stride); break;
                case DType::Float64: run_max_pool2d(static_cast<const float64_t*>(t_data) + t_offset, static_cast<float64_t*>(out_data) + out_offset, N, C, H, W, Hout, Wout, kernel, stride); break;
                case DType::Int8: run_max_pool2d(static_cast<const int8_t*>(t_data) + t_offset, static_cast<int8_t*>(out_data) + out_offset, N, C, H, W, Hout, Wout, kernel, stride); break;
                case DType::UInt8: run_max_pool2d(static_cast<const uint8_t*>(t_data) + t_offset, static_cast<uint8_t*>(out_data) + out_offset, N, C, H, W, Hout, Wout, kernel, stride); break;
                case DType::Int16: run_max_pool2d(static_cast<const int16_t*>(t_data) + t_offset, static_cast<int16_t*>(out_data) + out_offset, N, C, H, W, Hout, Wout, kernel, stride); break;
                case DType::UInt16: run_max_pool2d(static_cast<const uint16_t*>(t_data) + t_offset, static_cast<uint16_t*>(out_data) + out_offset, N, C, H, W, Hout, Wout, kernel, stride); break;
                case DType::Int32: run_max_pool2d(static_cast<const int32_t*>(t_data) + t_offset, static_cast<int32_t*>(out_data) + out_offset, N, C, H, W, Hout, Wout, kernel, stride); break;
                case DType::UInt32: run_max_pool2d(static_cast<const uint32_t*>(t_data) + t_offset, static_cast<uint32_t*>(out_data) + out_offset, N, C, H, W, Hout, Wout, kernel, stride); break;
                case DType::Int64: run_max_pool2d(static_cast<const int64_t*>(t_data) + t_offset, static_cast<int64_t*>(out_data) + out_offset, N, C, H, W, Hout, Wout, kernel, stride); break;
                case DType::UInt64: run_max_pool2d(static_cast<const uint64_t*>(t_data) + t_offset, static_cast<uint64_t*>(out_data) + out_offset, N, C, H, W, Hout, Wout, kernel, stride); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
            HIP_CHECK_ERROR(hipStreamSynchronize(nullptr));
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        t.give_future(future);
        out.give_future(future);
    }

    /**
     * @brief GPU MaxPool2d backward.
     */
    __host__ void max_pool2d_g_gpu(const Tensor& dy, const Tensor& x, Tensor& dx, const uint32_t kernel, const uint32_t stride) {
        if (dy.device != Device::GPU || x.device != Device::GPU || dx.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(x.device));
        }
        const auto* const dy_data = dy.data;
        const auto* const x_data = x.data;
        auto* const dx_data = dx.data;
        const auto dy_offset = dy.offset;
        const auto x_offset = x.offset;
        const auto dx_offset = dx.offset;
        const auto dtype = x.dtype;
        const auto N = x.shape[0];
        const auto C = x.shape[1];
        const auto H = x.shape[2];
        const auto W = x.shape[3];
        const auto Hout = dy.shape[2];
        const auto Wout = dy.shape[3];
        const auto worker = [=] {
            HIP_CHECK_ERROR(hipMemset(static_cast<uint8_t*>(dx_data) + dx_offset * DataTypes::dtype_bytes(dtype), 0, dx.numel * DataTypes::dtype_bytes(dtype)));
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: run_max_pool2d_g(static_cast<const float32_t*>(dy_data) + dy_offset, static_cast<const float32_t*>(x_data) + x_offset, static_cast<float32_t*>(dx_data) + dx_offset, N, C, H, W, Hout, Wout, kernel, stride); break;
                case DType::Float64: run_max_pool2d_g(static_cast<const float64_t*>(dy_data) + dy_offset, static_cast<const float64_t*>(x_data) + x_offset, static_cast<float64_t*>(dx_data) + dx_offset, N, C, H, W, Hout, Wout, kernel, stride); break;
                case DType::Int32: run_max_pool2d_g(static_cast<const int32_t*>(dy_data) + dy_offset, static_cast<const int32_t*>(x_data) + x_offset, static_cast<int32_t*>(dx_data) + dx_offset, N, C, H, W, Hout, Wout, kernel, stride); break;
                case DType::UInt32: run_max_pool2d_g(static_cast<const uint32_t*>(dy_data) + dy_offset, static_cast<const uint32_t*>(x_data) + x_offset, static_cast<uint32_t*>(dx_data) + dx_offset, N, C, H, W, Hout, Wout, kernel, stride); break;
                case DType::UInt64: run_max_pool2d_g(static_cast<const uint64_t*>(dy_data) + dy_offset, static_cast<const uint64_t*>(x_data) + x_offset, static_cast<uint64_t*>(dx_data) + dx_offset, N, C, H, W, Hout, Wout, kernel, stride); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
            HIP_CHECK_ERROR(hipStreamSynchronize(nullptr));
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        dy.give_future(future);
        x.give_future(future);
        dx.give_future(future);
    }
}
#endif
