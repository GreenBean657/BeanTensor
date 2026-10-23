/**
 * @file EqualHIP.cpp
 * @author GreenBean
 * @brief HIP/ROCm backend for elementwise equal.
 * Divergent twin of Math/Compare/Equal/EqualCUDA.cu — the two backends are
 * written separately per project rule (no HIP-only unification), but must stay
 * behaviorally identical: same IEEE equality semantics (NaN never equal,
 * -0.0 equals 0.0), same per-lane packed compare.
 * @version 0.1.1
 * @date 2026-10-22
 */
#if defined(USE_HIP)
#include <cstdint>
#include <type_traits>

#include <hip/hip_runtime.h>

#include "Math/Compare/Equal/Equal.h"
#include "Math/Compare/CompareHipShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    using namespace compare_gpu;
    namespace {
        // --------------------------- equal kernel ---------------------------

        /**
         * Ordered-compare equality: NaN never equals anything (both ordered compares fail),
         * -0.0 equals 0.0.
         * @brief Float equality key (IEEE == semantics, NaN excluded).
         * @param a Left key.
         * @param b Right key.
         * @return T/F - the keys are equal.
         */
        __device__ __forceinline__ bool dev_float_eq(const float64_t a, const float64_t b) {
            return !(a < b) && !(b < a) && !isnan(a) && !isnan(b);
        }

        /**
         * Elementwise equality into a UInt8 (0/1) mask: floats compare on widened fp64 keys,
         * packed dtypes are equal when BOTH widened lanes are equal, int4x2 compares raw bytes,
         * integers compare directly.
         * @brief Elementwise equal kernel (all dtypes).
         * @param a Left input tensor data, numel elements.
         * @param b Right input tensor data, numel elements.
         * @param out Output UInt8 mask, numel elements.
         * @param numel Total element count of a/b/out.
         */
        template<typename T>
        __global__ void equal_kernel(const T* __restrict__ a, const T* __restrict__ b, uint8_t* __restrict__ out, const size_t numel) {
            const size_t stride = blockDim.x * gridDim.x;
            for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < numel; i += stride) {
                if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                    out[i] = dev_float_eq(static_cast<float64_t>(a[i]), static_cast<float64_t>(b[i])) ? uint8_t{1} : uint8_t{0};
                } else if constexpr (std::is_same_v<T, float16_t>) {
                    out[i] = dev_float_eq(static_cast<float64_t>(__half2float(a[i])), static_cast<float64_t>(__half2float(b[i]))) ? uint8_t{1} : uint8_t{0};
                } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                    out[i] = dev_float_eq(static_cast<float64_t>(__bfloat162float(a[i])), static_cast<float64_t>(__bfloat162float(b[i]))) ? uint8_t{1} : uint8_t{0};
                } else if constexpr (std::is_same_v<T, float16x2_t>) {
                    const auto lo = dev_float_eq(static_cast<float64_t>(__half2float(__low2half(a[i]))), static_cast<float64_t>(__half2float(__low2half(b[i]))));
                    const auto hi = dev_float_eq(static_cast<float64_t>(__half2float(__high2half(a[i]))), static_cast<float64_t>(__half2float(__high2half(b[i]))));
                    out[i] = lo && hi ? uint8_t{1} : uint8_t{0};
                } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                    const auto lo = dev_float_eq(static_cast<float64_t>(__bfloat162float(__low2bfloat16(a[i]))), static_cast<float64_t>(__bfloat162float(__low2bfloat16(b[i]))));
                    const auto hi = dev_float_eq(static_cast<float64_t>(__bfloat162float(__high2bfloat16(a[i]))), static_cast<float64_t>(__bfloat162float(__high2bfloat16(b[i]))));
                    out[i] = lo && hi ? uint8_t{1} : uint8_t{0};
                } else if constexpr (std::is_same_v<T, int4x2_t>) {
                    out[i] = __builtin_bit_cast(uint8_t, a[i]) == __builtin_bit_cast(uint8_t, b[i]) ? uint8_t{1} : uint8_t{0};
                } else {
                    out[i] = a[i] == b[i] ? uint8_t{1} : uint8_t{0};
                }
            }
        }

        /**
         * Skips the launch for empty tensors; the kernel is asynchronous, the caller syncs.
         * @brief Host-side equal launch.
         * @param a Left input tensor data, numel elements.
         * @param b Right input tensor data, numel elements.
         * @param out Output UInt8 mask, numel elements.
         * @param numel Total element count of a/b/out.
         */
        template<typename T>
        __host__ void run_equal(const T* a, const T* b, uint8_t* out, const size_t numel) {
            if (numel > 0) {
                equal_kernel<T><<<grid_for(numel), BT_GPU_THREADS>>>(a, b, out, numel);
            }
        }
    }

    /**
     * Queues one pool worker that dispatches on dtype and launches the equal kernel. The future
     * lands on all three tensors so teardown drains it.
     * @brief GPU elementwise equal: out[i] = a[i] == b[i] ? 1 : 0.
     * @param a Left input tensor, GPU resident, contiguous.
     * @param b Right input tensor, GPU resident, contiguous.
     * @param out Output UInt8 tensor, same shape/device as the inputs.
     * @note Throws DirectFunctionCallOnWrongDataSource when any tensor is not on the GPU.
     */
    __host__ void equal_gpu(const Tensor& a, const Tensor& b, const Tensor& out) {
        if (a.device != Device::GPU || b.device != Device::GPU || out.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(a.device));
        }
        const auto* const a_data = a.data;
        const auto* const b_data = b.data;
        auto* const out_data = out.data;
        const auto a_offset = a.offset;
        const auto b_offset = b.offset;
        const auto dtype = a.dtype;
        const auto numel = a.numel;
        /**
         * Dispatches on dtype to run_equal, then syncs the default stream so the future only
         * completes once the buffer writes are done.
         * @brief Pool worker for the equal launch.
         * @note Lambda subcall. Self-execution prohibited.
         */
        const auto worker = [=] {
            auto* const mask = static_cast<uint8_t*>(out_data);
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: run_equal(static_cast<const float32_t*>(a_data) + a_offset, static_cast<const float32_t*>(b_data) + b_offset, mask, numel); break;
                case DType::Float64: run_equal(static_cast<const float64_t*>(a_data) + a_offset, static_cast<const float64_t*>(b_data) + b_offset, mask, numel); break;
                case DType::Float16: run_equal(static_cast<const float16_t*>(a_data) + a_offset, static_cast<const float16_t*>(b_data) + b_offset, mask, numel); break;
                case DType::BFloat16: run_equal(static_cast<const bfloat16_t*>(a_data) + a_offset, static_cast<const bfloat16_t*>(b_data) + b_offset, mask, numel); break;
                case DType::Float16x2: run_equal(static_cast<const float16x2_t*>(a_data) + a_offset, static_cast<const float16x2_t*>(b_data) + b_offset, mask, numel); break;
                case DType::BFloat16x2: run_equal(static_cast<const bfloat16x2_t*>(a_data) + a_offset, static_cast<const bfloat16x2_t*>(b_data) + b_offset, mask, numel); break;
                case DType::Int8: run_equal(static_cast<const int8_t*>(a_data) + a_offset, static_cast<const int8_t*>(b_data) + b_offset, mask, numel); break;
                case DType::UInt8: run_equal(static_cast<const uint8_t*>(a_data) + a_offset, static_cast<const uint8_t*>(b_data) + b_offset, mask, numel); break;
                case DType::Int16: run_equal(static_cast<const int16_t*>(a_data) + a_offset, static_cast<const int16_t*>(b_data) + b_offset, mask, numel); break;
                case DType::UInt16: run_equal(static_cast<const uint16_t*>(a_data) + a_offset, static_cast<const uint16_t*>(b_data) + b_offset, mask, numel); break;
                case DType::Int32: run_equal(static_cast<const int32_t*>(a_data) + a_offset, static_cast<const int32_t*>(b_data) + b_offset, mask, numel); break;
                case DType::UInt32: run_equal(static_cast<const uint32_t*>(a_data) + a_offset, static_cast<const uint32_t*>(b_data) + b_offset, mask, numel); break;
                case DType::Int64: run_equal(static_cast<const int64_t*>(a_data) + a_offset, static_cast<const int64_t*>(b_data) + b_offset, mask, numel); break;
                case DType::UInt64: run_equal(static_cast<const uint64_t*>(a_data) + a_offset, static_cast<const uint64_t*>(b_data) + b_offset, mask, numel); break;
                case DType::Int4x2: run_equal(static_cast<const int4x2_t*>(a_data) + a_offset, static_cast<const int4x2_t*>(b_data) + b_offset, mask, numel); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
            HIP_CHECK_ERROR(hipStreamSynchronize(nullptr));
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        a.give_future(future);
        b.give_future(future);
        out.give_future(future);
    }
}
#endif
