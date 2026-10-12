/*
 * HIP/ROCm GPU kernels + host wrappers for basic tensor operations.
 * CUDA mirror: Tensor/Ops/TensorOps.cu — both backends must stay behaviorally
 * identical (same rounding via explicit _rn intrinsics, same fail_check
 * semantics, same async handoff to Threading::GetCPUThreadPool()).
 */
#include <cstdint>
#include <limits>
#include <type_traits>

#include "Tensor/Tensor.h"
#include "Tensor/DTypes.h"
#include "TensorOps.h"

#include "Helpers/CPUThreading.h"

#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

template<typename Ptr> requires (!BeanTensor::DataTypes::requires_unpack_v<Ptr>)
__global__ void impli_gpu_set_static_val_unpacked(Ptr* __restrict__ ptr, const size_t end, const int32_t val) {
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = blockDim.x * gridDim.x;
    if (tid >= end) return;
    for (size_t i = tid; i < end; i += stride) {
        ptr[i] = val;
    }
}

/**
 * Force static assertion failure without compilation.
 * @brief Trigger static assertion failure.
 */
template <typename>
inline constexpr bool dependent_false_v = false;

/**
 * Implementation - set a static value into a packed type, on the GPU.
 * @brief GPU kernel, set static value into packed data.
 * @tparam Ptr Target pointer, must be a packed type.
 * @tparam Val Target value for packing, must be a packable (but unpacked) type.
 * @param ptr Target pointer.
 * @param end End index of the range to set.
 * @param value Target value to set.
 */
template<typename Ptr, typename Val>
requires (BeanTensor::DataTypes::requires_unpack_v<Ptr> && BeanTensor::DataTypes::can_be_packed_v<Val>)
__global__ void impli_gpu_set_static_val_packed(Ptr* __restrict__ ptr, const size_t end, const Val value) {
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = blockDim.x * gridDim.x;

    __shared__ Ptr shared_value;

    if (threadIdx.x == 0) {
        if constexpr (std::is_same_v<Ptr, BeanTensor::float16x2_t>) {
            shared_value = __halves2half2(value, value);
        } else if constexpr (std::is_same_v<Ptr, BeanTensor::bfloat16x2_t>) {
            shared_value = __halves2bfloat162(value, value);
        } else {
            static_assert(dependent_false_v<Ptr>, "Unsupported packed type");
        }
    }
    __syncthreads();

    for (size_t i = tid; i < end; i += stride) {
        ptr[i] = shared_value;
    }
}

namespace BeanTensor::Tensors::detail::gpu {

    // Read-only cached load, sized so it is valid for every dtype on both backends.
    // Non-trivially-copyable wrapper types (e.g. __hip_bfloat162) can't be
    // bit_cast, so they fall back to a plain const __restrict__ dereference.
    template<typename T>
    __device__ __forceinline__ T ldg_load(const T* __restrict__ p) {
        if constexpr (std::is_trivially_copyable_v<T> && sizeof(T) == 2) {
            return __builtin_bit_cast(T, __ldg(reinterpret_cast<const uint16_t*>(p)));
        } else if constexpr (std::is_trivially_copyable_v<T> && sizeof(T) == 4) {
            return __builtin_bit_cast(T, __ldg(reinterpret_cast<const uint32_t*>(p)));
        } else if constexpr (std::is_trivially_copyable_v<T> && sizeof(T) == 8) {
            return __builtin_bit_cast(T, __ldg(reinterpret_cast<const unsigned long long*>(p)));
        } else {
            return *p;
        }
    }

    template<typename T> struct lane_of { using type = T; };
    template<> struct lane_of<float16x2_t> { using type = float16_t; };
    template<> struct lane_of<bfloat16x2_t> { using type = bfloat16_t; };
    template<typename T> using lane_of_t = typename lane_of<T>::type;

    /**
     * Widen an FP16 lane to FP32, mirroring unaccel_fp16_to_fp32_tracked
     * (Intrinsics/Native/DTypeConversions.h): NaN/Inf inputs flag only under
     * STRICT; the widening itself is exact.
     * @brief FP16 -> FP32 lane widening with CPU-matching flags.
     * @param v FP16 value to convert.
     * @param flag Reference set to true on a conversion event the active method cares about.
     * @param m Active ConversionClampMethod.
     * @return FP32 value converted from the FP16 input.
     */
    __device__ __forceinline__ float dev_fp16_to_fp32(const float16_t v, bool& flag, const ConversionClampMethod m) {
        const float f = __half2float(v);
        if (m == ConversionClampMethod::STRICT && (__isnanf(f) || __isinff(f))) flag = true;
        return f;
    }

    /**
     * Widen a BF16 lane to FP32, mirroring unaccel_bf16_to_fp32
     * (Intrinsics/Native/DTypeConversions.h): exact shift, never flags.
     * @brief BF16 -> FP32 lane widening.
     * @param v BF16 value to convert.
     * @return FP32 value converted from the BF16 input.
     */
    __device__ __forceinline__ float dev_bf16_to_fp32(const bfloat16_t v) {
        return __bfloat162float(v);
    }

    /**
     * Narrow FP32 to FP16, mirroring unaccel_fp32_to_fp16_tracked
     * (Intrinsics/Native/DTypeConversions.h): NaN/Inf flag only under STRICT and
     * pass through; finite out-of-range clamps to +-FP16_MAX and flags under any
     * non-PERMISSIVE method. Hardware RNE matches the CPU path's rounding
     * (including gradual underflow) bit-for-bit.
     * @brief FP32 -> FP16 narrowing with CPU-matching clamp/flags.
     * @param f FP32 value to convert.
     * @param flag Reference set to true on a conversion event the active method cares about.
     * @param m Active ConversionClampMethod.
     * @return FP16 value converted from the FP32 input.
     */
    __device__ __forceinline__ float16_t dev_fp32_to_fp16(const float f, bool& flag, const ConversionClampMethod m) {
        if (__isnanf(f) || __isinff(f)) {
            if (m == ConversionClampMethod::STRICT) flag = true;
            return __float2half_rn(f);
        }
        if (f > DataTypes::detail::FP16_MAX) {
            if (m != ConversionClampMethod::PERMISSIVE) flag = true;
            return __float2half_rn(DataTypes::detail::FP16_MAX);
        }
        if (f < DataTypes::detail::FP16_MIN) {
            if (m != ConversionClampMethod::PERMISSIVE) flag = true;
            return __float2half_rn(DataTypes::detail::FP16_MIN);
        }
        return __float2half_rn(f);
    }

    /**
     * Narrow FP32 to BF16, mirroring unaccel_fp32_to_bf16_tracked
     * (Intrinsics/Native/DTypeConversions.h): NaN/Inf flag only under STRICT and
     * pass through; FP32-subnormal inputs flush to signed zero; everything else
     * rounds RNE (finite overflow rounds to BF16 Inf, unflagged - identical to
     * the CPU path).
     * @brief FP32 -> BF16 narrowing with CPU-matching flush/flags.
     * @param f FP32 value to convert.
     * @param flag Reference set to true on a conversion event the active method cares about.
     * @param m Active ConversionClampMethod.
     * @return BF16 value converted from the FP32 input.
     */
    __device__ __forceinline__ bfloat16_t dev_fp32_to_bf16(const float f, bool& flag, const ConversionClampMethod m) {
        const uint32_t bits = __float_as_uint(f);
        if ((bits & 0x7F800000u) == 0x7F800000u) {
            if (m == ConversionClampMethod::STRICT) flag = true;
            return __float2bfloat16(f); // RNE (CUDA parity: __float2bfloat16_rn)
        }
        if ((bits & 0x7F800000u) == 0u) {
            return __float2bfloat16(__uint_as_float(bits & 0x80000000u)); // signed zero
        }
        return __float2bfloat16(f); // RNE (CUDA parity: __float2bfloat16_rn)
    }

    /**
     * Narrow FP64 to FP32, mirroring standard2standard<float64_t, float32_t>
     * (Intrinsics/Shared/SharedIntrinsics.h): finite overflow clamps to
     * +-FLT_MAX and flags under any non-PERMISSIVE method; NaN flags
     * non-PERMISSIVE; Inf flags STRICT only.
     * @brief FP64 -> FP32 narrowing with CPU-matching clamp/flags.
     * @param v FP64 value to convert.
     * @param flag Reference set to true on a conversion event the active method cares about.
     * @param m Active ConversionClampMethod.
     * @return FP32 value converted from the FP64 input.
     */
    __device__ __forceinline__ float dev_fp64_to_fp32(const float64_t v, bool& flag, const ConversionClampMethod m) {
        if (v > DataTypes::detail::FP32_MAX_D) {
            if (m != ConversionClampMethod::PERMISSIVE) flag = true;
            return static_cast<float>(DataTypes::detail::FP32_MAX_D);
        }
        if (v < DataTypes::detail::FP32_MIN_D) {
            if (m != ConversionClampMethod::PERMISSIVE) flag = true;
            return static_cast<float>(DataTypes::detail::FP32_MIN_D);
        }
        if (__isnan(v)) {
            if (m != ConversionClampMethod::PERMISSIVE) flag = true;
        } else if (__isinf(v)) {
            if (m == ConversionClampMethod::STRICT) flag = true;
        }
        return static_cast<float>(v);
    }

    /**
     * Widen FP32 to FP64, mirroring standard2standard<float32_t, float64_t>
     * (Intrinsics/Shared/SharedIntrinsics.h): exact; NaN flags non-PERMISSIVE;
     * Inf flags STRICT only.
     * @brief FP32 -> FP64 widening with CPU-matching flags.
     * @param v FP32 value to convert.
     * @param flag Reference set to true on a conversion event the active method cares about.
     * @param m Active ConversionClampMethod.
     * @return FP64 value converted from the FP32 input.
     */
    __device__ __forceinline__ float64_t dev_fp32_to_fp64(const float32_t v, bool& flag, const ConversionClampMethod m) {
        if (__isnanf(v)) {
            if (m != ConversionClampMethod::PERMISSIVE) flag = true;
        } else if (__isinff(v)) {
            if (m == ConversionClampMethod::STRICT) flag = true;
        }
        return static_cast<float64_t>(v);
    }

    /**
     * Convert a single scalar lane between two float-kind dtypes, mirroring the
     * CPU dispatcher's scalar_convert (Intrinsics/Dispatcher.h) branch for
     * branch. Only float-kind dtypes reach the GPU kernels; integer conversion
     * is CPU-only.
     * @brief Scalar lane conversion with CPU-matching semantics.
     * @tparam Src Source lane dtype.
     * @tparam Dst Destination lane dtype.
     * @param v Value to convert.
     * @param flag Reference set to true on a conversion event the active method cares about.
     * @param m Active ConversionClampMethod.
     * @return Value converted to Dst.
     */
    template<typename Src, typename Dst>
    __device__ __forceinline__ Dst dev_scalar_convert(const Src v, bool& flag, const ConversionClampMethod m) {
        if constexpr (std::is_same_v<Src, Dst>) {
            return v;
        } else if constexpr (std::is_same_v<Src, bfloat16_t>) {
            return dev_scalar_convert<float32_t, Dst>(dev_bf16_to_fp32(v), flag, m);
        } else if constexpr (std::is_same_v<Src, float16_t>) {
            return dev_scalar_convert<float32_t, Dst>(dev_fp16_to_fp32(v, flag, m), flag, m);
        } else if constexpr (std::is_same_v<Dst, bfloat16_t>) {
            if constexpr (std::is_same_v<Src, float32_t>) {
                return dev_fp32_to_bf16(v, flag, m);
            } else {
                return dev_fp32_to_bf16(dev_fp64_to_fp32(v, flag, m), flag, m);
            }
        } else if constexpr (std::is_same_v<Dst, float16_t>) {
            if constexpr (std::is_same_v<Src, float32_t>) {
                return dev_fp32_to_fp16(v, flag, m);
            } else {
                return dev_fp32_to_fp16(dev_fp64_to_fp32(v, flag, m), flag, m);
            }
        } else if constexpr (std::is_same_v<Dst, float32_t>) {
            return dev_fp64_to_fp32(v, flag, m);
        } else {
            return dev_fp32_to_fp64(v, flag, m);
        }
    }

    template<typename Dst>
    __device__ __forceinline__ Dst pack_lanes(const lane_of_t<Dst> lo, const lane_of_t<Dst> hi) {
        if constexpr (std::is_same_v<Dst, float16x2_t>) {
            return __halves2half2(lo, hi);
        } else if constexpr (std::is_same_v<Dst, bfloat16x2_t>) {
            return __halves2bfloat162(lo, hi);
        } else {
            static_assert(dependent_false_v<Dst>, "pack_lanes requires a packed type");
        }
    }

    /**
     * Generic float-kind conversion kernel. Mirrors the CPU dispatcher's
     * element_convert/convert_range semantics exactly (Intrinsics/Dispatcher.h):
     * unpacked->packed duplicates the converted value into both lanes,
     * packed->unpacked converts the first lane, packed->packed converts both
     * lanes. Every element is written (clamped values included); conversion
     * events the active ConversionClampMethod cares about accumulate into a
     * thread-local flag and publish with a single atomicOr on exit - the
     * per-element hot loop never touches shared state, same as the CPU path.
     * @brief GPU dtype conversion kernel.
     * @tparam Src Source dtype.
     * @tparam Dst Destination dtype.
     * @param src Source buffer.
     * @param dst Destination buffer.
     * @param end Number of elements.
     * @param method Active ConversionClampMethod.
     * @param fail_check Shared flag for the whole conversion (nonzero = event occurred).
     */
    template<typename Src, typename Dst>
    __global__ void impli_gpu_convert(const Src* __restrict__ src, Dst* __restrict__ dst,
                                      const size_t end, const ConversionClampMethod method,
                                      int32_t* __restrict__ fail_check) {
        const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
        const size_t stride = blockDim.x * gridDim.x;
        bool local_flag = false;

        for (size_t i = tid; i < end; i += stride) {
            if constexpr (!DataTypes::requires_unpack_v<Src> && !DataTypes::requires_unpack_v<Dst>) {
                dst[i] = dev_scalar_convert<Src, Dst>(ldg_load(src + i), local_flag, method);
            } else if constexpr (DataTypes::requires_unpack_v<Dst> && !DataTypes::requires_unpack_v<Src>) {
                using H = lane_of_t<Dst>;
                const H h = dev_scalar_convert<Src, H>(ldg_load(src + i), local_flag, method);
                dst[i] = pack_lanes<Dst>(h, h);
            } else if constexpr (DataTypes::requires_unpack_v<Src> && !DataTypes::requires_unpack_v<Dst>) {
                using H = lane_of_t<Src>;
                const Src v = ldg_load(src + i);
                dst[i] = dev_scalar_convert<H, Dst>(v.x, local_flag, method);
            } else {
                using HS = lane_of_t<Src>;
                using HD = lane_of_t<Dst>;
                const Src v = ldg_load(src + i);
                const HD lo = dev_scalar_convert<HS, HD>(v.x, local_flag, method);
                const HD hi = dev_scalar_convert<HS, HD>(v.y, local_flag, method);
                dst[i] = pack_lanes<Dst>(lo, hi);
            }
        }
        if (local_flag) atomicOr(fail_check, 1);
    }

    template<typename Src>
    void launch_convert_dst(const DataTypes::DType new_dtype, const Src* src, void* dst, const size_t numel,
                            const size_t blocks, const size_t threads, const ConversionClampMethod method, int32_t* fail_check) {
        using DType = DataTypes::DType;
        switch (new_dtype) {
            case DType::Float16:     impli_gpu_convert<Src, float16_t><<<blocks, threads>>>(src, static_cast<float16_t*>(dst), numel, method, fail_check); break;
            case DType::Float16x2:   impli_gpu_convert<Src, float16x2_t><<<blocks, threads>>>(src, static_cast<float16x2_t*>(dst), numel, method, fail_check); break;
            case DType::BFloat16:    impli_gpu_convert<Src, bfloat16_t><<<blocks, threads>>>(src, static_cast<bfloat16_t*>(dst), numel, method, fail_check); break;
            case DType::BFloat16x2:  impli_gpu_convert<Src, bfloat16x2_t><<<blocks, threads>>>(src, static_cast<bfloat16x2_t*>(dst), numel, method, fail_check); break;
            case DType::Float32:     impli_gpu_convert<Src, float32_t><<<blocks, threads>>>(src, static_cast<float32_t*>(dst), numel, method, fail_check); break;
            case DType::Float64:     impli_gpu_convert<Src, float64_t><<<blocks, threads>>>(src, static_cast<float64_t*>(dst), numel, method, fail_check); break;
            default: throw ErrorHandling::NotImplemented();
        }
    }
}

namespace BeanTensor::Tensors::detail {
    void set_amount_gpu(Tensor& t, const int32_t amount) {
        if (t.device != Device::GPU) {
            assert(false && "Call of GPU function on CPU tensor"); std::unreachable();
        }
        t.sync();
        static auto& threadMgr = Threading::GetCPUThreadPool();
        const auto worker = [&t, amount] {
            constexpr size_t threads = 256;
            const size_t blocks = (t.numel + threads - 1) / threads;
            using DType = DataTypes::DType;
            switch (t.dtype) {
                case DType::Int8:      impli_gpu_set_static_val_unpacked<int8_t><<<blocks, threads>>>(static_cast<int8_t*>(t.data), t.numel, amount); break;
                case DType::Int16:     impli_gpu_set_static_val_unpacked<int16_t><<<blocks, threads>>>(static_cast<int16_t*>(t.data), t.numel, amount); break;
                case DType::Int32:     impli_gpu_set_static_val_unpacked<int32_t><<<blocks, threads>>>(static_cast<int32_t*>(t.data), t.numel, amount); break;
                case DType::Int64:     impli_gpu_set_static_val_unpacked<int64_t><<<blocks, threads>>>(static_cast<int64_t*>(t.data), t.numel, amount); break;
                case DType::UInt8:     impli_gpu_set_static_val_unpacked<uint8_t><<<blocks, threads>>>(static_cast<uint8_t*>(t.data), t.numel, amount); break;
                case DType::UInt16:    impli_gpu_set_static_val_unpacked<uint16_t><<<blocks, threads>>>(static_cast<uint16_t*>(t.data), t.numel, amount); break;
                case DType::UInt32:    impli_gpu_set_static_val_unpacked<uint32_t><<<blocks, threads>>>(static_cast<uint32_t*>(t.data), t.numel, amount); break;
                case DType::UInt64:    impli_gpu_set_static_val_unpacked<uint64_t><<<blocks, threads>>>(static_cast<uint64_t*>(t.data), t.numel, amount); break;
                case DType::Float16:   impli_gpu_set_static_val_unpacked<float16_t><<<blocks, threads>>>(static_cast<float16_t*>(t.data), t.numel, amount); break;
                case DType::BFloat16:  impli_gpu_set_static_val_unpacked<bfloat16_t><<<blocks, threads>>>(static_cast<bfloat16_t*>(t.data), t.numel, amount); break;
                case DType::Float32:   impli_gpu_set_static_val_unpacked<float32_t><<<blocks, threads>>>(static_cast<float32_t*>(t.data), t.numel, amount); break;
                case DType::Float64:   impli_gpu_set_static_val_unpacked<float64_t><<<blocks, threads>>>(static_cast<float64_t*>(t.data), t.numel, amount); break;
                case DType::Float16x2: {
                    const auto h = __float2half_rn(static_cast<float>(amount));
                    impli_gpu_set_static_val_packed<float16x2_t, float16_t><<<blocks, threads>>>(static_cast<float16x2_t*>(t.data), t.numel, h);
                    break;
                }
                case DType::BFloat16x2: {
                    const auto h = __float2bfloat16(static_cast<float>(amount)); // RNE (CUDA parity: __float2bfloat16_rn)
                    impli_gpu_set_static_val_packed<bfloat16x2_t, bfloat16_t><<<blocks, threads>>>(static_cast<bfloat16x2_t*>(t.data), t.numel, h);
                    break;
                }
                default: throw ErrorHandling::NotImplemented();
            }
            HIP_CHECK_ERROR(hipGetLastError());
        };
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        t.give_future(future);
    }

    void set_random_gpu(Tensor& t, const double min, const double max, const std::int64_t seed) {
        if (t.device != Device::GPU) return;
        t.sync();
        const auto worker = [&t, min, max, seed] {
            auto temp = Tensor({t.numel}, t.dtype, Device::CPU, false);
            temp.random(min, max, seed);
            temp.sync();
            move_to_gpu(temp, 0);
            temp.sync();
            // NEVER EXECUTE .SYNC WITHIN A TENSORS OWN WORKER
            t.kill_gpu_data(); // free the previous allocation
            t.data = temp.data;
            temp.data = nullptr;
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        t.give_future(future);
    }

    void move_to_gpu(Tensor& t, const uint32_t device) {
        if (t.device != Device::CPU) return;
        if (device != 0) throw ErrorHandling::NotImplemented(); // multi-GPU dropped in V5
        t.sync();
        t.kill_buffer();
        const auto worker = [&t] {
            {
                int count = 0;
                HIP_CHECK_ERROR(hipGetDeviceCount(&count));
                if (count < 1) {
                    throw std::invalid_argument("Invalid GPU device ID");
                }
                size_t free = 0, total = 0;
                HIP_CHECK_ERROR(hipMemGetInfo(&free, &total));
                if (free <= t.nbytes) throw ErrorHandling::OutOfMemory();
            }
            HIP_CHECK_ERROR(hipSetDevice(0));
            void* buffer = nullptr;
            HIP_CHECK_ERROR(hipMalloc(&buffer, t.numel * DataTypes::dtype_bytes(t.dtype)));
            HIP_CHECK_ERROR(hipMemcpy(buffer, t.data, t.nbytes, hipMemcpyHostToDevice));
            t.kill_cpu_data();
            t.data = buffer;
            t.device = Device::GPU;
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        t.give_future(future);
    }

    void move_to_cpu(Tensor& t) {
        if (t.device != Device::GPU) return;
        t.sync();
        t.kill_buffer();
        const auto worker = [&t] {
            HIP_CHECK_ERROR(hipSetDevice(0));
            void* buffer = std::aligned_alloc(64, (t.nbytes + 63) / 64 * 64);
            HIP_CHECK_ERROR(hipMemcpy(buffer, t.data, t.nbytes, hipMemcpyDeviceToHost));
            HIP_CHECK_ERROR(hipFree(t.data));
            t.data = buffer;
            t.device = Device::CPU;
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        t.give_future(future);
    }

    void convert_gpu(Tensor& t, const DataTypes::DType& new_dtype, const ConversionClampMethod method) {
        if (t.device != Device::GPU) return;
        if (t.dtype == new_dtype) return;
        t.sync();
        t.kill_buffer();
        // NOTE: new_dtype/method must be captured BY VALUE — the worker runs
        // async on the thread pool after convert_gpu returns, by which point new_dtype (a
        // reference bound to a temporary in the caller) has been destroyed.
        const auto worker = [&t, new_dtype, method] {
            const size_t target_buffer_size = t.numel * DataTypes::dtype_bytes(new_dtype);
            if (t.convert_buffer == nullptr || t.convert_buffer_size < target_buffer_size) {
                HIP_CHECK_ERROR(hipFree(t.convert_buffer));
                HIP_CHECK_ERROR(hipMalloc(&t.convert_buffer, target_buffer_size));
                t.convert_buffer_size = target_buffer_size;
            }
            constexpr size_t threads = 256;
            const size_t blocks = (t.numel + threads - 1) / threads;
            int32_t* check_err = nullptr;
            HIP_CHECK_ERROR(hipMalloc(&check_err, sizeof(int32_t)));
            constexpr int32_t zero = 0;
            HIP_CHECK_ERROR(hipMemcpy(check_err, &zero, sizeof(int32_t), hipMemcpyHostToDevice));

            using DType = DataTypes::DType;
            switch (t.dtype) {
                case DType::Float16:     gpu::launch_convert_dst<float16_t>(new_dtype, static_cast<const float16_t*>(t.data), t.convert_buffer, t.numel, blocks, threads, method, check_err); break;
                case DType::Float16x2:   gpu::launch_convert_dst<float16x2_t>(new_dtype, static_cast<const float16x2_t*>(t.data), t.convert_buffer, t.numel, blocks, threads, method, check_err); break;
                case DType::BFloat16:    gpu::launch_convert_dst<bfloat16_t>(new_dtype, static_cast<const bfloat16_t*>(t.data), t.convert_buffer, t.numel, blocks, threads, method, check_err); break;
                case DType::BFloat16x2:  gpu::launch_convert_dst<bfloat16x2_t>(new_dtype, static_cast<const bfloat16x2_t*>(t.data), t.convert_buffer, t.numel, blocks, threads, method, check_err); break;
                case DType::Float32:     gpu::launch_convert_dst<float32_t>(new_dtype, static_cast<const float32_t*>(t.data), t.convert_buffer, t.numel, blocks, threads, method, check_err); break;
                case DType::Float64:     gpu::launch_convert_dst<float64_t>(new_dtype, static_cast<const float64_t*>(t.data), t.convert_buffer, t.numel, blocks, threads, method, check_err); break;
                default: throw ErrorHandling::NotImplemented(); // integer GPU conversions unsupported (CPU path covers them)
            }
            HIP_CHECK_ERROR(hipGetLastError());

            int32_t ret = 0;
            HIP_CHECK_ERROR(hipMemcpy(&ret, check_err, sizeof(int32_t), hipMemcpyDeviceToHost));
            HIP_CHECK_ERROR(hipFree(check_err));
            if (ret != 0) {
                throw std::range_error("convert_gpu: conversion event the active ConversionClampMethod rejects (NaN/Inf/out-of-range)");
            }

            void* old_data = t.data;
            t.data = t.convert_buffer;
            t.convert_buffer = old_data;
            t.convert_buffer_size = t.nbytes;
            t.dtype = new_dtype;
            t.nbytes = target_buffer_size;
            t.owns_data = true;
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        t.give_future(future);
    }
}

#include "TensorOps.hpp"
