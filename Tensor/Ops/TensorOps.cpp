/**
 * @file TensorOps.cpp
 * @author GreenBean
 * @brief HIP GPU kernels + host wrappers for basic tensor operations.
 * Mirror pair of Tensor/Ops/TensorOps.cu (CUDA); both backends must stay behaviorally identical (same rounding via
 * explicit _rn intrinsics, same fail_check semantics, same async handoff to Threading::GetCPUThreadPool()).
 * @version 0.1.1
 * @date 2026-10-18
 */
#include <cstdint>
#include <type_traits>

#include "Tensor/Tensor.h"
#include "Tensor/DTypes.h"
#include "TensorOps.h"

#include "Helpers/CPUThreading.h"

#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>


namespace BeanTensor::Tensors::detail::gpu {

    namespace {
        /**
         * Map a dtype to scalers. Used for packing validation.
         * @brief Maps a dtype to the scaler lane used for packing.
         * @tparam T The dtype to map. When unspecialized, lane type is T itself.
         */
        template<typename T> struct lane_of { using type = T; };

        /**
         * @brief lane_of specialization for float16x2_t (two float16_t lanes)
         */
        template<> struct lane_of<float16x2_t> { using type = float16_t; };

        /**
         * @brief lane_of specialization for bfloat16x2_t (two bfloat16_t lanes)
         */
        template<> struct lane_of<bfloat16x2_t> { using type = bfloat16_t; };
        /**
         * float16x2_t -> float16_t, bfloat16x2_t -> bfloat16_t, all other T -> T.
         * @brief Alias for lane_of<T>::type, used to get the scalar lane type of a given dtype.
         * @tparam T the dtype whos scalar lane is requested.
         */
        template<typename T> using lane_of_t = lane_of<T>::type;

        /**
         * Set a tensor to a static value on the GPU. This kernel is used for unpacked types.
         * @brief GPU kernel to set a tensor to a static value.
         * @tparam Ptr Source pointer type.
         * @param ptr Pointer to modify.
         * @param end End index of the source pointer.
         * @param val Value to set.
         */
        template<typename Ptr> requires (!BeanTensor::DataTypes::requires_unpack_v<Ptr>)
        __global__ void impli_gpu_set_static_val_unpacked(Ptr* __restrict__ ptr, const size_t end, const Ptr val) {
            const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t stride = blockDim.x * gridDim.x;
            if (tid >= end) return;
            for (size_t i = tid; i < end; i += stride) {
                ptr[i] = val;
            }
        }

        /**
         * @brief Always return false.
         */
        template <typename>
        inline constexpr bool dependent_false_v = false;

        /**
         * @brief GPU kernel to set a packed tensor to a static value.
         * @tparam Ptr Source pointer type (packed).
         * @tparam Val Value type (unpacked).
         * @param ptr Source pointer (value that should be packed INTO)
         * @param end End index of the source pointer.
         * @param value Value pointer (value that the kernel reads to write INTO the source pointer)
         */
        template<typename Ptr, typename Val>
        requires (BeanTensor::DataTypes::requires_unpack_v<Ptr> && BeanTensor::DataTypes::can_be_packed_v<Val>)
        __global__ void impli_gpu_set_static_val_packed(Ptr* __restrict__ ptr, const size_t end, const Val value) {
            const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t stride = blockDim.x * gridDim.x;

            __shared__ Ptr shared_value;

            if (threadIdx.x == 0) {
                if constexpr (std::is_same_v<Ptr, float16x2_t>) {
                    shared_value = __halves2half2(value, value);
                } else if constexpr (std::is_same_v<Ptr, bfloat16x2_t>) {
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

        /**
         * Loads a value through the GPU read-only cache path.
         * When the value cannot be put in cache manually, default to allowing the GPU to decide behavior.
         * @brief Load note: This function will force provided values into lgd - never write to something fed here.
         * @tparam T The type of the value to load.
         * @param p A pointer to the value to load.
         * @return The loaded value.
         */
        template<typename T>
        __device__ __forceinline__ T ldg_load(const T* __restrict__ p) {
            if constexpr (std::is_trivially_copyable_v<T> && sizeof(T) == 2) {
                return __builtin_bit_cast(T, __ldg(reinterpret_cast<const uint16_t*>(p)));
            } else if constexpr (std::is_trivially_copyable_v<T> && sizeof(T) == 4) {
                return __builtin_bit_cast(T, __ldg(reinterpret_cast<const uint32_t*>(p)));
            } else if constexpr (std::is_trivially_copyable_v<T> && sizeof(T) == 8) {
                return __builtin_bit_cast(T, __ldg(reinterpret_cast<const uint64_t*>(p)));
            } else {
                return *p;
            }
        }

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
        __device__ __forceinline__ float32_t dev_fp16_to_fp32(const float16_t v, bool& flag, const ConversionClampMethod m) {
            const float32_t f = __half2float(v);
            if (m == ConversionClampMethod::STRICT && (__isnanf(f) || __isinff(f))) flag = true;
            return f;
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
        __device__ __forceinline__ bfloat16_t dev_fp32_to_bf16(const float32_t f, bool& flag, const ConversionClampMethod m) {
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
         * Widen a BF16 lane to FP32, mirroring unaccel_bf16_to_fp32
         * (Intrinsics/Native/DTypeConversions.h): exact shift, never flags.
         * @brief BF16 -> FP32 lane widening.
         * @param v BF16 value to convert.
         * @return FP32 value converted from the BF16 input.
         */
        __device__ __forceinline__ float32_t dev_bf16_to_fp32(const bfloat16_t v) {
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
        __device__ __forceinline__ float16_t dev_fp32_to_fp16(const float32_t f, bool& flag, const ConversionClampMethod m) {
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
        __device__ __forceinline__ float32_t dev_fp64_to_fp32(const float64_t v, bool& flag, const ConversionClampMethod m) {
            if (v > DataTypes::detail::FP32_MAX_D) {
                if (m != ConversionClampMethod::PERMISSIVE) flag = true;
                return DataTypes::detail::FP32_MAX_D;
            }
            if (v < DataTypes::detail::FP32_MIN_D) {
                if (m != ConversionClampMethod::PERMISSIVE) flag = true;
                return DataTypes::detail::FP32_MIN_D;
            }
            if (__isnan(v)) {
                if (m != ConversionClampMethod::PERMISSIVE) flag = true;
            } else if (__isinf(v)) {
                if (m == ConversionClampMethod::STRICT) flag = true;
            }
            return static_cast<float32_t>(v);
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
         * Integer limits for device code. std::numeric_limits is unavailable in
         * device code (nvcc rejects it - see Tensor/DTypes.h), so bounds come
         * from <cstdint> macros and digit counts are hand-computed.
         * @brief Compile-time integer bounds/digits for the conversion kernels.
         * @tparam T Integer type to inspect.
         */
        template<typename T> struct dev_int_limits;
        template<> struct dev_int_limits<int8_t>   { static constexpr int8_t   min_v = INT8_MIN;  static constexpr int8_t   max_v = INT8_MAX;   static constexpr int32_t digits = 7;  };
        template<> struct dev_int_limits<int16_t>  { static constexpr int16_t  min_v = INT16_MIN; static constexpr int16_t  max_v = INT16_MAX;  static constexpr int32_t digits = 15; };
        template<> struct dev_int_limits<int32_t>  { static constexpr int32_t  min_v = INT32_MIN; static constexpr int32_t  max_v = INT32_MAX;  static constexpr int32_t digits = 31; };
        template<> struct dev_int_limits<int64_t>  { static constexpr int64_t  min_v = INT64_MIN; static constexpr int64_t  max_v = INT64_MAX;  static constexpr int32_t digits = 63; };
        template<> struct dev_int_limits<uint8_t>  { static constexpr uint8_t  min_v = 0;         static constexpr uint8_t  max_v = UINT8_MAX;  static constexpr int32_t digits = 8;  };
        template<> struct dev_int_limits<uint16_t> { static constexpr uint16_t min_v = 0;         static constexpr uint16_t max_v = UINT16_MAX; static constexpr int32_t digits = 16; };
        template<> struct dev_int_limits<uint32_t> { static constexpr uint32_t min_v = 0;         static constexpr uint32_t max_v = UINT32_MAX; static constexpr int32_t digits = 32; };
        template<> struct dev_int_limits<uint64_t> { static constexpr uint64_t min_v = 0;         static constexpr uint64_t max_v = UINT64_MAX; static constexpr int32_t digits = 64; };

        /**
         * Sign-safe comparisons, mirroring std::cmp_greater/std::cmp_less
         * (unusable in device code). Mixed-sign compares never convert a
         * negative value to unsigned.
         * @brief Value-preserving greater-than across integer types.
         * @tparam Src Left-hand integer type.
         * @tparam Dst Right-hand integer type.
         * @param a Left-hand value.
         * @param b Right-hand value.
         * @return True iff a compares greater than b.
         */
        template<typename Src, typename Dst>
        __device__ __forceinline__ bool dev_cmp_greater(const Src a, const Dst b) {
            if constexpr (std::is_signed_v<Src> == std::is_signed_v<Dst>) {
                return a > b;
            } else if constexpr (std::is_signed_v<Src>) {
                return a >= 0 && static_cast<std::make_unsigned_t<Src>>(a) > b;
            } else {
                return b < 0 || a > static_cast<std::make_unsigned_t<Dst>>(b);
            }
        }

        /**
         * @brief Value-preserving less-than across integer types (see dev_cmp_greater).
         */
        template<typename Src, typename Dst>
        __device__ __forceinline__ bool dev_cmp_less(const Src a, const Dst b) {
            return dev_cmp_greater(b, a);
        }

        /**
         * Convert between integer dtypes, mirroring the integer branch of
         * standard2standard (Intrinsics/Shared/SharedIntrinsics.h): out-of-range
         * values clamp to the destination bound and flag under any
         * non-PERMISSIVE method.
         * @brief Integer -> integer conversion with CPU-matching clamp/flags.
         * @tparam Src Source integer dtype.
         * @tparam Dst Destination integer dtype.
         * @param v Value to convert.
         * @param flag Reference set to true on a conversion event the active method cares about.
         * @param m Active ConversionClampMethod.
         * @return Value converted to Dst.
         */
        template<typename Src, typename Dst>
        __device__ __forceinline__ Dst dev_int_to_int(const Src v, bool& flag, const ConversionClampMethod m) {
            if (dev_cmp_greater(v, dev_int_limits<Dst>::max_v)) {
                if (m != ConversionClampMethod::PERMISSIVE) {
                    flag = true;
                }
                return dev_int_limits<Dst>::max_v;
            }
            if (dev_cmp_less(v, dev_int_limits<Dst>::min_v)) {
                if (m != ConversionClampMethod::PERMISSIVE) {
                    flag = true;
                }
                return dev_int_limits<Dst>::min_v;
            }
            return static_cast<Dst>(v);
        }

        /**
         * Extract the low signed 4-bit lane of a packed int4x2_t byte
         * (lo = bits[3:0]), sign-extended to int8_t.
         * @brief int4x2_t low-lane extraction.
         * @param v Packed byte to unpack.
         * @return Low lane sign-extended into [-8,7].
         */
        __device__ __forceinline__ int8_t int4_lane_lo(const int4x2_t v) {
            const auto nibble = static_cast<uint8_t>(v) & 0x0Fu;
            return static_cast<int8_t>(nibble > 7 ? nibble - 16 : nibble);
        }

        /**
         * Extract the high signed 4-bit lane of a packed int4x2_t byte
         * (hi = bits[7:4]), sign-extended to int8_t.
         * @brief int4x2_t high-lane extraction.
         * @param v Packed byte to unpack.
         * @return High lane sign-extended into [-8,7].
         */
        __device__ __forceinline__ int8_t int4_lane_hi(const int4x2_t v) {
            const auto nibble = static_cast<uint8_t>(v) >> 4;
            return static_cast<int8_t>(nibble > 7 ? nibble - 16 : nibble);
        }

        /**
         * Pack two int8_t lanes into one int4x2_t byte, truncating each lane to
         * its low 4 bits (& 0xF). Callers clamp into [-8,7] first.
         * @brief int4x2_t lane packing.
         * @param lo Low lane (bits[3:0]).
         * @param hi High lane (bits[7:4]).
         * @return Packed byte.
         */
        __host__ __device__ __forceinline__ int4x2_t pack_int4x2(const int8_t lo, const int8_t hi) {
            const auto lo_nib = static_cast<uint8_t>(static_cast<uint8_t>(lo) & 0x0Fu);
            const auto hi_nib = static_cast<uint8_t>(static_cast<uint8_t>(hi) & 0x0Fu);
            return static_cast<int4x2_t>(static_cast<uint8_t>(lo_nib | static_cast<uint8_t>(hi_nib << 4)));
        }

        /**
         * Clamp an int8_t lane into the signed int4 range [-8,7], mirroring
         * dev_int_to_int's clamp/flag idiom: out-of-range values clamp to the
         * bound and flag under any non-PERMISSIVE method.
         * @brief int8_t -> int4 range clamp with CPU-matching flags.
         * @param v Lane value to clamp.
         * @param flag Reference set to true on a conversion event the active method cares about.
         * @param m Active ConversionClampMethod.
         * @return v clamped into [-8,7].
         */
        __device__ __forceinline__ int8_t dev_clamp_int4(const int8_t v, bool& flag, const ConversionClampMethod m) {
            if (v > 7) {
                if (m != ConversionClampMethod::PERMISSIVE) {
                    flag = true;
                }
                return 7;
            }
            if (v < -8) {
                if (m != ConversionClampMethod::PERMISSIVE) {
                    flag = true;
                }
                return -8;
            }
            return v;
        }

        /**
         * Convert a floating-point lane to an integer dtype, mirroring the
         * float -> int branch of the CPU dispatcher's scalar_convert
         * (Intrinsics/Dispatcher.h): NaN flags under any non-PERMISSIVE method
         * and yields 0; overflow clamps to Dst's max (when Src cannot represent
         * that max exactly, the bound rounds up to a value already outside Dst's
         * range, so equality there is also overflow); Dst's lowest is a power
         * of two (or zero) and always exact in Src.
         * @brief Float -> integer conversion with CPU-matching clamp/flags.
         * @tparam Src Source floating dtype (float32_t or float64_t).
         * @tparam Dst Destination integer dtype.
         * @param v Value to convert.
         * @param flag Reference set to true on a conversion event the active method cares about.
         * @param m Active ConversionClampMethod.
         * @return Value converted to Dst.
         */
        template<typename Src, typename Dst>
        __device__ __forceinline__ Dst dev_float_to_int(const Src v, bool& flag, const ConversionClampMethod m) {
            if constexpr (std::is_same_v<Src, float32_t>) {
                if (__isnanf(v)) {
                    if (m != ConversionClampMethod::PERMISSIVE) {
                        flag = true;
                    }
                    return Dst{0};
                }
            } else {
                if (__isnan(v)) {
                    if (m != ConversionClampMethod::PERMISSIVE) {
                        flag = true;
                    }
                    return Dst{0};
                }
            }
            constexpr auto src_digits = std::is_same_v<Src, float32_t> ? 24 : 53;
            constexpr auto max_bound = static_cast<Src>(dev_int_limits<Dst>::max_v);
            // !(v < max_bound) is v == max_bound here (NaN is handled above) —
            // written this way to stay -Werror=float-equal clean. Mirrors Dispatcher.h.
            if (
                constexpr bool max_exact = src_digits >= dev_int_limits<Dst>::digits;
                v > max_bound || (!max_exact && !(v < max_bound))
            ) {
                if (m != ConversionClampMethod::PERMISSIVE) {
                    flag = true;
                }
                return dev_int_limits<Dst>::max_v;
            }
            if (v < static_cast<Src>(dev_int_limits<Dst>::min_v)) {
                if (m != ConversionClampMethod::PERMISSIVE) {
                    flag = true;
                }
                return dev_int_limits<Dst>::min_v;
            }
            return static_cast<Dst>(v);
        }

        /**
         * Convert a single scalar lane between two dtypes, mirroring the
         * CPU dispatcher's scalar_convert (Intrinsics/Dispatcher.h) branch for
         * branch, integer lanes included.
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
                } else if constexpr (std::is_floating_point_v<Src>) {
                    // FP64 narrows through FP32 (double rounding, <=1 ulp at boundaries);
                    // scalar and AVX512 paths funnel identically, so they stay bit-exact.
                    return dev_fp32_to_bf16(dev_fp64_to_fp32(v, flag, m), flag, m);
                } else {
                    return dev_fp32_to_bf16(static_cast<float32_t>(v), flag, m);
                }
            } else if constexpr (std::is_same_v<Dst, float16_t>) {
                if constexpr (std::is_same_v<Src, float32_t>) {
                    return dev_fp32_to_fp16(v, flag, m);
                } else if constexpr (std::is_floating_point_v<Src>) {
                    // FP64 narrows through FP32 (double rounding, <=1 ulp at boundaries);
                    // scalar and AVX512 paths funnel identically, so they stay bit-exact.
                    return dev_fp32_to_fp16(dev_fp64_to_fp32(v, flag, m), flag, m);
                } else {
                    return dev_fp32_to_fp16(static_cast<float32_t>(v), flag, m);
                }
            } else if constexpr (std::is_integral_v<Src> && std::is_integral_v<Dst>) {
                return dev_int_to_int<Src, Dst>(v, flag, m);
            } else if constexpr (std::is_floating_point_v<Src> && std::is_floating_point_v<Dst>) {
                if constexpr (std::is_same_v<Dst, float32_t>) {
                    return dev_fp64_to_fp32(v, flag, m);
                } else {
                    return dev_fp32_to_fp64(v, flag, m);
                }
            } else if constexpr (std::is_integral_v<Src>) {
                // int -> float: always representable in range.
                return static_cast<Dst>(v);
            } else {
                // float -> int: range-check against the integer bounds.
                return dev_float_to_int<Src, Dst>(v, flag, m);
            }

        }

        /**
         * Pack two scaler lanes into one value. 2 unique BF16's to one BF16x2, for example.
         * @note Packs two lanes into a single packed value.
         * @tparam Dst Packed datatype.
         * @param lo Low lane of the packed value.
         * @param hi High lane of the packed value.
         * @return Packed value.
         */
        template<typename Dst> requires (BeanTensor::DataTypes::requires_unpack_v<Dst> && BeanTensor::DataTypes::can_be_packed_v<lane_of_t<Dst>>)
        __device__ __forceinline__ Dst pack_lanes(const lane_of_t<Dst> lo, const lane_of_t<Dst> hi) {
            if constexpr (std::is_same_v<Dst, float16x2_t>) {
                return __halves2half2(lo, hi);
            } else if constexpr (std::is_same_v<Dst, bfloat16x2_t>) {
                return __halves2bfloat162(lo, hi);
            } else {
                static_assert(dependent_false_v<Dst>, "pack_lanes requires a packed type");
                __builtin_unreachable();
            }
        }

        /**
         * Generic conversion kernel. Mirrors the CPU dispatcher's
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
        __global__ void impli_gpu_convert(
            const Src* __restrict__ src,
            Dst* __restrict__ dst,
            const size_t end,
            const ConversionClampMethod method,
            int32_t* __restrict__ fail_check
            ) {
            const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t stride = blockDim.x * gridDim.x;
            bool local_flag = false;

            for (size_t i = tid; i < end; i += stride) {
                if constexpr (std::is_same_v<Src, int4x2_t> && std::is_same_v<Dst, int4x2_t>) {
                    dst[i] = ldg_load(src + i);
                } else if constexpr (std::is_same_v<Src, int4x2_t> && !DataTypes::requires_unpack_v<Dst>) {
                    // int4x2 -> unpacked: convert the low lane only.
                    dst[i] = dev_scalar_convert<int8_t, Dst>(int4_lane_lo(ldg_load(src + i)), local_flag, method);
                } else if constexpr (std::is_same_v<Src, int4x2_t>) {
                    // int4x2 -> packed 16-bit: convert each lane.
                    using HD = lane_of_t<Dst>;
                    const auto v = ldg_load(src + i);
                    const auto lo = dev_scalar_convert<int8_t, HD>(int4_lane_lo(v), local_flag, method);
                    const auto hi = dev_scalar_convert<int8_t, HD>(int4_lane_hi(v), local_flag, method);
                    dst[i] = pack_lanes<Dst>(lo, hi);
                } else if constexpr (std::is_same_v<Dst, int4x2_t> && !DataTypes::requires_unpack_v<Src>) {
                    // unpacked -> int4x2: convert to int8, clamp into [-8,7], duplicate lanes.
                    const auto lane = dev_clamp_int4(dev_scalar_convert<Src, int8_t>(ldg_load(src + i), local_flag, method), local_flag, method);
                    dst[i] = pack_int4x2(lane, lane);
                } else if constexpr (std::is_same_v<Dst, int4x2_t>) {
                    // packed 16-bit -> int4x2: convert each lane to int8, clamp into [-8,7], pack.
                    using HS = lane_of_t<Src>;
                    const auto v = ldg_load(src + i);
                    const auto lo = dev_clamp_int4(dev_scalar_convert<HS, int8_t>(v.x, local_flag, method), local_flag, method);
                    const auto hi = dev_clamp_int4(dev_scalar_convert<HS, int8_t>(v.y, local_flag, method), local_flag, method);
                    dst[i] = pack_int4x2(lo, hi);
                } else if constexpr (!DataTypes::requires_unpack_v<Src> && !DataTypes::requires_unpack_v<Dst>) {
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

        /**
         * @brief Dispatches a dtype-conversion kernel to the GPU for a fixed type.
         * @tparam Src Source element type.
         * @param new_dtype Enum selecting destination dtype.
         * @param src Device pointer to source buffer.
         * @param dst Device pointer to destination buffer.
         * @param numel Number of elements to convert.
         * @param blocks Number of blocks to use. (blocks per grid)
         * @param threads Number of threads per block. (threads per block)
         * @param method Conversion method to use.
         * @param fail_check Device pointer to error flag. Nonzero written on error.
         */
        template<typename Src>
        __host__ void launch_convert_dst(const DataTypes::DType new_dtype, const Src* src, void* dst, const size_t numel,
                                const size_t blocks, const size_t threads, const ConversionClampMethod method, int32_t* fail_check) {
            using DType = DataTypes::DType;
            switch (new_dtype) {
                case DType::Float16: {
                    impli_gpu_convert<Src, float16_t><<<blocks, threads>>>(src, static_cast<float16_t*>(dst), numel, method, fail_check);
                    break;
                }
                case DType::Float16x2: {
                    impli_gpu_convert<Src, float16x2_t><<<blocks, threads>>>(src, static_cast<float16x2_t*>(dst), numel, method, fail_check);
                    break;
                }
                case DType::BFloat16: {
                    impli_gpu_convert<Src, bfloat16_t><<<blocks, threads>>>(src, static_cast<bfloat16_t*>(dst), numel, method, fail_check);
                    break;
                }
                case DType::BFloat16x2: {
                    impli_gpu_convert<Src, bfloat16x2_t><<<blocks, threads>>>(src, static_cast<bfloat16x2_t*>(dst), numel, method, fail_check);
                    break;
                }
                case DType::Float32: {
                    impli_gpu_convert<Src, float32_t><<<blocks, threads>>>(src, static_cast<float32_t*>(dst), numel, method, fail_check);
                    break;
                }
                case DType::Float64: {
                    impli_gpu_convert<Src, float64_t><<<blocks, threads>>>(src, static_cast<float64_t*>(dst), numel, method, fail_check);
                    break;
                }
                case DType::Int64: {
                    impli_gpu_convert<Src, int64_t><<<blocks, threads>>>(src, static_cast<int64_t*>(dst), numel, method, fail_check);
                    break;
                }
                case DType::Int32: {
                    impli_gpu_convert<Src, int32_t><<<blocks, threads>>>(src, static_cast<int32_t*>(dst), numel, method, fail_check);
                    break;
                }
                case DType::Int16: {
                    impli_gpu_convert<Src, int16_t><<<blocks, threads>>>(src, static_cast<int16_t*>(dst), numel, method, fail_check);
                    break;
                }
                case DType::Int8: {
                    impli_gpu_convert<Src, int8_t><<<blocks, threads>>>(src, static_cast<int8_t*>(dst), numel, method, fail_check);
                    break;
                }
                case DType::UInt64: {
                    impli_gpu_convert<Src, uint64_t><<<blocks, threads>>>(src, static_cast<uint64_t*>(dst), numel, method, fail_check);
                    break;
                }
                case DType::UInt32: {
                    impli_gpu_convert<Src, uint32_t><<<blocks, threads>>>(src, static_cast<uint32_t*>(dst), numel, method, fail_check);
                    break;
                }
                case DType::UInt16: {
                    impli_gpu_convert<Src, uint16_t><<<blocks, threads>>>(src, static_cast<uint16_t*>(dst), numel, method, fail_check);
                    break;
                }
                case DType::UInt8: {
                    impli_gpu_convert<Src, uint8_t><<<blocks, threads>>>(src, static_cast<uint8_t*>(dst), numel, method, fail_check);
                    break;
                }
                case DType::Int4x2: {
                    impli_gpu_convert<Src, int4x2_t><<<blocks, threads>>>(src, static_cast<int4x2_t*>(dst), numel, method, fail_check);
                    break;
                }
                [[unlikely]] default: {
                    throw ErrorHandling::NotImplemented();
                }
            }
        }

        /**
         * @brief Dispatches a static-value fill kernel to the GPU for a fixed dtype.
         * @tparam Val Value type the caller supplied (int64_t, uint64_t, or float64_t).
         * @param data Device pointer to the tensor buffer.
         * @param numel Number of elements.
         * @param dtype Active dtype of the tensor.
         * @param amount Value to set, converted per dtype.
         */
        template<typename Val>
        __host__ void launch_set_static(void* data, const size_t numel, const DataTypes::DType dtype, const Val amount) {
            constexpr size_t threads = 256;
            const size_t blocks = (numel + threads - 1) / threads;
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Int8: {
                    impli_gpu_set_static_val_unpacked<int8_t><<<blocks, threads>>>(static_cast<int8_t*>(data), numel, static_cast<int8_t>(amount));
                    break;
                }
                case DType::Int16: {
                    impli_gpu_set_static_val_unpacked<int16_t><<<blocks, threads>>>(static_cast<int16_t*>(data), numel, static_cast<int16_t>(amount));
                    break;
                }
                case DType::Int32: {
                    impli_gpu_set_static_val_unpacked<int32_t><<<blocks, threads>>>(static_cast<int32_t*>(data), numel, static_cast<int32_t>(amount));
                    break;
                }
                case DType::Int64: {
                    impli_gpu_set_static_val_unpacked<int64_t><<<blocks, threads>>>(static_cast<int64_t*>(data), numel, static_cast<int64_t>(amount));
                    break;
                }
                case DType::UInt8: {
                    impli_gpu_set_static_val_unpacked<uint8_t><<<blocks, threads>>>(static_cast<uint8_t*>(data), numel, static_cast<uint8_t>(amount));
                    break;
                }
                case DType::UInt16: {
                    impli_gpu_set_static_val_unpacked<uint16_t><<<blocks, threads>>>(static_cast<uint16_t*>(data), numel, static_cast<uint16_t>(amount));
                    break;
                }
                case DType::UInt32: {
                    impli_gpu_set_static_val_unpacked<uint32_t><<<blocks, threads>>>(static_cast<uint32_t*>(data), numel, static_cast<uint32_t>(amount));
                    break;
                }
                case DType::UInt64: {
                    impli_gpu_set_static_val_unpacked<uint64_t><<<blocks, threads>>>(static_cast<uint64_t*>(data), numel, static_cast<uint64_t>(amount));
                    break;
                }
                case DType::Float16: {
                    const auto h = __float2half_rn(static_cast<float32_t>(amount));
                    impli_gpu_set_static_val_unpacked<float16_t><<<blocks, threads>>>(static_cast<float16_t*>(data), numel, h);
                    break;
                }
                case DType::BFloat16: {
                    const auto h = __float2bfloat16(static_cast<float32_t>(amount)); // RNE (CUDA parity: __float2bfloat16_rn)
                    impli_gpu_set_static_val_unpacked<bfloat16_t><<<blocks, threads>>>(static_cast<bfloat16_t*>(data), numel, h);
                    break;
                }
                case DType::Float32: {
                    impli_gpu_set_static_val_unpacked<float32_t><<<blocks, threads>>>(static_cast<float32_t*>(data), numel, static_cast<float32_t>(amount));
                    break;
                }
                case DType::Float64: {
                    impli_gpu_set_static_val_unpacked<float64_t><<<blocks, threads>>>(static_cast<float64_t*>(data), numel, static_cast<float64_t>(amount));
                    break;
                }
                case DType::Float16x2: {
                    const auto h = __float2half_rn(static_cast<float32_t>(amount));
                    impli_gpu_set_static_val_packed<float16x2_t, float16_t><<<blocks, threads>>>(static_cast<float16x2_t*>(data), numel, h);
                    break;
                }
                case DType::BFloat16x2: {
                    const auto h = __float2bfloat16(static_cast<float32_t>(amount)); // RNE (CUDA parity: __float2bfloat16_rn)
                    impli_gpu_set_static_val_packed<bfloat16x2_t, bfloat16_t><<<blocks, threads>>>(static_cast<bfloat16x2_t*>(data), numel, h);
                    break;
                }
                case DType::Int4x2: {
                    // Clamp the fill value into the signed int4 range [-8,7]; both lanes get it.
                    auto lane = static_cast<int8_t>(0);
                    if constexpr (std::is_unsigned_v<Val>) {
                        if (amount > static_cast<Val>(7)) {
                            lane = static_cast<int8_t>(7);
                        } else {
                            lane = static_cast<int8_t>(amount);
                        }
                    } else {
                        if (amount > static_cast<Val>(7)) {
                            lane = static_cast<int8_t>(7);
                        } else if (amount < static_cast<Val>(-8)) {
                            lane = static_cast<int8_t>(-8);
                        } else {
                            lane = static_cast<int8_t>(amount);
                        }
                    }
                    impli_gpu_set_static_val_unpacked<int4x2_t><<<blocks, threads>>>(static_cast<int4x2_t*>(data), numel, pack_int4x2(lane, lane));
                    break;
                }
                [[unlikely]] default: {
                    throw ErrorHandling::NotImplemented();
                }
            }
            HIP_CHECK_ERROR(hipGetLastError());
        }
    }
}

namespace BeanTensor::Tensors::detail {
    /**
     * Sets a tensor to a static value on the GPU.
     * @note GPU Tensors only. Else, throw error.
     * @param t Tensors to receive modification.
     * @param amount Amount to set the tensors value.
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if the tensor is not on the GPU.
     */
    __host__ void set_amount_gpu(Tensor& t, const int64_t amount) {
        if (t.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(t.device));
        }
        t.sync();
        static auto& threadMgr = Threading::GetCPUThreadPool();
        const auto worker = [&t, amount] {
            gpu::launch_set_static(t.data, t.numel, t.dtype, amount);
        };
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        t.give_future(future);
    }

    /**
     * Sets a tensor to a static value on the GPU.
     * @note GPU Tensors only. Else, throw error.
     * @param t Tensors to receive modification.
     * @param amount Amount to set the tensors value.
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if the tensor is not on the GPU.
     */
    __host__ void set_amount_gpu(Tensor& t, const uint64_t amount) {
        if (t.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(t.device));
        }
        t.sync();
        static auto& threadMgr = Threading::GetCPUThreadPool();
        const auto worker = [&t, amount] {
            gpu::launch_set_static(t.data, t.numel, t.dtype, amount);
        };
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        t.give_future(future);
    }

    /**
     * Sets a tensor to a static value on the GPU.
     * @note GPU Tensors only. Else, throw error.
     * @param t Tensors to receive modification.
     * @param amount Amount to set the tensors value.
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if the tensor is not on the GPU.
     */
    __host__ void set_amount_gpu(Tensor& t, const float64_t amount) {
        if (t.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(t.device));
        }
        t.sync();
        static auto& threadMgr = Threading::GetCPUThreadPool();
        const auto worker = [&t, amount] {
            gpu::launch_set_static(t.data, t.numel, t.dtype, amount);
        };
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        t.give_future(future);
    }

    /**
     * Sets a tensor to a random value on the GPU.
     * @param t Tensor to receive modification.
     * @param min Lowest possible value for this tensor.
     * @param max Highest possible value for this tensor.
     * @param seed Seed to use.
     */
    __host__ void set_random_gpu(Tensor& t, const float64_t min, const float64_t max, const std::int64_t seed) {
        if (t.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(t.device));
        }
        t.sync();
        const auto worker = [&t, min, max, seed] {
            auto temp = Tensor({t.numel}, t.dtype, Device::CPU, false);
            temp.random(min, max, seed);
            temp.sync();
            move_to_gpu(temp);
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

    /**
     * Moves a tensor to the GPU.
     * @param t Tensor to move to the GPU.
     */
    __host__ void move_to_gpu(Tensor& t) {
        if (t.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(t.device));
        }
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

    /**
     * Moves a tensor to the CPU.
     * @param t Tensor to move to the CPU.
     */
    __host__ void move_to_cpu(Tensor& t) {
        if (t.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(t.device));
        }
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

    /**
     * Converts a tensor's data type on the GPU.
     * @param t The tensor to convert.
     * @param new_dtype The new data type.
     * @param method The conversion method.
     */
    __host__ void convert_gpu(Tensor& t, const DataTypes::DType& new_dtype, const ConversionClampMethod method) {
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
                case DType::Float16: {
                    gpu::launch_convert_dst<float16_t>(new_dtype, static_cast<const float16_t*>(t.data), t.convert_buffer, t.numel, blocks, threads, method, check_err);
                    break;
                }
                case DType::Float16x2: {
                    gpu::launch_convert_dst<float16x2_t>(new_dtype, static_cast<const float16x2_t*>(t.data), t.convert_buffer, t.numel, blocks, threads, method, check_err);
                    break;
                }
                case DType::BFloat16: {
                    gpu::launch_convert_dst<bfloat16_t>(new_dtype, static_cast<const bfloat16_t*>(t.data), t.convert_buffer, t.numel, blocks, threads, method, check_err);
                    break;
                }
                case DType::BFloat16x2: {
                    gpu::launch_convert_dst<bfloat16x2_t>(new_dtype, static_cast<const bfloat16x2_t*>(t.data), t.convert_buffer, t.numel, blocks, threads, method, check_err);
                    break;
                }
                case DType::Float32: {
                    gpu::launch_convert_dst<float32_t>(new_dtype, static_cast<const float32_t*>(t.data), t.convert_buffer, t.numel, blocks, threads, method, check_err);
                    break;
                }
                case DType::Float64: {
                    gpu::launch_convert_dst<float64_t>(new_dtype, static_cast<const float64_t*>(t.data), t.convert_buffer, t.numel, blocks, threads, method, check_err);
                    break;
                }
                case DType::Int64: {
                    gpu::launch_convert_dst<int64_t>(new_dtype, static_cast<const int64_t*>(t.data), t.convert_buffer, t.numel, blocks, threads, method, check_err);
                    break;
                }
                case DType::Int32: {
                    gpu::launch_convert_dst<int32_t>(new_dtype, static_cast<const int32_t*>(t.data), t.convert_buffer, t.numel, blocks, threads, method, check_err);
                    break;
                }
                case DType::Int16: {
                    gpu::launch_convert_dst<int16_t>(new_dtype, static_cast<const int16_t*>(t.data), t.convert_buffer, t.numel, blocks, threads, method, check_err);
                    break;
                }
                case DType::Int8: {
                    gpu::launch_convert_dst<int8_t>(new_dtype, static_cast<const int8_t*>(t.data), t.convert_buffer, t.numel, blocks, threads, method, check_err);
                    break;
                }
                case DType::UInt64: {
                    gpu::launch_convert_dst<uint64_t>(new_dtype, static_cast<const uint64_t*>(t.data), t.convert_buffer, t.numel, blocks, threads, method, check_err);
                    break;
                }
                case DType::UInt32: {
                    gpu::launch_convert_dst<uint32_t>(new_dtype, static_cast<const uint32_t*>(t.data), t.convert_buffer, t.numel, blocks, threads, method, check_err);
                    break;
                }
                case DType::UInt16: {
                    gpu::launch_convert_dst<uint16_t>(new_dtype, static_cast<const uint16_t*>(t.data), t.convert_buffer, t.numel, blocks, threads, method, check_err);
                    break;
                }
                case DType::UInt8: {
                    gpu::launch_convert_dst<uint8_t>(new_dtype, static_cast<const uint8_t*>(t.data), t.convert_buffer, t.numel, blocks, threads, method, check_err);
                    break;
                }
                case DType::Int4x2: {
                    gpu::launch_convert_dst<int4x2_t>(new_dtype, static_cast<const int4x2_t*>(t.data), t.convert_buffer, t.numel, blocks, threads, method, check_err);
                    break;
                }
                [[unlikely]] default: {
                    throw ErrorHandling::NotImplemented();
                }
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
