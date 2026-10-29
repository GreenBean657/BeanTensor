/**
 * @file ElementwiseHipShared.h
 * @author GreenBean
 * @brief HIP-side shared helpers for the elementwise GPU backend: grid
 * constants, device twins of the CPU unaccel fp32->fp16/bf16 narrowing
 * (bit-exact, including the fp16 overflow clamp and the BF16 subnormal
 * flush), half/packed/int4 lane helpers, the broadcast offset decomposition,
 * and the per-dtype scalar apply functions.
 * Divergent twin of Math/Elementwise/ElementwiseCudaShared.h — the two backends are
 * written separately per project rule (no HIP-only unification), but must stay
 * behaviorally identical: same fp32/fp64 compute widths, same RNE narrowing
 * (CPU unaccel pathway), same wrap integer ops, same div-by-zero/min-rules.
 * @version 0.1.3
 * @date 2026-10-22
 */
#pragma once

#if defined(USE_HIP)
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include "Math/Elementwise/Elementwise.h"
#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors::detail::elementwise_kernels {
    /**
     * @brief Threads per block for every elementwise kernel.
     */
    constexpr size_t BT_GPU_THREADS = 256;
    /**
     * Grid-stride loops cover any remainder, so large outputs never need more blocks than this.
     * @brief Upper bound on the block count of one launch.
     */
    constexpr size_t BT_GPU_MAX_BLOCKS = 1024;

    /**
     * Clamped to [1, BT_GPU_MAX_BLOCKS]; the grid-stride loop covers any remainder.
     * @brief Block count for a grid-stride kernel over numel elements.
     * @param numel Element count the kernel walks.
     * @return Block count of BT_GPU_THREADS wide blocks.
     */
    [[nodiscard]] inline __host__ size_t grid_for(const size_t numel) {
        const auto blocks = (numel + BT_GPU_THREADS - 1) / BT_GPU_THREADS;
        return blocks < 1 ? 1 : blocks > BT_GPU_MAX_BLOCKS ? BT_GPU_MAX_BLOCKS : blocks;
    }

    // --------------------------- narrowing twins ---------------------------
    // Both mirror the CPU unaccel pathway (Intrinsics/Native/DTypeConversions.h)
    // bit-for-bit so GPU results are identical to the CPU dispatcher's, per the
    // Project parity invariant. Pure bit manipulation — no intrinsics.

    /**
     * Mirror of unaccel_fp32_to_bf16: NaN/Inf truncate the payload (NaN sets
     * the quiet bit), fp32-subnormal/zero inputs flush to signed zero,
     * everything else rounds RNE via the add-half-plus-LSB bit trick.
     * @brief Device twin of the CPU unaccel FP32 -> BF16 narrowing.
     * @param f FP32 value to narrow.
     * @return BF16 value, bit-identical to the CPU unaccel pathway.
     */
    __device__ __forceinline__ bfloat16_t dev_narrow_bf16(const float32_t f) {
        const auto to_bits = __builtin_bit_cast(uint32_t, f);
        constexpr auto exponent_mask = 0x7F800000u;
        if ((to_bits & exponent_mask) == exponent_mask) {
            // NaN or Inf: truncate the payload; NaN sets the quiet bit so it can never become Inf.
            auto truncated = static_cast<uint16_t>(to_bits >> 16);
            if (
                constexpr auto mantissa_mask = 0x007FFFFFu;
                (to_bits & mantissa_mask) != 0u
                ) {
                truncated = static_cast<uint16_t>(truncated | 0x0040u);
            }
            return __builtin_bit_cast(bfloat16_t, truncated);
        }
        if ((to_bits & exponent_mask) == 0u) {
            // FP32 subnormal or zero: flush to signed zero (matches the CPU unaccel/AVX pathway).
            return __builtin_bit_cast(bfloat16_t, static_cast<uint16_t>((to_bits & 0x80000000u) >> 16));
        }
        auto rounded = to_bits;
        rounded += 0x7FFFu + ((rounded >> 16) & 1u);
        rounded >>= 16;
        return __builtin_bit_cast(bfloat16_t, static_cast<uint16_t>(rounded));
    }

    /**
     * Mirror of unaccel_fp32_to_fp16: NaN keeps a quieted payload, Inf stays
     * Inf, finite overflow clamps to +/-65504 (NOT Inf), subnormals round RNE
     * with gradual underflow, everything else rounds RNE via the bit trick.
     * @brief Device twin of the CPU unaccel FP32 -> FP16 narrowing.
     * @param val FP32 value to narrow.
     * @return FP16 value, bit-identical to the CPU unaccel pathway.
     */
    __device__ __forceinline__ float16_t dev_narrow_fp16(const float32_t val) {
        const auto to_bits = __builtin_bit_cast(uint32_t, val);
        auto converted = static_cast<uint16_t>((to_bits & 0x80000000u) >> 16);
        const auto mantissa_mask = to_bits & 0x007FFFFFu;
        const auto exponent_mask = to_bits & 0x7F800000u;
        if (mantissa_mask != 0u && exponent_mask == 0x7F800000u) {
            // NaN: keep the payload with the quiet bit set so it never truncates to Inf.
            converted = static_cast<uint16_t>(converted | 0x7C00u | 0x0200u | (mantissa_mask >> 13));
            return __builtin_bit_cast(float16_t, converted);
        }
        if (mantissa_mask == 0u && exponent_mask == 0x7F800000u) {
            // Inf.
            converted = static_cast<uint16_t>(converted | 0x7C00u);
            return __builtin_bit_cast(float16_t, converted);
        }
        if (val > DataTypes::detail::FP16_MAX) {
            return __builtin_bit_cast(float16_t, static_cast<uint16_t>(converted | 0x7BFFu));
        }
        if (val < DataTypes::detail::FP16_MIN) {
            return __builtin_bit_cast(float16_t, static_cast<uint16_t>(converted | 0xFBFFu));
        }
        if (const auto exp_field = (to_bits >> 23) & 0xFFu; exp_field < 113u) {
            // FP16-subnormal result: shift the mantissa into place with RNE gradual underflow.
            const auto mant24 = (to_bits & 0x007FFFFFu) | 0x00800000u;
            if (
                const auto shift = 126u - exp_field; // distance to the fp16 subnormal LSB (2^-24)
                shift <= 25u
                ) {
                const auto halfway = 1u << (shift - 1u);
                converted = static_cast<uint16_t>(converted | ((mant24 + halfway - 1u + ((mant24 >> shift) & 1u)) >> shift));
            }
            return __builtin_bit_cast(float16_t, converted);
        }
        // Round to nearest even: add half + the result LSB, then truncate.
        auto rounded = to_bits;
        rounded += 0xFFFu + ((rounded >> 13) & 1u);
        const auto exponent = ((rounded >> 23) & 0xFFu) - 112u; // -127 + 15 re-bias
        converted = static_cast<uint16_t>(converted | (exponent << 10) | ((rounded & 0x007FFFFFu) >> 13));
        return __builtin_bit_cast(float16_t, converted);
    }

    // --------------------------- lane helpers ---------------------------

    /**
     * @brief Widen one fp16 lane to fp32 (exact, hardware conversion).
     * @param v Half-width lane value.
     * @return The lane widened to fp32.
     */
    __device__ __forceinline__ float32_t dev_widen(const float16_t v) { return __half2float(v); }
    /**
     * @brief Widen one bf16 lane to fp32 (exact, hardware conversion).
     * @param v Half-width lane value.
     * @return The lane widened to fp32.
     */
    __device__ __forceinline__ float32_t dev_widen(const bfloat16_t v) { return __bfloat162float(v); }

    /**
     * Only float16x2_t and bfloat16x2_t are specialized; any other P fails to compile.
     * @brief Lane type of a packed dtype (float16x2_t -> float16_t, bfloat16x2_t -> bfloat16_t).
     */
    template<typename P> struct packed_lane;
    template<> struct packed_lane<float16x2_t> { using type = float16_t; };
    template<> struct packed_lane<bfloat16x2_t> { using type = bfloat16_t; };
    template<typename P> using packed_lane_t = packed_lane<P>::type;

    /**
     * @brief Low lane of a packed value.
     * @param v Packed value.
     * @return The low lane.
     */
    __device__ __forceinline__ float16_t dev_lane_lo(const float16x2_t v) { return __low2half(v); }
    /**
     * @brief Low lane of a packed value.
     * @param v Packed value.
     * @return The low lane.
     */
    __device__ __forceinline__ bfloat16_t dev_lane_lo(const bfloat16x2_t& v) { return __low2bfloat16(v); }
    /**
     * @brief High lane of a packed value.
     * @param v Packed value.
     * @return The high lane.
     */
    __device__ __forceinline__ float16_t dev_lane_hi(const float16x2_t v) { return __high2half(v); }
    /**
     * @brief High lane of a packed value.
     * @param v Packed value.
     * @return The high lane.
     */
    __device__ __forceinline__ bfloat16_t dev_lane_hi(const bfloat16x2_t& v) { return __high2bfloat16(v); }

    /**
     * @brief Pack two narrowed fp16 lanes back into a packed value.
     * @param lo Low lane.
     * @param hi High lane.
     * @return Packed value with lo in the low half, hi in the high half.
     */
    __device__ __forceinline__ float16x2_t dev_pack_lanes(const float16_t lo, const float16_t hi) { return __halves2half2(lo, hi); }
    /**
     * @brief Pack two narrowed bf16 lanes back into a packed value.
     * @param lo Low lane.
     * @param hi High lane.
     * @return Packed value with lo in the low half, hi in the high half.
     */
    __device__ __forceinline__ bfloat16x2_t dev_pack_lanes(const bfloat16_t lo, const bfloat16_t hi) { return __halves2bfloat162(lo, hi); }

    // --------------------------- int4x2 helpers ---------------------------

    /**
     * Low 4 bits, sign-extended from bit 3.
     * @brief Sign-extended low nibble of a packed int4 pair.
     * @param v Packed int4 pair.
     * @return Low nibble in [-8, 7].
     */
    __device__ __forceinline__ int32_t dev_int4_lo(const int4x2_t v) {
        const auto n = static_cast<int32_t>(static_cast<uint8_t>(v) & 0x0Fu);
        return n >= 8 ? n - 16 : n;
    }

    /**
     * High 4 bits, sign-extended from bit 3.
     * @brief Sign-extended high nibble of a packed int4 pair.
     * @param v Packed int4 pair.
     * @return High nibble in [-8, 7].
     */
    __device__ __forceinline__ int32_t dev_int4_hi(const int4x2_t v) {
        const auto n = static_cast<uint8_t>(v) >> 4;
        return n >= 8 ? n - 16 : n;
    }

    /**
     * Only the low 4 bits of each input survive, so out-of-range sums wrap.
     * @brief Pack two signed values into an int4 pair (modular, low 4 bits each).
     * @param lo Value for the low nibble.
     * @param hi Value for the high nibble.
     * @return Packed int4 pair.
     */
    __device__ __forceinline__ int4x2_t dev_pack_int4(const int32_t lo, const int32_t hi) {
        return static_cast<int4x2_t>(static_cast<uint8_t>((lo & 0xF) | ((hi & 0xF) << 4)));
    }

    // --------------------------- broadcast offsets ---------------------------

    /**
     * Device twin of elementwise_shared::bcast_offsets: decomposes the flat
     * output index into output coordinates and accumulates each operand's
     * element offset from its broadcast-adjusted strides. The exact fast path
     * (identical shapes) maps both offsets to the flat index. The two must
     * stay behaviorally identical to the host loop.
     * @brief Element offsets of both operands for output flat index i.
     * @param plan Broadcast execution plan.
     * @param i Flat output index.
     * @param ao Out: left operand element offset.
     * @param bo Out: right operand element offset.
     */
    __device__ __forceinline__ void dev_bcast_offsets(const ElementwiseBcast& plan, const size_t i, size_t& ao, size_t& bo) {
        if (plan.exact) {
            ao = i;
            bo = i;
            return;
        }
        ao = 0;
        bo = 0;
        auto rem = i;
        for (auto d = size_t{0}; d < plan.out_ndim; ++d) {
            const auto coord = rem / plan.out_strides[d];
            rem -= coord * plan.out_strides[d];
            ao += coord * plan.a_strides[d];
            bo += coord * plan.b_strides[d];
        }
    }

    // --------------------------- arithmetic applies ---------------------------

    /**
     * Single-rounding native op; Div is IEEE (x/0 -> inf/nan), Pow is powf (the
     * device-libm twin of std::pow on floats).
     * @brief Scalar binary apply (fp32).
     */
    template<ElementwiseOp Op>
    __device__ __forceinline__ float32_t dev_apply_native(const float32_t a, const float32_t b) {
        if constexpr (Op == ElementwiseOp::Add) {
            return a + b;
        } else if constexpr (Op == ElementwiseOp::Sub) {
            return a - b;
        } else if constexpr (Op == ElementwiseOp::Mul) {
            return a * b;
        } else if constexpr (Op == ElementwiseOp::Div) {
            return a / b;
        } else {
            return powf(a, b);
        }
    }

    /**
     * Single-rounding native op; Div is IEEE, Pow is pow (the device-libm twin
     * of std::pow on doubles).
     * @brief Scalar binary apply (fp64).
     */
    template<ElementwiseOp Op>
    __device__ __forceinline__ float64_t dev_apply_native(const float64_t a, const float64_t b) {
        if constexpr (Op == ElementwiseOp::Add) {
            return a + b;
        } else if constexpr (Op == ElementwiseOp::Sub) {
            return a - b;
        } else if constexpr (Op == ElementwiseOp::Mul) {
            return a * b;
        } else if constexpr (Op == ElementwiseOp::Div) {
            return a / b;
        } else {
            return pow(a, b);
        }
    }

    /**
     * Widen to fp32, apply, narrow through the CPU-matching unaccel twin.
     * @brief Scalar binary apply (fp16/bf16, fp32 compute).
     */
    template<typename H, ElementwiseOp Op>
    __device__ __forceinline__ H dev_apply_half(const H a, const H b) {
        const auto wa = dev_widen(a);
        const auto wb = dev_widen(b);
        if constexpr (Op == ElementwiseOp::Add) {
            const auto r = wa + wb;
            if constexpr (std::is_same_v<H, float16_t>) {
                return dev_narrow_fp16(r);
            } else {
                return dev_narrow_bf16(r);
            }
        } else if constexpr (Op == ElementwiseOp::Sub) {
            const auto r = wa - wb;
            if constexpr (std::is_same_v<H, float16_t>) {
                return dev_narrow_fp16(r);
            } else {
                return dev_narrow_bf16(r);
            }
        } else if constexpr (Op == ElementwiseOp::Mul) {
            const auto r = wa * wb;
            if constexpr (std::is_same_v<H, float16_t>) {
                return dev_narrow_fp16(r);
            } else {
                return dev_narrow_bf16(r);
            }
        } else if constexpr (Op == ElementwiseOp::Div) {
            const auto r = wa / wb;
            if constexpr (std::is_same_v<H, float16_t>) {
                return dev_narrow_fp16(r);
            } else {
                return dev_narrow_bf16(r);
            }
        } else {
            const auto r = powf(wa, wb);
            if constexpr (std::is_same_v<H, float16_t>) {
                return dev_narrow_fp16(r);
            } else {
                return dev_narrow_bf16(r);
            }
        }
    }

    /**
     * Per-lane widen/apply/narrow, repack on store.
     * @brief Scalar binary apply (fp16x2/bf16x2, per-lane fp32 compute).
     */
    template<typename P, ElementwiseOp Op>
    __device__ __forceinline__ P dev_apply_packed(const P& a, const P& b) {
        using H = packed_lane_t<P>;
        const auto r_lo = dev_apply_half<H, Op>(dev_lane_lo(a), dev_lane_lo(b));
        const auto r_hi = dev_apply_half<H, Op>(dev_lane_hi(a), dev_lane_hi(b));
        return dev_pack_lanes(r_lo, r_hi);
    }

    /**
     * Modular (wrap) add/sub/mul through the unsigned twin; Div truncates
     * toward zero, division by zero yields 0 and min/-1 yields min (wrap) —
     * the same contract as the integer CPU kernels, never UB. Pow on integers
     * is rejected before dispatch and never reaches here.
     * @brief Scalar binary apply (all integer dtypes).
     */
    template<typename T, ElementwiseOp Op>
    __device__ __forceinline__ T dev_apply_int(const T a, const T b) {
        using U = std::make_unsigned_t<T>;
        if constexpr (Op == ElementwiseOp::Add) {
            return static_cast<T>(static_cast<U>(static_cast<U>(a) + static_cast<U>(b)));
        } else if constexpr (Op == ElementwiseOp::Sub) {
            return static_cast<T>(static_cast<U>(static_cast<U>(a) - static_cast<U>(b)));
        } else if constexpr (Op == ElementwiseOp::Mul) {
            return static_cast<T>(static_cast<U>(static_cast<U>(a) * static_cast<U>(b)));
        } else if constexpr (Op == ElementwiseOp::Div) {
            if (b == T{0}) {
                return T{0};
            }
            if constexpr (std::is_signed_v<T>) {
                // The min pattern is built through the unsigned twin because
                // nvcc rejects std::numeric_limits in device code (the CPU
                // kernel documents the same branch).
                constexpr auto min_pattern = static_cast<T>(U{1} << (sizeof(T) * 8u - 1u));
                if (a == min_pattern && b == T{-1}) {
                    return a;
                }
            }
            return static_cast<T>(a / b);
        } else {
            return T{0}; // Pow: unreachable — integer pow is rejected at the Tensor method.
        }
    }

    /**
     * Per-nibble apply, repack on store. Add/Sub/Mul wrap per nibble (the
     * pack keeps the low 4 bits); Div truncates per nibble with the same
     * div-by-zero -> 0 rule.
     * @brief Scalar binary apply (int4x2, per-nibble).
     */
    template<ElementwiseOp Op>
    __device__ __forceinline__ int4x2_t dev_apply_int4x2(const int4x2_t a, const int4x2_t b) {
        const auto alo = dev_int4_lo(a);
        const auto ahi = dev_int4_hi(a);
        const auto blo = dev_int4_lo(b);
        const auto bhi = dev_int4_hi(b);
        if constexpr (Op == ElementwiseOp::Add) {
            return dev_pack_int4(alo + blo, ahi + bhi);
        } else if constexpr (Op == ElementwiseOp::Sub) {
            return dev_pack_int4(alo - blo, ahi - bhi);
        } else if constexpr (Op == ElementwiseOp::Mul) {
            return dev_pack_int4(alo * blo, ahi * bhi);
        } else if constexpr (Op == ElementwiseOp::Div) {
            return dev_pack_int4(blo == 0 ? 0 : alo / blo, bhi == 0 ? 0 : ahi / bhi);
        } else {
            return a; // Pow: unreachable — integer pow is rejected at the Tensor method.
        }
    }

    /**
     * @brief Category dispatch for the binary applies.
     */
    template<typename T, ElementwiseOp Op>
    __device__ __forceinline__ T dev_apply_any(const T& a, const T& b) {
        if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
            return dev_apply_native<Op>(a, b);
        } else if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
            return dev_apply_half<T, Op>(a, b);
        } else if constexpr (DataTypes::requires_unpack_v<T>) {
            return dev_apply_packed<T, Op>(a, b);
        } else if constexpr (std::is_same_v<T, int4x2_t>) {
            return dev_apply_int4x2<Op>(a, b);
        } else {
            return dev_apply_int<T, Op>(a, b);
        }
    }

    // --------------------------- unary applies ---------------------------

    /**
     * expf/logf (device-libm twins of std::exp/std::log on floats).
     * @brief Scalar unary apply (fp32).
     */
    template<ElementwiseUnaryOp Op>
    __device__ __forceinline__ float32_t dev_apply_unary_native(const float32_t v) {
        if constexpr (Op == ElementwiseUnaryOp::Exp) {
            return expf(v);
        } else if constexpr (Op == ElementwiseUnaryOp::Log) {
            return logf(v);
        } else if constexpr (Op == ElementwiseUnaryOp::Sigmoid) {
            return float32_t{1.0f} / (float32_t{1.0f} + expf(-v));
        } else {
            return float32_t{1.0f} - float32_t{2.0f} / (expf(float32_t{2.0f} * v) + float32_t{1.0f});
        }
    }

    /**
     * exp/log (device-libm twins of std::exp/std::log on doubles).
     * @brief Scalar unary apply (fp64).
     */
    template<ElementwiseUnaryOp Op>
    __device__ __forceinline__ float64_t dev_apply_unary_native(const float64_t v) {
        if constexpr (Op == ElementwiseUnaryOp::Exp) {
            return exp(v);
        } else if constexpr (Op == ElementwiseUnaryOp::Log) {
            return log(v);
        } else if constexpr (Op == ElementwiseUnaryOp::Sigmoid) {
            return float64_t{1.0} / (float64_t{1.0} + exp(-v));
        } else {
            return float64_t{1.0} - float64_t{2.0} / (exp(float64_t{2.0} * v) + float64_t{1.0});
        }
    }

    /**
     * Widen to fp32, apply, narrow through the CPU-matching unaccel twin.
     * @brief Scalar unary apply (fp16/bf16, fp32 compute).
     */
    template<typename H, ElementwiseUnaryOp Op>
    __device__ __forceinline__ H dev_apply_unary_half(const H v) {
        const auto wide = dev_widen(v);
        if constexpr (Op == ElementwiseUnaryOp::Exp) {
            if constexpr (std::is_same_v<H, float16_t>) {
                return dev_narrow_fp16(expf(wide));
            } else {
                return dev_narrow_bf16(expf(wide));
            }
        } else if constexpr (Op == ElementwiseUnaryOp::Log) {
            if constexpr (std::is_same_v<H, float16_t>) {
                return dev_narrow_fp16(logf(wide));
            } else {
                return dev_narrow_bf16(logf(wide));
            }
        } else if constexpr (Op == ElementwiseUnaryOp::Sigmoid) {
            const auto r = float32_t{1.0f} / (float32_t{1.0f} + expf(-wide));
            if constexpr (std::is_same_v<H, float16_t>) {
                return dev_narrow_fp16(r);
            } else {
                return dev_narrow_bf16(r);
            }
        } else {
            const auto r = float32_t{1.0f} - float32_t{2.0f} / (expf(float32_t{2.0f} * wide) + float32_t{1.0f});
            if constexpr (std::is_same_v<H, float16_t>) {
                return dev_narrow_fp16(r);
            } else {
                return dev_narrow_bf16(r);
            }
        }
    }

    /**
     * Per-lane widen/apply/narrow, repack on store.
     * @brief Scalar unary apply (fp16x2/bf16x2, per-lane fp32 compute).
     */
    template<typename P, ElementwiseUnaryOp Op>
    __device__ __forceinline__ P dev_apply_unary_packed(const P& v) {
        using H = packed_lane_t<P>;
        return dev_pack_lanes(dev_apply_unary_half<H, Op>(dev_lane_lo(v)), dev_apply_unary_half<H, Op>(dev_lane_hi(v)));
    }

    /**
     * @brief Category dispatch for the unary applies.
     */
    template<typename T, ElementwiseUnaryOp Op>
    __device__ __forceinline__ T dev_apply_any_unary(const T& v) {
        if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
            return dev_apply_unary_native<Op>(v);
        } else if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
            return dev_apply_unary_half<T, Op>(v);
        } else {
            return dev_apply_unary_packed<T, Op>(v);
        }
    }
}

#endif

