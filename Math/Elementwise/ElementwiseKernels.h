/**
 * @file ElementwiseKernels.h
 * @author GreenBean
 * @brief SIMD kernels + runtime probes for binary elementwise arithmetic
 * (add/sub/mul/div). Phase 1 scope: same-shape, contiguous, both operands
 * dense. fp16/bf16 buffers are processed as FLAT uint16 lane arrays — one
 * 16-bit lane per element for Float16/BFloat16, two independent lanes per
 * packed element for Float16x2/BFloat16x2 (per-lane elementwise semantics are
 * identical, so the packed dispatch reuses the same kernels over 2n lanes).
 * Phase 2 adds: tensor-scalar SPLAT kernels (the scalar is converted ONCE
 * into the tensor dtype by the caller — the narrow_scalar PERMISSIVE contract
 * in ElementwiseShared.h — then broadcast into a vector register) over the
 * same dtype/op set, and per-nibble Int4x2 add/sub/mul kernels (Int4x2 div
 * stays scalar under the same no-integer-div rule; broadcast Int4x2 rows
 * also stay scalar, see ElementwiseCPU.cpp).
 * Still NOT covered: broadcast plans with two dense operands (handled by row
 * decomposition onto these kernels in ElementwiseCPU.cpp), Pow, and integer
 * div (x86 SIMD has no integer division; div-by-zero/min-over-(-1) semantics
 * stay scalar by design).
 * @version 0.3
 * @date 2026-10-23
 */
#pragma once

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "Math/Elementwise/Elementwise.h"
#include "Tensor/DTypes.h"

namespace BeanTensor::Math::Elementwise {
    /**
     * Sets the quiet bit and keeps the payload. Only meaningful on a NaN.
     * Internal linkage: the AVX2/AVX512 translation units compile this header with
     * their ISA flags, so each TU keeps its own copy and no weak symbol is shared
     * with the scalar paths.
     * @brief Quiets a floating-point NaN.
     * @param v A NaN of type F.
     * @return The same NaN with the quiet bit set.
     */
    template<typename F>
    [[nodiscard]] static F quiet_nan(const F v) {
        if constexpr (std::is_same_v<F, float32_t>) {
            return std::bit_cast<float32_t>(std::bit_cast<uint32_t>(v) | (uint32_t{1} << 22));
        } else {
            return std::bit_cast<float64_t>(std::bit_cast<uint64_t>(v) | (uint64_t{1} << 51));
        }
    }

    /**
     * Single-rounding IEEE add/sub/mul/div under one NaN contract: a NaN operand
     * propagates quieted, and when both operands are NaN the first operand's payload
     * wins (the x86 rule for the SSE/AVX arithmetic instructions). The SIMD kernels
     * apply the same select explicitly, so every path returns the same bits on every
     * host compiler. Non-NaN inputs use the plain IEEE op.
     * @brief Scalar fp32/fp64 binary op under the elementwise NaN contract.
     * @param a The left operand.
     * @param b The right operand.
     * @return The op result under the NaN contract.
     */
    template<typename F, Tensors::ElementwiseOp Op>
    [[nodiscard]] static F fp_binop(const F a, const F b) {
        static_assert(Op != Tensors::ElementwiseOp::Pow, "fp_binop covers add/sub/mul/div only");
        if (std::isnan(a) || std::isnan(b)) {
            return quiet_nan<F>(std::isnan(a) ? a : b);
        }
        if constexpr (Op == Tensors::ElementwiseOp::Add) {
            return a + b;
        } else if constexpr (Op == Tensors::ElementwiseOp::Sub) {
            return a - b;
        } else if constexpr (Op == Tensors::ElementwiseOp::Mul) {
            return a * b;
        } else {
            return a / b;
        }
    }

    /**
     * Cached through IncludedCPUFeatures(); callers gate every AVX2 kernel on this.
     * @brief Runtime probe: are the AVX2+F16C elementwise kernels safe to execute?
     * @return True when the CPU supports AVX2 and F16C.
     */
    [[nodiscard]] bool elementwise_avx2_supported();

    /**
     * Cached through IncludedCPUFeatures(); callers gate every AVX512 kernel on this.
     * @brief Runtime probe: are the AVX512 elementwise kernels safe to execute?
     * @return True when the CPU supports AVX512F, AVX512BW, AVX512BF16, AVX512DQ and F16C
     * (the flag union the AVX512 kernel TU is compiled with).
     */
    [[nodiscard]] bool elementwise_avx512_supported();

    // --------------------------- fp32 (native) ---------------------------

    /**
     * Computes out[i] = a[i] + b[i] over [begin, end). IEEE single-rounding add, bit-identical to the scalar apply.
     * @brief AVX2/AVX512 range add (fp32).
     */
    void add_f32_range_avx2(float32_t*, const float32_t*, const float32_t*, size_t, size_t);
    /**
     * Computes out[i] = a[i] - b[i] over [begin, end). IEEE single-rounding sub, bit-identical to the scalar apply.
     * @brief AVX2/AVX512 range sub (fp32).
     */
    void sub_f32_range_avx2(float32_t*, const float32_t*, const float32_t*, size_t, size_t);
    /**
     * Computes out[i] = a[i] * b[i] over [begin, end). IEEE single-rounding mul, bit-identical to the scalar apply.
     * @brief AVX2/AVX512 range mul (fp32).
     */
    void mul_f32_range_avx2(float32_t*, const float32_t*, const float32_t*, size_t, size_t);
    /**
     * Computes out[i] = a[i] / b[i] over [begin, end). IEEE div (x/0 -> inf/nan), bit-identical to the scalar apply.
     * @brief AVX2/AVX512 range div (fp32).
     */
    void div_f32_range_avx2(float32_t*, const float32_t*, const float32_t*, size_t, size_t);

    /**
     * Computes out[i] = a[i] + b[i] over [begin, end). IEEE single-rounding add, bit-identical to the scalar apply.
     * @brief AVX512 range add (fp32).
     */
    void add_f32_range_avx512(float32_t*, const float32_t*, const float32_t*, size_t, size_t);
    /**
     * Computes out[i] = a[i] - b[i] over [begin, end). IEEE single-rounding sub, bit-identical to the scalar apply.
     * @brief AVX512 range sub (fp32).
     */
    void sub_f32_range_avx512(float32_t*, const float32_t*, const float32_t*, size_t, size_t);
    /**
     * Computes out[i] = a[i] * b[i] over [begin, end). IEEE single-rounding mul, bit-identical to the scalar apply.
     * @brief AVX512 range mul (fp32).
     */
    void mul_f32_range_avx512(float32_t*, const float32_t*, const float32_t*, size_t, size_t);
    /**
     * Computes out[i] = a[i] / b[i] over [begin, end). IEEE div (x/0 -> inf/nan), bit-identical to the scalar apply.
     * @brief AVX512 range div (fp32).
     */
    void div_f32_range_avx512(float32_t*, const float32_t*, const float32_t*, size_t, size_t);

    // --------------------------- fp64 (native) ---------------------------

    /**
     * Computes out[i] = a[i] + b[i] over [begin, end). IEEE single-rounding add, bit-identical to the scalar apply.
     * @brief AVX2 range add (fp64).
     */
    void add_f64_range_avx2(float64_t*, const float64_t*, const float64_t*, size_t, size_t);
    /**
     * Computes out[i] = a[i] - b[i] over [begin, end). IEEE single-rounding sub, bit-identical to the scalar apply.
     * @brief AVX2 range sub (fp64).
     */
    void sub_f64_range_avx2(float64_t*, const float64_t*, const float64_t*, size_t, size_t);
    /**
     * Computes out[i] = a[i] * b[i] over [begin, end). IEEE single-rounding mul, bit-identical to the scalar apply.
     * @brief AVX2 range mul (fp64).
     */
    void mul_f64_range_avx2(float64_t*, const float64_t*, const float64_t*, size_t, size_t);
    /**
     * Computes out[i] = a[i] / b[i] over [begin, end). IEEE div (x/0 -> inf/nan), bit-identical to the scalar apply.
     * @brief AVX2 range div (fp64).
     */
    void div_f64_range_avx2(float64_t*, const float64_t*, const float64_t*, size_t, size_t);

    /**
     * Computes out[i] = a[i] + b[i] over [begin, end). IEEE single-rounding add, bit-identical to the scalar apply.
     * @brief AVX512 range add (fp64).
     */
    void add_f64_range_avx512(float64_t*, const float64_t*, const float64_t*, size_t, size_t);
    /**
     * Computes out[i] = a[i] - b[i] over [begin, end). IEEE single-rounding sub, bit-identical to the scalar apply.
     * @brief AVX512 range sub (fp64).
     */
    void sub_f64_range_avx512(float64_t*, const float64_t*, const float64_t*, size_t, size_t);
    /**
     * Computes out[i] = a[i] * b[i] over [begin, end). IEEE single-rounding mul, bit-identical to the scalar apply.
     * @brief AVX512 range mul (fp64).
     */
    void mul_f64_range_avx512(float64_t*, const float64_t*, const float64_t*, size_t, size_t);
    /**
     * Computes out[i] = a[i] / b[i] over [begin, end). IEEE div (x/0 -> inf/nan), bit-identical to the scalar apply.
     * @brief AVX512 range div (fp64).
     */
    void div_f64_range_avx512(float64_t*, const float64_t*, const float64_t*, size_t, size_t);

    // ------------- fp16 lanes (Float16 AND Float16x2 buffers) -------------

    /**
     * Per-lane: widen fp16->fp32, add, narrow RNE with the scalar overflow clamp
     * (finite |x| > 65504 -> +/-65504) over [begin, end) flat 16-bit lanes.
     * Bit-identical to the scalar widen/apply/narrow apply.
     * @brief AVX2 range add (fp16 lanes).
     */
    void add_f16_range_avx2(uint16_t*, const uint16_t*, const uint16_t*, size_t, size_t);
    /**
     * Per-lane: widen fp16->fp32, sub, narrow RNE with the scalar overflow clamp
     * (finite |x| > 65504 -> +/-65504) over [begin, end) flat 16-bit lanes.
     * Bit-identical to the scalar widen/apply/narrow apply.
     * @brief AVX2 range sub (fp16 lanes).
     */
    void sub_f16_range_avx2(uint16_t*, const uint16_t*, const uint16_t*, size_t, size_t);
    /**
     * Per-lane: widen fp16->fp32, mul, narrow RNE with the scalar overflow clamp
     * (finite |x| > 65504 -> +/-65504) over [begin, end) flat 16-bit lanes.
     * Bit-identical to the scalar widen/apply/narrow apply.
     * @brief AVX2 range mul (fp16 lanes).
     */
    void mul_f16_range_avx2(uint16_t*, const uint16_t*, const uint16_t*, size_t, size_t);
    /**
     * Per-lane: widen fp16->fp32, div, narrow RNE with the scalar overflow clamp
     * (finite |x| > 65504 -> +/-65504) over [begin, end) flat 16-bit lanes.
     * Bit-identical to the scalar widen/apply/narrow apply.
     * @brief AVX2 range div (fp16 lanes).
     */
    void div_f16_range_avx2(uint16_t*, const uint16_t*, const uint16_t*, size_t, size_t);

    /**
     * Per-lane: widen fp16->fp32, add, narrow RNE with the scalar overflow clamp
     * (finite |x| > 65504 -> +/-65504) over [begin, end) flat 16-bit lanes.
     * Bit-identical to the scalar widen/apply/narrow apply.
     * @brief AVX512 range add (fp16 lanes).
     */
    void add_f16_range_avx512(uint16_t*, const uint16_t*, const uint16_t*, size_t, size_t);
    /**
     * Per-lane: widen fp16->fp32, sub, narrow RNE with the scalar overflow clamp
     * (finite |x| > 65504 -> +/-65504) over [begin, end) flat 16-bit lanes.
     * Bit-identical to the scalar widen/apply/narrow apply.
     * @brief AVX512 range sub (fp16 lanes).
     */
    void sub_f16_range_avx512(uint16_t*, const uint16_t*, const uint16_t*, size_t, size_t);
    /**
     * Per-lane: widen fp16->fp32, mul, narrow RNE with the scalar overflow clamp
     * (finite |x| > 65504 -> +/-65504) over [begin, end) flat 16-bit lanes.
     * Bit-identical to the scalar widen/apply/narrow apply.
     * @brief AVX512 range mul (fp16 lanes).
     */
    void mul_f16_range_avx512(uint16_t*, const uint16_t*, const uint16_t*, size_t, size_t);
    /**
     * Per-lane: widen fp16->fp32, div, narrow RNE with the scalar overflow clamp
     * (finite |x| > 65504 -> +/-65504) over [begin, end) flat 16-bit lanes.
     * Bit-identical to the scalar widen/apply/narrow apply.
     * @brief AVX512 range div (fp16 lanes).
     */
    void div_f16_range_avx512(uint16_t*, const uint16_t*, const uint16_t*, size_t, size_t);

    // ------------- bf16 lanes (BFloat16 AND BFloat16x2 buffers) -------------

    /**
     * Per-lane: widen bf16->fp32 (16-bit shift), add, narrow RNE with the
     * scalar special-case semantics (NaN quieting, fp32-subnormal flush-to-zero)
     * over [begin, end) flat 16-bit lanes. Bit-identical to the scalar apply.
     * @brief AVX2 range add (bf16 lanes).
     */
    void add_bf16_range_avx2(uint16_t*, const uint16_t*, const uint16_t*, size_t, size_t);
    /**
     * Per-lane: widen bf16->fp32 (16-bit shift), sub, narrow RNE with the
     * scalar special-case semantics (NaN quieting, fp32-subnormal flush-to-zero)
     * over [begin, end) flat 16-bit lanes. Bit-identical to the scalar apply.
     * @brief AVX2 range sub (bf16 lanes).
     */
    void sub_bf16_range_avx2(uint16_t*, const uint16_t*, const uint16_t*, size_t, size_t);
    /**
     * Per-lane: widen bf16->fp32 (16-bit shift), mul, narrow RNE with the
     * scalar special-case semantics (NaN quieting, fp32-subnormal flush-to-zero)
     * over [begin, end) flat 16-bit lanes. Bit-identical to the scalar apply.
     * @brief AVX2 range mul (bf16 lanes).
     */
    void mul_bf16_range_avx2(uint16_t*, const uint16_t*, const uint16_t*, size_t, size_t);
    /**
     * Per-lane: widen bf16->fp32 (16-bit shift), div, narrow RNE with the
     * scalar special-case semantics (NaN quieting, fp32-subnormal flush-to-zero)
     * over [begin, end) flat 16-bit lanes. Bit-identical to the scalar apply.
     * @brief AVX2 range div (bf16 lanes).
     */
    void div_bf16_range_avx2(uint16_t*, const uint16_t*, const uint16_t*, size_t, size_t);

    /**
     * Per-lane: widen bf16->fp32 (16-bit shift), add, narrow RNE via
     * _mm512_cvtneps_pbh (hardware matches the scalar NaN/flush/RNE semantics)
     * over [begin, end) flat 16-bit lanes. Bit-identical to the scalar apply.
     * @brief AVX512 range add (bf16 lanes).
     */
    void add_bf16_range_avx512(uint16_t*, const uint16_t*, const uint16_t*, size_t, size_t);
    /**
     * Per-lane: widen bf16->fp32 (16-bit shift), sub, narrow RNE via
     * _mm512_cvtneps_pbh (hardware matches the scalar NaN/flush/RNE semantics)
     * over [begin, end) flat 16-bit lanes. Bit-identical to the scalar apply.
     * @brief AVX512 range sub (bf16 lanes).
     */
    void sub_bf16_range_avx512(uint16_t*, const uint16_t*, const uint16_t*, size_t, size_t);
    /**
     * Per-lane: widen bf16->fp32 (16-bit shift), mul, narrow RNE via
     * _mm512_cvtneps_pbh (hardware matches the scalar NaN/flush/RNE semantics)
     * over [begin, end) flat 16-bit lanes. Bit-identical to the scalar apply.
     * @brief AVX512 range mul (bf16 lanes).
     */
    void mul_bf16_range_avx512(uint16_t*, const uint16_t*, const uint16_t*, size_t, size_t);
    /**
     * Per-lane: widen bf16->fp32 (16-bit shift), div, narrow RNE via
     * _mm512_cvtneps_pbh (hardware matches the scalar NaN/flush/RNE semantics)
     * over [begin, end) flat 16-bit lanes. Bit-identical to the scalar apply.
     * @brief AVX512 range div (bf16 lanes).
     */
    void div_bf16_range_avx512(uint16_t*, const uint16_t*, const uint16_t*, size_t, size_t);

    // --------------------------- integers ---------------------------
    // Wrap (modular) add/sub/mul only — one kernel per width serves the signed
    // AND unsigned dtype (the bit patterns of +,-,* are sign-agnostic). Integer
    // div stays on the scalar path: no x86 integer division, and the
    // div-by-zero -> 0 / min/-1 -> min contract is scalar-defined.

    /**
     * Computes out[i] = (a[i] + b[i]) mod 2^8 over [begin, end). Wrap add, bit-identical for int8 and uint8.
     * @brief AVX2 range add (8-bit lanes).
     */
    void add_i8_range_avx2(int8_t*, const int8_t*, const int8_t*, size_t, size_t);
    /**
     * Computes out[i] = (a[i] - b[i]) mod 2^8 over [begin, end). Wrap sub, bit-identical for int8 and uint8.
     * @brief AVX2 range sub (8-bit lanes).
     */
    void sub_i8_range_avx2(int8_t*, const int8_t*, const int8_t*, size_t, size_t);
    /**
     * Computes out[i] = (a[i] * b[i]) mod 2^8 over [begin, end). Wrap mul (low byte), bit-identical for int8 and uint8.
     * @brief AVX2 range mul (8-bit lanes).
     */
    void mul_i8_range_avx2(int8_t*, const int8_t*, const int8_t*, size_t, size_t);

    /**
     * Computes out[i] = (a[i] + b[i]) mod 2^8 over [begin, end). Wrap add, bit-identical for int8 and uint8.
     * @brief AVX512 range add (8-bit lanes).
     */
    void add_i8_range_avx512(int8_t*, const int8_t*, const int8_t*, size_t, size_t);
    /**
     * Computes out[i] = (a[i] - b[i]) mod 2^8 over [begin, end). Wrap sub, bit-identical for int8 and uint8.
     * @brief AVX512 range sub (8-bit lanes).
     */
    void sub_i8_range_avx512(int8_t*, const int8_t*, const int8_t*, size_t, size_t);
    /**
     * Computes out[i] = (a[i] * b[i]) mod 2^8 over [begin, end). Wrap mul (low byte), bit-identical for int8 and uint8.
     * @brief AVX512 range mul (8-bit lanes).
     */
    void mul_i8_range_avx512(int8_t*, const int8_t*, const int8_t*, size_t, size_t);

    /**
     * Computes out[i] = (a[i] + b[i]) mod 2^16 over [begin, end). Wrap add, bit-identical for int16 and uint16.
     * @brief AVX2 range add (16-bit lanes).
     */
    void add_i16_range_avx2(int16_t*, const int16_t*, const int16_t*, size_t, size_t);
    /**
     * Computes out[i] = (a[i] - b[i]) mod 2^16 over [begin, end). Wrap sub, bit-identical for int16 and uint16.
     * @brief AVX2 range sub (16-bit lanes).
     */
    void sub_i16_range_avx2(int16_t*, const int16_t*, const int16_t*, size_t, size_t);
    /**
     * Computes out[i] = (a[i] * b[i]) mod 2^16 over [begin, end). Wrap mul (low word), bit-identical for int16 and uint16.
     * @brief AVX2 range mul (16-bit lanes).
     */
    void mul_i16_range_avx2(int16_t*, const int16_t*, const int16_t*, size_t, size_t);

    /**
     * Computes out[i] = (a[i] + b[i]) mod 2^16 over [begin, end). Wrap add, bit-identical for int16 and uint16.
     * @brief AVX512 range add (16-bit lanes).
     */
    void add_i16_range_avx512(int16_t*, const int16_t*, const int16_t*, size_t, size_t);
    /**
     * Computes out[i] = (a[i] - b[i]) mod 2^16 over [begin, end). Wrap sub, bit-identical for int16 and uint16.
     * @brief AVX512 range sub (16-bit lanes).
     */
    void sub_i16_range_avx512(int16_t*, const int16_t*, const int16_t*, size_t, size_t);
    /**
     * Computes out[i] = (a[i] * b[i]) mod 2^16 over [begin, end). Wrap mul (low word), bit-identical for int16 and uint16.
     * @brief AVX512 range mul (16-bit lanes).
     */
    void mul_i16_range_avx512(int16_t*, const int16_t*, const int16_t*, size_t, size_t);

    /**
     * Computes out[i] = (a[i] + b[i]) mod 2^32 over [begin, end). Wrap add, bit-identical for int32 and uint32.
     * @brief AVX2 range add (32-bit lanes).
     */
    void add_i32_range_avx2(int32_t*, const int32_t*, const int32_t*, size_t, size_t);
    /**
     * Computes out[i] = (a[i] - b[i]) mod 2^32 over [begin, end). Wrap sub, bit-identical for int32 and uint32.
     * @brief AVX2 range sub (32-bit lanes).
     */
    void sub_i32_range_avx2(int32_t*, const int32_t*, const int32_t*, size_t, size_t);
    /**
     * Computes out[i] = (a[i] * b[i]) mod 2^32 over [begin, end). Wrap mul (low dword), bit-identical for int32 and uint32.
     * @brief AVX2 range mul (32-bit lanes).
     */
    void mul_i32_range_avx2(int32_t*, const int32_t*, const int32_t*, size_t, size_t);

    /**
     * Computes out[i] = (a[i] + b[i]) mod 2^32 over [begin, end). Wrap add, bit-identical for int32 and uint32.
     * @brief AVX512 range add (32-bit lanes).
     */
    void add_i32_range_avx512(int32_t*, const int32_t*, const int32_t*, size_t, size_t);
    /**
     * Computes out[i] = (a[i] - b[i]) mod 2^32 over [begin, end). Wrap sub, bit-identical for int32 and uint32.
     * @brief AVX512 range sub (32-bit lanes).
     */
    void sub_i32_range_avx512(int32_t*, const int32_t*, const int32_t*, size_t, size_t);
    /**
     * Computes out[i] = (a[i] * b[i]) mod 2^32 over [begin, end). Wrap mul (low dword), bit-identical for int32 and uint32.
     * @brief AVX512 range mul (32-bit lanes).
     */
    void mul_i32_range_avx512(int32_t*, const int32_t*, const int32_t*, size_t, size_t);

    /**
     * Computes out[i] = (a[i] + b[i]) mod 2^64 over [begin, end). Wrap add, bit-identical for int64 and uint64.
     * @brief AVX2 range add (64-bit lanes).
     */
    void add_i64_range_avx2(int64_t*, const int64_t*, const int64_t*, size_t, size_t);
    /**
     * Computes out[i] = (a[i] - b[i]) mod 2^64 over [begin, end). Wrap sub, bit-identical for int64 and uint64.
     * @brief AVX2 range sub (64-bit lanes).
     */
    void sub_i64_range_avx2(int64_t*, const int64_t*, const int64_t*, size_t, size_t);
    /**
     * Computes out[i] = (a[i] * b[i]) mod 2^64 over [begin, end). Wrap mul (low qword,
     * _mm256_mul_epu32 pairs — the low half is sign-agnostic), bit-identical for int64 and uint64.
     * @brief AVX2 range mul (64-bit lanes).
     */
    void mul_i64_range_avx2(int64_t*, const int64_t*, const int64_t*, size_t, size_t);

    /**
     * Computes out[i] = (a[i] + b[i]) mod 2^64 over [begin, end). Wrap add, bit-identical for int64 and uint64.
     * @brief AVX512 range add (64-bit lanes).
     */
    void add_i64_range_avx512(int64_t*, const int64_t*, const int64_t*, size_t, size_t);
    /**
     * Computes out[i] = (a[i] - b[i]) mod 2^64 over [begin, end). Wrap sub, bit-identical for int64 and uint64.
     * @brief AVX512 range sub (64-bit lanes).
     */
    void sub_i64_range_avx512(int64_t*, const int64_t*, const int64_t*, size_t, size_t);
    /**
     * Computes out[i] = (a[i] * b[i]) mod 2^64 over [begin, end). Wrap mul (low qword,
     * _mm512_mullo_epi64 under -mavx512dq), bit-identical for int64 and uint64.
     * @brief AVX512 range mul (64-bit lanes).
     */
    void mul_i64_range_avx512(int64_t*, const int64_t*, const int64_t*, size_t, size_t);

    // ==================== tensor-scalar splat kernels ====================
    // The scalar arrives ALREADY narrowed into the element type (the caller
    // applies the narrow_scalar PERMISSIVE contract exactly once); the kernel
    // broadcasts it into a vector register and runs the same loop shape as
    // the binary kernels above. Op set matches the binary kernels: fp
    // add/sub/mul/div, integer wrap add/sub/mul (no integer div, no Pow).

    // --------------------------- fp32 splat ---------------------------

    /**
     * Computes out[i] = a[i] (op) s over [begin, end), s splat into every lane. IEEE
     * single-rounding op, bit-identical to the scalar apply with the narrowed scalar.
     * @brief AVX2/AVX512 range splat (fp32).
     */
    void add_f32_splat_avx2(float32_t*, const float32_t*, float32_t, size_t, size_t);
    /**
     * Computes out[i] = a[i] (op) s over [begin, end), s splat into every lane. IEEE
     * single-rounding op, bit-identical to the scalar apply with the narrowed scalar.
     * @brief AVX2/AVX512 range splat (fp32).
     */
    void sub_f32_splat_avx2(float32_t*, const float32_t*, float32_t, size_t, size_t);
    /**
     * Computes out[i] = a[i] (op) s over [begin, end), s splat into every lane. IEEE
     * single-rounding op, bit-identical to the scalar apply with the narrowed scalar.
     * @brief AVX2/AVX512 range splat (fp32).
     */
    void mul_f32_splat_avx2(float32_t*, const float32_t*, float32_t, size_t, size_t);
    /**
     * Computes out[i] = a[i] (op) s over [begin, end), s splat into every lane. IEEE
     * div (x/0 -> inf/nan), bit-identical to the scalar apply with the narrowed scalar.
     * @brief AVX2/AVX512 range splat (fp32).
     */
    void div_f32_splat_avx2(float32_t*, const float32_t*, float32_t, size_t, size_t);

    /**
     * Computes out[i] = a[i] (op) s over [begin, end), s splat into every lane. IEEE
     * single-rounding op, bit-identical to the scalar apply with the narrowed scalar.
     * @brief AVX512 range splat (fp32).
     */
    void add_f32_splat_avx512(float32_t*, const float32_t*, float32_t, size_t, size_t);
    /**
     * Computes out[i] = a[i] (op) s over [begin, end), s splat into every lane. IEEE
     * single-rounding op, bit-identical to the scalar apply with the narrowed scalar.
     * @brief AVX512 range splat (fp32).
     */
    void sub_f32_splat_avx512(float32_t*, const float32_t*, float32_t, size_t, size_t);
    /**
     * Computes out[i] = a[i] (op) s over [begin, end), s splat into every lane. IEEE
     * single-rounding op, bit-identical to the scalar apply with the narrowed scalar.
     * @brief AVX512 range splat (fp32).
     */
    void mul_f32_splat_avx512(float32_t*, const float32_t*, float32_t, size_t, size_t);
    /**
     * Computes out[i] = a[i] (op) s over [begin, end), s splat into every lane. IEEE
     * div (x/0 -> inf/nan), bit-identical to the scalar apply with the narrowed scalar.
     * @brief AVX512 range splat (fp32).
     */
    void div_f32_splat_avx512(float32_t*, const float32_t*, float32_t, size_t, size_t);

    // --------------------------- fp64 splat ---------------------------

    /**
     * Computes out[i] = a[i] (op) s over [begin, end), s splat into every lane. IEEE
     * single-rounding op, bit-identical to the scalar apply with the narrowed scalar.
     * @brief AVX2 range splat (fp64).
     */
    void add_f64_splat_avx2(float64_t*, const float64_t*, float64_t, size_t, size_t);
    /**
     * Computes out[i] = a[i] (op) s over [begin, end), s splat into every lane. IEEE
     * single-rounding op, bit-identical to the scalar apply with the narrowed scalar.
     * @brief AVX2 range splat (fp64).
     */
    void sub_f64_splat_avx2(float64_t*, const float64_t*, float64_t, size_t, size_t);
    /**
     * Computes out[i] = a[i] (op) s over [begin, end), s splat into every lane. IEEE
     * single-rounding op, bit-identical to the scalar apply with the narrowed scalar.
     * @brief AVX2 range splat (fp64).
     */
    void mul_f64_splat_avx2(float64_t*, const float64_t*, float64_t, size_t, size_t);
    /**
     * Computes out[i] = a[i] (op) s over [begin, end), s splat into every lane. IEEE
     * div (x/0 -> inf/nan), bit-identical to the scalar apply with the narrowed scalar.
     * @brief AVX2 range splat (fp64).
     */
    void div_f64_splat_avx2(float64_t*, const float64_t*, float64_t, size_t, size_t);

    /**
     * Computes out[i] = a[i] (op) s over [begin, end), s splat into every lane. IEEE
     * single-rounding op, bit-identical to the scalar apply with the narrowed scalar.
     * @brief AVX512 range splat (fp64).
     */
    void add_f64_splat_avx512(float64_t*, const float64_t*, float64_t, size_t, size_t);
    /**
     * Computes out[i] = a[i] (op) s over [begin, end), s splat into every lane. IEEE
     * single-rounding op, bit-identical to the scalar apply with the narrowed scalar.
     * @brief AVX512 range splat (fp64).
     */
    void sub_f64_splat_avx512(float64_t*, const float64_t*, float64_t, size_t, size_t);
    /**
     * Computes out[i] = a[i] (op) s over [begin, end), s splat into every lane. IEEE
     * single-rounding op, bit-identical to the scalar apply with the narrowed scalar.
     * @brief AVX512 range splat (fp64).
     */
    void mul_f64_splat_avx512(float64_t*, const float64_t*, float64_t, size_t, size_t);
    /**
     * Computes out[i] = a[i] (op) s over [begin, end), s splat into every lane. IEEE
     * div (x/0 -> inf/nan), bit-identical to the scalar apply with the narrowed scalar.
     * @brief AVX512 range splat (fp64).
     */
    void div_f64_splat_avx512(float64_t*, const float64_t*, float64_t, size_t, size_t);

    // ------------- fp16 lane splats (Float16 AND Float16x2 buffers) -------------

    /**
     * Per-lane: widen fp16->fp32, op against the splat widened scalar, narrow RNE
     * with the scalar overflow clamp (finite |x| > 65504 -> +/-65504) over
     * [begin, end) flat 16-bit lanes. The scalar is one 16-bit lane bit pattern
     * (for Float16x2 the caller narrows it into both lanes, so any lane works).
     * Bit-identical to the scalar widen/apply/narrow apply with the narrowed scalar.
     * @brief AVX2 range splat (fp16 lanes).
     */
    void add_f16_splat_avx2(uint16_t*, const uint16_t*, uint16_t, size_t, size_t);
    /**
     * Per-lane: widen fp16->fp32, op against the splat widened scalar, narrow RNE
     * with the scalar overflow clamp over [begin, end) flat 16-bit lanes. Bit-identical
     * to the scalar widen/apply/narrow apply with the narrowed scalar.
     * @brief AVX2 range splat (fp16 lanes).
     */
    void sub_f16_splat_avx2(uint16_t*, const uint16_t*, uint16_t, size_t, size_t);
    /**
     * Per-lane: widen fp16->fp32, op against the splat widened scalar, narrow RNE
     * with the scalar overflow clamp over [begin, end) flat 16-bit lanes. Bit-identical
     * to the scalar widen/apply/narrow apply with the narrowed scalar.
     * @brief AVX2 range splat (fp16 lanes).
     */
    void mul_f16_splat_avx2(uint16_t*, const uint16_t*, uint16_t, size_t, size_t);
    /**
     * Per-lane: widen fp16->fp32, op against the splat widened scalar, narrow RNE
     * with the scalar overflow clamp over [begin, end) flat 16-bit lanes. Bit-identical
     * to the scalar widen/apply/narrow apply with the narrowed scalar.
     * @brief AVX2 range splat (fp16 lanes).
     */
    void div_f16_splat_avx2(uint16_t*, const uint16_t*, uint16_t, size_t, size_t);

    /**
     * Per-lane: widen fp16->fp32, op against the splat widened scalar, narrow RNE
     * with the scalar overflow clamp over [begin, end) flat 16-bit lanes. Bit-identical
     * to the scalar widen/apply/narrow apply with the narrowed scalar.
     * @brief AVX512 range splat (fp16 lanes).
     */
    void add_f16_splat_avx512(uint16_t*, const uint16_t*, uint16_t, size_t, size_t);
    /**
     * Per-lane: widen fp16->fp32, op against the splat widened scalar, narrow RNE
     * with the scalar overflow clamp over [begin, end) flat 16-bit lanes. Bit-identical
     * to the scalar widen/apply/narrow apply with the narrowed scalar.
     * @brief AVX512 range splat (fp16 lanes).
     */
    void sub_f16_splat_avx512(uint16_t*, const uint16_t*, uint16_t, size_t, size_t);
    /**
     * Per-lane: widen fp16->fp32, op against the splat widened scalar, narrow RNE
     * with the scalar overflow clamp over [begin, end) flat 16-bit lanes. Bit-identical
     * to the scalar widen/apply/narrow apply with the narrowed scalar.
     * @brief AVX512 range splat (fp16 lanes).
     */
    void mul_f16_splat_avx512(uint16_t*, const uint16_t*, uint16_t, size_t, size_t);
    /**
     * Per-lane: widen fp16->fp32, op against the splat widened scalar, narrow RNE
     * with the scalar overflow clamp over [begin, end) flat 16-bit lanes. Bit-identical
     * to the scalar widen/apply/narrow apply with the narrowed scalar.
     * @brief AVX512 range splat (fp16 lanes).
     */
    void div_f16_splat_avx512(uint16_t*, const uint16_t*, uint16_t, size_t, size_t);

    // ------------- bf16 lane splats (BFloat16 AND BFloat16x2 buffers) -------------

    /**
     * Per-lane: widen bf16->fp32 (16-bit shift), op against the splat widened scalar,
     * narrow RNE with the scalar special-case semantics (NaN quieting, fp32-subnormal
     * flush-to-zero) over [begin, end) flat 16-bit lanes. Bit-identical to the scalar
     * apply with the narrowed scalar.
     * @brief AVX2 range splat (bf16 lanes).
     */
    void add_bf16_splat_avx2(uint16_t*, const uint16_t*, uint16_t, size_t, size_t);
    /**
     * Per-lane: widen bf16->fp32, op against the splat widened scalar, narrow RNE with
     * the scalar special-case semantics over [begin, end) flat 16-bit lanes.
     * Bit-identical to the scalar apply with the narrowed scalar.
     * @brief AVX2 range splat (bf16 lanes).
     */
    void sub_bf16_splat_avx2(uint16_t*, const uint16_t*, uint16_t, size_t, size_t);
    /**
     * Per-lane: widen bf16->fp32, op against the splat widened scalar, narrow RNE with
     * the scalar special-case semantics over [begin, end) flat 16-bit lanes.
     * Bit-identical to the scalar apply with the narrowed scalar.
     * @brief AVX2 range splat (bf16 lanes).
     */
    void mul_bf16_splat_avx2(uint16_t*, const uint16_t*, uint16_t, size_t, size_t);
    /**
     * Per-lane: widen bf16->fp32, op against the splat widened scalar, narrow RNE with
     * the scalar special-case semantics over [begin, end) flat 16-bit lanes.
     * Bit-identical to the scalar apply with the narrowed scalar.
     * @brief AVX2 range splat (bf16 lanes).
     */
    void div_bf16_splat_avx2(uint16_t*, const uint16_t*, uint16_t, size_t, size_t);

    /**
     * Per-lane: widen bf16->fp32, op against the splat widened scalar, narrow RNE via
     * _mm512_cvtneps_pbh (hardware matches the scalar NaN/flush/RNE semantics) over
     * [begin, end) flat 16-bit lanes. Bit-identical to the scalar apply with the narrowed scalar.
     * @brief AVX512 range splat (bf16 lanes).
     */
    void add_bf16_splat_avx512(uint16_t*, const uint16_t*, uint16_t, size_t, size_t);
    /**
     * Per-lane: widen bf16->fp32, op against the splat widened scalar, narrow RNE via
     * _mm512_cvtneps_pbh over [begin, end) flat 16-bit lanes. Bit-identical to the
     * scalar apply with the narrowed scalar.
     * @brief AVX512 range splat (bf16 lanes).
     */
    void sub_bf16_splat_avx512(uint16_t*, const uint16_t*, uint16_t, size_t, size_t);
    /**
     * Per-lane: widen bf16->fp32, op against the splat widened scalar, narrow RNE via
     * _mm512_cvtneps_pbh over [begin, end) flat 16-bit lanes. Bit-identical to the
     * scalar apply with the narrowed scalar.
     * @brief AVX512 range splat (bf16 lanes).
     */
    void mul_bf16_splat_avx512(uint16_t*, const uint16_t*, uint16_t, size_t, size_t);
    /**
     * Per-lane: widen bf16->fp32, op against the splat widened scalar, narrow RNE via
     * _mm512_cvtneps_pbh over [begin, end) flat 16-bit lanes. Bit-identical to the
     * scalar apply with the narrowed scalar.
     * @brief AVX512 range splat (bf16 lanes).
     */
    void div_bf16_splat_avx512(uint16_t*, const uint16_t*, uint16_t, size_t, size_t);

    // --------------------------- integer splats ---------------------------
    // Wrap (modular) add/sub/mul only — one kernel per width serves the signed
    // AND unsigned dtype (the bit patterns of +,-,* are sign-agnostic). Integer
    // div stays on the scalar path: no x86 integer division, and the
    // div-by-zero -> 0 / min-over-(-1) contract is scalar-defined.

    /**
     * Computes out[i] = (a[i] (op) s) mod 2^8 over [begin, end), s splat into every
     * lane. Wrap arithmetic, bit-identical for int8 and uint8.
     * @brief AVX2/AVX512 range splat (8-bit lanes).
     */
    void add_i8_splat_avx2(int8_t*, const int8_t*, int8_t, size_t, size_t);
    /**
     * Computes out[i] = (a[i] (op) s) mod 2^8 over [begin, end), s splat into every
     * lane. Wrap arithmetic, bit-identical for int8 and uint8.
     * @brief AVX2/AVX512 range splat (8-bit lanes).
     */
    void sub_i8_splat_avx2(int8_t*, const int8_t*, int8_t, size_t, size_t);
    /**
     * Computes out[i] = (a[i] (op) s) mod 2^8 over [begin, end), s splat into every
     * lane. Wrap arithmetic, bit-identical for int8 and uint8.
     * @brief AVX2/AVX512 range splat (8-bit lanes).
     */
    void mul_i8_splat_avx2(int8_t*, const int8_t*, int8_t, size_t, size_t);

    /**
     * Computes out[i] = (a[i] (op) s) mod 2^8 over [begin, end), s splat into every
     * lane. Wrap arithmetic, bit-identical for int8 and uint8.
     * @brief AVX512 range splat (8-bit lanes).
     */
    void add_i8_splat_avx512(int8_t*, const int8_t*, int8_t, size_t, size_t);
    /**
     * Computes out[i] = (a[i] (op) s) mod 2^8 over [begin, end), s splat into every
     * lane. Wrap arithmetic, bit-identical for int8 and uint8.
     * @brief AVX512 range splat (8-bit lanes).
     */
    void sub_i8_splat_avx512(int8_t*, const int8_t*, int8_t, size_t, size_t);
    /**
     * Computes out[i] = (a[i] (op) s) mod 2^8 over [begin, end), s splat into every
     * lane. Wrap arithmetic, bit-identical for int8 and uint8.
     * @brief AVX512 range splat (8-bit lanes).
     */
    void mul_i8_splat_avx512(int8_t*, const int8_t*, int8_t, size_t, size_t);

    /**
     * Computes out[i] = (a[i] (op) s) mod 2^16 over [begin, end), s splat into every
     * lane. Wrap arithmetic, bit-identical for int16 and uint16.
     * @brief AVX2 range splat (16-bit lanes).
     */
    void add_i16_splat_avx2(int16_t*, const int16_t*, int16_t, size_t, size_t);
    /**
     * Computes out[i] = (a[i] (op) s) mod 2^16 over [begin, end), s splat into every
     * lane. Wrap arithmetic, bit-identical for int16 and uint16.
     * @brief AVX2 range splat (16-bit lanes).
     */
    void sub_i16_splat_avx2(int16_t*, const int16_t*, int16_t, size_t, size_t);
    /**
     * Computes out[i] = (a[i] (op) s) mod 2^16 over [begin, end), s splat into every
     * lane. Wrap arithmetic, bit-identical for int16 and uint16.
     * @brief AVX2 range splat (16-bit lanes).
     */
    void mul_i16_splat_avx2(int16_t*, const int16_t*, int16_t, size_t, size_t);

    /**
     * Computes out[i] = (a[i] (op) s) mod 2^16 over [begin, end), s splat into every
     * lane. Wrap arithmetic, bit-identical for int16 and uint16.
     * @brief AVX512 range splat (16-bit lanes).
     */
    void add_i16_splat_avx512(int16_t*, const int16_t*, int16_t, size_t, size_t);
    /**
     * Computes out[i] = (a[i] (op) s) mod 2^16 over [begin, end), s splat into every
     * lane. Wrap arithmetic, bit-identical for int16 and uint16.
     * @brief AVX512 range splat (16-bit lanes).
     */
    void sub_i16_splat_avx512(int16_t*, const int16_t*, int16_t, size_t, size_t);
    /**
     * Computes out[i] = (a[i] (op) s) mod 2^16 over [begin, end), s splat into every
     * lane. Wrap arithmetic, bit-identical for int16 and uint16.
     * @brief AVX512 range splat (16-bit lanes).
     */
    void mul_i16_splat_avx512(int16_t*, const int16_t*, int16_t, size_t, size_t);

    /**
     * Computes out[i] = (a[i] (op) s) mod 2^32 over [begin, end), s splat into every
     * lane. Wrap arithmetic, bit-identical for int32 and uint32.
     * @brief AVX2 range splat (32-bit lanes).
     */
    void add_i32_splat_avx2(int32_t*, const int32_t*, int32_t, size_t, size_t);
    /**
     * Computes out[i] = (a[i] (op) s) mod 2^32 over [begin, end), s splat into every
     * lane. Wrap arithmetic, bit-identical for int32 and uint32.
     * @brief AVX2 range splat (32-bit lanes).
     */
    void sub_i32_splat_avx2(int32_t*, const int32_t*, int32_t, size_t, size_t);
    /**
     * Computes out[i] = (a[i] (op) s) mod 2^32 over [begin, end), s splat into every
     * lane. Wrap arithmetic, bit-identical for int32 and uint32.
     * @brief AVX2 range splat (32-bit lanes).
     */
    void mul_i32_splat_avx2(int32_t*, const int32_t*, int32_t, size_t, size_t);

    /**
     * Computes out[i] = (a[i] (op) s) mod 2^32 over [begin, end), s splat into every
     * lane. Wrap arithmetic, bit-identical for int32 and uint32.
     * @brief AVX512 range splat (32-bit lanes).
     */
    void add_i32_splat_avx512(int32_t*, const int32_t*, int32_t, size_t, size_t);
    /**
     * Computes out[i] = (a[i] (op) s) mod 2^32 over [begin, end), s splat into every
     * lane. Wrap arithmetic, bit-identical for int32 and uint32.
     * @brief AVX512 range splat (32-bit lanes).
     */
    void sub_i32_splat_avx512(int32_t*, const int32_t*, int32_t, size_t, size_t);
    /**
     * Computes out[i] = (a[i] (op) s) mod 2^32 over [begin, end), s splat into every
     * lane. Wrap arithmetic, bit-identical for int32 and uint32.
     * @brief AVX512 range splat (32-bit lanes).
     */
    void mul_i32_splat_avx512(int32_t*, const int32_t*, int32_t, size_t, size_t);

    /**
     * Computes out[i] = (a[i] (op) s) mod 2^64 over [begin, end), s splat into every
     * lane. Wrap arithmetic, bit-identical for int64 and uint64.
     * @brief AVX2 range splat (64-bit lanes).
     */
    void add_i64_splat_avx2(int64_t*, const int64_t*, int64_t, size_t, size_t);
    /**
     * Computes out[i] = (a[i] (op) s) mod 2^64 over [begin, end), s splat into every
     * lane. Wrap arithmetic, bit-identical for int64 and uint64.
     * @brief AVX2 range splat (64-bit lanes).
     */
    void sub_i64_splat_avx2(int64_t*, const int64_t*, int64_t, size_t, size_t);
    /**
     * Computes out[i] = (a[i] (op) s) mod 2^64 over [begin, end), s splat into every
     * lane. Wrap mul (low qword, _mm256_mul_epu32 pairs — sign-agnostic).
     * @brief AVX2 range splat (64-bit lanes).
     */
    void mul_i64_splat_avx2(int64_t*, const int64_t*, int64_t, size_t, size_t);

    /**
     * Computes out[i] = (a[i] (op) s) mod 2^64 over [begin, end), s splat into every
     * lane. Wrap arithmetic, bit-identical for int64 and uint64.
     * @brief AVX512 range splat (64-bit lanes).
     */
    void add_i64_splat_avx512(int64_t*, const int64_t*, int64_t, size_t, size_t);
    /**
     * Computes out[i] = (a[i] (op) s) mod 2^64 over [begin, end), s splat into every
     * lane. Wrap arithmetic, bit-identical for int64 and uint64.
     * @brief AVX512 range splat (64-bit lanes).
     */
    void sub_i64_splat_avx512(int64_t*, const int64_t*, int64_t, size_t, size_t);
    /**
     * Computes out[i] = (a[i] (op) s) mod 2^64 over [begin, end), s splat into every
     * lane. Wrap mul (low qword, _mm512_mullo_epi64 under -mavx512dq).
     * @brief AVX512 range splat (64-bit lanes).
     */
    void mul_i64_splat_avx512(int64_t*, const int64_t*, int64_t, size_t, size_t);

    // ------------------- reverse (scalar-on-the-left) splats -------------------
    // Broadcast rows where operand A has inner stride 0 need out[i] = s (op) b[i]
    // — the scalar on the LEFT. The plain splat kernels put the scalar on the
    // right, which is a different operation for sub/div (and selects a different
    // NaN payload for commutative fp ops), so these kernels replay the exact
    // scalar operand order. Integer add/mul are bitwise commutative under wrap,
    // so only integer sub needs a left variant. Div-by-zero/min-over-(-1) stays
    // scalar (no x86 integer div), as does Pow.

    /**
     * Computes out[i] = s (op) a[i] over [begin, end) — the scalar on the LEFT,
     * matching the scalar apply's operand order bit-for-bit (incl. NaN payload
     * selection). Same lane widths and tail semantics as the plain splat kernels.
     * @brief AVX2/AVX512 left splat (fp32).
     */
    void add_f32_splat_lhs_avx2(float32_t*, const float32_t*, float32_t, size_t, size_t);
    /**
     * Computes out[i] = s (op) a[i] over [begin, end) — the scalar on the LEFT.
     * @brief AVX2/AVX512 left splat (fp32).
     */
    void sub_f32_splat_lhs_avx2(float32_t*, const float32_t*, float32_t, size_t, size_t);
    /**
     * Computes out[i] = s (op) a[i] over [begin, end) — the scalar on the LEFT.
     * @brief AVX2/AVX512 left splat (fp32).
     */
    void mul_f32_splat_lhs_avx2(float32_t*, const float32_t*, float32_t, size_t, size_t);
    /**
     * Computes out[i] = s (op) a[i] over [begin, end) — the scalar on the LEFT.
     * @brief AVX2/AVX512 left splat (fp32).
     */
    void div_f32_splat_lhs_avx2(float32_t*, const float32_t*, float32_t, size_t, size_t);

    /**
     * Computes out[i] = s (op) a[i] over [begin, end) — the scalar on the LEFT.
     * @brief AVX512 left splat (fp32).
     */
    void add_f32_splat_lhs_avx512(float32_t*, const float32_t*, float32_t, size_t, size_t);
    /**
     * Computes out[i] = s (op) a[i] over [begin, end) — the scalar on the LEFT.
     * @brief AVX512 left splat (fp32).
     */
    void sub_f32_splat_lhs_avx512(float32_t*, const float32_t*, float32_t, size_t, size_t);
    /**
     * Computes out[i] = s (op) a[i] over [begin, end) — the scalar on the LEFT.
     * @brief AVX512 left splat (fp32).
     */
    void mul_f32_splat_lhs_avx512(float32_t*, const float32_t*, float32_t, size_t, size_t);
    /**
     * Computes out[i] = s (op) a[i] over [begin, end) — the scalar on the LEFT.
     * @brief AVX512 left splat (fp32).
     */
    void div_f32_splat_lhs_avx512(float32_t*, const float32_t*, float32_t, size_t, size_t);

    /**
     * Computes out[i] = s (op) a[i] over [begin, end) — the scalar on the LEFT.
     * @brief AVX2 left splat (fp64).
     */
    void add_f64_splat_lhs_avx2(float64_t*, const float64_t*, float64_t, size_t, size_t);
    /**
     * Computes out[i] = s (op) a[i] over [begin, end) — the scalar on the LEFT.
     * @brief AVX2 left splat (fp64).
     */
    void sub_f64_splat_lhs_avx2(float64_t*, const float64_t*, float64_t, size_t, size_t);
    /**
     * Computes out[i] = s (op) a[i] over [begin, end) — the scalar on the LEFT.
     * @brief AVX2 left splat (fp64).
     */
    void mul_f64_splat_lhs_avx2(float64_t*, const float64_t*, float64_t, size_t, size_t);
    /**
     * Computes out[i] = s (op) a[i] over [begin, end) — the scalar on the LEFT.
     * @brief AVX2 left splat (fp64).
     */
    void div_f64_splat_lhs_avx2(float64_t*, const float64_t*, float64_t, size_t, size_t);

    /**
     * Computes out[i] = s (op) a[i] over [begin, end) — the scalar on the LEFT.
     * @brief AVX512 left splat (fp64).
     */
    void add_f64_splat_lhs_avx512(float64_t*, const float64_t*, float64_t, size_t, size_t);
    /**
     * Computes out[i] = s (op) a[i] over [begin, end) — the scalar on the LEFT.
     * @brief AVX512 left splat (fp64).
     */
    void sub_f64_splat_lhs_avx512(float64_t*, const float64_t*, float64_t, size_t, size_t);
    /**
     * Computes out[i] = s (op) a[i] over [begin, end) — the scalar on the LEFT.
     * @brief AVX512 left splat (fp64).
     */
    void mul_f64_splat_lhs_avx512(float64_t*, const float64_t*, float64_t, size_t, size_t);
    /**
     * Computes out[i] = s (op) a[i] over [begin, end) — the scalar on the LEFT.
     * @brief AVX512 left splat (fp64).
     */
    void div_f64_splat_lhs_avx512(float64_t*, const float64_t*, float64_t, size_t, size_t);

    /**
     * Per-lane: widen fp16->fp32, op with the scalar on the LEFT (s op a[i]),
     * narrow RNE with the scalar overflow clamp over [begin, end) flat lanes.
     * @brief AVX2 left splat (fp16 lanes).
     */
    void add_f16_splat_lhs_avx2(uint16_t*, const uint16_t*, uint16_t, size_t, size_t);
    /**
     * Per-lane: widen fp16->fp32, op with the scalar on the LEFT, narrow RNE with
     * the scalar overflow clamp over [begin, end) flat lanes.
     * @brief AVX2 left splat (fp16 lanes).
     */
    void sub_f16_splat_lhs_avx2(uint16_t*, const uint16_t*, uint16_t, size_t, size_t);
    /**
     * Per-lane: widen fp16->fp32, op with the scalar on the LEFT, narrow RNE with
     * the scalar overflow clamp over [begin, end) flat lanes.
     * @brief AVX2 left splat (fp16 lanes).
     */
    void mul_f16_splat_lhs_avx2(uint16_t*, const uint16_t*, uint16_t, size_t, size_t);
    /**
     * Per-lane: widen fp16->fp32, op with the scalar on the LEFT, narrow RNE with
     * the scalar overflow clamp over [begin, end) flat lanes.
     * @brief AVX2 left splat (fp16 lanes).
     */
    void div_f16_splat_lhs_avx2(uint16_t*, const uint16_t*, uint16_t, size_t, size_t);

    /**
     * Per-lane: widen fp16->fp32, op with the scalar on the LEFT, narrow RNE with
     * the scalar overflow clamp over [begin, end) flat lanes.
     * @brief AVX512 left splat (fp16 lanes).
     */
    void add_f16_splat_lhs_avx512(uint16_t*, const uint16_t*, uint16_t, size_t, size_t);
    /**
     * Per-lane: widen fp16->fp32, op with the scalar on the LEFT, narrow RNE with
     * the scalar overflow clamp over [begin, end) flat lanes.
     * @brief AVX512 left splat (fp16 lanes).
     */
    void sub_f16_splat_lhs_avx512(uint16_t*, const uint16_t*, uint16_t, size_t, size_t);
    /**
     * Per-lane: widen fp16->fp32, op with the scalar on the LEFT, narrow RNE with
     * the scalar overflow clamp over [begin, end) flat lanes.
     * @brief AVX512 left splat (fp16 lanes).
     */
    void mul_f16_splat_lhs_avx512(uint16_t*, const uint16_t*, uint16_t, size_t, size_t);
    /**
     * Per-lane: widen fp16->fp32, op with the scalar on the LEFT, narrow RNE with
     * the scalar overflow clamp over [begin, end) flat lanes.
     * @brief AVX512 left splat (fp16 lanes).
     */
    void div_f16_splat_lhs_avx512(uint16_t*, const uint16_t*, uint16_t, size_t, size_t);

    /**
     * Per-lane: widen bf16->fp32, op with the scalar on the LEFT (s op a[i]), narrow
     * RNE with the scalar special-case semantics over [begin, end) flat lanes.
     * @brief AVX2 left splat (bf16 lanes).
     */
    void add_bf16_splat_lhs_avx2(uint16_t*, const uint16_t*, uint16_t, size_t, size_t);
    /**
     * Per-lane: widen bf16->fp32, op with the scalar on the LEFT, narrow RNE with
     * the scalar special-case semantics over [begin, end) flat lanes.
     * @brief AVX2 left splat (bf16 lanes).
     */
    void sub_bf16_splat_lhs_avx2(uint16_t*, const uint16_t*, uint16_t, size_t, size_t);
    /**
     * Per-lane: widen bf16->fp32, op with the scalar on the LEFT, narrow RNE with
     * the scalar special-case semantics over [begin, end) flat lanes.
     * @brief AVX2 left splat (bf16 lanes).
     */
    void mul_bf16_splat_lhs_avx2(uint16_t*, const uint16_t*, uint16_t, size_t, size_t);
    /**
     * Per-lane: widen bf16->fp32, op with the scalar on the LEFT, narrow RNE with
     * the scalar special-case semantics over [begin, end) flat lanes.
     * @brief AVX2 left splat (bf16 lanes).
     */
    void div_bf16_splat_lhs_avx2(uint16_t*, const uint16_t*, uint16_t, size_t, size_t);

    /**
     * Per-lane: widen bf16->fp32, op with the scalar on the LEFT, narrow RNE via
     * _mm512_cvtneps_pbh over [begin, end) flat lanes.
     * @brief AVX512 left splat (bf16 lanes).
     */
    void add_bf16_splat_lhs_avx512(uint16_t*, const uint16_t*, uint16_t, size_t, size_t);
    /**
     * Per-lane: widen bf16->fp32, op with the scalar on the LEFT, narrow RNE via
     * _mm512_cvtneps_pbh over [begin, end) flat lanes.
     * @brief AVX512 left splat (bf16 lanes).
     */
    void sub_bf16_splat_lhs_avx512(uint16_t*, const uint16_t*, uint16_t, size_t, size_t);
    /**
     * Per-lane: widen bf16->fp32, op with the scalar on the LEFT, narrow RNE via
     * _mm512_cvtneps_pbh over [begin, end) flat lanes.
     * @brief AVX512 left splat (bf16 lanes).
     */
    void mul_bf16_splat_lhs_avx512(uint16_t*, const uint16_t*, uint16_t, size_t, size_t);
    /**
     * Per-lane: widen bf16->fp32, op with the scalar on the LEFT, narrow RNE via
     * _mm512_cvtneps_pbh over [begin, end) flat lanes.
     * @brief AVX512 left splat (bf16 lanes).
     */
    void div_bf16_splat_lhs_avx512(uint16_t*, const uint16_t*, uint16_t, size_t, size_t);

    // Integer left splats: only sub needs one — wrap add/mul are bitwise
    // commutative, so the plain splat kernels already match the scalar operand
    // order bit-for-bit.

    /**
     * Computes out[i] = (s - a[i]) mod 2^8 over [begin, end). Wrap sub with the
     * scalar on the LEFT, bit-identical for int8 and uint8.
     * @brief AVX2/AVX512 left splat sub (8-bit lanes).
     */
    void sub_i8_splat_lhs_avx2(int8_t*, const int8_t*, int8_t, size_t, size_t);
    /**
     * Computes out[i] = (s - a[i]) mod 2^16 over [begin, end). Wrap sub with the
     * scalar on the LEFT, bit-identical for int16 and uint16.
     * @brief AVX2 left splat sub (16-bit lanes).
     */
    void sub_i16_splat_lhs_avx2(int16_t*, const int16_t*, int16_t, size_t, size_t);
    /**
     * Computes out[i] = (s - a[i]) mod 2^32 over [begin, end). Wrap sub with the
     * scalar on the LEFT, bit-identical for int32 and uint32.
     * @brief AVX2 left splat sub (32-bit lanes).
     */
    void sub_i32_splat_lhs_avx2(int32_t*, const int32_t*, int32_t, size_t, size_t);
    /**
     * Computes out[i] = (s - a[i]) mod 2^64 over [begin, end). Wrap sub with the
     * scalar on the LEFT, bit-identical for int64 and uint64.
     * @brief AVX2 left splat sub (64-bit lanes).
     */
    void sub_i64_splat_lhs_avx2(int64_t*, const int64_t*, int64_t, size_t, size_t);

    /**
     * Computes out[i] = (s - a[i]) mod 2^8 over [begin, end). Wrap sub with the
     * scalar on the LEFT, bit-identical for int8 and uint8.
     * @brief AVX512 left splat sub (8-bit lanes).
     */
    void sub_i8_splat_lhs_avx512(int8_t*, const int8_t*, int8_t, size_t, size_t);
    /**
     * Computes out[i] = (s - a[i]) mod 2^16 over [begin, end). Wrap sub with the
     * scalar on the LEFT, bit-identical for int16 and uint16.
     * @brief AVX512 left splat sub (16-bit lanes).
     */
    void sub_i16_splat_lhs_avx512(int16_t*, const int16_t*, int16_t, size_t, size_t);
    /**
     * Computes out[i] = (s - a[i]) mod 2^32 over [begin, end). Wrap sub with the
     * scalar on the LEFT, bit-identical for int32 and uint32.
     * @brief AVX512 left splat sub (32-bit lanes).
     */
    void sub_i32_splat_lhs_avx512(int32_t*, const int32_t*, int32_t, size_t, size_t);
    /**
     * Computes out[i] = (s - a[i]) mod 2^64 over [begin, end). Wrap sub with the
     * scalar on the LEFT, bit-identical for int64 and uint64.
     * @brief AVX512 left splat sub (64-bit lanes).
     */
    void sub_i64_splat_lhs_avx512(int64_t*, const int64_t*, int64_t, size_t, size_t);

    // --------------------------- int4x2 (per-nibble) ---------------------------
    // Buffers are flat uint8_t arrays (one packed int4 pair per byte). Add/Sub/Mul
    // wrap per nibble: the scalar apply sign-extends each nibble, applies the op in
    // int32 and keeps the low 4 bits, which the unsigned-nibble identity
    // (a & 0xF) OP (b & 0xF) masked to 4 bits reproduces bit-for-bit regardless of
    // signedness. Div stays scalar (no x86 integer division + the per-nibble
    // div-by-zero -> 0 rule is scalar-defined); broadcast Int4x2 rows stay scalar
    // too (see ElementwiseCPU.cpp).

    /**
     * Per-nibble wrap op over [begin, end) packed bytes: unpack nibbles to byte lanes
     * (vpand 0x0F / vpsrlw 4 + vpand), byte-lane arithmetic, mask 0x0F, repack
     * (vpsllw 4 + vpor — 16-bit shifts stay within byte pairs, so no lane
     * interleaving). Bit-identical to the scalar apply_int4x2.
     * @brief AVX2 range add (int4x2, 32 packed bytes per step).
     */
    void add_i4x2_range_avx2(uint8_t*, const uint8_t*, const uint8_t*, size_t, size_t);
    /**
     * Per-nibble wrap op over [begin, end) packed bytes; bit-identical to the
     * scalar apply_int4x2.
     * @brief AVX2 range sub (int4x2, 32 packed bytes per step).
     */
    void sub_i4x2_range_avx2(uint8_t*, const uint8_t*, const uint8_t*, size_t, size_t);
    /**
     * Per-nibble wrap op over [begin, end) packed bytes; bit-identical to the
     * scalar apply_int4x2.
     * @brief AVX2 range mul (int4x2, 32 packed bytes per step).
     */
    void mul_i4x2_range_avx2(uint8_t*, const uint8_t*, const uint8_t*, size_t, size_t);

    /**
     * Per-nibble wrap op over [begin, end) packed bytes; bit-identical to the
     * scalar apply_int4x2.
     * @brief AVX512 range add (int4x2, 64 packed bytes per step).
     */
    void add_i4x2_range_avx512(uint8_t*, const uint8_t*, const uint8_t*, size_t, size_t);
    /**
     * Per-nibble wrap op over [begin, end) packed bytes; bit-identical to the
     * scalar apply_int4x2.
     * @brief AVX512 range sub (int4x2, 64 packed bytes per step).
     */
    void sub_i4x2_range_avx512(uint8_t*, const uint8_t*, const uint8_t*, size_t, size_t);
    /**
     * Per-nibble wrap op over [begin, end) packed bytes; bit-identical to the
     * scalar apply_int4x2.
     * @brief AVX512 range mul (int4x2, 64 packed bytes per step).
     */
    void mul_i4x2_range_avx512(uint8_t*, const uint8_t*, const uint8_t*, size_t, size_t);
}
