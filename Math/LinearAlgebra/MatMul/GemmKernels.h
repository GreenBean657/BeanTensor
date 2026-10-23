/**
 * @file GemmKernels.h
 * @author GreenBean
 * @brief GEMM kernels for different backends.
 * @version 0.1
 * @date 2026-10-18
 */
#pragma once

#include "Tensor/DTypes.h"

namespace BeanTensor::Math::LinearAlgebra {
    /**
     * @brief Runtime probe: is the AVX2+FMA GEMM kernel safe to execute?
     * @return True when the CPU supports AVX2 and FMA.
     */
    [[nodiscard]] bool gemm_avx2_available() noexcept;

    /**
     * @brief Runtime probe: is the AVX512F GEMM kernel safe to execute?
     * @return True when the CPU supports AVX512F.
     */
    [[nodiscard]] bool gemm_avx512_available() noexcept;

    /**
     * @brief Runtime probe: are the VDPBF16PS (bf16 dot-product) GEMM kernels
     * safe to execute? Requires AVX512BW for the 16-bit permutes as well.
     * @return True when the CPU supports AVX512BF16 and AVX512BW.
     */
    [[nodiscard]] bool gemm_avx512_bf16_available() noexcept;

    /**
     * @brief Runtime probe: are the F16C-widened fp16 GEMM kernels safe to
     * execute? Requires AVX512BW for the packed-lane permutes as well.
     * @return True when the CPU supports AVX512F, AVX512BW, and F16C.
     */
    [[nodiscard]] bool gemm_avx512_f16c_available() noexcept;

    /**
     * @brief Runtime probe: are the AVX512-VNNI integer dot-product GEMM
     * kernels safe to execute? Requires AVX512BW for byte/word permutes.
     * @return True when the CPU supports AVX512VNNI, AVX512F, and AVX512BW.
     */
    [[nodiscard]] bool gemm_avx512_vnni_available() noexcept;

    /**
     * @brief Runtime probe: are the AVX512DQ 64-bit integer GEMM kernels safe
     * to execute?
     * @return True when the CPU supports AVX512F and AVX512DQ.
     */
    [[nodiscard]] bool gemm_avx512_dq_available() noexcept;


    /**
     * @brief AVX2+FMA row-range GEMM (FP32 FMA accumulate).
     */
    void gemm_f32_rows_avx2(const float32_t*, const float32_t*, float32_t*, size_t, size_t, size_t, size_t);

    /**
     * @brief AVX2+FMA row-range GEMM (FP64 FMA accumulate).
     */
    void gemm_f64_rows_avx2(const float64_t*, const float64_t*, float64_t*, size_t, size_t, size_t, size_t);

    /**
     * @brief AVX2 row-range GEMM (int8_t A and B, wrapping 32-bit accumulate).
     */
    void gemm_i8_rows_avx2(const int8_t*, const int8_t*, int8_t*, size_t, size_t, size_t, size_t);

    /**
     * @brief AVX2 row-range GEMM (uint8_t A and B, wrapping 32-bit accumulate).
     */
    void gemm_u8_rows_avx2(const uint8_t*, const uint8_t*, uint8_t*, size_t, size_t, size_t, size_t);

    /**
     * @brief AVX2 row-range GEMM (int16_t A and B, wrapping 32-bit accumulate).
     */
    void gemm_i16_rows_avx2(const int16_t*, const int16_t*, int16_t*, size_t, size_t, size_t, size_t);

    /**
     * @brief AVX2 row-range GEMM (uint16_t A and B, wrapping 32-bit accumulate).
     */
    void gemm_u16_rows_avx2(const uint16_t*, const uint16_t*, uint16_t*, size_t, size_t, size_t, size_t);

    /**
     * @brief AVX2 row-range GEMM (int32_t elements, mod-2^32 arithmetic).
     */
    void gemm_i32_rows_avx2(const int32_t*, const int32_t*, int32_t*, size_t, size_t, size_t, size_t);

    /**
     * @brief AVX2 row-range GEMM (uint32_t elements, mod-2^32 arithmetic).
     */
    void gemm_u32_rows_avx2(const uint32_t*, const uint32_t*, uint32_t*, size_t, size_t, size_t, size_t);

    /**
     * @brief AVX2 row-range GEMM (int64_t elements, mod-2^64 arithmetic).
     */
    void gemm_i64_rows_avx2(const int64_t*, const int64_t*, int64_t*, size_t, size_t, size_t, size_t);

    /**
     * @brief AVX2 row-range GEMM (uint64_t elements, mod-2^64 arithmetic).
     */
    void gemm_u64_rows_avx2(const uint64_t*, const uint64_t*, uint64_t*, size_t, size_t, size_t, size_t);

    /**
     * @brief AVX512F row-range GEMM (FP32 FMA accumulate).
     */
    void gemm_f32_rows_avx512(const float32_t*, const float32_t*, float32_t*, size_t, size_t, size_t, size_t);

    /**
     * @brief AVX512F row-range GEMM (FP64 FMA accumulate).
     */
    void gemm_f64_rows_avx512(const float64_t*, const float64_t*, float64_t*, size_t, size_t, size_t, size_t);

    /**
     * @brief AVX512BF16 row-range GEMM (bf16, FP32 accumulate).
     */
    void gemm_bf16_rows_avx512bf16(const bfloat16_t*, const bfloat16_t*, bfloat16_t*, size_t, size_t, size_t, size_t);

    /**
     * @brief AVX512BF16 row-range GEMM (bf16x2, per-lane FP32 accumulate).
     */
    void gemm_bf16x2_rows_avx512bf16(const bfloat16x2_t*, const bfloat16x2_t*, bfloat16x2_t*, size_t, size_t, size_t, size_t);

    /**
     * @brief AVX512F+F16C row-range GEMM (fp16, FP32 accumulate).
     */
    void gemm_f16_rows_avx512f16c(const float16_t*, const float16_t*, float16_t*, size_t, size_t, size_t, size_t);

    /**
     * @brief AVX512F+F16C row-range GEMM (fp16x2, per-lane FP32 accumulate).
     */
    void gemm_f16x2_rows_avx512f16c(const float16x2_t*, const float16x2_t*, float16x2_t*, size_t, size_t, size_t, size_t);

    /**
     * @brief AVX512VNNI row-range GEMM (int8_t A and B, wrapping 32-bit accumulate).
     */
    void gemm_i8_rows_avx512vnni(const int8_t*, const int8_t*, int8_t*, size_t, size_t, size_t, size_t);

    /**
     * @brief AVX512VNNI row-range GEMM (uint8_t A and B, wrapping 32-bit accumulate).
     */
    void gemm_u8_rows_avx512vnni(const uint8_t*, const uint8_t*, uint8_t*, size_t, size_t, size_t, size_t);

    /**
     * @brief AVX512VNNI row-range GEMM (int16_t A and B, wrapping 32-bit accumulate).
     */
    void gemm_i16_rows_avx512vnni(const int16_t*, const int16_t*, int16_t*, size_t, size_t, size_t, size_t);

    /**
     * @brief AVX512VNNI row-range GEMM (uint16_t A and B, wrapping 32-bit accumulate).
     */
    void gemm_u16_rows_avx512vnni(const uint16_t*, const uint16_t*, uint16_t*, size_t, size_t, size_t, size_t);

    /**
     * @brief AVX512F row-range GEMM (int32_t elements, mod-2^32 arithmetic).
     */
    void gemm_i32_rows_avx512(const int32_t*, const int32_t*, int32_t*, size_t, size_t, size_t, size_t);

    /**
     * @brief AVX512F row-range GEMM (uint32_t elements, mod-2^32 arithmetic).
     */
    void gemm_u32_rows_avx512(const uint32_t*, const uint32_t*, uint32_t*, size_t, size_t, size_t, size_t);

    /**
     * @brief AVX512DQ row-range GEMM (int64_t elements, mod-2^64 arithmetic).
     */
    void gemm_i64_rows_avx512dq(const int64_t*, const int64_t*, int64_t*, size_t, size_t, size_t, size_t);

    /**
     * @brief AVX512DQ row-range GEMM (uint64_t elements, mod-2^64 arithmetic).
     */
    void gemm_u64_rows_avx512dq(const uint64_t*, const uint64_t*, uint64_t*, size_t, size_t, size_t, size_t);

    /**
     * @brief AVX2 row-range GEMM (packed int4x2, per-lane wrapping accumulate).
     */
    void gemm_i4x2_rows_avx2(const int4x2_t*, const int4x2_t*, int4x2_t*, size_t, size_t, size_t, size_t);

    /**
     * @brief AVX512VNNI row-range GEMM (packed int4x2, per-lane wrapping accumulate).
     */
    void gemm_i4x2_rows_avx512vnni(const int4x2_t*, const int4x2_t*, int4x2_t*, size_t, size_t, size_t, size_t);
}
