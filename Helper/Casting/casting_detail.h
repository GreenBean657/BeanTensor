#pragma once

#include "Helper/hardware.h"
#include "Tensor/Dtypes.h"
// Claims

#define HAS_F16C = __builtin_cpu_supports("f16c")

#if defined(USE_HIP) && defined(__clang__)
    #include <hip/hip_fp16.h>
    #include <hip/hip_bf16.h>
#elif defined(USE_CUDA)
    #include <cuda_fp16.h>
    #include <cuda_bf16.h>
#else
    #include <cmath>
    #include <cstring>
    #include <cstdint>
    #include <immintrin.h>
#endif

namespace BeanTensor::Casting::detail {

        using float32_t = float;
        using float64_t = double;


        constexpr float32_t F16_MAX      =  65504.0f;
        constexpr float32_t F16_MIN      = -65504.0f;
        constexpr float32_t F16_MIN_POS  =  0.000060975552f;
        constexpr float32_t F16_EPSILON  =  0.000977f;

        constexpr float32_t BF16_MAX     =  3.38953139e+38f;
        constexpr float32_t BF16_MIN     = -3.38953139e+38f;
        constexpr float32_t BF16_MIN_POS =  1.175494351e-38f;
        constexpr float32_t BF16_EPSILON =  0.0078125f;

#if (defined(USE_HIP) && defined(__clang__)) || defined(USE_CUDA)
        inline float32_t f16_to_f32(const float16_t f16) {
            return __half2float(f16);
        }
#else
        inline float32_t f16_to_f32(const float16_t f16) {
            if (BeanTensor::Hardware::CPU().f16c) {
                return _cvtsh_ss(raw(f16));
            }
            const uint16_t bits = raw(f16);
            const uint16_t sign = (bits >> 15) & 0x1;
            const uint16_t exp16 = (bits >> 10) & 0x1F;
            const uint16_t mant16 = bits & 0x3FF;

            // NaN
            if (exp16 == 0x1F && mant16 != 0) {
                const uint32_t nan = (static_cast<uint32_t>(sign) << 31) | 0x7F800000 | (static_cast<uint32_t>(mant16) << 13);
                float32_t f;
                std::memcpy(&f, &nan, sizeof(nan));
                return f;
            }

            // Infinity
            if (exp16 == 0x1F) {
                const uint32_t inf = (static_cast<uint32_t>(sign) << 31) | 0x7F800000;
                float32_t f;
                std::memcpy(&f, &inf, sizeof(inf));
                return f;
            }

            // Zero
            if (exp16 == 0 && mant16 == 0) {
                const uint32_t zero = static_cast<uint32_t>(sign) << 31;
                float32_t f;
                std::memcpy(&f, &zero, sizeof(zero));
                return f;
            }

            const uint32_t exp32  = static_cast<uint32_t>(exp16) - 15 + 127;
            const uint32_t mant32 = static_cast<uint32_t>(mant16) << 13;
            const uint32_t out    = (static_cast<uint32_t>(sign) << 31) | (exp32 << 23) | mant32;

            float32_t f;
            std::memcpy(&f, &out, sizeof(out));
            return f;
        }

#endif

#if (defined(USE_HIP) && defined(__clang__)) || defined(USE_CUDA)
        inline float32_t bf16_to_f32(const bfloat16_t bf16) {
            return __bfloat162float(bf16);
        }
#else

        inline float32_t bf16_to_f32(const bfloat16_t bf16) {
            float32_t res;
            uint32_t bits = raw(bf16);
            bits = bits << 16;
            std::memcpy(&res, &bits, 4);
            return res;
        }
#endif

#if (defined(USE_HIP) && defined(__clang__)) || defined(USE_CUDA)
        inline float16_t f32_to_f16(const float32_t f32) {
            return __float2half(f32);
        }


#else
        inline float16_t f32_to_f16(const float32_t f32) {
            if (BeanTensor::Hardware::CPU().f16c)
                return as_f16(_cvtss_sh(f32, 0));

            uint32_t bits;
            std::memcpy(&bits, &f32, 4);

            const uint16_t sign   =  (bits >> 31) & 0x1;
            const auto     exp32  = static_cast<int32_t>((bits >> 23) & 0xFF);
            const uint32_t mant32 =   bits & 0x7FFFFF;

            // NaN
            if (exp32 == 0xFF && mant32 != 0)
                return as_f16((sign << 15) | 0x7C00 | 0x0200);

            // Infinity
            if (exp32 == 0xFF)
                return as_f16((sign << 15) | 0x7C00);

            int32_t exp16 = exp32 - 127 + 15;

            // Underflow — flush to zero
            if (exp16 <= 0)
                return as_f16(sign << 15);

            // Overflow — flush to infinity
            if (exp16 >= 31)
                return as_f16((sign << 15) | 0x7C00);

            // Round to nearest even
            uint32_t mant16 = mant32 >> 13;
            if (const uint32_t round = mant32 & 0x1FFF;
                round > 0x1000 || (round == 0x1000 && (mant16 & 1)))
                mant16++;

            // Rounding overflowed mantissa into exponent
            if (mant16 >= 0x400) {
                mant16 = 0;
                exp16++;
                if (exp16 >= 31)
                    return as_f16((sign << 15) | 0x7C00);
            }

            return as_f16(
                (sign << 15) | (static_cast<uint16_t>(exp16) << 10) | static_cast<uint16_t>(mant16)
            );
        }

#endif

#if (defined(USE_HIP) && defined(__clang__)) || defined(USE_CUDA)
        inline bfloat16_t f32_to_bf16(const float32_t f32) {
            return __float2bfloat16(f32);
        }
#else
        inline bfloat16_t f32_to_bf16(const float32_t f32) {
            uint32_t bits;
            std::memcpy(&bits, &f32, sizeof(bits));
            bits += 0x7FFF + ((bits >> 16) & 1); //Rounding
            return as_bf16(bits >> 16);
        }
#endif

#if (defined(USE_HIP) && defined(__clang__)) || defined(USE_CUDA)
        inline bfloat16_t f16_to_bf16(const float16_t f16) {
            return __float2bfloat16(__half2float(f16));
        }

        inline float16_t fbf16_to_16(const bfloat16_t bf16) {
            return __float2half(__bfloat162float(bf16));
        }
#else

        inline bfloat16_t f16_to_bf16(const float16_t f16) {
            // expand f16 bits into fp32 layout, then take top 16
            const uint16_t bits   = raw(f16);
            const uint32_t sign   = (bits >> 15) & 0x1;
            const uint32_t exp16  = (bits >> 10) & 0x1F;
            const uint32_t mant16 = bits & 0x3FF;

            // rebias exponent 15 → 127
            const uint32_t exp32  = exp16 - 15 + 127;
            // shift mantissa from 10 bits to 23 bits
            const uint32_t mant32 = mant16 << 13;

            const uint32_t fp32bits = (sign << 31) | (exp32 << 23) | mant32;
            // bf16 is just the top 16 bits of fp32
            return as_bf16(fp32bits >> 16);
        }

        inline float16_t bf16_to_f16(const bfloat16_t bf16) {
            const uint32_t fp32bits = static_cast<uint32_t>(raw(bf16)) << 16;
            const uint32_t sign     = (fp32bits >> 31) & 0x1;
            const uint32_t exp32    = (fp32bits >> 23) & 0xFF;
            const uint32_t mant32   =  fp32bits        & 0x7FFFFF;

            // rebias exponent 127 > 15, shift mantissa 23 > 10
            const uint32_t exp16  = exp32 - 127 + 15;
            const uint16_t mant16 = mant32 >> 13;

            return as_f16((sign << 15) | (exp16 << 10) | mant16);
        }

#endif

        // f64 conversions
        inline float16_t f64_to_f16(const float64_t f64) {
            return f32_to_f16(static_cast<float32_t>(f64));
        }

        inline bfloat16_t f64_to_bf16(const float64_t f64) {
            return f32_to_bf16(static_cast<float32_t>(f64));
        }

        inline float32_t f64_to_f32(const float64_t f64) {
            return static_cast<float32_t>(f64);
        }

        inline float64_t f32_to_f64(const float32_t f32) {
            return f32;
        }


        inline float64_t f16_to_f64(const float16_t f16) {
            return f16_to_f32(f16);
        }

        inline float64_t bf16_to_f64(const bfloat16_t bf16) {
            return bf16_to_f32(bf16);
        }
    } //namespace BeanTensor::Casting