#pragma once
//
// Created by greenbean on 2026-05-30.
//
#include "Intrinsics/ConversionMethods.h"
#include "Tensor/Tensor.h"

namespace BeanTensor::Intrinsics::detail {
    inline std::vector<std::shared_future<void>> unaccelerated_bf16_to_fp32(const Casting::bfloat16_t* src, float* dst, const size_t end) {
        const size_t t = Hardware::CPU().threads;

        auto compute = [src, dst](const size_t start, const size_t end_l) {
            for (size_t pos = start; pos < end_l; ++pos) {
                uint16_t raw;
                std::memcpy(&raw, &src[pos], sizeof(raw));
                uint32_t bits = static_cast<uint32_t>(raw) << 16;
                dst[pos] = std::bit_cast<Casting::float32_t>(bits);
            }
        };

        std::vector<std::shared_future<void>> futures;
        if (end <= t * 1024) {
            compute(0, end);
            return futures;
        }

        static auto& threadMgr = Threading::get_cpu_thread_pool();
        const size_t chunk_size = ((end / t + 15) / 16) * 16;

        auto worker = [compute](const size_t start, const size_t end_l) {
            compute(start, end_l);
        };
        for (size_t thread_id = 0; thread_id < t; ++thread_id) {
            const size_t start = thread_id * chunk_size;
            const size_t end_l = std::min(start + chunk_size, end);
            if (start >= end) break;
            futures.push_back(threadMgr.submit(
                [=] { worker(start, end_l); },
                Threading::ThreadPriority::Normal
            ).share());
        }
        return futures;
    }

    /**
     * Convert a single BF16 to a FP32.
     * @param val BF16 Value
     * @return FP32 Value
     */
    inline Casting::float32_t bf16_to_fp32(const Casting::bfloat16_t val) {
#if (!defined(USE_CUDA) && !defined(USE_HIP))
        uint32_t bits = static_cast<uint32_t>(Casting::raw(val)) << 16;
        Casting::float32_t f; std::memcpy(&f, &bits, 4);
        return f;
#else
        return __bfloat162float(val);
#endif
    }

    /**
     * Convert a FP32 to a BF16
     * @param val FP32 Value
     * @return BF16 Value
     */
    inline Casting::bfloat16_t fp32_to_bf16(const Casting::float32_t val) {
#if (!defined(USE_CUDA) && !defined(USE_HIP))
        uint32_t x; std::memcpy(&x, &val, 4);
        if (((x >> 23) & 0xFF) == 0xFF && (x & 0x7FFFFF)) {
            // NaN -> keep sign, force a non-zero mantissa top bit (quiet)
            return Casting::as_bf16(static_cast<uint16_t>((x >> 16) | 0x0040));
        }
        // AI ASSISTED
        const uint32_t lsb = (x >> 16) & 1u;
        const uint32_t bias = 0x7FFFu + lsb;
        x += bias;
        return Casting::as_bf16(static_cast<uint16_t>(x >> 16));
#else
        return __float2bfloat16(val);
#endif
    }

    /**
     * Convert a FP16 to a FP32
     * @param val FP16 Value
     * @return FP32 Value
     */
    inline Casting::float32_t fp16_to_fp32(const Casting::float16_t val) {
#if (!defined(USE_CUDA) && !defined(USE_HIP))
        const uint16_t v = Casting::raw(val);
        uint32_t sign = static_cast<uint32_t>(v & 0x8000u) << 16;
        uint32_t exp  = (v >> 10) & 0x1F;
        uint32_t mant = v & 0x3FF;
        uint32_t f;
        if (exp == 0) {
            if (mant == 0) {
                f = sign; // +/- 0
            } else {
                // subnormal half -> normalized float
                int e = -1;
                do { mant <<= 1; ++e; } while ((mant & 0x400) == 0);
                mant &= 0x3FF;
                f = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) | (mant << 13);
            }
        } else if (exp == 0x1F) {
            f = sign | 0x7F800000u | (mant << 13);          // inf / NaN
        } else {
            f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
        float out; std::memcpy(&out, &f, 4);
        return out;
#else
        return __half2float(val);
#endif
    }

    inline Casting::float16_t fp32_to_fp16(const Casting::float32_t val, [[maybe_unused]] bool& was_nan, [[maybe_unused]] bool& was_inf,
                                [[maybe_unused]] bool& overflow_finite) {
#if (!defined(USE_CUDA) && !defined(USE_HIP))
        // AI ASSISTED
        was_nan = was_inf = overflow_finite = false;
        uint32_t x; std::memcpy(&x, &val, 4);
        uint32_t sign = (x >> 16) & 0x8000u;
        uint32_t aexp = (x >> 23) & 0xFF;
        uint32_t mant = x & 0x7FFFFF;

        if (aexp == 0xFF) {
            if (mant) { was_nan = true; return Casting::as_f16(static_cast<uint16_t>(sign | 0x7E00)); }
            was_inf = true; return Casting::as_f16(static_cast<uint16_t>(sign | 0x7C00));
        }

        int e = static_cast<int>(aexp) - 127 + 15;          // half-biased exponent
        if (e >= 0x1F) { overflow_finite = true; return Casting::as_f16(static_cast<uint16_t>(sign | 0x7C00)); }
        if (e <= 0) {
            if (e < -10) return Casting::as_f16(static_cast<uint16_t>(sign)); // underflow -> +/- 0
            mant |= 0x800000;                                // restore implicit 1
            const int shift = 14 - e;
            uint32_t hm = mant >> shift;
            const uint32_t rem = mant & ((1u << shift) - 1);
            if (const uint32_t half = 1u << (shift - 1); rem > half || (rem == half && (hm & 1))) ++hm; // RNE
            return Casting::as_f16(static_cast<uint16_t>(sign | hm));
        }
        auto h = static_cast<uint16_t>(sign | (static_cast<uint32_t>(e) << 10) | (mant >> 13));
        if (const uint32_t rem = mant & 0x1FFF; rem > 0x1000 || (rem == 0x1000 && (h & 1))) {
            ++h;
            if ((h & 0x7FFF) == 0x7C00) overflow_finite = true;
        }
        return Casting::as_f16(h);
#else
    return __float2half(val);
#endif
    }

    /**
     * Convert a Floating Point value to FP64
     * @tparam T Floating Point type to convert
     * @param v Value to convert
     * @return FP64 value
     */
    template <typename T> requires Casting::is_float_kind_v<T>
    [[nodiscard]] Casting::float64_t decode_to_fp64(T v) {
        if constexpr (std::is_same_v<T, Casting::bfloat16_t>) {
            return static_cast<Casting::float64_t>(bf16_to_fp32(v));
        } else if constexpr (std::is_same_v<T, Casting::float16_t>) {
            return static_cast<Casting::float64_t>(fp16_to_fp32(v));
        } else {
            return static_cast<Casting::float64_t>(v);
        }
    }

    /**
     * Verify compatibility during conversions.
     * @note float-source -> integer-destination
     * @tparam dst Destination DType
     * @param dv Double value.
     * @param method Conversion Method, used to determine errors.
     * @param flag Flag, used to determine errors.
     * @return Converted value.
     */
    template <typename dst> requires std::is_integral_v<dst>
    [[nodiscard]] dst sat_float_to_int(double dv, const ConversionClampMethod method, bool& flag) {
        constexpr auto DMAX = static_cast<double>(std::numeric_limits<dst>::max());
        constexpr auto DMIN = static_cast<double>(std::numeric_limits<dst>::min());
        if (std::isnan(dv)) {
            if (method != ConversionClampMethod::ALLOW_ALL) flag = true;
            return static_cast<dst>(0);
        }
        if (std::isinf(dv)) {
            if (method == ConversionClampMethod::HARD_ERROR) flag = true;
            return dv > 0 ? std::numeric_limits<dst>::max()
                          : std::numeric_limits<dst>::min();
        }
        if (dv > DMAX) {
            if (method != ConversionClampMethod::ALLOW_ALL) flag = true;
            return std::numeric_limits<dst>::max();
        }
        if (dv < DMIN) {
            if (method != ConversionClampMethod::ALLOW_ALL) flag = true;
            return std::numeric_limits<dst>::min();
        }
        return static_cast<dst>(dv);
    }

    /**
     * Verify compatibility during conversions.
     * @note integer-source -> integer-destination
     * @tparam dst Destination DType
     * @tparam src Source DType
     * @param v Value to test
     * @param method Conversion method, used to determine errors.
     * @param flag Flag, used to determine errors.
     * @return Converted value
     */
    template<typename dst, typename src> requires std::is_integral_v<dst> && std::is_integral_v<src>
    [[nodiscard]] dst sat_int_to_int(src v, const ConversionClampMethod method, bool& flag) {
        const auto lv = static_cast<long double>(v);
        const auto DMAX = static_cast<long double>(std::numeric_limits<dst>::max());
        const auto DMIN = static_cast<long double>(std::numeric_limits<dst>::min());
        if (lv > DMAX) { if (method != ConversionClampMethod::ALLOW_ALL) flag = true;
            return std::numeric_limits<dst>::max(); }
        if (lv < DMIN) { if (method != ConversionClampMethod::ALLOW_ALL) flag = true;
            return std::numeric_limits<dst>::min(); }
        return static_cast<dst>(v);
    }

    /**
     * Write a double into a float, with protections.
     * @note Write a double into a float-like destination.
     * @tparam dst Destination Dtype.
     * @param dv Double to write with.
     * @param src_was_nan Was the source NaN?
     * @param src_was_inf Was the source infinity?
     * @param method ConversionMethod
     * @param flag Flag, used to determine errors.
     * @return Written value
     */
    template <typename dst> requires Casting::is_float_kind_v<dst>
    [[nodiscard]] dst encode_float_dst(Casting::float64_t dv, const bool src_was_nan, const bool src_was_inf, const ConversionClampMethod method, bool& flag) {
        constexpr bool is_bfloat16 = std::is_same_v<dst, Casting::bfloat16_t>;
        constexpr bool is_float16  = std::is_same_v<dst, Casting::float16_t>;
        if (src_was_nan) {
            if constexpr (is_bfloat16) {
                return fp32_to_bf16(std::nanf(""));
            } else if constexpr (is_float16) {
                bool a = false;
                bool b = false;
                bool c = false;
                return fp32_to_fp16(std::nanf(""), a,b,c);
            } else {
                return static_cast<dst>(std::numeric_limits<double>::quiet_NaN());
            }
        }
        if (src_was_inf) {
            if (method == ConversionClampMethod::HARD_ERROR) {
                flag = true;
            }
            constexpr Casting::float64_t inf = std::numeric_limits<Casting::float64_t>::infinity();
            const Casting::float64_t sv = std::signbit(dv) ? -inf : inf;
            if constexpr (is_bfloat16) {
                return fp32_to_bf16(static_cast<Casting::float32_t>(sv));
            } else if constexpr (is_float16) {
                bool a = false;
                bool b = false;
                bool c = false;
                return fp32_to_fp16(static_cast<Casting::float32_t>(sv), a,b,c);
            } else {
                return static_cast<dst>(dv);   // only float32/float64 land here
            }
        }
        //Finite
        Casting::float64_t maxf;
        if constexpr (std::is_same_v<dst, Casting::bfloat16_t>) {
            maxf = 3.3895314e38;
        } else if constexpr (std::is_same_v<dst, Casting::float16_t>) {
            maxf = 65504.0;
        } else {
            maxf = static_cast<Casting::float64_t>(std::numeric_limits<dst>::max());
        }
        if (std::fabs(dv) > maxf) {
            if (method != ConversionClampMethod::ALLOW_ALL) {
                flag = true;
            }
            dv = std::copysign(maxf, dv);
        }
        if constexpr (is_bfloat16) {
            return fp32_to_bf16(static_cast<Casting::float32_t>(dv));
        } else if constexpr (is_float16) {
            bool a = false;
            bool b = false;
            bool c = false;
            return fp32_to_fp16(static_cast<Casting::float32_t>(dv), a,b,c);
        } else {
            return static_cast<dst>(dv);
        }
        __builtin_unreachable();
    }

    /**
     *
     * @tparam dst
     * @tparam src
     * @param value
     * @param method
     * @param flag
     * @return
     */
    template <typename dst, typename src>
    [[nodiscard]] dst scalar_cast_one(src value, ConversionClampMethod method, bool& flag) {
        if constexpr (Casting::is_float_kind_v<src> && Casting::is_float_kind_v<dst>) {
            // Float > Float
            const double dv = decode_to_fp64(value);
            return encode_float_dst<dst>(dv, std::isnan(dv), std::isinf(dv), method, flag);
        } else if constexpr (Casting::is_float_kind_v<src> && !Casting::is_float_kind_v<dst>) {
            // Float > Int
            return sat_float_to_int<dst>(decode_to_fp64(value), method, flag);
        } else if constexpr (!Casting::is_float_kind_v<src> && Casting::is_float_kind_v<dst>) {
            // Int > Float (Precision loss is not overflow!!)
            const double dv = static_cast<Casting::float64_t>(value);
            if constexpr (std::is_same_v<dst, Casting::bfloat16_t>) {
                return fp32_to_bf16(static_cast<Casting::float32_t>(dv));
            } else if constexpr (std::is_same_v<dst, Casting::float16_t>) {
                bool a = false;
                bool b = false;
                bool c = false;
                return fp32_to_fp16(static_cast<Casting::float32_t>(dv), a,b,c);
            } else {
                return static_cast<dst>(value);
            }
        } else {
            // Int > Int
            return sat_int_to_int<dst>(value, method, flag);
        }
    }

    template <typename dst, typename src>
    void scalar_bulk(const src* source, dst* destination, const size_t start, const size_t end, const ConversionClampMethod method, bool& flag) {
        for (size_t i = start; i < end; i++) {
            destination[i] = scalar_cast_one<dst, src>(source[i], method, flag);
        }
    }
}