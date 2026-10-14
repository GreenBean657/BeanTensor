#pragma once

#include <cstdint>
#include <cstring>

#include "Intrinsics/ConversionMethods.h"
#include "Tensor/DTypes.h"

namespace BeanTensor::Intrinsics::detail {

    /**
     * Perform a bit_cast via a memcpy instruction. Requires size_of(to) == size_of(from).
     * @brief std::bit_cast via memcpy, for non-trivially-copyable types.
     * @note Mimics std::bit_cast, but uses memcpy to avoid the trivially-copyable requirement. This is necessary for HIP's __hip_bfloat16 and __hip_bfloat162 types, which have user-provided copy constructors and are therefore not trivially copyable.
     * @tparam To Source type to cast to.
     * @tparam From Target type to cast from.
     * @param from Value to cast from.
     * @return Casted value of type To.
     */
    template<typename To, typename From>
    [[nodiscard]] To bit_cast_memcpy(const From& from) noexcept {
        static_assert(sizeof(To) == sizeof(From));
        To to{};
        std::memcpy(static_cast<void*>(&to), static_cast<const void*>(&from), sizeof(To));
        return to;
    }

    /**
     * Perform a BF16 to FP32 conversion without using hardware acceleration on a single value.
     * @brief Software emulated BF16 > FP32
     * @param val BF16 Value to convert.
     * @return FP32 value converted from the BF16 input.
     */
    [[nodiscard]] inline float32_t unaccel_bf16_to_fp32(const bfloat16_t& val) {
        const std::uint32_t prior_bf = static_cast<std::uint32_t>(bit_cast_memcpy<std::uint16_t>(val)) << 16;
        return bit_cast_memcpy<float32_t>(prior_bf);
    }

    /**
     * Perform a FP32 to BF16 conversion without using hardware acceleration on a single value.
     * @note Does not throw on ConversionClampMethod due to hyperthreading divergence support. If you need to throw on a conversion event, TRACK THE FLAG ARGUMENT.
     * @brief Software emulated FP32 > BF16
     * @param val FP32 Value to convert.
     * @param flag_met Reference to a boolean flag that will be set to true if a conversion error occurs.
     * @param method Active ConversionClampMethod - How should conversion tolerance be handled?
     * @return BF16 value converted from the FP32 input.
     */
    [[nodiscard]] inline bfloat16_t unaccel_fp32_to_bf16_tracked(const float32_t val, bool& flag_met, const ConversionClampMethod method = ConversionClampMethod::PERMISSIVE) {
        const auto to_bits = bit_cast_memcpy<std::uint32_t>(val);

        //Check if NaN
        const uint32_t mantissa_mask = to_bits & 0x7FFFFF;
        const uint32_t exponent_mask = to_bits & 0x7F800000;
        const bool exp_met = exponent_mask == 0x7F800000;
        if (exp_met) {
            //NaN && INF: NaN flags under any non-PERMISSIVE method, Inf under STRICT only.
            if (mantissa_mask != 0) {
                if (method != ConversionClampMethod::PERMISSIVE) flag_met = true;
            } else if (method == ConversionClampMethod::STRICT) {
                flag_met = true;
            }
            auto truncated = static_cast<std::uint16_t>(to_bits >> 16);
            if (mantissa_mask != 0) {
                // NaN: set quiet bit so payload dosen't truncate to zero (inf yield), to match AVX.
                truncated |= 0x0040;
            }
            return bit_cast_memcpy<bfloat16_t>(truncated);
        }

        // FP32 subnormals / zero input: flush to zero, preserving only the sign. Matches AVX.
        if (exponent_mask == 0) {
            return bit_cast_memcpy<bfloat16_t>(static_cast<std::uint16_t>((to_bits & 0x80000000) >> 16));
        }
        // Round to nearest even: add half + the result LSB, then truncate.
        std::uint32_t rounded = to_bits;
        rounded += 0x7FFF + ((rounded >> 16) & 1);
        rounded >>= 16;

        return bit_cast_memcpy<bfloat16_t>(static_cast<std::uint16_t>(rounded));
    }

    /**
     * Perform a FP32 to BF16 conversion without using hardware acceleration on a single value. Discards Conversion errors.
     * @note Untracked FP32 > BF16. Mimics a tracked call but discards the flag.
     * @brief Software emulated FP32 > BF16
     * @param val FP32 Value to convert.
     * @return BF16 value converted from the FP32 input.
     */
    [[nodiscard]] inline bfloat16_t unaccel_fp32_to_bf16_untracked(const float32_t& val) {
        bool ignored_flag = false;
        return unaccel_fp32_to_bf16_tracked(val, ignored_flag, ConversionClampMethod::PERMISSIVE);
    }

    /**
     * Perform a FP16 to FP32 conversion without using hardware acceleration on a single value.
     * @note Does not throw on ConversionClampMethod due to hyperthreading divergence support. If you need to throw on a conversion event, TRACK THE FLAG ARGUMENT.
     * @brief Software emulated FP16 > FP32.
     * @param val FP16 Value to convert.
     * @param flag_met Reference to a boolean flag that will be set if a conversion error occurs.
     * @param method The conversion clamp method to use.
     * @return The converted FP32 value.
     */
    [[nodiscard]] inline float32_t unaccel_fp16_to_fp32_tracked(const float16_t& val, bool& flag_met, const ConversionClampMethod method = ConversionClampMethod::PERMISSIVE) {
        const auto to_bits = bit_cast_memcpy<std::uint16_t>(val);
        std::uint32_t converted = (static_cast<std::uint32_t>(to_bits & 0x8000) << 16); // Move sign to position 31
        const uint16_t mantissa_mask = to_bits & 0x03FF;
        const uint16_t exponent_mask = to_bits & 0x7C00;
        const bool exp_met = exponent_mask == 0x7C00;
        if (mantissa_mask != 0 && exp_met) {
            //NaN — set the fp32 quiet bit so the payload can never become zero.
            converted |= 0x7F800000; // Move the Exponent.
            converted |= 0x00400000; // Quiet bit.
            converted |= static_cast<std::uint32_t>(mantissa_mask) << 13; // Move the Mantissa.
            if (method != ConversionClampMethod::PERMISSIVE) {
                flag_met = true;
            }
            return bit_cast_memcpy<float32_t>(converted);
        }
        if (mantissa_mask == 0 && exp_met) {
            //Inf. Mantissa is zero.
            converted |= 0x7F800000; // Move the Exponent.
            if (method == ConversionClampMethod::STRICT) {
                flag_met = true;
            }
            return bit_cast_memcpy<float32_t>(converted);
        }
        if ((to_bits & 0x7FFF) == 0) {
            // +-0
            return bit_cast_memcpy<float32_t>(converted);
        }
        if (exponent_mask == 0 && mantissa_mask != 0) {
            //Subnormal
            uint32_t mant = mantissa_mask;
            uint32_t shift = 0;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                shift++;
            }
            mant &= 0x3FF;
            converted |= mant << 13;
            converted |= (113 - shift) << 23;
            return bit_cast_memcpy<float32_t>(converted);
        }
        uint32_t exponent = (to_bits >> 10) & 0x1F; // 5-bit raw exponent
        exponent = exponent - 15 + 127; // re-bias
        exponent <<= 23;// place at bits 30-23
        converted |= exponent;
        converted |= static_cast<std::uint32_t>(mantissa_mask) << 13;
        return bit_cast_memcpy<float32_t>(converted);
    }

    /**
     * Perform a FP16 to FP32 conversion without using hardware acceleration on a single value. Discards Conversion errors.
     * @note Untracked FP16 > FP32. Mimics a tracked call but discards the flag.
     * @brief Software emulated FP16 > FP32
     * @param val FP16 Value to convert.
     * @return FP32 value converted from the FP16 input.
     */
    [[nodiscard]] inline float32_t unaccel_fp16_to_fp32_untracked(const float16_t& val) {
        bool ignored_flag = false;
        return unaccel_fp16_to_fp32_tracked(val, ignored_flag, ConversionClampMethod::PERMISSIVE);
    }

    /**
     * Perform a FP32 to FP16 conversion without using hardware acceleration on a single value.
     * @note Does not throw on ConversionClampMethod due to hyperthreading divergence support. If you need to throw on a conversion event, TRACK THE FLAG ARGUMENT.
     * @param val FP32 Value to convert.
     * @param flag_met Reference to a boolean flag that will be set if a conversion error occurs.
     * @param method The conversion clamp method to use.
     * @return The converted FP16 value.
     */
    [[nodiscard]] inline float16_t unaccel_fp32_to_fp16_tracked(const float32_t& val, bool& flag_met, const ConversionClampMethod method = ConversionClampMethod::PERMISSIVE) {
        std::uint16_t converted = 0;
        auto to_bits = bit_cast_memcpy<std::uint32_t>(val);

        const uint32_t mantissa_mask = to_bits & 0x007FFFFF;
        const uint32_t exponent_mask = to_bits & 0x7F800000;
        const bool exp_met = exponent_mask == 0x7F800000;
        converted |= ((to_bits & 0x80000000) >> 16);

        if (mantissa_mask != 0 && exp_met) {
            // NaN - set FP16 quiet bit so payload never truncates to zero (which would yield Inf).
            converted |= 0x7C00;
            converted |= 0x0200;
            converted |= (mantissa_mask >> 13);
            if (method != ConversionClampMethod::PERMISSIVE) flag_met = true;
            return bit_cast_memcpy<float16_t>(converted);
        }
        if (mantissa_mask == 0 && exp_met) {
            converted |= 0x7C00;
            if (method == ConversionClampMethod::STRICT) flag_met = true;
            return bit_cast_memcpy<float16_t>(converted);
        }
        if (val > DataTypes::detail::FP16_MAX || val < DataTypes::detail::FP16_MIN) {
            if (method != ConversionClampMethod::PERMISSIVE) flag_met = true;
            auto ignore_flag = false;
            if (val > DataTypes::detail::FP16_MAX)
                return unaccel_fp32_to_fp16_tracked(DataTypes::detail::FP16_MAX, ignore_flag);
            return unaccel_fp32_to_fp16_tracked(DataTypes::detail::FP16_MIN, ignore_flag);
        }

        // Result is an FP16 subnormal (or zero) when the FP32 exponent field is below 113 (value < 2^-14).
        // Shift mantissa into place && round-to-nearest-even for gradual underflow.
        if (
            const uint32_t exp_field = (to_bits >> 23) & 0xFF;
            exp_field < 113
            ) {
            const uint32_t mant24 = (to_bits & 0x007FFFFF) | 0x00800000;
            const uint32_t shift = 126 - exp_field; // distance to the fp16 subnormal LSB (2^-24)
            uint32_t sub = 0;
            if (shift <= 25) {
                const uint32_t halfway = 1u << (shift - 1);
                sub = (mant24 + halfway - 1 + ((mant24 >> shift) & 1)) >> shift;
            }
            converted |= static_cast<std::uint16_t>(sub);
            return bit_cast_memcpy<float16_t>(converted);
        }

        // Round to nearest even: add half + the result LSB, then truncate.
        to_bits += 0xFFF + ((to_bits >> 13) & 1);

        uint32_t exponent = (to_bits >> 23) & 0xFF;
        exponent = exponent - 127 + 15;
        exponent <<= 10;
        converted |= exponent;
        converted |= (to_bits & 0x007FFFFF) >> 13;

        return bit_cast_memcpy<float16_t>(converted);
    }

    /**
     * Perform a FP32 to FP16 conversion without using hardware acceleration on a single value. Discards Conversion errors.
     * @note Untracked FP32 > FP16. Mimics a tracked call but discards the flag.
     * @brief Software emulated FP32 > FP16
     * @param val FP32 Value to convert.
     * @return FP16 value converted from the FP32 input.
     */
    [[nodiscard]] inline float16_t unaccel_fp32_to_fp16_untracked(const float32_t& val) {
        bool ignored_flag = false;
        return unaccel_fp32_to_fp16_tracked(val, ignored_flag, ConversionClampMethod::PERMISSIVE);
    }

}
