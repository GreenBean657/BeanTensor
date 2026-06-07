#pragma once
//
// Created by greenbean on 2026-05-30
//
#include "Intrinsics/ConversionMethods.h"
#include "Tensor/Tensor.h"

namespace BeanTensor::Intrinsics::detail {



    [[nodiscard]] inline Casting::float32_t unaccel_bf16_to_fp32(const Casting::bfloat16_t& val) {
        const std::uint32_t prior_bf = static_cast<std::uint32_t>(val) << 16;
        Casting::float32_t f{0};
        std::memcpy(&f, &prior_bf, sizeof(prior_bf));
        return f;
    }

    [[nodiscard]] inline Casting::bfloat16_t unaccel_fp32_to_bf16_tracked(const Casting::float32_t val, bool& flag_met, const ConversionClampMethod method = ConversionClampMethod::PERMISSIVE) {
        uint32_t to_bits = 0;
        Casting::bfloat16_t bf16{0};
        std::memcpy(&to_bits, &val, sizeof(val));

        //Check if NaN
        const uint32_t mantissa_mask = to_bits & 0x7FFFFF;
        const uint32_t exponent_mask = to_bits & 0x7F800000;
        const bool exp_met = exponent_mask == 0x7F800000;
        if (exp_met) {
            //NaN && INF
            if (method == ConversionClampMethod::STRICT) {
                flag_met = true;
            }
            to_bits >>= 16;
            std::memcpy(&bf16, &to_bits, sizeof(bf16));
            return bf16;
        }
        // Round to nearest even
        const uint32_t round_bit  = (to_bits >> 15) & 1;
        const uint32_t sticky_bit = (to_bits & 0x7FFF) != 0;

        to_bits += round_bit & (sticky_bit | (to_bits >> 16) & 1);
        to_bits >>= 16;

        memcpy(&bf16, &to_bits, sizeof(bf16));
        return bf16;
    }

    [[nodiscard]] inline Casting::bfloat16_t unaccel_fp32_to_bf16_untracked(const Casting::float32_t& val) {
        bool ignored_flag = false;
        return unaccel_fp32_to_bf16_tracked(val, ignored_flag, ConversionClampMethod::PERMISSIVE);
    }

    [[nodiscard]] inline Casting::float32_t unaccel_fp16_to_fp32_tracked(const Casting::float16_t& val, bool& flag_met, const ConversionClampMethod method = ConversionClampMethod::PERMISSIVE) {
        uint16_t to_bits = 0;
        uint32_t converted = 0;
        Casting::float32_t returned_fp32{0};
        std::memcpy(&to_bits, &val, sizeof(val));
        converted = (static_cast<uint32_t>(to_bits & 0x8000) << 16); // Move sign to position 31
        const uint16_t mantissa_mask = to_bits & 0x03FF;
        const uint16_t exponent_mask = to_bits & 0x7C00;
        const bool exp_met = exponent_mask == 0x7C00;
        if (mantissa_mask != 0 && exp_met) {
            //NaN
            converted |= 0x7F800000; // Move the Exponent.
            converted |= static_cast<uint32_t>(mantissa_mask) << 13; // Move the Mantissa.
            if (method == ConversionClampMethod::STRICT) {
                flag_met = true;
            }
            memcpy(&returned_fp32, &converted, sizeof(returned_fp32));
            return returned_fp32;
        }
        if (mantissa_mask == 0 && exp_met) {
            //Inf
            converted |= 0x7F800000; // Move the Exponent.
            //Mantissa is zero.
            if (method == ConversionClampMethod::STRICT) {
                flag_met = true;
            }
            memcpy(&returned_fp32, &converted, sizeof(returned_fp32));
            return returned_fp32;
        }
        if (exponent_mask == 0 && mantissa_mask != 0) {
            uint32_t mant = mantissa_mask;
            uint32_t shift = 0;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                shift++;
            }
            mant &= 0x3FF;
            converted |= mant << 13;
            converted |= (112 - shift) << 23;
            memcpy(&returned_fp32, &converted, sizeof(returned_fp32));
            return returned_fp32;
        }
        uint32_t exponent = (to_bits >> 10) & 0x1F; // 5-bit raw exponent
        exponent = exponent - 15 + 127; // re-bias
        exponent <<= 23;// place at bits 30-23
        converted |= exponent;
        converted |= static_cast<uint32_t>(mantissa_mask) << 13;
        memcpy(&returned_fp32, &converted, sizeof(returned_fp32));
        return returned_fp32;
    }

    [[nodiscard]] inline Casting::float32_t unaccel_fp16_to_fp32_untracked(const Casting::float16_t& val) {
        bool ignored_flag = false;
        return unaccel_fp16_to_fp32_tracked(val, ignored_flag, ConversionClampMethod::PERMISSIVE);
    }

    [[nodiscard]] inline Casting::float16_t unaccel_fp32_to_fp16_tracked(const Casting::float32_t& val, bool& flag_met, const ConversionClampMethod method = ConversionClampMethod::PERMISSIVE) {
        uint16_t converted = 0;
        uint32_t to_bits = 0;
        Casting::float16_t returned_fp16{0};
        std::memcpy(&to_bits, &val, sizeof(val));

        const uint32_t mantissa_mask = to_bits & 0x007FFFFF;
        const uint32_t exponent_mask = to_bits & 0x7F800000;
        const bool exp_met = exponent_mask == 0x7F800000;
        converted |= ((to_bits & 0x80000000) >> 16);

        if (mantissa_mask != 0 && exp_met) {
            converted |= 0x7C00;
            converted |= (mantissa_mask >> 13);
            if (method == ConversionClampMethod::STRICT) flag_met = true;
            memcpy(&returned_fp16, &converted, sizeof(returned_fp16));
            return returned_fp16;
        }
        if (mantissa_mask == 0 && exp_met) {
            converted |= 0x7C00;
            if (method == ConversionClampMethod::STRICT) flag_met = true;
            memcpy(&returned_fp16, &converted, sizeof(returned_fp16));
            return returned_fp16;
        }
        if (val > Casting::detail::F16_MAX || val < Casting::detail::F16_MIN) {
            if (method != ConversionClampMethod::PERMISSIVE) flag_met = true;
            auto ignore_flag = false;
            if (val > Casting::detail::F16_MAX)
                return unaccel_fp32_to_fp16_tracked(Casting::detail::F16_MAX, ignore_flag);
            return unaccel_fp32_to_fp16_tracked(Casting::detail::F16_MIN, ignore_flag);
        }

        // Round to nearest even — normal path only
        const uint32_t round_bit  = (to_bits >> 12) & 1;
        const uint32_t sticky_bit = (to_bits & 0x0FFF) != 0;
        to_bits += round_bit & (sticky_bit | (to_bits >> 13) & 1);

        uint32_t exponent = (to_bits >> 23) & 0xFF;
        exponent = exponent - 127 + 15;
        exponent <<= 10;
        converted |= exponent;
        converted |= (to_bits & 0x007FFFFF) >> 13;

        memcpy(&returned_fp16, &converted, sizeof(returned_fp16));
        return returned_fp16;
    }

    [[nodiscard]] inline Casting::float16_t unaccel_fp32_to_fp16_untracked(const Casting::float32_t& val) {
        bool ignored_flag = false;
        return unaccel_fp32_to_fp16_tracked(val, ignored_flag, ConversionClampMethod::PERMISSIVE);
    }



}