/**
 * @file ConversionMethods.h
 * @author GreenBean
 * @brief Conversion clamp tolerance modes shared by every dtype conversion path.
 * @version 0.1
 * @date 2026-10-09
 */
#pragma once
/**
 * Assign a ConversionClamp tolerance for conversions.
 */
enum class ConversionClampMethod {
    /**
     * Error on source-inf AND on finite overflow/underflow (and NaN)
     */
    STRICT,
    /**
     * Error on finite overflow/underflow, and NaN. Do NOT error on source-inf.
     */
    ALLOW_INF,
    /**
     * Clamp every overflow/underflow, NaN stays NaN. Never error.
     */
    PERMISSIVE,

};
