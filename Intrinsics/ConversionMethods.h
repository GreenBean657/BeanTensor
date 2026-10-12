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
