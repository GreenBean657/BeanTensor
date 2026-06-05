#pragma once
/**
 *
 */
enum class ConversionClampMethod {
    /**
     * Error on source-inf AND on finite overflow/underflow (and NaN>int)
     */
    HARD_ERROR,
    /**
     * Error on finite overflow/underflow (and NaN>int). Do NOT error on source-inf.
     */
    ALLOW_INF,
    /**
     * Clamp every overflow/underflow, NaN -> 0. Never error.
     */
    ALLOW_ALL

};
