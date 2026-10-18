/**
 * @file Hardware.h
 * @author GreenBean
 * @brief CPU feature set detection and cached runtime feature queries.
 * @version 0.1
 * @date 2026-10-18
 */
#pragma once

#include <cstdint>
#include <thread>

/**
 * CPU Feature Set Types, used to determine what is or isn't available.
 * If a instruction set is not listed here, its not used by BeanTensor. It likely wasn't compiled in the binary - don't try to use it.
 * @brief CPU Feature Set Types, used to determine and test features are available on the current CPU.
 */
enum CPUFeatureSetTypes {
    CPUFeatureSetType_F16C = 1,
    CPUFeatureSetType_FP16 = 2,
    CPUFeatureSetType_AVX= 3,
    CPUFeatureSetType_AVX2 = 4,
    CPUFeatureSetType_AVX512F = 5,
    CPUFeatureSetType_AVX512BF16 = 6,
    CPUFeatureSetType_FMA = 7,
    CPUFeatureSetType_NPU_STRIXHALO = 8,
    CPUFeatureSetType_AVX512DQ = 9,
    CPUFeatureSetType_AVX512BW = 10,
    CPUFeatureSetType_AVX512VNNI = 11
    // Bits 12-32 RESERVED for future checks. Do not downscale features to uint16_t.
};

/**
 * Get the CPU feature set. If you didn't obtain this value by calling IncludedCPUFeatures(), you wont get what you are looking for. This is a static value, and will not change during runtime.
 * @brief CPU Feature Set, used to determine what features are available on the current CPU.
 */
struct alignas(8) CPUFeatureSets {
    uint32_t features;
    uint32_t threads; // Likely not running this many threads.. ever.. but 2x uint32_t means 1 register can store, in full.
    // TODO - (Marked TODO for V1 final check.) if features ends up being uint16_t, compact threads to align at 32 bit total.

    /**
     * @brief INTERNAL - Set a feature as present.
     * Set a feature as present on this CPU. Affects this CPUFeatureSets instance only.
     * @param n CPUFeatureSetTypes to set.
     * @note You probably shouldn't call this - you will desync your CPUFeatureSets instance. Use has_feature.
     */
    void set_feature(const CPUFeatureSetTypes n) noexcept {
        features |=  (1u << n);
    }

    /**
     * @brief INTERNAL - Clear a feature as present.
     * Clear a feature as present on this CPU. Affects this CPUFeatureSets instance only
     * @param n CPUFeatureSetTypes to clear.
     * @note You probably shouldn't call this - you will desync your CPUFeatureSets instance. Use has_feature.
     */
    void clear_feature(const CPUFeatureSetTypes n) noexcept {
        features &= ~(1u << n);
    }

    /**
     * @brief Check if a feature is present.
     * Check if a feature is present on this CPU, based on what this CPUFeatureSets instance was told.
     * @param n CPUFeatureSetTypes to check.
     * @return [TRUE/FALSE] Present.
     */
    [[nodiscard]] bool has_feature(const CPUFeatureSetTypes n) const noexcept {
        return (features >> n) & 1u;
    }
};

/**
 * Get the CPU feature set, based on both compile-time and start-time runtime checks. This is a static value, and will not change during runtime.
 * @brief Get the CPU feature set.
 * @return The CPU feature set.
 */
[[nodiscard]] __inline__ __attribute__((always_inline)) static const CPUFeatureSets& IncludedCPUFeatures() {
    // NOLINTNEXTLINE(*-dynamic-static-initializers)
    static const CPUFeatureSets cpu_features = []{
        CPUFeatureSets f {
            .features = 0,
            .threads  = std::thread::hardware_concurrency()
        };

// NOTE - NEVER, EVER removed #if defined() checks.
// __builtin_cpu_supports is *ONLY* a run-time check.
// IncludedCPUFeatures() is static W/ Internal linkage - Every TU will get it's OWN copy.
// It's NOT a violation when TU X lacks the compile-time flag, but TU Y has it. If you remove the #if defined() checks, TU X will call __builtin_cpu_supports("X") and crash at runtime because the compiler didn't compile in the instruction set for that TU.
#if defined(__F16C__)
        if (__builtin_cpu_supports("f16c")) {
            f.set_feature(CPUFeatureSetType_F16C);
        }
#endif

#if defined(__AVX512FP16__)
        if (__builtin_cpu_supports("avx512fp16")) {
            f.set_feature(CPUFeatureSetType_FP16);
        }
#endif

#if defined(__AVX__)
        if (__builtin_cpu_supports("avx")) {
            f.set_feature(CPUFeatureSetType_AVX);
        }
#endif

#if defined(__AVX2__)
        if (__builtin_cpu_supports("avx2")) {
            f.set_feature(CPUFeatureSetType_AVX2);
        }
#endif

#if defined(__AVX512F__)
        if (__builtin_cpu_supports("avx512f")) {
            f.set_feature(CPUFeatureSetType_AVX512F);
        }
#endif

#if defined(__AVX512BF16__)
        if (__builtin_cpu_supports("avx512bf16")) {
            f.set_feature(CPUFeatureSetType_AVX512BF16);
        }
#endif

#if defined(__AVX512DQ__)
        if (__builtin_cpu_supports("avx512dq")) {
            f.set_feature(CPUFeatureSetType_AVX512DQ);
        }
#endif

#if defined(__AVX512BW__)
        if (__builtin_cpu_supports("avx512bw")) {
            f.set_feature(CPUFeatureSetType_AVX512BW);
        }
#endif

#if defined(__AVX512VNNI__)
        if (__builtin_cpu_supports("avx512vnni")) {
            f.set_feature(CPUFeatureSetType_AVX512VNNI);
        }
#endif

#if defined(__FMA__)
        if (__builtin_cpu_supports("fma")) {
            f.set_feature(CPUFeatureSetType_FMA);
        }
#endif

#if defined(USE_NPU_STRIXHALO)
        //TODO - Actually probe driver.
        f.set_feature(CPUFeatureSetType_NPU_STRIXHALO);
#endif

        return f;
    }();
    return cpu_features;
}