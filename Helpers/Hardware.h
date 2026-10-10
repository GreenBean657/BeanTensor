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
    CPUFeatureSetType_NPU_STRIXHALO = 8
};

/**
 * Get the CPU feature set. If you didn't obtain this value by calling IncludedCPUFeatures(), you wont get what you are looking for. This is a static value, and will not change during runtime.
 * @brief CPU Feature Set, used to determine what features are available on the current CPU.
 */
struct alignas(8) CPUFeatureSets {
    uint32_t features;
    uint32_t threads;

    /**
     * @brief INTERNAL - Set a feature as present.
     * Set a feature as present on this CPU. Affects this CPUFeatureSets instance only.
     * @param n CPUFeatureSetTypes to set.
     */
    void set_feature(const CPUFeatureSetTypes n) {
        features |=  (1u << n);
    }

    /**
     * @brief INTERNAL - Clear a feature as present.
     * Clear a feature as present on this CPU. Affects this CPUFeatureSets instance only
     * @param n CPUFeatureSetTypes to clear.
     */
    void clear_feature(const CPUFeatureSetTypes n) {
        features &= ~(1u << n);
    }

    /**
     * @brief INTERNAL - Check if a feature is present.
     * Check if a feature is present on this CPU, based on what this CPUFeatureSets instance was told.
     * @param n CPUFeatureSetTypes to check.
     * @return [TRUE/FALSE] Present.
     */
    [[nodiscard]] bool has_feature(const CPUFeatureSetTypes n) const {
        return (features >> n) & 1u;
    }
};

/**
 * Get the CPU feature set, based on both compile-time and start-time runtime checks. This is a static value, and will not change during runtime.
 * @brief Get the CPU feature set.
 * @return The CPU feature set.
 */
inline static const CPUFeatureSets& IncludedCPUFeatures() {
    // NOLINTNEXTLINE(*-dynamic-static-initializers)
    static const CPUFeatureSets cpu_features = []{
        CPUFeatureSets f {
            .features = 0,
            .threads  = std::thread::hardware_concurrency()
        };

#if defined(__F16C__)
        if (__builtin_cpu_supports("f16c"))
            f.set_feature(CPUFeatureSetType_F16C);
#endif

#if defined(__AVX512FP16__)
        if (__builtin_cpu_supports("avx512fp16"))
            f.set_feature(CPUFeatureSetType_FP16);
#endif

#if defined(__AVX__)
        if (__builtin_cpu_supports("avx"))
            f.set_feature(CPUFeatureSetType_AVX);
#endif

#if defined(__AVX2__)
        if (__builtin_cpu_supports("avx2"))
            f.set_feature(CPUFeatureSetType_AVX2);
#endif

#if defined(__AVX512F__)
        if (__builtin_cpu_supports("avx512f"))
            f.set_feature(CPUFeatureSetType_AVX512F);
#endif

#if defined(__AVX512BF16__)
        if (__builtin_cpu_supports("avx512bf16"))
            f.set_feature(CPUFeatureSetType_AVX512BF16);
#endif

#if defined(__FMA__)
        if (__builtin_cpu_supports("fma"))
            f.set_feature(CPUFeatureSetType_FMA);
#endif

#if defined(USE_NPU_STRIXHALO)
        //TODO - Actually probe driver.
        f.set_feature(CPUFeatureSetType_NPU_STRIXHALO);
#endif

        return f;
    }();
    return cpu_features;
}