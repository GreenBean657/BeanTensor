#pragma once
#include <cstdint>
#include <thread>
namespace BeanTensor::Hardware {
    struct CPUFeatureSet {
        bool f16c;
        bool avx;
        bool avx2;
        bool avx512f;
        bool avx512bf16;
        bool fma;
        uint8_t threads;
    };

    inline CPUFeatureSet detect_cpu() {
        CPUFeatureSet cpuFeatureSet{};
        cpuFeatureSet.f16c = __builtin_cpu_supports("f16c");
        cpuFeatureSet.avx = __builtin_cpu_supports("avx");
        cpuFeatureSet.avx2 = __builtin_cpu_supports("avx2");
        cpuFeatureSet.avx512f = __builtin_cpu_supports("avx512f");
        cpuFeatureSet.avx512bf16 = __builtin_cpu_supports("avx512bf16");
        cpuFeatureSet.fma = __builtin_cpu_supports("fma");
        cpuFeatureSet.threads = std::thread::hardware_concurrency();
        if (cpuFeatureSet.threads == 0) {
            cpuFeatureSet.threads = 4;
        }
        return cpuFeatureSet;
    }

    inline const CPUFeatureSet& CPU() {
        static const CPUFeatureSet CPUFeatures = detect_cpu();
        return CPUFeatures;
    }
}