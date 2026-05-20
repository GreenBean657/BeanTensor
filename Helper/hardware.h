#pragma once
#include <cstdint>
#include <thread>
namespace BeanTensor::Hardware {
    struct CPUFeatureSet {
        bool f16c;
        bool fp16;
        bool avx;
        bool avx2;
        bool avx512f;
        bool avx512bf16;
        bool fma;
        bool npu_strix_halo;
        uint8_t threads;
    };

    inline CPUFeatureSet detect_cpu() {
        CPUFeatureSet cpuFeatureSet{};
        cpuFeatureSet.f16c = __builtin_cpu_supports("f16c");
        cpuFeatureSet.fp16 = __builtin_cpu_supports("avx512fp16");
        cpuFeatureSet.avx = __builtin_cpu_supports("avx");
        cpuFeatureSet.avx2 = __builtin_cpu_supports("avx2");
        cpuFeatureSet.avx512f = __builtin_cpu_supports("avx512f");
        cpuFeatureSet.avx512bf16 = __builtin_cpu_supports("avx512bf16");
        cpuFeatureSet.fma = __builtin_cpu_supports("fma");
        cpuFeatureSet.threads = std::thread::hardware_concurrency();
#ifdef USE_NPU_STRIXHALO
        cpuFeatureSet.npu_strix_halo = true;
#else
        cpuFeatureSet.npu_strix_halo = false;
#endif
        if (cpuFeatureSet.threads == 0) {
            cpuFeatureSet.threads = 4;
        }
        return cpuFeatureSet;
    }

    inline const CPUFeatureSet& CPU() {
        static const CPUFeatureSet CPUFeatures = detect_cpu();
        return CPUFeatures;
    }
    inline const std::string& cputostring() {
        static const std::string CPUString = "CPU Features: " +
            std::string(CPU().f16c ? "f16c " : "") +
            std::string(CPU().fp16 ? "fp16 " : "") +
            std::string(CPU().avx ? "avx " : "") +
            std::string(CPU().avx2 ? "avx2 " : "") +
            std::string(CPU().avx512f ? "avx512f " : "") +
            std::string(CPU().avx512bf16 ? "avx512bf16 " : "") +
            std::string(CPU().fma ? "fma " : "") +
            std::string(CPU().npu_strix_halo ? "npu_strix_halo " : "") +
            "\nThreads: " + std::to_string(CPU().threads);
        return CPUString;
    }
}