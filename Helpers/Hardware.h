/**
 * @file Hardware.h
 * @author GreenBean
 * @brief CPU feature set detection, cached runtime feature queries, cache sizing, and CCD topology detection.
 * @version 0.4
 * @date 2026-10-26
 */
#pragma once

#include <cpuid.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

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
        features |= 1u << n;
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

/**
 * CCD (Core Complex Die) topology snapshot, detected once at first use.
 * Groups are the sets of logical CPUs that share a last-level cache (one group per CCD).
 * @brief CCD topology: logical CPU groups per die, plus a CPU -> group reverse index.
 */
struct CpuCcdGroups {
    std::vector<std::vector<uint32_t>> groups; // groups[i] = sorted logical CPU indices of CCD i.
    std::vector<uint32_t> cpu_to_group;        // Indexed by logical CPU; size covers the max CPU id seen in topology (at least hardware_concurrency()).
};

/**
 * Detect the last-level-cache size in bytes via the CPUID deterministic cache leafs
 * (Intel leaf 4, AMD leaf 0x8000001D - same field layout). The largest data/unified
 * cache level wins. Legacy AMD leaf 0x80000006 is used only as a last-resort fallback
 * (it reports the total across CCDs, which is NOT what NT-store gating wants).
 * Results outside [256 KB, 1 GB] are distrusted and replaced with a 16 MB fallback.
 * Memoized; this is a static value and will not change during runtime.
 * @brief Get the last-level-cache size in bytes.
 * @return The clamped LLC size in bytes.
 */
[[nodiscard]] __inline__ __attribute__((always_inline)) static size_t LastLevelCacheBytes() {
    // NOLINTNEXTLINE(*-dynamic-static-initializers)
    static const size_t llc_bytes = [] {
        constexpr size_t kMinSaneBytes = size_t{256} * 1024u;
        constexpr size_t kMaxSaneBytes = size_t{1} * 1024u * 1024u * 1024u;
        constexpr size_t kFallbackBytes = size_t{16} * 1024u * 1024u;

        // Probe one deterministic cache leaf across all its subleafs; returns the
        // largest data (type 1) or unified (type 3) cache size in bytes, 0 if none.
        const auto probe_deterministic = [](const uint32_t leaf, const uint32_t max_leaf) -> size_t {
            if (leaf > max_leaf) {
                return 0;
            }
            size_t best_size = 0;
            for (uint32_t subleaf = 0;; subleaf++) {
                uint32_t eax = 0;
                uint32_t ebx = 0;
                uint32_t ecx = 0;
                uint32_t edx = 0;
                if (!__get_cpuid_count(leaf, subleaf, &eax, &ebx, &ecx, &edx)) {
                    break;
                }
                if (
                    const uint32_t type = eax & 0x1Fu;
                    type == 0u
                ) {
                    break; // null descriptor - end of subleaf list
                }
                else if (type == 1u || type == 3u) {
                    const size_t ways = (ebx >> 22) & 0x3FFu;
                    const size_t partitions = (ebx >> 12) & 0x3FFu;
                    const size_t line_size = ebx & 0xFFFu;
                    const size_t sets = ecx;
                    if (
                        const size_t size = (ways + 1u) * (partitions + 1u) * (line_size + 1u) * (sets + 1u);
                        size > best_size
                    ) {
                        best_size = size;
                    }
                }
            }
            return best_size;
        };

        uint32_t signature = 0;
        const uint32_t max_basic_leaf = __get_cpuid_max(0u, &signature);
        const uint32_t max_extended_leaf = __get_cpuid_max(0x80000000u, nullptr);

        size_t llc = 0;
        if (
            const size_t intel = probe_deterministic(4u, max_basic_leaf);
            intel != 0u
        ) {
            llc = intel;
        }
        if (
            const size_t amd = probe_deterministic(0x8000001Du, max_extended_leaf);
            amd > llc
        ) {
            llc = amd;
        }
        if (llc == 0u && max_extended_leaf >= 0x80000006u) {
            uint32_t eax = 0;
            uint32_t ebx = 0;
            uint32_t ecx = 0;
            uint32_t edx = 0;
            if (__get_cpuid_count(0x80000006u, 0u, &eax, &ebx, &ecx, &edx)) {
                llc = static_cast<size_t>((edx >> 18) & 0x3FFFu) * 512u * 1024u;
            }
        }
        if (llc < kMinSaneBytes || llc > kMaxSaneBytes) {
            llc = kFallbackBytes;
        }
        return llc;
    }();
    return llc_bytes;
}

/**
 * Minimum destination size in bytes before non-temporal (streaming) stores pay off.
 * Below the LLC, NT stores measured a 1.8x LOSS; past it, a 25-31% WIN. Half the
 * LLC is the gate. Memoized; this is a static value and will not change during runtime.
 * @brief Get the minimum destination buffer size for non-temporal stores.
 * @return max(LLC / 2, 64 KB) in bytes.
 */
[[nodiscard]] __inline__ __attribute__((always_inline)) static size_t NtStoreMinBytes() {
    // NOLINTNEXTLINE(*-dynamic-static-initializers)
    static const size_t threshold = std::max(LastLevelCacheBytes() / 2u, size_t{64} * 1024u);
    return threshold;
}

/**
 * Detect CCD groups from sysfs (/sys/devices/system/cpu/cpuN/cache/indexM/). Each
 * distinct shared_cpu_list among level-3 Unified indices is one CCD group. When
 * sysfs is unavailable or unparseable, degrades to a single group containing all
 * logical CPUs. Memoized; this is a static value and will not change during runtime.
 * @brief Get the CCD topology of this CPU.
 * @return The CCD groups and the CPU -> group reverse index.
 */
[[nodiscard]] __inline__ __attribute__((always_inline)) static const CpuCcdGroups& IncludedCcdTopology() {
    // NOLINTNEXTLINE(*-dynamic-static-initializers)
    static const CpuCcdGroups topology = [] {
        const uint32_t num_cpus = std::thread::hardware_concurrency();

        const auto fallback_single = [num_cpus] {
            CpuCcdGroups fallback;
            std::vector<uint32_t>& all = fallback.groups.emplace_back();
            fallback.cpu_to_group.resize(num_cpus);
            for (uint32_t cpu = 0; cpu < num_cpus; cpu++) {
                all.push_back(cpu);
                fallback.cpu_to_group[cpu] = 0;
            }
            return fallback;
        };

#if defined(__linux__)
        const std::filesystem::path base{"/sys/devices/system/cpu"};
        std::error_code ec;

        const auto read_line = [](const std::filesystem::path& file) -> std::string {
            std::ifstream in(file);
            std::string line;
            std::getline(in, line);
            return line;
        };

        // "0-7,16-23" -> {0..7, 16..23}. Malformed tokens are skipped; returns empty only on parse failure.
        const auto parse_cpu_list = [](const std::string& list) -> std::vector<uint32_t> {
            std::vector<uint32_t> cpus;
            try {
                size_t pos = 0;
                while (pos < list.size()) {
                    const auto comma = list.find(',', pos);
                    const auto token = list.substr(
                        pos,
                        comma == std::string::npos ? std::string::npos : comma - pos
                    );
                    pos = comma == std::string::npos ? list.size() : comma + 1u;

                    if (token.empty()) {
                        continue;
                    }

                    if (
                        const auto dash = token.find('-');
                        dash == std::string::npos
                    ) {
                        cpus.push_back(static_cast<uint32_t>(std::stoul(token)));
                    }
                    else {
                        const auto first = static_cast<uint32_t>(std::stoul(token.substr(0, dash)));
                        const auto last = static_cast<uint32_t>(std::stoul(token.substr(dash + 1u)));
                        if (first > last) {
                            continue;
                        }
                        for (auto v = first; v <= last; v++) {
                            cpus.push_back(v);
                        }
                    }
                }
            }
            catch (const std::exception& e) {
                if (
                    const auto* ia = dynamic_cast<const std::invalid_argument*>(&e);
                    ia != nullptr
                ) {
                    return {};
                }
                if (
                    const auto* oor = dynamic_cast<const std::out_of_range*>(&e);
                    oor != nullptr
                ) {
                    return {};
                }
                throw;
            }
            return cpus;
        };

        if (std::filesystem::exists(base, ec) && !ec) {
            std::map<std::string, std::vector<uint32_t>> lists_by_shared_cpu;
            uint32_t max_cpu_id = 0;
            for (const auto& cpu_entry : std::filesystem::directory_iterator(base, ec)) {
                if (ec) {
                    break;
                }
                const std::filesystem::path cache_dir = cpu_entry.path() / "cache";
                if (
                    const bool is_cache_dir = std::filesystem::is_directory(cache_dir, ec);
                    !is_cache_dir
                ) {
                    ec.clear(); // ENOTDIR for non-CPU entries (crash_hotplug, online, ...) - not an iteration error.
                    continue;
                }
                for (const auto& index_entry : std::filesystem::directory_iterator(cache_dir, ec)) {
                    if (ec) {
                        break;
                    }
                    if (read_line(index_entry.path() / "level") != "3") {
                        continue;
                    }
                    if (read_line(index_entry.path() / "type") != "Unified") {
                        continue;
                    }
                    const std::string shared = read_line(index_entry.path() / "shared_cpu_list");
                    if (shared.empty()) {
                        continue;
                    }
                    if (std::vector<uint32_t> cpus = parse_cpu_list(shared); !cpus.empty()) {
                        for (const auto cpu : cpus) {
                            if (cpu > max_cpu_id) {
                                max_cpu_id = cpu;
                            }
                        }
                        lists_by_shared_cpu.emplace(shared, std::move(cpus));
                    }
                }
            }
            if (!lists_by_shared_cpu.empty()) {
                CpuCcdGroups result;
                result.groups.reserve(lists_by_shared_cpu.size());
                const auto map_size = std::max(num_cpus, max_cpu_id + 1u);
                result.cpu_to_group.assign(map_size, 0);
                uint32_t group_idx = 0;
                for (auto& entry : lists_by_shared_cpu) {
                    std::vector<uint32_t>& cpus = entry.second;
                    std::ranges::sort(cpus);
                    for (const uint32_t cpu : cpus) {
                        if (cpu < result.cpu_to_group.size()) {
                            result.cpu_to_group[cpu] = group_idx;
                        }
                    }
                    result.groups.push_back(std::move(cpus));
                    group_idx++;
                }
                return result;
            }
        }
#endif

        return fallback_single();
    }();
    return topology;
}

/**
 * Ask the OS which logical CPU the calling thread runs on, then map it through
 * IncludedCcdTopology(). On any failure (non-Linux, sched_getcpu() < 0, or the
 * CPU is outside the detected topology), returns group 0.
 * @brief Get the CCD group of the CPU the calling thread currently runs on.
 * @return CCD group index, or 0 if it cannot be determined.
 */
[[nodiscard]] __inline__ __attribute__((always_inline)) static uint32_t CcdGroupOfCurrentCpu() {
#if defined(__linux__)
    if (
        const auto cpu = sched_getcpu();
        cpu >= 0
    ) {
        const auto&[groups, cpu_to_group] = IncludedCcdTopology();
        if (const auto idx = static_cast<size_t>(cpu); idx < cpu_to_group.size()) {
            return cpu_to_group[idx];
        }
    }
#endif
    return 0;
}

/**
 * Cross-CCD traffic is expensive: writes from the wrong CCD cost ~8x, reads of
 * remote-written data cost ~15-20%. Buffers that fit one CCD's share of the LLC
 * should be worked on by that CCD's threads only.
 * @brief Should work on this buffer be kept within a single CCD group?
 * @param working_set_bytes Size of the buffer in question.
 * @return true when the buffer fits within the LLC (schedule on one CCD group).
 */
[[nodiscard]] __inline__ __attribute__((always_inline)) static bool KeepWorkLocal(const size_t working_set_bytes) noexcept {
    return working_set_bytes <= LastLevelCacheBytes();
}