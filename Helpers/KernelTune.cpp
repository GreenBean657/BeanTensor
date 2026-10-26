/**
 * @file KernelTune.cpp
 * @author GreenBean
 * @brief Implementation of the KernelTune API: default block-size table,
 * lazy thread-safe cache loading, explicit GPU autotune sweep, test hooks, and
 * explicit __host__ annotations on GPU-side host helpers/launchers.
 * @version 0.4
 * @date 2026-10-26
 */
#include "Helpers/KernelTune.h"

#include "Tensor/DTypes.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <system_error>
#include <vector>

#include "Helpers/Errors.h"

#if defined(USE_HIP)
#include <hip/hip_runtime.h>
#elif defined(USE_CUDA)
#include <cuda_runtime.h>
#endif

namespace BeanTensor::KernelTune {

    constexpr size_t kTinyBoundary = 65536;
    constexpr size_t kSmallBoundary = 2097152;
    constexpr size_t kMediumBoundary = 8388608;

    constexpr size_t kSlackBytes = 1024 * 1024;
    constexpr std::array<size_t, 4> kClassFloor = {1, kTinyBoundary + 1, kSmallBoundary + 1, kMediumBoundary + 1};
    constexpr std::array<size_t, 4> kClassCeil = {kTinyBoundary, kSmallBoundary, kMediumBoundary, std::numeric_limits<size_t>::max()};

namespace {

    constexpr std::array<size_t, 4> kDefaults = {128, 128, 256, 512};
    constexpr size_t kMaxBlockSize = 1024;
    constexpr size_t kMinBlockSize = 1;

    [[nodiscard]] size_t size_class_index(const size_t numel) noexcept {
        if (numel <= kTinyBoundary) {
            return 0;
        }
        if (numel <= kSmallBoundary) {
            return 1;
        }
        if (numel <= kMediumBoundary) {
            return 2;
        }
        return 3;
    }

    struct OverrideTable {
        std::array<std::atomic<size_t>, 4> values;
        std::array<std::atomic<bool>, 4> active;

        OverrideTable() noexcept {
            for (auto& v : values) {
                v.store(0, std::memory_order_relaxed);
            }
            for (auto& a : active) {
                a.store(false, std::memory_order_relaxed);
            }
        }
    };

    [[nodiscard]] OverrideTable& overrides() noexcept {
        static OverrideTable table;
        return table;
    }

    struct CacheState {
        std::once_flag once;
        std::string device_name;
        std::array<size_t, 4> values = kDefaults;
        std::shared_mutex mutex;
    };

    [[nodiscard]] CacheState& cache() noexcept {
        static CacheState state;
        return state;
    }

    [[nodiscard]] std::string compute_cache_path() {
        if (
            const char* const env = std::getenv("BEANTENSOR_KERNEL_TUNE");
            env != nullptr && env[0] != '\0'
        ) {
            return std::string(env);
        }
        std::string base;
        if (
            const char* const xdg = std::getenv("XDG_CACHE_HOME");
            xdg != nullptr && xdg[0] != '\0'
        ) {
            base = xdg;
        }
        else {
            const char* const home = std::getenv("HOME");
            base = (home != nullptr && home[0] != '\0')
                       ? std::string(home) + "/.cache"
                       : std::string("/tmp");
        }
        return base + "/beantensor/kernel_tune.conf";
    }

#if defined(USE_HIP) || defined(USE_CUDA)
    constexpr std::array<size_t, 4> kCandidates = {128, 256, 512, 1024};
    constexpr size_t kRunsPerCandidate = 9;

    __global__ void autotune_partial(const float32_t* __restrict__ data, float32_t* __restrict__ partials, const size_t numel) {
        extern __shared__ float32_t shared[];
        const auto tid = threadIdx.x;
        const auto gid = threadIdx.x + blockIdx.x * blockDim.x;
        const auto stride = blockDim.x * gridDim.x;
        auto acc = 0.0f;
        for (size_t i = gid; i < numel; i += stride) {
            acc += data[i];
        }
        shared[tid] = acc;
        __syncthreads();
        for (auto s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared[tid] += shared[tid + s];
            }
            __syncthreads();
        }
        if (tid == 0) {
            partials[blockIdx.x] = shared[0];
        }
    }

    __global__ void autotune_final(const float32_t* __restrict__ partials, const size_t count, float32_t* __restrict__ out) {
        extern __shared__ float32_t shared[];
        const auto tid = threadIdx.x;
        auto acc = 0.0f;
        for (auto i = tid; i < count; i += blockDim.x) {
            acc += partials[i];
        }
        shared[tid] = acc;
        __syncthreads();
        for (auto s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared[tid] += shared[tid + s];
            }
            __syncthreads();
        }
        if (tid == 0) {
            out[0] = shared[0];
        }
    }

    [[nodiscard]] __host__ size_t grid_for(const size_t numel, const size_t block_size) noexcept {
        const auto blocks = (numel + block_size - 1) / block_size;
        constexpr size_t kMaxBlocks = 1024;
        if (blocks < 1) {
            return 1;
        }
        if (blocks > kMaxBlocks) {
            return kMaxBlocks;
        }
        return blocks;
    }

#if defined(USE_HIP)
    using gpu_event_t = hipEvent_t;
    using gpu_stream_t = hipStream_t;

    inline __host__ void gpu_event_create(gpu_event_t* const e) { HIP_CHECK_ERROR(hipEventCreate(e)); }
    inline __host__ void gpu_event_destroy(const gpu_event_t e) { HIP_CHECK_ERROR(hipEventDestroy(e)); }
    inline __host__ void gpu_event_record(const gpu_event_t e, const gpu_stream_t s) { HIP_CHECK_ERROR(hipEventRecord(e, s)); }
    inline __host__ void gpu_event_sync(const gpu_event_t e) { HIP_CHECK_ERROR(hipEventSynchronize(e)); }

    [[nodiscard]] __host__ float32_t gpu_event_elapsed(const gpu_event_t start, const gpu_event_t stop) {
        float32_t ms = 0.0f;
        HIP_CHECK_ERROR(hipEventElapsedTime(&ms, start, stop));
        return ms;
    }

    inline __host__ void gpu_malloc(void** const p, const size_t n) { HIP_CHECK_ERROR(hipMalloc(p, n)); }
    inline __host__ void gpu_free(void* const p) { HIP_CHECK_ERROR(hipFree(p)); }
    inline __host__ void gpu_memcpy_h2d(void* const dst, const void* const src, const size_t n) { HIP_CHECK_ERROR(hipMemcpy(dst, src, n, hipMemcpyHostToDevice)); }
#elif defined(USE_CUDA)
    using gpu_event_t = cudaEvent_t;
    using gpu_stream_t = cudaStream_t;

    inline __host__ void gpu_event_create(gpu_event_t* const e) { CUDA_CHECK_ERROR(cudaEventCreate(e)); }
    inline __host__ void gpu_event_destroy(const gpu_event_t e) { CUDA_CHECK_ERROR(cudaEventDestroy(e)); }
    inline __host__ void gpu_event_record(const gpu_event_t e, const gpu_stream_t s) { CUDA_CHECK_ERROR(cudaEventRecord(e, s)); }
    inline __host__ void gpu_event_sync(const gpu_event_t e) { CUDA_CHECK_ERROR(cudaEventSynchronize(e)); }

    [[nodiscard]] __host__ float32_t gpu_event_elapsed(const gpu_event_t start, const gpu_event_t stop) {
        float32_t ms = 0.0f;
        CUDA_CHECK_ERROR(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }

    inline __host__ void gpu_malloc(void** const p, const size_t n) { CUDA_CHECK_ERROR(cudaMalloc(p, n)); }
    inline __host__ void gpu_free(void* const p) { CUDA_CHECK_ERROR(cudaFree(p)); }
    inline __host__ void gpu_memcpy_h2d(void* const dst, const void* const src, const size_t n) { CUDA_CHECK_ERROR(cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice)); }
#endif

    [[nodiscard]] __host__ float32_t measure_reduction(
        const size_t numel,
        const size_t block_size,
        const float32_t* d_data,
        float32_t* d_partials,
        float32_t* d_out,
        const gpu_event_t start,
        const gpu_event_t stop
    ) {
        const auto blocks = grid_for(numel, block_size);
        gpu_event_record(start, nullptr);
        autotune_partial<<<blocks, block_size, block_size * sizeof(float32_t)>>>(d_data, d_partials, numel);
        autotune_final<<<1, block_size, block_size * sizeof(float32_t)>>>(d_partials, blocks, d_out);
        gpu_event_record(stop, nullptr);
        gpu_event_sync(stop);
        return gpu_event_elapsed(start, stop);
    }

    [[nodiscard]] __host__ std::string gpu_device_name() {
#if defined(USE_HIP)
        int32_t dev = 0;
        HIP_CHECK_ERROR(hipGetDevice(&dev));
        hipDeviceProp_t props{};
        HIP_CHECK_ERROR(hipGetDeviceProperties(&props, dev));
        return std::string(props.name);
#elif defined(USE_CUDA)
        int32_t dev = 0;
        CUDA_CHECK_ERROR(cudaGetDevice(&dev));
        cudaDeviceProp props{};
        CUDA_CHECK_ERROR(cudaGetDeviceProperties(&props, dev));
        return std::string(props.name);
#endif
    }

    [[nodiscard]] __host__ bool gpu_device_available() {
#if defined(USE_HIP)
        int32_t count = 0;
        return hipGetDeviceCount(&count) == hipSuccess && count > 0;
#elif defined(USE_CUDA)
        int32_t count = 0;
        return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
#endif
    }
#endif // USE_HIP || USE_CUDA

    [[nodiscard]] std::string trim(const std::string& s) {
        const auto first = s.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) {
            return {};
        }
        const auto last = s.find_last_not_of(" \t\r\n");
        return s.substr(first, last - first + 1);
    }

    [[nodiscard]] bool parse_size_t(const std::string& s, size_t& out) {
        if (s.empty()) {
            return false;
        }
        size_t value = 0;
        for (const char c : s) {
            if (c < '0' || c > '9') {
                return false;
            }
            const auto digit = static_cast<size_t>(c - '0');
            if (
                const auto limit = (std::numeric_limits<size_t>::max() - digit) / 10;
                value > limit
            ) {
                return false;
            }
            value = value * 10 + digit;
        }
        out = value;
        return true;
    }

    void parse_cache(const std::string& path, const std::string& expected_device, std::array<size_t, 4>& out_values) {
        std::ifstream in(path);
        if (!in.is_open()) {
            return;
        }
        std::string line;
        if (!std::getline(in, line)) {
            return;
        }
        constexpr const char* kDevicePrefix = "device=";
        if (line.rfind(kDevicePrefix, 0) != 0) {
            return;
        }
        if (trim(line.substr(std::strlen(kDevicePrefix))) != expected_device) {
            return;
        }
        while (std::getline(in, line)) {
            const auto trimmed = trim(line);
            if (trimmed.empty() || trimmed[0] == '#') {
                continue;
            }
            const auto eq = trimmed.find('=');
            if (eq == std::string::npos) {
                continue;
            }
            const auto key = trimmed.substr(0, eq);
            const auto val_str = trimmed.substr(eq + 1);
            size_t idx = 0;
            if (key == "ReductionTiny") {
                idx = 0;
            }
            else if (key == "ReductionSmall") {
                idx = 1;
            }
            else if (key == "ReductionMedium") {
                idx = 2;
            }
            else if (key == "ReductionLarge") {
                idx = 3;
            }
            else {
                continue;
            }
            if (size_t val = 0;
                parse_size_t(val_str, val) &&
                val >= 1 && val <= kMaxBlockSize &&
                (val & (val - 1)) == 0
            ) {
                out_values[idx] = val;
            }
        }
    }

    void do_load_cache(const std::string& expected_device) {
        const auto path = compute_cache_path();
        auto values = kDefaults;
        parse_cache(path, expected_device, values);
        auto& state = cache();
        std::unique_lock lock(state.mutex);
        state.device_name = expected_device;
        state.values = values;
    }

} // anonymous namespace

    size_t block_size_for(const GpuFamily, const size_t numel) noexcept {
        const auto idx = size_class_index(numel);
        if (
            const auto& over = overrides();
            over.active[idx].load(std::memory_order_acquire)
            ) {
            return over.values[idx].load(std::memory_order_relaxed);
        }

        auto& state = cache();
#if defined(USE_HIP) || defined(USE_CUDA)
        try {
            std::call_once(state.once, [&] {
                try {
                    do_load_cache(gpu_device_name());
                }
                catch (...) {
                    // Unreadable or mismatched cache: keep defaults.
                }
            });
        }
        catch (...) {
            // std::call_once itself failed: keep defaults.
        }
#endif
        std::shared_lock lock(state.mutex);
        return state.values[idx];
    }

    void autotune() {
#if defined(USE_HIP) || defined(USE_CUDA)
        if (!gpu_device_available()) {
            throw ErrorHandling::NotImplemented();
        }
        const auto device_name = gpu_device_name();
        auto best_values = kDefaults;

        size_t free_bytes = 0;
        size_t total_bytes = 0;
#if defined(USE_HIP)
        HIP_CHECK_ERROR(hipMemGetInfo(&free_bytes, &total_bytes));
#elif defined(USE_CUDA)
        CUDA_CHECK_ERROR(cudaMemGetInfo(&free_bytes, &total_bytes));
#endif
        const auto reps = detail::autotune_representatives(free_bytes);
        const auto max_numel = reps.back();

        struct ResourceGuard {
            gpu_event_t start{};
            gpu_event_t stop{};
            float32_t* data = nullptr;
            float32_t* partials = nullptr;
            float32_t* out = nullptr;
            bool has_start = false;
            bool has_stop = false;

            ResourceGuard() = default;
            ResourceGuard(const ResourceGuard&) = delete;
            ResourceGuard& operator=(const ResourceGuard&) = delete;
            ResourceGuard(ResourceGuard&&) = delete;
            ResourceGuard& operator=(ResourceGuard&&) = delete;

            ~ResourceGuard() {
#if defined(USE_HIP)
                if (has_start) {
                    static_cast<void>(hipEventDestroy(start));
                }
                if (has_stop) {
                    static_cast<void>(hipEventDestroy(stop));
                }
                if (data != nullptr) {
                    static_cast<void>(hipFree(data));
                }
                if (partials != nullptr) {
                    static_cast<void>(hipFree(partials));
                }
                if (out != nullptr) {
                    static_cast<void>(hipFree(out));
                }
#elif defined(USE_CUDA)
                if (has_start) {
                    static_cast<void>(cudaEventDestroy(start));
                }
                if (has_stop) {
                    static_cast<void>(cudaEventDestroy(stop));
                }
                if (data != nullptr) {
                    static_cast<void>(cudaFree(data));
                }
                if (partials != nullptr) {
                    static_cast<void>(cudaFree(partials));
                }
                if (out != nullptr) {
                    static_cast<void>(cudaFree(out));
                }
#endif
            }
        };

        ResourceGuard guard;
        gpu_malloc(reinterpret_cast<void**>(&guard.data), max_numel * sizeof(float32_t));
        gpu_malloc(reinterpret_cast<void**>(&guard.partials), 1024 * sizeof(float32_t));
        gpu_malloc(reinterpret_cast<void**>(&guard.out), sizeof(float32_t));

        std::vector<float32_t> host_data(max_numel, 1.0f);
        gpu_memcpy_h2d(guard.data, host_data.data(), max_numel * sizeof(float32_t));

        gpu_event_create(&guard.start);
        guard.has_start = true;
        gpu_event_create(&guard.stop);
        guard.has_stop = true;

        static_cast<void>(measure_reduction(reps[0], kCandidates[1], guard.data, guard.partials, guard.out, guard.start, guard.stop));

        for (size_t size_idx = 0; size_idx < reps.size(); ++size_idx) {
            const auto numel = reps[size_idx];
            std::array<std::vector<float32_t>, kCandidates.size()> times{};
            for (auto& t : times) {
                t.reserve(kRunsPerCandidate);
            }
            for (size_t run = 0; run < kRunsPerCandidate; ++run) {
                for (size_t cand = 0; cand < kCandidates.size(); ++cand) {
                    const auto ms = measure_reduction(numel, kCandidates[cand], guard.data, guard.partials, guard.out, guard.start, guard.stop);
                    times[cand].push_back(ms);
                }
            }
            size_t best_cand = 1;
            float32_t best_ms = std::numeric_limits<float32_t>::max();
            for (size_t cand = 0; cand < kCandidates.size(); ++cand) {
                auto& t = times[cand];
                std::sort(t.begin(), t.end());
                const auto median = t[t.size() / 2];
                if (median < best_ms) {
                    best_ms = median;
                    best_cand = cand;
                }
            }
            best_values[size_idx] = kCandidates[best_cand];
        }

        guard.has_start = false;
        gpu_event_destroy(guard.start);
        guard.has_stop = false;
        gpu_event_destroy(guard.stop);
        guard.data = nullptr;
        gpu_free(guard.data);
        guard.partials = nullptr;
        gpu_free(guard.partials);
        guard.out = nullptr;
        gpu_free(guard.out);

        detail::write_cache_file(compute_cache_path(), device_name, best_values);
        do_load_cache(device_name);
#else
        throw ErrorHandling::NotImplemented();
#endif
    }

    const char* cache_path() {
        thread_local std::string path;
        path = compute_cache_path();
        return path.c_str();
    }

    void set_override(const GpuFamily, const SizeClass sc, const size_t block_size) {
        if (
            block_size < kMinBlockSize ||
            block_size > kMaxBlockSize ||
            (block_size & (block_size - 1)) != 0
        ) {
            throw ErrorHandling::InvalidKernelBlockSize(std::to_string(block_size));
        }
        const auto idx = static_cast<size_t>(sc);
        auto& over = overrides();
        over.values[idx].store(block_size, std::memory_order_relaxed);
        over.active[idx].store(true, std::memory_order_release);
    }

    void clear_override() noexcept {
        auto& over = overrides();
        for (auto& a : over.active) {
            a.store(false, std::memory_order_release);
        }
    }

    namespace detail {
        void write_cache_file(const std::string& path, const std::string& device_name, const std::array<size_t, 4>& values) {
            std::error_code ec;
            const auto parent = std::filesystem::path(path).parent_path();
            if (!parent.empty()) {
                std::filesystem::create_directories(parent, ec);
            }
            std::ofstream out(path);
            if (!out.is_open()) {
                throw ErrorHandling::CacheWriteFailed(path);
            }
            out << "device=" << device_name << "\n";
            out << "ReductionTiny=" << values[0] << "\n";
            out << "ReductionSmall=" << values[1] << "\n";
            out << "ReductionMedium=" << values[2] << "\n";
            out << "ReductionLarge=" << values[3] << "\n";
            if (!out.good()) {
                throw ErrorHandling::CacheWriteFailed(path + " (short write)");
            }
        }

        void load_cache_for_device(const std::string& device_name) {
            do_load_cache(device_name);
        }

        void reset_cache() {
            auto& state = cache();
            std::unique_lock lock(state.mutex);
            state.device_name.clear();
            state.values = kDefaults;
        }

        std::array<size_t, 4> autotune_representatives(const size_t free_bytes) {
            constexpr size_t kElementSize = sizeof(float32_t);
            constexpr std::array<size_t, 4> kFull = {32768, 1048576, 8388608, 67108864};

            if (
                constexpr size_t kSmallRepBytes = kFull[1] * kElementSize + kSlackBytes;
                free_bytes < kSmallRepBytes) {
                throw ErrorHandling::NotImplemented();
            }

            if (
                constexpr auto full_bytes = kFull[3] * kElementSize + kSlackBytes;
                free_bytes >= full_bytes
                ) {
                return kFull;
            }

            const auto usable = free_bytes - kSlackBytes;
            const auto factor = static_cast<float64_t>(usable) / static_cast<float64_t>(kFull[3] * kElementSize);
            std::array<size_t, 4> reps{};
            for (size_t i = 0; i < reps.size(); ++i) {
                const auto scaled = static_cast<size_t>(static_cast<float64_t>(kFull[i]) * factor);
                reps[i] = std::clamp(scaled, kClassFloor[i], kClassCeil[i]);
            }
            for (size_t i = 1; i < reps.size(); ++i) {
                if (reps[i] <= reps[i - 1]) {
                    reps[i] = reps[i - 1] + 1;
                }
            }
            if (reps[3] * kElementSize > usable) {
                throw ErrorHandling::NotImplemented();
            }
            return reps;
        }
    }
}
