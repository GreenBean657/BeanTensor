/**
 * @file KernelTune.h
 * @author GreenBean
 * @brief Kernel autotuning API: per-(GPU family, size-class) block-size table
 * with lazy thread-safe cache loading, an explicit autotune sweep, and
 * in-memory test overrides.
 * @version 0.4
 * @date 2026-10-26
 */
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace BeanTensor::KernelTune {

    /**
     * GPU kernel families that participate in autotuning.
     * @brief Kernel family identifier.
     */
    enum class GpuFamily : uint32_t { Reduction };

    /**
     * Element-count bucket used to pick a tuned block size.
     * @brief Size class for tuning decisions.
     */
    enum class SizeClass : uint32_t { Tiny, Small, Medium, Large };

    /**
     * Returns the tuned block size for a kernel launch, given the kernel family
     * and the element count of the operation. Consults in-memory overrides
     * first, then the lazy-loaded cache, then the compiled defaults. Never
     * blocks and never throws, so it is safe to call from pool workers.
     * @brief Tuned block size for a kernel family and element count.
     * @return Selected block size.
     */
    [[nodiscard]] size_t block_size_for(GpuFamily, size_t) noexcept;

    /**
     * Runs an in-process GPU sweep of the two-stage reduction launch pattern
     * and writes the per-size-class winners to the cache file. Available only
     * in HIP/CUDA builds when a GPU device is present; CPU-only builds throw
     * ErrorHandling::NotImplemented.
     * @brief Autotune reduction block sizes for the current GPU.
     */
    void autotune();

    /**
     * Path to the kernel-tuning cache file. BEANTENSOR_KERNEL_TUNE overrides
     * the location; otherwise XDG_CACHE_HOME or ~/.cache is used, with file
     * name beantensor/kernel_tune.conf. The returned pointer is valid for the
     * duration of the call.
     * @brief Cache file path.
     * @return Null-terminated path string.
     */
    [[nodiscard]] const char* cache_path();

    /**
     * Install an in-memory override for one (family, size-class) pair. Overrides
     * are consulted before cache or defaults and are intended for tests.
     * @brief Set a test override for a kernel family, size class, and block size.
     */
    void set_override(GpuFamily, SizeClass, size_t);

    /**
     * Clear all in-memory overrides installed by set_override().
     * @brief Remove test overrides.
     */
    void clear_override() noexcept;

    /**
     * Test-facing helpers for cache file round-trips. Not part of the public
     * API contract.
     */
    namespace detail {
        /**
         * Write a canonical cache file to an arbitrary path with the supplied
         * device name and per-size-class block sizes in Tiny/Small/Medium/Large
         * order. Cached block sizes must be powers of two in the range
         * [1, 1024]; non-pow2 values are ignored when the cache is loaded.
         * @brief Write a cache file (path, device name, block-size table).
         */
        void write_cache_file(const std::string&, const std::string&, const std::array<size_t, 4>&);

        /**
         * Load a cache file from cache_path() and apply it if the header device
         * matches the supplied name.
         * @brief Load a cache file for the expected device name.
         */
        void load_cache_for_device(const std::string&);

        /**
         * Reset in-memory cache to compiled defaults. Does not touch the cache file.
         */
        void reset_cache();

        /**
         * Pure CPU helper that selects autotune representative element counts for
         * the four size classes from the available free device memory (reported
         * by hipMemGetInfo/cudaMemGetInfo). Throws ErrorHandling::NotImplemented
         * if the GPU does not have enough memory to tune even the Small-class
         * representative.
         * @brief Representative element counts in Tiny/Small/Medium/Large order.
         * @return Representative element counts in Tiny/Small/Medium/Large order.
         */
        [[nodiscard]] std::array<size_t, 4> autotune_representatives(size_t);
    }
}
