/**
 * @file Random/Generator.h
 * @author GreenBean
 * @brief Process-wide random seed generator for BeanTensor.
 * @version 0.1
 * @date 2026-10-28
 */
#pragma once

#include <atomic>
#include <cstdint>
#include <random>

static_assert(std::atomic<uint64_t>::is_always_lock_free);

namespace BeanTensor::Random {

namespace detail {

/**
 * @brief Splitmix64 finalizer.
 * @param x The 64-bit input value.
 * @return The mixed 64-bit output.
 */
[[nodiscard]] constexpr uint64_t mix64(uint64_t x) noexcept {
    x ^= x >> 30;
    x *= uint64_t{0xbf58476d1ce4e5b9};
    x ^= x >> 27;
    x *= uint64_t{0x94d049bb133111eb};
    x ^= x >> 31;
    return x;
}

} // namespace detail

/**
 * @class Generator
 * @brief Deterministic, thread-safe random seed generator.
 *
 * Each Generator instance holds a 64-bit seed and an atomic draw counter.
 * The stream of values is deterministic: the k-th draw from any Generator
 * constructed (or re-seeded) with the same seed value maps to the same 64-bit
 * result via @ref detail::mix64. This mapping k -> stream seed is independent
 * of how work is distributed across threads.
 *
 * Concurrent draws are serialized by the atomic counter: no draw index is
 * skipped or repeated, but which thread receives which index depends on
 * scheduling. Thread safety is provided by std::atomic members only; no
 * mutexes are used.
 */
class Generator {
public:
    /**
     * @brief Construct a Generator seeded from the system random device.
     *
     * Combines two independent std::random_device draws into the initial seed.
     */
    Generator() {
        std::random_device rd;
        const auto lo = static_cast<uint64_t>(rd());
        const auto hi = static_cast<uint64_t>(rd());
        seed_.store((hi << 32) | lo, std::memory_order_relaxed);
        counter_.store(uint64_t{0}, std::memory_order_relaxed);
    }

    /**
     * @brief Construct a Generator with an explicit seed.
     * @param seed The 64-bit seed value.
     */
    explicit Generator(uint64_t seed) {
        seed_.store(seed, std::memory_order_relaxed);
        counter_.store(uint64_t{0}, std::memory_order_relaxed);
    }

    Generator(const Generator&) = delete;
    Generator& operator=(const Generator&) = delete;

    /**
     * @brief Move-construct a Generator from another.
     * @param other The Generator to move from. Left deterministically reseedable.
     */
    Generator(Generator&& other) noexcept
        : seed_(other.seed_.load(std::memory_order_relaxed))
        , counter_(other.counter_.load(std::memory_order_relaxed)) {
        other.seed_.store(uint64_t{0}, std::memory_order_relaxed);
        other.counter_.store(uint64_t{0}, std::memory_order_relaxed);
    }

    /**
     * @brief Move-assign a Generator from another.
     * @param other The Generator to move from. Left deterministically reseedable.
     * @return A reference to this Generator.
     */
    Generator& operator=(Generator&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        const auto seed = other.seed_.load(std::memory_order_relaxed);
        const auto counter = other.counter_.load(std::memory_order_relaxed);
        seed_.store(seed, std::memory_order_relaxed);
        counter_.store(counter, std::memory_order_relaxed);
        other.seed_.store(uint64_t{0}, std::memory_order_relaxed);
        other.counter_.store(uint64_t{0}, std::memory_order_relaxed);
        return *this;
    }

    /**
     * @brief Replace the current seed and reset the draw counter.
     * @param new_seed The new 64-bit seed value.
     */
    void seed(uint64_t new_seed) noexcept {
        seed_.store(new_seed, std::memory_order_relaxed);
        counter_.store(uint64_t{0}, std::memory_order_relaxed);
    }

    /**
     * @brief Draw the next 64-bit seed value.
     * @return The k-th deterministic seed derived from the stored seed.
     */
    [[nodiscard]] uint64_t draw_seed() noexcept {
        const auto count = counter_.fetch_add(uint64_t{1}, std::memory_order_relaxed);
        return detail::mix64(seed_.load(std::memory_order_relaxed) + count);
    }

private:
    std::atomic<uint64_t> seed_;
    std::atomic<uint64_t> counter_;
};

/**
 * @brief Return the process-wide default Generator.
 * @return A reference to the function-local static Generator instance.
 */
[[nodiscard]] inline Generator& default_generator() {
    static Generator gen;
    return gen;
}

/**
 * @brief Reseed the process-wide default Generator.
 * @param new_seed The new 64-bit seed value.
 */
inline void set_seed(uint64_t new_seed) {
    default_generator().seed(new_seed);
}

} // namespace BeanTensor::Random
