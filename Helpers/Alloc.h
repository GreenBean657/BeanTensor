/**
 * @file Alloc.h
 * @author GreenBean
 * @brief Single funnel for every project CPU buffer allocation. TSAN intercepts
 * aligned_alloc and strictly enforces the C11 rule that the size must be a
 * multiple of the alignment; glibc tolerates unrounded sizes, so any raw
 * aligned_alloc(64, odd_size) aborts under TSAN before tests even start.
 * All project CPU buffers must be allocated through bt_aligned_alloc, which
 * rounds the size up to a 64-byte multiple so the C11 contract always holds.
 * Freeing stays std::free. Raw aligned_alloc is banned project-wide and
 * enforced by Verifications/aligned_alloc_banned.sh.
 * @version 0.1
 * @date 2026-10-25
 */
#pragma once

#include <cstddef>
#include <cstdlib>

/**
 * Allocate a 64-byte-aligned CPU buffer of at least `bytes` bytes.
 * @brief Allocate a 64-byte-aligned CPU buffer.
 * @param bytes Requested size in bytes. Zero is allowed: a minimum 64-byte
 * block is returned because aligned_alloc(64, 0) is implementation-defined
 * and may return nullptr, while project code treats nullptr as OutOfMemory.
 * @return Pointer to the aligned buffer, or nullptr only on allocation
 * failure. Never returns nullptr for bytes == 0.
 */
[[nodiscard]] inline void* bt_aligned_alloc(const size_t bytes) noexcept {
    return std::aligned_alloc(64, ((bytes + 63) / 64) * 64);
}
