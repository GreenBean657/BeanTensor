/**
 * @file Lz77.cpp
 * @author GreenBean
 * @brief Implementation of the from-scratch LZ77 codec declared in Lz77.h.
 * @version 0.1
 * @date 2026-11-05
 */
#include "Serialization/Lz77.h"

#include <cstring>

#include "Helpers/Errors.h"

namespace BeanTensor::Serialization {
    namespace {
        constexpr uint32_t k_window = 65535;
        constexpr size_t k_hash_size = 1 << 16;
        constexpr size_t k_min_match = 3;
        constexpr size_t k_max_match = 258;

        /**
         * Hashes three consecutive bytes into the match table index.
         * @brief 3-byte prefix hash.
         */
        size_t hash3(const uint8_t* p) {
            const auto v = (static_cast<uint32_t>(p[0]) << 16) |
                           (static_cast<uint32_t>(p[1]) << 8) |
                           static_cast<uint32_t>(p[2]);
            return ((v * uint32_t{2654435761}) >> 16) & (k_hash_size - 1);
        }

        void push_literal_run(std::vector<uint8_t>& out, const uint8_t* run, size_t len) {
            size_t i = 0;
            while (i < len) {
                const auto chunk = static_cast<uint8_t>((len - i) > 255 ? 255 : (len - i));
                out.push_back(0x00);
                out.push_back(chunk);
                out.insert(out.end(), run + i, run + i + chunk);
                i += chunk;
            }
        }
    }

    std::vector<uint8_t> lz_compress(const uint8_t* src, const size_t n) {
        auto out = std::vector<uint8_t>();
        out.reserve(n / 2 + 16);

        auto table = std::vector<size_t>(k_hash_size, SIZE_MAX);
        size_t lit_start = 0;
        size_t pos = 0;
        while (pos < n) {
            size_t best_len = 0;
            size_t best_off = 0;
            if (pos + k_min_match <= n) {
                const size_t cand = table[hash3(src + pos)];
                if (cand != SIZE_MAX && cand < pos && pos - cand <= k_window) {
                    const size_t limit = (n - pos) < k_max_match ? (n - pos) : k_max_match;
                    size_t len = 0;
                    while (len < limit && src[cand + len] == src[pos + len]) {
                        ++len;
                    }
                    if (len >= k_min_match) {
                        best_len = len;
                        best_off = pos - cand;
                    }
                }
                table[hash3(src + pos)] = pos;
            }
            if (best_len >= k_min_match) {
                push_literal_run(out, src + lit_start, pos - lit_start);
                out.push_back(0x01);
                out.push_back(static_cast<uint8_t>(best_len - k_min_match));
                out.push_back(static_cast<uint8_t>(best_off & 0xFF));
                out.push_back(static_cast<uint8_t>((best_off >> 8) & 0xFF));
                for (size_t j = 1; j < best_len; ++j) {
                    if (pos + j + k_min_match <= n) {
                        table[hash3(src + pos + j)] = pos + j;
                    }
                }
                pos += best_len;
                lit_start = pos;
            } else {
                ++pos;
            }
        }
        push_literal_run(out, src + lit_start, n - lit_start);
        out.push_back(0x02);
        return out;
    }

    std::vector<uint8_t> lz_decompress(const uint8_t* src, const size_t n, const size_t expected_size) {
        auto out = std::vector<uint8_t>();
        out.reserve(expected_size);
        size_t i = 0;
        bool ended = false;
        while (i < n && !ended) {
            if (
                const auto token = src[i++];
                token == 0x00
                ) {
                if (i >= n) {
                    throw ErrorHandling::CorruptFile("lz stream: truncated literal run");
                }
                const auto count = static_cast<size_t>(src[i++]);
                if (i + count > n) {
                    throw ErrorHandling::CorruptFile("lz stream: literal run overruns input");
                }
                if (count > expected_size - out.size()) {
                    throw ErrorHandling::CorruptFile("lz stream: literal run overruns expected size");
                }
                out.insert(out.end(), src + i, src + i + count);
                i += count;
            } else if (token == 0x01) {
                if (i + 3 > n) {
                    throw ErrorHandling::CorruptFile("lz stream: truncated match token");
                }
                const auto len = static_cast<size_t>(src[i]) + k_min_match;
                ++i;
                const auto off_lo = static_cast<size_t>(src[i]);
                ++i;
                const auto off_hi = static_cast<size_t>(src[i]);
                ++i;
                const auto off = off_lo | (off_hi << 8);
                if (off == 0 || off > out.size()) {
                    throw ErrorHandling::CorruptFile("lz stream: match offset out of range");
                }
                if (len > expected_size - out.size()) {
                    throw ErrorHandling::CorruptFile("lz stream: match overruns expected size");
                }
                for (size_t j = 0; j < len; ++j) {
                    out.push_back(out[out.size() - off]);
                }
            } else if (token == 0x02) {
                if (i != n) {
                    throw ErrorHandling::CorruptFile("lz stream: bytes after the end token");
                }
                ended = true;
            } else {
                throw ErrorHandling::CorruptFile("lz stream: unknown token");
            }
        }
        if (!ended) {
            throw ErrorHandling::CorruptFile("lz stream: missing end token");
        }
        if (out.size() != expected_size) {
            throw ErrorHandling::CorruptFile("lz stream: decoded size mismatch");
        }
        return out;
    }
}
