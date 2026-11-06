/**
 * @file Lz77.h
 * @author GreenBean
 * @brief From-scratch LZ77 codec for .bt tensor records: greedy matcher with a
 * 64 KiB window and a single-entry 3-byte-prefix hash table. Chosen because
 * the project bans external libraries (zlib is not linkable), and a record
 * codec only needs "smaller on repetitive data, exactly reversible" — not
 * ratio. Token stream: 0x00 + u8 count + literal bytes (runs of 1..255),
 * 0x01 + u8 (len-3) + u16 LE offset for matches (len 3..258, offset
 * 1..65535), 0x02 end of stream. Decompression validates every byte and the
 * decoded size against the caller's expectation.
 * @version 0.1
 * @date 2026-11-05
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace BeanTensor::Serialization {

    /**
     * Compresses a byte buffer. The empty input compresses to a valid
     * zero-payload stream (a bare end token), so decompress always succeeds.
     * @brief LZ77-compress a byte buffer.
     * @param src Source bytes.
     * @param n Source length in bytes.
     * @return The compressed token stream.
     */
    [[nodiscard]] std::vector<uint8_t> lz_compress(const uint8_t* src, size_t n);

    /**
     * Decompresses a token stream produced by lz_compress. Every malformed
     * token, over-long match, and size mismatch raises — a .bt reader must
     * never accept a stream that does not decode exactly.
     * @brief LZ77-decompress a token stream.
     * @param src Token stream bytes.
     * @param n Token stream length in bytes.
     * @param expected_size Exact decoded size the stream must produce.
     * @return The decoded bytes (size == expected_size).
     * @throws ErrorHandling::CorruptFile the stream is malformed or decodes to the wrong size.
     */
    [[nodiscard]] std::vector<uint8_t> lz_decompress(const uint8_t* src, size_t n, size_t expected_size);
}
