/**
 * @file Dispatcher.h
 * @author GreenBean
 * @brief CPU dtype conversion dispatcher; selects the scalar, AVX2 or AVX512 conversion path.
 * @version 0.1
 * @date 2026-10-18
 */
#pragma once


#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <stdexcept>
#include <vector>

#include "Helpers/CPUThreading.h"
#include "Helpers/Errors.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/AVX2/DTypeConversions.h"
#include "Intrinsics/AVX512/DTypeConversions.h"
#include "Intrinsics/ConversionMethods.h"
#include "Intrinsics/Native/DTypeConversions.h"
#include "Intrinsics/Shared/SharedIntrinsics.h"
#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors { class Tensor; }

namespace BeanTensor::Intrinsics::detail {

    /**
     * Map a dtype to its lane type: identity for unpacked dtypes, the half type
     * for packed dtypes (Float16x2 -> Float16, BFloat16x2 -> BFloat16).
     * @brief Lane type of a (possibly packed) dtype.
     * @tparam T Dtype to inspect.
     */
    template<typename T> struct lane_of { using type = T; };
    template<> struct lane_of<float16x2_t> { using type = float16_t; };
    template<> struct lane_of<bfloat16x2_t> { using type = bfloat16_t; };
    template<typename T> using lane_of_t = typename lane_of<T>::type;

    /**
     * Extract the low signed 4-bit lane (bits [3:0]) of an Int4x2 byte,
     * sign-extended to int8.
     * @brief Low Int4x2 lane.
     * @param v Packed byte.
     * @return Lane value in [-8, 7].
     */
    [[nodiscard]] inline int8_t int4_lane_lo(const int4x2_t v) noexcept {
        const auto nib = static_cast<uint8_t>(bit_cast_memcpy<uint8_t>(v) & 0x0F);
        return static_cast<int8_t>(nib < 8 ? nib : static_cast<uint8_t>(nib | 0xF0));
    }

    /**
     * Extract the high signed 4-bit lane (bits [7:4]) of an Int4x2 byte,
     * sign-extended to int8.
     * @brief High Int4x2 lane.
     * @param v Packed byte.
     * @return Lane value in [-8, 7].
     */
    [[nodiscard]] inline int8_t int4_lane_hi(const int4x2_t v) noexcept {
        const auto nib = static_cast<uint8_t>(bit_cast_memcpy<uint8_t>(v) >> 4);
        return static_cast<int8_t>(nib < 8 ? nib : static_cast<uint8_t>(nib | 0xF0));
    }

    /**
     * Pack two signed values into an Int4x2 byte, truncating each to its low
     * nibble (lo = bits [3:0], hi = bits [7:4]).
     * @brief Pack an Int4x2 byte.
     * @param lo Value for the low lane.
     * @param hi Value for the high lane.
     * @return Packed byte.
     */
    [[nodiscard]] inline int4x2_t pack_int4x2(const int8_t lo, const int8_t hi) noexcept {
        const auto lo_nib = static_cast<uint8_t>(static_cast<uint8_t>(lo) & 0x0F);
        const auto hi_nib = static_cast<uint8_t>((static_cast<uint8_t>(hi) & 0x0F) << 4);
        return bit_cast_memcpy<int4x2_t>(static_cast<uint8_t>(lo_nib | hi_nib));
    }

    /**
     * Clamp a signed 8-bit value into the Int4 lane range [-8, 7]. Out-of-range
     * input sets flag under any non-PERMISSIVE method.
     * @brief Clamp into the Int4 lane range.
     * @param v Value to clamp.
     * @param flag Reference set to true on clamp under a non-PERMISSIVE method.
     * @param method Active ConversionClampMethod.
     * @return Clamped value in [-8, 7].
     */
    [[nodiscard]] inline int8_t clamp_int4_lane(const int8_t v, bool& flag, const ConversionClampMethod method) noexcept {
        if (v > 7) {
            if (method != ConversionClampMethod::PERMISSIVE) {
                flag = true;
            }
            return int8_t{7};
        }
        if (v < -8) {
            if (method != ConversionClampMethod::PERMISSIVE) {
                flag = true;
            }
            return int8_t{-8};
        }
        return v;
    }

    /**
     * Convert a single scalar (non-packed) value between two dtypes.
     * @brief Scalar (non-packed) lane conversion.
     * @tparam Src Source dtype.
     * @tparam Dst Destination dtype.
     * @param v Value to convert.
     * @param flag Reference set to true if a conversion error the active method cares about occurs.
     * @param method Active ConversionClampMethod - How should conversion tolerance be handled?
     * @return Value converted to Dst.
     */
    template<typename Src, typename Dst>
    [[nodiscard]] Dst scalar_convert(const Src& v, bool& flag, const ConversionClampMethod method) {
        if constexpr (std::is_same_v<Src, Dst>) {
            return v;
        } else if constexpr (std::is_same_v<Src, bfloat16_t>) {
            return scalar_convert<float32_t, Dst>(unaccel_bf16_to_fp32(v), flag, method);
        } else if constexpr (std::is_same_v<Src, float16_t>) {
            return scalar_convert<float32_t, Dst>(unaccel_fp16_to_fp32_tracked(v, flag, method), flag, method);
        } else if constexpr (std::is_same_v<Dst, bfloat16_t>) {
            if constexpr (std::is_same_v<Src, float32_t>) {
                return unaccel_fp32_to_bf16_tracked(v, flag, method);
            } else if constexpr (std::is_floating_point_v<Src>) {
                // FP64 narrows through FP32 (double rounding, <=1 ulp at boundaries);
                // scalar and AVX512 paths funnel identically, so they stay bit-exact.
                const float32_t f = standard2standard<Src, float32_t>(v, flag, method);
                return unaccel_fp32_to_bf16_tracked(f, flag, method);
            } else {
                return unaccel_fp32_to_bf16_tracked(static_cast<float32_t>(v), flag, method);
            }
        } else if constexpr (std::is_same_v<Dst, float16_t>) {
            if constexpr (std::is_same_v<Src, float32_t>) {
                return unaccel_fp32_to_fp16_tracked(v, flag, method);
            } else if constexpr (std::is_floating_point_v<Src>) {
                // FP64 narrows through FP32 (double rounding, <=1 ulp at boundaries);
                // scalar and AVX512 paths funnel identically, so they stay bit-exact.
                const float32_t f = standard2standard<Src, float32_t>(v, flag, method);
                return unaccel_fp32_to_fp16_tracked(f, flag, method);
            } else {
                return unaccel_fp32_to_fp16_tracked(static_cast<float32_t>(v), flag, method);
            }
        } else if constexpr ((std::is_floating_point_v<Src> && std::is_floating_point_v<Dst>) ||
                             (std::is_integral_v<Src> && std::is_integral_v<Dst>)) {
            return standard2standard<Src, Dst>(v, flag, method);
        } else if constexpr (std::is_integral_v<Src>) {
            // int -> float: always representable in range.
            return static_cast<Dst>(v);
        } else {
            // float -> int: range-check against the integer bounds.
            if (std::isnan(v)) {
                if (method != ConversionClampMethod::PERMISSIVE) flag = true;
                return Dst{0};
            }
            // Compare in the source type. When Src cannot represent Dst's max
            // exactly (e.g. int64 in float64_t), the bound rounds up to a value
            // already outside Dst's range, so equality there is also overflow.
            // Written as !(v < max_bound) rather than v == max_bound: identical
            // semantics (NaN is handled above), but -Werror=float-equal clean.
            const Src max_bound = static_cast<Src>(std::numeric_limits<Dst>::max());
            if (
                constexpr bool max_exact = std::numeric_limits<Src>::digits >= std::numeric_limits<Dst>::digits; v > max_bound || (!max_exact && !(v < max_bound))
                ) {
                if (method != ConversionClampMethod::PERMISSIVE) flag = true;
                return std::numeric_limits<Dst>::max();
            }
            // lowest() is a power of two (or zero): always exact in Src.
            if (v < static_cast<Src>(std::numeric_limits<Dst>::lowest())) {
                if (method != ConversionClampMethod::PERMISSIVE) flag = true;
                return std::numeric_limits<Dst>::lowest();
            }
            return static_cast<Dst>(v);
        }
    }

    /**
     * Convert a single element between two dtypes, packed-dtype aware.
     * @note numel/shape are always preserved: unpacked -> packed duplicates the
     * converted value into both lanes, packed -> unpacked converts the first
     * lane, packed -> packed converts both lanes independently.
     * @brief Element conversion, packed-dtype aware.
     * @tparam Src Source dtype.
     * @tparam Dst Destination dtype.
     * @param v Value to convert.
     * @param flag Reference set to true if a conversion error the active method cares about occurs.
     * @param method Active ConversionClampMethod.
     * @return Value converted to Dst.
     */
    template<typename Src, typename Dst>
    [[nodiscard]] inline Dst element_convert(const Src& v, bool& flag, const ConversionClampMethod method) {
        if constexpr (std::is_same_v<Src, Dst>) {
            return v;
        } else if constexpr (std::is_same_v<Src, int4x2_t>) {
            if constexpr (std::is_same_v<Dst, float16x2_t> || std::is_same_v<Dst, bfloat16x2_t>) {
                // Int4x2 -> packed 16-bit: convert both lanes independently.
                using H = lane_of_t<Dst>;
                const auto lo_c = static_cast<std::uint16_t>(bit_cast_memcpy<std::uint16_t>(scalar_convert<int8_t, H>(int4_lane_lo(v), flag, method)));
                const auto hi_c = static_cast<std::uint16_t>(bit_cast_memcpy<std::uint16_t>(scalar_convert<int8_t, H>(int4_lane_hi(v), flag, method)));
                return bit_cast_memcpy<Dst>(static_cast<std::uint32_t>(lo_c) | (static_cast<std::uint32_t>(hi_c) << 16));
            } else {
                // Int4x2 -> unpacked: first (lo) lane only.
                return scalar_convert<int8_t, Dst>(int4_lane_lo(v), flag, method);
            }
        } else if constexpr (std::is_same_v<Dst, int4x2_t>) {
            if constexpr (std::is_same_v<Src, float16x2_t> || std::is_same_v<Src, bfloat16x2_t>) {
                // Packed 16-bit -> Int4x2: convert both lanes independently.
                using H = lane_of_t<Src>;
                const std::uint32_t bits = bit_cast_memcpy<std::uint32_t>(v);
                const H lo = bit_cast_memcpy<H>(static_cast<std::uint16_t>(bits & 0xFFFF));
                const H hi = bit_cast_memcpy<H>(static_cast<std::uint16_t>(bits >> 16));
                return pack_int4x2(
                    clamp_int4_lane(scalar_convert<H, int8_t>(lo, flag, method), flag, method),
                    clamp_int4_lane(scalar_convert<H, int8_t>(hi, flag, method), flag, method));
            } else {
                // Unpacked -> Int4x2: duplicate the clamped value into both lanes.
                const auto c = clamp_int4_lane(scalar_convert<Src, int8_t>(v, flag, method), flag, method);
                return pack_int4x2(c, c);
            }
        } else if constexpr (!DataTypes::requires_unpack_v<Src> && !DataTypes::requires_unpack_v<Dst>) {
            return scalar_convert<Src, Dst>(v, flag, method);
        } else if constexpr (DataTypes::requires_unpack_v<Dst> && !DataTypes::requires_unpack_v<Src>) {
            using H = lane_of_t<Dst>;
            const auto bits = static_cast<std::uint16_t>(bit_cast_memcpy<std::uint16_t>(scalar_convert<Src, H>(v, flag, method)));
            return bit_cast_memcpy<Dst>(static_cast<std::uint32_t>(bits) | (static_cast<std::uint32_t>(bits) << 16));
        } else if constexpr (DataTypes::requires_unpack_v<Src> && !DataTypes::requires_unpack_v<Dst>) {
            using H = lane_of_t<Src>;
            const std::uint32_t bits = bit_cast_memcpy<std::uint32_t>(v);
            const H lo = bit_cast_memcpy<H>(static_cast<std::uint16_t>(bits & 0xFFFF));
            return scalar_convert<H, Dst>(lo, flag, method);
        } else {
            using HS = lane_of_t<Src>;
            using HD = lane_of_t<Dst>;
            const std::uint32_t bits = bit_cast_memcpy<std::uint32_t>(v);
            const HS lo = bit_cast_memcpy<HS>(static_cast<std::uint16_t>(bits & 0xFFFF));
            const HS hi = bit_cast_memcpy<HS>(static_cast<std::uint16_t>(bits >> 16));
            const auto lo_c = static_cast<std::uint16_t>(bit_cast_memcpy<std::uint16_t>(scalar_convert<HS, HD>(lo, flag, method)));
            const auto hi_c = static_cast<std::uint16_t>(bit_cast_memcpy<std::uint16_t>(scalar_convert<HS, HD>(hi, flag, method)));
            return bit_cast_memcpy<Dst>(static_cast<std::uint32_t>(lo_c) | (static_cast<std::uint32_t>(hi_c) << 16));
        }
    }

    /**
     * Convert a contiguous range of elements. Runs on the calling thread.
     * @note Accumulates conversion events into a thread-local flag and performs
     * a single relaxed store to the shared flag on exit — the per-element hot
     * loop never touches shared state.
     * @brief Convert [begin, end) of src into dst.
     * @tparam Src Source dtype.
     * @tparam Dst Destination dtype.
     * @param src Source buffer.
     * @param dst Destination buffer.
     * @param begin First element index.
     * @param end One past the last element index.
     * @param flag Shared atomic flag for the whole conversion.
     * @param method Active ConversionClampMethod.
     */
    template<typename Src, typename Dst>
    void convert_range(const void* src, void* dst, const size_t begin, const size_t end,
                       std::atomic<bool>& flag, const ConversionClampMethod method) {
        const auto* s = static_cast<const Src*>(src);
        auto* d = static_cast<Dst*>(dst);
        bool local_flag = false;
        for (size_t i = begin; i < end; ++i)
            d[i] = element_convert<Src, Dst>(s[i], local_flag, method);
        if (local_flag) flag.store(true, std::memory_order_relaxed);
    }

    /**
     * Convert a buffer, chunking across the CPU thread pool when the tensor is
     * large enough to be worth it (numel > threads * 1024); smaller buffers
     * convert inline on the calling thread.
     * @brief Chunked, thread-pool dispatched conversion.
     * @tparam Src Source dtype.
     * @tparam Dst Destination dtype.
     * @param src Source buffer.
     * @param dst Destination buffer.
     * @param numel Number of elements.
     * @param flag Shared atomic flag for the whole conversion.
     * @param method Active ConversionClampMethod.
     * @param futures Out-vector receiving the chunk futures (empty when run inline).
     */
    template<typename Src, typename Dst>
    void convert_chunked(const void* src, void* dst, const size_t numel,
                         std::atomic<bool>& flag, const ConversionClampMethod method,
                         std::vector<std::shared_future<void>>& futures) {
        const size_t t_count = IncludedCPUFeatures().threads;
        if (numel <= t_count * 1024 || t_count < 2) {
            convert_range<Src, Dst>(src, dst, 0, numel, flag, method);
            return;
        }
        auto& pool = Threading::GetCPUThreadPool();
        const size_t chunk_size = (numel + t_count - 1) / t_count;
        for (size_t t = 0; t < t_count; ++t) {
            const size_t t_start = t * chunk_size;
            const size_t t_end = std::min(t_start + chunk_size, numel);
            if (t_start >= numel) break;
            futures.push_back(pool.submit(
                [=, &flag] { convert_range<Src, Dst>(src, dst, t_start, t_end, flag, method); }
            ).share());
        }
    }

    /**
     * Dispatch a conversion from Src to the runtime-selected destination dtype.
     * @brief Runtime dtype switch over convert_chunked.
     * @tparam Src Source dtype.
     * @param new_dtype Destination dtype.
     * @param src Source buffer.
     * @param dst Destination buffer.
     * @param numel Number of elements.
     * @param flag Shared atomic flag for the whole conversion.
     * @param method Active ConversionClampMethod.
     * @param futures Out-vector receiving the chunk futures.
     * @throws ErrorHandling::NotImplemented Unknown destination dtype.
     */
    template<typename Src>
    void convert_to_dst(const DataTypes::DType new_dtype, const void* src, void* dst, const size_t numel,
                        std::atomic<bool>& flag, const ConversionClampMethod method,
                        std::vector<std::shared_future<void>>& futures) {
        using DType = DataTypes::DType;
        switch (new_dtype) {
            case DType::BFloat16:   convert_chunked<Src, bfloat16_t>(src, dst, numel, flag, method, futures); break;
            case DType::BFloat16x2: convert_chunked<Src, bfloat16x2_t>(src, dst, numel, flag, method, futures); break;
            case DType::Float16:    convert_chunked<Src, float16_t>(src, dst, numel, flag, method, futures); break;
            case DType::Float16x2:  convert_chunked<Src, float16x2_t>(src, dst, numel, flag, method, futures); break;
            case DType::Float32:    convert_chunked<Src, float32_t>(src, dst, numel, flag, method, futures); break;
            case DType::Float64:    convert_chunked<Src, float64_t>(src, dst, numel, flag, method, futures); break;
            case DType::Int64:      convert_chunked<Src, int64_t>(src, dst, numel, flag, method, futures); break;
            case DType::Int32:      convert_chunked<Src, int32_t>(src, dst, numel, flag, method, futures); break;
            case DType::Int16:      convert_chunked<Src, int16_t>(src, dst, numel, flag, method, futures); break;
            case DType::Int8:       convert_chunked<Src, int8_t>(src, dst, numel, flag, method, futures); break;
            case DType::UInt64:     convert_chunked<Src, uint64_t>(src, dst, numel, flag, method, futures); break;
            case DType::UInt32:     convert_chunked<Src, uint32_t>(src, dst, numel, flag, method, futures); break;
            case DType::UInt16:     convert_chunked<Src, uint16_t>(src, dst, numel, flag, method, futures); break;
            case DType::UInt8:      convert_chunked<Src, uint8_t>(src, dst, numel, flag, method, futures); break;
            case DType::Int4x2:      convert_chunked<Src, int4x2_t>(src, dst, numel, flag, method, futures); break;
            [[unlikely]] default: throw ErrorHandling::NotImplemented();
        }
    }

    /**
     * Queue a CPU tensor's conversion to a new dtype, ping-ponging through the
     * tensor's convert_buffer. Asynchronous: the conversion chunks and a
     * finalize task (buffer swap + dtype flip) run on the thread pool; the
     * tensor's sync() collects them. Defined in Intrinsics/Dispatcher.cpp
     * (needs Tensor internals).
     * @throws std::range_error At the tensor's sync(): values were
     * clamped/invalid under a non-PERMISSIVE method.
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource Non-CPU tensor.
     */
    void launch_conv_cpu(Tensors::Tensor& t, const DataTypes::DType& new_dtype, ConversionClampMethod method);
}
