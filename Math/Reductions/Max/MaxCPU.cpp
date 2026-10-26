/**
 * @file MaxCPU.cpp
 * @author GreenBean
 * @brief CPU backend for max. Hosts the Tensor::max method definition
 * (mirrors how Tensor::sum lives in SumCPU.cpp); max has no autograd node —
 * the output never joins the computation graph.
 * @version 0.1
 * @date 2026-10-25
 */
#include <cstdint>
#include <future>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "Math/Reductions/Max/Max.h"
#include "Math/Reductions/ReductionsShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/Hardware.h"
#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Tensors::detail {
    using namespace reductions_shared;

    namespace {
        /**
         * The winning value is always carried as the raw bits of an actual input element, so the
         * non-NaN result is bit-for-bit one of the inputs (+0.0 included). A NaN anywhere
         * canonicalizes to the quiet NaN with an empty payload, which keeps the combine
         * associative and commutative, so every backend and chunking agrees bit-for-bit.
         * @brief Canonical quiet NaN of a scalar float dtype, by bit pattern.
         * @return The canonical NaN as T.
         */
        template<typename T>
        [[nodiscard]] T fp_canon_nan() {
            if constexpr (std::is_same_v<T, float32_t>) {
                return Intrinsics::detail::bit_cast_memcpy<float32_t>(UINT32_C(0x7FC00000));
            } else {
                static_assert(std::is_same_v<T, float64_t>, "fp_canon_nan: scalar float dtype expected");
                return Intrinsics::detail::bit_cast_memcpy<float64_t>(UINT64_C(0x7FF8000000000000));
            }
        }

        /**
         * @brief Positive zero of a scalar float dtype, by bit pattern.
         * @return +0.0 as T.
         */
        template<typename T>
        [[nodiscard]] T fp_pos_zero() {
            if constexpr (std::is_same_v<T, float32_t>) {
                return Intrinsics::detail::bit_cast_memcpy<float32_t>(UINT32_C(0x00000000));
            } else {
                static_assert(std::is_same_v<T, float64_t>, "fp_pos_zero: scalar float dtype expected");
                return Intrinsics::detail::bit_cast_memcpy<float64_t>(UINT64_C(0x0000000000000000));
            }
        }

        /**
         * @brief Is a scalar float exactly +/-0.0? (bit test, no float ==)
         * @param v Value to test.
         * @return T/F - the magnitude bits are all zero.
         */
        template<typename T>
        [[nodiscard]] bool fp_is_zero(const T v) {
            if constexpr (std::is_same_v<T, float32_t>) {
                return (Intrinsics::detail::bit_cast_memcpy<uint32_t>(v) & UINT32_C(0x7FFFFFFF)) == 0u;
            } else {
                static_assert(std::is_same_v<T, float64_t>, "fp_is_zero: scalar float dtype expected");
                return (Intrinsics::detail::bit_cast_memcpy<uint64_t>(v) & UINT64_C(0x7FFFFFFFFFFFFFFF)) == 0u;
            }
        }

        /**
         * NaN-propagating max combine. Distinct-bit values that compare equal can only be
         * +0.0 vs -0.0 (NaN is handled above), so the zero tie canonicalizes to +0.0 — that
         * is what keeps the result bit-identical across backends and chunk counts.
         * @brief Combine two scalar float partials, max semantics.
         * @param a Left partial.
         * @param b Right partial.
         * @return The winning partial (any NaN propagates as the canonical quiet NaN).
         */
        template<typename T>
        [[nodiscard]] T fp_max_combine(const T a, const T b) {
            if (__builtin_isnan(a) || __builtin_isnan(b)) {
                return fp_canon_nan<T>();
            }
            if (b > a) {
                return b;
            }
            if (a > b) {
                return a;
            }
            if (fp_is_zero(a)) {
                return fp_pos_zero<T>();
            }
            return a;
        }

        /**
         * @brief Canonical quiet NaN lane bits (fp16 0x7E00, bf16 0x7FC0).
         * @return The canonical NaN bits of the lane type.
         */
        template<typename H>
        [[nodiscard]] uint16_t half_nan_bits() {
            if constexpr (std::is_same_v<H, float16_t>) {
                return uint16_t{0x7E00};
            } else {
                static_assert(std::is_same_v<H, bfloat16_t>, "half_nan_bits: half lane type expected");
                return uint16_t{0x7FC0};
            }
        }

        /**
         * @brief -inf lane bits, the max identity (fp16 0xFC00, bf16 0xFF80).
         * @return The identity bits of the lane type.
         */
        template<typename H>
        [[nodiscard]] uint16_t half_identity_bits() {
            if constexpr (std::is_same_v<H, float16_t>) {
                return uint16_t{0xFC00};
            } else {
                static_assert(std::is_same_v<H, bfloat16_t>, "half_identity_bits: half lane type expected");
                return uint16_t{0xFF80};
            }
        }

        /**
         * The winner is carried as raw lane bits; comparison runs on the widened fp32 lane.
         * fp16/bf16 -> fp32 is injective, so a comparison tie means identical lane bits,
         * except for the +0.0/-0.0 pair, which canonicalizes to +0.0 for max.
         * @brief Combine two half-precision lane partials, max semantics.
         * @param a_bits Left partial lane bits.
         * @param b_bits Right partial lane bits.
         * @return The winning lane bits (any NaN propagates as the canonical quiet NaN).
         */
        template<typename H>
        [[nodiscard]] uint16_t half_max_combine(const uint16_t a_bits, const uint16_t b_bits) {
            const auto a = widen_lane(Intrinsics::detail::bit_cast_memcpy<H>(a_bits));
            const auto b = widen_lane(Intrinsics::detail::bit_cast_memcpy<H>(b_bits));
            if (__builtin_isnan(a) || __builtin_isnan(b)) {
                return half_nan_bits<H>();
            }
            if (b > a) {
                return b_bits;
            }
            if (a > b) {
                return a_bits;
            }
            if ((a_bits & uint16_t{0x7FFF}) == 0) {
                return uint16_t{0x0000}; // +0.0
            }
            return a_bits;
        }

        /**
         * @brief Scalar range max (fp32/fp64), -inf identity, NaN-propagating combine.
         * @param data Input elements.
         * @param begin First element compared.
         * @param end One past the last element compared.
         * @return Max of the elements in [begin, end).
         */
        template<typename T>
        [[nodiscard]] T max_range_scalar(const T* data, const size_t begin, const size_t end) {
            auto best = -std::numeric_limits<T>::infinity();
            for (auto i = begin; i < end; ++i) {
                best = fp_max_combine(best, data[i]);
            }
            return best;
        }

        /**
         * @brief Scalar range max (fp16/bf16), raw winning lane bits.
         * @param data Input elements.
         * @param begin First element compared.
         * @param end One past the last element compared.
         * @return Bits of the max element in [begin, end).
         */
        template<typename H>
        [[nodiscard]] uint16_t max_range_half(const H* data, const size_t begin, const size_t end) {
            auto best = half_identity_bits<H>();
            for (auto i = begin; i < end; ++i) {
                best = half_max_combine<H>(best, Intrinsics::detail::bit_cast_memcpy<uint16_t>(data[i]));
            }
            return best;
        }

        /**
         * Lanes reduce independently; the per-lane winners may come from different elements
         * (same convention as Sum's per-lane totals).
         * @brief Scalar range max (fp16x2/bf16x2, per-lane, raw winning lane bits).
         * @param data Input elements.
         * @param begin First element compared.
         * @param end One past the last element compared.
         * @return {max_lo_bits, max_hi_bits} of the elements in [begin, end).
         */
        template<typename P>
        [[nodiscard]] std::pair<uint16_t, uint16_t> max_range_packed(const P* data, const size_t begin, const size_t end) {
            using H = packed_lane_t<P>;
            auto best_lo = half_identity_bits<H>();
            auto best_hi = half_identity_bits<H>();
            for (auto i = begin; i < end; ++i) {
                const auto bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(data[i]);
                best_lo = half_max_combine<H>(best_lo, static_cast<uint16_t>(bits & 0xFFFFu));
                best_hi = half_max_combine<H>(best_hi, static_cast<uint16_t>(bits >> 16));
            }
            return {best_lo, best_hi};
        }

        /**
         * @brief Scalar range max (all integer dtypes), lowest() identity, plain compare.
         * @param data Input elements.
         * @param begin First element compared.
         * @param end One past the last element compared.
         * @return Max of the elements in [begin, end).
         */
        template<typename T>
        [[nodiscard]] T max_range_int(const T* data, const size_t begin, const size_t end) {
            auto best = std::numeric_limits<T>::lowest();
            for (auto i = begin; i < end; ++i) {
                if (data[i] > best) {
                    best = data[i];
                }
            }
            return best;
        }

        /**
         * @brief Scalar range max (int4x2, per-lane signed compare).
         * @param data Input elements.
         * @param begin First element compared.
         * @param end One past the last element compared.
         * @return {max_lo, max_hi} of the elements in [begin, end).
         */
        [[nodiscard]] std::pair<int8_t, int8_t> max_range_i4x2(const int4x2_t* data, const size_t begin, const size_t end) {
            auto best_lo = int8_t{-8};
            auto best_hi = int8_t{-8};
            for (auto i = begin; i < end; ++i) {
                const auto lo = i4_lane_lo(data[i]);
                const auto hi = i4_lane_hi(data[i]);
                if (lo > best_lo) {
                    best_lo = lo;
                }
                if (hi > best_hi) {
                    best_hi = hi;
                }
            }
            return {best_lo, best_hi};
        }
    }

    /**
     * Dispatches on dtype; per-chunk partials are merged in chunk order on the caller thread.
     * Comparison reduction: the combine is associative and commutative, so every chunking
     * agrees bit-for-bit, and the non-NaN result is always one of the input elements.
     * @brief CPU full reduction, largest element written into the scalar output tensor.
     * @param t Input tensor.
     * @param out Single element output tensor, same dtype as t.
     */
    void max_cpu(const Tensor& t, const Tensor& out) {
        if (t.device != Device::CPU || out.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(t.device));
        }
        using DType = DataTypes::DType;
        const auto numel = t.numel;
        switch (t.dtype) {
            case DType::Float32: {
                const auto* data = static_cast<const float32_t*>(t.data) + t.offset;
                std::vector partials(IncludedCPUFeatures().threads, -std::numeric_limits<float32_t>::infinity());
                drain(launch_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = max_range_scalar<float32_t>(data, b, e); }, t, numel * sizeof(float32_t)));
                auto best = -std::numeric_limits<float32_t>::infinity();
                for (const auto p : partials) {
                    best = fp_max_combine(best, p);
                }
                *static_cast<float32_t*>(out.data) = best;
                break;
            }
            case DType::Float64: {
                const auto* data = static_cast<const float64_t*>(t.data) + t.offset;
                std::vector partials(IncludedCPUFeatures().threads, -std::numeric_limits<float64_t>::infinity());
                drain(launch_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = max_range_scalar<float64_t>(data, b, e); }, t, numel * sizeof(float64_t)));
                auto best = -std::numeric_limits<float64_t>::infinity();
                for (const auto p : partials) {
                    best = fp_max_combine(best, p);
                }
                *static_cast<float64_t*>(out.data) = best;
                break;
            }
            case DType::Float16: {
                const auto* data = static_cast<const float16_t*>(t.data) + t.offset;
                std::vector partials(IncludedCPUFeatures().threads, half_identity_bits<float16_t>());
                drain(launch_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = max_range_half<float16_t>(data, b, e); }, t, numel * sizeof(float16_t)));
                auto best_bits = half_identity_bits<float16_t>();
                for (const auto p : partials) {
                    best_bits = half_max_combine<float16_t>(best_bits, p);
                }
                *static_cast<float16_t*>(out.data) = Intrinsics::detail::bit_cast_memcpy<float16_t>(best_bits);
                break;
            }
            case DType::BFloat16: {
                const auto* data = static_cast<const bfloat16_t*>(t.data) + t.offset;
                std::vector partials(IncludedCPUFeatures().threads, half_identity_bits<bfloat16_t>());
                drain(launch_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = max_range_half<bfloat16_t>(data, b, e); }, t, numel * sizeof(bfloat16_t)));
                auto best_bits = half_identity_bits<bfloat16_t>();
                for (const auto p : partials) {
                    best_bits = half_max_combine<bfloat16_t>(best_bits, p);
                }
                *static_cast<bfloat16_t*>(out.data) = Intrinsics::detail::bit_cast_memcpy<bfloat16_t>(best_bits);
                break;
            }
            case DType::Float16x2: {
                const auto* data = static_cast<const float16x2_t*>(t.data) + t.offset;
                std::vector<std::pair<uint16_t, uint16_t>> partials(IncludedCPUFeatures().threads, {half_identity_bits<float16_t>(), half_identity_bits<float16_t>()});
                drain(launch_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = max_range_packed<float16x2_t>(data, b, e); }, t, numel * sizeof(float16x2_t)));
                auto best_lo = half_identity_bits<float16_t>();
                auto best_hi = half_identity_bits<float16_t>();
                for (const auto& [p_lo, p_hi] : partials) {
                    best_lo = half_max_combine<float16_t>(best_lo, p_lo);
                    best_hi = half_max_combine<float16_t>(best_hi, p_hi);
                }
                *static_cast<float16x2_t*>(out.data) = pack_lanes<float16x2_t>(best_lo, best_hi);
                break;
            }
            case DType::BFloat16x2: {
                const auto* data = static_cast<const bfloat16x2_t*>(t.data) + t.offset;
                std::vector<std::pair<uint16_t, uint16_t>> partials(IncludedCPUFeatures().threads, {half_identity_bits<bfloat16_t>(), half_identity_bits<bfloat16_t>()});
                drain(launch_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = max_range_packed<bfloat16x2_t>(data, b, e); }, t, numel * sizeof(bfloat16x2_t)));
                auto best_lo = half_identity_bits<bfloat16_t>();
                auto best_hi = half_identity_bits<bfloat16_t>();
                for (const auto& [p_lo, p_hi] : partials) {
                    best_lo = half_max_combine<bfloat16_t>(best_lo, p_lo);
                    best_hi = half_max_combine<bfloat16_t>(best_hi, p_hi);
                }
                *static_cast<bfloat16x2_t*>(out.data) = pack_lanes<bfloat16x2_t>(best_lo, best_hi);
                break;
            }
            case DType::Int8: case DType::UInt8: case DType::Int16: case DType::UInt16:
            case DType::Int32: case DType::UInt32: case DType::Int64: case DType::UInt64: {
                /**
                 * Runs max_range_int<T> over every chunk and drains the futures.
                 * @brief Threaded integer max for one concrete element type.
                 * @param data Base pointer, offset already applied.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                auto launch_int = [&]<typename T>(const T* data) {
                    std::vector partials(IncludedCPUFeatures().threads, std::numeric_limits<T>::lowest());
                    drain(launch_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = max_range_int<T>(data, b, e); }, t, numel * sizeof(T)));
                    auto best = std::numeric_limits<T>::lowest();
                    for (const auto p : partials) {
                        if (p > best) {
                            best = p;
                        }
                    }
                    *static_cast<T*>(out.data) = best;
                };
                switch (t.dtype) {
                    case DType::Int8: launch_int(static_cast<const int8_t*>(t.data) + t.offset); break;
                    case DType::UInt8: launch_int(static_cast<const uint8_t*>(t.data) + t.offset); break;
                    case DType::Int16: launch_int(static_cast<const int16_t*>(t.data) + t.offset); break;
                    case DType::UInt16: launch_int(static_cast<const uint16_t*>(t.data) + t.offset); break;
                    case DType::Int32: launch_int(static_cast<const int32_t*>(t.data) + t.offset); break;
                    case DType::UInt32: launch_int(static_cast<const uint32_t*>(t.data) + t.offset); break;
                    case DType::Int64: launch_int(static_cast<const int64_t*>(t.data) + t.offset); break;
                    case DType::UInt64: launch_int(static_cast<const uint64_t*>(t.data) + t.offset); break;
                    [[unlikely]] default: throw ErrorHandling::UnReachableCode();
                }
                break;
            }
            case DType::Int4x2: {
                const auto* data = static_cast<const int4x2_t*>(t.data) + t.offset;
                std::vector<std::pair<int8_t, int8_t>> partials(IncludedCPUFeatures().threads, {int8_t{-8}, int8_t{-8}});
                drain(launch_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = max_range_i4x2(data, b, e); }, t, numel * sizeof(int4x2_t)));
                auto best_lo = int8_t{-8};
                auto best_hi = int8_t{-8};
                for (const auto& [p_lo, p_hi] : partials) {
                    if (p_lo > best_lo) {
                        best_lo = p_lo;
                    }
                    if (p_hi > best_hi) {
                        best_hi = p_hi;
                    }
                }
                *static_cast<int4x2_t*>(out.data) = i4_pack(best_lo, best_hi);
                break;
            }
            [[unlikely]] default: {
                throw ErrorHandling::NotImplemented();
            }
        }
    }
}

namespace BeanTensor::Tensors {

    /**
     * Syncs, rejects empty/non-contiguous inputs (same contract as mean), then routes to
     * the CPU or GPU backend. No autograd node is wired: the output never joins the
     * computation graph (same stance as argmax).
     * @brief Max of every element.
     * @return Single element tensor of the same dtype and device.
     * @throws ErrorHandling::IndexOutOfRange Tensor is empty.
     * @throws ErrorHandling::NotImplemented Tensor is non-contiguous.
     */
    Tensor Tensor::max() const {
        sync();
        if (this->numel == 0) {
            throw ErrorHandling::IndexOutOfRange();
        }
        if (!this->is_contiguous) {
            throw ErrorHandling::NotImplemented();
        }
        auto out = Tensor(std::vector<size_t>{1}, this->dtype, this->device, false);
        if (this->device == Device::CPU) {
            detail::max_cpu(*this, out);
        } else if (this->device == Device::GPU) {
            detail::max_gpu(*this, out);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        return out;
    }
}
