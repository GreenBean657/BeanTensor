/**
 * @file MeanCPU.cpp
 * @author GreenBean
 * @brief CPU backend for mean. Hosts the Tensor::mean method definition
 * (mirrors how Tensor::sum lives in SumCPU.cpp).
 * @version 0.1
 * @date 2026-10-22
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <future>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "Math/Reductions/Mean/Mean.h"
#include "Math/Reductions/ReductionsShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Tensors::detail {
    using namespace reductions_shared;

    namespace {
        /**
         * Sequential order; float means are not guaranteed bit-identical across dispatch tiers
         * or backends (the accumulation order is backend-defined, same as sum).
         * @brief Scalar range sum for the mean (fp32/fp64).
         * @param data Input elements.
         * @param begin First element summed.
         * @param end One past the last element summed.
         * @return Sum of the elements in [begin, end).
         */
        template<typename T>
        [[nodiscard]] T mean_range_scalar(const T* data, const size_t begin, const size_t end) {
            auto total = T{0};
            for (auto i = begin; i < end; ++i) {
                total += data[i];
            }
            return total;
        }

        /**
         * @brief Scalar range sum for the mean (fp16/bf16, fp32 accumulate).
         * @param data Input elements.
         * @param begin First element summed.
         * @param end One past the last element summed.
         * @return FP32 sum of the elements in [begin, end).
         */
        template<typename H>
        [[nodiscard]] float32_t mean_range_half(const H* data, const size_t begin, const size_t end) {
            auto total = 0.0f;
            for (auto i = begin; i < end; ++i) {
                total += widen_lane(data[i]);
            }
            return total;
        }

        /**
         * Lanes accumulate independently in fp32.
         * @brief Scalar range sum for the mean (fp16x2/bf16x2, per-lane fp32 accumulate).
         * @param data Input elements.
         * @param begin First element summed.
         * @param end One past the last element summed.
         * @return {sum_lo, sum_hi}.
         */
        template<typename P>
        [[nodiscard]] std::pair<float32_t, float32_t> mean_range_packed(const P* data, const size_t begin, const size_t end) {
            using H = packed_lane_t<P>;
            auto total_lo = 0.0f;
            auto total_hi = 0.0f;
            for (auto i = begin; i < end; ++i) {
                const auto bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(data[i]);
                total_lo += widen_lane(lane_lo<H>(bits));
                total_hi += widen_lane(lane_hi<H>(bits));
            }
            return {total_lo, total_hi};
        }

        /**
         * Unlike sum()'s modular uint64 contract, the mean accumulates in fp64: a wrapped sum
         * divided by numel would be meaningless. Values past 2^53 in magnitude can lose low
         * bits (documented); narrowing truncates toward zero and clamps to the dtype range.
         * @brief Scalar range sum for the mean (all integer dtypes, fp64 accumulate).
         * @param data Input elements.
         * @param begin First element summed.
         * @param end One past the last element summed.
         * @return FP64 sum of the elements in [begin, end).
         */
        template<typename T>
        [[nodiscard]] float64_t mean_range_int(const T* data, const size_t begin, const size_t end) {
            auto total = float64_t{0.0};
            for (auto i = begin; i < end; ++i) {
                total += static_cast<float64_t>(data[i]);
            }
            return total;
        }

        /**
         * Signed nibble values accumulate per lane in fp32 (every sum with |value| < 2^24 is exact).
         * @brief Scalar range sum for the mean (int4x2, per-lane fp32 accumulate).
         * @param data Input elements.
         * @param begin First element summed.
         * @param end One past the last element summed.
         * @return {sum_lo, sum_hi} of the signed lane values.
         */
        [[nodiscard]] inline std::pair<float32_t, float32_t> mean_range_i4x2(const int4x2_t* data, const size_t begin, const size_t end) {
            auto total_lo = 0.0f;
            auto total_hi = 0.0f;
            for (auto i = begin; i < end; ++i) {
                total_lo += static_cast<float32_t>(i4_lane_lo(data[i]));
                total_hi += static_cast<float32_t>(i4_lane_hi(data[i]));
            }
            return {total_lo, total_hi};
        }

        /**
         * Clamp to the dtype range, then truncate toward zero — the same narrowing rule as the
         * GradDType float->int boundary cast. A NaN mean stores 0 (there is no integer NaN).
         * The uint64/int64 maxima round UP to 2^64/2^63 in fp64, so the >= guard must fire
         * before the static_cast (which would be UB at exactly 2^64/2^63).
         * @brief Narrow an fp64 mean to an integer dtype.
         * @param v FP64 mean.
         * @return The mean narrowed into T's range.
         */
        template<typename T>
        [[nodiscard]] T narrow_mean_int(const float64_t v) {
            constexpr auto lo = static_cast<float64_t>(std::numeric_limits<T>::lowest());
            constexpr auto hi = static_cast<float64_t>(std::numeric_limits<T>::max());
            if (std::isnan(v)) {
                return T{0};
            }
            if (v <= lo) {
                return std::numeric_limits<T>::lowest();
            }
            if (v >= hi) {
                return std::numeric_limits<T>::max();
            }
            return static_cast<T>(v);
        }
    }

    /**
     * Dispatches on dtype; per-chunk partial sums are merged in chunk order, then divided by
     * numel once and narrowed to the output dtype.
     * @brief CPU full reduction, averages every element into the scalar output tensor.
     * @param t Input tensor.
     * @param out Single element output tensor, same dtype as t.
     */
    void mean_cpu(const Tensor& t, Tensor& out) {
        if (t.device != Device::CPU || out.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(t.device));
        }
        using DType = DataTypes::DType;
        const auto numel = t.numel;
        switch (t.dtype) {
            case DType::Float32: {
                const auto* data = static_cast<const float32_t*>(t.data) + t.offset;
                std::vector<float32_t> partials(IncludedCPUFeatures().threads, 0.0f);
                drain(launch_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = mean_range_scalar<float32_t>(data, b, e); }, t));
                auto total = 0.0f;
                for (const auto p : partials) {
                    total += p;
                }
                *static_cast<float32_t*>(out.data) = total / static_cast<float32_t>(numel);
                break;
            }
            case DType::Float64: {
                const auto* data = static_cast<const float64_t*>(t.data) + t.offset;
                std::vector<float64_t> partials(IncludedCPUFeatures().threads, 0.0);
                drain(launch_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = mean_range_scalar<float64_t>(data, b, e); }, t));
                auto total = 0.0;
                for (const auto p : partials) {
                    total += p;
                }
                *static_cast<float64_t*>(out.data) = total / static_cast<float64_t>(numel);
                break;
            }
            case DType::Float16: {
                const auto* data = static_cast<const float16_t*>(t.data) + t.offset;
                std::vector<float32_t> partials(IncludedCPUFeatures().threads, 0.0f);
                drain(launch_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = mean_range_half<float16_t>(data, b, e); }, t));
                auto total = 0.0f;
                for (const auto p : partials) {
                    total += p;
                }
                *static_cast<float16_t*>(out.data) = narrow_lane<float16_t>(total / static_cast<float32_t>(numel));
                break;
            }
            case DType::BFloat16: {
                const auto* data = static_cast<const bfloat16_t*>(t.data) + t.offset;
                std::vector<float32_t> partials(IncludedCPUFeatures().threads, 0.0f);
                drain(launch_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = mean_range_half<bfloat16_t>(data, b, e); }, t));
                auto total = 0.0f;
                for (const auto p : partials) {
                    total += p;
                }
                *static_cast<bfloat16_t*>(out.data) = narrow_lane<bfloat16_t>(total / static_cast<float32_t>(numel));
                break;
            }
            case DType::Float16x2: {
                const auto* data = static_cast<const float16x2_t*>(t.data) + t.offset;
                std::vector<std::pair<float32_t, float32_t>> partials(IncludedCPUFeatures().threads, {0.0f, 0.0f});
                drain(launch_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = mean_range_packed<float16x2_t>(data, b, e); }, t));
                auto total_lo = 0.0f;
                auto total_hi = 0.0f;
                for (const auto& p : partials) {
                    total_lo += p.first;
                    total_hi += p.second;
                }
                const auto n = static_cast<float32_t>(numel);
                *static_cast<float16x2_t*>(out.data) = pack_lanes<float16x2_t>(
                    Intrinsics::detail::bit_cast_memcpy<uint16_t>(narrow_lane<float16_t>(total_lo / n)),
                    Intrinsics::detail::bit_cast_memcpy<uint16_t>(narrow_lane<float16_t>(total_hi / n)));
                break;
            }
            case DType::BFloat16x2: {
                const auto* data = static_cast<const bfloat16x2_t*>(t.data) + t.offset;
                std::vector<std::pair<float32_t, float32_t>> partials(IncludedCPUFeatures().threads, {0.0f, 0.0f});
                drain(launch_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = mean_range_packed<bfloat16x2_t>(data, b, e); }, t));
                auto total_lo = 0.0f;
                auto total_hi = 0.0f;
                for (const auto& p : partials) {
                    total_lo += p.first;
                    total_hi += p.second;
                }
                const auto n = static_cast<float32_t>(numel);
                *static_cast<bfloat16x2_t*>(out.data) = pack_lanes<bfloat16x2_t>(
                    Intrinsics::detail::bit_cast_memcpy<uint16_t>(narrow_lane<bfloat16_t>(total_lo / n)),
                    Intrinsics::detail::bit_cast_memcpy<uint16_t>(narrow_lane<bfloat16_t>(total_hi / n)));
                break;
            }
            case DType::Int8: case DType::UInt8: case DType::Int16: case DType::UInt16:
            case DType::Int32: case DType::UInt32: case DType::Int64: case DType::UInt64: {
                std::vector<float64_t> partials(IncludedCPUFeatures().threads, 0.0);
                /**
                 * Runs mean_range_int<T> over every chunk and drains the futures.
                 * @brief Threaded integer mean for one concrete element type.
                 * @param data Base pointer, offset already applied.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                auto launch_int = [&]<typename T>(const T* data) {
                    drain(launch_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = mean_range_int<T>(data, b, e); }, t));
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
                auto total = float64_t{0.0};
                for (const auto p : partials) {
                    total += p;
                }
                const auto mean = total / static_cast<float64_t>(numel);
                switch (t.dtype) {
                    case DType::Int8: *static_cast<int8_t*>(out.data) = narrow_mean_int<int8_t>(mean); break;
                    case DType::UInt8: *static_cast<uint8_t*>(out.data) = narrow_mean_int<uint8_t>(mean); break;
                    case DType::Int16: *static_cast<int16_t*>(out.data) = narrow_mean_int<int16_t>(mean); break;
                    case DType::UInt16: *static_cast<uint16_t*>(out.data) = narrow_mean_int<uint16_t>(mean); break;
                    case DType::Int32: *static_cast<int32_t*>(out.data) = narrow_mean_int<int32_t>(mean); break;
                    case DType::UInt32: *static_cast<uint32_t*>(out.data) = narrow_mean_int<uint32_t>(mean); break;
                    case DType::Int64: *static_cast<int64_t*>(out.data) = narrow_mean_int<int64_t>(mean); break;
                    case DType::UInt64: *static_cast<uint64_t*>(out.data) = narrow_mean_int<uint64_t>(mean); break;
                    [[unlikely]] default: throw ErrorHandling::UnReachableCode();
                }
                break;
            }
            case DType::Int4x2: {
                const auto* data = static_cast<const int4x2_t*>(t.data) + t.offset;
                std::vector<std::pair<float32_t, float32_t>> partials(IncludedCPUFeatures().threads, {0.0f, 0.0f});
                drain(launch_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = mean_range_i4x2(data, b, e); }, t));
                auto total_lo = 0.0f;
                auto total_hi = 0.0f;
                for (const auto& p : partials) {
                    total_lo += p.first;
                    total_hi += p.second;
                }
                const auto n = static_cast<float32_t>(numel);
                // Truncate toward zero, clamped to [-8, 7] (the signed nibble range).
                const auto mean_lo = static_cast<int8_t>(std::clamp(total_lo / n, -8.0f, 7.0f));
                const auto mean_hi = static_cast<int8_t>(std::clamp(total_hi / n, -8.0f, 7.0f));
                *static_cast<int4x2_t*>(out.data) = i4_pack(mean_lo, mean_hi);
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
     * @brief Mean of every element.
     * @return Single element tensor of the same dtype and device.
     */
    Tensor Tensor::mean() const {
        sync();
        if (this->numel == 0) {
            throw ErrorHandling::IndexOutOfRange();
        }
        if (!this->is_contiguous) {
            throw ErrorHandling::NotImplemented();
        }
        auto out = Tensor(std::vector<size_t>{1}, this->dtype, this->device, false);
        if (this->device == Device::CPU) {
            detail::mean_cpu(*this, out);
        } else if (this->device == Device::GPU) {
            detail::mean_gpu(*this, out);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        return out;
    }
}
