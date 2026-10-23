/**
 * @file IndexingShared.h
 * @author GreenBean
 * @brief Shared helpers for the indexing ops (index_select, gather, cat):
 * integer-index dtype gate, bounds-checked index reads, and shape-span products.
 * Header-only: everything here is a template or inline.
 * @version 0.1
 * @date 2026-10-22
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "Helpers/Errors.h"
#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors::detail::indexing_shared {

    /**
     * The plain integer dtypes (int4x2 excluded: nibble lanes are not valid indices here).
     * @brief Is dt an integer dtype usable as an index tensor?
     * @param dt DType to check.
     * @return True for Int8..Int64 and UInt8..UInt64.
     */
    [[nodiscard]] inline bool is_integer_index_dtype(const DataTypes::DType& dt) {
        using DType = DataTypes::DType;
        switch (dt) {
            case DType::Int8: case DType::UInt8: case DType::Int16: case DType::UInt16:
            case DType::Int32: case DType::UInt32: case DType::Int64: case DType::UInt64: {
                return true;
            }
            default: {
                return false;
            }
        }
    }

    /**
     * Signed indices reject negatives; every index must lie below the axis extent.
     * @brief Read and bounds-check one index-tensor element.
     * @tparam T Index element type (a plain integer dtype).
     * @param data Index elements, view offset already applied.
     * @param flat Flat element position to read.
     * @param extent Axis extent the index selects from.
     * @return The index as a size_t in [0, extent).
     * @throws ErrorHandling::IndexOutOfRange Negative or >= extent.
     */
    template<typename T> requires std::is_integral_v<T>
    [[nodiscard]] size_t checked_axis_index(const T* data, const size_t flat, const size_t extent) {
        const auto v = data[flat];
        if constexpr (std::is_signed_v<T>) {
            if (v < 0) {
                throw ErrorHandling::IndexOutOfRange();
            }
        }
        // v >= 0 on the signed path (negatives threw above), so the unsigned
        // twin preserves the value on the way to the uint64 comparison.
        const auto u = static_cast<uint64_t>(static_cast<std::make_unsigned_t<T>>(v));
        if (u >= extent) {
            throw ErrorHandling::IndexOutOfRange();
        }
        return u;
    }

    /**
     * @brief Product of shape[begin, end) (1 for an empty span).
     * @param shape Shape array.
     * @param begin First dimension multiplied.
     * @param end One past the last dimension multiplied.
     * @return Product of the extents in [begin, end).
     */
    [[nodiscard]] inline size_t span_product(const size_t* shape, const size_t begin, const size_t end) {
        auto product = size_t{1};
        for (auto d = begin; d < end; ++d) {
            product *= shape[d];
        }
        return product;
    }
}
