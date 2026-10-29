/**
 * @file SoftmaxCPU.cpp
 * @author GreenBean
 * @brief CPU backend for standalone rank-2 softmax, plus the Tensor::softmax
 * method definition.
 * @version 0.1
 * @date 2026-10-28
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "Math/Activations/ActivationsShared.h"
#include "Math/Activations/Softmax/Softmax.h"
#include "Math/Activations/Softmax/SoftmaxKernels.h"
#include "Math/Activations/Softmax/SoftmaxShared.h"
#include "Math/Autograd/SoftmaxG/SoftmaxG.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"
#include "Helpers/Errors.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Tensors::detail {
    using namespace softmax_shared;

    namespace {
        /**
         * @brief Row-range worker for softmax_cpu over the row range [b, e).
         * Dispatches on the logits dtype and calls softmax_row / softmax_row_packed.
         * @param b First row index of the range (inclusive).
         * @param e One past the last row index of the range.
         */
        template<typename T, typename Acc>
        void softmax_rows(const T* logits, T* out, const size_t cols_store, const size_t cols_eff, const size_t b, const size_t e) {
            for (auto r = b; r < e; ++r) {
                if constexpr (std::is_same_v<T, float16x2_t> || std::is_same_v<T, bfloat16x2_t> || std::is_same_v<T, int4x2_t>) {
                    softmax_row_packed<T, Acc>(logits + r * cols_store, out + r * cols_store, cols_eff);
                } else {
                    softmax_row<T, Acc>(logits + r * cols_store, out + r * cols_store, cols_eff);
                }
            }
        }
    }

    /**
     * Last-axis softmax over a rank-2 tensor. Numerically stable per-row:
     * max-subtract, exp, sum, normalize. Supports all dtypes; fp32/fp16/bf16/
     * packed accumulate in fp32, fp64 and integer kinds in fp64. Integer outputs
     * clamp/truncate and map NaN to 0. The work is chunked across the CPU pool
     * by rows; futures are drained before the function returns.
     * @brief CPU standalone rank-2 softmax.
     * @param t Rank-2 logits tensor (rows x classes), CPU resident, contiguous.
     * @param out Output tensor, same shape/dtype/device as t (written).
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if any tensor is not on the CPU.
     * @throws ErrorHandling::NotImplemented for an unsupported dtype.
     */
    void softmax_cpu(const Tensor& t, Tensor& out) {
        if (t.device != Device::CPU || out.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(t.device));
        }
        using DType = DataTypes::DType;
        const auto rows = t.shape[0];
        const auto cols_store = t.shape[1];
        const auto packed = t.dtype == DType::Float16x2 || t.dtype == DType::BFloat16x2 || t.dtype == DType::Int4x2;
        const auto cols_eff = cols_store * (packed ? size_t{2} : size_t{1});
        switch (t.dtype) {
            case DType::Float32: {
                const auto* logits = static_cast<const float32_t*>(t.data) + t.offset;
                auto* out_ptr = static_cast<float32_t*>(out.data) + out.offset;
                const auto kernel = Math::Activations::softmax_avx512_available()
                    ? Math::Activations::softmax_f32_range_avx512
                    : Math::Activations::softmax_avx2_available() ? Math::Activations::softmax_f32_range_avx2 : nullptr;
                if (kernel != nullptr) {
                    drain(launch_rows(rows, [&](const size_t b, const size_t e) { kernel(out_ptr, logits, rows, cols_eff, b, e); }, rows * cols_store * sizeof(float32_t), t, out));
                } else {
                    drain(launch_rows(rows, [&](const size_t b, const size_t e) { softmax_rows<float32_t, float32_t>(logits, out_ptr, cols_store, cols_eff, b, e); }, rows * cols_store * sizeof(float32_t), t, out));
                }
                break;
            }
            case DType::Float64: {
                const auto* logits = static_cast<const float64_t*>(t.data) + t.offset;
                auto* out_ptr = static_cast<float64_t*>(out.data) + out.offset;
                const auto kernel = Math::Activations::softmax_avx512_available()
                    ? Math::Activations::softmax_f64_range_avx512
                    : Math::Activations::softmax_avx2_available() ? Math::Activations::softmax_f64_range_avx2 : nullptr;
                if (kernel != nullptr) {
                    drain(launch_rows(rows, [&](const size_t b, const size_t e) { kernel(out_ptr, logits, rows, cols_eff, b, e); }, rows * cols_store * sizeof(float64_t), t, out));
                } else {
                    drain(launch_rows(rows, [&](const size_t b, const size_t e) { softmax_rows<float64_t, float64_t>(logits, out_ptr, cols_store, cols_eff, b, e); }, rows * cols_store * sizeof(float64_t), t, out));
                }
                break;
            }
            case DType::Float16: {
                const auto* logits = static_cast<const float16_t*>(t.data) + t.offset;
                auto* out_ptr = static_cast<float16_t*>(out.data) + out.offset;
                drain(launch_rows(rows, [&](const size_t b, const size_t e) { softmax_rows<float16_t, float32_t>(logits, out_ptr, cols_store, cols_eff, b, e); }, rows * cols_store * sizeof(float16_t), t, out));
                break;
            }
            case DType::BFloat16: {
                const auto* logits = static_cast<const bfloat16_t*>(t.data) + t.offset;
                auto* out_ptr = static_cast<bfloat16_t*>(out.data) + out.offset;
                drain(launch_rows(rows, [&](const size_t b, const size_t e) { softmax_rows<bfloat16_t, float32_t>(logits, out_ptr, cols_store, cols_eff, b, e); }, rows * cols_store * sizeof(bfloat16_t), t, out));
                break;
            }
            case DType::Float16x2: {
                const auto* logits = static_cast<const float16x2_t*>(t.data) + t.offset;
                auto* out_ptr = static_cast<float16x2_t*>(out.data) + out.offset;
                drain(launch_rows(rows, [&](const size_t b, const size_t e) { softmax_rows<float16x2_t, float32_t>(logits, out_ptr, cols_store, cols_eff, b, e); }, rows * cols_store * sizeof(float16x2_t), t, out));
                break;
            }
            case DType::BFloat16x2: {
                const auto* logits = static_cast<const bfloat16x2_t*>(t.data) + t.offset;
                auto* out_ptr = static_cast<bfloat16x2_t*>(out.data) + out.offset;
                drain(launch_rows(rows, [&](const size_t b, const size_t e) { softmax_rows<bfloat16x2_t, float32_t>(logits, out_ptr, cols_store, cols_eff, b, e); }, rows * cols_store * sizeof(bfloat16x2_t), t, out));
                break;
            }
            case DType::Int8: case DType::UInt8: case DType::Int16: case DType::UInt16:
            case DType::Int32: case DType::UInt32: case DType::Int64: case DType::UInt64:
            case DType::Int4x2: {
                auto launch_int = [&]<typename T>(const T* logits, T* out_ptr) {
                    drain(launch_rows(rows, [&](const size_t b, const size_t e) { softmax_rows<T, float64_t>(logits, out_ptr, cols_store, cols_eff, b, e); }, rows * cols_store * sizeof(T), t, out));
                };
                switch (t.dtype) {
                    case DType::Int8: launch_int(static_cast<const int8_t*>(t.data) + t.offset, static_cast<int8_t*>(out.data) + out.offset); break;
                    case DType::UInt8: launch_int(static_cast<const uint8_t*>(t.data) + t.offset, static_cast<uint8_t*>(out.data) + out.offset); break;
                    case DType::Int16: launch_int(static_cast<const int16_t*>(t.data) + t.offset, static_cast<int16_t*>(out.data) + out.offset); break;
                    case DType::UInt16: launch_int(static_cast<const uint16_t*>(t.data) + t.offset, static_cast<uint16_t*>(out.data) + out.offset); break;
                    case DType::Int32: launch_int(static_cast<const int32_t*>(t.data) + t.offset, static_cast<int32_t*>(out.data) + out.offset); break;
                    case DType::UInt32: launch_int(static_cast<const uint32_t*>(t.data) + t.offset, static_cast<uint32_t*>(out.data) + out.offset); break;
                    case DType::Int64: launch_int(static_cast<const int64_t*>(t.data) + t.offset, static_cast<int64_t*>(out.data) + out.offset); break;
                    case DType::UInt64: launch_int(static_cast<const uint64_t*>(t.data) + t.offset, static_cast<uint64_t*>(out.data) + out.offset); break;
                    case DType::Int4x2: launch_int(static_cast<const int4x2_t*>(t.data) + t.offset, static_cast<int4x2_t*>(out.data) + out.offset); break;
                    [[unlikely]] default: throw ErrorHandling::UnReachableCode();
                }
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
     * Numerically-stable last-axis softmax over the class dimension: this must be
     * rank-2 (rows x classes). Returns a tensor of the same shape/dtype/device
     * where each row sums to 1. Computed as exp(x - row_max) / sum(exp(x -
     * row_max)); a NaN logit poisons its whole row. Packed logits dtypes
     * (Float16x2/BFloat16x2) and Int4x2 unpack each element's two lanes as
     * consecutive classes (effective class count is 2 x shape[1]). Accumulation:
     * fp32 for fp32/fp16/bf16/packed logits, fp64 for fp64 and integer logits.
     * Integer outputs clamp/truncate and map NaN to 0. Blocks before returning.
     * When the input has requires_grad=true the output joins the computation
     * graph (softmax_G backward: dx = y * (dy - rowsum(dy*y)), with y recomputed
     * from the saved input).
     * @brief Standalone rank-2 softmax.
     * @return New tensor (rows x classes) holding softmax probabilities.
     * @throws ErrorHandling::MismatchedShapes Input is not rank-2 or is empty.
     * @throws ErrorHandling::NotImplemented Non-contiguous operand.
     */
    Tensor Tensor::softmax() const {
        if (this->ndim != 2) {
            throw ErrorHandling::MismatchedShapes("softmax requires rank-2 input (rows x classes)");
        }
        if (this->shape[0] == 0 || this->shape[1] == 0) {
            throw ErrorHandling::MismatchedShapes("softmax requires non-empty input");
        }
        if (!this->is_contiguous) {
            throw ErrorHandling::NotImplemented();
        }
        sync();
        auto out = Tensor(std::vector(this->shape, this->shape + this->ndim), this->dtype, this->device, false);
        if (this->device == Device::CPU) {
            detail::softmax_cpu(*this, out);
        } else if (this->device == Device::GPU) {
            detail::softmax_gpu(*this, out);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        if (this->requires_grad) {
            out.wire_node(detail::softmax_g_bridge, *this);
        }
        return out;
    }
}
