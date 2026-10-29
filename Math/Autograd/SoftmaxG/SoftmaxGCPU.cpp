/**
 * @file SoftmaxGCPU.cpp
 * @author GreenBean
 * @brief CPU backend for standalone softmax backward, plus the softmax backward
 * trampoline (softmax_g_node).
 * @version 0.1
 * @date 2026-10-28
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <future>
#include <limits>
#include <type_traits>
#include <vector>

#include "Math/Activations/Softmax/SoftmaxShared.h"
#include "Math/Autograd/AutogradShared.h"
#include "Math/Autograd/GradDType.h"
#include "Math/Autograd/SoftmaxG/SoftmaxG.h"
#include "Math/Elementwise/ElementwiseShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Tensors::detail {
    using namespace softmax_shared;
    using namespace elementwise_shared;

    namespace {
        /**
         * @brief Row softmax backward: dx_j = y_j * (dy_j - sum_c dy_c * y_c),
         * with y recomputed from the forward input row. Non-packed dtypes.
         * @param x_row Forward input row.
         * @param dy_row Upstream gradient row.
         * @param dx_row Output gradient row (written).
         * @param cols Effective class count per row.
         */
        template<typename T, typename Acc>
        void softmax_g_row(const T* x_row, const T* dy_row, T* dx_row, const size_t cols) {
            auto widen_x = [&](const size_t j) -> Acc {
                if constexpr (std::is_same_v<Acc, float32_t>) {
                    return widen_class_f32(x_row, j);
                } else {
                    return widen_class_f64(x_row, j);
                }
            };
            auto widen_dy = [&](const size_t j) -> Acc {
                if constexpr (std::is_same_v<Acc, float32_t>) {
                    return widen_class_f32(dy_row, j);
                } else {
                    return widen_class_f64(dy_row, j);
                }
            };
            auto m = std::numeric_limits<Acc>::lowest();
            for (auto j = size_t{0}; j < cols; ++j) {
                if (
                    const auto x = widen_x(j);
                    x > m
                ) {
                    m = x;
                }
            }
            auto s = Acc{0};
            for (auto j = size_t{0}; j < cols; ++j) {
                s += std::exp(widen_x(j) - m);
            }
            std::vector<Acc> y(cols);
            for (auto j = size_t{0}; j < cols; ++j) {
                y[j] = std::exp(widen_x(j) - m) / s;
            }
            auto row_sum = Acc{0};
            for (auto j = size_t{0}; j < cols; ++j) {
                row_sum += widen_dy(j) * y[j];
            }
            for (auto j = size_t{0}; j < cols; ++j) {
                dx_row[j] = narrow_prob<T, Acc>(y[j] * (widen_dy(j) - row_sum));
            }
        }

        /**
         * @brief Row softmax backward for packed x2 / int4x2 rows.
         * @param x_row Forward input row of packed storage elements.
         * @param dy_row Upstream gradient row of packed storage elements.
         * @param dx_row Output gradient row (written).
         * @param cols_eff Effective class count (two per packed element).
         */
        template<typename T, typename Acc>
        void softmax_g_row_packed(const T* x_row, const T* dy_row, T* dx_row, const size_t cols_eff) {
            auto widen_x = [&](const size_t j) -> Acc {
                if constexpr (std::is_same_v<Acc, float32_t>) {
                    return widen_class_f32(x_row, j);
                } else {
                    return widen_class_f64(x_row, j);
                }
            };
            auto widen_dy = [&](const size_t j) -> Acc {
                if constexpr (std::is_same_v<Acc, float32_t>) {
                    return widen_class_f32(dy_row, j);
                } else {
                    return widen_class_f64(dy_row, j);
                }
            };
            auto m = std::numeric_limits<Acc>::lowest();
            for (auto j = size_t{0}; j < cols_eff; ++j) {
                if (
                    const auto x = widen_x(j);
                    x > m
                ) {
                    m = x;
                }
            }
            auto s = Acc{0};
            for (auto j = size_t{0}; j < cols_eff; ++j) {
                s += std::exp(widen_x(j) - m);
            }
            std::vector<Acc> y(cols_eff);
            for (auto j = size_t{0}; j < cols_eff; ++j) {
                y[j] = std::exp(widen_x(j) - m) / s;
            }
            auto row_sum = Acc{0};
            for (auto j = size_t{0}; j < cols_eff; ++j) {
                row_sum += widen_dy(j) * y[j];
            }
            for (auto e = size_t{0}; e < cols_eff / 2; ++e) {
                const auto j = 2 * e;
                const auto glo = y[j] * (widen_dy(j) - row_sum);
                const auto ghi = y[j + 1] * (widen_dy(j + 1) - row_sum);
                if constexpr (std::is_same_v<T, int4x2_t>) {
                    dx_row[e] = pack_int4(
                        std::clamp(static_cast<int32_t>(narrow_prob<int8_t, Acc>(glo)), -8, 7),
                        std::clamp(static_cast<int32_t>(narrow_prob<int8_t, Acc>(ghi)), -8, 7));
                } else {
                    using H = packed_lane_t<T>;
                    const auto lo = narrow_lane<H>(static_cast<float32_t>(glo));
                    const auto hi = narrow_lane<H>(static_cast<float32_t>(ghi));
                    dx_row[e] = pack_lanes<T>(Intrinsics::detail::bit_cast_memcpy<uint16_t>(lo),
                                              Intrinsics::detail::bit_cast_memcpy<uint16_t>(hi));
                }
            }
        }
    }

    /**
     * dx = y * (dy - rowsum(dy*y)), row by row, with y recomputed from the saved
     * forward input x. x/dy/dx share shape and dtype. The work is chunked across
     * the CPU pool by rows; futures are drained before returning.
     * @brief CPU standalone softmax backward.
     * @param dy Upstream gradient tensor.
     * @param x Forward input tensor.
     * @param dx Output gradient (written).
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if any tensor is not on the CPU.
     * @throws ErrorHandling::NotImplemented for an unsupported dtype.
     */
    void softmax_g_cpu(const Tensor& dy, const Tensor& x, Tensor& dx) {
        if (dy.device != Device::CPU || x.device != Device::CPU || dx.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(x.device));
        }
        using DType = DataTypes::DType;
        const auto rows = x.shape[0];
        const auto cols_store = x.shape[1];
        const auto packed = x.dtype == DType::Float16x2 || x.dtype == DType::BFloat16x2 || x.dtype == DType::Int4x2;
        const auto cols_eff = cols_store * (packed ? size_t{2} : size_t{1});
        auto run = [&]<typename T, typename Acc>(const T* x_ptr, const T* dy_ptr, T* dx_ptr) {
            drain(launch_rows(rows, [&](const size_t b, const size_t e) {
                for (auto r = b; r < e; ++r) {
                    if constexpr (std::is_same_v<T, float16x2_t> || std::is_same_v<T, bfloat16x2_t> || std::is_same_v<T, int4x2_t>) {
                        softmax_g_row_packed<T, Acc>(x_ptr + r * cols_store, dy_ptr + r * cols_store, dx_ptr + r * cols_store, cols_eff);
                    } else {
                        softmax_g_row<T, Acc>(x_ptr + r * cols_store, dy_ptr + r * cols_store, dx_ptr + r * cols_store, cols_eff);
                    }
                }
            }, x.numel * sizeof(T), x, dx));
        };
        switch (x.dtype) {
            case DType::Float32: run.operator()<float32_t, float32_t>(static_cast<const float32_t*>(x.data) + x.offset, static_cast<const float32_t*>(dy.data) + dy.offset, static_cast<float32_t*>(dx.data) + dx.offset); break;
            case DType::Float64: run.operator()<float64_t, float64_t>(static_cast<const float64_t*>(x.data) + x.offset, static_cast<const float64_t*>(dy.data) + dy.offset, static_cast<float64_t*>(dx.data) + dx.offset); break;
            case DType::Float16: run.operator()<float16_t, float32_t>(static_cast<const float16_t*>(x.data) + x.offset, static_cast<const float16_t*>(dy.data) + dy.offset, static_cast<float16_t*>(dx.data) + dx.offset); break;
            case DType::BFloat16: run.operator()<bfloat16_t, float32_t>(static_cast<const bfloat16_t*>(x.data) + x.offset, static_cast<const bfloat16_t*>(dy.data) + dy.offset, static_cast<bfloat16_t*>(dx.data) + dx.offset); break;
            case DType::Float16x2: run.operator()<float16x2_t, float32_t>(static_cast<const float16x2_t*>(x.data) + x.offset, static_cast<const float16x2_t*>(dy.data) + dy.offset, static_cast<float16x2_t*>(dx.data) + dx.offset); break;
            case DType::BFloat16x2: run.operator()<bfloat16x2_t, float32_t>(static_cast<const bfloat16x2_t*>(x.data) + x.offset, static_cast<const bfloat16x2_t*>(dy.data) + dy.offset, static_cast<bfloat16x2_t*>(dx.data) + dx.offset); break;
            case DType::Int8: run.operator()<int8_t, float64_t>(static_cast<const int8_t*>(x.data) + x.offset, static_cast<const int8_t*>(dy.data) + dy.offset, static_cast<int8_t*>(dx.data) + dx.offset); break;
            case DType::UInt8: run.operator()<uint8_t, float64_t>(static_cast<const uint8_t*>(x.data) + x.offset, static_cast<const uint8_t*>(dy.data) + dy.offset, static_cast<uint8_t*>(dx.data) + dx.offset); break;
            case DType::Int16: run.operator()<int16_t, float64_t>(static_cast<const int16_t*>(x.data) + x.offset, static_cast<const int16_t*>(dy.data) + dy.offset, static_cast<int16_t*>(dx.data) + dx.offset); break;
            case DType::UInt16: run.operator()<uint16_t, float64_t>(static_cast<const uint16_t*>(x.data) + x.offset, static_cast<const uint16_t*>(dy.data) + dy.offset, static_cast<uint16_t*>(dx.data) + dx.offset); break;
            case DType::Int32: run.operator()<int32_t, float64_t>(static_cast<const int32_t*>(x.data) + x.offset, static_cast<const int32_t*>(dy.data) + dy.offset, static_cast<int32_t*>(dx.data) + dx.offset); break;
            case DType::UInt32: run.operator()<uint32_t, float64_t>(static_cast<const uint32_t*>(x.data) + x.offset, static_cast<const uint32_t*>(dy.data) + dy.offset, static_cast<uint32_t*>(dx.data) + dx.offset); break;
            case DType::Int64: run.operator()<int64_t, float64_t>(static_cast<const int64_t*>(x.data) + x.offset, static_cast<const int64_t*>(dy.data) + dy.offset, static_cast<int64_t*>(dx.data) + dx.offset); break;
            case DType::UInt64: run.operator()<uint64_t, float64_t>(static_cast<const uint64_t*>(x.data) + x.offset, static_cast<const uint64_t*>(dy.data) + dy.offset, static_cast<uint64_t*>(dx.data) + dx.offset); break;
            case DType::Int4x2: run.operator()<int4x2_t, float64_t>(static_cast<const int4x2_t*>(x.data) + x.offset, static_cast<const int4x2_t*>(dy.data) + dy.offset, static_cast<int4x2_t*>(dx.data) + dx.offset); break;
            [[unlikely]] default: throw ErrorHandling::NotImplemented();
        }
    }

    /**
     * inputs are (x). Only x receives a gradient. The kernel keeps its
     * operand-dtype semantics — it runs on x/dy in their OWN dtype (the integer
     * and packed-lane class structure are forward-op contracts), with dy and x
     * boundary-cast from grad_dtype into the work dtype first (GradDType.h, the
     * single conversion rule); the computed dx delta is then boundary-cast into
     * grad_dtype so the delta leaving the node IS the assigned dtype.
     * @brief Backward trampoline for standalone softmax.
     * @param output_grad Gradient of this node's output tensor, in grad_dtype.
     * @param inputs Input tensors of the forward op.
     * @param n_inputs Number of entries in inputs.
     * @param grad_dtype The calling backward()'s assigned gradient dtype.
     * @throws ErrorHandling::UnReachableCode if n_inputs is not 1 or the device is neither CPU nor GPU.
     */
    void softmax_g_node(Node*, Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        if (n_inputs != 1) [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        auto& x = *inputs[0];
        if (!x.requires_grad) {
            return;
        }
        output_grad->sync();
        x.sync();
        const auto work = is_float_kind(grad_dtype)
            ? grad_dtype
            : DataTypes::DType::Float32;
        const auto xw = grad_converted_copy(x, work);
        auto dy_cast = std::optional<Tensor>{};
        const Tensor* dyw = output_grad;
        if (output_grad->dtype != work) {
            dy_cast = grad_converted_copy(*output_grad, work);
            dyw = &*dy_cast;
        }
        auto dx = Tensor(std::vector(x.shape, x.shape + x.ndim), work, x.device, false);
        if (x.device == Device::CPU) {
            softmax_g_cpu(*dyw, xw, dx);
        } else if (x.device == Device::GPU) {
            softmax_g_gpu(*dyw, xw, dx);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        grad_convert_inplace(dx, grad_dtype);
        if (x.node != nullptr) {
            grad_probe_hit(dx.dtype);
        }
        x.accumulate_grad(std::move(dx));
    }

    /**
     * wire_node's function-pointer type takes a mutable output_grad while
     * softmax_g_node only reads it. The trampoline forwards unchanged.
     * @brief Signature bridge between wire_node and softmax_g_node.
     */
    void softmax_g_bridge(Node* node, Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        softmax_g_node(node, output_grad, inputs, n_inputs, grad_dtype);
    }
}
