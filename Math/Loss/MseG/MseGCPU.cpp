/**
 * @file MseGCPU.cpp
 * @author GreenBean
 * @brief CPU backend for mse backward, plus the mse backward trampoline
 * (mse_g_node). The update is a pure elementwise scaling of the stored
 * difference, so there are no SIMD kernels (same reasoning as relu_g's
 * select and softmax_ce_g's scalar exp/log dominance).
 * @version 0.1
 * @date 2026-10-23
 */
#include <cstdint>
#include <future>
#include <limits>
#include <type_traits>
#include <vector>

#include "Math/Autograd/GradDType.h"
#include "Math/Loss/LossShared.h"
#include "Math/Loss/MseG/MseG.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Tensors::detail {
    using namespace loss_shared;

    namespace {
        // --------------------------- mse_g kernels ---------------------------

        /**
         * Packed grad dtypes read the low lane (the forward store_loss_mean
         * duplicates the mean into both lanes); int4x2 reads the low nibble.
         * @brief Widen the scalar dloss gradient element to the accumulator width.
         * @param g Pointer to the single dloss gradient element, of dtype T.
         * @return The dloss value in the accumulator type Acc.
         */
        template<typename T, typename Acc>
        [[nodiscard]] Acc widen_dloss(const void* g) {
            if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                return *static_cast<const T*>(g);
            } else if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
                return widen_lane(*static_cast<const T*>(g));
            } else if constexpr (std::is_same_v<T, float16x2_t> || std::is_same_v<T, bfloat16x2_t>) {
                using H = packed_lane_t<T>;
                const auto bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(*static_cast<const T*>(g));
                return widen_lane(lane_lo<H>(bits));
            } else if constexpr (std::is_same_v<T, int4x2_t>) {
                return static_cast<float64_t>(i4_lane_lo(*static_cast<const int4x2_t*>(g)));
            } else {
                return static_cast<float64_t>(*static_cast<const T*>(g));
            }
        }

        /**
         * Narrow one gradient value into the operand dtype: floats pass (halfs
         * narrow RNE through the unaccel pathway); integers clamp to the dtype
         * range and truncate toward zero, mapping NaN to 0 (no integer NaN) —
         * the same contract as softmax_ce_g's narrow_grad.
         * @brief Store one mse_g gradient value narrowed to T.
         * @param v_raw Gradient value in the accumulator width.
         * @return The value narrowed to T.
         */
        template<typename T, typename Acc>
        [[nodiscard]] T narrow_grad(const Acc v_raw) {
            if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                return static_cast<T>(v_raw);
            } else if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
                return narrow_lane<T>(static_cast<float32_t>(v_raw));
            } else {
                auto v = static_cast<float64_t>(v_raw);
                if (__builtin_isnan(v)) {
                    v = 0.0;
                }
                const auto lo = static_cast<float64_t>(std::numeric_limits<T>::lowest());
                const auto hi = static_cast<float64_t>(std::numeric_limits<T>::max());
                if (v <= lo) {
                    v = lo;
                } else if (v >= hi) {
                    v = hi;
                }
                return static_cast<T>(v);
            }
        }

        /**
         * @brief Scalar range mse backward (fp32/fp64, storage-width compute).
         * @param p Predictions, read only.
         * @param t Targets, read only.
         * @param out Output gradient, same shape/dtype as p (written).
         * @param scale 2 * dloss / logical element count, in Acc.
         * @param begin First element processed.
         * @param end One past the last element processed.
         */
        template<typename T, typename Acc>
        void mse_g_range_scalar(const T* p, const T* t, T* out, const Acc scale, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                out[i] = narrow_grad<T, Acc>(scale * (static_cast<Acc>(p[i]) - static_cast<Acc>(t[i])));
            }
        }

        /**
         * @brief Scalar range mse backward (fp16/bf16, widened fp32 compute).
         * @param p Predictions, read only.
         * @param t Targets, read only.
         * @param out Output gradient, same shape/dtype as p (written).
         * @param scale 2 * dloss / logical element count, in fp32.
         * @param begin First element processed.
         * @param end One past the last element processed.
         */
        template<typename H>
        void mse_g_range_half(const H* p, const H* t, H* out, const float32_t scale, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                out[i] = narrow_lane<H>(scale * (widen_lane(p[i]) - widen_lane(t[i])));
            }
        }

        /**
         * Per-lane widen/scale/narrow, repacked on store. The scale already
         * carries the 1 / (2 * numel) lane count (n_logical), so each lane
         * gradient is scale * (lane_p - lane_t).
         * @brief Scalar range mse backward (fp16x2/bf16x2, per-lane fp32 compute).
         * @param p Predictions, read only.
         * @param t Targets, read only.
         * @param out Output gradient, same shape/dtype as p (written).
         * @param scale 2 * dloss / logical element count, in fp32.
         * @param begin First element processed.
         * @param end One past the last element processed.
         */
        template<typename P>
        void mse_g_range_packed(const P* p, const P* t, P* out, const float32_t scale, const size_t begin, const size_t end) {
            using H = packed_lane_t<P>;
            for (auto i = begin; i < end; ++i) {
                const auto p_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(p[i]);
                const auto t_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(t[i]);
                const auto g_lo = scale * (widen_lane(lane_lo<H>(p_bits)) - widen_lane(lane_lo<H>(t_bits)));
                const auto g_hi = scale * (widen_lane(lane_hi<H>(p_bits)) - widen_lane(lane_hi<H>(t_bits)));
                out[i] = pack_lanes<P>(Intrinsics::detail::bit_cast_memcpy<uint16_t>(narrow_lane<H>(g_lo)),
                                       Intrinsics::detail::bit_cast_memcpy<uint16_t>(narrow_lane<H>(g_hi)));
            }
        }

        /**
         * Per-lane fp64 compute, clamped to [-8, 7] (the int4 store contract),
         * after the int8 NaN -> 0 / clamp / truncate narrowing.
         * @brief Scalar range mse backward (int4x2, per-nibble fp64 compute).
         * @param p Predictions, read only.
         * @param t Targets, read only.
         * @param out Output gradient, same shape/dtype as p (written).
         * @param scale 2 * dloss / logical element count, in fp64.
         * @param begin First element processed.
         * @param end One past the last element processed.
         */
        void mse_g_range_i4x2(const int4x2_t* p, const int4x2_t* t, int4x2_t* out, const float64_t scale, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                const auto g_lo = scale * (static_cast<float64_t>(i4_lane_lo(p[i])) - static_cast<float64_t>(i4_lane_lo(t[i])));
                const auto g_hi = scale * (static_cast<float64_t>(i4_lane_hi(p[i])) - static_cast<float64_t>(i4_lane_hi(t[i])));
                out[i] = i4_pack(std::clamp(static_cast<int32_t>(narrow_grad<int8_t, float64_t>(g_lo)), -8, 7),
                                 std::clamp(static_cast<int32_t>(narrow_grad<int8_t, float64_t>(g_hi)), -8, 7));
            }
        }
    }

    /**
     * dpred = 2 * (pred - target) / n_logical * dloss, elementwise, with the
     * operand-dtype semantics: fp32/fp64 compute in their storage width,
     * fp16/bf16 and the lanes of packed x2 widen to fp32 (RNE narrow on
     * store), integer dtypes and int4x2 compute in fp64 and narrow with
     * clamp + truncate (NaN -> 0). n_logical is numel, doubled for packed
     * dtypes and int4x2 (their lanes/nibbles are the logical elements the
     * forward mean divided by). dloss_grad is a single element (packed dtypes
     * read the low lane, int4x2 the low nibble). Blocks until the workers finish.
     * @brief CPU mse backward.
     * @param dloss_grad Scalar upstream gradient of the loss.
     * @param pred Forward predictions.
     * @param targets Forward targets (read for the difference only).
     * @param dpred Output gradient, same shape/dtype/device as pred (written).
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if any tensor is not on the CPU.
     * @throws ErrorHandling::NotImplemented for an unsupported pred dtype.
     */
    void mse_g_cpu(const Tensor& dloss_grad, const Tensor& pred, const Tensor& targets, Tensor& dpred) {
        if (dloss_grad.device != Device::CPU || pred.device != Device::CPU || targets.device != Device::CPU || dpred.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(pred.device));
        }
        using DType = DataTypes::DType;
        const auto numel = pred.numel;
        const auto packed_kind = pred.dtype == DType::Float16x2 || pred.dtype == DType::BFloat16x2;
        const auto lane_factor = (packed_kind || pred.dtype == DType::Int4x2) ? size_t{2} : size_t{1};
        const auto n_logical = numel * lane_factor;
        /**
         * Widens the scalar upstream gradient, forms the scale 2 * dloss /
         * n_logical in the accumulator type, then drains a launch_loss_rows
         * over the elements; a (b, e) range is a block of storage elements.
         * @brief Pred-dtype driver for mse_g_cpu.
         * @param T Storage element type of pred and dpred.
         * @param Acc Accumulator type (float32_t for fp32/halfs/packed, float64_t for fp64/integers/int4x2).
         * @param p_ptr First pred element, offset applied.
         * @param t_ptr First targets element, offset applied.
         * @param o_ptr First dpred element, offset applied (written).
         * @param g_data Raw data pointer of the scalar dloss gradient tensor.
         */
        auto run_pred = [&]<typename T, typename Acc>(const T* p_ptr, const T* t_ptr, T* o_ptr, const void* g_data) {
            const auto dloss = widen_dloss<T, Acc>(g_data);
            const auto scale = (Acc{2} * dloss) / static_cast<Acc>(n_logical);
            drain(launch_loss_rows(numel, [&](const size_t b, const size_t e) { mse_g_range_scalar<T, Acc>(p_ptr, t_ptr, o_ptr, scale, b, e); }, pred, targets, dpred));
        };
        switch (pred.dtype) {
            case DType::Float32: run_pred.operator()<float32_t, float32_t>(static_cast<const float32_t*>(pred.data) + pred.offset, static_cast<const float32_t*>(targets.data) + targets.offset, static_cast<float32_t*>(dpred.data) + dpred.offset, dloss_grad.data); break;
            case DType::Float64: run_pred.operator()<float64_t, float64_t>(static_cast<const float64_t*>(pred.data) + pred.offset, static_cast<const float64_t*>(targets.data) + targets.offset, static_cast<float64_t*>(dpred.data) + dpred.offset, dloss_grad.data); break;
            case DType::Float16: {
                const auto dloss = widen_dloss<float16_t, float32_t>(dloss_grad.data);
                const auto scale = (float32_t{2} * dloss) / static_cast<float32_t>(n_logical);
                drain(launch_loss_rows(numel, [&](const size_t b, const size_t e) { mse_g_range_half<float16_t>(static_cast<const float16_t*>(pred.data) + pred.offset, static_cast<const float16_t*>(targets.data) + targets.offset, static_cast<float16_t*>(dpred.data) + dpred.offset, scale, b, e); }, pred, targets, dpred));
                break;
            }
            case DType::BFloat16: {
                const auto dloss = widen_dloss<bfloat16_t, float32_t>(dloss_grad.data);
                const auto scale = (float32_t{2} * dloss) / static_cast<float32_t>(n_logical);
                drain(launch_loss_rows(numel, [&](const size_t b, const size_t e) { mse_g_range_half<bfloat16_t>(static_cast<const bfloat16_t*>(pred.data) + pred.offset, static_cast<const bfloat16_t*>(targets.data) + targets.offset, static_cast<bfloat16_t*>(dpred.data) + dpred.offset, scale, b, e); }, pred, targets, dpred));
                break;
            }
            case DType::Float16x2: {
                const auto dloss = widen_dloss<float16x2_t, float32_t>(dloss_grad.data);
                const auto scale = (float32_t{2} * dloss) / static_cast<float32_t>(n_logical);
                drain(launch_loss_rows(numel, [&](const size_t b, const size_t e) { mse_g_range_packed<float16x2_t>(static_cast<const float16x2_t*>(pred.data) + pred.offset, static_cast<const float16x2_t*>(targets.data) + targets.offset, static_cast<float16x2_t*>(dpred.data) + dpred.offset, scale, b, e); }, pred, targets, dpred));
                break;
            }
            case DType::BFloat16x2: {
                const auto dloss = widen_dloss<bfloat16x2_t, float32_t>(dloss_grad.data);
                const auto scale = (float32_t{2} * dloss) / static_cast<float32_t>(n_logical);
                drain(launch_loss_rows(numel, [&](const size_t b, const size_t e) { mse_g_range_packed<bfloat16x2_t>(static_cast<const bfloat16x2_t*>(pred.data) + pred.offset, static_cast<const bfloat16x2_t*>(targets.data) + targets.offset, static_cast<bfloat16x2_t*>(dpred.data) + dpred.offset, scale, b, e); }, pred, targets, dpred));
                break;
            }
            case DType::Int8: case DType::UInt8: case DType::Int16: case DType::UInt16:
            case DType::Int32: case DType::UInt32: case DType::Int64: case DType::UInt64: {
                /**
                 * Runs the fp64 int range over every chunk and drains the futures.
                 * @brief Threaded integer mse backward for one concrete element type.
                 * @param p_ptr Predictions buffer typed as T, already offset.
                 * @param t_ptr Targets buffer typed as T, already offset.
                 * @param o_ptr Output buffer typed as T, already offset.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                auto launch_int = [&]<typename T>(const T* p_ptr, const T* t_ptr, T* o_ptr) {
                    const auto dloss = widen_dloss<T, float64_t>(dloss_grad.data);
                    const auto scale = (float64_t{2} * dloss) / static_cast<float64_t>(n_logical);
                    drain(launch_loss_rows(numel, [&](const size_t b, const size_t e) { mse_g_range_scalar<T, float64_t>(p_ptr, t_ptr, o_ptr, scale, b, e); }, pred, targets, dpred));
                };
                switch (pred.dtype) {
                    case DType::Int8: launch_int(static_cast<const int8_t*>(pred.data) + pred.offset, static_cast<const int8_t*>(targets.data) + targets.offset, static_cast<int8_t*>(dpred.data) + dpred.offset); break;
                    case DType::UInt8: launch_int(static_cast<const uint8_t*>(pred.data) + pred.offset, static_cast<const uint8_t*>(targets.data) + targets.offset, static_cast<uint8_t*>(dpred.data) + dpred.offset); break;
                    case DType::Int16: launch_int(static_cast<const int16_t*>(pred.data) + pred.offset, static_cast<const int16_t*>(targets.data) + targets.offset, static_cast<int16_t*>(dpred.data) + dpred.offset); break;
                    case DType::UInt16: launch_int(static_cast<const uint16_t*>(pred.data) + pred.offset, static_cast<const uint16_t*>(targets.data) + targets.offset, static_cast<uint16_t*>(dpred.data) + dpred.offset); break;
                    case DType::Int32: launch_int(static_cast<const int32_t*>(pred.data) + pred.offset, static_cast<const int32_t*>(targets.data) + targets.offset, static_cast<int32_t*>(dpred.data) + dpred.offset); break;
                    case DType::UInt32: launch_int(static_cast<const uint32_t*>(pred.data) + pred.offset, static_cast<const uint32_t*>(targets.data) + targets.offset, static_cast<uint32_t*>(dpred.data) + dpred.offset); break;
                    case DType::Int64: launch_int(static_cast<const int64_t*>(pred.data) + pred.offset, static_cast<const int64_t*>(targets.data) + targets.offset, static_cast<int64_t*>(dpred.data) + dpred.offset); break;
                    case DType::UInt64: launch_int(static_cast<const uint64_t*>(pred.data) + pred.offset, static_cast<const uint64_t*>(targets.data) + targets.offset, static_cast<uint64_t*>(dpred.data) + dpred.offset); break;
                    [[unlikely]] default: throw ErrorHandling::UnReachableCode();
                }
                break;
            }
            case DType::Int4x2: {
                const auto dloss = widen_dloss<int4x2_t, float64_t>(dloss_grad.data);
                const auto scale = (float64_t{2} * dloss) / static_cast<float64_t>(n_logical);
                drain(launch_loss_rows(numel, [&](const size_t b, const size_t e) { mse_g_range_i4x2(static_cast<const int4x2_t*>(pred.data) + pred.offset, static_cast<const int4x2_t*>(targets.data) + targets.offset, static_cast<int4x2_t*>(dpred.data) + dpred.offset, scale, b, e); }, pred, targets, dpred));
                break;
            }
            [[unlikely]] default: throw ErrorHandling::NotImplemented();
        }
    }

    /**
     * inputs are (pred, target). Only pred receives a gradient (targets
     * never require one); skipped when pred does not require grad. The kernel
     * keeps its operand-dtype semantics — it runs on pred in its OWN dtype
     * (the packed-lane / integer-accumulation structure is a forward-op
     * contract, not a grad-precision choice), with the scalar dloss
     * boundary-cast from grad_dtype into the pred dtype first (GradDType.h,
     * the single conversion rule); the computed dpred delta is then
     * boundary-cast into grad_dtype so the delta leaving the node IS the
     * assigned dtype. Syncs the tensors it reads.
     * @brief Backward trampoline for mse.
     * @param output_grad Gradient of this node's output tensor, in grad_dtype.
     * @param inputs Input tensors of the forward op.
     * @param n_inputs Number of entries in inputs.
     * @param grad_dtype The calling backward()'s assigned gradient dtype.
     * @throws ErrorHandling::UnReachableCode if n_inputs is not 2 or the device is neither CPU nor GPU.
     */
    void mse_g_node(const Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        if (n_inputs != 2) [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        auto& pred = *inputs[0];
        const auto& target = *inputs[1];
        if (!pred.requires_grad) {
            return;
        }
        output_grad->sync();
        pred.sync();
        target.sync();
        const auto dloss = grad_converted_copy(*output_grad, pred.dtype);
        auto dpred = Tensor(std::vector(pred.shape, pred.shape + pred.ndim), pred.dtype, pred.device, false);
        if (pred.device == Device::CPU) {
            mse_g_cpu(dloss, pred, target, dpred);
        } else if (pred.device == Device::GPU) {
            mse_g_gpu(dloss, pred, target, dpred);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        grad_convert_inplace(dpred, grad_dtype); // the delta leaves in the assigned dtype
        if (pred.node != nullptr) {
            grad_probe_hit(dpred.dtype); // test hook: an intermediate received a delta
        }
        pred.accumulate_grad(std::move(dpred));
    }

    /**
     * wire_node's function-pointer type takes a mutable output_grad while
     * mse_g_node takes a const one (a function-pointer type mismatch the
     * Clang build accepts as an extension but GCC rejects). The trampoline
     * only reads output_grad, so forward through a matching bridge.
     * @brief Signature bridge between wire_node and mse_g_node.
     * @param output_grad Gradient of the forward output, in grad_dtype.
     * @param inputs Input tensors of the forward op.
     * @param n_inputs Number of entries in inputs.
     * @param grad_dtype The calling backward()'s assigned gradient dtype.
     */
    void mse_g_bridge(Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        mse_g_node(output_grad, inputs, n_inputs, grad_dtype);
    }
}
