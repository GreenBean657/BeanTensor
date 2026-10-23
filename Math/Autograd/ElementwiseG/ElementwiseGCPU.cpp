/**
 * @file ElementwiseGCPU.cpp
 * @author GreenBean
 * @brief CPU backend for elementwise autograd backward: the add/sub/mul/div/pow
 * (tensor-tensor and tensor-scalar) and exp/log backward trampolines, plus the
 * broadcast gradient reduction (bcast_sum_to_cpu). The trampolines compose the
 * full (output-shaped) per-operand delta from the public forward ops (the
 * GemmG reuse precedent) in the assigned grad dtype — or in fp32 when an
 * integer assigned dtype meets pow/exp, whose deltas need transcendentals — and
 * hand it to ew_deliver_delta, which sum-reduces it over the operand's
 * broadcast dims (bcast_sum_to) and boundary-casts it into grad_dtype per the
 * single conversion rule (GradDType.h).
 * @version 0.1
 * @date 2026-10-23
 */
#include <optional>
#include <type_traits>
#include <vector>

#include "Math/Autograd/AutogradShared.h"
#include "Math/Autograd/ElementwiseG/ElementwiseG.h"
#include "Math/Autograd/ElementwiseG/ElementwiseGShared.h"
#include "Math/Autograd/GradDType.h"
#include "Math/Elementwise/Elementwise.h"
#include "Math/Elementwise/ElementwiseShared.h"
#include "Tensor/Tensor.h"

#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Tensors::detail {
    using namespace autograd_shared;
    using namespace elementwise_shared;

    namespace {
        // --------------------------- broadcast gradient reduction kernels ---------------------------

        /**
         * One accumulator per operand element (each element is reduced by
         * exactly one worker, in a fixed enumeration order), so floats sum in
         * the storage width with a chunking-independent, GPU-bit-exact order.
         * @brief Scalar range broadcast-reduce (fp32/fp64).
         * @param dx Output gradient buffer, op.numel elements (written).
         * @param delta Out-shaped delta buffer.
         * @param plan Broadcast-reduction plan.
         * @param begin First operand element processed.
         * @param end One past the last operand element processed.
         */
        template<typename T>
        void bcast_sum_range_float(T* dx, const T* delta, const BcastReducePlan& plan, const size_t begin, const size_t end) {
            for (auto j = begin; j < end; ++j) {
                auto acc = T{0};
                for_each_preimage(plan, j, [&](const size_t idx) { acc += delta[idx]; });
                dx[j] = acc;
            }
        }

        /**
         * Half dtypes accumulate in fp32 and narrow RNE once at store time
         * (the same contract as the bias_add_g column sums).
         * @brief Scalar range broadcast-reduce (fp16/bf16, fp32 accumulate).
         * @param dx Output gradient buffer, op.numel elements (written).
         * @param delta Out-shaped delta buffer.
         * @param plan Broadcast-reduction plan.
         * @param begin First operand element processed.
         * @param end One past the last operand element processed.
         */
        template<typename H>
        void bcast_sum_range_half(H* dx, const H* delta, const BcastReducePlan& plan, const size_t begin, const size_t end) {
            for (auto j = begin; j < end; ++j) {
                auto acc = float32_t{0.0f};
                for_each_preimage(plan, j, [&](const size_t idx) { acc += widen_lane(delta[idx]); });
                dx[j] = narrow_lane<H>(acc);
            }
        }

        /**
         * Packed dtypes reduce each lane independently in fp32 and repack on
         * store.
         * @brief Scalar range broadcast-reduce (fp16x2/bf16x2, per-lane fp32 accumulate).
         * @param dx Output gradient buffer, op.numel elements (written).
         * @param delta Out-shaped delta buffer.
         * @param plan Broadcast-reduction plan.
         * @param begin First operand element processed.
         * @param end One past the last operand element processed.
         */
        template<typename P>
        void bcast_sum_range_packed(P* dx, const P* delta, const BcastReducePlan& plan, const size_t begin, const size_t end) {
            using H = packed_lane_t<P>;
            for (auto j = begin; j < end; ++j) {
                auto lo = float32_t{0.0f};
                auto hi = float32_t{0.0f};
                for_each_preimage(plan, j, [&](const size_t idx) {
                    const auto bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(delta[idx]);
                    lo += widen_lane(lane_lo<H>(bits));
                    hi += widen_lane(lane_hi<H>(bits));
                });
                dx[j] = pack_lanes<P>(Intrinsics::detail::bit_cast_memcpy<uint16_t>(narrow_lane<H>(lo)),
                                      Intrinsics::detail::bit_cast_memcpy<uint16_t>(narrow_lane<H>(hi)));
            }
        }

        /**
         * Integer sums wrap modularly through the unsigned twin, bit-identical
         * on every backend and chunking.
         * @brief Scalar range broadcast-reduce (all integer dtypes, wrap).
         * @param dx Output gradient buffer, op.numel elements (written).
         * @param delta Out-shaped delta buffer.
         * @param plan Broadcast-reduction plan.
         * @param begin First operand element processed.
         * @param end One past the last operand element processed.
         */
        template<typename T>
        void bcast_sum_range_int(T* dx, const T* delta, const BcastReducePlan& plan, const size_t begin, const size_t end) {
            using U = std::make_unsigned_t<T>;
            for (auto j = begin; j < end; ++j) {
                auto acc = U{0};
                for_each_preimage(plan, j, [&](const size_t idx) { acc = static_cast<U>(acc + static_cast<U>(delta[idx])); });
                dx[j] = static_cast<T>(acc);
            }
        }

        /**
         * Int4x2 sums accumulate per nibble in int32 and wrap at pack time
         * (pack_int4's low-4-bit mask), equivalent to per-add wrap.
         * @brief Scalar range broadcast-reduce (int4x2, per-nibble).
         * @param dx Output gradient buffer, op.numel elements (written).
         * @param delta Out-shaped delta buffer.
         * @param plan Broadcast-reduction plan.
         * @param begin First operand element processed.
         * @param end One past the last operand element processed.
         */
        void bcast_sum_range_int4x2(int4x2_t* dx, const int4x2_t* delta, const BcastReducePlan& plan, const size_t begin, const size_t end) {
            for (auto j = begin; j < end; ++j) {
                auto lo = int32_t{0};
                auto hi = int32_t{0};
                for_each_preimage(plan, j, [&](const size_t idx) {
                    lo += int4_lo(delta[idx]);
                    hi += int4_hi(delta[idx]);
                });
                dx[j] = pack_int4(lo, hi);
            }
        }

        // --------------------------- delta composition ---------------------------

        /**
         * The analytic local derivatives composed onto dy, all through the
         * public (broadcast-capable) forward ops in the compute dtype:
         * add: dA = dB = dy; sub: dA = dy, dB = -dy; mul: dA = dy*b, dB = dy*a;
         * div: dA = dy/b, dB = -dy*a/b^2; pow: dA = dy*b*a^(b-1), dB = dy*a^b*ln(a).
         * Float div/pow/log follow the forward contracts (IEEE div, backend
         * libm pow/log); integer compute keeps the forward wrap/truncate
         * semantics, inherent to requesting an integer gradient chain.
         * @brief Full output-shaped delta for one side of a binary elementwise op.
         * @param dy Upstream gradient (output-shaped).
         * @param ag Left operand in the compute dtype (borrowed or cast).
         * @param bg Right operand in the compute dtype (borrowed or cast).
         * @param op Operation selector.
         * @param left True to build the left operand's delta, false the right's.
         * @return Fresh output-shaped delta in the compute dtype.
         */
        // clone() is non-const only because it queues onto the tensor's pool;
        // it never mutates the source's observable state (GradDType.h).
        [[nodiscard]] Tensor clone_delta(const Tensor& dy) {
            return const_cast<Tensor&>(dy).clone(true);
        }

        [[nodiscard]] Tensor binary_delta_full(const Tensor& dy, const Tensor& ag, const Tensor& bg, const ElementwiseOp op, const bool left) {
            switch (op) {
                case ElementwiseOp::Add: {
                    return clone_delta(dy);
                }
                case ElementwiseOp::Sub: {
                    if (left) {
                        return clone_delta(dy);
                    }
                    return dy.mul(-1.0);
                }
                case ElementwiseOp::Mul: {
                    if (left) {
                        return dy.mul(bg);
                    }
                    return dy.mul(ag);
                }
                case ElementwiseOp::Div: {
                    if (left) {
                        return dy.div(bg);
                    }
                    return dy.mul(ag).div(bg.mul(bg)).mul(-1.0);
                }
                case ElementwiseOp::Pow: {
                    if (left) {
                        return dy.mul(bg).mul(ag.pow(bg.sub(1.0)));
                    }
                    return dy.mul(ag.pow(bg)).mul(ag.log());
                }
                [[unlikely]] default: {
                    throw ErrorHandling::UnReachableCode();
                }
            }
        }

        /**
         * Tensor-scalar local derivatives composed onto dy: add/sub pass dy
         * through; mul scales by s; div divides by s; pow differentiates
         * a^s -> s*a^(s-1) (the scalar is narrowed into the compute dtype by
         * the scalar op itself, per the PERMISSIVE contract). The scalar
         * variant preserves the input shape, so no reduction is ever needed.
         * @brief Full delta for a tensor-scalar elementwise op.
         * @param dy Upstream gradient (operand-shaped).
         * @param ag Operand in the compute dtype (borrowed or cast; pow only).
         * @param op Operation selector.
         * @param s Scalar operand (as passed to the forward op).
         * @return Fresh operand-shaped delta in the compute dtype.
         */
        [[nodiscard]] Tensor scalar_delta_full(const Tensor& dy, const Tensor& ag, const ElementwiseOp op, const float64_t s) {
            switch (op) {
                case ElementwiseOp::Add:
                case ElementwiseOp::Sub: {
                    return clone_delta(dy);
                }
                case ElementwiseOp::Mul: {
                    return dy.mul(s);
                }
                case ElementwiseOp::Div: {
                    return dy.div(s);
                }
                case ElementwiseOp::Pow: {
                    return dy.mul(ag.pow(s - 1.0)).mul(s);
                }
                [[unlikely]] default: {
                    throw ErrorHandling::UnReachableCode();
                }
            }
        }
    }

    /**
     * Deep copy of op in dst under the single grad-conversion rule (see the
     * declaration in ElementwiseG.h for why the copy drops requires_grad).
     * @brief Graph-clean operand copy in the compute dtype.
     * @param op Forward operand.
     * @param dst Compute dtype (the assigned grad dtype, or the fp32 fallback).
     * @return A fresh non-grad tensor holding op's values in dst.
     */
    Tensor ew_operand_copy(const Tensor& op, const DataTypes::DType& dst) {
        auto copy = const_cast<Tensor&>(op).clone(true);
        if (copy.dtype != dst) {
            copy.convert_dtype(dst, ConversionClampMethod::PERMISSIVE);
            copy.sync();
        }
        copy.requires_grad = false;
        return copy;
    }

    /**
     * dx[j] = sum of delta[i] over every output index i that maps to operand
     * element j under the broadcast plan built from (delta, op). dx is
     * written (each element by exactly one worker — chunking over the
     * operand range is race-free by construction), so no zero-fill is
     * required. Floats sum in the storage width per element in the fixed
     * for_each_preimage order: bit-identical across chunkings and against
     * the GPU twin. Futures land on delta and dx; the caller's accumulate
     * (or teardown) drains them.
     * @brief CPU broadcast gradient reduction.
     * @param delta Out-shaped delta, CPU resident, contiguous.
     * @param op The broadcast operand (provides the target shape/strides).
     * @param dx Output gradient, op-shaped (written).
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if any tensor is not on the CPU.
     * @throws ErrorHandling::NotImplemented for an unsupported dtype.
     */
    void bcast_sum_to_cpu(const Tensor& delta, const Tensor& op, Tensor& dx) {
        if (delta.device != Device::CPU || op.device != Device::CPU || dx.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(delta.device));
        }
        using DType = DataTypes::DType;
        const auto plan = delta.build_bcast_reduce_plan(op);
        const auto numel = dx.numel;
        switch (delta.dtype) {
            case DType::Float32: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the operand element range [b, e). Captures by value via init-captures (dx_ptr, d_ptr, offsets already applied; plan by value); no default capture.
                 * Reduces delta into dx over [b, e) in the storage width.
                 * @brief Worker for bcast_sum_to_cpu, float32_t path.
                 * @param b First operand element index of the range (inclusive).
                 * @param e One past the last operand element index of the range.
                 */
                launch_elementwise([dx_ptr = static_cast<float32_t*>(dx.data) + dx.offset, d_ptr = static_cast<const float32_t*>(delta.data) + delta.offset, plan](const size_t b, const size_t e) { bcast_sum_range_float<float32_t>(dx_ptr, d_ptr, plan, b, e); }, numel, delta, delta, dx);
                break;
            }
            case DType::Float64: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the operand element range [b, e). Captures by value via init-captures (dx_ptr, d_ptr, offsets already applied; plan by value); no default capture.
                 * Reduces delta into dx over [b, e) in the storage width.
                 * @brief Worker for bcast_sum_to_cpu, float64_t path.
                 * @param b First operand element index of the range (inclusive).
                 * @param e One past the last operand element index of the range.
                 */
                launch_elementwise([dx_ptr = static_cast<float64_t*>(dx.data) + dx.offset, d_ptr = static_cast<const float64_t*>(delta.data) + delta.offset, plan](const size_t b, const size_t e) { bcast_sum_range_float<float64_t>(dx_ptr, d_ptr, plan, b, e); }, numel, delta, delta, dx);
                break;
            }
            case DType::Float16: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the operand element range [b, e). Captures by value via init-captures (dx_ptr, d_ptr, offsets already applied; plan by value); no default capture.
                 * Accumulates in fp32 and narrows RNE once at store time.
                 * @brief Worker for bcast_sum_to_cpu, float16_t path.
                 * @param b First operand element index of the range (inclusive).
                 * @param e One past the last operand element index of the range.
                 */
                launch_elementwise([dx_ptr = static_cast<float16_t*>(dx.data) + dx.offset, d_ptr = static_cast<const float16_t*>(delta.data) + delta.offset, plan](const size_t b, const size_t e) { bcast_sum_range_half<float16_t>(dx_ptr, d_ptr, plan, b, e); }, numel, delta, delta, dx);
                break;
            }
            case DType::BFloat16: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the operand element range [b, e). Captures by value via init-captures (dx_ptr, d_ptr, offsets already applied; plan by value); no default capture.
                 * Accumulates in fp32 and narrows RNE once at store time.
                 * @brief Worker for bcast_sum_to_cpu, bfloat16_t path.
                 * @param b First operand element index of the range (inclusive).
                 * @param e One past the last operand element index of the range.
                 */
                launch_elementwise([dx_ptr = static_cast<bfloat16_t*>(dx.data) + dx.offset, d_ptr = static_cast<const bfloat16_t*>(delta.data) + delta.offset, plan](const size_t b, const size_t e) { bcast_sum_range_half<bfloat16_t>(dx_ptr, d_ptr, plan, b, e); }, numel, delta, delta, dx);
                break;
            }
            case DType::Float16x2: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the operand element range [b, e). Captures by value via init-captures (dx_ptr, d_ptr, offsets already applied; plan by value); no default capture.
                 * Each element carries two lanes; both are reduced per element in fp32.
                 * @brief Worker for bcast_sum_to_cpu, float16x2_t packed path.
                 * @param b First operand element index of the range (inclusive).
                 * @param e One past the last operand element index of the range.
                 */
                launch_elementwise([dx_ptr = static_cast<float16x2_t*>(dx.data) + dx.offset, d_ptr = static_cast<const float16x2_t*>(delta.data) + delta.offset, plan](const size_t b, const size_t e) { bcast_sum_range_packed<float16x2_t>(dx_ptr, d_ptr, plan, b, e); }, numel, delta, delta, dx);
                break;
            }
            case DType::BFloat16x2: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the operand element range [b, e). Captures by value via init-captures (dx_ptr, d_ptr, offsets already applied; plan by value); no default capture.
                 * Each element carries two lanes; both are reduced per element in fp32.
                 * @brief Worker for bcast_sum_to_cpu, bfloat16x2_t packed path.
                 * @param b First operand element index of the range (inclusive).
                 * @param e One past the last operand element index of the range.
                 */
                launch_elementwise([dx_ptr = static_cast<bfloat16x2_t*>(dx.data) + dx.offset, d_ptr = static_cast<const bfloat16x2_t*>(delta.data) + delta.offset, plan](const size_t b, const size_t e) { bcast_sum_range_packed<bfloat16x2_t>(dx_ptr, d_ptr, plan, b, e); }, numel, delta, delta, dx);
                break;
            }
            case DType::Int4x2: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the operand element range [b, e). Captures by value via init-captures (dx_ptr, d_ptr, offsets already applied; plan by value); no default capture.
                 * Each element carries two nibbles; both are reduced per element in int32.
                 * @brief Worker for bcast_sum_to_cpu, int4x2_t path.
                 * @param b First operand element index of the range (inclusive).
                 * @param e One past the last operand element index of the range.
                 */
                launch_elementwise([dx_ptr = static_cast<int4x2_t*>(dx.data) + dx.offset, d_ptr = static_cast<const int4x2_t*>(delta.data) + delta.offset, plan](const size_t b, const size_t e) { bcast_sum_range_int4x2(dx_ptr, d_ptr, plan, b, e); }, numel, delta, delta, dx);
                break;
            }
            default: {
                /**
                 * Captures by reference: plan, numel, delta and dx from the enclosing function. Generic lambda; T is deduced from d_ptr.
                 * Chunks the operand range across the pool (each element is written by exactly one worker, so no partials are needed) and reduces delta into dx with wrapping integer sums.
                 * @brief Integer-dtype broadcast-reduce driver for element type T.
                 * @param T Integer element type of delta (int8_t ... uint64_t).
                 * @param d_ptr Pointer to the first element of delta, offset already applied.
                 */
                auto run_int = [&]<typename T>(const T* d_ptr) {
                    auto* dx_ptr = static_cast<T*>(dx.data) + dx.offset;
                    /**
                     * Runs on a pool worker (or inline for small inputs) over the operand element range [b, e). Captures by value via init-captures (dx_ptr, d_ptr already applied); no default capture.
                     * Integer sums wrap in T through the unsigned twin.
                     * @brief Worker for bcast_sum_to_cpu, integer path of type T.
                     * @param b First operand element index of the range (inclusive).
                     * @param e One past the last operand element index of the range.
                     */
                    launch_elementwise([dx_ptr, d_ptr, plan](const size_t b, const size_t e) { bcast_sum_range_int<T>(dx_ptr, d_ptr, plan, b, e); }, numel, delta, delta, dx);
                };
                switch (delta.dtype) {
                    case DType::Int8: run_int(static_cast<const int8_t*>(delta.data) + delta.offset); break;
                    case DType::UInt8: run_int(static_cast<const uint8_t*>(delta.data) + delta.offset); break;
                    case DType::Int16: run_int(static_cast<const int16_t*>(delta.data) + delta.offset); break;
                    case DType::UInt16: run_int(static_cast<const uint16_t*>(delta.data) + delta.offset); break;
                    case DType::Int32: run_int(static_cast<const int32_t*>(delta.data) + delta.offset); break;
                    case DType::UInt32: run_int(static_cast<const uint32_t*>(delta.data) + delta.offset); break;
                    case DType::Int64: run_int(static_cast<const int64_t*>(delta.data) + delta.offset); break;
                    case DType::UInt64: run_int(static_cast<const uint64_t*>(delta.data) + delta.offset); break;
                    [[unlikely]] default: throw ErrorHandling::NotImplemented();
                }
                break;
            }
        }
    }

    /**
     * Shared tail of every elementwise trampoline (see the declaration in
     * ElementwiseG.h for the contract).
     * @brief Reduce (if broadcast), boundary-cast, probe and accumulate a full delta.
     * @param full Output-shaped delta, consumed (moved from when shapes match).
     * @param operand The forward operand the delta belongs to.
     * @param work Compute dtype the delta was built in.
     * @param grad_dtype The calling backward()'s assigned gradient dtype.
     * @throws ErrorHandling::UnReachableCode if the operand is on neither CPU nor GPU.
     */
    void ew_deliver_delta(Tensor&& full, Tensor& operand, const DataTypes::DType& work, const DataTypes::DType& grad_dtype) {
        const auto same_shape = full.ndim == operand.ndim
            && std::equal(operand.shape, operand.shape + operand.ndim, full.shape);
        if (same_shape) {
            if (full.dtype != grad_dtype) {
                grad_convert_inplace(full, grad_dtype);
            }
            if (operand.node != nullptr) {
                grad_probe_hit(full.dtype); // test hook: an intermediate received a delta
            }
            operand.accumulate_grad(std::move(full));
            return;
        }
        // The reduce workers read full's buffer raw: land its producers first.
        full.sync();
        auto delta = Tensor(std::vector(operand.shape, operand.shape + operand.ndim), work, operand.device, false);
        if (operand.device == Device::CPU) {
            bcast_sum_to_cpu(full, operand, delta);
        } else if (operand.device == Device::GPU) {
            bcast_sum_to_gpu(full, operand, delta);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        if (delta.dtype != grad_dtype) {
            grad_convert_inplace(delta, grad_dtype);
        }
        if (operand.node != nullptr) {
            grad_probe_hit(delta.dtype);
        }
        operand.accumulate_grad(std::move(delta));
    }

    /**
     * inputs are (a, b). Each side's delta is composed onto output_grad in
     * the compute dtype (grad_dtype, or fp32 when an integer assigned dtype
     * meets pow — its deltas need transcendentals), boundary-casting the
     * operands it touches per the single conversion rule; broadcast operands
     * then receive the delta sum-reduced over their broadcast dims. Each
     * side is computed only if its tensor requires grad.
     * @brief Shared trampoline body for the tensor-tensor elementwise backward nodes.
     * @param output_grad Gradient of this node's output tensor, in grad_dtype.
     * @param inputs Input tensors of the forward op.
     * @param n_inputs Number of entries in inputs.
     * @param grad_dtype The calling backward()'s assigned gradient dtype.
     * @param op Operation selector (Add/Sub/Mul/Div/Pow).
     * @throws ErrorHandling::UnReachableCode if n_inputs is not 2.
     */
    void ew_binary_g_node_impl(Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype, const ElementwiseOp op) {
        if (n_inputs != 2) [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        auto& a = *inputs[0];
        auto& b = *inputs[1];
        if (!a.requires_grad && !b.requires_grad) {
            return;
        }
        output_grad->sync();
        const auto work = op == ElementwiseOp::Pow && !is_float_kind(grad_dtype)
            ? DataTypes::DType::Float32
            : grad_dtype;
        // pow under an integer assigned grad dtype computes in fp32, so the
        // upstream delta boundary-casts into the working dtype too (the
        // single conversion rule); the common path borrows output_grad as-is.
        auto dy_cast = std::optional<Tensor>{};
        const Tensor* dyw = output_grad;
        if (output_grad->dtype != work) {
            dy_cast = grad_converted_copy(*output_grad, work);
            dyw = &*dy_cast;
        }
        const bool uses_operands = op == ElementwiseOp::Mul || op == ElementwiseOp::Div || op == ElementwiseOp::Pow;
        // An operand that requires grad must be copied (graph-clean) before
        // it feeds the composed products — borrowing it would wire throwaway
        // nodes onto them. An operand that does not require grad cannot wire
        // anything and is borrowed as-is.
        auto a_copy = std::optional<Tensor>{};
        auto b_copy = std::optional<Tensor>{};
        const Tensor* ag = &a;
        const Tensor* bg = &b;
        if (uses_operands && a.requires_grad) {
            a_copy = ew_operand_copy(a, work);
            ag = &*a_copy;
        }
        if (uses_operands && b.requires_grad) {
            b_copy = ew_operand_copy(b, work);
            bg = &*b_copy;
        }
        if (a.requires_grad) {
            auto full = binary_delta_full(*dyw, *ag, *bg, op, true);
            ew_deliver_delta(std::move(full), a, work, grad_dtype);
        }
        if (b.requires_grad) {
            auto full = binary_delta_full(*dyw, *ag, *bg, op, false);
            ew_deliver_delta(std::move(full), b, work, grad_dtype);
        }
    }

    /**
     * inputs are (a) with the forward scalar bit-cast into inputs[1]
     * (scalar_payload — never dereferenced, never walked). The tensor-scalar
     * ops preserve the operand shape, so the composed delta is accumulated
     * directly after the same compute-dtype selection as the binary path.
     * @brief Shared trampoline body for the tensor-scalar elementwise backward nodes.
     * @param output_grad Gradient of this node's output tensor, in grad_dtype.
     * @param inputs Input slot 0 is the operand; slot 1 carries the scalar payload.
     * @param n_inputs Number of walked entries in inputs (always 1 here).
     * @param grad_dtype The calling backward()'s assigned gradient dtype.
     * @param op Operation selector (Add/Sub/Mul/Div/Pow).
     * @throws ErrorHandling::UnReachableCode if n_inputs is not 1.
     */
    void ew_scalar_g_node_impl(Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype, const ElementwiseOp op) {
        if (n_inputs != 1) [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        auto& a = *inputs[0];
        if (!a.requires_grad) {
            return;
        }
        const auto s = scalar_payload_value(inputs[1]);
        output_grad->sync();
        const auto work = op == ElementwiseOp::Pow && !is_float_kind(grad_dtype)
            ? DataTypes::DType::Float32
            : grad_dtype;
        // pow under an integer assigned grad dtype computes in fp32: the
        // upstream delta boundary-casts into the working dtype too.
        auto dy_cast = std::optional<Tensor>{};
        const Tensor* dyw = output_grad;
        if (output_grad->dtype != work) {
            dy_cast = grad_converted_copy(*output_grad, work);
            dyw = &*dy_cast;
        }
        // pow's delta recomputes a^(s-1) through the public pow op; a
        // requires grad here, so the copy keeps the product off the graph
        // (the other scalar ops touch only dy and the scalar itself).
        auto a_copy = std::optional<Tensor>{};
        const Tensor* ag = &a;
        if (op == ElementwiseOp::Pow) {
            a_copy = ew_operand_copy(a, work);
            ag = &*a_copy;
        }
        auto full = scalar_delta_full(*dyw, *ag, op, s);
        ew_deliver_delta(std::move(full), a, work, grad_dtype);
    }

    /**
     * inputs are (x). exp's delta is dy * exp(x) recomputed in the compute
     * dtype (grad_dtype when it is a float kind, else fp32 — exp needs a
     * transcendental and integer assigned dtypes boundary-cast the result at
     * delivery); log's delta is dy / x, pure division, computable directly in
     * any assigned dtype.
     * @brief Shared trampoline body for the unary elementwise backward nodes.
     * @param output_grad Gradient of this node's output tensor, in grad_dtype.
     * @param inputs Input tensors of the forward op.
     * @param n_inputs Number of entries in inputs.
     * @param grad_dtype The calling backward()'s assigned gradient dtype.
     * @param op Operation selector (Exp/Log).
     * @throws ErrorHandling::UnReachableCode if n_inputs is not 1.
     */
    void ew_unary_g_node_impl(Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype, const ElementwiseUnaryOp op) {
        if (n_inputs != 1) [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        auto& x = *inputs[0];
        if (!x.requires_grad) {
            return;
        }
        output_grad->sync();
        if (op == ElementwiseUnaryOp::Exp) {
            const auto work = is_float_kind(grad_dtype)
                ? grad_dtype
                : DataTypes::DType::Float32;
            // x requires grad here: copy (graph-clean) so exp's product
            // stays off the graph. dy is a grad tensor (never requires
            // grad), so it feeds the products as-is.
            const auto xw = ew_operand_copy(x, work);
            auto dy_cast = std::optional<Tensor>{};
            const Tensor* dyw = output_grad;
            if (output_grad->dtype != work) {
                dy_cast = grad_converted_copy(*output_grad, work);
                dyw = &*dy_cast;
            }
            auto dx = dyw->mul(xw.exp());
            ew_deliver_delta(std::move(dx), x, work, grad_dtype);
            return;
        }
        const auto xw = ew_operand_copy(x, grad_dtype);
        auto dx = output_grad->div(xw);
        ew_deliver_delta(std::move(dx), x, grad_dtype, grad_dtype);
    }

    void add_g_node(Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        ew_binary_g_node_impl(output_grad, inputs, n_inputs, grad_dtype, ElementwiseOp::Add);
    }

    void sub_g_node(Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        ew_binary_g_node_impl(output_grad, inputs, n_inputs, grad_dtype, ElementwiseOp::Sub);
    }

    void mul_g_node(Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        ew_binary_g_node_impl(output_grad, inputs, n_inputs, grad_dtype, ElementwiseOp::Mul);
    }

    void div_g_node(Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        ew_binary_g_node_impl(output_grad, inputs, n_inputs, grad_dtype, ElementwiseOp::Div);
    }

    void pow_g_node(Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        ew_binary_g_node_impl(output_grad, inputs, n_inputs, grad_dtype, ElementwiseOp::Pow);
    }

    void add_scalar_g_node(Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        ew_scalar_g_node_impl(output_grad, inputs, n_inputs, grad_dtype, ElementwiseOp::Add);
    }

    void sub_scalar_g_node(Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        ew_scalar_g_node_impl(output_grad, inputs, n_inputs, grad_dtype, ElementwiseOp::Sub);
    }

    void mul_scalar_g_node(Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        ew_scalar_g_node_impl(output_grad, inputs, n_inputs, grad_dtype, ElementwiseOp::Mul);
    }

    void div_scalar_g_node(Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        ew_scalar_g_node_impl(output_grad, inputs, n_inputs, grad_dtype, ElementwiseOp::Div);
    }

    void pow_scalar_g_node(Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        ew_scalar_g_node_impl(output_grad, inputs, n_inputs, grad_dtype, ElementwiseOp::Pow);
    }

    void exp_g_node(Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        ew_unary_g_node_impl(output_grad, inputs, n_inputs, grad_dtype, ElementwiseUnaryOp::Exp);
    }

    void log_g_node(Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        ew_unary_g_node_impl(output_grad, inputs, n_inputs, grad_dtype, ElementwiseUnaryOp::Log);
    }
}

namespace BeanTensor::Tensors {
    /**
     * this is the out-shaped delta; op is the broadcast operand being reduced
     * into. Only the shapes/strides are needed: the reduction maps free
     * (size-1 or left-padded missing) operand dims to full output extents
     * (see for_each_preimage).
     * @brief Build the broadcast-reduction plan for (this, op).
     * @param op The operand the delta will be reduced into.
     * @return The broadcast-reduction execution plan.
     */
    detail::BcastReducePlan Tensor::build_bcast_reduce_plan(const Tensor& op) const {
        detail::BcastReducePlan plan{};
        plan.out_ndim = this->ndim;
        for (auto d = size_t{0}; d < this->ndim; ++d) {
            plan.out_shape[d] = this->shape[d];
        }
        plan.out_strides[this->ndim - 1] = 1;
        for (auto d = this->ndim - 1; d-- > 0;) {
            plan.out_strides[d] = plan.out_strides[d + 1] * plan.out_shape[d + 1];
        }
        plan.op_ndim = op.ndim;
        for (auto od = size_t{0}; od < op.ndim; ++od) {
            plan.op_shape[od] = op.shape[od];
        }
        return plan;
    }
}
