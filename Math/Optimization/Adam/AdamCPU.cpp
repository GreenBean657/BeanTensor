/**
 * @file AdamCPU.cpp
 * @author GreenBean
 * @brief CPU backend for adam_update. Hosts the Tensor::adam_update and
 * Tensor::adam_step method definitions (mirrors how Tensor::sgd_update lives
 * in SgdUpdateCPU.cpp), plus the optimizer-side moment state map: Adam needs
 * per-parameter first/second moments and a step count, and Tensor's verified
 * sizeof layout must not grow, so the state lives in an external map keyed by
 * the parameter's address (this file) instead of new Tensor members.
 * @version 0.1.3
 * @date 2026-10-23
 */
#include <cmath>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "Math/Autograd/GradDType.h"
#include "Math/Optimization/Adam/Adam.h"
#include "Math/Optimization/OptimizationShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Tensors::detail {
    using namespace optimization_shared;

    namespace {
        // --------------------------- Adam moment state ---------------------------

        /**
         * First/second moment buffers (same shape/dtype/device as the
         * parameter, zero-initialized on first use) plus the step counter
         * driving the bias correction. Heap-owned through unique_ptrs so map
         * rehashing never moves a Tensor (moves are legal but would needlessly
         * drain pending workers).
         */
        struct AdamStateEntry {
            std::unique_ptr<Tensor> m;
            std::unique_ptr<Tensor> v;
            uint64_t t = 0;
            Device device = Device::CPU;
        };

        /**
         * @brief Mutex serializing a whole Adam step on the state map: entry
         * lookup, stale-entry reset, step-count increment, and dispatch all
         * run under it, so no thread observes or resets an entry mid-step.
         * @return The map's mutex.
         */
        std::mutex& adam_state_mutex() {
            static std::mutex m;
            return m;
        }

        /**
         * Function-local static: one state map per process, initialized on
         * first use. Keyed by the parameter's address — the parameters must
         * outlive the optimizer's use of them (the same raw-pointer
         * discipline as backward()'s graph parents); a destroyed parameter
         * whose address is reused by a same-shape tensor would inherit stale
         * moments, which the shape/dtype/device validation below resets, but
         * callers must not rely on that.
         * @brief The optimizer-side Adam moment state map.
         * @return Map from parameter address to its moment state.
         */
        std::unordered_map<const Tensor*, std::unique_ptr<AdamStateEntry>>& adam_states() {
            static std::unordered_map<const Tensor*, std::unique_ptr<AdamStateEntry>> states;
            return states;
        }

        // --------------------------- adam kernels ---------------------------

        /**
         * m = fma(b1, m, (1-b1)*g); v = fma(b2, v, (1-b2)*g*g);
         * w = fma(-lr, (m/bc1) / (sqrt(v/bc2) + eps), w). All math in T's
         * storage width with single-rounding fma (matches the FMA-based
         * dispatch of the other optimization kernels); std::fma/std::sqrt
         * resolve to the fp32 (fmaf/sqrtf) or fp64 forms by overload.
         * @brief Scalar range Adam (fp32/fp64).
         * @param w Weights, updated in place.
         * @param m First moments, updated in place.
         * @param v Second moments, updated in place.
         * @param g Gradients, read only.
         * @param lr Learning rate.
         * @param b1 Beta 1.
         * @param b2 Beta 2.
         * @param bc1 1 - beta1^t (bias correction).
         * @param bc2 1 - beta2^t (bias correction).
         * @param eps Epsilon.
         * @param begin First element processed.
         * @param end One past the last element processed.
         */
        template<typename T>
        void adam_range_scalar(T* w, T* m, T* v, const T* g,
                               const T lr, const T b1, const T b2, const T bc1, const T bc2, const T eps,
                               const size_t begin, const size_t end) {
            const auto omb1 = T{1} - b1;
            const auto omb2 = T{1} - b2;
            const auto neg_lr = -lr;
            for (auto i = begin; i < end; ++i) {
                const auto mm = std::fma(b1, m[i], omb1 * g[i]);
                const auto vv = std::fma(b2, v[i], omb2 * g[i] * g[i]);
                m[i] = mm;
                v[i] = vv;
                const auto step = (mm / bc1) / (std::sqrt(vv / bc2) + eps);
                w[i] = std::fma(neg_lr, step, w[i]);
            }
        }

        /**
         * Widen to fp32, compute the fp32 Adam update, narrow m/v/w RNE
         * (unaccel pathway). The stored (narrowed) moments feed the next
         * step, exactly like the storage-width kernels.
         * @brief Scalar range Adam (fp16/bf16, fp32 compute).
         * @param w Weights, updated in place.
         * @param m First moments, updated in place.
         * @param v Second moments, updated in place.
         * @param g Gradients, read only.
         * @param lr Learning rate.
         * @param b1 Beta 1.
         * @param b2 Beta 2.
         * @param bc1 1 - beta1^t (bias correction).
         * @param bc2 1 - beta2^t (bias correction).
         * @param eps Epsilon.
         * @param begin First element processed.
         * @param end One past the last element processed.
         */
        template<typename H>
        void adam_range_half(H* w, H* m, H* v, const H* g,
                             const float32_t lr, const float32_t b1, const float32_t b2,
                             const float32_t bc1, const float32_t bc2, const float32_t eps,
                             const size_t begin, const size_t end) {
            const auto omb1 = 1.0f - b1;
            const auto omb2 = 1.0f - b2;
            const auto neg_lr = -lr;
            for (auto i = begin; i < end; ++i) {
                const auto gm = widen_lane(g[i]);
                const auto mm = std::fma(b1, widen_lane(m[i]), omb1 * gm);
                const auto vv = std::fma(b2, widen_lane(v[i]), omb2 * gm * gm);
                m[i] = narrow_lane<H>(mm);
                v[i] = narrow_lane<H>(vv);
                const auto step = (mm / bc1) / (std::sqrt(vv / bc2) + eps);
                w[i] = narrow_lane<H>(std::fma(neg_lr, step, widen_lane(w[i])));
            }
        }

        /**
         * Per-lane widen/compute/narrow, repacked on store.
         * @brief Scalar range Adam (fp16x2/bf16x2, per-lane fp32 compute).
         * @param w Weights, updated in place.
         * @param m First moments, updated in place.
         * @param v Second moments, updated in place.
         * @param g Gradients, read only.
         * @param lr Learning rate.
         * @param b1 Beta 1.
         * @param b2 Beta 2.
         * @param bc1 1 - beta1^t (bias correction).
         * @param bc2 1 - beta2^t (bias correction).
         * @param eps Epsilon.
         * @param begin First element processed.
         * @param end One past the last element processed.
         */
        template<typename P>
        void adam_range_packed(P* w, P* m, P* v, const P* g,
                               const float32_t lr, const float32_t b1, const float32_t b2,
                               const float32_t bc1, const float32_t bc2, const float32_t eps,
                               const size_t begin, const size_t end) {
            using H = packed_lane_t<P>;
            const auto omb1 = 1.0f - b1;
            const auto omb2 = 1.0f - b2;
            const auto neg_lr = -lr;
            for (auto i = begin; i < end; ++i) {
                const auto w_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(w[i]);
                const auto m_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(m[i]);
                const auto v_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(v[i]);
                const auto g_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(g[i]);
                /**
                 * Runs the fp32 Adam lane update for one packed lane. Captures b1/b2/omb1/omb2/bc1/bc2/lr/eps/neg_lr by reference.
                 * @brief Update one packed lane.
                 * @param w_lane Current weight lane bits.
                 * @param m_lane Current first-moment lane bits.
                 * @param v_lane Current second-moment lane bits.
                 * @param g_lane Gradient lane bits.
                 * @return The narrowed (weight, first moment, second moment) lane bits.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                auto lane_step = [&](const uint16_t w_lane, const uint16_t m_lane, const uint16_t v_lane, const uint16_t g_lane) {
                    const auto gm = widen_lane(lane_lo<H>(static_cast<uint32_t>(g_lane)));
                    const auto mm = std::fma(b1, widen_lane(lane_lo<H>(static_cast<uint32_t>(m_lane))), omb1 * gm);
                    const auto vv = std::fma(b2, widen_lane(lane_lo<H>(static_cast<uint32_t>(v_lane))), omb2 * gm * gm);
                    const auto step = (mm / bc1) / (std::sqrt(vv / bc2) + eps);
                    const auto new_w = narrow_lane<H>(std::fma(neg_lr, step, widen_lane(lane_lo<H>(static_cast<uint32_t>(w_lane)))));
                    return std::tuple{Intrinsics::detail::bit_cast_memcpy<uint16_t>(new_w),
                                      Intrinsics::detail::bit_cast_memcpy<uint16_t>(narrow_lane<H>(mm)),
                                      Intrinsics::detail::bit_cast_memcpy<uint16_t>(narrow_lane<H>(vv))};
                };
                const auto [w_lo, m_lo, v_lo] = lane_step(static_cast<uint16_t>(w_bits & 0xFFFFu), static_cast<uint16_t>(m_bits & 0xFFFFu), static_cast<uint16_t>(v_bits & 0xFFFFu), static_cast<uint16_t>(g_bits & 0xFFFFu));
                const auto [w_hi, m_hi, v_hi] = lane_step(static_cast<uint16_t>(w_bits >> 16), static_cast<uint16_t>(m_bits >> 16), static_cast<uint16_t>(v_bits >> 16), static_cast<uint16_t>(g_bits >> 16));
                w[i] = pack_lanes<P>(w_lo, w_hi);
                m[i] = pack_lanes<P>(m_lo, m_hi);
                v[i] = pack_lanes<P>(v_lo, v_hi);
            }
        }

        /**
         * Futures land on all four tensors so teardown drains the workers
         * (they hold raw buffer pointers). Small inputs run inline. The worker
         * is captured by value, the futures outlive this call.
         * @brief Threaded Adam launcher, chunks an elementwise four-operand range worker across the CPU thread pool.
         * @param worker Callable taking (begin, end).
         * @param numel Element count to cover.
         * @param w Weight tensor, receives the futures.
         * @param grad Gradient tensor, receives the futures.
         * @param m First-moment tensor, receives the futures.
         * @param v Second-moment tensor, receives the futures.
         */
        template<typename Worker>
        void launch_adam(Worker&& worker, const size_t numel, const Tensor& w, const Tensor& grad, const Tensor& m, const Tensor& v) {
            const auto t_count = static_cast<size_t>(IncludedCPUFeatures().threads);
            if (t_count <= 1 || numel <= t_count * 1024) {
                worker(0, numel);
                return;
            }
            auto& threadMgr = Threading::GetCPUThreadPool();
            std::vector<std::shared_future<void>> futures;
            const auto chunk = (numel + t_count - 1) / t_count;
            for (auto t = size_t{0}; t < t_count; ++t) {
                const auto begin = t * chunk;
                if (begin >= numel) {
                    break;
                }
                // worker is captured BY VALUE: these futures outlive this call
                // (they land on the tensors), so a reference would dangle.
                futures.push_back(threadMgr.submit([worker, begin, end = std::min(begin + chunk, numel)] { worker(begin, end); }, Threading::TaskPriority::Normal).share());
            }
            w.give_futures(futures);
            grad.give_futures(futures);
            m.give_futures(futures);
            v.give_futures(futures);
        }
    }

    /**
     * Dispatches on dtype over the float kinds (the Tensor method rejects
     * integer kinds before reaching here): fp32/fp64 in their storage width,
     * fp16/bf16 and the lanes of packed x2 widened to fp32 with RNE narrowing
     * on store. Chunked across the CPU thread pool; hyperparameters arrive as
     * fp64 and cast once per range.
     * @brief CPU in-place Adam moment step.
     * @param w Weights, updated in place.
     * @param grad Gradients, same shape/dtype/device as w.
     * @param m First moments, same shape/dtype/device as w (updated in place).
     * @param v Second moments, same shape/dtype/device as w (updated in place).
     * @param lr Learning rate.
     * @param beta1 Beta 1.
     * @param beta2 Beta 2.
     * @param beta1_corr 1 - beta1^t (bias correction).
     * @param beta2_corr 1 - beta2^t (bias correction).
     * @param epsilon Epsilon.
     */
    void adam_update_cpu(const Tensor& w, const Tensor& grad, const Tensor& m, const Tensor& v,
                         const float64_t lr, const float64_t beta1, const float64_t beta2,
                         const float64_t beta1_corr, const float64_t beta2_corr, const float64_t epsilon) {
        if (w.device != Device::CPU || grad.device != Device::CPU || m.device != Device::CPU || v.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(w.device));
        }
        using DType = DataTypes::DType;
        const auto numel = w.numel;
        switch (w.dtype) {
            case DType::Float32: {
                auto* w_ptr = static_cast<float32_t*>(w.data) + w.offset;
                auto* m_ptr = static_cast<float32_t*>(m.data) + m.offset;
                auto* v_ptr = static_cast<float32_t*>(v.data) + v.offset;
                const auto* g_ptr = static_cast<const float32_t*>(grad.data) + grad.offset;
                launch_adam([=](const size_t b, const size_t e) { adam_range_scalar<float32_t>(w_ptr, m_ptr, v_ptr, g_ptr, static_cast<float32_t>(lr), static_cast<float32_t>(beta1), static_cast<float32_t>(beta2), static_cast<float32_t>(beta1_corr), static_cast<float32_t>(beta2_corr), static_cast<float32_t>(epsilon), b, e); }, numel, w, grad, m, v);
                break;
            }
            case DType::Float64: {
                auto* w_ptr = static_cast<float64_t*>(w.data) + w.offset;
                auto* m_ptr = static_cast<float64_t*>(m.data) + m.offset;
                auto* v_ptr = static_cast<float64_t*>(v.data) + v.offset;
                const auto* g_ptr = static_cast<const float64_t*>(grad.data) + grad.offset;
                launch_adam([=](const size_t b, const size_t e) { adam_range_scalar<float64_t>(w_ptr, m_ptr, v_ptr, g_ptr, lr, beta1, beta2, beta1_corr, beta2_corr, epsilon, b, e); }, numel, w, grad, m, v);
                break;
            }
            case DType::Float16: {
                auto* w_ptr = static_cast<float16_t*>(w.data) + w.offset;
                auto* m_ptr = static_cast<float16_t*>(m.data) + m.offset;
                auto* v_ptr = static_cast<float16_t*>(v.data) + v.offset;
                const auto* g_ptr = static_cast<const float16_t*>(grad.data) + grad.offset;
                launch_adam([=](const size_t b, const size_t e) { adam_range_half<float16_t>(w_ptr, m_ptr, v_ptr, g_ptr, static_cast<float32_t>(lr), static_cast<float32_t>(beta1), static_cast<float32_t>(beta2), static_cast<float32_t>(beta1_corr), static_cast<float32_t>(beta2_corr), static_cast<float32_t>(epsilon), b, e); }, numel, w, grad, m, v);
                break;
            }
            case DType::BFloat16: {
                auto* w_ptr = static_cast<bfloat16_t*>(w.data) + w.offset;
                auto* m_ptr = static_cast<bfloat16_t*>(m.data) + m.offset;
                auto* v_ptr = static_cast<bfloat16_t*>(v.data) + v.offset;
                const auto* g_ptr = static_cast<const bfloat16_t*>(grad.data) + grad.offset;
                launch_adam([=](const size_t b, const size_t e) { adam_range_half<bfloat16_t>(w_ptr, m_ptr, v_ptr, g_ptr, static_cast<float32_t>(lr), static_cast<float32_t>(beta1), static_cast<float32_t>(beta2), static_cast<float32_t>(beta1_corr), static_cast<float32_t>(beta2_corr), static_cast<float32_t>(epsilon), b, e); }, numel, w, grad, m, v);
                break;
            }
            case DType::Float16x2: {
                auto* w_ptr = static_cast<float16x2_t*>(w.data) + w.offset;
                auto* m_ptr = static_cast<float16x2_t*>(m.data) + m.offset;
                auto* v_ptr = static_cast<float16x2_t*>(v.data) + v.offset;
                const auto* g_ptr = static_cast<const float16x2_t*>(grad.data) + grad.offset;
                launch_adam([=](const size_t b, const size_t e) { adam_range_packed<float16x2_t>(w_ptr, m_ptr, v_ptr, g_ptr, static_cast<float32_t>(lr), static_cast<float32_t>(beta1), static_cast<float32_t>(beta2), static_cast<float32_t>(beta1_corr), static_cast<float32_t>(beta2_corr), static_cast<float32_t>(epsilon), b, e); }, numel, w, grad, m, v);
                break;
            }
            case DType::BFloat16x2: {
                auto* w_ptr = static_cast<bfloat16x2_t*>(w.data) + w.offset;
                auto* m_ptr = static_cast<bfloat16x2_t*>(m.data) + m.offset;
                auto* v_ptr = static_cast<bfloat16x2_t*>(v.data) + v.offset;
                const auto* g_ptr = static_cast<const bfloat16x2_t*>(grad.data) + grad.offset;
                launch_adam([=](const size_t b, const size_t e) { adam_range_packed<bfloat16x2_t>(w_ptr, m_ptr, v_ptr, g_ptr, static_cast<float32_t>(lr), static_cast<float32_t>(beta1), static_cast<float32_t>(beta2), static_cast<float32_t>(beta1_corr), static_cast<float32_t>(beta2_corr), static_cast<float32_t>(epsilon), b, e); }, numel, w, grad, m, v);
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
     * Both tensors are synced before dispatch; requires identical shape,
     * device, contiguous, and a float-kind dtype (Adam's sqrt/moment math is
     * float-only; integer kinds throw, mirroring pow/exp/log's contract). A
     * grad dtype differing from the weight dtype (backward(DType) makes the
     * grad dtype a per-call knob) converts a private copy into the weight
     * dtype via the single grad-conversion rule (GradDType.h), same as
     * sgd_update. The first/second moments and the step counter live in the
     * optimizer-side state map (AdamCPU.cpp), keyed by this tensor's address:
     * this tensor must outlive the optimizer's use of it, and the moments
     * persist across calls — zeroing them is the caller's job (drop the
     * optimizer or use a fresh parameter). Requires 0 <= beta1, beta2 < 1
     * and epsilon > 0 (documented contract; garbage in, garbage out beyond
     * that). Async on both devices: the update queues onto the thread pool
     * and errors materialize at sync().
     * @brief In-place Adam step, this -= lr * m_hat / (sqrt(v_hat) + eps).
     * @param gradient Gradient tensor (same shape/device as this; any dtype —
     * a mismatch converts a copy into this tensor's dtype).
     * @param lr Learning rate.
     * @param beta1 First-moment decay.
     * @param beta2 Second-moment decay.
     * @param epsilon Denominator epsilon.
     * @throws ErrorHandling::MismatchedShapes Shapes differ.
     * @throws ErrorHandling::MismatchedDevices Devices differ.
     * @throws ErrorHandling::NotImplemented Non-contiguous operand, or a non-float dtype.
     */
    void Tensor::adam_update(const Tensor& gradient, const float64_t lr, const float64_t beta1,
                             const float64_t beta2, const float64_t epsilon) {
        if (!this->is_same_shape(gradient)) {
            throw ErrorHandling::MismatchedShapes("adam_update requires identical shapes");
        }
        if (this->device != gradient.device) {
            throw ErrorHandling::MismatchedDevices();
        }
        if (!this->is_contiguous || !gradient.is_contiguous) {
            throw ErrorHandling::NotImplemented();
        }
        {
            using DType = DataTypes::DType;
            switch (this->dtype) {
                case DType::Float16: case DType::BFloat16: case DType::Float16x2: case DType::BFloat16x2:
                case DType::Float32: case DType::Float64: {
                    break;
                }
                default: {
                    throw ErrorHandling::NotImplemented();
                }
            }
        }
        if (this->dtype != gradient.dtype) {
            const auto g = detail::grad_converted_copy(gradient, this->dtype);
            this->adam_update(g, lr, beta1, beta2, epsilon);
            return;
        }
        sync();
        gradient.sync();
        // Moment state: external optimizer-side map, keyed by this tensor's
        // address (Tensor's verified sizeof layout stays untouched). The
        // mutex is held across lookup, the stale-entry check/reset, the step
        // increment, and the dispatch: a stale entry — device rewritten by
        // to(), or a reused address with a different shape/dtype — resets
        // atomically with respect to every other step on the map, and two
        // threads stepping the SAME parameter serialize their whole step
        // instead of interleaving a reset with an in-flight dispatch.
        auto& states = detail::adam_states();
        auto& mutex = detail::adam_state_mutex();
        const auto lock = std::lock_guard{mutex};
        auto& slot = states[this];
        if (!slot) {
            slot = std::make_unique<detail::AdamStateEntry>();
        }
        auto* const entry = slot.get();
        if (entry->m == nullptr || entry->device != this->device ||
            !entry->m->is_same_shape(*this) || entry->m->get_dtype() != this->dtype) {
            entry->m = std::make_unique<Tensor>(std::vector(this->shape, this->shape + this->ndim), this->dtype, this->device, false);
            entry->v = std::make_unique<Tensor>(std::vector(this->shape, this->shape + this->ndim), this->dtype, this->device, false);
            entry->m->set_zero();
            entry->v->set_zero();
            entry->m->sync();
            entry->v->sync();
            entry->t = 0;
            entry->device = this->device;
        }
        entry->t += 1;
        const auto t_f = static_cast<float64_t>(entry->t);
        const auto beta1_corr = 1.0 - std::pow(beta1, t_f);
        const auto beta2_corr = 1.0 - std::pow(beta2, t_f);
        if (this->device == Device::CPU) {
            detail::adam_update_cpu(*this, gradient, *entry->m, *entry->v, lr, beta1, beta2, beta1_corr, beta2_corr, epsilon);
        } else if (this->device == Device::GPU) {
            detail::adam_update_gpu(*this, gradient, *entry->m, *entry->v, lr, beta1, beta2, beta1_corr, beta2_corr, epsilon);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
    }

    /**
     * Uses the gradient attached to this tensor.
     * @brief In-place Adam step from this tensor's own gradient.
     * @param lr Learning rate.
     * @param beta1 First-moment decay.
     * @param beta2 Second-moment decay.
     * @param epsilon Denominator epsilon.
     * @throws ErrorHandling::NoGradient requires_grad is false or grad is null.
     */
    void Tensor::adam_step(const float64_t lr, const float64_t beta1, const float64_t beta2, const float64_t epsilon) {
        if (!this->requires_grad || this->grad == nullptr) {
            throw ErrorHandling::NoGradient();
        }
        adam_update(*this->grad, lr, beta1, beta2, epsilon);
    }

    /**
     * Reads the optimizer's map under the same mutex the step holds, so a
     * concurrent step cannot resize the map mid-lookup. Does not create an entry.
     * @brief Byte size of this tensor's Adam moment state.
     * @return Bytes held for first and second moments, or 0 if none.
     */
    size_t Tensor::adam_state_byte_count() const {
        const auto lock = std::lock_guard{detail::adam_state_mutex()};
        const auto& states = detail::adam_states();
        const auto found = states.find(this);
        if (found == states.end() || !found->second) {
            return 0;
        }
        const auto& entry = *found->second;
        if (entry.m == nullptr || entry.v == nullptr) {
            return 0;
        }
        return entry.m->byte_count() + entry.v->byte_count();
    }

    /**
     * Erases this tensor's entry under the same mutex a step holds. Erasing
     * destroys the moment tensors, which drain their own pending work.
     * @brief Drop this tensor's Adam moment state from the optimizer's map.
     */
    void Tensor::forget_adam_state() const {
        const auto lock = std::lock_guard{detail::adam_state_mutex()};
        detail::adam_states().erase(this);
    }
}
