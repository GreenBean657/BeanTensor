/**
 * @file Backward.cpp
 * @author GreenBean
 * @brief Tensor gradient plumbing: ensure_grad, accumulate_grad, wire_node,
 * zero_grad, set_grad_dtype (with its validate_grad_dtype precheck), and
 * backward() (reverse-mode backpropagation, Kahn's algorithm over consumer
 * counts). accumulate_grad dispatches to the grad_acc cpu/gpu backends.
 * @version 0.3
 * @date 2026-10-23
 */
#include <limits>
#include <unordered_set>
#include <vector>

#include "Math/Autograd/Autograd.h"
#include "Math/Autograd/GradDType.h"
#include "Tensor/Tensor.h"

namespace BeanTensor::Tensors {

    /**
     * Allocates grad with this tensor's shape, grad_dtype and device, zero-filled, and syncs
     * so the fill lands before any reader queues work. No-op if grad already exists.
     * Ownership: this tensor owns the heap-allocated grad.
     * @brief Lazily allocate a zeroed gradient tensor in grad_dtype.
     * @param grad_dtype Dtype of the gradient buffer for this backward() call.
     */
    void Tensor::ensure_grad(const DataTypes::DType grad_dtype) {
        if (this->grad != nullptr) {
            return;
        }
        this->grad = new Tensor(std::vector(this->shape, this->shape + this->ndim), grad_dtype, this->device, false);
        this->grad->set_zero();
        this->grad->sync(); // the zero-fill must land before any reader queues work
    }

    /**
     * If no grad exists, adopts delta as the grad (the move drains delta's workers);
     * the adopted buffer's dtype is the delta's, so a leaf's first backward() call
     * pins its grad dtype to that call's assigned grad dtype. Otherwise, when the
     * dtypes differ (a later call assigned a different grad dtype), delta is
     * converted into the pinned buffer's dtype via the single grad-conversion rule
     * (GradDType.h) — accumulation semantics are preserved across calls. Then syncs
     * delta and the existing grad and adds delta in place on this tensor's device.
     * delta must match this tensor's shape.
     * @brief Add a gradient contribution into this tensor's grad.
     * @param delta Gradient contribution; consumed (moved from or left drained).
     * @throws ErrorHandling::UnReachableCode if the device is neither CPU nor GPU.
     */
    void Tensor::accumulate_grad(Tensor&& delta) {
        if (this->grad == nullptr) {
            this->grad = new Tensor(std::move(delta)); // the move drains delta's workers
            return;
        }
        if (delta.dtype != this->grad->dtype) {
            // The existing buffer's dtype wins (leaf grads accumulate across
            // calls); the incoming delta converts into it, in place.
            detail::grad_convert_inplace(delta, this->grad->dtype);
        }
        delta.sync();
        // grad may still have queued writers (backward()'s async set_zero on
        // GPU, or an earlier grad_acc from a fan-out sibling); accumulating
        // before they land would race them on the same buffer.
        this->grad->sync();
        if (this->device == Device::CPU) {
            detail::grad_acc_cpu(*this->grad, delta);
        } else if (this->device == Device::GPU) {
            detail::grad_acc_gpu(*this->grad, delta);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
    }

    /**
     * No-op when neither input requires grad. Otherwise marks this tensor as a non-leaf
     * that requires grad, allocates a Node recording backward_fn and the input pointers
     * (non-owning: the inputs must outlive this tensor), and replaces any stale node.
     * @brief Attach an autograd node to this (output) tensor.
     * @param backward_fn Backward trampoline invoked as (owning node, output grad, inputs, input count, grad dtype).
     * @param a First input tensor.
     * @param b Optional second input tensor; nullptr for a unary op.
     */
    void Tensor::wire_node(void (*backward_fn)(Node*, Tensor*, Tensor**, size_t, DataTypes::DType), const Tensor& a, const Tensor* b) {
        if (!a.requires_grad && (b == nullptr || !b->requires_grad)) {
            return;
        }
        this->requires_grad = true;
        this->is_leaf = false;
        auto* new_node = new Node{};
        new_node->_backward_fn = backward_fn;
        new_node->_inputs[0] = const_cast<Tensor*>(&a);
        new_node->_n_inputs = 1;
        this->parents[0] = new_node->_inputs[0];
        if (b != nullptr) {
            new_node->_inputs[1] = const_cast<Tensor*>(b);
            new_node->_n_inputs = 2;
            this->parents[1] = new_node->_inputs[1];
        }
        this->n_parents = new_node->_n_inputs;
        delete this->node; // re-wiring replaces any stale node
        this->node = new_node;
    }

    /**
     * Deletes grad (its destructor drains pending workers) and resets it to nullptr;
     * the next accumulation reallocates it.
     * @brief Drop this tensor's accumulated gradient.
     */
    void Tensor::zero_grad() {
        delete this->grad; // ~Tensor drains grad's own pending workers first
        this->grad = nullptr;
    }

    /**
     * Refuses non-leaf tensors and tensors that do not require grad, then, when a
     * gradient is held with a dtype different from dt, refuses a narrowing change
     * under Reject. Syncs the held gradient first, since a pending conversion only
     * flips its dtype at sync().
     * @brief Check set_grad_dtype's preconditions without changing anything.
     * @param dt Dtype the gradient buffer would have.
     * @param narrowing Narrowing policy the change would use.
     * @throws ErrorHandling::NonLeafGradDType this tensor owns a graph node.
     * @throws ErrorHandling::NoGradient requires_grad is false.
     * @throws ErrorHandling::NarrowingGradConversion narrowing refused under Reject.
     */
    void Tensor::validate_grad_dtype(const DataTypes::DType dt, const DataTypes::GradNarrowing narrowing) const {
        if (this->node != nullptr) {
            throw ErrorHandling::NonLeafGradDType();
        }
        if (!this->requires_grad) {
            throw ErrorHandling::NoGradient();
        }
        if (this->grad == nullptr) {
            return;
        }
        this->grad->sync();
        if (
            const auto old = this->grad->dtype;
            old != dt && narrowing == DataTypes::GradNarrowing::Reject && DataTypes::is_narrowing_conversion(old, dt)
        ) {
            throw ErrorHandling::NarrowingGradConversion();
        }
    }

    /**
     * Validates via validate_grad_dtype, then: with no gradient held, allocates a
     * zeroed one in dt (so the leaf is pinned to dt for its next backward); with a
     * gradient of dt already, does nothing; otherwise converts the held buffer in
     * place under the single grad-conversion rule and syncs, so the conversion has
     * landed before the next backward() accumulates into it.
     * @brief Pin or change this leaf's gradient dtype.
     * @param dt Dtype the gradient buffer must have.
     * @param narrowing Reject refuses a narrowing change; Allow converts it.
     * @throws ErrorHandling::NonLeafGradDType this tensor owns a graph node.
     * @throws ErrorHandling::NoGradient requires_grad is false.
     * @throws ErrorHandling::NarrowingGradConversion narrowing refused under Reject.
     */
    void Tensor::set_grad_dtype(const DataTypes::DType dt, const DataTypes::GradNarrowing narrowing) {
        this->validate_grad_dtype(dt, narrowing);
        if (this->grad == nullptr) {
            this->ensure_grad(dt);
            return;
        }
        if (this->grad->dtype == dt) {
            return;
        }
        detail::grad_convert_inplace(*this->grad, dt);
    }

    /**
     * Walks the graph from this tensor, rebuilds/zeros the grads of node-owning
     * (intermediate) tensors in grad_dtype so they hold only this call's delta in
     * this call's dtype (leaf grads accumulate across calls; a leaf buffer keeps
     * the dtype its first call pinned), seeds this tensor's grad with ones in
     * grad_dtype, then fires each node's backward function in topological order
     * (Kahn's algorithm over consumer counts), handing every node the assigned
     * grad_dtype. Syncs every visited tensor and its grad before returning.
     * @brief Reverse-mode backpropagation from this tensor.
     * @param grad_dtype Dtype of the seed and every intermediate delta; free per
     * call, any supported dtype, no engine default.
     * @throws ErrorHandling::NoGradient if this tensor neither requires grad nor has a node.
     * @throws ErrorHandling::DimensionTooLarge if a tensor's consumer count exceeds 255.
     * @throws ErrorHandling::UnReachableCode if not every node fires (a cycle).
     */
    void Tensor::backward(const DataTypes::DType grad_dtype) {
        if (!this->requires_grad && this->node == nullptr) {
            throw ErrorHandling::NoGradient();
        }
        // Discover every tensor reachable through the graph.
        std::vector<Tensor*> visited;
        std::unordered_set<Tensor*> seen;
        std::vector stack{this};
        while (!stack.empty()) {
            auto* t = stack.back();
            stack.pop_back();
            if (!seen.insert(t).second) {
                continue;
            }
            visited.push_back(t);
            if (t->node != nullptr) {
                for (auto i = size_t{0}; i < t->node->_n_inputs; ++i) {
                    stack.push_back(t->node->_inputs[i]);
                }
            }
        }
        // Intermediate (node-owning) grads hold only this call's delta, in this
        // call's grad dtype: a stale buffer of a different dtype is dropped and
        // rebuilt. Leaf grads accumulate across calls (see accumulate_grad).
        for (const auto t : visited) {
            if (t->node != nullptr) {
                if (t->grad != nullptr && t->grad->dtype != grad_dtype) {
                    t->zero_grad();
                }
                t->ensure_grad(grad_dtype);
                t->grad->set_zero();
            }
        }
        // Seed the root with ones in the assigned grad dtype.
        this->ensure_grad(grad_dtype);
        auto ones = Tensor(std::vector(this->shape, this->shape + this->ndim), grad_dtype, this->device, false);
        ones.set_one();
        this->accumulate_grad(std::move(ones));
        // Kahn's algorithm over consumer counts: a node fires once every
        // consumer of its tensor has delivered a gradient contribution.
        for (const auto t : visited) {
            if (t->node != nullptr) {
                t->node->_n_outputs_pending = 0;
            }
        }
        for (const auto t : visited) {
            if (t->node == nullptr) {
                continue;
            }
            for (auto i = size_t{0}; i < t->node->_n_inputs; ++i) {
                if (
                    const auto* in = t->node->_inputs[i];
                    in->node != nullptr
                    ) {
                    if (in->node->_n_outputs_pending == std::numeric_limits<uint8_t>::max()) {
                        throw ErrorHandling::DimensionTooLarge("backward graph fan-in exceeds 255");
                    }
                    ++in->node->_n_outputs_pending;
                }
            }
        }
        std::vector<Tensor*> ready;
        auto node_count = size_t{0};
        for (const auto t : visited) {
            if (t->node != nullptr) {
                ++node_count;
                if (t->node->_n_outputs_pending == 0) {
                    ready.push_back(t);
                }
            }
        }
        auto fired = size_t{0};
        while (!ready.empty()) {
            const auto* t = ready.back();
            ready.pop_back();
            auto* const n = t->node;
            n->_backward_fn(n, t->grad, n->_inputs, n->_n_inputs, grad_dtype);
            ++fired;
            for (auto i = size_t{0}; i < n->_n_inputs; ++i) {
                if (
                    auto* const in = n->_inputs[i];
                    in->node != nullptr && --in->node->_n_outputs_pending == 0
                    ) {
                    ready.push_back(in);
                }
            }
        }
        if (fired != node_count) [[unlikely]] {
            // The graph is built exclusively by binary/unary ops on fresh
            // outputs, so a cycle means memory corruption, not user error.
            throw ErrorHandling::UnReachableCode();
        }
        for (const auto t : visited) {
            t->sync();
            if (t->grad != nullptr) {
                t->grad->sync();
            }
        }
    }
}
