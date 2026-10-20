/**
 * @file Backward.cpp
 * @author GreenBean
 * @brief Tensor gradient plumbing: ensure_grad, accumulate_grad, wire_node,
 * zero_grad, and backward() (reverse-mode backpropagation, Kahn's algorithm
 * over consumer counts). accumulate_grad dispatches to the grad_acc
 * cpu/gpu backends.
 * @version 0.1
 * @date 2026-10-20
 */
#include <limits>
#include <unordered_set>
#include <vector>

#include "Math/Autograd/Autograd.h"
#include "Tensor/Tensor.h"

namespace BeanTensor::Tensors {

    /**
     * Allocates grad with this tensor's shape, dtype and device, zero-filled, and syncs
     * so the fill lands before any reader queues work. No-op if grad already exists.
     * Ownership: this tensor owns the heap-allocated grad.
     * @brief Lazily allocate a zeroed gradient tensor.
     */
    void Tensor::ensure_grad() {
        if (this->grad != nullptr) {
            return;
        }
        this->grad = new Tensor(std::vector<size_t>(this->shape, this->shape + this->ndim), this->dtype, this->device, false);
        this->grad->set_zero();
        this->grad->sync(); // the zero-fill must land before any reader queues work
    }

    /**
     * If no grad exists, adopts delta as the grad (the move drains delta's workers).
     * Otherwise syncs delta and the existing grad, then adds delta in place on this
     * tensor's device. delta must match this tensor's shape and dtype.
     * @brief Add a gradient contribution into this tensor's grad.
     * @param delta Gradient contribution; consumed (moved from or left drained).
     * @throws ErrorHandling::UnReachableCode if the device is neither CPU nor GPU.
     */
    void Tensor::accumulate_grad(Tensor&& delta) {
        if (this->grad == nullptr) {
            this->grad = new Tensor(std::move(delta)); // the move drains delta's workers
            return;
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
     * @param backward_fn Backward trampoline invoked as (output grad, inputs, input count).
     * @param a First input tensor.
     * @param b Optional second input tensor; nullptr for a unary op.
     */
    void Tensor::wire_node(void (*backward_fn)(Tensor*, Tensor**, size_t), const Tensor& a, const Tensor* b) {
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
     * Walks the graph from this tensor, zeroes the grads of node-owning (intermediate)
     * tensors so they hold only this call's delta (leaf grads accumulate across calls),
     * seeds this tensor's grad with ones, then fires each node's backward function in
     * topological order (Kahn's algorithm over consumer counts). Syncs every visited
     * tensor and its grad before returning.
     * @brief Reverse-mode backpropagation from this tensor.
     * @throws ErrorHandling::NoGradient if this tensor neither requires grad nor has a node.
     * @throws ErrorHandling::DimensionTooLarge if a tensor's consumer count exceeds 255.
     * @throws ErrorHandling::UnReachableCode if not every node fires (a cycle).
     */
    void Tensor::backward() {
        if (!this->requires_grad && this->node == nullptr) {
            throw ErrorHandling::NoGradient();
        }
        // Discover every tensor reachable through the graph.
        std::vector<Tensor*> visited;
        std::unordered_set<Tensor*> seen;
        std::vector<Tensor*> stack{this};
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
        // Intermediate (node-owning) grads hold only this call's delta; leaf
        // grads accumulate across calls.
        for (const auto t : visited) {
            if (t->node != nullptr) {
                t->ensure_grad();
                t->grad->set_zero();
            }
        }
        // Seed the root with ones.
        this->ensure_grad();
        auto ones = Tensor(std::vector<size_t>(this->shape, this->shape + this->ndim), this->dtype, this->device, false);
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
                auto* const in = t->node->_inputs[i];
                if (in->node != nullptr) {
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
            auto* const t = ready.back();
            ready.pop_back();
            auto* const n = t->node;
            n->_backward_fn(t->grad, n->_inputs, n->_n_inputs);
            ++fired;
            for (auto i = size_t{0}; i < n->_n_inputs; ++i) {
                auto* const in = n->_inputs[i];
                if (in->node != nullptr && --in->node->_n_outputs_pending == 0) {
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
