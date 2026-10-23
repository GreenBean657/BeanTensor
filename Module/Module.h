/**
 * @file Module.h
 * @author GreenBean
 * @brief Abstract Module base class: an ordered registry of named parameters
 * and named child modules, a virtual forward(Tensor) -> Tensor, named
 * parameter access with manual gradient access (each parameter is a plain
 * Tensors::Tensor whose accumulated gradient lives in Tensor::grad), layer
 * injection helpers for container subclasses, Adam/zero_grad training
 * conveniences over the whole tree, and a human-readable std::ostream tree
 * printer (terminal only — serialization to .bt files comes later). Also
 * declares mse_loss, the module-facing entry point to the MSE loss
 * (Math/Loss/Loss.h) for forward/backward training loops.
 * @version 0.1
 * @date 2026-10-23
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "Tensor/Tensor.h"

namespace BeanTensor::Modules {

    /**
     * Base class of every neural-network layer. A Module owns zero or more
     * named parameters (Linear's weight/bias) and zero or more named child
     * modules (Sequential's layers), both kept in insertion order.
     * Parameters live in std::deque slots so their Tensor addresses are
     * stable for the module's whole lifetime — Tensor::adam_step keys its
     * moment state on the parameter address, so a module's parameters must
     * never be reallocated (register them once in the constructor).
     * Graph lifetime: forward() outputs join the computation graph through
     * the module's parameters (and the input tensor), which are borrowed raw
     * pointers internally — every tensor participating in a forward,
     * including the input and every parameter of every layer on the path,
     * must stay alive until backward() returns. In a training loop that
     * means keeping the input/target as named locals outside the step body
     * and destroying neither the module nor its parameters mid-loop.
     * Forward intermediates: because the graph stores raw pointers, a
     * module's own forward intermediates (Linear's matmul output feeding
     * bias_add, Sequential's stage outputs feeding the next layer) are kept
     * alive by the module itself, in its forward_cache: hold_intermediate()
     * moves them into stable heap slots, and clear_forward_cache() (called
     * at the start of every forward() implementation) drops the PREVIOUS
     * pass's chain. The contract this creates: backward() a module's output
     * BEFORE calling forward() on that module again — the natural training
     * loop order (forward -> loss -> backward -> step -> zero_grad).
     * @brief Abstract module: named parameter/layer registry + virtual forward.
     */
    class Module {
    public:
        virtual ~Module() = default;

        /**
         * Run the module. Implementations chain tensor ops
         * (matmul/bias_add/relu/...); when any participating tensor has
         * requires_grad the output joins the computation graph and can be
         * backward()'d into the module's parameters.
         * @brief Compute the module's output for one input.
         * @param x Input tensor.
         * @return Output tensor.
         */
        [[nodiscard]] virtual Tensors::Tensor forward(const Tensors::Tensor& x) = 0;

        /**
         * @brief Human-readable layer type for printing.
         * @return The class name, e.g. "Linear" or "Sequential".
         */
        [[nodiscard]] virtual std::string name() const = 0;

        /**
         * Look up one of THIS module's own registered parameters by its
         * registration name ("weight", "bias", ...). Child-module parameters
         * are reached through named_parameters() (qualified names) or by
         * descending into the child.
         * @brief Direct parameter access by registration name.
         * @param param_name Registration name given at add_parameter time.
         * @return The parameter tensor.
         * @throws ErrorHandling::IndexOutOfRange No own parameter holds that name.
         */
        [[nodiscard]] Tensors::Tensor& parameter(const std::string& param_name);
        /**
         * const overload: read-only view of the parameter.
         * @brief Direct parameter access by registration name (const).
         * @param param_name Registration name given at add_parameter time.
         * @return The parameter tensor.
         * @throws ErrorHandling::IndexOutOfRange No own parameter holds that name.
         */
        [[nodiscard]] const Tensors::Tensor& parameter(const std::string& param_name) const;

        /**
         * Manual gradient access: the accumulated gradient of the named own
         * parameter, i.e. parameter(name).gradient(). Only meaningful after a
         * backward() whose graph passed through that parameter.
         * @brief Read the accumulated gradient of a named own parameter.
         * @param param_name Registration name of the parameter.
         * @return The parameter's gradient tensor.
         * @throws ErrorHandling::IndexOutOfRange No own parameter holds that name.
         * @throws ErrorHandling::NoGradient The parameter has no accumulated gradient yet.
         */
        [[nodiscard]] Tensors::Tensor& grad(const std::string& param_name);

        /**
         * Flattened, ordered view of every parameter in the subtree:
         * this module's own parameters first, then each child's, recursively.
         * Names are qualified with the child names on the path
         * ("0.weight", "0.bias", "1.weight"), so a Sequential can be trained
         * without descending its registry by hand. The returned pointers are
         * owned by the module tree and live as long as it does.
         * @brief Ordered qualified parameter list of the whole subtree.
         * @return (qualified name, parameter) pairs in registry order.
         */
        [[nodiscard]] std::vector<std::pair<std::string, Tensors::Tensor*>> named_parameters();
        /**
         * const overload of named_parameters.
         * @brief Ordered qualified parameter list of the whole subtree (const).
         * @return (qualified name, const parameter) pairs in registry order.
         */
        [[nodiscard]] std::vector<std::pair<std::string, const Tensors::Tensor*>> named_parameters() const;

        /**
         * Drop every accumulated parameter gradient in the subtree
         * (Tensor::zero_grad on each named parameter).
         * @brief Clear all parameter gradients.
         */
        void zero_grad();

        /**
         * One Adam step over the whole subtree: Tensor::adam_step(lr, beta1,
         * beta2, epsilon) on every parameter that currently holds an
         * accumulated gradient; parameters without one are skipped (e.g. a
         * layer the loss did not reach). Adam's moment state lives in an
         * external map keyed by parameter address, which is why parameters
         * are deque-allocated and must not move.
         * @brief Adam update of every parameter with an accumulated gradient.
         * @param lr Learning rate.
         * @param beta1 First-moment decay.
         * @param beta2 Second-moment decay.
         * @param epsilon Denominator epsilon.
         */
        void adam_step(float64_t lr, float64_t beta1, float64_t beta2, float64_t epsilon);

        /**
         * Human-readable tree dump, one line per parameter/child, children
         * printed recursively under "[name] <Type> {" headers. Terminal
         * output only — no serialization contract.
         * @brief Print the module tree (names, layer types, shapes).
         * @param os Stream to write to.
         * @param indent Nesting depth in steps of two spaces (0 at top level).
         */
        virtual void print(std::ostream& os, size_t indent) const;

    protected:
        /**
         * Register one own parameter. The tensor is heap-allocated in a
         * deque slot; its address is stable for the module's lifetime.
         * @brief Add a named parameter to this module's registry.
         * @param param_name Registration name; must be unique among this module's parameters.
         * @param tensor The parameter tensor (moved into the registry).
         * @throws ErrorHandling::IndexOutOfRange The name is empty or already taken.
         */
        void add_parameter(std::string param_name, Tensors::Tensor tensor);

        /**
         * Append a named child module to this module's registry.
         * @brief Add a named child module (append).
         * @param module_name Registry name; must be unique among this module's children. Empty auto-names to the next index.
         * @param module The child module (adopted).
         * @throws ErrorHandling::IndexOutOfRange The name is already taken.
         */
        void add_module(std::string module_name, std::unique_ptr<Module> module);
        /**
         * Insert a named child module at a registry position.
         * @brief Add a named child module (insert).
         * @param pos Insertion position; must be <= the child count.
         * @param module_name Registry name; must be unique among this module's children. Empty auto-names to the next index.
         * @param module The child module (adopted).
         * @throws ErrorHandling::IndexOutOfRange pos out of range, or the name is already taken.
         */
        void insert_module(size_t pos, std::string module_name, std::unique_ptr<Module> module);
        /**
         * Swap the child module at a registry position for another one; the
         * existing registry name is kept (the slot is the identity, not the
         * module). The displaced module is returned for reuse.
         * @brief Replace the child module at a position, keeping its name.
         * @param pos Position of the slot to replace; must be < the child count.
         * @param module The replacement module (adopted).
         * @return The displaced module.
         * @throws ErrorHandling::IndexOutOfRange pos out of range.
         */
        [[nodiscard]] std::unique_ptr<Module> replace_module(size_t pos, std::unique_ptr<Module> module);
        /**
         * Remove the child module at a registry position and return it.
         * @brief Remove a child module.
         * @param pos Position of the slot to remove; must be < the child count.
         * @return The removed module.
         * @throws ErrorHandling::IndexOutOfRange pos out of range.
         */
        [[nodiscard]] std::unique_ptr<Module> remove_module(size_t pos);

        /**
         * Drop every intermediate held from the previous forward() pass.
         * Every forward() implementation must call this FIRST, before
         * recording new intermediates — the previous pass's graph must
         * already have been backward()'d by then (see the class doc).
         * @brief Clear the per-forward intermediate cache.
         */
        void clear_forward_cache();
        /**
         * Move a forward intermediate into a stable heap slot owned by this
         * module, keeping it alive for the current pass's backward(). Use it
         * for any tensor produced inside forward() that a LATER op in the
         * same forward() references (the raw graph pointers would otherwise
         * dangle when the stack temporary dies). The returned reference's
         * address is stable until the next clear_forward_cache().
         * @brief Keep a forward intermediate alive until the next forward().
         * @param t The intermediate tensor (moved into the cache).
         * @return Reference to the cached tensor (stable address).
         */
        Tensors::Tensor& hold_intermediate(Tensors::Tensor t);

        /**
         * Look up a child's registry name by position (used by containers).
         * @brief Name of the child at a registry position.
         * @param pos Position; must be < the child count.
         * @return The child's registry name.
         * @throws ErrorHandling::IndexOutOfRange pos out of range.
         */
        [[nodiscard]] const std::string& module_name_at(size_t pos) const;
        /**
         * Position of the child holding a registry name.
         * @brief Index of the child registered under a name.
         * @param module_name Name to look up.
         * @return The child's registry position.
         * @throws ErrorHandling::IndexOutOfRange No child holds that name.
         */
        [[nodiscard]] size_t module_index_of(const std::string& module_name) const;

        struct ParameterEntry {
            std::string name;
            std::unique_ptr<Tensors::Tensor> tensor;
        };
        struct ModuleEntry {
            std::string name;
            std::unique_ptr<Module> module;
        };
        std::deque<ParameterEntry> parameters;
        std::deque<ModuleEntry> modules;
        /**
         * Forward-pass intermediates of the CURRENT pass only. Held so the
         * raw graph parent pointers a module's forward creates stay valid
         * until backward(); wiped by clear_forward_cache() at the start of
         * the next forward() (deque of unique_ptr: slot addresses — and
         * therefore held tensor addresses — are stable).
         */
        std::deque<std::unique_ptr<Tensors::Tensor>> forward_cache;

        /**
         * Recursive worker behind named_parameters(): appends this module's
         * own parameters under prefix, then descends into each child under
         * prefix + child name + ".". unique_ptr::get() on a const deque slot
         * still yields a mutable Tensors::Tensor*, so the one template serves
         * both the mutable and the const overload.
         * @tparam Ptr Tensors::Tensor* or const Tensors::Tensor*.
         * @param prefix Qualified-name prefix ("" at the root).
         * @param out Accumulating (name, parameter) list.
         */
        template<typename Ptr>
        void collect_named_parameters(const std::string& prefix,
                                      std::vector<std::pair<std::string, Ptr>>& out) const {
            for (const auto& entry : this->parameters) {
                out.emplace_back(prefix + entry.name, entry.tensor.get());
            }
            for (const auto& entry : this->modules) {
                entry.module->collect_named_parameters<Ptr>(prefix + entry.name + ".", out);
            }
        }

        /**
         * @brief Reject a child name that is already registered.
         * @param module_name Name to check.
         * @throws ErrorHandling::IndexOutOfRange The name is taken.
         */
        void require_unique_module_name(const std::string& module_name) const;
    };

    /**
     * Stream the whole module tree via Module::print at indent 0.
     * @brief Human-readable module tree to a stream.
     * @param os Destination stream.
     * @param module Module to print.
     * @return The stream, for chaining.
     */
    std::ostream& operator<<(std::ostream& os, const Module& module);

    /**
     * Module-facing loss entry point for training loops:
     * loss = mse_loss(net.forward(x), target); loss.backward(grad_dtype);
     * net.adam_step(...). Thin forwarder to the MSE loss free function
     * BeanTensor::Loss::mse (Math/Loss/Loss.h) — mean of (pred - target)^2
     * over all elements, pred is the only operand that can receive a
     * gradient.
     * @brief Mean squared error loss for module training loops.
     * @param pred Module output; the only operand that can receive a gradient.
     * @param target Targets; identical shape/dtype/device as pred.
     * @return Shape-{1} tensor holding the mean loss, in pred's dtype/device.
     */
    [[nodiscard]] Tensors::Tensor mse_loss(const Tensors::Tensor& pred, const Tensors::Tensor& target);
}
