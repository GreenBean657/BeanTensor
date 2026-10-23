/**
 * @file Sequential.h
 * @author GreenBean
 * @brief Ordered container module: an append/insert/replace/remove registry
 * of named child modules whose forward chains outputs through in order.
 * @version 0.1
 * @date 2026-10-23
 */
#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "Module/Module.h"

namespace BeanTensor::Modules {

    /**
     * Chains children in registry order: out = last(...second(first(x))).
     * Children are registered under names; an empty name auto-registers as
     * the next index ("0", "1", ...), so `Sequential` + `append` reads like
     * an ordered list. Injection mutates both the registry and forward
     * behavior immediately — a replaced slot keeps its registry name, only
     * the module behind it changes.
     * @brief Ordered module chain (container).
     */
    class Sequential : public Module {
    public:
        Sequential() = default;
        // Unique ownership of children: copying a container would orphan the
        // registry's raw Tensor pointers (Adam keys on parameter addresses).
        Sequential(const Sequential&) = delete;
        Sequential& operator=(const Sequential&) = delete;
        Sequential(Sequential&&) = default;
        Sequential& operator=(Sequential&&) = default;
        ~Sequential() override = default;

        /**
         * Feed x through every child in registry order. An empty Sequential
         * has no layers to forward through and throws
         * ErrorHandling::NotImplemented.
         * @brief Chain outputs through all children.
         * @param x Input tensor.
         * @return The last child's output.
         * @throws ErrorHandling::NotImplemented The registry is empty.
         */
        [[nodiscard]] Tensors::Tensor forward(const Tensors::Tensor& x) override;

        /**
         * @brief Human-readable layer type.
         * @return "Sequential".
         */
        [[nodiscard]] std::string name() const override;

        /**
         * Append a child at the end of the registry.
         * @brief Append a named child module.
         * @param module_name Registry name; empty auto-names to the next index.
         * @param module The child module (adopted).
         * @return This container, for chaining.
         * @throws ErrorHandling::IndexOutOfRange The name is already taken.
         */
        Sequential& append(std::string module_name, std::unique_ptr<Module> module);

        /**
         * Insert a child before the slot currently at pos.
         * @brief Insert a named child module at a position.
         * @param pos Insertion position; must be <= size().
         * @param module_name Registry name; empty auto-names to the next index.
         * @param module The child module (adopted).
         * @return This container, for chaining.
         * @throws ErrorHandling::IndexOutOfRange pos out of range, or the name is already taken.
         */
        Sequential& insert(size_t pos, std::string module_name, std::unique_ptr<Module> module);

        /**
         * Swap the module behind a registry slot; the slot's name and
         * position (and therefore every later forward) change meaning, the
         * identity of the slot does not.
         * @brief Replace the child module at a position, keeping its name.
         * @param pos Position of the slot; must be < size().
         * @param module The replacement module (adopted).
         * @return The displaced module, for reuse.
         * @throws ErrorHandling::IndexOutOfRange pos out of range.
         */
        [[nodiscard]] std::unique_ptr<Module> replace(size_t pos, std::unique_ptr<Module> module);

        /**
         * Remove the child at a registry position.
         * @brief Remove a child module.
         * @param pos Position of the slot; must be < size().
         * @return The removed module, for reuse.
         * @throws ErrorHandling::IndexOutOfRange pos out of range.
         */
        [[nodiscard]] std::unique_ptr<Module> remove(size_t pos);

        /**
         * @brief Child count.
         * @return The registry size.
         */
        [[nodiscard]] size_t size() const;
        /**
         * @brief Whether the registry is empty.
         * @return size() == 0.
         */
        [[nodiscard]] bool empty() const;

        /**
         * @brief Child at a registry position.
         * @param pos Position; must be < size().
         * @return The child module.
         * @throws ErrorHandling::IndexOutOfRange pos out of range.
         */
        [[nodiscard]] Module& operator[](size_t pos);
        /**
         * const overload of operator[].
         * @brief Child at a registry position (const).
         * @param pos Position; must be < size().
         * @return The child module.
         * @throws ErrorHandling::IndexOutOfRange pos out of range.
         */
        [[nodiscard]] const Module& operator[](size_t pos) const;

        /**
         * @brief Child registered under a name.
         * @param module_name Registry name to look up.
         * @return The child module.
         * @throws ErrorHandling::IndexOutOfRange No child holds that name.
         */
        [[nodiscard]] Module& at(const std::string& module_name);

        /**
         * @brief Position of the child registered under a name.
         * @param module_name Registry name to look up.
         * @return The child's registry position.
         * @throws ErrorHandling::IndexOutOfRange No child holds that name.
         */
        [[nodiscard]] size_t index_of(const std::string& module_name) const;

        /**
         * @brief Registry name of the child at a position.
         * @param pos Position; must be < size().
         * @return The child's registry name.
         * @throws ErrorHandling::IndexOutOfRange pos out of range.
         */
        [[nodiscard]] const std::string& name_at(size_t pos) const;
    };
}
