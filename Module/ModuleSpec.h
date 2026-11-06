/**
 * @file ModuleSpec.h
 * @author GreenBean
 * @brief The .bt architecture description: a tree mirroring a Module tree,
 * carrying each layer's type name, its constructor hyperparameters as
 * key/value strings, and its named children. Parameter and state values are
 * NOT in the spec — they travel in the file's tensor records, written and
 * re-read in registry traversal order, with dtype/shape verified against the
 * reconstructed architecture. Hyperparameters are strings so the spec stays
 * trivially serializable; each layer's factory parses and validates its own
 * keys.
 * @version 0.2
 * @date 2026-11-05
 */
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "Tensor/DTypes.h"

namespace BeanTensor::Modules {

    /**
     * Architecture description of one module node, recursively.
     * @brief One node of a module's .bt architecture spec.
     */
    struct ModuleSpec {
        /**
         * The layer's factory key, equal to the class's name()
         * ("Linear", "Conv2d", "Sequential", ...).
         */
        std::string type;
        /**
         * Constructor hyperparameters as ordered key/value string pairs
         * (e.g. Linear: in_features/out_features/bias). Each layer's factory
         * owns the key set and the parsing.
         */
        std::vector<std::pair<std::string, std::string>> hyperparameters;
        /**
         * Named child modules in registry order; empty for leaf layers.
         */
        std::vector<std::pair<std::string, ModuleSpec>> children;
    };

    /**
     * Finds a hyperparameter by key.
     * @brief Look up a hyperparameter value.
     * @return Pointer to the value, or nullptr when the key is absent.
     */
    [[nodiscard]] const std::string* find_hyper(const ModuleSpec&, const std::string&);

    /**
     * Parses an unsigned hyperparameter (full string must consume).
     * @brief Required unsigned hyperparameter.
     * @return The parsed value.
     * @throws ErrorHandling::CorruptFile the key is absent, not numeric, or overflows.
     */
    [[nodiscard]] uint64_t require_hyper_u64(const ModuleSpec&, const std::string&);

    /**
     * Parses a floating-point hyperparameter (full string must consume).
     * @brief Required floating-point hyperparameter.
     * @return The parsed value.
     * @throws ErrorHandling::CorruptFile the key is absent or not numeric.
     */
    [[nodiscard]] float64_t require_hyper_f64(const ModuleSpec&, const std::string&);

    /**
     * Parses a boolean hyperparameter stored as "0"/"1".
     * @brief Required boolean hyperparameter.
     * @return The parsed value.
     * @throws ErrorHandling::CorruptFile the key is absent or not "0"/"1".
     */
    [[nodiscard]] bool require_hyper_bool(const ModuleSpec&, const std::string&);

    /**
     * Formats a floating-point hyperparameter for a .bt spec: hexadecimal
     * float (std::to_chars, chars_format::hex), which is roundtrip-bit-exact
     * and locale-independent, unlike std::to_string's six decimal digits.
     * @brief Format a float64_t hyperparameter losslessly.
     * @return The spec string (parses back to the identical bits).
     */
    [[nodiscard]] std::string format_hyper_f64(float64_t);
}
