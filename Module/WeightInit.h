/**
 * @file WeightInit.h
 * @author GreenBean
 * @brief Weight initialization modes for trainable layers. He (Kaiming)
 * uniform on the fan-in is the historical default (ReLU-MLP networks);
 * Xavier (Glorot) uniform on fan-in + fan_out is the counterpart for
 * symmetric/tanh/linear activations. The spec-string forms ("he"/"xavier")
 * round-trip through the .bt architecture hyperparameters.
 * @version 0.1
 * @date 2026-11-26
 */
#pragma once

#include <string>

#include "Helpers/Errors.h"

namespace BeanTensor::Modules {

    /**
     * @brief Weight initialization mode.
     */
    enum class WeightInit {
        He,     ///< He (Kaiming) uniform: U[-b, b], b = sqrt(6 / fan_in).
        Xavier, ///< Xavier (Glorot) uniform: U[-b, b], b = sqrt(6 / (fan_in + fan_out)).
    };

    /**
     * @brief Spec-string form of a weight init mode ("he" / "xavier").
     */
    [[nodiscard]] inline const char* weight_init_name(const WeightInit init) {
        switch (init) {
            case WeightInit::He: return "he";
            case WeightInit::Xavier: return "xavier";
        }
        throw ErrorHandling::UnReachableCode();
    }

    /**
     * @brief Parse the spec-string form of a weight init mode.
     * @throws ErrorHandling::CorruptFile the name is not a known mode.
     */
    [[nodiscard]] inline WeightInit weight_init_from_name(const std::string& name) {
        if (name == "he") {
            return WeightInit::He;
        }
        if (name == "xavier") {
            return WeightInit::Xavier;
        }
        throw ErrorHandling::CorruptFile("unknown weight init mode '" + name + "'");
    }
}
