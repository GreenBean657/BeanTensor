/**
 * @file ModuleStream.h
 * @author GreenBean
 * @brief C++-only stream output convenience for Module. This header is
 * deliberately separate from Module.h so that the public module API stays
 * bindable: python wrappers call Module::print() directly.
 * @version 0.1
 * @date 2026-10-27
 */
#pragma once

#include <ostream>

#include "Module/Module.h"

namespace BeanTensor::Modules {

    /**
     * Stream the whole module tree via Module::print at indent 0.
     * C++-only convenience, deliberately NOT in the bindable public API
     * checked by bind_clean_public_api.sh; never bound to Python.
     * @brief Human-readable module tree to a stream.
     * @param os Destination stream.
     * @param module Module to print.
     * @return The stream, for chaining.
     */
    inline std::ostream& operator<<(std::ostream& os, const Module& module) {
        return os << module.print();
    }

}
