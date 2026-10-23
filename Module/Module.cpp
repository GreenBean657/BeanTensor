/**
 * @file Module.cpp
 * @author GreenBean
 * @brief Implementation of the Module base class registry (named parameters
 * and named child modules), named/manual-gradient parameter access,
 * subtree-wide zero_grad/adam_step, the ostream tree printer, and the
 * mse_loss forwarder to BeanTensor::Loss::mse.
 * @version 0.1
 * @date 2026-10-23
 */
#include "Module/Module.h"

#include <ostream>
#include <utility>

#include "Math/Loss/Loss.h"

namespace BeanTensor::Modules {
    namespace {
        /**
         * Indent helper: two spaces per nesting level.
         * @param os Stream to pad.
         * @param indent Nesting depth.
         */
        void write_indent(std::ostream& os, const size_t indent) {
            for (auto i = size_t{0}; i < indent; ++i) {
                os << "  ";
            }
        }
    }

    Tensors::Tensor& Module::parameter(const std::string& param_name) {
        for (auto& entry : this->parameters) {
            if (entry.name == param_name) {
                return *entry.tensor;
            }
        }
        throw ErrorHandling::IndexOutOfRange();
    }

    const Tensors::Tensor& Module::parameter(const std::string& param_name) const {
        for (const auto& entry : this->parameters) {
            if (entry.name == param_name) {
                return *entry.tensor;
            }
        }
        throw ErrorHandling::IndexOutOfRange();
    }

    Tensors::Tensor& Module::grad(const std::string& param_name) {
        return this->parameter(param_name).gradient();
    }

    std::vector<std::pair<std::string, Tensors::Tensor*>> Module::named_parameters() {
        auto out = std::vector<std::pair<std::string, Tensors::Tensor*>>();
        this->collect_named_parameters<Tensors::Tensor*>("", out);
        return out;
    }

    std::vector<std::pair<std::string, const Tensors::Tensor*>> Module::named_parameters() const {
        auto out = std::vector<std::pair<std::string, const Tensors::Tensor*>>();
        this->collect_named_parameters<const Tensors::Tensor*>("", out);
        return out;
    }

    void Module::zero_grad() {
        for (auto& entry : this->parameters) {
            entry.tensor->zero_grad();
        }
        for (auto& entry : this->modules) {
            entry.module->zero_grad();
        }
    }

    void Module::adam_step(const float64_t lr, const float64_t beta1, const float64_t beta2, const float64_t epsilon) {
        for (auto& entry : this->parameters) {
            if (entry.tensor->has_grad()) {
                entry.tensor->adam_step(lr, beta1, beta2, epsilon);
            }
        }
        for (auto& entry : this->modules) {
            entry.module->adam_step(lr, beta1, beta2, epsilon);
        }
    }

    void Module::print(std::ostream& os, const size_t indent) const {
        // Header at the CURRENT stream position (the caller wrote any line
        // indent / "[name] " prefix); only the body and closing brace indent.
        os << this->name() << " {\n";
        for (const auto& entry : this->parameters) {
            write_indent(os, indent + 1);
            os << entry.name << ": " << DataTypes::dtype_to_string(entry.tensor->get_dtype())
               << " " << entry.tensor->shape_to_string() << "\n";
        }
        for (const auto& entry : this->modules) {
            write_indent(os, indent + 1);
            os << "[" << entry.name << "] ";
            entry.module->print(os, indent + 1);
        }
        write_indent(os, indent);
        os << "}\n";
    }

    std::ostream& operator<<(std::ostream& os, const Module& module) {
        module.print(os, 0);
        return os;
    }

    Tensors::Tensor mse_loss(const Tensors::Tensor& pred, const Tensors::Tensor& target) {
        return Loss::mse(pred, target);
    }

    void Module::add_parameter(std::string param_name, Tensors::Tensor tensor) {
        if (param_name.empty()) {
            throw ErrorHandling::IndexOutOfRange();
        }
        for (const auto& entry : this->parameters) {
            if (entry.name == param_name) {
                throw ErrorHandling::IndexOutOfRange();
            }
        }
        auto slot = std::make_unique<Tensors::Tensor>(std::move(tensor));
        this->parameters.push_back(ParameterEntry{std::move(param_name), std::move(slot)});
    }

    void Module::add_module(std::string module_name, std::unique_ptr<Module> module) {
        if (module_name.empty()) {
            module_name = std::to_string(this->modules.size());
        }
        this->require_unique_module_name(module_name);
        this->modules.push_back(ModuleEntry{std::move(module_name), std::move(module)});
    }

    void Module::insert_module(const size_t pos, std::string module_name, std::unique_ptr<Module> module) {
        if (pos > this->modules.size()) {
            throw ErrorHandling::IndexOutOfRange();
        }
        if (module_name.empty()) {
            module_name = std::to_string(this->modules.size());
        }
        this->require_unique_module_name(module_name);
        this->modules.insert(this->modules.begin() + static_cast<std::ptrdiff_t>(pos),
                             ModuleEntry{std::move(module_name), std::move(module)});
    }

    std::unique_ptr<Module> Module::replace_module(const size_t pos, std::unique_ptr<Module> module) {
        if (pos >= this->modules.size()) {
            throw ErrorHandling::IndexOutOfRange();
        }
        auto& slot = this->modules[pos];
        auto displaced = std::move(slot.module);
        slot.module = std::move(module);
        return displaced;
    }

    std::unique_ptr<Module> Module::remove_module(const size_t pos) {
        if (pos >= this->modules.size()) {
            throw ErrorHandling::IndexOutOfRange();
        }
        auto it = this->modules.begin() + static_cast<std::ptrdiff_t>(pos);
        auto removed = std::move(it->module);
        this->modules.erase(it);
        return removed;
    }

    const std::string& Module::module_name_at(const size_t pos) const {
        if (pos >= this->modules.size()) {
            throw ErrorHandling::IndexOutOfRange();
        }
        return this->modules[pos].name;
    }

    size_t Module::module_index_of(const std::string& module_name) const {
        for (auto i = size_t{0}; i < this->modules.size(); ++i) {
            if (this->modules[i].name == module_name) {
                return i;
            }
        }
        throw ErrorHandling::IndexOutOfRange();
    }

    void Module::require_unique_module_name(const std::string& module_name) const {
        for (const auto& entry : this->modules) {
            if (entry.name == module_name) {
                throw ErrorHandling::IndexOutOfRange();
            }
        }
    }

    void Module::clear_forward_cache() {
        this->forward_cache.clear();
    }

    Tensors::Tensor& Module::hold_intermediate(Tensors::Tensor t) {
        auto slot = std::make_unique<Tensors::Tensor>(std::move(t));
        auto& ref = *slot;
        this->forward_cache.push_back(std::move(slot));
        return ref;
    }
}
