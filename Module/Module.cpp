/**
 * @file Module.cpp
 * @author GreenBean
 * @brief Implementation of the Module base class registry (named parameters
 * and named child modules), named/manual-gradient parameter access,
 * subtree-wide zero_grad/adam_step, the ostream tree printer, and the
 * mse_loss forwarder to BeanTensor::Loss::mse, the softmax_cross_entropy
 * forwarder to Tensor::softmax_ce, and the subtree-wide gradient-dtype
 * setter (validate-all-then-apply) with its per-parameter getter.
 * @version 0.5
 * @date 2026-10-28
 */
#include "Module/Module.h"

#include <iomanip>
#include <sstream>
#include <string>
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
        for (auto&[name, tensor] : this->parameters) {
            if (name == param_name) {
                return *tensor;
            }
        }
        throw ErrorHandling::NameError("no parameter named '" + param_name + "'");
    }

    const Tensors::Tensor& Module::parameter(const std::string& param_name) const {
        for (const auto&[name, tensor] : this->parameters) {
            if (name == param_name) {
                return *tensor;
            }
        }
        throw ErrorHandling::NameError("no parameter named '" + param_name + "'");
    }

    Tensors::Tensor& Module::grad(const std::string& param_name) {
        return this->parameter(param_name).gradient();
    }

    std::optional<DataTypes::DType> Module::grad_dtype(const std::string& param_name) const {
        return this->parameter(param_name).grad_dtype();
    }

    void Module::set_grad_dtype(const DataTypes::DType dt, const DataTypes::GradNarrowing narrowing) {
        const auto params = this->named_parameters();
        for (const auto&[name, tensor] : params) {
            tensor->validate_grad_dtype(dt, narrowing);
        }
        for (const auto&[name, tensor] : params) {
            tensor->set_grad_dtype(dt, narrowing);
        }
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

    size_t Module::parameter_count() const {
        auto total = size_t{0};
        for (const auto&[_, snd] : this->named_parameters()) {
            total += snd->element_count();
        }
        return total;
    }

    size_t Module::parameter_bytes() const {
        auto total = size_t{0};
        for (const auto&[_, snd] : this->named_parameters()) {
            total += snd->byte_count();
        }
        return total;
    }

    void Module::forget_optimizer_state() const {
        for (const auto& entry : this->named_parameters()) {
            entry.second->forget_adam_state();
        }
    }

    Module::~Module() {
        this->forget_optimizer_state();
    }

    Module& Module::operator=(Module&& other) noexcept {
        if (this != &other) {
            this->forget_optimizer_state();
            this->parameters = std::move(other.parameters);
            this->modules = std::move(other.modules);
            this->forward_cache = std::move(other.forward_cache);
        }
        return *this;
    }

    void Module::zero_grad() {
        for (auto&[name, tensor] : this->parameters) {
            tensor->zero_grad();
        }
        for (auto&[name, module] : this->modules) {
            module->zero_grad();
        }
    }

    void Module::adam_step(const float64_t lr, const float64_t beta1, const float64_t beta2, const float64_t epsilon) const {
        for (const auto&[name, tensor] : this->parameters) {
            if (tensor->has_grad()) {
                tensor->adam_step(lr, beta1, beta2, epsilon);
            }
        }
        for (const auto&[name, module] : this->modules) {
            module->adam_step(lr, beta1, beta2, epsilon);
        }
    }

    bool Module::is_training() const {
        return this->training;
    }

    void Module::train(const bool mode) {
        this->training = mode;
        for (auto&[name, module] : this->modules) {
            module->train(mode);
        }
    }

    void Module::eval() {
        this->train(false);
    }

    std::string Module::print(const size_t indent) const {
        auto os = std::ostringstream();
        // Header at the CURRENT stream position (the caller wrote any line
        // indent / "[name] " prefix); only the body and closing brace indent.
        os << this->name() << " {\n";
        for (const auto&[name, tensor] : this->parameters) {
            write_indent(os, indent + 1);
            os << name << ": " << DataTypes::dtype_to_string(tensor->get_dtype())
               << " " << tensor->shape_to_string()
               << "  " << format_bytes(tensor->byte_count())
               << "  grad " << format_bytes(tensor->grad_byte_count())
               << "  adam " << format_bytes(tensor->adam_state_byte_count()) << "\n";
        }
        for (const auto&[name, module] : this->modules) {
            write_indent(os, indent + 1);
            os << "[" << name << "] ";
            os << module->print(indent + 1);
        }
        write_indent(os, indent);
        os << "}\n";
        if (indent == 0) {
            os << "total: " << format_count(this->parameter_count()) << " parameters\n"
               << "  weights " << format_bytes(this->parameter_bytes())
               << ", grads " << format_bytes(this->parameter_grad_bytes())
               << ", Adam state " << format_bytes(this->optimizer_state_bytes()) << "\n";
        }
        return os.str();
    }

    std::string format_count(const size_t count) {
        const auto digits = std::to_string(count);
        auto out = std::string{};
        for (auto i = size_t{0}; i < digits.size(); ++i) {
            if (
                const auto separator = i > 0 && (digits.size() - i) % 3 == 0;
                separator
            ) {
                out.push_back(',');
            }
            out.push_back(digits[i]);
        }
        return out;
    }

    std::string format_bytes(const size_t bytes) {
        constexpr float64_t kKibi = 1024;
        const auto value = static_cast<float64_t>(bytes);
        auto out = std::ostringstream();
        if (
            const auto plain = value < kKibi;
            plain
        ) {
            out << bytes << " B";
            return out.str();
        }
        out << std::fixed << std::setprecision(1);
        if (
            const auto kb = value < kKibi * kKibi;
            kb
        ) {
            out << value / kKibi << " KB";
        } else if (
            const auto mb = value < kKibi * kKibi * kKibi;
            mb
        ) {
            out << value / (kKibi * kKibi) << " MB";
        } else {
            out << value / (kKibi * kKibi * kKibi) << " GB";
        }
        return out.str();
    }

    size_t Module::parameter_grad_bytes() const {
        auto total = size_t{0};
        for (const auto& entry : this->named_parameters()) {
            total += entry.second->grad_byte_count();
        }
        return total;
    }

    size_t Module::optimizer_state_bytes() const {
        auto total = size_t{0};
        for (const auto& entry : this->named_parameters()) {
            total += entry.second->adam_state_byte_count();
        }
        return total;
    }

    Tensors::Tensor mse_loss(const Tensors::Tensor& pred, const Tensors::Tensor& target) {
        return Loss::mse(pred, target);
    }

    Tensors::Tensor softmax_cross_entropy(const Tensors::Tensor& logits, const Tensors::Tensor& targets) {
        return logits.softmax_ce(targets);
    }

    void Module::add_parameter(std::string param_name, Tensors::Tensor tensor) {
        if (param_name.empty()) {
            throw ErrorHandling::NameError("parameter name is empty");
        }
        for (const auto&[name, _] : this->parameters) {
            if (name == param_name) {
                throw ErrorHandling::NameError("parameter name already taken: '" + param_name + "'");
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
        const auto it = this->modules.begin() + static_cast<std::ptrdiff_t>(pos);
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
        throw ErrorHandling::NameError("no child module named '" + module_name + "'");
    }

    void Module::require_unique_module_name(const std::string& module_name) const {
        for (const auto&[name, module] : this->modules) {
            if (name == module_name) {
                throw ErrorHandling::NameError("child module name already taken: '" + module_name + "'");
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
