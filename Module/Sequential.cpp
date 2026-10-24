/**
 * @file Sequential.cpp
 * @author GreenBean
 * @brief Sequential container implementation: the chained forward and the
 * append/insert/replace/remove registry wrappers.
 * @version 0.1
 * @date 2026-10-23
 */
#include "Module/Sequential.h"

#include <utility>

#include "Tensor/Tensor.h"

namespace BeanTensor::Modules {

    Tensors::Tensor Sequential::forward(const Tensors::Tensor& x) {
        if (this->empty()) {
            throw ErrorHandling::EmptyModule();
        }
        // Every stage output is referenced by the NEXT stage's graph node, so
        // each one (except the last, which is returned) is held in this
        // container's forward cache until the next forward() call.
        this->clear_forward_cache();
        auto it = this->modules.begin();
        auto out = std::make_unique<Tensors::Tensor>(it->module->forward(x));
        ++it;
        for (; it != this->modules.end(); ++it) {
            auto& held = this->hold_intermediate(std::move(*out));
            out = std::make_unique<Tensors::Tensor>(it->module->forward(held));
        }
        return std::move(*out);
    }

    std::string Sequential::name() const {
        return "Sequential";
    }

    Sequential& Sequential::append(std::string module_name, std::unique_ptr<Module> module) {
        this->add_module(std::move(module_name), std::move(module));
        return *this;
    }

    Sequential& Sequential::insert(const size_t pos, std::string module_name, std::unique_ptr<Module> module) {
        this->insert_module(pos, std::move(module_name), std::move(module));
        return *this;
    }

    std::unique_ptr<Module> Sequential::replace(const size_t pos, std::unique_ptr<Module> module) {
        return this->replace_module(pos, std::move(module));
    }

    std::unique_ptr<Module> Sequential::remove(const size_t pos) {
        return this->remove_module(pos);
    }

    size_t Sequential::size() const {
        return this->modules.size();
    }

    bool Sequential::empty() const {
        return this->modules.empty();
    }

    Module& Sequential::operator[](const size_t pos) {
        if (pos >= this->modules.size()) {
            throw ErrorHandling::IndexOutOfRange();
        }
        return *this->modules[pos].module;
    }

    const Module& Sequential::operator[](const size_t pos) const {
        if (pos >= this->modules.size()) {
            throw ErrorHandling::IndexOutOfRange();
        }
        return *this->modules[pos].module;
    }

    Module& Sequential::at(const std::string& module_name) const {
        return *this->modules[this->module_index_of(module_name)].module;
    }

    size_t Sequential::index_of(const std::string& module_name) const {
        return this->module_index_of(module_name);
    }

    const std::string& Sequential::name_at(const size_t pos) const {
        return this->module_name_at(pos);
    }
}
