#include <cstdint>
#include <exception>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Data/DataLoader.h"
#include "Metrics/Metrics.h"
#include "Module/Gelu.h"
#include "Module/Linear.h"
#include "Module/Module.h"
#include "Module/Relu.h"
#include "Module/Sequential.h"
#include "Random/Generator.h"
#include "Serialization/Serialization.h"
#include "Tensor/DTypes.h"
#include "Tensor/Tensor.h"
#include "demos/Mnist.h"

namespace py = pybind11;

using BeanTensor::Data::DataLoader;
using BeanTensor::DataTypes::DType;
using BeanTensor::float32_t;
using BeanTensor::float64_t;
using BeanTensor::Modules::Gelu;
using BeanTensor::Modules::Linear;
using BeanTensor::Modules::Module;
using BeanTensor::Modules::Relu;
using BeanTensor::Modules::Sequential;
using BeanTensor::Serialization::CheckpointLevel;
using BeanTensor::Tensors::Device;
using BeanTensor::Tensors::Tensor;

namespace {

    /**
     * @brief Raise a Python RuntimeError; pybind11 ships no py::runtime_error,
     * so this is the documented idiom for the generic Python error type.
     * @param msg Error message, copied into the Python exception.
     */
    [[noreturn]] void raise_runtime_error(const char* msg) {
        PyErr_SetString(PyExc_RuntimeError, msg);
        throw py::error_already_set();
    }

    /**
     * @brief Map a NumPy dtype onto the engine dtype for the four dtypes the
     * buffer exchange supports.
     * @param dt NumPy dtype of the source array.
     * @return Matching engine dtype.
     * @throws py::value_error The dtype is not float32/float64/int32/int64.
     */
    [[nodiscard]] DType dtype_from_numpy(const py::dtype& dt) {
        if (dt.is(py::dtype::of<float32_t>())) {
            return DType::Float32;
        }
        if (dt.is(py::dtype::of<float64_t>())) {
            return DType::Float64;
        }
        if (dt.is(py::dtype::of<int32_t>())) {
            return DType::Int32;
        }
        if (dt.is(py::dtype::of<int64_t>())) {
            return DType::Int64;
        }
        throw py::value_error("from_numpy: unsupported dtype (supported: float32, float64, int32, int64)");
    }

    /**
     * @brief Copy a NumPy array into a new CPU tensor, one bulk host write.
     * @param arr Source array; must be C-contiguous with a supported dtype.
     * @return Owning CPU tensor with the array's shape and dtype.
     * @throws py::value_error The array is not C-contiguous or the dtype is unsupported.
     */
    [[nodiscard]] Tensor from_numpy(py::array arr) {
        const DType dt = dtype_from_numpy(arr.dtype());
        if ((arr.flags() & py::array::c_style) == 0) {
            throw py::value_error("from_numpy: array must be C-contiguous");
        }
        const py::buffer_info info = arr.request();
        std::vector<size_t> shape(info.shape.begin(), info.shape.end());
        Tensor t(std::move(shape), dt, Device::CPU);
        const size_t total = static_cast<size_t>(info.size) * static_cast<size_t>(info.itemsize);
        t.write_element(0, info.ptr, total);
        return t;
    }

    /**
     * @brief Copy one element out to a freshly allocated NumPy array of the
     * matching dtype, so the result survives the tensor's destruction.
     * @tparam T Element type matching the tensor's dtype.
     * @param t Synced CPU tensor.
     * @return New 1-element NumPy array; the copy owns its buffer.
     */
    template <typename T>
    [[nodiscard]] py::array to_numpy_typed(const Tensor& t) {
        std::vector<ssize_t> dims(t.rank());
        for (size_t i = 0; i < t.rank(); ++i) {
            dims[i] = static_cast<ssize_t>(t.dim(i));
        }
        py::array_t<T> out(dims);
        t.read_element(0, out.mutable_data(), static_cast<size_t>(out.size()) * sizeof(T));
        return out;
    }

    /**
     * @brief Sync a CPU tensor and copy its elements out to a new NumPy array.
     * @param t Source tensor.
     * @return New NumPy array owning a copy of the tensor's data.
     * @throws RuntimeError The tensor is on GPU or holds an unsupported dtype.
     * @note The bulk read copies the tensor's base buffer. Every tensor the
     *       current binding surface can produce is freshly allocated and
     *       contiguous; engine-side strided views would need per-element
     *       readback once view-producing ops are bound.
     */
    [[nodiscard]] py::array to_numpy(const Tensor& t) {
        if (t.get_device() != Device::CPU) {
            raise_runtime_error("to_numpy: tensor lives on GPU; move it to CPU first");
        }
        t.sync();
        switch (t.get_dtype()) {
            case DType::Float32:
                return to_numpy_typed<float32_t>(t);
            case DType::Float64:
                return to_numpy_typed<float64_t>(t);
            case DType::Int32:
                return to_numpy_typed<int32_t>(t);
            case DType::Int64:
                return to_numpy_typed<int64_t>(t);
            default:
                raise_runtime_error("to_numpy: unsupported dtype (supported: float32, float64, int32, int64)");
        }
    }

    /**
     * @brief Read the single element of a scalar tensor as a Python float.
     * @param t Source tensor.
     * @return The element's value widened to float64.
     * @throws py::value_error The tensor does not hold exactly one element.
     * @throws RuntimeError The dtype is unsupported for scalar readback.
     */
    [[nodiscard]] float64_t item(const Tensor& t) {
        t.sync();
        if (t.element_count() != 1) {
            throw py::value_error("item(): tensor must have exactly one element");
        }
        switch (t.get_dtype()) {
            case DType::Float32: {
                float32_t v;
                t.read_element(0, &v, sizeof(v));
                return static_cast<float64_t>(v);
            }
            case DType::Float64: {
                float64_t v;
                t.read_element(0, &v, sizeof(v));
                return v;
            }
            case DType::Int32: {
                int32_t v;
                t.read_element(0, &v, sizeof(v));
                return static_cast<float64_t>(v);
            }
            case DType::Int64: {
                int64_t v;
                t.read_element(0, &v, sizeof(v));
                return static_cast<float64_t>(v);
            }
            default:
                raise_runtime_error("item(): unsupported dtype (supported: float32, float64, int32, int64)");
        }
    }

    /**
     * @brief Pack the tensor's dimensions into a Python tuple.
     * @param t Source tensor.
     * @return Shape as a tuple of ints, e.g. (2, 3, 4).
     */
    [[nodiscard]] py::tuple shape_tuple(const Tensor& t) {
        py::tuple out(t.rank());
        for (size_t i = 0; i < t.rank(); ++i) {
            out[i] = py::int_(t.dim(i));
        }
        return out;
    }

    /**
     * @brief Flatten the module's subtree parameters into a Python dict.
     * @param self Python handle of the module (used as the keep-alive parent).
     * @return dict mapping qualified names ("0.weight", "2.bias", ...) to
     *         Tensor handles. Each handle is a fresh Python object sharing
     *         the C++ parameter owned by the module — writes through it (e.g.
     *         set_zero) mutate the module's parameter — and each keeps the
     *         module alive via reference_internal.
     */
    [[nodiscard]] py::dict named_parameters_dict(const py::object& self) {
        Module& mod = self.cast<Module&>();
        py::dict out;
        for (auto& [name, tensor] : mod.named_parameters()) {
            out[name.c_str()] = py::cast(tensor, py::return_value_policy::reference_internal, self);
        }
        return out;
    }

    /**
     * The one-step container form: pybind11 3.1 moves the default
     * std::unique_ptr<Module> holder out of each list element, so the
     * Python-side temporaries transfer ownership into the registry and may be
     * deleted immediately. Children auto-number ("0", "1", ...) via
     * Sequential::append with an empty name, matching the C++ make() path.
     * @brief Build a Sequential by moving each list element's holder.
     * @param layers Python list of Module instances (Linear/Relu/Gelu/...).
     * @return The assembled Sequential as a Module handle.
     */
    [[nodiscard]] std::unique_ptr<Module> make_sequential(const py::list& layers) {
        auto seq = std::make_unique<Sequential>();
        for (const py::handle& item : layers) {
            seq->append("", py::cast<std::unique_ptr<Module>>(item));
        }
        return seq;
    }

    /**
     * @brief Load one MNIST split as flat rank-2 rows through the header-only
     * demos::mnist utilities.
     * @param dir Directory holding the uncompressed IDX files.
     * @param train True selects the 60k train pair (t10k-* when false).
     * @param dtype Model dtype for the pixel rows; the host staging buffer is
     *        Float64 only when the model dtype is Float64, else Float32.
     * @param device Device both tensors are uploaded to.
     * @return (images {N, 784} in @p dtype, labels {N} Int64).
     * @throws RuntimeError The IDX files are missing/invalid or the upload
     *         failed; the message names @p dir.
     */
    [[nodiscard]] std::pair<Tensor, Tensor> load_mnist_flat(
        const std::string& dir,
        bool train,
        DType dtype,
        Device device
    ) {
        auto split = demos::mnist::Split{};
        const auto image_file = train
            ? std::string_view{demos::mnist::kTrainImages}
            : std::string_view{demos::mnist::kTestImages};
        const auto label_file = train
            ? std::string_view{demos::mnist::kTrainLabels}
            : std::string_view{demos::mnist::kTestLabels};
        if (
            const auto ok = demos::mnist::load_split(std::filesystem::path{dir}, image_file, label_file, split);
            !ok
        ) {
            const auto msg = std::string{"load_mnist_flat: failed to read the MNIST split from '"} + dir + "'";
            raise_runtime_error(msg.c_str());
        }
        const auto host_dtype = dtype == DType::Float64 ? DType::Float64 : DType::Float32;
        auto uploaded = demos::mnist::upload_flat(split, device, dtype, host_dtype);
        if (!uploaded.has_value()) {
            const auto msg = std::string{"load_mnist_flat: failed to upload the MNIST split from '"} + dir + "'";
            raise_runtime_error(msg.c_str());
        }
        return std::pair{std::move(uploaded->images), std::move(uploaded->labels)};
    }

    /**
     * @brief Run Metrics::evaluate and unpack the host-side result into a
     * plain (accuracy, mean_loss) tuple of Python floats.
     * @param model Module to evaluate; its parameter requires_grad flags are
     *        saved and restored by the engine.
     * @param loader DataLoader over the evaluation set; the engine resets it
     *        before the pass.
     * @return (accuracy, mean_loss) as float64 values.
     */
    [[nodiscard]] std::tuple<float64_t, float64_t> evaluate_pair(Module& model, DataLoader& loader) {
        const auto result = BeanTensor::Metrics::evaluate(model, loader);
        return std::make_tuple(result.accuracy, result.mean_loss);
    }

} // namespace

PYBIND11_MODULE(beantensor, m) {
    m.attr("__version__") = "0.1";

    py::enum_<DType>(m, "DType")
        .value("Float16", DType::Float16)
        .value("Float32", DType::Float32)
        .value("Float64", DType::Float64)
        .value("Int32", DType::Int32)
        .value("Int64", DType::Int64);

    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("GPU", Device::GPU);

    /**
     * Optimizer-state depth of a .bt save. Weights writes architecture +
     * parameters (v1 layout); WeightsAndGrads appends per-parameter gradient
     * records; Full additionally appends Adam moments and the step counter,
     * so a mid-train checkpoint resumes training bit-reproducibly.
     */
    py::enum_<CheckpointLevel>(m, "CheckpointLevel")
        .value("Weights", CheckpointLevel::Weights)
        .value("WeightsAndGrads", CheckpointLevel::WeightsAndGrads)
        .value("Full", CheckpointLevel::Full);

    py::class_<Tensor>(m, "Tensor")
        .def_static("from_numpy", &from_numpy)
        .def("to_numpy", &to_numpy)
        .def_property_readonly("shape", &shape_tuple)
        .def_property_readonly("dtype", &Tensor::get_dtype)
        .def_property_readonly("device", &Tensor::get_device)
        .def("set_zero", &Tensor::set_zero)
        .def("sync", &Tensor::sync)
        // In-place, blocking move between devices (Tensor::to); returns None.
        // The GPU smoke (ctest bindings_gpu) stages its data with
        // from_numpy on CPU and uploads it through this.
        .def("to", &Tensor::to)
        // add wires a Node onto the result whenever an operand requires_grad,
        // so the output may join the graph and must keep BOTH operands alive
        // until backward() returns (the engine's graph parents are raw
        // pointers — see the Tensor class docs and Module::forward below).
        .def("add", [](const Tensor& self, const Tensor& other) { return self.add(other); },
             py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        .def("item", &item)
        .def("backward", &Tensor::backward);

    // The engine loader stores clone(false) views over the caller's tensor
    // buffers, so the Python-side sources must outlive the loader; keep_alive
    // pins both constructor args to it.
    py::class_<DataLoader>(m, "DataLoader")
        .def(py::init<const Tensor&, const Tensor&, size_t, bool, std::optional<uint64_t>, bool>(),
             py::arg("images"), py::arg("labels"), py::arg("batch"),
             py::arg("shuffle") = true, py::arg("seed") = py::none(), py::arg("drop_last") = false,
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>())
        .def("reset", &DataLoader::reset)
        // std::optional<std::pair<Tensor, Tensor>> maps to None | (Tensor, Tensor).
        .def("next_batch", &DataLoader::next_batch);

    // Abstract base: every layer factory below returns a
    // std::unique_ptr<Module> and Sequential's list ctor disowns each
    // element's holder into one — pybind11 3 requires py::smart_holder for
    // both directions of unique_ptr ownership transfer (classic holders
    // refuse with "requires py::class_<T, py::smart_holder>"), so Python
    // still sees one bt.Module type and Sequential can consume any factory
    // product.
    py::class_<Module, py::smart_holder>(m, "Module")
        // The returned output joins the computation graph through the
        // module's parameters and the input, which the engine borrows as raw
        // pointers (Module/Module.h graph-lifetime note), so the output must
        // keep BOTH the module and the input alive until backward() returns.
        // Engine rule that comes with it: a module's output must be
        // backwarded BEFORE that module's next forward() — every forward()
        // replaces the module's forward-cache (clear_forward_cache), dropping
        // the previous pass's intermediates. The natural training-loop order
        // (forward -> loss -> backward -> step -> zero_grad) satisfies it.
        .def("forward", [](Module& self, const Tensor& x) { return self.forward(x); },
             py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        .def("zero_grad", &Module::zero_grad)
        .def("adam_step", &Module::adam_step)
        .def("train", [](Module& self, bool mode) { self.train(mode); }, py::arg("mode") = true)
        .def("eval", &Module::eval)
        .def("named_parameters", &named_parameters_dict)
        .def("__str__", [](const Module& self) { return self.print(0); });

    m.def("Linear",
          [](size_t in_features, size_t out_features, DType dtype, Device device) {
              return std::unique_ptr<Module>(std::make_unique<Linear>(in_features, out_features, dtype, device));
          });
    m.def("Relu", []() { return std::unique_ptr<Module>(std::make_unique<Relu>()); });
    m.def("Gelu", []() { return std::unique_ptr<Module>(std::make_unique<Gelu>()); });
    m.def("Sequential", &make_sequential);
    // The loss's graph node borrows the logits and labels tensors as raw
    // pointers, so the returned loss must keep both alive until backward()
    // returns (same contract as Module.forward above).
    m.def("softmax_cross_entropy", &BeanTensor::Modules::softmax_cross_entropy,
          py::keep_alive<0, 1>(), py::keep_alive<0, 2>());
    m.def("evaluate", &evaluate_pair, py::arg("model"), py::arg("loader"));
    m.def("load_mnist_flat", &load_mnist_flat,
          py::arg("dir"), py::arg("train"), py::arg("dtype"), py::arg("device"));
    // Training-loop determinism starts from the process-wide default
    // generator: Linear draws its init seed from it, so reseeding here makes
    // a from-scratch net construction reproducible across processes.
    m.def("set_seed", &BeanTensor::Random::set_seed);

    // .bt export/import. save is synchronous and quiesces nothing itself —
    // sync and settle the model first for a reproducible checkpoint (the
    // engine's contract, see Serialization.h). load's unique_ptr<Module>
    // return rides the same smart_holder ownership transfer as the layer
    // factories, so Python sees one bt.Module. Engine-side failures
    // (missing/corrupt file, unsupported version, unknown module type)
    // arrive as RuntimeError through the translator below.
    m.def("save", &BeanTensor::Serialization::save,
          py::arg("module"), py::arg("path"), py::arg("compress") = true,
          py::arg("level") = CheckpointLevel::Weights);
    m.def("load", &BeanTensor::Serialization::load,
          py::arg("path"), py::arg("device") = Device::CPU);

    // Every ErrorHandling::* type in Helpers/Errors.h derives directly from a
    // std exception; the project has no deeper common root, so catching
    // std::exception here covers the whole engine surface and lands engine
    // errors as Python RuntimeError carrying e.what(). pybind11 3.x runs user
    // translators BEFORE its builtin mapping, and its own exception types
    // (py::value_error et al.) derive from std::exception too — rethrow them
    // untouched so the builtin fallback restores their Python types.
    py::register_exception_translator([](std::exception_ptr p) {
        if (!p) {
            return;
        }
        try {
            std::rethrow_exception(p);
        } catch (const py::error_already_set &) {
            throw;
        } catch (const py::builtin_exception &) {
            throw;
        } catch (const std::exception& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
            throw py::error_already_set();
        }
    });
}
