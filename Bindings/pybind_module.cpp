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

#include "Data/Csv.h"
#include "Data/DataLoader.h"
#include "Metrics/Metrics.h"
#include "Module/BatchNorm.h"
#include "Module/Conv2d.h"
#include "Module/Dropout.h"
#include "Module/Flatten.h"
#include "Module/Gelu.h"
#include "Module/Linear.h"
#include "Module/MaxPool2d.h"
#include "Module/Module.h"
#include "Module/Relu.h"
#include "Module/Sequential.h"
#include "Random/Generator.h"
#include "Serialization/Serialization.h"
#include "Tensor/DTypes.h"
#include "Tensor/Tensor.h"
#include "demos/Mnist.h"

namespace py = pybind11;

using BeanTensor::Data::CsvDataset;
using BeanTensor::Data::CsvNormalize;
using BeanTensor::Data::CsvNullPolicy;
using BeanTensor::Data::CsvOptions;
using BeanTensor::Data::DataLoader;
using BeanTensor::DataTypes::DType;
using BeanTensor::float16_t;
using BeanTensor::float32_t;
using BeanTensor::float64_t;
using BeanTensor::Modules::BatchNorm;
using BeanTensor::Modules::Conv2d;
using BeanTensor::Modules::Dropout;
using BeanTensor::Modules::Flatten;
using BeanTensor::Modules::Gelu;
using BeanTensor::Modules::Linear;
using BeanTensor::Modules::MaxPool2d;
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
     * @brief Map a NumPy dtype onto the engine dtype for every NumPy-native
     * scalar dtype with an engine counterpart (float16 through uint64).
     * @param dt NumPy dtype of the source array.
     * @return Matching engine dtype.
     * @throws py::value_error The dtype has no engine counterpart (bool,
     *         float128, complex, object, str).
     */
    [[nodiscard]] DType dtype_from_numpy(const py::dtype& dt) {
        if (dt.is(py::dtype("float16"))) {
            return DType::Float16;
        }
        if (dt.is(py::dtype::of<float32_t>())) {
            return DType::Float32;
        }
        if (dt.is(py::dtype::of<float64_t>())) {
            return DType::Float64;
        }
        if (dt.is(py::dtype::of<int8_t>())) {
            return DType::Int8;
        }
        if (dt.is(py::dtype::of<int16_t>())) {
            return DType::Int16;
        }
        if (dt.is(py::dtype::of<int32_t>())) {
            return DType::Int32;
        }
        if (dt.is(py::dtype::of<int64_t>())) {
            return DType::Int64;
        }
        if (dt.is(py::dtype::of<uint8_t>())) {
            return DType::UInt8;
        }
        if (dt.is(py::dtype::of<uint16_t>())) {
            return DType::UInt16;
        }
        if (dt.is(py::dtype::of<uint32_t>())) {
            return DType::UInt32;
        }
        if (dt.is(py::dtype::of<uint64_t>())) {
            return DType::UInt64;
        }
        throw py::value_error(
            "from_numpy: unsupported dtype (supported: float16, float32, float64, "
            "int8, int16, int32, int64, uint8, uint16, uint32, uint64)"
        );
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
     * @tparam T Element type matching the tensor's dtype; only sizeof(T) is
     *         used, so the packed/enum engine element types work too.
     * @param t Synced CPU tensor.
     * @param np_dt NumPy dtype of the output array (numpy has no dtype for the
     *        engine's BFloat16 or packed lanes, hence the explicit argument).
     * @return New 1-element NumPy array; the copy owns its buffer.
     */
    template <typename T>
    [[nodiscard]] py::array to_numpy_typed(const Tensor& t, const py::dtype& np_dt) {
        std::vector<ssize_t> dims(t.rank());
        for (size_t i = 0; i < t.rank(); ++i) {
            dims[i] = static_cast<ssize_t>(t.dim(i));
        }
        py::array out(np_dt, dims);
        t.read_element(0, out.mutable_data(), static_cast<size_t>(out.size()) * sizeof(T));
        return out;
    }

    /**
     * @brief Sync a CPU tensor and copy its elements out to a new NumPy array.
     * @param t Source tensor.
     * @return New NumPy array owning a copy of the tensor's data.
     * @throws RuntimeError The tensor is on GPU.
     * @throws py::value_error The dtype has no NumPy-native counterpart
     *         (BFloat16, BFloat16x2, Float16x2, Int4x2); read those with
     *         to_bytes() instead.
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
            case DType::Float16:
                return to_numpy_typed<float16_t>(t, py::dtype("float16"));
            case DType::Float32:
                return to_numpy_typed<float32_t>(t, py::dtype::of<float32_t>());
            case DType::Float64:
                return to_numpy_typed<float64_t>(t, py::dtype::of<float64_t>());
            case DType::Int8:
                return to_numpy_typed<int8_t>(t, py::dtype::of<int8_t>());
            case DType::Int16:
                return to_numpy_typed<int16_t>(t, py::dtype::of<int16_t>());
            case DType::Int32:
                return to_numpy_typed<int32_t>(t, py::dtype::of<int32_t>());
            case DType::Int64:
                return to_numpy_typed<int64_t>(t, py::dtype::of<int64_t>());
            case DType::UInt8:
                return to_numpy_typed<uint8_t>(t, py::dtype::of<uint8_t>());
            case DType::UInt16:
                return to_numpy_typed<uint16_t>(t, py::dtype::of<uint16_t>());
            case DType::UInt32:
                return to_numpy_typed<uint32_t>(t, py::dtype::of<uint32_t>());
            case DType::UInt64:
                return to_numpy_typed<uint64_t>(t, py::dtype::of<uint64_t>());
            default: {
                const auto name = BeanTensor::DataTypes::dtype_to_string(t.get_dtype());
                throw py::value_error(
                    std::string{"to_numpy: dtype "} + name.c_str() +
                    " has no NumPy-native counterpart; use to_bytes() for the raw element bytes"
                );
            }
        }
    }

    /**
     * @brief Build a tensor from raw element bytes (the only host path for the
     *        dtypes NumPy cannot represent: BFloat16, the x2 float packs and
     *        Int4x2 — and any future dtype).
     * @param data Little-endian element bytes, exactly element_count times the
     *        dtype's element size.
     * @param shape Tensor shape.
     * @param dtype Engine dtype the bytes are interpreted as.
     * @param device Device the tensor is created on.
     * @return Owning tensor with the given shape and dtype.
     * @throws py::value_error The byte count does not match shape times dtype.
     */
    [[nodiscard]] Tensor from_bytes(py::bytes data, std::vector<size_t> shape, const DType dtype, const Device device) {
        char* buffer = nullptr;
        Py_ssize_t length = 0;
        if (PyBytes_AsStringAndSize(data.ptr(), &buffer, &length) != 0) {
            throw py::error_already_set();
        }
        auto count = size_t{1};
        for (const size_t dim : shape) {
            count *= dim;
        }
        if (
            const size_t expected = count * BeanTensor::DataTypes::dtype_bytes(dtype);
            static_cast<size_t>(length) != expected
        ) {
            const auto msg = std::string{"from_bytes: got "} + std::to_string(length) +
                " bytes, but shape and dtype require exactly " + std::to_string(expected);
            throw py::value_error(msg);
        }
        Tensor t(std::move(shape), dtype, device);
        t.write_element(0, buffer, static_cast<size_t>(length));
        return t;
    }

    /**
     * @brief Sync the tensor and copy its element bytes out to a Python bytes
     *        object — the exact inverse of from_bytes.
     * @param t Source tensor.
     * @return Exactly element_count times the dtype's element size bytes.
     */
    [[nodiscard]] py::bytes to_bytes(const Tensor& t) {
        t.sync();
        const size_t n = t.byte_count();
        std::string out(n, '\0');
        t.read_element(0, out.data(), n);
        return py::bytes(out);
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
     * @brief Load one MNIST split as NCHW rank-4 {N, 1, 28, 28} images
     *        through the header-only demos::mnist utilities (the CNN demo's
     *        data path; C=1 NCHW is pixel-order contiguous, so the flat IDX
     *        payload maps 1:1).
     * @param dir Directory holding the uncompressed IDX files.
     * @param train True selects the 60k train pair (t10k-* when false).
     * @param dtype Model dtype for the images; the host staging buffer is
     *        Float64 only when the model dtype is Float64, else Float32.
     * @param device Device both tensors are uploaded to.
     * @return (images {N, 1, 28, 28} in @p dtype, labels {N} Int64).
     * @throws RuntimeError The IDX files are missing/invalid or the upload
     *         failed; the message names @p dir.
     */
    [[nodiscard]] std::pair<Tensor, Tensor> load_mnist_nchw(
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
            const auto msg = std::string{"load_mnist_nchw: failed to read the MNIST split from '"} + dir + "'";
            raise_runtime_error(msg.c_str());
        }
        const auto host_dtype = dtype == DType::Float64 ? DType::Float64 : DType::Float32;
        auto uploaded = demos::mnist::upload_nchw(split, device, dtype, host_dtype);
        if (!uploaded.has_value()) {
            const auto msg = std::string{"load_mnist_nchw: failed to upload the MNIST split from '"} + dir + "'";
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

    /**
     * @brief Load a CSV file through data::load_csv and hand the whole
     *        dataset (tensor pair + inspection metadata) to Python.
     * @param path Filesystem path to the CSV file.
     * @param opts Parsing options; only the bound fields (feature dtype, null
     *        policy, normalization) depart from the engine defaults.
     * @return The CsvDataset: features {N, F} in opts.feature_dtype, labels
     *         {N} Int32 class ids, plus null_count, feature/label names and
     *         the NaN-skipped per-column stats csv_demo.cpp prints.
     * @throws RuntimeError The file cannot be opened or a cell violates the
     *         chosen policies; the engine's message is preserved through the
     *         module's exception translator.
     */
    [[nodiscard]] CsvDataset load_csv_dataset(const std::string& path, const CsvOptions& opts) {
        return BeanTensor::Data::load_csv(path, opts);
    }

} // namespace

PYBIND11_MODULE(beantensor, m) {
    m.attr("__version__") = "0.1";

    py::enum_<DType>(m, "DType")
        .value("BFloat16", DType::BFloat16)
        .value("BFloat16x2", DType::BFloat16x2)
        .value("Float16", DType::Float16)
        .value("Float16x2", DType::Float16x2)
        .value("Float32", DType::Float32)
        .value("Float64", DType::Float64)
        .value("Int64", DType::Int64)
        .value("Int32", DType::Int32)
        .value("Int16", DType::Int16)
        .value("Int8", DType::Int8)
        .value("UInt64", DType::UInt64)
        .value("UInt32", DType::UInt32)
        .value("UInt16", DType::UInt16)
        .value("UInt8", DType::UInt8)
        .value("Int4x2", DType::Int4x2);

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

    /**
     * CSV parsing policies; member names mirror Data/Csv.h exactly. Note
     * CsvNormalize.None is reachable only through getattr — "None" is not a
     * dotted-access name in Python syntax.
     */
    py::enum_<CsvNullPolicy>(m, "CsvNullPolicy")
        .value("KeepNaN", CsvNullPolicy::KeepNaN)
        .value("Error", CsvNullPolicy::Error)
        .value("FillMean", CsvNullPolicy::FillMean)
        .value("FillMedian", CsvNullPolicy::FillMedian)
        .value("FillConstant", CsvNullPolicy::FillConstant);

    py::enum_<CsvNormalize>(m, "CsvNormalize")
        .value("None", CsvNormalize::None)
        .value("MinMax", CsvNormalize::MinMax)
        .value("Standardize", CsvNormalize::Standardize);

    // The bound subset of data::CsvOptions (Data/Csv.h): the three fields
    // the csv demo exercises. The unbound fields keep the engine defaults
    // (automatic header/delimiter detection, last-column labels, KeepNaN
    // nulls). load_csv takes the struct by value, so mutating the Python
    // handle after a call never retroactively changes that call.
    py::class_<CsvOptions>(m, "CsvOptions")
        .def(py::init<>())
        .def_readwrite("feature_dtype", &CsvOptions::feature_dtype)
        .def_readwrite("nulls", &CsvOptions::nulls)
        .def_readwrite("normalize", &CsvOptions::normalize);

    // The C++ load_csv return (Data/Csv.h): the tensor pair plus the column
    // metadata the csv demo's inspection act prints. Tensor members are
    // handles onto the dataset's members (reference_internal keeps the
    // dataset alive); the name/stat vectors come back as plain Python lists.
    py::class_<CsvDataset>(m, "CsvDataset")
        .def_readonly("features", &CsvDataset::features, py::return_value_policy::reference_internal)
        .def_readonly("labels", &CsvDataset::labels, py::return_value_policy::reference_internal)
        .def_readonly("label_names", &CsvDataset::label_names)
        .def_readonly("feature_names", &CsvDataset::feature_names)
        .def_readonly("null_count", &CsvDataset::null_count)
        .def_readonly("column_min", &CsvDataset::column_min)
        .def_readonly("column_max", &CsvDataset::column_max)
        .def_readonly("column_mean", &CsvDataset::column_mean);

    py::class_<Tensor>(m, "Tensor")
        .def_static("from_numpy", &from_numpy)
        // Raw-bits construction for the dtypes NumPy cannot represent
        // (BFloat16, the x2 float packs, Int4x2): one bulk host write after an
        // exact byte-size check.
        .def_static("from_bytes", &from_bytes,
                    py::arg("data"), py::arg("shape"), py::arg("dtype"),
                    py::arg("device") = Device::CPU)
        .def("to_numpy", &to_numpy)
        // Exact inverse of from_bytes; works for every engine dtype.
        .def("to_bytes", &to_bytes)
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
        // Same op and graph contract as the softmax_cross_entropy free
        // function: the loss node borrows this tensor and the labels, so the
        // output keeps BOTH alive until backward() returns.
        .def("softmax_ce", [](const Tensor& self, const Tensor& labels) { return self.softmax_ce(labels); },
             py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        // The evaluation verbs the csv demo replays (csv_demo.cpp:
        // argmax(1) -> convert_dtype(Int32) -> equal(labels) ->
        // convert_dtype(Float32) -> mean). argmax and equal outputs never
        // join the autograd graph (the Tensor.h op docs say so explicitly),
        // so they carry no keep_alive; mean attaches a MeanG node when the
        // input requires_grad, so its output keeps the input alive until
        // backward() returns (same contract as add above).
        .def("argmax", [](const Tensor& self, const size_t axis) { return self.argmax(axis); })
        .def("equal", [](const Tensor& self, const Tensor& other) { return self.equal(other); })
        .def("mean", [](const Tensor& self) { return self.mean(); },
             py::keep_alive<0, 1>())
        // In-place dtype conversion (Tensor::convert_dtype with the default
        // PERMISSIVE clamp method); returns None.
        .def("convert_dtype", [](Tensor& self, const DType new_dtype) { self.convert_dtype(new_dtype); })
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

    // CNN stack factories. Conv2d's argument order mirrors cnn.cpp's call
    // exactly — (in_channels, out_channels, kernel, stride, pad, dtype,
    // device) — so a user copying the demo gets the same topology; the
    // remaining ctors expose the demo-used subset of their headers' defaults
    // (BatchNorm keeps its momentum/eps, MaxPool2d its stride-is-kernel).
    m.def("Conv2d",
          [](uint32_t in_channels, uint32_t out_channels, uint32_t kernel,
             uint32_t stride, uint32_t pad, DType dtype, Device device) {
              return std::unique_ptr<Module>(std::make_unique<Conv2d>(
                  in_channels, out_channels, kernel, stride, pad, dtype, device));
          });
    m.def("BatchNorm",
          [](size_t channels, DType dtype, Device device) {
              return std::unique_ptr<Module>(std::make_unique<BatchNorm>(channels, dtype, device));
          });
    m.def("MaxPool2d",
          [](uint32_t kernel) {
              return std::unique_ptr<Module>(std::make_unique<MaxPool2d>(kernel));
          });
    m.def("Flatten", []() { return std::unique_ptr<Module>(std::make_unique<Flatten>()); });
    m.def("Dropout",
          [](float64_t p) { return std::unique_ptr<Module>(std::make_unique<Dropout>(p)); });
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
    m.def("load_mnist_nchw", &load_mnist_nchw,
          py::arg("dir"), py::arg("train"), py::arg("dtype"), py::arg("device"));
    m.def("load_csv", &load_csv_dataset, py::arg("path"), py::arg("opts"));
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
