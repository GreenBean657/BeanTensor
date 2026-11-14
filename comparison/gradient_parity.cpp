/**
 * @file gradient_parity.cpp
 * @author GreenBean
 * @brief C++ side of the PyTorch gradient parity harness. Generates identical
 * inputs for every test case, executes the autograd graph, and writes raw
 * little-endian tensor bytes plus a text manifest under the requested output
 * directory (default comparison/generated-data). The Python driver
 * (comparison/gradient_parity.py) rebuilds the same graphs in PyTorch and
 * compares every gradient against the bytes written here.
 * @version 0.1
 * @date 2026-11-14
 */
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "BeanTensor.h"
#include "Helpers/Errors.h"
#include "Math/Dropout/Dropout.h"

using BeanTensor::DataTypes::DType;
using BeanTensor::Modules::Linear;
using BeanTensor::Modules::Conv2d;
using BeanTensor::Modules::BatchNorm;
using BeanTensor::Modules::Relu;
using BeanTensor::Modules::MaxPool2d;
using BeanTensor::Modules::Flatten;
using BeanTensor::Modules::Sequential;
using BeanTensor::Tensors::Device;
using BeanTensor::Tensors::Tensor;
using BeanTensor::float32_t;
using BeanTensor::float64_t;
using BeanTensor::ErrorHandling::NotImplemented;
using BeanTensor::ErrorHandling::SerializationError;

namespace {

    struct CaseConfig {
        std::string name;
        DType grad_dtype;
        float64_t abs_tol;
        float64_t rel_tol;
    };

    [[nodiscard]] std::string manifest_dtype_string(const DType dt) {
        switch (dt) {
            case DType::Float32: return "Float32";
            case DType::Float64: return "Float64";
            case DType::Int32: return "Int32";
            case DType::Int64: return "Int64";
            [[unlikely]] default: throw NotImplemented();
        }
    }

    [[nodiscard]] size_t dtype_bytes(const DType dt) {
        switch (dt) {
            case DType::Float32: return sizeof(float32_t);
            case DType::Float64: return sizeof(float64_t);
            case DType::Int32: return sizeof(int32_t);
            case DType::Int64: return sizeof(int64_t);
            [[unlikely]] default: throw NotImplemented();
        }
    }

    [[nodiscard]] std::string shape_string(const std::vector<size_t>& shape) {
        auto out = std::string{};
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i != 0) {
                out += "x";
            }
            out += std::to_string(shape[i]);
        }
        return out;
    }

    void write_bytes(const std::string& path, const void* data, const size_t bytes) {
        auto file = std::ofstream(path, std::ios::binary);
        if (!file) {
            throw SerializationError("cannot open " + path);
        }
        file.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
        if (!file) {
            throw SerializationError("write failed: " + path);
        }
    }

    void write_tensor(const Tensor& t, const std::string& path) {
        t.sync();
        const auto bytes = t.byte_count();
        auto buffer = std::vector<char>(bytes);
        t.read_element(0, buffer.data(), bytes);
        write_bytes(path, buffer.data(), bytes);
    }

    [[nodiscard]] Tensor random_tensor(const std::vector<size_t>& shape, const DType dt,
                                       const Device device, const int64_t seed) {
        auto t = Tensor(shape, dt, device, true);
        t.random(-1.0, 1.0, seed);
        return t;
    }

    [[nodiscard]] Tensor make_label_tensor(const size_t rows, const size_t classes,
                                           const Device device, const int64_t seed) {
        auto labels = Tensor(std::vector<size_t>{rows}, DType::Int32, device, false);
        auto gen = BeanTensor::Random::Generator(static_cast<uint64_t>(seed));
        for (size_t i = 0; i < rows; ++i) {
            const auto cls = static_cast<int32_t>(gen.draw_seed() % classes);
            labels.write_element(i, &cls, sizeof(int32_t));
        }
        return labels;
    }

    void write_manifest_entry(std::ofstream& manifest, const std::string& case_name,
                              const DType grad_dtype, const float64_t abs_tol,
                              const float64_t rel_tol) {
        manifest << "case " << case_name << " " << manifest_dtype_string(grad_dtype)
                 << " " << std::setprecision(17) << abs_tol
                 << " " << std::setprecision(17) << rel_tol << "\n";
    }

    void write_manifest_tensor(std::ofstream& manifest, const std::string& name,
                               const Tensor& t, const std::string& rel_path) {
        manifest << "tensor " << name << " " << manifest_dtype_string(t.get_dtype());
        for (size_t i = 0; i < t.rank(); ++i) {
            manifest << " " << t.dim(i);
        }
        manifest << " " << rel_path << "\n";
    }

    void write_gradient(std::ofstream& manifest, const std::string& name,
                        const Tensor& param, const std::string& case_name,
                        const std::filesystem::path& dir) {
        const auto rel_path = case_name + "/" + name + ".bin";
        const auto path = dir / rel_path;
        write_tensor(param.gradient(), path.string());
        write_manifest_tensor(manifest, name, param.gradient(), rel_path);
    }

    void reset_case_gradients(Sequential& model) {
        model.zero_grad();
    }

    void run_broadcast_add(const CaseConfig& cfg, const Device device,
                           const std::filesystem::path& dir, std::ofstream& manifest) {
        write_manifest_entry(manifest, cfg.name, cfg.grad_dtype, cfg.abs_tol, cfg.rel_tol);
        const auto case_dir = dir / cfg.name;
        std::filesystem::create_directories(case_dir);

        auto a = random_tensor({4, 4}, DType::Float64, device, 1001);
        auto b = random_tensor({4, 1}, DType::Float64, device, 1002);
        write_tensor(a, (case_dir / "a.bin").string());
        write_manifest_tensor(manifest, "a", a, cfg.name + "/a.bin");
        write_tensor(b, (case_dir / "b.bin").string());
        write_manifest_tensor(manifest, "b", b, cfg.name + "/b.bin");

        auto c = a.add(b);
        auto loss = c.sum();
        loss.backward(cfg.grad_dtype);

        write_gradient(manifest, "grad_a", a, cfg.name, dir);
        write_gradient(manifest, "grad_b", b, cfg.name, dir);
    }

    void run_broadcast_mul(const CaseConfig& cfg, const Device device,
                           const std::filesystem::path& dir, std::ofstream& manifest) {
        write_manifest_entry(manifest, cfg.name, cfg.grad_dtype, cfg.abs_tol, cfg.rel_tol);
        const auto case_dir = dir / cfg.name;
        std::filesystem::create_directories(case_dir);

        auto a = random_tensor({3, 1, 4}, DType::Float64, device, 2001);
        auto b = random_tensor({1, 5, 4}, DType::Float64, device, 2002);
        write_tensor(a, (case_dir / "a.bin").string());
        write_manifest_tensor(manifest, "a", a, cfg.name + "/a.bin");
        write_tensor(b, (case_dir / "b.bin").string());
        write_manifest_tensor(manifest, "b", b, cfg.name + "/b.bin");

        auto c = a.mul(b);
        auto loss = c.sum();
        loss.backward(cfg.grad_dtype);

        write_gradient(manifest, "grad_a", a, cfg.name, dir);
        write_gradient(manifest, "grad_b", b, cfg.name, dir);
    }

    void run_broadcast_div(const CaseConfig& cfg, const Device device,
                           const std::filesystem::path& dir, std::ofstream& manifest) {
        write_manifest_entry(manifest, cfg.name, cfg.grad_dtype, cfg.abs_tol, cfg.rel_tol);
        const auto case_dir = dir / cfg.name;
        std::filesystem::create_directories(case_dir);

        auto a = random_tensor({2, 3}, DType::Float64, device, 3001);
        auto b_raw = random_tensor({1, 3}, DType::Float64, device, 3002);
        // Keep divisor away from zero for finite gradients.  b_raw stays alive
        // as an autograd parent; b is the shifted tensor used in the div.
        auto b = b_raw.add(float64_t{0.5});
        write_tensor(a, (case_dir / "a.bin").string());
        write_manifest_tensor(manifest, "a", a, cfg.name + "/a.bin");
        write_tensor(b, (case_dir / "b.bin").string());
        write_manifest_tensor(manifest, "b", b, cfg.name + "/b.bin");

        auto c = a.div(b);
        auto loss = c.sum();
        loss.backward(cfg.grad_dtype);

        write_gradient(manifest, "grad_a", a, cfg.name, dir);
        write_gradient(manifest, "grad_b", b, cfg.name, dir);
    }

    void run_matmul_bias_relu_softmax_ce(const CaseConfig& cfg, const Device device,
                                          const std::filesystem::path& dir, std::ofstream& manifest) {
        write_manifest_entry(manifest, cfg.name, cfg.grad_dtype, cfg.abs_tol, cfg.rel_tol);
        const auto case_dir = dir / cfg.name;
        std::filesystem::create_directories(case_dir);

        const auto rows = size_t{8};
        const auto in_features = size_t{16};
        const auto out_features = size_t{10};
        auto x = random_tensor({rows, in_features}, DType::Float32, device, 4001);
        auto labels = make_label_tensor(rows, out_features, device, 4002);

        auto w = random_tensor({in_features, out_features}, DType::Float32, device, 4003);
        auto bias = Tensor(std::vector<size_t>{out_features}, DType::Float32, device, true);
        bias.set_zero();

        write_tensor(x, (case_dir / "x.bin").string());
        write_manifest_tensor(manifest, "x", x, cfg.name + "/x.bin");
        write_tensor(labels, (case_dir / "labels.bin").string());
        write_manifest_tensor(manifest, "labels", labels, cfg.name + "/labels.bin");
        write_tensor(w, (case_dir / "weight.bin").string());
        write_manifest_tensor(manifest, "weight", w, cfg.name + "/weight.bin");
        write_tensor(bias, (case_dir / "bias.bin").string());
        write_manifest_tensor(manifest, "bias", bias, cfg.name + "/bias.bin");

        auto h = x.matmul(w);
        auto biased = h.bias_add(bias);
        auto y = biased.relu();
        auto loss = y.softmax_ce(labels);
        loss.backward(cfg.grad_dtype);

        write_gradient(manifest, "grad_x", x, cfg.name, dir);
        write_gradient(manifest, "grad_weight", w, cfg.name, dir);
        write_gradient(manifest, "grad_bias", bias, cfg.name, dir);
    }

    void run_mse_linear(const CaseConfig& cfg, const Device device,
                        const std::filesystem::path& dir, std::ofstream& manifest) {
        write_manifest_entry(manifest, cfg.name, cfg.grad_dtype, cfg.abs_tol, cfg.rel_tol);
        const auto case_dir = dir / cfg.name;
        std::filesystem::create_directories(case_dir);

        const auto rows = size_t{8};
        const auto in_features = size_t{16};
        const auto out_features = size_t{4};
        auto x = random_tensor({rows, in_features}, DType::Float32, device, 5001);
        auto target = random_tensor({rows, out_features}, DType::Float32, device, 5002);
        auto w = random_tensor({in_features, out_features}, DType::Float32, device, 5003);
        auto bias = Tensor(std::vector<size_t>{out_features}, DType::Float32, device, true);
        bias.set_zero();

        write_tensor(x, (case_dir / "x.bin").string());
        write_manifest_tensor(manifest, "x", x, cfg.name + "/x.bin");
        write_tensor(target, (case_dir / "target.bin").string());
        write_manifest_tensor(manifest, "target", target, cfg.name + "/target.bin");
        write_tensor(w, (case_dir / "weight.bin").string());
        write_manifest_tensor(manifest, "weight", w, cfg.name + "/weight.bin");
        write_tensor(bias, (case_dir / "bias.bin").string());
        write_manifest_tensor(manifest, "bias", bias, cfg.name + "/bias.bin");

        auto h = x.matmul(w);
        auto pred = h.bias_add(bias);
        auto loss = pred.mse(target);
        loss.backward(cfg.grad_dtype);

        write_gradient(manifest, "grad_x", x, cfg.name, dir);
        write_gradient(manifest, "grad_weight", w, cfg.name, dir);
        write_gradient(manifest, "grad_bias", bias, cfg.name, dir);
    }

    void run_conv_bn_pool_flatten(const CaseConfig& cfg, const Device device,
                                   const std::filesystem::path& dir, std::ofstream& manifest) {
        write_manifest_entry(manifest, cfg.name, cfg.grad_dtype, cfg.abs_tol, cfg.rel_tol);
        const auto case_dir = dir / cfg.name;
        std::filesystem::create_directories(case_dir);

        const auto n = size_t{2};
        const auto c = size_t{3};
        const auto h = size_t{8};
        const auto out_c = size_t{4};
        const auto classes = size_t{3};
        auto x = random_tensor({n, c, h, h}, DType::Float32, device, 6001);
        auto labels = make_label_tensor(n, classes, device, 6002);

        write_tensor(x, (case_dir / "x.bin").string());
        write_manifest_tensor(manifest, "x", x, cfg.name + "/x.bin");
        write_tensor(labels, (case_dir / "labels.bin").string());
        write_manifest_tensor(manifest, "labels", labels, cfg.name + "/labels.bin");

        auto conv = Conv2d(static_cast<uint32_t>(c), static_cast<uint32_t>(out_c),
                           3, 1, 1, DType::Float32, device, true, 6003);
        auto bn = BatchNorm(out_c, DType::Float32, device, 0.1, 1e-5);
        auto relu = Relu();
        auto pool = MaxPool2d(2, 2);
        auto flatten = Flatten();
        // After {2,4,8,8} -> conv {2,4,8,8} -> pool {2,4,4,4} -> flatten {2,64}
        auto linear = Linear(64, classes, DType::Float32, device, true, 6004);

        auto& conv_w = conv.parameter("weight");
        auto& conv_b = conv.parameter("bias");
        auto& bn_g = bn.parameter("gamma");
        auto& bn_b = bn.parameter("beta");
        auto& lin_w = linear.parameter("weight");
        auto& lin_b = linear.parameter("bias");

        write_tensor(conv_w, (case_dir / "conv_weight.bin").string());
        write_manifest_tensor(manifest, "conv_weight", conv_w, cfg.name + "/conv_weight.bin");
        write_tensor(conv_b, (case_dir / "conv_bias.bin").string());
        write_manifest_tensor(manifest, "conv_bias", conv_b, cfg.name + "/conv_bias.bin");
        write_tensor(bn_g, (case_dir / "bn_gamma.bin").string());
        write_manifest_tensor(manifest, "bn_gamma", bn_g, cfg.name + "/bn_gamma.bin");
        write_tensor(bn_b, (case_dir / "bn_beta.bin").string());
        write_manifest_tensor(manifest, "bn_beta", bn_b, cfg.name + "/bn_beta.bin");
        write_tensor(lin_w, (case_dir / "linear_weight.bin").string());
        write_manifest_tensor(manifest, "linear_weight", lin_w, cfg.name + "/linear_weight.bin");
        write_tensor(lin_b, (case_dir / "linear_bias.bin").string());
        write_manifest_tensor(manifest, "linear_bias", lin_b, cfg.name + "/linear_bias.bin");

        auto t1 = conv.forward(x);
        auto t2 = bn.forward(t1);
        auto t3 = relu.forward(t2);
        auto t4 = pool.forward(t3);
        auto t5 = flatten.forward(t4);
        auto logits = linear.forward(t5);
        auto loss = logits.softmax_ce(labels);
        loss.backward(cfg.grad_dtype);

        write_gradient(manifest, "grad_x", x, cfg.name, dir);
        write_gradient(manifest, "grad_conv_weight", conv_w, cfg.name, dir);
        write_gradient(manifest, "grad_conv_bias", conv_b, cfg.name, dir);
        write_gradient(manifest, "grad_bn_gamma", bn_g, cfg.name, dir);
        write_gradient(manifest, "grad_bn_beta", bn_b, cfg.name, dir);
        write_gradient(manifest, "grad_linear_weight", lin_w, cfg.name, dir);
        write_gradient(manifest, "grad_linear_bias", lin_b, cfg.name, dir);
    }

    void run_dropout_fixed_mask(const CaseConfig& cfg, const Device device,
                                const std::filesystem::path& dir, std::ofstream& manifest) {
        write_manifest_entry(manifest, cfg.name, cfg.grad_dtype, cfg.abs_tol, cfg.rel_tol);
        const auto case_dir = dir / cfg.name;
        std::filesystem::create_directories(case_dir);

        const auto shape = std::vector<size_t>{2, 4};
        auto x = random_tensor(shape, DType::Float32, device, 7001);
        auto mask = BeanTensor::Math::Dropout::train_mask(x, 0.5, 12345);
        mask.sync();

        write_tensor(x, (case_dir / "x.bin").string());
        write_manifest_tensor(manifest, "x", x, cfg.name + "/x.bin");
        write_tensor(mask, (case_dir / "mask.bin").string());
        write_manifest_tensor(manifest, "mask", mask, cfg.name + "/mask.bin");

        auto y = x.mul(mask);
        auto loss = y.sum();
        loss.backward(cfg.grad_dtype);

        write_gradient(manifest, "grad_x", x, cfg.name, dir);
    }

    void run_chained_fanout(const CaseConfig& cfg, const Device device,
                            const std::filesystem::path& dir, std::ofstream& manifest) {
        write_manifest_entry(manifest, cfg.name, cfg.grad_dtype, cfg.abs_tol, cfg.rel_tol);
        const auto case_dir = dir / cfg.name;
        std::filesystem::create_directories(case_dir);

        auto x = random_tensor({2, 4}, DType::Float64, device, 8001);
        write_tensor(x, (case_dir / "x.bin").string());
        write_manifest_tensor(manifest, "x", x, cfg.name + "/x.bin");

        auto y = x.mul(float64_t{2.0});
        auto z1 = y.add(float64_t{1.0});
        auto z2 = y.mul(y);
        auto prod = z1.mul(z2);
        auto loss = prod.mean();
        loss.backward(cfg.grad_dtype);

        write_gradient(manifest, "grad_x", x, cfg.name, dir);
    }

    using CaseFn = void (*)(const CaseConfig&, Device, const std::filesystem::path&, std::ofstream&);

    [[nodiscard]] std::vector<std::pair<CaseConfig, CaseFn>> all_cases(const bool smoke) {
        if (smoke) {
            return {
                {{"broadcast_add", DType::Float64, 1e-12, 1e-12}, run_broadcast_add},
                {{"chained_fanout", DType::Float64, 1e-12, 1e-12}, run_chained_fanout},
            };
        }
        return {
            {{"broadcast_add", DType::Float64, 1e-12, 1e-12}, run_broadcast_add},
            {{"broadcast_mul", DType::Float64, 1e-12, 1e-12}, run_broadcast_mul},
            {{"broadcast_div", DType::Float64, 1e-12, 1e-12}, run_broadcast_div},
            {{"matmul_bias_relu_softmax_ce", DType::Float32, 1e-4, 1e-4}, run_matmul_bias_relu_softmax_ce},
            {{"mse_linear", DType::Float32, 1e-4, 1e-4}, run_mse_linear},
            {{"conv_bn_pool_flatten", DType::Float32, 1e-4, 1e-4}, run_conv_bn_pool_flatten},
            {{"dropout_fixed_mask", DType::Float32, 1e-5, 1e-5}, run_dropout_fixed_mask},
            {{"chained_fanout", DType::Float64, 1e-12, 1e-12}, run_chained_fanout},
        };
    }

    [[nodiscard]] bool parse_device(const std::string& s, Device& out) {
        if (s == "cpu" || s == "CPU") {
            out = Device::CPU;
            return true;
        }
        if (s == "gpu" || s == "GPU") {
            out = Device::GPU;
            return true;
        }
        return false;
    }

} // namespace

int main(const int argc, char** argv) {
    const auto args = std::vector<std::string>(argv + 1, argv + argc);
    auto out_dir = std::filesystem::path("comparison/generated-data");
    auto device = Device::CPU;
    auto smoke = false;

    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--out" && i + 1 < args.size()) {
            out_dir = args[i + 1];
            ++i;
        } else if (args[i] == "--device" && i + 1 < args.size()) {
            if (!parse_device(args[i + 1], device)) {
                std::cerr << "invalid device: " << args[i + 1] << "\n";
                return 1;
            }
            ++i;
        } else if (args[i] == "--smoke") {
            smoke = true;
        } else {
            std::cerr << "unknown arg: " << args[i] << "\n";
            return 1;
        }
    }

    if (
        const auto created = std::filesystem::create_directories(out_dir);
        !created && !std::filesystem::exists(out_dir)
    ) {
        std::cerr << "failed to create output directory: " << out_dir << "\n";
        return 1;
    }

    const auto manifest_path = out_dir / "manifest.txt";
    auto manifest = std::ofstream(manifest_path);
    if (!manifest) {
        std::cerr << "failed to open manifest: " << manifest_path << "\n";
        return 1;
    }

    const auto cases = all_cases(smoke);
    auto failed = int32_t{0};
    for (const auto& [cfg, fn] : cases) {
        try {
            fn(cfg, device, out_dir, manifest);
            std::cout << "ok: " << cfg.name << "\n";
        } catch (const std::exception& e) {
            std::cerr << "FAIL: " << cfg.name << ": " << e.what() << "\n";
            ++failed;
        }
    }

    manifest.close();
    if (failed != 0) {
        std::cerr << failed << " case(s) failed\n";
        return 1;
    }
    return 0;
}
