/**
 * @file gradient_parity_perf.cpp
 * @author GreenBean
 * @brief Gradient parity performance harness. Reads the shape plan emitted by
 * the forward op benchmark (`BeanTensorV5Benchmarks --emit-shapes`) and times
 * one forward + one backward pass for each differentiable family. Output is a
 * CSV matching `comparison/gradient_parity_perf.py` so the runner can build a
 * side-by-side BeanTensor vs PyTorch table.
 * @version 0.1
 * @date 2026-11-14
 */
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "BeanTensor.h"
#include "Helpers/Errors.h"

using BeanTensor::DataTypes::DType;
using BeanTensor::Tensors::Device;
using BeanTensor::Tensors::Tensor;
using BeanTensor::float32_t;
using BeanTensor::float64_t;

using BeanTensor::ErrorHandling::NotImplemented;

namespace {

    [[nodiscard]] float64_t median(std::vector<float64_t> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    }

    template <typename F>
    [[nodiscard]] float64_t time_it(F&& fn, const int32_t reps) {
        fn();
        auto samples = std::vector<float64_t>();
        samples.reserve(static_cast<size_t>(reps));
        for (auto i = int32_t{0}; i < reps; ++i) {
            const auto start = std::chrono::steady_clock::now();
            fn();
            const auto end = std::chrono::steady_clock::now();
            samples.push_back(
                std::chrono::duration<float64_t>(end - start).count() * float64_t{1000.0}
            );
        }
        return median(std::move(samples));
    }

    [[nodiscard]] std::string dims_string(const std::vector<size_t>& dims) {
        auto out = std::string{};
        for (size_t i = 0; i < dims.size(); ++i) {
            if (i != 0) {
                out += "x";
            }
            out += std::to_string(dims[i]);
        }
        return out;
    }

    [[nodiscard]] Tensor rand_tensor(const std::vector<size_t>& shape, const DType dt,
                                     const Device device, const uint64_t seed) {
        auto t = Tensor(shape, dt, device, true);
        t.random(float64_t{-1.0}, float64_t{1.0}, static_cast<int64_t>(seed));
        return t;
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

    [[nodiscard]] std::optional<size_t> parse_size_t(const std::string& s) {
        uint64_t value = 0;
        const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), value);
        if (ec != std::errc{} || ptr != s.data() + s.size()) {
            return std::nullopt;
        }
        return static_cast<size_t>(value);
    }

    void sync_loss_and_leaves(Tensor& loss, const std::vector<Tensor*>& leaves) {
        loss.sync();
        for (auto* leaf : leaves) {
            leaf->gradient().sync();
        }
    }

    void print_row(const std::string& family, const std::string& op,
                   const std::vector<size_t>& dims, const float64_t median_ms) {
        std::cout << family << "," << op << "," << dims_string(dims) << ","
                  << std::setprecision(6) << median_ms << "\n";
    }

    void run_elementwise(const std::string& op, const std::vector<size_t>& dims,
                         const Device device, const int32_t reps) {
        const auto n = dims[0];
        auto a = rand_tensor({n}, DType::Float32, device, 7);
        auto b = rand_tensor({n}, DType::Float32, device, 8);

        const auto median_ms = time_it(
            [&] {
                auto tmp = op == "add" ? a.add(b) : a.mul(b);
                auto loss = tmp.sum();
                loss.backward(DType::Float32);
                sync_loss_and_leaves(loss, {&a, &b});
            },
            reps
        );
        print_row("elementwise", op, dims, median_ms);
    }

    void run_gemm(const std::string& op, const std::vector<size_t>& dims,
                  const Device device, const int32_t reps) {
        const auto m = dims[0];
        const auto k = dims[1];
        const auto n = dims[2];
        auto a = rand_tensor({m, k}, DType::Float32, device, 7);
        auto b = rand_tensor({k, n}, DType::Float32, device, 8);

        const auto median_ms = time_it(
            [&] {
                auto tmp = a.matmul(b);
                auto loss = tmp.sum();
                loss.backward(DType::Float32);
                sync_loss_and_leaves(loss, {&a, &b});
            },
            reps
        );
        print_row("gemm", op, dims, median_ms);
    }

    void run_reduction(const std::string& op, const std::vector<size_t>& dims,
                       const Device device, const int32_t reps) {
        const auto rows = dims[0];
        const auto cols = dims[1];
        auto a = rand_tensor({rows, cols}, DType::Float32, device, 7);

        const auto median_ms = time_it(
            [&] {
                auto loss = op == "sum" ? a.sum() : a.max();
                loss.backward(DType::Float32);
                sync_loss_and_leaves(loss, {&a});
            },
            reps
        );
        print_row("reduction", op, dims, median_ms);
    }

    void run_loss(const std::string& op, const std::vector<size_t>& dims,
                  const Device device, const int32_t reps) {
        const auto rows = dims[0];
        const auto classes = dims[1];
        auto logits = rand_tensor({rows, classes}, DType::Float32, device, 7);
        auto labels = Tensor(std::vector<size_t>{rows}, DType::Int32, device, false);
        auto gen = BeanTensor::Random::Generator(9);
        for (size_t i = 0; i < rows; ++i) {
            const auto cls = static_cast<int32_t>(gen.draw_seed() % classes);
            labels.write_element(i, &cls, sizeof(int32_t));
        }

        const auto median_ms = time_it(
            [&] {
                auto loss = logits.softmax_ce(labels);
                loss.backward(DType::Float32);
                sync_loss_and_leaves(loss, {&logits});
            },
            reps
        );
        print_row("loss", op, dims, median_ms);
    }

    void run_convolution(const std::string& op, const std::vector<size_t>& dims,
                         const Device device, const int32_t reps) {
        const auto n = dims[0];
        const auto cin = dims[1];
        const auto side = dims[2];
        const auto cout = dims[4];
        const auto k = dims[5];
        auto x = rand_tensor({n, cin, side, side}, DType::Float32, device, 7);
        auto w = rand_tensor({cin, k, k, cout}, DType::Float32, device, 8);
        auto bias = rand_tensor({cout}, DType::Float32, device, 9);

        const auto median_ms = time_it(
            [&] {
                auto tmp = x.conv2d(w, bias, 1, 0);
                auto loss = tmp.sum();
                loss.backward(DType::Float32);
                sync_loss_and_leaves(loss, {&x, &w, &bias});
            },
            reps
        );
        print_row("convolution", op, dims, median_ms);
    }

    void run_spatial(const std::string& op, const std::vector<size_t>& dims,
                     const Device device, const int32_t reps) {
        const auto n = dims[0];
        const auto c = dims[1];
        const auto side = dims[2];
        auto x = rand_tensor({n, c, side, side}, DType::Float32, device, 7);

        const auto median_ms = time_it(
            [&] {
                auto tmp = x.max_pool2d(2, 2);
                auto loss = tmp.sum();
                loss.backward(DType::Float32);
                sync_loss_and_leaves(loss, {&x});
            },
            reps
        );
        print_row("spatial", op, dims, median_ms);
    }

    void run_normalization(const std::string& op, const std::vector<size_t>& dims,
                           const Device device, const int32_t reps) {
        const auto n = dims[0];
        const auto c = dims[1];
        const auto side = dims[2];
        auto x = rand_tensor({n, c, side, side}, DType::Float32, device, 7);
        auto gamma = rand_tensor({c}, DType::Float32, device, 8);
        auto beta = rand_tensor({c}, DType::Float32, device, 9);
        auto run_mean = Tensor(std::vector<size_t>{c}, DType::Float32, device, false);
        auto run_var = Tensor(std::vector<size_t>{c}, DType::Float32, device, false);
        run_mean.set_zero();
        run_var.set_one();

        const auto median_ms = time_it(
            [&] {
                auto tmp = x.batch_norm(gamma, beta, run_mean, run_var, true, 0.1, 1e-5);
                auto loss = tmp.sum();
                loss.backward(DType::Float32);
                sync_loss_and_leaves(loss, {&x, &gamma, &beta});
            },
            reps
        );
        print_row("normalization", op, dims, median_ms);
    }

    void run_case(const std::string& family, const std::string& op,
                  const std::vector<size_t>& dims, const Device device,
                  const int32_t reps) {
        if (family == "elementwise" && (op == "add" || op == "mul")) {
            run_elementwise(op, dims, device, reps);
        } else if (family == "gemm" && op == "matmul") {
            run_gemm(op, dims, device, reps);
        } else if (family == "reduction" && op == "sum") {
            run_reduction(op, dims, device, reps);
        } else if (family == "loss" && op == "softmax_ce") {
            run_loss(op, dims, device, reps);
        } else if (family == "convolution" && op == "conv2d") {
            run_convolution(op, dims, device, reps);
        } else if (family == "spatial" && op == "max_pool2d") {
            run_spatial(op, dims, device, reps);
        } else if (family == "normalization" && op == "batch_norm") {
            run_normalization(op, dims, device, reps);
        }
    }

} // namespace

int main(const int argc, char** argv) {
    const auto args = std::vector<std::string>(argv + 1, argv + argc);
    auto shapes_path = std::filesystem::path("benchmark/shapes.txt");
    auto device = Device::CPU;
    auto reps = int32_t{5};
    auto smoke = false;

    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--shapes" && i + 1 < args.size()) {
            shapes_path = args[i + 1];
            ++i;
        } else if (args[i] == "--device" && i + 1 < args.size()) {
            if (!parse_device(args[i + 1], device)) {
                std::cerr << "invalid device: " << args[i + 1] << "\n";
                return 1;
            }
            ++i;
        } else if (args[i] == "--reps" && i + 1 < args.size()) {
            if (
                const auto parsed = parse_size_t(args[i + 1]);
                parsed.has_value()
            ) {
                reps = static_cast<int32_t>(parsed.value());
            } else {
                std::cerr << "invalid reps: " << args[i + 1] << "\n";
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

    if (smoke) {
        reps = 1;
    }

    auto file = std::ifstream(shapes_path);
    if (!file) {
        std::cerr << "cannot open shapes file: " << shapes_path << "\n";
        return 1;
    }

    std::string line;
    while (std::getline(file, line)) {
        auto parts = std::vector<std::string>{};
        {
            auto iss = std::istringstream(line);
            auto token = std::string{};
            while (iss >> token) {
                parts.push_back(token);
            }
        }
        if (parts.size() < 4 || parts[0] != "shape") {
            continue;
        }
        const auto& family = parts[1];
        const auto& op = parts[2];
        auto dims = std::vector<size_t>();
        dims.reserve(parts.size() - 3);
        bool dims_ok = true;
        for (size_t i = 3; i < parts.size(); ++i) {
            if (const auto parsed = parse_size_t(parts[i]); parsed.has_value()) {
                dims.push_back(parsed.value());
            } else {
                dims_ok = false;
                break;
            }
        }
        if (!dims_ok) {
            continue;
        }
        run_case(family, op, dims, device, reps);
    }
    return 0;
}
