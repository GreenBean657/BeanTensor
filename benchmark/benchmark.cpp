/**
 * @file benchmark.cpp
 * @author GreenBean
 * @brief Op-level performance harness: times one forward op per row across
 * the kernel families (elementwise add/mul, GEMM, reductions sum/max/argmax,
 * softmax_ce, conv2d, max_pool2d, batch_norm) at sizes derived from the
 * detected last-level cache (never from the dev box), warmup + median of N
 * repetitions. Two modes: --smoke (tiny sizes, one rep; wired into the test
 * suite so a broken benchmark fails loudly) and full (default), which prints
 * one CSV row per benchmark to stdout. --emit-shapes additionally prints the
 * concrete shape plan so the PyTorch comparison column
 * (comparison/bench_pytorch.py) benches identical sizes.
 * @version 0.1
 * @date 2026-11-06
 */
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "Helpers/Hardware.h"
#include "Tensor/DTypes.h"
#include "Tensor/Tensor.h"

using BeanTensor::Tensors::Device;
using BeanTensor::Tensors::Tensor;
using DType = BeanTensor::DataTypes::DType;
using BeanTensor::float64_t;

namespace {

    struct Row {
        std::string family;
        std::string op;
        std::vector<size_t> dims;
        float64_t median_ms = 0;
    };

    [[nodiscard]] float64_t median(std::vector<float64_t> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    }

    // Time one callable: one warmup iteration, then reps timed iterations,
    // median reported. Every iteration syncs so queued async work lands
    // inside its own measurement.
    template <typename F>
    [[nodiscard]] float64_t time_it(F&& fn, const int32_t reps) {
        fn();
        auto samples = std::vector<float64_t>();
        samples.reserve(static_cast<size_t>(reps));
        for (auto i = int32_t{0}; i < reps; ++i) {
            const auto start = std::chrono::steady_clock::now();
            fn();
            const auto end = std::chrono::steady_clock::now();
            samples.push_back(std::chrono::duration<float64_t>(end - start).count() * 1000.0);
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

    void print_row(const Row& row) {
        std::cout << row.family << "," << row.op << "," << dims_string(row.dims) << ","
                  << row.median_ms << "\n";
    }

    void print_shape(const Row& row) {
        std::cout << "shape " << row.family << " " << row.op;
        for (const auto d : row.dims) {
            std::cout << " " << d;
        }
        std::cout << "\n";
    }

    // Size plan: one small (cache-resident) and one large (~4x the LLC)
    // problem per family, floors so smoke machines stay meaningful.
    struct Plan {
        size_t vec_small = 0;
        size_t vec_large = 0;
        size_t gemm_m = 0;
        size_t gemm_k = 0;
        size_t gemm_n = 0;
        size_t gemm_m_large = 0;
        size_t rows = 0;
        size_t cols = 0;
        size_t classes = 0;
        size_t conv_n = 0;
        size_t conv_cin = 0;
        size_t conv_side = 0;
        size_t conv_cout = 0;
        size_t pool_n = 0;
        size_t pool_c = 0;
        size_t pool_side = 0;
        size_t bn_n = 0;
        size_t bn_c = 0;
        size_t bn_side = 0;
    };

    [[nodiscard]] Plan make_plan(const bool smoke) {
        if (smoke) {
            return Plan{64, 256, 8, 16, 8, 16, 4, 8, 3, 1, 1, 8, 8, 2, 2, 8, 1, 4, 4};
        }
        const auto llc = LastLevelCacheBytes();
        const auto elem = sizeof(float);
        // Vectors: ~1/16 LLC and ~4x LLC worth of float elements (two tensors
        // touched for add, so the resident estimate is approximate by design).
        const auto vec_small = std::max<size_t>(llc / 16 / elem, size_t{1} << 14);
        const auto vec_large = std::max<size_t>(llc * 4 / elem, size_t{1} << 20);
        const auto gemm_k = size_t{256};
        const auto gemm_n = size_t{256};
        const auto gemm_m = std::max<size_t>(llc / (gemm_k * elem) / 4, size_t{64});
        const auto gemm_m_large = gemm_m * 4;
        const auto cols = size_t{512};
        const auto rows = std::max<size_t>(llc / (cols * elem) / 4, size_t{64});
        const auto classes = size_t{10};
        return Plan{vec_small, vec_large, gemm_m, gemm_k, gemm_n, gemm_m_large,
                    rows, cols, classes,
                    /*conv_n=*/8, /*conv_cin=*/4, /*conv_side=*/56, /*conv_cout=*/16,
                    /*pool_n=*/16, /*pool_c=*/8, /*pool_side=*/28,
                    /*bn_n=*/16, /*bn_c=*/16, /*bn_side=*/14};
    }
}

int main(const int argc, char** argv) {
    const auto args = std::vector<std::string>(argv + 1, argv + argc);
    const auto smoke = std::find(args.begin(), args.end(), "--smoke") != args.end();
    const auto emit_shapes = std::find(args.begin(), args.end(), "--emit-shapes") != args.end();
    const auto reps = smoke ? 1 : 5;
    const auto plan = make_plan(smoke);
    const auto dt = DType::Float32;
    const auto device = Device::CPU;
    auto rows = std::vector<Row>{};

    auto rand2 = [&](const std::vector<size_t>& dims) {
        auto t = Tensor(dims, dt, device, false);
        t.random(-1.0, 1.0, 7);
        return t;
    };

    // Elementwise.
    for (const auto n : {plan.vec_small, plan.vec_large}) {
        auto a = rand2({n});
        auto b = rand2({n});
        rows.push_back({"elementwise", "add", {n}, time_it([&] { static_cast<void>(a.add(b)); }, reps)});
        rows.push_back({"elementwise", "mul", {n}, time_it([&] { static_cast<void>(a.mul(b)); }, reps)});
    }

    // GEMM.
    for (const auto m : {plan.gemm_m, plan.gemm_m_large}) {
        auto a = rand2({m, plan.gemm_k});
        auto b = rand2({plan.gemm_k, plan.gemm_n});
        rows.push_back({"gemm", "matmul", {m, plan.gemm_k, plan.gemm_n},
                        time_it([&] { static_cast<void>(a.matmul(b)); }, reps)});
    }

    // Reductions over rank-2 rows x cols.
    {
        auto a = rand2({plan.rows, plan.cols});
        rows.push_back({"reduction", "sum", {plan.rows, plan.cols},
                        time_it([&] { static_cast<void>(a.sum()); }, reps)});
        rows.push_back({"reduction", "max", {plan.rows, plan.cols},
                        time_it([&] { static_cast<void>(a.max()); }, reps)});
        rows.push_back({"reduction", "argmax1", {plan.rows, plan.cols},
                        time_it([&] { static_cast<void>(a.argmax(1)); }, reps)});
    }

    // Loss.
    {
        auto logits = rand2({plan.rows, plan.classes});
        auto labels = Tensor(std::vector<size_t>{plan.rows}, DType::Int32, device, false);
        for (auto i = int32_t{0}; i < static_cast<int32_t>(plan.rows); ++i) {
            const auto cls = i % static_cast<int32_t>(plan.classes);
            labels.write_element(static_cast<size_t>(i), &cls, sizeof(int32_t));
        }
        rows.push_back({"loss", "softmax_ce", {plan.rows, plan.classes},
                        time_it([&] { static_cast<void>(logits.softmax_ce(labels)); }, reps)});
    }

    // Convolution forward: NCHW input, {R,S,Cin,Cout} weight, bias.
    {
        auto x = rand2({plan.conv_n, plan.conv_cin, plan.conv_side, plan.conv_side});
        auto w = rand2({plan.conv_cin, 3, 3, plan.conv_cout});
        auto bias = rand2({plan.conv_cout});
        rows.push_back({"convolution", "conv2d",
                        {plan.conv_n, plan.conv_cin, plan.conv_side, plan.conv_side, plan.conv_cout, 3, 3},
                        time_it([&] { static_cast<void>(x.conv2d(w, bias, 1, 0)); }, reps)});
    }

    // Spatial pooling.
    {
        auto x = rand2({plan.pool_n, plan.pool_c, plan.pool_side, plan.pool_side});
        rows.push_back({"spatial", "max_pool2d", {plan.pool_n, plan.pool_c, plan.pool_side, plan.pool_side},
                        time_it([&] { static_cast<void>(x.max_pool2d(2, 2)); }, reps)});
    }

    // Batch normalization forward (train mode).
    {
        auto x = rand2({plan.bn_n, plan.bn_c, plan.bn_side, plan.bn_side});
        auto gamma = rand2({plan.bn_c});
        auto beta = rand2({plan.bn_c});
        auto run_mean = Tensor(std::vector<size_t>{plan.bn_c}, dt, device, false);
        auto run_var = Tensor(std::vector<size_t>{plan.bn_c}, dt, device, false);
        run_mean.set_zero();
        run_var.set_one();
        rows.push_back({"normalization", "batch_norm", {plan.bn_n, plan.bn_c, plan.bn_side, plan.bn_side},
                        time_it([&] {
                            static_cast<void>(x.batch_norm(gamma, beta, run_mean, run_var, true, 0.1, 1e-5));
                        }, reps)});
    }

    for (const auto& row : rows) {
        if (emit_shapes) {
            print_shape(row);
        } else {
            print_row(row);
        }
    }
    if (smoke) {
        // Every family must produce at least one row: a broken benchmark is
        // a test failure, not a silent skip.
        const auto families = {"elementwise", "gemm", "reduction", "loss", "convolution", "spatial", "normalization"};
        for (const auto* fam : families) {
            const auto found = std::any_of(rows.begin(), rows.end(), [&](const Row& r) { return r.family == fam; });
            if (!found) {
                std::cerr << "benchmark smoke: family produced no rows: " << fam << "\n";
                return 1;
            }
        }
        std::cerr << "benchmark smoke ok (" << rows.size() << " rows)\n";
    }
    return 0;
}
