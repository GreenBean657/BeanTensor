/**
 * @file csv_loader_tests.cpp
 * @author GreenBean
 * @brief CSV dataset-loading tests - framework-free assertion suite. Covers
 * the parsing contract end to end: header on/off/auto, delimiter inference
 * and override, numeric and categorical columns, label-name mapping
 * stability, the NaN/empty-cell policy, every named rejection (missing
 * file, empty file, ragged rows, non-numeric cells), dtype selection
 * including integer-kind rejection of fractional values, and a full
 * CSV -> DataLoader -> Linear -> optimizer-step composition proving the
 * loaded pair plugs straight into the training stack.
 * @version 0.2
 * @date 2026-11-10
 */
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "Data/Csv.h"
#include "Data/DataLoader.h"
#include "Helpers/Errors.h"
#include "Module/Linear.h"
#include "Module/Relu.h"
#include "Module/Sequential.h"
#include "Tensor/DTypes.h"
#include "Tensor/Tensor.h"

using BeanTensor::Tensors::Device;
using BeanTensor::Tensors::Tensor;
using BeanTensor::Data::CsvDelimiter;
using BeanTensor::Data::CsvOptions;
using BeanTensor::Data::load_csv;
using DType = BeanTensor::DataTypes::DType;
namespace eh = BeanTensor::ErrorHandling;
using BeanTensor::float32_t;
using BeanTensor::float64_t;

static int32_t failures = 0;
#define CHECK(...) do { \
    if (!(__VA_ARGS__)) { ++failures; std::cerr << "FAIL: " << #__VA_ARGS__ << " @ line " << __LINE__ << "\n"; } \
} while (0)

static void write_csv(const std::string& path, const std::string& content) {
    std::ofstream out(path);
    out << content;
}

static std::string scratch_path(const std::string& name) {
    std::error_code ec;
    std::filesystem::create_directories("scratch", ec);
    return std::string{"scratch/csv_test_"} + name + ".csv";
}

static void test_roundtrip_and_header() {
    const auto path = scratch_path("basic");
    write_csv(path,
              "age,height,label\n"
              "25,1.80,cat\n"
              "30,1.65,dog\n"
              "35,1.70,cat\n");
    auto ds = load_csv(path);
    CHECK(ds.features.rank() == 2);
    CHECK(ds.features.dim(0) == 3);
    CHECK(ds.features.dim(1) == 2);
    CHECK(ds.features.get_dtype() == DType::Float64);
    CHECK(ds.labels.get_dtype() == DType::Int32);
    CHECK(ds.labels.dim(0) == 3);
    CHECK(std::fabs(ds.features.at<float64_t>({0, 0}) - 25.0) < 1e-9);
    CHECK(std::fabs(ds.features.at<float64_t>({0, 1}) - 1.80) < 1e-9);
    CHECK(std::fabs(ds.features.at<float64_t>({2, 1}) - 1.70) < 1e-9);
    CHECK(ds.labels.at<int32_t>({0}) == 0);
    CHECK(ds.labels.at<int32_t>({1}) == 1);
    CHECK(ds.labels.at<int32_t>({2}) == 0);
    CHECK(ds.label_names.size() == 2);
    CHECK(ds.label_names[0] == "cat");
    CHECK(ds.label_names[1] == "dog");
    CHECK(ds.feature_names.size() == 2);
    CHECK(ds.feature_names[0] == "age");
    CHECK(ds.feature_names[1] == "height");
    std::cout << "csv roundtrip + header ok\n";
}

static void test_no_header_and_synthetic_names() {
    const auto path = scratch_path("noheader");
    write_csv(path,
              "1.0,2.0,0\n"
              "3.0,4.0,1\n");
    auto opts = CsvOptions{};
    opts.has_header = false;
    auto ds = load_csv(path, opts);
    CHECK(ds.features.dim(0) == 2);
    CHECK(ds.features.dim(1) == 2);
    CHECK(ds.feature_names[0] == "col0");
    CHECK(ds.feature_names[1] == "col1");
    CHECK(std::fabs(ds.features.at<float64_t>({1, 1}) - 4.0) < 1e-9);
    std::cout << "csv no-header ok\n";
}

static void test_delimiter_inference_and_override() {
    {
        const auto path = scratch_path("semi");
        write_csv(path, "a;b;l\n1;2;x\n3;4;y\n");
        auto ds = load_csv(path);
        CHECK(ds.features.dim(1) == 2);
        CHECK(ds.label_names[0] == "x");
    }
    {
        const auto path = scratch_path("tab");
        write_csv(path, "a\tb\tl\n1\t2\t0\n");
        auto ds = load_csv(path);
        CHECK(ds.features.dim(1) == 2);
        CHECK(std::fabs(ds.features.at<float64_t>({0, 1}) - 2.0) < 1e-9);
    }
    {
        // A semicolon file misread as comma must produce ragged rejection,
        // and the explicit override must fix it.
        const auto path = scratch_path("override");
        write_csv(path, "a;b;l\n1;2;x\n");
        auto opts = CsvOptions{};
        opts.delimiter = CsvDelimiter::Semicolon;
        auto ds = load_csv(path, opts);
        CHECK(ds.features.dim(1) == 2);
    }
    std::cout << "csv delimiter inference + override ok\n";
}

static void test_categorical_mapping_stability() {
    // ids follow first-seen order and are stable regardless of row order.
    const auto path = scratch_path("mapping");
    write_csv(path, "f,l\n1,b\n2,a\n3,b\n4,a\n");
    auto ds = load_csv(path);
    CHECK(ds.label_names[0] == "b");
    CHECK(ds.label_names[1] == "a");
    CHECK(ds.labels.at<int32_t>({0}) == 0);
    CHECK(ds.labels.at<int32_t>({1}) == 1);
    CHECK(ds.labels.at<int32_t>({2}) == 0);
    std::cout << "csv categorical mapping ok\n";
}

static void test_nan_and_empty_policy() {
    const auto path = scratch_path("nan");
    write_csv(path, "a,b,l\n1,,x\n,4,y\n");
    auto ds = load_csv(path);
    CHECK(std::isnan(ds.features.at<float64_t>({0, 1})));
    CHECK(std::isnan(ds.features.at<float64_t>({1, 0})));
    CHECK(std::fabs(ds.features.at<float64_t>({0, 0}) - 1.0) < 1e-9);
    std::cout << "csv nan policy ok\n";
}

template <typename Ex>
static bool throws_on(const std::string& name, const std::string& content, const CsvOptions& opts = {}) {
    const auto path = scratch_path(name);
    write_csv(path, content);
    try {
        (void)load_csv(path, opts);
    } catch (const Ex&) {
        return true;
    }
    return false;
}

static void test_all_rejections() {
    bool threw = false;
    try {
        (void)load_csv("scratch/definitely_not_here.csv");
    } catch (const eh::SerializationError&) {
        threw = true;
    }
    CHECK(threw);

    CHECK(throws_on<eh::EmptyFile>("emptyfile", "\n\n  \n", {}));
    CHECK(throws_on<eh::RaggedCsv>("ragged", "a,b,l\n1,2,x\n3,4\n", {}));
    CHECK(throws_on<eh::NonNumericValue>("nonnumeric", "a,l\nhello,x\n", CsvOptions{}));
    // A categorical column outside the label position is a contract
    // violation: feature columns must be numeric.
    CHECK(throws_on<eh::NonNumericValue>("categorical_feature", "a,city,l\n1,paris,x\n", {}));

    // Header override on a file whose first row names the columns.
    auto force_header = CsvOptions{};
    force_header.has_header = true;
    const auto forced = scratch_path("forced");
    write_csv(forced, "a,l\n1,x\n2,y\n");
    auto ds = load_csv(forced, force_header);
    CHECK(ds.features.dim(0) == 2);
    std::cout << "csv rejections ok\n";
}

static void test_dtype_selection() {
    const auto path = scratch_path("dtype");
    write_csv(path, "a,l\n1,x\n2,y\n");
    auto opts = CsvOptions{};
    opts.feature_dtype = DType::Float32;
    auto ds = load_csv(path, opts);
    CHECK(ds.features.get_dtype() == DType::Float32);
    CHECK(std::fabs(ds.features.at<float32_t>({1, 0}) - 2.0f) < 1e-6f);

    auto int_opts = CsvOptions{};
    int_opts.feature_dtype = DType::Int32;
    auto ids = load_csv(path, int_opts);
    CHECK(ids.features.get_dtype() == DType::Int32);
    CHECK(ids.features.at<int32_t>({0, 0}) == 1);

    // Fractional values cannot satisfy an integer feature dtype.
    const auto frac_path = scratch_path("frac");
    write_csv(frac_path, "a,l\n1.5,x\n");
    auto frac_opts = CsvOptions{};
    frac_opts.feature_dtype = DType::Int32;
    bool threw = false;
    try {
        (void)load_csv(frac_path, frac_opts);
    } catch (const eh::NonNumericValue&) {
        threw = true;
    }
    CHECK(threw);
    std::cout << "csv dtype selection ok\n";
}

static void test_end_to_end_training() {
    const auto path = scratch_path("train");
    write_csv(path,
              "x1,x2,cls\n"
              "0.0,0.0,a\n"
              "0.1,0.0,a\n"
              "0.0,0.1,a\n"
              "1.0,1.0,b\n"
              "1.0,0.9,b\n"
              "0.9,1.0,b\n");
    auto opts = CsvOptions{};
    opts.feature_dtype = DType::Float32;
    auto ds = load_csv(path, opts);

    // Seeded init: an unseeded Linear draws from the process-wide Generator
    // (random_device-seeded), which made this test's outcome vary per run.
    auto net = BeanTensor::Modules::Sequential();
    net.append("fc", std::make_unique<BeanTensor::Modules::Linear>(2, 2, DType::Float32, Device::CPU, true, 42));

    // Progress is measured on the full dataset before vs after training.
    // Comparing two arbitrary shuffled batches compares batch difficulty,
    // not learning — a well-trained net can score worse on a harder batch.
    const auto eval_loss = [&net, &ds]() {
        auto logits = net.forward(ds.features);
        auto loss = logits.softmax_ce(ds.labels);
        loss.sync();
        return static_cast<float64_t>(loss.at<float32_t>({0}));
    };
    const auto before = eval_loss();

    auto loader = BeanTensor::Data::DataLoader(ds.features, ds.labels, 2, true, 5);
    for (auto step = int32_t{0}; step < 12; ++step) {
        auto batch = loader.next_batch();
        if (!batch.has_value()) {
            loader.reset();
            batch = loader.next_batch();
        }
        net.zero_grad();
        auto logits = net.forward(batch->first);
        auto loss = logits.softmax_ce(batch->second);
        loss.backward(DType::Float32);
        net.adam_step(1e-2, 0.9, 0.999, 1e-8);
    }
    const auto after = eval_loss();
    CHECK(after < before);
    std::cout << "csv end-to-end training ok (" << before << " -> " << after << ")\n";
}


// ---- Null policies, stats, and normalization ----

static void test_null_count_and_stats() {
    const auto path = scratch_path("nullstats");
    write_csv(path, "a,b,l\n1,10,x\n,20,y\n3,,x\n");
    auto opts = CsvOptions{};
    opts.nulls = BeanTensor::Data::CsvNullPolicy::KeepNaN;
    auto ds = load_csv(path, opts);
    CHECK(ds.null_count == 2);
    CHECK(ds.column_min.size() == 2);
    CHECK(std::fabs(ds.column_min[0] - 1.0) < 1e-9);
    CHECK(std::fabs(ds.column_max[0] - 3.0) < 1e-9);
    CHECK(std::fabs(ds.column_mean[0] - 2.0) < 1e-9);   // NaN-skipped
    CHECK(std::fabs(ds.column_min[1] - 10.0) < 1e-9);
    CHECK(std::fabs(ds.column_max[1] - 20.0) < 1e-9);
    CHECK(std::isnan(ds.features.at<float64_t>({1, 0})));
    CHECK(std::isnan(ds.features.at<float64_t>({2, 1})));
    std::cout << "csv null count + stats ok\n";
}

static void test_null_error_policy() {
    const auto path = scratch_path("nullerr");
    write_csv(path, "a,l\n1,x\n,y\n");
    auto opts = CsvOptions{};
    opts.nulls = BeanTensor::Data::CsvNullPolicy::Error;
    bool threw = false;
    try {
        (void)load_csv(path, opts);
    } catch (const eh::NullValue&) {
        threw = true;
    }
    CHECK(threw);
    std::cout << "csv null error policy ok\n";
}

static void test_null_fill_policies() {
    using BeanTensor::Data::CsvNullPolicy;
    {
        const auto path = scratch_path("fillmean");
        write_csv(path, "a,l\n1,x\n,y\n3,z\n");
        auto opts = CsvOptions{};
        opts.nulls = CsvNullPolicy::FillMean;
        auto ds = load_csv(path, opts);
        CHECK(!std::isnan(ds.features.at<float64_t>({1, 0})));
        CHECK(std::fabs(ds.features.at<float64_t>({1, 0}) - 2.0) < 1e-9);
    }
    {
        const auto path = scratch_path("fillmedian");
        write_csv(path, "a,l\n1,x\n,y\n10,z\n4,w\n");
        auto opts = CsvOptions{};
        opts.nulls = CsvNullPolicy::FillMedian;
        auto ds = load_csv(path, opts);
        CHECK(std::fabs(ds.features.at<float64_t>({1, 0}) - 4.0) < 1e-9);
    }
    {
        const auto path = scratch_path("fillconst");
        write_csv(path, "a,l\n1,x\n,y\n");
        auto opts = CsvOptions{};
        opts.nulls = CsvNullPolicy::FillConstant;
        opts.fill_value = 0.5;
        auto ds = load_csv(path, opts);
        CHECK(std::fabs(ds.features.at<float64_t>({1, 0}) - 0.5) < 1e-9);
    }
    std::cout << "csv null fill policies ok\n";
}

static void test_normalize_minmax() {
    const auto path = scratch_path("normmm");
    write_csv(path, "a,b,l\n1,7,x\n5,7,y\n3,7,z\n");
    auto opts = CsvOptions{};
    opts.normalize = BeanTensor::Data::CsvNormalize::MinMax;
    auto ds = load_csv(path, opts);
    CHECK(std::fabs(ds.features.at<float64_t>({0, 0}) - 0.0) < 1e-9);
    CHECK(std::fabs(ds.features.at<float64_t>({1, 0}) - 1.0) < 1e-9);
    CHECK(std::fabs(ds.features.at<float64_t>({2, 0}) - 0.5) < 1e-9);
    // constant column maps to all zeros
    CHECK(std::fabs(ds.features.at<float64_t>({0, 1}) - 0.0) < 1e-9);
    CHECK(std::fabs(ds.features.at<float64_t>({1, 1}) - 0.0) < 1e-9);
    // labels untouched
    CHECK(ds.labels.at<int32_t>({0}) == 0);
    CHECK(ds.label_names[0] == "x");
    std::cout << "csv minmax normalization ok\n";
}

static void test_normalize_standardize() {
    const auto path = scratch_path("normstd");
    write_csv(path, "a,l\n1,x\n2,y\n3,z\n");
    auto opts = CsvOptions{};
    opts.normalize = BeanTensor::Data::CsvNormalize::Standardize;
    auto ds = load_csv(path, opts);
    // Population standard deviation of [1,2,3] is sqrt(2/3), so the z-scores
    // are +/-sqrt(3/2), not +/-1 (that would be the sample std).
    const auto expect = std::sqrt(1.5);
    CHECK(std::fabs(ds.features.at<float64_t>({0, 0}) - (-expect)) < 1e-9);
    CHECK(std::fabs(ds.features.at<float64_t>({1, 0}) - 0.0) < 1e-9);
    CHECK(std::fabs(ds.features.at<float64_t>({2, 0}) - expect) < 1e-9);
    std::cout << "csv standardize normalization ok\n";
}

static void test_normalize_with_int_dtype() {
    const auto path = scratch_path("normint");
    write_csv(path, "a,l\n0,0\n10,1\n5,0\n");
    auto opts = CsvOptions{};
    opts.normalize = BeanTensor::Data::CsvNormalize::MinMax;
    opts.feature_dtype = DType::Int32;
    auto ds = load_csv(path, opts);
    CHECK(ds.features.at<int32_t>({0, 0}) == 0);
    CHECK(ds.features.at<int32_t>({1, 0}) == 1);
    CHECK(ds.features.at<int32_t>({2, 0}) == 0);
    std::cout << "csv normalize + int dtype ok\n";
}

int main() {
    test_roundtrip_and_header();
    test_no_header_and_synthetic_names();
    test_delimiter_inference_and_override();
    test_categorical_mapping_stability();
    test_nan_and_empty_policy();
    test_all_rejections();
    test_dtype_selection();
    test_null_count_and_stats();
    test_null_error_policy();
    test_null_fill_policies();
    test_normalize_minmax();
    test_normalize_standardize();
    test_normalize_with_int_dtype();
    test_end_to_end_training();
    if (failures == 0) {
        std::cout << "ALL CSV TESTS PASSED\n";
        return 0;
    }
    std::cerr << failures << " FAILURE(S)\n";
    return 1;
}
