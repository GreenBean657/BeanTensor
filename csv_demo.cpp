#include "BeanTensor.h"

#include <fstream>
#include <iostream>
#include <string>

using namespace BeanTensor;
using DType = DataTypes::DType;

namespace {

    void write_file(const std::string& path, const std::string& content) {
        auto out = std::ofstream(path);
        out << content;
    }

    // Act 1: inspection — stats, nulls, scaling, and the label mapping.
    void inspection_act() {
        const auto path = std::string{"scratch/csv_demo_inspection.csv"};
        auto content = std::string{"length,width,noise,fruit\n"};
        for (auto i = int32_t{0}; i < 40; ++i) {
            const auto is_apple = (i % 2) == 0;
            const auto length = is_apple ? 6.0 + 0.1 * static_cast<float64_t>(i % 7)
                                         : 8.0 + 0.1 * static_cast<float64_t>(i % 5);
            const auto width = is_apple ? 7.0 + 0.1 * static_cast<float64_t>(i % 5)
                                        : 6.0 + 0.1 * static_cast<float64_t>(i % 7);
            const auto noise_cell = (i % 9 == 0) ? std::string{""}
                                                 : std::to_string(0.01 * static_cast<float64_t>(i));
            content += std::to_string(length) + "," + std::to_string(width) + "," +
                       noise_cell + "," + (is_apple ? "apple" : "pear") + "\n";
        }
        write_file(path, content);

        auto opts = data::CsvOptions{};
        opts.nulls = data::CsvNullPolicy::FillMean;
        opts.normalize = data::CsvNormalize::Standardize;
        auto ds = data::load_csv(path, opts);

        std::cout << "== inspection: " << ds.features.dim(0) << " rows x " << ds.features.dim(1)
                  << " features from " << path << "\n";
        std::cout << "null cells filled with column means: " << ds.null_count << "\n";
        for (size_t f = 0; f < ds.feature_names.size(); ++f) {
            std::cout << "  " << ds.feature_names[f]
                      << "  min " << ds.column_min[f]
                      << "  max " << ds.column_max[f]
                      << "  mean " << ds.column_mean[f] << "\n";
        }
        std::cout << "labels:";
        for (size_t i = 0; i < ds.label_names.size(); ++i) {
            std::cout << " [" << i << "]=" << ds.label_names[i];
        }
        std::cout << "\n\n";
    }

    // Act 2: training — the proven tiny recipe end to end.
    [[nodiscard]] bool training_act() {
        const auto path = std::string{"scratch/csv_demo_train.csv"};
        write_file(path,
                   "x1,x2,cls\n"
                   "0.0,0.0,a\n"
                   "0.1,0.0,a\n"
                   "0.0,0.1,a\n"
                   "1.0,1.0,b\n"
                   "1.0,0.9,b\n"
                   "0.9,1.0,b\n");

        auto opts = data::CsvOptions{};
        opts.feature_dtype = DType::Float32;
        auto ds = data::load_csv(path, opts);
        std::cout << "== training: " << ds.features.dim(0) << " rows from " << path
                  << " (labels";
        for (const auto& n : ds.label_names) {
            std::cout << " " << n;
        }
        std::cout << ")\n";

        auto loader = data::DataLoader(ds.features, ds.labels, 2, true, 5);
        auto net = nn::Sequential::make(std::make_unique<nn::Linear>(2, 2));

        auto first_loss = float64_t{0};
        auto last_loss = float64_t{0};
        for (auto step = int32_t{0}; step < 60; ++step) {
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
            loss.sync();
            const auto v = static_cast<float64_t>(loss.at<float32_t>({0}));
            if (step == 0) {
                first_loss = v;
            }
            last_loss = v;
        }

        net.eval();
        auto logits = net.forward(ds.features);
        auto predicted = logits.argmax(1);
        predicted.convert_dtype(DType::Int32);
        predicted.sync();
        auto mask = predicted.equal(ds.labels);
        mask.convert_dtype(DType::Float32);
        mask.sync();
        auto accuracy = mask.mean();
        accuracy.sync();

        std::cout << "loss " << first_loss << " -> " << last_loss
                  << "  accuracy " << (static_cast<float64_t>(accuracy.at<float32_t>({0})) * float64_t{100})
                  << "%\n";
        return last_loss < first_loss;
    }
}

int main() {
    rnd::set_seed(7);
    std::error_code ec;
    std::filesystem::create_directories("scratch", ec);

    inspection_act();
    if (!training_act()) {
        std::cerr << "csv demo: loss did not decrease\n";
        return 1;
    }
    return 0;
}
