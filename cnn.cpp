#include "BeanTensor.h"
#include "Helpers/Dashboard/Dashboard.h"
#include "demos/Mnist.h"

#include <cstdio>
#include <sstream>
#include <unistd.h>

using namespace BeanTensor;
using DType = DataTypes::DType;
using demos::mnist::Split;


namespace {

    // 28 -> conv3 -> 26 -> pool2 -> 13 -> conv3 -> 11 -> pool2 -> 5, so the
    // flatten width below is 16 * 5 * 5.
    constexpr size_t kChannels1 = 8;
    constexpr size_t kChannels2 = 16;
    constexpr size_t kFinalSide = 5;
    constexpr size_t kFlattened = kChannels2 * kFinalSide * kFinalSide;
    constexpr size_t kHidden = 64;
    constexpr size_t kClasses = 10;
    constexpr size_t kBatch = 128;
    constexpr size_t kEpochs = 2;
    constexpr float64_t kDropout = 0.25;
    constexpr float64_t kLearningRate = 0.001;
    constexpr float64_t kBeta1 = 0.9;
    constexpr float64_t kBeta2 = 0.999;
    constexpr float64_t kEpsilon = 1e-8;
    constexpr uint32_t kShuffleSeed = 1234;

    constexpr auto kDType = DType::Float32;
    constexpr auto gDType = DType::Float32;
    constexpr auto kHostDType = DType::Float32;

    // A CPU epoch over the full 60k-image set is minutes long; the ctest
    // entry uses a small subsample so the suite (and CI) stay fast, and the
    // recorded results run picks a size that finishes in sensible wall time.
    // The PyTorch counterpart takes the same flags for an apples-to-apples
    // comparison.
    struct RunConfig {
        size_t train_rows = 60000;
        size_t test_rows = 10000;
        size_t epochs = kEpochs;
        bool no_dashboard = false;
    };

    [[nodiscard]] RunConfig parse_args(const int argc, char** argv) {
        auto cfg = RunConfig{};
        for (auto i = int32_t{1}; i < argc; ++i) {
            const auto arg = std::string(argv[i]);
            const auto next = [&](const char* flag) -> std::optional<size_t> {
                if (arg == flag && i + 1 < argc) {
                    ++i;
                    return static_cast<size_t>(std::stoull(argv[i]));
                }
                return std::nullopt;
            };
            if (
                const auto v = next("--train-rows");
                v.has_value()
            ) {
                cfg.train_rows = *v;
            } else if (
                const auto v = next("--test-rows");
                v.has_value()
            ) {
                cfg.test_rows = *v;
            } else if (
                const auto v = next("--epochs");
                v.has_value()
            ) {
                cfg.epochs = *v;
            } else if (arg == "--no-dashboard") {
                cfg.no_dashboard = true;
            }
        }
        cfg.train_rows = std::min(cfg.train_rows, size_t{60000});
        cfg.test_rows = std::min(cfg.test_rows, size_t{10000});
        return cfg;
    }

    demos::mnist::Split subsample(const demos::mnist::Split& in, const size_t rows) {
        auto out = demos::mnist::Split{};
        out.count = std::min(rows, in.count);
        out.images.resize(out.count * demos::mnist::kPixels);
        out.labels.resize(out.count);
        std::copy_n(in.images.begin(), static_cast<std::ptrdiff_t>(out.count * demos::mnist::kPixels), out.images.begin());
        std::copy_n(in.labels.begin(), static_cast<std::ptrdiff_t>(out.count), out.labels.begin());
        return out;
    }

    [[nodiscard]] bool run_training(const demos::mnist::Split& train, const demos::mnist::Split& test,
                                    const Device device, const size_t epochs,
                                    Helpers::Dashboard& dash, std::vector<std::string>& epoch_lines,
                                    std::vector<std::string>& failure_lines) {
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "Device: " << Tensors::stringify_device_source(device) << "\n";
        std::cout << std::string(60, '=') << "\n";

        // Construct the layers as separate statements in layer order: the
        // Conv2ds and Linears draw their init seeds from the process-wide
        // default generator at construction, and one-expression
        // Sequential::make(...) would leave that draw order to the
        // compiler's (unspecified) argument evaluation order — GCC evaluates
        // right-to-left, which silently reassigns the seeds and changes the
        // training trajectory.
        auto conv1 = std::make_unique<nn::Conv2d>(1, kChannels1, 3, 1, 0, kDType, device);
        auto bn1 = std::make_unique<nn::BatchNorm>(kChannels1, kDType, device);
        auto relu1 = std::make_unique<nn::Relu>();
        auto pool1 = std::make_unique<nn::MaxPool2d>(2);
        auto conv2 = std::make_unique<nn::Conv2d>(kChannels1, kChannels2, 3, 1, 0, kDType, device);
        auto bn2 = std::make_unique<nn::BatchNorm>(kChannels2, kDType, device);
        auto relu2 = std::make_unique<nn::Relu>();
        auto pool2 = std::make_unique<nn::MaxPool2d>(2);
        auto flatten = std::make_unique<nn::Flatten>();
        auto drop = std::make_unique<nn::Dropout>(kDropout);
        auto fc1 = std::make_unique<nn::Linear>(kFlattened, kHidden, kDType, device);
        auto relu3 = std::make_unique<nn::Relu>();
        auto fc2 = std::make_unique<nn::Linear>(kHidden, kClasses, kDType, device);
        auto net = nn::Sequential::make(
            std::move(conv1),
            std::move(bn1),
            std::move(relu1),
            std::move(pool1),
            std::move(conv2),
            std::move(bn2),
            std::move(relu2),
            std::move(pool2),
            std::move(flatten),
            std::move(drop),
            std::move(fc1),
            std::move(relu3),
            std::move(fc2));
        std::cout << net;

        auto train_data = demos::mnist::upload_nchw(train, device, kDType, kHostDType);
        auto test_data = demos::mnist::upload_nchw(test, device, kDType, kHostDType);
        if (
            const auto uploaded = train_data.has_value() && test_data.has_value();
            !uploaded
        ) {
            // Live mode: the dtype message just printed sits inside the
            // dashboard's redraw region and is overwritten by the next frame
            // — defer a replacement diagnostic so it survives past the
            // persistent final frame stop() leaves on screen.
            if (dash.active()) {
                failure_lines.emplace_back("mnist upload failed (dtype mismatch reported during upload)\n");
            }
            return false;
        }

        auto train_loader = data::DataLoader(
            train_data->images,
            train_data->labels,
            kBatch,
            true,
            kShuffleSeed,
            true
        );
        auto test_loader = data::DataLoader(
            test_data->images,
            test_data->labels,
            kBatch,
            false,
            std::nullopt,
            true
        );

        const auto run_start = std::chrono::steady_clock::now();
        for (auto epoch = size_t{0}; epoch < epochs; ++epoch) {
            const auto epoch_start = std::chrono::steady_clock::now();
            train_loader.reset();
            auto loss_acc = Tensor(std::vector<size_t>{1}, kDType, device);
            loss_acc.set_zero();
            auto steps = size_t{0};
            while (true) {
                auto batch_opt = train_loader.next_batch();
                if (!batch_opt.has_value()) {
                    break;
                }
                auto& [batch, labels] = batch_opt.value();
                if (
                    const auto batch_ok = demos::mnist::expect_dtype(batch, kDType, "train: input batch");
                    !batch_ok
                ) {
                    if (dash.active()) {
                        auto msg = std::ostringstream{};
                        msg << "train: input batch has dtype " << DataTypes::dtype_to_string(batch.get_dtype())
                            << ", expected " << DataTypes::dtype_to_string(kDType) << "\n";
                        failure_lines.push_back(msg.str());
                    }
                    return false;
                }

                net.zero_grad();
                auto out = net.forward(batch);
                auto loss = nn::softmax_cross_entropy(out, labels);
                loss.backward(gDType);
                net.adam_step(kLearningRate, kBeta1, kBeta2, kEpsilon);

                if (
                    const auto loss_ok = demos::mnist::expect_dtype(loss, kDType, "train loss");
                    !loss_ok
                ) {
                    if (dash.active()) {
                        auto msg = std::ostringstream{};
                        msg << "train loss has dtype " << DataTypes::dtype_to_string(loss.get_dtype())
                            << ", expected " << DataTypes::dtype_to_string(kDType) << "\n";
                        failure_lines.push_back(msg.str());
                    }
                    return false;
                }
                loss_acc = loss_acc.add(loss);
                ++steps;
                // The per-batch scalar read costs a sync, so it only happens
                // when the dashboard is live; the inert path never pays it.
                if (dash.active()) {
                    dash.report_batch(demos::mnist::read_scalar(loss, kHostDType));
                }
            }
            // Drain pending optimizer/gradient work before the synchronous
            // eval pass; BatchNorm must also be in eval mode so the running
            // statistics (not batch statistics) drive the accuracy number.
            for (auto& [name, param] : net.named_parameters()) {
                static_cast<void>(name);
                param->sync();
            }
            net.eval();
            const auto result = metrics::evaluate(net, test_loader);
            net.train();
            const auto epoch_seconds = std::chrono::duration<float64_t>(std::chrono::steady_clock::now() - epoch_start).count();
            const auto train_mean_loss = demos::mnist::read_scalar(loss_acc, kHostDType) / static_cast<float64_t>(steps);
            auto epoch_line = std::ostringstream{};
            epoch_line << "epoch " << (epoch + 1) << "/" << epochs
                       << "  train mean loss " << train_mean_loss
                       << "  test accuracy " << (result.accuracy * float64_t{100}) << "%"
                       << "  test mean loss " << result.mean_loss
                       << "  time " << epoch_seconds << " s\n";
            if (dash.active()) {
                // Live mode defers the canonical epoch lines until stop()
                // persists its final frame; the dashboard gets the numeric record.
                dash.report_epoch(static_cast<uint32_t>(epoch + 1), static_cast<uint32_t>(epochs),
                                  train_mean_loss, result.accuracy, result.mean_loss);
                epoch_lines.push_back(epoch_line.str());
            } else {
                std::cout << epoch_line.str();
            }
        }
        const auto total_seconds = std::chrono::duration<float64_t>(std::chrono::steady_clock::now() - run_start).count();
        if (dash.active()) {
            // Live mode: nothing post-training may print while the dashboard
            // owns its region — defer the tail behind stop() like the epoch
            // lines, so the persisting final frame is never overwritten.
            auto tail = std::ostringstream{};
            tail << "total time " << total_seconds << " s\n";
            tail << "\nafter training:\n" << net;
            epoch_lines.push_back(tail.str());
        } else {
            std::cout << "total time " << total_seconds << " s\n";
            std::cout << "\nafter training:\n" << net;
        }
        return true;
    }
}

int main(const int argc, char** argv) {
    rnd::set_seed(kShuffleSeed);
    const auto cfg = [&] {
        try {
            return parse_args(argc, argv);
        } catch (const std::exception& e) {
            std::cerr << "arguments: [--train-rows N] [--test-rows N] [--epochs N] [--no-dashboard]\n";
            throw;
        }
    }();

    const auto dir = std::filesystem::path(demos::mnist::kDir);
    auto train = demos::mnist::Split{};
    auto test = demos::mnist::Split{};
    if (
        const auto loaded = demos::mnist::load_split(dir, demos::mnist::kTrainImages, demos::mnist::kTrainLabels, train)
            && demos::mnist::load_split(dir, demos::mnist::kTestImages, demos::mnist::kTestLabels, test);
        !loaded
    ) {
        std::cerr << "Place the uncompressed MNIST IDX files under " << dir.string() << "/ and rerun.\n";
        return 1;
    }
    std::cout << "Loaded " << train.count << " training and " << test.count << " test images\n";

    const auto train_sub = subsample(train, cfg.train_rows);
    const auto test_sub = subsample(test, cfg.test_rows);
    std::cout << "Using " << train_sub.count << " training and " << test_sub.count << " test images\n";

    // One dashboard for the whole run (CPU pass and, when compiled in, the
    // GPU pass). The TTY assertion is fd-level because the sink is std::cerr.
    auto dash = Helpers::Dashboard(std::cerr, isatty(fileno(stderr)) != 0 && !cfg.no_dashboard);
    auto epoch_lines = std::vector<std::string>{};
    auto failure_lines = std::vector<std::string>{};

    // stop() leaves the final frame on screen and prints its summary below
    // it; the deferred canonical epoch lines and any failure diagnostics
    // captured during the run follow, so nothing the live dashboard drew over
    // is lost.
    const auto finish = [&](const int code) {
        dash.stop();
        for (const auto& line : epoch_lines) {
            std::cout << line;
        }
        for (const auto& line : failure_lines) {
            std::cerr << line;
        }
        return code;
    };

    if (!run_training(train_sub, test_sub, Device::CPU, cfg.epochs, dash, epoch_lines, failure_lines)) {
        return finish(1);
    }

#if defined(USE_CUDA) || defined(USE_HIP)
    if (!run_training(train_sub, test_sub, Device::GPU, cfg.epochs, dash, epoch_lines, failure_lines)) {
        return finish(1);
    }
#else
    std::cout << "\nNo GPU backend compiled in; skipping GPU run.\n";
#endif

    return finish(0);
}
