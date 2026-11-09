# Quickstart

This page is a complete, copy-paste MNIST training example. It mirrors the
repository's `main.cpp` but is stripped to a single CPU-only run so you can
build and run it immediately.

## Build

```bash
cd <your-beantensor-clone>
cmake -B build-cpu -S . -DUSE_CUDA=OFF -DUSE_HIP=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --target BeanTensorV5Core -j 6
```

Ninja is recommended if it is installed (`-G Ninja` on the configure line). The
example below links against the resulting `build-cpu/libBeanTensorV5Core.a`.

## MNIST training example

Save as `mnist_quickstart.cpp` in the repository root and fetch the MNIST data:

```bash
./fetch_mnist.sh
```

This places the uncompressed IDX files under `datasets/mnist/`. Then create the
source file:

```cpp
#include "BeanTensor.h"
#include "demos/Mnist.h"

#include <iostream>

using namespace BeanTensor;
using DType = DataTypes::DType;

int main() {
    rnd::set_seed(1234);

    auto train = demos::mnist::Split{};
    auto test = demos::mnist::Split{};
    const auto dir = std::filesystem::path{demos::mnist::kDir};
    if (!demos::mnist::load_split(dir, demos::mnist::kTrainImages, demos::mnist::kTrainLabels, train) ||
        !demos::mnist::load_split(dir, demos::mnist::kTestImages, demos::mnist::kTestLabels, test)) {
        std::cerr << "Place the uncompressed MNIST IDX files under " << dir << "/ (run ./fetch_mnist.sh)\n";
        return 1;
    }
    auto train_data = demos::mnist::upload_flat(train, Device::CPU, DType::Float32, DType::Float32);
    auto test_data = demos::mnist::upload_flat(test, Device::CPU, DType::Float32, DType::Float32);

    auto net = nn::Sequential::make(
        std::make_unique<nn::Linear>(784, 128),
        std::make_unique<nn::Relu>(),
        std::make_unique<nn::Linear>(128, 10));

    auto train_loader = data::DataLoader(train_data->images, train_data->labels, 100, true, 1234, true);
    auto test_loader = data::DataLoader(test_data->images, test_data->labels, 100, false, std::nullopt, true);

    for (auto epoch = size_t{0}; epoch < 3; ++epoch) {
        train_loader.reset();
        while (const auto batch = train_loader.next_batch()) {
            // Name the forward output: the loss's graph holds parents that
            // must outlive backward(), and a chained temporary would not.
            auto out = net.forward(batch->first);
            net.zero_grad();
            auto loss = nn::softmax_cross_entropy(out, batch->second);
            loss.backward(DType::Float32);
            net.adam_step(1e-3, 0.9, 0.999, 1e-8);
        }
        net.eval();
        const auto result = metrics::evaluate(net, test_loader);
        net.train();
        std::cout << "epoch " << (epoch + 1) << "/3  test accuracy "
                  << (result.accuracy * float64_t{100}) << "%\n";
    }
    return 0;
}
```

Compile and run:

```bash
./fetch_mnist.sh
g++ -std=c++23 -O2 -I. \
    mnist_quickstart.cpp \
    -Lbuild-cpu -lBeanTensorV5Core \
    -lpthread -o mnist_quickstart
./mnist_quickstart
```

You should see three epochs of training output ending with test accuracy around
97% on MNIST. See [`docs/mnist-results.md`](mnist-results.md) for the full
acceptance numbers.
