# MNIST end-to-end acceptance results

## Model, optimizer, and data

- Architecture: `Sequential(Linear(784 -> 128), ReLU, Linear(128 -> 10))`
- Hidden activation: ReLU
- Loss: softmax cross-entropy against class labels
- Optimizer: Adam (`lr=1e-3`, `beta1=0.9`, `beta2=0.999`, `eps=1e-8`)
- Epochs: 3
- Batch size: 100
- Random seed: 1234 (process-wide RNG seeded before model construction; train DataLoader seeded explicitly)
- Data: uncompressed MNIST IDX files in `datasets/mnist/` (relative to the repository root; run `./fetch_mnist.sh` to download them)
- DType: Float32 for weights, activations, and gradients

## Results

| backend | test accuracy | test mean loss | wall time |
|--------:|--------------:|---------------:|----------:|
| BeanTensor CPU | 96.88% | 0.104828 | 5.88 s |
| BeanTensor CUDA | 96.95% | 0.101921 | 5.59 s |
| PyTorch CPU | 96.60% | 0.128132 | 1.887 s |
| PyTorch CUDA | N/A | N/A | N/A |

PyTorch CUDA is not available in the reference environment used for this run (`torch.cuda.is_available()` returned False), so no GPU number was produced.

## Repro commands

### BeanTensor CPU

```bash
cd <your-beantensor-clone>
./fetch_mnist.sh
cmake -B build-cpu -S . -DUSE_CUDA=OFF -DUSE_HIP=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build build-cpu --target BeanTensorV5 -j 6
./build-cpu/BeanTensorV5
```

### BeanTensor CUDA

```bash
cd <your-beantensor-clone>
./fetch_mnist.sh
cmake -B build-cuda -S . -DUSE_CUDA=ON -DUSE_HIP=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build build-cuda --target BeanTensorV5 -j 6
./gpu-test.sh mnist build-cuda -L mnist -V
```

### PyTorch CPU

```bash
cd <your-beantensor-clone>
python3 -m venv comparison/.venv
comparison/.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
comparison/.venv/bin/python comparison/mnist_pytorch.py
```

## Pass-bar verdict

- BeanTensor CPU (96.88%) ≥ PyTorch CPU (96.60%) − 1 pp: **PASS**.
- BeanTensor CUDA (96.95%) ≥ PyTorch CPU (96.60%) − 1 pp: **PASS**.
- BeanTensor CPU vs CUDA accuracy difference: 0.07 pp, within the 1 pp tolerance: **PASS**.
