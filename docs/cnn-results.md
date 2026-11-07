# CNN end-to-end acceptance results (Task 5)

## Model, optimizer, and data

- Architecture: `Conv2d(1->8, 3) -> BatchNorm(8) -> Relu -> MaxPool2d(2)
  -> Conv2d(8->16, 3) -> BatchNorm(16) -> Relu -> MaxPool2d(2)
  -> Flatten -> Dropout(0.25) -> Linear(400->64) -> Relu -> Linear(64->10)`
- Loss: softmax cross-entropy against class labels; optimizer Adam
  (lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8); batch 128; 2 epochs; Float32.
- Seed: process-wide generator seeded with 1234 before model construction;
  train DataLoader seeded 1234.
- Data: uncompressed MNIST IDX files in `datasets/mnist/` (run
  `./fetch_mnist.sh`). The demo accepts `--train-rows`, `--test-rows`, and
  `--epochs`; the ctest entry uses 2048/512/1 so the suite stays fast.

## Results (2026-11-06, full set: 60000 train / 10000 test, 2 epochs)

| framework | backend | test accuracy | wall |
|---|---|---:|---:|
| BeanTensor | CPU | **97.78%** | 56.2 s |
| BeanTensor | CUDA GPU | 97.32% / 97.27% (two runs) | 16.3 s |
| PyTorch | CPU | 98.35% | 78.8 s |

Epoch detail (BeanTensor CPU): epoch 1 → 96.67%, epoch 2 → 97.78%.
GPU runs: 97.32% and 97.27% final test accuracy across two runs.

## Pass-bar verdict: PASS with a documented hairline exception

- BeanTensor CPU (97.78%) >= PyTorch CPU (98.35%) - 1pp (97.35%): **PASS**.
- BeanTensor CPU vs GPU: 0.46-0.51pp apart, within the 1pp backend
  tolerance: **PASS**.
- BeanTensor GPU (97.27-97.32%) vs the 97.35% bar: misses by 0.03-0.08pp —
  a hairline, reproducible miss, not noise. Closing it wants one tuning pass
  (epochs or learning rate) with both frameworks rerun at the new config;
  recorded as follow-up rather than papered over.

## Repro

```bash
./fetch_mnist.sh
cmake -B build-cpu -S . -DUSE_CUDA=OFF -DUSE_HIP=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build build-cpu --target BeanTensorV5CNN -j 6
./build-cpu/BeanTensorV5CNN            # full set, CPU
# PyTorch counterpart (same architecture/optimizer/epochs):
comparison/.venv/bin/python comparison/cnn_pytorch.py
```

## Enabling note: pool scheduler fixes this demo forced

The first sustained convolution-backward consumer in the project exposed
four layers of pre-existing concurrency defects, all fixed on this branch and
covered by regression tests in `tests/platform_topology_tests.cpp`:

1. Tensor destructors inside pool workers blocked on pending tasks — now
   drained through the helping wait.
2. `pending_tasks` mutation raced between submit and worker threads — now
   serialized (TSAN-clean).
3. The conv backward's raw-pointer kernel chain had no producer/consumer
   ordering — now ordered with in-worker helping waits (TSAN-clean; this was
   also a latent wrong-gradient bug).
4. `ThreadPool::steal()` skipped lone tasks in parked workers' deques and
   `sync()` blocked inside workers — steal now takes one task from the first
   non-empty victim (lone tasks stay claimable) and `sync()` routes through
   the helping wait on workers.
