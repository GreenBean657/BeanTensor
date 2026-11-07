# Op-level benchmark results

Machine-readable history: `benchmark/results/<timestamp>.log` (appended by
`benchmark/run.sh`). This file holds the published table and the repro
commands; numbers are from the run recorded below and are hardware-dependent
— sizes derive from the detected last-level cache at runtime, never from the
dev box.

## Harness

- Binary: `BeanTensorV5Benchmarks` (wired into the test suite as
  `benchmark_smoke`, which fails the suite if any benchmark family stops
  producing rows).
- Method: one warmup iteration, then 5 timed iterations per row, median
  reported. Timed region includes the op's `sync()` so queued async work
  lands inside its own measurement.
- Families: elementwise (add/mul), gemm (matmul), reduction (sum/max/argmax
  axis-1), loss (softmax_ce), convolution (conv2d), spatial (max_pool2d),
  normalization (batch_norm train mode).
- Scope ruling: GPU op-level benchmarking is out of scope for this harness —
  timing queued async GPU work needs per-iteration device-sync semantics the
  harness does not implement, and GPU test traffic must go through
  `gpu-test.sh`. End-to-end GPU numbers live in `docs/mnist-results.md` and
  `docs/cnn-results.md`.

## Repro

```bash
./benchmark/run.sh
```

CPU build + run unconditionally; the PyTorch comparison column runs when the
comparison venv exists.

## Results (2026-11-06)

BeanTensor CPU vs PyTorch CPU, same shapes (shape plan emitted by
`--emit-shapes`, consumed by `comparison/bench_pytorch.py`):

| family | op | dims | BeanTensor ms | PyTorch ms |
|---|---|---|---:|---:|
| elementwise | add | 393216 | 26.9395 | 11.9863 |
| elementwise | mul | 393216 | 27.0681 | 6.9796 |
| elementwise | add | 25165824 | 34.6617 | 35.3645 |
| elementwise | mul | 25165824 | 47.3423 | 33.4015 |
| gemm | matmul | 6144x256x256 | 14.0379 | 12.8334 |
| gemm | matmul | 24576x256x256 | 30.2742 | 26.8438 |
| reduction | sum | 3072x512 | 29.7654 | 6.0352 |
| reduction | max | 3072x512 | 2.64979 | 5.9094 |
| reduction | argmax1 | 3072x512 | 3.73499 | 8.8549 |
| loss | softmax_ce | 3072x10 | 1.24738 | 9.0210 |
| convolution | conv2d | 8x4x56x56x16x3x3 | 14.6919 | 15.8624 |
| spatial | max_pool2d | 16x8x28x28 | 13.3848 | 15.1622 |
| normalization | batch_norm | 16x16x14x14 | 6.01416 | 27.9649 |
