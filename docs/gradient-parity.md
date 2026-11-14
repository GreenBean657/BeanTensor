# PyTorch gradient parity harness

This harness verifies that BeanTensor’s autograd produces the same gradients as
PyTorch for identical graphs built from identical raw inputs. It is a
regression gate: the C++ driver emits per-case data and a text manifest, and the
Python driver rebuilds each graph in PyTorch and compares every gradient with
NaN/Inf checks and blind tolerances.

## How it works

- **C++ driver**: `comparison/gradient_parity.cpp` builds each parity case from
  deterministic random little-endian tensors, runs `backward()`, and writes the
  input tensors and every parameter gradient to `comparison/generated-data/`.
  A `manifest.txt` lists the case tolerances and tensor metadata.
- **Python driver**: `comparison/gradient_parity.py` parses the manifest, loads
  the raw bytes into PyTorch tensors, reconstructs the same graph, and compares
  each gradient. It fails loudly on the first mismatch, printing the worst
  element and max abs/rel error.
- **Negative controls**: `--perturb <gradient>` scales one PyTorch gradient by
  `1.1x` to prove the harness would catch a real drift; `--corrupt-manifest`
  proves manifest parsing is guarded.
- **Runner**: `comparison/run_gradient_parity.sh` ties the two sides together.
  GPU runs are routed through `/srv/dev/BeanTensor/gpu-test.sh` so they respect
  the shared GPU lock.

## Cases

| case | grad dtype | abs/rel tol | what it exercises |
|---|---|---|---|
| `broadcast_add` | Float64 | 1e-12 | elementwise broadcast + sum |
| `broadcast_mul` | Float64 | 1e-12 | elementwise broadcast + sum |
| `broadcast_div` | Float64 | 1e-12 | elementwise broadcast + shifted divisor |
| `matmul_bias_relu_softmax_ce` | Float32 | 1e-4 | linear layer + ReLU + cross-entropy |
| `mse_linear` | Float32 | 1e-4 | linear layer + MSE loss |
| `conv_bn_pool_flatten` | Float32 | 1e-4 | Conv2d → BatchNorm → ReLU → MaxPool2d → Flatten → Linear → CE |
| `dropout_fixed_mask` | Float32 | 1e-5 | elementwise `mul` with a pre-generated dropout mask |
| `chained_fanout` | Float64 | 1e-12 | fanout + mean (tests multi-use gradients) |

## Running

```bash
# CPU (requires a PyTorch venv at comparison/.venv)
bash comparison/run_gradient_parity.sh --build-dir scratch/build-cpu --device cpu

# CUDA (must go through gpu-test.sh)
bash comparison/run_gradient_parity.sh --build-dir scratch/build-cuda --device gpu
```

## Performance: forward + backward

The same shape plan used by the forward op benchmark
(`BeanTensorV5Benchmarks --emit-shapes`) is timed with one forward pass and one
backward pass per row. `argmax` and `Tensor::max()` are omitted because they are
not wired into the autograd graph.

<!-- perf-results -->
_Generated 2026-11-14 17:21 UTC_

| family | op | dims | BeanTensor ms | PyTorch ms | ratio |
|---|---|---:|---:|---:|---:|
| elementwise | add | 393216 | 10.5656 | 24.9817 | 0.42 |
| elementwise | mul | 393216 | 18.5509 | 42.9837 | 0.43 |
| elementwise | add | 25165824 | 277.628 | 53.2575 | 5.21 |
| elementwise | mul | 25165824 | 340.337 | 129.3446 | 2.63 |
| gemm | matmul | 6144x256x256 | 69.2444 | 96.9709 | 0.71 |
| gemm | matmul | 24576x256x256 | 250.999 | 150.0299 | 1.67 |
| reduction | sum | 3072x512 | 5.09278 | 21.0648 | 0.24 |
| loss | softmax_ce | 3072x10 | 3.76322 | 18.0403 | 0.21 |
| convolution | conv2d | 8x4x56x56x16x3x3 | 33.1421 | 91.0031 | 0.36 |
| spatial | max_pool2d | 16x8x28x28 | 3.20142 | 23.9882 | 0.13 |
| normalization | batch_norm | 16x16x14x14 | 6.36833 | 43.9527 | 0.14 |

<!-- /perf-results -->
