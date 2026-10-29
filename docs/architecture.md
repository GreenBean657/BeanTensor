# BeanTensor Architecture

This document describes the user-visible semantics of BeanTensor: the rules a
caller must follow to use the library correctly.

## What BeanTensor is

BeanTensor is a from-scratch C++23 tensor and machine-learning framework that
runs on CPU (scalar, AVX2, and AVX-512), CUDA, and HIP/ROCm with no external ML
libraries. It provides tensors, autograd, neural-network modules, and
first-order optimizers built directly on top of the CUDA/ROCm runtimes and
compiler intrinsics.

## Backends and CPU-feature dispatch

A build targets exactly one GPU backend, or none at all. Pass both flags
explicitly so the CMake default never chooses for you:
`-DUSE_CUDA=ON -DUSE_HIP=OFF`, `-DUSE_HIP=ON -DUSE_CUDA=OFF`, or both off for
a CPU-only build. GPU tensors live on `Device::GPU`; CPU tensors live on
`Device::CPU`. `Tensor::to()` moves a tensor between CPU and GPU.

CPU kernels compile scalar, AVX2, and AVX-512 variants in the same binary and
select one at runtime through `IncludedCPUFeatures()`. Do not perform your own
CPU feature checks.

Dtype conversion between devices and precisions follows a single bit-exact rule
set (`ConversionClampMethod`). When you call `convert_dtype` with a non-`PERMISSIVE`
method, out-of-range or NaN values may raise at `sync()`.

## Execution model

Tensor operations are asynchronous. They queue work onto a thread pool and
return immediately; errors surface when the tensor's `sync()` is called. You
must call `sync()` before you read values with `at()`, print contents, or
assume an operation has completed.

Never call `sync()` from inside a pool worker: it is a documented deadlock.
Inside pool tasks, use `Threading::HelpWait` instead of blocking on futures.
`Tensor::to()` is blocking and syncs before it returns. Destructors drain any
pending work and discard unobserved errors, so if you care about a failure,
sync first.

## Tensors and memory

Tensor dtypes are selected through `DataTypes::DType` and exposed as fixed-width
aliases such as `float32_t`, `float64_t`, `int32_t`, and `bfloat16_t`. The library
uses these aliases internally, and values read back through `at()` are typed with
them, so user code can name tensor dtypes portably.

A tensor's data buffer is 64-byte aligned on CPU and 256-byte aligned on GPU.

`owns_data=false` means the tensor is a view of another tensor. `reshape()`,
`slice()`, and `split()` create metadata-only views; writes through a view are
visible through every tensor sharing that buffer. Each tensor also owns a cached
`convert_buffer` used internally for dtype conversions; you do not manage it.

Copying a `Tensor` is disabled; use `clone(true)` for a deep copy and
`clone(false)` for a view. Moving is `noexcept` and preserves any pending-task
errors on the destination.

## Autograd

`matmul`, `bias_add`, `relu`, `softmax_ce`, and the elementwise
add/sub/mul/div/pow, exp/log, `sum`, `mean`, and `mse` ops attach a graph node
to their output whenever any input has `requires_grad=true`. You run
backpropagation by calling `backward(grad_dtype)` on a scalar or shape-{1} loss
tensor.

The gradient dtype is a per-call decision. The seed and every node-to-node
delta are computed in the `grad_dtype` you pass; boundary casts use the single
grad-conversion rule in `Math/Autograd/GradDType.h` (`PERMISSIVE` semantics).

Leaf gradients accumulate across calls: the first `backward()` call pins the
leaf's gradient buffer dtype, and later calls convert deltas into that pinned
dtype. Intermediate gradients are rebuilt in `grad_dtype` and zeroed at the
start of each call.

Graph parents are stored as raw pointers. You must keep every tensor that
participates in the graph alive until `backward()` returns. Destroying a parent
early leaves dangling pointers and undefined behavior.

## Modules

A `Module` owns named parameters in `std::deque` slots so their addresses stay
stable for the module's lifetime. Optimizer state is keyed by parameter address,
so parameters must never be reallocated after registration.

Every `forward()` implementation clears the module's forward-cache at the start,
dropping the previous pass's intermediate tensors. Therefore, call `backward()`
on a module's output before you call `forward()` on that module again. The usual
training loop order is: forward, loss, backward, optimizer step, `zero_grad()`.
`zero_grad()` and `adam_step()` walk the whole module tree.

## Randomness and determinism

Random numbers come from `BeanTensor::Random::Generator` plus `set_seed(uint64_t)`.
The process-wide generator covers weight initialization, shuffling, and any op
whose seed is omitted. CPU fills use a fixed chunk grid, so results are identical
regardless of how many threads run. Explicit-seed calls are reproducible.

`sum` and `mean` of floating-point tensors use backend-defined reduction order,
so their exact bits may differ between CPU scalar, SIMD, and GPU paths. Min,
max, and argmax are bit-identical across backends.

## Scope

Decisions binding the MVP, recorded 2026-10-27, updated 2026-10-28 for the
landed layer set:

- Python bindings are out of scope for the MVP. The library ships as C++;
  bindings are a later phase. The public API is already shaped for them —
  value-in/value-out signatures, no non-const reference or mutable pointer
  parameters — enforced on every build by the bind-clean verification.
- Weight initialization is He (Kaiming) uniform, fan-in — bound
  sqrt(6 / in_features), bias zero — replacing the old ad-hoc range, in
  `Module/Linear` and `Module/Conv2d`.
- The accuracy pass bar is the PyTorch comparison harness
  (`comparison/mnist_pytorch.py`) on the same architecture, minus one
  percentage point. That is the Task 6 acceptance criterion.
- The Task 6 acceptance model is a single MLP — 784 → hidden → 10 for MNIST.
  Convolutions are available for non-MLP models but are not part of that
  acceptance run.
- The layer set shipped for the MVP now includes `Linear`, `ReLU`, `Sigmoid`,
  `Tanh`, `Softmax`, `Flatten`, `Dropout`, `Conv2d`, and `MaxPool2d`, each with
  CPU (scalar + AVX2 + AVX-512), CUDA, and HIP paths where applicable, plus
  matching autograd nodes. `Conv2d` is implemented as im2col + GEMM with a
  scatter epilogue writing NCHW directly, an owner-pinned `Conv2dG` backward,
  He fan-in initialization, and CPU/CUDA/HIP backends.
