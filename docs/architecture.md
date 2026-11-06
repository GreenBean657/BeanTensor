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

## Python binding readiness

The public headers in `Module/`, `Random/`, `Data/`, and `Metrics/` are kept
pybind11-friendly by `verifications/bind_clean_public_api.sh`, which fails the
verify target on: non-const references to fundamental or fixed-width-alias
types, non-const references to `std::optional<...>`, mutable pointer
parameters, and C-style variadic parameters. Function-pointer parameters are
caught by the mutable-pointer rule. New public API must stay inside these
rules; `verifications/bind_clean_negative_controls.sh` proves each rule fires.

Deliberate exceptions — C++-centric surface that bindings will wrap with
explicit policies rather than reshape:

- `Module::named_parameters()` returns `std::vector<std::pair<std::string,
  Tensor*>>`: the pointer is the address-stable identity that
  `Tensor::adam_step` keys optimizer state on, so it cannot become a value.
- `Module::parameter()` / `Module::grad()` return `Tensors::Tensor&` into
  those address-stable slots (bind with a reference return policy).
- `BatchNorm::running_stats()` returns a pair of const tensor references
  (views of the running statistic buffers).
- `DataLoader::Iterator`'s dereference returns a reference pair; it is a
  C++-only convenience layer — language bindings use `next_batch()`.
- `Random::default_generator()` returns the process-wide generator by
  reference; its copy constructor is deleted, so a by-value binding fails to
  compile. Bindings route through the `set_seed` / `draw_seed` free
  functions instead.
- `Tensor::read_element()` / `Tensor::write_element()` are C++-only bulk
  byte I/O (bounds-checked; the offset is in elements of size `bytes`). They
  are the primitives behind `.bt` export; language bindings get bulk access
  via a future buffer protocol, not these.

`Tensor/Tensor.h` and `Tensor/Ops/TensorOps.h` are excluded from the checked
set: their scanner findings are namespace-scope `detail` internals (autograd
trampolines, dispatcher hooks) — not binding surface — plus the explicit
byte-I/O exception above. See `docs/roadmap.md`.

## Model export (.bt)

`Serialization::save(module, path)` / `Serialization::load(path, device)`
persist and rebuild a module tree. A `.bt` file holds a versioned
architecture spec (layer types, constructor hyperparameters as lossless
hexfloat strings, named children) plus CRC32-verified tensor records —
parameters and BatchNorm running statistics — in registry order. The format
byte layout lives in `docs/roadmap.md`. Bit-exactness contract: a
save → load roundtrip on one machine is byte-identical per tensor (dtypes
are restored from the records), and a CPU-saved file loads on GPU with
identical values because only the device move is backend-defined.

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

## Training utilities

`BeanTensor::Data::DataLoader` produces shuffled or sequential batches from a
feature/label tensor pair on the device where the data already lives. It holds
views into the supplied tensors, so the source tensors must outlive the loader.
Call `reset()` at the start of each epoch and `next_batch()` until it returns
`std::nullopt`; set `drop_last=true` to discard a ragged final chunk.

`BeanTensor::Metrics::evaluate` runs a full evaluation epoch over a DataLoader.
It saves `requires_grad` flags, disables gradient tracking for the forward pass,
and restores the flags on exit (even when an exception propagates), returning
`EvalResult` with top-1 accuracy and a row-weighted mean of
`softmax_cross_entropy`. It does not change the module's train/eval mode, so
stochastic layers such as Dropout stay stochastic unless the caller switches to
`model.eval()` first. Input batches with `requires_grad=true` still wire graph
nodes during the forward pass, though no `backward()` is called. The repository's
`main.cpp` uses DataLoader for both training and test splits and calls
`Metrics::evaluate` after each training epoch to report test accuracy and loss.

## Scope

Decisions binding the MVP, recorded 2026-10-27, updated 2026-10-28 for the
landed layer set:

- Python bindings are out of scope for the MVP. The library ships as C++;
  bindings are a later phase. The public API is already shaped for them —
  value-in/value-out signatures, no non-const references to fundamental or
  alias types and no mutable pointer parameters; class-type non-const references
  are permitted — enforced on every build by the bind-clean verification.
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
