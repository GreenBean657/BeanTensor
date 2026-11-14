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
`Device::CPU`. `Tensor::to()` moves a tensor between CPU and GPU. See [Device transfers and failure atomicity](#device-transfers-and-failure-atomicity) below for the exact state machine.

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

## CSV datasets

`Data::load_csv(path, options)` turns a CSV file into the tensor pair
`DataLoader` consumes directly: features `{N, F}` in `CsvOptions::feature_dtype`
(Float32/Float64/Int32/Int64; integer kinds reject fractional values) and
labels `{N}` Int32 class ids. The label column (last, or `CsvOptions::label_column`)
is categorical: ids follow first-seen order and `CsvDataset::label_names`
maps an id back to its original string. Every other column must be numeric —
a stringy feature column raises `ErrorHandling::NonNumericValue`. Header
presence is automatic (the first line is a header when any cell fails to parse
as a number) or forced via `CsvOptions::has_header`; the delimiter is inferred
from comma/semicolon/tab or forced via `CsvOptions::delimiter`. Empty cells in
numeric columns become NaN by default (`CsvNullPolicy::KeepNaN`); the policy
can instead raise (`Error` → `NullValue`), or fill with the column's
NaN-skipped mean, median, or a constant (`CsvOptions::fill_value`). Each load
reports `CsvDataset::null_count` and the NaN-skipped per-column
`column_min`/`column_max`/`column_mean` of the parsed values. Features can be
scaled per column after filling and before dtype conversion:
`CsvNormalize::MinMax` maps to [0, 1] (constant columns become all zeros) and
`Standardize` z-scores with the population standard deviation (constant
columns become all zeros); the label column is never scaled. Empty label
cells, ragged rows (`RaggedCsv`), a file with no data rows (`EmptyFile`), and
an unopenable path (`SerializationError`) all raise. Blank lines are skipped. Errors raise
synchronously at the call; loading is host-side and not part of the async
tensor-op model.

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

## Device transfers and failure atomicity

`Tensor::to(Device)` moves a tensor between CPU and GPU. The implementation
follows a strict validate → allocate destination → copy → hand-off views →
commit state machine:

1. **Validate** the source device and target device pair. Non-contiguous views
   are rejected here, before anything is mutated.
2. **Allocate** the destination buffer on the target device.
3. **Copy** the source bytes into the destination buffer.
4. **Hand off child views**, after the copy has landed: an owner clones each
   child out of the old buffer before it is freed; a view re-homes its children
   onto the view's parents.
5. **Commit** only after the allocation and copy both succeed: free the old
   buffer, flip `device`/`data`/`offset`/`is_contiguous`, and recompute strides.

Because the source `data` pointer, `offset`, and `device` are not touched until
step 5, an allocation failure (OOM) or a copy failure leaves the source tensor
fully usable. The failure is non-sticky: calling `to()` again on the same tensor
retries the transfer from the intact source. Contiguous views become owned,
contiguous copies on the destination device; non-contiguous views raise
`ErrorHandling::NotImplemented` rather than silently copying the wrong bytes.

The accumulated gradient (`Tensor::grad`) rides along recursively after the data
buffer moves, so a parameter and its gradient always share a device.

A classification hook, `BeanTensor::Tensors::detail::classify_transfer_failure`,
produces a `TransferFailureClassification` with four flags:
`alloc_failed_pre_change`, `copy_failed_with_temp`, `h2d_fallback_eligible`, and
`fallback_meaningless`. The OOM policy consumes this hook to decide whether to
propagate the error or fall back to CPU compute. See [OOM policy](#oom-policy)
below.

Test builds compile in a fault-injection seam
(`BeanTensor::Tensors::detail::TransferFaultInjection`) that can arm alloc or
copy failures deterministically. It is compiled out of production builds.

## OOM policy

`BeanTensor::Helpers::OOMPolicy` is a process-wide singleton configured once at
startup via `OOMPolicy::configure(OOMPolicyConfig)`. The policy is consulted by
`Tensor::atomic_transfer_to()` on allocation or copy failure and by compute-
dispatcher boundaries on OOM. It supports three exclusive modes:

1. **`PreserveLogThrow`** (default): log a diagnostic line and rethrow the
   original exception. The source tensor remains untouched because the transfer
   is failure-atomic.
2. **`H2DFallback`**: when a CPU→GPU transfer fails with
   `h2d_fallback_eligible == true`, swallow the error and leave the tensor on
   CPU so execution can continue there. Compute-dispatcher boundaries can also
   fall back to CPU when the dispatcher marks the op as fallback-eligible.
3. **`CheckpointAndThrow`**: call the application-supplied checkpoint hook,
   log a diagnostic line, and rethrow. The hook is expected to save the model
   via `Serialization::save(..., CheckpointLevel::Full)` using an atomic
   temp-file + rename; the policy does not itself perform I/O beyond logging.

Configuration is explicit C++ API only (no environment variables). `configure()`
validates the config and throws `ErrorHandling::InvalidOOMPolicy` if checkpoint
mode is selected without both a `checkpoint_hook` and a non-empty
`checkpoint_destination`. The diagnostic log is append-only, line-oriented, and
mutex-serialized; a logging failure is swallowed so it can never mask the
original OOM.

The compute-dispatcher fallback eligibility is normally derived from op/device
metadata by the dispatcher. A test-only seam
(`OOMPolicy::set_compute_fallback_eligible_for_test`) lets the unit tests
exercise the compute-fallback path without a real dispatcher integration.

## Autograd

`matmul`, `bias_add`, `relu`, `softmax_ce`, and the elementwise
add/sub/mul/div/pow, exp/log, `sum`, `mean`, `mse`, `bce`, and `bce_with_logits` ops attach a graph node
to their output whenever any input has `requires_grad=true`. The
indexing/view ops do the same: `index_select`, `gather`, `cat`, `slice`,
`split` (each chunk wires the slice node), and `reshape` (which shares
flatten's backward). You run
backpropagation by calling `backward(grad_dtype)` on a scalar or shape-{1} loss
tensor.

Indexing/view nodes carry their per-op state in an owned `NodeMeta` payload
hung off the node (the axis, the slice ranges and source shape, and — for
`cat` with more than three inputs — the overflow graph-parent pointers);
the node deletes it when the output tensor dies or is re-wired. Index tensors
are not copied into the metadata: they ride in the node's saved-tensor slots
as borrowed graph constants and, exactly like input parents, must outlive
`backward()`. Duplicate-index backward (`index_select`, `gather`)
scatter-adds into a zeroed gradient in ascending pick order per destination
element on every backend, so the accumulation is deterministic and
bit-identical between CPU and GPU.

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

## Binary cross-entropy

`Tensor::bce(target)` is the probability-form BCE: `mean(-y*log(p̂) - (1-y)*log(1-p̂))`, where `p̂ = clamp(p, eps, 1-eps)`. `Tensor::bce_with_logits(target)` is the numerically stable logit form: `mean(max(x, 0) - x*y + log1p(exp(-|x|)))`. Both return a shape-{1} tensor in the prediction/logit dtype and block before returning. Both forms attach a graph node when the prediction/logit tensor has `requires_grad=true`; targets never receive a gradient.

Validation contract: operands must share shape, dtype, and device; both must be contiguous; integer storage is rejected (`NotImplemented`). Targets must be finite and lie in `[0, 1]`; otherwise `ErrorHandling::InvalidProbability` is thrown. Empty tensors and non-broadcastable shape mismatches raise `MismatchedShapes`.

Clamp contract: for fp32-compute dtypes (fp32, and the widened fp16/bf16/packed paths) `eps = 1e-7`; for fp64 `eps = 1e-15`. Forward and backward use the same clamped `p̂`, so the gradient denominators match the forward loss exactly. The logit form is implemented directly in log-space; it is never `sigmoid` followed by probability-BCE, so overflow cannot reappear.

Non-finite logit behavior: the stable formula is evaluated exactly as written. At `x = +Inf` with `y = 1` it evaluates `Inf - Inf = NaN`; at `x = -Inf` with `y = 0` the product `-Inf * 0` is also `NaN`, so the loss is `NaN`. At `x = +Inf` with `y = 0` the product `Inf * 0` is `NaN`, so the loss is `NaN`. At `x = -Inf` with `y = 1` the `-x*y` term yields `+Inf` and the loss is `+Inf`. This is formula-exact and matches PyTorch. Callers must ensure logits are finite if a finite loss is required.

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

`BeanTensor::Optimization::GradScaler` is a mixed-precision training-loop
utility, not an engine precision mode. It scales the loss (`scaler.scale(loss)`),
then after `backward()` unscales parameter gradients in place and checks them for
finite values (`scaler.unscale_and_check(module)`). When every gradient is
finite it applies dynamic growth/backoff through `scaler.update(finite)`; on a
non-finite step it backs off the scale and the caller simply does not invoke the
optimizer. Convenience wrappers `scaler.step(module, lr, ...)` for Adam and
`scaler.step(module, lr)` for SGD perform unscale-check-update-step in one call.
Because a skipped step does not call `Module::adam_step` or `Module::sgd_step`,
Adam's per-parameter step counter and first/second moment buffers are untouched
and SGD leaves parameters unchanged. Static mode disables dynamic growth/backoff
while still applying the fixed scale. The finite scan uses existing device ops
(`sub`, `equal`, `min`) and performs no host fp16 arithmetic.

## GELU activation

`Tensor::gelu()` computes the Gaussian Error Linear Unit exactly as
`0.5 * x * (1 + erf(x / sqrt(2)))`. The scalar CPU path and the CUDA/HIP
kernels evaluate `erf` through the standard library or device math library
respectively.

The AVX2 and AVX-512 paths are width-dependent:

- `float32_t` SIMD uses the Abramowitz & Stegun 7.1.26 polynomial approximation
  of `erf` with a lane-wise `exp` inside vector load/store loops. Forward and
  backward results match the scalar `std::erf` reference to within the
  polynomial's absolute-error bound (`~1.5e-7`); the regression suite asserts
  `5e-7`.
- `float64_t` SIMD falls back to scalar `std::erf` per lane inside the vector
  loop, so forward and backward are bit-exact against the scalar reference on
  AVX2 and AVX-512.

For large magnitudes every tier clamps to the asymptotic value (`x` for
positive inputs, `0` for negative inputs). Cutoffs are `5.7f` for `float32_t`
and `9.0` for `float64_t`. The non-finite contract is `NaN -> NaN`,
`+Inf -> +Inf`, `-Inf -> 0`; the backward contract is `deriv(+Inf) = 1`,
`deriv(-Inf) = 0`, `deriv(NaN) = NaN`.

`gelu` is defined for floating-point dtypes (`Float16`, `Float16x2`,
`BFloat16`, `BFloat16x2`, `Float32`, and `Float64`); reduced-precision lanes
are compared against the `std::erf` reference at `5e-3`. Integer inputs raise
`ErrorHandling::NotImplemented`. Non-contiguous inputs also raise
`NotImplemented` because the activation is implemented as a single 1-D parallel
loop over contiguous elements.

The matching autograd node computes
`dx = dy * (0.5 * (1 + erf(x / sqrt(2))) + x * exp(-0.5 * x^2) / sqrt(2*pi))`,
using the same piecewise branches as forward so gradient and forward stay
consistent on every backend.

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
- The layer set shipped for the MVP now includes `Linear`, `ReLU`, `GELU`,
  `Sigmoid`, `Tanh`, `Softmax`, `Flatten`, `Dropout`, `Conv2d`, `MaxPool2d`, and
  `AvgPool2d`, each with CPU (scalar + AVX2 + AVX-512), CUDA, and HIP paths where applicable,
  plus matching autograd nodes. `Conv2d` is implemented as im2col + GEMM with a
  scatter epilogue writing NCHW directly, an owner-pinned `Conv2dG` backward,
  He fan-in initialization, and CPU/CUDA/HIP backends.

## PyTorch gradient parity harness

The project ships a C++/PyTorch parity harness under `comparison/` whose goal is
to catch autograd drift early. The C++ driver (`comparison/gradient_parity.cpp`)
builds each parity case from deterministic raw tensors, runs `backward()`, and
writes the inputs and every gradient to `comparison/generated-data/` plus a text
manifest. The Python driver (`comparison/gradient_parity.py`) parses that
manifest, reconstructs the identical graph in PyTorch, and compares every
gradient with NaN/Inf checks followed by an abs-or-rel tolerance. It exits
nonzero on the first mismatch and prints the worst element.

The harness includes loud negative controls (`--perturb <gradient>` and
`--corrupt-manifest`) and a shell runner (`comparison/run_gradient_parity.sh`)
that routes GPU C++ data generation through `gpu-test.sh` so it respects the
shared GPU lock. A performance extension (`benchmark/gradient_parity_perf.cpp`,
`comparison/gradient_parity_perf.py`, and
`comparison/run_gradient_parity_perf.sh`) reuses the forward op benchmark's
shape plan to time one forward + one backward pass per family and publishes a
side-by-side BeanTensor vs PyTorch table in `docs/gradient-parity.md`.

## AvgPool2d

`Tensor::avg_pool2d(kernel)` and `Tensor::avg_pool2d(kernel, stride)` perform 2D
average pooling on rank-4 NCHW tensors. The output spatial size is
`floor((H - kernel) / stride) + 1` with no padding; the one-argument overload sets
`stride = kernel`. Each output element is the arithmetic mean of the
`kernel × kernel` input window.

Dtype and accumulation policy mirrors `Tensor::mean`. Floating-point dtypes sum in
their storage width and divide by `kernel²` in that width. `Float16`/`BFloat16`
and the packed `Float16x2`/`BFloat16x2` dtypes widen each lane to `Float32`, sum,
divide, and narrow RNE back to storage. Integer dtypes sum in `Float64`, then
truncate toward zero and clamp to the dtype range (NaN stores 0). `Int4x2` sums
per nibble in `Float32` and clamps to the signed nibble range `[-8, 7]`.

Backward scatters `dy / (kernel²)` to every element of the source window;
overlapping windows accumulate all contributions. CPU backward serializes to a
single-threaded read-modify-write loop when `stride < kernel` so the accumulation
order is deterministic, and supports all floating dtypes including `Float16` and
`BFloat16`. GPU backward uses atomic adds in output-major order and is currently
implemented only for `Float32` and `Float64`; calling `backward()` with a half
gradient dtype on a float32/float64 GPU result also throws `ErrorHandling::NotImplemented`
(the GPU forward path does not support half dtypes at all).
