# Python API

The opt-in `beantensor` extension module (`Bindings/pybind_module.cpp`, built with
`-DBUILD_PYTHON_BINDINGS=ON`, pybind11 >= 3.0 from pip) exposes the training
surface the repository's Python programs use: `python/digits.py` and
`python/cnn.py` replay the C++ MNIST demos through it, and `python/btui.py`
builds an interactive training TUI on top of it. This page is the reference for
that module; `bt.__version__` is `"0.1"`.

Everything below was checked against `Bindings/pybind_module.cpp` — signatures,
parameter names, and defaults are the real ones. If a feature you expect is not
listed, see [Not yet bound](#not-yet-bound).

## Build and run requirements

You need one Python 3 interpreter with development headers (Debian/Ubuntu:
`python3-dev`), a venv holding `pybind11 >= 3.0` and `numpy`, and a build
configured with `-DBUILD_PYTHON_BINDINGS=ON`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install "pybind11>=3.0" numpy

cmake -B <build-dir> -S . -DUSE_CUDA=OFF -DUSE_HIP=OFF \
      -DBUILD_PYTHON_BINDINGS=ON -DCMAKE_BUILD_TYPE=Release -G Ninja \
      -DPython3_EXECUTABLE=$PWD/.venv/bin/python
cmake --build <build-dir> --target beantensor
```

Pass both backend flags explicitly so the CMake default never picks one for you
(`-DUSE_CUDA=ON -DUSE_HIP=OFF` for CUDA, `-DUSE_HIP=ON -DUSE_CUDA=OFF` for
ROCm). The build produces `beantensor` as a shared object under
`<build-dir>/Bindings`; put that directory on `PYTHONPATH` to import it:

```bash
PYTHONPATH=<build-dir>/Bindings python3 -c "import beantensor as bt; print(bt.__version__)"
```

`numpy` is required at runtime for the array interop functions. Everything else
in this document is stdlib-plus-`beantensor`.

A CPU-only build rejects every `Device.GPU` use with `RuntimeError`; a
CUDA/HIP build trains on GPU by passing `bt.Device.GPU` where a `device`
argument is documented below. See [`docs/architecture.md`](architecture.md) for
backend semantics.

## Conventions

Read these before the tables; they are the engine's rules, and the bindings
preserve them rather than paper over them.

- **Asynchronous execution.** Tensor ops queue onto a thread pool and return
  immediately; errors surface at `sync()`. `item()`, `to_numpy()`,
  `write_numpy()`, and `evaluate()` sync for you (and release the GIL across
  the wait, so other Python threads stay responsive). `convert_dtype()` is
  asynchronous: the dtype flips when the queued conversion finalizes, so sync
  before asserting `dtype` or feeding the result into the next op.
- **Exceptions.** Engine errors arrive as `RuntimeError` carrying the engine
  message; binding-side validation (contiguity, dtype/shape mismatch, byte
  counts) raises `ValueError`. A rejected `write_numpy()` leaves the payload
  untouched.
- **Graph lifetime.** A module's output joins the computation graph through
  the module's parameters and the input (borrowed raw pointers), so
  `backward()` a module's output BEFORE calling `forward()` on that module
  again — every `forward()` drops the previous pass's intermediates. The
  natural loop order (forward, loss, backward, step, `zero_grad()`) satisfies
  it. Graph-output tensors pin their parents alive from Python, so deleting
  the Python-side temporaries (input, logits, even the module) is safe while
  the loss you backward from stays referenced.
- **Determinism.** `bt.set_seed(n)` seeds the process-wide generator that
  weight initialization and shuffling draw from. Layers draw one init seed per
  layer in construction order, so construct layers as separate statements in
  layer order — a one-expression `bt.Sequential([...])` around bare factory
  calls leaves the draw order to argument evaluation order. A Python loop that
  replays a C++ program call-for-call reproduces it bit-for-bit (CPU builds).
- **`to()` is in-place and blocking** (syncs before returning) and returns
  `None`: `x.to(bt.Device.GPU)` moves `x`, it does not produce a new tensor.

## Quickstart

A complete, copy-paste program: a tiny MLP trained a few steps on synthetic
data, printing the loss. No dataset download needed.

```python
import numpy as np
import beantensor as bt

bt.set_seed(1234)

x = bt.Tensor.from_numpy(np.array([
    [1.0, 0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0, 0.5],
    [0.0, 0.0, 1.0, 0.25],
    [1.0, 1.0, 0.0, -0.5],
    [0.0, 1.0, 1.0, 0.75],
    [1.0, 0.0, 1.0, -0.25],
], dtype=np.float32))
y = bt.Tensor.from_numpy(np.array([0, 1, 2, 3, 1, 2], dtype=np.int64))

# Named statements in layer order: each layer draws its init seed from the
# process-wide generator at construction.
fc1 = bt.Linear(4, 8, bt.DType.Float32, bt.Device.CPU)
relu = bt.Relu()
fc2 = bt.Linear(8, 4, bt.DType.Float32, bt.Device.CPU)
net = bt.Sequential([fc1, relu, fc2])
net.train()

for step in range(30):
    net.zero_grad()
    logits = net.forward(x)                       # backward this before the next forward
    loss = bt.softmax_cross_entropy(logits, y)    # shape {1}, keeps logits+y alive
    loss.backward(bt.DType.Float32)               # grad dtype is a per-call decision
    net.adam_step(1e-2, 0.9, 0.999, 1e-8)         # lr, beta1, beta2, epsilon
    if step % 10 == 0 or step == 29:
        print(f"step {step:2d}  loss {loss.item():.6f}")
```

The printed loss decreases over the 30 steps and is bit-reproducible across
runs of the same build with the same seed. Real-data versions of this loop:
`python/digits.py`
(MNIST MLP) and `python/cnn.py` (MNIST CNN), both runnable with
`PYTHONPATH=<build-dir>/Bindings`.

## Tensors

Create tensors with the two static factories — there is no shape-only
constructor (see [Not yet bound](#not-yet-bound)).

### Creation and NumPy interop

| Member | Signature | Notes |
|---|---|---|
| `Tensor.from_numpy(arr)` | staticmethod | Copies a C-contiguous NumPy array of a supported dtype into a new CPU tensor with the array's shape and dtype. Non-contiguous arrays and unsupported dtypes raise `ValueError`. Supported: `float16`, `float32`, `float64`, `int8`–`int64`, `uint8`–`uint64` (bool, complex, float128, object, str have no engine counterpart). |
| `Tensor.from_bytes(data, shape, dtype, device=Device.CPU)` | staticmethod | Builds a tensor from raw little-endian element bytes — the only host path for the dtypes NumPy cannot represent (`BFloat16`, `BFloat16x2`, `Float16x2`, `Int4x2`). `data` must be exactly `prod(shape) * dtype_bytes` long, else `ValueError`. |
| `to_numpy()` | method | New NumPy array owning a copy of the data; the result survives the tensor's destruction. CPU tensors only — a GPU tensor raises `RuntimeError`; dtypes with no NumPy counterpart raise `ValueError` pointing at `to_bytes()`. Syncs (GIL released). |
| `write_numpy(arr)` | method | In-place overwrite of the whole payload from a C-contiguous array of exactly the tensor's dtype and shape; syncs first. Every check (device, contiguity, dtype, shape) runs before the single write, so a rejected call changes nothing. GPU tensor → `RuntimeError`. |
| `to_bytes()` | method | Exact inverse of `from_bytes`; works for every engine dtype. Syncs. |

```python
t = bt.Tensor.from_numpy(np.arange(12, dtype=np.float32).reshape(3, 4))
t.write_numpy(np.zeros((3, 4), dtype=np.float32))   # in-place, validated
arr = t.to_numpy()                                  # fresh copy out
raw = t.to_bytes()                                  # raw element bytes

# Raw bits for a dtype NumPy cannot hold (BFloat16 0x3F80 = 1.0):
bf16 = bt.Tensor.from_bytes(bytes([0x80, 0x3F]), (1,), bt.DType.BFloat16)
```

### Properties, lifecycle, and dtype conversion

| Member | Signature | Notes |
|---|---|---|
| `shape` | property | Tuple of ints, e.g. `(2, 3)`. |
| `dtype` | property | A `bt.DType` value. |
| `device` | property | `bt.Device.CPU` or `bt.Device.GPU`. |
| `sync()` | method | Blocks until queued work touching this tensor has finished and any pending error is raised. Never call it from a pool worker (documented deadlock); from Python user threads it is the normal drain. |
| `set_zero()` | method | Zeroes the payload in place. |
| `to(device)` | method | In-place, blocking move between CPU and GPU; returns `None`. The accumulated gradient rides along, so a parameter and its grad always share a device. Raises `RuntimeError` on a CPU-only build. |
| `convert_dtype(new_dtype)` | method | In-place dtype conversion with the engine's default `PERMISSIVE` clamp rule (float narrows RNE, float→int truncates toward zero and clamps, NaN→0); returns `None`. Asynchronous — sync before reading `dtype` or feeding the result onward. |

```python
pred = logits.argmax(1)                 # Int64 {N}
mask = pred.equal(y)                    # UInt8 0/1 mask; operands must share a dtype
mask.convert_dtype(bt.DType.Float32)    # async...
mask.sync()                             # ...so sync before the next op
accuracy = mask.mean().item()           # mean consumes the mask directly
```

### Math, evaluation verbs, and autograd

| Member | Signature | Notes |
|---|---|---|
| `add(other)` | method | Elementwise add; wires a graph node whenever an operand has `requires_grad`, so the output keeps both operands alive until `backward()` returns. |
| `mean()` | method | Scalar mean; blocks before returning. Float reduction order is backend-defined. Joins the graph when the input `requires_grad` (keeping the input alive); consumes `equal()` masks directly. |
| `argmax(axis)` | method | Int64 tensor of per-slice argmax indices (input shape minus `axis`); never joins the graph. |
| `equal(other)` | method | Elementwise equality as a UInt8 0/1 mask of this tensor's shape and device. Operands must share dtype, shape, and device; never joins the graph. |
| `softmax_ce(labels)` | method | Method form of `bt.softmax_cross_entropy` — same op, same graph contract, bitwise-identical results. |
| `backward(grad_dtype)` | method | Backpropagates from this scalar/shape-{1} loss; `grad_dtype` is required and is the dtype of the seed and every node-to-node delta. Gradients accumulate into leaf (parameter) tensors across calls. |
| `item()` | method | Reads the single element of a scalar tensor as a Python float (widened to float64). Supports `Float32`, `Float64`, `Int32`, `Int64` only; any other shape raises `ValueError`, any other dtype `RuntimeError`. Syncs (GIL released). |
| `adam_step_count()` | method | Adam optimizer-state introspection: `None` until the first Adam step has touched this tensor, then the step count as an int. |

## Modules

Layer factories are module-level functions returning `bt.Module` handles (the
holder transfers into `Sequential`, so temporaries are fine). Every factory
argument is required — there are no defaults. `dtype` is the parameter dtype
(`bt.DType.Float32` in all shipped demos) and `device` is where the parameters
live.

| Factory | Signature | Notes |
|---|---|---|
| `Linear` | `(in_features, out_features, dtype, device)` | Weight shape `(in_features, out_features)`, bias `(out_features,)`; He fan-in uniform init, zero bias. |
| `Conv2d` | `(in_channels, out_channels, kernel, stride, pad, dtype, device)` | 2-D convolution over rank-4 NCHW input; weight `(in_channels, kernel, kernel, out_channels)`, bias `(out_channels,)`; He fan-in init. |
| `BatchNorm` | `(channels, dtype, device)` | Normalization over the channel dimension; parameters `gamma`/`beta` of shape `(channels,)` (starts 1/0). Keeps its momentum/eps engine defaults. |
| `MaxPool2d` | `(kernel)` | Non-overlapping 2-D max pool (stride = kernel) over rank-4 NCHW. |
| `Flatten` | `()` | Flattens all dims after the batch into one rank-2 `{N, ...}` row; accepts any input rank. |
| `Dropout` | `(p)` | Inverted dropout, stochastic in train mode and the identity in eval mode (`p` is the drop probability). |
| `Relu` | `()` | ReLU activation. |
| `Gelu` | `()` | GELU activation (`0.5x(1 + erf(x/√2))`). |
| `Sequential` | `([layer, ...])` | Chains layers in order; children auto-number (`"0"`, `"1"`, ...) for `named_parameters()` keys. |

### Module methods

Every layer factory product and `bt.load`'s return shares the abstract
`bt.Module` type:

| Member | Signature | Notes |
|---|---|---|
| `forward(x)` | method | Runs the input through the module; the output joins the computation graph through the module's parameters and the input, so it keeps both alive until `backward()` returns. Must be backwarded before this module's next `forward()`. |
| `zero_grad()` | method | Drops every accumulated parameter gradient in the subtree. |
| `adam_step(lr, beta1, beta2, epsilon)` | method | One Adam step over every parameter that currently holds a gradient; parameters without one are skipped. All four arguments required, e.g. `adam_step(1e-3, 0.9, 0.999, 1e-8)`. |
| `sgd_step(lr)` | method | One SGD step over every parameter that currently holds a gradient; `lr` is keyword-able (`sgd_step(lr=0.1)`). |
| `train(mode=True)` | method | Sets train/eval mode recursively (`train(False)` ≡ `eval()`). Stochastic layers (Dropout, BatchNorm) follow this. |
| `eval()` | method | Convenience alias for `train(False)`. |
| `named_parameters()` | method | Dict mapping qualified names (`"0.weight"`, `"2.bias"`, `"1.gamma"`, ...) to `Tensor` handles that share the module's parameters — writes through a handle (e.g. `set_zero()`) mutate the module's parameter. |
| `str(mod)` | — | Human-readable tree dump: layer types, names, shapes, parameter totals. |

```python
params = net.named_parameters()
for name, p in params.items():
    print(name, p.shape, p.dtype)
w = params["0.weight"]
w.adam_step_count()        # None before the first net.adam_step(...)
```

## Losses and evaluation

| Function | Signature | Notes |
|---|---|---|
| `softmax_cross_entropy(logits, labels)` | `(logits, labels)` | Mean softmax cross-entropy over rank-2 logits `{N, C}` and integer targets `{N}` (Int64 in the shipped demos); returns a shape-`{1}` tensor in the logits dtype. The returned loss keeps both operands alive until `backward()` returns. Identical to the `Tensor.softmax_ce` method. |
| `evaluate(model, loader)` | `(model, loader)` | One full evaluation pass: resets the loader, runs it to exhaustion with gradient tracking disabled, and returns `(accuracy, mean_loss)` as Python floats. Saves and restores the parameters' `requires_grad` flags, but does NOT switch train/eval mode — call `model.eval()` first when the model has stochastic layers. Releases the GIL for the whole pass and resets the loader itself, so a second call over the same loader re-runs the pass. |

```python
test_loader = bt.DataLoader(x, y, batch=2, shuffle=False)
net.eval()
accuracy, mean_loss = bt.evaluate(net, test_loader)
net.train()
```

## Data

### DataLoader

```text
loader = bt.DataLoader(images, labels, batch, shuffle=True, seed=None, drop_last=False)
```

| Member | Signature | Notes |
|---|---|---|
| constructor | `(images, labels, batch, shuffle=True, seed=None, drop_last=False)` | Batches a feature/label tensor pair along dim 0 on the device where the data already lives. `seed` makes the shuffled order reproducible; `drop_last=False` keeps a ragged final batch. The loader holds engine-side views of the two tensors — they must outlive the loader (the binding pins the constructor arguments). |
| `reset()` | method | Rewinds to the start of the epoch; call at the top of every training loop. |
| `next_batch()` | method | Returns `(features, labels)` for the next batch, or `None` at epoch end (and on every call after, until `reset()`). |

### Built-in loaders

| Function | Signature | Returns |
|---|---|---|
| `load_mnist_flat(dir, train, dtype, device)` | `(dir, train, dtype, device)` | `(images {N, 784} in dtype, labels {N} Int64)` from the uncompressed IDX files under `dir` (run `./fetch_mnist.sh`). `train=True` selects the 60k split, `False` the 10k test split. `RuntimeError` names `dir` on missing/invalid files. |
| `load_mnist_nchw(dir, train, dtype, device)` | `(dir, train, dtype, device)` | `(images {N, 1, 28, 28} in dtype, labels {N} Int64)` — the CNN layout; pixel payload is bitwise identical to `load_mnist_flat`. |
| `load_csv(path, opts)` | `(path, opts)` | A `CsvDataset` (below). Errors raise synchronously with the engine message. |

```python
train_x, train_y = bt.load_mnist_flat("datasets/mnist", True, bt.DType.Float32, bt.Device.CPU)
loader = bt.DataLoader(train_x, train_y, batch=100, shuffle=True, seed=1234, drop_last=True)
```

### CSV datasets

`bt.CsvOptions()` is default-constructed, then three bound fields depart from
the engine defaults (the rest — automatic header/delimiter detection,
last-column labels, `KeepNaN` nulls — stay):

| Field | Type | Notes |
|---|---|---|
| `feature_dtype` | `bt.DType` | Features dtype: `Float32`/`Float64`/`Int32`/`Int64` (integer kinds reject fractional values). |
| `nulls` | `bt.CsvNullPolicy` | `KeepNaN` (default), `Error`, `FillMean`, `FillMedian`, `FillConstant`. |
| `normalize` | `bt.CsvNormalize` | `MinMax`, `Standardize`, or `None`. Quirk: `None` is not a dotted-access name in Python syntax — reach it with `getattr(bt.CsvNormalize, "None")`. |

`bt.load_csv(path, opts)` returns a `CsvDataset` with readonly fields:
`features` (`{N, F}` in `feature_dtype`), `labels` (`{N}` Int32 class ids in
first-seen order), `feature_names`, `label_names` (id → original string),
`null_count`, and the NaN-skipped per-column stats `column_min`,
`column_max`, `column_mean` (plain lists). Details and error taxonomy:
[`docs/architecture.md`](architecture.md) § CSV datasets.

```python
opts = bt.CsvOptions()
opts.feature_dtype = bt.DType.Float32
opts.nulls = bt.CsvNullPolicy.FillMean
opts.normalize = bt.CsvNormalize.Standardize
ds = bt.load_csv("scratch/csv_demo_train.csv", opts)
loader = bt.DataLoader(ds.features, ds.labels, 32)
```

## Optimizers

There are no optimizer objects — the update rules are methods on `Module`
(walking the whole subtree) over the gradients `backward()` accumulated into
each parameter:

- `net.adam_step(lr, beta1, beta2, epsilon)` — Adam; moment state is keyed by
  parameter identity and never reallocated, and `Tensor.adam_step_count()`
  exposes the per-parameter step counter (`None` until stepped).
- `net.sgd_step(lr)` — plain SGD; parameters without an accumulated gradient
  are skipped by both rules.
- `net.zero_grad()` — clears the accumulated gradients between steps.

The standard loop is `zero_grad()` → `forward()` → loss → `backward(grad_dtype)`
→ step. The `backward` argument sets the gradient dtype for that call — there is
no global default; the demos pass `bt.DType.Float32`.

## Checkpoints (.bt)

| Function | Signature | Notes |
|---|---|---|
| `save(module, path, compress=True, level=CheckpointLevel.Weights, metadata="")` | see signature | Writes architecture + parameters (+ more per `level`) to `path`. Synchronous, but quiesces nothing itself — sync and settle the model first for a reproducible checkpoint. `metadata` writes one opaque trailing blob (e.g. `"optimizer=adam\nlr=0.001"`); any metadata makes the file format v3. |
| `load(path, device=Device.CPU)` | `(path, device=Device.CPU)` | Rebuilds the module tree; returns a `bt.Module`. dtypes are restored from the records, and a CPU-saved file loads onto `device` with identical values. Missing/corrupt/unsupported files raise `RuntimeError`. |
| `read_metadata(path)` | `(path)` | The CRC-verified v3 metadata blob, or `""` when the file carries none. |

`level` selects the optimizer-state depth:

| `bt.CheckpointLevel` | Records written |
|---|---|
| `Weights` | Architecture + parameters (v1 layout). Gradients and optimizer state are dropped. |
| `WeightsAndGrads` | Plus per-parameter gradient records. |
| `Full` | Plus Adam moments and the step counter — a mid-train checkpoint that resumes training bit-reproducibly. |

```python
bt.save(net, "model.bt", True, bt.CheckpointLevel.Full,
        metadata="optimizer=adam\nlr=0.01")
restored = bt.load("model.bt", bt.Device.CPU)
assert bt.read_metadata("model.bt") == "optimizer=adam\nlr=0.01"
restored.eval()
accuracy, mean_loss = bt.evaluate(restored, test_loader)
```

## Introspection and telemetry

| Function / class | Signature | Notes |
|---|---|---|
| `export_spec(module)` | `(module)` | The `.bt` architecture spec as plain dicts: `{type, hyperparameters: {k: v}, children: [{name, ...}]}` in registry order. Hyperparameter values stay strings (decimal counts, `"0"`/`"1"` bools, hexfloat floats — parse with `float.fromhex`). Scalars only, no tensor handles. |
| `pool_snapshot()` | `()` | Live CPU-pool telemetry: `{worker_count, pending_tasks_total, stopping, workers: [...]}`; each worker dict carries `worker_id`, stringified `status`/`task_priority`, and counters `task_sequence`, `task_running_ms`, `futures_awaited`, `tasks_completed`, `tasks_stolen`, `tasks_helped`, `local_queue_depth`, `pinned_group`. Never throws; returns in microseconds. |
| `gpu_memory()` | `()` | `{total_bytes, free_bytes, used_bytes}` when a GPU backend is compiled in and the runtime answers; `None` on CPU-only builds, call failure, or no live GPU. Telemetry never throws. |
| `Dashboard(tty=True, refresh_hz=2.0)` | class | In-process terminal dashboard; methods `report_batch(batch_loss)`, `report_epoch(epoch, epochs, train_mean_loss, test_accuracy, test_mean_loss)`, `render_once()`, `stop()` (idempotent), `last_frame()` (test seam). Frames sink to the process's real fd 2 (`std::cerr`), not Python's `sys.stderr`; with `tty=False` it is fully inert. See § Training dashboard in [`docs/architecture.md`](architecture.md). |

## dtype and device reference

`bt.DType` exposes all fifteen engine dtypes. Eleven have NumPy-native
counterparts and work with `from_numpy`/`to_numpy`/`write_numpy`; the packed
and sub-byte dtypes only exist on the raw-bytes path.

| `bt.DType` member | NumPy counterpart | Notes |
|---|---|---|
| `BFloat16` | — | Raw bytes only. |
| `BFloat16x2` | — | Two bf16 lanes per element; bytes only. |
| `Float16` | `float16` | IEEE binary16, same layout as NumPy. |
| `Float16x2` | — | Two f16 lanes per element; bytes only. |
| `Float32` | `float32` | |
| `Float64` | `float64` | |
| `Int8` | `int8` | |
| `Int16` | `int16` | |
| `Int32` | `int32` | CSV labels and `equal()` operands are typically Int32. |
| `Int64` | `int64` | Class-label dtype (`load_mnist_*`, `softmax_cross_entropy` targets, `argmax` output). |
| `UInt8` | `uint8` | `equal()`'s 0/1 mask dtype. |
| `UInt16` | `uint16` | |
| `UInt32` | `uint32` | |
| `UInt64` | `uint64` | |
| `Int4x2` | — | Two signed 4-bit lanes per byte; bytes only. |

`bt.Device` has exactly two members: `CPU` and `GPU` (single-GPU scope; there
is no multi-device API). `bt.CheckpointLevel`: `Weights`, `WeightsAndGrads`,
`Full`. `item()` supports only `Float32`, `Float64`, `Int32`, `Int64`.

## btui

`python/btui.py` is a standalone interactive training TUI over this module —
an editable layer stack, data-source and hyperparameter panels, live loss
graph, CPU-pool panel, and `.bt` checkpoint save/load/export-import keys — with
a headless replay mode for tests. Run `python3 python/btui.py` (MNIST IDX files
from `./fetch_mnist.sh` required for the built-in sources; `--data-dir`
overrides). Full behavior description: [`docs/architecture.md`](architecture.md)
§ btui. Its logic and UI are covered by `tests/btui_logic_tests.py` and
`tests/btui_ui_tests.py`.

## Not yet bound

The engine has these capabilities; the Python surface deliberately does not
(yet). Workarounds noted where one exists:

- **Tensor math beyond `add`.** `sub`/`mul`/`div`/`pow`, `matmul`/`bias_add`,
  `exp`/`log`/`sum`, and the `relu`/`sigmoid`/`tanh`/`gelu`/`softmax` tensor
  methods are engine-side only. Use the corresponding `Module` layers for the
  bound activations.
- **Other losses.** `mse` and `bce`/`bce_with_logits` (and the `mse_loss`
  module entry point) are engine-side only; `softmax_cross_entropy` is the one
  bound loss.
- **Tensor creation and views.** No shape-only constructor and no
  `zeros`/`ones`/`full`/`rand` factories — `from_numpy`/`from_bytes` are the
  only creation paths. No `clone`, `reshape`/`slice`/`split`/`cat`,
  `index_select`/`gather`, element access (`at`/`set`), or `requires_grad`
  control from Python; no `Tensor.grad` readback (gradients are observable only
  through optimizer steps and `WeightsAndGrads`/`Full` checkpoints).
  `argmax_lanes` (packed-logit class indices) is not bound either.
- **Modules.** Engine-side `Sigmoid`, `Tanh`, `Softmax`, and `AvgPool2d` have
  no Python factories.
- **Optimizers.** No optimizer objects, state dicts, or schedulers;
  `GradScaler` (mixed-precision loop utility) is not bound.
- **Engine internals.** Kernel autotuning (`KernelTune::autotune`), the OOM
  policy configuration, CPU-feature queries, and `named_children` /
  `parameter_count` style tree introspection are C++-only. Architecture
  introspection from Python is `export_spec`.
- **NumPy interop for `BFloat16`/`BFloat16x2`/`Float16x2`/`Int4x2`.** By
  design: use `from_bytes`/`to_bytes`.
