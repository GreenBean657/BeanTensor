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
  through the `from_numpy`/`to_numpy`/`write_numpy` and
  `from_bytes`/`to_bytes` pairs built on top of them, not these directly.

`Tensor/Tensor.h` and `Tensor/Ops/TensorOps.h` are excluded from the checked
set: their scanner findings are namespace-scope `detail` internals (autograd
trampolines, dispatcher hooks) — not binding surface — plus the explicit
byte-I/O exception above. See `docs/roadmap.md`.

## Python bindings

The opt-in `beantensor` extension module (`Bindings/pybind_module.cpp`, built
with `-DBUILD_PYTHON_BINDINGS=ON`, pybind11 >= 3.0 from pip) exposes the training
surface used by the demos: `set_seed`; the `Device` enum and the full
fifteen-value `DType` enum;
`Tensor` (`from_numpy`/`to_numpy` for the eleven NumPy-native dtypes,
`write_numpy` for a validated in-place overwrite from a NumPy array,
`from_bytes`/`to_bytes` raw-bits exchange for the rest, `shape`/`dtype`/`device`,
`set_zero`, `sync`, `to` (in-place, blocking device
move), `add`, `softmax_ce`, the evaluation verbs `argmax`/`equal`/`mean` with
in-place `convert_dtype`, `item`, `backward`);
`Linear`/`Relu`/`Gelu`/`Conv2d`/`BatchNorm`/`MaxPool2d`/`Flatten`/`Dropout`/
`Sequential` (`forward`, `zero_grad`, `adam_step`, `sgd_step`,
`train`/`eval`, `named_parameters`); `softmax_cross_entropy`;
`load_mnist_flat` and `load_mnist_nchw`; `load_csv` returning the `CsvDataset`
mirror — the features/labels tensor pair plus the inspection fields
`label_names`, `feature_names`, `null_count`, and
`column_min`/`column_max`/`column_mean` — with `CsvOptions`
(`feature_dtype`, `nulls`, `normalize`) plus the `CsvNullPolicy` and
`CsvNormalize` enums; `DataLoader` (`reset`, `next_batch`); `evaluate` returning
`(accuracy, mean_loss)`; `pool_snapshot` returning the CPU pool's live
telemetry as plain dicts; `gpu_memory` returning
{total_bytes, free_bytes, used_bytes} (None on CPU-only builds, on call
failure, or with no live GPU — telemetry never throws);
`Tensor.adam_step_count` (None until the first Adam step);
`export_spec` returning a module's architecture
spec as a plain dict tree (type, string hyperparameters, ordered children);
and `save`/`load` with `CheckpointLevel`.
The blocking calls on this surface (`to_numpy`, `write_numpy`, `item`,
`evaluate`) release the GIL across their sync/compute regions so other
Python threads (a UI redraw loop, a telemetry poller) stay responsive
while the training thread waits on the pool.

Determinism contract: the bindings add no RNG draws, conversions, or
reordering of their own, so a Python training loop that replays a C++ program
call-for-call reproduces it bit-for-bit. `python/digits.py` does exactly this
against the `BeanTensorV5` MNIST demo, and `python/cnn.py` does it against the
`BeanTensorV5CNN` convolutional demo: the epoch accuracy/loss lines are
identical to the C++ binary's on the same machine, data, and commit (only
wall-time fields differ). The `mnist_parity` and `cnn_parity` ctests enforce
this on CPU-only builds; on GPU-backend builds they skip themselves, because
the C++ demos then print both a CPU and a GPU set of epoch lines that a line
diff cannot attribute to a side. One replay rule follows from the engine's seed
model: layers draw their init seeds from the process-wide generator in
construction order, so Python must construct layers in the same statement
order the C++ reference uses — C++ argument evaluation order is unspecified,
and collapsing layer construction into one call expression would make the
draw order compiler-dependent.

Graph lifetime: the engine's graph nodes borrow raw pointers to their inputs
and to module parameters, so every graph-output tensor pins its parents with
result-to-parent `py::keep_alive` — `Module.forward`'s output keeps the module
and the input alive, `softmax_cross_entropy`'s loss keeps the logits and
labels alive (the `Tensor.softmax_ce` method is the same op under the same
contract), `Tensor.add`'s output keeps both operands alive, and `Tensor.mean`'s
output keeps its input alive whenever that input requires_grad. Deleting
the Python-side temporaries is safe while the output you backward from stays
alive. One engine rule comes with it: because every `forward()` replaces the
module's forward-cache (`clear_forward_cache`, see `Module/Module.h`), a
module's output must be backwarded BEFORE that module's next `forward()` —
the natural training-loop order (forward, loss, backward, step, zero_grad)
satisfies it.

Build and run:

```bash
cmake -B <build-dir> -S . -DUSE_CUDA=OFF -DUSE_HIP=OFF \
      -DBUILD_TESTING=ON -DBUILD_PYTHON_BINDINGS=ON -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build <build-dir> --target beantensor -j 6
PYTHONPATH=<build-dir>/Bindings python3 python/digits.py
```

`python/digits.py` is stdlib-only apart from `beantensor` itself. It accepts
`--device gpu` (requires a CUDA/HIP build), `--data-dir`, and `--save`; after
training it writes a `.bt` checkpoint and reloads it, printing the reloaded
model's eval numbers. `python/cnn.py` takes the same three flags and replays
the `BeanTensorV5CNN` convolutional demo (Conv2d → BatchNorm → ReLU →
MaxPool2d twice, Flatten, Dropout, two Linears) with the same save/reload
pattern.

## btui

`python/btui.py` is a standalone interactive training TUI over the
`beantensor` bindings: a stdlib-curses front-end (imported lazily, so the
module loads in terminal-less environments) plus a curses-free run core.
Launch it with `python3 python/btui.py` (`--data-dir` overrides the MNIST
directory). v1 trains in Float32 only, on CPU always and on GPU when the
build has a backend — availability is probed once at startup (a tiny tensor
move), and the device field locks to cpu with the
`cpu (no gpu in this build)` message when the probe fails.

Panels: a header strip (run state, `[edits pending]` once the spec or config
has diverged from the live run, `[.bt seed loaded]` after a `W` or `L`
load); a
Model panel — the layer stack with each layer's constructor params, edited
in place, plus a `data input:` info line showing the data source's logical
per-sample shape (`784` for mnist_flat, `(1, 28, 28)` for mnist_nchw, the
feature count for CSV — read through the current null/normalize policies —
and `unknown` when detection fails); editing the FIRST layer's input param
(Linear `in_features` / Conv2d `in_channels`) pre-fills the detected scalar
into the edit form, stated in the prompt and freely overridable; a Params
panel — data source (`mnist_flat`, `mnist_nchw`, `csv`),
data dir, CSV path, CSV null and normalize policies (`KeepNaN`/`FillMean`/
`FillMedian`/`FillConstant` and `None`/`MinMax`/`Standardize`, mapped onto
`CsvOptions`), train/test row caps, optimizer (`adam`, `sgd`),
lr/beta1/beta2/eps, epochs, batch, seed, and a `device` field
(cpu/gpu — editable when the startup probe found a GPU, otherwise greyed
to `cpu (no gpu in this build)` and locked); a Training panel — batch count and
rate, last/mean loss over a bounded window with a vertical eighth-block
graph (non-finite cells render as `?`), and the per-epoch
train-loss/test-accuracy/test-loss
line; and a Pool panel — the worker grid with status tags, stuck flagging,
and done/stolen/helped counters, the same rendering vocabulary as the
embedded dashboard (see [Training dashboard](#training-dashboard)). Below
them sit a two-line log strip and a footer that shows either the help line
or the active edit form.

Keys: arrows move the selection (LEFT/RIGHT walk a layer's params), TAB
switches Model/Params focus, ENTER edits the selected field, `a` starts the
add-layer form (one prompt per param; an empty commit keeps the default),
`d` deletes the selected layer, `j`/`k` move the selected layer down/up one
slot, `r` starts or restarts the run, `p` pauses/resumes a live run (batch
boundary; the header shows `run: paused`), `s`
requests a stop, `w` saves the last finished net to a `.bt` checkpoint
(three-step form: path, default `btui_model.bt`; then
`strip for sharing? (y/n)`, default `n` — `y` writes a recipient-safe copy:
Weights level, no gradients/moments, and NO config metadata, logged as
`shared copy: <path> — weights only, config stripped`; `n` continues to the
checkpoint level —
`Weights`, `WeightsAndGrads`, or `Full`, default `Weights`; the file also
carries the current data/hyperparam config as key=value metadata),
`W` loads a `.bt` checkpoint as seed weights
only, `L` imports a `.bt` checkpoint's ARCHITECTURE — the model stack is
replaced with the file's layers (via `export_spec`; unknown layer types are
rejected with a log error and no partial import) and the loaded module
becomes the pending seed-weights source for the next run; when the file
carries config metadata (written by `w`), the settings that parse are
restored and named in a `restored config:` log line — unknown keys are
ignored and malformed values keep defaults. `r` refuses
shape-incompatible runs BEFORE any engine call: the first layer's input
rank is checked against the data source, and a mismatch logs the shape
contract (`first layer Conv2d needs rank-4 NCHW input {batch, channels,
height, width}; this source gives rank-2 feature rows`) instead of the
engine's raw rank error — the message states the requirement, with the
built-in sources mentioned as one way to satisfy it, never the only way.
Flatten accepts any input rank. Details the
app's layer schema cannot express — Linear/Conv2d `bias=0`, MaxPool2d
`stride != kernel`, BatchNorm `momentum`/`eps` — are dropped on import and
each drop is named in the log line (`dropped: layer N ...`), `q` quits. Footer
forms accept text with BACKSPACE editing; an empty commit keeps the current
value, and ESC cancels the whole in-progress form sequence.

Run semantics: the model and config stay editable while a run is live; edits
are flagged pending and take effect when `r` restarts — the live run stops
and the new run rebuilds from the current spec and config. A restart reuses
weights from the last finished net: every parameter matching by name and
shape is copied into the fresh build and everything else reinitializes
(BatchNorm running statistics are module state, not named parameters, and
are not carried). The copy is host-mediated, so it works across devices: a
GPU-resident old param is moved to CPU in place for the read (safe — the
old net is discarded post-restart) and a GPU-bound new param detours
through CPU for the write before moving back. `W` instead seeds the matching parameters of the next
run's freshly built net from the loaded checkpoint. Data always loads and
slices (row caps) on CPU; with `device: gpu` each split tensor then uploads
via `.to()` before its DataLoader is built, and the train/evaluate loop
runs on-device, and the Training box shows a `vram: used/total GiB` readout
(from `bt.gpu_memory()`, polled at the pool rate; GPU utilization % is out
of scope — it needs NVML, beyond the no-external-libraries bar). A stop is
cooperative:
it lands at a batch boundary, at most one batch after `s`, and the pending
per-epoch evaluate is skipped. Pause is likewise batch-boundary (plus one
check before the per-epoch evaluate): loader position, epoch state, and
optimizer state are preserved and the loss ring simply stops growing.
Resume has two modes: right after an L import the spec is clean, so the
loaded module trains DIRECTLY — a Full checkpoint's Adam moments and step
counters survive (true resume, logged `resumed: optimizer state intact`);
any spec edit marks it dirty and the next run rebuilds + re-seeds as above
(moments reset, logged `rebuilt: weights re-seeded`). The ready net is
consumed when the run starts — a second `r` rebuilds from the spec and
re-seeds from the last finished net. Resume additionally requires the ready
net to already live on the target device: Adam moments are address-keyed
optimizer state and do NOT move with `Tensor.to`, so a CPU-saved
checkpoint resumed onto GPU is rebuilt and re-seeded instead (logged
`device changed: rebuilt with weight re-seed, optimizer state reset`)
rather than silently zeroing its moments mid-run. Two data-path disclosures: CSV runs train
AND evaluate on the same single split (a CSV has no held-out half; the row
caps apply to both uses), and the app's loaders keep the ragged final batch
(`drop_last=false`) rather than discarding it. Quit waits for the training
thread: the first `q` during a live run requests the stop and the UI keeps
redrawing until the thread joins; a second `q` force-quits immediately, and
the (non-daemon) thread then finishes on its own before interpreter
teardown. The same spec and seed give identical init
draws — one seed draw per layer in spec order, the construction-order rule
from [Python bindings](#python-bindings) above.

Color: on terminals with color support the log strip renders error lines
(`run failed:`, `cannot start:`, invalid input) red, the header run state
is green (running) / yellow (paused) / dim (idle), and the Pool worker
cells carry the IDLE/OWN/STEAL/HELP/WAIT color vocabulary from
[Training dashboard](#training-dashboard). Graphs are btop-style VERTICAL
bar graphs (Unicode eighth-block chars U+2581..U+2588, 8 levels per cell,
bottom row is the base), with two flavors: the Pool box renders a 3-row
ABSOLUTE graph of tasks
completed-per-interval under its `workers=`/`pending=`/`done=` header
(activity magnitude is its meaningful signal), while the Training box
renders a 4-row CENTERED DELTA graph of per-batch loss — a dim mid-height
centerline, each column the change from the previous sample (the first
visible sample deltas against the window mean), up = loss rose, down =
loss fell, heights scaled by the window's largest |delta| so the two
directions read comparably; a zero delta renders exactly the centerline
glyph and a flat series renders centerline only,
and the `last=`/`mean=` readout above still carries the level.
The absolute flavor keeps the shared normalization rules: exact zero
renders blank, nonzero floors at one level, flat-but-busy sits
mid-height — an idle graph slot
is blank rows, never filler characters. Cells with no activity in the
interval RECEDE as dim lowercase `#NN idle`; cells whose counter advanced
DOMINATE in bright green as `#NN ~N` (the interval count, never the IDLE
string), and the box header keeps the `workers=`/`pending=`/`done=`
readouts above the interval graph — monochrome-safe, the block characters
carry the information. At startup the app logs one
color-diagnostic line to the log strip: `colors on (N pairs)`, or off with
the reason — `colors off: NO_COLOR set` when the NO_COLOR convention is
set (honored before any color init), or
`colors off: TERM=<term> reports no color support (try TERM=xterm-256color)`.
This is diagnose-only; the app never forces colors past terminfo.
Instantaneous status is
truthful but structurally blind to microsecond pool tasks (a worker
finishing a sub-ms task between samples reads Idle even under full load),
so the panel derives ACTIVITY from the cumulative `tasks_completed`
counters: a worker whose counter advanced between polls rendered with a
bright active style and a `+N` count, the header shows the interval total
(`done=N`), and unchanged workers keep their resting rendering — the
first poll has no baseline and falls back to the instantaneous tags. The
Pool box is labeled
`CPU pool` for honesty — on GPU runs the CPU pool genuinely idles between
kernel-launch microtasks while the GPU does the work, so with
`device: gpu` a dim hint row says
`gpu run: pool enqueues kernels; watch batches/s + vram` (pool telemetry
refreshes at 5 Hz; the snapshot costs ~0.1 ms). The Training box shows an
`evaluating...` state on its counts line during each per-epoch evaluate
(set by the training thread around the call, cleared in a finally), so
the eval stall reads as intentional instead of a hang. Terminals without color
(`TERM=dumb` or `has_colors()` false) get the identical layout in
monochrome — style decisions live in the controller, the screen backends
(CursesScreen, and BufferScreen which records the tokens for tests)
merely apply them.

Test seams: `python3 python/btui.py --headless <keys-file>` replays a
scripted session — one event per line (named keys like `DOWN`/`ENTER`/`ESC`,
`TYPE:<text>` for literal strings, `SET:<slug>=<value>` for config fields
routed through the same validators as the forms, `#` comments) — and prints
the final screen text to stdout. The exit code is 0 on a clean run and
nonzero when the run failed (the shared error slot is set at completion)
or a force quit left the training thread live, so headless CI never
reports success after a failed run. The screen
backend is dependency-injected: `BufferScreen` is a plain string grid
implementing the curses subset the app draws through, so the UI and its
tests run without a terminal.

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
parameters and BatchNorm running statistics — in registry order. A file may
additionally carry ONE optional trailing metadata record (an opaque named
blob, `btui.config` today, that the engine transports without parsing):
any metadata payload makes the file version 3, so older builds fail cleanly
with `UnsupportedVersion`, while new builds read versions 1–3 and skip
unknown extra-record kinds inside v3. `save(..., metadata)` writes it and
`Serialization::read_metadata(path)` returns the CRC-verified payload or
"". The format
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

## Training dashboard

`BeanTensor::Helpers::Dashboard` (see `Helpers/Dashboard/Dashboard.h`) is an
in-process terminal dashboard for training runs. Each frame is a framed,
fixed-geometry block: a header strip, a Training box (batch count with a
batches/second rate, a compact latest/mean loss readout over the bounded
recent-loss ring with a sparkline — non-finite losses render as `?` cells and
are excluded from the statistics — and an epoch line), a Pool box (the worker
grid with OWN/STEAL/HELP/WAIT colors, counters, and stuck-flagging, reusing
`PoolMonitor::render_pool_body` clipped to the box interior), and a one-line
footer, all drawn with box-drawing borders at a fixed 76-cell width. It is a
training-loop utility like the ones above: the training thread feeds it, and
it only observes engine state, never participates in it. The standalone
training TUI (`python/btui.py`, see [btui](#btui)) reuses the
worker-status tags for its own Pool panel (its graphs use the vertical
eighth-block idiom, not this dashboard's horizontal ramp).

Activation is TTY-gated. The dashboard is active only when fd 2 (its sink,
`std::cerr`) is a TTY and the run allows it; otherwise it is fully inert — no
render thread is started, `report_batch`/`report_epoch` are no-ops, and the
sink is never touched. Non-TTY runs (CI, pipes, the parity ctests) are
byte-identical to running without a dashboard.

The feed API is `report_batch(loss)` for the per-batch loss and
`report_epoch(epoch, epochs, train_mean_loss, test_accuracy, test_mean_loss)`
for each finished epoch; the Python binding exposes the same surface as
`bt.Dashboard`. The per-batch loss read in dashboard mode syncs that batch —
the scalar costs whatever the caller paid to obtain it (typically a sync via
`item()`), so dashboard mode is a watching mode, not a benchmarking mode, and
the demos only pay it while the dashboard is live. `--no-dashboard` restores
the exact old pipeline with zero overhead.

Lifecycle: when active, one plain render thread (never a pool worker) redraws
at `refresh_hz` (default 2 Hz); the renderer reads the pool only through
`PoolMonitor`'s snapshot API and never calls `sync()`. There is no alternate
screen and no cursor hide: the dashboard claims a fixed-height region, and
after the first frame each tick moves the cursor up by the frame's line count
and rewrites the whole block (every content line is padded to the frame
width, so the rewrite is exact). `stop()` is idempotent, joins the render
thread, and does NOT erase — the last complete frame persists on screen and
the recorded epoch history and final summary print below it; the destructor
backstops `stop()`. In dashboard mode the demos defer their canonical epoch
lines AND any caught failure diagnostics until after `stop()` so they print
below the persisting frame in order. Ctrl+C simply dies with the partial
frame on screen (nothing to restore — same as any killed full-screen TUI).

Geometry contract: each frame re-checks the terminal size (TIOCGWINSZ on fd
2; the `terminal_size()` virtual is the test seam). A terminal narrower than
the 76-cell frame or shorter than frame height + 2 gets ONE
"terminal too small" notice line and the dashboard goes render-inert for the
run — the render thread exits after the notice, no partial frames — while
reports keep recording and `stop()` still prints the deferred history.
Mid-run resizes garble at most one tick and self-correct on the next full
redraw; while the dashboard is active, nothing else may write to the sink
(the demos defer their prints).

Usage: pass `--no-dashboard` to `BeanTensorV5CNN`, `python/digits.py`, or
`python/cnn.py` to suppress the dashboard (it defaults to on whenever stderr
is a TTY). For custom training loops, construct `bt.Dashboard(tty=...,
refresh_hz=...)` in Python, or `Helpers::Dashboard(std::cerr, sink_is_tty,
refresh_hz)` in C++, feed it from the training thread, and call `stop()` at
the end.

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
