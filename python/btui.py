#!/usr/bin/env python3
"""btui.py — standalone interactive training TUI for BeanTensor.

Core (curses-free, unit-tested without a terminal):

- LayerSpec / ModelSpec: the architecture source of truth as plain data.
- build_net: deterministic net construction (CPU + Float32 only in v1).
- diff_keep_weights: restart-with-weight-reuse copy over matching
  name+shape parameters.
- DataConfig / HyperConfig: validated run configuration.
- SharedState: lock-guarded metrics store the UI thread reads.
- run_training: the training-thread target function.
- TrainJob: thread + stop-event wrapper around run_training.

Front-end (below the core, lazily imported curses):

- BtuiApp: the controller (model editor, params forms, run lifecycle,
  key dispatch) drawing through a dependency-injected Screen.
- CursesScreen: the real-terminal backend (imports curses at construction).
- BufferScreen + run_headless: the string-grid backend and scripted driver
  behind `--headless <keys-file>`, so tests never need a terminal.

The curses import is deliberately LAZY (inside ui()/CursesScreen) so the
core imports in any environment, including terminals-less test runners.
"""

import argparse
import collections
import dataclasses
import math
import os
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field

import numpy as np

import beantensor as bt

F32 = bt.DType.Float32
CPU = bt.Device.CPU

# v1 is Float32-only. Device is CPU or — when the build has a GPU backend —
# GPU, probed once per process (weight reuse reads params through to_numpy,
# which requires CPU, so the copy path moves them CPU-ward in place).

_GPU_PROBE = {"done": False, "available": False}


def probe_gpu() -> bool:
    """One-time GPU availability probe: move a tiny tensor to bt.Device.GPU
    and see whether the engine allows it (no backend, no driver, or a busy
    device all read as unavailable). The result is cached for the process."""
    if not _GPU_PROBE["done"]:
        try:
            bt.Tensor.from_numpy(np.zeros(1, dtype=np.float32)).to(bt.Device.GPU)
            _GPU_PROBE["available"] = True
        except Exception:  # noqa: BLE001 - any engine failure means no GPU
            _GPU_PROBE["available"] = False
        _GPU_PROBE["done"] = True
    return _GPU_PROBE["available"]


def _gpu_utilization():
    """SM utilization percent (0-100) from nvidia-smi, or None when the
    probe is unavailable or unparseable. Stdlib-only by btui's no-dependency
    rule: no pynvml, just a bounded subprocess with a 2 s timeout."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        return int(proc.stdout.strip().splitlines()[0].strip())
    except (ValueError, IndexError):
        return None

MNIST_FLAT = "mnist_flat"
MNIST_NCHW = "mnist_nchw"
CSV = "csv"
DATA_SOURCES = (MNIST_FLAT, MNIST_NCHW, CSV)

ADAM = "adam"
SGD = "sgd"
OPTIMIZERS = (ADAM, SGD)

# CSV policy vocabularies exposed in the Params panel; mapped onto the
# bt.CsvNullPolicy / bt.CsvNormalize enums via getattr ("None" is not a
# dotted-access name) at load time.
CSV_NULL_POLICIES = ("KeepNaN", "FillMean", "FillMedian", "FillConstant")
CSV_NORMALIZERS = ("None", "MinMax", "Standardize")

# Checkpoint levels offered by the save form (bt.CheckpointLevel names).
SAVE_LEVELS = ("Weights", "WeightsAndGrads", "Full")

DEFAULT_TEST_ROWS = 2000


@dataclass(frozen=True)
class LayerSpec:
    """One layer: a bound module type name plus its constructor params."""

    type: str
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ModelSpec:
    """The full architecture: an ordered stack of LayerSpec entries.

    dirty: True once the layer stack has been edited away from a built or
    imported net. A clean spec with a ready (L-loaded) net resumes that net
    directly — optimizer moments and step counters intact; a dirty spec
    forces a rebuild + weight re-seed (moments reset)."""

    layers: list = field(default_factory=list)
    dirty: bool = False


def _require(layer: LayerSpec, *names) -> list:
    missing = [name for name in names if name not in layer.params]
    if missing:
        raise ValueError(
            f"layer {layer.type!r} is missing param(s): {', '.join(missing)}"
        )
    return [layer.params[name] for name in names]


def _build_layer(layer: LayerSpec, device):
    """Construct one bound module from its spec. One statement per layer, in
    spec order, so each layer's init-seed draw is pinned to its position."""
    kind = layer.type
    if kind == "Linear":
        in_features, out_features = _require(layer, "in_features", "out_features")
        return bt.Linear(int(in_features), int(out_features), F32, device)
    if kind == "Conv2d":
        in_channels, out_channels, kernel = _require(
            layer, "in_channels", "out_channels", "kernel"
        )
        return bt.Conv2d(
            int(in_channels), int(out_channels), int(kernel),
            int(layer.params.get("stride", 1)), int(layer.params.get("pad", 0)),
            F32, device,
        )
    if kind == "BatchNorm":
        (channels,) = _require(layer, "channels")
        return bt.BatchNorm(int(channels), F32, device)
    if kind == "MaxPool2d":
        (kernel,) = _require(layer, "kernel")
        return bt.MaxPool2d(int(kernel))
    if kind == "Flatten":
        return bt.Flatten()
    if kind == "Dropout":
        (p,) = _require(layer, "p")
        return bt.Dropout(float(p))
    if kind == "Relu":
        return bt.Relu()
    if kind == "Gelu":
        return bt.Gelu()
    raise ValueError(f"unknown layer type: {kind!r}")


def build_net(spec: ModelSpec, seed: int, device: str = "cpu"):
    """Build a Float32 bt.Module from a ModelSpec on the requested device.

    Determinism rule: a fresh bt.set_seed(seed), then one construction
    statement per layer in spec order — the same spec and seed always draw
    byte-identical init weights (the engine fills GPU random tensors from
    the same seeded host stream, so the draws match across devices).
    """
    dev = bt.Device.GPU if device == "gpu" else bt.Device.CPU
    bt.set_seed(int(seed))
    layers = []
    for layer in spec.layers:
        layers.append(_build_layer(layer, dev))
    return bt.Sequential(layers)


def diff_keep_weights(old, new) -> list:
    """Copy reusable parameters from a previous run's net into a fresh one.

    For every named_parameters entry present in BOTH nets with an identical
    shape, the old values overwrite the new ones. The copy path is
    host-mediated (to_numpy -> write_numpy, both CPU-only): a GPU-resident
    OLD param is moved to CPU in place first — in-place and blocking, which
    is safe because the old net is discarded post-restart — and a
    GPU-resident NEW param takes a CPU detour for the write and is moved
    back to its device afterwards. CPU-to-CPU copies never move anything.
    Everything else keeps the fresh net's reinitialized values — this
    includes BatchNorm running statistics, which are module state, not
    named parameters, and are deliberately NOT carried over in v1.

    Returns the copied parameter names in the new net's registry order.
    """
    old_params = old.named_parameters()
    new_params = new.named_parameters()
    copied = []
    for name, new_p in new_params.items():
        old_p = old_params.get(name)
        if old_p is None or old_p.shape != new_p.shape:
            continue
        if old_p.device != CPU:
            old_p.to(CPU)
        values = old_p.to_numpy()
        new_device = new_p.device
        if new_device != CPU:
            new_p.to(CPU)
        new_p.write_numpy(values)
        if new_device != CPU:
            new_p.to(new_device)
        copied.append(name)
    return copied


@dataclass(frozen=True)
class DataConfig:
    """Where the training data comes from.

    source: one of "mnist_flat", "mnist_nchw", "csv".
    train_rows: cap on the training split (None = all rows); applied on CPU
        via a to_numpy -> numpy slice -> from_numpy roundtrip before the
        tensors are used (to_numpy requires CPU, which v1 guarantees).
    test_rows: cap on the evaluation split (the per-epoch bounded evaluate).
    CSV has a single split: the same rows feed training and evaluation.
    csv_nulls / csv_normalize: the CsvNullPolicy / CsvNormalize policy names
        (as strings; "None" is reachable via getattr on the enum).
    device: "cpu" or "gpu" — gpu is valid only when the process probe found
        a GPU backend (validate rejects it otherwise).
    """

    source: str
    data_dir: str = "datasets/mnist"
    csv_path: str = ""
    train_rows: int | None = None
    test_rows: int = DEFAULT_TEST_ROWS
    csv_nulls: str = "KeepNaN"
    csv_normalize: str = "None"
    device: str = "cpu"

    def validate(self, has_gpu: bool | None = None) -> None:
        if has_gpu is None:
            has_gpu = probe_gpu()
        if self.source not in DATA_SOURCES:
            raise ValueError(
                f"data source must be one of {DATA_SOURCES}, got {self.source!r}"
            )
        if self.source == CSV and not self.csv_path:
            raise ValueError("csv data source requires csv_path")
        if self.train_rows is not None and int(self.train_rows) < 1:
            raise ValueError(f"train_rows must be >= 1 or None, got {self.train_rows!r}")
        if int(self.test_rows) < 1:
            raise ValueError(f"test_rows must be >= 1, got {self.test_rows!r}")
        if self.csv_nulls not in CSV_NULL_POLICIES:
            raise ValueError(
                f"csv_nulls must be one of {CSV_NULL_POLICIES}, got {self.csv_nulls!r}"
            )
        if self.csv_normalize not in CSV_NORMALIZERS:
            raise ValueError(
                f"csv_normalize must be one of {CSV_NORMALIZERS}, got {self.csv_normalize!r}"
            )
        if self.device not in ("cpu", "gpu"):
            raise ValueError(f"device must be 'cpu' or 'gpu', got {self.device!r}")
        if self.device == "gpu" and not has_gpu:
            raise ValueError(
                "device 'gpu' requested, but there is no gpu in this build"
            )


@dataclass(frozen=True)
class HyperConfig:
    """Optimizer and loop hyperparameters."""

    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    epochs: int = 3
    batch: int = 100
    seed: int = 1234
    optimizer: str = ADAM

    def validate(self) -> None:
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(
                f"optimizer must be one of {OPTIMIZERS}, got {self.optimizer!r}"
            )
        if int(self.epochs) < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs!r}")
        if int(self.batch) < 1:
            raise ValueError(f"batch must be >= 1, got {self.batch!r}")
        # float("inf") > 0.0 is True: finiteness must be checked explicitly,
        # otherwise lr/eps=inf trains to NaN while presenting as valid.
        if not math.isfinite(self.lr) or not self.lr > 0.0:
            raise ValueError(f"lr must be finite and > 0, got {self.lr!r}")
        if not math.isfinite(self.eps) or not self.eps > 0.0:
            raise ValueError(f"eps must be finite and > 0, got {self.eps!r}")
        if not 0.0 < self.beta1 < 1.0:
            raise ValueError(f"beta1 must be in (0, 1), got {self.beta1!r}")
        if not 0.0 < self.beta2 < 1.0:
            raise ValueError(f"beta2 must be in (0, 1), got {self.beta2!r}")


class SharedState:
    """Lock-guarded metrics store shared between the training thread (writer)
    and the UI thread (reader). All reads return copies."""

    def __init__(self, loss_capacity: int = 240):
        self._lock = threading.Lock()
        self._loss_capacity = int(loss_capacity)
        self._losses = collections.deque(maxlen=self._loss_capacity)
        self._epoch_stats = []
        self._batches_per_s = 0.0
        self._batches_done = 0
        self._running = False
        self._error = None
        self._pending = False
        self._seeding = False
        self._paused = False
        self._evaluating = False
        self._run_start = None

    def begin_run(self) -> None:
        with self._lock:
            self._losses = collections.deque(maxlen=self._loss_capacity)
            self._epoch_stats = []
            self._batches_per_s = 0.0
            self._batches_done = 0
            self._error = None
            self._running = True
            self._seeding = False
            self._paused = False
            self._evaluating = False
            self._run_start = time.perf_counter()

    def end_run(self) -> None:
        with self._lock:
            self._running = False
            self._paused = False
            self._evaluating = False

    def record_batch(self, loss: float) -> None:
        with self._lock:
            self._losses.append(loss)
            self._batches_done += 1
            elapsed = time.perf_counter() - self._run_start
            if elapsed > 0.0:
                self._batches_per_s = self._batches_done / elapsed

    def push_epoch(self, stats: dict) -> None:
        with self._lock:
            self._epoch_stats.append(stats)

    def set_error(self, text) -> None:
        with self._lock:
            self._error = text

    def set_pending(self, flag: bool) -> None:
        with self._lock:
            self._pending = bool(flag)

    def set_seeding(self, flag: bool) -> None:
        """Restart-reuse copy window: held by the training thread only while
        diff_keep_weights mutates the previous net's params in place. The
        UI refuses checkpoint saves of the aliased last_net during it."""
        with self._lock:
            self._seeding = bool(flag)

    @property
    def seeding(self) -> bool:
        with self._lock:
            return self._seeding

    def set_paused(self, flag: bool) -> None:
        """Pause flag for the training thread; checked between batches and
        before the per-epoch evaluate. UI-only control — the engine has no
        pause concept, the loop simply waits."""
        with self._lock:
            self._paused = bool(flag)

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    def set_evaluating(self, flag: bool) -> None:
        """Held by the training thread exactly around the per-epoch
        evaluate (try/finally), so the UI can render the stall as an
        intentional 'evaluating...' state instead of reading as a hang."""
        with self._lock:
            self._evaluating = bool(flag)

    @property
    def evaluating(self) -> bool:
        with self._lock:
            return self._evaluating

    @property
    def losses(self):
        with self._lock:
            return list(self._losses)

    @property
    def epoch_stats(self):
        with self._lock:
            return list(self._epoch_stats)

    @property
    def batches_per_s(self) -> float:
        with self._lock:
            return self._batches_per_s

    @property
    def batches_done(self) -> int:
        with self._lock:
            return self._batches_done

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def error(self):
        with self._lock:
            return self._error

    @property
    def loss_capacity(self) -> int:
        with self._lock:
            return self._loss_capacity

    @property
    def pending(self) -> bool:
        with self._lock:
            return self._pending

    def snapshot(self) -> dict:
        """One consistent copy of every field, for the UI redraw."""
        with self._lock:
            return {
                "losses": list(self._losses),
                "epoch_stats": list(self._epoch_stats),
                "batches_per_s": self._batches_per_s,
                "batches_done": self._batches_done,
                "running": self._running,
                "error": self._error,
                "pending": self._pending,
                "seeding": self._seeding,
                "paused": self._paused,
                "evaluating": self._evaluating,
            }


def _slice_rows(images, labels, rows):
    """Cap a split to `rows` rows on CPU: to_numpy -> numpy slice ->
    from_numpy. to_numpy requires CPU, and v1 never leaves it, so the
    roundtrip is sound (and happens before any consumer sees the tensors)."""
    if rows is None:
        return images, labels
    return (
        bt.Tensor.from_numpy(images.to_numpy()[:rows]),
        bt.Tensor.from_numpy(labels.to_numpy()[:rows]),
    )


def _labels_int64(labels):
    """evaluate() compares argmax (Int64) against the labels with a
    shared-dtype equal(), so every label tensor is normalized to Int64.
    CSV labels arrive as Int32; MNIST labels are Int64 already."""
    if labels.dtype == bt.DType.Int64:
        return labels
    return bt.Tensor.from_numpy(labels.to_numpy().astype(np.int64))


def _load_data(data_cfg: DataConfig):
    """Load the (train, test) tensor pairs per the data config."""
    if data_cfg.source == CSV:
        opts = bt.CsvOptions()
        opts.feature_dtype = F32
        # getattr because CsvNormalize.None is not a dotted-access name.
        opts.nulls = getattr(bt.CsvNullPolicy, data_cfg.csv_nulls)
        opts.normalize = getattr(bt.CsvNormalize, data_cfg.csv_normalize)
        dataset = bt.load_csv(data_cfg.csv_path, opts)
        features = dataset.features
        labels = _labels_int64(dataset.labels)
        # A CSV has one split: training and the bounded evaluate share it.
        train = _slice_rows(features, labels, data_cfg.train_rows)
        test = _slice_rows(features, labels, int(data_cfg.test_rows))
        return train, test
    load = bt.load_mnist_flat if data_cfg.source == MNIST_FLAT else bt.load_mnist_nchw
    train_images, train_labels = load(data_cfg.data_dir, True, F32, CPU)
    test_images, test_labels = load(data_cfg.data_dir, False, F32, CPU)
    train = _slice_rows(train_images, train_labels, data_cfg.train_rows)
    test = _slice_rows(test_images, test_labels, int(data_cfg.test_rows))
    return train, test


def detect_input_shape(data_cfg: DataConfig):
    """The data source's logical per-sample input shape, or None.

    MNIST is static ((784,) flat, (1, 28, 28) NCHW) and never touches the
    disk. CSV loads the file with the config's current policies and reads
    features.shape[1:]; any load failure (missing path, parse error)
    returns None — the Model panel shows "unknown" and nothing breaks.
    """
    if data_cfg.source == MNIST_FLAT:
        return (784,)
    if data_cfg.source == MNIST_NCHW:
        return (1, 28, 28)
    if data_cfg.source == CSV:
        if not data_cfg.csv_path:
            return None
        try:
            opts = bt.CsvOptions()
            opts.feature_dtype = F32
            opts.nulls = getattr(bt.CsvNullPolicy, data_cfg.csv_nulls)
            opts.normalize = getattr(bt.CsvNormalize, data_cfg.csv_normalize)
            dataset = bt.load_csv(data_cfg.csv_path, opts)
        except Exception:  # noqa: BLE001 - detection is best-effort
            return None
        return tuple(dataset.features.shape[1:])
    return None


def _parse_spec_float(text: str) -> float:
    """Spec float hypers are lossless hexfloats (Module/ModuleSpec.h
    format_hyper_f64); plain decimal parses first for robustness."""
    try:
        return float(text)
    except ValueError:
        return float.fromhex(text)


def spec_to_layer_spec(exported: dict):
    """Inverse of build_net's key tables: turn an export_spec dict (the .bt
    architecture tree) back into a ModelSpec. Sequential children give the
    layer order. dtype/device are NOT in the spec — rebuilds always use the
    app's Float32 and current device. Unknown layer types raise ValueError
    before anything is applied (no partial import).

    Key mapping from the modules' spec() emitters: Linear in_features/
    out_features, Conv2d in_channels/out_channels/kernel/stride with
    "padding" -> pad, BatchNorm "num_features" -> channels,
    MaxPool2d kernel (stride unpinned — the bound ctor always uses
    stride=kernel), Dropout p, Flatten/Relu/Gelu none.

    Returns (ModelSpec, warnings): one warning per non-default value the
    app's layer schema cannot express and the import therefore DROPS —
    Linear/Conv2d bias=0, MaxPool2d stride != kernel, BatchNorm
    momentum/eps (always present in a spec, always dropped). Callers must
    surface these; a bias=0 file rebuilt as bias=1 is an architecture
    change, not a detail.
    """
    if exported.get("type") == "Sequential":
        nodes = exported["children"]
    else:
        nodes = [exported]
    layers = []
    warnings = []
    for i, node in enumerate(nodes):
        kind = node["type"]
        hypers = node["hyperparameters"]
        if kind == "Linear":
            if hypers.get("bias") == "0":
                warnings.append(f"layer {i} Linear bias=0")
            layers.append(LayerSpec("Linear", {
                "in_features": int(hypers["in_features"]),
                "out_features": int(hypers["out_features"]),
            }))
        elif kind == "Conv2d":
            if hypers.get("bias") == "0":
                warnings.append(f"layer {i} Conv2d bias=0")
            layers.append(LayerSpec("Conv2d", {
                "in_channels": int(hypers["in_channels"]),
                "out_channels": int(hypers["out_channels"]),
                "kernel": int(hypers["kernel"]),
                "stride": int(hypers["stride"]),
                "pad": int(hypers["padding"]),
            }))
        elif kind == "BatchNorm":
            if "momentum" in hypers or "eps" in hypers:
                warnings.append(f"layer {i} BatchNorm momentum/eps")
            layers.append(LayerSpec("BatchNorm", {
                "channels": int(hypers["num_features"]),
            }))
        elif kind == "MaxPool2d":
            kernel = int(hypers["kernel"])
            stride = int(hypers.get("stride", kernel))
            if stride != kernel:
                warnings.append(f"layer {i} MaxPool2d stride={stride} (!= kernel)")
            layers.append(LayerSpec("MaxPool2d", {"kernel": kernel}))
        elif kind == "Dropout":
            layers.append(LayerSpec("Dropout", {
                "p": _parse_spec_float(hypers["p"]),
            }))
        elif kind in ("Flatten", "Relu", "Gelu"):
            layers.append(LayerSpec(kind, {}))
        else:
            raise ValueError(f"unknown layer type in file: {kind!r}")
    return ModelSpec(layers=layers), warnings


def _wait_while_paused(shared: SharedState, stop_event) -> None:
    """Block while paused, polling the stop event so s still lands promptly.
    Called between batches and before the per-epoch evaluate; loader
    position, epoch state, and optimizer state are untouched by the wait."""
    while shared.paused and not stop_event.is_set():
        time.sleep(0.05)


def ready_net_matches_device(ready_net, device: str) -> bool:
    """Whether the ready net's params already live on the target device.
    The true-resume path requires this: Adam moments live in an external
    address-keyed map and do NOT move with Tensor.to (only data and grad
    move), so a cross-device "resume" would silently zero the moments at
    the first optimizer step. A param-less net (e.g. Relu-only) matches
    any device — it carries no optimizer state."""
    target = bt.Device.GPU if device == "gpu" else bt.Device.CPU
    for _name, param in ready_net.named_parameters().items():
        if param.device != target:
            return False
    return True


def validate_arch_data_fit(spec: ModelSpec, data_cfg: DataConfig):
    """Pre-run architecture/data fit check.

    Returns None when the run is compatible, else a one-line friendly
    refusal reason. The first layer decides the effective input rank:
    Conv2d consumes rank-4 NCHW, Linear/Flatten consume rank-2 feature
    rows; mnist_nchw produces the former, mnist_flat and csv the latter.
    Rank-agnostic first layers (Relu/Gelu/Dropout/MaxPool2d/BatchNorm)
    pass either way. This turns first-layer rank errors (the raw engine
    "conv2d input must be rank-4 NCHW" family) into an actionable log line
    BEFORE the run — deeper rank errors still surface from the engine.
    """
    if not spec.layers:
        return "model has no layers"
    first = spec.layers[0].type
    nchw_data = data_cfg.source == MNIST_NCHW
    if first == "Conv2d" and not nchw_data:
        return ("first layer Conv2d needs rank-4 NCHW input "
                "{batch, channels, height, width}; this source gives rank-2 "
                "feature rows. Use image data shaped NCHW (the mnist_nchw "
                "source is one option), not a specific dataset requirement.")
    if first == "Linear" and nchw_data:
        return ("first layer Linear needs rank-2 feature rows "
                "{batch, features}; this source gives rank-4 images "
                "{batch, channels, height, width}. Use a flat source "
                "(mnist_flat or csv), or insert a Flatten layer first.")
    # Flatten accepts any contiguous input and reshapes to rank 2, so it is
    # compatible with both flat and NCHW sources.
    return None


def run_training(spec: ModelSpec, data_cfg: DataConfig, hp_cfg: HyperConfig,
                 shared: SharedState, stop_event,
                 seed_weights_from=None, has_gpu=None, ready_net=None):
    """Training-thread target: build the net, load data, run the epoch loop.

    - Determinism: a fresh bt.set_seed(hp.seed) at the start of every call
      (build_net reseeds with the same value before constructing).
    - Device: data_cfg.device ("cpu"/"gpu", validated against the probe).
      The net is built on the device; data still LOADS and SLICES on CPU
      (to_numpy requires it), then each split tensor uploads via .to()
      before the DataLoader is constructed; forward/backward/step and the
      per-epoch evaluate all run on-device.
    - Weight reuse: when seed_weights_from is given, matching name+shape
      parameters are copied in via diff_keep_weights before training
      (host-mediated across devices; see its docstring).
    - True resume: when ready_net is given AND the spec is clean
      (spec.dirty False — no edits since the L import) AND the ready net's
      params already live on the target device, the ready module trains
      directly: build_net and the copy are skipped, so Adam moments and
      step counters survive. A dirty spec, a missing ready net, or a
      DEVICE MISMATCH rebuilds and re-seeds instead — moments live in an
      external address-keyed map and do not move with Tensor.to, so
      resuming a CPU checkpoint straight onto GPU would silently zero them.
    - Pause: the shared paused flag is checked between batches (after each
      completed batch) and once more before the per-epoch evaluate; the
      wait polls the stop event so s still lands at the batch boundary.
    - Stop: stop_event is checked between batches (after each completed
      batch). The engine offers no task cancellation, so a stop lands at a
      batch boundary — up to one batch after the event is set. A stopped run
      skips the pending per-epoch evaluate.
    - Failure: any engine/runtime exception is captured as traceback text in
      shared.error (running cleared) and the call returns None instead of
      raising. Configuration errors (invalid DataConfig/HyperConfig) raise
      ValueError to the caller before the run starts.

    Returns the trained bt.Module on success (stopped or finished), None on
    failure.
    """
    if has_gpu is None:
        has_gpu = probe_gpu()
    data_cfg.validate(has_gpu)
    hp_cfg.validate()

    shared.begin_run()
    try:
        device = bt.Device.GPU if data_cfg.device == "gpu" else bt.Device.CPU
        bt.set_seed(int(hp_cfg.seed))
        if (ready_net is not None and not spec.dirty
                and ready_net_matches_device(ready_net, data_cfg.device)):
            # True resume: train the ready module itself. Adam moments and
            # step counters ride along — they only survive because the
            # device match above guarantees no .to() ever ran on these
            # params (moments are address-keyed and do not move).
            net = ready_net
        else:
            net = build_net(spec, hp_cfg.seed, device=data_cfg.device)
            if seed_weights_from is not None:
                # The copy mutates the old net's params in place (GPU params
                # move CPU-ward); last_net aliases it until this run lands, so
                # checkpoint saves of it must be refused for the window.
                shared.set_seeding(True)
                try:
                    diff_keep_weights(seed_weights_from, net)
                finally:
                    shared.set_seeding(False)

        (train_images, train_labels), (test_images, test_labels) = _load_data(data_cfg)
        if device is not CPU:
            # Upload AFTER the CPU-side slice, per tensor, before any
            # DataLoader sees the data.
            for tensor in (train_images, train_labels, test_images, test_labels):
                tensor.to(device)
        train_loader = bt.DataLoader(
            train_images, train_labels, int(hp_cfg.batch), True, int(hp_cfg.seed), False
        )
        test_loader = bt.DataLoader(
            test_images, test_labels, int(hp_cfg.batch), False, None, False
        )

        net.train()
        stopped = False
        for epoch in range(int(hp_cfg.epochs)):
            epoch_start = time.perf_counter()
            train_loader.reset()
            loss_sum = 0.0
            steps = 0
            while True:
                batch_opt = train_loader.next_batch()
                if batch_opt is None:
                    break
                features, labels = batch_opt
                net.zero_grad()
                logits = net.forward(features)
                loss = bt.softmax_cross_entropy(logits, labels)
                loss.backward(F32)
                if hp_cfg.optimizer == ADAM:
                    net.adam_step(hp_cfg.lr, hp_cfg.beta1, hp_cfg.beta2, hp_cfg.eps)
                else:
                    net.sgd_step(lr=hp_cfg.lr)
                value = loss.item()
                loss_sum += value
                steps += 1
                shared.record_batch(value)
                # Pause granularity: batch boundary. The wait polls the stop
                # event, so s still lands promptly while paused.
                _wait_while_paused(shared, stop_event)
                if stop_event.is_set():
                    stopped = True
                    break
            if stopped:
                break
            # The pause check runs before the per-epoch evaluate too.
            _wait_while_paused(shared, stop_event)
            if stop_event.is_set():
                stopped = True
                break
            for param in net.named_parameters().values():
                param.sync()
            net.eval()
            shared.set_evaluating(True)
            try:
                accuracy, mean_loss = bt.evaluate(net, test_loader)
            finally:
                shared.set_evaluating(False)
            net.train()
            train_mean_loss = loss_sum / steps if steps > 0 else 0.0
            shared.push_epoch({
                "epoch": epoch + 1,
                "epochs": int(hp_cfg.epochs),
                "train_mean_loss": train_mean_loss,
                "test_accuracy": accuracy,
                "test_mean_loss": mean_loss,
                "seconds": time.perf_counter() - epoch_start,
            })
        return net
    except Exception:  # noqa: BLE001 - the thread must never die silently
        shared.set_error(traceback.format_exc())
        return None
    finally:
        shared.end_run()


class TrainJob:
    """Thread + stop-event wrapper around run_training.

    The UI owns one TrainJob per run: start() launches the training thread,
    stop() requests a stop at the next batch boundary, join() waits for the
    engine to quiesce. After join(), job.net holds the trained module (None
    if the run failed; shared.error carries the traceback).

    The config is validated synchronously here, at construction — a bad run
    must never vanish silently inside the thread. The spec's layer list is
    copied, so UI-side edits during a run never race the build.
    """

    def __init__(self, spec: ModelSpec, data_cfg: DataConfig, hp_cfg: HyperConfig,
                 shared: SharedState, seed_weights_from=None, has_gpu=None,
                 ready_net=None):
        if has_gpu is None:
            has_gpu = probe_gpu()
        data_cfg.validate(has_gpu)
        hp_cfg.validate()
        if (
            fit_reason := validate_arch_data_fit(spec, data_cfg)
        ) is not None:
            # Direct callers get the same pre-run refusal as the UI.
            raise ValueError(fit_reason)
        self.spec = ModelSpec(layers=list(spec.layers), dirty=spec.dirty)
        self.data_cfg = data_cfg
        self.hp_cfg = hp_cfg
        self.shared = shared
        self.seed_weights_from = seed_weights_from
        self.ready_net = ready_net
        self.has_gpu = has_gpu
        self.stop_event = threading.Event()
        self.net = None
        # Non-daemon as a safety net: if the UI is force-quit with the run
        # still live, interpreter shutdown waits for the thread instead of
        # tearing the engine down underneath it.
        self._thread = threading.Thread(
            target=self._target, name="btui-train", daemon=False
        )

    def _target(self) -> None:
        self.net = run_training(
            self.spec, self.data_cfg, self.hp_cfg, self.shared,
            self.stop_event, self.seed_weights_from, has_gpu=self.has_gpu,
            ready_net=self.ready_net,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def join(self, timeout=None) -> None:
        self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()


# ---------------------------------------------------------------------------
# Terminal front-end
#
# The UI is built on a dependency-injected Screen abstraction: CursesScreen
# wraps a real curses stdscr, BufferScreen is a plain string grid for tests
# and the --headless scripted mode. Neither the app controller nor the
# headless driver imports curses — only CursesScreen does, at construction.
# ---------------------------------------------------------------------------

# Sparkline ramp and status vocabulary shared with the embedded dashboard
# (Helpers/Dashboard, Helpers/PoolMonitor.h): non-finite cells render '?',
# worker cells use the IDLE/OWN/STEAL/HELP/WAIT tags and their colors.
STATUS_TAGS = {
    "Idle": "IDLE",
    "RunningOwn": "OWN",
    "RunningStolen": "STEAL",
    "RunningHelped": "HELP",
    "HelpWaiting": "WAIT",
}
STUCK_THRESHOLD_MS = 10000
POOL_REFRESH_SECONDS = 0.2
REDRAW_SECONDS = 0.1
# Minimum gap between redraws triggered by input events. Terminals deliver
# wheel-scroll as bursts of arrow-key events; an unconditional per-key
# redraw fires a full erase+draw per tick, which stomps the periodic pacing
# and starves the training thread. Coalescing bursts at 30 Hz keeps input
# feedback imperceptibly fresh without the storm.
KEY_REDRAW_MIN_SECONDS = 1.0 / 30.0
MIN_ROWS = 24
MIN_COLS = 80
HELP_LINE = ("arrows move | tab panel | enter edit | a add | d del | j/k reorder | "
             "r run | p pause | s stop | w save | W seed | L import | q quit")

# Logical style tokens. The controller passes style NAMES to the screens;
# BufferScreen maps names to these tokens (recorded per cell for tests),
# CursesScreen maps them to real color pairs. With color disabled both
# return attr 0 — the monochrome fallback.
STYLE_HEADER = 1
STYLE_SELECTED = 2
STYLE_DIM = 3
STYLE_ERROR = 4
STYLE_STUCK = 5
STYLE_IDLE = 6
STYLE_OWN = 7
STYLE_STEAL = 8
STYLE_HELP = 9
STYLE_WAIT = 10
STYLE_STATE_RUNNING = 11
STYLE_STATE_PAUSED = 12
STYLE_STATE_IDLE = 13
STYLE_BUSY = 14

_BUFFER_STYLE_TOKENS = {
    "header": STYLE_HEADER,
    "selected": STYLE_SELECTED,
    "dim": STYLE_DIM,
    "error": STYLE_ERROR,
    "stuck": STYLE_STUCK,
    "st:Idle": STYLE_IDLE,
    "st:RunningOwn": STYLE_OWN,
    "st:RunningStolen": STYLE_STEAL,
    "st:RunningHelped": STYLE_HELP,
    "st:HelpWaiting": STYLE_WAIT,
    "st:busy": STYLE_BUSY,
    "state:running": STYLE_STATE_RUNNING,
    "state:paused": STYLE_STATE_PAUSED,
    "state:idle": STYLE_STATE_IDLE,
}

# Layer param schema for the editor: (param name, validator kind, add-flow
# default) per layer type, in display order. Validator kinds: "int1" (>= 1),
# "int0" (>= 0), "prob" ([0, 1)).
_LAYER_PARAMS = {
    "Linear": (("in_features", "int1", 2), ("out_features", "int1", 2)),
    "Conv2d": (("in_channels", "int1", 1), ("out_channels", "int1", 4),
               ("kernel", "int1", 3), ("stride", "int1", 1), ("pad", "int0", 0)),
    "BatchNorm": (("channels", "int1", 4),),
    "MaxPool2d": (("kernel", "int1", 2),),
    "Dropout": (("p", "prob", 0.25),),
    "Flatten": (),
    "Relu": (),
    "Gelu": (),
}
LAYER_TYPES = tuple(_LAYER_PARAMS)


def _parse_int(text, minimum):
    try:
        value = int(text.strip())
    except ValueError:
        raise ValueError("not an integer") from None
    if value < minimum:
        raise ValueError(f"must be >= {minimum}")
    return value


def _parse_float(text):
    try:
        return float(text.strip())
    except ValueError:
        raise ValueError("not a number") from None


def _parse_prob(text):
    value = _parse_float(text)
    if not 0.0 <= value < 1.0:
        raise ValueError("must be in [0, 1)")
    return value


def _validate_value(kind, text):
    if kind == "int1":
        return _parse_int(text, 1)
    if kind == "int0":
        return _parse_int(text, 0)
    if kind == "prob":
        return _parse_prob(text)
    raise ValueError(f"unknown validator kind {kind!r}")


def _fmt6(value):
    """C++ default ostream formatting: 6 significant digits, general."""
    return f"{value:.6g}"


VGRAPH_RAMP = " ▁▂▃▄▅▆▇█"


def _vgraph(values, height):
    """Map a series to `height` rows of eighth-block bars (btop idiom).

    Each column is one sample; each row is one string of len(values)
    chars, TOP row first (the last string is the base row). Normalization
    mirrors the sparkline family: exact zero renders blank, any nonzero
    value floors at one level, and a flat-but-busy series sits at
    mid-height. Non-finite samples render '?'. Empty input yields `height`
    empty strings; a zero series yields blank (space) rows — an idle
    graph slot is blank rows, never dotted filler.
    """
    if height < 1 or not values:
        return [""] * max(0, height)
    finite = [v for v in values if math.isfinite(v)]
    lo = min(finite) if finite else 0.0
    hi = max(finite) if finite else 0.0
    span = hi - lo
    top = height * 8
    rows = []
    for r in range(height):
        row = []
        for value in values:
            if not math.isfinite(value):
                row.append("?")
                continue
            level = 0
            if value == 0.0:
                level = 0  # exact zero stays blank even inside a busy window
            elif span > 0.0:
                level = min(top, max(1, int((value - lo) / span * top)))
            else:
                level = top // 2  # flat but busy: mid-height, never blank
            fill = max(0, min(8, level - (height - 1 - r) * 8))
            row.append(VGRAPH_RAMP[fill])
        rows.append("".join(row))
    return rows


DGRAPH_CENTER = "─"


def _dgraph(values, height):
    """Centered DELTA graph with a RESERVED baseline row.

    Layout for an odd height (the caller clamps to odd): the middle row is a
    continuous dim centerline ('─' in every finite column — bars never touch
    it), rows above hold positive deltas, rows below hold negative deltas.
    Each side-cell carries two levels via half blocks: ▀/▄ = one level
    (partial, hugging the baseline), █ = two levels (full). Up bars fill
    bottom-up (▄ partial at their top end), down bars fill top-down (▀
    partial at their bottom end), so a bar of ANY size visually connects to
    the centerline. Per column: the first visible sample deltas against the
    window mean, the rest against their predecessor; heights are scaled by
    the largest |delta| in the window under a SHARED per-side budget, so up
    and down read comparably. Zero delta and empty/flat windows render the
    centerline only; non-finite samples mark the centerline with '?'.
    """
    if height < 3 or not values:
        return [""] * max(0, height)
    # Odd height keeps the baseline centered with equal sides.
    height |= 1
    finite_vals = [v for v in values if math.isfinite(v)]
    mean = sum(finite_vals) / len(finite_vals) if finite_vals else 0.0
    deltas = []
    for i, value in enumerate(values):
        if not math.isfinite(value):
            deltas.append(float("nan"))
            continue
        if i == 0:
            deltas.append(value - mean)
        else:
            prev = values[i - 1]
            deltas.append(value - prev if math.isfinite(prev) else float("nan"))
    scale = max((abs(d) for d in deltas if math.isfinite(d)), default=0.0)
    center = (height - 1) // 2
    side_rows = center
    side_levels = side_rows * 2
    rows = []
    for r in range(height):
        row = []
        for d in deltas:
            if not math.isfinite(d):
                row.append("?" if r == center else " ")
                continue
            if r == center:
                row.append(DGRAPH_CENTER)
                continue
            if scale == 0.0 or d == 0.0:
                row.append(" ")
                continue
            level = min(side_levels, max(1, round(abs(d) / scale * side_levels)))
            if d > 0.0:
                # Up: bottom-up fill; row center-1 is adjacent to the line.
                if r >= center:
                    row.append(" ")
                    continue
                dist = center - r
                remaining = level - (dist - 1) * 2
                if remaining <= 0:
                    row.append(" ")
                else:
                    # Partial or full; a half always fills the side of the
                    # cell toward the baseline so the bar stays connected.
                    row.append("▄" if remaining == 1 else "█")
            else:
                # Down: top-down fill; row center+1 is adjacent to the line.
                if r <= center:
                    row.append(" ")
                    continue
                dist = r - center
                remaining = level - (dist - 1) * 2
                if remaining <= 0:
                    row.append(" ")
                else:
                    row.append("▀" if remaining == 1 else "█")
        rows.append("".join(row))
    return rows


def parse_keys(text):
    """Parse a headless keys file into input events.

    One event per line; blank lines and #-comments are skipped. "TYPE:<txt>"
    expands to one character event per character. "SET:<slug>=<value>" is a
    config directive routed through the same validators as the forms. Any
    other line is a key name ("DOWN", "ENTER", "TAB", "ESC", "BACKSPACE") or
    a single character ("r", "q", ...).
    """
    events = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("TYPE:"):
            events.extend(line[len("TYPE:"):])
        elif line.startswith("SET:"):
            slug, sep, value = line[len("SET:"):].partition("=")
            if not sep:
                raise ValueError(f"bad SET line (want SET:slug=value): {raw!r}")
            events.append(("SET", slug.strip(), value.strip()))
        else:
            events.append(line)
    return events


class BufferScreen:
    """Plain string-grid Screen for tests and --headless: addstr writes into
    a char grid (clipped silently, like a curses window that never errors),
    text() dumps the frame for assertions. Every cell also records the style
    token it was written with (see the STYLE_* constants) so tests can
    assert color decisions; color=False is the monochrome fallback and
    records attr 0 everywhere."""

    def __init__(self, rows=34, cols=110, color=True):
        self.rows = int(rows)
        self.cols = int(cols)
        self.color = bool(color)
        self.refresh_count = 0
        self._grid = [[" "] * self.cols for _ in range(self.rows)]
        self._attrs = [[0] * self.cols for _ in range(self.rows)]

    def getmaxyx(self):
        return self.rows, self.cols

    def style(self, name):
        if not self.color:
            return 0
        return _BUFFER_STYLE_TOKENS.get(name, 0)

    def addstr(self, y, x, text, attr=0):
        if y < 0 or y >= self.rows:
            return
        cx = x
        for ch in text:
            if 0 <= cx < self.cols:
                self._grid[y][cx] = ch
                self._attrs[y][cx] = attr
            cx += 1

    def erase(self):
        for row, arow in zip(self._grid, self._attrs):
            for i in range(self.cols):
                row[i] = " "
                arow[i] = 0

    def refresh(self):
        self.refresh_count += 1

    def poll_key(self):
        return None

    def attr_at(self, y, x):
        return self._attrs[y][x]

    def attrs_in_row(self, y):
        return list(self._attrs[y])

    def all_attrs(self):
        return [attr for row in self._attrs for attr in row]

    def text(self):
        lines = ["".join(row).rstrip() for row in self._grid]
        while lines and not lines[-1]:
            lines.pop()
        return "\n".join(lines)


class CursesScreen:
    """Real-terminal Screen. Imports curses at construction, keeping the
    module-level import surface curses-free (the lazy-curses contract)."""

    def __init__(self, stdscr):
        import curses

        self._curses = curses
        self.stdscr = stdscr
        try:
            curses.curs_set(0)  # some terminals can't hide the cursor; fine
        except curses.error:
            pass
        stdscr.keypad(True)  # decode arrow/function keys into KEY_* codes
        stdscr.timeout(50)
        self._colors = False
        # Startup color diagnostics, logged once to the log strip by the
        # app: a monochrome session must explain itself. NO_COLOR (the
        # convention) is honored before any color init — diagnose only,
        # never force colors past terminfo.
        pairs = ((1, curses.COLOR_GREEN), (2, curses.COLOR_CYAN),
                 (3, curses.COLOR_YELLOW), (4, curses.COLOR_MAGENTA),
                 (5, curses.COLOR_RED), (6, curses.COLOR_WHITE))
        if os.environ.get("NO_COLOR"):
            self.color_info = "colors off: NO_COLOR set"
        elif curses.has_colors():
            curses.start_color()
            try:
                curses.use_default_colors()
                bg = -1  # terminal default background
            except curses.error:
                # Some color-capable terminfo lacks default-color support;
                # fall back to an explicit black background instead of
                # failing app init.
                bg = curses.COLOR_BLACK
            # 1 green (OWN), 2 cyan (STEAL), 3 yellow (HELP), 4 magenta
            # (WAIT), 5 red (stuck/error), 6 white (dim idle).
            for pair, color in pairs:
                curses.init_pair(pair, color, bg)
            self._colors = True
            self.color_info = f"colors on ({len(pairs)} pairs)"
        else:
            term = os.environ.get("TERM", "<unset>")
            self.color_info = (
                f"colors off: TERM={term} reports no color support "
                "(try TERM=xterm-256color)")

    def getmaxyx(self):
        return self.stdscr.getmaxyx()

    def style(self, name):
        if not self._colors:
            return 0  # monochrome fallback: attr 0 everywhere
        curses = self._curses

        def pair(n):
            return curses.color_pair(n) if self._colors else 0

        if name == "header":
            return curses.A_BOLD
        if name == "selected":
            return curses.A_REVERSE
        if name == "dim":
            return curses.A_DIM
        if name in ("error", "stuck"):
            return pair(5) | curses.A_BOLD
        status_styles = {
            "st:Idle": pair(6) | curses.A_DIM,
            "st:RunningOwn": pair(1),
            "st:RunningStolen": pair(2),
            "st:RunningHelped": pair(3),
            "st:HelpWaiting": pair(4) | curses.A_BOLD,
            # Interval-active worker (counter advanced since last poll):
            # bright green so activity visually dominates the resting dim
            # cells (bold-white read as plain text in the user's theme).
            "st:busy": pair(1) | curses.A_BOLD,
        }
        run_state_styles = {
            "state:running": pair(1),           # green: live run
            "state:paused": pair(3),            # yellow: paused
            "state:idle": pair(6) | curses.A_DIM,
        }
        return status_styles.get(name, run_state_styles.get(name, 0))

    def addstr(self, y, x, text, attr=0):
        try:
            self.stdscr.addstr(y, x, text, attr)
        except self._curses.error:
            pass  # edge/bottom-right writes raise under curses; clip instead

    def erase(self):
        self.stdscr.erase()

    def refresh(self):
        self.stdscr.refresh()

    def poll_key(self):
        """Non-blocking key read normalized to the app's key vocabulary."""
        curses = self._curses
        try:
            ch = self.stdscr.getch()
        except curses.error:
            return None
        if ch == -1:
            return None
        named = {
            curses.KEY_UP: "UP", curses.KEY_DOWN: "DOWN",
            curses.KEY_LEFT: "LEFT", curses.KEY_RIGHT: "RIGHT",
            curses.KEY_BTAB: "TAB", 9: "TAB",
            10: "ENTER", 13: "ENTER", curses.KEY_ENTER: "ENTER",
            27: "ESC",
            curses.KEY_BACKSPACE: "BACKSPACE", 127: "BACKSPACE", 8: "BACKSPACE",
        }
        if ch in named:
            return named[ch]
        if 32 <= ch < 127:
            return chr(ch)
        return None


class _Form:
    """One-line footer edit. text starts empty: an empty commit keeps the
    current value (edit forms) or the default (add/param forms); a form with
    empty_default commits that value on an empty commit instead."""

    __slots__ = ("prompt", "name", "commit", "empty_default", "text")

    def __init__(self, prompt, name, commit, empty_default=None):
        self.prompt = prompt
        self.name = name
        self.commit = commit
        self.empty_default = empty_default
        self.text = ""


class BtuiApp:
    """The UI controller: owns the working spec/config, the live TrainJob,
    selection/focus state, the log strip, and the input dispatch. All
    drawing goes through the injected Screen; all engine mutation happens
    on the training thread."""

    def __init__(self, screen, data_dir="datasets/mnist", has_gpu=None):
        self.screen = screen
        # GPU availability, probed once per process unless the caller
        # injects it (tests do); gates the device field and run validation.
        self.has_gpu = probe_gpu() if has_gpu is None else bool(has_gpu)
        self._launch_data_dir = data_dir
        self.shared = SharedState()
        self.spec = ModelSpec()
        self.data_cfg = DataConfig(source=MNIST_FLAT, data_dir=data_dir)
        self.hp_cfg = HyperConfig()
        self.job = None
        self.last_net = None
        self.seed_weights = None
        self.ready_net = None
        self.focus = "model"
        self.layer_sel = 0
        self.param_cursor = 0
        self.field_sel = 0
        self.form = None
        self.quit_requested = False
        self._stopping = False
        self._force_quit = False
        self._form_queue = []
        self._add_type = None
        self._add_params = None
        self._log = collections.deque(maxlen=200)
        self._pool = None
        self._pool_at = 0.0
        self._pool_warned = False
        self._pool_deltas = None
        self._pool_spark = collections.deque(maxlen=60)
        self._vram = None
        self._gpu_util = None
        self._save_path = None
        self._save_level_form = None
        self._detected_key = None
        self._detected_shape = None
        self._fields = self._make_fields()
        self.log("press a to add a layer, r to run, q to quit")
        color_info = getattr(screen, "color_info", None)
        if color_info:
            # Startup color diagnostics (see CursesScreen): visible once in
            # the log strip so a monochrome session explains itself.
            self.log(color_info)

    def detected_input_shape(self):
        """detect_input_shape for the current data config, cached and
        recomputed only when the detection-relevant fields change (a CSV
        probe reads the file, so it must not run per redraw)."""
        cfg = self.data_cfg
        key = (cfg.source, cfg.csv_path, cfg.csv_nulls, cfg.csv_normalize)
        if key != self._detected_key:
            self._detected_key = key
            self._detected_shape = detect_input_shape(cfg)
        return self._detected_shape

    def _detected_input_scalar(self, pname):
        """The detected shape reduced to the scalar an input param wants:
        the flattened element count for Linear.in_features, the channel
        count for Conv2d.in_channels. None when detection failed."""
        shape = self.detected_input_shape()
        if shape is None:
            return None
        if pname == "in_channels":
            return shape[0]
        total = 1
        for dim in shape:
            total *= dim
        return total

    # -- logging -----------------------------------------------------------

    def log(self, line, error=False):
        # Entries carry an error flag so the log strip can render failures
        # (run failed / cannot start / invalid input) in the ERROR style.
        self._log.append((line, bool(error)))

    @property
    def log_lines(self):
        return [line for line, _error in self._log]

    # -- run lifecycle -----------------------------------------------------

    @property
    def busy(self):
        return self.job is not None

    def _collect_finished(self):
        if self.job is None or self.job.is_alive():
            return
        self.job.join()
        if self.job.net is not None:
            self.last_net = self.job.net
            self.log("run finished; net kept for save/reuse")
        else:
            error = self.shared.error
            if error:
                self.log(f"run failed: {error.strip().splitlines()[-1]}",
                         error=True)
        self.job = None
        if self._stopping:
            # The quit requested while the run was draining lands now that
            # the engine is quiesced.
            self.quit_requested = True

    def shutdown(self):
        """Stop a live run and wait for it, staying responsive.

        Never returns while the training thread is alive, unless the user
        force-quits: the wait keeps redrawing and processing input, and a
        second q forces an early return (documented escape hatch — the
        non-daemon thread then finishes on its own before interpreter
        teardown). Idempotent.
        """
        if self.job is None:
            return
        self._stopping = True
        self.job.stop()
        self.log("waiting for the training thread (press q again to force quit)")
        while self.job is not None and self.job.is_alive() and not self._force_quit:
            self.job.join(0.05)
            self.redraw()
            key = self.screen.poll_key()
            if key is not None:
                self.handle_key(key)
        self._collect_finished()

    def _restart(self):
        if self.job is not None:
            self.job.stop()
            self.job.join(10.0)
            self._collect_finished()
            if self.job is not None:
                self.log("cannot restart: the live run did not stop in time")
                return
        if not self.spec.layers:
            self.log("cannot start: the model has no layers (press a to add one)",
                     error=True)
            return
        if (
            fit_reason := validate_arch_data_fit(self.spec, self.data_cfg)
        ) is not None:
            # Refuse before any engine call: the friendly fix, not the raw
            # rank error the engine would raise mid-run.
            self.log(f"cannot start: {fit_reason}", error=True)
            return
        use_ready = (
            self.ready_net is not None
            and not self.spec.dirty
            and ready_net_matches_device(self.ready_net, self.data_cfg.device)
        )
        device_changed = (
            self.ready_net is not None
            and not self.spec.dirty
            and not use_ready
        )
        seed_from = None if use_ready else (
            self.seed_weights if self.seed_weights is not None else self.last_net
        )
        try:
            job = TrainJob(self.spec, self.data_cfg, self.hp_cfg, self.shared,
                           seed_weights_from=seed_from, has_gpu=self.has_gpu,
                           ready_net=self.ready_net if use_ready else None)
        except ValueError as exc:
            self.log(f"cannot start: {exc}", error=True)
            return
        self.job = job
        job.start()
        self.shared.set_pending(False)
        if use_ready:
            # The L-loaded module trains directly: moments and step
            # counters survive (true resume — same device, verified above).
            self.log("resumed: optimizer state intact")
        elif device_changed:
            # Moments cannot move across devices (address-keyed state), so
            # a cross-device resume is rebuilt honestly, weights carried.
            self.log("device changed: rebuilt with weight re-seed, "
                     "optimizer state reset")
        elif seed_from is not None:
            self.log("rebuilt: weights re-seeded")
        else:
            self.log("run started")
        self.ready_net = None
        self.seed_weights = None

    def _stop(self):
        if self.job is None:
            self.log("no run to stop")
            return
        self.job.stop()
        self.log("stop requested (lands at a batch boundary)")

    def _toggle_pause(self):
        if self.job is None or not self.job.is_alive():
            self.log("no run to pause")
            return
        paused = not self.shared.paused
        self.shared.set_paused(paused)
        self.log("paused (p resumes)" if paused else "resumed")

    # -- input dispatch ----------------------------------------------------

    def handle_key(self, key):
        if self.quit_requested:
            return
        if self.form is not None:
            self._handle_form_key(key)
            return
        if key == "UP":
            self._move(-1)
        elif key == "DOWN":
            self._move(1)
        elif key == "LEFT":
            self._move_param(-1)
        elif key == "RIGHT":
            self._move_param(1)
        elif key == "TAB":
            self.focus = "params" if self.focus == "model" else "model"
        elif key == "ENTER":
            self._open_edit()
        elif key == "a":
            self._start_add()
        elif key == "d":
            self._delete_layer()
        elif key == "j":
            self._reorder_layer(1)
        elif key == "k":
            self._reorder_layer(-1)
        elif key == "r":
            self._restart()
        elif key == "s":
            self._stop()
        elif key == "p":
            self._toggle_pause()
        elif key == "w":
            self._start_save()
        elif key == "W":
            self._start_load()
        elif key == "L":
            self._start_import()
        elif key == "q":
            if self.job is not None and self.job.is_alive() and not self._force_quit:
                if self._stopping:
                    # Documented escape hatch: leave immediately. The
                    # non-daemon training thread finishes on its own before
                    # interpreter teardown.
                    self._force_quit = True
                    self.quit_requested = True
                    self.log("force quit: the training thread finishes on its own")
                else:
                    self._stopping = True
                    self.job.stop()
                    self.log("stopping the run before quit (press q again to force)")
            else:
                self.quit_requested = True

    def _handle_form_key(self, key):
        if key == "ESC":
            self.form = None
            self._form_queue = []
            self._cancel_add()
            self.log("edit cancelled")
            return
        if key == "ENTER":
            form = self.form
            text = form.text
            if text == "" and form.empty_default is None:
                self._advance_form()  # keep the current value / default
                return
            try:
                form.commit(text if text else form.empty_default)
            except ValueError as exc:
                self.log(f"invalid {form.name} '{text}': {exc}", error=True)
                self.form = None
                self._form_queue = []
                self._cancel_add()
                return
            self._advance_form()
            return
        if key == "BACKSPACE":
            self.form.text = self.form.text[:-1]
            return
        if len(key) == 1 and key.isprintable():
            self.form.text += key

    def _advance_form(self):
        if self._form_queue:
            self.form = self._form_queue.pop(0)
        else:
            self.form = None
            self._finish_add()

    def _move(self, delta):
        if self.focus == "model":
            count = len(self.spec.layers)
            if count == 0:
                return
            self.layer_sel = min(count - 1, max(0, self.layer_sel + delta))
            self.param_cursor = 0
        else:
            self.field_sel = min(len(self._fields) - 1, max(0, self.field_sel + delta))

    def _move_param(self, delta):
        if self.focus != "model" or not self.spec.layers:
            return
        count = len(self.spec.layers[self.layer_sel].params)
        if count == 0:
            return
        self.param_cursor = min(count - 1, max(0, self.param_cursor + delta))

    # -- model editing -----------------------------------------------------

    def _mark_pending(self):
        self.shared.set_pending(True)

    @staticmethod
    def _describe_layer(layer):
        if not layer.params:
            return layer.type
        params = " ".join(f"{key}={value}" for key, value in layer.params.items())
        return f"{layer.type}({params})"

    def _start_add(self):
        self._cancel_add()
        known = ", ".join(LAYER_TYPES)
        self._form_queue = []
        self.form = _Form(f"layer type ({known}): ", "layer type", self._commit_add_type)

    def _cancel_add(self):
        self._add_type = None
        self._add_params = None

    def _finish_add(self):
        if self._add_type is None:
            return
        layer = LayerSpec(self._add_type, dict(self._add_params))
        self.spec = ModelSpec(layers=[*self.spec.layers, layer], dirty=True)
        self.layer_sel = len(self.spec.layers) - 1
        self.param_cursor = 0
        self._cancel_add()
        self._mark_pending()
        self.log(f"added {self._describe_layer(layer)}")

    def _commit_add_type(self, text):
        value = text.strip()
        if value not in _LAYER_PARAMS:
            raise ValueError(f"unknown layer type '{value}'")
        self._add_type = value
        self._add_params = {}
        queue = []
        for pname, kind, default in _LAYER_PARAMS[value]:
            self._add_params[pname] = default
            queue.append(_Form(
                f"{value} {pname} (default {default}): ", pname,
                self._make_add_param_commit(pname, kind),
            ))
        self._form_queue = queue

    def _make_add_param_commit(self, pname, kind):
        def commit(text):
            self._add_params[pname] = _validate_value(kind, text)

        return commit

    def _delete_layer(self):
        if not self.spec.layers:
            self.log("nothing to delete")
            return
        layers = list(self.spec.layers)
        removed = layers.pop(self.layer_sel)
        self.spec = ModelSpec(layers=layers, dirty=True)
        self.layer_sel = min(self.layer_sel, max(0, len(layers) - 1))
        self.param_cursor = 0
        self._mark_pending()
        self.log(f"removed {self._describe_layer(removed)}")

    def _reorder_layer(self, delta):
        """Move the selected layer one slot (j = down, k = up). Edges clamp
        silently; the selection follows the moved layer."""
        count = len(self.spec.layers)
        if count < 2:
            return
        target = self.layer_sel + delta
        if target < 0 or target >= count:
            return
        layers = list(self.spec.layers)
        layers[self.layer_sel], layers[target] = layers[target], layers[self.layer_sel]
        self.spec = ModelSpec(layers=layers, dirty=True)
        self.layer_sel = target
        self._mark_pending()

    def _open_edit(self):
        if self.focus == "model":
            if not self.spec.layers:
                self.log("no layers yet - press a to add one")
                return
            layer = self.spec.layers[self.layer_sel]
            schema = _LAYER_PARAMS.get(layer.type, ())
            if not schema:
                self.log(f"{layer.type} has no editable params")
                return
            idx = min(self.param_cursor, len(schema) - 1)
            pname, kind, _default = schema[idx]
            current = layer.params.get(pname, "")
            self.form = _Form(
                f"{layer.type} {pname} (current {current}): ", pname,
                self._make_layer_param_commit(self.layer_sel, pname, kind),
            )
            # First layer's input dim pre-fills from the detected data shape
            # (still user-editable: clear or replace before committing).
            if self.layer_sel == 0 and pname in ("in_features", "in_channels"):
                detected = self._detected_input_scalar(pname)
                if detected is not None:
                    self.form.prompt = (
                        f"{layer.type} {pname} (current {current}, "
                        f"detected {detected} from data - edit or accept): "
                    )
                    self.form.text = str(detected)
            return
        slug, label, get, apply = self._fields[self.field_sel]
        if slug == "device" and not self.has_gpu:
            self.log("device: no gpu in this build")
            return
        self.form = _Form(f"{label} (current {get()}): ", slug, apply)

    def _make_layer_param_commit(self, layer_index, pname, kind):
        def commit(text):
            value = _validate_value(kind, text)
            layers = list(self.spec.layers)
            layer = layers[layer_index]
            params = dict(layer.params)
            params[pname] = value
            layers[layer_index] = LayerSpec(layer.type, params)
            self.spec = ModelSpec(layers=layers, dirty=True)
            self._mark_pending()

        return commit

    # -- save / load -------------------------------------------------------

    def _start_save(self):
        if self.last_net is None:
            self.log("nothing to save yet (no finished run)")
            return
        if self.shared.seeding:
            # The restart copy is mutating the old net's params in place
            # right now; last_net aliases it — a checkpoint write would race
            # the buffer swap.
            self.log("save busy: restart copy in flight")
            return
        levels = ", ".join(SAVE_LEVELS)
        self._save_level_form = _Form(
            f"save level ({levels}, default Weights): ", "save level",
            self._commit_save_level, empty_default="Weights",
        )
        # Strip step first: after the path, before the level prompt (which
        # only runs when not stripping).
        self._form_queue = [_Form(
            "strip for sharing? (y/n, default n): ", "strip",
            self._commit_strip, empty_default="n",
        )]
        self.form = _Form("save path (default btui_model.bt): ", "save path",
                          self._commit_save_path, empty_default="btui_model.bt")

    def _commit_save_path(self, text):
        self._save_path = text.strip()

    def _commit_strip(self, text):
        """strip=y writes a recipient-safe copy on the spot: Weights level
        (no gradients or Adam moments) and NO config metadata. strip=n
        queues the level prompt and the metadata-embedding save."""
        answer = text.strip().lower()
        if answer not in ("y", "n"):
            raise ValueError("must be y or n")
        if answer == "n":
            self._form_queue.append(self._save_level_form)
            return
        try:
            bt.save(self.last_net, self._save_path, True,
                    bt.CheckpointLevel.Weights, metadata="")
        except Exception as exc:  # engine errors arrive as RuntimeError
            raise ValueError(str(exc)) from None
        self.log(f"shared copy: {self._save_path} — weights only, config stripped")
        self._save_path = None

    def _commit_save_level(self, text):
        level_name = text.strip()
        if level_name not in SAVE_LEVELS:
            raise ValueError(f"must be one of {', '.join(SAVE_LEVELS)}")
        # Embed the run config as opaque key=value metadata (the engine
        # carries it unparsed; L restores it defensively on import).
        try:
            bt.save(self.last_net, self._save_path, True,
                    getattr(bt.CheckpointLevel, level_name),
                    metadata=self._config_metadata())
        except Exception as exc:  # engine errors arrive as RuntimeError
            raise ValueError(str(exc)) from None
        self.log(f"saved {self._save_path} ({level_name})")
        self._save_path = None

    def _config_metadata(self):
        """The DataConfig + HyperConfig as key=value lines for the .bt
        metadata record. Floats use 17 significant digits so the values
        round-trip exactly."""
        data = self.data_cfg
        hp = self.hp_cfg
        return "\n".join([
            f"source={data.source}",
            f"data_dir={data.data_dir}",
            f"csv_path={data.csv_path}",
            f"train_rows={'all' if data.train_rows is None else data.train_rows}",
            f"test_rows={data.test_rows}",
            f"csv_nulls={data.csv_nulls}",
            f"csv_normalize={data.csv_normalize}",
            f"device={data.device}",
            f"optimizer={hp.optimizer}",
            f"lr={hp.lr:.17g}",
            f"beta1={hp.beta1:.17g}",
            f"beta2={hp.beta2:.17g}",
            f"eps={hp.eps:.17g}",
            f"epochs={hp.epochs}",
            f"batch={hp.batch}",
            f"seed={hp.seed}",
        ])

    def _apply_config_metadata(self, text):
        """Apply a .bt metadata blob (key=value lines) onto the current
        config, defensively: unknown keys are ignored, values that fail the
        field validators keep the current setting, and the log line names
        what actually changed."""
        applied = []
        for line in text.splitlines():
            key, sep, value = line.partition("=")
            if not sep:
                continue
            slug = "data_source" if key == "source" else key
            for fslug, _label, _get, apply in self._fields:
                if fslug != slug:
                    continue
                try:
                    apply(value)
                except ValueError:
                    break  # malformed value: keep the current setting
                applied.append(f"{key}={value}")
                break
        if applied:
            self.log(f"restored config: {' '.join(applied)}")

    def _start_load(self):
        self.form = _Form("load .bt path: ", "load path", self._commit_load)

    def _commit_load(self, text):
        path = text.strip()
        if not path:
            raise ValueError("must not be empty")
        try:
            self.seed_weights = bt.load(path)
        except Exception as exc:
            raise ValueError(str(exc)) from None
        self.log(f"loaded {path}; seeds matching params on the next run")

    def _start_import(self):
        self.form = _Form("import architecture .bt path: ", "import path",
                          self._commit_import)

    def _commit_import(self, text):
        """L: load a .bt, export its architecture spec, and replace the
        ModelSpec with it; the loaded module becomes the pending
        seed-weights source for the next r. An unknown layer type raises
        before anything is applied — no partial import."""
        path = text.strip()
        if not path:
            raise ValueError("must not be empty")
        try:
            module = bt.load(path)
            exported = bt.export_spec(module)
            new_spec, drops = spec_to_layer_spec(exported)
        except ValueError:
            raise
        except Exception as exc:  # engine load errors arrive as RuntimeError
            raise ValueError(str(exc)) from None
        self.spec = new_spec  # clean: it describes the loaded module exactly
        self.seed_weights = module
        self.ready_net = module
        self.layer_sel = 0
        self.param_cursor = 0
        self._mark_pending()
        dropped = f" (dropped: {', '.join(drops)})" if drops else ""
        self.log(f"imported {len(new_spec.layers)} layers from {path}{dropped}; "
                 "weights seed the next run")
        # The optional training-config metadata rides the same file. EVERY
        # import starts from fresh defaults (the launch --data-dir aside) and
        # then applies what the metadata holds — absent keys default,
        # malformed values default, and a metadata-less file voids to pure
        # defaults, never leaving stale values from the previous session.
        self.data_cfg = DataConfig(source=MNIST_FLAT, data_dir=self._launch_data_dir)
        self.hp_cfg = HyperConfig()
        try:
            metadata = bt.read_metadata(path)
        except Exception:
            metadata = ""
        if metadata:
            self._apply_config_metadata(metadata)

    # -- params panel fields -------------------------------------------------

    def _make_fields(self):
        return [
            ("data_source", "data source", lambda: self.data_cfg.source,
             self._apply_data_source),
            ("data_dir", "data dir", lambda: self.data_cfg.data_dir,
             self._apply_data_dir),
            ("csv_path", "csv path", lambda: self.data_cfg.csv_path,
             self._apply_csv_path),
            ("csv_nulls", "csv nulls", lambda: self.data_cfg.csv_nulls,
             self._apply_csv_nulls),
            ("csv_normalize", "csv normalize", lambda: self.data_cfg.csv_normalize,
             self._apply_csv_normalize),
            ("train_rows", "train rows",
             lambda: "all" if self.data_cfg.train_rows is None
             else str(self.data_cfg.train_rows),
             self._apply_train_rows),
            ("test_rows", "test rows", lambda: str(self.data_cfg.test_rows),
             self._apply_test_rows),
            ("optimizer", "optimizer", lambda: self.hp_cfg.optimizer,
             self._apply_optimizer),
            ("lr", "lr", lambda: _fmt6(self.hp_cfg.lr), self._apply_lr),
            ("beta1", "beta1", lambda: _fmt6(self.hp_cfg.beta1),
             self._apply_beta1),
            ("beta2", "beta2", lambda: _fmt6(self.hp_cfg.beta2),
             self._apply_beta2),
            ("eps", "eps", lambda: _fmt6(self.hp_cfg.eps), self._apply_eps),
            ("epochs", "epochs", lambda: str(self.hp_cfg.epochs),
             self._apply_epochs),
            ("batch", "batch", lambda: str(self.hp_cfg.batch), self._apply_batch),
            ("seed", "seed", lambda: str(self.hp_cfg.seed), self._apply_seed),
            ("device", "device", self._device_display, self._apply_device),
        ]

    def _device_display(self):
        if not self.has_gpu:
            return "cpu (no gpu in this build)"
        return self.data_cfg.device

    def _apply_device(self, text):
        value = text.strip().lower()
        if value not in ("cpu", "gpu"):
            raise ValueError("must be cpu or gpu")
        if value == "gpu" and not self.has_gpu:
            raise ValueError("no gpu in this build")
        self.data_cfg = dataclasses.replace(self.data_cfg, device=value)
        self._mark_pending()

    def apply_setting(self, slug, text):
        """Headless SET: directive — same validators as the forms."""
        for fslug, _label, get, apply in self._fields:
            if fslug == slug:
                try:
                    apply(text)
                except ValueError as exc:
                    self.log(f"invalid {slug} '{text}': {exc}", error=True)
                else:
                    self.log(f"set {slug} = {get()}")
                return
        self.log(f"unknown setting '{slug}'")

    def _apply_data_source(self, text):
        value = text.strip()
        if value not in DATA_SOURCES:
            raise ValueError(f"must be one of {', '.join(DATA_SOURCES)}")
        self.data_cfg = dataclasses.replace(self.data_cfg, source=value)
        self._mark_pending()

    def _apply_data_dir(self, text):
        value = text.strip()
        if not value:
            raise ValueError("must not be empty")
        self.data_cfg = dataclasses.replace(self.data_cfg, data_dir=value)
        self._mark_pending()

    def _apply_csv_path(self, text):
        self.data_cfg = dataclasses.replace(self.data_cfg, csv_path=text.strip())
        self._mark_pending()

    def _apply_csv_nulls(self, text):
        value = text.strip()
        if value not in CSV_NULL_POLICIES:
            raise ValueError(f"must be one of {', '.join(CSV_NULL_POLICIES)}")
        self.data_cfg = dataclasses.replace(self.data_cfg, csv_nulls=value)
        self._mark_pending()

    def _apply_csv_normalize(self, text):
        value = text.strip()
        if value not in CSV_NORMALIZERS:
            raise ValueError(f"must be one of {', '.join(CSV_NORMALIZERS)}")
        self.data_cfg = dataclasses.replace(self.data_cfg, csv_normalize=value)
        self._mark_pending()

    def _apply_train_rows(self, text):
        value = text.strip().lower()
        rows = None if value == "all" else _parse_int(value, 1)
        self.data_cfg = dataclasses.replace(self.data_cfg, train_rows=rows)
        self._mark_pending()

    def _apply_test_rows(self, text):
        self.data_cfg = dataclasses.replace(self.data_cfg,
                                            test_rows=_parse_int(text, 1))
        self._mark_pending()

    def _apply_optimizer(self, text):
        value = text.strip().lower()
        if value not in OPTIMIZERS:
            raise ValueError(f"must be one of {', '.join(OPTIMIZERS)}")
        self.hp_cfg = dataclasses.replace(self.hp_cfg, optimizer=value)
        self._mark_pending()

    def _apply_lr(self, text):
        value = _parse_float(text)
        if not math.isfinite(value) or not value > 0.0:
            raise ValueError("must be finite and > 0")
        self.hp_cfg = dataclasses.replace(self.hp_cfg, lr=value)
        self._mark_pending()

    def _apply_beta1(self, text):
        self.hp_cfg = dataclasses.replace(self.hp_cfg,
                                          beta1=self._parse_unit_interval(text))
        self._mark_pending()

    def _apply_beta2(self, text):
        self.hp_cfg = dataclasses.replace(self.hp_cfg,
                                          beta2=self._parse_unit_interval(text))
        self._mark_pending()

    @staticmethod
    def _parse_unit_interval(text):
        value = _parse_float(text)
        if not 0.0 < value < 1.0:
            raise ValueError("must be in (0, 1)")
        return value

    def _apply_eps(self, text):
        value = _parse_float(text)
        if not math.isfinite(value) or not value > 0.0:
            raise ValueError("must be finite and > 0")
        self.hp_cfg = dataclasses.replace(self.hp_cfg, eps=value)
        self._mark_pending()

    def _apply_epochs(self, text):
        self.hp_cfg = dataclasses.replace(self.hp_cfg, epochs=_parse_int(text, 1))
        self._mark_pending()

    def _apply_batch(self, text):
        self.hp_cfg = dataclasses.replace(self.hp_cfg, batch=_parse_int(text, 1))
        self._mark_pending()

    def _apply_seed(self, text):
        self.hp_cfg = dataclasses.replace(self.hp_cfg, seed=_parse_int(text, 0))
        self._mark_pending()

    # -- drawing -------------------------------------------------------------

    def _put(self, y, x, text, style_name=None, width=None):
        if width is not None:
            text = text[:width]
        attr = self.screen.style(style_name) if style_name else 0
        self.screen.addstr(y, x, text, attr)

    def _box(self, y, x, h, w, title):
        if h < 2 or w < 4:
            return
        label = f"─ {title} "[:w - 2]
        self._put(y, x, "┌" + label + "─" * (w - 2 - len(label)) + "┐")
        for row in range(1, h - 1):
            self._put(y + row, x, "│")
            self._put(y + row, x + w - 1, "│")
        self._put(y + h - 1, x, "└" + "─" * (w - 2) + "┘")

    def redraw(self):
        self._collect_finished()
        now = time.perf_counter()
        if self._pool is None or now - self._pool_at >= POOL_REFRESH_SECONDS:
            try:
                new_pool = bt.pool_snapshot()
                # Interval activity: the cumulative counters integrate the
                # whole poll window, catching microsecond pool tasks that
                # instantaneous status sampling structurally misses. The
                # first poll has no baseline and falls back to the tags.
                self._pool_deltas = None
                if self._pool is not None:
                    prev = {wk["worker_id"]: wk["tasks_completed"]
                            for wk in self._pool["workers"]}
                    self._pool_deltas = {
                        wk["worker_id"]: wk["tasks_completed"]
                        - prev.get(wk["worker_id"], 0)
                        for wk in new_pool["workers"]
                    }
                    # Aggregate activity ring: tasks completed this interval.
                    self._pool_spark.append(sum(self._pool_deltas.values()))
                self._pool = new_pool
                self._pool_warned = False
            except Exception as exc:  # best-effort panel; never kill the UI
                if not self._pool_warned:
                    self.log(f"pool_snapshot failed: {exc}", error=True)
                    self._pool_warned = True
            # VRAM + SM-utilization telemetry rides the same 2 Hz poll, only
            # while the run targets GPU; gpu_memory returns None on CPU
            # builds / failures by contract, and _gpu_utilization is None on
            # any probe failure, so no try/except.
            if self.has_gpu and self.data_cfg.device == "gpu":
                self._vram = bt.gpu_memory()
                self._gpu_util = _gpu_utilization()
            else:
                self._vram = None
                self._gpu_util = None
            self._pool_at = now

        snap = self.shared.snapshot()
        screen = self.screen
        screen.erase()
        rows, cols = screen.getmaxyx()
        if rows < MIN_ROWS or cols < MIN_COLS:
            self._put(0, 0,
                      f"terminal too small (need at least {MIN_COLS}x{MIN_ROWS})",
                      "error")
            screen.refresh()
            return

        header_state = ("paused" if snap["paused"]
                        else "running" if snap["running"] else "idle")
        header_prefix = "btui - BeanTensor training TUI   run: "
        self._put(0, 0, header_prefix, "header")
        self._put(0, len(header_prefix), header_state, f"state:{header_state}")
        suffix = ""
        if snap["pending"]:
            suffix += "   [edits pending]"
        if self.seed_weights is not None:
            suffix += "   [.bt seed loaded]"
        self._put(0, len(header_prefix) + len(header_state), suffix, "header")

        main_h = rows - 4  # header + 2 log rows + footer
        left_w = max(34, min(cols - 30, (cols * 2) // 5))
        right_w = cols - left_w
        # Params holds 16 fields; give it the larger share so the device row
        # is visible without scrolling at the default geometry.
        model_h = max(5, (main_h * 2) // 5)
        params_h = main_h - model_h
        train_h = max(6, (main_h * 2) // 5)
        pool_h = main_h - train_h
        self._draw_model(1, 0, model_h, left_w)
        self._draw_params(1 + model_h, 0, params_h, left_w)
        self._draw_train(1, left_w, train_h, right_w, snap)
        self._draw_pool(1 + train_h, left_w, pool_h, right_w)

        log_render = list(self._log)[-2:]
        while len(log_render) < 2:
            log_render.insert(0, ("", False))
        for offset, (line, is_error) in enumerate(log_render):
            self._put(rows - 3 + offset, 0, line[:cols],
                      "error" if is_error else "dim")

        if self.form is not None:
            self._put(rows - 1, 0,
                      f"{self.form.prompt}{self.form.text}█"[:cols], "header")
        else:
            self._put(rows - 1, 0, HELP_LINE[:cols], "dim")
        screen.refresh()

    def _draw_model(self, y, x, h, w):
        self._box(y, x, h, w, "Model")
        inner = w - 2
        # Bottom interior row is reserved for the detected data input shape.
        if h >= 4:
            shape = self.detected_input_shape()
            if shape is None:
                shown = "unknown"
            elif len(shape) == 1:
                shown = str(shape[0])
            else:
                shown = "(" + ", ".join(str(dim) for dim in shape) + ")"
            self._put(y + h - 2, x + 1, f"  data input: {shown}"[:inner], "dim")
        layers = self.spec.layers
        if not layers:
            self._put(y + 1, x + 1, "(empty - press a to add a layer)"[:inner], "dim")
            return
        visible = h - 3
        start = max(0, self.layer_sel - visible + 1)
        for row in range(visible):
            i = start + row
            if i >= len(layers):
                break
            layer = layers[i]
            selected = self.focus == "model" and i == self.layer_sel
            marker = ">" if selected else " "
            parts = []
            for j, (key, value) in enumerate(layer.params.items()):
                piece = f"{key}={value}"
                if selected and j == self.param_cursor:
                    piece = f"[{piece}]"
                parts.append(piece)
            line = f"{marker} {i} {layer.type}"
            if parts:
                line += "  " + " ".join(parts)
            self._put(y + 1 + row, x + 1, line[:inner],
                      "selected" if selected else None)

    def _draw_params(self, y, x, h, w):
        self._box(y, x, h, w, "Params")
        inner = w - 2
        visible = h - 2
        start = max(0, self.field_sel - visible + 1)
        for row in range(visible):
            i = start + row
            if i >= len(self._fields):
                break
            slug, label, get, _apply = self._fields[i]
            selected = self.focus == "params" and i == self.field_sel
            marker = ">" if selected else " "
            # The locked device row is greyed (and refuses edits) when the
            # probe found no GPU.
            style = "selected" if selected else (
                "dim" if slug == "device" and not self.has_gpu else None
            )
            self._put(y + 1 + row, x + 1, f"{marker} {label}: {get()}"[:inner],
                      style)

    def _draw_train(self, y, x, h, w, snap):
        self._box(y, x, h, w, "Training")
        inner = w - 2
        losses = snap["losses"]
        counts = (f"batches={snap['batches_done']}  "
                  f"b/s={_fmt6(snap['batches_per_s'])}  "
                  f"window={len(losses)}/{self.shared.loss_capacity}")
        if self.data_cfg.device == "gpu" and self._vram is not None:
            gib = 2 ** 30
            counts += (f"  vram: {_fmt6(self._vram['used_bytes'] / gib)}"
                       f"/{_fmt6(self._vram['total_bytes'] / gib)} GiB")
            if self._gpu_util is not None:
                counts += f"  sm: {self._gpu_util}%"
        if snap["evaluating"]:
            counts += "  evaluating..."
        self._put(y + 1, x + 1, counts[:inner], "dim")
        last = _fmt6(losses[-1]) if losses else "-"
        finite = [v for v in losses if math.isfinite(v)]
        mean = _fmt6(sum(finite) / len(finite)) if finite else "-"
        self._put(y + 2, x + 1, f"last={last}  mean={mean}"[:inner])
        # btop-style vertical loss graph: the recent-window tail, as many
        # samples as the box is wide, 4 rows tall when the box allows
        # (clamped so the epoch line still fits below it).
        # Centered delta loss graph (btop-style vertical bars around a dim
        # centerline): up = worse, down = improving, scaled by the window's
        # largest |delta|. The centerline row renders dim, the bars default.
        # Height floor is 2 so the graph always carries direction — at 80x24
        # (h=8) it gets 3 rows; the epoch line below yields when h allows it.
        graph_h = max(3, min(7, h - 5))
        visible = losses[-inner:] if inner > 0 else []
        center = (graph_h - 1) // 2
        for offset, graph_row in enumerate(_dgraph(visible, graph_h)):
            self._put(y + 3 + offset, x + 1, graph_row[:inner],
                      "dim" if offset == center else None)
        if snap["epoch_stats"]:
            st = snap["epoch_stats"][-1]
            eline = (f"epoch {st['epoch']}/{st['epochs']}  "
                     f"train loss {_fmt6(st['train_mean_loss'])}  "
                     f"test acc {_fmt6(st['test_accuracy'] * 100)}%  "
                     f"test loss {_fmt6(st['test_mean_loss'])}")
        else:
            eline = "epoch -  waiting for the first epoch"
        epoch_row = y + 3 + graph_h
        if epoch_row < y + h - 1:
            self._put(epoch_row, x + 1, eline[:inner])

    def _draw_pool(self, y, x, h, w):
        # The label is honest about WHAT is shown: the CPU pool. On GPU
        # runs the CPU pool idles between kernel-launch microtasks while
        # the GPU does the work — the hint row says so instead of letting
        # an all-IDLE grid read as a fault.
        self._box(y, x, h, w, "CPU pool")
        inner = w - 2
        pool = self._pool
        if pool is None:
            self._put(y + 1, x + 1, "pool snapshot unavailable"[:inner], "dim")
            return
        header_fields = (f"workers={pool['worker_count']}  "
                         f"pending={pool['pending_tasks_total']}")
        if pool["stopping"]:
            header_fields += "  STOPPING"
        deltas = self._pool_deltas
        if deltas is not None:
            header_fields += f"  done={sum(deltas.values())}"
        self._put(y + 1, x + 1, header_fields[:inner])
        # Aggregate activity as a 3-row vertical block graph (btop idiom):
        # tasks completed per poll interval over the recent ring window.
        # Blank rows mean idle — no filler characters. The graph is fed
        # only the ring tail that fits the box: the ring holds 60 samples,
        # but a narrow box interior clips the window, never the border.
        visible = list(self._pool_spark)[-inner:] if inner > 0 else []
        graph_rows = _vgraph([float(v) for v in visible], 3)
        for offset, graph_row in enumerate(graph_rows):
            self._put(y + 2 + offset, x + 1, graph_row[:inner])
        grid_start = y + 5
        if self.data_cfg.device == "gpu" and grid_start < y + h - 2:
            self._put(grid_start, x + 1,
                      "gpu run: pool enqueues kernels; watch batches/s + vram"[:inner],
                      "dim")
            grid_start += 1
        workers = pool["workers"]
        cell_w = 14
        per_row = max(1, inner // cell_w)
        max_rows = max(0, y + h - 2 - grid_start)
        need_rows = (len(workers) + per_row - 1) // per_row
        shown = 0
        for r in range(min(need_rows, max_rows)):
            cx = x + 1
            for c in range(per_row):
                idx = r * per_row + c
                if idx >= len(workers):
                    break
                wkr = workers[idx]
                delta = deltas.get(wkr["worker_id"], 0) if deltas is not None else 0
                if delta > 0:
                    # Active cells DOMINATE: bright work tag + count, and
                    # never the IDLE string (it read as idle to users).
                    cell = f"#{wkr['worker_id']:02d} ~{delta}"
                    style = "st:busy"
                elif deltas is not None and wkr["status"] == "Idle":
                    # Resting cells RECEDE: dim lowercase, visually background.
                    cell = f"#{wkr['worker_id']:02d} idle"
                    style = "dim"
                else:
                    # First poll (no baseline) or a worker mid-task with no
                    # completions this interval: instantaneous tag.
                    tag = STATUS_TAGS.get(wkr["status"], "????")
                    cell = f"#{wkr['worker_id']:02d} {tag}"
                    style = f"st:{wkr['status']}"
                    if wkr["status"] != "Idle":
                        cell += f" {wkr['task_running_ms']}ms"
                        if wkr["task_running_ms"] > STUCK_THRESHOLD_MS:
                            style = "stuck"
                self._put(grid_start + r, cx, cell, style)
                cx += cell_w
            shown += 1
        totals_row = grid_start + shown
        if totals_row < y + h - 1:
            hidden = len(workers) - min(len(workers), max_rows * per_row)
            totals = (f"done={sum(wk['tasks_completed'] for wk in workers)}"
                      f"  stolen={sum(wk['tasks_stolen'] for wk in workers)}"
                      f"  helped={sum(wk['tasks_helped'] for wk in workers)}")
            if hidden > 0:
                totals += f"  (+{hidden} workers)"
            self._put(totals_row, x + 1, totals[:inner], "dim")


def run_headless(keys_path, data_dir="datasets/mnist", rows=34, cols=110):
    """Scripted session on the BufferScreen backend: same input dispatch and
    redraw as the live UI, key events read from a file (see parse_keys).
    After the events are consumed the driver keeps ticking until any live
    run finishes (run-to-idle), then returns (exit_code, final_screen_text).
    The exit code is nonzero when the run failed (shared.error set at
    completion) or shutdown left a live job behind (force quit) — headless
    CI must never report success after a failed run.
    """
    screen = BufferScreen(rows, cols)
    app = BtuiApp(screen, data_dir=data_dir)
    with open(keys_path, "r", encoding="utf-8") as fh:
        events = parse_keys(fh.read())
    app.redraw()
    idx = 0
    while not app.quit_requested:
        if idx < len(events):
            event = events[idx]
            idx += 1
            if isinstance(event, tuple):
                _tag, slug, value = event
                app.apply_setting(slug, value)
            else:
                app.handle_key(event)
            app.redraw()
        elif app.busy:
            time.sleep(0.005)
            app.redraw()
        else:
            break
    app.shutdown()
    app.redraw()
    failed = app.shared.error is not None or app.busy
    return (1 if failed else 0), screen.text()


def ui(data_dir="datasets/mnist"):
    """The curses front-end. Kept lazy: importing btui never touches curses,
    so the core is importable (and testable) without a terminal. A terminal
    curses cannot drive (bad TERM, no terminfo) gets one concise fallback
    line instead of a traceback, and exits nonzero."""
    import curses
    import locale

    locale.setlocale(locale.LC_ALL, "")
    try:
        curses.wrapper(_curses_loop, data_dir)
    except curses.error as exc:
        print(f"btui: curses could not initialize on this terminal ({exc}); "
              "check TERM/terminfo, or use --headless <keys-file> for the "
              "scripted backend", file=sys.stderr)
        raise SystemExit(1) from exc


def _curses_loop(stdscr, data_dir):
    screen = CursesScreen(stdscr)
    app = BtuiApp(screen, data_dir=data_dir)
    app.redraw()
    last_draw = time.perf_counter()
    try:
        while not app.quit_requested:
            key = screen.poll_key()
            if key is not None:
                app.handle_key(key)
                now = time.perf_counter()
                if now - last_draw >= KEY_REDRAW_MIN_SECONDS:
                    app.redraw()
                    last_draw = now
            elif time.perf_counter() - last_draw >= REDRAW_SECONDS:
                app.redraw()
                last_draw = time.perf_counter()
    finally:
        app.shutdown()
        app.redraw()  # final frame persists after curses restores the tty


def main() -> int:
    parser = argparse.ArgumentParser(description="BeanTensor interactive training TUI")
    parser.add_argument("--data-dir", default="datasets/mnist",
                        help="directory holding the uncompressed MNIST IDX files")
    parser.add_argument("--headless", metavar="KEYS_FILE", default=None,
                        help="run a scripted session on the headless test backend "
                             "and print the final screen text to stdout")
    args = parser.parse_args()
    if args.headless is not None:
        code, text = run_headless(args.headless, data_dir=args.data_dir)
        print(text)
        return code
    ui(data_dir=args.data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
