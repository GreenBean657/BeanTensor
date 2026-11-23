import sys

import numpy as np

import beantensor as bt

# --- Task 8: GPU smoke ---
# ctest bindings_gpu (registered only on GPU-backend builds, and invoked only
# through gpu-test.sh) passes --gpu; ONLY this section runs in that mode, the
# CPU surface below is covered by the plain bindings_smoke ctest on the same
# build. Same synthetic mini-training as Task 4(b) — 30 steps of forward /
# softmax_cross_entropy / backward(Float32) / adam_step on a 4->8->4 net —
# trained once on CPU and once on Device::GPU, each after a fresh
# set_seed(1234) so both nets draw identical init weights (the engine fills
# GPU random tensors from the same seeded host stream, Tensor/Ops/
# TensorOps.cpp set_random_gpu). The two final losses must agree within 1e-3:
# an approximate sanity bound only. Float reduction order is backend-defined
# (Tensor.h reduction docs), so cross-device bit parity is NOT claimed here.
# The same discipline then repeats on the conv stack: a tiny
# Conv2d(1,4,3)->BatchNorm(4)->Relu->MaxPool2d(2)->Flatten->Linear net
# trained 5 steps,
# once on CPU and once on Device::GPU after a fresh set_seed(4321), with the
# same 1e-3 approximate cross-device bound — reduction order is
# backend-defined, so this is a drift sanity check, not a parity claim.
if "--gpu" in sys.argv:
    import math

    GPU = bt.Device.GPU
    F32_GPU = bt.DType.Float32
    GPU_X = np.array(
        [
            [1.0, 0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0, 0.5],
            [0.0, 0.0, 1.0, 0.25],
            [1.0, 1.0, 0.0, -0.5],
            [0.0, 1.0, 1.0, 0.75],
            [1.0, 0.0, 1.0, -0.25],
        ],
        dtype=np.float32,
    )
    GPU_LABELS = np.array([0, 1, 2, 3, 1, 2], dtype=np.int64)

    def gpu_smoke_final_loss(device):
        bt.set_seed(1234)
        net = bt.Sequential([
            bt.Linear(4, 8, F32_GPU, device),
            bt.Relu(),
            bt.Linear(8, 4, F32_GPU, device),
        ])
        net.train()
        x = bt.Tensor.from_numpy(GPU_X)
        labels = bt.Tensor.from_numpy(GPU_LABELS)
        x.to(device)
        labels.to(device)
        final = 0.0
        for _ in range(30):
            logits = net.forward(x)
            loss = bt.softmax_cross_entropy(logits, labels)
            net.zero_grad()
            loss.backward(F32_GPU)
            net.adam_step(1e-2, 0.9, 0.999, 1e-8)
            final = loss.item()
        return final

    cpu_loss = gpu_smoke_final_loss(bt.Device.CPU)
    gpu_loss = gpu_smoke_final_loss(GPU)
    print(f"bindings_gpu: cpu_loss={cpu_loss:.6f} gpu_loss={gpu_loss:.6f} "
          f"diff={abs(cpu_loss - gpu_loss):.3e}")
    assert math.isfinite(cpu_loss) and math.isfinite(gpu_loss), (
        f"non-finite final loss: cpu={cpu_loss} gpu={gpu_loss}"
    )
    assert abs(cpu_loss - gpu_loss) < 1e-3, (
        "cpu/gpu final losses diverged beyond the sanity bound "
        f"(reduction order is backend-defined, 1e-3 is approximate): "
        f"cpu={cpu_loss} gpu={gpu_loss}"
    )

    # Conv stack twin: Conv2d(1,4,3,1,0) -> BatchNorm(4) -> Relu ->
    # MaxPool2d(2) -> Flatten
    # -> Linear(36,3) on {6,1,8,8} input (conv3 drops to 6x6, pool2 to 3x3,
    # flatten width 4*3*3 = 36), 5 train steps. Named statements pin each
    # layer's init-seed draw to its position, same as the CNN stack probe
    # below (BatchNorm draws nothing — gamma/beta start at 1/0 — so inserting
    # it does not shift the Linear's draw). One net is trained on CPU, one on
    # Device::GPU, each after a
    # fresh set_seed(4321) so both draw identical init weights; the final
    # losses must agree within the same 1e-3 approximate bound as the MLP
    # above (float reduction order is backend-defined per the Tensor.h
    # reduction docs — cross-device bit parity is NOT claimed here).
    CONV_X = (
        np.arange(6 * 1 * 8 * 8, dtype=np.float32).reshape(6, 1, 8, 8) - 191.5
    ) / 64.0
    CONV_LABELS = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)

    def gpu_smoke_conv_losses(device):
        bt.set_seed(4321)
        conv = bt.Conv2d(1, 4, 3, 1, 0, F32_GPU, device)
        bn = bt.BatchNorm(4, F32_GPU, device)
        relu = bt.Relu()
        pool = bt.MaxPool2d(2)
        flat = bt.Flatten()
        fc = bt.Linear(36, 3, F32_GPU, device)
        net = bt.Sequential([conv, bn, relu, pool, flat, fc])
        net.train()
        x = bt.Tensor.from_numpy(CONV_X)
        labels = bt.Tensor.from_numpy(CONV_LABELS)
        x.to(device)
        labels.to(device)
        losses = []
        for _ in range(5):
            logits = net.forward(x)
            loss = bt.softmax_cross_entropy(logits, labels)
            net.zero_grad()
            loss.backward(F32_GPU)
            net.adam_step(1e-2, 0.9, 0.999, 1e-8)
            losses.append(loss.item())
        return losses

    cpu_conv = gpu_smoke_conv_losses(bt.Device.CPU)
    gpu_conv = gpu_smoke_conv_losses(GPU)
    print(f"bindings_gpu: conv cpu_loss={cpu_conv[-1]:.6f} "
          f"gpu_loss={gpu_conv[-1]:.6f} "
          f"diff={abs(cpu_conv[-1] - gpu_conv[-1]):.3e}")
    assert all(math.isfinite(v) for v in cpu_conv + gpu_conv), (
        f"non-finite conv loss: cpu={cpu_conv} gpu={gpu_conv}"
    )
    assert abs(cpu_conv[-1] - gpu_conv[-1]) < 1e-3, (
        "conv cpu/gpu final losses diverged beyond the sanity bound "
        f"(reduction order is backend-defined, 1e-3 is approximate): "
        f"cpu={cpu_conv[-1]} gpu={gpu_conv[-1]}"
    )
    print("BINDINGS GPU SMOKE OK")
    raise SystemExit(0)

assert isinstance(bt.__version__, str) and bt.__version__

# Enums exposed via py::enum_.
assert bt.DType.Float32 is not None
assert bt.DType.Float64 is not None
assert bt.DType.Int32 is not None
assert bt.DType.Int64 is not None
assert bt.DType.Float16 is not None
assert bt.Device.CPU is not None
assert bt.Device.GPU is not None

DTYPE_MAP = {
    np.dtype(np.float32): bt.DType.Float32,
    np.dtype(np.float64): bt.DType.Float64,
    np.dtype(np.int32): bt.DType.Int32,
    np.dtype(np.int64): bt.DType.Int64,
}


def expect_raises(exc_type, fn, label):
    try:
        fn()
    except exc_type:
        return
    except Exception as e:  # noqa: BLE001 - report the actual failure type
        raise AssertionError(
            f"{label}: expected {exc_type.__name__}, got {type(e).__name__}: {e}"
        )
    raise AssertionError(f"{label}: expected {exc_type.__name__}, nothing raised")


# (a) from_numpy/to_numpy roundtrip preserves values and shape per dtype.
for np_dt, bt_dt in DTYPE_MAP.items():
    src = (np.arange(12, dtype=np.float64) - 5.5).reshape(3, 4).astype(np_dt)
    t = bt.Tensor.from_numpy(src)
    assert t.dtype == bt_dt, f"dtype mismatch for {np_dt}"
    assert t.shape == (3, 4), f"shape mismatch for {np_dt}"
    out = t.to_numpy()
    assert out.dtype == src.dtype, f"numpy dtype mismatch for {np_dt}"
    assert out.shape == src.shape, f"numpy shape mismatch for {np_dt}"
    assert np.array_equal(out, src), f"values not preserved for {np_dt}"

# (b) to_numpy result stays valid after the tensor is destroyed.
t = bt.Tensor.from_numpy(np.array([[1.5, -2.5]], dtype=np.float32))
out = t.to_numpy()
del t
assert np.array_equal(out, np.array([[1.5, -2.5]], dtype=np.float32))

# (c) non-contiguous arrays and unmapped dtypes are rejected with ValueError.
base = np.arange(12, dtype=np.float32).reshape(3, 4)
expect_raises(ValueError, lambda: bt.Tensor.from_numpy(base[:, ::2]), "strided view")
expect_raises(ValueError, lambda: bt.Tensor.from_numpy(base.T), "transposed view")
expect_raises(ValueError, lambda: bt.Tensor.from_numpy(np.arange(10)[::2]), "step-2 view")
expect_raises(ValueError, lambda: bt.Tensor.from_numpy(np.zeros(8, dtype=np.bool_)), "bool")

# (d) shape/dtype/device report correctly.
t = bt.Tensor.from_numpy(np.zeros((2, 3, 4), dtype=np.float64))
assert t.shape == (2, 3, 4)
assert t.dtype == bt.DType.Float64
assert t.device == bt.Device.CPU

# (e) item() reads back scalars; non-scalars raise.
t = bt.Tensor.from_numpy(np.array([3.25], dtype=np.float32))
assert t.item() == 3.25
t = bt.Tensor.from_numpy(np.array([-7], dtype=np.int64))
assert t.item() == -7.0
expect_raises(
    ValueError,
    lambda: bt.Tensor.from_numpy(np.ones((2,), dtype=np.float32)).item(),
    "item on non-scalar",
)

# (f) add works; shape mismatch raises instead of crashing.
a = bt.Tensor.from_numpy(np.ones((2, 3), dtype=np.float32))
b = bt.Tensor.from_numpy(np.full((2, 3), 2.0, dtype=np.float32))
s = a.add(b)
assert np.array_equal(s.to_numpy(), np.full((2, 3), 3.0, dtype=np.float32))
bad = bt.Tensor.from_numpy(np.ones((3, 2), dtype=np.float32))
expect_raises(Exception, lambda: a.add(bad), "shape-mismatched add")

# (g) set_zero zeroes the tensor.
t = bt.Tensor.from_numpy(np.array([5.5], dtype=np.float32))
t.set_zero()
assert t.item() == 0.0

# backward is callable through the binding and surfaces engine errors as
# Python exceptions (full autograd coverage lands with Task 4).
t = bt.Tensor.from_numpy(np.array([1.0], dtype=np.float32))
expect_raises(Exception, lambda: t.backward(bt.DType.Float32), "backward without graph")

# --- Task 3: module surface (Linear/Relu/Gelu, Sequential) ---

# (a) Sequential([Linear, Relu, Linear]) constructs via unique_ptr holder
# transfer from the Python list; forward of a {5,4} input returns {5,2}.
F32 = bt.DType.Float32
CPU = bt.Device.CPU
net = bt.Sequential([bt.Linear(4, 3, F32, CPU), bt.Relu(), bt.Linear(3, 2, F32, CPU)])
x = bt.Tensor.from_numpy(np.ones((5, 4), dtype=np.float32))
y = net.forward(x)
assert y.shape == (5, 2), f"sequential forward shape: {y.shape}"
assert y.dtype == F32

# (b) named_parameters() yields a dict with exactly the engine's registry
# names (children auto-numbered in insertion order: 0, 1, 2).
params = net.named_parameters()
assert isinstance(params, dict)
assert set(params.keys()) == {"0.weight", "0.bias", "2.weight", "2.bias"}, (
    f"named_parameters keys: {sorted(params.keys())}"
)
for name, p in params.items():
    assert isinstance(p, bt.Tensor), f"{name} is not a Tensor"
assert params["0.weight"].shape == (4, 3)
assert params["0.bias"].shape == (3,)
assert params["2.weight"].shape == (3, 2)
assert params["2.bias"].shape == (2,)
# Handles are copies sharing the C++ parameter: zeroing through the handle is
# visible to a later named_parameters() call, and the handle outlives the dict.
w = params["0.weight"]
del params
w.set_zero()
assert np.array_equal(net.named_parameters()["0.weight"].to_numpy(),
                      np.zeros((4, 3), dtype=np.float32))

# (c) str(net) is non-empty.
assert isinstance(str(net), str) and str(net)

# (d) train()/eval() don't throw; zero_grad/adam_step are callable.
net.train()
net.eval()
net.train()
net.zero_grad()
net.adam_step(1e-3, 0.9, 0.999, 1e-8)

# (e) Ownership transfer is real: the Linear survives its Python temporary.
lin = bt.Linear(4, 3, F32, CPU)
net2 = bt.Sequential([lin, bt.Gelu()])
del lin
import gc

gc.collect()
y2 = net2.forward(bt.Tensor.from_numpy(np.ones((2, 4), dtype=np.float32)))
assert y2.shape == (2, 3), f"post-del forward shape: {y2.shape}"
assert set(net2.named_parameters().keys()) == {"0.weight", "0.bias"}
del net, net2, w, x, y, y2
gc.collect()

# --- CNN module stack (conv2d, batchnorm, maxpool2d, flatten, dropout) ---

# (a) The full CNN stack as named statements in layer order — statement
# sequencing pins each layer's init-seed draw to its position (one-expression
# construction would leave the draw order to the caller, per the digits MVP).
# Geometry follows cnn.cpp's first batch: 28 -> conv3 -> 26 -> pool2 -> 13 ->
# conv3 -> 11 -> pool2 -> 5, so a C-channel second conv flattens to C * 5 * 5.
# Keeping 16 channels makes the flatten width 16 * 5 * 5 = 400, which is what
# pins fc1's fan-in below — any other channel count must change fc1's width
# to C * 25 to match.
conv1 = bt.Conv2d(1, 16, 3, 1, 0, F32, CPU)
bn1 = bt.BatchNorm(16, F32, CPU)
relu1 = bt.Relu()
pool1 = bt.MaxPool2d(2)
conv2 = bt.Conv2d(16, 16, 3, 1, 0, F32, CPU)
bn2 = bt.BatchNorm(16, F32, CPU)
relu2 = bt.Relu()
pool2 = bt.MaxPool2d(2)
flatten = bt.Flatten()
drop = bt.Dropout(0.25)
fc1 = bt.Linear(400, 128, F32, CPU)
relu3 = bt.Relu()
fc2 = bt.Linear(128, 10, F32, CPU)

# (b) Per-layer shape probe: 26 -> 13 -> 11 -> 5 -> 400, F32 throughout.
# Runs before the Sequential assembly below: the container moves each
# element's unique_ptr holder, disowning these Python handles (same ownership
# transfer as the Task 3 module-surface check).
cnn_img = bt.Tensor.from_numpy(np.ones((2, 1, 28, 28), dtype=np.float32))
probe = cnn_img
for layer, want in (
    (conv1, (2, 16, 26, 26)),
    (bn1, (2, 16, 26, 26)),
    (relu1, (2, 16, 26, 26)),
    (pool1, (2, 16, 13, 13)),
    (conv2, (2, 16, 11, 11)),
    (bn2, (2, 16, 11, 11)),
    (relu2, (2, 16, 11, 11)),
    (pool2, (2, 16, 5, 5)),
    (flatten, (2, 400)),
    (drop, (2, 400)),
):
    probe = layer.forward(probe)
    assert probe.shape == want, f"cnn probe shape: {probe.shape} != {want}"
    assert probe.dtype == F32

cnn = bt.Sequential([
    conv1, bn1, relu1, pool1, conv2, bn2, relu2, pool2,
    flatten, drop, fc1, relu3, fc2,
])

# (c) named_parameters: the registry numbers every child in insertion order
# (parameterless layers occupy their index without contributing keys).
# Conv2d and Linear register weight+bias; BatchNorm registers gamma+beta —
# the key set plus the Conv2d weight layout {C,R,S,K} pins the topology.
params = cnn.named_parameters()
assert set(params.keys()) == {
    "0.weight", "0.bias", "1.gamma", "1.beta",
    "4.weight", "4.bias", "5.gamma", "5.beta",
    "10.weight", "10.bias", "12.weight", "12.bias",
}, f"cnn named_parameters keys: {sorted(params.keys())}"
assert params["0.weight"].shape == (1, 3, 3, 16)
assert params["0.bias"].shape == (16,)
assert params["1.gamma"].shape == (16,)
assert params["1.beta"].shape == (16,)
assert params["4.weight"].shape == (16, 3, 3, 16)
assert params["4.bias"].shape == (16,)
assert params["5.gamma"].shape == (16,)
assert params["5.beta"].shape == (16,)
assert params["10.weight"].shape == (400, 128)
assert params["10.bias"].shape == (128,)
assert params["12.weight"].shape == (128, 10)
assert params["12.bias"].shape == (10,)

# (d) Full-stack forward of {2,1,28,28} ones returns {2,10}. Eval mode makes
# dropout an identity; the assertion is shape/dtype only.
cnn.eval()
logits = cnn.forward(cnn_img)
assert logits.shape == (2, 10), f"cnn forward shape: {logits.shape}"
assert logits.dtype == F32
assert str(cnn) and isinstance(str(cnn), str)

# (e) Dropout-only train/eval difference on the same input. With input ones,
# eval forward returns exactly ones while train forward returns 0 or
# 1/(1-0.5) = 2 per element, so the raw output bytes differ deterministically
# (the assertion is byte inequality, never float == on engine values).
drop_mod = bt.Dropout(0.5)
drop_x = bt.Tensor.from_numpy(np.ones((4, 8), dtype=np.float32))
drop_mod.train()
drop_train = drop_mod.forward(drop_x).to_numpy()
drop_mod.eval()
drop_eval = drop_mod.forward(drop_x).to_numpy()
assert np.array_equal(drop_eval, np.ones((4, 8), dtype=np.float32)), (
    "eval-mode dropout must be an identity on the input"
)
assert drop_train.tobytes() != drop_eval.tobytes(), (
    "train-mode dropout must differ from eval-mode on the same input"
)

del cnn, probe, logits, params, drop_mod, drop_train, drop_eval
gc.collect()

# --- Task 4: training verbs + determinism ---

import math
import os
import subprocess
import sys

# (a) softmax_cross_entropy on a synthetic 4-class problem matches a value
# hand-computed with Python's own math (tolerance 1e-6, this one check is
# against Python, not against the engine).
CE_LOGITS = np.array(
    [
        [2.0, -1.0, 0.5, 3.0],
        [0.0, 0.0, 0.0, 0.0],
        [-4.0, 4.0, 1.0, -2.0],
    ],
    dtype=np.float32,
)
CE_LABELS = np.array([3, 0, 1], dtype=np.int64)


def reference_mean_cross_entropy(logits, labels):
    # Engine semantics from Tensor::softmax_ce: per row
    # m + log(sum(exp(x - m))) - x_target with m = row max, meaned over rows.
    total = 0.0
    for row, target in zip(logits, labels):
        row_max = float(np.max(row))
        logsumexp = row_max + math.log(
            sum(math.exp(float(x) - row_max) for x in row)
        )
        total += logsumexp - float(row[int(target)])
    return total / float(len(labels))


ce_loss = bt.softmax_cross_entropy(
    bt.Tensor.from_numpy(CE_LOGITS), bt.Tensor.from_numpy(CE_LABELS)
)
assert ce_loss.shape == (1,), f"softmax_cross_entropy shape: {ce_loss.shape}"
assert ce_loss.dtype == F32
expected_ce = reference_mean_cross_entropy(CE_LOGITS, CE_LABELS)
assert abs(ce_loss.item() - expected_ce) <= 1e-6, (
    f"softmax_cross_entropy: engine {ce_loss.item()} vs reference {expected_ce}"
)

# (b) determinism: a seeded 30-step softmax_ce + backward(F32) + adam_step
# loop, with zero_grad between steps, run twice in FRESH SUBPROCESSES must
# print bit-identical losses. (c) zero_grad in the loop is what keeps the two
# runs identical: gradients never accumulate across steps.
# Division of labor: this A/B pins CROSS-PROCESS REPRODUCIBILITY of the
# binding surface (unseeded entropy, address-dependent ordering, kernel
# nondeterminism). It does NOT pin absolute RNG-stream parity with the C++
# engine: both subprocesses run the same compiled module with the same seed,
# so a change that adds an extra draw from the seeded default generator in
# the binding layer shifts both runs identically and stays green here.
# Absolute parity is pinned downstream by the Task 7 end-to-end diff of
# digits.py epoch lines against the C++ BeanTensorV5 binary on the same
# machine (docs/superpowers/plans/2026-11-14-pybind-digits-mvp.md, Task 7).
# A green A/B here does NOT prove C++ parity.
TRAIN_SCRIPT = r"""
import numpy as np
import beantensor as bt

bt.set_seed(20261114)
net = bt.Sequential([
    bt.Linear(4, 8, bt.DType.Float32, bt.Device.CPU),
    bt.Relu(),
    bt.Linear(8, 4, bt.DType.Float32, bt.Device.CPU),
])
net.train()
x = bt.Tensor.from_numpy(np.array([
    [1.0, 0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0, 0.5],
    [0.0, 0.0, 1.0, 0.25],
    [1.0, 1.0, 0.0, -0.5],
    [0.0, 1.0, 1.0, 0.75],
    [1.0, 0.0, 1.0, -0.25],
], dtype=np.float32))
labels = bt.Tensor.from_numpy(np.array([0, 1, 2, 3, 1, 2], dtype=np.int64))
losses = []
for _ in range(30):
    logits = net.forward(x)
    loss = bt.softmax_cross_entropy(logits, labels)
    net.zero_grad()
    loss.backward(bt.DType.Float32)
    net.adam_step(1e-2, 0.9, 0.999, 1e-8)
    # Hex-float: a bit-level trace, so the A/B comparison cannot hide a
    # 1-ulp drift behind %.6f rounding.
    losses.append(loss.item().hex())
print(";".join(losses))
"""


def run_training_once():
    env = dict(os.environ)
    mod_dir = os.path.dirname(os.path.abspath(bt.__file__))
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (mod_dir, env.get("PYTHONPATH", "")) if p
    )
    proc = subprocess.run(
        [sys.executable, "-c", TRAIN_SCRIPT],
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
        env=env,
    )
    return proc.stdout.strip()


run_a = run_training_once()
run_b = run_training_once()
assert run_a, "training subprocess printed nothing"
assert run_a == run_b, (
    "seeded training diverged across fresh subprocesses:\n"
    f"  run A: {run_a}\n  run B: {run_b}"
)
loss_seq = [float.fromhex(v) for v in run_a.split(";")]
assert len(loss_seq) == 30
assert all(math.isfinite(v) for v in loss_seq)
assert loss_seq[-1] < loss_seq[0], (
    f"training did not reduce loss: first {loss_seq[0]}, last {loss_seq[-1]}"
)

# --- Task 2: data path — NCHW MNIST loader, CSV loader, softmax_ce method ---

# (a) load_csv with the policy enums: a null cell under FillMean is filled
# with the NaN-skipped column mean; Standardize then centers on that same
# column_mean (Data/Csv.cpp), so the filled cell standardizes to exactly 0.0
# — asserted byte-wise, never float ==. Features come back in
# CsvOptions.feature_dtype; labels are Int32 ids in first-seen order.
SCRATCH_BINDINGS = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scratch", "bindings")
)
os.makedirs(SCRATCH_BINDINGS, exist_ok=True)

CSV_PATH = os.path.join(SCRATCH_BINDINGS, "task2_null_fill.csv")
with open(CSV_PATH, "w") as fh:
    fh.write(
        "f1,f2,cls\n"
        "1.0,10.0,a\n"
        "2.0,,a\n"
        "3.0,30.0,b\n"
        "4.0,40.0,b\n"
    )

csv_opts = bt.CsvOptions()
csv_opts.feature_dtype = bt.DType.Float32
csv_opts.nulls = bt.CsvNullPolicy.FillMean
csv_opts.normalize = bt.CsvNormalize.Standardize

# load_csv returns the full dataset object (the C++ CsvDataset mirror): the
# tensor pair plus the inspection metadata csv_demo.cpp prints — null count,
# feature/label names, and the NaN-skipped per-column stats (computed before
# filling/normalization, Data/Csv.cpp). The fixture has exactly one null
# (row 2, f2), so null_count pins the parse; the stats are NaN-skipped
# references of f1=[1,2,3,4], f2=[10,null,30,40] (mean 80/3 computed in the
# same row order the engine sums in, so exact == is safe here).
ds = bt.load_csv(CSV_PATH, csv_opts)
features, labels = ds.features, ds.labels
assert ds.null_count == 1, f"csv null_count: {ds.null_count}"
assert ds.feature_names == ["f1", "f2"], f"csv feature_names: {ds.feature_names}"
assert ds.label_names == ["a", "b"], f"csv label_names: {ds.label_names}"
assert ds.column_min == [1.0, 10.0], f"csv column_min: {ds.column_min}"
assert ds.column_max == [4.0, 40.0], f"csv column_max: {ds.column_max}"
assert ds.column_mean == [2.5, (10.0 + 30.0 + 40.0) / 3.0], (
    f"csv column_mean: {ds.column_mean}"
)
assert features.shape == (4, 2), f"csv features shape: {features.shape}"
assert labels.shape == (4,), f"csv labels shape: {labels.shape}"
assert features.dtype == F32
assert labels.dtype == bt.DType.Int32
assert np.array_equal(labels.to_numpy(), np.array([0, 0, 1, 1], dtype=np.int32)), (
    f"csv labels not in first-seen order: {labels.to_numpy()}"
)

# In-test reference: parse in float64, NaN-skip mean-fill, population-std
# standardize, narrow to float32 — the pipeline Data/Csv.cpp runs.
ref = np.array(
    [[1.0, 10.0], [2.0, np.nan], [3.0, 30.0], [4.0, 40.0]],
    dtype=np.float64,
)
col_mean = np.nanmean(ref, axis=0)
filled = np.where(np.isnan(ref), col_mean, ref)
std_dev = np.sqrt(np.mean((filled - col_mean) ** 2, axis=0))
expected_features = ((filled - col_mean) / std_dev).astype(np.float32)
got_features = features.to_numpy()
assert got_features.dtype == np.float32
assert np.isclose(got_features, expected_features, rtol=1e-6, atol=1e-7).all(), (
    f"standardized features diverged:\n{got_features}\nvs\n{expected_features}"
)
# The filled cell (row 1, col 1) holds the column mean pre-standardization,
# so it must land on exactly +0.0.
assert got_features[1, 1].tobytes() == np.float32(0.0).tobytes(), (
    f"FillMean cell must standardize to exactly 0.0, got {got_features[1, 1]!r}"
)

# A missing CSV raises RuntimeError naming the file.
MISSING_CSV = os.path.join(SCRATCH_BINDINGS, "__no_such_file__.csv")
expect_raises(RuntimeError, lambda: bt.load_csv(MISSING_CSV, csv_opts), "load_csv on a missing file")

# (b) load_mnist_nchw: rank-4 {60000,1,28,28} F32 images + Int64 labels when
# the IDX files are present (SKIP-print, never download); a missing dir
# raises RuntimeError naming the dir, same contract as load_mnist_flat.
NCHW_BAD_DIR = "datasets/__definitely_not_mnist__"
try:
    bt.load_mnist_nchw(NCHW_BAD_DIR, True, F32, CPU)
    raise AssertionError("load_mnist_nchw on a missing dir did not raise")
except RuntimeError as e:
    assert NCHW_BAD_DIR in str(e), f"error message does not mention the dir: {e}"

mnist_dir = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "mnist")
)
need = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
]
if all(os.path.isfile(os.path.join(mnist_dir, f)) for f in need):
    images4, labels4 = bt.load_mnist_nchw(mnist_dir, True, F32, CPU)
    assert images4.shape == (60000, 1, 28, 28), f"nchw train images shape: {images4.shape}"
    assert labels4.shape == (60000,), f"nchw train labels shape: {labels4.shape}"
    assert images4.dtype == F32 and labels4.dtype == bt.DType.Int64
    assert np.array_equal(
        labels4.to_numpy()[:5], np.array([5, 0, 4, 1, 9], dtype=np.int64)
    ), f"unexpected MNIST train label head: {labels4.to_numpy()[:5]}"
    img4 = images4.to_numpy()
    assert float(img4.min()) >= 0.0 and float(img4.max()) <= 1.0
    # C=1 NCHW is pixel-order contiguous, so the flat loader's rows and the
    # NCHW images carry the same payload bitwise.
    flat_images, _ = bt.load_mnist_flat(mnist_dir, True, F32, CPU)
    assert flat_images.to_numpy()[:8].tobytes() == img4[:8, 0].reshape(8, 784).tobytes(), (
        "flat and NCHW MNIST payloads diverge bitwise"
    )
    print("MNIST NCHW OK")
else:
    print("SKIP: datasets/mnist IDX files not present")

# (c) The softmax_ce METHOD is the same op as the bound free function:
# identical shape/dtype and bitwise-equal loss bytes on the same inputs.
ce_logits_t = bt.Tensor.from_numpy(CE_LOGITS)
ce_labels_t = bt.Tensor.from_numpy(CE_LABELS)
method_loss = ce_logits_t.softmax_ce(ce_labels_t)
free_loss = bt.softmax_cross_entropy(
    bt.Tensor.from_numpy(CE_LOGITS), bt.Tensor.from_numpy(CE_LABELS)
)
assert method_loss.shape == (1,) and method_loss.dtype == F32
assert free_loss.shape == (1,) and free_loss.dtype == F32
assert method_loss.to_numpy().tobytes() == free_loss.to_numpy().tobytes(), (
    "Tensor.softmax_ce and bt.softmax_cross_entropy disagree bitwise"
)

# --- Task 5: data path (DataLoader, evaluate, MNIST flat loader) ---

# (a) batching: 10 rows, batch 4. drop_last=True yields two {4,4} batches then
# None (repeatedly); drop_last=False yields {4,4}, {4,4}, {2,4}, then None.
# Labels ride along dim 0.
feat_np = np.arange(40, dtype=np.float32).reshape(10, 4)
lab_np = np.arange(10, dtype=np.int64)
feat = bt.Tensor.from_numpy(feat_np)
lab = bt.Tensor.from_numpy(lab_np)

loader_drop = bt.DataLoader(feat, lab, 4, True, None, True)
b1 = loader_drop.next_batch()
assert b1 is not None and b1[0].shape == (4, 4) and b1[1].shape == (4,)
b2 = loader_drop.next_batch()
assert b2 is not None and b2[0].shape == (4, 4)
assert loader_drop.next_batch() is None
assert loader_drop.next_batch() is None  # stays None past epoch end

loader_keep = bt.DataLoader(
    feat, lab, batch=4, shuffle=False, seed=None, drop_last=False
)
k1 = loader_keep.next_batch()
k2 = loader_keep.next_batch()
k3 = loader_keep.next_batch()
assert k3 is not None and k3[0].shape == (2, 4) and k3[1].shape == (2,)
assert loader_keep.next_batch() is None

# (b) shuffle=False preserves row order; batch contents match the source rows.
assert np.array_equal(k1[0].to_numpy(), feat_np[0:4])
assert np.array_equal(k1[1].to_numpy(), lab_np[0:4])
assert np.array_equal(k2[0].to_numpy(), feat_np[4:8])
assert np.array_equal(k3[0].to_numpy(), feat_np[8:10])
# Same explicit seed -> same shuffled order across two loaders.
s1 = bt.DataLoader(feat, lab, 10, True, 7, False)
s2 = bt.DataLoader(feat, lab, 10, True, 7, False)
assert np.array_equal(s1.next_batch()[0].to_numpy(), s2.next_batch()[0].to_numpy())
# The loader holds engine-side views of the caller's tensors; the binding must
# keep the source tensors alive after their Python temporaries are gone.
def _loader_from_temps():
    f = bt.Tensor.from_numpy(feat_np)
    l = bt.Tensor.from_numpy(lab_np)
    return bt.DataLoader(f, l, 4, False, None, False)

tmp_loader = _loader_from_temps()
assert np.array_equal(tmp_loader.next_batch()[0].to_numpy(), feat_np[0:4])

# (c) evaluate: a parameter-free Relu module separates a trivial 2-class set
# (class 0 rows [2,-1] relu to [2,0] -> argmax 0; class 1 rows [-1,2] -> [0,2]
# -> argmax 1). Ragged final batch (3,3,2) exercises the weighted mean loss.
ev_x = np.array([[2.0, -1.0]] * 4 + [[-1.0, 2.0]] * 4, dtype=np.float32)
ev_y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
ev_loader = bt.DataLoader(
    bt.Tensor.from_numpy(ev_x), bt.Tensor.from_numpy(ev_y),
    batch=3, shuffle=False, seed=None, drop_last=False,
)
model = bt.Sequential([bt.Relu()])
acc, mean_loss = bt.evaluate(model, ev_loader)
assert isinstance(acc, float) and isinstance(mean_loss, float)
assert acc == 1.0, f"evaluate accuracy on separable set: {acc}"
assert math.isfinite(mean_loss) and mean_loss < 1.0, (
    f"evaluate mean_loss on separable set: {mean_loss}"
)
# evaluate resets the loader itself: a second pass over the exhausted loader
# must agree with the first.
acc2, mean_loss2 = bt.evaluate(model, ev_loader)
assert acc2 == 1.0 and math.isfinite(mean_loss2)

# (d) a missing MNIST dir raises RuntimeError naming the dir.
BAD_DIR = "datasets/__definitely_not_mnist__"
try:
    bt.load_mnist_flat(BAD_DIR, True, F32, CPU)
    raise AssertionError("load_mnist_flat on a missing dir did not raise")
except RuntimeError as e:
    assert BAD_DIR in str(e), f"error message does not mention the dir: {e}"

# (e) real MNIST when the IDX files are present locally (SKIP, never download).
mnist_dir = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "mnist")
)
need = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
]
if all(os.path.isfile(os.path.join(mnist_dir, f)) for f in need):
    images, labels = bt.load_mnist_flat(mnist_dir, True, F32, CPU)
    assert images.shape == (60000, 784), f"train images shape: {images.shape}"
    assert labels.shape == (60000,), f"train labels shape: {labels.shape}"
    assert images.dtype == F32 and labels.dtype == bt.DType.Int64
    label_head = labels.to_numpy()[:5]
    assert np.array_equal(label_head, np.array([5, 0, 4, 1, 9], dtype=np.int64)), (
        f"unexpected MNIST train label head: {label_head}"
    )
    img = images.to_numpy()
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0
    print("MNIST OK")
else:
    print("SKIP: datasets/mnist IDX files not present")

# --- Task 6: .bt serialization (save/load with all CheckpointLevels) ---

# (a) A trained tiny Sequential saved at Full loads back with
# named_parameters() bitwise-identical to the original, compared via
# to_numpy() bytes. (b) One more identical training step on the original and
# loaded model produces bitwise-identical losses and params — the mirror of
# the C++ OOM-checkpoint test's bit-reproducible-resume contract. (c) Loading
# a garbage path raises. (d) CheckpointLevel.Weights roundtrips params only:
# optimizer state is lost by design, so the post-step params diverge even
# though the pre-step loss — a pure function of the params — still matches.
SCRATCH_BINDINGS = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scratch", "bindings")
)
os.makedirs(SCRATCH_BINDINGS, exist_ok=True)

CL = bt.CheckpointLevel
assert CL.Weights is not None and CL.WeightsAndGrads is not None and CL.Full is not None


def build_tiny_net():
    return bt.Sequential([
        bt.Linear(4, 8, F32, CPU),
        bt.Relu(),
        bt.Linear(8, 4, F32, CPU),
    ])


def param_bytes(model):
    return {name: t.to_numpy().tobytes() for name, t in model.named_parameters().items()}


def train_step(model, x, labels):
    # Mirrors the C++ oom_policy_tests train_step: zero_grad, forward, loss,
    # backward, read the loss, then adam_step. The loss bytes are captured
    # before adam_step, so the comparison is a pure function of the params.
    model.zero_grad()
    logits = model.forward(x)
    loss = bt.softmax_cross_entropy(logits, labels)
    loss.backward(F32)
    loss_bytes = loss.to_numpy().tobytes()
    model.adam_step(1e-2, 0.9, 0.999, 1e-8)
    return loss_bytes


bt.set_seed(20261114)
net = build_tiny_net()
net.train()
x = bt.Tensor.from_numpy(np.array([
    [1.0, 0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0, 0.5],
    [0.0, 0.0, 1.0, 0.25],
    [1.0, 1.0, 0.0, -0.5],
], dtype=np.float32))
labels = bt.Tensor.from_numpy(np.array([0, 1, 2, 3], dtype=np.int64))
for _ in range(3):
    train_step(net, x, labels)
orig_params = param_bytes(net)

# All three checkpoint levels roundtrip the parameters bitwise.
for level, tag in (
    (CL.Weights, "weights"),
    (CL.WeightsAndGrads, "weights_and_grads"),
    (CL.Full, "full"),
):
    path = os.path.join(SCRATCH_BINDINGS, f"task6_roundtrip_{tag}.bt")
    bt.save(net, path, True, level)
    restored = bt.load(path)
    assert isinstance(restored, bt.Module), f"{tag}: load did not return a bt.Module"
    assert param_bytes(restored) == orig_params, f"{tag}: params diverged across save/load"

# Uncompressed Full also roundtrips bitwise.
uncompressed = os.path.join(SCRATCH_BINDINGS, "task6_roundtrip_full_uncompressed.bt")
bt.save(net, uncompressed, False, CL.Full)
assert param_bytes(bt.load(uncompressed)) == orig_params

# (b) Full resumes training bit-reproducibly: one identical step on the
# original and the loaded model agrees on the loss and the params, bitwise.
full_path = os.path.join(SCRATCH_BINDINGS, "task6_full.bt")
bt.save(net, full_path, True, CL.Full)
loaded = bt.load(full_path)
loss_orig = train_step(net, x, labels)
loss_loaded = train_step(loaded, x, labels)
assert loss_orig == loss_loaded, "Full checkpoint: resumed step loss diverged bitwise"
assert param_bytes(loaded) == param_bytes(net), (
    "Full checkpoint: post-step params diverged bitwise"
)

# (c) Loading a missing path or a corrupt file raises RuntimeError.
expect_raises(
    RuntimeError,
    lambda: bt.load(os.path.join(SCRATCH_BINDINGS, "__no_such_checkpoint__.bt")),
    "load of missing path",
)
garbage_path = os.path.join(SCRATCH_BINDINGS, "task6_garbage.bt")
with open(garbage_path, "wb") as fh:
    fh.write(b"this is not a .bt checkpoint")
expect_raises(RuntimeError, lambda: bt.load(garbage_path), "load of corrupt file")

# (d) Weights drops optimizer state by design: params roundtrip bitwise, and
# the next step's loss — a pure function of the params — still matches
# bitwise, but the post-step params diverge because the loaded model's Adam
# moments and step counter were reset.
bt.set_seed(20261114)
w_net = build_tiny_net()
w_net.train()
for _ in range(3):
    train_step(w_net, x, labels)
w_path = os.path.join(SCRATCH_BINDINGS, "task6_weights.bt")
bt.save(w_net, w_path, True, CL.Weights)
w_loaded = bt.load(w_path)
assert param_bytes(w_loaded) == param_bytes(w_net)
w_loss = train_step(w_loaded, x, labels)
ref_loss = train_step(w_net, x, labels)
assert w_loss == ref_loss, "Weights roundtrip changed the pre-step loss"
assert param_bytes(w_loaded) != param_bytes(w_net), (
    "Weights checkpoint unexpectedly preserved optimizer state"
)

# --- Task 9: graph-output lifetime (result-to-parent keep_alive) ---

# The engine's graph nodes borrow raw pointers to their inputs and to module
# parameters (Module/Module.h, Tensor/Tensor.h), so a graph-output tensor must
# keep its parents alive until backward() returns. The bindings pin them with
# result-to-parent keep_alive: Module.forward's output keeps the module AND
# the input alive, softmax_cross_entropy's loss keeps logits and labels
# alive, and Tensor.add's output keeps both operands alive. The checks below
# delete the Python-side parents (explicit del + gc.collect) and compare —
# bitwise, via numpy bytes / file bytes, never float == — the loss, the
# gradients, and the post-step params against a named-local reference run.
# Gradients are compared as a WeightsAndGrads checkpoint written after
# backward but before adam_step (the format gates each record on has_grad):
# post-step params alone cannot prove grad equality because Adam's first
# step saturates to sign(g).
LIFETIME_X = np.array(
    [
        [1.0, 0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0, 0.5],
        [0.0, 0.0, 1.0, 0.25],
        [1.0, 1.0, 0.0, -0.5],
    ],
    dtype=np.float32,
)
LIFETIME_LABELS = np.array([0, 1, 2, 3], dtype=np.int64)


def lifetime_grads_blob(model):
    path = os.path.join(SCRATCH_BINDINGS, "lifetime_run_grads.bt")
    bt.save(model, path, False, CL.WeightsAndGrads)
    with open(path, "rb") as fh:
        return fh.read()


def lifetime_reference_run():
    bt.set_seed(20261114)
    net = build_tiny_net()
    net.train()
    x = bt.Tensor.from_numpy(LIFETIME_X)
    labels = bt.Tensor.from_numpy(LIFETIME_LABELS)
    logits = net.forward(x)
    loss = bt.softmax_cross_entropy(logits, labels)
    net.zero_grad()
    loss.backward(F32)
    loss_bytes = loss.to_numpy().tobytes()
    grads_blob = lifetime_grads_blob(net)
    net.adam_step(1e-2, 0.9, 0.999, 1e-8)
    return loss_bytes, grads_blob, param_bytes(net)


ref_loss_bytes, ref_grads_blob, ref_post_params = lifetime_reference_run()

# (a) Chained expression: no named local for the logits — the temporary
# forward output must stay alive through the loss alone.
bt.set_seed(20261114)
net = build_tiny_net()
net.train()
x = bt.Tensor.from_numpy(LIFETIME_X)
labels = bt.Tensor.from_numpy(LIFETIME_LABELS)
loss = bt.softmax_cross_entropy(net.forward(x), labels)
loss_bytes = loss.to_numpy().tobytes()
net.zero_grad()
loss.backward(F32)
assert loss_bytes == ref_loss_bytes, "chained: loss diverged bitwise"
assert lifetime_grads_blob(net) == ref_grads_blob, "chained: gradients diverged bitwise"
net.adam_step(1e-2, 0.9, 0.999, 1e-8)
assert param_bytes(net) == ref_post_params, "chained: post-step params diverged bitwise"
del net, x, labels, loss
gc.collect()

# (b) Explicit del + gc.collect() of the forward output, the input, and the
# labels before backward; the module stays named so grads/params stay
# readable. The loss alone must hold the whole graph.
bt.set_seed(20261114)
net = build_tiny_net()
net.train()
x = bt.Tensor.from_numpy(LIFETIME_X)
labels = bt.Tensor.from_numpy(LIFETIME_LABELS)
logits = net.forward(x)
loss = bt.softmax_cross_entropy(logits, labels)
loss_bytes = loss.to_numpy().tobytes()
net.zero_grad()
del logits, x, labels
gc.collect()
loss.backward(F32)
assert loss_bytes == ref_loss_bytes, "del parents: loss diverged bitwise"
assert lifetime_grads_blob(net) == ref_grads_blob, "del parents: gradients diverged bitwise"
net.adam_step(1e-2, 0.9, 0.999, 1e-8)
assert param_bytes(net) == ref_post_params, "del parents: post-step params diverged bitwise"
del net, loss
gc.collect()

# (c) The review repro itself: delete the MODEL after constructing the loss
# (together with the forward output, input, and labels); only `loss` remains.
# Without the keep_alive chain this is backward() into freed parameters
# (RuntimeError: std::bad_alloc / Unreachable code); with it, backward must
# simply succeed and the already-read loss bytes must be unaffected.
bt.set_seed(20261114)
net = build_tiny_net()
net.train()
x = bt.Tensor.from_numpy(LIFETIME_X)
labels = bt.Tensor.from_numpy(LIFETIME_LABELS)
logits = net.forward(x)
loss = bt.softmax_cross_entropy(logits, labels)
loss_bytes = loss.to_numpy().tobytes()
del net, logits, x, labels
gc.collect()
loss.backward(F32)
assert loss_bytes == ref_loss_bytes, "del model: loss diverged bitwise"
del loss
gc.collect()

# (d) Tensor.add joins the graph through its requires_grad operand (a module
# parameter), so its output must keep BOTH operands alive.
def add_reference_run():
    bt.set_seed(20261114)
    net = build_tiny_net()
    net.train()
    w = net.named_parameters()["0.weight"]  # (4, 8) for Linear(4, 8)
    c = bt.Tensor.from_numpy(np.ones((4, 8), dtype=np.float32))
    s = w.add(c)
    net.zero_grad()
    s.backward(F32)
    return lifetime_grads_blob(net)


add_ref_grads_blob = add_reference_run()

bt.set_seed(20261114)
net = build_tiny_net()
net.train()
w = net.named_parameters()["0.weight"]
c = bt.Tensor.from_numpy(np.ones((4, 8), dtype=np.float32))
s = w.add(c)
net.zero_grad()
del w, c
gc.collect()
s.backward(F32)
assert lifetime_grads_blob(net) == add_ref_grads_blob, "add: gradients diverged bitwise"
del net, s
gc.collect()

# --- All engine dtypes: full enum, NumPy-native inference, raw-bits I/O ---

# (a) bt.DType exposes every engine dtype (15 values — Tensor/DTypes.h:89-105;
# the task brief said 14, the enum itself carries 15).
ALL_DTYPES = (
    bt.DType.BFloat16, bt.DType.BFloat16x2, bt.DType.Float16, bt.DType.Float16x2,
    bt.DType.Float32, bt.DType.Float64, bt.DType.Int64, bt.DType.Int32,
    bt.DType.Int16, bt.DType.Int8, bt.DType.UInt64, bt.DType.UInt32,
    bt.DType.UInt16, bt.DType.UInt8, bt.DType.Int4x2,
)
assert len(ALL_DTYPES) == 15 and len(set(ALL_DTYPES)) == 15, (
    f"bt.DType must expose 15 distinct values, got {len(set(ALL_DTYPES))}"
)

# (b) from_numpy -> to_numpy roundtrip, byte-compared, for every NumPy-native
# dtype with an engine counterpart. Engine Float16 is IEEE-754 binary16 — same
# sign/exp/mantissa layout as numpy float16 — so byte equality is the
# assertion; float16 values are exactly representable halves (0.5, -2.25).
NATIVE_MAP = {
    np.dtype(np.float16): bt.DType.Float16,
    np.dtype(np.float32): bt.DType.Float32,
    np.dtype(np.float64): bt.DType.Float64,
    np.dtype(np.int8): bt.DType.Int8,
    np.dtype(np.int16): bt.DType.Int16,
    np.dtype(np.int32): bt.DType.Int32,
    np.dtype(np.int64): bt.DType.Int64,
    np.dtype(np.uint8): bt.DType.UInt8,
    np.dtype(np.uint16): bt.DType.UInt16,
    np.dtype(np.uint32): bt.DType.UInt32,
    np.dtype(np.uint64): bt.DType.UInt64,
}
# Sign/exponent-exercising values every listed dtype holds exactly (uint8 tops
# out at 255, so no 300 there).
NATIVE_VALUES = {
    np.dtype(np.float16): [0.5, -2.25, 0.0, 300.0, -1.5],
    np.dtype(np.float32): [0.5, -2.25, 0.0, 300.0, -1.5],
    np.dtype(np.float64): [0.5, -2.25, 0.0, 300.0, -1.5],
    np.dtype(np.int8): [-1, 0, 127],
    np.dtype(np.int16): [-1, 0, 300],
    np.dtype(np.int32): [-1, 0, 300],
    np.dtype(np.int64): [-1, 0, 300],
    np.dtype(np.uint8): [0, 255],
    np.dtype(np.uint16): [0, 300, 65535],
    np.dtype(np.uint32): [0, 300, 4294967295],
    np.dtype(np.uint64): [0, 300, 18446744073709551615],
}
for np_dt, bt_dt in NATIVE_MAP.items():
    src = np.array(NATIVE_VALUES[np_dt], dtype=np_dt)
    t = bt.Tensor.from_numpy(src)
    assert t.dtype == bt_dt, f"dtype mismatch for {np_dt}"
    assert t.shape == src.shape, f"shape mismatch for {np_dt}"
    out = t.to_numpy()
    assert out.dtype == np_dt, f"numpy dtype mismatch for {np_dt}"
    assert out.tobytes() == src.tobytes(), f"bytes not preserved for {np_dt}"

# Raw-bits tie-out for Float16: the bytes path and the numpy path must agree
# on the IEEE half patterns (0.5 = 0x3800, -2.25 = 0xC080).
F16_BITS = bytes([0x00, 0x38, 0x80, 0xC0])
t = bt.Tensor.from_bytes(F16_BITS, (2,), bt.DType.Float16)
assert t.to_numpy().tobytes() == F16_BITS, "float16 raw bits disagree with numpy"
assert bt.Tensor.from_numpy(np.array([0.5, -2.25], dtype=np.float16)).to_bytes() == F16_BITS

# (c) from_bytes/to_bytes roundtrip for the non-NumPy-native dtypes, crafted
# bit patterns: BFloat16 0x3F80 = 1.0; Int4x2 packs two signed 4-bit lanes per
# byte (0x21 -> lo=1, hi=2; 0xFE -> lo=-2, hi=-1); the x2 float packs hold two
# lanes per 4-byte element.
BF16_BITS = bytes([0x80, 0x3F, 0x00, 0x40, 0x80, 0xBF])  # 1.0, 2.0, -1.0
BF16X2_BITS = bytes([0x80, 0x3F, 0x00, 0x40, 0x80, 0xBF, 0x00, 0xC0])  # [1,2], [-1,-2]
F16X2_BITS = bytes([0x00, 0x3C, 0x80, 0xC0, 0x00, 0x38, 0x00, 0xBE])  # [1,-2.25], [0.5,-1.5]
I4X2_BITS = bytes([0x21, 0xFE])  # (1,2), (-2,-1)
for bits, shape, dt in (
    (BF16_BITS, (3,), bt.DType.BFloat16),
    (BF16X2_BITS, (2,), bt.DType.BFloat16x2),
    (F16X2_BITS, (2,), bt.DType.Float16x2),
    (I4X2_BITS, (2,), bt.DType.Int4x2),
):
    t = bt.Tensor.from_bytes(bits, shape, dt)
    assert t.dtype == dt, f"dtype mismatch for {dt}"
    assert t.shape == shape, f"shape mismatch for {dt}"
    assert t.device == CPU, f"device mismatch for {dt}"
    assert t.to_bytes() == bits, f"raw bytes not preserved for {dt}"

# Wrong-size from_bytes raises ValueError (2 bytes given, 3 * 2 required; and
# 3 bytes given, 2 * 1 required).
expect_raises(
    ValueError,
    lambda: bt.Tensor.from_bytes(bytes([0x80, 0x3F]), (3,), bt.DType.BFloat16),
    "from_bytes wrong size",
)
expect_raises(
    ValueError,
    lambda: bt.Tensor.from_bytes(bytes([0x21, 0x00, 0x00]), (2,), bt.DType.Int4x2),
    "from_bytes wrong size (Int4x2)",
)

# (d) to_numpy on a non-NumPy-native dtype raises ValueError pointing at
# to_bytes; the packed dtypes raise the same way.
bf16_t = bt.Tensor.from_bytes(BF16_BITS, (3,), bt.DType.BFloat16)
try:
    bf16_t.to_numpy()
    raise AssertionError("to_numpy on BFloat16 did not raise")
except ValueError as e:
    assert "to_bytes" in str(e), f"error message must mention to_bytes: {e}"
expect_raises(
    ValueError,
    lambda: bt.Tensor.from_bytes(I4X2_BITS, (2,), bt.DType.Int4x2).to_numpy(),
    "to_numpy on Int4x2",
)

# (e) NumPy dtypes with no engine counterpart keep raising ValueError: bool,
# complex, and a strided view of a supported dtype still raises too.
expect_raises(ValueError, lambda: bt.Tensor.from_numpy(np.array([True, False])), "bool array")
expect_raises(
    ValueError,
    lambda: bt.Tensor.from_numpy(np.array([1.0 + 2.0j], dtype=np.complex128)),
    "complex array",
)
expect_raises(
    ValueError,
    lambda: bt.Tensor.from_numpy(np.arange(12, dtype=np.uint16).reshape(3, 4)[:, ::2]),
    "strided uint16 view",
)

# --- Evaluation verbs: argmax(axis), equal, mean, convert_dtype ---

# The chain csv_demo.cpp evaluates with: argmax(1) -> convert_dtype(Int32) ->
# equal(labels) -> convert_dtype(Float32) -> mean. argmax/equal outputs never
# join the autograd graph (Tensor.h op docs); mean attaches a MeanG node only
# when the input requires_grad. Values are compared byte-wise, never float ==.
ev_logits = bt.Tensor.from_numpy(
    np.array([[0.1, 0.9, 0.0], [2.0, 1.0, 0.5]], dtype=np.float32)
)
ev_labels32 = bt.Tensor.from_numpy(np.array([1, 0], dtype=np.int32))
ev_pred = ev_logits.argmax(1)
assert ev_pred.shape == (2,) and ev_pred.dtype == bt.DType.Int64
# convert_dtype is async: the dtype flips when the queued conversion
# finalizes, so sync before asserting or feeding the next op (same syncs
# csv_demo.cpp makes).
ev_pred.convert_dtype(bt.DType.Int32)
ev_pred.sync()
assert ev_pred.dtype == bt.DType.Int32
ev_mask = ev_pred.equal(ev_labels32)
assert ev_mask.shape == (2,) and ev_mask.dtype == bt.DType.UInt8
assert ev_mask.to_numpy().tobytes() == np.array([1, 1], dtype=np.uint8).tobytes()
ev_mask.convert_dtype(F32)
ev_mask.sync()
ev_acc = ev_mask.mean()
assert ev_acc.shape == (1,) and ev_acc.dtype == F32
assert ev_acc.to_numpy().tobytes() == np.array([1.0], dtype=np.float32).tobytes()
# equal enforces the shared-dtype contract: an Int64 left operand against the
# Int32 labels raises instead of comparing across widths.
expect_raises(
    Exception,
    lambda: ev_logits.argmax(1).equal(ev_labels32),
    "dtype-mismatched equal",
)

# --- Training dashboard: inert contract and the active last_frame seam ---

# The bound dashboard sinks to the C++ std::cerr stream (fd 2): frames and the
# stop() summary go to the process's real stderr, so Python-level sys.stderr
# redirection does not intercept them. Activity is gated on the ctor's tty
# flag, not on the stream (Helpers/Dashboard/Dashboard.h), so under ctest —
# where stderr is always a captured pipe, never a TTY — a tty=True dashboard
# is live and renders into the log. The active checks are therefore guarded
# to run only when stderr is NOT a TTY, so a manual run on a real terminal
# never starts the framed region-redraw render thread.
#
# (a) Inert: tty=False constructs the fully inert dashboard — every verb is
# callable, the sink is never touched, and the last-frame seam stays empty.
dash_inert = bt.Dashboard(tty=False)
dash_inert.report_batch(0.5)
dash_inert.report_epoch(1, 3, 0.3, 0.9, 0.4)
dash_inert.render_once()
assert dash_inert.last_frame() == "", (
    f"inert dashboard must never compose a frame, got {dash_inert.last_frame()!r}"
)
dash_inert.stop()  # idempotent no-op on an inert dashboard
dash_inert.stop()

if not sys.stderr.isatty():
    # (b) Active on a non-TTY sink: feed two losses plus one epoch summary,
    # render_once() composes on demand, and last_frame() must hold the exact
    # composed bytes — the compact readout of the fed losses (latest 0.1,
    # ring mean 0.3) with the sparkline, the epoch progress "1/3", and a pool
    # worker status tag from the grid (IDLE, or OWN while a pool task runs).
    dash = bt.Dashboard(tty=True, refresh_hz=10.0)
    dash.report_batch(0.5)
    dash.report_batch(0.1)
    dash.report_epoch(1, 3, 0.3, 0.9, 0.4)
    dash.render_once()
    frame = dash.last_frame()
    assert "last=0.1" in frame and "mean=0.3" in frame and "spark: " in frame, (
        f"frame must carry the fed-loss readout and sparkline: {frame!r}"
    )
    assert "1/3" in frame, f"frame must carry the epoch progress: {frame!r}"
    assert "IDLE" in frame or "OWN" in frame, (
        f"frame must carry a pool worker status tag: {frame!r}"
    )
    dash.stop()
    dash.stop()  # stop() is idempotent on an active dashboard too
    print("DASHBOARD SMOKE OK")
else:
    print("SKIP: active-dashboard checks need a non-TTY stderr (run under ctest)")

# --- Task 1 (btui): pool snapshots, in-place numpy writes, sgd_step ---

# (a) pool_snapshot: plain POD dicts of the global CPU pool's live telemetry
# (Helpers/CPUThreading.h snapshot()) — no tensor handles, no keep_alive.
# The top-level key set pins the dict schema; each worker dict is keyed
# exactly by the WorkerSnapshot member names; status/task_priority arrive
# stringified; pending_tasks_total is the sum the snapshot was built from, so
# it must equal the sum of the per-worker local depths exactly.
snap1 = bt.pool_snapshot()
assert isinstance(snap1, dict), f"pool_snapshot returned {type(snap1)}"
assert set(snap1.keys()) == {"worker_count", "pending_tasks_total", "stopping", "workers"}, (
    f"pool_snapshot keys: {sorted(snap1.keys())}"
)
assert isinstance(snap1["worker_count"], int) and snap1["worker_count"] > 0
assert isinstance(snap1["pending_tasks_total"], int) and snap1["pending_tasks_total"] >= 0
assert snap1["stopping"] is False, f"pool must not be stopping mid-run: {snap1['stopping']}"
assert isinstance(snap1["workers"], list)
assert len(snap1["workers"]) == snap1["worker_count"], (
    f"worker list length {len(snap1['workers'])} != worker_count {snap1['worker_count']}"
)
WORKER_SNAPSHOT_KEYS = {
    "worker_id", "status", "pinned_group", "task_sequence", "task_priority",
    "task_running_ms", "futures_awaited", "tasks_completed", "tasks_stolen",
    "tasks_helped", "local_queue_depth",
}
STATUS_STRINGS = {"Idle", "RunningOwn", "RunningStolen", "RunningHelped", "HelpWaiting"}
PRIORITY_STRINGS = {"Low", "Normal", "High"}
depth_sum = 0
for w in snap1["workers"]:
    assert set(w.keys()) == WORKER_SNAPSHOT_KEYS, f"worker keys: {sorted(w.keys())}"
    assert w["status"] in STATUS_STRINGS, f"unexpected status string: {w['status']!r}"
    # task_priority is valid only when status != Idle (WorkerSnapshot docs);
    # an idle worker's raw slot may stringify to "?".
    if w["status"] != "Idle":
        assert w["task_priority"] in PRIORITY_STRINGS, (
            f"unexpected priority string: {w['task_priority']!r}"
        )
    for counter in (
        "worker_id", "pinned_group", "task_sequence", "task_running_ms",
        "futures_awaited", "tasks_completed", "tasks_stolen", "tasks_helped",
        "local_queue_depth",
    ):
        assert isinstance(w[counter], int) and w[counter] >= 0, (
            f"{counter} not a sane int: {w[counter]!r}"
        )
    assert w["worker_id"] < snap1["worker_count"], (
        f"worker_id out of range: {w['worker_id']}"
    )
    depth_sum += w["local_queue_depth"]
assert depth_sum == snap1["pending_tasks_total"], (
    f"pending_tasks_total {snap1['pending_tasks_total']} != sum of local depths {depth_sum}"
)

# The counters are monotonic lifetime totals of the process-wide pool, so two
# back-to-back snapshots must not show a decrease on any shared worker.
snap2 = bt.pool_snapshot()
assert snap2["worker_count"] == snap1["worker_count"]
for earlier, later in zip(snap1["workers"], snap2["workers"]):
    for counter in ("futures_awaited", "tasks_completed", "tasks_stolen", "tasks_helped"):
        assert later[counter] >= earlier[counter], (
            f"{counter} decreased for worker {earlier['worker_id']}: "
            f"{earlier[counter]} -> {later[counter]}"
        )

# (b) write_numpy: in-place payload overwrite, the exact inverse direction of
# to_numpy's bulk read. Roundtrips are byte-compared per NumPy-native dtype;
# every rejected call raises ValueError BEFORE anything is written, so the
# payload stays byte-identical across all of them.
for np_dt in (np.float16, np.float32, np.float64, np.int32, np.int64, np.uint8):
    src = (np.arange(12, dtype=np.float64) - 5.5).reshape(3, 4).astype(np_dt)
    t = bt.Tensor.from_numpy(np.zeros((3, 4), dtype=np_dt))
    t.write_numpy(src)
    assert t.to_numpy().tobytes() == src.tobytes(), (
        f"write_numpy roundtrip bytes diverged for {np_dt}"
    )

wt = bt.Tensor.from_numpy(np.zeros((3, 4), dtype=np.float32))
payload_before = wt.to_numpy().tobytes()
expect_raises(ValueError, lambda: wt.write_numpy(np.zeros((3, 4), dtype=np.float64)), "write_numpy dtype mismatch")
expect_raises(ValueError, lambda: wt.write_numpy(np.zeros((12,), dtype=np.float32)), "write_numpy shape mismatch")
expect_raises(ValueError, lambda: wt.write_numpy(np.zeros((4, 3), dtype=np.float32).T), "write_numpy non-contiguous")
expect_raises(ValueError, lambda: wt.write_numpy(np.zeros((3, 4), dtype=np.bool_)), "write_numpy unsupported dtype")
assert wt.to_numpy().tobytes() == payload_before, (
    "a rejected write_numpy must leave the payload byte-identical"
)
# A well-formed write still lands after the rejected ones.
fix_up = np.arange(12, dtype=np.float32).reshape(3, 4)
wt.write_numpy(fix_up)
assert wt.to_numpy().tobytes() == fix_up.tobytes()

# The GPU guard mirrors to_numpy's: a GPU-resident tensor raises RuntimeError
# before any host write. On a CPU-only build no GPU tensor can exist, so the
# guard's presence is pinned by the dtype-mismatch case above (same check
# body); on GPU builds a staged tensor exercises the branch itself.
try:
    gpu_stage = bt.Tensor.from_numpy(np.zeros((2, 3), dtype=np.float32))
    gpu_stage.to(bt.Device.GPU)
except Exception:
    gpu_stage = None
if gpu_stage is not None:
    expect_raises(RuntimeError, lambda: gpu_stage.write_numpy(np.zeros((2, 3), dtype=np.float32)), "write_numpy on a GPU tensor")
    print("WRITE_NUMPY GPU GUARD OK")
else:
    print("SKIP: write_numpy GPU guard needs a GPU-backend build with a live device")

# (c) sgd_step: one step moves every parameter holding an accumulated
# gradient (byte inequality, never float ==), and a step with no gradients
# accumulated is a no-op. lr is passed by keyword to pin the py::arg.
bt.set_seed(20261119)
sgd_net = bt.Sequential([
    bt.Linear(4, 8, F32, CPU),
    bt.Relu(),
    bt.Linear(8, 4, F32, CPU),
])
sgd_net.train()
sgd_x = bt.Tensor.from_numpy(np.array([
    [1.0, 0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0, 0.5],
    [0.0, 0.0, 1.0, 0.25],
    [1.0, 1.0, 0.0, -0.5],
    [0.0, 1.0, 1.0, 0.75],
    [1.0, 0.0, 1.0, -0.25],
], dtype=np.float32))
sgd_labels = bt.Tensor.from_numpy(np.array([0, 1, 2, 3, 1, 2], dtype=np.int64))
sgd_logits = sgd_net.forward(sgd_x)
sgd_loss = bt.softmax_cross_entropy(sgd_logits, sgd_labels)
sgd_net.zero_grad()
sgd_loss.backward(F32)
sgd_before = {name: t.to_numpy().tobytes() for name, t in sgd_net.named_parameters().items()}
sgd_net.sgd_step(lr=0.1)
sgd_after = {name: t.to_numpy().tobytes() for name, t in sgd_net.named_parameters().items()}
assert sgd_before.keys() == sgd_after.keys()
for name in sgd_before:
    assert sgd_before[name] != sgd_after[name], f"sgd_step left {name} byte-identical"

# No accumulated gradient (zeroed, no backward since) -> the step applies
# nothing, whichever of the engine's zero_grad semantics is in effect (a
# cleared grad is skipped; an all-zero grad updates by exactly zero).
sgd_net.zero_grad()
sgd_frozen = {name: t.to_numpy().tobytes() for name, t in sgd_net.named_parameters().items()}
sgd_net.sgd_step(0.1)
for name, p in sgd_net.named_parameters().items():
    assert p.to_numpy().tobytes() == sgd_frozen[name], (
        f"sgd_step with no accumulated grads moved {name}"
    )

# --- Task 7 (btui): architecture spec export ---

# export_spec mirrors Module::spec() (Module/ModuleSpec.h) as a plain POD
# dict tree: {type, hyperparameters: {k: v}, children: [{name, ...}]} with
# Sequential children in registry order. Hyperparameter values stay strings
# exactly as the spec stores them — decimal for counts, "0"/"1" for bools,
# hexfloat for floats (format_hyper_f64).
spec_net = bt.Sequential([
    bt.Linear(4, 8, F32, CPU),
    bt.Relu(),
    bt.Dropout(0.25),
    bt.Linear(8, 3, F32, CPU),
])
exported = bt.export_spec(spec_net)
assert isinstance(exported, dict), f"export_spec returned {type(exported)}"
assert exported["type"] == "Sequential", f"root type: {exported['type']}"
assert exported["hyperparameters"] == {}
spec_children = exported["children"]
assert [c["name"] for c in spec_children] == ["0", "1", "2", "3"], (
    f"children names/order: {[c['name'] for c in spec_children]}"
)
assert [c["type"] for c in spec_children] == ["Linear", "Relu", "Dropout", "Linear"]
lin0_hyper = spec_children[0]["hyperparameters"]
assert lin0_hyper == {"in_features": "4", "out_features": "8", "bias": "1"}, (
    f"linear hypers: {lin0_hyper}"
)
assert spec_children[1]["hyperparameters"] == {}
assert spec_children[1]["children"] == []
# Dropout's p is a hexfloat; float.fromhex parses it back bit-exactly.
assert float.fromhex(spec_children[2]["hyperparameters"]["p"]) == 0.25
assert spec_children[3]["hyperparameters"]["in_features"] == "8"
assert spec_children[3]["hyperparameters"]["out_features"] == "3"
print("EXPORT SPEC SMOKE OK")

# --- Task 8 (btui): gpu memory telemetry + adam step counters ---

# gpu_memory is telemetry: on a CPU-only build (or with no live GPU) it must
# return None, never raise. When a GPU is reachable it returns sane bytes.
try:
    gm_probe = bt.Tensor.from_numpy(np.zeros(1, dtype=np.float32))
    gm_probe.to(bt.Device.GPU)
    gm_has_gpu = True
except Exception:
    gm_has_gpu = False
gm = bt.gpu_memory()
if gm_has_gpu:
    assert isinstance(gm, dict), f"gpu_memory returned {type(gm)}"
    assert set(gm.keys()) == {"total_bytes", "free_bytes", "used_bytes"}
    assert gm["total_bytes"] > 0
    assert 0 <= gm["used_bytes"] <= gm["total_bytes"]
    assert gm["used_bytes"] + gm["free_bytes"] == gm["total_bytes"]
    print("GPU MEMORY SMOKE OK")
else:
    assert gm is None, f"gpu_memory on a CPU-only build must be None, got {gm}"

# adam_step_count: None before any Adam step on the param, then counts.
asc_net = bt.Sequential([bt.Linear(4, 4, F32, CPU)])
asc_w = asc_net.named_parameters()["0.weight"]
assert asc_w.adam_step_count() is None, "fresh param must have no Adam state"
asc_net.train()
asc_x = bt.Tensor.from_numpy(np.ones((3, 4), dtype=np.float32))
asc_y = bt.Tensor.from_numpy(np.array([0, 1, 2], dtype=np.int64))
for expected in (1, 2, 3):
    asc_net.zero_grad()
    asc_loss = bt.softmax_cross_entropy(asc_net.forward(asc_x), asc_y)
    asc_loss.backward(F32)
    asc_net.adam_step(1e-2, 0.9, 0.999, 1e-8)
    assert asc_net.named_parameters()["0.weight"].adam_step_count() == expected, (
        f"adam_step_count after {expected} steps"
    )
print("ADAM STEP COUNT SMOKE OK")

# --- Task 14 (btui): optional training-config metadata in .bt ---

META_PATH = os.path.join(SCRATCH_BINDINGS, "task14_meta.bt")
META_PLAIN_PATH = os.path.join(SCRATCH_BINDINGS, "task14_plain.bt")
meta_net = bt.Sequential([bt.Linear(4, 4, F32, CPU)])
bt.save(meta_net, META_PATH, True, bt.CheckpointLevel.Weights,
        metadata="optimizer=adam\nlr=0.001")
assert bt.read_metadata(META_PATH) == "optimizer=adam\nlr=0.001", (
    f"metadata roundtrip: {bt.read_metadata(META_PATH)!r}"
)
# Params survive the metadata-bearing file.
assert bt.load(META_PATH).named_parameters()["0.weight"].to_numpy().tobytes() == (
    meta_net.named_parameters()["0.weight"].to_numpy().tobytes()
)
# A code-written checkpoint carries no metadata; read_metadata returns "".
bt.save(meta_net, META_PLAIN_PATH, True, bt.CheckpointLevel.Full)
assert bt.read_metadata(META_PLAIN_PATH) == ""
print("METADATA SMOKE OK")

print("BTUI BINDING SMOKE OK")

print("BINDINGS SMOKE OK")
