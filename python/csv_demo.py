#!/usr/bin/env python3
"""csv_demo.py — BeanTensor Python-bindings CSV demo.

Replays csv_demo.cpp (the C++ BeanTensorV5CSVDemo demo) act-for-act through
the pybind11 surface: the inspection act (null fill + standardize, then the
null count, per-column min/max/mean, and label mapping the C++ demo prints)
and the training act (the exact 6-row separable CSV, a Linear(2,2) softmax
classifier, 60 steps of softmax_ce + backward(Float32) + adam_step), ending
in the same loss/accuracy line and the same loss-did-not-decrease exit
contract. The fixtures are created under ./scratch (the same paths the C++
demo writes), so the demo runs green from any cwd.

Run from the repository root:
    PYTHONPATH=scratch/build-py2/Bindings python3 python/csv_demo.py
"""

import sys
from pathlib import Path

import beantensor as bt

SCRATCH = Path("scratch")
INSPECTION_CSV = SCRATCH / "csv_demo_inspection.csv"
TRAIN_CSV = SCRATCH / "csv_demo_train.csv"

K_BATCH = 2
K_LOADER_SEED = 5
K_STEPS = 60
K_LEARNING_RATE = 1e-2
K_BETA1 = 0.9
K_BETA2 = 0.999
K_EPSILON = 1e-8


def write_csv(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def fmt(value: float) -> str:
    # C++ default ostream formatting: 6 significant digits, general format.
    return f"{value:.6g}"


# Act 1: inspection — stats, nulls, scaling, and the label mapping.
def inspection_act() -> None:
    lines = ["length,width,noise,fruit"]
    for i in range(40):
        is_apple = (i % 2) == 0
        length = 6.0 + 0.1 * (i % 7) if is_apple else 8.0 + 0.1 * (i % 5)
        width = 7.0 + 0.1 * (i % 5) if is_apple else 6.0 + 0.1 * (i % 7)
        noise_cell = "" if (i % 9 == 0) else f"{0.01 * i:.6f}"
        label = "apple" if is_apple else "pear"
        # std::to_string in the C++ demo writes %f (six decimals) per cell.
        lines.append(f"{length:.6f},{width:.6f},{noise_cell},{label}")
    write_csv(INSPECTION_CSV, "\n".join(lines) + "\n")

    opts = bt.CsvOptions()
    # csv_demo.cpp leaves feature_dtype at the Float64 default in this act;
    # the printed stats are NaN-skipped float64 values computed before the
    # dtype conversion (Data/Csv.cpp), so the numbers below are identical
    # either way.
    opts.nulls = bt.CsvNullPolicy.FillMean
    opts.normalize = bt.CsvNormalize.Standardize
    ds = bt.load_csv(str(INSPECTION_CSV), opts)

    print(
        f"== inspection: {ds.features.shape[0]} rows x {ds.features.shape[1]} "
        f"features from {INSPECTION_CSV}"
    )
    print(f"null cells filled with column means: {ds.null_count}")
    for name, col_min, col_max, col_mean in zip(
        ds.feature_names, ds.column_min, ds.column_max, ds.column_mean
    ):
        print(f"  {name}  min {fmt(col_min)}  max {fmt(col_max)}  mean {fmt(col_mean)}")
    label_line = " ".join(f"[{i}]={name}" for i, name in enumerate(ds.label_names))
    print(f"labels: {label_line}")
    print()


# Act 2: training — the proven tiny recipe end to end.
def training_act() -> bool:
    write_csv(
        TRAIN_CSV,
        "x1,x2,cls\n"
        "0.0,0.0,a\n"
        "0.1,0.0,a\n"
        "0.0,0.1,a\n"
        "1.0,1.0,b\n"
        "1.0,0.9,b\n"
        "0.9,1.0,b\n",
    )

    opts = bt.CsvOptions()
    opts.feature_dtype = bt.DType.Float32
    ds = bt.load_csv(str(TRAIN_CSV), opts)
    label_names = " ".join(ds.label_names)
    print(f"== training: {ds.features.shape[0]} rows from {TRAIN_CSV} (labels {label_names})")

    loader = bt.DataLoader(ds.features, ds.labels, K_BATCH, True, K_LOADER_SEED)
    net = bt.Sequential([bt.Linear(2, 2, bt.DType.Float32, bt.Device.CPU)])

    first_loss = 0.0
    last_loss = 0.0
    for step in range(K_STEPS):
        batch = loader.next_batch()
        if batch is None:
            loader.reset()
            batch = loader.next_batch()
        features, labels = batch
        net.zero_grad()
        logits = net.forward(features)
        loss = logits.softmax_ce(labels)
        loss.backward(bt.DType.Float32)
        net.adam_step(K_LEARNING_RATE, K_BETA1, K_BETA2, K_EPSILON)
        value = loss.item()
        if step == 0:
            first_loss = value
        last_loss = value

    net.eval()
    # Engine-side evaluation, the exact chain csv_demo.cpp runs:
    # argmax(1) -> Int32 -> equal(labels) -> UInt8 mask -> Float32 -> mean.
    # convert_dtype is async (the dtype flips when the queued conversion
    # finalizes), so each conversion is synced before the next consumer —
    # the same syncs the C++ demo makes.
    eval_logits = net.forward(ds.features)
    predicted = eval_logits.argmax(1)
    predicted.convert_dtype(bt.DType.Int32)
    predicted.sync()
    mask = predicted.equal(ds.labels)
    mask.convert_dtype(bt.DType.Float32)
    mask.sync()
    accuracy = mask.mean().item()

    print(f"loss {fmt(first_loss)} -> {fmt(last_loss)}  accuracy {fmt(accuracy * 100)}%")
    return last_loss < first_loss


def main() -> None:
    bt.set_seed(7)
    try:
        SCRATCH.mkdir(exist_ok=True)
    except OSError:
        pass

    inspection_act()
    if not training_act():
        print("csv demo: loss did not decrease", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
