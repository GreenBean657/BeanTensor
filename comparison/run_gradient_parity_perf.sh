#!/usr/bin/env bash
# comparison/run_gradient_parity_perf.sh — forward+backward performance comparison.
# Emits the shape plan from the existing forward benchmark, times each
# differentiable family in both BeanTensor and PyTorch, and writes a Markdown
# table to docs/gradient-parity.md.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DEVICE="cpu"
BUILD_DIR=""
SMOKE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --smoke)
            SMOKE="--smoke"
            shift
            ;;
        *)
            echo "unknown arg: $1" >&2
            exit 1
            ;;
    esac
done

VENV="$REPO_ROOT/comparison/.venv/bin/python"
if [[ ! -x "$VENV" ]]; then
    MAIN_VENV="/srv/dev/BeanTensor/comparison/.venv/bin/python"
    if [[ -x "$MAIN_VENV" ]]; then
        VENV="$MAIN_VENV"
    else
        echo "SKIP: no comparison venv found"
        exit 0
    fi
fi

if [[ -z "$BUILD_DIR" ]]; then
    echo "usage: $0 --build-dir <dir> [--device cpu|gpu] [--smoke]" >&2
    exit 1
fi

BENCHMARK_BINARY="$BUILD_DIR/BeanTensorV5Benchmarks"
PERF_BINARY="$BUILD_DIR/BeanTensorV5GradientParityPerf"
if [[ ! -x "$BENCHMARK_BINARY" ]]; then
    echo "benchmark binary not found: $BENCHMARK_BINARY" >&2
    exit 1
fi
if [[ ! -x "$PERF_BINARY" ]]; then
    echo "perf binary not found: $PERF_BINARY" >&2
    exit 1
fi

mkdir -p "$REPO_ROOT/comparison/perf-results"
SHAPES_FILE="$REPO_ROOT/benchmark/shapes.txt"
"$BENCHMARK_BINARY" --emit-shapes > "$SHAPES_FILE"

BT_CSV="$REPO_ROOT/comparison/perf-results/bt.csv"
PT_CSV="$REPO_ROOT/comparison/perf-results/pt.csv"

if [[ "$DEVICE" == "gpu" ]]; then
    "/srv/dev/BeanTensor/gpu-test.sh" gradient-parity-perf "$BUILD_DIR" -- \
        "$PERF_BINARY" --shapes "$SHAPES_FILE" --device gpu $SMOKE > "$BT_CSV"
else
    "$PERF_BINARY" --shapes "$SHAPES_FILE" --device cpu $SMOKE > "$BT_CSV"
fi

PYTHONPATH="$REPO_ROOT" "$VENV" "$REPO_ROOT/comparison/gradient_parity_perf.py" \
    --shapes "$SHAPES_FILE" --device "$DEVICE" $SMOKE > "$PT_CSV"

DOCS_FILE="$REPO_ROOT/docs/gradient-parity.md"
TABLE_MARKER_START="<!-- perf-results -->"
TABLE_MARKER_END="<!-- /perf-results -->"

python3 - "$BT_CSV" "$PT_CSV" "$DOCS_FILE" "$TABLE_MARKER_START" "$TABLE_MARKER_END" <<'PY'
import csv
import sys
from datetime import datetime, timezone

bt_path, pt_path, docs_path, marker_start, marker_end = sys.argv[1:6]

with open(bt_path, encoding="utf-8") as handle:
    bt_rows = list(csv.reader(handle))
with open(pt_path, encoding="utf-8") as handle:
    pt_rows = list(csv.reader(handle))

if len(bt_rows) != len(pt_rows):
    print("mismatched row counts between BeanTensor and PyTorch CSVs", file=sys.stderr)
    sys.exit(1)

table_lines = [
    "| family | op | dims | BeanTensor ms | PyTorch ms | ratio |",
    "|---|---|---:|---:|---:|---:|",
]
for bt_row, pt_row in zip(bt_rows, pt_rows):
    if len(bt_row) < 4 or len(pt_row) < 4:
        continue
    family, op, dims, bt_ms = bt_row[0], bt_row[1], bt_row[2], bt_row[3]
    pt_ms = pt_row[3]
    try:
        ratio = float(bt_ms) / float(pt_ms)
    except (ValueError, ZeroDivisionError):
        ratio = float("nan")
    table_lines.append(
        f"| {family} | {op} | {dims} | {bt_ms} | {pt_ms} | {ratio:.2f} |"
    )

timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
new_section = "\n".join([
    marker_start,
    f"_Generated {timestamp}_",
    "",
] + table_lines + [
    "",
    marker_end,
])

with open(docs_path, encoding="utf-8") as handle:
    original = handle.read()

if marker_start in original and marker_end in original:
    before = original.split(marker_start, 1)[0]
    after = original.split(marker_end, 1)[1]
    updated = before + new_section + after
else:
    updated = original + "\n" + new_section + "\n"

with open(docs_path, "w", encoding="utf-8") as handle:
    handle.write(updated)
PY

echo "perf table written to $DOCS_FILE"
