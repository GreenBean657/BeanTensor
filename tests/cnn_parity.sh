#!/usr/bin/env bash
# tests/cnn_parity.sh — hermetic ctest for the Python-bindings CNN parity
# contract: python/cnn.py reproduces the C++ BeanTensorV5CNN demo's epoch
# lines bit-for-bit (accuracy and mean-loss fields; wall time is stripped
# before the diff). Skips gracefully with exit 0 when the MNIST IDX files are
# absent, so ctest stays runnable on machines without the dataset.
#
# Usage: cnn_parity.sh <path-to-BeanTensorV5CNN> <beantensor-module-dir>
# Both paths come from CMake as generator expressions; PYTHONPATH for
# cnn.py is set to the module dir.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <path-to-BeanTensorV5CNN> <beantensor-module-dir>" >&2
    exit 2
fi
CPP_BIN="$1"
MODULE_DIR="$2"

DATA_DIR="$REPO_ROOT/datasets/mnist"
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
    if [[ ! -f "$DATA_DIR/$f" ]]; then
        echo "SKIP: $DATA_DIR/$f not found"
        exit 0
    fi
done

# GPU-backend builds make the C++ demo run training twice (CPU pass, then GPU
# pass), printing two sets of identically-shaped epoch lines that grep cannot
# attribute to a side — the diff would see 4 C++ lines against 2 Python lines
# and could never pass. CMake sets BT_GPU_BACKEND=1 on this test exactly when
# USE_CUDA or USE_HIP was ON at configure time, so the skip is hermetic: no
# hardware probing, and CPU-only builds never see the variable.
if [[ "${BT_GPU_BACKEND:-0}" == "1" ]]; then
    echo "SKIP: GPU-backend build; the C++ demo emits CPU and GPU epoch lines that this diff cannot separate"
    exit 0
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

"$CPP_BIN" > "$WORK/cpp.out"
PYTHONPATH="$MODULE_DIR" python3 python/cnn.py --save "$WORK/cnn_mnist.bt" > "$WORK/py.out"

strip_epoch_lines() {
    grep '^epoch ' "$1" | sed -E 's/ {2}time .*//'
}

# The contract is exact: each side runs 2 epochs, so each output must carry
# exactly 2 epoch lines — an empty or truncated stream on either side is a
# failure, never a silent pass through an empty diff.
EXPECTED_EPOCH_LINES=2
cpp_epochs="$(grep -c '^epoch ' "$WORK/cpp.out" || true)"
py_epochs="$(grep -c '^epoch ' "$WORK/py.out" || true)"
if [[ "$cpp_epochs" -ne "$EXPECTED_EPOCH_LINES" || "$py_epochs" -ne "$EXPECTED_EPOCH_LINES" ]]; then
    echo "cnn_parity FAIL: expected $EXPECTED_EPOCH_LINES epoch lines per side, got cpp=$cpp_epochs py=$py_epochs" >&2
    exit 1
fi

if diff <(strip_epoch_lines "$WORK/cpp.out") <(strip_epoch_lines "$WORK/py.out"); then
    echo "cnn_parity PASS: epoch lines identical (wall time excluded)"
else
    echo "cnn_parity FAIL: epoch lines differ" >&2
    exit 1
fi
