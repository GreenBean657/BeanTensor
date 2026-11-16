#!/usr/bin/env bash
# tests/mnist_parity.sh — hermetic ctest for the Python-bindings MNIST parity
# contract: python/digits.py reproduces the C++ BeanTensorV5 demo's epoch
# lines bit-for-bit (accuracy and mean-loss fields; wall time is stripped
# before the diff). Skips gracefully with exit 0 when the MNIST IDX files are
# absent, so ctest stays runnable on machines without the dataset.
#
# Usage: mnist_parity.sh <path-to-BeanTensorV5> <beantensor-module-dir>
# Both paths come from CMake as generator expressions; PYTHONPATH for
# digits.py is set to the module dir.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <path-to-BeanTensorV5> <beantensor-module-dir>" >&2
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

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

"$CPP_BIN" > "$WORK/cpp.out"
PYTHONPATH="$MODULE_DIR" python3 python/digits.py --save "$WORK/digits_mlp.bt" > "$WORK/py.out"

strip_epoch_lines() {
    grep '^epoch ' "$1" | sed -E 's/ {2}time .*//'
}

if diff <(strip_epoch_lines "$WORK/cpp.out") <(strip_epoch_lines "$WORK/py.out"); then
    echo "mnist_parity PASS: epoch lines identical (wall time excluded)"
else
    echo "mnist_parity FAIL: epoch lines differ" >&2
    exit 1
fi
