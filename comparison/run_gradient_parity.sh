#!/usr/bin/env bash
# comparison/run_gradient_parity.sh — run the C++/PyTorch gradient parity harness.
# Skips gracefully when the comparison venv is missing, so ctest stays runnable
# on machines without PyTorch. GPU runs go through gpu-test.sh when a GPU build
# directory is supplied.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DEVICE="cpu"
BUILD_DIR=""
NEGATIVE=""
SMOKE=""
SKIP_GPU_LOCK=""

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
        --negative-gradient)
            NEGATIVE="--perturb $2"
            shift 2
            ;;
        --corrupt-manifest)
            NEGATIVE="--corrupt-manifest"
            shift
            ;;
        --smoke)
            SMOKE="--smoke"
            shift
            ;;
        --skip-gpu-lock)
            SKIP_GPU_LOCK=1
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
    # Worktrees share the venv from the main checkout.
    MAIN_VENV="/srv/dev/BeanTensor/comparison/.venv/bin/python"
    if [[ -x "$MAIN_VENV" ]]; then
        VENV="$MAIN_VENV"
    else
        echo "SKIP: no comparison venv found"
        exit 0
    fi
fi

if [[ -z "$BUILD_DIR" ]]; then
    echo "usage: $0 --build-dir <dir> [--device cpu|gpu] [--smoke] [--negative-gradient <grad>|--corrupt-manifest]" >&2
    exit 1
fi

BINARY="$BUILD_DIR/BeanTensorV5GradientParity"
if [[ ! -x "$BINARY" ]]; then
    echo "binary not found: $BINARY" >&2
    exit 1
fi

DATA_DIR="$REPO_ROOT/comparison/generated-data"
# The C++ driver truncates/overwrites files, so a full wipe is unnecessary.
mkdir -p "$DATA_DIR"

if [[ "$DEVICE" == "gpu" && -z "$SKIP_GPU_LOCK" ]]; then
    # The C++ driver touches Device::GPU; gpu-test.sh serializes GPU traffic
    # by running the matching ctest target. The ctest target re-enters this
    # script with --skip-gpu-lock so the lock is not acquired twice.
    "/srv/dev/BeanTensor/gpu-test.sh" gradient-parity "$BUILD_DIR" -R gradient_parity_gpu
    exit 0
fi

"$BINARY" --out "$DATA_DIR" --device "$DEVICE" $SMOKE

# The reference comparison always runs on PyTorch CPU; the comparison venv is
# CPU-only and the raw bytes are portable across devices.
PYTHONPATH="$REPO_ROOT" "$VENV" "$REPO_ROOT/comparison/gradient_parity.py" --data-dir "$DATA_DIR" --device cpu $NEGATIVE
