#!/usr/bin/env bash
# benchmark/run.sh — run the op-level benchmark harness and record results.
# CPU set runs unconditionally; a GPU set runs through the gpu-test.sh gate
# when a GPU build directory exists; the PyTorch comparison column runs when
# the comparison venv exists. Results append to a dated log under
# benchmark/results/ for trend comparison; docs/benchmarks.md holds the
# published table with repro commands. Memory gate per project rule.
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

FREE_KB="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
if [[ "$FREE_KB" -gt 33554432 ]]; then
    JOBS="$(nproc)"
elif [[ "$FREE_KB" -gt 5242880 ]]; then
    JOBS=6
else
    echo "run.sh: only ${FREE_KB}KB available; refusing to build under 5GB." >&2
    exit 1
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
mkdir -p benchmark/results
LOG="benchmark/results/${STAMP}.log"
{
    echo "# benchmark run ${STAMP}"
    echo "# cpu: $(grep -m1 'model name' /proc/cpuinfo || true)"
} >> "$LOG"

echo "== CPU build =="
cmake -S . -B scratch/build-bench-cpu -DUSE_CUDA=OFF -DUSE_HIP=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build scratch/build-bench-cpu --target BeanTensorV5Benchmarks -j "$JOBS"

echo "== CPU benchmarks =="
./scratch/build-bench-cpu/BeanTensorV5Benchmarks | tee -a "$LOG"

echo "== PyTorch comparison column =="
VENV="comparison/.venv/bin/python"
if [[ ! -x "$VENV" && -x /srv/dev/BeanTensor/comparison/.venv/bin/python ]]; then
    VENV="/srv/dev/BeanTensor/comparison/.venv/bin/python"
fi
if [[ -x "$VENV" ]]; then
    ./scratch/build-bench-cpu/BeanTensorV5Benchmarks --emit-shapes > benchmark/shapes.txt
    {
        echo "# pytorch column"
        "$VENV" comparison/bench_pytorch.py benchmark/shapes.txt
    } | tee -a "$LOG"
else
    echo "no comparison venv found; skipping pytorch column"
fi

# GPU op-level benchmarking is deliberately out of scope for this harness:
# timing queued async GPU work needs per-iteration device sync semantics the
# harness does not have, and GPU test traffic must go through gpu-test.sh
# (recorded in docs/roadmap.md). End-to-end GPU numbers live in docs/mnist-results.md and
# docs/cnn-results.md instead. Recorded in docs/roadmap.md.
echo "gpu op-level benchmarks: out of scope for this harness (see docs/roadmap.md)"

echo "results appended to $LOG"
