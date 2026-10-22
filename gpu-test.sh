#!/usr/bin/env bash
# gpu-test.sh — MANDATORY wrapper for every GPU (HIP/CUDA) test run.
# This box is a single Strix Halo APU: the same GPU that runs ROCm/CUDA
# compute also drives the desktop display. Unserialized GPU test runs
# (especially concurrent with full builds) have wedged the amdgpu driver and
# killed the desktop session (2026-10-22, kwin pageflip timeouts). flock on
# scratch/.gpu-test.lock serializes all GPU test traffic; the lock file's
# contents are an audit trail of the current/last holder.
# Contract: gpu-test.sh <label> <build-dir> [extra ctest args...]
#   <label>    identifies who is running (agent/task name), recorded in the lock.
#   Exit non-zero = failure (lock timeout, bad build dir, or test failure).
#   GPU_TEST_WAIT overrides the lock wait in seconds (default 600).

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK="${GPU_TEST_LOCK:-$ROOT/scratch/.gpu-test.lock}"
HISTORY="${LOCK%.lock}.history"
WAIT="${GPU_TEST_WAIT:-600}"

LABEL="${1:-}"
BUILD="${2:-}"
if [[ -z "$LABEL" || -z "$BUILD" ]]; then
    echo "gpu-test_sh ERROR: usage: gpu-test.sh <label> <build-dir> [ctest args...]" >&2
    exit 2
fi
shift 2
if [[ ! -f "$BUILD/CMakeCache.txt" ]]; then
    echo "gpu-test_sh ERROR: '$BUILD' is not a configured build dir." >&2
    exit 2
fi
if ! grep -qE '^USE_(HIP|CUDA):BOOL=ON$' "$BUILD/CMakeCache.txt"; then
    echo "gpu-test_sh ERROR: '$BUILD' is CPU-only; run ctest directly, this wrapper is for GPU builds." >&2
    exit 2
fi

mkdir -p "$ROOT/scratch"
exec 9>"$LOCK"
if ! flock -w "$WAIT" 9; then
    echo "gpu-test_sh ERROR: timed out after ${WAIT}s waiting for $LOCK (another GPU test holds it)." >&2
    exit 2
fi

printf 'pid=%s label=%s started=%s build=%s host=%s\n' \
    "$$" "$LABEL" "$(date -Is)" "$BUILD" "$(hostname)" > "$LOCK"
printf '%s pid=%s label=%s build=%s ACQUIRED\n' "$(date -Is)" "$$" "$LABEL" "$BUILD" >> "$HISTORY"

STATUS=0
ctest --test-dir "$BUILD" --output-on-failure "$@" || STATUS=$?

printf '%s pid=%s label=%s build=%s RELEASED exit=%s\n' "$(date -Is)" "$$" "$LABEL" "$BUILD" "$STATUS" >> "$HISTORY"
flock -u 9
exit "$STATUS"
