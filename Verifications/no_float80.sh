#!/usr/bin/env bash
# Prohibit long double / float80_t anywhere in BeanTensor sources.
# Exit 1 = Violation. Repo root as $1 argument.

set -euo pipefail
ROOT="${1:-.}"
cd "$ROOT"

DIRS=(Tensor Helpers Intrinsics tests)
FILES=(main.cpp)

for d in "${DIRS[@]}"; do
    if [[ ! -d "$d" ]]; then
        echo "nofloat80_sh ERROR: expected source directory '$d' not found." >&2
        exit 2
    fi
done

if grep -rEn \
    --include='*.cpp' --include='*.h' --include='*.hpp' \
    --include='*.cu'  --include='*.cuh' \
    --include='*.hip' --include='*.hip.hpp' \
    '\b(long[[:space:]]+double|float80)\b' \
    "${DIRS[@]}" "${FILES[@]}"; then
    echo "nofloat80_sh ERROR: Attempted usage of float80_t / long double - remember, this is incompatible with GPUs and thus encourages CPU>GPU pathway divergence." >&2
    exit 1
fi
exit 0
