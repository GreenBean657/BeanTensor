#!/usr/bin/env bash
# Bind-clean public API check.
# Public headers must stay value-in/value-out so a future pybind11 surface can
# bind them without non-const reference or mutable pointer parameters.
#
# Exit codes:
#   0 = all checked public signatures are bind-clean.
#   1 = a non-const reference or mutable pointer parameter was found.
#   2 = internal error (scanner missing) or no checked headers exist.
#
# Contract: $1 = repo root, $2 = build dir (unused). Optional extra headers may
# be passed as $3+; this is the negative-control hook used by the test suite.

set -euo pipefail

ROOT="${1:-.}"
# $2 is the build dir, unused by this check.

cd "$ROOT"

BASE_HEADERS=(
    Module/Module.h
    Module/Linear.h
    Module/Relu.h
    Module/Sequential.h
    Random/Generator.h
    Data/DataLoader.h
)

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCANNER="$SCRIPT_DIR/bind_clean_scanner.awk"

if [[ ! -f "$SCANNER" ]]; then
    echo "bind_clean_public_api.sh ERROR: scanner '$SCANNER' not found." >&2
    exit 2
fi

HEADERS=()
for h in "${BASE_HEADERS[@]}"; do
    if [[ -f "$h" ]]; then
        HEADERS+=("$h")
    fi
done

for ((i = 3; i <= $#; i++)); do
    h="${!i}"
    if [[ -f "$h" ]]; then
        HEADERS+=("$h")
    fi
done

if [[ ${#HEADERS[@]} -eq 0 ]]; then
    echo "bind_clean_public_api.sh ERROR: none of the checked headers exist." >&2
    exit 2
fi

for h in "${HEADERS[@]}"; do
    awk -f "$SCANNER" "$h"
done

exit 0
