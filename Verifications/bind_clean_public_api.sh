#!/usr/bin/env bash
# Bind-clean public API check.
# Public headers must stay value-in/value-out so a future pybind11 surface can
# bind them without non-const references to fundamental / fixed-width-alias
# types or mutable pointer parameters. Non-const references to class types are
# permitted (e.g. Metrics::evaluate(Module&, DataLoader&)).
#
# Exit codes:
#   0 = all checked public signatures are bind-clean.
#   1 = a banned parameter shape was found.
#   2 = internal error (scanner missing) or no checked headers exist.
#
# Contract: $1 = repo root, $2 = build dir (unused). Optional extra headers may
# be passed as $3+; this is the negative-control hook used by the test suite.

set -euo pipefail
shopt -s nullglob

ROOT="${1:-.}"
# $2 is the build dir, unused by this check.

cd "$ROOT"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCANNER="$SCRIPT_DIR/bind_clean_scanner.awk"

if [[ ! -f "$SCANNER" ]]; then
    echo "bind_clean_public_api.sh ERROR: scanner '$SCANNER' not found." >&2
    exit 2
fi

# Glob public headers from the known surface directories. ModuleStream.h is
# excluded by design because its streaming operators take std::ostream&.
HEADERS=()
for h in Module/*.h Random/*.h Data/*.h Metrics/*.h; do
    if [[ "$(basename "$h")" == "ModuleStream.h" ]]; then
        continue
    fi
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
