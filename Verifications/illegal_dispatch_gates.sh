#!/usr/bin/env bash
# Per-TU dispatch-gate invariant.
# IncludedCPUFeatures() is header-defined with internal linkage: every TU
# compiles its own copy under that TU's flags. An unflagged TU's copy comes
# up empty, so a has_feature() gate called from a TU without per-file ISA
# flags silently returns false and dispatch falls to scalar - the flags look
# right, the kernels exist, everything runs, just slow.
# Invariant: a .cpp calling has_feature() must carry per-file ISA flags via
# set_source_files_properties(... PROPERTIES COMPILE_OPTIONS ...) in the
# top-level CMakeLists.txt. Ungated TUs may only read .threads.
# Contract: $1 = repo root, $2 = build dir. Exit non-zero = violation.
# A check that can't run is a failure.

set -euo pipefail
ROOT="${1:-.}"
# shellcheck disable=SC2034
BUILD="${2:-}"
cd "$ROOT"

if ! command -v git >/dev/null 2>&1; then
    echo "dispatch_gates_flagged_sh ERROR: git not available - a check that can't run is a failure." >&2
    exit 2
fi
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "dispatch_gates_flagged_sh ERROR: '$ROOT' is not a git work tree - a check that can't run is a failure." >&2
    exit 2
fi
if [[ ! -f CMakeLists.txt ]]; then
    echo "dispatch_gates_flagged_sh ERROR: CMakeLists.txt not found in '$ROOT'." >&2
    exit 2
fi

# Every source the build flags per-file (set_source_files_properties
# ... PROPERTIES COMPILE_OPTIONS ...), extracted from the CMake lists.
FLAGGED=$(awk '
    /set_source_files_properties\(/ { inblock=1 }
    inblock {
        line=$0
        if (line ~ /\)/) { inblock=0 }
        gsub(/[()]/, "", line)
        n=split(line, f, /[[:space:]]+/)
        for (i=1;i<=n;i++) {
            t=f[i]
            if (t=="PROPERTIES" || t=="COMPILE_OPTIONS") continue
            if (t ~ /^-/) continue
            if (t ~ /\.(cpp|cu|hip)$/) print t
        }
    }
' CMakeLists.txt | sort -u)

if [[ -z "$FLAGGED" ]]; then
    echo "dispatch_gates_flagged_sh ERROR: no per-file flagged sources found in CMakeLists.txt." >&2
    exit 2
fi

# Tracked .cpp callers of has_feature() (headers excluded by construction;
# Hardware.h defines the gate bodies under their own #if guards by design).
CALLERS=$(git grep -l 'has_feature(' -- '*.cpp' || true)

VIOLATIONS=""
if [[ -n "$CALLERS" ]]; then
    while IFS= read -r f; do
        if ! grep -qxF "$f" <<< "$FLAGGED"; then
            VIOLATIONS="${VIOLATIONS}${f}
"
        fi
    done <<< "$CALLERS"
fi

if [[ -n "$VIOLATIONS" ]]; then
    echo "dispatch_gates_flagged_sh ERROR: has_feature() called from a TU without per-file ISA flags - that TU's IncludedCPUFeatures() copy is blind and the gate always returns false:" >&2
    printf '%s' "$VIOLATIONS" >&2
    echo "dispatch_gates_flagged_sh: fix = add the TU to a set_source_files_properties(... COMPILE_OPTIONS -m...) block, or move the gate into an already-flagged kernel TU." >&2
    exit 1
fi
echo "dispatch_gates_flagged_sh: OK - every has_feature() caller carries per-file ISA flags."
exit 0
