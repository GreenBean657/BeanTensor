#!/usr/bin/env bash
# Raw aligned_alloc is banned project-wide. All CPU buffers must come through
# bt_aligned_alloc (Helpers/Alloc.h), which rounds sizes to a 64-byte multiple:
# TSAN intercepts aligned_alloc and enforces the C11 size-alignment rule
# strictly, so any raw aligned_alloc(64, odd_size) aborts before tests run.
# Contract: $1 = repo root, $2 = build dir. Exit non-zero = violation.
# A check that can't run is a failure.

set -euo pipefail
ROOT="${1:-.}"
cd "$ROOT"

if ! command -v git >/dev/null 2>&1; then
    echo "aligned_alloc_banned_sh ERROR: git not available - a check that can't run is a failure." >&2
    exit 2
fi
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "aligned_alloc_banned_sh ERROR: '$ROOT' is not a git work tree - a check that can't run is a failure." >&2
    exit 2
fi

# Tracked sources only (git excludes .git, scratch, and untracked build dirs
# by construction). Helpers/Alloc.h is the one legal caller of aligned_alloc.
HITS=$(git grep -n -E '(^|[^_[:alnum:]])aligned_alloc[[:space:]]*\(' -- '*.h' '*.hpp' '*.cpp' '*.cu' '*.cuh' \
    | grep -v '^Helpers/Alloc\.h:' || true)

if [[ -n "$HITS" ]]; then
    echo "aligned_alloc_banned_sh ERROR: raw aligned_alloc is banned project-wide." >&2
    echo "Allocate CPU buffers via bt_aligned_alloc (Helpers/Alloc.h) instead:" >&2
    echo "$HITS" >&2
    exit 1
fi
echo "aligned_alloc_banned_sh: OK - every aligned_alloc lives in Helpers/Alloc.h."
exit 0
