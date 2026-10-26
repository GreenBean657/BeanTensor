#!/usr/bin/env bash
# CUDA 12.9 + glibc >= 2.41 compatibility invariant.
# glibc's C23 math prototypes (cospi/sinpi/rsqrt + f variants) are declared
# noexcept; CUDA 12.9's crt/math_functions.h redeclares the same 6 functions
# without it, which is a hard compile error under _GNU_SOURCE. The fix is a
# 6-line noexcept(true) patch to the toolkit header (needs sudo, once per
# toolkit install). This check FAILS LOUD with the exact command when a CUDA
# build is configured against an unpatched header.
# Contract: $1 = repo root, $2 = build dir. Exit non-zero = violation.

set -euo pipefail
# shellcheck disable=SC2034
ROOT="${1:-.}"
BUILD="${2:-}"

# Only applies when the CUDA backend is actually selected.
CACHE="$BUILD/CMakeCache.txt"
if [[ -z "$BUILD" || ! -f "$CACHE" ]]; then
    echo "cuda_glibc_noexcept_sh ERROR: cannot read '$CACHE' - a check that can't run is a failure." >&2
    exit 2
fi
if ! grep -q '^USE_CUDA:BOOL=ON$' "$CACHE"; then
    exit 0
fi

HEADER="/usr/local/cuda-12.9/targets/x86_64-linux/include/crt/math_functions.h"
if [[ ! -f "$HEADER" ]]; then
    echo "cuda_glibc_noexcept_sh ERROR: USE_CUDA=ON but CUDA 12.9 header '$HEADER' not found." >&2
    exit 2
fi

MISSING=0
for fn in cospi cospif sinpi sinpif rsqrt rsqrtf; do
    if ! grep -Eq " $fn\((double|float) x\) noexcept\(true\);" "$HEADER"; then
        echo "cuda_glibc_noexcept_sh ERROR: '$fn' in $HEADER lacks noexcept(true)." >&2
        MISSING=1
    fi
done

if [[ "$MISSING" -ne 0 ]]; then
    cat >&2 <<'EOF'
cuda_glibc_noexcept_sh ERROR: CUDA 12.9 header conflicts with glibc >= 2.41 C23 math prototypes.
Apply the documented one-time toolkit patch (see the project build notes):

  sudo sed -i -E 's/^(extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ +(double|float) +(cospi|cospif|sinpi|sinpif|rsqrt|rsqrtf)\((double|float) x\));/\1 noexcept(true);/' /usr/local/cuda-12.9/targets/x86_64-linux/include/crt/math_functions.h
EOF
    exit 1
fi
exit 0
