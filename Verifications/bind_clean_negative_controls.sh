#!/usr/bin/env bash
# Negative and positive controls for bind_clean_scanner.awk.
# Ensures apostrophes inside comments do not mask subsequent banned declarations,
# and that every packed dtype alias is rejected as a non-const lvalue reference.
#
# Contract: $1 = repo root, $2 = build dir. Probe headers are written
# idempotently under $2/bind_clean_probes/ so the build dir owns their lifecycle;
# no temporary directories or deletions are used.

set -euo pipefail

ROOT="${1:-.}"
BUILD_DIR="${2:-build}"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCANNER="$SCRIPT_DIR/bind_clean_scanner.awk"

cd "$ROOT"

if [[ ! -f "$SCANNER" ]]; then
    echo "bind_clean_negative_controls.sh ERROR: scanner '$SCANNER' not found." >&2
    exit 2
fi

PROBE_DIR="$BUILD_DIR/bind_clean_probes"
mkdir -p "$PROBE_DIR"

fail=0

# Helper: run scanner, record failure if exit code differs from expected.
expect() {
    local file="$1"
    local want="$2"
    local label="$3"
    local got
    if awk -f "$SCANNER" "$file" >/dev/null 2>&1; then
        got=0
    else
        got=1
    fi
    if [[ "$got" != "$want" ]]; then
        echo "bind_clean_negative_controls.sh FAIL: $label (expected exit $want, got $got)" >&2
        fail=1
    fi
}

# Positive controls: allowed declarations must pass.
cat > "$PROBE_DIR/const_ref_alias.h" <<'EOF'
const float64_t& g();
EOF
cat > "$PROBE_DIR/class_ref.h" <<'EOF'
void h(Tensor& t);
void i(Generator& g);
EOF
expect "$PROBE_DIR/const_ref_alias.h" 0 "const float64_t& is allowed"
expect "$PROBE_DIR/class_ref.h" 0 "class-type non-const references are allowed"

# Negative controls: banned declarations must fail.
cat > "$PROBE_DIR/bare_int_ref.h" <<'EOF'
void f(int& value);
EOF
expect "$PROBE_DIR/bare_int_ref.h" 1 "bare int& is banned"

# Negative controls after apostrophe-containing comments must still fail.
cat > "$PROBE_DIR/line_comment_apostrophe.h" <<'EOF'
// caller's flag
void f(int& value);
EOF
cat > "$PROBE_DIR/block_comment_apostrophe.h" <<'EOF'
/** caller's flag */
void f(int& value);
EOF
cat > "$PROBE_DIR/block_comment_multi_line_apostrophe.h" <<'EOF'
/*
 * caller's flag
 */
void f(int& value);
EOF
expect "$PROBE_DIR/line_comment_apostrophe.h" 1 "int& after // apostrophe comment is banned"
expect "$PROBE_DIR/block_comment_apostrophe.h" 1 "int& after /** apostrophe comment is banned"
expect "$PROBE_DIR/block_comment_multi_line_apostrophe.h" 1 "int& after /* apostrophe comment is banned"

# Packed dtype aliases must be banned as non-const lvalue references.
cat > "$PROBE_DIR/packed_float16x2.h" <<'EOF'
void f(float16x2_t& value);
EOF
cat > "$PROBE_DIR/packed_bfloat16x2.h" <<'EOF'
void f(bfloat16x2_t& value);
EOF
cat > "$PROBE_DIR/packed_int4x2.h" <<'EOF'
void f(int4x2_t& value);
EOF
expect "$PROBE_DIR/packed_float16x2.h" 1 "float16x2_t& is banned"
expect "$PROBE_DIR/packed_bfloat16x2.h" 1 "bfloat16x2_t& is banned"
expect "$PROBE_DIR/packed_int4x2.h" 1 "int4x2_t& is banned"

if [[ "$fail" -ne 0 ]]; then
    exit 1
fi

echo "bind_clean_negative_controls.sh OK"
exit 0
