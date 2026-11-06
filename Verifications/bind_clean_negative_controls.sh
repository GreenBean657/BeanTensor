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

# std::optional positive controls: by value and by const reference are allowed.
cat > "$PROBE_DIR/optional_value.h" <<'EOF'
void j(std::optional<int32_t> value);
EOF
cat > "$PROBE_DIR/optional_const_ref.h" <<'EOF'
void k(const std::optional<float32_t>& value);
EOF
expect "$PROBE_DIR/optional_value.h" 0 "std::optional by value is allowed"
expect "$PROBE_DIR/optional_const_ref.h" 0 "const std::optional& is allowed"

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

# Non-const lvalue reference to std::optional is banned.
cat > "$PROBE_DIR/optional_ref.h" <<'EOF'
void f(std::optional<int32_t>& value);
EOF
cat > "$PROBE_DIR/optional_alias_ref.h" <<'EOF'
void f(std::optional<float64_t>& value);
EOF
expect "$PROBE_DIR/optional_ref.h" 1 "std::optional<int32_t>& is banned"
expect "$PROBE_DIR/optional_alias_ref.h" 1 "std::optional<float64_t>& is banned"

# C-style variadic parameters are banned.
cat > "$PROBE_DIR/variadic.h" <<'EOF'
void f(const char* fmt, ...);
EOF
expect "$PROBE_DIR/variadic.h" 1 "C-style variadic parameter is banned"

# Function-pointer parameters are banned.
cat > "$PROBE_DIR/function_pointer.h" <<'EOF'
void f(void (*callback)(int32_t));
EOF
cat > "$PROBE_DIR/function_pointer_alias_args.h" <<'EOF'
void g(int32_t (*hook)(float32_t, Tensor&));
EOF
cat > "$PROBE_DIR/const_function_pointer.h" <<'EOF'
void f(void (* const callback)(int32_t));
EOF
expect "$PROBE_DIR/function_pointer.h" 1 "function-pointer parameter is banned"
expect "$PROBE_DIR/function_pointer_alias_args.h" 1 "function-pointer with alias args is banned"
expect "$PROBE_DIR/const_function_pointer.h" 1 "const function-pointer parameter is banned"

# Template parameter packs are NOT C-style variadics; similarly-named class
# types are NOT std::optional.
cat > "$PROBE_DIR/template_pack.h" <<'EOF'
template <typename... Args>
void f(Args... args);
EOF
cat > "$PROBE_DIR/template_forwarding_pack.h" <<'EOF'
template <typename... Args>
void g(Args&&... args);
EOF
cat > "$PROBE_DIR/optional_lookalike_class.h" <<'EOF'
void h(my_namespace::optional_config& cfg);
EOF
expect "$PROBE_DIR/template_pack.h" 0 "template parameter pack is allowed"
expect "$PROBE_DIR/template_forwarding_pack.h" 0 "template forwarding pack is allowed"
expect "$PROBE_DIR/optional_lookalike_class.h" 0 "class type containing 'optional' is allowed"

if [[ "$fail" -ne 0 ]]; then
    exit 1
fi

echo "bind_clean_negative_controls.sh OK"
exit 0
