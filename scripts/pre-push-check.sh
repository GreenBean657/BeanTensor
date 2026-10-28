#!/usr/bin/env bash
# pre-push-check.sh — the BeanTensor pre-push gate. Every phase must pass before
# anything reaches the remote; any failure refuses the push with the evidence.
# Phases exist because each one caught (or would have caught) a real incident:
#   A. tree clean           — uncommitted stragglers do not get left behind
#   B. history shape        — the pushed range is the shape we intend (one commit
#                             ahead of origin/main by default; merge commits in
#                             the range are refused — they are how dead branch
#                             names leak into public history)
#   C. message + tree audit — no agentic tooling markers in commit messages or
#                             tracked files (project-instructions references,
#                             mission IDs, orchestration vocabulary)
#   D. workflow lint        — any workflow touched in the range must declare a
#                             permissions: block and pin every action to a full
#                             commit SHA (GitHub repo policy)
#   E. tests on the tree    — full CPU ctest + platform loop + gated HIP run
#                             (any test failure refuses the push outright)
# Bypass for a deliberate exception: BEANTENSOR_SKIP_PREPUSH=1 (the hook warns).
set -uo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
FAIL=0
phase() { printf '\n=== [%s] %s\n' "$(date +%H:%M:%S)" "$1"; }
pass() { echo "PASS: $1"; }
fail() { echo "FAIL: $1"; FAIL=1; }

# Pick build parallelism from available memory, matching the project rule.
FREE_KB="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
if [[ "$FREE_KB" -gt 33554432 ]]; then
    JOBS="$(nproc)"
elif [[ "$FREE_KB" -gt 4194304 ]]; then
    JOBS=8
else
    echo "pre-push: only ${FREE_KB}KB available; refusing to build under 4GB." >&2
    exit 1
fi

# ---- A. tree clean ----
phase "A. working tree clean"
if [[ -z "$(git status --porcelain)" ]]; then
    pass "no uncommitted changes"
else
    fail "uncommitted changes present:"; git status --porcelain | head -5
fi

# ---- B. history shape ----
phase "B. history shape"
BRANCH="$(git branch --show-current)"
if [[ "$BRANCH" != "main" ]]; then
    fail "on branch '$BRANCH', expected main — pushing a side branch is a decision, not a default"
else
    pass "on main"
fi
RANGE="$(git rev-parse origin/main 2>/dev/null)..HEAD"
AHEAD="$(git rev-list --count "$RANGE")"
if [[ "$AHEAD" == "0" ]]; then
    fail "nothing to push"
elif [[ "$AHEAD" != "1" && "${BEANTENSOR_ALLOW_MULTICOMMIT:-0}" != "1" ]]; then
    fail "$AHEAD commits ahead of origin/main; the house shape is ONE commit (set BEANTENSOR_ALLOW_MULTICOMMIT=1 to declare this is intentional)"
else
    pass "$AHEAD commit(s) ahead of origin/main"
fi
MERGES="$(git log --merges --format='%h %s' "$RANGE")"
if [[ -n "$MERGES" ]]; then
    fail "merge commits in the push range (dead branch names leak into public history):"; echo "$MERGES"
else
    pass "no merge commits in the push range"
fi

# ---- C. agentic-marker audit ----
phase "C. agentic-marker audit (messages + tracked files)"
PATTERNS='AGENTS\.md|AGENTS\b|[Tt]ower|swarm|subagent|kimi|claude|copilot|reviewer|mission M[0-9]|^M[0-9]+[ :]|review round'
MSG_HITS="$(git log "$RANGE" --format='%H %s%n%b' | grep -inE "$PATTERNS" || true)"
if [[ -n "$MSG_HITS" ]]; then
    fail "markers in commit messages:"; echo "$MSG_HITS" | head -6
else
    pass "commit messages clean"
fi
# The auditor is exempt from its own scan: its pattern literal names the
# markers it bans, so it can never be clean under its own regex.
TREE_HITS="$(git grep -inE "$PATTERNS" -- . ':(exclude)scripts/pre-push-check.sh' || true)"
if [[ -n "$TREE_HITS" ]]; then
    fail "markers in tracked files:"; echo "$TREE_HITS" | head -6
else
    pass "tracked files clean"
fi

# ---- D. workflow lint (only when workflows changed) ----
phase "D. workflow lint"
WF_CHANGED="$(git diff --name-only "$RANGE" -- .github/workflows/ || true)"
if [[ -z "$WF_CHANGED" ]]; then
    pass "no workflow changes in range"
else
    echo "workflows changed: $WF_CHANGED"
    for f in $WF_CHANGED; do
        if ! grep -q '^permissions:' "$f"; then
            fail "$f has no permissions: block (CodeQL actions/missing-workflow-permissions)"
        fi
        # every uses: pin must be a full 40-hex SHA
        BAD_PINS="$(grep -E 'uses:' "$f" | grep -vE '@[0-9a-f]{40}\b' || true)"
        if [[ -n "$BAD_PINS" ]]; then
            fail "$f has non-SHA-pinned actions:"; echo "$BAD_PINS"
        fi
    done
    [[ "$FAIL" == "0" ]] && pass "workflow lint clean"
fi

# ---- E. tests on the exact tree ----
phase "E. test matrix"
BUILD=scratch/build-verify-fresh
if [[ ! -f "$BUILD/CMakeCache.txt" ]]; then
    cmake -S . -B "$BUILD" -DUSE_HIP=OFF -DUSE_CUDA=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
fi
BUILD_LOG="$BUILD/pre-push-build.log"
BUILD_OK=1
cmake --build "$BUILD" -j "$JOBS" >"$BUILD_LOG" 2>&1 || BUILD_OK=0
if [[ "$BUILD_OK" == "0" ]] || grep -qE 'error:' "$BUILD_LOG"; then
    if grep -qE 'error:' "$BUILD_LOG"; then
        fail "CPU build has compile errors"
    else
        fail "CPU build failed"
    fi
else
    pass "CPU build clean"
fi
CTEST_OUT="$(ctest --test-dir "$BUILD" --output-on-failure 2>&1)"
if echo "$CTEST_OUT" | grep -q '100% tests passed'; then
    pass "ctest: $(echo "$CTEST_OUT" | grep 'tests passed')"
else
    fail "ctest:"; echo "$CTEST_OUT" | tail -5
fi
LOOP_OK=1
for i in 1 2 3; do
    "./$BUILD/BeanTensorV5PlatformTests" > /dev/null 2>&1 || LOOP_OK=0
done
if [[ "$LOOP_OK" == "1" ]]; then
    pass "platform suite 3x loop clean"
else
    fail "platform suite loop failed"
fi
if [[ -f scratch/build-soak-hip/CMakeCache.txt ]]; then
    exec 9>scratch/.gpu-test.lock
    if flock -w 300 9; then
        HIP_OUT="$(bash gpu-test.sh pre-push-gate scratch/build-soak-hip 2>&1)"
        if echo "$HIP_OUT" | grep -q '100% tests passed'; then
            pass "gated HIP ctest clean"
        else
            fail "gated HIP ctest:"; echo "$HIP_OUT" | tail -3
        fi
    else
        fail "GPU lock timeout — refused to push without the gated HIP run"
    fi
fi

phase "SUMMARY"
if [[ "$FAIL" == "0" ]]; then
    echo "PRE-PUSH GATE: ALL PHASES PASS"
    exit 0
fi
echo "PRE-PUSH GATE: REFUSED — fix the failures above (or declare intent with BEANTENSOR_SKIP_PREPUSH=1)"
exit 1
