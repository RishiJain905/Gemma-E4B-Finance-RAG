#!/usr/bin/env bash
# scripts/verify.sh — deterministic verification gate for agent loops
# Runs ruff + the offline pytest suite; the FINAL line is a machine-readable verdict.
# Usage: bash scripts/verify.sh [tests/test_x.py]
#   An argument scopes pytest to one file for fast inner-loop iteration (ruff still runs on everything).
# Output contract: last line is "VERIFY: PASS" (exit 0) or "VERIFY: FAIL (<stages>)" (exit 1).
# This line is the stop condition for /goal loops — see how-to-loop.md.

set -u
cd "$(dirname "$0")/.."

if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON=".venv/Scripts/python.exe"
else
    PYTHON="python"
fi

TARGET="${1:-tests/}"
FAILED=""

echo "== ruff =="
"$PYTHON" -m ruff check . || FAILED="$FAILED ruff"

echo "== pytest (offline: $TARGET) =="
"$PYTHON" -m pytest "$TARGET" -q -m "not live" --maxfail=10 || FAILED="$FAILED pytest"

if [ -z "$FAILED" ]; then
    echo "VERIFY: PASS"
    exit 0
else
    echo "VERIFY: FAIL ($(echo $FAILED | tr ' ' ','))"
    exit 1
fi
