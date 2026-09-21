#!/usr/bin/env bash
# jev-laya-free — clean, repository-local training/test environment.
#
# Builds a fresh venv in THIS repository (never the shared ComfyUI env or any
# global site-packages), installs the exact-pinned training/test stack, verifies
# it with `pip check`, and (with --test) runs the full suite.
#
# Usage:
#   scripts/setup_env.sh            # create .venv + install + pip check
#   scripts/setup_env.sh --test    # ... then run the full test suite
#   scripts/setup_env.sh --check   # only verify an existing .venv (pip check)
#
# Idempotent: re-running into an existing .venv just re-verifies/reinstalls.
# Offline note: the base package is standard-library only; only the [training]
# stack needs network (PyPI). No credentials or hosted API are required.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

MODE="${1:-setup}"
VENV="$REPO_DIR/.venv"

# Pick the venv interpreter (created or existing).
if [[ "$MODE" != "--check" && ! -x "$VENV/bin/python" ]]; then
  echo ">> Creating a fresh repository-local venv at $VENV"
  python3 -m venv "$VENV"
fi
PY="$VENV/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "ERROR: no venv python at $PY. Run without --check first." >&2
  exit 1
fi

echo ">> Upgrading pip"
"$PY" -m pip install --upgrade pip

if [[ "$MODE" == "--check" ]]; then
  echo ">> Verifying existing environment with pip check"
  "$PY" -m pip check
  echo "OK: environment verified."
  exit 0
fi

echo ">> Installing the exact-pinned training/test stack (requirements-training.txt)"
"$PY" -m pip install -r requirements-training.txt

echo ">> Verifying with pip check"
"$PY" -m pip check

echo "OK: clean training/test environment is ready at $VENV"
echo "    Run the suite with:"
echo "      PYTHONPATH=src $PY -m pytest tests/ -q"

if [[ "$MODE" == "--test" ]]; then
  echo ">> Running the full test suite"
  PYTHONPATH=src "$PY" -m pytest tests/ -q
fi
