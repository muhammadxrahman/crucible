#!/usr/bin/env bash
# Regression gate. Runs the lint + test suite that must pass before pushing.
# Invoked by the git pre-push hook and runnable by hand:  ./scripts/check.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> ruff format --check"
uv run ruff format --check .

echo "==> ruff check"
uv run ruff check .

echo "==> pytest"
uv run pytest

echo "==> all checks passed"
