#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --locked
cmake --preset release
cmake --build --preset release -j 4
ctest --preset release
uv run pytest -q
uv run python tools/codegen/generate.py --check
uv run python tools/check_dependencies.py
uv run swarm-walkthrough "$@"
