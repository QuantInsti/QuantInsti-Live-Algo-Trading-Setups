#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"
python -m build
pip install dist/ibkr_multi_asset-1.0.0-py3-none-any.whl --force-reinstall --no-deps
cd "${SCRIPT_DIR}"
python main.py
