#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_BUILD="${1:-cu124}"

cd "$ROOT_DIR"

"$PYTHON_BIN" - <<'PY'
import sys

if not ((3, 9) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit(
        f"需要 Python 3.9-3.12，当前版本为 {sys.version.split()[0]}"
    )
PY

"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

case "$TORCH_BUILD" in
  cu118|cu124|cu126)
    python -m pip install torch==2.6.0 \
      --index-url "https://download.pytorch.org/whl/${TORCH_BUILD}"
    ;;
  cpu)
    python -m pip install torch==2.6.0 \
      --index-url https://download.pytorch.org/whl/cpu
    ;;
  *)
    echo "用法: bash setup_linux.sh [cu118|cu124|cu126|cpu]" >&2
    exit 2
    ;;
esac

python -m pip install -r requirements-linux.txt
python verify_package.py
python run_ipgdo_parameter_search.py --check

echo "环境准备完成。先运行: bash plan.sh"
