#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
LOG_FILE="$ROOT_DIR/logs/network_capacity_search.log"

cd "$ROOT_DIR"
mkdir -p logs
export PYTHONUNBUFFERED=1

echo "[$(date --iso-8601=seconds)] 启动或继续IPGDO网络容量参数搜索" | tee -a "$LOG_FILE"
"$PYTHON_BIN" run_ipgdo_parameter_search.py "$@" 2>&1 | tee -a "$LOG_FILE"
