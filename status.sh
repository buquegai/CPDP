#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

cd "$ROOT_DIR"
"$PYTHON_BIN" search_status.py

if [[ -f logs/network_capacity_search.pid ]]; then
  PID="$(cat logs/network_capacity_search.pid)"
  if kill -0 "$PID" 2>/dev/null; then
    echo "后台进程: 正在运行，PID=$PID"
  else
    echo "后台进程: PID文件存在，但进程已结束，PID=$PID"
  fi
fi
