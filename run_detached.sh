#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$ROOT_DIR/logs/network_capacity_search.nohup.log"
PID_FILE="$ROOT_DIR/logs/network_capacity_search.pid"

cd "$ROOT_DIR"
mkdir -p logs

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "实验已经在运行，PID=$(cat "$PID_FILE")"
  exit 1
fi

nohup bash "$ROOT_DIR/run_search.sh" "$@" >"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"
echo "后台实验已启动，PID=$(cat "$PID_FILE")"
echo "日志: $LOG_FILE"
