#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/yamromano/Documents/PythonF4/PythonF4/drone_phase1_replay_demo_20260706"
PORT="${1:-8767}"
PID_FILE="$ROOT/.atlas_runtime/atlas_server_${PORT}.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No ATLAS PID file for port $PORT."
  exit 0
fi

PID="$(cat "$PID_FILE" || true)"
if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  echo "Stopped ATLAS on port $PORT (pid $PID)."
else
  echo "ATLAS process for port $PORT was not running."
fi

rm -f "$PID_FILE"
