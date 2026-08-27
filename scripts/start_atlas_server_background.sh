#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/yamromano/Documents/PythonF4/PythonF4/drone_phase1_replay_demo_20260706"
PY="/Users/yamromano/Documents/PythonF4/PythonF4/.venv-metis/bin/python"
PORT="${1:-8767}"
RUN_DIR="$ROOT/.atlas_runtime"
PID_FILE="$RUN_DIR/atlas_server_${PORT}.pid"
LOG_FILE="$RUN_DIR/atlas_server_${PORT}.log"

mkdir -p "$RUN_DIR"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" || true)"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "ATLAS is already running on port $PORT (pid $OLD_PID)."
    echo "URL: http://127.0.0.1:$PORT/"
    exit 0
  fi
fi

cd "$ROOT"
nohup "$PY" scripts/atlas_app_server.py --port "$PORT" >> "$LOG_FILE" 2>&1 &
PID="$!"
echo "$PID" > "$PID_FILE"

echo "Started ATLAS on port $PORT (pid $PID)."
echo "URL: http://127.0.0.1:$PORT/"
echo "Log: $LOG_FILE"
