#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/yamromano/Documents/PythonF4/PythonF4/drone_phase1_replay_demo_20260706"
PY="/Users/yamromano/Documents/PythonF4/PythonF4/.venv-metis/bin/python"
PORT="${1:-8767}"

cd "$ROOT"
mkdir -p "$ROOT/.atlas_runtime"

echo "ATLAS keepalive server"
echo "URL: http://127.0.0.1:$PORT/"
echo "Press Ctrl-C to stop."

while true; do
  "$PY" scripts/atlas_app_server.py --port "$PORT"
  code="$?"
  echo "ATLAS server exited with code $code; restarting in 1 second..."
  sleep 1
done
