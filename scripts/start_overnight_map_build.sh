#!/bin/zsh
set -u

PROJECT=/Users/yamromano/Documents/PythonF4/PythonF4/drone_phase1_replay_demo_20260706
RUNTIME="$PROJECT/runtime/overnight_map_build_20260727"
PYTHON=/Users/yamromano/Documents/PythonF4/PythonF4/.venv-metis/bin/python
VIDEO="$PROJECT/data/source_videos/IMG_9961.MOV"

mkdir -p "$RUNTIME"
cd "$PROJECT" || exit 70
exec /usr/bin/caffeinate -dimsu "$PYTHON" \
  "$PROJECT/scripts/run_direct_video_map_build.py" \
  "$VIDEO" \
  --log "$RUNTIME/build.log" \
  --status "$RUNTIME/status.json" \
  >>"$RUNTIME/launcher.log" 2>&1
