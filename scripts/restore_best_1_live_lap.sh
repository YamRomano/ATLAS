#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_DIR="$ROOT_DIR/restore_assets/best_1_live_lap"
MODE="${1:---verify-only}"

if [[ "$MODE" != "--verify-only" && "$MODE" != "--restore" ]]; then
  echo "Usage: $0 [--verify-only|--restore]" >&2
  exit 2
fi

cd "$ASSET_DIR"
shasum -a 256 -c SHA256SUMS

if [[ "$MODE" == "--verify-only" ]]; then
  echo "All best-1-live-lap Git LFS archive parts passed SHA-256 verification."
  exit 0
fi

cd "$ROOT_DIR"
cat "$ASSET_DIR"/active_colmap_map.tar.part-* | tar -xf -
cat "$ASSET_DIR"/best_full_lap_run.tar.part-* | tar -xf -
cat "$ASSET_DIR"/colmap_runtime_macos_arm64.tar.part-* | tar -xf -

required=(
  "results/maps/map_copy_20260730_114851_cfefdc/colmap/database.db"
  "results/maps/map_copy_20260730_114851_cfefdc/colmap/sparse/0/images.bin"
  "viewer/public/live_dji_sessions/atlas_dji_live_20260823_154451_94903f/query_frames/query_004051.jpg"
  "viewer/public/maps/map_copy_20260730_114851_cfefdc/replays/dji_live_20260823_154451_94903f/poses.json"
  "tools/colmap-env/bin/colmap"
  "vendor/opendji/OpenDJI.py"
)
for path in "${required[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "Restore verification failed: missing $path" >&2
    exit 1
  fi
done

echo "Restored the active COLMAP map, exact full-lap evidence, and macOS arm64 COLMAP runtime."
echo "Create a Python environment from requirements.txt, then start scripts/atlas_app_server.py."
