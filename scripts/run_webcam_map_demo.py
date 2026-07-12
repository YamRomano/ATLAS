#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run(cmd: list[object]) -> None:
    cmd = [str(x) for x in cmd]
    print("\n+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Capture a live webcam scan, run COLMAP, and export it to ATLAS.")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--fps", type=float, default=1.5)
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--matcher", choices=["sequential", "exhaustive"], default="sequential")
    ap.add_argument("--reuse-frames", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path(cfg["work_dir"]).resolve()
    py = Path(cfg["python"])
    scripts = root / "scripts"
    frames = root / "data" / "webcam_map_frames"
    colmap_out = root / "results" / "webcam_colmap_map"
    viewer_public = root / "viewer" / "public"

    if not args.reuse_frames:
        capture_cmd = [
            py,
            scripts / "capture_webcam_frames.py",
            "--out-dir",
            frames,
            "--camera-index",
            args.camera_index,
            "--duration",
            args.duration,
            "--fps",
            args.fps,
            "--max-size",
            cfg["max_image_size"],
            "--prefix",
            "webcam_map",
        ]
        if args.no_preview:
            capture_cmd.append("--no-preview")
        run(capture_cmd)

    run(
        [
            py,
            scripts / "run_colmap_map_only.py",
            "--colmap",
            cfg["colmap_bin"],
            "--frames",
            frames,
            "--out-dir",
            colmap_out,
            "--max-image-size",
            cfg["max_image_size"],
            "--camera-model",
            cfg["map_camera_model"],
            "--matcher",
            args.matcher,
        ]
    )

    run(
        [
            py,
            scripts / "build_map_only_viewer_data.py",
            "--model-text",
            colmap_out / "sparse_text",
            "--out-public",
            viewer_public,
            "--preserve-media",
        ]
    )

    print("\nLive webcam COLMAP map is ready.")
    print(f"Frames: {frames}")
    print(f"COLMAP output: {colmap_out}")
    print(f"Viewer data: {viewer_public}")
    print("Run:")
    print(f"  cd {root}")
    print("  python3 scripts/serve_viewer.py")
    print("  open http://127.0.0.1:8765")


if __name__ == "__main__":
    main()
