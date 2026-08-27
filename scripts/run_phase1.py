#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def run(cmd: list[object]) -> None:
    cmd = [str(x) for x in cmd]
    print("\n+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--reuse", action="store_true", help="Reuse existing frame/COLMAP outputs when present.")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path(cfg["work_dir"]).resolve()
    # Keep the configured venv path exactly. Calling resolve() follows the
    # symlink back to Apple's system Python and loses the venv site-packages.
    py = Path(cfg["python"])
    scripts = root / "scripts"
    data = root / "data"
    results = root / "results"

    map_frames = data / "map_frames"
    query_frames = data / "query_frames"
    colmap_out = results / "colmap"
    tsolve_inputs = results / "tsolve_inputs"
    tsolve_runtime = results / "tsolve_runtime"
    runtime_dir = results / "tsolve_runtime_code"
    viewer_public = root / "viewer" / "public"

    data.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    if not args.reuse or not map_frames.exists():
        run(
            [
                py,
                scripts / "extract_frames.py",
                "--video",
                cfg["iphone_map_video"],
                "--out-dir",
                map_frames,
                "--fps",
                cfg["map_frame_fps"],
                "--max-size",
                cfg["max_image_size"],
                "--prefix",
                "map",
            ]
        )

    if not args.reuse or not query_frames.exists():
        run(
            [
                py,
                scripts / "extract_frames.py",
                "--video",
                cfg["drone_query_video"],
                "--out-dir",
                query_frames,
                "--fps",
                cfg["query_frame_fps"],
                "--max-size",
                cfg["max_image_size"],
                "--prefix",
                "query",
            ]
        )

    if not args.reuse or not (colmap_out / "localized_model_text" / "images.txt").exists():
        run(
            [
                py,
                scripts / "run_colmap_pipeline.py",
                "--colmap",
                cfg["colmap_bin"],
                "--map-frames",
                map_frames,
                "--query-frames",
                query_frames,
                "--out-dir",
                colmap_out,
                "--max-image-size",
                cfg["max_image_size"],
                "--map-camera-model",
                cfg["map_camera_model"],
                "--query-camera-model",
                cfg["query_camera_model"],
            ]
        )

    run(
        [
            py,
            scripts / "export_tsolve_inputs_from_colmap.py",
            "--localized-model-text",
            colmap_out / "localized_model_text",
            "--query-frames",
            query_frames,
            "--out-dir",
            tsolve_inputs,
            "--min-points",
            cfg["min_query_correspondences"],
            "--max-points",
            cfg["max_query_correspondences"],
        ]
    )

    run(
        [
            py,
            scripts / "setup_tsolve_runtime.py",
            "--base-yam-code-dir",
            cfg["base_yam_code_dir"],
            "--dropin-patch-dir",
            cfg["dropin_patch_dir"],
            "--base-harness-dir",
            cfg["base_harness_dir"],
            "--out-dir",
            runtime_dir,
        ]
    )

    run(
        [
            py,
            scripts / "run_tsolve_replay.py",
            "--python",
            py,
            "--runtime-dir",
            runtime_dir,
            "--solver-dir",
            cfg["solver_dir"],
            "--inputs-dir",
            tsolve_inputs,
            "--out-dir",
            tsolve_runtime,
            "--count",
            cfg["tsolve_count"],
            "--train-count",
            cfg["tsolve_train_count"],
            "--prime",
            cfg["tsolve_prime"],
            "--degree",
            cfg["tsolve_degree"],
            "--action-weights",
            cfg["tsolve_action_weights"],
            "--fallback-action-weights",
            cfg["tsolve_fallback_action_weights"],
        ]
    )

    run(
        [
            py,
            scripts / "build_viewer_data.py",
            "--localized-model-text",
            colmap_out / "localized_model_text",
            "--tsolve-runtime-dir",
            tsolve_runtime,
            "--drone-video",
            cfg["drone_query_video"],
            "--out-public",
            viewer_public,
        ]
    )

    print("\nPhase-1 replay demo is ready.")
    print("Viewer:")
    print(f"  cd {root}")
    print("  python3 scripts/serve_viewer.py")
    print("  open http://127.0.0.1:8765")


if __name__ == "__main__":
    main()
