#!/usr/bin/env python3
"""Run ATLAS/TSolve localization on a DJI live frame bank.

Use this after or while `atlas_dji_live_bridge.py` has written frames to:

    data/dji_live/<session>/query_frames/

This script uses the same bounded localizer used by the uploaded-video replay:
first accepted frame is globally registered to the selected COLMAP map, then
later frames are tracked by optical flow and solved by TSolve.

It writes a replay asset back into the selected ATLAS map.  It does not send any
drone movement commands.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import atlas_app_server as atlas  # noqa: E402


def image_count(path: Path) -> int:
    return sum(
        1
        for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def copy_final_partial_to_poses(partial_path: Path, final_path: Path) -> int:
    payload = json.loads(partial_path.read_text(encoding="utf-8"))
    payload["mode"] = "dji_live_tsolve_replay"
    payload["description"] = "TSolve R,t estimates produced from DJI MSDK live frames."
    poses = payload.get("poses") or []
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return len(poses)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Localize frames captured by atlas_dji_live_bridge.py against an ATLAS map."
    )
    ap.add_argument("--query-frames", required=True, type=Path)
    ap.add_argument("--map-id", default="", help="ATLAS map id. Default: current selected map.")
    ap.add_argument("--title", default="", help="Replay title shown in ATLAS.")
    ap.add_argument("--max-frames", type=int, default=0, help="Optionally copy only first N frames.")
    args = ap.parse_args()

    query_frames_src = args.query_frames.resolve()
    if not query_frames_src.exists():
        raise FileNotFoundError(query_frames_src)
    if not (query_frames_src / "frames.csv").exists():
        raise FileNotFoundError(query_frames_src / "frames.csv")
    if image_count(query_frames_src) == 0:
        raise RuntimeError(f"No frames found in {query_frames_src}")

    cfg = atlas.load_config()
    py = Path(cfg["python"])
    selected = atlas.set_selected_map(args.map_id) if args.map_id else atlas.selected_map_entry()
    full_map_frames = atlas.frames_for_entry(selected)
    map_artifacts = atlas.colmap_artifacts_for_entry(selected)
    faiss_spec = atlas.faiss_index_command(cfg, map_artifacts)
    if faiss_spec is None:
        raise RuntimeError("Direct OpenCV SIFT localization requires the Faiss map index.")
    faiss_index_dir, faiss_build_cmd = faiss_spec

    replay_id = atlas.make_map_id("dji_live")
    replay_title = args.title.strip() or f"DJI Live Path {time.strftime('%H:%M:%S')}"
    base_asset_dir = atlas.VIEWER / selected["asset_base"]
    if not base_asset_dir.exists():
        base_asset_dir = atlas.MAPS_DIR / selected["id"]
    out_asset_dir = base_asset_dir / "replays" / replay_id
    run_root = ROOT / "results" / "dji_live_runs" / selected["id"] / replay_id
    query_frames = run_root / "query_frames"
    tsolve_inputs = run_root / "tsolve_inputs"
    runtime_dir = run_root / "tsolve_runtime_code"
    tsolve_runtime = run_root / "tsolve_runtime"
    stream_work = run_root / "live_existing_map_stream"
    partial_pose_path = out_asset_dir / "poses_partial.json"

    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    out_asset_dir.mkdir(parents=True, exist_ok=True)

    if args.max_frames > 0:
        query_frames.mkdir(parents=True, exist_ok=True)
        images = sorted(
            [
                p
                for p in query_frames_src.iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
        )[: args.max_frames]
        for img in images:
            shutil.copy2(img, query_frames / img.name)
        shutil.copy2(query_frames_src / "frames.csv", query_frames / "frames.csv")
    else:
        query_frames = query_frames_src

    expected_count = image_count(query_frames)
    print("ATLAS DJI live frame-bank localizer")
    print(f"  selected map: {selected['title']} ({selected['id']})")
    print(f"  query frames: {query_frames}")
    print(f"  expected:     {expected_count}")
    print(f"  replay:       {replay_title}")
    print("  control:      disabled\n")

    atlas.atomic_write_json(
        partial_pose_path,
        {
            "mode": "dji_live_tsolve_partial",
            "replay_id": replay_id,
            "frame_source": str(query_frames),
            "expected_count": expected_count,
            "processed_count": 0,
            "complete": False,
            "updated_at": time.time(),
            "poses": [],
        },
    )

    atlas.set_job("drone", "running", f"Preparing TSolve runtime for DJI live frame bank: {expected_count} frames.")
    atlas.run_cmd(
        "drone",
        [
            py,
            SCRIPTS / "setup_tsolve_runtime.py",
            "--base-yam-code-dir",
            cfg["base_yam_code_dir"],
            "--dropin-patch-dir",
            cfg["dropin_patch_dir"],
            "--base-harness-dir",
            cfg["base_harness_dir"],
            "--out-dir",
            runtime_dir,
        ],
        atlas.DRONE_STOP_EVENT,
    )

    atlas.set_job("drone", "running", "Preparing the persistent all-map SIFT/Faiss index.")
    atlas.run_cmd("drone", faiss_build_cmd, atlas.DRONE_STOP_EVENT)

    atlas.set_job("drone", "running", "Running bounded TSolve localization on DJI live frames.")
    localize_command: list[object] = [
            py,
            SCRIPTS / "run_bounded_tsolve_video_stream.py",
            "--colmap",
            cfg["colmap_bin"],
            "--map-database",
            map_artifacts["database"],
            "--map-images",
            map_artifacts["images"],
            "--map-sparse-model",
            map_artifacts["sparse_model"],
            "--map-sparse-text",
            map_artifacts["sparse_text"],
            "--query-frames",
            query_frames,
            "--runtime-dir",
            runtime_dir,
            "--solver-dir",
            cfg["solver_dir"],
            "--inputs-out-dir",
            tsolve_inputs,
            "--out-dir",
            tsolve_runtime,
            "--work-dir",
            stream_work,
            "--max-image-size",
            cfg["max_image_size"],
            "--query-camera-model",
            cfg["query_camera_model"],
            "--query-camera-params",
            cfg.get("query_camera_params", ""),
            "--sift-max-num-features",
            cfg.get("live_sift_max_num_features", 1024),
            "--min-points",
            cfg["min_query_correspondences"],
            "--max-points",
            cfg["max_query_correspondences"],
            "--max-reference-images",
            cfg.get("live_reference_image_cap", 48),
            "--tracking-reference-images",
            cfg.get("live_tracking_reference_image_cap", 10),
            "--track-pool-size",
            cfg.get("live_tracking_pool_size", 900),
            "--relocalize-every",
            cfg.get("live_relocalize_every", 0),
            "--flow-max-error",
            cfg.get("live_flow_max_error", 34.0),
            "--flow-backtrack-error",
            cfg.get("live_flow_backtrack_error", 2.5),
            "--partial-pose-out",
            partial_pose_path,
            "--replay-id",
            replay_id,
            "--expected-count",
            expected_count,
            "--prime",
            cfg["tsolve_prime"],
            "--degree",
            cfg["tsolve_degree"],
            "--action-weights",
            cfg["tsolve_action_weights"],
            "--fallback-action-weights",
            cfg["tsolve_fallback_action_weights"],
            "--scene-json",
            base_asset_dir / "scene.json",
            "--display-z-sign",
            selected.get("display_z_sign", -1),
        ]
    atlas.add_faiss_live_arguments(localize_command, cfg, faiss_index_dir)
    atlas.run_cmd("drone", localize_command, atlas.DRONE_STOP_EVENT)

    final_pose_path = out_asset_dir / "poses.json"
    pose_count = copy_final_partial_to_poses(partial_pose_path, final_pose_path)
    replay = {
        "id": replay_id,
        "title": replay_title,
        "asset_base": atlas.public_rel(out_asset_dir),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_video": "DJI MSDK live frame bank",
        "counts": {"poses": pose_count},
    }
    atlas.add_replay_to_map(selected["id"], replay, select=True)
    atlas.set_job("drone", "done", f"DJI live TSolve path ready: {replay_title} ({pose_count} poses).")
    print(json.dumps({"ok": True, "replay": replay, "poses": pose_count, "out": str(out_asset_dir)}, indent=2))


if __name__ == "__main__":
    main()
