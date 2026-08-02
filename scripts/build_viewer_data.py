#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np

from colmap_io import camera_center, read_images_text, read_points3d_text, qvec_to_rotmat


def instance_sort_key(instance_id: str) -> tuple[str, int, str]:
    prefix, separator, suffix = instance_id.rpartition("_")
    if separator and suffix.isdigit():
        return prefix, int(suffix), instance_id
    return instance_id, -1, instance_id


def load_static_results(static_dir: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(static_dir.glob("*.json"), key=lambda path: instance_sort_key(path.stem)):
        out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    return out


def load_instance_meta(instances_dir: Path, instance_id: str) -> dict:
    p = instances_dir / instance_id / "input.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def load_live_summary(tsolve_runtime_dir: Path) -> dict:
    summary_path = tsolve_runtime_dir / "live_stream_summary.json"
    if not summary_path.is_file():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def load_output_rejected_case_ids(tsolve_runtime_dir: Path) -> set[str]:
    summary = load_live_summary(tsolve_runtime_dir)
    return {
        str(item["case_id"])
        for item in summary.get("output_rejected", [])
        if isinstance(item, dict) and item.get("case_id")
    }


def colmap_reference_from_meta(meta: dict) -> dict | None:
    qvec = meta.get("colmap_qvec_world_to_camera")
    tvec = meta.get("colmap_tvec_world_to_camera")
    image_name = meta.get("image_name")
    if qvec is None or tvec is None:
        return None
    try:
        R = qvec_to_rotmat(np.asarray(qvec, dtype=float))
        t = np.asarray(tvec, dtype=float).reshape(3)
        center = -R.T @ t
    except Exception:
        return None
    return {
        "image_name": image_name,
        "R": R.tolist(),
        "t": t.tolist(),
        "center": center.tolist(),
        "registered_points": int(meta.get("colmap_registered_points") or meta.get("points") or 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--localized-model-text", required=True, type=Path)
    ap.add_argument("--tsolve-runtime-dir", required=True, type=Path)
    ap.add_argument("--drone-video", required=True, type=Path)
    ap.add_argument("--out-public", required=True, type=Path)
    args = ap.parse_args()

    out = args.out_public
    if out.exists():
        shutil.rmtree(out)
    (out / "media").mkdir(parents=True, exist_ok=True)

    points = read_points3d_text(args.localized_model_text / "points3D.txt")
    images = read_images_text(args.localized_model_text / "images.txt")
    point_rows = [
        {"id": int(pid), "xyz": pt.xyz.tolist(), "rgb": list(pt.rgb), "error": pt.error}
        for pid, pt in sorted(points.items())
    ]
    map_cameras = []
    colmap_query_pose_by_name = {}
    for im in images.values():
        R = qvec_to_rotmat(im.qvec)
        C = camera_center(im)
        row = {
            "image_name": im.name,
            "R": R.tolist(),
            "t": im.tvec.tolist(),
            "center": C.tolist(),
            "registered_points": int(np.sum(im.point3d_ids >= 0)),
        }
        if im.name.startswith("query/"):
            colmap_query_pose_by_name[im.name] = row
        else:
            map_cameras.append(row)

    static_dir = args.tsolve_runtime_dir / "persistent_static_json"
    instances_dir = args.tsolve_runtime_dir / "instances_all"
    results = load_static_results(static_dir)
    live_summary = load_live_summary(args.tsolve_runtime_dir)
    output_rejected_case_ids = load_output_rejected_case_ids(args.tsolve_runtime_dir)
    poses = []
    for instance_id, result in sorted(results.items(), key=lambda item: instance_sort_key(item[0])):
        # The live localizer may solve a mathematically valid pose that its
        # temporal/objective guard rejects.  Such a result remains on disk for
        # diagnostics, but it must never reappear in the final viewer path.
        if instance_id in output_rejected_case_ids:
            continue
        meta = load_instance_meta(instances_dir, instance_id)
        R = np.asarray(result.get("R"), dtype=float) if result.get("R") is not None else None
        t = np.asarray(result.get("t"), dtype=float).reshape(3) if result.get("t") is not None else None
        center = (-R.T @ t).tolist() if R is not None and t is not None else None
        poses.append(
            {
                "instance_id": instance_id,
                "success": bool(result.get("success")),
                "time_sec": meta.get("time_sec"),
                "image_name": meta.get("image_name"),
                "R": None if R is None else R.tolist(),
                "t": None if t is None else t.tolist(),
                "center": center,
                "objective": result.get("objective"),
                "total_ms": result.get("total_ms"),
                "stages_ms": result.get("stages_ms", {}),
                "colmap_reference": colmap_query_pose_by_name.get(str(meta.get("image_name")))
                or colmap_reference_from_meta(meta),
            }
        )

    scene = {
        "points3D": point_rows,
        "map_cameras": map_cameras,
        "coordinate_note": "COLMAP/TSolve world coordinates. Camera center is -R^T t.",
    }
    (out / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
    pose_payload = {
        "mode": "simulated_live_tsolve_replay",
        "frame_source": str(args.drone_video),
        "description": "Timestamped TSolve R,t estimates produced from uploaded drone-video frames.",
        "processed_count": int(live_summary.get("processed_frames") or len(results)),
        "expected_count": int(live_summary.get("query_frames") or len(results)),
        "accepted_count": len(poses),
        "output_rejected_count": len(output_rejected_case_ids),
        "complete": True,
        "poses": poses,
    }
    (out / "poses.json").write_text(json.dumps(pose_payload, indent=2), encoding="utf-8")

    source_video = args.drone_video.resolve()
    video_dst = out / "media" / "drone_query.mp4"
    try:
        os.symlink(source_video, video_dst)
    except FileExistsError:
        pass
    except OSError:
        shutil.copy2(source_video, video_dst)

    print(json.dumps({"points": len(point_rows), "map_cameras": len(map_cameras), "poses": len(poses), "output_rejected": len(output_rejected_case_ids), "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
