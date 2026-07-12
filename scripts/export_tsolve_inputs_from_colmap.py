#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from colmap_io import read_cameras_text, read_images_text, read_points3d_text


def farthest_spread_indices(xy: np.ndarray, count: int) -> np.ndarray:
    if len(xy) <= count:
        return np.arange(len(xy), dtype=int)
    center = xy.mean(axis=0)
    first = int(np.argmax(np.linalg.norm(xy - center, axis=1)))
    chosen = [first]
    dist = np.linalg.norm(xy - xy[first], axis=1)
    while len(chosen) < count:
        idx = int(np.argmax(dist))
        chosen.append(idx)
        dist = np.minimum(dist, np.linalg.norm(xy - xy[idx], axis=1))
    return np.array(chosen, dtype=int)


def sha256_case(K: np.ndarray, p3d: np.ndarray, p2d: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(K, dtype=np.float64).tobytes())
    h.update(np.asarray(p3d, dtype=np.float64).tobytes())
    h.update(np.asarray(p2d, dtype=np.float64).tobytes())
    return h.hexdigest()


def read_frame_times(frames_csv: Path) -> dict[str, dict[str, str]]:
    if not frames_csv.exists():
        return {}
    with frames_csv.open(newline="", encoding="utf-8") as f:
        return {row["image_name"]: row for row in csv.DictReader(f)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--localized-model-text", required=True, type=Path)
    ap.add_argument("--query-frames", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--query-prefix", default="query/")
    ap.add_argument("--min-points", type=int, default=40)
    ap.add_argument("--max-points", type=int, default=40)
    args = ap.parse_args()

    model = args.localized_model_text
    cameras = read_cameras_text(model / "cameras.txt")
    images = read_images_text(model / "images.txt")
    points = read_points3d_text(model / "points3D.txt")
    frame_times = read_frame_times(args.query_frames / "frames.csv")

    out = args.out_dir
    if out.exists():
        shutil.rmtree(out)
    inputs = out / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    rejected = []
    query_images = [im for im in images.values() if im.name.startswith(args.query_prefix)]
    query_images.sort(key=lambda im: im.name)

    for case_idx, image in enumerate(query_images):
        valid = image.point3d_ids >= 0
        valid &= np.array([int(pid) in points for pid in image.point3d_ids], dtype=bool)
        valid_idx = np.where(valid)[0]
        if len(valid_idx) < args.min_points:
            rejected.append({"image": image.name, "valid_2d3d": int(len(valid_idx)), "reason": "too_few_correspondences"})
            continue

        xy_all = image.xys[valid_idx]
        chosen_local = farthest_spread_indices(xy_all, min(args.max_points, len(valid_idx)))
        chosen_idx = valid_idx[chosen_local]

        p2d = image.xys[chosen_idx].astype(float)
        p3d = np.asarray([points[int(pid)].xyz for pid in image.point3d_ids[chosen_idx]], dtype=float)
        K = cameras[image.camera_id].K()

        raw_name = image.name.split("/", 1)[-1]
        frame_row = frame_times.get(raw_name, {})
        case_id = f"query_{len(manifest_rows):06d}"
        case_dir = inputs / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(case_dir / "p3d.csv", p3d, delimiter=",", fmt="%.12g")
        np.savetxt(case_dir / "p2d.csv", p2d, delimiter=",", fmt="%.12g")

        input_meta = {
            "K": K.tolist(),
            "image_name": image.name,
            "source_frame": frame_row.get("source_frame"),
            "time_sec": float(frame_row["time_sec"]) if frame_row.get("time_sec") else None,
            "points": int(len(chosen_idx)),
            "input_sha256": sha256_case(K, p3d, p2d),
            "colmap_image_id": image.image_id,
            "colmap_camera_id": image.camera_id,
            "colmap_qvec_world_to_camera": image.qvec.tolist(),
            "colmap_tvec_world_to_camera": image.tvec.tolist(),
        }
        (case_dir / "input.json").write_text(json.dumps(input_meta, indent=2), encoding="utf-8")

        manifest_rows.append(
            {
                "experiment": "drone_replay_colmap_localized",
                "case_id": case_id,
                "p3d_csv": f"inputs/{case_id}/p3d.csv",
                "p2d_csv": f"inputs/{case_id}/p2d.csv",
                "input_json": f"inputs/{case_id}/input.json",
                "points": int(len(chosen_idx)),
                "image_name": image.name,
                "time_sec": input_meta["time_sec"],
            }
        )

    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["experiment", "case_id", "p3d_csv", "p2d_csv", "input_json", "points", "image_name", "time_sec"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    (out / "export_summary.json").write_text(
        json.dumps(
            {
                "localized_query_images": len(query_images),
                "accepted_cases": len(manifest_rows),
                "rejected_cases": len(rejected),
                "min_points": args.min_points,
                "max_points": args.max_points,
                "rejected": rejected[:50],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"accepted_cases": len(manifest_rows), "rejected_cases": len(rejected), "out_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()
