#!/usr/bin/env python3
"""Compare a saved patrol camera sequence with the current patrol geometry.

This is read-only.  It audits registered COLMAP cameras, room-frame patrol
targets, and the image assets packaged for live relocalization.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colmap_io import qvec_to_rotmat, read_images_text  # noqa: E402
from run_bounded_tsolve_video_stream import build_room_transform  # noqa: E402


def horizontal_distance(a: list[float], b: list[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[2]) - float(b[2]))


def frame_sort_key(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)(?=\.[^.]+$)", name)
    return (int(match.group(1)) if match else sys.maxsize, name)


def load_patrol(manifest_path: Path, map_id: str, patrol_id: str) -> tuple[dict[str, Any], list[list[float]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next((item for item in manifest.get("maps", []) if item.get("id") == map_id), None)
    if not isinstance(entry, dict):
        raise RuntimeError(f"Map not found: {map_id}")
    patrol = next((item for item in entry.get("patrols", []) if item.get("id") == patrol_id), None)
    if not isinstance(patrol, dict):
        raise RuntimeError(f"Patrol not found: {patrol_id}")
    points: list[list[float]] = []
    for item in patrol.get("points", []):
        try:
            value = item["rxyz"]
            point = [float(value[0]), float(value[1]), float(value[2])]
        except (KeyError, IndexError, TypeError, ValueError):
            raise RuntimeError("Patrol has an invalid room-frame point.") from None
        points.append(point)
    if len(points) < 2:
        raise RuntimeError("Patrol needs at least two points.")
    return patrol, points


def load_map_entry(manifest_path: Path, map_id: str) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next((item for item in manifest.get("maps", []) if item.get("id") == map_id), None)
    if not isinstance(entry, dict):
        raise RuntimeError(f"Map not found: {map_id}")
    return entry


def explicit_room_transform(entry: dict[str, Any]):
    matrix = (entry.get("room_alignment") or {}).get("matrix")
    if not isinstance(matrix, list) or len(matrix) != 3:
        return None
    try:
        rows = [[float(value) for value in row] for row in matrix]
    except (TypeError, ValueError):
        return None
    if any(len(row) != 4 or not all(math.isfinite(value) for value in row) for row in rows):
        return None

    def transform(xyz: list[float]) -> list[float]:
        return [sum(row[index] * float(xyz[index]) for index in range(3)) + row[3] for row in rows]

    return transform


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("viewer/public/maps/manifest.json"))
    parser.add_argument("--map-id", required=True, help="Map containing the current patrol definition.")
    parser.add_argument(
        "--alignment-map-id",
        help="Map providing the room alignment; defaults to --map-id.",
    )
    parser.add_argument("--patrol-id", required=True)
    parser.add_argument("--sparse-text", type=Path, required=True)
    parser.add_argument("--scene-json", type=Path, required=True)
    parser.add_argument("--display-z-sign", type=float, default=1.0)
    parser.add_argument("--sequence-prefix", default="enhancement/manual_patrol_")
    parser.add_argument("--map-images", type=Path)
    parser.add_argument("--arrival-radius", type=float, default=0.24)
    parser.add_argument("--soft-radius", type=float, default=0.38)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    patrol, points = load_patrol(args.manifest, args.map_id, args.patrol_id)
    alignment_map_id = args.alignment_map_id or args.map_id
    alignment_entry = load_map_entry(args.manifest, alignment_map_id)
    transform = explicit_room_transform(alignment_entry)
    transform_source = "explicit_room_alignment" if transform is not None else "scene_pca"
    if transform is None:
        transform = build_room_transform(args.scene_json, args.display_z_sign)
    if transform is None:
        raise RuntimeError(f"Could not build room transform from {args.scene_json}")

    images = read_images_text(args.sparse_text / "images.txt")
    cameras: list[dict[str, Any]] = []
    packaged = 0
    missing_examples: list[str] = []
    for image in images.values():
        center = -(qvec_to_rotmat(image.qvec).T @ image.tvec)
        room_center = transform(center.tolist())
        if room_center is None:
            continue
        exists = None
        if args.map_images is not None:
            exists = (args.map_images / image.name).is_file()
            if exists:
                packaged += 1
            elif len(missing_examples) < 30:
                missing_examples.append(image.name)
        cameras.append(
            {
                "name": image.name,
                "rcenter": room_center,
                "registered_points": int(np.sum(image.point3d_ids >= 0)),
                "asset_packaged": exists,
            }
        )

    sequence = sorted(
        [item for item in cameras if item["name"].startswith(args.sequence_prefix)],
        key=lambda item: frame_sort_key(item["name"]),
    )
    nearest: list[dict[str, Any]] = []
    for point_index, point in enumerate(points):
        best = min(sequence, key=lambda item: horizontal_distance(item["rcenter"], point))
        distance = horizontal_distance(best["rcenter"], point)
        nearest.append(
            {
                "point": point_index + 1,
                "distance": distance,
                "arrival": (
                    "strict" if distance <= args.arrival_radius
                    else "soft" if distance <= args.soft_radius
                    else "miss"
                ),
                "camera": best["name"],
                "rcenter": best["rcenter"],
                "registered_points": best["registered_points"],
            }
        )

    ordered_targets = points + [points[0]]
    ordered_hits: list[dict[str, Any]] = []
    cursor = 0
    for target_index, target in enumerate(ordered_targets):
        hit = next(
            (
                index
                for index in range(cursor, len(sequence))
                if horizontal_distance(sequence[index]["rcenter"], target) <= args.soft_radius
            ),
            None,
        )
        if hit is None:
            break
        item = sequence[hit]
        ordered_hits.append(
            {
                "point": (target_index % len(points)) + 1,
                "camera": item["name"],
                "distance": horizontal_distance(item["rcenter"], target),
                "sequence_index": hit,
            }
        )
        cursor = hit + 1

    failures: list[str] = []
    misses = [item for item in nearest if item["arrival"] == "miss"]
    if misses:
        failures.append(
            "saved correct-order patrol misses current soft arrival radius at points "
            + ", ".join(str(item["point"]) for item in misses)
        )
    if len(ordered_hits) != len(ordered_targets):
        failures.append(
            f"no complete ordered loop at {args.soft_radius:.2f}; next missing point is "
            f"{(len(ordered_hits) % len(points)) + 1}"
        )
    if args.map_images is not None and packaged != len(cameras):
        failures.append(
            f"map image package contains {packaged}/{len(cameras)} registered camera assets"
        )

    report = {
        "kind": "atlas_patrol_map_camera_coverage_audit",
        "safe": not failures,
        "map_id": args.map_id,
        "alignment_map_id": alignment_map_id,
        "transform_source": transform_source,
        "patrol_id": args.patrol_id,
        "patrol_title": patrol.get("title"),
        "sequence_prefix": args.sequence_prefix,
        "thresholds": {
            "arrival_radius": args.arrival_radius,
            "soft_radius": args.soft_radius,
        },
        "registered_camera_count": len(cameras),
        "sequence_camera_count": len(sequence),
        "packaged_camera_assets": (
            {
                "present": packaged,
                "required": len(cameras),
                "missing": len(cameras) - packaged,
                "missing_examples": missing_examples,
            }
            if args.map_images is not None
            else None
        ),
        "nearest_points": nearest,
        "ordered_loop": {
            "complete": len(ordered_hits) == len(ordered_targets),
            "required_hits": len(ordered_targets),
            "hits": ordered_hits,
        },
        "failures": failures,
    }
    rendered = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if report["safe"] else 2)


if __name__ == "__main__":
    main()
