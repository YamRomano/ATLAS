#!/usr/bin/env python3
"""Build a compact, read-only display mesh from an existing ATLAS map.

The production COLMAP model is never modified.  The output is a voxel-surface
asset used only by the standalone Camera Path Lab page.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MAP_MANIFEST = ROOT / "viewer" / "public" / "maps" / "manifest.json"


def map_entry(map_id: str) -> dict:
    payload = json.loads(MAP_MANIFEST.read_text(encoding="utf-8"))
    for entry in payload.get("maps", []):
        if str(entry.get("id")) == map_id:
            return entry
    raise RuntimeError(f"Unknown ATLAS map id: {map_id}")


def room_matrix(entry: dict) -> np.ndarray:
    matrix = (entry.get("room_alignment") or {}).get("matrix")
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (3, 4) or not np.isfinite(array).all():
        raise RuntimeError("Camera Path Lab requires a validated 3x4 room alignment matrix.")
    return array


def scene_path(entry: dict) -> Path:
    asset_base = str(entry.get("asset_base") or "").strip()
    if not asset_base:
        raise RuntimeError("The selected map has no viewer asset directory.")
    path = ROOT / "viewer" / asset_base / "scene.json"
    if not path.is_file():
        raise RuntimeError(f"Map scene is missing: {path}")
    return path


def compact(value: float) -> float:
    return round(float(value), 4)


def build_asset(entry: dict, voxel_size: float, min_samples: int, max_voxels: int) -> dict:
    scene = json.loads(scene_path(entry).read_text(encoding="utf-8"))
    rows = scene.get("dense_points3D") or scene.get("points3D") or []
    if not rows:
        raise RuntimeError("The selected map contains no 3D points.")

    xyz = np.asarray([row["xyz"] for row in rows], dtype=np.float64)
    rgb = np.asarray([row.get("rgb", [118, 148, 166]) for row in rows], dtype=np.float64)
    usable = np.isfinite(xyz).all(axis=1) & (np.abs(xyz) < 1_000_000.0).all(axis=1)
    xyz = xyz[usable]
    rgb = rgb[usable]
    rgb = np.nan_to_num(rgb, nan=128.0, posinf=255.0, neginf=0.0)
    matrix = room_matrix(entry)
    room_xyz = np.column_stack(
        [np.sum(xyz * matrix[axis, :3], axis=1) + matrix[axis, 3] for axis in range(3)]
    )

    low = np.asarray(
        [np.quantile(room_xyz[:, 0], 0.01), np.quantile(room_xyz[:, 1], 0.02), np.quantile(room_xyz[:, 2], 0.01)]
    )
    high = np.asarray(
        [np.quantile(room_xyz[:, 0], 0.99), np.quantile(room_xyz[:, 1], 0.98), np.quantile(room_xyz[:, 2], 0.99)]
    )
    margin = np.maximum((high - low) * np.asarray([0.06, 0.10, 0.06]), voxel_size)
    keep = np.logical_and(room_xyz >= low - margin, room_xyz <= high + margin).all(axis=1)
    room_xyz = room_xyz[keep]
    rgb = rgb[keep]

    origin = low - margin
    cells = np.floor((room_xyz - origin) / voxel_size).astype(np.int32)
    unique_cells, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    voxel_count = len(unique_cells)
    sums = np.column_stack(
        [np.bincount(inverse, weights=room_xyz[:, index], minlength=voxel_count) for index in range(3)]
    )
    color_sums = np.column_stack(
        [np.bincount(inverse, weights=rgb[:, index], minlength=voxel_count) for index in range(3)]
    )
    centers = sums / counts[:, None]
    colors = np.clip(np.rint(color_sums / counts[:, None]), 0, 255).astype(np.uint8)

    selected = np.flatnonzero(counts >= max(1, min_samples))
    if len(selected) > max_voxels:
        order = np.argsort(counts[selected], kind="stable")[::-1]
        selected = selected[order[:max_voxels]]
    selected = selected[np.lexsort((centers[selected, 2], centers[selected, 1], centers[selected, 0]))]

    voxels = [
        [
            compact(centers[index, 0]),
            compact(centers[index, 1]),
            compact(centers[index, 2]),
            int(colors[index, 0]),
            int(colors[index, 1]),
            int(colors[index, 2]),
            int(counts[index]),
        ]
        for index in selected
    ]
    return {
        "format": "atlas-camera-path-lab-voxel-mesh-v1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_map_id": entry["id"],
        "source_map_title": entry.get("title") or entry["id"],
        "read_only_source": True,
        "source_points": int(len(rows)),
        "display_voxels": len(voxels),
        "voxel_size": float(voxel_size),
        "bounds": {
            "min": [compact(value) for value in np.min(centers[selected], axis=0)],
            "max": [compact(value) for value in np.max(centers[selected], axis=0)],
        },
        "room_alignment": entry.get("room_alignment"),
        "safety_barriers": entry.get("safety_barriers") or [],
        "voxels": voxels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-id", required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "viewer" / "public" / "camera_path_lab" / "good_copy_mesh.json",
    )
    parser.add_argument("--voxel-size", type=float, default=0.105)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--max-voxels", type=int, default=52000)
    args = parser.parse_args()
    if args.voxel_size <= 0:
        raise SystemExit("--voxel-size must be positive")
    if args.max_voxels < 1000:
        raise SystemExit("--max-voxels must be at least 1000")

    output = build_asset(map_entry(args.map_id), args.voxel_size, args.min_samples, args.max_voxels)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_name(f".{args.out.name}.{time.time_ns()}.tmp")
    temp.write_text(json.dumps(output, separators=(",", ":")), encoding="utf-8")
    temp.replace(args.out)
    print(
        f"Camera Path Lab mesh ready: {output['display_voxels']} voxels from "
        f"{output['source_points']} points -> {args.out}"
    )


if __name__ == "__main__":
    main()
