#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path

import numpy as np

from colmap_io import camera_center, qvec_to_rotmat, read_images_text, read_points3d_text


PLY_SCALAR_TYPES = {
    "char": ("b", 1),
    "int8": ("b", 1),
    "uchar": ("B", 1),
    "uint8": ("B", 1),
    "short": ("h", 2),
    "int16": ("h", 2),
    "ushort": ("H", 2),
    "uint16": ("H", 2),
    "int": ("i", 4),
    "int32": ("i", 4),
    "uint": ("I", 4),
    "uint32": ("I", 4),
    "float": ("f", 4),
    "float32": ("f", 4),
    "double": ("d", 8),
    "float64": ("d", 8),
}


def read_dense_ply(path: Path | None, limit: int) -> list[dict]:
    if path is None or not path.exists():
        return []
    with path.open("rb") as f:
        first = f.readline().decode("ascii", errors="replace").strip()
        if first != "ply":
            raise RuntimeError(f"Not a PLY file: {path}")
        header: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError(f"Unexpected EOF in PLY header: {path}")
            text = line.decode("ascii", errors="replace").strip()
            header.append(text)
            if text == "end_header":
                break

        fmt = "ascii"
        vertex_count = 0
        vertex_props: list[tuple[str, str]] = []
        current_element = None
        for text in header:
            parts = text.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and current_element == "vertex":
                if len(parts) >= 3 and parts[1] != "list":
                    vertex_props.append((parts[1], parts[2]))

        if not vertex_count or not vertex_props:
            return []
        prop_names = [name for _, name in vertex_props]
        try:
            x_idx = prop_names.index("x")
            y_idx = prop_names.index("y")
            z_idx = prop_names.index("z")
        except ValueError as exc:
            raise RuntimeError(f"PLY vertex properties must include x,y,z: {path}") from exc
        rgb_idx = []
        for name in ("red", "green", "blue"):
            rgb_idx.append(prop_names.index(name) if name in prop_names else None)

        stride = max(1, int(np.ceil(vertex_count / max(limit, 1)))) if limit else 1
        rows: list[dict] = []
        if fmt == "ascii":
            for i in range(vertex_count):
                line = f.readline()
                if not line:
                    break
                if i % stride:
                    continue
                values = line.decode("ascii", errors="replace").split()
                if len(values) < len(vertex_props):
                    continue
                xyz = [float(values[x_idx]), float(values[y_idx]), float(values[z_idx])]
                if all(idx is not None for idx in rgb_idx):
                    rgb = [int(float(values[idx])) for idx in rgb_idx]
                else:
                    rgb = [118, 220, 255]
                rows.append({"id": i, "xyz": xyz, "rgb": rgb, "dense": True})
        elif fmt in {"binary_little_endian", "binary_big_endian"}:
            endian = "<" if fmt == "binary_little_endian" else ">"
            try:
                fmt_chars = [PLY_SCALAR_TYPES[t][0] for t, _ in vertex_props]
            except KeyError as exc:
                raise RuntimeError(f"Unsupported PLY scalar type {exc.args[0]} in {path}") from exc
            record_fmt = endian + "".join(fmt_chars)
            record_size = struct.calcsize(record_fmt)
            for i in range(vertex_count):
                data = f.read(record_size)
                if len(data) < record_size:
                    break
                if i % stride:
                    continue
                values = struct.unpack(record_fmt, data)
                xyz = [float(values[x_idx]), float(values[y_idx]), float(values[z_idx])]
                if all(idx is not None for idx in rgb_idx):
                    rgb = [int(values[idx]) for idx in rgb_idx]
                else:
                    rgb = [118, 220, 255]
                rows.append({"id": i, "xyz": xyz, "rgb": rgb, "dense": True})
        else:
            raise RuntimeError(f"Unsupported PLY format {fmt}: {path}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a COLMAP sparse map to the ATLAS viewer format.")
    ap.add_argument("--model-text", required=True, type=Path)
    ap.add_argument("--out-public", required=True, type=Path)
    ap.add_argument("--preserve-media", action="store_true")
    ap.add_argument("--dense-points", type=Path, default=None)
    ap.add_argument("--dense-point-limit", type=int, default=180000)
    args = ap.parse_args()

    out = args.out_public
    old_media = out / "media"
    media_backup = None
    if args.preserve_media and old_media.exists():
        media_backup = out.parent / "_media_backup_for_map_only"
        if media_backup.exists():
            shutil.rmtree(media_backup)
        shutil.copytree(old_media, media_backup)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    if media_backup is not None:
        shutil.copytree(media_backup, out / "media")
        shutil.rmtree(media_backup)

    points = read_points3d_text(args.model_text / "points3D.txt")
    images = read_images_text(args.model_text / "images.txt")

    point_rows = [
        {"id": int(pid), "xyz": pt.xyz.tolist(), "rgb": list(pt.rgb), "error": pt.error}
        for pid, pt in sorted(points.items())
    ]
    dense_rows = read_dense_ply(args.dense_points, args.dense_point_limit)
    map_cameras = []
    for im in images.values():
        R = qvec_to_rotmat(im.qvec)
        C = camera_center(im)
        map_cameras.append(
            {
                "image_name": im.name,
                "R": R.tolist(),
                "t": im.tvec.tolist(),
                "center": C.tolist(),
                "registered_points": int(np.sum(im.point3d_ids >= 0)),
            }
        )

    scene = {
        "points3D": point_rows,
        "map_cameras": map_cameras,
        "coordinate_note": "COLMAP map-only world coordinates from live webcam capture.",
        "source": str(args.model_text),
    }
    if dense_rows:
        scene["dense_points3D"] = dense_rows
        scene["dense_source"] = str(args.dense_points)
    (out / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
    (out / "poses.json").write_text(json.dumps({"poses": []}, indent=2), encoding="utf-8")
    print(json.dumps({"points": len(point_rows), "dense_points": len(dense_rows), "map_cameras": len(map_cameras), "poses": 0, "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
