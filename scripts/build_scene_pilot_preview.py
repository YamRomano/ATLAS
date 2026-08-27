#!/usr/bin/env python3
"""Build a bounded, colored GLB point preview from a COLMAP fused cloud."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


FUSED_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("nx", "<f4"),
        ("ny", "<f4"),
        ("nz", "<f4"),
        ("r", "u1"),
        ("g", "u1"),
        ("b", "u1"),
    ]
)


def read_header(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        vertex_count = None
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Incomplete PLY header: {path}")
            text = line.decode("ascii", errors="strict").strip()
            if text.startswith("element vertex "):
                vertex_count = int(text.split()[-1])
            if text == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"PLY has no vertex count: {path}")
        return handle.tell(), vertex_count


def quaternion_rotation(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def read_camera_centers(images_bin: Path) -> np.ndarray:
    centers: list[np.ndarray] = []
    with images_bin.open("rb") as handle:
        image_count = struct.unpack("<Q", handle.read(8))[0]
        for _ in range(image_count):
            handle.read(4)  # image id
            qvec = np.asarray(struct.unpack("<4d", handle.read(32)))
            tvec = np.asarray(struct.unpack("<3d", handle.read(24)))
            handle.read(4)  # camera id
            while handle.read(1) != b"\0":
                pass
            point_count = struct.unpack("<Q", handle.read(8))[0]
            handle.seek(point_count * 24, 1)
            centers.append(-quaternion_rotation(qvec).T @ tvec)
    if not centers:
        raise ValueError(f"No registered cameras in {images_bin}")
    return np.asarray(centers)


def padded(data: bytes, pad: bytes = b"\0") -> bytes:
    return data + pad * ((-len(data)) % 4)


def build_glb(positions: np.ndarray, colors: np.ndarray, output: Path) -> None:
    positions = np.ascontiguousarray(positions, dtype="<f4")
    colors = np.ascontiguousarray(colors, dtype="u1")
    position_blob = padded(positions.tobytes())
    color_offset = len(position_blob)
    binary_blob = padded(position_blob + colors.tobytes())

    document = {
        "asset": {"version": "2.0", "generator": "Camera Path scene-pilot preview"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "Scene pilot fused cloud"}],
        "meshes": [
            {
                "name": "Scene pilot fused cloud",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "COLOR_0": 1},
                        "mode": 0,
                    }
                ],
            }
        ],
        "buffers": [{"byteLength": len(binary_blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": positions.nbytes, "target": 34962},
            {"buffer": 0, "byteOffset": color_offset, "byteLength": colors.nbytes, "target": 34962},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(positions),
                "type": "VEC3",
                "min": positions.min(axis=0).astype(float).tolist(),
                "max": positions.max(axis=0).astype(float).tolist(),
            },
            {
                "bufferView": 1,
                "componentType": 5121,
                "count": len(colors),
                "type": "VEC3",
                "normalized": True,
            },
        ],
    }
    json_blob = padded(json.dumps(document, separators=(",", ":")).encode("utf-8"), b" ")
    total_length = 12 + 8 + len(json_blob) + 8 + len(binary_blob)
    with output.open("wb") as handle:
        handle.write(struct.pack("<4sII", b"glTF", 2, total_length))
        handle.write(struct.pack("<I4s", len(json_blob), b"JSON"))
        handle.write(json_blob)
        handle.write(struct.pack("<I4s", len(binary_blob), b"BIN\0"))
        handle.write(binary_blob)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--images-bin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-radius", type=float, default=8.0)
    parser.add_argument("--voxel-size", type=float, default=0.075)
    args = parser.parse_args()

    offset, vertex_count = read_header(args.fused)
    cloud = np.memmap(args.fused, dtype=FUSED_DTYPE, mode="r", offset=offset, shape=(vertex_count,))
    points = np.column_stack((cloud["x"], cloud["y"], cloud["z"]))
    colors = np.column_stack((cloud["r"], cloud["g"], cloud["b"]))
    centers = read_camera_centers(args.images_bin)

    keep = np.zeros(vertex_count, dtype=bool)
    radius_squared = args.camera_radius**2
    for start in range(0, vertex_count, 250_000):
        stop = min(vertex_count, start + 250_000)
        block = points[start:stop]
        nearest_squared = ((block[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).min(axis=1)
        keep[start:stop] = np.isfinite(block).all(axis=1) & (nearest_squared <= radius_squared)

    selected_points = points[keep]
    selected_colors = colors[keep]
    voxel_keys = np.floor(selected_points / args.voxel_size).astype(np.int16)
    packed_keys = np.ascontiguousarray(voxel_keys).view(
        np.dtype((np.void, voxel_keys.dtype.itemsize * voxel_keys.shape[1]))
    ).ravel()
    _, representative_indices = np.unique(packed_keys, return_index=True)
    representative_indices.sort()

    preview_points = selected_points[representative_indices]
    preview_colors = selected_colors[representative_indices]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_glb(preview_points, preview_colors, args.output)
    print(
        json.dumps(
            {
                "source_points": vertex_count,
                "bounded_points": int(keep.sum()),
                "preview_points": len(preview_points),
                "registered_cameras": len(centers),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
