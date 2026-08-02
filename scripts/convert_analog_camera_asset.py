#!/usr/bin/env python3
"""Convert the supplied analog-camera OBJ into a compact ATLAS-colored GLB."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from convert_camera_path_lab_mesh import write_glb


ROOT = Path(__file__).resolve().parents[1]
MATERIAL_COLORS = {
    "Cuerpo": (24, 38, 47),
    "Lente": (52, 190, 230),
    "Accesorios": (181, 207, 216),
}


def obj_index(raw: str, length: int) -> int:
    value = int(raw)
    return value - 1 if value > 0 else length + value


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_vertices: list[tuple[float, float, float]] = []
    source_normals: list[tuple[float, float, float]] = []
    positions: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    colors: list[tuple[int, int, int]] = []
    indices: list[int] = []
    vertex_lookup: dict[tuple[int, int, str], int] = {}
    material = "Cuerpo"

    with path.open(encoding="utf-8", errors="strict") as handle:
        for raw_line in handle:
            parts = raw_line.split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                source_vertices.append(tuple(float(value) for value in parts[1:4]))
            elif parts[0] == "vn" and len(parts) >= 4:
                source_normals.append(tuple(float(value) for value in parts[1:4]))
            elif parts[0] == "usemtl" and len(parts) >= 2:
                material = " ".join(parts[1:])
            elif parts[0] == "f" and len(parts) >= 4:
                corners = parts[1:]
                for triangle_index in range(1, len(corners) - 1):
                    for token in (corners[0], corners[triangle_index], corners[triangle_index + 1]):
                        values = token.split("/")
                        vertex_index = obj_index(values[0], len(source_vertices))
                        normal_index = -1
                        normal = (0.0, 1.0, 0.0)
                        if len(values) >= 3 and values[2]:
                            normal_index = obj_index(values[2], len(source_normals))
                            normal = source_normals[normal_index]
                        key = (vertex_index, normal_index, material)
                        output_index = vertex_lookup.get(key)
                        if output_index is None:
                            output_index = len(positions)
                            vertex_lookup[key] = output_index
                            positions.append(source_vertices[vertex_index])
                            normals.append(normal)
                            colors.append(MATERIAL_COLORS.get(material, MATERIAL_COLORS["Accesorios"]))
                        indices.append(output_index)

    if not positions:
        raise RuntimeError(f"OBJ contains no usable faces: {path}")
    position_array = np.asarray(positions, dtype=np.float32)
    normal_array = np.asarray(normals, dtype=np.float32)
    normal_lengths = np.linalg.norm(normal_array, axis=1)
    normal_array[normal_lengths > 1e-8] /= normal_lengths[normal_lengths > 1e-8, None]
    return (
        position_array,
        normal_array,
        np.asarray(indices, dtype=np.uint32),
        np.asarray(colors, dtype=np.uint8),
    )


def center_and_scale(positions: np.ndarray, target_width: float) -> np.ndarray:
    minimum = positions.min(axis=0)
    maximum = positions.max(axis=0)
    width = float(maximum[0] - minimum[0])
    if not np.isfinite(width) or width <= 1e-9:
        raise RuntimeError("Camera OBJ has an invalid width.")
    center = (minimum + maximum) * 0.5
    return np.ascontiguousarray((positions - center) * (target_width / width), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "viewer" / "public" / "camera_path_lab" / "analog_camera.glb",
    )
    parser.add_argument("--width", type=float, default=0.42)
    args = parser.parse_args()
    if args.width <= 0:
        raise SystemExit("--width must be positive")
    positions, normals, indices, colors = read_obj(args.input)
    positions = center_and_scale(positions, args.width)
    write_glb(args.output, positions, normals, indices, colors)
    print(
        f"Analog camera GLB ready: {len(positions):,} vertices, "
        f"{len(indices) // 3:,} faces, {args.output.stat().st_size / 1_048_576:.1f} MiB"
    )


if __name__ == "__main__":
    main()
