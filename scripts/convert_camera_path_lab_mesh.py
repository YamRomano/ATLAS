#!/usr/bin/env python3
"""Convert a COLMAP PLY mesh into a compact, room-aligned GLB display asset."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MAP_MANIFEST = ROOT / "viewer" / "public" / "maps" / "manifest.json"
PLY_TYPES = {
    "char": ("i1", "b"),
    "uchar": ("u1", "B"),
    "int8": ("i1", "b"),
    "uint8": ("u1", "B"),
    "short": ("<i2", "h"),
    "ushort": ("<u2", "H"),
    "int16": ("<i2", "h"),
    "uint16": ("<u2", "H"),
    "int": ("<i4", "i"),
    "uint": ("<u4", "I"),
    "int32": ("<i4", "i"),
    "uint32": ("<u4", "I"),
    "float": ("<f4", "f"),
    "float32": ("<f4", "f"),
    "double": ("<f8", "d"),
    "float64": ("<f8", "d"),
}


def reference_map(map_id: str) -> dict:
    payload = json.loads(MAP_MANIFEST.read_text(encoding="utf-8"))
    for entry in payload.get("maps", []):
        if str(entry.get("id")) == map_id:
            return entry
    raise RuntimeError(f"Unknown map id: {map_id}")


def parse_header(handle) -> tuple[str, list[dict]]:
    first = handle.readline()
    if first.strip() != b"ply":
        raise RuntimeError("Input is not a PLY mesh.")
    file_format = ""
    elements: list[dict] = []
    current = None
    while True:
        raw = handle.readline()
        if not raw:
            raise RuntimeError("PLY header ends unexpectedly.")
        line = raw.decode("ascii", errors="strict").strip()
        parts = line.split()
        if not parts or parts[0] in {"comment", "obj_info"}:
            continue
        if parts[0] == "format":
            file_format = parts[1]
        elif parts[0] == "element":
            current = {"name": parts[1], "count": int(parts[2]), "properties": []}
            elements.append(current)
        elif parts[0] == "property":
            if current is None:
                raise RuntimeError("PLY property appears before an element.")
            if parts[1] == "list":
                current["properties"].append(
                    {"kind": "list", "count_type": parts[2], "value_type": parts[3], "name": parts[4]}
                )
            else:
                current["properties"].append({"kind": "scalar", "type": parts[1], "name": parts[2]})
        elif parts[0] == "end_header":
            break
    if file_format not in {"ascii", "binary_little_endian"}:
        raise RuntimeError(f"Unsupported PLY format: {file_format}")
    return file_format, elements


def read_binary_scalar(handle, type_name: str):
    try:
        numpy_type, struct_type = PLY_TYPES[type_name]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported PLY scalar type: {type_name}") from exc
    size = np.dtype(numpy_type).itemsize
    data = handle.read(size)
    if len(data) != size:
        raise RuntimeError("PLY data ends unexpectedly.")
    return struct.unpack("<" + struct_type, data)[0]


def read_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    with path.open("rb") as handle:
        file_format, elements = parse_header(handle)
        vertices = None
        colors = None
        faces: list[list[int]] = []
        for element in elements:
            name = element["name"]
            count = element["count"]
            properties = element["properties"]
            if file_format == "ascii":
                rows = [handle.readline().decode("ascii").split() for _ in range(count)]
                if name == "vertex":
                    scalar_names = [prop["name"] for prop in properties if prop["kind"] == "scalar"]
                    scalar_index = {value: index for index, value in enumerate(scalar_names)}
                    vertices = np.asarray(
                        [[float(row[scalar_index[axis]]) for axis in ("x", "y", "z")] for row in rows],
                        dtype=np.float64,
                    )
                    if all(channel in scalar_index for channel in ("red", "green", "blue")):
                        colors = np.asarray(
                            [[int(row[scalar_index[channel]]) for channel in ("red", "green", "blue")] for row in rows],
                            dtype=np.uint8,
                        )
                elif name == "face":
                    for row in rows:
                        size = int(row[0])
                        faces.append([int(value) for value in row[1 : size + 1]])
                continue

            if name == "vertex" and all(prop["kind"] == "scalar" for prop in properties):
                dtype = np.dtype([(prop["name"], PLY_TYPES[prop["type"]][0]) for prop in properties])
                data = np.fromfile(handle, dtype=dtype, count=count)
                if len(data) != count:
                    raise RuntimeError("PLY vertex block ends unexpectedly.")
                vertices = np.column_stack([data[axis] for axis in ("x", "y", "z")]).astype(np.float64)
                if all(channel in data.dtype.names for channel in ("red", "green", "blue")):
                    colors = np.column_stack([data[channel] for channel in ("red", "green", "blue")]).astype(np.uint8)
                continue

            for _ in range(count):
                row_lists: dict[str, list[int]] = {}
                for prop in properties:
                    if prop["kind"] == "scalar":
                        read_binary_scalar(handle, prop["type"])
                    else:
                        size = int(read_binary_scalar(handle, prop["count_type"]))
                        row_lists[prop["name"]] = [
                            int(read_binary_scalar(handle, prop["value_type"])) for _ in range(size)
                        ]
                if name == "face":
                    values = row_lists.get("vertex_indices") or row_lists.get("vertex_index")
                    if values:
                        faces.append(values)

    if vertices is None or not faces:
        raise RuntimeError("PLY contains no usable vertices or faces.")
    triangles: list[tuple[int, int, int]] = []
    for face in faces:
        if len(face) < 3:
            continue
        for index in range(1, len(face) - 1):
            triangles.append((face[0], face[index], face[index + 1]))
    return vertices, np.asarray(triangles, dtype=np.int64), colors


def prepare_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray | None,
    matrix: np.ndarray,
    max_faces: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    valid = (faces >= 0).all(axis=1) & (faces < len(vertices)).all(axis=1)
    faces = faces[valid]
    if len(faces) > max_faces:
        stride = max(1, math.ceil(len(faces) / max_faces))
        faces = faces[::stride][:max_faces]
    used, remapped = np.unique(faces.reshape(-1), return_inverse=True)
    selected = vertices[used]
    positions = np.column_stack(
        [np.sum(selected * matrix[axis, :3], axis=1) + matrix[axis, 3] for axis in range(3)]
    )
    indices = remapped.reshape(-1, 3).astype(np.uint32)
    keep = np.isfinite(positions).all(axis=1) & (np.abs(positions) < 1_000_000).all(axis=1)
    if not keep.all():
        lookup = np.full(len(positions), -1, dtype=np.int64)
        lookup[keep] = np.arange(int(keep.sum()))
        face_keep = keep[indices].all(axis=1)
        indices = lookup[indices[face_keep]].astype(np.uint32)
        used = used[keep]
        positions = positions[keep]
    normals = np.zeros_like(positions, dtype=np.float64)
    if len(indices):
        p0 = positions[indices[:, 0]]
        p1 = positions[indices[:, 1]]
        p2 = positions[indices[:, 2]]
        face_normals = np.cross(p1 - p0, p2 - p0)
        for corner in range(3):
            np.add.at(normals, indices[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    normals[lengths > 1e-12] /= lengths[lengths > 1e-12, None]
    normals[lengths <= 1e-12] = (0.0, 1.0, 0.0)
    selected_colors = colors[used] if colors is not None else None
    return (
        np.ascontiguousarray(positions, dtype=np.float32),
        np.ascontiguousarray(normals, dtype=np.float32),
        np.ascontiguousarray(indices.reshape(-1), dtype=np.uint32),
        np.ascontiguousarray(selected_colors, dtype=np.uint8) if selected_colors is not None else None,
    )


def append_aligned(target: bytearray, data: bytes) -> tuple[int, int]:
    while len(target) % 4:
        target.append(0)
    offset = len(target)
    target.extend(data)
    return offset, len(data)


def write_glb(path: Path, positions: np.ndarray, normals: np.ndarray, indices: np.ndarray, colors) -> None:
    binary = bytearray()
    views = []
    accessors = []

    def add_accessor(array, component_type, accessor_type, target, normalized=False, bounds=False):
        offset, length = append_aligned(binary, array.tobytes(order="C"))
        view_index = len(views)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": length, "target": target})
        payload = {
            "bufferView": view_index,
            "componentType": component_type,
            "count": int(array.shape[0] if array.ndim > 1 else len(array)),
            "type": accessor_type,
        }
        if normalized:
            payload["normalized"] = True
        if bounds:
            payload["min"] = [float(value) for value in array.min(axis=0)]
            payload["max"] = [float(value) for value in array.max(axis=0)]
        accessors.append(payload)
        return len(accessors) - 1

    position_accessor = add_accessor(positions, 5126, "VEC3", 34962, bounds=True)
    normal_accessor = add_accessor(normals, 5126, "VEC3", 34962)
    attributes = {"POSITION": position_accessor, "NORMAL": normal_accessor}
    if colors is not None:
        attributes["COLOR_0"] = add_accessor(colors, 5121, "VEC3", 34962, normalized=True)
    index_accessor = add_accessor(indices, 5125, "SCALAR", 34963)
    gltf = {
        "asset": {"version": "2.0", "generator": "ATLAS Camera Path Lab"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "Read-only COLMAP Delaunay mesh"}],
        "meshes": [{"primitives": [{"attributes": attributes, "indices": index_accessor, "material": 0}]}],
        "materials": [{
            "name": "ATLAS lab surface",
            "doubleSided": True,
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.72, 0.84, 0.88, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.92,
            },
        }],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": views,
        "accessors": accessors,
    }
    json_chunk = bytearray(json.dumps(gltf, separators=(",", ":")).encode("utf-8"))
    while len(json_chunk) % 4:
        json_chunk.append(0x20)
    while len(binary) % 4:
        binary.append(0)
    total_length = 12 + 8 + len(json_chunk) + 8 + len(binary)
    output = bytearray(struct.pack("<4sII", b"glTF", 2, total_length))
    output.extend(struct.pack("<I4s", len(json_chunk), b"JSON"))
    output.extend(json_chunk)
    output.extend(struct.pack("<I4s", len(binary), b"BIN\x00"))
    output.extend(binary)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_bytes(output)
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-id", required=True)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "runtime" / "camera_path_lab_mesh" / "good_copy_sparse_delaunay.ply",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "viewer" / "public" / "camera_path_lab" / "good_copy_mesh.glb",
    )
    parser.add_argument("--max-faces", type=int, default=220_000)
    args = parser.parse_args()
    if args.max_faces < 1_000:
        raise SystemExit("--max-faces must be at least 1000")
    entry = reference_map(args.map_id)
    matrix = np.asarray((entry.get("room_alignment") or {}).get("matrix"), dtype=np.float64)
    if matrix.shape != (3, 4) or not np.isfinite(matrix).all():
        raise RuntimeError("Reference map has no validated 3x4 room alignment matrix.")
    vertices, faces, colors = read_mesh(args.input)
    positions, normals, indices, colors = prepare_mesh(vertices, faces, colors, matrix, args.max_faces)
    if len(indices) < 3:
        raise RuntimeError("The converted mesh has no valid triangles.")
    write_glb(args.output, positions, normals, indices, colors)
    print(
        f"Camera Path Lab GLB ready: {len(positions):,} vertices, {len(indices) // 3:,} faces, "
        f"{args.output.stat().st_size / 1_048_576:.1f} MiB -> {args.output}"
    )


if __name__ == "__main__":
    main()
