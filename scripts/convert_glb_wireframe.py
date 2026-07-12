#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import struct
import zipfile
from pathlib import Path


COMPONENTS = {
    5120: ("b", 1),
    5121: ("B", 1),
    5122: ("h", 2),
    5123: ("H", 2),
    5125: ("I", 4),
    5126: ("f", 4),
}
TYPE_COUNTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


def read_glb(data: bytes) -> tuple[dict, bytes]:
    if data[:4] != b"glTF":
        raise ValueError("Not a GLB file")
    version, length = struct.unpack_from("<II", data, 4)
    if version != 2:
        raise ValueError(f"Unsupported GLB version {version}")
    off = 12
    gltf = None
    bin_chunk = None
    while off < length:
        chunk_len, chunk_type = struct.unpack_from("<II", data, off)
        off += 8
        chunk = data[off:off + chunk_len]
        off += chunk_len
        if chunk_type == 0x4E4F534A:
            gltf = json.loads(chunk.rstrip(b" \t\r\n\0").decode("utf-8"))
        elif chunk_type == 0x004E4942:
            bin_chunk = chunk
    if gltf is None or bin_chunk is None:
        raise ValueError("GLB is missing JSON or BIN chunk")
    return gltf, bin_chunk


def load_accessor(gltf: dict, blob: bytes, accessor_index: int) -> list[list[float]] | list[int]:
    accessor = gltf["accessors"][accessor_index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    component_type = accessor["componentType"]
    fmt, comp_size = COMPONENTS[component_type]
    elem_count = TYPE_COUNTS[accessor["type"]]
    count = accessor["count"]
    byte_offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    stride = view.get("byteStride", comp_size * elem_count)

    out = []
    for i in range(count):
        base = byte_offset + i * stride
        vals = struct.unpack_from("<" + fmt * elem_count, blob, base)
        if elem_count == 1:
            out.append(vals[0])
        else:
            out.append([float(v) for v in vals])
    return out


def find_glb_bytes(zip_path: Path) -> tuple[str, bytes]:
    with zipfile.ZipFile(zip_path, "r") as z:
        names = [n for n in z.namelist() if n.lower().endswith(".glb") and "__MACOSX" not in n]
        if not names:
            raise FileNotFoundError(f"No .glb found in {zip_path}")
        name = names[0]
        return name, z.read(name)


def normalize_vertices(vertices: list[list[float]]) -> tuple[list[list[float]], dict]:
    mins = [min(v[i] for v in vertices) for i in range(3)]
    maxs = [max(v[i] for v in vertices) for i in range(3)]
    center = [(mins[i] + maxs[i]) * 0.5 for i in range(3)]
    span = max(maxs[i] - mins[i] for i in range(3))
    scale = 1.0 / span if span > 0 else 1.0
    normalized = [[(v[i] - center[i]) * scale for i in range(3)] for v in vertices]
    return normalized, {"min": mins, "max": maxs, "center": center, "span": span}


def triangle_edges(indices: list[int]) -> list[tuple[int, int]]:
    seen = set()
    edges = []
    for i in range(0, len(indices) - 2, 3):
        tri = [indices[i], indices[i + 1], indices[i + 2]]
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            if a == b:
                continue
            e = (a, b) if a < b else (b, a)
            if e not in seen:
                seen.add(e)
                edges.append(e)
    return edges


def sample_even(items: list, limit: int) -> list:
    if len(items) <= limit:
        return items
    step = len(items) / limit
    return [items[min(len(items) - 1, int(i * step))] for i in range(limit)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--out-glb", type=Path)
    ap.add_argument("--edge-limit", type=int, default=1700)
    ap.add_argument("--point-limit", type=int, default=2200)
    args = ap.parse_args()

    source_name, glb_data = find_glb_bytes(args.zip)
    gltf, blob = read_glb(glb_data)

    vertices: list[list[float]] = []
    edges: list[tuple[int, int]] = []
    primitive_count = 0
    triangle_count = 0

    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            if prim.get("mode", 4) != 4:
                continue
            pos_idx = prim.get("attributes", {}).get("POSITION")
            if pos_idx is None:
                continue
            local_positions = load_accessor(gltf, blob, pos_idx)
            if not local_positions:
                continue
            base = len(vertices)
            vertices.extend(local_positions)
            if "indices" in prim:
                local_indices = load_accessor(gltf, blob, prim["indices"])
            else:
                local_indices = list(range(len(local_positions)))
            local_indices = [base + int(i) for i in local_indices]
            edges.extend(triangle_edges(local_indices))
            triangle_count += len(local_indices) // 3
            primitive_count += 1

    if not vertices:
        raise RuntimeError("No triangle mesh POSITION data found in GLB")

    vertices, bounds = normalize_vertices(vertices)
    sampled_edges = sample_even(edges, args.edge_limit)
    sampled_points = sample_even(list(range(len(vertices))), args.point_limit)

    out = {
        "name": "DJI Mini 3 Pro",
        "source_zip": str(args.zip),
        "source_glb_entry": source_name,
        "primitive_count": primitive_count,
        "source_vertex_count": len(vertices),
        "source_triangle_count": triangle_count,
        "source_edge_count": len(edges),
        "bounds": bounds,
        "vertices": [[round(x, 6), round(y, 6), round(z, 6)] for x, y, z in vertices],
        "edges": sampled_edges,
        "point_indices": sampled_points,
        "note": "Normalized wireframe extracted from the uploaded GLB for dependency-free canvas rendering.",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out), encoding="utf-8")
    if args.out_glb:
        args.out_glb.parent.mkdir(parents=True, exist_ok=True)
        args.out_glb.write_bytes(glb_data)
    print(json.dumps({
        "source": source_name,
        "out_json": str(args.out_json),
        "out_glb": None if args.out_glb is None else str(args.out_glb),
        "vertices": len(vertices),
        "triangles": triangle_count,
        "edges_total": len(edges),
        "edges_used": len(sampled_edges),
        "points_used": len(sampled_points),
        "bounds": bounds,
    }, indent=2))


if __name__ == "__main__":
    main()
