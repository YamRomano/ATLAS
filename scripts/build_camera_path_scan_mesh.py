#!/usr/bin/env python3
"""Build a room-aligned, textured GLB from an ARKit scan archive.

The source scan is used only as a visual surface. Camera localization continues
to use the validated COLMAP reference map.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import struct
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "viewer" / "public" / "camera_path_lab" / "good_copy_mesh.json"
DEFAULT_OUTPUT = ROOT / "viewer" / "public" / "camera_path_lab" / "room_scan_textured.glb"


def archive_member(archive: zipfile.ZipFile, basename: str) -> str:
    matches = [name for name in archive.namelist() if Path(name).name == basename]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {basename!r} in the scan archive; found {len(matches)}.")
    return matches[0]


def read_scan_archive(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, bytes, dict]:
    with zipfile.ZipFile(path) as archive:
        info = json.loads(archive.read(archive_member(archive, "info.json")))
        vertex_count = int(info.get("numVertices") or 0)
        face_count = int(info.get("numFaces") or 0)
        if vertex_count < 3 or face_count < 1:
            raise RuntimeError("Scan metadata does not describe a usable textured mesh.")

        positions = np.empty((vertex_count, 3), dtype=np.float32)
        texcoords = np.empty((vertex_count, 2), dtype=np.float32)
        indices = np.empty((face_count, 3), dtype=np.uint32)
        vertex_index = texture_index = face_index = 0
        obj_name = archive_member(archive, "textured_output.obj")
        with archive.open(obj_name) as raw:
            for encoded in raw:
                if encoded.startswith(b"v "):
                    if vertex_index >= vertex_count:
                        raise RuntimeError("OBJ has more vertices than declared by info.json.")
                    positions[vertex_index] = [float(value) for value in encoded.split()[1:4]]
                    vertex_index += 1
                elif encoded.startswith(b"vt "):
                    if texture_index >= vertex_count:
                        raise RuntimeError("OBJ has more texture coordinates than vertices.")
                    u, v = (float(value) for value in encoded.split()[1:3])
                    texcoords[texture_index] = (u, 1.0 - v)
                    texture_index += 1
                elif encoded.startswith(b"f "):
                    values = encoded.split()[1:]
                    if len(values) != 3:
                        raise RuntimeError("The textured scan must contain triangular faces.")
                    if face_index >= face_count:
                        raise RuntimeError("OBJ has more faces than declared by info.json.")
                    for corner, value in enumerate(values):
                        parts = value.split(b"/")
                        position_id = int(parts[0])
                        texture_id = int(parts[1]) if len(parts) > 1 and parts[1] else position_id
                        if position_id != texture_id:
                            raise RuntimeError("OBJ position/texture indices are not one-to-one.")
                        indices[face_index, corner] = position_id - 1
                    face_index += 1

        if vertex_index != vertex_count or texture_index != vertex_count or face_index != face_count:
            raise RuntimeError(
                "Scan mesh count mismatch: "
                f"vertices {vertex_index}/{vertex_count}, texture coordinates "
                f"{texture_index}/{vertex_count}, faces {face_index}/{face_count}."
            )
        if not np.isfinite(positions).all() or not np.isfinite(texcoords).all():
            raise RuntimeError("Scan mesh contains non-finite geometry or texture coordinates.")
        if int(indices.max(initial=0)) >= vertex_count:
            raise RuntimeError("Scan mesh contains an out-of-range face index.")
        texture = archive.read(archive_member(archive, "textured_output.jpg"))
    return positions, texcoords, indices, texture, info


def voxel_centers(points: np.ndarray, size: float, max_points: int = 120_000) -> np.ndarray:
    finite = points[np.isfinite(points).all(axis=1)]
    keys = np.floor(finite / float(size)).astype(np.int32)
    _, unique = np.unique(keys, axis=0, return_index=True)
    sampled = finite[np.sort(unique)]
    if len(sampled) > max_points:
        stride = int(math.ceil(len(sampled) / max_points))
        sampled = sampled[::stride][:max_points]
    return np.asarray(sampled, dtype=np.float64)


def trimmed_center(points: np.ndarray) -> np.ndarray:
    low, high = np.quantile(points, [0.015, 0.985], axis=0)
    return (low + high) * 0.5


def horizontal_principal_angle(points: np.ndarray) -> float:
    xz = points[:, (0, 2)]
    centered = xz - np.median(xz, axis=0)
    covariance = np.cov(centered, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    return math.atan2(float(axis[1]), float(axis[0]))


def yaw_matrix(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.asarray(
        [[cosine, 0.0, -sine], [0.0, 1.0, 0.0], [sine, 0.0, cosine]],
        dtype=np.float64,
    )


def refine_yaw_alignment(
    source: np.ndarray,
    target: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    iterations: int = 28,
) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(target)
    for iteration in range(iterations):
        if (
            not np.isfinite(rotation).all()
            or not np.isfinite(translation).all()
            or float(np.linalg.norm(rotation)) > 5.0
            or float(np.linalg.norm(translation)) > 100.0
        ):
            break
        transformed = np.einsum("ni,ji->nj", source, rotation, optimize=False) + translation
        distances, neighbor_indices = tree.query(transformed, workers=-1)
        cutoff = min(1.35 if iteration < 5 else 0.72, float(np.quantile(distances, 0.68)))
        keep = distances <= max(0.20, cutoff)
        if int(keep.sum()) < 500:
            break

        source_xz = transformed[keep][:, (0, 2)]
        target_xz = target[neighbor_indices[keep]][:, (0, 2)]
        source_mean = source_xz.mean(axis=0)
        target_mean = target_xz.mean(axis=0)
        covariance = np.einsum(
            "ni,nj->ij",
            source_xz - source_mean,
            target_xz - target_mean,
            optimize=False,
        )
        left, _, right = np.linalg.svd(covariance)
        delta_2d = right.T @ left.T
        if np.linalg.det(delta_2d) < 0:
            right[-1] *= -1
            delta_2d = right.T @ left.T
        delta_angle = math.atan2(float(delta_2d[1, 0]), float(delta_2d[0, 0]))
        # Nearest-neighbor ICP can briefly select the opposite side of this
        # mostly rectangular room. Keep every correction local so one
        # ambiguous iteration cannot send an otherwise good candidate away.
        delta_angle = float(np.clip(delta_angle, -math.radians(5.0), math.radians(5.0)))
        cosine = math.cos(delta_angle)
        sine = math.sin(delta_angle)
        delta_2d = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
        delta_rotation = yaw_matrix(delta_angle)
        delta_translation = np.zeros(3, dtype=np.float64)
        delta_translation[[0, 2]] = target_mean - delta_2d @ source_mean
        horizontal_length = float(np.linalg.norm(delta_translation[[0, 2]]))
        if horizontal_length > 0.24:
            delta_translation[[0, 2]] *= 0.24 / horizontal_length
        y_delta = np.median(target[neighbor_indices[keep], 1] - transformed[keep, 1])
        delta_translation[1] = float(np.clip(y_delta, -0.18, 0.18))
        if not np.isfinite(delta_translation).all():
            break
        rotation = delta_rotation @ rotation
        translation = delta_rotation @ translation + delta_translation
        if abs(delta_angle) < 1e-5 and np.linalg.norm(delta_translation) < 2e-4:
            break
    return rotation, translation


def alignment_score(source: np.ndarray, target: np.ndarray, rotation, translation) -> dict:
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        return {
            "score": float("inf"),
            "median_source_to_reference_m": float("inf"),
            "median_reference_to_source_m": float("inf"),
            "source_within_0_50m_ratio": 0.0,
            "reference_within_0_50m_ratio": 0.0,
        }
    transformed = np.einsum("ni,ji->nj", source, rotation, optimize=False) + translation
    forward = cKDTree(target).query(transformed, workers=-1)[0]
    reverse = cKDTree(transformed).query(target, workers=-1)[0]
    forward_clip = np.minimum(forward, 1.25)
    reverse_clip = np.minimum(reverse, 1.25)
    score = float(np.mean(forward_clip) * 0.45 + np.mean(reverse_clip) * 0.55)
    return {
        "score": score,
        "median_source_to_reference_m": float(np.median(forward)),
        "median_reference_to_source_m": float(np.median(reverse)),
        "source_within_0_50m_ratio": float(np.mean(forward <= 0.50)),
        "reference_within_0_50m_ratio": float(np.mean(reverse <= 0.50)),
    }


def align_scan_to_reference(
    scan_positions: np.ndarray,
    reference_points: np.ndarray,
    preferred_candidate: int | None = None,
) -> tuple[np.ndarray, dict]:
    source = voxel_centers(scan_positions, 0.13)
    target = voxel_centers(reference_points, 0.12)
    source_angle = horizontal_principal_angle(source)
    target_angle = horizontal_principal_angle(target)
    source_center = trimmed_center(source)
    target_center = trimmed_center(target)
    candidates = []
    for quarter_turn in range(4):
        angle = target_angle - source_angle + quarter_turn * math.pi / 2.0
        rotation = yaw_matrix(angle)
        translation = target_center - rotation @ source_center
        rotation, translation = refine_yaw_alignment(source, target, rotation, translation)
        metrics = alignment_score(source, target, rotation, translation)
        metrics["candidate"] = quarter_turn
        metrics["yaw_deg"] = math.degrees(math.atan2(float(rotation[2, 0]), float(rotation[0, 0])))
        candidates.append((metrics["score"], rotation, translation, metrics))
    candidates.sort(key=lambda value: value[0])
    automatic_best = candidates[0]
    selected = automatic_best
    if preferred_candidate is not None:
        selected = next(
            value for value in candidates if value[3]["candidate"] == preferred_candidate
        )
    _, rotation, translation, best = selected
    runner_up = next(value[3] for value in candidates if value is not selected)
    best["runner_up_score"] = runner_up["score"]
    best["score_margin"] = float(runner_up["score"] - best["score"])
    best["selection_mode"] = "forced" if preferred_candidate is not None else "automatic"
    best["automatic_best_candidate"] = int(automatic_best[3]["candidate"])
    best["source_alignment_points"] = int(len(source))
    best["reference_alignment_points"] = int(len(target))
    best["candidates"] = [
        {key: value for key, value in candidate[3].items() if key != "candidates"}
        for candidate in candidates
    ]
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform, best


def resize_jpeg(texture: bytes, max_size: int) -> bytes:
    with Image.open(io.BytesIO(texture)) as image:
        image = image.convert("RGB")
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, "JPEG", quality=88, optimize=True, progressive=True)
        return output.getvalue()


def append_aligned(target: bytearray, data: bytes) -> tuple[int, int]:
    while len(target) % 4:
        target.append(0)
    offset = len(target)
    target.extend(data)
    return offset, len(data)


def write_textured_glb(
    path: Path,
    positions: np.ndarray,
    texcoords: np.ndarray,
    indices: np.ndarray,
    texture: bytes,
) -> None:
    binary = bytearray()
    views = []
    accessors = []

    def add_accessor(array, component_type, accessor_type, target, bounds=False):
        offset, length = append_aligned(binary, np.ascontiguousarray(array).tobytes(order="C"))
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": length, "target": target})
        accessor = {
            "bufferView": len(views) - 1,
            "componentType": component_type,
            "count": int(array.shape[0]),
            "type": accessor_type,
        }
        if bounds:
            accessor["min"] = [float(value) for value in array.min(axis=0)]
            accessor["max"] = [float(value) for value in array.max(axis=0)]
        accessors.append(accessor)
        return len(accessors) - 1

    position_accessor = add_accessor(positions, 5126, "VEC3", 34962, bounds=True)
    texture_accessor = add_accessor(texcoords, 5126, "VEC2", 34962)
    index_accessor = add_accessor(indices.reshape(-1), 5125, "SCALAR", 34963)
    image_offset, image_length = append_aligned(binary, texture)
    views.append({"buffer": 0, "byteOffset": image_offset, "byteLength": image_length})
    gltf = {
        "asset": {"version": "2.0", "generator": "Camera Path Room Scan Builder"},
        "extensionsUsed": ["KHR_materials_unlit"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "Aligned textured room scan"}],
        "meshes": [{"primitives": [{
            "attributes": {"POSITION": position_accessor, "TEXCOORD_0": texture_accessor},
            "indices": index_accessor,
            "material": 0,
        }]}],
        "materials": [{
            "name": "Room scan texture",
            "doubleSided": True,
            "extensions": {"KHR_materials_unlit": {}},
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0},
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
        }],
        "textures": [{"sampler": 0, "source": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}],
        "images": [{"bufferView": len(views) - 1, "mimeType": "image/jpeg"}],
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
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(output)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--texture-size", type=int, default=4096)
    parser.add_argument(
        "--alignment-candidate",
        type=int,
        choices=range(4),
        help=(
            "Force one of the four 90-degree room-axis candidates. Use this only "
            "to resolve a visually verified symmetric-room orientation ambiguity."
        ),
    )
    args = parser.parse_args()
    if args.texture_size < 1024:
        raise SystemExit("--texture-size must be at least 1024")

    positions, texcoords, indices, texture, scan_info = read_scan_archive(args.archive)
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    reference_points = np.asarray(reference.get("voxels") or [], dtype=np.float64)[:, :3]
    if len(reference_points) < 1_000:
        raise RuntimeError("Reference surface contains too few points for alignment.")
    transform, alignment = align_scan_to_reference(
        positions,
        reference_points,
        preferred_candidate=args.alignment_candidate,
    )
    aligned = (
        np.einsum("ni,ji->nj", positions.astype(np.float64), transform[:3, :3], optimize=False)
        + transform[:3, 3]
    )
    aligned = np.ascontiguousarray(aligned, dtype=np.float32)
    web_texture = resize_jpeg(texture, args.texture_size)
    write_textured_glb(args.output, aligned, texcoords, indices, web_texture)

    metadata = {
        "format": "camera-path-textured-room-scan-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_archive": str(args.archive),
        "source_scan_title": scan_info.get("title"),
        "reference_map_id": reference.get("source_map_id"),
        "visual_only": True,
        "vertices": int(len(positions)),
        "faces": int(len(indices)),
        "texture_size": args.texture_size,
        "bounds": {"min": aligned.min(axis=0).tolist(), "max": aligned.max(axis=0).tolist()},
        "scan_to_room_matrix": transform.tolist(),
        "alignment": alignment,
        "glb": args.output.name,
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"Textured room GLB ready: {len(positions):,} vertices, {len(indices):,} faces, "
        f"{args.output.stat().st_size / 1_048_576:.1f} MiB; "
        f"alignment score {alignment['score']:.4f}, yaw {alignment['yaw_deg']:.2f} deg -> {args.output}"
    )


if __name__ == "__main__":
    main()
