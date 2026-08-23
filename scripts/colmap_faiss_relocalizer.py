#!/usr/bin/env python3
"""Global COLMAP SIFT-to-3D retrieval with a persistent Faiss IVF index.

The index contains every SIFT descriptor observation in the registered COLMAP
model that is attached to a 3D point.  Untriangulated database descriptors are
intentionally excluded because they cannot create a PnP/TSolve correspondence.

At runtime OpenCV SIFT descriptors from the current frame search the complete
map index. ANN neighbors are grouped by 3D point before the ratio test, made
one-to-one, and geometrically checked against registered source images. The
accepted output is only a 2D-to-3D correspondence pool: TSolve remains the
live pose solver. The older database-backed PnP entry point remains available
for offline compatibility, but it is not used by the live pipeline.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from colmap_io import CAMERA_MODEL_BY_ID, Camera, read_images_model, read_points3d_text


INDEX_FORMAT_VERSION = 1
DESCRIPTOR_DIMENSION = 128


def require_faiss():
    try:
        import faiss  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised by deployment, not tests
        raise RuntimeError(
            "Faiss is required for global SIFT retrieval. Install faiss-cpu in the "
            "ATLAS Python environment."
        ) from exc
    return faiss


def file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def source_fingerprint(database: Path, sparse_model: Path) -> dict[str, Any]:
    model_file = sparse_model / "images.bin"
    if not model_file.exists():
        model_file = sparse_model / "images.txt"
    points_file = sparse_model / "points3D.bin"
    if not points_file.exists():
        points_file = sparse_model / "points3D.txt"
    return {
        "database": file_fingerprint(database),
        "images_model": file_fingerprint(model_file),
        "points_model": file_fingerprint(points_file),
    }


def index_is_current(
    out_dir: Path,
    *,
    database: Path | None = None,
    sparse_model: Path | None = None,
) -> bool:
    required = [
        out_dir / "index.faiss",
        out_dir / "point3d_ids.npy",
        out_dir / "source_image_ids.npy",
        out_dir / "source_feature_indices.npy",
        out_dir / "manifest.json",
    ]
    if not all(path.exists() for path in required):
        return False
    try:
        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    if int(manifest.get("format_version") or -1) != INDEX_FORMAT_VERSION:
        return False
    if database is None or sparse_model is None:
        return True
    try:
        return manifest.get("source_fingerprint") == source_fingerprint(database, sparse_model)
    except OSError:
        return False


def _database_images(conn: sqlite3.Connection) -> dict[str, tuple[int, int]]:
    return {
        str(name): (int(image_id), int(camera_id))
        for image_id, name, camera_id in conn.execute(
            "SELECT image_id, name, camera_id FROM images"
        )
    }


def _read_descriptors(
    conn: sqlite3.Connection,
    image_id: int,
) -> np.ndarray | None:
    row = conn.execute(
        "SELECT rows, cols, data FROM descriptors WHERE image_id = ?",
        (int(image_id),),
    ).fetchone()
    if row is None:
        return None
    rows, cols, blob = row
    rows = int(rows)
    cols = int(cols)
    if rows <= 0 or cols != DESCRIPTOR_DIMENSION or blob is None:
        return None
    values = np.frombuffer(blob, dtype=np.uint8)
    if values.size != rows * cols:
        return None
    return values.reshape(rows, cols)


def _registered_sources(
    database: Path,
    sparse_model: Path,
) -> list[tuple[int, Any]]:
    images = read_images_model(sparse_model)
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as conn:
        database_images = _database_images(conn)
    sources: list[tuple[int, Any]] = []
    for image in images.values():
        database_row = database_images.get(str(image.name))
        if database_row is None:
            continue
        sources.append((int(database_row[0]), image))
    sources.sort(key=lambda item: str(item[1].name))
    return sources


def _mapped_rows(
    conn: sqlite3.Connection,
    database_image_id: int,
    model_image: Any,
) -> tuple[np.ndarray, np.ndarray] | None:
    descriptors = _read_descriptors(conn, database_image_id)
    if descriptors is None:
        return None
    point_ids = np.asarray(model_image.point3d_ids, dtype=np.int64).reshape(-1)
    usable = min(len(descriptors), len(point_ids))
    if usable <= 0:
        return None
    point_ids = point_ids[:usable]
    valid_indices = np.flatnonzero(point_ids >= 0).astype(np.int32)
    if len(valid_indices) == 0:
        return None
    return descriptors[valid_indices], np.column_stack(
        [point_ids[valid_indices], valid_indices]
    ).astype(np.int64)


def _training_sample(
    *,
    database: Path,
    sources: list[tuple[int, Any]],
    requested_size: int,
) -> tuple[np.ndarray, int]:
    per_image = max(1, int(math.ceil(max(1, requested_size) / max(1, len(sources)))))
    samples: list[np.ndarray] = []
    mapped_count = 0
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as conn:
        for database_image_id, model_image in sources:
            rows = _mapped_rows(conn, database_image_id, model_image)
            if rows is None:
                continue
            descriptors, _metadata = rows
            mapped_count += int(len(descriptors))
            if len(descriptors) <= per_image:
                samples.append(descriptors.copy())
            else:
                indices = np.linspace(
                    0,
                    len(descriptors) - 1,
                    num=per_image,
                    dtype=np.int64,
                )
                samples.append(descriptors[indices].copy())
    if not samples or mapped_count <= 0:
        raise RuntimeError("No COLMAP SIFT descriptors linked to 3D points were found.")
    sample = np.concatenate(samples, axis=0)
    if len(sample) > requested_size > 0:
        indices = np.linspace(0, len(sample) - 1, num=requested_size, dtype=np.int64)
        sample = sample[indices]
    return np.ascontiguousarray(sample, dtype=np.float32), mapped_count


def build_faiss_index(
    *,
    database: Path,
    sparse_model: Path,
    out_dir: Path,
    nlist: int = 4096,
    nprobe: int = 32,
    training_sample_size: int = 262144,
    force: bool = False,
) -> dict[str, Any]:
    """Build an IVF-SQ8 index over all mapped COLMAP descriptor observations."""
    faiss = require_faiss()
    database = database.resolve()
    sparse_model = sparse_model.resolve()
    out_dir = out_dir.resolve()
    if index_is_current(out_dir, database=database, sparse_model=sparse_model) and not force:
        return json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))

    sources = _registered_sources(database, sparse_model)
    if not sources:
        raise RuntimeError("No registered COLMAP images have matching database descriptors.")
    train, mapped_count = _training_sample(
        database=database,
        sources=sources,
        requested_size=max(4096, int(training_sample_size)),
    )
    # Faiss recommends substantially more training vectors than IVF cells.
    # Reduce only when a small test/model cannot support the configured count.
    effective_nlist = min(
        max(1, int(nlist)),
        max(1, int(len(train) // 39)),
    )
    quantizer = faiss.IndexFlatL2(DESCRIPTOR_DIMENSION)
    index = faiss.IndexIVFScalarQuantizer(
        quantizer,
        DESCRIPTOR_DIMENSION,
        effective_nlist,
        faiss.ScalarQuantizer.QT_8bit,
        faiss.METRIC_L2,
    )
    started = time.perf_counter()
    index.train(train)
    index.nprobe = min(max(1, int(nprobe)), effective_nlist)

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{out_dir.name}.building-", dir=str(out_dir.parent))
    )
    try:
        point_ids_file = np.lib.format.open_memmap(
            temporary / "point3d_ids.npy",
            mode="w+",
            dtype=np.int64,
            shape=(mapped_count,),
        )
        source_images_file = np.lib.format.open_memmap(
            temporary / "source_image_ids.npy",
            mode="w+",
            dtype=np.int32,
            shape=(mapped_count,),
        )
        source_features_file = np.lib.format.open_memmap(
            temporary / "source_feature_indices.npy",
            mode="w+",
            dtype=np.int32,
            shape=(mapped_count,),
        )
        cursor = 0
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as conn:
            for source_number, (database_image_id, model_image) in enumerate(sources, start=1):
                rows = _mapped_rows(conn, database_image_id, model_image)
                if rows is None:
                    continue
                descriptors, metadata = rows
                count = int(len(descriptors))
                end = cursor + count
                if end > mapped_count:
                    raise RuntimeError("Mapped descriptor count changed while building the index.")
                vector_ids = np.arange(cursor, end, dtype=np.int64)
                index.add_with_ids(
                    np.ascontiguousarray(descriptors, dtype=np.float32),
                    vector_ids,
                )
                point_ids_file[cursor:end] = metadata[:, 0]
                source_images_file[cursor:end] = int(model_image.image_id)
                source_features_file[cursor:end] = metadata[:, 1].astype(np.int32)
                cursor = end
                if source_number % 100 == 0 or source_number == len(sources):
                    print(
                        f"Faiss map index: {source_number}/{len(sources)} images, "
                        f"{cursor}/{mapped_count} mapped descriptors",
                        flush=True,
                    )
        if cursor != mapped_count or int(index.ntotal) != mapped_count:
            raise RuntimeError(
                f"Faiss index count mismatch: metadata={cursor}, index={index.ntotal}, "
                f"expected={mapped_count}"
            )
        point_ids_file.flush()
        source_images_file.flush()
        source_features_file.flush()
        del point_ids_file, source_images_file, source_features_file
        faiss.write_index(index, str(temporary / "index.faiss"))
        sparse_points_file = sparse_model / "points3D.txt"
        unique_points = (
            len(read_points3d_text(sparse_points_file))
            if sparse_points_file.exists()
            else None
        )
        manifest = {
            "format_version": INDEX_FORMAT_VERSION,
            "algorithm": "faiss_ivf_sq8_colmap_sift_2d3d",
            "descriptor_type": "COLMAP_SIFT_UINT8_AS_FLOAT_L2",
            "descriptor_dimension": DESCRIPTOR_DIMENSION,
            "indexed_observations": int(mapped_count),
            "registered_source_images": int(len(sources)),
            "map_point_count": int(unique_points) if unique_points is not None else None,
            "nlist": int(effective_nlist),
            "nprobe_default": int(index.nprobe),
            "training_vectors": int(len(train)),
            "build_seconds": float(time.perf_counter() - started),
            "faiss_version": str(getattr(faiss, "__version__", "unknown")),
            "source_fingerprint": source_fingerprint(database, sparse_model),
            "created_at": time.time(),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        stale = None
        if out_dir.exists():
            stale = out_dir.with_name(f".{out_dir.name}.stale-{os.getpid()}")
            if stale.exists():
                shutil.rmtree(stale)
            out_dir.rename(stale)
        try:
            os.replace(temporary, out_dir)
        except Exception:
            if stale is not None and stale.exists() and not out_dir.exists():
                stale.rename(out_dir)
            raise
        if stale is not None:
            shutil.rmtree(stale, ignore_errors=True)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _query_features(
    database_path: Path,
    image_name: str,
) -> tuple[int, Camera, np.ndarray, np.ndarray]:
    with sqlite3.connect(str(database_path)) as conn:
        image_row = conn.execute(
            "SELECT image_id, camera_id FROM images WHERE name = ?",
            (image_name,),
        ).fetchone()
        if image_row is None:
            raise RuntimeError("faiss_query_image_missing")
        image_id, camera_id = int(image_row[0]), int(image_row[1])
        keypoint_row = conn.execute(
            "SELECT rows, cols, data FROM keypoints WHERE image_id = ?",
            (image_id,),
        ).fetchone()
        descriptor_row = conn.execute(
            "SELECT rows, cols, data FROM descriptors WHERE image_id = ?",
            (image_id,),
        ).fetchone()
        camera_row = conn.execute(
            "SELECT model, width, height, params FROM cameras WHERE camera_id = ?",
            (camera_id,),
        ).fetchone()
    if keypoint_row is None or descriptor_row is None or camera_row is None:
        raise RuntimeError("faiss_query_features_missing")
    keypoint_rows, keypoint_cols, keypoint_blob = keypoint_row
    descriptor_rows, descriptor_cols, descriptor_blob = descriptor_row
    keypoints = np.frombuffer(keypoint_blob, dtype=np.float32).reshape(
        int(keypoint_rows), int(keypoint_cols)
    )
    descriptors = np.frombuffer(descriptor_blob, dtype=np.uint8).reshape(
        int(descriptor_rows), int(descriptor_cols)
    )
    usable = min(len(keypoints), len(descriptors))
    if usable <= 0 or descriptors.shape[1] != DESCRIPTOR_DIMENSION:
        raise RuntimeError("faiss_query_descriptor_shape_invalid")
    model_id, width, height, params_blob = camera_row
    model_name, _ = CAMERA_MODEL_BY_ID.get(int(model_id), (f"MODEL_{model_id}", 0))
    camera = Camera(
        camera_id=camera_id,
        model=model_name,
        width=int(width),
        height=int(height),
        params=np.frombuffer(params_blob, dtype=np.float64).tolist(),
    )
    return image_id, camera, keypoints[:usable, :2].copy(), descriptors[:usable].copy()


def opencv_sift_features(
    image_path: Path,
    *,
    max_features: int = 2400,
    n_octave_layers: int = 3,
    contrast_threshold: float = 0.02,
    edge_threshold: float = 12.0,
    sigma: float = 1.6,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Extract COLMAP-compatible 128-D SIFT directly from a live frame.

    COLMAP stores its normalized SIFT descriptor as uint8 values on the usual
    approximately-512 L2 scale. OpenCV returns the same descriptor convention
    as float32, so no RootSIFT or unit normalization may be inserted here: the
    persistent map index was trained on the raw COLMAP descriptor scale.
    """
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None or gray.ndim != 2:
        raise RuntimeError("opencv_sift_frame_unreadable")
    detector = cv2.SIFT_create(
        nfeatures=max(64, int(max_features)),
        nOctaveLayers=max(1, int(n_octave_layers)),
        contrastThreshold=max(1.0e-5, float(contrast_threshold)),
        edgeThreshold=max(1.0, float(edge_threshold)),
        sigma=max(0.1, float(sigma)),
    )
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    if not keypoints or descriptors is None:
        raise RuntimeError("opencv_sift_no_features")
    descriptors = np.ascontiguousarray(descriptors, dtype=np.float32)
    if descriptors.ndim != 2 or descriptors.shape[1] != DESCRIPTOR_DIMENSION:
        raise RuntimeError("opencv_sift_descriptor_shape_invalid")
    xy = np.asarray([keypoint.pt for keypoint in keypoints], dtype=np.float32)
    usable = min(len(xy), len(descriptors))
    if usable <= 0:
        raise RuntimeError("opencv_sift_no_usable_features")
    height, width = gray.shape
    return xy[:usable], descriptors[:usable], int(width), int(height)


def calibrated_frame_camera(
    *,
    model: str,
    params: str | list[float] | np.ndarray,
    width: int,
    height: int,
) -> Camera:
    if isinstance(params, str):
        values = [float(value.strip()) for value in params.split(",") if value.strip()]
    else:
        values = np.asarray(params, dtype=np.float64).reshape(-1).astype(float).tolist()
    model_name = str(model or "").strip().upper()
    expected = next(
        (count for name, count in CAMERA_MODEL_BY_ID.values() if name == model_name),
        None,
    )
    if expected is None or len(values) != int(expected):
        raise RuntimeError(
            f"opencv_sift_camera_invalid:{model_name}:{len(values)}_params"
        )
    return Camera(
        camera_id=-1,
        model=model_name,
        width=int(width),
        height=int(height),
        params=values,
    )


def unique_point_matches(
    *,
    distances: np.ndarray,
    neighbor_ids: np.ndarray,
    point3d_ids: np.ndarray,
    source_image_ids: np.ndarray,
    ratio: float,
) -> list[dict[str, Any]]:
    """Apply a SIFT ratio test between different 3D points, then one-to-one."""
    ratio_squared = float(ratio) ** 2
    candidates: list[dict[str, Any]] = []
    for query_index, (query_distances, query_neighbors) in enumerate(
        zip(np.asarray(distances), np.asarray(neighbor_ids))
    ):
        # Faiss returns every row in ascending distance order. The first valid
        # vector is therefore the best observation of its 3D point; scan only
        # until the first *different* point rather than building a Python dict
        # over every neighbor. This preserves the grouped-point ratio test and
        # removes most of the per-frame matching overhead.
        best: tuple[float, int, int, int] | None = None
        second_distance: float | None = None
        for distance, vector_id in zip(query_distances, query_neighbors):
            vector_id = int(vector_id)
            distance = float(distance)
            if vector_id < 0 or vector_id >= len(point3d_ids) or not math.isfinite(distance):
                continue
            point_id = int(point3d_ids[vector_id])
            if best is None:
                best = (
                    distance,
                    point_id,
                    vector_id,
                    int(source_image_ids[vector_id]),
                )
                continue
            if point_id != best[1]:
                second_distance = distance
                break
        if best is None or second_distance is None:
            continue
        if best[0] > ratio_squared * max(second_distance, 1.0e-12):
            continue
        candidates.append(
            {
                "query_index": int(query_index),
                "point3d_id": int(best[1]),
                "vector_id": int(best[2]),
                "source_image_id": int(best[3]),
                "distance": float(best[0]),
                "second_point_distance": float(second_distance),
            }
        )
    # A 3D point must be assigned to only one query keypoint. Keep its strongest
    # descriptor match; query keypoints are already unique by construction.
    best_by_point: dict[int, dict[str, Any]] = {}
    for candidate in candidates:
        point_id = int(candidate["point3d_id"])
        previous = best_by_point.get(point_id)
        if previous is None or (
            float(candidate["distance"]), int(candidate["query_index"])
        ) < (
            float(previous["distance"]), int(previous["query_index"])
        ):
            best_by_point[point_id] = candidate
    return sorted(best_by_point.values(), key=lambda row: int(row["query_index"]))


def _distortion(camera: Camera) -> np.ndarray:
    model = camera.model.upper()
    params = np.asarray(camera.params, dtype=np.float64)
    if model == "SIMPLE_RADIAL" and len(params) >= 4:
        return np.array([params[3], 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if model == "RADIAL" and len(params) >= 5:
        return np.array([params[3], params[4], 0.0, 0.0, 0.0], dtype=np.float64)
    if model == "OPENCV" and len(params) >= 8:
        return np.array([params[4], params[5], params[6], params[7], 0.0], dtype=np.float64)
    if model == "FULL_OPENCV" and len(params) >= 12:
        return np.asarray(params[4:12], dtype=np.float64)
    return np.zeros((5,), dtype=np.float64)


def _rotmat_to_qvec(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = math.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) * 2.0
            quaternion = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        elif axis == 1:
            scale = math.sqrt(max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = math.sqrt(max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])) * 2.0
            quaternion = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=np.float64,
            )
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise ValueError("Invalid zero quaternion from rotation matrix")
    quaternion /= norm
    return -quaternion if quaternion[0] < 0.0 else quaternion


def _spread(xy: np.ndarray, width: int, height: int) -> dict[str, Any]:
    points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if len(points) < 4 or width <= 0 or height <= 0:
        return {"ok": False, "occupied_grid_cells": 0, "span_x_fraction": 0.0, "span_y_fraction": 0.0}
    normalized = points / np.array([float(width), float(height)], dtype=np.float64)
    span = np.ptp(normalized, axis=0)
    grid = np.clip(np.floor(normalized * 3.0).astype(int), 0, 2)
    occupied = len({(int(cell[0]), int(cell[1])) for cell in grid})
    return {
        "ok": bool(occupied >= 4 and float(span[0]) >= 0.20 and float(span[1]) >= 0.15),
        "occupied_grid_cells": int(occupied),
        "span_x_fraction": float(span[0]),
        "span_y_fraction": float(span[1]),
    }


def fixed_center_orientation_pool(
    *,
    image_name: str,
    image_id: int,
    camera: Camera,
    keypoints: np.ndarray,
    matches: list[dict[str, Any]],
    map_points: dict[int, Any],
    min_points: int,
    expected_center: np.ndarray,
    max_angle_degrees: float = 1.5,
    sample_count: int = 16000,
) -> dict[str, Any]:
    """Filter all-map SIFT matches at a controller-proved fixed position.

    At a verified patrol waypoint the aircraft is neutrally hovering and its
    metric center from the preceding lap is already known.  Translation is
    therefore not an unknown during correspondence verification: a correct
    3D point and image feature must produce world/camera rays related by one
    rotation.  Robustly estimate only that rotation to reject repeated-room
    aliases, then return the inlier 2D->3D set to TSolve.  This function never
    calls OpenCV PnP and its temporary rotation is not the published pose.
    """
    usable = [match for match in matches if int(match["point3d_id"]) in map_points]
    required = max(8, int(min_points))
    if len(usable) < required:
        return {
            "accepted": False,
            "reason": "faiss_fixed_center_too_few_unique_2d3d",
            "faiss_unique_matches": int(len(usable)),
        }
    xy = np.asarray(
        [keypoints[int(match["query_index"]), :2] for match in usable],
        dtype=np.float64,
    )
    pids = np.asarray([int(match["point3d_id"]) for match in usable], dtype=np.int64)
    p3d = np.asarray([map_points[int(point_id)].xyz for point_id in pids], dtype=np.float64)
    finite = np.all(np.isfinite(p3d), axis=1) & (np.max(np.abs(p3d), axis=1) < 1.0e6)
    xy = xy[finite]
    pids = pids[finite]
    p3d = p3d[finite]
    usable = [match for match, keep in zip(usable, finite) if bool(keep)]
    if len(usable) < required:
        return {
            "accepted": False,
            "reason": "faiss_fixed_center_too_few_finite_2d3d",
            "faiss_unique_matches": int(len(usable)),
        }

    center = np.asarray(expected_center, dtype=np.float64).reshape(3)
    K = np.asarray(camera.K(), dtype=np.float64)
    normalized_xy = cv2.undistortPoints(
        xy.reshape(-1, 1, 2),
        K,
        _distortion(camera),
    ).reshape(-1, 2)
    camera_rays = np.column_stack(
        [normalized_xy, np.ones(len(normalized_xy), dtype=np.float64)]
    )
    world_rays = p3d - center.reshape(1, 3)
    world_norms = np.linalg.norm(world_rays, axis=1)
    camera_norms = np.linalg.norm(camera_rays, axis=1)
    ray_valid = (
        np.all(np.isfinite(world_rays), axis=1)
        & np.all(np.isfinite(camera_rays), axis=1)
        & (world_norms > 1.0e-9)
        & (camera_norms > 1.0e-9)
    )
    xy = xy[ray_valid]
    pids = pids[ray_valid]
    p3d = p3d[ray_valid]
    usable = [match for match, keep in zip(usable, ray_valid) if bool(keep)]
    world = world_rays[ray_valid] / world_norms[ray_valid, None]
    camera_rays = camera_rays[ray_valid] / camera_norms[ray_valid, None]
    if len(world) < required:
        return {
            "accepted": False,
            "reason": "faiss_fixed_center_too_few_valid_rays",
            "faiss_unique_matches": int(len(world)),
        }

    cosine_threshold = math.cos(math.radians(max(0.1, float(max_angle_degrees))))
    rng = np.random.default_rng(20260823)
    best_rotation: np.ndarray | None = None
    best_indices = np.empty((0,), dtype=np.int64)
    best_rank: tuple[int, int, int, float] | None = None

    def fixed_spread_rank(indices: np.ndarray) -> tuple[int, int, float]:
        candidate_spread = _spread(xy[indices], camera.width, camera.height)
        occupied = int(candidate_spread.get("occupied_grid_cells") or 0)
        span_x = float(candidate_spread.get("span_x_fraction") or 0.0)
        span_y = float(candidate_spread.get("span_y_fraction") or 0.0)
        spread_ok = int(occupied >= 2 and span_x >= 0.20 and span_y >= 0.15)
        return spread_ok, occupied, span_x + span_y

    remaining = max(1, int(sample_count))
    while remaining > 0:
        batch_count = min(128, remaining)
        remaining -= batch_count
        samples = np.argpartition(
            rng.random((batch_count, len(world))),
            kth=2,
            axis=1,
        )[:, :3]
        covariance = np.einsum(
            "bni,bnj->bij",
            world[samples],
            camera_rays[samples],
        )
        try:
            left, _singular, right_t = np.linalg.svd(covariance)
        except np.linalg.LinAlgError:
            continue
        rotations = np.matmul(
            right_t.transpose(0, 2, 1),
            left.transpose(0, 2, 1),
        )
        reflected = np.linalg.det(rotations) < 0.0
        if np.any(reflected):
            right_t[reflected, -1, :] *= -1.0
            rotations[reflected] = np.matmul(
                right_t[reflected].transpose(0, 2, 1),
                left[reflected].transpose(0, 2, 1),
            )
        predicted = np.einsum("bij,nj->bni", rotations, world)
        agreement = np.einsum("bni,ni->bn", predicted, camera_rays)
        counts = np.sum(agreement >= cosine_threshold, axis=1)
        # Repeated wall texture can produce the numerically largest consensus
        # in one small image patch. Keep the strongest spatially constraining
        # hypothesis instead of discovering only after RANSAC that the winner
        # cannot constrain translation for TSolve.
        top_count = min(8, batch_count)
        top = np.argpartition(counts, -top_count)[-top_count:]
        for candidate_index in top:
            candidate_indices = np.flatnonzero(
                agreement[int(candidate_index)] >= cosine_threshold
            ).astype(np.int64)
            spread_ok, occupied, total_span = fixed_spread_rank(
                candidate_indices
            )
            candidate_rank = (
                int(spread_ok and len(candidate_indices) >= required),
                int(len(candidate_indices)),
                occupied,
                total_span,
            )
            if best_rank is None or candidate_rank > best_rank:
                best_rank = candidate_rank
                best_rotation = rotations[int(candidate_index)].copy()
                best_indices = candidate_indices

    if best_rotation is None or len(best_indices) < required:
        return {
            "accepted": False,
            "reason": "faiss_fixed_center_no_orientation_consensus",
            "faiss_unique_matches": int(len(world)),
            "faiss_fixed_center_inliers": int(len(best_indices)),
        }

    for _ in range(3):
        covariance = world[best_indices].T @ camera_rays[best_indices]
        try:
            left, _singular, right_t = np.linalg.svd(covariance)
        except np.linalg.LinAlgError:
            break
        refined = right_t.T @ left.T
        if float(np.linalg.det(refined)) < 0.0:
            right_t[-1, :] *= -1.0
            refined = right_t.T @ left.T
        best_rotation = refined
        agreement = np.sum((world @ best_rotation.T) * camera_rays, axis=1)
        updated = np.flatnonzero(agreement >= cosine_threshold).astype(np.int64)
        if np.array_equal(updated, best_indices):
            break
        best_indices = updated

    spread = _spread(xy[best_indices], camera.width, camera.height)
    fixed_center_spread_ok = bool(
        int(spread.get("occupied_grid_cells") or 0) >= 2
        and float(spread.get("span_x_fraction") or 0.0) >= 0.20
        and float(spread.get("span_y_fraction") or 0.0) >= 0.15
    )
    spread["fixed_center_ok"] = fixed_center_spread_ok
    agreement = np.sum(
        (world[best_indices] @ best_rotation.T) * camera_rays[best_indices],
        axis=1,
    )
    angular_errors = np.degrees(np.arccos(np.clip(agreement, -1.0, 1.0)))
    median_error = (
        float(np.median(angular_errors)) if len(angular_errors) else float("inf")
    )
    if len(best_indices) < required:
        reason = "faiss_fixed_center_refined_below_minimum"
    elif not fixed_center_spread_ok:
        reason = "faiss_fixed_center_low_spatial_concentration"
    elif not math.isfinite(median_error) or median_error > 1.0:
        reason = "faiss_fixed_center_angular_error_too_high"
    else:
        reason = ""
    if reason:
        return {
            "accepted": False,
            "reason": reason,
            "faiss_unique_matches": int(len(world)),
            "faiss_fixed_center_inliers": int(len(best_indices)),
            "faiss_fixed_center_median_angle_degrees": median_error,
            "correspondence_spread": spread,
        }

    source_ids = {
        int(usable[index]["source_image_id"]) for index in best_indices
    }
    translation = -best_rotation @ center
    return {
        "accepted": True,
        "image_name": image_name,
        "xy": xy[best_indices].astype(np.float32),
        "p3d": p3d[best_indices],
        "point3d_ids": pids[best_indices],
        "K": K,
        "colmap_image_id": int(image_id),
        "colmap_camera_id": int(camera.camera_id),
        "colmap_registered_points": int(len(best_indices)),
        "valid_2d3d": int(len(best_indices)),
        "faiss_unique_matches": int(len(world)),
        "faiss_fixed_center_inliers": int(len(best_indices)),
        "faiss_source_images": int(len(source_ids)),
        "faiss_fixed_center_median_angle_degrees": median_error,
        "faiss_fixed_center_hypotheses": int(sample_count),
        "correspondence_spread": spread,
        # This exact center is controller/previous-lap evidence used to select
        # correspondences. TSolve remains the only final R,t estimator.
        "colmap_qvec_world_to_camera": _rotmat_to_qvec(best_rotation).tolist(),
        "colmap_tvec_world_to_camera": translation.astype(float).tolist(),
        "pose_prior_center": center.astype(float).tolist(),
        "trusted_recovery": True,
        "tsolve_only_correspondences": True,
        "fixed_center_position_lock": True,
    }


def source_geometry_tsolve_pool(
    *,
    image_name: str,
    image_id: int,
    camera: Camera,
    keypoints: np.ndarray,
    matches: list[dict[str, Any]],
    neighbor_ids: np.ndarray,
    point3d_ids: np.ndarray,
    source_image_ids: np.ndarray,
    source_feature_indices: np.ndarray,
    map_images: dict[int, Any],
    map_points: dict[int, Any],
    min_points: int,
    max_source_hypotheses: int = 24,
) -> dict[str, Any]:
    """Verify ANN matches in 2D-to-2D source views, then hand them to TSolve.

    This is the position-unknown live path. A correct 3D point is normally
    observed in several nearby registered map images. For each source image we
    use its stored COLMAP feature coordinate and the current OpenCV SIFT
    coordinate to estimate a fundamental-matrix consensus. Candidates must be
    supported by several independent source views. This rejects descriptor
    aliases without estimating a camera pose and never calls PnP; TSolve is the
    only component that turns the resulting 2D-to-3D pool into R,t.
    """
    usable = [match for match in matches if int(match["point3d_id"]) in map_points]
    required = max(8, int(min_points))
    if len(usable) < required:
        return {
            "accepted": False,
            "reason": "faiss_tsolve_too_few_unique_2d3d",
            "faiss_unique_matches": int(len(usable)),
        }

    groups: dict[int, dict[int, np.ndarray]] = {}
    match_by_query = {int(match["query_index"]): match for match in usable}
    neighbors = np.asarray(neighbor_ids)
    for match in usable:
        query_index = int(match["query_index"])
        point_id = int(match["point3d_id"])
        if query_index < 0 or query_index >= len(neighbors):
            continue
        for vector_id_raw in neighbors[query_index]:
            vector_id = int(vector_id_raw)
            if (
                vector_id < 0
                or vector_id >= len(point3d_ids)
                or int(point3d_ids[vector_id]) != point_id
            ):
                continue
            source_id = int(source_image_ids[vector_id])
            feature_index = int(source_feature_indices[vector_id])
            source_image = map_images.get(source_id)
            if (
                source_image is None
                or feature_index < 0
                or feature_index >= len(source_image.xys)
                or feature_index >= len(source_image.point3d_ids)
                or int(source_image.point3d_ids[feature_index]) != point_id
            ):
                continue
            groups.setdefault(source_id, {})[query_index] = np.asarray(
                source_image.xys[feature_index], dtype=np.float32
            )

    ranked_groups = sorted(
        groups.items(),
        key=lambda row: (-len(row[1]), int(row[0])),
    )[: max(1, int(max_source_hypotheses))]
    votes: dict[int, int] = {}
    successful_sources = 0
    source_inliers: dict[int, int] = {}
    method = getattr(cv2, "USAC_MAGSAC", cv2.FM_RANSAC)
    for source_id, query_to_source in ranked_groups:
        rows = list(query_to_source.items())
        if len(rows) < 8:
            continue
        source_xy = np.asarray([row[1] for row in rows], dtype=np.float32)
        query_xy = np.asarray(
            [keypoints[int(row[0]), :2] for row in rows], dtype=np.float32
        )
        try:
            _fundamental, mask = cv2.findFundamentalMat(
                source_xy,
                query_xy,
                method,
                1.5,
                0.999,
                2000,
            )
        except cv2.error:
            continue
        if mask is None:
            continue
        inliers = np.flatnonzero(np.asarray(mask).reshape(-1) > 0)
        if len(inliers) < 8 or float(len(inliers)) / float(len(rows)) < 0.55:
            continue
        successful_sources += 1
        source_inliers[int(source_id)] = int(len(inliers))
        for local_index in inliers:
            query_index = int(rows[int(local_index)][0])
            votes[query_index] = votes.get(query_index, 0) + 1

    vote_threshold = 3 if successful_sources >= 6 else (2 if successful_sources >= 3 else 1)
    selected_matches = [
        match_by_query[query_index]
        for query_index, count in votes.items()
        if count >= vote_threshold and query_index in match_by_query
    ]
    selected_matches.sort(key=lambda row: int(row["query_index"]))
    if len(selected_matches) < required:
        return {
            "accepted": False,
            "reason": "faiss_tsolve_source_geometry_below_minimum",
            "faiss_unique_matches": int(len(usable)),
            "faiss_source_geometry_inliers": int(len(selected_matches)),
            "faiss_source_geometry_hypotheses": int(successful_sources),
            "faiss_source_vote_threshold": int(vote_threshold),
        }

    xy = np.asarray(
        [keypoints[int(match["query_index"]), :2] for match in selected_matches],
        dtype=np.float32,
    )
    pids = np.asarray(
        [int(match["point3d_id"]) for match in selected_matches], dtype=np.int64
    )
    p3d = np.asarray([map_points[int(point_id)].xyz for point_id in pids], dtype=np.float64)
    finite = np.all(np.isfinite(p3d), axis=1) & (np.max(np.abs(p3d), axis=1) < 1.0e6)
    xy = xy[finite]
    pids = pids[finite]
    p3d = p3d[finite]
    if len(xy) < required:
        return {
            "accepted": False,
            "reason": "faiss_tsolve_source_geometry_nonfinite",
            "faiss_unique_matches": int(len(usable)),
            "faiss_source_geometry_inliers": int(len(xy)),
        }
    spread = _spread(xy, camera.width, camera.height)
    if not bool(spread.get("ok")):
        return {
            "accepted": False,
            "reason": "faiss_tsolve_source_geometry_low_spread",
            "faiss_unique_matches": int(len(usable)),
            "faiss_source_geometry_inliers": int(len(xy)),
            "correspondence_spread": spread,
        }
    return {
        "accepted": True,
        "image_name": image_name,
        "xy": xy,
        "p3d": p3d,
        "point3d_ids": pids,
        "K": np.asarray(camera.K(), dtype=np.float64),
        "colmap_image_id": int(image_id),
        "colmap_camera_id": int(camera.camera_id),
        "colmap_registered_points": int(len(xy)),
        "valid_2d3d": int(len(xy)),
        "faiss_unique_matches": int(len(usable)),
        "faiss_source_images": int(successful_sources),
        "faiss_source_geometry_inliers": int(len(xy)),
        "faiss_source_geometry_hypotheses": int(successful_sources),
        "faiss_source_vote_threshold": int(vote_threshold),
        "faiss_source_inliers": source_inliers,
        "correspondence_spread": spread,
        "trusted_recovery": True,
        "tsolve_only_correspondences": True,
        "ann_geometric_verification": "multi_source_fundamental_consensus",
    }


def verified_pnp_pool(
    *,
    image_name: str,
    image_id: int,
    camera: Camera,
    keypoints: np.ndarray,
    matches: list[dict[str, Any]],
    map_points: dict[int, Any],
    min_points: int,
    expected_center: np.ndarray | None,
    reprojection_error: float,
    position_locked: bool = False,
) -> dict[str, Any]:
    if bool(position_locked) and expected_center is not None:
        return fixed_center_orientation_pool(
            image_name=image_name,
            image_id=image_id,
            camera=camera,
            keypoints=keypoints,
            matches=matches,
            map_points=map_points,
            min_points=min_points,
            expected_center=np.asarray(expected_center, dtype=np.float64),
        )
    usable = [match for match in matches if int(match["point3d_id"]) in map_points]
    required = max(8, int(min_points))
    if len(usable) < required:
        return {
            "accepted": False,
            "reason": "faiss_too_few_unique_2d3d",
            "faiss_unique_matches": int(len(usable)),
        }
    xy = np.asarray(
        [keypoints[int(match["query_index"]), :2] for match in usable],
        dtype=np.float64,
    )
    pids = np.asarray([int(match["point3d_id"]) for match in usable], dtype=np.int64)
    p3d = np.asarray([map_points[int(point_id)].xyz for point_id in pids], dtype=np.float64)
    finite_geometry = np.all(np.isfinite(p3d), axis=1) & (
        np.max(np.abs(p3d), axis=1) < 1.0e6
    )
    if not np.all(finite_geometry):
        xy = xy[finite_geometry]
        pids = pids[finite_geometry]
        p3d = p3d[finite_geometry]
        usable = [match for match, keep in zip(usable, finite_geometry) if bool(keep)]
    if len(usable) < required:
        return {
            "accepted": False,
            "reason": "faiss_too_few_finite_2d3d",
            "faiss_unique_matches": int(len(usable)),
        }
    K = np.asarray(camera.K(), dtype=np.float64)
    distortion = _distortion(camera)
    expected = None
    if expected_center is not None:
        candidate = np.asarray(expected_center, dtype=np.float64).reshape(3)
        if np.all(np.isfinite(candidate)):
            expected = candidate
    source_ids = np.asarray([int(match["source_image_id"]) for match in usable], dtype=np.int64)
    source_counts: dict[int, int] = {}
    for source_id in source_ids:
        source_counts[int(source_id)] = source_counts.get(int(source_id), 0) + 1
    hypotheses: list[tuple[str, np.ndarray]] = [("global", np.arange(len(xy), dtype=np.int64))]
    for source_id, count in sorted(source_counts.items(), key=lambda row: (-row[1], row[0]))[:8]:
        if count >= required:
            hypotheses.append(
                (f"source_image_{source_id}", np.flatnonzero(source_ids == source_id).astype(np.int64))
            )

    solutions: list[dict[str, Any]] = []
    most_inliers = 0
    for hypothesis_name, hypothesis_indices in hypotheses:
        object_points = p3d[hypothesis_indices]
        image_points = xy[hypothesis_indices]
        seeds = 8 if hypothesis_name == "global" else 3
        for seed in range(seeds):
            cv2.setRNGSeed(2903 + seed)
            solved, rvec, tvec, initial_inliers = cv2.solvePnPRansac(
                object_points,
                image_points,
                K,
                distortion,
                iterationsCount=1200,
                reprojectionError=float(reprojection_error),
                confidence=0.999,
                flags=cv2.SOLVEPNP_EPNP,
            )
            if not solved or initial_inliers is None:
                continue
            local_indices = np.asarray(initial_inliers, dtype=np.int64).reshape(-1)
            most_inliers = max(most_inliers, int(len(local_indices)))
            if len(local_indices) < required:
                continue
            refined, rvec, tvec = cv2.solvePnP(
                object_points[local_indices],
                image_points[local_indices],
                K,
                distortion,
                rvec,
                tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not refined or not (
                np.all(np.isfinite(rvec)) and np.all(np.isfinite(tvec))
            ):
                continue
            if float(np.max(np.abs(tvec))) >= 1.0e6:
                continue
            projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, distortion)
            errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
            local_indices = np.flatnonzero(
                np.isfinite(errors) & (errors <= float(reprojection_error))
            ).astype(np.int64)
            most_inliers = max(most_inliers, int(len(local_indices)))
            if len(local_indices) < required:
                continue
            global_indices = hypothesis_indices[local_indices]
            spread = _spread(xy[global_indices], camera.width, camera.height)
            if not spread["ok"]:
                continue
            rotation, _ = cv2.Rodrigues(rvec)
            translation = np.asarray(tvec, dtype=np.float64).reshape(3)
            if not (
                np.all(np.isfinite(rotation)) and np.all(np.isfinite(translation))
            ):
                continue
            if float(np.max(np.abs(rotation))) > 1.000001:
                continue
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                camera_points = (rotation @ p3d[global_indices].T).T + translation
            if not np.all(np.isfinite(camera_points)):
                continue
            if float(np.mean(camera_points[:, 2] > 1.0e-6)) < 0.95:
                continue
            center = -rotation.T @ translation
            solutions.append(
                {
                    "hypothesis": hypothesis_name,
                    "R": rotation,
                    "t": translation,
                    "center": center,
                    "indices": global_indices,
                    "inliers": int(len(global_indices)),
                    "median_error": float(np.median(errors[local_indices])),
                    "center_step": (
                        float(np.linalg.norm(center - expected)) if expected is not None else 0.0
                    ),
                    "spread": spread,
                }
            )
    if not solutions:
        return {
            "accepted": False,
            "reason": "faiss_pnp_geometric_verification_failed",
            "faiss_unique_matches": int(len(usable)),
            "faiss_pnp_inliers": int(most_inliers),
            "faiss_source_images": int(len(source_counts)),
        }
    best_inliers = max(int(solution["inliers"]) for solution in solutions)
    credible = [
        solution
        for solution in solutions
        if int(solution["inliers"]) >= max(required, int(math.ceil(best_inliers * 0.75)))
    ]
    if expected is not None:
        selected = min(
            credible,
            key=lambda solution: (
                float(solution["center_step"]),
                -int(solution["inliers"]),
                float(solution["median_error"]),
            ),
        )
    else:
        selected = min(
            credible,
            key=lambda solution: (
                -int(solution["inliers"]),
                float(solution["median_error"]),
            ),
        )
    indices = np.asarray(selected["indices"], dtype=np.int64)
    return {
        "accepted": True,
        "image_name": image_name,
        "xy": xy[indices].astype(np.float32),
        "p3d": p3d[indices],
        "point3d_ids": pids[indices],
        "K": K,
        "colmap_image_id": int(image_id),
        "colmap_camera_id": int(camera.camera_id),
        "colmap_registered_points": int(len(indices)),
        "valid_2d3d": int(len(indices)),
        "faiss_unique_matches": int(len(usable)),
        "faiss_pnp_inliers": int(len(indices)),
        "faiss_source_images": int(len(source_counts)),
        "faiss_pnp_hypotheses": int(len(solutions)),
        "faiss_selected_hypothesis": str(selected["hypothesis"]),
        "faiss_pnp_center_step": float(selected["center_step"]),
        "faiss_median_reprojection_error": float(selected["median_error"]),
        "correspondence_spread": selected["spread"],
        "colmap_qvec_world_to_camera": _rotmat_to_qvec(selected["R"]).tolist(),
        "colmap_tvec_world_to_camera": np.asarray(selected["t"], dtype=float).tolist(),
    }


class FaissIVF3DRelocalizer:
    def __init__(
        self,
        index_dir: Path,
        *,
        nprobe: int = 8,
        top_k: int = 48,
        ratio: float = 0.86,
        min_points: int = 40,
        reprojection_error: float = 6.0,
        opencv_sift_max_features: int = 2400,
        opencv_sift_n_octave_layers: int = 3,
        opencv_sift_contrast_threshold: float = 0.02,
        opencv_sift_edge_threshold: float = 12.0,
        opencv_sift_sigma: float = 1.6,
    ) -> None:
        faiss = require_faiss()
        self.index_dir = index_dir.resolve()
        if not index_is_current(self.index_dir):
            raise FileNotFoundError(f"Incomplete Faiss map index: {self.index_dir}")
        self.manifest = json.loads(
            (self.index_dir / "manifest.json").read_text(encoding="utf-8")
        )
        self.index = faiss.read_index(str(self.index_dir / "index.faiss"))
        self.point3d_ids = np.load(
            self.index_dir / "point3d_ids.npy", mmap_mode="r", allow_pickle=False
        )
        self.source_image_ids = np.load(
            self.index_dir / "source_image_ids.npy", mmap_mode="r", allow_pickle=False
        )
        self.source_feature_indices = np.load(
            self.index_dir / "source_feature_indices.npy", mmap_mode="r", allow_pickle=False
        )
        count = int(self.index.ntotal)
        if not (
            len(self.point3d_ids) == count
            and len(self.source_image_ids) == count
            and len(self.source_feature_indices) == count
        ):
            raise RuntimeError("Faiss index and 3D metadata lengths differ.")
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = min(max(1, int(nprobe)), int(self.index.nlist))
        self.top_k = max(2, int(top_k))
        self.ratio = min(0.99, max(0.1, float(ratio)))
        self.min_points = max(8, int(min_points))
        self.reprojection_error = max(1.0, float(reprojection_error))
        self.opencv_sift_max_features = max(64, int(opencv_sift_max_features))
        self.opencv_sift_n_octave_layers = max(1, int(opencv_sift_n_octave_layers))
        self.opencv_sift_contrast_threshold = max(
            1.0e-5, float(opencv_sift_contrast_threshold)
        )
        self.opencv_sift_edge_threshold = max(1.0, float(opencv_sift_edge_threshold))
        self.opencv_sift_sigma = max(0.1, float(opencv_sift_sigma))

    def localize_frame(
        self,
        *,
        image_path: Path,
        image_name: str,
        camera_model: str,
        camera_params: str | list[float] | np.ndarray,
        map_images: dict[int, Any],
        map_points: dict[int, Any],
        expected_center: np.ndarray | None,
        position_locked: bool = False,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """OpenCV-SIFT -> Faiss-IVF -> TSolve-correspondence live path."""
        started = time.perf_counter()
        diagnostic: dict[str, Any] = {
            "feature_extract_ms": 0.0,
            "search_ms": 0.0,
            "match_filter_ms": 0.0,
            "geometric_verification_ms": 0.0,
            "pnp_ms": 0.0,
            "reason": "",
            "feature_backend": "opencv_sift",
            "ann_backend": "faiss_ivf_sq8_l2",
            "pose_backend": "tsolve_only",
            "pnp_ransac_used": False,
        }
        try:
            feature_started = time.perf_counter()
            keypoints, descriptors, width, height = opencv_sift_features(
                image_path,
                max_features=self.opencv_sift_max_features,
                n_octave_layers=self.opencv_sift_n_octave_layers,
                contrast_threshold=self.opencv_sift_contrast_threshold,
                edge_threshold=self.opencv_sift_edge_threshold,
                sigma=self.opencv_sift_sigma,
            )
            diagnostic["feature_extract_ms"] = 1000.0 * (
                time.perf_counter() - feature_started
            )
            camera = calibrated_frame_camera(
                model=camera_model,
                params=camera_params,
                width=width,
                height=height,
            )
            search_started = time.perf_counter()
            distances, neighbors = self.index.search(descriptors, self.top_k)
            diagnostic["search_ms"] = 1000.0 * (time.perf_counter() - search_started)
            filter_started = time.perf_counter()
            matches = unique_point_matches(
                distances=distances,
                neighbor_ids=neighbors,
                point3d_ids=self.point3d_ids,
                source_image_ids=self.source_image_ids,
                ratio=self.ratio,
            )
            diagnostic["match_filter_ms"] = 1000.0 * (
                time.perf_counter() - filter_started
            )
            verify_started = time.perf_counter()
            if bool(position_locked) and expected_center is not None:
                pool = fixed_center_orientation_pool(
                    image_name=image_name,
                    image_id=-1,
                    camera=camera,
                    keypoints=keypoints,
                    matches=matches,
                    map_points=map_points,
                    min_points=self.min_points,
                    expected_center=np.asarray(expected_center, dtype=np.float64),
                    sample_count=8000,
                )
                verification_profile = "fixed_center_rotation_consensus_to_tsolve"
            else:
                pool = source_geometry_tsolve_pool(
                    image_name=image_name,
                    image_id=-1,
                    camera=camera,
                    keypoints=keypoints,
                    matches=matches,
                    neighbor_ids=neighbors,
                    point3d_ids=self.point3d_ids,
                    source_image_ids=self.source_image_ids,
                    source_feature_indices=self.source_feature_indices,
                    map_images=map_images,
                    map_points=map_points,
                    min_points=self.min_points,
                )
                verification_profile = "multi_source_2d_geometry_to_tsolve"
            diagnostic["geometric_verification_ms"] = 1000.0 * (
                time.perf_counter() - verify_started
            )
            diagnostic["verification_profile"] = verification_profile
            diagnostic["query_descriptors"] = int(len(descriptors))
            diagnostic["unique_2d3d_candidates"] = int(len(matches))
            diagnostic["index_vectors"] = int(self.index.ntotal)
            diagnostic["nprobe"] = int(self.index.nprobe)
            diagnostic["top_k"] = int(self.top_k)
            diagnostic["ratio"] = float(self.ratio)
            diagnostic["total_ms"] = 1000.0 * (time.perf_counter() - started)
            if not pool.get("accepted"):
                diagnostic["reason"] = str(
                    pool.get("reason") or "faiss_tsolve_relocalization_failed"
                )
                diagnostic.update(
                    {key: value for key, value in pool.items() if key != "accepted"}
                )
                return None, diagnostic
            pool["faiss_query_descriptors"] = int(len(descriptors))
            pool["faiss_index_vectors"] = int(self.index.ntotal)
            pool["localization_method"] = "opencv_sift_faiss_ivf_to_tsolve"
            return pool, diagnostic
        except (RuntimeError, ValueError, cv2.error) as exc:
            diagnostic["reason"] = f"faiss_tsolve_relocalization_error:{exc}"
            diagnostic["total_ms"] = 1000.0 * (time.perf_counter() - started)
            return None, diagnostic

    def localize(
        self,
        *,
        database_path: Path,
        image_name: str,
        map_points: dict[int, Any],
        expected_center: np.ndarray | None,
        position_locked: bool = False,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        started = time.perf_counter()
        diagnostic: dict[str, Any] = {
            "feature_read_ms": 0.0,
            "search_ms": 0.0,
            "pnp_ms": 0.0,
            "reason": "",
        }
        try:
            read_started = time.perf_counter()
            image_id, camera, keypoints, descriptors = _query_features(
                database_path, image_name
            )
            diagnostic["feature_read_ms"] = 1000.0 * (time.perf_counter() - read_started)
            search_started = time.perf_counter()
            distances, neighbors = self.index.search(
                np.ascontiguousarray(descriptors, dtype=np.float32),
                self.top_k,
            )
            diagnostic["search_ms"] = 1000.0 * (time.perf_counter() - search_started)
            filter_started = time.perf_counter()
            matches = unique_point_matches(
                distances=distances,
                neighbor_ids=neighbors,
                point3d_ids=self.point3d_ids,
                source_image_ids=self.source_image_ids,
                ratio=self.ratio,
            )
            diagnostic["match_filter_ms"] = 1000.0 * (
                time.perf_counter() - filter_started
            )
            pnp_started = time.perf_counter()
            pool = verified_pnp_pool(
                image_name=image_name,
                image_id=image_id,
                camera=camera,
                keypoints=keypoints,
                matches=matches,
                map_points=map_points,
                min_points=self.min_points,
                expected_center=expected_center,
                reprojection_error=self.reprojection_error,
                position_locked=position_locked,
            )
            verification_ms = 1000.0 * (time.perf_counter() - pnp_started)
            diagnostic["pnp_ms"] = 0.0 if position_locked else verification_ms
            diagnostic["fixed_center_verification_ms"] = (
                verification_ms if position_locked else 0.0
            )
            diagnostic["verification_profile"] = (
                "fixed_center_rotation_consensus_to_tsolve"
                if position_locked
                else "pnp_ransac"
            )
            diagnostic["query_descriptors"] = int(len(descriptors))
            diagnostic["unique_2d3d_candidates"] = int(len(matches))
            diagnostic["index_vectors"] = int(self.index.ntotal)
            diagnostic["total_ms"] = 1000.0 * (time.perf_counter() - started)
            if not pool.get("accepted"):
                diagnostic["reason"] = str(pool.get("reason") or "faiss_relocalization_failed")
                diagnostic.update({key: value for key, value in pool.items() if key != "accepted"})
                return None, diagnostic
            pool["faiss_query_descriptors"] = int(len(descriptors))
            pool["faiss_index_vectors"] = int(self.index.ntotal)
            return pool, diagnostic
        except (RuntimeError, sqlite3.Error, ValueError, cv2.error) as exc:
            diagnostic["reason"] = f"faiss_relocalization_error:{exc}"
            diagnostic["total_ms"] = 1000.0 * (time.perf_counter() - started)
            return None, diagnostic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--map-sparse-model", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--nlist", type=int, default=4096)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--training-sample-size", type=int, default=262144)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = build_faiss_index(
        database=args.database,
        sparse_model=args.map_sparse_model,
        out_dir=args.out_dir,
        nlist=args.nlist,
        nprobe=args.nprobe,
        training_sample_size=args.training_sample_size,
        force=args.force,
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
