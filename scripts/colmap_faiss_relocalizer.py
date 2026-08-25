#!/usr/bin/env python3
"""Global COLMAP SIFT-to-3D retrieval with a persistent Faiss IVF index.

The index contains every SIFT descriptor observation in the registered COLMAP
model that is attached to a 3D point.  Untriangulated database descriptors are
intentionally excluded because they cannot create a PnP/TSolve correspondence.

At runtime the current query descriptors search the complete map index.  ANN
neighbors are grouped by 3D point before the ratio test, made one-to-one, and
then verified by calibrated PnP RANSAC.  Faiss similarity alone never produces
an accepted pose.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from colmap_io import CAMERA_MODEL_BY_ID, Camera, read_images_model, read_points3d_text


INDEX_FORMAT_VERSION = 1
DESCRIPTOR_DIMENSION = 128


def opencv_sift_to_colmap_descriptors(descriptors: np.ndarray) -> np.ndarray:
    """Convert OpenCV SIFT rows to COLMAP's default uint8 RootSIFT format.

    OpenCV returns the conventional L2-normalized SIFT histogram scaled by
    512. COLMAP's default ``l1_root`` normalization instead L1-normalizes the
    non-negative histogram, takes its square root, scales by 512, and stores
    uint8 values. The persistent Faiss map index contains that COLMAP format,
    so query descriptors must use the same representation.
    """
    rows = np.asarray(descriptors, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != DESCRIPTOR_DIMENSION:
        raise ValueError(
            f"Expected SIFT descriptors with shape (N, {DESCRIPTOR_DIMENSION}), "
            f"got {rows.shape}"
        )
    if len(rows) == 0:
        return np.empty((0, DESCRIPTOR_DIMENSION), dtype=np.uint8)
    if not np.all(np.isfinite(rows)):
        raise ValueError("OpenCV SIFT descriptors contain non-finite values")
    nonnegative = np.maximum(rows, 0.0)
    l1 = np.sum(nonnegative, axis=1, keepdims=True)
    rooted = np.sqrt(nonnegative / np.maximum(l1, np.finfo(np.float32).eps))
    return np.clip(np.rint(rooted * 512.0), 0.0, 255.0).astype(np.uint8)


class OpenCVSiftFeatureExtractor:
    """Reusable in-process query extractor compatible with a COLMAP SIFT map."""

    def __init__(
        self,
        *,
        max_num_features: int = 1024,
        max_image_size: int = 1200,
    ) -> None:
        self.max_num_features = max(64, int(max_num_features))
        self.max_image_size = int(max_image_size)
        # COLMAP's CPU defaults are 3 octave layers, edge threshold 10, sigma
        # 1.6 and peak threshold 0.0066667. OpenCV divides contrastThreshold by
        # nOctaveLayers internally, hence 0.02 / 3 gives the same threshold.
        self._sift = cv2.SIFT_create(
            nfeatures=self.max_num_features,
            nOctaveLayers=3,
            contrastThreshold=0.02,
            edgeThreshold=10.0,
            sigma=1.6,
        )
        self._lock = threading.Lock()

    def extract(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image = np.asarray(gray)
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim != 2 or image.size == 0:
            raise ValueError("OpenCV SIFT requires a non-empty grayscale image")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        height, width = image.shape
        scale = 1.0
        if self.max_image_size > 0 and max(width, height) > self.max_image_size:
            scale = float(self.max_image_size) / float(max(width, height))
            image = cv2.resize(
                image,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        # A periodic recovery may still be finishing when a stopped checkpoint
        # requests a current-frame solve. OpenCV does not promise that one SIFT
        # instance is re-entrant, so serialize only the extraction call.
        with self._lock:
            keypoints, descriptors = self._sift.detectAndCompute(image, None)
        if descriptors is None or not keypoints:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, DESCRIPTOR_DIMENSION), dtype=np.uint8),
            )
        xy = np.asarray([keypoint.pt for keypoint in keypoints], dtype=np.float32)
        if scale != 1.0:
            xy /= np.float32(scale)
        return xy, opencv_sift_to_colmap_descriptors(descriptors)


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
) -> dict[str, Any]:
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
        nprobe: int = 32,
        top_k: int = 32,
        ratio: float = 0.80,
        min_points: int = 40,
        reprojection_error: float = 6.0,
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

    def localize(
        self,
        *,
        database_path: Path,
        image_name: str,
        map_points: dict[int, Any],
        expected_center: np.ndarray | None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        started = time.perf_counter()
        try:
            read_started = time.perf_counter()
            image_id, camera, keypoints, descriptors = _query_features(
                database_path, image_name
            )
            feature_read_ms = 1000.0 * (time.perf_counter() - read_started)
            pool, diagnostic = self.localize_features(
                image_name=image_name,
                image_id=image_id,
                camera=camera,
                keypoints=keypoints,
                descriptors=descriptors,
                map_points=map_points,
                expected_center=expected_center,
            )
            diagnostic["feature_read_ms"] = feature_read_ms
            diagnostic["total_ms"] = 1000.0 * (time.perf_counter() - started)
            return pool, diagnostic
        except (RuntimeError, sqlite3.Error, ValueError, cv2.error) as exc:
            return None, {
                "feature_read_ms": 0.0,
                "search_ms": 0.0,
                "match_filter_ms": 0.0,
                "pnp_ms": 0.0,
                "reason": f"faiss_relocalization_error:{exc}",
                "total_ms": 1000.0 * (time.perf_counter() - started),
            }

    def localize_features(
        self,
        *,
        image_name: str,
        image_id: int,
        camera: Camera,
        keypoints: np.ndarray,
        descriptors: np.ndarray,
        map_points: dict[int, Any],
        expected_center: np.ndarray | None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Search already-extracted query features without a COLMAP database."""
        started = time.perf_counter()
        diagnostic: dict[str, Any] = {
            "feature_read_ms": 0.0,
            "search_ms": 0.0,
            "match_filter_ms": 0.0,
            "pnp_ms": 0.0,
            "reason": "",
        }
        try:
            xy = np.asarray(keypoints, dtype=np.float32)
            rows = np.asarray(descriptors)
            if xy.ndim != 2 or xy.shape[1] < 2:
                raise ValueError(f"Invalid query keypoint shape: {xy.shape}")
            if rows.ndim != 2 or rows.shape[1] != DESCRIPTOR_DIMENSION:
                raise ValueError(f"Invalid query descriptor shape: {rows.shape}")
            usable = min(len(xy), len(rows))
            if usable <= 0:
                raise ValueError("No query SIFT features")
            xy = np.ascontiguousarray(xy[:usable, :2], dtype=np.float32)
            rows = np.ascontiguousarray(rows[:usable], dtype=np.float32)
            if not np.all(np.isfinite(xy)) or not np.all(np.isfinite(rows)):
                raise ValueError("Query SIFT features contain non-finite values")

            search_started = time.perf_counter()
            distances, neighbors = self.index.search(rows, self.top_k)
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
                keypoints=xy,
                matches=matches,
                map_points=map_points,
                min_points=self.min_points,
                expected_center=expected_center,
                reprojection_error=self.reprojection_error,
            )
            diagnostic["pnp_ms"] = 1000.0 * (time.perf_counter() - pnp_started)
            diagnostic["query_descriptors"] = int(usable)
            diagnostic["unique_2d3d_candidates"] = int(len(matches))
            diagnostic["index_vectors"] = int(self.index.ntotal)
            diagnostic["total_ms"] = 1000.0 * (time.perf_counter() - started)
            if not pool.get("accepted"):
                diagnostic["reason"] = str(pool.get("reason") or "faiss_relocalization_failed")
                diagnostic.update({key: value for key, value in pool.items() if key != "accepted"})
                return None, diagnostic
            pool["faiss_query_descriptors"] = int(usable)
            pool["faiss_index_vectors"] = int(self.index.ntotal)
            return pool, diagnostic
        except (RuntimeError, ValueError, cv2.error) as exc:
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
