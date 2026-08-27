#!/usr/bin/env python3
"""Align one ATLAS COLMAP map to a preserved reference map without mutating either model.

The source images are independently localized against the reference map's
persistent SIFT/Faiss 2D-to-3D index.  Common camera centers and orientations
then estimate a robust Sim(3) from source COLMAP coordinates to reference
COLMAP coordinates.  The resulting transform can be composed with the
reference map's explicit room alignment and written only to the source map's
manifest entry.

The sparse models and reference manifest entry are always read-only.  Use
``--apply`` only after the audit report passes its quality gates.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from colmap_faiss_relocalizer import FaissIVF3DRelocalizer  # noqa: E402
from colmap_io import (  # noqa: E402
    camera_center,
    qvec_to_rotmat,
    read_cameras_model,
    read_images_model,
    read_points3d_text,
)


def _map_entry(manifest: dict[str, Any], map_id: str) -> dict[str, Any]:
    entry = next(
        (item for item in manifest.get("maps", []) if item.get("id") == map_id),
        None,
    )
    if not isinstance(entry, dict):
        raise RuntimeError(f"Map not found in manifest: {map_id}")
    return entry


def _colmap_root(entry: dict[str, Any]) -> Path:
    configured = (entry.get("validation") or {}).get("colmap_root")
    if configured:
        return Path(configured).expanduser().resolve()
    return (ROOT / "results" / "maps" / str(entry["id"]) / "colmap").resolve()


def _sparse_model(root: Path) -> Path:
    candidates = [root / "sparse" / "0", root / "sparse", root / "dense" / "sparse"]
    for candidate in candidates:
        if (candidate / "images.bin").exists() or (candidate / "images.txt").exists():
            return candidate
    raise FileNotFoundError(f"No sparse COLMAP model below {root}")


def _sparse_text(root: Path) -> Path:
    candidate = root / "sparse_text"
    if (candidate / "points3D.txt").exists():
        return candidate
    raise FileNotFoundError(f"Missing sparse text export below {root}")


def _rotation_angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    relative = left @ right.T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _camera_center_from_pose(qvec: Any, tvec: Any) -> np.ndarray:
    rotation = qvec_to_rotmat(np.asarray(qvec, dtype=float))
    translation = np.asarray(tvec, dtype=float).reshape(3)
    return -rotation.T @ translation


def _umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if len(source) < 3 or source.shape != target.shape or source.shape[1] != 3:
        raise ValueError("A Sim(3) needs at least three paired 3D points")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(3, dtype=float)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt
    variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    if variance <= 1.0e-12:
        raise ValueError("Source camera centers are degenerate")
    scale = float(np.sum(singular * sign) / variance)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Estimated map scale is invalid")
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def _fit_robust_sim3(
    observations: list[dict[str, Any]],
    *,
    max_position_error: float,
    max_rotation_error_deg: float,
) -> tuple[dict[str, Any], list[int]]:
    source = np.asarray([item["source_center"] for item in observations], dtype=float)
    target = np.asarray([item["reference_center"] for item in observations], dtype=float)
    orientation = [np.asarray(item["orientation_transform"], dtype=float) for item in observations]
    if len(source) < 3:
        raise RuntimeError("Fewer than three shared localized cameras")

    combinations = list(itertools.combinations(range(len(source)), 3))
    if len(combinations) > 5000:
        rng = np.random.default_rng(20260823)
        selected = rng.choice(len(combinations), size=5000, replace=False)
        combinations = [combinations[int(index)] for index in selected]

    best: tuple[tuple[int, float, float], tuple[float, np.ndarray, np.ndarray], list[int]] | None = None
    for triple in combinations:
        try:
            candidate = _umeyama(source[list(triple)], target[list(triple)])
        except (ValueError, np.linalg.LinAlgError):
            continue
        scale, rotation, translation = candidate
        predicted = (scale * (rotation @ source.T)).T + translation
        position_errors = np.linalg.norm(predicted - target, axis=1)
        rotation_errors = np.asarray(
            [_rotation_angle_degrees(rotation, item) for item in orientation],
            dtype=float,
        )
        inliers = np.flatnonzero(
            (position_errors <= max_position_error)
            & (rotation_errors <= max_rotation_error_deg)
        ).tolist()
        if len(inliers) < 3:
            continue
        score = (
            len(inliers),
            -float(np.median(position_errors[inliers])),
            -float(np.median(rotation_errors[inliers])),
        )
        if best is None or score > best[0]:
            best = (score, candidate, inliers)
    if best is None:
        raise RuntimeError("No consistent source-to-reference Sim(3) hypothesis")

    inliers = best[2]
    for _ in range(4):
        scale, rotation, translation = _umeyama(source[inliers], target[inliers])
        predicted = (scale * (rotation @ source.T)).T + translation
        position_errors = np.linalg.norm(predicted - target, axis=1)
        rotation_errors = np.asarray(
            [_rotation_angle_degrees(rotation, item) for item in orientation],
            dtype=float,
        )
        updated = np.flatnonzero(
            (position_errors <= max_position_error)
            & (rotation_errors <= max_rotation_error_deg)
        ).tolist()
        if updated == inliers or len(updated) < 3:
            break
        inliers = updated

    scale, rotation, translation = _umeyama(source[inliers], target[inliers])
    predicted = (scale * (rotation @ source.T)).T + translation
    position_errors = np.linalg.norm(predicted - target, axis=1)
    rotation_errors = np.asarray(
        [_rotation_angle_degrees(rotation, item) for item in orientation],
        dtype=float,
    )
    inlier_positions = position_errors[inliers]
    inlier_rotations = rotation_errors[inliers]
    return {
        "scale": scale,
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "matrix_3x4": np.column_stack([scale * rotation, translation]).tolist(),
        "position_error_m": {
            "median": float(np.median(inlier_positions)),
            "p90": float(np.quantile(inlier_positions, 0.90)),
            "max": float(np.max(inlier_positions)),
        },
        "rotation_error_deg": {
            "median": float(np.median(inlier_rotations)),
            "p90": float(np.quantile(inlier_rotations, 0.90)),
            "max": float(np.max(inlier_rotations)),
        },
    }, inliers


def _compose_room_alignment(
    reference_room_matrix: list[list[float]],
    sim3_matrix: list[list[float]],
) -> list[list[float]]:
    reference = np.asarray(reference_room_matrix, dtype=float)
    sim3 = np.asarray(sim3_matrix, dtype=float)
    if reference.shape != (3, 4) or sim3.shape != (3, 4):
        raise ValueError("Both room and Sim(3) matrices must be 3x4")
    homogeneous = np.eye(4, dtype=float)
    homogeneous[:3, :] = sim3
    return (reference @ homogeneous).tolist()


def _run_feature_extractor(
    *,
    colmap: Path,
    database: Path,
    image_root: Path,
    image_name: str,
    image_list: Path,
    camera_model: str,
    camera_params: str,
    max_image_size: int,
    max_features: int,
) -> None:
    image_list.write_text(image_name + "\n", encoding="utf-8")
    command = [
        str(colmap),
        "feature_extractor",
        "--database_path",
        str(database),
        "--image_path",
        str(image_root),
        "--image_list_path",
        str(image_list),
        "--ImageReader.camera_model",
        camera_model,
        "--ImageReader.single_camera_per_folder",
        "1",
        "--SiftExtraction.max_image_size",
        str(max_image_size),
        "--SiftExtraction.max_num_features",
        str(max_features),
        "--SiftExtraction.use_gpu",
        "0",
    ]
    if camera_params:
        command.extend(["--ImageReader.camera_params", camera_params])
    environment = os.environ.copy()
    environment.setdefault("QT_QPA_PLATFORM", "minimal")
    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stdout[-4000:])


def align(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_entry = _map_entry(manifest, args.source_map_id)
    reference_entry = _map_entry(manifest, args.reference_map_id)
    source_root = _colmap_root(source_entry)
    reference_root = _colmap_root(reference_entry)
    source_sparse = _sparse_model(source_root)
    reference_sparse = _sparse_model(reference_root)
    reference_text = _sparse_text(reference_root)
    source_frames = Path(source_entry["frames_path"]).expanduser().resolve()
    reference_room = (reference_entry.get("room_alignment") or {}).get("matrix")
    if not isinstance(reference_room, list):
        raise RuntimeError("Reference map has no explicit room_alignment matrix")

    source_images = read_images_model(source_sparse)
    source_cameras = read_cameras_model(source_sparse)
    registered = sorted(source_images.values(), key=lambda item: item.name)
    if len(registered) < 3:
        raise RuntimeError("Source reconstruction has fewer than three cameras")
    step = max(1, int(math.ceil(len(registered) / max(3, args.max_samples))))
    sampled = registered[::step]
    if registered[-1].name != sampled[-1].name:
        sampled.append(registered[-1])

    camera = source_cameras[sampled[0].camera_id]
    camera_params = ",".join(f"{float(value):.16g}" for value in camera.params)
    reference_points = read_points3d_text(reference_text / "points3D.txt")
    relocalizer = FaissIVF3DRelocalizer(
        reference_root / "faiss_sift_3d_ivf",
        nprobe=args.faiss_nprobe,
        top_k=args.faiss_top_k,
        ratio=args.faiss_ratio,
        min_points=args.min_pnp_inliers,
        reprojection_error=args.pnp_reprojection_error,
    )

    work_root = args.work_dir.resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    observations: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="queries-", dir=work_root) as temporary:
        temporary_root = Path(temporary)
        image_root = temporary_root / "images"
        query_root = image_root / "query"
        query_root.mkdir(parents=True)
        image_list = temporary_root / "image.txt"
        for number, source_image in enumerate(sampled):
            frame = source_frames / source_image.name
            if not frame.exists():
                attempts.append({"image_name": source_image.name, "accepted": False, "reason": "frame_missing"})
                continue
            query_name = f"query/{source_image.name}"
            linked = query_root / source_image.name
            try:
                os.symlink(frame, linked)
            except OSError:
                shutil.copy2(frame, linked)
            database = temporary_root / f"query_{number:04d}.db"
            frame_started = time.perf_counter()
            try:
                _run_feature_extractor(
                    colmap=args.colmap,
                    database=database,
                    image_root=image_root,
                    image_name=query_name,
                    image_list=image_list,
                    camera_model=camera.model,
                    camera_params=camera_params,
                    max_image_size=args.max_image_size,
                    max_features=args.max_features,
                )
                pool, diagnostic = relocalizer.localize(
                    database_path=database,
                    image_name=query_name,
                    map_points=reference_points,
                    expected_center=None,
                )
            finally:
                database.unlink(missing_ok=True)
                linked.unlink(missing_ok=True)
            attempt = {
                "image_name": source_image.name,
                "accepted": pool is not None,
                "elapsed_ms": 1000.0 * (time.perf_counter() - frame_started),
                "diagnostic": diagnostic,
            }
            attempts.append(attempt)
            if pool is None:
                print(f"[{number + 1}/{len(sampled)}] rejected {source_image.name}: {diagnostic.get('reason')}", flush=True)
                continue
            source_rotation = qvec_to_rotmat(source_image.qvec)
            reference_rotation = qvec_to_rotmat(np.asarray(pool["colmap_qvec_world_to_camera"], dtype=float))
            observation = {
                "image_name": source_image.name,
                "source_center": camera_center(source_image).tolist(),
                "reference_center": _camera_center_from_pose(
                    pool["colmap_qvec_world_to_camera"],
                    pool["colmap_tvec_world_to_camera"],
                ).tolist(),
                "orientation_transform": (reference_rotation.T @ source_rotation).tolist(),
                "pnp_inliers": int(pool.get("faiss_pnp_inliers") or 0),
                "source_images": int(pool.get("faiss_source_images") or 0),
                "median_reprojection_error": float(pool.get("faiss_median_reprojection_error") or 0.0),
            }
            observations.append(observation)
            print(
                f"[{number + 1}/{len(sampled)}] localized {source_image.name}: "
                f"{observation['pnp_inliers']} inliers",
                flush=True,
            )

    transform, inliers = _fit_robust_sim3(
        observations,
        max_position_error=args.max_position_error,
        max_rotation_error_deg=args.max_rotation_error_deg,
    )
    inlier_set = set(inliers)
    for index, observation in enumerate(observations):
        observation["alignment_inlier"] = index in inlier_set
    room_matrix = _compose_room_alignment(reference_room, transform["matrix_3x4"])
    inlier_ratio = len(inliers) / max(1, len(observations))
    passed = bool(
        len(inliers) >= args.min_alignment_inliers
        and inlier_ratio >= args.min_alignment_inlier_ratio
        and transform["position_error_m"]["median"] <= args.max_median_position_error
        and transform["position_error_m"]["p90"] <= args.max_p90_position_error
        and transform["rotation_error_deg"]["median"] <= args.max_median_rotation_error_deg
    )
    report = {
        "format": "atlas-reference-map-sim3-alignment-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_map_id": args.source_map_id,
        "reference_map_id": args.reference_map_id,
        "source_model": str(source_sparse),
        "reference_model": str(reference_sparse),
        "sampled_images": len(sampled),
        "localized_images": len(observations),
        "alignment_inliers": len(inliers),
        "alignment_inlier_ratio": inlier_ratio,
        "quality_passed": passed,
        "sim3_source_to_reference": transform,
        "source_to_room_matrix_3x4": room_matrix,
        "thresholds": {
            "max_position_error": args.max_position_error,
            "max_rotation_error_deg": args.max_rotation_error_deg,
            "min_alignment_inliers": args.min_alignment_inliers,
            "min_alignment_inlier_ratio": args.min_alignment_inlier_ratio,
            "max_median_position_error": args.max_median_position_error,
            "max_p90_position_error": args.max_p90_position_error,
            "max_median_rotation_error_deg": args.max_median_rotation_error_deg,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "observations": observations,
        "attempts": attempts,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.apply:
        if not passed:
            raise RuntimeError(f"Alignment quality gates failed; audit saved to {args.report}")
        source_entry["room_alignment"] = {
            "matrix": room_matrix,
            "method": "reference-map-sim3-from-cross-camera-sift-pnp",
            "reference_map_id": args.reference_map_id,
            "alignment_audit": {
                "report": str(args.report.resolve()),
                "localized_images": len(observations),
                "inliers": len(inliers),
                "inlier_ratio": inlier_ratio,
                "median_position_error_m": transform["position_error_m"]["median"],
                "p90_position_error_m": transform["position_error_m"]["p90"],
                "median_rotation_error_deg": transform["rotation_error_deg"]["median"],
                "scale_source_to_reference": transform["scale"],
            },
        }
        # Coordinate equivalence must not be represented with source_map_id:
        # ATLAS also uses that field to inherit localization artifacts.  This
        # map keeps its own Xiaomi reconstruction while sharing only the
        # audited room coordinate frame with the reference.
        source_entry["coordinate_frame_id"] = args.reference_map_id
        source_entry["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-map-id", required=True)
    parser.add_argument("--reference-map-id", required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "viewer" / "public" / "maps" / "manifest.json")
    parser.add_argument("--colmap", type=Path, default=ROOT / "tools" / "colmap-env" / "bin" / "colmap")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "runtime" / "map_reference_alignment")
    parser.add_argument("--report", type=Path, default=ROOT / "runtime" / "map_reference_alignment" / "alignment_audit.json")
    parser.add_argument("--max-samples", type=int, default=60)
    parser.add_argument("--max-image-size", type=int, default=1200)
    parser.add_argument("--max-features", type=int, default=4096)
    parser.add_argument("--faiss-nprobe", type=int, default=32)
    parser.add_argument("--faiss-top-k", type=int, default=32)
    parser.add_argument("--faiss-ratio", type=float, default=0.80)
    parser.add_argument("--min-pnp-inliers", type=int, default=35)
    parser.add_argument("--pnp-reprojection-error", type=float, default=6.0)
    parser.add_argument("--max-position-error", type=float, default=0.35)
    parser.add_argument("--max-rotation-error-deg", type=float, default=12.0)
    parser.add_argument("--min-alignment-inliers", type=int, default=8)
    parser.add_argument("--min-alignment-inlier-ratio", type=float, default=0.35)
    parser.add_argument("--max-median-position-error", type=float, default=0.15)
    parser.add_argument("--max-p90-position-error", type=float, default=0.30)
    parser.add_argument("--max-median-rotation-error-deg", type=float, default=6.0)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    report = align(args)
    print(json.dumps({key: report[key] for key in (
        "source_map_id",
        "reference_map_id",
        "sampled_images",
        "localized_images",
        "alignment_inliers",
        "alignment_inlier_ratio",
        "quality_passed",
        "sim3_source_to_reference",
        "source_to_room_matrix_3x4",
        "elapsed_seconds",
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
