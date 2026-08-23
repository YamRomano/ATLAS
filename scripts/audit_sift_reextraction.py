#!/usr/bin/env python3
"""Re-extract COLMAP CPU SIFT twice from one preserved recovery frame.

The audit uses two clean databases so it can distinguish a feature-extraction
problem from a downstream Faiss/matching problem.  Optionally, both descriptor
sets are passed through the persistent global 2D-to-3D Faiss relocalizer.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from colmap_faiss_relocalizer import FaissIVF3DRelocalizer, _query_features
from colmap_io import read_points3d_text


def run_checked(command: list[object]) -> None:
    subprocess.run([str(value) for value in command], check=True)


def extract_once(
    *,
    colmap: Path,
    database: Path,
    image_root: Path,
    image_list: Path,
    camera_model: str,
    camera_params: str,
    max_image_size: int,
    max_num_features: int,
) -> None:
    database.unlink(missing_ok=True)
    run_checked([colmap, "database_creator", "--database_path", database])
    command: list[object] = [
        colmap,
        "feature_extractor",
        "--database_path",
        database,
        "--image_path",
        image_root,
        "--image_list_path",
        image_list,
        "--ImageReader.camera_model",
        camera_model,
        "--ImageReader.single_camera_per_folder",
        "1",
        "--SiftExtraction.max_image_size",
        max_image_size,
        "--SiftExtraction.max_num_features",
        max_num_features,
        "--SiftExtraction.use_gpu",
        "0",
    ]
    if camera_params.strip():
        command.extend(["--ImageReader.camera_params", camera_params.strip()])
    run_checked(command)


def feature_summary(database: Path, image_name: str) -> dict[str, Any]:
    _image_id, camera, keypoints, descriptors = _query_features(database, image_name)
    return {
        "camera_model": camera.model,
        "camera_width": int(camera.width),
        "camera_height": int(camera.height),
        "camera_params": [float(value) for value in camera.params],
        "keypoints": keypoints,
        "descriptors": descriptors,
    }


def comparison_payload(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    first_keypoints = np.asarray(first["keypoints"], dtype=np.float32)
    second_keypoints = np.asarray(second["keypoints"], dtype=np.float32)
    first_descriptors = np.asarray(first["descriptors"], dtype=np.uint8)
    second_descriptors = np.asarray(second["descriptors"], dtype=np.uint8)
    same_keypoint_shape = first_keypoints.shape == second_keypoints.shape
    same_descriptor_shape = first_descriptors.shape == second_descriptors.shape
    keypoints_exact = bool(
        same_keypoint_shape and np.array_equal(first_keypoints, second_keypoints)
    )
    descriptors_exact = bool(
        same_descriptor_shape and np.array_equal(first_descriptors, second_descriptors)
    )
    return {
        "first_keypoints": int(len(first_keypoints)),
        "second_keypoints": int(len(second_keypoints)),
        "first_descriptors": int(len(first_descriptors)),
        "second_descriptors": int(len(second_descriptors)),
        "keypoint_shapes_equal": bool(same_keypoint_shape),
        "descriptor_shapes_equal": bool(same_descriptor_shape),
        "keypoints_exactly_equal": keypoints_exact,
        "descriptors_exactly_equal": descriptors_exact,
        "max_keypoint_delta_px": (
            float(np.max(np.abs(first_keypoints - second_keypoints)))
            if same_keypoint_shape and first_keypoints.size
            else None
        ),
        "descriptor_values_different": (
            int(np.count_nonzero(first_descriptors != second_descriptors))
            if same_descriptor_shape
            else None
        ),
        "reproducible": bool(keypoints_exact and descriptors_exact),
    }


def parse_center(value: str) -> np.ndarray | None:
    if not value.strip():
        return None
    center = np.asarray([float(item) for item in value.split(",")], dtype=np.float64)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("--expected-center must contain three finite comma-separated values")
    return center


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--colmap", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--image-name", default="query/recovery.jpg")
    parser.add_argument("--camera-model", default="SIMPLE_RADIAL")
    parser.add_argument("--camera-params", default="")
    parser.add_argument("--max-image-size", type=int, default=1200)
    parser.add_argument("--max-num-features", type=int, default=8192)
    parser.add_argument("--faiss-index-dir", type=Path)
    parser.add_argument("--map-points", type=Path)
    parser.add_argument("--expected-center", default="")
    parser.add_argument("--faiss-nprobe", type=int, default=32)
    parser.add_argument("--faiss-top-k", type=int, default=32)
    parser.add_argument("--faiss-ratio", type=float, default=0.80)
    parser.add_argument("--faiss-min-points", type=int, default=40)
    parser.add_argument("--faiss-reprojection-error", type=float, default=6.0)
    args = parser.parse_args()

    image = args.image.resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    out_dir = args.out_dir.resolve()
    image_root = out_dir / "images"
    copied_image = image_root / args.image_name
    copied_image.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, copied_image)
    image_list = out_dir / "image_list.txt"
    image_list.write_text(args.image_name + "\n", encoding="utf-8")

    databases = [out_dir / "sift_first.db", out_dir / "sift_second.db"]
    summaries: list[dict[str, Any]] = []
    for database in databases:
        extract_once(
            colmap=args.colmap.resolve(),
            database=database,
            image_root=image_root,
            image_list=image_list,
            camera_model=args.camera_model,
            camera_params=args.camera_params,
            max_image_size=max(64, int(args.max_image_size)),
            max_num_features=max(64, int(args.max_num_features)),
        )
        summaries.append(feature_summary(database, args.image_name))

    result: dict[str, Any] = {
        "image": str(image),
        "preserved_copy": str(copied_image),
        "image_name": args.image_name,
        "comparison": comparison_payload(summaries[0], summaries[1]),
    }

    if args.faiss_index_dir is not None or args.map_points is not None:
        if args.faiss_index_dir is None or args.map_points is None:
            raise ValueError("--faiss-index-dir and --map-points must be supplied together")
        map_points = read_points3d_text(args.map_points.resolve())
        relocalizer = FaissIVF3DRelocalizer(
            args.faiss_index_dir.resolve(),
            nprobe=args.faiss_nprobe,
            top_k=args.faiss_top_k,
            ratio=args.faiss_ratio,
            min_points=args.faiss_min_points,
            reprojection_error=args.faiss_reprojection_error,
        )
        expected_center = parse_center(args.expected_center)
        result["faiss"] = []
        for database in databases:
            pool, diagnostic = relocalizer.localize(
                database_path=database,
                image_name=args.image_name,
                map_points=map_points,
                expected_center=expected_center,
            )
            result["faiss"].append(
                {
                    "database": str(database),
                    "accepted": pool is not None,
                    "diagnostic": diagnostic,
                    "valid_2d3d": pool.get("valid_2d3d") if pool else None,
                }
            )

    output = out_dir / "sift_reextraction_audit.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
