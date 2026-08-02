#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from colmap_io import camera_center, qvec_to_rotmat, read_images_model, read_points3d_text


def run(command: list[object]) -> None:
    cmd = [str(value) for value in command]
    print("+", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "minimal")
    subprocess.run(cmd, check=True, env=env)


def sqlite_backup(source: Path, destination: Path) -> None:
    # Resolve staging symlinks before constructing SQLite's read-only URI.
    # The macOS SQLite build can reject an otherwise readable relative symlink
    # when the URI is opened with mode=ro.
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Some macOS volumes intermittently reject SQLite's page-by-page backup
    # when the destination database is created inside a freshly populated
    # staging tree. Build and verify the snapshot on the local temporary
    # volume, then copy the completed file into staging.
    with tempfile.TemporaryDirectory(prefix="atlas-colmap-db-") as temp_dir:
        snapshot = Path(temp_dir) / "database.db"
        with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as src:
            with sqlite3.connect(snapshot) as dst:
                src.backup(dst)
                result = dst.execute("PRAGMA quick_check").fetchone()
                if not result or result[0] != "ok":
                    raise RuntimeError(f"COLMAP database snapshot failed integrity check: {result}")
        staging_copy = destination.with_name(f".{destination.name}.copying")
        if staging_copy.exists():
            staging_copy.unlink()
        shutil.copy2(snapshot, staging_copy)
        os.replace(staging_copy, destination)


def first_camera_id(reference_text: Path) -> int:
    for line in (reference_text / "cameras.txt").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return int(stripped.split()[0])
    raise RuntimeError("Reference model has no camera.")


def choose_largest_sparse_model(sparse_root: Path) -> Path:
    candidates = [
        path
        for path in sorted(sparse_root.iterdir())
        if path.is_dir() and (path / "images.bin").exists() and (path / "points3D.bin").exists()
    ]
    if not candidates:
        raise RuntimeError(f"No COLMAP sparse model found under {sparse_root}")
    return max(candidates, key=lambda path: (path / "points3D.bin").stat().st_size)


def resolve_output_model(output_root: Path) -> Path:
    if (output_root / "images.bin").exists() and (output_root / "points3D.bin").exists():
        return output_root
    return choose_largest_sparse_model(output_root)


def evenly_sample(values: list[str], limit: int) -> list[str]:
    if len(values) <= limit:
        return values
    indices = np.linspace(0, len(values) - 1, limit, dtype=int)
    return [values[int(index)] for index in indices]


def write_match_pairs(
    path: Path,
    new_names: list[str],
    reference_names: list[str],
    *,
    reference_limit: int,
    reference_stride: int,
    sequential_window: int,
) -> tuple[int, int]:
    references = evenly_sample(reference_names, max(1, reference_limit))
    pairs: set[tuple[str, str]] = set()
    # A continuous enhancement flight only needs periodic map anchors. Matching
    # every video frame to hundreds of reference images wastes hours and makes
    # repeated indoor views more likely to create a false global alias. The
    # incremental mapper below registers the intervening views through the
    # dense chronological overlap.
    anchor_names = [
        name
        for index, name in enumerate(new_names)
        if index < sequential_window or index % max(1, reference_stride) == 0
    ]
    for name in anchor_names:
        for reference in references:
            pairs.add((name, reference))
    for index, name in enumerate(new_names):
        for other in new_names[index + 1:index + 1 + max(1, sequential_window)]:
            pairs.add((name, other))
    path.write_text("".join(f"{left} {right}\n" for left, right in sorted(pairs)), encoding="utf-8")
    return len(pairs), len(anchor_names)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else float("inf")


def main() -> int:
    parser = argparse.ArgumentParser(description="Enhance a COLMAP map while keeping the reference coordinate frame fixed.")
    parser.add_argument("--colmap", type=Path, required=True)
    parser.add_argument("--reference-colmap", type=Path, required=True)
    parser.add_argument("--new-frames", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-image-size", type=int, default=1200)
    parser.add_argument("--min-registration-ratio", type=float, default=0.15)
    parser.add_argument("--max-median-anchor-distance", type=float, default=1.20)
    parser.add_argument("--max-p95-anchor-distance", type=float, default=3.00)
    parser.add_argument("--reference-image-limit", type=int, default=180)
    parser.add_argument("--reference-stride", type=int, default=4)
    parser.add_argument("--sequential-window", type=int, default=12)
    parser.add_argument(
        "--query-camera-params",
        default="",
        help="Optional COLMAP camera params for the enhancement camera, e.g. f,cx,cy,k.",
    )
    parser.add_argument("--min-temporal-coverage", type=float, default=0.75)
    parser.add_argument("--max-consecutive-center-step", type=float, default=0.85)
    args = parser.parse_args()

    reference = args.reference_colmap.resolve()
    reference_database = reference / "database.db"
    reference_images_dir = reference / "images"
    reference_sparse_root = reference / "sparse"
    reference_text = reference / "sparse_text"
    for required in (reference_database, reference_images_dir, reference_sparse_root, reference_text):
        if not required.exists():
            raise RuntimeError(f"Missing fixed-reference artifact: {required}")
    reference_sparse = choose_largest_sparse_model(reference_sparse_root)

    new_sources = sorted(args.new_frames.glob("*.jpg"))
    if len(new_sources) < 8:
        raise RuntimeError(f"Need at least 8 enhancement frames; found {len(new_sources)}.")

    out = args.out_dir.resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    images_dir = out / "images"
    shutil.copytree(reference_images_dir, images_dir)
    reference_names = sorted(path.relative_to(reference_images_dir).as_posix() for path in reference_images_dir.rglob("*.jpg"))
    reference_name_set = set(reference_names)
    new_names: list[str] = []
    enhancement_dir = images_dir / "enhancement"
    # A fixed-reference candidate may already contain frames from an earlier
    # enhancement pass. Reuse that namespace while keeping the new filenames
    # collision-checked against the copied reference image bank above.
    enhancement_dir.mkdir(exist_ok=True)
    for source in new_sources:
        relative_name = f"enhancement/{source.name}"
        if relative_name in reference_name_set:
            continue
        shutil.copy2(source, images_dir / relative_name)
        new_names.append(relative_name)
    if len(new_names) < 8:
        raise RuntimeError("Enhancement frames duplicate the fixed reference; no useful new frame bank remains.")

    database = out / "database.db"
    sqlite_backup(reference_database, database)
    new_list = out / "new_images.txt"
    new_list.write_text("\n".join(new_names) + "\n", encoding="utf-8")
    feature_command: list[object] = [
        args.colmap,
        "feature_extractor",
        "--database_path",
        database,
        "--image_path",
        images_dir,
        "--image_list_path",
        new_list,
        "--ImageReader.camera_model",
        "SIMPLE_RADIAL",
        "--ImageReader.single_camera_per_folder",
        "1",
        "--SiftExtraction.max_image_size",
        args.max_image_size,
        "--SiftExtraction.use_gpu",
        "0",
    ]
    if args.query_camera_params.strip():
        feature_command.extend(["--ImageReader.camera_params", args.query_camera_params.strip()])
    run(feature_command)

    pairs_path = out / "fixed_reference_pairs.txt"
    pair_count, anchor_frame_count = write_match_pairs(
        pairs_path,
        new_names,
        reference_names,
        reference_limit=args.reference_image_limit,
        reference_stride=args.reference_stride,
        sequential_window=args.sequential_window,
    )
    run(
        [
            args.colmap,
            "matches_importer",
            "--database_path",
            database,
            "--match_list_path",
            pairs_path,
            "--match_type",
            "pairs",
            "--SiftMatching.guided_matching",
            "1",
            "--SiftMatching.use_gpu",
            "0",
            "--SiftMatching.num_threads",
            "8",
        ]
    )

    incremental_root = out / "incremental"
    incremental_root.mkdir()
    run(
        [
            args.colmap,
            "mapper",
            "--database_path",
            database,
            "--image_path",
            images_dir,
            "--input_path",
            reference_sparse,
            "--output_path",
            incremental_root,
            "--Mapper.fix_existing_images",
            "1",
            "--Mapper.ba_refine_focal_length",
            "0",
            "--Mapper.ba_refine_principal_point",
            "0",
            "--Mapper.ba_refine_extra_params",
            "0",
            "--Mapper.abs_pose_min_num_inliers",
            "20",
            "--Mapper.abs_pose_min_inlier_ratio",
            "0.15",
            "--Mapper.abs_pose_max_error",
            "8",
            "--Mapper.max_reg_trials",
            "5",
            "--Mapper.filter_max_reproj_error",
            "3",
        ]
    )
    registered = resolve_output_model(incremental_root)

    reference_images = read_images_model(reference_sparse)
    registered_images = read_images_model(registered)
    by_name = {image.name: image for image in registered_images.values()}
    registered_new = [by_name[name] for name in new_names if name in by_name]
    minimum_registered = max(12, int(np.ceil(len(new_names) * max(0.05, args.min_registration_ratio))))
    if len(registered_new) < minimum_registered:
        raise RuntimeError(
            f"Only {len(registered_new)}/{len(new_names)} enhancement frames registered; "
            f"need at least {minimum_registered}. Fixed Lab map was preserved."
        )
    registered_indices = [index for index, name in enumerate(new_names) if name in by_name]
    temporal_coverage = (registered_indices[-1] + 1) / len(new_names)
    if temporal_coverage < args.min_temporal_coverage:
        raise RuntimeError(
            f"Enhancement registered only through {temporal_coverage:.1%} of the continuous capture; "
            f"need at least {args.min_temporal_coverage:.1%}. Fixed map was preserved."
        )

    consecutive_steps: list[float] = []
    consecutive_rotations_deg: list[float] = []
    for left_index, right_index in zip(registered_indices, registered_indices[1:]):
        if right_index != left_index + 1:
            continue
        left = by_name[new_names[left_index]]
        right = by_name[new_names[right_index]]
        consecutive_steps.append(float(np.linalg.norm(camera_center(right) - camera_center(left))))
        relative_rotation = qvec_to_rotmat(right.qvec) @ qvec_to_rotmat(left.qvec).T
        cosine = float(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))
        consecutive_rotations_deg.append(float(np.degrees(np.arccos(cosine))))
    max_consecutive_step = max(consecutive_steps, default=float("inf"))
    if max_consecutive_step > args.max_consecutive_center_step:
        raise RuntimeError(
            "Registered enhancement contains an implausible position jump "
            f"({max_consecutive_step:.3f} > {args.max_consecutive_center_step:.3f} map units). "
            "No map artifacts were replaced."
        )

    reference_by_name = {image.name: image for image in reference_images.values()}
    reference_pose_deltas = [
        float(np.linalg.norm(camera_center(by_name[name]) - camera_center(image)))
        for name, image in reference_by_name.items()
        if name in by_name
    ]
    max_reference_pose_delta = max(reference_pose_deltas, default=float("inf"))
    if len(reference_pose_deltas) != len(reference_images) or max_reference_pose_delta > 1e-7:
        raise RuntimeError(
            "The incremental result moved or dropped fixed reference cameras "
            f"(max center delta {max_reference_pose_delta:.3g}). No map artifacts were replaced."
        )
    reference_centers = np.asarray([camera_center(image) for image in reference_images.values()], dtype=float)
    anchor_distances = []
    for image in registered_new:
        center = camera_center(image)
        anchor_distances.append(float(np.min(np.linalg.norm(reference_centers - center, axis=1))))
    median_anchor = percentile(anchor_distances, 50)
    p95_anchor = percentile(anchor_distances, 95)
    if median_anchor > args.max_median_anchor_distance or p95_anchor > args.max_p95_anchor_distance:
        raise RuntimeError(
            "Registered enhancement trajectory is inconsistent with the fixed Lab map "
            f"(anchor distance median {median_anchor:.2f}, p95 {p95_anchor:.2f}). "
            "No map artifacts were replaced."
        )

    sparse = out / "sparse"
    final_model = sparse / "0"
    sparse.mkdir()
    shutil.copytree(registered, final_model)
    sparse_text = out / "sparse_text"
    sparse_text.mkdir()
    run([args.colmap, "model_converter", "--input_path", final_model, "--output_path", sparse_text, "--output_type", "TXT"])

    final_images = read_images_model(final_model)
    final_points = read_points3d_text(sparse_text / "points3D.txt")
    reference_points = read_points3d_text(reference_text / "points3D.txt")
    (out / "image_list.txt").write_text(
        "\n".join(sorted(path.relative_to(images_dir).as_posix() for path in images_dir.rglob("*.jpg"))) + "\n",
        encoding="utf-8",
    )
    summary = {
        "mode": "fixed_reference_enhancement",
        "reference_colmap": str(reference),
        "new_frame_count": len(new_names),
        "match_pair_count": pair_count,
        "anchor_frame_count": anchor_frame_count,
        "reference_sparse_model": str(reference_sparse),
        "registered_new_images": len(registered_new),
        "registration_ratio": len(registered_new) / len(new_names),
        "temporal_coverage": temporal_coverage,
        "max_consecutive_center_step": max_consecutive_step,
        "p95_consecutive_center_step": percentile(consecutive_steps, 95),
        "max_consecutive_rotation_deg": max(consecutive_rotations_deg, default=float("inf")),
        "max_reference_pose_delta": max_reference_pose_delta,
        "anchor_distance_median": median_anchor,
        "anchor_distance_p95": p95_anchor,
        "reference_registered_images": len(reference_images),
        "final_registered_images": len(final_images),
        "reference_points": len(reference_points),
        "final_points": len(final_points),
        "added_points": max(0, len(final_points) - len(reference_points)),
        "coordinate_frame_preserved": True,
    }
    (out / "fixed_reference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
