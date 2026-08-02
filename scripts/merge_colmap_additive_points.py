#!/usr/bin/env python3
"""Merge triangulated support into a COLMAP text model without deleting reference data.

Reference cameras, image poses, existing 2D observations, and 3D point
coordinates are preserved. New registered images are copied from a
triangulated candidate. Candidate observations may extend existing point
tracks, and candidate-only points are retained only when their observations do
not replace an existing reference observation.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from colmap_io import Image, read_images_text


@dataclass
class PointLine:
    point_id: int
    xyz: tuple[float, float, float]
    rgb: tuple[int, int, int]
    error: float
    track: list[tuple[int, int]]


def data_lines(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                yield stripped


def point_ids(path: Path) -> set[int]:
    return {int(line.split(maxsplit=1)[0]) for line in data_lines(path)}


def parse_point(line: str) -> PointLine:
    fields = line.split()
    if len(fields) < 8 or (len(fields) - 8) % 2:
        raise RuntimeError(f"Invalid COLMAP point line: {line[:200]}")
    return PointLine(
        point_id=int(fields[0]),
        xyz=(float(fields[1]), float(fields[2]), float(fields[3])),
        rgb=(int(fields[4]), int(fields[5]), int(fields[6])),
        error=float(fields[7]),
        track=[
            (int(fields[index]), int(fields[index + 1]))
            for index in range(8, len(fields), 2)
        ],
    )


def clone_image(image: Image) -> Image:
    return Image(
        image_id=image.image_id,
        qvec=np.asarray(image.qvec, dtype=float).copy(),
        tvec=np.asarray(image.tvec, dtype=float).copy(),
        camera_id=image.camera_id,
        name=image.name,
        xys=np.asarray(image.xys, dtype=float).copy(),
        point3d_ids=np.asarray(image.point3d_ids, dtype=np.int64).copy(),
    )


def write_images(path: Path, images: dict[int, Image]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "# Image list with two lines of data per image:\n"
            "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
            "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
            f"# Number of images: {len(images)}\n"
        )
        for image_id in sorted(images):
            image = images[image_id]
            header = " ".join(
                [
                    str(image.image_id),
                    *(f"{float(value):.17g}" for value in image.qvec),
                    *(f"{float(value):.17g}" for value in image.tvec),
                    str(image.camera_id),
                    image.name,
                ]
            )
            observations = " ".join(
                f"{float(xy[0]):.17g} {float(xy[1]):.17g} {int(point_id)}"
                for xy, point_id in zip(image.xys, image.point3d_ids)
            )
            handle.write(f"{header}\n{observations}\n")


def point_line(record: PointLine, track: list[tuple[int, int]]) -> str:
    values = [
        str(record.point_id),
        *(f"{value:.17g}" for value in record.xyz),
        *(str(value) for value in record.rgb),
        f"{record.error:.17g}",
        *(str(value) for pair in track for value in pair),
    ]
    return " ".join(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-text", type=Path, required=True)
    parser.add_argument("--candidate-text", type=Path, required=True)
    parser.add_argument("--out-text", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--min-new-image-observations", type=int, default=1)
    parser.add_argument("--min-track-length", type=int, default=2)
    parser.add_argument("--max-fixed-center-delta", type=float, default=1e-7)
    args = parser.parse_args()

    reference = args.reference_text.resolve()
    candidate = args.candidate_text.resolve()
    out = args.out_text.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite additive merge output: {out}")
    out.mkdir(parents=True)

    reference_images = read_images_text(reference / "images.txt")
    candidate_images = read_images_text(candidate / "images.txt")
    reference_names = {image.name: image_id for image_id, image in reference_images.items()}
    candidate_names = {image.name: image_id for image_id, image in candidate_images.items()}
    missing_reference_names = sorted(set(reference_names) - set(candidate_names))
    if missing_reference_names:
        raise RuntimeError(
            f"Candidate is missing {len(missing_reference_names)} reference images."
        )

    max_fixed_center_delta = 0.0
    for name, reference_id in reference_names.items():
        candidate_id = candidate_names[name]
        if candidate_id != reference_id:
            raise RuntimeError(
                f"Reference image id changed for {name}: {reference_id} -> {candidate_id}"
            )
        left = reference_images[reference_id]
        right = candidate_images[candidate_id]
        left_center = -qvec_to_rotation(left.qvec).T @ left.tvec
        right_center = -qvec_to_rotation(right.qvec).T @ right.tvec
        max_fixed_center_delta = max(
            max_fixed_center_delta,
            float(np.linalg.norm(right_center - left_center)),
        )
    if max_fixed_center_delta > args.max_fixed_center_delta:
        raise RuntimeError(
            f"Candidate moved a reference camera by {max_fixed_center_delta:.6g} m."
        )

    new_image_ids = set(candidate_images) - set(reference_images)
    if not new_image_ids:
        raise RuntimeError("Candidate has no new registered images.")
    merged_images = {
        image_id: clone_image(image) for image_id, image in reference_images.items()
    }
    for image_id in new_image_ids:
        merged_images[image_id] = clone_image(candidate_images[image_id])

    reference_point_ids = point_ids(reference / "points3D.txt")
    candidate_only_points: dict[int, PointLine] = {}
    for line in data_lines(candidate / "points3D.txt"):
        record = parse_point(line)
        if record.point_id not in reference_point_ids:
            candidate_only_points[record.point_id] = record

    kept_new_points: dict[int, PointLine] = {}
    dropped_new_points: dict[str, int] = {
        "short_track": 0,
        "no_new_image_observation": 0,
    }
    for point_id, record in candidate_only_points.items():
        filtered_track: list[tuple[int, int]] = []
        new_image_observations = 0
        for image_id, point2d_index in record.track:
            candidate_image = candidate_images.get(image_id)
            if (
                candidate_image is None
                or point2d_index < 0
                or point2d_index >= len(candidate_image.point3d_ids)
                or int(candidate_image.point3d_ids[point2d_index]) != point_id
            ):
                raise RuntimeError(
                    f"Candidate point {point_id} has an inconsistent track observation."
                )
            if image_id in new_image_ids:
                filtered_track.append((image_id, point2d_index))
                new_image_observations += 1
                continue
            reference_image = reference_images.get(image_id)
            if (
                reference_image is not None
                and point2d_index < len(reference_image.point3d_ids)
                and int(reference_image.point3d_ids[point2d_index]) == -1
            ):
                filtered_track.append((image_id, point2d_index))
        if len(filtered_track) < args.min_track_length:
            dropped_new_points["short_track"] += 1
            continue
        if new_image_observations < args.min_new_image_observations:
            dropped_new_points["no_new_image_observation"] += 1
            continue
        kept_new_points[point_id] = PointLine(
            point_id=record.point_id,
            xyz=record.xyz,
            rgb=record.rgb,
            error=record.error,
            track=filtered_track,
        )

    reference_track_additions: dict[int, list[tuple[int, int]]] = {}
    for image_id in new_image_ids:
        image = merged_images[image_id]
        for point2d_index, raw_point_id in enumerate(image.point3d_ids):
            point_id = int(raw_point_id)
            if point_id in reference_point_ids:
                reference_track_additions.setdefault(point_id, []).append(
                    (image_id, point2d_index)
                )
            elif point_id not in kept_new_points:
                image.point3d_ids[point2d_index] = -1

    for point_id, record in kept_new_points.items():
        for image_id, point2d_index in record.track:
            image = merged_images[image_id]
            current = int(image.point3d_ids[point2d_index])
            if image_id in reference_images:
                if current != -1:
                    raise RuntimeError(
                        f"New point {point_id} would replace reference observation "
                        f"{image_id}:{point2d_index} -> {current}."
                    )
                image.point3d_ids[point2d_index] = point_id
            elif current != point_id:
                raise RuntimeError(
                    f"New image observation {image_id}:{point2d_index} is inconsistent."
                )

    shutil.copy2(reference / "cameras.txt", out / "cameras.txt")
    write_images(out / "images.txt", merged_images)
    with (out / "points3D.txt").open("w", encoding="utf-8") as handle:
        total_points = len(reference_point_ids) + len(kept_new_points)
        handle.write(
            "# 3D point list with one line of data per point:\n"
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as "
            "(IMAGE_ID, POINT2D_IDX)\n"
            f"# Number of points: {total_points}\n"
        )
        seen_reference_ids: set[int] = set()
        for line in data_lines(reference / "points3D.txt"):
            record = parse_point(line)
            additions = reference_track_additions.get(record.point_id, [])
            handle.write(f"{point_line(record, [*record.track, *additions])}\n")
            seen_reference_ids.add(record.point_id)
        if seen_reference_ids != reference_point_ids:
            raise RuntimeError("Reference point parsing changed during additive merge.")
        for point_id in sorted(kept_new_points):
            record = kept_new_points[point_id]
            handle.write(f"{point_line(record, record.track)}\n")

    new_image_support = [
        int(np.sum(merged_images[image_id].point3d_ids >= 0))
        for image_id in sorted(new_image_ids)
    ]
    summary = {
        "mode": "additive_colmap_point_merge",
        "reference_images": len(reference_images),
        "new_images": len(new_image_ids),
        "merged_images": len(merged_images),
        "reference_points_preserved": len(reference_point_ids),
        "candidate_only_points": len(candidate_only_points),
        "added_points": len(kept_new_points),
        "merged_points": len(reference_point_ids) + len(kept_new_points),
        "extended_reference_point_tracks": len(reference_track_additions),
        "dropped_new_points": dropped_new_points,
        "max_fixed_center_delta": max_fixed_center_delta,
        "new_image_support_min": min(new_image_support),
        "new_image_support_median": float(np.median(new_image_support)),
        "new_image_support_max": max(new_image_support),
        "new_images_with_zero_support": sum(value == 0 for value in new_image_support),
    }
    summary_path = args.summary_out or out / "additive_merge_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def qvec_to_rotation(qvec: np.ndarray) -> np.ndarray:
    q = np.asarray(qvec, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=float,
    )


if __name__ == "__main__":
    raise SystemExit(main())
