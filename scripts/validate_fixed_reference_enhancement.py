#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from colmap_io import camera_center, qvec_to_rotmat, read_images_model


def point_count(model: Path) -> int:
    text = model / "points3D.txt"
    if text.exists():
        return sum(
            1
            for line in text.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        )
    binary = model / "points3D.bin"
    with binary.open("rb") as handle:
        raw = handle.read(8)
    if len(raw) != 8:
        raise RuntimeError(f"Invalid COLMAP points file: {binary}")
    return int(struct.unpack("<Q", raw)[0])


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


def rotation_distance_degrees(left: np.ndarray, right: np.ndarray) -> float:
    relative = qvec_to_rotmat(right) @ qvec_to_rotmat(left).T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def longest_missing_run(registered: set[int], count: int) -> int:
    longest = 0
    current = 0
    for index in range(count):
        if index in registered:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a fixed-reference COLMAP enhancement without modifying either model."
    )
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--selected-frames", type=Path, required=True)
    parser.add_argument("--new-prefix", default="enhancement/")
    parser.add_argument("--min-registration-ratio", type=float, default=0.50)
    parser.add_argument("--min-temporal-coverage", type=float, default=0.90)
    parser.add_argument("--span-start-index", type=int)
    parser.add_argument("--span-end-index", type=int)
    parser.add_argument("--min-span-registration-ratio", type=float)
    parser.add_argument("--max-consecutive-center-step", type=float, default=0.85)
    parser.add_argument("--max-reference-center-delta", type=float, default=1e-7)
    parser.add_argument(
        "--allow-unchanged-reference-step",
        action="store_true",
        help=(
            "Allow a consecutive center step above the limit only when the exact "
            "same edge is present in the fixed reference model."
        ),
    )
    parser.add_argument("--summary-out", type=Path)
    args = parser.parse_args()

    reference = args.reference_model.resolve()
    candidate = args.candidate_model.resolve()
    selected_sources = sorted(args.selected_frames.glob("*.jpg"))
    selected_names = [f"{args.new_prefix}{path.name}" for path in selected_sources]
    if not selected_names:
        raise RuntimeError(f"No selected JPEG frames found in {args.selected_frames}")

    reference_images = read_images_model(reference)
    candidate_images = read_images_model(candidate)
    reference_by_name = {image.name: image for image in reference_images.values()}
    candidate_by_name = {image.name: image for image in candidate_images.values()}

    missing_reference = sorted(set(reference_by_name) - set(candidate_by_name))
    reference_center_deltas: list[float] = []
    reference_rotation_deltas: list[float] = []
    for name, image in reference_by_name.items():
        current = candidate_by_name.get(name)
        if current is None:
            continue
        reference_center_deltas.append(
            float(np.linalg.norm(camera_center(current) - camera_center(image)))
        )
        reference_rotation_deltas.append(
            rotation_distance_degrees(image.qvec, current.qvec)
        )

    registered_indices = [
        index for index, name in enumerate(selected_names) if name in candidate_by_name
    ]
    registered_index_set = set(registered_indices)
    registered_new = [candidate_by_name[selected_names[index]] for index in registered_indices]
    registration_ratio = len(registered_indices) / len(selected_names)
    temporal_coverage = (
        (registered_indices[-1] + 1) / len(selected_names) if registered_indices else 0.0
    )
    if (args.span_start_index is None) != (args.span_end_index is None):
        raise ValueError("--span-start-index and --span-end-index must be supplied together.")
    span_registration_ratio: float | None = None
    span_registered_frames: int | None = None
    span_frame_count: int | None = None
    if args.span_start_index is not None and args.span_end_index is not None:
        if not (
            0 <= args.span_start_index <= args.span_end_index < len(selected_names)
        ):
            raise ValueError(
                f"Invalid selected-frame span {args.span_start_index}..{args.span_end_index}"
            )
        span_frame_count = args.span_end_index - args.span_start_index + 1
        span_registered_frames = sum(
            args.span_start_index <= index <= args.span_end_index
            for index in registered_indices
        )
        span_registration_ratio = span_registered_frames / span_frame_count

    consecutive: list[dict[str, object]] = []
    for left_index, right_index in zip(registered_indices, registered_indices[1:]):
        if right_index != left_index + 1:
            continue
        left = candidate_by_name[selected_names[left_index]]
        right = candidate_by_name[selected_names[right_index]]
        consecutive.append(
            {
                "left": selected_names[left_index],
                "right": selected_names[right_index],
                "center_step": float(
                    np.linalg.norm(camera_center(right) - camera_center(left))
                ),
                "rotation_deg": rotation_distance_degrees(left.qvec, right.qvec),
            }
        )
    unchanged_reference_step_exceptions: list[dict[str, object]] = []
    for item in consecutive:
        if float(item["center_step"]) <= args.max_consecutive_center_step:
            continue
        left = reference_by_name.get(str(item["left"]))
        right = reference_by_name.get(str(item["right"]))
        if left is None or right is None:
            continue
        reference_step = float(
            np.linalg.norm(camera_center(right) - camera_center(left))
        )
        if (
            args.allow_unchanged_reference_step
            and abs(reference_step - float(item["center_step"]))
            <= args.max_reference_center_delta
        ):
            unchanged_reference_step_exceptions.append(
                {**item, "reference_center_step": reference_step}
            )
    center_steps = [float(item["center_step"]) for item in consecutive]
    rotation_steps = [float(item["rotation_deg"]) for item in consecutive]
    top_center_steps = sorted(
        consecutive, key=lambda item: float(item["center_step"]), reverse=True
    )[:10]

    reference_centers = np.asarray(
        [camera_center(image) for image in reference_images.values()], dtype=float
    )
    anchor_distances = [
        float(
            np.min(
                np.linalg.norm(
                    reference_centers - camera_center(image),
                    axis=1,
                )
            )
        )
        for image in registered_new
    ]

    max_center_step = max(center_steps, default=float("inf"))
    max_reference_center_delta = max(reference_center_deltas, default=float("inf"))
    max_reference_rotation_delta = max(reference_rotation_deltas, default=float("inf"))
    exception_edges = {
        (str(item["left"]), str(item["right"]))
        for item in unchanged_reference_step_exceptions
    }
    new_center_steps = [
        float(item["center_step"])
        for item in consecutive
        if (str(item["left"]), str(item["right"])) not in exception_edges
    ]
    checks = {
        "reference_images_preserved": not missing_reference,
        "reference_centers_fixed": (
            len(reference_center_deltas) == len(reference_images)
            and max_reference_center_delta <= args.max_reference_center_delta
        ),
        "registration_ratio": registration_ratio >= args.min_registration_ratio,
        "temporal_coverage": temporal_coverage >= args.min_temporal_coverage,
        "consecutive_center_steps": (
            bool(center_steps)
            and all(
                float(item["center_step"]) <= args.max_consecutive_center_step
                or (str(item["left"]), str(item["right"])) in exception_edges
                for item in consecutive
            )
        ),
        "points_added": point_count(candidate) > point_count(reference),
    }
    if args.min_span_registration_ratio is not None:
        if span_registration_ratio is None:
            raise ValueError(
                "--min-span-registration-ratio requires --span-start-index and --span-end-index."
            )
        checks["span_registration_ratio"] = (
            span_registration_ratio >= args.min_span_registration_ratio
        )
    summary = {
        "valid": all(checks.values()),
        "checks": checks,
        "reference_model": str(reference),
        "candidate_model": str(candidate),
        "selected_frame_count": len(selected_names),
        "registered_selected_frames": len(registered_indices),
        "registration_ratio": registration_ratio,
        "first_registered_index": registered_indices[0] if registered_indices else None,
        "last_registered_index": registered_indices[-1] if registered_indices else None,
        "temporal_coverage": temporal_coverage,
        "span_start_index": args.span_start_index,
        "span_end_index": args.span_end_index,
        "span_frame_count": span_frame_count,
        "span_registered_frames": span_registered_frames,
        "span_registration_ratio": span_registration_ratio,
        "missing_selected_frames": len(selected_names) - len(registered_indices),
        "longest_missing_selected_run": longest_missing_run(
            registered_index_set, len(selected_names)
        ),
        "reference_registered_images": len(reference_images),
        "candidate_registered_images": len(candidate_images),
        "missing_reference_images": len(missing_reference),
        "max_reference_center_delta": max_reference_center_delta,
        "max_reference_rotation_delta_deg": max_reference_rotation_delta,
        "reference_points": point_count(reference),
        "candidate_points": point_count(candidate),
        "added_points": point_count(candidate) - point_count(reference),
        "max_consecutive_center_step": max_center_step,
        "max_new_consecutive_center_step": max(
            new_center_steps, default=float("inf")
        ),
        "unchanged_reference_step_exceptions": unchanged_reference_step_exceptions,
        "p95_consecutive_center_step": percentile(center_steps, 95),
        "max_consecutive_rotation_deg": max(rotation_steps, default=float("inf")),
        "p95_consecutive_rotation_deg": percentile(rotation_steps, 95),
        "anchor_distance_median": percentile(anchor_distances, 50),
        "anchor_distance_p95": percentile(anchor_distances, 95),
        "top_consecutive_center_steps": top_center_steps,
    }
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
