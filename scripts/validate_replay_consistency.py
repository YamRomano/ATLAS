#!/usr/bin/env python3
"""Independently validate a finalized uploaded-video localization replay."""

from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def finite_center(pose: dict[str, Any]) -> list[float] | None:
    value = pose.get("center")
    if not isinstance(value, list) or len(value) < 3:
        return None
    try:
        center = [float(value[index]) for index in range(3)]
    except (TypeError, ValueError):
        return None
    return center if all(math.isfinite(item) for item in center) else None


def accepted_poses(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("poses", payload) if isinstance(payload, dict) else payload
    poses = [
        pose
        for pose in raw
        if isinstance(pose, dict)
        and pose.get("success") is not False
        and not pose.get("held_pose")
        and not pose.get("output_rejected")
        and finite_center(pose) is not None
        and pose.get("time_sec") is not None
    ]
    return payload if isinstance(payload, dict) else {}, poses


def distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))


def rotation_delta_deg(left: dict[str, Any], right: dict[str, Any]) -> float | None:
    left_r, right_r = left.get("R"), right.get("R")
    if not (
        isinstance(left_r, list)
        and isinstance(right_r, list)
        and len(left_r) == 3
        and len(right_r) == 3
    ):
        return None
    try:
        trace = sum(
            sum(float(right_r[row][axis]) * float(left_r[row][axis]) for row in range(3))
            for axis in range(3)
        )
        cosine = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
        return math.degrees(math.acos(cosine))
    except (IndexError, TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--trusted", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--extraction-metadata", type=Path, required=True)
    parser.add_argument("--max-step", type=float, default=0.55)
    parser.add_argument("--max-gap-seconds", type=float, default=2.0)
    parser.add_argument("--trusted-time-tolerance", type=float, default=0.15)
    parser.add_argument("--critical-start", type=float, default=295.0)
    parser.add_argument("--critical-end", type=float, default=330.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    payload, poses = accepted_poses(args.candidate)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    extraction = json.loads(args.extraction_metadata.read_text(encoding="utf-8"))
    times = [float(pose["time_sec"]) for pose in poses]
    centers = [finite_center(pose) for pose in poses]
    centers = [center for center in centers if center is not None]
    steps = [distance(left, right) for left, right in zip(centers, centers[1:])]
    gaps = [right - left for left, right in zip(times, times[1:])]
    rotations = [
        value
        for left, right in zip(poses, poses[1:])
        if (value := rotation_delta_deg(left, right)) is not None
    ]
    colmap_reference_deltas = [
        distance(center, reference_center)
        for pose, center in zip(poses, centers)
        if isinstance(pose.get("colmap_reference"), dict)
        and (reference_center := finite_center(pose["colmap_reference"])) is not None
    ]
    output_rejected = [
        item for item in summary.get("output_rejected", []) if isinstance(item, dict)
    ]
    rejected_ids = {str(item.get("case_id")) for item in output_rejected if item.get("case_id")}
    exported_ids = {str(pose.get("instance_id")) for pose in poses}
    processed = int(summary.get("processed_frames") or 0)
    solved = int(summary.get("accepted_cases") or 0)
    expected_accepted = solved - int(summary.get("output_rejected_cases") or 0)
    duration = float(extraction.get("duration_sec") or 0.0)
    temporal_coverage = (times[-1] - times[0]) / duration if times and duration > 0 else 0.0

    critical_indices = [
        index for index, value in enumerate(times) if args.critical_start <= value <= args.critical_end
    ]
    critical_steps = [
        steps[index]
        for index in critical_indices
        if index < len(steps) and times[index + 1] <= args.critical_end
    ]
    critical_rotations = [
        rotations[index]
        for index in critical_indices
        if index < len(rotations) and times[index + 1] <= args.critical_end
    ]

    trusted_comparison = None
    if args.trusted:
        _, trusted = accepted_poses(args.trusted)
        trusted_times = [float(pose["time_sec"]) for pose in trusted]
        matched_distances: list[float] = []
        matched_time_deltas: list[float] = []
        for pose, time_value, center in zip(poses, times, centers):
            insertion = bisect.bisect_left(trusted_times, time_value)
            candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(trusted)]
            if not candidates:
                continue
            nearest = min(candidates, key=lambda index: abs(trusted_times[index] - time_value))
            delta = abs(trusted_times[nearest] - time_value)
            trusted_center = finite_center(trusted[nearest])
            if delta <= args.trusted_time_tolerance and trusted_center is not None:
                matched_time_deltas.append(delta)
                matched_distances.append(distance(center, trusted_center))
        trusted_comparison = {
            "path": str(args.trusted),
            "accepted_records": len(trusted),
            "matched_records": len(matched_distances),
            "median_time_delta_seconds": median(matched_time_deltas) if matched_time_deltas else None,
            "median_center_delta_m": median(matched_distances) if matched_distances else None,
            "p95_center_delta_m": percentile(matched_distances, 0.95),
            "max_center_delta_m": max(matched_distances, default=None),
        }

    checks = {
        "complete": bool(payload.get("complete", True)),
        "all_frames_processed": processed == int(summary.get("query_frames") or 0),
        "accepted_count_matches_guard": len(poses) == expected_accepted,
        "timestamps_strictly_increasing": all(right > left for left, right in zip(times, times[1:])),
        "temporal_coverage": temporal_coverage >= 0.98,
        "no_unguarded_output_rejections": not (rejected_ids & exported_ids),
        "consecutive_center_steps": bool(steps) and max(steps) <= args.max_step,
        "pose_gaps": not gaps or max(gaps) <= args.max_gap_seconds,
        "critical_turn_continuity": bool(critical_indices) and max(critical_steps, default=0.0) <= args.max_step,
    }
    report = {
        "valid": all(checks.values()),
        "checks": checks,
        "candidate": str(args.candidate),
        "counts": {
            "query_frames": int(summary.get("query_frames") or 0),
            "processed_frames": processed,
            "solver_accepted": solved,
            "output_rejected": len(output_rejected),
            "final_accepted_poses": len(poses),
        },
        "time": {
            "first_seconds": times[0] if times else None,
            "last_seconds": times[-1] if times else None,
            "duration_seconds": duration,
            "temporal_coverage": temporal_coverage,
            "max_pose_gap_seconds": max(gaps, default=None),
        },
        "continuity": {
            "max_center_step_m": max(steps, default=None),
            "p95_center_step_m": percentile(steps, 0.95),
            "max_rotation_step_deg": max(rotations, default=None),
            "p95_rotation_step_deg": percentile(rotations, 0.95),
        },
        "colmap_reference_agreement": {
            "compared_poses": len(colmap_reference_deltas),
            "median_center_delta_m": median(colmap_reference_deltas) if colmap_reference_deltas else None,
            "p95_center_delta_m": percentile(colmap_reference_deltas, 0.95),
            "max_center_delta_m": max(colmap_reference_deltas, default=None),
        },
        "critical_turn": {
            "start_seconds": args.critical_start,
            "end_seconds": args.critical_end,
            "accepted_poses": len(critical_indices),
            "max_center_step_m": max(critical_steps, default=None),
            "max_rotation_step_deg": max(critical_rotations, default=None),
        },
        "trusted_comparison": trusted_comparison,
    }
    rendered = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if report["valid"] else 2)


if __name__ == "__main__":
    main()
