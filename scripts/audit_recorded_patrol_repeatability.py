#!/usr/bin/env python3
"""Measure frame-to-pose lag and repeatability for recorded patrol simulations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


LEGS = {
    "point3_to_point4": (2901, 3900),
    "point4_to_point1": (4261, 4400),
}


def source_frame(pose: dict[str, Any]) -> int | None:
    value = str(pose.get("instance_id") or "").rsplit("_", 1)[-1]
    try:
        return int(value)
    except ValueError:
        return None


def center(pose: dict[str, Any] | None) -> list[float] | None:
    value = pose.get("rcenter") if isinstance(pose, dict) else None
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        result = [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def distance_xz(left: list[float] | None, right: list[float] | None) -> float | None:
    if left is None or right is None:
        return None
    return math.hypot(left[0] - right[0], left[2] - right[2])


def load_poses(path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    indexed: dict[int, dict[str, Any]] = {}
    for pose in document.get("poses") or []:
        if not isinstance(pose, dict):
            continue
        frame = source_frame(pose)
        if frame is not None:
            indexed[frame] = pose
    return document, indexed


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def freeze_intervals(
    run: dict[int, dict[str, Any]],
    physical: dict[int, dict[str, Any]],
    first: int,
    last: int,
    *,
    model_step_epsilon: float = 0.000001,
    minimum_physical_displacement: float = 0.02,
) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    active_start: int | None = None
    previous_frame: int | None = None

    def finish(end_frame: int | None) -> None:
        nonlocal active_start
        if active_start is None or end_frame is None or end_frame <= active_start:
            active_start = None
            return
        physical_displacement = distance_xz(
            center(physical.get(active_start)),
            center(physical.get(end_frame)),
        )
        if physical_displacement is not None and physical_displacement >= minimum_physical_displacement:
            intervals.append(
                {
                    "start_frame": active_start,
                    "end_frame": end_frame,
                    "frame_span": end_frame - active_start + 1,
                    "physical_baseline_displacement_m": physical_displacement,
                    "model_displacement_m": distance_xz(
                        center(run.get(active_start)),
                        center(run.get(end_frame)),
                    ),
                }
            )
        active_start = None

    for frame in range(first + 1, last + 1):
        previous = center(run.get(frame - 1))
        current = center(run.get(frame))
        step = distance_xz(previous, current)
        frozen = step is not None and step <= model_step_epsilon
        if frozen and active_start is None:
            active_start = frame - 1
        elif not frozen:
            finish(previous_frame)
        previous_frame = frame
    finish(previous_frame)
    return sorted(
        intervals,
        key=lambda item: float(item["physical_baseline_displacement_m"]),
        reverse=True,
    )


def leg_metrics(
    run: dict[int, dict[str, Any]],
    physical: dict[int, dict[str, Any]],
    first: int,
    last: int,
) -> dict[str, Any]:
    errors: list[float] = []
    maximum_error: tuple[float, int] | None = None
    held_frames = 0
    visual_frames = 0
    for frame in range(first, last + 1):
        pose = run.get(frame)
        error = distance_xz(center(pose), center(physical.get(frame)))
        if error is not None:
            errors.append(error)
            if maximum_error is None or error > maximum_error[0]:
                maximum_error = (error, frame)
        held_frames += bool(isinstance(pose, dict) and pose.get("held_pose"))
        visual_frames += bool(
            isinstance(pose, dict)
            and pose.get("pose_source") == "patrol_visual_route_recovery"
        )
    freezes = freeze_intervals(run, physical, first, last)
    return {
        "first_frame": first,
        "last_frame": last,
        "frame_count": last - first + 1,
        "mean_position_error_m": sum(errors) / len(errors) if errors else None,
        "p95_position_error_m": percentile(errors, 0.95),
        "maximum_position_error_m": maximum_error[0] if maximum_error else None,
        "maximum_position_error_frame": maximum_error[1] if maximum_error else None,
        "held_frames": held_frames,
        "visual_recovery_frames": visual_frames,
        "freeze_interval_count": len(freezes),
        "largest_freeze_intervals": freezes[:12],
    }


def run_metrics(
    path: Path,
    document: dict[str, Any],
    indexed: dict[int, dict[str, Any]],
    physical: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "path": str(path),
        "replay_id": document.get("replay_id") or path.parent.name,
        "expected_count": document.get("expected_count"),
        "processed_count": document.get("processed_count"),
        "accepted_count": document.get("accepted_count"),
        "held_count": document.get("held_count"),
        "failed_count": document.get("failed_count"),
        "complete": document.get("complete"),
        "legs": {
            name: leg_metrics(indexed, physical, first, last)
            for name, (first, last) in LEGS.items()
        },
    }


def cross_run_metrics(
    left: dict[int, dict[str, Any]],
    right: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    errors: list[float] = []
    maximum: tuple[float, int] | None = None
    held_state_mismatches = 0
    for frame in sorted(set(left) & set(right)):
        error = distance_xz(center(left.get(frame)), center(right.get(frame)))
        if error is not None:
            errors.append(error)
            if maximum is None or error > maximum[0]:
                maximum = (error, frame)
        held_state_mismatches += bool(left[frame].get("held_pose")) != bool(
            right[frame].get("held_pose")
        )
    return {
        "compared_frames": len(errors),
        "mean_center_delta_m": sum(errors) / len(errors) if errors else None,
        "p95_center_delta_m": percentile(errors, 0.95),
        "maximum_center_delta_m": maximum[0] if maximum else None,
        "maximum_center_delta_frame": maximum[1] if maximum else None,
        "frames_over_0_05m": sum(error > 0.05 for error in errors),
        "frames_over_0_10m": sum(error > 0.10 for error in errors),
        "held_state_mismatches": held_state_mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-baseline", required=True, type=Path)
    parser.add_argument("--runs", required=True, nargs="+", type=Path)
    args = parser.parse_args()

    _physical_document, physical = load_poses(args.physical_baseline)
    loaded = [(path, *load_poses(path)) for path in args.runs]
    report: dict[str, Any] = {
        "physical_baseline": str(args.physical_baseline),
        "runs": [
            run_metrics(path, document, indexed, physical)
            for path, document, indexed in loaded
        ],
        "cross_run": [],
    }
    for index in range(1, len(loaded)):
        report["cross_run"].append(
            {
                "left": str(loaded[0][0]),
                "right": str(loaded[index][0]),
                **cross_run_metrics(loaded[0][2], loaded[index][2]),
            }
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
