#!/usr/bin/env python3
"""Stitch accepted bounded-localization segments onto a target frame timeline.

The input streams may overlap and may contain held/rejected poses.  Only
successful, non-held, non-rejected poses with finite camera transforms are
eligible.  Each target frame is assigned the nearest eligible pose in time,
within a small caller-provided tolerance.  This is useful when localization
was audited at 10 FPS but COLMAP enhancement images were extracted at 1 FPS.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def finite_vector(value: object, length: int) -> bool:
    if not isinstance(value, list) or len(value) != length:
        return False
    try:
        return all(math.isfinite(float(item)) for item in value)
    except (TypeError, ValueError):
        return False


def finite_rotation(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(finite_vector(row, 3) for row in value)
    )


def accepted_pose(pose: object) -> bool:
    return (
        isinstance(pose, dict)
        and pose.get("success") is True
        and not pose.get("held_pose")
        and not pose.get("output_rejected")
        and finite_rotation(pose.get("R"))
        and finite_vector(pose.get("t"), 3)
        and finite_vector(pose.get("center"), 3)
        and finite_vector(pose.get("rcenter"), 3)
        and isinstance(pose.get("time_sec"), (int, float))
        and math.isfinite(float(pose["time_sec"]))
    )


def pose_rank(pose: dict[str, Any]) -> tuple[float, float]:
    reference = pose.get("colmap_reference")
    registered = (
        float(reference.get("registered_points", 0))
        if isinstance(reference, dict)
        else 0.0
    )
    objective = pose.get("objective")
    objective_rank = (
        -float(objective)
        if isinstance(objective, (int, float)) and math.isfinite(float(objective))
        else float("-inf")
    )
    return registered, objective_rank


def load_accepted(paths: list[Path]) -> list[dict[str, Any]]:
    by_source_name: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        poses = payload.get("poses", payload) if isinstance(payload, dict) else payload
        if not isinstance(poses, list):
            raise RuntimeError(f"Invalid pose stream: {path}")
        for raw in poses:
            if not accepted_pose(raw):
                continue
            pose = dict(raw)
            source_name = str(pose.get("image_name", ""))
            pose["_source_pose_json"] = str(path.resolve())
            current = by_source_name.get(source_name)
            if current is None or pose_rank(pose) > pose_rank(current):
                by_source_name[source_name] = pose
    accepted = sorted(by_source_name.values(), key=lambda item: float(item["time_sec"]))
    if not accepted:
        raise RuntimeError("No trusted poses were found in the supplied streams.")
    return accepted


def load_target_frames(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            try:
                time_sec = float(row["time_sec"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"Invalid frame row {index} in {path}") from exc
            rows.append(
                {
                    "target_index": index,
                    "image_name": str(row["image_name"]),
                    "time_sec": time_sec,
                }
            )
    if not rows:
        raise RuntimeError(f"No target frames found in {path}")
    return rows


def nearest_pose(
    accepted: list[dict[str, Any]],
    times: list[float],
    target_time: float,
) -> dict[str, Any]:
    insertion = bisect.bisect_left(times, target_time)
    candidates = accepted[max(0, insertion - 1) : min(len(accepted), insertion + 1)]
    return min(candidates, key=lambda pose: abs(float(pose["time_sec"]) - target_time))


def gap_runs(selected_indices: set[int], first: int, last: int) -> list[dict[str, int]]:
    gaps: list[dict[str, int]] = []
    start: int | None = None
    for index in range(first, last + 1):
        if index not in selected_indices and start is None:
            start = index
        elif index in selected_indices and start is not None:
            gaps.append({"first_index": start, "last_index": index - 1, "frames": index - start})
            start = None
    if start is not None:
        gaps.append({"first_index": start, "last_index": last, "frames": last - start + 1})
    return gaps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses-json", type=Path, action="append", required=True)
    parser.add_argument("--target-frames-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--max-time-offset", type=float, default=0.12)
    args = parser.parse_args()

    accepted = load_accepted(args.poses_json)
    accepted_times = [float(pose["time_sec"]) for pose in accepted]
    targets = load_target_frames(args.target_frames_csv)

    stitched: list[dict[str, Any]] = []
    for target in targets:
        nearest = nearest_pose(accepted, accepted_times, float(target["time_sec"]))
        offset = abs(float(nearest["time_sec"]) - float(target["time_sec"]))
        if offset > args.max_time_offset:
            continue
        pose = {
            key: value
            for key, value in nearest.items()
            if not key.startswith("_")
        }
        pose["source_image_name"] = pose.get("image_name")
        pose["source_time_sec"] = pose.get("time_sec")
        pose["source_pose_json"] = nearest["_source_pose_json"]
        pose["target_frame_index"] = target["target_index"]
        pose["time_offset_sec"] = offset
        pose["image_name"] = f"query/{target['image_name']}"
        pose["time_sec"] = target["time_sec"]
        stitched.append(pose)

    if not stitched:
        raise RuntimeError("No target frames were within the requested time tolerance.")

    selected_indices = {int(pose["target_frame_index"]) for pose in stitched}
    first_index = min(selected_indices)
    last_index = max(selected_indices)
    gaps = gap_runs(selected_indices, first_index, last_index)

    consecutive_steps: list[dict[str, Any]] = []
    for left, right in zip(stitched, stitched[1:]):
        if int(right["target_frame_index"]) != int(left["target_frame_index"]) + 1:
            continue
        center_step = float(
            np.linalg.norm(
                np.asarray(right["center"], dtype=float)
                - np.asarray(left["center"], dtype=float)
            )
        )
        room_step = float(
            np.linalg.norm(
                np.asarray(right["rcenter"], dtype=float)
                - np.asarray(left["rcenter"], dtype=float)
            )
        )
        consecutive_steps.append(
            {
                "left": left["image_name"],
                "right": right["image_name"],
                "center_step": center_step,
                "room_center_step": room_step,
            }
        )

    span_count = last_index - first_index + 1
    summary = {
        "mode": "trusted_bounded_pose_stitch",
        "source_pose_streams": [str(path.resolve()) for path in args.poses_json],
        "target_frames_csv": str(args.target_frames_csv.resolve()),
        "target_frame_count": len(targets),
        "trusted_source_pose_count": len(accepted),
        "stitched_pose_count": len(stitched),
        "first_target_index": first_index,
        "last_target_index": last_index,
        "covered_span_frames": span_count,
        "coverage_within_span": len(stitched) / span_count,
        "coverage_of_all_targets": len(stitched) / len(targets),
        "max_time_offset_sec": max(float(pose["time_offset_sec"]) for pose in stitched),
        "missing_gap_count": len(gaps),
        "longest_missing_gap_frames": max((gap["frames"] for gap in gaps), default=0),
        "missing_gaps": gaps,
        "max_consecutive_center_step": max(
            (step["center_step"] for step in consecutive_steps), default=None
        ),
        "max_consecutive_room_center_step": max(
            (step["room_center_step"] for step in consecutive_steps), default=None
        ),
        "top_consecutive_steps": sorted(
            consecutive_steps,
            key=lambda step: float(step["center_step"]),
            reverse=True,
        )[:10],
    }
    output = {
        "mode": "trusted_bounded_pose_stitch",
        "complete": True,
        "processed_count": len(stitched),
        "accepted_count": len(stitched),
        "held_count": 0,
        "failed_count": 0,
        "poses": stitched,
        "summary": summary,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    summary_path = args.summary_out or args.out.with_name(f"{args.out.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
