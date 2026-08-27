#!/usr/bin/env python3
"""Compare saved Live ATLAS pose streams against one taught patrol loop.

This audit intentionally reads real live runs.  It does not transform and
replay the baseline's own images, because that only measures self-consistency
of the reference bank.  The report is meant to stay stable across controller
and localizer revisions so regressions are visible before another flight.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


def finite_center(pose: dict[str, Any]) -> list[float] | None:
    value = pose.get("rcenter")
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        center = [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None
    return center if all(math.isfinite(item) for item in center) else None


def horizontal_distance(left: list[float], right: list[float]) -> float:
    return math.hypot(left[0] - right[0], left[2] - right[2])


def patrol_waypoints(reference: dict[str, Any]) -> list[list[float]]:
    legs = reference.get("legs")
    if not isinstance(legs, list) or len(legs) < 4:
        raise ValueError("patrol reference must contain at least four legs")
    first = legs[0].get("from") if isinstance(legs[0], dict) else None
    points = [first]
    points.extend(leg.get("to") if isinstance(leg, dict) else None for leg in legs[:4])
    result: list[list[float]] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 3:
            raise ValueError("patrol reference contains an invalid leg endpoint")
        result.append([float(point[0]), float(point[1]), float(point[2])])
    return result


def ordered_waypoint_hits(
    poses: Iterable[dict[str, Any]],
    waypoints: list[list[float]],
    *,
    radius: float,
) -> list[dict[str, Any]]:
    """Return consecutive P1, P2, P3, P4, P1 hits in publication order."""
    hits: list[dict[str, Any]] = []
    target_index = 0
    for pose_index, pose in enumerate(poses):
        if target_index >= len(waypoints):
            break
        center = finite_center(pose)
        if center is None:
            continue
        distance = horizontal_distance(center, waypoints[target_index])
        if distance > radius:
            continue
        hits.append(
            {
                "waypoint": 1 if target_index == 4 else target_index + 1,
                "sequence_index": target_index,
                "pose_index": pose_index,
                "time_sec": pose.get("time_sec"),
                "distance": distance,
                "held_pose": bool(pose.get("held_pose")),
                "pose_source": pose.get("pose_source") or "metric_tsolve",
            }
        )
        target_index += 1
    return hits


def analyze_run(
    pose_document: dict[str, Any],
    waypoints: list[list[float]],
    *,
    radius: float,
) -> dict[str, Any]:
    poses = [item for item in pose_document.get("poses") or [] if isinstance(item, dict)]
    hits = ordered_waypoint_hits(poses, waypoints, radius=radius)
    held = sum(bool(pose.get("held_pose")) for pose in poses)
    visual = sum(
        str(pose.get("pose_source") or "") == "patrol_visual_route_recovery"
        for pose in poses
    )
    timeouts = sum(
        "timeout" in str(pose.get("hold_reason") or pose.get("rejected_reason") or "").lower()
        for pose in poses
    )
    reached = [int(hit["waypoint"]) for hit in hits]
    return {
        "replay_id": pose_document.get("replay_id"),
        "pose_count": len(poses),
        "reached_waypoints": reached,
        "reached_point_4": len(hits) >= 4,
        "closed_first_loop": len(hits) >= 5,
        "held_count": held,
        "held_ratio": held / max(1, len(poses)),
        "visual_route_count": visual,
        "visual_route_ratio": visual / max(1, len(poses)),
        "timeout_frame_count": timeouts,
        "hits": hits,
    }


def replay_date(replay_id: str) -> str:
    parts = replay_id.split("_")
    return parts[2] if len(parts) > 2 and len(parts[2]) == 8 else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--from-date", default="")
    parser.add_argument("--to-date", default="99999999")
    parser.add_argument("--radius", type=float, default=0.25)
    parser.add_argument("--minimum-poses", type=int, default=50)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    waypoints = patrol_waypoints(reference)
    rows: list[dict[str, Any]] = []
    for pose_path in sorted(args.replays.glob("dji_live_*/poses.json")):
        replay_id = pose_path.parent.name
        date = replay_date(replay_id)
        if not date or date < args.from_date or date > args.to_date:
            continue
        document = json.loads(pose_path.read_text(encoding="utf-8"))
        row = analyze_run(document, waypoints, radius=max(0.01, args.radius))
        if row["pose_count"] >= max(1, args.minimum_poses):
            rows.append(row)

    progress_counts = {str(index): 0 for index in range(6)}
    for row in rows:
        progress_counts[str(min(5, len(row["reached_waypoints"])))] += 1
    point4_rows = [row for row in rows if row["reached_point_4"]]
    summary = {
        "scope": "real_saved_live_pose_streams",
        "from_date": args.from_date or None,
        "to_date": args.to_date,
        "waypoint_radius": max(0.01, args.radius),
        "minimum_poses": max(1, args.minimum_poses),
        "run_count": len(rows),
        "progress_counts_by_consecutive_hits": progress_counts,
        "reached_point_4_count": len(point4_rows),
        "closed_first_loop_count": sum(row["closed_first_loop"] for row in rows),
        "point4_run_ids": [row["replay_id"] for row in point4_rows],
        "point4_held_ratios": {
            str(row["replay_id"]): row["held_ratio"] for row in point4_rows
        },
        "runs": rows,
    }
    rendered = json.dumps(summary, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
