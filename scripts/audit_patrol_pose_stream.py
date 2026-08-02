#!/usr/bin/env python3
"""Audit a saved/localized pose stream as a command-free patrol shadow run.

The audit uses the same room-frame X/Z geometry and arrival radii as the live
DJI patrol controller.  It never opens the DJI bridge and never writes to a
map, manifest, patrol, or replay.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


def horizontal_distance(a: list[float], b: list[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[2]) - float(b[2]))


def segment_distance(point: list[float], start: list[float], end: list[float]) -> float:
    px, pz = float(point[0]), float(point[2])
    ax, az = float(start[0]), float(start[2])
    bx, bz = float(end[0]), float(end[2])
    dx, dz = bx - ax, bz - az
    length_sq = dx * dx + dz * dz
    if length_sq <= 1e-12:
        return math.hypot(px - ax, pz - az)
    fraction = max(0.0, min(1.0, ((px - ax) * dx + (pz - az) * dz) / length_sq))
    return math.hypot(px - (ax + fraction * dx), pz - (az + fraction * dz))


def finite_vector(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 3:
        return None
    try:
        result = [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def pose_time(pose: dict[str, Any]) -> float | None:
    try:
        value = float(pose.get("time_sec"))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def load_patrol(manifest_path: Path, map_id: str, patrol_id: str) -> tuple[dict[str, Any], list[list[float]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next((item for item in manifest.get("maps", []) if item.get("id") == map_id), None)
    if not isinstance(entry, dict):
        raise RuntimeError(f"Map not found in manifest: {map_id}")
    patrol = next((item for item in entry.get("patrols", []) if item.get("id") == patrol_id), None)
    if not isinstance(patrol, dict):
        raise RuntimeError(f"Patrol not found on {map_id}: {patrol_id}")
    points = [finite_vector(item.get("rxyz")) for item in patrol.get("points", []) if isinstance(item, dict)]
    if len(points) < 2 or any(point is None for point in points):
        raise RuntimeError("Patrol must contain at least two valid room-frame points.")
    return patrol, [point for point in points if point is not None]


def load_map_entry(manifest_path: Path, map_id: str) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next((item for item in manifest.get("maps", []) if item.get("id") == map_id), None)
    if not isinstance(entry, dict):
        raise RuntimeError(f"Map not found in manifest: {map_id}")
    return entry


def explicit_room_transform(entry: dict[str, Any]):
    matrix = (entry.get("room_alignment") or {}).get("matrix")
    if not isinstance(matrix, list) or len(matrix) != 3:
        return None
    try:
        rows = [[float(value) for value in row] for row in matrix]
    except (TypeError, ValueError):
        return None
    if any(len(row) != 4 or not all(math.isfinite(value) for value in row) for row in rows):
        return None

    def transform(xyz: list[float]) -> list[float]:
        return [sum(row[index] * float(xyz[index]) for index in range(3)) + row[3] for row in rows]

    return transform


def load_pose_stream(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    poses = raw.get("poses", raw) if isinstance(raw, dict) else raw
    if not isinstance(poses, list):
        raise RuntimeError(f"Pose stream has no pose list: {path}")
    return raw if isinstance(raw, dict) else {}, [pose for pose in poses if isinstance(pose, dict)]


def accepted_pose(pose: dict[str, Any]) -> bool:
    return bool(
        pose.get("success") is not False
        and not pose.get("held_pose")
        and not pose.get("output_rejected")
        and finite_vector(pose.get("rcenter")) is not None
        and pose_time(pose) is not None
    )


def endpoint_drift(reference_path: Path | None, points: list[list[float]]) -> dict[str, Any] | None:
    if reference_path is None or not reference_path.exists():
        return None
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    legs = reference.get("legs", []) if isinstance(reference, dict) else []
    by_point: dict[int, list[float]] = {}
    for leg in legs:
        if not isinstance(leg, dict):
            continue
        for key, field in (("from_point", "from"), ("to_point", "to")):
            try:
                index = int(leg.get(key)) - 1
            except (TypeError, ValueError):
                continue
            vector = finite_vector(leg.get(field))
            if vector is not None and index not in by_point:
                by_point[index] = vector
    differences = [
        {
            "point": index + 1,
            "horizontal_drift": horizontal_distance(points[index], value),
            "current": points[index],
            "reference": value,
        }
        for index, value in sorted(by_point.items())
        if index < len(points)
    ]
    return {
        "path": str(reference_path),
        "complete_loop": reference.get("complete_loop"),
        "point_differences": differences,
        "max_horizontal_drift": max((item["horizontal_drift"] for item in differences), default=0.0),
        "stale_for_current_patrol": any(item["horizontal_drift"] > 0.05 for item in differences),
    }


def audit(
    *,
    pose_path: Path,
    manifest_path: Path,
    map_id: str,
    alignment_map_id: str | None,
    patrol_id: str,
    arrival_radius: float,
    soft_radius: float,
    max_step: float,
    max_pose_gap_seconds: float,
    max_cross_track: float,
    min_acceptance_ratio: float,
    taught_reference: Path | None,
) -> dict[str, Any]:
    patrol, points = load_patrol(manifest_path, map_id, patrol_id)
    stream, all_poses = load_pose_stream(pose_path)
    alignment_map_id = alignment_map_id or map_id
    room_transform = explicit_room_transform(load_map_entry(manifest_path, alignment_map_id))
    if room_transform is not None:
        all_poses = [
            {
                **pose,
                "rcenter": room_transform(center),
            }
            if (center := finite_vector(pose.get("center"))) is not None
            else pose
            for pose in all_poses
        ]
    accepted = [pose for pose in all_poses if accepted_pose(pose)]
    centers = [finite_vector(pose.get("rcenter")) for pose in accepted]
    times = [pose_time(pose) for pose in accepted]
    centers = [center for center in centers if center is not None]
    times = [value for value in times if value is not None]

    held_count = sum(bool(pose.get("held_pose")) for pose in all_poses)
    rejected_count = sum(bool(pose.get("output_rejected")) for pose in all_poses)
    failed_count = sum(pose.get("success") is False for pose in all_poses)
    denominator = max(1, int(stream.get("processed_count") or len(all_poses)))
    acceptance_ratio = len(accepted) / denominator

    steps: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    for left_pose, right_pose, left, right, left_time, right_time in zip(
        accepted, accepted[1:], centers, centers[1:], times, times[1:]
    ):
        step = horizontal_distance(left, right)
        gap = max(0.0, right_time - left_time)
        if step > max_step:
            steps.append(
                {
                    "from_time": left_time,
                    "to_time": right_time,
                    "step": step,
                    "from_image": left_pose.get("image_name"),
                    "to_image": right_pose.get("image_name"),
                }
            )
        if gap > max_pose_gap_seconds:
            gaps.append(
                {
                    "from_time": left_time,
                    "to_time": right_time,
                    "gap_seconds": gap,
                    "from_image": left_pose.get("image_name"),
                    "to_image": right_pose.get("image_name"),
                }
            )

    nearest: list[dict[str, Any]] = []
    for index, target in enumerate(points):
        if not accepted:
            nearest.append({"point": index + 1, "distance": None, "time_sec": None})
            continue
        best_index, best_distance = min(
            enumerate(centers), key=lambda item: horizontal_distance(item[1], target)
        )
        nearest.append(
            {
                "point": index + 1,
                "distance": horizontal_distance(centers[best_index], target),
                "time_sec": times[best_index],
                "image_name": accepted[best_index].get("image_name"),
            }
        )

    # Find one ordered loop exactly as the live circle controller expects it:
    # point 1 -> ... -> final point -> point 1.
    ordered_targets = points + [points[0]]
    ordered_hits: list[dict[str, Any]] = []
    cursor = 0
    for target_index, target in enumerate(ordered_targets):
        hit = next(
            (
                index
                for index in range(cursor, len(centers))
                if horizontal_distance(centers[index], target) <= soft_radius
            ),
            None,
        )
        if hit is None:
            break
        distance = horizontal_distance(centers[hit], target)
        ordered_hits.append(
            {
                "point": (target_index % len(points)) + 1,
                "time_sec": times[hit],
                "distance": distance,
                "arrival": "strict" if distance <= arrival_radius else "soft",
                "pose_index": hit,
                "image_name": accepted[hit].get("image_name"),
            }
        )
        cursor = hit + 1

    legs: list[dict[str, Any]] = []
    if len(ordered_hits) == len(ordered_targets):
        for leg_index, (left_hit, right_hit) in enumerate(zip(ordered_hits, ordered_hits[1:])):
            left_index = int(left_hit["pose_index"])
            right_index = int(right_hit["pose_index"])
            start = points[leg_index % len(points)]
            end = points[(leg_index + 1) % len(points)]
            errors = [segment_distance(center, start, end) for center in centers[left_index : right_index + 1]]
            legs.append(
                {
                    "from_point": (leg_index % len(points)) + 1,
                    "to_point": ((leg_index + 1) % len(points)) + 1,
                    "duration_seconds": times[right_index] - times[left_index],
                    "samples": len(errors),
                    "median_cross_track": median(errors) if errors else None,
                    "max_cross_track": max(errors) if errors else None,
                    "within_live_corridor": bool(errors) and max(errors) <= max_cross_track,
                }
            )

    stale_reference = endpoint_drift(taught_reference, points)
    failures: list[str] = []
    if not bool(stream.get("complete", True)):
        failures.append("pose stream is not marked complete")
    if not accepted:
        failures.append("no accepted room-frame poses")
    if acceptance_ratio < min_acceptance_ratio:
        failures.append(
            f"accepted-pose ratio {acceptance_ratio:.3f} is below {min_acceptance_ratio:.3f}"
        )
    if steps:
        failures.append(f"{len(steps)} accepted room-position jumps exceed {max_step:.2f}")
    if gaps:
        failures.append(f"{len(gaps)} accepted-pose gaps exceed {max_pose_gap_seconds:.1f}s")
    if len(ordered_hits) != len(ordered_targets):
        next_point = (len(ordered_hits) % len(points)) + 1
        failures.append(f"ordered patrol loop is incomplete; next missing target is point {next_point}")
    bad_legs = [leg for leg in legs if not leg["within_live_corridor"]]
    if bad_legs:
        failures.append(f"{len(bad_legs)} patrol legs leave the {max_cross_track:.2f} live corridor")
    if stale_reference and stale_reference["stale_for_current_patrol"]:
        failures.append("taught patrol reference endpoints do not match the current patrol coordinates")

    return {
        "kind": "atlas_command_free_patrol_shadow_audit",
        "safe": not failures,
        "pose_stream": str(pose_path),
        "map_id": map_id,
        "alignment_map_id": alignment_map_id,
        "transform_source": "explicit_room_alignment" if room_transform is not None else "pose_rcenter",
        "patrol_id": patrol_id,
        "patrol_title": patrol.get("title"),
        "thresholds": {
            "arrival_radius": arrival_radius,
            "soft_radius": soft_radius,
            "max_pose_step": max_step,
            "max_pose_gap_seconds": max_pose_gap_seconds,
            "max_cross_track": max_cross_track,
            "min_acceptance_ratio": min_acceptance_ratio,
        },
        "pose_counts": {
            "processed": denominator,
            "records": len(all_poses),
            "accepted": len(accepted),
            "held": held_count,
            "rejected": rejected_count,
            "failed": failed_count,
            "acceptance_ratio": acceptance_ratio,
        },
        "time_span_seconds": (times[-1] - times[0]) if len(times) >= 2 else 0.0,
        "nearest_points": nearest,
        "ordered_loop": {
            "complete": len(ordered_hits) == len(ordered_targets),
            "required_hits": len(ordered_targets),
            "hits": ordered_hits,
        },
        "legs": legs,
        "continuity": {
            "oversized_step_count": len(steps),
            "top_oversized_steps": sorted(steps, key=lambda item: item["step"], reverse=True)[:20],
            "long_gap_count": len(gaps),
            "top_long_gaps": sorted(gaps, key=lambda item: item["gap_seconds"], reverse=True)[:20],
        },
        "taught_reference": stale_reference,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-stream", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("viewer/public/maps/manifest.json"))
    parser.add_argument("--map-id", required=True)
    parser.add_argument(
        "--alignment-map-id",
        help="Map providing the explicit room alignment for center-to-rcenter conversion.",
    )
    parser.add_argument("--patrol-id", required=True)
    parser.add_argument("--arrival-radius", type=float, default=0.24)
    parser.add_argument("--soft-radius", type=float, default=0.38)
    parser.add_argument("--max-step", type=float, default=0.55)
    parser.add_argument("--max-pose-gap-seconds", type=float, default=8.0)
    parser.add_argument("--max-cross-track", type=float, default=0.80)
    parser.add_argument("--min-acceptance-ratio", type=float, default=0.75)
    parser.add_argument("--taught-reference", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = audit(
        pose_path=args.pose_stream,
        manifest_path=args.manifest,
        map_id=args.map_id,
        alignment_map_id=args.alignment_map_id,
        patrol_id=args.patrol_id,
        arrival_radius=args.arrival_radius,
        soft_radius=args.soft_radius,
        max_step=args.max_step,
        max_pose_gap_seconds=args.max_pose_gap_seconds,
        max_cross_track=args.max_cross_track,
        min_acceptance_ratio=args.min_acceptance_ratio,
        taught_reference=args.taught_reference,
    )
    rendered = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if report["safe"] else 2)


if __name__ == "__main__":
    main()
