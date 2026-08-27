#!/usr/bin/env python3
"""Build a conservative visual reference loop from complementary live runs.

This utility deliberately does *not* create raw RC commands.  The Android
bridge does not expose manual-stick telemetry, so treating video timing as a
command trace would be unsafe.  Instead it records the best accepted visual
segments, their coverage, and the expected room-frame headings for each patrol
leg.  The live bridge may use this only while yawing in place; forward motion
remains gated by a fresh accepted TSolve position.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MAPS_DIR = ROOT / "viewer" / "public" / "maps"
MANIFEST_PATH = MAPS_DIR / "manifest.json"


def horizontal_distance(a: list[float], b: list[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[2]) - float(b[2]))


def heading_degrees(a: list[float], b: list[float]) -> float:
    return math.degrees(math.atan2(float(b[2]) - float(a[2]), float(b[0]) - float(a[0])))


def signed_heading_delta(from_degrees: float, to_degrees: float) -> float:
    delta = float(to_degrees) - float(from_degrees)
    while delta > 180.0:
        delta -= 360.0
    while delta < -180.0:
        delta += 360.0
    return delta


def accepted_poses(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    poses = raw.get("poses", raw) if isinstance(raw, dict) else raw
    if not isinstance(poses, list):
        raise RuntimeError(f"Invalid replay pose file: {path}")
    out: list[dict[str, Any]] = []
    for pose in poses:
        if not isinstance(pose, dict):
            continue
        if pose.get("success") is False or pose.get("held_pose") or pose.get("output_rejected"):
            continue
        center = pose.get("rcenter")
        if not isinstance(center, list) or len(center) < 3:
            continue
        try:
            xyz = [float(center[0]), float(center[1]), float(center[2])]
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in xyz):
            continue
        out.append({**pose, "rcenter": xyz})
    if not out:
        raise RuntimeError(f"Replay has no accepted room-frame TSolve poses: {path}")
    return out


def nearest_index(poses: list[dict[str, Any]], point: list[float]) -> tuple[int, float]:
    return min(
        (
            (index, horizontal_distance(pose["rcenter"], point))
            for index, pose in enumerate(poses)
        ),
        key=lambda item: item[1],
    )


def compact_samples(poses: list[dict[str, Any]], *, stride: int = 8) -> list[dict[str, Any]]:
    """Keep a small, auditable visual trace rather than a huge replay copy."""
    selected = poses[::max(1, stride)]
    if poses and (not selected or selected[-1] is not poses[-1]):
        selected.append(poses[-1])
    result: list[dict[str, Any]] = []
    for pose in selected:
        heading = pose.get("rheading")
        result.append(
            {
                "time_sec": pose.get("time_sec"),
                "image_name": pose.get("image_name"),
                "rcenter": pose["rcenter"],
                "rheading": heading if isinstance(heading, list) and len(heading) >= 3 else None,
            }
        )
    return result


def read_patrol(map_id: str, patrol_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    maps = manifest.get("maps") if isinstance(manifest, dict) else None
    entry = next((item for item in maps or [] if item.get("id") == map_id), None)
    if not isinstance(entry, dict):
        raise RuntimeError(f"Unknown map: {map_id}")
    patrol = next((item for item in entry.get("patrols", []) if item.get("id") == patrol_id), None)
    if not isinstance(patrol, dict):
        raise RuntimeError(f"Unknown patrol on map {map_id}: {patrol_id}")
    points = patrol.get("points")
    if not isinstance(points, list) or len(points) < 4:
        raise RuntimeError("A four-point patrol is required to stitch this reference loop.")
    return entry, patrol


def point_xyz(point: dict[str, Any]) -> list[float]:
    value = point.get("rxyz") if isinstance(point, dict) else None
    if not isinstance(value, list) or len(value) < 3:
        raise RuntimeError("Patrol point is missing rxyz coordinates.")
    return [float(value[0]), float(value[1]), float(value[2])]


def replay_pose_path(map_dir: Path, replay_id: str) -> Path:
    replay_dir = map_dir / "replays" / replay_id
    for name in ("poses.json", "poses_partial.json"):
        candidate = replay_dir / name
        if candidate.exists():
            return candidate
    raise RuntimeError(f"Replay poses were not found: {replay_dir}")


def make_leg(
    *,
    point_from: int,
    point_to: int,
    start: list[float],
    end: list[float],
    source_replay: str,
    source_direction: str,
    poses: list[dict[str, Any]],
    source_indices: tuple[int, int],
) -> dict[str, Any]:
    if len(poses) < 2:
        raise RuntimeError(f"Taught leg {point_from}->{point_to} has fewer than two accepted poses.")
    return {
        "from_point": point_from,
        "to_point": point_to,
        "from": start,
        "to": end,
        "expected_heading_deg": heading_degrees(start, end),
        "source_replay": source_replay,
        "source_direction": source_direction,
        "source_indices": list(source_indices),
        "accepted_pose_count": len(poses),
        "samples": compact_samples(poses),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-id", required=True)
    parser.add_argument("--patrol-id", required=True)
    parser.add_argument("--forward-replay", required=True, help="Replay containing point 1 -> 2 -> 3.")
    second_source = parser.add_mutually_exclusive_group(required=True)
    second_source.add_argument(
        "--return-replay",
        help="Replay containing point 1 -> 4 -> 3; it is reversed into 3 -> 4 -> 1.",
    )
    second_source.add_argument(
        "--continuation-replay",
        help="Replay taught directly from point 3 -> 4 -> 1; use this to finish an existing 1 -> 2 -> 3 run.",
    )
    second_source.add_argument(
        "--final-leg-replay",
        help="Replay taught directly from point 4 -> 1; the forward replay must already contain 1 -> 2 -> 3 -> 4.",
    )
    parser.add_argument("--max-point-error", type=float, default=0.55)
    parser.add_argument("--max-junction-error", type=float, default=0.55)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    _entry, patrol = read_patrol(args.map_id, args.patrol_id)
    points = [point_xyz(point) for point in patrol["points"][:4]]
    map_dir = MAPS_DIR / args.map_id
    forward_path = replay_pose_path(map_dir, args.forward_replay)
    second_replay = str(args.final_leg_replay or args.continuation_replay or args.return_replay)
    second_path = replay_pose_path(map_dir, second_replay)
    forward = accepted_poses(forward_path)
    returning = accepted_poses(second_path)

    f1, f1_error = nearest_index(forward, points[0])
    f2, f2_error = nearest_index(forward, points[1])
    f3, f3_error = nearest_index(forward, points[2])
    f4, f4_error = nearest_index(forward, points[3])
    r1, r1_error = nearest_index(returning, points[0])
    r4, r4_error = nearest_index(returning, points[3])
    r3, r3_error = nearest_index(returning, points[2])
    if not (f1 < f2 < f3):
        raise RuntimeError("Forward replay does not visit patrol points 1 -> 2 -> 3 in order.")
    point_4_junction = 0.0
    if args.final_leg_replay:
        if not (f1 < f2 < f3 < f4):
            raise RuntimeError("Forward replay does not visit patrol points 1 -> 2 -> 3 -> 4 in order.")
        if not (r4 < r1):
            raise RuntimeError("Final-leg replay does not visit patrol points 4 -> 1 in order.")
        leg_34_poses = forward[f3:f4 + 1]
        leg_41_poses = returning[r4:r1 + 1]
        leg_34_indices = (f3, f4)
        leg_41_indices = (r4, r1)
        leg_34_replay = args.forward_replay
        leg_34_direction = "forward"
        leg_41_direction = "forward_final_leg"
        point_4_junction = horizontal_distance(forward[f4]["rcenter"], returning[r4]["rcenter"])
        reversed_trace_note = None
    elif args.continuation_replay:
        if not (r3 < r4 < r1):
            raise RuntimeError("Continuation replay does not visit patrol points 3 -> 4 -> 1 in order.")
        leg_34_poses = returning[r3:r4 + 1]
        leg_41_poses = returning[r4:r1 + 1]
        leg_34_indices = (r3, r4)
        leg_41_indices = (r4, r1)
        leg_34_replay = second_replay
        leg_34_direction = "forward_continuation"
        leg_41_direction = "forward_continuation"
        reversed_trace_note = None
    else:
        if not (r1 < r4 < r3):
            raise RuntimeError("Return replay does not visit patrol points 1 -> 4 -> 3 in order for safe reversal.")

        # The second recording is physically traversed 1 -> 4 -> 3.  Its accepted
        # position trace is reversed to form a reference *path* 3 -> 4 -> 1.  Its
        # camera headings are intentionally not copied as control headings: the
        # aircraft must face the opposite way on the reversed route.
        return_reversed = list(reversed(returning[r1:r3 + 1]))
        reverse_p3_index = len(return_reversed) - 1 - (r3 - r1)
        reverse_p4_index = len(return_reversed) - 1 - (r4 - r1)
        reverse_p1_index = len(return_reversed) - 1
        if not (0 <= reverse_p3_index < reverse_p4_index < reverse_p1_index):
            raise RuntimeError("Unable to form an ordered reversed 3 -> 4 -> 1 trace.")
        leg_34_poses = return_reversed[reverse_p3_index:reverse_p4_index + 1]
        leg_41_poses = return_reversed[reverse_p4_index:reverse_p1_index + 1]
        leg_34_indices = (r3, r4)
        leg_41_indices = (r4, r1)
        leg_34_replay = second_replay
        leg_34_direction = "reversed_position_trace"
        leg_41_direction = "reversed_position_trace"
        reversed_trace_note = (
            "Legs 3->4 and 4->1 use reversed position traces; their original camera headings "
            "must not be replayed because the aircraft faces the opposite direction."
        )

    legs = [
        make_leg(
            point_from=1,
            point_to=2,
            start=points[0],
            end=points[1],
            source_replay=args.forward_replay,
            source_direction="forward",
            poses=forward[f1:f2 + 1],
            source_indices=(f1, f2),
        ),
        make_leg(
            point_from=2,
            point_to=3,
            start=points[1],
            end=points[2],
            source_replay=args.forward_replay,
            source_direction="forward",
            poses=forward[f2:f3 + 1],
            source_indices=(f2, f3),
        ),
        make_leg(
            point_from=3,
            point_to=4,
            start=points[2],
            end=points[3],
            source_replay=leg_34_replay,
            source_direction=leg_34_direction,
            poses=leg_34_poses,
            source_indices=leg_34_indices,
        ),
        make_leg(
            point_from=4,
            point_to=1,
            start=points[3],
            end=points[0],
            source_replay=second_replay,
            source_direction=leg_41_direction,
            poses=leg_41_poses,
            source_indices=leg_41_indices,
        ),
    ]
    headings = [leg["expected_heading_deg"] for leg in legs]
    for index, leg in enumerate(legs):
        leg["turn_from_previous_deg"] = signed_heading_delta(headings[index - 1], headings[index])

    coverage = {
        "point_1_forward_error": f1_error,
        "point_2_error": f2_error,
        "point_3_forward_error": f3_error,
        "point_3_return_error": f3_error if args.final_leg_replay else r3_error,
        "point_4_error": max(f4_error, r4_error) if args.final_leg_replay else r4_error,
        "point_1_return_error": r1_error,
    }
    p3_junction = (
        0.0
        if args.final_leg_replay
        else horizontal_distance(forward[f3]["rcenter"], returning[r3]["rcenter"])
    )
    p1_junction = horizontal_distance(returning[r1]["rcenter"], forward[f1]["rcenter"])
    coverage["point_3_junction_error"] = p3_junction
    coverage["point_1_junction_error"] = p1_junction
    coverage["point_4_junction_error"] = point_4_junction
    complete = (
        max(coverage[key] for key in (
            "point_1_forward_error", "point_2_error", "point_3_forward_error",
            "point_3_return_error", "point_4_error", "point_1_return_error",
        )) <= args.max_point_error
        and max(p3_junction, point_4_junction, p1_junction) <= args.max_junction_error
    )

    output = {
        "version": 1,
        "kind": "atlas_taught_patrol_reference",
        "map_id": args.map_id,
        "patrol_id": args.patrol_id,
        "created_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
        "source_replays": [args.forward_replay, second_replay],
        "second_replay_mode": (
            "direct_final_leg"
            if args.final_leg_replay
            else ("direct_continuation" if args.continuation_replay else "reversed_return")
        ),
        "frame": "atlas_room",
        "complete_loop": complete,
        "enabled_for_turn_recovery": complete,
        "max_point_error_map_units": args.max_point_error,
        "max_junction_error_map_units": args.max_junction_error,
        "coverage": coverage,
        "legs": legs,
        "safety_note": (
            "Reference supplies expected turn headings only. It never authorizes forward movement "
            "without a fresh accepted TSolve room position."
        ),
        "reversed_trace_note": reversed_trace_note,
    }
    out_path = args.out or (map_dir / "taught_patrols" / args.patrol_id / "reference.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "legs"}, indent=2))
    print(f"wrote {out_path}")
    if not complete:
        raise SystemExit("Reference did not meet the requested coverage/junction limits.")


if __name__ == "__main__":
    main()
