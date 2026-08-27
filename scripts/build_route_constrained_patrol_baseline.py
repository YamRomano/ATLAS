#!/usr/bin/env python3
"""Build a frame-backed, route-constrained patrol baseline.

This utility is for a taught/manual flight whose video is complete but whose
absolute pose stream contains false localization jumps.  It does not pretend
that the rejected poses are measurements.  Instead it associates every source
frame with one of four known patrol legs, freezes X/Y/Z during each waypoint
turn, and advances monotonically along the corresponding patrol segment.

The generated replay is a visual/reference path.  It is never a raw RC command
trace and it never authorizes live translation without a fresh accepted pose.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MAPS_DIR = ROOT / "viewer" / "public" / "maps"
LIVE_SESSIONS_DIR = ROOT / "viewer" / "public" / "live_dji_sessions"
MANIFEST_PATH = MAPS_DIR / "manifest.json"
FRAME_INDEX_RE = re.compile(r"(\d{6})(?:\.[^.]+)?$")


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def finite_vector(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 3:
        return None
    try:
        result = [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def frame_index(image_name: Any) -> int | None:
    match = FRAME_INDEX_RE.search(str(image_name or ""))
    return int(match.group(1)) if match else None


def heading_degrees(start: list[float], end: list[float]) -> float:
    return math.degrees(math.atan2(end[2] - start[2], end[0] - start[0]))


def signed_heading_delta(start: float, end: float) -> float:
    return (float(end) - float(start) + 180.0) % 360.0 - 180.0


def smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    return value * value * (3.0 - 2.0 * value)


def _forward_expansion(previous: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Return radial image expansion and median flow for one recorded step."""
    points = cv2.goodFeaturesToTrack(
        previous,
        maxCorners=600,
        qualityLevel=0.01,
        minDistance=6,
    )
    if points is None:
        return 0.0, 0.0
    moved, ok, _ = cv2.calcOpticalFlowPyrLK(
        previous,
        current,
        points,
        None,
        winSize=(25, 25),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 24, 0.01),
    )
    if moved is None or ok is None:
        return 0.0, 0.0
    valid = ok.reshape(-1).astype(bool)
    p = points.reshape(-1, 2)[valid]
    q = moved.reshape(-1, 2)[valid]
    if len(p) < 30:
        return 0.0, 0.0
    height, width = previous.shape
    design = np.column_stack(
        [
            np.ones(len(p)),
            (p[:, 0] - 0.5 * width) / width,
            (p[:, 1] - 0.5 * height) / height,
        ]
    )
    flow = q - p
    coef_x = np.linalg.lstsq(design, flow[:, 0], rcond=None)[0]
    coef_y = np.linalg.lstsq(design, flow[:, 1], rcond=None)[0]
    expansion = 0.5 * (coef_x[1] / width + coef_y[2] / height)
    median_flow = float(np.median(np.linalg.norm(flow, axis=1)))
    return float(expansion), median_flow


def _clean_motion_activity(raw: list[bool]) -> list[bool]:
    """Close one-frame holes and reject isolated autofocus/noise spikes."""
    active = list(raw)
    for index in range(1, len(active) - 1):
        if not active[index] and active[index - 1] and active[index + 1]:
            active[index] = True
    start = 0
    while start < len(active):
        if not active[start]:
            start += 1
            continue
        end = start + 1
        while end < len(active) and active[end]:
            end += 1
        if end - start < 2:
            active[start:end] = [False] * (end - start)
        start = end
    return active


def recorded_forward_motion_progress(
    frame_dir: Path,
    start_frame: int,
    end_frame: int,
) -> tuple[dict[int, float], dict[str, Any]]:
    """Build a monotonic leg clock that moves only with physical pulses.

    The old baseline advanced position continuously as wall-clock frame number
    increased, including during the long neutral gaps between joystick pulses.
    Radial optical expansion is a direct image-space observation of forward
    camera motion.  The result still begins/ends at the map-picked patrol
    points, but remains stationary when the recorded aircraft was stationary.
    """
    if end_frame <= start_frame:
        raise RuntimeError("Motion-weighted patrol leg must contain at least two frames.")

    def gray(index: int) -> np.ndarray:
        path = frame_dir / f"query_{index:06d}.jpg"
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Missing recorded patrol frame: {path}")
        return cv2.resize(image, (600, 338), interpolation=cv2.INTER_AREA)

    previous = gray(start_frame)
    raw_activity: list[bool] = []
    expansions: list[float] = []
    for index in range(start_frame, end_frame):
        current = gray(index + 1)
        expansion, flow = _forward_expansion(previous, current)
        expansions.append(expansion)
        # At 600 px width, a real forward pulse begins around 0.0035 radial
        # expansion and 0.45 px median motion.  Requiring both rejects neutral
        # camera noise and the almost-uniform horizontal field of a yaw pulse.
        raw_activity.append(expansion >= 0.0035 and flow >= 0.45)
        previous = current
    activity = _clean_motion_activity(raw_activity)
    active_count = sum(activity)
    if active_count < 4:
        raise RuntimeError(
            f"Only {active_count} forward-motion frames found in {start_frame}:{end_frame}."
        )
    progress: dict[int, float] = {start_frame: 0.0}
    elapsed_active = 0
    pulse_count = 0
    was_active = False
    for offset, moving in enumerate(activity):
        if moving and not was_active:
            pulse_count += 1
        if moving:
            elapsed_active += 1
        progress[start_frame + offset + 1] = elapsed_active / active_count
        was_active = moving
    progress[end_frame] = 1.0
    return progress, {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "active_frame_steps": active_count,
        "pulse_count": pulse_count,
        "maximum_expansion": max(expansions) if expansions else 0.0,
        "method": "radial_optical_expansion_binary_pulse_clock",
    }


def interpolate(start: list[float], end: list[float], fraction: float) -> list[float]:
    return [start[index] + (end[index] - start[index]) * fraction for index in range(3)]


def heading_vector(degrees: float) -> list[float]:
    radians = math.radians(degrees)
    return [math.cos(radians), 0.0, math.sin(radians)]


def read_manifest_entry(map_id: str, patrol_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    maps = manifest.get("maps") if isinstance(manifest, dict) else None
    entry = next((item for item in maps or [] if item.get("id") == map_id), None)
    if not isinstance(entry, dict):
        raise RuntimeError(f"Unknown map: {map_id}")
    patrol = next((item for item in entry.get("patrols", []) if item.get("id") == patrol_id), None)
    if not isinstance(patrol, dict):
        raise RuntimeError(f"Unknown patrol on {map_id}: {patrol_id}")
    points = patrol.get("points")
    if not isinstance(points, list) or len(points) != 4:
        raise RuntimeError("A four-point patrol is required.")
    if any(finite_vector(point.get("rxyz") if isinstance(point, dict) else None) is None for point in points):
        raise RuntimeError("Every patrol point must contain a finite rxyz vector.")
    return manifest, entry, patrol


def read_frame_rows(frames_csv: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with frames_csv.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            index = int(raw["source_frame"])
            rows[index] = {
                "image_name": str(raw["image_name"]),
                "time_sec": float(raw["time_sec"]),
                "received_unix": float(raw["received_unix"]),
            }
    if not rows:
        raise RuntimeError(f"No frame timestamps found: {frames_csv}")
    return rows


def read_source_poses(path: Path) -> dict[int, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    poses = raw.get("poses", raw) if isinstance(raw, dict) else raw
    if not isinstance(poses, list):
        raise RuntimeError(f"Source replay has no pose list: {path}")
    result: dict[int, dict[str, Any]] = {}
    for pose in poses:
        if not isinstance(pose, dict):
            continue
        index = frame_index(pose.get("image_name"))
        if index is not None:
            result[index] = pose
    return result


def reliable_cruise_y(source_poses: dict[int, dict[str, Any]], first: int, last: int) -> float:
    values: list[float] = []
    for index, pose in source_poses.items():
        if index < first or index > last:
            continue
        if pose.get("success") is False or pose.get("held_pose") or pose.get("output_rejected"):
            continue
        center = finite_vector(pose.get("rcenter"))
        if center is not None:
            values.append(center[1])
    if not values:
        raise RuntimeError("The reliable prefix has no accepted room-frame height samples.")
    return float(median(values))


def validate_boundaries(boundaries: dict[str, int], available: set[int]) -> None:
    ordered_names = [
        "point1",
        "point2_arrival",
        "point2_departure",
        "point3_arrival",
        "point3_departure",
        "point4_arrival",
        "point4_departure",
        "point1_return",
    ]
    ordered = [boundaries[name] for name in ordered_names]
    if any(right <= left for left, right in zip(ordered, ordered[1:])):
        raise RuntimeError("Patrol phase boundaries must be strictly increasing.")
    missing = [value for value in ordered if value not in available]
    if missing:
        raise RuntimeError(f"Boundary frames are absent from frames.csv: {missing}")


def phase_for_frame(index: int, boundaries: dict[str, int]) -> tuple[str, int, int, int, int]:
    """Return phase, start/end frames, from point, and to point (zero based)."""
    if index <= boundaries["point2_arrival"]:
        return "leg_1_2", boundaries["point1"], boundaries["point2_arrival"], 0, 1
    if index <= boundaries["point2_departure"]:
        return "turn_at_2", boundaries["point2_arrival"], boundaries["point2_departure"], 1, 2
    if index <= boundaries["point3_arrival"]:
        return "leg_2_3", boundaries["point2_departure"], boundaries["point3_arrival"], 1, 2
    if index <= boundaries["point3_departure"]:
        return "turn_at_3", boundaries["point3_arrival"], boundaries["point3_departure"], 2, 3
    if index <= boundaries["point4_arrival"]:
        return "leg_3_4", boundaries["point3_departure"], boundaries["point4_arrival"], 2, 3
    if index <= boundaries["point4_departure"]:
        return "turn_at_4", boundaries["point4_arrival"], boundaries["point4_departure"], 3, 0
    return "leg_4_1", boundaries["point4_departure"], boundaries["point1_return"], 3, 0


def build_poses(
    *,
    frame_rows: dict[int, dict[str, Any]],
    points: list[list[float]],
    boundaries: dict[str, int],
    cruise_y: float,
    motion_progress: dict[str, dict[int, float]] | None = None,
) -> list[dict[str, Any]]:
    headings = [heading_degrees(points[index], points[(index + 1) % 4]) for index in range(4)]
    poses: list[dict[str, Any]] = []
    for index in range(boundaries["point1"], boundaries["point1_return"] + 1):
        row = frame_rows.get(index)
        if row is None:
            raise RuntimeError(f"Missing source frame {index} inside the taught loop.")
        phase, start_frame, end_frame, from_point, to_point = phase_for_frame(index, boundaries)
        raw_progress = (index - start_frame) / max(1, end_frame - start_frame)
        progress = smoothstep(raw_progress)
        phase_progress = (motion_progress or {}).get(phase)
        if phase_progress is not None:
            progress = float(phase_progress.get(index, progress))
        turning = phase.startswith("turn_at_")
        if turning:
            center = points[from_point].copy()
            heading = headings[from_point - 1] + signed_heading_delta(
                headings[from_point - 1], headings[from_point]
            ) * progress
            route_progress = 0.0
        else:
            center = interpolate(points[from_point], points[to_point], progress)
            heading = headings[from_point]
            route_progress = progress
        center[1] = cruise_y
        vector = heading_vector(heading)
        source_image_name = row["image_name"]
        poses.append(
            {
                "instance_id": f"baseline_{index:06d}",
                "success": True,
                "time_sec": row["time_sec"],
                "received_unix": row["received_unix"],
                "image_name": f"query/{source_image_name}",
                "source_frame": index,
                "rcenter": center,
                "rheading": vector,
                "rotation_heading": vector,
                "held_pose": False,
                "output_rejected": False,
                "route_phase": phase,
                "route_progress": route_progress,
                "rotation_position_locked": turning,
                "pose_source": "route_constrained_taught_baseline",
                "position_observation": "patrol_geometry_associated_with_manual_video_frame",
            }
        )
    return poses


def compact_leg_samples(
    poses: list[dict[str, Any]], phase: str, stride: int = 10
) -> list[dict[str, Any]]:
    phase_poses = [pose for pose in poses if pose["route_phase"] == phase]
    selected = phase_poses[:: max(1, stride)]
    if phase_poses and (not selected or selected[-1] is not phase_poses[-1]):
        selected.append(phase_poses[-1])
    return [
        {
            "time_sec": pose["time_sec"],
            "image_name": pose["image_name"],
            "source_frame": pose["source_frame"],
            "rcenter": pose["rcenter"],
            "rheading": pose["rheading"],
        }
        for pose in selected
    ]


def build_reference(
    *,
    map_id: str,
    patrol_id: str,
    source_replay_id: str,
    points: list[list[float]],
    poses: list[dict[str, Any]],
    boundaries: dict[str, int],
    cruise_y: float = 0.0,
    visual_recovery_bank: str | None = None,
    visual_recovery_audit: str | None = None,
) -> dict[str, Any]:
    # Patrol points are picked from the map and may retain historical
    # localization error in their vertical component. The flight planner
    # deliberately commands every patrol target at one fixed altitude; the
    # reference/control contract must describe that same physical route.
    route_points = [
        [float(point[0]), float(cruise_y), float(point[2])]
        for point in points
    ]
    legs = []
    for point_index, phase in enumerate(("leg_1_2", "leg_2_3", "leg_3_4", "leg_4_1")):
        target_index = (point_index + 1) % 4
        # The short 4->1 return passes close to chairs and the window desk, so
        # its image scale changes much faster than on the three long legs.
        # Five-frame anchors keep that real motion visually continuous without
        # weakening the matcher or broadening it to another patrol leg.
        sample_stride = 5 if phase == "leg_4_1" else 10
        legs.append(
            {
                "from_point": point_index + 1,
                "to_point": target_index + 1,
                "from": route_points[point_index],
                "to": route_points[target_index],
                "expected_heading_deg": heading_degrees(route_points[point_index], route_points[target_index]),
                "source_replay": source_replay_id,
                "source_direction": "forward_manual_full_loop",
                "samples": compact_leg_samples(poses, phase, stride=sample_stride),
            }
        )
    reference = {
        "version": 2,
        "kind": "atlas_route_constrained_patrol_reference_candidate",
        "map_id": map_id,
        "patrol_id": patrol_id,
        "source_replays": [source_replay_id],
        "frame": "atlas_room",
        "complete_loop": True,
        "enabled_for_turn_recovery": False,
        "enabled_for_live_route_gate": True,
        "phase_boundaries": boundaries,
        "legs": legs,
        "safety_note": (
            "Frame-backed visual baseline only. Positions during false-localization coverage are "
            "route constraints, not solved R,t measurements. Live translation still requires a "
            "fresh accepted localizer pose and route-progress validation."
        ),
    }
    if visual_recovery_bank and visual_recovery_audit:
        reference.update(
            {
                "enabled_for_visual_route_recovery": True,
                "visual_route_recovery_bank": str(visual_recovery_bank),
                "visual_route_recovery_audit": str(visual_recovery_audit),
            }
        )
    return reference


def publish_replay(
    *,
    manifest: dict[str, Any],
    entry: dict[str, Any],
    replay: dict[str, Any],
    select: bool,
) -> None:
    replays = [item for item in entry.get("replays", []) if item.get("id") != replay["id"]]
    replays.append(replay)
    entry["replays"] = replays
    if select:
        entry["active_replay_id"] = replay["id"]
    entry["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    atomic_write_json(MANIFEST_PATH, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-id", required=True)
    parser.add_argument("--patrol-id", required=True)
    parser.add_argument("--source-replay", required=True)
    parser.add_argument("--source-session", required=True)
    parser.add_argument("--replay-id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--point1-frame", type=int, required=True)
    parser.add_argument("--point2-arrival-frame", type=int, required=True)
    parser.add_argument("--point2-departure-frame", type=int, required=True)
    parser.add_argument("--point3-arrival-frame", type=int, required=True)
    parser.add_argument("--point3-departure-frame", type=int, required=True)
    parser.add_argument("--point4-arrival-frame", type=int, required=True)
    parser.add_argument("--point4-departure-frame", type=int, required=True)
    parser.add_argument("--point1-return-frame", type=int, required=True)
    parser.add_argument("--point1-next-departure-frame", type=int)
    parser.add_argument("--motion-weighted-weak-legs", action="store_true")
    parser.add_argument("--cruise-y", type=float)
    parser.add_argument("--visual-recovery-bank")
    parser.add_argument("--visual-recovery-audit")
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if bool(args.visual_recovery_bank) != bool(args.visual_recovery_audit):
        parser.error("--visual-recovery-bank and --visual-recovery-audit must be provided together")

    manifest, entry, patrol = read_manifest_entry(args.map_id, args.patrol_id)
    points = [finite_vector(point["rxyz"]) for point in patrol["points"]]
    assert all(point is not None for point in points)
    points = [point for point in points if point is not None]
    session_dir = LIVE_SESSIONS_DIR / args.source_session
    frames_csv = session_dir / "query_frames" / "frames.csv"
    source_pose_path = MAPS_DIR / args.map_id / "replays" / args.source_replay / "poses.json"
    frame_rows = read_frame_rows(frames_csv)
    # A fixed audited cruise height makes the old raw (jump-corrupted) replay
    # unnecessary.  Some installations intentionally retain only its frame
    # bank and the route-constrained baseline.
    source_poses = (
        read_source_poses(source_pose_path)
        if args.cruise_y is None
        else {}
    )
    boundaries = {
        "point1": args.point1_frame,
        "point2_arrival": args.point2_arrival_frame,
        "point2_departure": args.point2_departure_frame,
        "point3_arrival": args.point3_arrival_frame,
        "point3_departure": args.point3_departure_frame,
        "point4_arrival": args.point4_arrival_frame,
        "point4_departure": args.point4_departure_frame,
        "point1_return": args.point1_return_frame,
    }
    if args.point1_next_departure_frame is not None:
        if args.point1_next_departure_frame <= args.point1_return_frame:
            parser.error("--point1-next-departure-frame must follow --point1-return-frame")
        boundaries["point1_next_departure"] = args.point1_next_departure_frame
    validate_boundaries(boundaries, set(frame_rows))
    cruise_y = (
        float(args.cruise_y)
        if args.cruise_y is not None
        else reliable_cruise_y(source_poses, args.point1_frame, args.point3_arrival_frame)
    )
    motion_progress: dict[str, dict[int, float]] = {}
    motion_progress_audit: dict[str, Any] = {}
    if args.motion_weighted_weak_legs:
        frame_dir = session_dir / "query_frames"
        for phase, start_name, end_name in (
            ("leg_3_4", "point3_departure", "point4_arrival"),
            ("leg_4_1", "point4_departure", "point1_return"),
        ):
            profile, audit = recorded_forward_motion_progress(
                frame_dir,
                boundaries[start_name],
                boundaries[end_name],
            )
            motion_progress[phase] = profile
            motion_progress_audit[phase] = audit
    poses = build_poses(
        frame_rows=frame_rows,
        points=points,
        boundaries=boundaries,
        cruise_y=cruise_y,
        motion_progress=motion_progress,
    )
    query_frame_base_url = f"public/live_dji_sessions/{args.source_session}/query_frames"
    payload = {
        "mode": "atlas_route_constrained_taught_baseline",
        "description": (
            "Full manual patrol frame sequence with route-monotonic waypoint geometry and "
            "position-locked waypoint rotations."
        ),
        "complete": True,
        "processed_count": len(poses),
        "accepted_count": len(poses),
        "held_count": 0,
        "failed_count": 0,
        "replay_id": args.replay_id,
        "source_replay_id": args.source_replay,
        "source_session": args.source_session,
        "query_frame_base_url": query_frame_base_url,
        "frame_source": query_frame_base_url,
        "phase_boundaries": boundaries,
        "cruise_y": cruise_y,
        "route_constraints": {
            "monotonic_leg_progress": True,
            "rotation_position_locked": True,
            "direct_flight_command_trace": False,
            "requires_fresh_live_localization_for_translation": True,
            "weak_leg_progress_clock": (
                "recorded_forward_image_pulses"
                if args.motion_weighted_weak_legs
                else "frame_time_smoothstep"
            ),
        },
        "motion_progress_audit": motion_progress_audit,
        "updated_at": time.time(),
        "poses": poses,
    }
    out_dir = MAPS_DIR / args.map_id / "replays" / args.replay_id
    reference = build_reference(
        map_id=args.map_id,
        patrol_id=args.patrol_id,
        source_replay_id=args.source_replay,
        points=points,
        poses=poses,
        boundaries=boundaries,
        cruise_y=cruise_y,
        visual_recovery_bank=args.visual_recovery_bank,
        visual_recovery_audit=args.visual_recovery_audit,
    )
    summary = {
        "replay_id": args.replay_id,
        "title": args.title,
        "pose_count": len(poses),
        "first_frame": poses[0]["source_frame"],
        "last_frame": poses[-1]["source_frame"],
        "duration_seconds": poses[-1]["time_sec"] - poses[0]["time_sec"],
        "cruise_y": cruise_y,
        "phase_boundaries": boundaries,
        "selected": bool(args.select and not args.dry_run),
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return

    atomic_write_json(out_dir / "poses.json", payload)
    atomic_write_json(out_dir / "reference_candidate.json", reference)
    replay = {
        "id": args.replay_id,
        "title": args.title,
        "asset_base": f"public/maps/{args.map_id}/replays/{args.replay_id}",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_video": "Live ATLAS 15:47:14 full manual patrol frames",
        "source_replay_id": args.source_replay,
        "kind": "route_constrained_taught_baseline",
        "counts": {
            "poses": len(poses),
            "processed": len(poses),
            "accepted": len(poses),
            "frames": len(poses),
            "held": 0,
            "failed": 0,
        },
        "query_frame_base_url": query_frame_base_url,
        "route_constraints": payload["route_constraints"],
    }
    publish_replay(manifest=manifest, entry=entry, replay=replay, select=args.select)
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_dir / 'poses.json'}")
    print(f"wrote {out_dir / 'reference_candidate.json'}")


if __name__ == "__main__":
    main()
