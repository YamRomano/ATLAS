#!/usr/bin/env python3
"""Publish a bounded camera-path slice for fast visual alignment review."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import atlas_app_server as atlas


ROOT = Path(__file__).resolve().parents[1]


def distance(left: list[float], right: list[float]) -> float:
    return math.dist([float(value) for value in left], [float(value) for value in right])


def floor_distance(left: list[float], right: list[float]) -> float:
    return math.hypot(float(right[0]) - float(left[0]), float(right[2]) - float(left[2]))


def parse_frame_range(value: str) -> tuple[int, int]:
    try:
        start_text, end_text = value.split(":", 1)
        start, end = int(start_text), int(end_text)
    except (AttributeError, TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("frame range must use START:END") from error
    if start < 0 or end <= start:
        raise argparse.ArgumentTypeError("frame range must satisfy 0 <= START < END")
    return start, end


def smooth_path_pacing(
    poses: list[dict],
    frame_range: tuple[int, int],
    *,
    radius: int,
    uniform_blend: float = 0.0,
) -> dict:
    """Retimes a bounded path interval without changing its spatial curve.

    Fixed-map recovery can leave several frames with almost no translation and
    then insert a fast correction.  Smooth the distance travelled per frame,
    and sample the original polyline at that smoother cumulative distance.  The
    first and last positions, frame timestamps, and complete path geometry are
    preserved.
    """
    start, end = frame_range
    if end >= len(poses):
        raise ValueError(f"pacing range {start}:{end} exceeds frame {len(poses) - 1}")
    radius = max(1, int(radius))
    uniform_blend = min(1.0, max(0.0, float(uniform_blend)))
    centers: list[list[float]] = []
    for pose in poses[start : end + 1]:
        center = pose.get("rcenter")
        if not isinstance(center, list) or len(center) != 3:
            raise ValueError("pacing range contains a pose without a valid room center")
        centers.append([float(value) for value in center])

    # Pacing is judged on the visible room floor.  Counting vertical jitter as
    # travel can leave the top-view marker apparently frozen even though the
    # 3D distance is changing, followed by a horizontal catch-up burst.
    steps = [floor_distance(left, right) for left, right in zip(centers, centers[1:])]
    total = sum(steps)
    if total <= 1e-9:
        return {
            "frame_start": start,
            "frame_end": end,
            "radius_frames": radius,
            "distance_m": total,
            "adjusted_pose_count": 0,
        }

    smoothed_steps = []
    for index in range(len(steps)):
        window = steps[max(0, index - radius) : min(len(steps), index + radius + 1)]
        smoothed_steps.append(sum(window) / len(window))
    normalization = total / sum(smoothed_steps)
    smoothed_steps = [step * normalization for step in smoothed_steps]
    if uniform_blend > 0:
        uniform_step = total / len(smoothed_steps)
        smoothed_steps = [
            step * (1.0 - uniform_blend) + uniform_step * uniform_blend
            for step in smoothed_steps
        ]

    source_distances = [0.0]
    for step in steps:
        source_distances.append(source_distances[-1] + step)
    target_distances = [0.0]
    for step in smoothed_steps:
        target_distances.append(target_distances[-1] + step)
    target_distances[-1] = total

    source_segment = 0
    adjusted = 0
    for offset, target_distance in enumerate(target_distances):
        while (
            source_segment + 1 < len(source_distances) - 1
            and source_distances[source_segment + 1] < target_distance
        ):
            source_segment += 1
        left_distance = source_distances[source_segment]
        right_distance = source_distances[source_segment + 1]
        alpha = (
            (target_distance - left_distance) / (right_distance - left_distance)
            if right_distance > left_distance + 1e-12
            else 0.0
        )
        center = [
            left + (right - left) * alpha
            for left, right in zip(centers[source_segment], centers[source_segment + 1])
        ]
        pose = poses[start + offset]
        original = [float(value) for value in pose["rcenter"]]
        if distance(original, center) > 1e-9:
            adjusted += 1
        pose["raw_unpaced_rcenter"] = original
        pose["rcenter"] = center
        pose["preview_pacing_smoothed"] = True
        pose["preview_pacing_range"] = [start, end]

    return {
        "frame_start": start,
        "frame_end": end,
        "radius_frames": radius,
        "uniform_blend": uniform_blend,
        "pacing_plane": "XZ",
        "distance_m": total,
        "adjusted_pose_count": adjusted,
        "original_min_step_m": min(steps),
        "original_max_step_m": max(steps),
        "smoothed_min_step_m": min(smoothed_steps),
        "smoothed_max_step_m": max(smoothed_steps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--preview-id", required=True)
    parser.add_argument("--title", default="Camera Path Alignment Preview")
    parser.add_argument("--target-start-x", type=float)
    parser.add_argument("--target-start-z", type=float)
    parser.add_argument("--movement-scale", type=float, default=1.0)
    parser.add_argument("--smooth-pacing-range", type=parse_frame_range)
    parser.add_argument("--smooth-pacing-radius", type=int, default=30)
    parser.add_argument("--smooth-pacing-uniform-blend", type=float, default=0.0)
    args = parser.parse_args()

    if (args.target_start_x is None) != (args.target_start_z is None):
        parser.error("--target-start-x and --target-start-z must be supplied together")
    if not 0.05 <= float(args.movement_scale) <= 2.0:
        parser.error("--movement-scale must be between 0.05 and 2.0")
    if not 0.0 <= float(args.smooth_pacing_uniform_blend) <= 1.0:
        parser.error("--smooth-pacing-uniform-blend must be between 0 and 1")

    source_run = args.source_run.resolve()
    source_poses_path = source_run / "poses.json"
    source_manifest_path = source_run / "manifest.json"
    source = json.loads(source_poses_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_stream = source_manifest.get("stream") or {}
    all_poses = list(source.get("poses") or [])
    frame_count = min(max(2, int(args.frames)), len(all_poses))
    poses = [copy.deepcopy(pose) for pose in all_poses[:frame_count]]
    if len(poses) < 2:
        raise RuntimeError("The source replay does not contain enough poses for a preview.")

    pacing_smoothing = None
    if args.smooth_pacing_range:
        pacing_smoothing = smooth_path_pacing(
            poses,
            args.smooth_pacing_range,
            radius=max(1, args.smooth_pacing_radius),
            uniform_blend=args.smooth_pacing_uniform_blend,
        )

    alignment_translation_xz = None
    if (
        args.target_start_x is not None
        and args.target_start_z is not None
    ) or not math.isclose(float(args.movement_scale), 1.0):
        first_center = poses[0].get("rcenter")
        if not isinstance(first_center, list) or len(first_center) < 3:
            raise RuntimeError("The first preview pose does not contain a valid room center.")
        source_start_x = float(first_center[0])
        source_start_z = float(first_center[2])
        target_start_x = (
            float(args.target_start_x)
            if args.target_start_x is not None
            else source_start_x
        )
        target_start_z = (
            float(args.target_start_z)
            if args.target_start_z is not None
            else source_start_z
        )
        movement_scale = float(args.movement_scale)
        delta_x = target_start_x - source_start_x
        delta_z = target_start_z - source_start_z
        alignment_translation_xz = [delta_x, delta_z]
        for pose in poses:
            center = pose.get("rcenter")
            if not isinstance(center, list) or len(center) < 3:
                raise RuntimeError("A preview pose does not contain a valid room center.")
            pose["raw_preview_rcenter"] = list(center)
            center[0] = target_start_x + (float(center[0]) - source_start_x) * movement_scale
            center[2] = target_start_z + (float(center[2]) - source_start_z) * movement_scale
            pose["preview_alignment_translation_xz"] = alignment_translation_xz
            pose["preview_movement_scale"] = movement_scale

    indices = [int(pose.get("frame_index")) for pose in poses]
    times = [float(pose.get("time_sec") or 0.0) for pose in poses]
    if indices != list(range(frame_count)):
        raise RuntimeError("The preview source is not a contiguous zero-based pose stream.")
    if any(right <= left for left, right in zip(times, times[1:])):
        raise RuntimeError("The preview source timestamps are not strictly increasing.")

    steps = [
        distance(left["rcenter"], right["rcenter"])
        for left, right in zip(poses, poses[1:])
    ]
    direct_count = sum(not pose.get("interpolated_pose") for pose in poses)
    interpolated_count = frame_count - direct_count
    first = poses[0]
    anchor = first.get("colmap_reference") or {}
    preview_validation = {
        "requires_user_validation": True,
        "scope": "initial placement, first walk, first major turn, and video/pose synchronization",
        "frame_start": 0,
        "frame_end": frame_count - 1,
        "frame_count": frame_count,
        "duration_sec": times[-1] - times[0],
        "direct_pose_count": direct_count,
        "interpolated_pose_count": interpolated_count,
        "max_display_step_m": max(steps, default=0.0),
        "first_room_position": first.get("rcenter"),
        "first_room_heading": first.get("rheading"),
        "first_colmap_registered_points": int(anchor.get("registered_points") or 0),
        "source_replay_id": source_stream.get("replay_id"),
        "operator_selected_start_xz": (
            [float(args.target_start_x), float(args.target_start_z)]
            if args.target_start_x is not None
            else None
        ),
        "preview_alignment_translation_xz": alignment_translation_xz,
        "preview_movement_scale": float(args.movement_scale),
        "pacing_smoothing": pacing_smoothing,
    }
    duration_sec = float(preview_validation["duration_sec"])
    duration_label = (
        f"{max(1, round(duration_sec / 60.0))}-minute"
        if duration_sec >= 90.0
        else f"{max(1, round(duration_sec))}-second"
    )

    output_dir = atlas.CAMERA_PATH_LAB_DIR / "validation_previews" / args.preview_id
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": "camera_path_alignment_validation_preview",
        "description": "Short candidate trajectory for operator review; not a validated full path.",
        "complete": True,
        "expected_count": frame_count,
        "processed_count": frame_count,
        "accepted_count": direct_count,
        "interpolated_count": interpolated_count,
        "validation_preview": preview_validation,
        "poses": poses,
        "updated_at": time.time(),
    }
    poses_path = output_dir / "poses.json"
    atlas.atomic_write_json(poses_path, payload)

    stream = {
        "map_id": source_stream.get("map_id"),
        "replay_id": args.preview_id,
        "title": args.title,
        "asset_base": atlas.public_rel(output_dir),
        "final_pose_url": atlas.public_rel(poses_path),
        "media_url": source_stream.get("media_url"),
        "pose_count": frame_count,
        "accepted_pose_count": direct_count,
        "interpolated_pose_count": interpolated_count,
        "expected_count": frame_count,
        "complete": True,
        "side_project": True,
        "offline_validated": False,
        "validation_preview": True,
        "requires_user_validation": True,
        "source_map_title": source_stream.get("source_map_title"),
        "preview_validation": preview_validation,
        "pacing_smoothed": bool(pacing_smoothing),
    }
    manifest = {
        "generated_at": time.time(),
        "status": "preview",
        "message": (
            f"{duration_label} alignment preview ready: frames 0–{frame_count - 1}. "
            "The start was aligned to the operator-selected point; check the first turn "
            "and video synchronization."
            + (" Path pacing was smoothed across the selected preview interval." if pacing_smoothing else "")
        ),
        "stream": stream,
    }
    atlas.atomic_write_json(output_dir / "manifest.json", manifest)
    atlas.atomic_write_json(atlas.CAMERA_PATH_LAB_DIR / "offline_latest.json", manifest)
    print(json.dumps({"ok": True, "preview": stream}, indent=2))


if __name__ == "__main__":
    main()
