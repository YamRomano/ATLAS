#!/usr/bin/env python3
"""Build and validate a complete Camera Path replay against a fixed ATLAS map.

This is deliberately separate from the simulated-live UI job.  It processes a
finite video offline, permits synchronous fixed-map SIFT/Faiss recovery, and only
publishes a replay after absolute-anchor and continuity validation passes.  It
never edits the selected map, its COLMAP model, patrols, or safety barriers.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import atlas_app_server as atlas  # noqa: E402


FRAME_RE = re.compile(r"(\d+)(?=\.[^.]+$)")


def run_logged(command: list[object], log_path: Path) -> None:
    command = [str(item) for item in command]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("+ " + " ".join(command), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("+ " + " ".join(command) + "\n")
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def frame_index(pose: dict[str, Any]) -> int | None:
    name = str(pose.get("image_name") or "")
    match = FRAME_RE.search(name)
    return int(match.group(1)) if match else None


def load_frame_rows(frames_csv: Path) -> list[dict[str, str]]:
    with frames_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        image_name = row.get("filename") or row.get("image_name")
        if not image_name:
            raise RuntimeError("Extracted frame metadata is missing filename/image_name.")
        # Older extractor output used filename; the current extractor uses
        # image_name.  Normalize once so replay interpolation remains stable.
        row["filename"] = image_name
    rows.sort(key=lambda row: int(FRAME_RE.search(row["filename"]).group(1)))
    return rows


def vec(pose: dict[str, Any], key: str) -> list[float] | None:
    value = pose.get(key)
    if not isinstance(value, list) or len(value) != 3:
        return None
    numbers = [float(item) for item in value]
    return numbers if all(math.isfinite(item) for item in numbers) else None


def lerp(left: list[float], right: list[float], alpha: float) -> list[float]:
    return [a + (b - a) * alpha for a, b in zip(left, right)]


def normalized_lerp(left: list[float], right: list[float], alpha: float) -> list[float]:
    value = lerp(left, right, alpha)
    norm = math.sqrt(sum(item * item for item in value))
    return [item / norm for item in value] if norm > 1e-9 else list(left)


def trusted_pose(pose: dict[str, Any]) -> bool:
    return bool(
        pose.get("success")
        and not pose.get("held_pose")
        and not pose.get("output_rejected")
        and vec(pose, "center")
        and vec(pose, "rcenter")
    )


def smooth_absolute_anchor_drift(
    trusted_by_frame: dict[int, dict[str, Any]],
    *,
    min_correction_m: float = 0.08,
) -> tuple[dict[int, dict[str, Any]], int]:
    """Distribute fixed-map recovery corrections across their tracking interval.

    Optical flow is continuous but slowly drifts.  A later independent COLMAP
    pose corrects that drift in one frame, which is mathematically useful but
    produces a visible path spike.  Preserve every independent anchor exactly
    and spread only its accumulated correction over the frames since the prior
    anchor.  Missing frames are filled *after* this pass, so they cannot retain
    the pre-correction spike.
    """
    smoothed = {index: copy.deepcopy(pose) for index, pose in trusted_by_frame.items()}
    anchor_indices = sorted(
        index
        for index, pose in smoothed.items()
        if isinstance(pose.get("colmap_reference"), dict)
    )
    adjusted = 0
    for left_anchor, right_anchor in zip(anchor_indices, anchor_indices[1:]):
        interior = sorted(index for index in smoothed if left_anchor < index < right_anchor)
        if not interior:
            continue
        last_index = interior[-1]
        last_center = vec(smoothed[last_index], "rcenter")
        anchor_center = vec(smoothed[right_anchor], "rcenter")
        if not last_center or not anchor_center:
            continue

        # Estimate the final local-tracking velocity without including the
        # absolute recovery step itself.  Component medians resist one noisy
        # optical-flow frame while retaining real walking motion.
        recent = [index for index in [left_anchor, *interior] if index <= last_index][-6:]
        velocities: list[list[float]] = []
        for first_index, second_index in zip(recent, recent[1:]):
            first = smoothed[first_index]
            second = smoothed[second_index]
            elapsed = float(second.get("time_sec") or 0.0) - float(first.get("time_sec") or 0.0)
            first_center = vec(first, "rcenter")
            second_center = vec(second, "rcenter")
            if elapsed > 1e-6 and first_center and second_center:
                velocities.append(
                    [(second_center[axis] - first_center[axis]) / elapsed for axis in range(3)]
                )
        velocity = (
            [statistics.median(value[axis] for value in velocities) for axis in range(3)]
            if velocities
            else [0.0, 0.0, 0.0]
        )
        tail_elapsed = max(
            0.0,
            float(smoothed[right_anchor].get("time_sec") or 0.0)
            - float(smoothed[last_index].get("time_sec") or 0.0),
        )
        predicted_anchor = [
            last_center[axis] + velocity[axis] * tail_elapsed for axis in range(3)
        ]
        correction = [
            anchor_center[axis] - predicted_anchor[axis] for axis in range(3)
        ]
        if math.dist(correction, [0.0, 0.0, 0.0]) < min_correction_m:
            continue

        left_time = float(smoothed[left_anchor].get("time_sec") or 0.0)
        right_time = float(smoothed[right_anchor].get("time_sec") or left_time)
        duration = max(1e-6, right_time - left_time)
        for index in interior:
            pose = smoothed[index]
            center = vec(pose, "rcenter")
            if not center:
                continue
            alpha = min(1.0, max(0.0, (float(pose.get("time_sec") or left_time) - left_time) / duration))
            pose["raw_offline_rcenter"] = center
            pose["rcenter"] = [
                center[axis] + alpha * correction[axis] for axis in range(3)
            ]
            pose["offline_anchor_drift_smoothed"] = True
            adjusted += 1
    return smoothed, adjusted


def prepare_browser_video(video: Path, asset_dir: Path) -> Path:
    """Publish a seekable web video, transcoding QuickTime phone media if needed."""
    media_dir = asset_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    if video.suffix.lower() not in {".mov", ".qt"}:
        atlas.copy_video_to_public(video, asset_dir)
        return media_dir / "drone_query.mp4"

    output = media_dir / "drone_query.m4v"
    if output.is_file() and output.stat().st_size > 0:
        old_link = media_dir / "drone_query.mp4"
        if old_link.is_symlink():
            old_link.unlink()
        return output
    pending = media_dir / "drone_query.pending.m4v"
    if pending.exists():
        pending.unlink()
    subprocess.run(
        [
            "/usr/bin/avconvert",
            "--source",
            str(video),
            "--preset",
            "PresetAppleM4V720pHD",
            "--output",
            str(pending),
            "--replace",
            "--disableMetadataFilter",
            "--progress",
        ],
        check=True,
        cwd=str(ROOT),
    )
    if not pending.is_file() or pending.stat().st_size <= 0:
        raise RuntimeError("The web-video conversion completed without an output file.")
    os.replace(pending, output)
    old_link = media_dir / "drone_query.mp4"
    if old_link.is_symlink():
        old_link.unlink()
    return output


def fill_short_gaps(
    trusted_by_frame: dict[int, dict[str, Any]],
    frame_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], int]:
    expected = len(frame_rows)
    indices = sorted(trusted_by_frame)
    if not indices or indices[0] != 0 or indices[-1] != expected - 1:
        raise RuntimeError(
            f"Trusted trajectory does not cover both video ends: "
            f"first={indices[0] if indices else None}, last={indices[-1] if indices else None}, expected={expected}."
        )
    complete: list[dict[str, Any]] = []
    interpolated = 0
    cursor = 0
    for index in range(expected):
        if index in trusted_by_frame:
            pose = copy.deepcopy(trusted_by_frame[index])
        else:
            while cursor + 1 < len(indices) and indices[cursor + 1] < index:
                cursor += 1
            left_index = indices[cursor]
            right_index = indices[cursor + 1]
            left = trusted_by_frame[left_index]
            right = trusted_by_frame[right_index]
            alpha = (index - left_index) / (right_index - left_index)
            pose = copy.deepcopy(left if alpha < 0.5 else right)
            for key in ("center", "rcenter"):
                pose[key] = lerp(vec(left, key), vec(right, key), alpha)
            left_heading = vec(left, "rheading")
            right_heading = vec(right, "rheading")
            if left_heading and right_heading:
                pose["rheading"] = normalized_lerp(left_heading, right_heading, alpha)
            pose.update(
                {
                    "instance_id": f"offline_interpolated_{index:06d}",
                    "image_name": frame_rows[index]["filename"],
                    "success": True,
                    "held_pose": False,
                    "output_rejected": False,
                    "interpolated_pose": True,
                    "interpolation_bounds": [left_index, right_index],
                }
            )
            interpolated += 1
        pose["frame_index"] = index
        pose["time_sec"] = float(frame_rows[index]["time_sec"])
        complete.append(pose)
    return complete, interpolated


def point_in_polygon(x: float, z: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, (xi, zi) in enumerate(polygon):
        xj, zj = polygon[j]
        crosses = (zi > z) != (zj > z)
        if crosses and x < (xj - xi) * (z - zi) / ((zj - zi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def validate_raw_trajectory(
    poses: list[dict[str, Any]],
    expected: int,
    map_entry: dict[str, Any],
    max_gap_frames: int,
    min_trusted_ratio: float,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    trusted_by_frame = {
        index: pose
        for pose in poses
        if trusted_pose(pose) and (index := frame_index(pose)) is not None
    }
    indices = sorted(trusted_by_frame)
    if not indices:
        raise RuntimeError("Offline localizer produced no trusted poses.")
    gaps = [right - left - 1 for left, right in zip(indices, indices[1:])]
    max_gap = max(gaps, default=0)
    accepted_ratio = len(indices) / max(1, expected)
    temporal_coverage = (indices[-1] - indices[0] + 1) / max(1, expected)

    steps: list[float] = []
    step_allowances: list[float] = []
    for left_index, right_index in zip(indices, indices[1:]):
        left = vec(trusted_by_frame[left_index], "center")
        right = vec(trusted_by_frame[right_index], "center")
        steps.append(math.dist(left, right))
        left_time = float(trusted_by_frame[left_index].get("time_sec") or 0.0)
        right_time = float(trusted_by_frame[right_index].get("time_sec") or left_time)
        elapsed = max(0.0, right_time - left_time)
        # A finite phone video can cross nearly the full long axis of this room
        # while optical flow is weak.  Keep the speed gate (which rejects an
        # instantaneous repeated-room alias) but let a sufficiently long gap
        # recover anywhere within the measured room-length envelope.
        step_allowances.append(min(8.0, max(1.6, 0.18 + 1.6 * elapsed)))
    max_step = max(steps, default=0.0)
    max_step_excess = max(
        (step - allowance for step, allowance in zip(steps, step_allowances)),
        default=0.0,
    )

    first = trusted_by_frame.get(0)
    if first is None:
        raise RuntimeError(f"Frame 0 did not receive a trusted absolute pose; first trusted frame is {indices[0]}.")
    reference = first.get("colmap_reference") or {}
    reference_center = reference.get("center")
    first_center = vec(first, "center")
    if not isinstance(reference_center, list) or len(reference_center) != 3:
        raise RuntimeError("Frame 0 is missing its independent COLMAP reference pose.")
    anchor_error = math.dist(first_center, [float(item) for item in reference_center])
    registered_points = int(reference.get("registered_points") or 0)

    barriers = map_entry.get("safety_barriers") or []
    polygon = []
    for barrier in barriers:
        a = barrier.get("a") or []
        if len(a) >= 3:
            polygon.append((float(a[0]), float(a[2])))
    first_room = vec(first, "rcenter")
    first_inside_room = bool(
        len(polygon) >= 3 and point_in_polygon(first_room[0], first_room[2], polygon)
    )

    validation = {
        "valid": bool(
            indices[0] == 0
            and indices[-1] == expected - 1
            and accepted_ratio >= min_trusted_ratio
            and temporal_coverage >= 0.98
            and max_gap <= max_gap_frames
            and max_step_excess <= 1e-6
            and anchor_error <= 1e-6
            and registered_points >= 100
            and first_inside_room
        ),
        "expected_frames": expected,
        "trusted_frames": len(indices),
        "accepted_ratio": accepted_ratio,
        "required_trusted_ratio": min_trusted_ratio,
        "temporal_coverage": temporal_coverage,
        "first_trusted_frame": indices[0],
        "last_trusted_frame": indices[-1],
        "max_missing_gap_frames": max_gap,
        "allowed_gap_frames": max_gap_frames,
        "max_trusted_center_step_m": max_step,
        "max_time_scaled_step_excess_m": max_step_excess,
        "recovery_max_speed_mps": 1.6,
        "recovery_max_total_step_m": 8.0,
        "first_anchor_center": first_center,
        "first_anchor_room_center": first_room,
        "first_anchor_colmap_error_m": anchor_error,
        "first_anchor_registered_points": registered_points,
        "first_anchor_inside_room": first_inside_room,
    }
    return validation, trusted_by_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--map-id", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--title", default="Offline Camera Path")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--relocalize-every", type=int, default=60)
    parser.add_argument("--max-gap-frames", type=int, default=45)
    parser.add_argument(
        "--min-trusted-ratio",
        type=float,
        default=0.60,
        help=(
            "Minimum directly trusted frame ratio for a bounded-interpolation replay. "
            "Interpolated gaps must still satisfy the independent endpoint, gap, and speed checks."
        ),
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Resume an interrupted validated-offline run from its last trusted TSolve case.",
    )
    args = parser.parse_args()

    video = args.video.resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    cfg = atlas.load_config()
    map_entry = atlas.get_map_entry(args.map_id)
    map_artifacts = atlas.colmap_artifacts_for_entry(map_entry)
    base_asset_dir = atlas.VIEWER / map_entry["asset_base"]
    run_id = args.run_id.strip() or f"offline_{time.strftime('%Y%m%d_%H%M%S')}"
    run_root = ROOT / "results" / "camera_path_offline" / map_entry["id"] / run_id
    query_frames = run_root / "query_frames"
    runtime_code = run_root / "tsolve_runtime_code"
    tsolve_inputs = run_root / "tsolve_inputs"
    tsolve_output = run_root / "tsolve_output"
    work_dir = run_root / "fixed_map_work"
    partial_path = run_root / "poses_partial.json"
    log_path = run_root / "offline_build.log"
    out_asset_dir = atlas.CAMERA_PATH_LAB_DIR / "offline_runs" / run_id
    final_pose_path = out_asset_dir / "poses.json"
    latest_manifest_path = atlas.CAMERA_PATH_LAB_DIR / "offline_latest.json"

    run_root.mkdir(parents=True, exist_ok=True)
    metadata_path = query_frames / "metadata.json"
    metadata = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = int(metadata.get("expected_frames") or 0)
    extracted = len(list(query_frames.glob("query_*.jpg"))) if query_frames.is_dir() else 0
    if not metadata.get("complete") or expected <= 0 or extracted != expected:
        if query_frames.exists():
            shutil.rmtree(query_frames)
        run_logged(
            [
                cfg["python"],
                SCRIPTS / "extract_frames.py",
                "--video",
                video,
                "--out-dir",
                query_frames,
                "--fps",
                args.fps,
                "--max-size",
                cfg["max_image_size"],
                "--prefix",
                "query",
            ],
            log_path,
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = int(metadata["expected_frames"])
        extracted = len(list(query_frames.glob("query_*.jpg")))
    if expected <= 0 or extracted != expected:
        raise RuntimeError(f"Frame extraction incomplete: {extracted}/{expected}.")

    if not (runtime_code / "yam_code").is_dir() or not (runtime_code / "harness").is_dir():
        run_logged(
            [
                cfg["python"],
                SCRIPTS / "setup_tsolve_runtime.py",
                "--base-yam-code-dir",
                cfg["base_yam_code_dir"],
                "--dropin-patch-dir",
                cfg["dropin_patch_dir"],
                "--base-harness-dir",
                cfg["base_harness_dir"],
                "--out-dir",
                runtime_code,
            ],
            log_path,
        )

    faiss_spec = atlas.faiss_index_command(cfg, map_artifacts)
    if faiss_spec is None:
        raise RuntimeError("Direct OpenCV SIFT localization requires the Faiss map index.")
    faiss_index_dir, faiss_build_cmd = faiss_spec
    run_logged(faiss_build_cmd, log_path)

    resume_args: list[object] = []
    if args.resume_existing:
        if not partial_path.is_file():
            raise FileNotFoundError(f"Cannot resume without {partial_path}")
        partial_doc = json.loads(partial_path.read_text(encoding="utf-8"))
        prior_poses = list(partial_doc.get("poses") or [])
        trusted = next((pose for pose in reversed(prior_poses) if trusted_pose(pose)), None)
        if trusted is None:
            raise RuntimeError("Cannot resume: partial stream has no trusted pose.")
        last_frame = frame_index(trusted)
        instance_id = str(trusted.get("instance_id") or "")
        resume_case = tsolve_inputs / "inputs" / instance_id
        if last_frame is None or not (resume_case / "input.json").is_file():
            raise RuntimeError(
                f"Cannot resume from frame={last_frame}, case={resume_case}; TSolve case is incomplete."
            )
        resume_args = [
            "--start-frame-index",
            last_frame + 1,
            "--resume-pose-stream",
            partial_path,
            "--resume-case-dir",
            resume_case,
        ]
        print(
            f"Resuming offline trajectory at frame {last_frame + 1}/{expected} "
            f"from {instance_id}.",
            flush=True,
        )
    else:
        atlas.atomic_write_json(
            partial_path,
            {
                "mode": "camera_path_offline_partial",
                "replay_id": run_id,
                "expected_count": expected,
                "processed_count": 0,
                "complete": False,
                "poses": [],
            },
        )
    command: list[object] = [
            cfg["python"],
            SCRIPTS / "run_bounded_tsolve_video_stream.py",
            "--colmap",
            cfg["colmap_bin"],
            "--map-database",
            map_artifacts["database"],
            "--map-images",
            map_artifacts["images"],
            "--map-sparse-model",
            map_artifacts["sparse_model"],
            "--map-sparse-text",
            map_artifacts["sparse_text"],
            "--query-frames",
            query_frames,
            "--runtime-dir",
            runtime_code,
            "--solver-dir",
            cfg["solver_dir"],
            "--inputs-out-dir",
            tsolve_inputs,
            "--out-dir",
            tsolve_output,
            "--work-dir",
            work_dir,
            "--max-image-size",
            cfg["max_image_size"],
            "--query-camera-model",
            cfg["query_camera_model"],
            "--query-camera-params",
            cfg.get("query_camera_params", ""),
            "--sift-max-num-features",
            cfg.get("live_sift_max_num_features", 1024),
            "--min-points",
            cfg["min_query_correspondences"],
            "--max-points",
            cfg["max_query_correspondences"],
            "--max-reference-images",
            48,
            "--tracking-reference-images",
            18,
            "--track-pool-size",
            1200,
            "--relocalize-every",
            max(0, args.relocalize_every),
            "--flow-max-error",
            cfg.get("live_flow_max_error", 34.0),
            "--flow-backtrack-error",
            cfg.get("live_flow_backtrack_error", 2.5),
            "--flow-window",
            cfg.get("live_flow_window", 31),
            "--flow-levels",
            cfg.get("live_flow_levels", 4),
            "--flow-iterations",
            cfg.get("live_flow_iterations", 24),
            "--min-track-points",
            15,
            "--min-track-ratio",
            0.10,
            "--proactive-relocalize-points",
            24,
            "--proactive-relocalize-cooldown-frames",
            60,
            "--global-recovery-after-failures",
            1,
            "--global-recovery-max-step",
            1.6,
            "--global-recovery-max-speed",
            1.6,
            "--global-recovery-max-total-step",
            8.0,
            "--output-max-step",
            8.0,
            "--output-max-speed",
            1.6,
            "--blocking-global-recovery",
            "--blocking-global-retry-interval",
            12,
            "--calibrate-output-to-first-global-anchor",
            "--partial-pose-out",
            partial_path,
            "--replay-id",
            run_id,
            "--drone-video",
            video,
            "--expected-count",
            expected,
            "--prime",
            cfg["tsolve_prime"],
            "--degree",
            cfg["tsolve_degree"],
            "--action-weights",
            cfg["tsolve_action_weights"],
            "--fallback-action-weights",
            cfg["tsolve_fallback_action_weights"],
            "--scene-json",
            base_asset_dir / "scene.json",
            "--display-z-sign",
            map_entry.get("display_z_sign", -1),
            "--room-alignment-json",
            json.dumps(map_entry.get("room_alignment") or {}),
        ]
    atlas.add_faiss_live_arguments(command, cfg, faiss_index_dir)
    command.extend(resume_args)
    run_logged(command, log_path)

    raw = json.loads(partial_path.read_text(encoding="utf-8"))
    raw_poses = raw.get("poses") if isinstance(raw.get("poses"), list) else []
    frame_rows = load_frame_rows(query_frames / "frames.csv")
    validation, trusted_by_frame = validate_raw_trajectory(
        raw_poses,
        expected,
        map_entry,
        max_gap_frames=max(0, args.max_gap_frames),
        min_trusted_ratio=min(1.0, max(0.0, args.min_trusted_ratio)),
    )
    atlas.atomic_write_json(run_root / "validation.json", validation)
    if not validation["valid"]:
        raise RuntimeError("Offline path failed validation: " + json.dumps(validation, sort_keys=True))

    display_poses, drift_smoothed = smooth_absolute_anchor_drift(trusted_by_frame)
    complete_poses, interpolated = fill_short_gaps(display_poses, frame_rows)
    out_asset_dir.mkdir(parents=True, exist_ok=True)
    browser_video = prepare_browser_video(video, out_asset_dir)
    final_payload = {
        **raw,
        "mode": "camera_path_offline_validated",
        "description": "Validated fixed-map offline camera trajectory with short bounded gaps interpolated for synchronized replay.",
        "complete": True,
        "expected_count": expected,
        "processed_count": expected,
        "accepted_count": validation["trusted_frames"],
        "interpolated_count": interpolated,
        "anchor_drift_smoothed_count": drift_smoothed,
        "validation": validation,
        "poses": complete_poses,
        "updated_at": time.time(),
    }
    atlas.atomic_write_json(final_pose_path, final_payload)
    stream = {
        "map_id": map_entry["id"],
        "replay_id": run_id,
        "title": args.title,
        "asset_base": atlas.public_rel(out_asset_dir),
        "final_pose_url": atlas.public_rel(final_pose_path),
        "media_url": atlas.public_rel(browser_video),
        "pose_count": expected,
        "accepted_pose_count": validation["trusted_frames"],
        "interpolated_pose_count": interpolated,
        "anchor_drift_smoothed_pose_count": drift_smoothed,
        "held_pose_count": 0,
        "failed_pose_count": 0,
        "expected_count": expected,
        "complete": True,
        "side_project": True,
        "offline_validated": True,
        "source_map_title": map_entry.get("title") or map_entry["id"],
        "validation": validation,
    }
    manifest = {
        "generated_at": time.time(),
        "status": "done",
        "message": f"Offline path ready: {validation['trusted_frames']}/{expected} trusted frames; {interpolated} short-gap interpolations.",
        "stream": stream,
    }
    atlas.atomic_write_json(out_asset_dir / "manifest.json", manifest)
    atlas.atomic_write_json(latest_manifest_path, manifest)
    print(json.dumps({"ok": True, "run": run_id, "validation": validation, "stream": stream}, indent=2))


if __name__ == "__main__":
    main()
