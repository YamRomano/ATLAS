#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from colmap_io import qvec_to_rotmat, read_cameras_model, read_images_model, read_points3d_text
from run_live_tsolve_existing_map_stream import (
    ReferenceSelector,
    copy_case_to_instance,
    farthest_spread_indices,
    image_files,
    import_runtime,
    prepare_image_root,
    read_frame_times,
    sha256_case,
    solve_case,
    write_manifest,
)


def run_timed(cmd: list[object]) -> float:
    cmd = [str(x) for x in cmd]
    print("+", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "minimal")
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        check=False,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.perf_counter() - t0
    print(f"  done in {elapsed:.2f}s", flush=True)
    if proc.returncode != 0:
        print(proc.stdout[-6000:], flush=True)
        proc.check_returncode()
    return elapsed


def load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    return img


def write_stage_header(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_index",
                "case_id",
                "image_name",
                "time_sec",
                "method",
                "accepted",
                "tracked_points",
                "selected_points",
                "feature_extract_ms",
                "match_ms",
                "register_ms",
                "optical_flow_ms",
                "tsolve_ms",
                "total_frame_ms",
                "reason",
            ],
        )
        writer.writeheader()


def append_stage(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_index",
                "case_id",
                "image_name",
                "time_sec",
                "method",
                "accepted",
                "tracked_points",
                "selected_points",
                "feature_extract_ms",
                "match_ms",
                "register_ms",
                "optical_flow_ms",
                "tsolve_ms",
                "total_frame_ms",
                "reason",
            ],
        )
        writer.writerow(row)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def matrix_list(M: Any) -> list[list[float]]:
    if hasattr(M, "tolist"):
        M = M.tolist()
    return [[float(x) for x in row] for row in M]


def vector_list(v: Any) -> list[float]:
    if hasattr(v, "tolist"):
        v = v.tolist()
    return [float(x) for x in v]


def camera_center_from_rt(R: Any, t: Any) -> list[float] | None:
    try:
        Rm = np.asarray(R, dtype=float)
        tv = np.asarray(t, dtype=float).reshape(3)
        if Rm.shape != (3, 3):
            return None
        return vector_list(-Rm.T @ tv)
    except Exception:
        return None


def _dot(a: list[float], b: list[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [float(x) - float(y) for x, y in zip(a, b)]


def _mul(a: list[float], s: float) -> list[float]:
    return [float(x) * float(s) for x in a]


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _normalize(a: list[float]) -> list[float]:
    n = max(float(sum(x * x for x in a)) ** 0.5, 1e-12)
    return [float(x) / n for x in a]


def _mat_vec(C: list[list[float]], v: list[float]) -> list[float]:
    return [_dot(row, v) for row in C]


def _power_eigen(C: list[list[float]], seed: list[float]) -> dict[str, Any]:
    v = _normalize(seed)
    for _ in range(64):
        v = _normalize(_mat_vec(C, v))
    Cv = _mat_vec(C, v)
    return {"v": v, "lambda": _dot(v, Cv)}


def _deflate(C: list[list[float]], eig: dict[str, Any]) -> list[list[float]]:
    v = eig["v"]
    lam = float(eig["lambda"])
    return [[C[r][c] - lam * v[r] * v[c] for c in range(3)] for r in range(3)]


def _covariance(points: list[list[float]], center: list[float]) -> list[list[float]]:
    C = [[0.0, 0.0, 0.0] for _ in range(3)]
    inv = 1.0 / max(1, len(points))
    for p in points:
        d = _sub(p, center)
        for r in range(3):
            for c in range(3):
                C[r][c] += d[r] * d[c] * inv
    return C


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    arr = sorted(float(v) for v in values)
    mid = len(arr) // 2
    if len(arr) % 2:
        return arr[mid]
    return 0.5 * (arr[mid - 1] + arr[mid])


def build_room_transform(scene_json: Path | None, display_z_sign: float) -> Any | None:
    """Match viewer/app.js buildRoomFrame() so bridge targets use room coordinates."""
    if scene_json is None or not scene_json.exists():
        return None
    try:
        scene = json.loads(scene_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    visual_rows = scene.get("dense_points3D") if scene.get("dense_points3D") else scene.get("points3D")
    if not isinstance(visual_rows, list) or not visual_rows:
        return None
    cloud = [list(map(float, row["xyz"])) for row in visual_rows if isinstance(row, dict) and row.get("xyz")]
    cameras = [
        list(map(float, row["center"]))
        for row in scene.get("map_cameras", [])
        if isinstance(row, dict) and row.get("center")
    ]
    if not cloud:
        return None
    stride = max(1, int(np.ceil(len(cloud) / 7000.0)))
    sample = cloud[::stride] + cameras
    center = [sum(p[i] for p in sample) / max(1, len(sample)) for i in range(3)]
    C = _covariance(sample, center)
    e0 = _power_eigen(C, [1.0, 0.2, 0.1])
    e1 = _power_eigen(_deflate(C, e0), [0.1, 1.0, 0.2])
    axis_x = _normalize(e0["v"])
    axis_z = _normalize(_sub(e1["v"], _mul(axis_x, _dot(e1["v"], axis_x))))
    if float(display_z_sign) < 0:
        axis_z = _mul(axis_z, -1.0)
    axis_y = _normalize(_cross(axis_z, axis_x))

    def raw_transform(xyz: list[float]) -> list[float]:
        d = _sub(xyz, center)
        return [_dot(d, axis_x), _dot(d, axis_y), _dot(d, axis_z)]

    point_y = [raw_transform(p)[1] for p in cloud[:5000]]
    cam_y = [raw_transform(p)[1] for p in cameras]
    if cam_y and _median(cam_y) < _median(point_y):
        axis_y = _mul(axis_y, -1.0)

    def transform(xyz: list[float] | None) -> list[float] | None:
        if xyz is None:
            return None
        try:
            p = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
        except (TypeError, ValueError, IndexError):
            return None
        d = _sub(p, center)
        return [_dot(d, axis_x), _dot(d, axis_y), _dot(d, axis_z)]

    def transform_direction(xyz: list[float] | None) -> list[float] | None:
        if xyz is None:
            return None
        try:
            p = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
        except (TypeError, ValueError, IndexError):
            return None
        return [_dot(p, axis_x), _dot(p, axis_y), _dot(p, axis_z)]

    transform.direction = transform_direction  # type: ignore[attr-defined]
    return transform


def room_heading_from_R(R: Any, room_transform: Any | None) -> list[float] | None:
    direction_transform = getattr(room_transform, "direction", None)
    if direction_transform is None or not isinstance(R, list) or len(R) < 3:
        return None
    try:
        # Keep this matched to viewer/app.js rawRotationYaw(): the third row
        # carries the camera optical-axis direction in the COLMAP frame.
        forward = [float(R[2][0]), float(R[2][1]), float(R[2][2])]
    except (TypeError, ValueError, IndexError):
        return None
    room_forward = direction_transform(forward)
    if not room_forward:
        return None
    room_forward[1] = 0.0
    n = float(np.linalg.norm(room_forward))
    if not np.isfinite(n) or n < 1e-9:
        return None
    return [float(room_forward[0] / n), 0.0, float(room_forward[2] / n)]


def colmap_reference_from_meta(meta: dict[str, Any]) -> dict[str, Any] | None:
    qvec = meta.get("colmap_qvec_world_to_camera")
    tvec = meta.get("colmap_tvec_world_to_camera")
    if qvec is None or tvec is None:
        return None
    try:
        R = qvec_to_rotmat(np.asarray(qvec, dtype=float))
        t = np.asarray(tvec, dtype=float).reshape(3)
        center = -R.T @ t
        return {
            "image_name": meta.get("image_name"),
            "R": matrix_list(R),
            "t": vector_list(t),
            "center": vector_list(center),
            "registered_points": int(meta.get("colmap_registered_points") or meta.get("points") or 0),
        }
    except Exception:
        return None


def read_case_meta(case_dir: Path) -> dict[str, Any]:
    try:
        return json.loads((case_dir / "input.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def partial_pose_from_result(
    case: dict[str, Any],
    result: dict[str, Any],
    *,
    room_transform: Any | None = None,
    output_rejection_reason: str | None = None,
) -> dict[str, Any]:
    meta = read_case_meta(Path(case["case_dir"])) if case.get("case_dir") else {}
    R = result.get("R")
    t = result.get("t")
    if isinstance(t, list) and len(t) == 1 and isinstance(t[0], list):
        t = t[0]
    center = camera_center_from_rt(R, t)
    success = bool(result.get("success")) and output_rejection_reason is None
    rcenter = room_transform(center) if success and room_transform is not None else None
    rheading = room_heading_from_R(R, room_transform) if success and room_transform is not None else None
    return {
        "instance_id": case.get("case_id"),
        "success": success,
        "time_sec": meta.get("time_sec", case.get("time_sec")),
        "image_name": meta.get("image_name", case.get("image_name")),
        "R": R,
        "t": t,
        "center": center if success else None,
        "rcenter": rcenter if success else None,
        "rheading": rheading if success else None,
        "raw_center": center if not success else None,
        "held_pose": False,
        "output_rejected": output_rejection_reason is not None,
        "rejected_reason": output_rejection_reason,
        "objective": result.get("objective"),
        "total_ms": result.get("total_ms"),
        "stages_ms": result.get("stages_ms", {}),
        "colmap_reference": colmap_reference_from_meta(meta),
    }


def held_pose_from_last(
    *,
    last_pose: dict[str, Any] | None,
    current_frame: dict[str, Any],
    reason: str,
) -> dict[str, Any] | None:
    """Emit a display-only pose while slow global recovery runs elsewhere."""
    if not last_pose or not last_pose.get("center"):
        return None
    pose = dict(last_pose)
    frame_index = current_frame.get("frame_index")
    pose.update(
        {
            "instance_id": f"hold_{int(frame_index):06d}" if frame_index is not None else "hold",
            "success": False,
            "time_sec": current_frame.get("time_sec"),
            "image_name": current_frame.get("image_name"),
            "held_pose": True,
            "output_rejected": True,
            "hold_reason": reason,
            "total_ms": 0.0,
            "stages_ms": {},
            "colmap_reference": None,
        }
    )
    return pose


def result_center_from_rt(result: dict[str, Any]) -> np.ndarray | None:
    R = result.get("R")
    t = result.get("t")
    if isinstance(t, list) and len(t) == 1 and isinstance(t[0], list):
        t = t[0]
    center = camera_center_from_rt(R, t)
    if center is None:
        return None
    arr = np.asarray(center, dtype=float).reshape(3)
    return arr if np.all(np.isfinite(arr)) else None


def pool_reference_center(pool: dict[str, Any]) -> np.ndarray | None:
    qvec = pool.get("colmap_qvec_world_to_camera")
    tvec = pool.get("colmap_tvec_world_to_camera")
    if qvec is None or tvec is None:
        return None
    try:
        R = qvec_to_rotmat(np.asarray(qvec, dtype=float))
        t = np.asarray(tvec, dtype=float).reshape(3)
        center = -R.T @ t
        return center if np.all(np.isfinite(center)) else None
    except Exception:
        return None


def pose_reference_center_from_case(case: dict[str, Any]) -> np.ndarray | None:
    if not case.get("case_dir"):
        return None
    ref = colmap_reference_from_meta(read_case_meta(Path(case["case_dir"])))
    center = ref.get("center") if ref else None
    if not isinstance(center, list) or len(center) < 3:
        return None
    arr = np.asarray(center[:3], dtype=float)
    return arr if np.all(np.isfinite(arr)) else None


def update_tracking_reference_center(
    *,
    case: dict[str, Any],
    result: dict[str, Any],
    previous_center: np.ndarray | None,
) -> tuple[np.ndarray | None, str]:
    """Keep global relocalization centered on the visual map, not on a bad root.

    COLMAP registration provides an independent camera center for global frames.
    Optical-flow frames do not have that absolute pose, so TSolve is used only
    when it is a locally plausible continuation of the existing track.
    """
    ref_center = pose_reference_center_from_case(case)
    if ref_center is not None:
        return ref_center, "colmap_registration"

    ts_center = result_center_from_rt(result)
    if ts_center is None:
        return previous_center, "missing_tsolve_center"
    if previous_center is None:
        return ts_center, "tsolve_bootstrap"

    try:
        current_time = float(case.get("time_sec"))
    except (TypeError, ValueError):
        current_time = None
    step = float(np.linalg.norm(ts_center - previous_center))
    # This is a reference-update sanity bound, not an output rejection.  It keeps
    # one wrong PnP root from poisoning the next COLMAP reference-image search.
    max_step = 4.0 if current_time is None else 5.0
    if step > max_step:
        return previous_center, f"ignored_tsolve_jump_{step:.3f}m"
    return ts_center, "tsolve_continuation"


def continuity_max_step(previous_time: float | None, current_time: float | None) -> float:
    if previous_time is None or current_time is None:
        return 0.75
    dt = max(0.0, float(current_time) - float(previous_time))
    # Indoor patrol is slow by design.  A meter-scale jump between adjacent
    # streamed frames is almost always a wrong root, not real drone motion.
    return min(1.35, max(0.42, 0.85 * dt + 0.28))


def output_objective_rejection(result: dict[str, Any]) -> str | None:
    try:
        objective = float(result.get("objective"))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(objective):
        return "nonfinite_objective"
    # Good live poses in our indoor runs are normally single-digit residuals.
    # Keep this as a broad sanity gate so only very poor roots are held back.
    if objective > 24.0:
        return f"objective_{objective:.3f}_gt_24.000"
    return None


def output_continuity_rejection(
    *,
    case: dict[str, Any],
    result: dict[str, Any],
    previous_center: np.ndarray | None,
    previous_time: float | None,
) -> tuple[str | None, np.ndarray | None, float | None]:
    if not result.get("success"):
        return None, previous_center, previous_time

    objective_reason = output_objective_rejection(result)
    if objective_reason is not None:
        return objective_reason, previous_center, previous_time

    center = result_center_from_rt(result)
    if center is None:
        return "missing_tsolve_center", previous_center, previous_time

    try:
        current_time = float(case.get("time_sec"))
    except (TypeError, ValueError):
        current_time = None

    if previous_center is None:
        return None, center, current_time

    step = float(np.linalg.norm(center - previous_center))
    max_step = continuity_max_step(previous_time, current_time)
    if step > max_step:
        return f"motion_jump_{step:.3f}m_gt_{max_step:.3f}m", previous_center, previous_time

    return None, center, current_time


def write_partial_pose_stream(
    *,
    path: Path | None,
    replay_id: str,
    drone_video: Path | None,
    expected_count: int,
    poses: list[dict[str, Any]],
    complete: bool,
    current_frame: dict[str, Any] | None = None,
) -> None:
    if path is None:
        return

    def pose_sort_key(pose: dict[str, Any]) -> tuple[float, str, str]:
        try:
            frame_time = float(pose.get("time_sec"))
        except (TypeError, ValueError):
            frame_time = float("inf")
        return (
            frame_time,
            str(pose.get("image_name") or ""),
            str(pose.get("instance_id") or ""),
        )

    accepted = 0
    held = 0
    failed = 0
    for pose in poses:
        if pose.get("held_pose"):
            held += 1
        elif bool(pose.get("success")) and pose.get("center"):
            accepted += 1
        else:
            failed += 1

    payload = {
        "mode": "simulated_live_tsolve_partial",
        "replay_id": replay_id,
        "frame_source": str(drone_video) if drone_video else None,
        "expected_count": int(expected_count),
        "processed_count": len(poses),
        "accepted_count": accepted,
        "held_count": held,
        "failed_count": failed,
        "complete": bool(complete),
        "updated_at": time.time(),
        "poses": sorted(poses, key=pose_sort_key),
    }
    if current_frame:
        payload["current_frame"] = current_frame
        payload["current_frame_time_sec"] = current_frame.get("time_sec")
    atomic_write_json(
        path,
        payload,
    )


def stop_file_requested(path: Path | None) -> bool:
    return path is not None and path.exists()


def expected_count_for_stream(args: argparse.Namespace) -> int:
    if args.expected_count:
        return int(args.expected_count)
    if getattr(args, "follow_dir", False):
        return 0
    return len(image_files(args.query_frames))


def iter_query_frames(args: argparse.Namespace):
    frame_idx = 0
    last_new_frame_time = time.perf_counter()
    while True:
        if args.follow_dir and stop_file_requested(args.stop_file):
            return
        frames = image_files(args.query_frames)
        if frame_idx < len(frames):
            while frame_idx < len(frames):
                if args.follow_dir and stop_file_requested(args.stop_file):
                    return
                last_new_frame_time = time.perf_counter()
                yield frame_idx, frames[frame_idx], len(frames)
                frame_idx += 1
            continue

        if not args.follow_dir:
            return
        if stop_file_requested(args.stop_file):
            return
        if args.follow_idle_timeout > 0 and time.perf_counter() - last_new_frame_time > args.follow_idle_timeout:
            return
        time.sleep(0.2)


class ReplayPacer:
    def __init__(self, *, enabled: bool, scale: float) -> None:
        self.enabled = bool(enabled)
        self.scale = max(0.01, float(scale))
        self.wall_start: float | None = None
        self.video_start: float | None = None

    def wait_until(self, frame_time: float | None) -> float:
        if not self.enabled or frame_time is None:
            return 0.0
        now = time.perf_counter()
        if self.wall_start is None:
            self.wall_start = now
            self.video_start = frame_time
            return 0.0

        video_elapsed = max(0.0, frame_time - float(self.video_start))
        target_wall = self.wall_start + video_elapsed * self.scale
        delay = target_wall - now
        if delay <= 0:
            return 0.0
        time.sleep(delay)
        return delay * 1000.0


def registered_correspondence_pool(
    *,
    localized_model: Path,
    map_points: dict[int, Any],
    image_name: str,
    min_points: int,
) -> dict[str, Any]:
    cameras = read_cameras_model(localized_model)
    images = read_images_model(localized_model)
    image = next((im for im in images.values() if im.name == image_name), None)
    if image is None:
        return {"accepted": False, "reason": "query_not_registered", "image_name": image_name}

    valid = image.point3d_ids >= 0
    valid &= np.array([int(pid) in map_points for pid in image.point3d_ids], dtype=bool)
    valid_idx = np.where(valid)[0]
    if len(valid_idx) < min_points:
        return {
            "accepted": False,
            "reason": "too_few_correspondences",
            "image_name": image_name,
            "valid_2d3d": int(len(valid_idx)),
        }

    pids = image.point3d_ids[valid_idx].astype(np.int64)
    xy = image.xys[valid_idx].astype(np.float32)
    p3d = np.asarray([map_points[int(pid)].xyz for pid in pids], dtype=np.float64)
    K = cameras[image.camera_id].K()
    return {
        "accepted": True,
        "image_name": image.name,
        "xy": xy,
        "p3d": p3d,
        "point3d_ids": pids,
        "K": K,
        "colmap_image_id": image.image_id,
        "colmap_camera_id": image.camera_id,
        "colmap_registered_points": int(np.sum(image.point3d_ids >= 0)),
        "colmap_qvec_world_to_camera": image.qvec.tolist(),
        "colmap_tvec_world_to_camera": image.tvec.tolist(),
    }


def cap_tracking_pool(
    pool: dict[str, Any],
    max_points: int,
    *,
    keep_point3d_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    xy = np.asarray(pool["xy"], dtype=np.float32)
    if max_points <= 0 or len(xy) <= max_points:
        return pool
    pids = np.asarray(pool.get("point3d_ids", np.arange(len(xy))), dtype=np.int64)
    keep: list[int] = []
    if keep_point3d_ids is not None:
        index_by_pid = {int(pid): i for i, pid in enumerate(pids)}
        keep = [index_by_pid[int(pid)] for pid in keep_point3d_ids if int(pid) in index_by_pid]
    if len(keep) >= max_points:
        idx = np.asarray(keep[:max_points], dtype=int)
    else:
        remaining_mask = np.ones(len(xy), dtype=bool)
        if keep:
            remaining_mask[np.asarray(keep, dtype=int)] = False
        remaining_idx = np.where(remaining_mask)[0]
        extra_count = max_points - len(keep)
        if len(remaining_idx) > extra_count:
            extra_local = farthest_spread_indices(xy[remaining_idx], extra_count)
            extra = remaining_idx[extra_local]
        else:
            extra = remaining_idx
        idx = np.concatenate([np.asarray(keep, dtype=int), np.asarray(extra, dtype=int)])
    out = dict(pool)
    out["xy"] = xy[idx]
    out["p3d"] = np.asarray(pool["p3d"], dtype=np.float64)[idx]
    out["point3d_ids"] = pids[idx]
    return out


def stable_case_indices(
    pool: dict[str, Any],
    *,
    max_points: int,
    preferred_point3d_ids: np.ndarray | None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    xy = np.asarray(pool["xy"], dtype=np.float64)
    pids = np.asarray(pool.get("point3d_ids", np.arange(len(xy))), dtype=np.int64)
    if preferred_point3d_ids is not None:
        index_by_pid = {int(pid): i for i, pid in enumerate(pids)}
        chosen: list[int] = []
        missing: list[int] = []
        for pid in preferred_point3d_ids:
            idx = index_by_pid.get(int(pid))
            if idx is None:
                missing.append(int(pid))
            else:
                chosen.append(idx)
        if missing:
            return None, {
                "accepted": False,
                "reason": "stable_solve_points_lost",
                "missing_stable_points": len(missing),
            }
        return np.asarray(chosen[:max_points], dtype=int), {
            "accepted": True,
            "stable_solve_set": True,
            "missing_stable_points": 0,
        }

    if len(xy) > max_points:
        chosen = farthest_spread_indices(xy, max_points)
    else:
        chosen = np.arange(len(xy), dtype=int)
    return chosen, {
        "accepted": True,
        "stable_solve_set": False,
        "missing_stable_points": 0,
    }


def track_pool(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    pool: dict[str, Any],
    *,
    max_error: float,
    backtrack_error: float,
    win_size: int,
    max_level: int,
    iterations: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    t0 = time.perf_counter()
    p0 = np.asarray(pool["xy"], dtype=np.float32).reshape(-1, 1, 2)
    if len(p0) == 0:
        return None, {"optical_flow_ms": 0.0, "tracked_points": 0, "reason": "empty_track_pool"}

    p1, st, err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        p0,
        None,
        winSize=(win_size, win_size),
        maxLevel=max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, 0.01),
    )
    if p1 is None or st is None:
        return None, {
            "optical_flow_ms": (time.perf_counter() - t0) * 1000.0,
            "tracked_points": 0,
            "reason": "lk_failed",
        }

    good = st.reshape(-1).astype(bool)
    if err is not None:
        good &= err.reshape(-1) <= max_error

    p0r, st_back, _ = cv2.calcOpticalFlowPyrLK(
        curr_gray,
        prev_gray,
        p1,
        None,
        winSize=(win_size, win_size),
        maxLevel=max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, 0.01),
    )
    if p0r is not None and st_back is not None:
        back = np.linalg.norm(p0.reshape(-1, 2) - p0r.reshape(-1, 2), axis=1)
        good &= st_back.reshape(-1).astype(bool)
        good &= back <= backtrack_error

    h, w = curr_gray.shape[:2]
    xy1 = p1.reshape(-1, 2)
    good &= xy1[:, 0] >= 0
    good &= xy1[:, 0] < w
    good &= xy1[:, 1] >= 0
    good &= xy1[:, 1] < h

    idx = np.where(good)[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if len(idx) == 0:
        return None, {"optical_flow_ms": elapsed_ms, "tracked_points": 0, "reason": "no_good_tracks"}

    out = dict(pool)
    out["xy"] = xy1[idx].astype(np.float32)
    out["p3d"] = np.asarray(pool["p3d"], dtype=np.float64)[idx]
    out["point3d_ids"] = np.asarray(pool.get("point3d_ids", np.arange(len(p0))), dtype=np.int64)[idx]
    out.pop("colmap_qvec_world_to_camera", None)
    out.pop("colmap_tvec_world_to_camera", None)
    return out, {"optical_flow_ms": elapsed_ms, "tracked_points": int(len(idx)), "reason": ""}


def write_case_from_pool(
    *,
    pool: dict[str, Any],
    frame_name: str,
    frame_times: dict[str, dict[str, str]],
    case_id: str,
    inputs_out: Path,
    max_points: int,
    method: str,
    tracking_parent: str | None = None,
    preferred_point3d_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    xy = np.asarray(pool["xy"], dtype=np.float64)
    p3d_all = np.asarray(pool["p3d"], dtype=np.float64)
    chosen, selection_meta = stable_case_indices(
        pool,
        max_points=max_points,
        preferred_point3d_ids=preferred_point3d_ids,
    )
    if chosen is None:
        return {
            "accepted": False,
            "case_id": case_id,
            "reason": selection_meta.get("reason", "case_selection_failed"),
            "tracked_pool_points": int(len(xy)),
            "missing_stable_points": int(selection_meta.get("missing_stable_points", 0)),
            "image_name": frame_name,
        }
    p2d = xy[chosen]
    p3d = p3d_all[chosen]
    pids = np.asarray(pool.get("point3d_ids", np.arange(len(xy))), dtype=np.int64)
    selected_point3d_ids = pids[chosen]
    K = np.asarray(pool["K"], dtype=np.float64)
    raw_name = frame_name.split("/", 1)[-1]
    frame_row = frame_times.get(raw_name, {})

    case_dir = inputs_out / "inputs" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(case_dir / "p3d.csv", p3d, delimiter=",", fmt="%.12g")
    np.savetxt(case_dir / "p2d.csv", p2d, delimiter=",", fmt="%.12g")
    meta = {
        "K": K.tolist(),
        "image_name": frame_name,
        "source_frame": frame_row.get("source_frame"),
        "time_sec": float(frame_row["time_sec"]) if frame_row.get("time_sec") else None,
        "points": int(len(chosen)),
        "tracked_pool_points": int(len(xy)),
        "selected_point3d_ids": selected_point3d_ids.astype(int).tolist(),
        "stable_solve_set": bool(selection_meta.get("stable_solve_set", False)),
        "missing_stable_points": int(selection_meta.get("missing_stable_points", 0)),
        "input_sha256": sha256_case(K, p3d, p2d),
        "localization_method": method,
        "tracking_parent": tracking_parent,
        "colmap_image_id": pool.get("colmap_image_id"),
        "colmap_camera_id": pool.get("colmap_camera_id"),
        "colmap_registered_points": pool.get("colmap_registered_points"),
        "colmap_qvec_world_to_camera": pool.get("colmap_qvec_world_to_camera"),
        "colmap_tvec_world_to_camera": pool.get("colmap_tvec_world_to_camera"),
    }
    (case_dir / "input.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {
        "accepted": True,
        "case_id": case_id,
        "case_dir": case_dir,
        "points": int(len(chosen)),
        "tracked_pool_points": int(len(xy)),
        "selected_point3d_ids": selected_point3d_ids.astype(int).tolist(),
        "stable_solve_set": bool(selection_meta.get("stable_solve_set", False)),
        "image_name": frame_name,
        "time_sec": meta["time_sec"],
        "method": method,
    }


class GlobalRelocalizer:
    def __init__(
        self,
        *,
        colmap: Path,
        map_database: Path,
        map_images: Path,
        map_sparse_model: Path,
        map_sparse_text: Path,
        all_images: Path,
        query_root: Path,
        work_dir: Path,
        max_image_size: int,
        query_camera_model: str,
        max_reference_images: int,
        tracking_reference_images: int,
        min_points: int,
    ):
        self.colmap = colmap
        self.map_database = map_database
        self.map_images = map_images
        self.map_sparse_model = map_sparse_model
        self.map_sparse_text = map_sparse_text
        self.all_images = all_images
        self.query_root = query_root
        self.work_dir = work_dir
        self.max_image_size = max_image_size
        self.query_camera_model = query_camera_model
        self.references = ReferenceSelector(
            map_sparse_text,
            bootstrap_count=max_reference_images,
            tracking_count=tracking_reference_images,
        )
        self.min_points = min_points

    def localize(
        self,
        *,
        frame: Path,
        frame_idx: int,
        query_name: str,
        map_points: dict[int, Any],
        last_center: np.ndarray | None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        stage = {
            "feature_extract_ms": 0.0,
            "match_ms": 0.0,
            "register_ms": 0.0,
            "reason": "",
        }
        shutil.copy2(frame, self.query_root / frame.name)
        image_list = self.work_dir / f"global_{frame_idx:06d}_image.txt"
        pair_list = self.work_dir / f"global_{frame_idx:06d}_pairs.txt"
        localized = self.work_dir / f"global_{frame_idx:06d}_localized"
        db = self.work_dir / f"global_{frame_idx:06d}.db"
        if db.exists():
            db.unlink()
        shutil.copy2(self.map_database, db)
        image_list.write_text(query_name + "\n", encoding="utf-8")

        stage["feature_extract_ms"] = 1000.0 * run_timed(
            [
                self.colmap,
                "feature_extractor",
                "--database_path",
                db,
                "--image_path",
                self.all_images,
                "--image_list_path",
                image_list,
                "--ImageReader.camera_model",
                self.query_camera_model,
                "--ImageReader.single_camera_per_folder",
                "1",
                "--SiftExtraction.max_image_size",
                self.max_image_size,
                "--SiftExtraction.use_gpu",
                "0",
            ]
        )

        attempts = [("tracking_global", self.references.near(last_center))]
        bootstrap = self.references.bootstrap()
        if attempts[0][1] != bootstrap:
            attempts.append(("bootstrap_global", bootstrap))

        for attempt_name, reference_names in attempts:
            print(
                f"global relocalization {query_name}: {len(reference_names)} {attempt_name} reference images",
                flush=True,
            )
            pair_list.write_text(
                "".join(f"{query_name} {ref_name}\n" for ref_name in reference_names),
                encoding="utf-8",
            )
            stage["match_ms"] += 1000.0 * run_timed(
                [
                    self.colmap,
                    "matches_importer",
                    "--database_path",
                    db,
                    "--match_list_path",
                    pair_list,
                    "--match_type",
                    "pairs",
                    "--SiftMatching.guided_matching",
                    "1",
                    "--SiftMatching.use_gpu",
                    "0",
                ]
            )
            if localized.exists():
                shutil.rmtree(localized)
            localized.mkdir(parents=True, exist_ok=True)
            stage["register_ms"] += 1000.0 * run_timed(
                [
                    self.colmap,
                    "image_registrator",
                    "--database_path",
                    db,
                    "--input_path",
                    self.map_sparse_model,
                    "--output_path",
                    localized,
                    "--Mapper.abs_pose_min_num_inliers",
                    "15",
                    "--Mapper.abs_pose_min_inlier_ratio",
                    "0.10",
                ]
            )
            pool = registered_correspondence_pool(
                localized_model=localized,
                map_points=map_points,
                image_name=query_name,
                min_points=self.min_points,
            )
            if pool.get("accepted"):
                pool["localization_method"] = attempt_name
                return pool, stage
            stage["reason"] = str(pool.get("reason") or "global_relocalization_failed")
            print("global relocalization rejected:", json.dumps(pool), flush=True)
        return None, stage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--colmap", required=True, type=Path)
    ap.add_argument("--map-database", required=True, type=Path)
    ap.add_argument("--map-images", required=True, type=Path)
    ap.add_argument("--map-sparse-model", required=True, type=Path)
    ap.add_argument("--map-sparse-text", required=True, type=Path)
    ap.add_argument("--query-frames", required=True, type=Path)
    ap.add_argument("--runtime-dir", required=True, type=Path)
    ap.add_argument("--solver-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--inputs-out-dir", required=True, type=Path)
    ap.add_argument("--work-dir", required=True, type=Path)
    ap.add_argument("--max-image-size", type=int, default=1200)
    ap.add_argument("--query-camera-model", default="SIMPLE_RADIAL")
    ap.add_argument("--min-points", type=int, default=40)
    ap.add_argument("--max-points", type=int, default=40)
    ap.add_argument("--max-reference-images", type=int, default=48)
    ap.add_argument("--tracking-reference-images", type=int, default=10)
    ap.add_argument("--track-pool-size", type=int, default=900)
    ap.add_argument("--relocalize-every", type=int, default=0)
    ap.add_argument("--flow-max-error", type=float, default=34.0)
    ap.add_argument("--flow-backtrack-error", type=float, default=2.5)
    ap.add_argument("--flow-window", type=int, default=21)
    ap.add_argument("--flow-levels", type=int, default=3)
    ap.add_argument("--flow-iterations", type=int, default=18)
    ap.add_argument("--min-track-points", type=int, default=0, help="Minimum tracked 2D/3D correspondences before local TSolve. 0 chooses a safe automatic value.")
    ap.add_argument("--min-track-ratio", type=float, default=0.10, help="Minimum fraction of the previous track pool that must survive LK tracking.")
    ap.add_argument("--global-recovery-after-failures", type=int, default=2, help="Run COLMAP recovery only after this many consecutive local failures.")
    ap.add_argument("--blocking-global-recovery", action="store_true", help="Debug mode: block the frame loop while COLMAP recovery runs.")
    ap.add_argument("--disable-background-recovery", action="store_true", help="Live mode: do not start asynchronous COLMAP recovery after local tracking drops.")
    ap.add_argument("--strict-stable-solve-set", action="store_true", help="Force the original 40 solve correspondences to survive; slower and mostly for debugging.")
    ap.add_argument("--prime", type=int, default=2147483647)
    ap.add_argument("--degree", type=int, default=11)
    ap.add_argument("--action-weights", default="branch")
    ap.add_argument("--fallback-action-weights", default="")
    ap.add_argument("--partial-pose-out", type=Path, default=None)
    ap.add_argument("--replay-id", default="live")
    ap.add_argument("--drone-video", type=Path, default=None)
    ap.add_argument("--expected-count", type=int, default=0)
    ap.add_argument("--scene-json", type=Path, default=None, help="ATLAS map scene.json used to export room-frame rcenter values.")
    ap.add_argument("--display-z-sign", type=float, default=-1.0, help="Map display z sign used by the ATLAS viewer room transform.")
    ap.add_argument("--follow-dir", action="store_true", help="Keep waiting for new query frames until --stop-file exists.")
    ap.add_argument("--stop-file", type=Path, default=None)
    ap.add_argument("--follow-idle-timeout", type=float, default=0.0, help="0 means wait indefinitely while following.")
    ap.add_argument("--pace-replay", action="store_true", help="For finite uploaded-video replay, process frames on their video timeline instead of as fast as possible.")
    ap.add_argument("--pace-scale", type=float, default=1.0, help="Timeline scale for --pace-replay. 1.0 means real video time; 0.5 means 2x faster.")
    args = ap.parse_args()
    if args.min_track_points <= 0:
        args.min_track_points = max(int(args.min_points), int(args.max_points) * 2)
    args.min_track_ratio = max(0.0, min(1.0, float(args.min_track_ratio)))
    args.flow_window = max(7, int(args.flow_window) | 1)
    args.flow_levels = max(0, int(args.flow_levels))
    args.flow_iterations = max(5, int(args.flow_iterations))
    args.global_recovery_after_failures = max(1, int(args.global_recovery_after_failures))

    for name in [
        "colmap",
        "map_database",
        "map_images",
        "map_sparse_model",
        "map_sparse_text",
        "query_frames",
        "runtime_dir",
        "solver_dir",
        "out_dir",
        "inputs_out_dir",
        "work_dir",
    ]:
        setattr(args, name, getattr(args, name).resolve())
    if args.partial_pose_out is not None:
        args.partial_pose_out = args.partial_pose_out.resolve()
    if args.drone_video is not None:
        args.drone_video = args.drone_video.resolve()
    if args.scene_json is not None:
        args.scene_json = args.scene_json.resolve()
    if args.stop_file is not None:
        args.stop_file = args.stop_file.resolve()

    for required in [
        args.colmap,
        args.map_database,
        args.map_images,
        args.map_sparse_model / "images.bin",
        args.map_sparse_text / "points3D.txt",
    ]:
        if not Path(required).exists():
            raise FileNotFoundError(required)

    frames = image_files(args.query_frames)
    if not frames and not args.follow_dir:
        raise RuntimeError(f"No query frames in {args.query_frames}")
    if not frames and args.follow_dir:
        print(f"Following {args.query_frames}; waiting for first DJI frame.", flush=True)

    for path in (args.out_dir, args.inputs_out_dir, args.work_dir):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    stage_csv = args.out_dir / "live_stage_times.csv"
    write_stage_header(stage_csv)
    frame_times = read_frame_times(args.query_frames / "frames.csv")
    room_transform = build_room_transform(args.scene_json, args.display_z_sign)
    if args.scene_json is not None:
        status = "enabled" if room_transform is not None else "unavailable"
        print(f"ATLAS room-frame rcenter export: {status} from {args.scene_json}", flush=True)
    map_points = read_points3d_text(args.map_sparse_text / "points3D.txt")
    all_images = args.work_dir / "all_images"
    query_root = prepare_image_root(args.map_images, all_images)

    relocalizer = GlobalRelocalizer(
        colmap=args.colmap,
        map_database=args.map_database,
        map_images=args.map_images,
        map_sparse_model=args.map_sparse_model,
        map_sparse_text=args.map_sparse_text,
        all_images=all_images,
        query_root=query_root,
        work_dir=args.work_dir,
        max_image_size=args.max_image_size,
        query_camera_model=args.query_camera_model,
        max_reference_images=args.max_reference_images,
        tracking_reference_images=args.tracking_reference_images,
        min_points=args.min_points,
    )

    runtime_api = import_runtime(args.runtime_dir, args.solver_dir.resolve())
    branch_dir = args.out_dir / "offline_branch"
    instances_dir = args.out_dir / "instances_all"
    branch_dir.mkdir(parents=True, exist_ok=True)
    instances_dir.mkdir(parents=True, exist_ok=True)

    print("=== Bounded TSolve runtime setup ===", flush=True)
    root_refiner = runtime_api["ensure_c_root_refiner"](
        yam_code_dir=runtime_api["yam_code"],
        out_dir=branch_dir,
        require_lapack=True,
    )
    direct_coeff_builder = runtime_api["ensure_direct_coeff_builder"](
        yam_code_dir=runtime_api["yam_code"],
        out_dir=branch_dir,
    )
    branches: list[Any] = []
    double_sos: dict[int, Path] = {}
    manifest_rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    output_rejected: list[dict[str, Any]] = []
    partial_poses: list[dict[str, Any]] = []
    last_output_pose: dict[str, Any] | None = None
    current_pool: dict[str, Any] | None = None
    prev_gray: np.ndarray | None = None
    prev_case_id: str | None = None
    last_center: np.ndarray | None = None
    last_output_center: np.ndarray | None = None
    last_output_time: float | None = None
    stable_solve_point3d_ids: np.ndarray | None = None
    consecutive_local_failures = 0
    global_relocalization_count = 0
    local_tracking_count = 0
    background_recovery_count = 0
    background_recovery_success_count = 0
    background_recovery_stale_count = 0
    pending_global: dict[str, Any] | None = None
    pacer = ReplayPacer(enabled=bool(args.pace_replay and not args.follow_dir), scale=args.pace_scale)

    def schedule_background_global_recovery(
        *,
        frame_idx: int,
        frame: Path,
        query_name: str,
        curr_gray: np.ndarray,
        reason: str,
    ) -> bool:
        nonlocal pending_global, background_recovery_count, background_recovery_stale_count
        if args.disable_background_recovery:
            return False
        if args.blocking_global_recovery:
            return False
        if pending_global is not None:
            pending_idx = int(pending_global.get("frame_idx", frame_idx))
            pending_age_frames = int(frame_idx) - pending_idx
            if args.follow_dir:
                stale_after_frames = max(2, int(args.global_recovery_after_failures) * 2)
            else:
                stale_after_frames = max(8, int(args.global_recovery_after_failures) * 4)
            if not pending_global.get("done") and pending_age_frames < stale_after_frames:
                return False
            if not pending_global.get("done"):
                pending_global["ignored"] = True
                pending_global["ignore_reason"] = (
                    f"stale_recovery_frame_{pending_idx}_behind_current_{int(frame_idx)}"
                )
                background_recovery_stale_count += 1
                print(
                    "BACKGROUND RECOVERY STALE:",
                    json.dumps(
                        {
                            "old_frame_index": pending_idx,
                            "new_frame_index": int(frame_idx),
                            "age_frames": pending_age_frames,
                        }
                    ),
                    flush=True,
                )
            pending_global = None
        center = None if last_center is None else np.asarray(last_center, dtype=float).copy()
        recovery: dict[str, Any] = {
            "done": False,
            "pool": None,
            "stage": {
                "feature_extract_ms": 0.0,
                "match_ms": 0.0,
                "register_ms": 0.0,
                "reason": "",
            },
            "error": None,
            "frame_idx": int(frame_idx),
            "frame": frame,
            "query_name": query_name,
            "gray": curr_gray.copy(),
            "started_at": time.perf_counter(),
            "reason": reason,
        }

        def worker() -> None:
            try:
                pool, stage = relocalizer.localize(
                    frame=frame,
                    frame_idx=frame_idx,
                    query_name=query_name,
                    map_points=map_points,
                    last_center=center,
                )
                recovery["pool"] = pool
                recovery["stage"] = stage
            except Exception as exc:  # pragma: no cover - defensive live path
                recovery["error"] = repr(exc)
                recovery["stage"]["reason"] = repr(exc)
            finally:
                recovery["done"] = True

        thread = threading.Thread(target=worker, name=f"atlas-global-recovery-{frame_idx:06d}", daemon=True)
        recovery["thread"] = thread
        pending_global = recovery
        background_recovery_count += 1
        thread.start()
        return True

    def append_global_recovery_pose(
        *,
        recovery: dict[str, Any],
        pool: dict[str, Any],
        stage: dict[str, Any],
        elapsed_ms: float,
    ) -> bool:
        nonlocal current_pool, prev_gray, prev_case_id
        nonlocal last_center, last_output_center, last_output_time, last_output_pose
        nonlocal stable_solve_point3d_ids, consecutive_local_failures

        frame_idx = int(recovery["frame_idx"])
        frame = Path(recovery["frame"])
        query_name = str(recovery["query_name"])
        curr_gray = np.asarray(recovery["gray"])
        case_id = f"instance_{len(manifest_rows):03d}"
        method = "global_colmap_background_recovery"
        frame_time_raw = frame_times.get(frame.name, {}).get("time_sec")
        try:
            frame_time = float(frame_time_raw) if frame_time_raw is not None and frame_time_raw != "" else None
        except (TypeError, ValueError):
            frame_time = None
        current_frame_meta = {
            "frame_index": frame_idx,
            "image_name": query_name,
            "time_sec": frame_time,
        }

        stable_solve_reset = False
        pool["K"] = np.asarray(pool["K"], dtype=np.float64)
        case = write_case_from_pool(
            pool=pool,
            frame_name=query_name,
            frame_times=frame_times,
            case_id=case_id,
            inputs_out=args.inputs_out_dir,
            max_points=args.max_points,
            method=method,
            tracking_parent=prev_case_id,
            preferred_point3d_ids=stable_solve_point3d_ids,
        )
        if (
            not case.get("accepted")
            and str(case.get("reason") or "") == "stable_solve_points_lost"
            and stable_solve_point3d_ids is not None
        ):
            print(
                "background recovery lost the preferred solve set; selecting a fresh local TSolve set",
                flush=True,
            )
            case = write_case_from_pool(
                pool=pool,
                frame_name=query_name,
                frame_times=frame_times,
                case_id=case_id,
                inputs_out=args.inputs_out_dir,
                max_points=args.max_points,
                method=method,
                tracking_parent=prev_case_id,
                preferred_point3d_ids=None,
            )
            stable_solve_reset = bool(case.get("accepted"))

        if not case.get("accepted"):
            consecutive_local_failures += 1
            reason = str(case.get("reason") or "background_recovery_case_selection_failed")
            rejected.append(
                {
                    "accepted": False,
                    "reason": reason,
                    "image_name": query_name,
                    "frame_index": frame_idx,
                }
            )
            append_stage(
                stage_csv,
                {
                    "frame_index": frame_idx,
                    "case_id": "",
                    "image_name": query_name,
                    "time_sec": frame_time_raw,
                    "method": method,
                    "accepted": False,
                    "tracked_points": int(case.get("tracked_pool_points", 0)),
                    "selected_points": 0,
                    "feature_extract_ms": stage.get("feature_extract_ms", 0.0),
                    "match_ms": stage.get("match_ms", 0.0),
                    "register_ms": stage.get("register_ms", 0.0),
                    "optical_flow_ms": 0.0,
                    "tsolve_ms": 0.0,
                    "total_frame_ms": elapsed_ms,
                    "reason": reason,
                },
            )
            print("BACKGROUND RECOVERY CASE SKIPPED:", json.dumps({"frame_index": frame_idx, "reason": reason}), flush=True)
            return False

        if stable_solve_point3d_ids is None or stable_solve_reset:
            stable_solve_point3d_ids = np.asarray(case["selected_point3d_ids"], dtype=np.int64)
            print(
                f"locked stable TSolve solve set: {len(stable_solve_point3d_ids)} 3D points",
                flush=True,
            )

        manifest_rows.append(
            {
                "experiment": "bounded_live_video_tsolve_stream",
                "case_id": case_id,
                "p3d_csv": f"inputs/{case_id}/p3d.csv",
                "p2d_csv": f"inputs/{case_id}/p2d.csv",
                "input_json": f"inputs/{case_id}/input.json",
                "points": int(case["points"]),
                "image_name": case["image_name"],
                "time_sec": case["time_sec"],
                "localization_attempt": method,
            }
        )
        write_manifest(args.inputs_out_dir, manifest_rows)
        instance_dir = instances_dir / case_id
        copy_case_to_instance(Path(case["case_dir"]), instance_dir)

        if not branches:
            print("Learning static-C branch from first accepted simulated-live frame.", flush=True)
            learn_t0 = time.perf_counter()
            branch, branch_json, double_so = runtime_api["learn_one_static_branch"](
                solver_dir=args.solver_dir.resolve(),
                yam_code_dir=runtime_api["yam_code"],
                instance_dir=instance_dir,
                branch_dir=branch_dir,
                seed=20260707,
                prime=args.prime,
                degree=args.degree,
                action_weights=args.action_weights,
            )
            print(f"learned branch={branch.index} in {time.perf_counter() - learn_t0:.3f}s", flush=True)
            print("branch json:", branch_json, flush=True)
            branches.append(branch)
            double_sos[branch.index] = double_so

        solve_t0 = time.perf_counter()
        result = solve_case(
            runtime_api=runtime_api,
            solver_dir=args.solver_dir.resolve(),
            out_dir=args.out_dir,
            branch_dir=branch_dir,
            branches=branches,
            double_sos=double_sos,
            root_refiner=root_refiner,
            direct_coeff_builder=direct_coeff_builder,
            instance_dir=instance_dir,
            instance_id=case_id,
            prime=args.prime,
            degree=args.degree,
            action_weights=args.action_weights,
            fallback_action_weights=args.fallback_action_weights,
            fork_seed=20260707 + frame_idx,
        )
        solve_ms = (time.perf_counter() - solve_t0) * 1000.0
        stages = result.get("stages_ms") or {}
        if result.get("success"):
            current_pool = cap_tracking_pool(
                pool,
                args.track_pool_size,
                keep_point3d_ids=stable_solve_point3d_ids,
            )
            prev_gray = curr_gray
            prev_case_id = case_id
            last_center, reference_update_reason = update_tracking_reference_center(
                case=case,
                result=result,
                previous_center=last_center,
            )
        else:
            consecutive_local_failures += 1
            reference_update_reason = "tsolve_failed"
            rejected.append(
                {
                    "accepted": False,
                    "reason": "tsolve_failed",
                    "image_name": query_name,
                    "frame_index": frame_idx,
                    "case_id": case_id,
                    "method": method,
                }
            )

        output_rejection_reason = None
        if result.get("success"):
            output_rejection_reason, last_output_center, last_output_time = output_continuity_rejection(
                case=case,
                result=result,
                previous_center=last_output_center,
                previous_time=last_output_time,
            )
            if output_rejection_reason is not None:
                consecutive_local_failures += 1
                output_rejected.append(
                    {
                        "accepted": False,
                        "reason": output_rejection_reason,
                        "image_name": query_name,
                        "frame_index": frame_idx,
                        "case_id": case_id,
                        "method": method,
                    }
                )
            else:
                consecutive_local_failures = 0

        if output_rejection_reason is not None and last_output_pose is not None:
            pose_payload = held_pose_from_last(
                last_pose=last_output_pose,
                current_frame=current_frame_meta,
                reason=output_rejection_reason,
            )
        else:
            pose_payload = partial_pose_from_result(
                case,
                result,
                room_transform=room_transform,
                output_rejection_reason=output_rejection_reason,
            )
        if pose_payload is None:
            pose_payload = partial_pose_from_result(
                case,
                result,
                room_transform=room_transform,
                output_rejection_reason=output_rejection_reason or "no_previous_pose_to_hold",
            )
        partial_poses.append(pose_payload)
        if pose_payload.get("success") and pose_payload.get("center"):
            last_output_pose = pose_payload
        write_partial_pose_stream(
            path=args.partial_pose_out,
            replay_id=args.replay_id,
            drone_video=args.drone_video,
            expected_count=expected_count_for_stream(args),
            poses=partial_poses,
            complete=False,
            current_frame=current_frame_meta,
        )

        append_stage(
            stage_csv,
            {
                "frame_index": frame_idx,
                "case_id": case_id,
                "image_name": query_name,
                "time_sec": case["time_sec"],
                "method": method,
                "accepted": bool(result.get("success")) and output_rejection_reason is None,
                "tracked_points": int(case["tracked_pool_points"]),
                "selected_points": int(case["points"]),
                "feature_extract_ms": stage.get("feature_extract_ms", 0.0),
                "match_ms": stage.get("match_ms", 0.0),
                "register_ms": stage.get("register_ms", 0.0),
                "optical_flow_ms": 0.0,
                "tsolve_ms": solve_ms,
                "total_frame_ms": elapsed_ms + solve_ms,
                "reason": output_rejection_reason or ("" if result.get("success") else "tsolve_failed"),
            },
        )
        print(
            "BACKGROUND POSE APPENDED:",
            json.dumps(
                {
                    "case_id": case_id,
                    "frame_index": frame_idx,
                    "success": bool(result.get("success")),
                    "output_accepted": bool(result.get("success")) and output_rejection_reason is None,
                    "output_reason": output_rejection_reason or "",
                    "tracked_points": int(case["tracked_pool_points"]),
                    "stable_solve_reset": stable_solve_reset,
                    "solve_ms": round(solve_ms, 2),
                    "total_ms": result.get("total_ms"),
                    "action_ms": stages.get("ysolve_static_action_double_ms"),
                    "root_ms": stages.get("ysolve_static_root_total_ms"),
                    "branch": result.get("branch_index"),
                    "new_branch": result.get("branch_new"),
                    "reference_center": reference_update_reason,
                }
            ),
            flush=True,
        )
        return bool(result.get("success")) and output_rejection_reason is None

    def apply_background_global_recovery() -> None:
        nonlocal pending_global, current_pool, prev_gray, prev_case_id
        nonlocal last_center, consecutive_local_failures, global_relocalization_count
        nonlocal background_recovery_success_count
        if pending_global is None or not pending_global.get("done"):
            return

        recovery = pending_global
        pending_global = None
        if recovery.get("ignored"):
            print(
                "BACKGROUND RECOVERY IGNORED:",
                json.dumps(
                    {
                        "frame_index": int(recovery.get("frame_idx", -1)),
                        "reason": str(recovery.get("ignore_reason") or "stale_recovery"),
                    }
                ),
                flush=True,
            )
            return
        stage = dict(recovery.get("stage") or {})
        pool = recovery.get("pool")
        elapsed_ms = (time.perf_counter() - float(recovery.get("started_at", time.perf_counter()))) * 1000.0
        accepted = pool is not None
        global_relocalization_count += 1
        if accepted:
            background_recovery_success_count += 1
            stage["reason"] = ""
            print(
                "BACKGROUND RECOVERY READY:",
                json.dumps(
                    {
                        "frame_index": recovery["frame_idx"],
                        "tracked_points": int(len(np.asarray(pool.get("xy", [])))),
                        "elapsed_ms": round(elapsed_ms, 1),
                    }
                ),
                flush=True,
            )
            append_global_recovery_pose(
                recovery=recovery,
                pool=pool,
                stage=stage,
                elapsed_ms=elapsed_ms,
            )
            return
        else:
            reason = str(stage.get("reason") or recovery.get("error") or "background_global_recovery_failed")
            stage["reason"] = reason
            print(
                "BACKGROUND RECOVERY FAILED:",
                json.dumps({"frame_index": recovery["frame_idx"], "reason": reason}),
                flush=True,
            )

        append_stage(
            stage_csv,
            {
                "frame_index": recovery["frame_idx"],
                "case_id": "",
                "image_name": recovery["query_name"],
                "time_sec": frame_times.get(recovery["frame"].name, {}).get("time_sec"),
                "method": "global_colmap_background_recovery",
                "accepted": accepted,
                "tracked_points": int(len(np.asarray(pool.get("xy", [])))) if accepted else 0,
                "selected_points": 0,
                "feature_extract_ms": stage.get("feature_extract_ms", 0.0),
                "match_ms": stage.get("match_ms", 0.0),
                "register_ms": stage.get("register_ms", 0.0),
                "optical_flow_ms": 0.0,
                "tsolve_ms": 0.0,
                "total_frame_ms": elapsed_ms,
                "reason": stage.get("reason", ""),
            },
        )

    write_partial_pose_stream(
        path=args.partial_pose_out,
        replay_id=args.replay_id,
        drone_video=args.drone_video,
        expected_count=expected_count_for_stream(args),
        poses=partial_poses,
        complete=False,
    )

    processed_frames = 0
    for frame_idx, frame, visible_frame_count in iter_query_frames(args):
        processed_frames = max(processed_frames, frame_idx + 1)
        frame_t0 = time.perf_counter()
        query_name = f"query/{frame.name}"
        case_id = f"instance_{len(manifest_rows):03d}"
        expected_label = str(args.expected_count) if args.expected_count else ("live" if args.follow_dir else str(visible_frame_count))
        print(f"\n=== BOUNDED STREAM FRAME {frame_idx + 1}/{expected_label}: {query_name} ===", flush=True)
        if args.follow_dir:
            frame_times = read_frame_times(args.query_frames / "frames.csv")
        raw_frame_row = frame_times.get(frame.name, {})
        try:
            current_frame_time = float(raw_frame_row["time_sec"]) if raw_frame_row.get("time_sec") else None
        except (TypeError, ValueError):
            current_frame_time = None
        current_frame_meta = {
            "frame_index": frame_idx,
            "image_name": query_name,
            "time_sec": current_frame_time,
        }
        write_partial_pose_stream(
            path=args.partial_pose_out,
            replay_id=args.replay_id,
            drone_video=args.drone_video,
            expected_count=expected_count_for_stream(args),
            poses=partial_poses,
            complete=False,
            current_frame=current_frame_meta,
        )
        pace_wait_ms = pacer.wait_until(current_frame_time)
        if pace_wait_ms > 0:
            apply_background_global_recovery()
        curr_gray = load_gray(frame)
        method = "optical_flow"
        stage = {
            "feature_extract_ms": 0.0,
            "match_ms": 0.0,
            "register_ms": 0.0,
            "optical_flow_ms": 0.0,
            "pace_wait_ms": pace_wait_ms,
            "reason": "",
        }
        stable_solve_reset = False
        apply_background_global_recovery()

        must_global = current_pool is None or prev_gray is None
        global_reason = "bootstrap" if must_global else ""
        if args.relocalize_every > 0 and frame_idx > 0 and frame_idx % args.relocalize_every == 0:
            if pending_global is not None and not args.blocking_global_recovery:
                method = "periodic_global_recovery_pending"
                stage["reason"] = "periodic_relocalization_background_recovery_pending"
            else:
                must_global = True
                global_reason = "periodic_relocalization"

        pool: dict[str, Any] | None = None
        if not must_global and current_pool is not None and prev_gray is not None:
            previous_pool_count = max(1, int(len(np.asarray(current_pool.get("xy", [])))))
            tracked, flow_stage = track_pool(
                prev_gray,
                curr_gray,
                current_pool,
                max_error=args.flow_max_error,
                backtrack_error=args.flow_backtrack_error,
                win_size=args.flow_window,
                max_level=args.flow_levels,
                iterations=args.flow_iterations,
            )
            stage.update(flow_stage)
            tracked_count = int(flow_stage.get("tracked_points") or 0)
            tracked_ratio = float(tracked_count) / float(previous_pool_count)
            if (
                tracked is not None
                and tracked_count >= args.min_track_points
                and tracked_ratio >= args.min_track_ratio
            ):
                pool = tracked
                method = "optical_flow"
                local_tracking_count += 1
            else:
                consecutive_local_failures += 1
                stage["reason"] = str(flow_stage.get("reason") or "too_few_tracked_points")
                if tracked_count < args.min_track_points:
                    stage["reason"] = f"{stage['reason']}_tracked_{tracked_count}_lt_{args.min_track_points}"
                elif tracked_ratio < args.min_track_ratio:
                    stage["reason"] = f"{stage['reason']}_ratio_{tracked_ratio:.3f}_lt_{args.min_track_ratio:.3f}"
                if consecutive_local_failures >= args.global_recovery_after_failures:
                    global_reason = f"recovery_after_{consecutive_local_failures}_local_failures"
                    if args.disable_background_recovery and args.follow_dir:
                        method = "optical_flow_recovery_disabled"
                        stage["reason"] = f"{stage['reason']}_live_recovery_disabled_holding_last_pose"
                    elif schedule_background_global_recovery(
                        frame_idx=frame_idx,
                        frame=frame,
                        query_name=query_name,
                        curr_gray=curr_gray,
                        reason=global_reason,
                    ):
                        method = "optical_flow_background_recovery_scheduled"
                        stage["reason"] = f"{stage['reason']}_background_recovery_scheduled"
                    elif pending_global is not None and not args.blocking_global_recovery:
                        method = "optical_flow_background_recovery_pending"
                        stage["reason"] = f"{stage['reason']}_background_recovery_pending"
                    else:
                        must_global = True
                else:
                    method = "optical_flow_wait_recovery"

        if must_global:
            method = f"global_colmap_{global_reason or 'recovery'}"
            pool, global_stage = relocalizer.localize(
                frame=frame,
                frame_idx=frame_idx,
                query_name=query_name,
                map_points=map_points,
                last_center=last_center,
            )
            stage.update(global_stage)
            global_relocalization_count += 1
            if pool is not None:
                consecutive_local_failures = 0

        if pool is None:
            rejected_row = {
                "accepted": False,
                "reason": stage.get("reason") or "localization_failed",
                "image_name": query_name,
                "frame_index": frame_idx,
            }
            rejected.append(rejected_row)
            append_stage(
                stage_csv,
                {
                    "frame_index": frame_idx,
                    "case_id": "",
                    "image_name": query_name,
                    "time_sec": frame_times.get(frame.name, {}).get("time_sec"),
                    "method": method,
                    "accepted": False,
                    "tracked_points": 0,
                    "selected_points": 0,
                    "feature_extract_ms": stage.get("feature_extract_ms", 0.0),
                    "match_ms": stage.get("match_ms", 0.0),
                    "register_ms": stage.get("register_ms", 0.0),
                    "optical_flow_ms": stage.get("optical_flow_ms", 0.0),
                    "tsolve_ms": 0.0,
                    "total_frame_ms": (time.perf_counter() - frame_t0) * 1000.0,
                    "reason": rejected_row["reason"],
                },
            )
            print("FRAME SKIPPED:", json.dumps(rejected_row), flush=True)
            held = held_pose_from_last(
                last_pose=last_output_pose,
                current_frame=current_frame_meta,
                reason=str(rejected_row["reason"]),
            )
            if held is not None:
                partial_poses.append(held)
            write_partial_pose_stream(
                path=args.partial_pose_out,
                replay_id=args.replay_id,
                drone_video=args.drone_video,
                expected_count=expected_count_for_stream(args),
                poses=partial_poses,
                complete=False,
                current_frame=current_frame_meta,
            )
            continue

        pool["K"] = np.asarray(pool["K"], dtype=np.float64)
        case = write_case_from_pool(
            pool=pool,
            frame_name=query_name,
            frame_times=frame_times,
            case_id=case_id,
            inputs_out=args.inputs_out_dir,
            max_points=args.max_points,
            method=method,
            tracking_parent=prev_case_id,
            preferred_point3d_ids=stable_solve_point3d_ids,
        )
        if not case.get("accepted") and method == "optical_flow" and args.strict_stable_solve_set:
            print(
                "stable solve set was lost during optical flow; doing bounded global relocalization",
                flush=True,
            )
            pool, global_stage = relocalizer.localize(
                frame=frame,
                frame_idx=frame_idx,
                query_name=query_name,
                map_points=map_points,
                last_center=last_center,
            )
            stage["feature_extract_ms"] += float(global_stage.get("feature_extract_ms", 0.0))
            stage["match_ms"] += float(global_stage.get("match_ms", 0.0))
            stage["register_ms"] += float(global_stage.get("register_ms", 0.0))
            method = "global_colmap"
            if pool is not None:
                pool["K"] = np.asarray(pool["K"], dtype=np.float64)
                case = write_case_from_pool(
                    pool=pool,
                    frame_name=query_name,
                    frame_times=frame_times,
                    case_id=case_id,
                    inputs_out=args.inputs_out_dir,
                    max_points=args.max_points,
                    method=method,
                    tracking_parent=prev_case_id,
                    preferred_point3d_ids=stable_solve_point3d_ids,
                )
                if not case.get("accepted") and stable_solve_point3d_ids is not None:
                    print(
                        "global relocalization did not contain the locked solve set; resetting solve set for this track segment",
                        flush=True,
                    )
                    case = write_case_from_pool(
                        pool=pool,
                        frame_name=query_name,
                        frame_times=frame_times,
                        case_id=case_id,
                        inputs_out=args.inputs_out_dir,
                        max_points=args.max_points,
                        method=method,
                        tracking_parent=prev_case_id,
                        preferred_point3d_ids=None,
                    )
                    stable_solve_reset = bool(case.get("accepted"))
        if (
            not case.get("accepted")
            and str(case.get("reason") or "") == "stable_solve_points_lost"
            and pool is not None
            and stable_solve_point3d_ids is not None
        ):
            print(
                "current frame lost the preferred solve set; selecting a fresh local TSolve set without global COLMAP",
                flush=True,
            )
            case = write_case_from_pool(
                pool=pool,
                frame_name=query_name,
                frame_times=frame_times,
                case_id=case_id,
                inputs_out=args.inputs_out_dir,
                max_points=args.max_points,
                method=method,
                tracking_parent=prev_case_id,
                preferred_point3d_ids=None,
            )
            stable_solve_reset = bool(case.get("accepted"))
        if not case.get("accepted"):
            consecutive_local_failures += 1
            rejected_row = {
                "accepted": False,
                "reason": str(case.get("reason") or "case_selection_failed"),
                "image_name": query_name,
                "frame_index": frame_idx,
            }
            rejected.append(rejected_row)
            append_stage(
                stage_csv,
                {
                    "frame_index": frame_idx,
                    "case_id": "",
                    "image_name": query_name,
                    "time_sec": frame_times.get(frame.name, {}).get("time_sec"),
                    "method": method,
                    "accepted": False,
                    "tracked_points": int(case.get("tracked_pool_points", 0)),
                    "selected_points": 0,
                    "feature_extract_ms": stage.get("feature_extract_ms", 0.0),
                    "match_ms": stage.get("match_ms", 0.0),
                    "register_ms": stage.get("register_ms", 0.0),
                    "optical_flow_ms": stage.get("optical_flow_ms", 0.0),
                    "tsolve_ms": 0.0,
                    "total_frame_ms": (time.perf_counter() - frame_t0) * 1000.0,
                    "reason": rejected_row["reason"],
                },
            )
            print("FRAME SKIPPED:", json.dumps(rejected_row), flush=True)
            held = held_pose_from_last(
                last_pose=last_output_pose,
                current_frame=current_frame_meta,
                reason=str(rejected_row["reason"]),
            )
            if held is not None:
                partial_poses.append(held)
            write_partial_pose_stream(
                path=args.partial_pose_out,
                replay_id=args.replay_id,
                drone_video=args.drone_video,
                expected_count=expected_count_for_stream(args),
                poses=partial_poses,
                complete=False,
                current_frame=current_frame_meta,
            )
            continue
        if stable_solve_point3d_ids is None or stable_solve_reset:
            stable_solve_point3d_ids = np.asarray(case["selected_point3d_ids"], dtype=np.int64)
            print(
                f"locked stable TSolve solve set: {len(stable_solve_point3d_ids)} 3D points",
                flush=True,
            )
        manifest_rows.append(
            {
                "experiment": "bounded_live_video_tsolve_stream",
                "case_id": case_id,
                "p3d_csv": f"inputs/{case_id}/p3d.csv",
                "p2d_csv": f"inputs/{case_id}/p2d.csv",
                "input_json": f"inputs/{case_id}/input.json",
                "points": int(case["points"]),
                "image_name": case["image_name"],
                "time_sec": case["time_sec"],
                "localization_attempt": method,
            }
        )
        write_manifest(args.inputs_out_dir, manifest_rows)
        instance_dir = instances_dir / case_id
        copy_case_to_instance(Path(case["case_dir"]), instance_dir)

        if not branches:
            print("Learning static-C branch from first accepted simulated-live frame.", flush=True)
            learn_t0 = time.perf_counter()
            branch, branch_json, double_so = runtime_api["learn_one_static_branch"](
                solver_dir=args.solver_dir.resolve(),
                yam_code_dir=runtime_api["yam_code"],
                instance_dir=instance_dir,
                branch_dir=branch_dir,
                seed=20260707,
                prime=args.prime,
                degree=args.degree,
                action_weights=args.action_weights,
            )
            print(f"learned branch={branch.index} in {time.perf_counter() - learn_t0:.3f}s", flush=True)
            print("branch json:", branch_json, flush=True)
            branches.append(branch)
            double_sos[branch.index] = double_so

        solve_t0 = time.perf_counter()
        result = solve_case(
            runtime_api=runtime_api,
            solver_dir=args.solver_dir.resolve(),
            out_dir=args.out_dir,
            branch_dir=branch_dir,
            branches=branches,
            double_sos=double_sos,
            root_refiner=root_refiner,
            direct_coeff_builder=direct_coeff_builder,
            instance_dir=instance_dir,
            instance_id=case_id,
            prime=args.prime,
            degree=args.degree,
            action_weights=args.action_weights,
            fallback_action_weights=args.fallback_action_weights,
            fork_seed=20260707 + frame_idx,
        )
        solve_ms = (time.perf_counter() - solve_t0) * 1000.0
        stages = result.get("stages_ms") or {}
        if not result.get("success"):
            consecutive_local_failures += 1
            print("TSolve failed frame; keeping it out of the tracking chain.", flush=True)
            rejected.append(
                {
                    "accepted": False,
                    "reason": "tsolve_failed",
                    "image_name": query_name,
                    "frame_index": frame_idx,
                    "case_id": case_id,
                    "method": method,
                }
            )
        else:
            current_pool = cap_tracking_pool(
                pool,
                args.track_pool_size,
                keep_point3d_ids=stable_solve_point3d_ids,
            )
            prev_gray = curr_gray
            prev_case_id = case_id

        reference_update_reason = "not_updated"
        if result.get("success"):
            last_center, reference_update_reason = update_tracking_reference_center(
                case=case,
                result=result,
                previous_center=last_center,
            )
        output_rejection_reason = None
        if result.get("success"):
            output_rejection_reason, last_output_center, last_output_time = output_continuity_rejection(
                case=case,
                result=result,
                previous_center=last_output_center,
                previous_time=last_output_time,
            )
            if output_rejection_reason is not None:
                consecutive_local_failures += 1
                output_rejected.append(
                    {
                        "accepted": False,
                        "reason": output_rejection_reason,
                        "image_name": query_name,
                        "frame_index": frame_idx,
                        "case_id": case_id,
                        "method": method,
                    }
                )
            else:
                consecutive_local_failures = 0
        if output_rejection_reason is not None and last_output_pose is not None:
            pose_payload = held_pose_from_last(
                last_pose=last_output_pose,
                current_frame=current_frame_meta,
                reason=output_rejection_reason,
            )
        else:
            pose_payload = partial_pose_from_result(
                case,
                result,
                room_transform=room_transform,
                output_rejection_reason=output_rejection_reason,
            )
        if pose_payload is None:
            pose_payload = partial_pose_from_result(
                case,
                result,
                room_transform=room_transform,
                output_rejection_reason=output_rejection_reason or "no_previous_pose_to_hold",
            )
        partial_poses.append(pose_payload)
        if pose_payload.get("success") and pose_payload.get("center"):
            last_output_pose = pose_payload
        write_partial_pose_stream(
            path=args.partial_pose_out,
            replay_id=args.replay_id,
            drone_video=args.drone_video,
            expected_count=expected_count_for_stream(args),
            poses=partial_poses,
            complete=False,
            current_frame=current_frame_meta,
        )

        append_stage(
            stage_csv,
            {
                "frame_index": frame_idx,
                "case_id": case_id,
                "image_name": query_name,
                "time_sec": case["time_sec"],
                "method": method,
                "accepted": bool(result.get("success")) and output_rejection_reason is None,
                "tracked_points": int(case["tracked_pool_points"]),
                "selected_points": int(case["points"]),
                "feature_extract_ms": stage.get("feature_extract_ms", 0.0),
                "match_ms": stage.get("match_ms", 0.0),
                "register_ms": stage.get("register_ms", 0.0),
                "optical_flow_ms": stage.get("optical_flow_ms", 0.0),
                "tsolve_ms": solve_ms,
                "total_frame_ms": (time.perf_counter() - frame_t0) * 1000.0,
                "reason": output_rejection_reason or ("" if result.get("success") else "tsolve_failed"),
            },
        )
        print(
            "POSE APPENDED:",
            json.dumps(
                {
                    "case_id": case_id,
                    "method": method,
                    "success": bool(result.get("success")),
                    "output_accepted": bool(result.get("success")) and output_rejection_reason is None,
                    "output_reason": output_rejection_reason or "",
                    "tracked_points": int(case["tracked_pool_points"]),
                    "stable_solve_set": bool(case.get("stable_solve_set")),
                    "stable_solve_reset": stable_solve_reset,
                    "solve_ms": round(solve_ms, 2),
                    "total_ms": result.get("total_ms"),
                    "action_ms": stages.get("ysolve_static_action_double_ms"),
                    "root_ms": stages.get("ysolve_static_root_total_ms"),
                    "branch": result.get("branch_index"),
                    "new_branch": result.get("branch_new"),
                    "reference_center": reference_update_reason,
                }
            ),
            flush=True,
        )

    if pending_global is not None:
        final_recovery_wait = 4.0 if args.follow_dir else 15.0
        deadline = time.perf_counter() + final_recovery_wait
        while pending_global is not None and not pending_global.get("done") and time.perf_counter() < deadline:
            time.sleep(0.05)
    apply_background_global_recovery()
    summary = {
        "mode": "bounded_live_video_tsolve_stream",
        "query_frames": len(image_files(args.query_frames)),
        "processed_frames": processed_frames,
        "follow_dir": bool(args.follow_dir),
        "accepted_cases": len(manifest_rows),
        "rejected_cases": len(rejected),
        "output_rejected_cases": len(output_rejected),
        "online_branch_count_final": len(branches),
        "inputs_out_dir": str(args.inputs_out_dir),
        "out_dir": str(args.out_dir),
        "stage_times_csv": str(stage_csv),
        "tracking_pool_size": args.track_pool_size,
        "relocalize_every": args.relocalize_every,
        "min_track_points": args.min_track_points,
        "min_track_ratio": args.min_track_ratio,
        "flow_window": args.flow_window,
        "flow_levels": args.flow_levels,
        "flow_iterations": args.flow_iterations,
        "global_recovery_after_failures": args.global_recovery_after_failures,
        "blocking_global_recovery": bool(args.blocking_global_recovery),
        "disable_background_recovery": bool(args.disable_background_recovery),
        "pace_replay": bool(args.pace_replay and not args.follow_dir),
        "pace_scale": float(args.pace_scale),
        "global_relocalization_count": global_relocalization_count,
        "background_recovery_count": background_recovery_count,
        "background_recovery_success_count": background_recovery_success_count,
        "background_recovery_stale_count": background_recovery_stale_count,
        "background_recovery_pending_at_finish": bool(pending_global is not None),
        "local_tracking_count": local_tracking_count,
        "strict_stable_solve_set": bool(args.strict_stable_solve_set),
        "stable_solve_point_count": 0 if stable_solve_point3d_ids is None else int(len(stable_solve_point3d_ids)),
        "rejected": rejected[:80],
        "output_rejected": output_rejected[:80],
    }
    (args.out_dir / "live_stream_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (args.inputs_out_dir / "export_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_partial_pose_stream(
        path=args.partial_pose_out,
        replay_id=args.replay_id,
        drone_video=args.drone_video,
        expected_count=expected_count_for_stream(args),
        poses=partial_poses,
        complete=True,
    )
    print("\n=== BOUNDED LIVE STREAM SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
