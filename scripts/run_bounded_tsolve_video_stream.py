#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
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


def explicit_room_transform(room_alignment: Any) -> Any | None:
    """Build the fixed map transform used by the viewer and patrol controller."""
    matrix = room_alignment.get("matrix") if isinstance(room_alignment, dict) else None
    if not isinstance(matrix, list) or len(matrix) != 3:
        return None
    try:
        rows = [[float(value) for value in row] for row in matrix]
    except (TypeError, ValueError):
        return None
    if any(len(row) != 4 or not all(math.isfinite(value) for value in row) for row in rows):
        return None

    def transform(xyz: list[float] | None) -> list[float] | None:
        if xyz is None:
            return None
        try:
            point = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
        except (TypeError, ValueError, IndexError):
            return None
        return [sum(row[index] * point[index] for index in range(3)) + row[3] for row in rows]

    def transform_direction(xyz: list[float] | None) -> list[float] | None:
        if xyz is None:
            return None
        try:
            direction = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
        except (TypeError, ValueError, IndexError):
            return None
        return [sum(row[index] * direction[index] for index in range(3)) for row in rows]

    transform.direction = transform_direction  # type: ignore[attr-defined]
    return transform


def parse_room_alignment_json(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("--room-alignment-json must contain valid JSON") from exc
    if isinstance(payload, dict) and isinstance(payload.get("room_alignment"), dict):
        payload = payload["room_alignment"]
    return payload if isinstance(payload, dict) else None


def build_room_transform(
    scene_json: Path | None,
    display_z_sign: float,
    room_alignment: Any = None,
) -> Any | None:
    """Match viewer/app.js buildRoomFrame() so bridge targets use room coordinates."""
    explicit = explicit_room_transform(room_alignment)
    if explicit is not None:
        return explicit
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


def normalize_room_heading(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        x = float(value[0])
        z = float(value[2])
    except (TypeError, ValueError):
        return None
    norm = math.hypot(x, z)
    if not math.isfinite(norm) or norm < 1e-9:
        return None
    return [x / norm, 0.0, z / norm]


def rotate_room_heading(heading: list[float] | None, yaw_radians: float) -> list[float] | None:
    base = normalize_room_heading(heading)
    if base is None or not math.isfinite(yaw_radians):
        return None
    c = math.cos(yaw_radians)
    s = math.sin(yaw_radians)
    return normalize_room_heading([c * base[0] - s * base[2], 0.0, s * base[0] + c * base[2]])


def room_heading_separation_degrees(first: Any, second: Any) -> float | None:
    """Return the unsigned planar angle between two room-frame headings."""
    a = normalize_room_heading(first)
    b = normalize_room_heading(second)
    if a is None or b is None:
        return None
    cross = a[0] * b[2] - a[2] * b[0]
    dot = a[0] * b[0] + a[2] * b[2]
    return abs(math.degrees(math.atan2(cross, dot)))


def optical_flow_yaw_delta(
    previous: np.ndarray | None,
    current: np.ndarray,
    focal_px: float,
) -> tuple[float | None, int]:
    """Estimate small camera yaw between adjacent images.

    This is intentionally a rotation-only hint.  It never changes the map
    position and is emitted only for a fresh, in-place patrol turn after the
    normal TSolve position has been rejected.
    """
    if previous is None or not math.isfinite(focal_px) or focal_px < 100.0:
        return None, 0
    points = cv2.goodFeaturesToTrack(previous, maxCorners=500, qualityLevel=0.01, minDistance=12)
    if points is None or len(points) < 20:
        return None, 0
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        previous,
        current,
        points,
        None,
        winSize=(31, 31),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 24, 0.01),
    )
    if tracked is None or status is None:
        return None, 0
    good = status.reshape(-1).astype(bool)
    if int(np.sum(good)) < 20:
        return None, int(np.sum(good))
    dx = tracked.reshape(-1, 2)[good, 0] - points.reshape(-1, 2)[good, 0]
    median = float(np.median(dx))
    mad = float(np.median(np.abs(dx - median)))
    stable = np.abs(dx - median) <= max(2.5, 3.0 * mad)
    count = int(np.sum(stable))
    if count < 16:
        return None, count
    delta = -math.atan2(float(np.median(dx[stable])), focal_px)
    # A larger adjacent-frame rotation is not a reliable optical-flow update.
    if not math.isfinite(delta) or abs(delta) > math.radians(6.0):
        return None, count
    return delta, count


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
        "received_unix": meta.get("received_unix", case.get("received_unix")),
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
    rotation_heading: list[float] | None = None,
    rotation_heading_tracks: int = 0,
    rotation_heading_delta_deg: float | None = None,
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
            "received_unix": current_frame.get("received_unix"),
            "image_name": current_frame.get("image_name"),
            "held_pose": True,
            "output_rejected": True,
            "hold_reason": reason,
            "total_ms": 0.0,
            "stages_ms": {},
            "colmap_reference": None,
        }
    )
    normalized_rotation = normalize_room_heading(rotation_heading)
    if normalized_rotation is not None and int(rotation_heading_tracks) >= 16:
        pose.update(
            {
                "rotation_heading": normalized_rotation,
                "rotation_heading_source": "optical_flow_yaw",
                "rotation_heading_tracks": int(rotation_heading_tracks),
                "rotation_heading_delta_deg": rotation_heading_delta_deg,
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


def global_recovery_continuity_rejection(
    *,
    pool: dict[str, Any],
    last_center: np.ndarray | None,
    max_step: float = 0.85,
) -> str | None:
    """Reject a COLMAP registration that contradicts the last trusted place.

    A live recovery starts only after the bridge has stopped translation (or
    while it is yawing in place), so its source camera cannot legitimately
    reappear several map units away.  Repeated indoor structure can otherwise
    produce hundreds of internally consistent but globally aliased matches.
    This check runs on COLMAP's own source-frame pose before its 2D/3D pool is
    propagated to the current frame and before TSolve can consume it.
    """
    if last_center is None:
        return None
    recovered_center = pool_reference_center(pool)
    if recovered_center is None:
        return "global_recovery_missing_colmap_center"
    previous = np.asarray(last_center, dtype=float).reshape(3)
    if not np.all(np.isfinite(previous)):
        return "global_recovery_invalid_trusted_center"
    step = float(np.linalg.norm(recovered_center - previous))
    limit = max(0.10, float(max_step))
    if step > limit:
        return f"global_recovery_alias_{step:.3f}m_gt_{limit:.3f}m"
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
        return 0.30
    dt = max(0.0, float(current_time) - float(previous_time))
    # Ten-FPS catch-up can intentionally skip queued images. Allow physically
    # continuous travel across that short capture gap, but retain a hard cap so
    # elapsed time can never make a meter-scale wrong root acceptable.
    return max(0.30, min(0.55, 0.18 + 0.85 * dt))


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
    frame_idx = max(0, int(args.start_frame_index))
    last_new_frame_time = time.perf_counter()
    while True:
        if args.follow_dir and stop_file_requested(args.stop_file):
            return
        frames = image_files(args.query_frames)
        if frame_idx < len(frames):
            if args.follow_dir and len(frames) - frame_idx > 3:
                skipped = len(frames) - frame_idx - 1
                frame_idx = len(frames) - 1
                print(
                    f"LIVE CATCH-UP: dropped {skipped} stale queued frames; "
                    f"localizing newest frame {frame_idx}.",
                    flush=True,
                )
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


def correspondence_spread_metrics(
    xy: np.ndarray,
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Validate that a small recovery pool constrains more than one image patch."""
    points = np.asarray(xy, dtype=float).reshape(-1, 2)
    if (
        len(points) < 4
        or int(width) <= 0
        or int(height) <= 0
        or not np.all(np.isfinite(points))
    ):
        return {
            "ok": False,
            "occupied_grid_cells": 0,
            "span_x_fraction": 0.0,
            "span_y_fraction": 0.0,
        }
    normalized = points / np.array([float(width), float(height)], dtype=float)
    span = np.ptp(normalized, axis=0)
    grid = np.clip(np.floor(normalized * 3.0).astype(int), 0, 2)
    occupied = len({(int(cell[0]), int(cell[1])) for cell in grid})
    span_x = float(span[0])
    span_y = float(span[1])
    return {
        "ok": bool(occupied >= 4 and span_x >= 0.20 and span_y >= 0.15),
        "occupied_grid_cells": int(occupied),
        "span_x_fraction": span_x,
        "span_y_fraction": span_y,
    }


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
    camera = cameras[image.camera_id]
    spread = correspondence_spread_metrics(
        xy,
        width=camera.width,
        height=camera.height,
    )
    if len(valid_idx) < 40 and not spread["ok"]:
        return {
            "accepted": False,
            "reason": "low_correspondence_spatial_concentration",
            "image_name": image_name,
            "valid_2d3d": int(len(valid_idx)),
            "correspondence_spread": spread,
        }
    p3d = np.asarray([map_points[int(pid)].xyz for pid in pids], dtype=np.float64)
    K = camera.K()
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
        "valid_2d3d": int(len(valid_idx)),
        "correspondence_spread": spread,
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
        "received_unix": float(frame_row["received_unix"]) if frame_row.get("received_unix") else None,
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
        recovery_max_step: float,
        matching_threads: int,
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
        self.recovery_max_step = max(0.10, float(recovery_max_step))
        self.matching_threads = max(1, int(matching_threads))

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
                    "--SiftMatching.num_threads",
                    self.matching_threads,
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
                continuity_reason = global_recovery_continuity_rejection(
                    pool=pool,
                    last_center=last_center,
                    max_step=self.recovery_max_step,
                )
                if continuity_reason is not None:
                    stage["reason"] = continuity_reason
                    print(
                        "global relocalization rejected:",
                        json.dumps(
                            {
                                "accepted": False,
                                "reason": continuity_reason,
                                "image_name": query_name,
                                "attempt": attempt_name,
                                "registered_points": pool.get("valid_2d3d"),
                                "recovered_center": (
                                    pool_reference_center(pool).tolist()
                                    if pool_reference_center(pool) is not None
                                    else None
                                ),
                                "last_trusted_center": (
                                    np.asarray(last_center, dtype=float).tolist()
                                    if last_center is not None
                                    else None
                                ),
                            }
                        ),
                        flush=True,
                    )
                    continue
                pool["localization_method"] = attempt_name
                stage["reason"] = ""
                # The database is a disposable copy of the fixed map DB. It
                # must never accumulate across live recovery attempts.
                db.unlink(missing_ok=True)
                image_list.unlink(missing_ok=True)
                pair_list.unlink(missing_ok=True)
                shutil.rmtree(localized, ignore_errors=True)
                return pool, stage
            stage["reason"] = str(pool.get("reason") or "global_relocalization_failed")
            print("global relocalization rejected:", json.dumps(pool), flush=True)
        db.unlink(missing_ok=True)
        image_list.unlink(missing_ok=True)
        pair_list.unlink(missing_ok=True)
        shutil.rmtree(localized, ignore_errors=True)
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
    ap.add_argument(
        "--sift-matching-threads",
        type=int,
        default=1,
        help="CPU matcher threads used by COLMAP recovery. One avoids a known macOS matcher race.",
    )
    ap.add_argument("--track-pool-size", type=int, default=900)
    ap.add_argument("--relocalize-every", type=int, default=0)
    ap.add_argument("--flow-max-error", type=float, default=34.0)
    ap.add_argument("--flow-backtrack-error", type=float, default=2.5)
    ap.add_argument("--flow-window", type=int, default=21)
    ap.add_argument("--flow-levels", type=int, default=3)
    ap.add_argument("--flow-iterations", type=int, default=18)
    ap.add_argument("--min-track-points", type=int, default=0, help="Minimum tracked 2D/3D correspondences before local TSolve. 0 chooses a safe automatic value.")
    ap.add_argument("--min-track-ratio", type=float, default=0.10, help="Minimum fraction of the previous track pool that must survive LK tracking.")
    ap.add_argument("--proactive-relocalize-points", type=int, default=28, help="Refresh map correspondences before optical flow reaches the hard minimum. 0 disables proactive refresh.")
    ap.add_argument("--proactive-relocalize-cooldown-frames", type=int, default=60, help="Minimum processed frames between proactive correspondence refresh attempts.")
    ap.add_argument("--global-recovery-after-failures", type=int, default=2, help="Run COLMAP recovery only after this many consecutive local failures.")
    ap.add_argument(
        "--global-recovery-max-step",
        type=float,
        default=0.85,
        help=(
            "Maximum map-space displacement accepted from a COLMAP recovery. "
            "Keep 0.85 for live hover recovery; finite low-FPS video may use a "
            "larger physically justified value while still rejecting room aliases."
        ),
    )
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
    ap.add_argument(
        "--start-frame-index",
        type=int,
        default=0,
        help="Begin the finite frame stream at this zero-based index.",
    )
    ap.add_argument(
        "--resume-pose-stream",
        type=Path,
        default=None,
        help="Existing partial pose JSON whose accepted history and last trusted pose seed this run.",
    )
    ap.add_argument(
        "--resume-case-dir",
        type=Path,
        default=None,
        help="Existing TSolve case containing p2d.csv, p3d.csv, and input.json for the frame before --start-frame-index.",
    )
    ap.add_argument("--scene-json", type=Path, default=None, help="ATLAS map scene.json used to export room-frame rcenter values.")
    ap.add_argument("--display-z-sign", type=float, default=-1.0, help="Map display z sign used by the ATLAS viewer room transform.")
    ap.add_argument(
        "--room-alignment-json",
        default=None,
        help=(
            "Saved map room_alignment JSON. When present, rcenter/rheading use this fixed "
            "viewer/patrol frame instead of recomputing a PCA frame from the point cloud."
        ),
    )
    ap.add_argument("--follow-dir", action="store_true", help="Keep waiting for new query frames until --stop-file exists.")
    ap.add_argument("--stop-file", type=Path, default=None)
    ap.add_argument("--follow-idle-timeout", type=float, default=0.0, help="0 means wait indefinitely while following.")
    ap.add_argument("--pace-replay", action="store_true", help="For finite uploaded-video replay, process frames on their video timeline instead of as fast as possible.")
    ap.add_argument("--pace-scale", type=float, default=1.0, help="Timeline scale for --pace-replay. 1.0 means real video time; 0.5 means 2x faster.")
    args = ap.parse_args()
    if args.min_track_points <= 0:
        args.min_track_points = max(int(args.min_points), int(args.max_points) * 2)
    args.proactive_relocalize_points = max(0, int(args.proactive_relocalize_points))
    if 0 < args.proactive_relocalize_points <= args.min_track_points:
        args.proactive_relocalize_points = args.min_track_points + 1
    args.proactive_relocalize_cooldown_frames = max(1, int(args.proactive_relocalize_cooldown_frames))
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
    room_alignment = parse_room_alignment_json(args.room_alignment_json)
    room_transform = build_room_transform(args.scene_json, args.display_z_sign, room_alignment)
    if args.scene_json is not None:
        status = "enabled" if room_transform is not None else "unavailable"
        source = "saved room_alignment" if explicit_room_transform(room_alignment) is not None else str(args.scene_json)
        print(f"ATLAS room-frame rcenter export: {status} from {source}", flush=True)
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
        recovery_max_step=args.global_recovery_max_step,
        matching_threads=args.sift_matching_threads,
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
    # This independent image-to-image chain is deliberately separate from the
    # accepted map-correspondence chain.  A rejected global TSolve root must
    # never update map position, but a small optical yaw can safely describe an
    # in-place turn while the bridge keeps translation locked.
    rotation_prev_gray: np.ndarray | None = None
    rotation_heading: list[float] | None = None
    rotation_heading_tracks = 0
    rotation_heading_delta_deg: float | None = None
    rotation_focal_px = 851.6865528775178
    rotation_last_center: list[float] | None = None
    rotation_last_received_unix: float | None = None
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
    proactive_relocalization_count = 0
    proactive_relocalization_success_count = 0
    proactive_relocalization_fallback_count = 0
    last_proactive_relocalize_frame = -args.proactive_relocalize_cooldown_frames
    pending_global: dict[str, Any] | None = None
    pacer = ReplayPacer(enabled=bool(args.pace_replay and not args.follow_dir), scale=args.pace_scale)

    if (args.resume_pose_stream is None) != (args.resume_case_dir is None):
        raise ValueError("--resume-pose-stream and --resume-case-dir must be provided together")
    if args.resume_pose_stream is not None and args.resume_case_dir is not None:
        resume_doc = json.loads(args.resume_pose_stream.read_text(encoding="utf-8"))
        resume_poses = list(resume_doc.get("poses") or [])
        trusted_pose = next(
            (
                pose
                for pose in reversed(resume_poses)
                if pose.get("success")
                and pose.get("center")
                and not pose.get("held_pose")
                and not pose.get("output_rejected")
            ),
            None,
        )
        if trusted_pose is None:
            raise RuntimeError("Resume pose stream has no trusted accepted pose")
        resume_meta = json.loads((args.resume_case_dir / "input.json").read_text(encoding="utf-8"))
        resume_xy = np.atleast_2d(np.loadtxt(args.resume_case_dir / "p2d.csv", delimiter=","))
        resume_xyz = np.atleast_2d(np.loadtxt(args.resume_case_dir / "p3d.csv", delimiter=","))
        resume_ids = np.asarray(resume_meta.get("selected_point3d_ids") or [], dtype=np.int64)
        if len(resume_xy) != len(resume_xyz) or len(resume_xy) != len(resume_ids):
            raise RuntimeError("Resume case correspondence arrays do not have matching lengths")
        resume_image_name = str(resume_meta.get("image_name") or "").split("/", 1)[-1]
        resume_image = args.query_frames / resume_image_name
        if not resume_image.exists():
            raise FileNotFoundError(resume_image)
        current_pool = {
            "accepted": True,
            "xy": resume_xy.astype(np.float32),
            "p3d": resume_xyz.astype(np.float64),
            "point3d_ids": resume_ids,
            "K": np.asarray(resume_meta["K"], dtype=np.float64),
            "colmap_image_id": resume_meta.get("colmap_image_id"),
            "colmap_camera_id": resume_meta.get("colmap_camera_id"),
            "colmap_registered_points": resume_meta.get("colmap_registered_points"),
            "colmap_qvec_world_to_camera": resume_meta.get("colmap_qvec_world_to_camera"),
            "colmap_tvec_world_to_camera": resume_meta.get("colmap_tvec_world_to_camera"),
        }
        prev_gray = load_gray(resume_image)
        rotation_prev_gray = prev_gray.copy()
        stable_solve_point3d_ids = resume_ids.copy()
        last_output_pose = trusted_pose
        last_center = np.asarray(trusted_pose["center"], dtype=np.float64)
        last_output_center = last_center.copy()
        last_output_time = (
            float(trusted_pose["time_sec"])
            if trusted_pose.get("time_sec") is not None
            else None
        )
        if trusted_pose.get("rheading"):
            rotation_heading = list(trusted_pose["rheading"])
        if trusted_pose.get("rcenter"):
            rotation_last_center = list(trusted_pose["rcenter"])
        if trusted_pose.get("received_unix") is not None:
            rotation_last_received_unix = float(trusted_pose["received_unix"])
        prev_case_id = f"resume_{args.resume_case_dir.name}"
        partial_poses = resume_poses
        print(
            "RESUMING TRUSTED TEMPORAL TRACK:",
            json.dumps(
                {
                    "start_frame_index": int(args.start_frame_index),
                    "resume_image": resume_image_name,
                    "correspondences": int(len(resume_ids)),
                    "prior_poses": int(len(partial_poses)),
                    "trusted_center": last_center.tolist(),
                }
            ),
            flush=True,
        )

    def attach_rotation_only_hint(pose: dict[str, Any] | None) -> dict[str, Any] | None:
        """Attach independent optical yaw; it never changes pose validity."""
        if pose is None:
            return pose
        heading = normalize_room_heading(rotation_heading)
        if heading is not None and rotation_heading_tracks >= 16:
            pose.update(
                {
                    "rotation_heading": heading,
                    "rotation_heading_source": "optical_flow_yaw",
                    "rotation_heading_tracks": int(rotation_heading_tracks),
                    "rotation_heading_delta_deg": rotation_heading_delta_deg,
                }
            )
        return pose

    def update_rotation_reference_from_accepted_pose(pose: dict[str, Any] | None) -> bool:
        """Seed optical yaw only from a locally continuous TSolve position.

        The bridge can reject a room-frame jump that passed a looser solver
        check.  Do not let that same candidate reset the optical yaw anchor.
        """
        nonlocal rotation_heading, rotation_last_center, rotation_last_received_unix
        if pose is None or not pose.get("success") or pose.get("held_pose"):
            return False
        heading = normalize_room_heading(pose.get("rheading"))
        center_raw = pose.get("rcenter")
        if heading is None or not isinstance(center_raw, list) or len(center_raw) < 3:
            return False
        try:
            center = [float(center_raw[0]), float(center_raw[1]), float(center_raw[2])]
            received = float(pose.get("received_unix"))
        except (TypeError, ValueError):
            return False
        if not all(math.isfinite(value) for value in center) or not math.isfinite(received):
            return False
        if rotation_last_center is not None:
            previous_time = rotation_last_received_unix
            dt = max(0.0, received - previous_time) if previous_time is not None else 0.0
            # This is intentionally stricter than the global TSolve output
            # gate.  Ten-FPS indoor flight cannot physically translate this
            # far between frames; a larger candidate is a heading-anchor risk.
            max_step = max(0.22, min(0.35, 0.18 + 0.17 * dt))
            if math.hypot(center[0] - rotation_last_center[0], center[2] - rotation_last_center[2]) > max_step:
                return False
        rotation_heading = heading
        rotation_last_center = center
        rotation_last_received_unix = received
        return True

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
            # A global map rematch may take several seconds on the full lab
            # map.  Never discard it and create another worker every few live
            # frames; that starves the live localizer and is exactly the
            # behaviour that made the Point-2 exit stall.
            return False
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
            # A recovery pool is tied to the camera view that produced it.
            # Preserve that view's independent yaw so a result returned after
            # a large in-place turn is discarded instead of synchronously
            # walking old correspondences through the entire turn.
            "rotation_heading": normalize_room_heading(rotation_heading),
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
            "received_unix": (
                float(frame_times.get(frame.name, {}).get("received_unix"))
                if frame_times.get(frame.name, {}).get("received_unix")
                else None
            ),
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
        if not result.get("success"):
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

        output_accepted = bool(result.get("success")) and output_rejection_reason is None
        if output_accepted:
            # Only a publicly accepted pose may advance the inherited
            # optical-flow chain and its map-reference center.
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
        elif result.get("success"):
            reference_update_reason = "held_rejected_pose_not_trusted"
            if consecutive_local_failures >= args.global_recovery_after_failures:
                # Force the next camera frame through a clean map rematch. The
                # flight bridge sees the held output and hovers in the meantime.
                current_pool = None
                prev_gray = None
                prev_case_id = None
                print(
                    "OUTPUT REJECTION RECOVERY ARMED:",
                    json.dumps(
                        {
                            "frame_index": frame_idx,
                            "consecutive_rejections": consecutive_local_failures,
                            "reason": output_rejection_reason,
                        }
                    ),
                    flush=True,
                )

        if output_rejection_reason is not None and last_output_pose is not None:
            pose_payload = held_pose_from_last(
                last_pose=last_output_pose,
                current_frame=current_frame_meta,
                reason=output_rejection_reason,
                rotation_heading=rotation_heading,
                rotation_heading_tracks=rotation_heading_tracks,
                rotation_heading_delta_deg=rotation_heading_delta_deg,
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
        pose_payload = attach_rotation_only_hint(pose_payload)
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

    def catch_up_background_recovery_pool(
        *,
        recovery: dict[str, Any],
        pool: dict[str, Any],
        current_frame_idx: int,
        current_gray: np.ndarray,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Propagate an old global match to the current live camera frame.

        A COLMAP rematch can take many seconds.  Its 2D/3D correspondences are
        useful, but its pose is not current enough to command the drone.  Track
        those correspondences through every captured intermediate image so the
        normal TSolve path can solve a genuinely current observation.
        """
        source_frame_idx = int(recovery["frame_idx"])
        source_gray = np.asarray(recovery["gray"])
        if current_frame_idx < source_frame_idx:
            return None, {
                "reason": "background_recovery_current_frame_precedes_source",
                "source_frame": source_frame_idx,
                "current_frame": current_frame_idx,
                "tracked_points": 0,
                "catchup_frames": 0,
                "catchup_ms": 0.0,
            }

        catchup_t0 = time.perf_counter()
        caught_pool = dict(pool)
        previous_gray = source_gray
        available_frames = image_files(args.query_frames)
        if current_frame_idx >= len(available_frames):
            return None, {
                "reason": "background_recovery_current_frame_not_available",
                "source_frame": source_frame_idx,
                "current_frame": current_frame_idx,
                "tracked_points": 0,
                "catchup_frames": 0,
                "catchup_ms": (time.perf_counter() - catchup_t0) * 1000.0,
            }

        optical_flow_ms = 0.0
        catchup_frames = 0
        catchup_span = current_frame_idx - source_frame_idx
        if catchup_span == 0:
            return caught_pool, {
                "reason": "",
                "source_frame": source_frame_idx,
                "current_frame": current_frame_idx,
                "tracked_points": int(len(np.asarray(caught_pool.get("xy", [])))),
                "catchup_frames": 0,
                "optical_flow_ms": 0.0,
                "catchup_ms": (time.perf_counter() - catchup_t0) * 1000.0,
                "direct_catchup": True,
            }

        # First try one direct optical-flow propagation.  When the drone has
        # finished turning and is hovering, the source and current images are
        # usually close enough for this to update an eight-second-old global
        # match in one bounded operation instead of blocking the live loop
        # while replaying 80+ intermediate frames.
        direct_previous_count = max(1, int(len(np.asarray(caught_pool.get("xy", [])))))
        direct_pool, direct_stage = track_pool(
            source_gray,
            current_gray,
            caught_pool,
            max_error=args.flow_max_error,
            backtrack_error=args.flow_backtrack_error,
            win_size=args.flow_window,
            max_level=args.flow_levels,
            iterations=args.flow_iterations,
        )
        direct_flow_ms = float(direct_stage.get("optical_flow_ms") or 0.0)
        optical_flow_ms += direct_flow_ms
        direct_count = int(direct_stage.get("tracked_points") or 0)
        direct_ratio = float(direct_count) / float(direct_previous_count)
        if (
            direct_pool is not None
            and direct_count >= args.min_track_points
            and direct_ratio >= args.min_track_ratio
        ):
            return direct_pool, {
                "reason": "",
                "source_frame": source_frame_idx,
                "current_frame": current_frame_idx,
                "tracked_points": direct_count,
                "catchup_frames": 1,
                "optical_flow_ms": optical_flow_ms,
                "catchup_ms": (time.perf_counter() - catchup_t0) * 1000.0,
                "direct_catchup": True,
            }

        # Sequential catch-up is retained for short gaps, but it must never
        # replay an entire multi-second in-place turn on the live thread.
        # Twelve ten-FPS frames keep this fallback bounded to roughly one
        # second; a longer failed direct catch-up is discarded and rematched
        # from the current view by the next background worker.
        max_sequential_catchup_frames = 12 if args.follow_dir else catchup_span
        if catchup_span > max_sequential_catchup_frames:
            direct_reason = str(direct_stage.get("reason") or "direct_optical_flow_failed")
            return None, {
                "reason": (
                    "background_recovery_catchup_span_"
                    f"{catchup_span}_gt_{max_sequential_catchup_frames}_after_{direct_reason}"
                ),
                "source_frame": source_frame_idx,
                "current_frame": current_frame_idx,
                "tracked_points": direct_count,
                "catchup_frames": 1,
                "optical_flow_ms": optical_flow_ms,
                "catchup_ms": (time.perf_counter() - catchup_t0) * 1000.0,
                "direct_catchup": False,
            }

        for intermediate_idx in range(source_frame_idx + 1, current_frame_idx + 1):
            # Reuse the image already loaded by the live loop for the final
            # step.  Earlier images remain in the session query directory.
            next_gray = (
                current_gray
                if intermediate_idx == current_frame_idx
                else load_gray(available_frames[intermediate_idx])
            )
            previous_count = max(1, int(len(np.asarray(caught_pool.get("xy", [])))))
            tracked, flow_stage = track_pool(
                previous_gray,
                next_gray,
                caught_pool,
                max_error=args.flow_max_error,
                backtrack_error=args.flow_backtrack_error,
                win_size=args.flow_window,
                max_level=args.flow_levels,
                iterations=args.flow_iterations,
            )
            optical_flow_ms += float(flow_stage.get("optical_flow_ms") or 0.0)
            tracked_count = int(flow_stage.get("tracked_points") or 0)
            tracked_ratio = float(tracked_count) / float(previous_count)
            catchup_frames += 1
            if (
                tracked is None
                or tracked_count < args.min_track_points
                or tracked_ratio < args.min_track_ratio
            ):
                reason = str(flow_stage.get("reason") or "too_few_tracked_points")
                if tracked_count < args.min_track_points:
                    reason = f"{reason}_tracked_{tracked_count}_lt_{args.min_track_points}"
                elif tracked_ratio < args.min_track_ratio:
                    reason = f"{reason}_ratio_{tracked_ratio:.3f}_lt_{args.min_track_ratio:.3f}"
                return None, {
                    "reason": f"background_recovery_catchup_failed_{reason}",
                    "source_frame": source_frame_idx,
                    "current_frame": current_frame_idx,
                    "failed_frame": intermediate_idx,
                    "tracked_points": tracked_count,
                    "catchup_frames": catchup_frames,
                    "optical_flow_ms": optical_flow_ms,
                    "catchup_ms": (time.perf_counter() - catchup_t0) * 1000.0,
                }
            caught_pool = tracked
            previous_gray = next_gray

        return caught_pool, {
            "reason": "",
            "source_frame": source_frame_idx,
            "current_frame": current_frame_idx,
            "tracked_points": int(len(np.asarray(caught_pool.get("xy", [])))),
            "catchup_frames": catchup_frames,
            "optical_flow_ms": optical_flow_ms,
            "catchup_ms": (time.perf_counter() - catchup_t0) * 1000.0,
        }

    def apply_background_global_recovery(
        *,
        current_frame_idx: int | None = None,
        current_gray: np.ndarray | None = None,
    ) -> None:
        nonlocal pending_global, current_pool, prev_gray, prev_case_id
        nonlocal last_center, consecutive_local_failures, global_relocalization_count
        nonlocal background_recovery_success_count, background_recovery_stale_count
        nonlocal stable_solve_point3d_ids
        if pending_global is None or not pending_global.get("done"):
            return
        # A live recovery result is consumed only when the loop has loaded a
        # current image.  Publishing the worker's old pose is never allowed.
        if args.follow_dir and (current_frame_idx is None or current_gray is None):
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
        recovery_age_frames = max(
            0,
            int(current_frame_idx if current_frame_idx is not None else processed_frames)
            - int(recovery.get("frame_idx", processed_frames)),
        )
        if args.follow_dir and accepted:
            live_frames = image_files(args.query_frames)
            current_image_name = (
                f"query/{live_frames[int(current_frame_idx)].name}"
                if 0 <= int(current_frame_idx) < len(live_frames)
                else str(recovery["query_name"])
            )
            recovery_heading_shift = room_heading_separation_degrees(
                recovery.get("rotation_heading"),
                rotation_heading,
            )
            if recovery_heading_shift is not None and recovery_heading_shift > 15.0:
                # This global match belongs to a camera view from before the
                # in-place turn.  It may contain a mathematically valid but
                # physically false translated solution (the repeated ~6.8 m
                # Point-3-to-4 jump).  Keep the last trusted position, consume
                # this stale worker, and immediately rematch the current view
                # in the background on the next live iteration.
                background_recovery_stale_count += 1
                stage["reason"] = (
                    "background_recovery_view_rotated_"
                    f"{recovery_heading_shift:.1f}deg_gt_15.0deg"
                )
                print(
                    "BACKGROUND RECOVERY DISCARDED AFTER TURN:",
                    json.dumps(
                        {
                            "source_frame": int(recovery["frame_idx"]),
                            "current_frame": int(current_frame_idx),
                            "heading_shift_deg": round(recovery_heading_shift, 2),
                            "age_frames": recovery_age_frames,
                            "elapsed_ms": round(elapsed_ms, 1),
                        }
                    ),
                    flush=True,
                )
                append_stage(
                    stage_csv,
                    {
                        "frame_index": current_frame_idx,
                        "case_id": "",
                        "image_name": current_image_name,
                        "time_sec": "",
                        "method": "global_colmap_background_recovery_turn_discard",
                        "accepted": False,
                        "tracked_points": int(len(np.asarray(pool.get("xy", [])))),
                        "selected_points": 0,
                        "feature_extract_ms": stage.get("feature_extract_ms", 0.0),
                        "match_ms": stage.get("match_ms", 0.0),
                        "register_ms": stage.get("register_ms", 0.0),
                        "optical_flow_ms": 0.0,
                        "tsolve_ms": 0.0,
                        "total_frame_ms": elapsed_ms,
                        "reason": stage["reason"],
                    },
                )
                return
            caught_pool, catchup = catch_up_background_recovery_pool(
                recovery=recovery,
                pool=pool,
                current_frame_idx=int(current_frame_idx),
                current_gray=np.asarray(current_gray),
            )
            if caught_pool is None:
                background_recovery_stale_count += 1
                stage["reason"] = str(catchup["reason"])
                print(
                    "BACKGROUND RECOVERY CATCH-UP FAILED:",
                    json.dumps(
                        {
                            **catchup,
                            "age_frames": recovery_age_frames,
                            "elapsed_ms": round(elapsed_ms, 1),
                        }
                    ),
                    flush=True,
                )
                append_stage(
                    stage_csv,
                    {
                        "frame_index": current_frame_idx,
                        "case_id": "",
                        "image_name": current_image_name,
                        "time_sec": "",
                        "method": "global_colmap_background_recovery_catchup",
                        "accepted": False,
                        "tracked_points": int(catchup.get("tracked_points") or 0),
                        "selected_points": 0,
                        "feature_extract_ms": stage.get("feature_extract_ms", 0.0),
                        "match_ms": stage.get("match_ms", 0.0),
                        "register_ms": stage.get("register_ms", 0.0),
                        "optical_flow_ms": catchup.get("optical_flow_ms", 0.0),
                        "tsolve_ms": 0.0,
                        "total_frame_ms": elapsed_ms + float(catchup.get("catchup_ms") or 0.0),
                        "reason": stage["reason"],
                    },
                )
                return

            # Install correspondences at the current image, not the delayed
            # COLMAP pose.  The normal code below this hook immediately runs
            # TSolve and continuity checks before any pose can be published.
            current_pool = cap_tracking_pool(caught_pool, args.track_pool_size)
            prev_gray = np.asarray(current_gray)
            prev_case_id = None
            stable_solve_point3d_ids = None
            consecutive_local_failures = 0
            background_recovery_success_count += 1
            print(
                "BACKGROUND RECOVERY CAUGHT UP:",
                json.dumps(
                    {
                        **catchup,
                        "age_frames": recovery_age_frames,
                        "source_points": int(len(np.asarray(pool.get("xy", [])))),
                        "elapsed_ms": round(elapsed_ms, 1),
                    }
                ),
                flush=True,
            )
            append_stage(
                stage_csv,
                {
                    "frame_index": current_frame_idx,
                    "case_id": "",
                    "image_name": current_image_name,
                    "time_sec": "",
                    "method": "global_colmap_background_recovery_catchup",
                    "accepted": True,
                    "tracked_points": int(catchup["tracked_points"]),
                    "selected_points": 0,
                    "feature_extract_ms": stage.get("feature_extract_ms", 0.0),
                    "match_ms": stage.get("match_ms", 0.0),
                    "register_ms": stage.get("register_ms", 0.0),
                    "optical_flow_ms": catchup.get("optical_flow_ms", 0.0),
                    "tsolve_ms": 0.0,
                    "total_frame_ms": elapsed_ms + float(catchup.get("catchup_ms") or 0.0),
                    "reason": "",
                },
            )
            return
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
            "received_unix": float(raw_frame_row["received_unix"]) if raw_frame_row.get("received_unix") else None,
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
        if current_pool is not None:
            try:
                focal_candidate = float(np.asarray(current_pool.get("K"), dtype=float)[0, 0])
                if math.isfinite(focal_candidate) and focal_candidate >= 100.0:
                    rotation_focal_px = focal_candidate
            except (TypeError, ValueError, IndexError):
                pass
        rotation_delta, rotation_tracks = optical_flow_yaw_delta(
            rotation_prev_gray,
            curr_gray,
            rotation_focal_px,
        )
        rotation_prev_gray = curr_gray
        rotation_heading_tracks = rotation_tracks
        rotation_heading_delta_deg = math.degrees(rotation_delta) if rotation_delta is not None else None
        if rotation_delta is not None and rotation_heading is not None:
            rotation_heading = rotate_room_heading(rotation_heading, rotation_delta)
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
        apply_background_global_recovery(
            current_frame_idx=frame_idx,
            current_gray=curr_gray,
        )

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
        proactive_fallback_pool: dict[str, Any] | None = None
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
                proactive_due = (
                    args.proactive_relocalize_points > 0
                    and tracked_count <= args.proactive_relocalize_points
                    and frame_idx - last_proactive_relocalize_frame
                    >= args.proactive_relocalize_cooldown_frames
                )
                if proactive_due:
                    global_reason = f"proactive_pool_refresh_{tracked_count}"
                    if not args.blocking_global_recovery and schedule_background_global_recovery(
                        frame_idx=frame_idx,
                        frame=frame,
                        query_name=query_name,
                        curr_gray=curr_gray,
                        reason=global_reason,
                    ):
                        # The current optical pool is valid.  Keep publishing
                        # it while the expensive map rematch runs in the
                        # background; a proactive refresh must never pause
                        # closed-loop travel.
                        method = "optical_flow_proactive_refresh_background"
                        stage["reason"] = "proactive_map_refresh_background"
                        proactive_relocalization_count += 1
                        last_proactive_relocalize_frame = frame_idx
                        print(
                            "PROACTIVE BACKGROUND POOL REFRESH:",
                            json.dumps(
                                {
                                    "frame_index": frame_idx,
                                    "tracked_points": tracked_count,
                                    "trigger_points": args.proactive_relocalize_points,
                                }
                            ),
                            flush=True,
                        )
                    elif pending_global is not None and not args.blocking_global_recovery:
                        method = "optical_flow_proactive_refresh_pending"
                        stage["reason"] = "proactive_map_refresh_pending_keep_valid_optical_flow"
                    else:
                        # Preserve the legacy blocking route for offline runs
                        # that explicitly request it.  Live flight defaults to
                        # the non-blocking branch above.
                        proactive_fallback_pool = tracked
                        pool = None
                        must_global = True
                        proactive_relocalization_count += 1
                        last_proactive_relocalize_frame = frame_idx
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

        if (
            must_global
            and args.follow_dir
            and last_output_pose is not None
            and not args.blocking_global_recovery
            and not args.disable_background_recovery
        ):
            # After two rejected translated roots, the trusted 3D tracking
            # pool is intentionally cleared.  During a live in-place turn,
            # however, a synchronous COLMAP retry takes ~8 seconds and makes
            # the valid optical yaw stale.  Recover map position in the
            # background while continuing to publish fresh held-position,
            # rotation-only observations.  Forward/lateral movement remains
            # locked until a new position passes the normal continuity gate.
            recovery_reason = global_reason or "missing_tracking_anchor"
            if recovery_reason == "bootstrap":
                recovery_reason = "missing_tracking_anchor_after_rejection"
            recovery_scheduled = schedule_background_global_recovery(
                frame_idx=frame_idx,
                frame=frame,
                query_name=query_name,
                curr_gray=curr_gray,
                reason=recovery_reason,
            )
            if recovery_scheduled or pending_global is not None:
                must_global = False
                method = (
                    "global_colmap_background_recovery_scheduled"
                    if recovery_scheduled
                    else "global_colmap_background_recovery_pending"
                )
                stage["reason"] = (
                    f"{recovery_reason}_holding_trusted_position_with_fresh_rotation_heading"
                )

        if must_global:
            method = f"global_colmap_{global_reason or 'recovery'}"
            refreshed_pool, global_stage = relocalizer.localize(
                frame=frame,
                frame_idx=frame_idx,
                query_name=query_name,
                map_points=map_points,
                last_center=last_center,
            )
            stage.update(global_stage)
            global_relocalization_count += 1
            if refreshed_pool is not None:
                pool = refreshed_pool
                consecutive_local_failures = 0
                if proactive_fallback_pool is not None:
                    proactive_relocalization_success_count += 1
            elif proactive_fallback_pool is not None:
                pool = proactive_fallback_pool
                method = "optical_flow_proactive_refresh_fallback"
                proactive_relocalization_fallback_count += 1
                stage["reason"] = "proactive_rematch_failed_using_valid_optical_flow_pool"
                print(
                    "PROACTIVE POOL REFRESH FALLBACK:",
                    json.dumps(
                        {
                            "frame_index": frame_idx,
                            "tracked_points": int(len(np.asarray(pool.get("xy", [])))),
                        }
                    ),
                    flush=True,
                )

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
                rotation_heading=rotation_heading,
                rotation_heading_tracks=rotation_heading_tracks,
                rotation_heading_delta_deg=rotation_heading_delta_deg,
            )
            if held is not None:
                partial_poses.append(attach_rotation_only_hint(held))
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
                rotation_heading=rotation_heading,
                rotation_heading_tracks=rotation_heading_tracks,
                rotation_heading_delta_deg=rotation_heading_delta_deg,
            )
            if held is not None:
                partial_poses.append(attach_rotation_only_hint(held))
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
        reference_update_reason = "not_updated"
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

        output_accepted = bool(result.get("success")) and output_rejection_reason is None
        if output_accepted:
            # A rejected TSolve root must not poison later optical-flow
            # correspondences or move the map-search reference center.
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
        elif result.get("success"):
            reference_update_reason = "held_rejected_pose_not_trusted"
            if consecutive_local_failures >= args.global_recovery_after_failures:
                # Clearing both tracking anchors makes must_global true on the
                # next visible frame, so recovery uses fresh map matches.
                current_pool = None
                prev_gray = None
                prev_case_id = None
                print(
                    "OUTPUT REJECTION RECOVERY ARMED:",
                    json.dumps(
                        {
                            "frame_index": frame_idx,
                            "consecutive_rejections": consecutive_local_failures,
                            "reason": output_rejection_reason,
                        }
                    ),
                    flush=True,
                )

        if output_rejection_reason is not None and last_output_pose is not None:
            pose_payload = held_pose_from_last(
                last_pose=last_output_pose,
                current_frame=current_frame_meta,
                reason=output_rejection_reason,
                rotation_heading=rotation_heading,
                rotation_heading_tracks=rotation_heading_tracks,
                rotation_heading_delta_deg=rotation_heading_delta_deg,
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
        pose_payload = attach_rotation_only_hint(pose_payload)
        partial_poses.append(pose_payload)
        if pose_payload.get("success") and pose_payload.get("center"):
            last_output_pose = pose_payload
            # Keep the optical heading anchor independent from the map-pose
            # publication path.  A TSolve candidate can pass this process's
            # broad output gate yet still be rejected later by the DJI bridge
            # as a physical motion jump.  Such a candidate must not replace
            # the heading used to safely finish an in-place turn.
            update_rotation_reference_from_accepted_pose(pose_payload)
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
        "proactive_relocalize_points": args.proactive_relocalize_points,
        "proactive_relocalize_cooldown_frames": args.proactive_relocalize_cooldown_frames,
        "proactive_relocalization_count": proactive_relocalization_count,
        "proactive_relocalization_success_count": proactive_relocalization_success_count,
        "proactive_relocalization_fallback_count": proactive_relocalization_fallback_count,
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
