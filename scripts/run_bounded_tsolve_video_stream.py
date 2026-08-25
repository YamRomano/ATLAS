#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from colmap_io import (
    CAMERA_MODEL_BY_ID,
    Camera,
    Image,
    qvec_to_rotmat,
    read_cameras_model,
    read_images_model,
    read_points3d_text,
)
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
)
from patrol_visual_route_recovery import PatrolVisualRouteRecovery
from taught_patrol_recovery import TaughtPatrolRecovery
from colmap_faiss_relocalizer import FaissIVF3DRelocalizer


def run_timed(cmd: list[object], *, timeout: float | None = None) -> float:
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
        timeout=timeout,
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


STAGE_CSV_FIELDS = [
    "frame_index",
    "case_id",
    "image_name",
    "time_sec",
    "method",
    "accepted",
    "extracted_features",
    "matched_features",
    "flow_input_points",
    "tracked_points",
    "pnp_inliers",
    "selected_points",
    "pruned_features",
    "frame_load_ms",
    "heading_flow_ms",
    "feature_extract_ms",
    "match_ms",
    "register_ms",
    "optical_flow_ms",
    "case_build_ms",
    "case_output_ms",
    "visual_route_ms",
    "visual_heading_ms",
    "route_logic_ms",
    "local_recovery_ms",
    "background_apply_ms",
    "background_worker_ms",
    "pose_update_ms",
    "stream_publish_ms",
    "tsolve_ms",
    "pace_wait_ms",
    "total_frame_ms",
    "reason",
]

POSE_STREAM_WINDOW = 180
POSE_STREAM_STATE: dict[str, dict[str, Any]] = {}
MANIFEST_FIELDS = [
    "experiment",
    "case_id",
    "p3d_csv",
    "p2d_csv",
    "input_json",
    "points",
    "image_name",
    "time_sec",
    "localization_attempt",
]


def write_stage_header(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=STAGE_CSV_FIELDS)
        writer.writeheader()


def append_stage(
    path: Path,
    row: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
) -> None:
    payload = dict(row)
    source = diagnostics if isinstance(diagnostics, dict) else {}
    for key in (
        "extracted_features",
        "matched_features",
        "flow_input_points",
        "pnp_inliers",
        "pruned_features",
        "pace_wait_ms",
        "frame_load_ms",
        "heading_flow_ms",
        "case_build_ms",
        "case_output_ms",
        "visual_route_ms",
        "visual_heading_ms",
        "route_logic_ms",
        "local_recovery_ms",
        "background_apply_ms",
        "background_worker_ms",
        "pose_update_ms",
        "stream_publish_ms",
    ):
        if key not in payload and source.get(key) is not None:
            payload[key] = source.get(key)

    def count(name: str) -> int | None:
        value = payload.get(name)
        if value in (None, ""):
            return None
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return None

    # "Pruned" is contextual: on optical frames it is the LK pool rejected
    # before pose solving; on a SIFT recovery it is the descriptors rejected
    # before a unique 2D-to-3D candidate survived.  If neither upstream count
    # is available, report the tracked pool not selected for the bounded
    # TSolve case.  The GUI also shows every constituent count, so this number
    # is never ambiguous in isolation.
    if count("pruned_features") is None:
        extracted = count("extracted_features")
        matched = count("matched_features")
        flow_input = count("flow_input_points")
        tracked = count("tracked_points")
        selected = count("selected_points")
        if extracted is not None and matched is not None:
            payload["pruned_features"] = max(0, extracted - matched)
        elif flow_input is not None and tracked is not None:
            payload["pruned_features"] = max(0, flow_input - tracked)
        elif tracked is not None and selected is not None:
            payload["pruned_features"] = max(0, tracked - selected)

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=STAGE_CSV_FIELDS)
        writer.writerow(payload)


def append_manifest_row(inputs_out: Path, row: dict[str, Any]) -> None:
    """Append one live case instead of rewriting the growing manifest."""
    path = inputs_out / "manifest.csv"
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # This is the live machine-to-machine pose stream.  Rewriting an indented
    # multi-megabyte history on every frame used more wall time than optical
    # tracking plus TSolve, so keep the exact payload but serialize it compactly.
    tmp.write_text(
        json.dumps(payload, separators=(",", ":"), default=str),
        encoding="utf-8",
    )
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


def rotation_frame_gap_status(
    previous_time: float | None,
    current_time: float | None,
    *,
    max_gap_seconds: float = 0.30,
) -> tuple[bool, float | None]:
    """Return whether two camera frames are close enough for optical yaw.

    The live consumer deliberately drops queued frames when localization falls
    behind.  Lucas-Kanade flow across that discontinuity is not an adjacent-
    frame measurement and can substantially under-count a real turn.  Treat
    the first frame after a gap as a new seed; the following contiguous frame
    can resume optical yaw normally.
    """
    try:
        previous = float(previous_time) if previous_time is not None else None
        current = float(current_time) if current_time is not None else None
        maximum = float(max_gap_seconds)
    except (TypeError, ValueError):
        return False, None
    if (
        previous is None
        or current is None
        or not math.isfinite(previous)
        or not math.isfinite(current)
        or not math.isfinite(maximum)
        or maximum <= 0.0
    ):
        return False, None
    gap = current - previous
    return bool(0.0 < gap <= maximum), gap


def rotation_heading_timing_policy(
    previous_time: float | None,
    current_time: float | None,
    *,
    route_leg_index: int,
    max_gap_seconds: float = 0.30,
) -> tuple[bool, float | None, str]:
    """Choose whether a frame pair may update live optical yaw.

    Live ATLAS 09:30:17 established the reliable Point-1 -> Point-2 ->
    Point-3 behavior by integrating every forward-in-time optical pair.  A
    later global discontinuity guard fixed under-counted yaw on the weak
    Point-3/Point-4 tail, but applying it to the established first two legs
    discarded real turn motion whenever localization briefly exceeded 300 ms.

    Keep the exact early-route behavior for legs 1 and 2.  The stricter
    adjacent-frame requirement remains active for legs 3 and 4, where a
    skipped burst previously produced the dangerous delayed-heading failure.
    """
    contiguous, gap = rotation_frame_gap_status(
        previous_time,
        current_time,
        max_gap_seconds=max_gap_seconds,
    )
    if contiguous:
        return True, gap, "adjacent_frame_pair"
    if int(route_leg_index) in {1, 2} and gap is not None and gap > 0.0:
        return True, gap, "legacy_093017_early_leg_continuity"
    return False, gap, "strict_tail_gap_reseed"


def optical_flow_yaw_delta(
    previous: np.ndarray | None,
    current: np.ndarray,
    focal_px: float,
) -> tuple[float | None, int]:
    """Estimate small camera yaw between adjacent images.

    This is intentionally a rotation-only hint.  It never derives map
    translation; the pose-stream stabilizer uses consecutive reliable yaw
    observations to freeze position during autonomous and manually taught
    in-place turns.
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


class RotationOnlyPositionStabilizer:
    """Keep room position fixed while consecutive frames show in-place yaw.

    TSolve still solves every frame and its raw center is retained for audit.
    The published room center is anchored during a turn, then the accumulated
    rotation-only center drift is removed before translation resumes.  This
    works for both autonomous patrol turns and manually taught turns.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        start_degrees: float = 0.20,
        stop_degrees: float = 0.06,
        minimum_tracks: int = 80,
        start_frames: int = 2,
        stop_frames: int = 4,
        max_reanchor_step: float = 0.75,
        bias_decay: float = 0.92,
    ) -> None:
        # Image-wide horizontal flow is not sufficient to distinguish yaw
        # from forward/lateral translation. Production localization must
        # therefore publish the solved room position unless an external
        # controller can explicitly prove that it commanded yaw-only motion.
        # Keep this stabilizer available for isolated/offline evaluation, but
        # fail safe by leaving it disabled in the live stream.
        self.enabled = bool(enabled)
        self.start_degrees = max(0.0, float(start_degrees))
        self.stop_degrees = max(0.0, min(float(stop_degrees), self.start_degrees))
        self.minimum_tracks = max(16, int(minimum_tracks))
        self.start_frames = max(1, int(start_frames))
        self.stop_frames = max(1, int(stop_frames))
        self.max_reanchor_step = max(0.05, float(max_reanchor_step))
        self.bias_decay = max(0.0, min(1.0, float(bias_decay)))
        self.start_streak = 0
        self.quiet_streak = 0
        self.candidate_anchor: np.ndarray | None = None
        self.position_anchor: np.ndarray | None = None
        self.position_anchor_commanded = False
        self.release_anchor: np.ndarray | None = None
        self.release_anchor_commanded = False
        self.room_bias = np.zeros(3, dtype=float)
        self.room_bias_commanded = False
        self.locked_frames = 0
        self.reanchor_count = 0
        self.stale_release_discard_count = 0
        self.active_route_key: tuple[Any, ...] | None = None
        self.route_transition_reset_count = 0

    @staticmethod
    def _center(value: Any) -> np.ndarray | None:
        if not isinstance(value, (list, tuple, np.ndarray)):
            return None
        try:
            center = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
        if center.size < 3 or not np.all(np.isfinite(center[:3])):
            return None
        return center[:3].copy()

    @property
    def active(self) -> bool:
        return self.position_anchor is not None

    def accept_absolute_position(self) -> None:
        """Drop turn-relative state after an independent absolute publication."""
        self.start_streak = 0
        self.quiet_streak = 0
        self.candidate_anchor = None
        self.position_anchor = None
        self.position_anchor_commanded = False
        self.release_anchor = None
        self.release_anchor_commanded = False
        self.room_bias[:] = 0.0
        self.room_bias_commanded = False

    def metric_route_room_bias(self) -> list[float] | None:
        """Return the commanded-yaw room correction used by metric TSolve.

        A yaw-only command cannot translate the aircraft, so after the turn
        ``apply()`` carries a constant room-frame correction that removes the
        monocular center drift accumulated during rotation. Legs 1 and 2 are
        TSolve-led; their route gate must evaluate that same corrected metric
        center instead of the pre-correction raw center. Expose only a bias
        established by an explicit controller yaw command, never an
        optical-flow-only inferred correction.
        """
        if (
            not self.enabled
            or self.active
            or not self.room_bias_commanded
            or not np.all(np.isfinite(self.room_bias))
        ):
            return None
        return self.room_bias.astype(float).tolist()
    def observe(
        self,
        *,
        delta_degrees: float | None,
        tracks: int,
        last_published_center: Any,
        commanded_yaw_only: bool | None = None,
        commanded_position_anchor: Any = None,
        route_key: tuple[Any, ...] | None = None,
    ) -> bool:
        if not self.enabled:
            return False
        if route_key is not None and route_key != self.active_route_key:
            # A waypoint transition is an ownership boundary.  No release
            # anchor, bias, or candidate from the previous leg may survive
            # into the next turn.  The controller's new waypoint anchor is
            # consumed below in the same frame, making the transition atomic.
            if self.active_route_key is not None:
                self.route_transition_reset_count += 1
            self.start_streak = 0
            self.quiet_streak = 0
            self.candidate_anchor = None
            self.position_anchor = None
            self.position_anchor_commanded = False
            self.release_anchor = None
            self.release_anchor_commanded = False
            self.room_bias[:] = 0.0
            self.room_bias_commanded = False
            self.active_route_key = route_key
        if commanded_yaw_only is False:
            # The controller has announced hover or translation. End any
            # optical yaw lock before processing this frame; apply() will
            # either reanchor within the 14:17 limit or keep translation
            # explicitly forbidden until localization returns near the anchor.
            self.start_streak = 0
            self.candidate_anchor = None
            self.quiet_streak = 0
            published = self._center(last_published_center)
            if self.release_anchor is not None and published is not None:
                release_drift = math.hypot(
                    float(published[0] - self.release_anchor[0]),
                    float(published[2] - self.release_anchor[2]),
                )
                if release_drift > 0.16:
                    # Visual-route recovery can carry real translation while
                    # metric TSolve has no pool. In that branch apply() is not
                    # called, so an already-released yaw anchor used to survive
                    # all the way from Point 3 to Point 4. The first held frame
                    # of the Point-4 turn then snapped the model back to Point 3.
                    self.release_anchor = None
                    self.release_anchor_commanded = False
                    self.room_bias[:] = 0.0
                    self.room_bias_commanded = False
                    self.stale_release_discard_count += 1
            if self.position_anchor is not None:
                published_drift = (
                    math.hypot(
                        float(published[0] - self.position_anchor[0]),
                        float(published[2] - self.position_anchor[2]),
                    )
                    if published is not None
                    else 0.0
                )
                if published_drift > 0.16:
                    # The model has already published real translation away
                    # from this yaw anchor. Restoring it would rewind the
                    # route. Live ATLAS 10:37:20 did exactly that after the
                    # Point-2 turn, replacing the newest Point-2→3 position
                    # with an older waypoint anchor.
                    self.release_anchor = None
                    self.release_anchor_commanded = False
                    self.room_bias[:] = 0.0
                    self.room_bias_commanded = False
                    self.stale_release_discard_count += 1
                else:
                    self.release_anchor = self.position_anchor.copy()
                    self.release_anchor_commanded = self.position_anchor_commanded
                self.position_anchor = None
                self.position_anchor_commanded = False
            return False
        if commanded_yaw_only is True:
            # The flight controller publishes its position anchor before it
            # sends a yaw-only pulse.  Use that explicit command boundary
            # immediately: waiting for two optical-flow yaw frames leaves a
            # race in which the first raw monocular center drift can reach the
            # command-side route gate and poison recovery.
            if self.position_anchor is None:
                anchor = self._center(commanded_position_anchor)
                if anchor is None:
                    anchor = self._center(last_published_center)
                if anchor is not None:
                    self.position_anchor = anchor
                    self.position_anchor_commanded = True
                    self.candidate_anchor = None
                    self.start_streak = 0
                    self.quiet_streak = 0
            else:
                # The optical hysteresis may have acquired the same turn one
                # frame before the controller status became visible. Once the
                # controller explicitly confirms yaw-only intent, its anchor
                # has position-truth semantics for the eventual release.
                self.position_anchor_commanded = True
            return self.active
        try:
            delta = abs(float(delta_degrees))
        except (TypeError, ValueError):
            delta = float("nan")
        reliable = math.isfinite(delta) and int(tracks) >= self.minimum_tracks
        strong_yaw = reliable and delta >= self.start_degrees
        quiet = reliable and delta <= self.stop_degrees

        if not self.active:
            if strong_yaw:
                if self.start_streak == 0:
                    self.candidate_anchor = self._center(last_published_center)
                self.start_streak += 1
                if self.start_streak >= self.start_frames and self.candidate_anchor is not None:
                    self.position_anchor = self.candidate_anchor.copy()
                    self.position_anchor_commanded = False
                    self.candidate_anchor = None
                    self.start_streak = 0
                    self.quiet_streak = 0
            else:
                self.start_streak = 0
                self.candidate_anchor = None
            return self.active

        if quiet:
            self.quiet_streak += 1
            if self.quiet_streak >= self.stop_frames:
                self.release_anchor = self.position_anchor.copy()
                self.release_anchor_commanded = self.position_anchor_commanded
                self.position_anchor = None
                self.position_anchor_commanded = False
                self.quiet_streak = 0
        else:
            # Missing/ambiguous flow must not release an established turn.
            self.quiet_streak = 0
        return self.active

    def apply(self, pose: dict[str, Any] | None) -> dict[str, Any] | None:
        if pose is None:
            return pose
        if not self.enabled:
            return pose
        center = self._center(pose.get("rcenter"))
        if center is None:
            return pose

        if self.active and self.position_anchor is not None:
            pose["rotation_raw_rcenter"] = center.tolist()
            pose["rcenter"] = self.position_anchor.tolist()
            pose["rotation_position_anchor"] = self.position_anchor.tolist()
            pose["rotation_position_locked"] = True
            pose["translation_allowed"] = False
            pose["rotation_position_source"] = "optical_flow_yaw_hysteresis"
            pose["rotation_anchor_commanded"] = self.position_anchor_commanded
            # The viewer and flight bridge consume `rheading`, not the audit-only
            # `rotation_heading` field.  During a held/rejected TSolve turn the
            # copied TSolve heading is stale, so publish the independent optical
            # heading while keeping the solver value for diagnostics.
            optical_heading = normalize_room_heading(pose.get("rotation_heading"))
            if optical_heading is not None:
                solver_heading = normalize_room_heading(pose.get("rheading"))
                if solver_heading is not None and "rheading_raw" not in pose:
                    pose["rheading_raw"] = solver_heading
                pose["rheading"] = optical_heading
                pose["rheading_source"] = "optical_flow_yaw"
            self.locked_frames += 1
            return pose

        # A held pose can contain a stale/raw center copied at the exact frame
        # where the controller releases yaw-only intent.  Live ATLAS 14:14:41
        # reached Point 4 correctly, then exposed that false center and rolled
        # the displayed route back before the localizer recovered.  A pure yaw
        # still cannot move the drone, so retain the release anchor (and the
        # independently tracked optical heading) until a fresh successful pose
        # is close enough to reanchor.
        if pose.get("held_pose") or not pose.get("success"):
            if self.release_anchor is not None:
                pose["rotation_raw_rcenter"] = center.tolist()
                pose["rcenter"] = self.release_anchor.tolist()
                pose["rotation_position_anchor"] = self.release_anchor.tolist()
                pose["rotation_position_locked"] = True
                pose["translation_allowed"] = False
                pose["rotation_reanchor_pending"] = True
                pose["rotation_position_source"] = "post_yaw_anchor_hold"
                pose["rotation_anchor_commanded"] = self.release_anchor_commanded
                optical_heading = normalize_room_heading(pose.get("rotation_heading"))
                if optical_heading is not None:
                    solver_heading = normalize_room_heading(pose.get("rheading"))
                    if solver_heading is not None and "rheading_raw" not in pose:
                        pose["rheading_raw"] = solver_heading
                    pose["rheading"] = optical_heading
                    pose["rheading_source"] = "optical_flow_yaw"
            return pose

        corrected = center + self.room_bias
        if self.release_anchor is not None:
            correction = self.release_anchor - corrected
            horizontal = math.hypot(float(correction[0]), float(correction[2]))
            commanded_reanchor = self.release_anchor_commanded
            if horizontal <= self.max_reanchor_step or commanded_reanchor:
                # A pure yaw cannot change room position.  The patrol anchor
                # is therefore the stronger translation reference than a
                # monocular center that drifted during the turn. Carry the
                # complete correction into the following optical track so its
                # real forward deltas remain relative to the saved route.
                self.room_bias = self.room_bias + correction
                self.room_bias_commanded = self.room_bias_commanded or commanded_reanchor
                corrected = center + self.room_bias
                self.reanchor_count += 1
                pose["rotation_reanchored_after_turn"] = True
                pose["rotation_anchor_is_position_truth"] = True
                pose["rotation_anchor_commanded"] = commanded_reanchor
            else:
                # Do not expose a large post-yaw coordinate snap. Keep the
                # route anchor published and wait for a later solution close
                # enough to reanchor safely.
                pose["rotation_raw_rcenter"] = center.tolist()
                pose["rcenter"] = self.release_anchor.tolist()
                pose["rotation_position_anchor"] = self.release_anchor.tolist()
                pose["rotation_position_locked"] = True
                pose["translation_allowed"] = False
                pose["rotation_reanchor_rejected"] = True
                pose["rotation_position_source"] = "post_yaw_anchor_hold"
                pose["rotation_release_correction"] = correction.tolist()
                return pose
            pose["rotation_release_correction"] = correction.tolist()
            self.release_anchor = None
            self.release_anchor_commanded = False

        if float(np.linalg.norm(self.room_bias)) > 1e-9:
            applied_bias = self.room_bias.copy()
            pose["rotation_raw_rcenter"] = center.tolist()
            pose["rcenter"] = corrected.tolist()
            pose["rotation_position_bias"] = applied_bias.tolist()
            pose["rotation_position_source"] = "post_yaw_room_bias"
            if self.room_bias_commanded:
                pose["rotation_anchor_is_position_truth"] = True
                pose["rotation_anchor_commanded"] = True
            # Position bias and heading authority are independent. This branch
            # is reached only by a fresh successful (non-held) pose after yaw
            # has ended. Replacing that current-frame TSolve heading with the
            # accumulated optical turn heading made a correct metric pose keep
            # the previous corner's yaw, most visibly at the second-lap Point
            # 1 -> Point 2 handoff. Optical yaw remains authoritative above
            # while yaw is active and while a held pose waits to reanchor; once
            # a fresh metric pose arrives, preserve its heading here. The
            # recorded-route absolute heading gate may still refine it later.
            if normalize_room_heading(pose.get("rheading")) is not None:
                pose["post_yaw_heading_authority"] = "fresh_metric_pose"
            # Keep the route-aligned offset while this local optical track is
            # active. Decaying it merely recreates the rotation-only drift as
            # fake translation and moves the displayed/controller position
            # away from the saved patrol corridor.
            if not self.room_bias_commanded:
                self.room_bias = self.room_bias * self.bias_decay
                if float(np.linalg.norm(self.room_bias)) < 1e-5:
                    self.room_bias[:] = 0.0
        return pose


def accept_lap_start_absolute_metric_position(
    stabilizer: RotationOnlyPositionStabilizer,
    *,
    pool: dict[str, Any] | None,
    output_accepted: bool,
) -> bool:
    """Make a fresh lap-start TSolve center absolute before publication.

    The repeated-lap checkpoint is independently registered from the newest
    camera image against the fixed COLMAP map.  It therefore supersedes every
    position anchor/bias accumulated while turning at Point 1.  Applying the
    old post-yaw room bias to this pose caused the captured lap-2 failure: a
    raw TSolve center about 1.25 m before Point 2 was shifted onto the saved
    Point-2 coordinate, so control stopped while the physical drone was still
    elsewhere.

    This reset is deliberately narrow.  Ordinary optical-flow/TSolve updates
    retain the existing turn stabilizer, and a failed/rejected rebootstrap can
    never clear the last trusted hold.
    """
    if not output_accepted or not isinstance(pool, dict):
        return False
    absolute_rebootstrap = bool(
        pool.get("lap_start_metric_rebootstrap") is True
        or pool.get("stopped_metric_rebootstrap") is True
    )
    if not absolute_rebootstrap:
        return False
    # ``cap_tracking_pool``/``track_pool`` intentionally preserve recovery
    # metadata. Consume this ownership marker once so later optical frames on
    # the leg cannot repeatedly reset a newly established stabilizer state.
    pool.pop("lap_start_metric_rebootstrap", None)
    pool.pop("stopped_metric_rebootstrap", None)
    stabilizer.accept_absolute_position()
    return True


def live_rotation_commanded(status_path: Path | None, max_age_seconds: float = 1.0) -> bool:
    """Return true only for a fresh controller-announced yaw-only interval."""
    commanded, _anchor = live_rotation_command_state(status_path, max_age_seconds)
    return commanded


def live_rotation_command_state(
    status_path: Path | None,
    max_age_seconds: float = 1.0,
) -> tuple[bool, list[float] | None]:
    """Return fresh yaw-only intent and the controller's exact room anchor."""
    if status_path is None or not status_path.exists():
        return False, None
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, None
    if str(payload.get("status") or "").strip().lower() != "running":
        return False, None
    if str(payload.get("command") or "").strip().lower() != "mission":
        return False, None
    try:
        age = time.time() - float(payload.get("updated_at") or 0.0)
    except (TypeError, ValueError):
        return False, None
    if age < -0.25 or age > max(0.1, float(max_age_seconds)):
        return False, None
    progress = payload.get("progress") if isinstance(payload.get("progress"), dict) else {}
    if progress.get("translation_locked") is not True:
        return False, None
    # `translation_locked` also describes a neutral recovery hover.  That
    # state must allow a fresh map/route observation to replace the rejected
    # position; treating it as an active yaw interval freezes every recovered
    # pose at the old anchor and creates a permanent localization deadlock.
    if progress.get("route_visual_recovery_allowed") is True:
        return False, None
    if str(progress.get("phase") or "").strip().lower() == "pose_recovery":
        # Compatibility with a bridge that predates the explicit flag above.
        return False, None
    try:
        forward = abs(float(progress.get("body_forward_gain") or 0.0))
        lateral = abs(float(progress.get("body_lateral_gain") or 0.0))
    except (TypeError, ValueError):
        return False, None
    commanded = forward <= 1e-6 and lateral <= 1e-6
    return commanded, finite_room_vector(progress.get("position_anchor")) if commanded else None


def finite_room_vector(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 3:
        return None
    try:
        result = [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def route_segment_projection_xz(
    point: Any,
    start: Any,
    end: Any,
) -> tuple[float, float] | None:
    """Return unclamped progress and finite-segment cross-track distance."""
    point = finite_room_vector(point)
    start = finite_room_vector(start)
    end = finite_room_vector(end)
    if point is None or start is None or end is None:
        return None
    dx = end[0] - start[0]
    dz = end[2] - start[2]
    length_sq = dx * dx + dz * dz
    if length_sq <= 1e-12:
        return None
    progress = ((point[0] - start[0]) * dx + (point[2] - start[2]) * dz) / length_sq
    clamped = max(0.0, min(1.0, progress))
    nearest_x = start[0] + clamped * dx
    nearest_z = start[2] + clamped * dz
    cross_track = math.hypot(point[0] - nearest_x, point[2] - nearest_z)
    return float(progress), float(cross_track)


class LivePatrolRouteGate:
    """Reject repeated-room aliases before they can update live tracking.

    The flight bridge publishes the exact active segment and whether the
    current command is rotation-only.  This gate validates that segment against
    the selected full-loop baseline, then enforces its corridor and monotonic
    direction.  It is inactive before/after the corresponding live mission.
    """

    def __init__(
        self,
        baseline_path: Path | None,
        status_path: Path | None,
        *,
        max_cross_track: float = 0.55,
        backward_tolerance: float = 0.08,
        turn_max_drift: float = 0.75,
        max_status_age: float = 5.0,
        endpoint_tolerance: float = 0.08,
    ) -> None:
        self.baseline_path = Path(baseline_path) if baseline_path else None
        self.status_path = Path(status_path) if status_path else None
        self.max_cross_track = max(0.10, float(max_cross_track))
        self.backward_tolerance = max(0.0, float(backward_tolerance))
        self.turn_max_drift = max(0.05, float(turn_max_drift))
        self.max_status_age = max(0.2, float(max_status_age))
        self.endpoint_tolerance = max(0.01, float(endpoint_tolerance))
        self.baseline_replay_id = self.baseline_path.parent.name if self.baseline_path else ""
        self.map_id = ""
        self.patrol_id = ""
        self.legs: list[tuple[list[float], list[float]]] = []
        self.last_key: tuple[Any, ...] | None = None
        self.last_progress: float | None = None
        self.last_publish_time: float | None = None
        self.departure_floor_reconciled_key: tuple[Any, ...] | None = None
        self.departure_progress_bias_key: tuple[Any, ...] | None = None
        self.departure_progress_bias = 0.0
        self.accepted_count = 0
        self.rejected_count = 0
        self._load()

    def _load(self) -> None:
        if self.baseline_path is None:
            return
        try:
            payload = json.loads(self.baseline_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if (
            not isinstance(payload, dict)
            or payload.get("complete_loop") is not True
            or payload.get("enabled_for_live_route_gate") is not True
        ):
            return
        legs: list[tuple[list[float], list[float]]] = []
        for leg in payload.get("legs", []):
            if not isinstance(leg, dict):
                continue
            start = finite_room_vector(leg.get("from"))
            end = finite_room_vector(leg.get("to"))
            if start is not None and end is not None:
                legs.append((start, end))
        if len(legs) < 4:
            return
        self.map_id = str(payload.get("map_id") or "")
        self.patrol_id = str(payload.get("patrol_id") or "")
        self.legs = legs

    @property
    def enabled(self) -> bool:
        return bool(self.legs and self.status_path is not None)

    @staticmethod
    def _distance(a: list[float], b: list[float]) -> float:
        return math.hypot(a[0] - b[0], a[2] - b[2])

    def _matches_baseline_leg(self, start: list[float], end: list[float]) -> bool:
        return any(
            self._distance(start, reference_start) <= self.endpoint_tolerance
            and self._distance(end, reference_end) <= self.endpoint_tolerance
            for reference_start, reference_end in self.legs
        )

    def _baseline_leg_index(self, start: list[float], end: list[float]) -> int | None:
        for index, (reference_start, reference_end) in enumerate(self.legs, start=1):
            if (
                self._distance(start, reference_start) <= self.endpoint_tolerance
                and self._distance(end, reference_end) <= self.endpoint_tolerance
            ):
                return index
        return None

    def active_context(self) -> dict[str, Any] | None:
        if not self.enabled or self.status_path is None:
            return None
        try:
            status = json.loads(self.status_path.read_text(encoding="utf-8"))
            age = time.time() - float(status.get("updated_at") or 0.0)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if age < -0.25 or age > self.max_status_age:
            return None
        if str(status.get("status") or "").strip().lower() != "running":
            return None
        if str(status.get("command") or "").strip().lower() != "mission":
            return None
        progress = status.get("progress") if isinstance(status.get("progress"), dict) else {}
        if self.map_id and str(progress.get("map_id") or "") != self.map_id:
            return None
        if self.patrol_id and str(progress.get("patrol_id") or "") != self.patrol_id:
            return None
        if (
            self.baseline_replay_id
            and str(progress.get("baseline_replay_id") or "") != self.baseline_replay_id
        ):
            return None
        start = finite_room_vector(progress.get("segment_start"))
        end = finite_room_vector(progress.get("target"))
        if start is None or end is None or not self._matches_baseline_leg(start, end):
            # The one-time entry leg from takeoff position to point 1 is not a
            # baseline loop leg and remains protected by the hard continuity
            # gate. Route monotonicity starts with the recorded circle itself.
            return None
        anchor = finite_room_vector(progress.get("position_anchor")) or start
        leg_index = self._baseline_leg_index(start, end)
        controller_translation_locked = progress.get("translation_locked") is True
        phase = str(progress.get("phase") or "").strip().lower()
        # Visual-route position is allowed to advance only while the bridge
        # has passed its final motion gate and is executing a real horizontal
        # command. `translation_locked=false` alone is insufficient: neutral
        # pre-yaw settling deliberately releases the metric rotation anchor,
        # and the landed 09:54 run then mistook changing yaw images for 1.39 m
        # of Point-3->4 translation.
        physical_translation_active = bool(
            progress.get("physical_translation_active") is True
            or (
                # Recorded live-clock simulations predate the explicit bridge
                # flag but publish an unambiguous translation-only phase.
                progress.get("physical_translation_active") is None
                and phase == "patrol_translation"
                and abs(float(progress.get("body_forward_gain") or 0.0)) > 1e-6
            )
        )
        recovery_hover = bool(
            progress.get("route_visual_recovery_allowed") is True
            or phase == "pose_recovery"
        )
        metric_position_recovery = bool(
            recovery_hover
            and progress.get("metric_position_recovery_allowed") is True
            and progress.get("require_metric_pose") is True
        )
        lap_start_metric_rebootstrap = bool(
            metric_position_recovery
            and progress.get("lap_start_metric_rebootstrap") is True
        )
        stopped_metric_rebootstrap = bool(
            metric_position_recovery
            and phase == "pose_recovery"
            and progress.get("stopped_metric_rebootstrap") is True
        )
        post_translation_progress_recovery = bool(
            recovery_hover
            and phase == "pose_recovery"
            and progress.get("post_translation_progress_recovery") is True
        )
        endpoint_position_recovery = bool(
            recovery_hover
            and phase == "pose_recovery"
            and progress.get("endpoint_position_recovery_allowed") is True
        )
        endpoint_overshoot_correction = bool(
            progress.get("endpoint_overshoot_correction") is True
        )
        try:
            command_progress_ceiling = float(
                progress.get("route_progress_command_ceiling")
            )
        except (TypeError, ValueError):
            command_progress_ceiling = None
        if command_progress_ceiling is not None:
            if not math.isfinite(command_progress_ceiling):
                command_progress_ceiling = None
            else:
                command_progress_ceiling = max(
                    0.0,
                    min(1.0, command_progress_ceiling),
                )
        try:
            route_pose_epoch = int(progress.get("route_pose_epoch") or 0)
            route_pose_epoch_unix = float(progress.get("route_pose_epoch_unix"))
        except (TypeError, ValueError):
            route_pose_epoch = 0
            route_pose_epoch_unix = float("nan")
        route_pose_epoch_reason = str(
            progress.get("route_pose_epoch_reason") or ""
        )
        verified_route_pose_epoch_reasons = {
            "verified_point4_handoff",
            "verified_point1_handoff",
            "lap_start_global_relocalization",
            "stopped_global_relocalization",
        }
        if (
            route_pose_epoch <= 0
            or not math.isfinite(route_pose_epoch_unix)
            or route_pose_epoch_reason not in verified_route_pose_epoch_reasons
        ):
            route_pose_epoch = 0
            route_pose_epoch_unix = None
            route_pose_epoch_reason = ""
        return {
            "start": start,
            "end": end,
            "anchor": anchor,
            # Visual route matching remains disabled during a real in-place
            # yaw, but it must run while the controller is neutrally hovering
            # specifically to regain localization.
            "translation_locked": bool(
                (controller_translation_locked or not physical_translation_active)
                and not recovery_hover
            ),
            # Keep metric route validation anchored whenever the controller is
            # physically forbidding translation.  Recovery hover deliberately
            # enables the visual matcher above, but the aircraft still did not
            # move: evaluating TSolve's post-yaw raw center as real translation
            # creates a circular wait before the rotation stabilizer can remove
            # that drift.  Visual matching and metric position guarding are
            # independent decisions.
            # During an explicit metric checkpoint the aircraft is receiving
            # neutral RC, not yaw.  Let a fresh 2D->3D solution expose the real
            # room position (including a small cross-track offset) instead of
            # projecting it back to the old route anchor. Ordinary yaw/recovery
            # hover keeps the conservative position guard.
            "position_guard_locked": bool(
                controller_translation_locked
                and not metric_position_recovery
                and not post_translation_progress_recovery
                and not endpoint_position_recovery
            ),
            "controller_translation_locked": controller_translation_locked,
            "physical_translation_active": physical_translation_active,
            "route_progress_command_ceiling": command_progress_ceiling,
            "route_progress_command_sequence": progress.get(
                "route_progress_command_sequence"
            ),
            "route_progress_command_budget_m": progress.get(
                "route_progress_command_budget_m"
            ),
            "route_pose_epoch": route_pose_epoch,
            "route_pose_epoch_unix": route_pose_epoch_unix,
            "route_pose_epoch_reason": route_pose_epoch_reason,
            "endpoint_overshoot_correction": endpoint_overshoot_correction,
            "recovery_hover": recovery_hover,
            "metric_position_recovery": metric_position_recovery,
            "lap_start_metric_rebootstrap": lap_start_metric_rebootstrap,
            "stopped_metric_rebootstrap": stopped_metric_rebootstrap,
            "post_translation_progress_recovery": (
                post_translation_progress_recovery
            ),
            "endpoint_position_recovery": endpoint_position_recovery,
            "require_metric_pose": progress.get("require_metric_pose") is True,
            "phase": progress.get("phase"),
            "leg_index": leg_index,
            "lap": progress.get("lap"),
            "step_index": progress.get("step_index"),
            "updated_at": status.get("updated_at"),
        }

    @staticmethod
    def _key(context: dict[str, Any]) -> tuple[Any, ...]:
        start = context["start"]
        end = context["end"]
        # ``step_index`` identifies a temporary mission command, not a patrol
        # leg.  Recovery/hover status updates intentionally omit it, so using
        # it here reset visual progress every time the controller entered or
        # left recovery on the same physical leg.  Keep one monotonic key for
        # the complete leg and change it only for a new lap/leg.
        key = (
            context.get("lap"),
            context.get("leg_index"),
            round(start[0], 4),
            round(start[2], 4),
            round(end[0], 4),
            round(end[2], 4),
        )
        # The controller can verify an endpoint after the following leg already
        # became active. Add a second ownership boundary only when that explicit
        # endpoint epoch exists. Ordinary legs and pre-handoff behavior keep
        # their exact historical key and therefore their established path.
        try:
            epoch = int(context.get("route_pose_epoch") or 0)
        except (TypeError, ValueError):
            epoch = 0
        if epoch > 0:
            return (*key, "route_pose_epoch", epoch)
        return key

    @staticmethod
    def frame_predates_route_pose_epoch(
        context: dict[str, Any] | None,
        received_unix: Any,
    ) -> bool:
        if not isinstance(context, dict):
            return False
        try:
            epoch = int(context.get("route_pose_epoch") or 0)
            cutoff = float(context.get("route_pose_epoch_unix"))
            received = float(received_unix)
        except (TypeError, ValueError):
            return bool(context.get("route_pose_epoch"))
        if epoch <= 0:
            return False
        if not math.isfinite(cutoff) or not math.isfinite(received):
            return True
        return received < cutoff

    @staticmethod
    def _anchor_progress(context: dict[str, Any]) -> float | None:
        """Return the controller-trusted starting floor for a patrol leg."""
        projection = route_segment_projection_xz(
            context.get("anchor"),
            context.get("start"),
            context.get("end"),
        )
        return float(projection[0]) if projection is not None else None

    def rejection(
        self,
        candidate: Any,
        previous: Any,
    ) -> tuple[str | None, dict[str, Any] | None]:
        context = self.active_context()
        candidate = finite_room_vector(candidate)
        previous = finite_room_vector(previous)
        if context is None or candidate is None:
            return None, None
        key = self._key(context)
        guarded_candidate = candidate
        raw_turn_drift = None
        if context["position_guard_locked"]:
            drift = self._distance(candidate, context["anchor"])
            raw_turn_drift = drift
            # The controller has physically forbidden translation and the
            # rotation stabilizer publishes this exact position anchor. Raw
            # monocular centers are not translation measurements during yaw:
            # Live ATLAS 12:08:57 accumulated 0.979 m of false center drift at
            # Point 2 while 489-500 optical-flow tracks consistently measured
            # the commanded rotation. Rejecting that raw center created a
            # circular recovery wait even though the published pose never
            # moved. Keep the raw drift for audit, but validate and publish the
            # controller anchor. Translation remains blocked at the final RC
            # command boundary until the stabilizer has released the turn.
            guarded_candidate = context["anchor"]
        projected = route_segment_projection_xz(guarded_candidate, context["start"], context["end"])
        if projected is None:
            return "route_projection_invalid", {"key": key, "context": context}
        progress, cross_track = projected
        unbiased_progress = float(progress)
        progress_bias = 0.0
        metric_led_leg = int(context.get("leg_index") or 0) in {1, 2}
        if (
            not context["position_guard_locked"]
            and key == self.departure_progress_bias_key
            and not metric_led_leg
        ):
            progress_bias = float(self.departure_progress_bias)
            progress += progress_bias
        if cross_track > self.max_cross_track:
            self.rejected_count += 1
            return (
                f"route_cross_track_{cross_track:.3f}m_gt_{self.max_cross_track:.3f}m",
                {"key": key, "progress": progress, "cross_track": cross_track, "context": context},
            )
        # Once this leg has a committed route observation, that guarded value
        # is the only monotonic floor. On a new leg, bootstrap from the
        # controller's position anchor (normally the shared waypoint), never
        # from ``previous``. The previous localizer pose can be temporarily
        # unguarded while mission status changes at the end of an in-place yaw.
        # Live ATLAS 13:51:24 exposed the failure deterministically: the real
        # Point-2 anchor projected to -0.0046 on leg 2, but five false yaw
        # centers projected to 0.3045. Treating the last false center as the new
        # floor rejected every correct Point-2-to-3 observation forever. The
        # successful 11:57:36 run passed only because its equivalent false
        # center happened to project near zero.
        floor = (
            float(self.last_progress)
            if self.last_key == key and self.last_progress is not None
            else self._anchor_progress(context)
        )
        endpoint_rollback = bool(
            context.get("endpoint_overshoot_correction") is True
            and floor is not None
            and float(floor) >= 0.90
            and 0.90 <= progress <= 1.20
            and progress <= float(floor) + self.backward_tolerance
        )
        if (
            floor is not None
            and progress < floor - self.backward_tolerance
            and not endpoint_rollback
        ):
            self.rejected_count += 1
            return (
                f"route_backward_{progress:.3f}_lt_{floor - self.backward_tolerance:.3f}",
                {"key": key, "progress": progress, "previous_progress": floor, "context": context},
            )
        # A waypoint is the hard end of a commanded leg.  The controller
        # already considers the aircraft arrived inside its endpoint radius;
        # accepting a monocular center more than 8% beyond that point can only
        # poison the next turn.  Live ATLAS 16:33 accepted 1.160 here while
        # the independent endpoint images were at 0.95-1.00, then rejected the
        # correct view for 330 frames as "backwards".
        endpoint_position_recovery = bool(
            context.get("endpoint_position_recovery") is True
        )
        if progress < -0.16 or (
            progress > 1.08
            and not endpoint_rollback
            and not endpoint_position_recovery
        ) or progress > 1.20:
            self.rejected_count += 1
            return (
                f"route_progress_outside_segment_{progress:.3f}",
                {"key": key, "progress": progress, "context": context},
            )
        observation = {
            "key": key,
            "progress": progress,
            "unbiased_progress": unbiased_progress,
            "departure_progress_bias": progress_bias,
            "cross_track": cross_track,
            "context": context,
            "endpoint_overshoot_rollback": endpoint_rollback,
        }
        if raw_turn_drift is not None:
            observation.update(
                {
                    "raw_turn_drift_m": raw_turn_drift,
                    "raw_turn_drift_limit_m": self.turn_max_drift,
                    "raw_turn_drift_anchored": raw_turn_drift > self.turn_max_drift,
                }
            )
        return None, observation

    def commit(self, observation: dict[str, Any] | None) -> None:
        if not isinstance(observation, dict) or observation.get("progress") is None:
            return
        key = observation.get("key")
        progress = float(observation["progress"])
        if key != self.last_key:
            self.last_key = key
            self.last_progress = progress
        elif observation.get("endpoint_overshoot_rollback") is True:
            # A bridge-commanded low-speed reverse is the only patrol state
            # allowed to lower the route clock. It is restricted to the final
            # 10% around the same endpoint and ends when the command phase
            # ends; ordinary localization jumps remain monotonic.
            self.last_progress = progress
        else:
            self.last_progress = progress if self.last_progress is None else max(self.last_progress, progress)
        self.accepted_count += 1

    def constrain_published_pose(
        self,
        pose: dict[str, Any] | None,
        observation: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Publish a monotonic, rate-limited pose on the commanded patrol leg."""
        if not isinstance(pose, dict) or not isinstance(observation, dict):
            return pose
        if pose.get("held_pose") is True or pose.get("output_rejected") is True:
            # A held payload is copied from the last trusted pose. Its route
            # observation belongs to the rejected TSolve candidate, not to the
            # aircraft position being published. Live ATLAS 12:52:19 rejected
            # a 0.754 m jump at Point 2, but this method still advanced the
            # displayed route floor from -0.004 to 0.063. Every later correct
            # anchor was then rejected as backwards. Keep both the model and
            # the monotonic floor unchanged until a real pose is accepted.
            pose["route_rejected_observation_ignored"] = True
            try:
                pose["route_rejected_observation_progress"] = float(
                    observation.get("progress")
                )
            except (TypeError, ValueError):
                pass
            return pose
        context = observation.get("context")
        key = observation.get("key")
        if not isinstance(context, dict) or key is None:
            return pose
        start = finite_room_vector(context.get("start"))
        end = finite_room_vector(context.get("end"))
        raw_center = finite_room_vector(pose.get("rcenter"))
        try:
            raw_progress = float(observation.get("progress"))
        except (TypeError, ValueError):
            return pose
        if start is None or end is None or raw_center is None or not math.isfinite(raw_progress):
            return pose
        segment_length = math.hypot(end[0] - start[0], end[2] - start[2])
        if segment_length <= 1e-9:
            return pose
        observed_metric_progress = raw_progress
        verified_visual_rewind = bool(
            context.get("recovery_hover") is True
            and observation.get("verified_visual_rewind") is True
            and pose.get("route_visual_verified_rewind") is True
            and int(pose.get("route_visual_verified_rewind_hits") or 0) >= 5
            and int(pose.get("route_visual_verified_rewind_inliers") or 0)
            >= 120
        )
        departure_visual_lock = bool(
            int(context.get("leg_index") or 0) == 3
            and context.get("controller_translation_locked") is not True
            and key != self.departure_floor_reconciled_key
        )
        if departure_visual_lock:
            # The weak repeated-wall 3->4 leg can yield a stable but wrong TSolve
            # root immediately after yaw.  Do not render that root or let it
            # become the monotonic floor before two current images verify the
            # departure.  The flight bridge treats the held anchor as no
            # progress and therefore hovers instead of sending another pulse.
            # Once route vision verifies the departure it calibrates the raw
            # metric offset and releases this lock for continuous TSolve.
            anchor_projection = route_segment_projection_xz(
                context.get("anchor") or start,
                start,
                end,
            )
            raw_progress = (
                float(anchor_projection[0])
                if anchor_projection is not None
                else 0.0
            )
        try:
            current_time = float(pose.get("time_sec"))
        except (TypeError, ValueError):
            current_time = None

        if key != self.last_key or self.last_progress is None:
            anchor_projection = route_segment_projection_xz(
                context.get("anchor") or start,
                start,
                end,
            )
            published_progress = (
                float(anchor_projection[0]) if anchor_projection is not None else raw_progress
            )
        else:
            published_progress = float(self.last_progress)
            dt = (
                max(0.0, current_time - self.last_publish_time)
                if current_time is not None and self.last_publish_time is not None
                else 0.10
            )
            # The aircraft travels at about 0.10 m/s. A 0.30 m/s catch-up
            # ceiling avoids sustained display lag, while the 10 cm absolute
            # ceiling prevents the 20-35 cm jumps seen in Live ATLAS 11:01:14.
            max_publish_step_m = min(0.10, 0.02 + 0.30 * dt)
            if (
                observation.get("endpoint_overshoot_rollback") is True
                or verified_visual_rewind
            ):
                requested_progress = raw_progress
                published_progress = max(
                    requested_progress,
                    published_progress - max_publish_step_m / segment_length,
                )
            else:
                requested_progress = max(published_progress, raw_progress)
                published_progress = min(
                    requested_progress,
                    published_progress + max_publish_step_m / segment_length,
                )

        # Normal patrol publication stops at the waypoint. During a neutral
        # endpoint-recovery hover, however, clipping an independently solved
        # overshoot to exactly 1.0 makes the controller believe it is at the
        # waypoint forever and prevents its existing bounded reverse logic
        # from engaging. Expose at most the already guarded 1.20 route limit
        # only for that recovery phase; no translation command is active.
        published_progress_limit = (
            1.20
            if context.get("endpoint_position_recovery") is True
            else 1.0
        )
        published_progress = max(
            -0.08,
            min(published_progress_limit, published_progress),
        )
        constrained = [
            start[index] + published_progress * (end[index] - start[index])
            for index in range(3)
        ]
        pose["route_raw_rcenter"] = raw_center
        pose["rcenter"] = constrained
        pose["route_position_constrained"] = True
        pose["route_raw_progress"] = raw_progress
        pose["route_published_progress"] = published_progress
        pose["route_cross_track_m"] = observation.get("cross_track")
        if verified_visual_rewind:
            pose["route_verified_visual_rewind_applied"] = True
        if departure_visual_lock:
            pose["route_departure_visual_lock"] = True
            pose["route_departure_unverified_metric_progress"] = observed_metric_progress
        if observation.get("raw_turn_drift_m") is not None:
            pose["route_raw_turn_drift_m"] = observation.get("raw_turn_drift_m")
            pose["route_raw_turn_drift_limit_m"] = observation.get(
                "raw_turn_drift_limit_m"
            )
            pose["route_raw_turn_drift_anchored"] = bool(
                observation.get("raw_turn_drift_anchored")
            )
        self.last_key = key
        self.last_progress = published_progress
        self.last_publish_time = current_time
        self.accepted_count += 1
        return pose

    def reconcile_verified_endpoint_floor(
        self,
        observation: dict[str, Any] | None,
        context: dict[str, Any] | None,
    ) -> bool:
        """Repair an older overrun floor from independent endpoint consensus."""
        if (
            not isinstance(observation, dict)
            or not isinstance(context, dict)
            or observation.get("endpoint_verified") is not True
        ):
            return False
        try:
            progress = float(observation.get("progress"))
        except (TypeError, ValueError):
            return False
        key = self._key(context)
        if (
            not math.isfinite(progress)
            or progress < 0.90
            or self.last_key != key
            or self.last_progress is None
            or float(self.last_progress) <= 1.0
        ):
            return False
        self.last_progress = 1.0
        return True

    def reconcile_verified_departure_floor(
        self,
        observation: dict[str, Any] | None,
        context: dict[str, Any] | None,
        metric_route_observation: dict[str, Any] | None = None,
    ) -> bool:
        """Repair only a just-started leg from a verified departure image.

        A repeated-room TSolve root can project several centimetres ahead on
        the first translation frame.  The monotonic route gate then preserves
        that false floor and rejects the correct departure view forever.  A
        strong two-frame visual observation may lower the floor only while
        both sources are still inside the first 8% of the same unlocked leg;
        this cannot rewind an established cruise track.
        """
        if (
            not isinstance(observation, dict)
            or not isinstance(context, dict)
            or observation.get("verified") is not True
            or observation.get("translation_safe") is False
            or context.get("controller_translation_locked") is True
            or int(context.get("leg_index") or 0) in {1, 2}
        ):
            return False
        key = self._key(context)
        if (
            key != self.last_key
            or self.last_progress is None
            or key == self.departure_floor_reconciled_key
        ):
            return False
        try:
            visual_progress = float(
                observation.get("matched_progress", observation.get("progress"))
            )
            current_progress = float(self.last_progress)
            inliers = int(observation.get("inliers") or 0)
            minimum_inliers = int(observation.get("minimum_inliers") or 120)
            acquisition_hits = int(observation.get("acquisition_hits") or 0)
        except (TypeError, ValueError):
            return False
        leg_index = int(context.get("leg_index") or 0)
        # Leg 4's first two strong visual hits straddle the first short
        # physical pulse: the second audited match is normally at 5.2%.
        # Keeping the generic 4% entry window meant the key-change frame only
        # initialized the route gate and the following hit arrived just
        # outside the window. The departure lock then remained set for the
        # entire 4->1 leg, freezing Point 4 until a final snap to Point 1.
        # This wider allowance is limited to the one-shot, two-hit, >=120
        # inlier repair at the start of leg 4; established progress still
        # cannot be rewound.
        visual_entry_limit = 0.12 if leg_index == 4 else 0.04
        current_entry_limit = 0.15 if leg_index == 4 else 0.08
        if (
            not math.isfinite(visual_progress)
            or visual_progress < -0.02
            or visual_progress > visual_entry_limit
            or current_progress < -0.02
            or current_progress > current_entry_limit
            or inliers < max(120, minimum_inliers)
            or acquisition_hits < 2
        ):
            return False
        try:
            metric_progress = float(
                metric_route_observation.get(
                    "unbiased_progress",
                    metric_route_observation.get("progress"),
                )
            )
        except (AttributeError, TypeError, ValueError):
            metric_progress = current_progress
        if not math.isfinite(metric_progress):
            metric_progress = current_progress
        self.last_progress = max(0.0, visual_progress)
        # This is a leg-entry state repair, not a general backwards-motion
        # permission. Once used, the shared monotonic route clock owns every
        # later observation until the lap/leg key changes.
        self.departure_floor_reconciled_key = key
        self.departure_progress_bias_key = key
        self.departure_progress_bias = visual_progress - metric_progress
        return True


def published_visual_route_observation(
    pose: dict[str, Any] | None,
    context: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Build route truth from the position actually published to the model.

    A visual matcher may identify a farther target view while
    ``visual_route_pose_from_last`` is still reconciling toward it over small
    position steps. Committing the target progress made the monotonic gate run
    ahead of the rendered/controller position in Live ATLAS 10:37:20. The
    following weak frame then looked backwards and froze Point 2→3 forever.
    """
    if not isinstance(pose, dict) or not isinstance(context, dict):
        return None
    center = finite_room_vector(pose.get("rcenter"))
    start = finite_room_vector(context.get("start"))
    end = finite_room_vector(context.get("end"))
    if center is None or start is None or end is None:
        return None
    projected = route_segment_projection_xz(center, start, end)
    if projected is None:
        return None
    progress, cross_track = projected
    key = LivePatrolRouteGate._key(context)
    try:
        target_progress = float(pose.get("route_visual_progress"))
    except (TypeError, ValueError):
        target_progress = progress
    pose["route_visual_published_progress"] = float(progress)
    pose["route_visual_progress_lag"] = max(0.0, target_progress - float(progress))
    observation = {
        "key": key,
        "progress": float(progress),
        "cross_track": float(cross_track),
        "context": context,
    }
    if pose.get("route_visual_verified_rewind") is True:
        observation["verified_visual_rewind"] = True
    return observation


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
    output_center_bias: np.ndarray | None = None,
) -> dict[str, Any]:
    meta = read_case_meta(Path(case["case_dir"])) if case.get("case_dir") else {}
    R = result.get("R")
    t = result.get("t")
    if isinstance(t, list) and len(t) == 1 and isinstance(t[0], list):
        t = t[0]
    center = camera_center_from_rt(R, t)
    success = bool(result.get("success")) and output_rejection_reason is None
    uncalibrated_center = center
    applied_bias = None
    if success and center is not None and output_center_bias is not None:
        bias = np.asarray(output_center_bias, dtype=float).reshape(3)
        corrected_center = np.asarray(center, dtype=float).reshape(3) + bias
        if np.all(np.isfinite(corrected_center)):
            center = vector_list(corrected_center)
            applied_bias = vector_list(bias)
            # Keep the published R/t/center tuple internally consistent. The
            # solver result itself remains untouched for tracking and gates.
            try:
                rotation = np.asarray(R, dtype=float).reshape(3, 3)
                t = vector_list(-rotation @ corrected_center)
            except (TypeError, ValueError):
                pass
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
        "uncalibrated_center": uncalibrated_center if success and applied_bias is not None else None,
        "output_center_bias": applied_bias,
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
    if not last_pose or not (last_pose.get("center") or last_pose.get("rcenter")):
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


def latest_published_pose(
    poses: object,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return the newest captured-frame pose, not the last completed worker.

    A background COLMAP worker can complete late and append its older frame
    after hundreds of newer live poses.  That recovery result may seed future
    tracking, but it must never become the display/control position inherited
    by a held pose for the current camera frame.
    """
    candidates = [
        pose for pose in poses
        if isinstance(pose, dict)
    ] if isinstance(poses, list) else []
    if not candidates:
        return fallback

    def order_key(pose: dict[str, Any]) -> tuple:
        image_name = str(pose.get("image_name") or "")
        try:
            frame_index = int(Path(image_name).stem.rsplit("_", 1)[-1])
        except (TypeError, ValueError):
            frame_index = -1

        def finite_number(value: object) -> tuple[int, float]:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return (0, float("-inf"))
            return (1, number) if math.isfinite(number) else (0, float("-inf"))

        received_valid, received_unix = finite_number(pose.get("received_unix"))
        time_valid, time_sec = finite_number(pose.get("time_sec"))
        return (
            1 if frame_index >= 0 else 0,
            frame_index,
            received_valid,
            received_unix,
            time_valid,
            time_sec,
        )

    return max(candidates, key=order_key)


def visual_route_pose_from_last(
    *,
    last_pose: dict[str, Any] | None,
    current_frame: dict[str, Any],
    observation: dict[str, Any],
    rotation_heading: list[float] | None = None,
    rotation_heading_tracks: int = 0,
    max_position_step: float = 0.18,
) -> dict[str, Any] | None:
    """Publish a verified, route-only visual observation without faking R/t.

    The source is explicitly distinguishable from a TSolve pose.  It carries
    only the room position and heading needed by the guarded patrol bridge;
    map-space center/R/t stay unset because the frame was not globally solved.
    """
    if not last_pose or observation.get("verified") is not True:
        return None
    room_center = finite_room_vector(observation.get("center"))
    recorded_heading = normalize_room_heading(observation.get("heading"))
    previous_heading = normalize_room_heading(last_pose.get("rheading"))
    optical_heading = (
        normalize_room_heading(rotation_heading)
        if int(rotation_heading_tracks) >= 16
        else None
    )
    # Route-image recovery is an absolute *position/progress* observation.
    # It must not silently replace the independently tracked live orientation.
    # The 2026-08-23 4->1 flight proved why: the physical camera completed the
    # turn, then this fallback moved the model along the leg while stamping the
    # prerecorded leg heading over the latest live/ORB heading.  Preserve the
    # already-published fused heading first; use current optical yaw only when
    # no live heading exists, and retain the bank heading as an explicit last
    # resort rather than presenting it as a new live measurement.
    if previous_heading is not None:
        heading = previous_heading
        heading_source = str(
            last_pose.get("rheading_source") or "previous_live_pose_heading"
        )
    elif optical_heading is not None:
        heading = optical_heading
        heading_source = "optical_flow_yaw"
    else:
        heading = recorded_heading
        heading_source = "recorded_patrol_leg_heading_fallback"
    if room_center is None or heading is None:
        return None
    target_room_center = list(room_center)
    published_room_center = list(room_center)
    previous_room_center = finite_room_vector(last_pose.get("rcenter"))
    remaining_horizontal = 0.0
    if previous_room_center is not None:
        dx = room_center[0] - previous_room_center[0]
        dz = room_center[2] - previous_room_center[2]
        horizontal = math.hypot(dx, dz)
        bounded_step = max(0.05, min(0.22, float(max_position_step)))
        previous_received = last_pose.get("received_unix")
        current_received = current_frame.get("received_unix")
        try:
            elapsed = max(
                0.0,
                min(0.625, float(current_received) - float(previous_received)),
            )
        except (TypeError, ValueError):
            elapsed = None
        if elapsed is not None:
            # The live model must not reconcile faster than an indoor patrol
            # can plausibly move.  The small floor keeps a hovering recovery
            # converging, while the time term preserves smooth 10-FPS motion
            # and the ceiling prevents a delayed solver result from snapping.
            bounded_step = min(bounded_step, max(0.02, 0.02 + 0.16 * elapsed))
        if horizontal > bounded_step:
            ratio = bounded_step / horizontal
            published_room_center = [
                previous_room_center[index]
                + (room_center[index] - previous_room_center[index]) * ratio
                for index in range(3)
            ]
            remaining_horizontal = horizontal - bounded_step
    reconciling = remaining_horizontal > 0.025
    translation_safe = observation.get("translation_safe") is not False
    frame_index = current_frame.get("frame_index")
    pose = dict(last_pose)
    # A visual route observation is a new absolute room-position publication,
    # not a continuation of the previous metric/turn stabilizer payload. Do
    # not carry an older waypoint's rotation-release metadata into a later
    # turn; Live ATLAS 11:57:36 reached Point 4, then inherited Point 3 here.
    for stale_key in (
        "rotation_position_anchor",
        "rotation_position_source",
        "rotation_reanchor_pending",
        "rotation_reanchor_rejected",
        "rotation_reanchored_after_turn",
        "rotation_anchor_is_position_truth",
        "rotation_anchor_commanded",
        "rotation_release_correction",
        "rotation_position_bias",
        "route_raw_rcenter",
        "route_raw_progress",
        "route_published_progress",
        "route_position_constrained",
        "route_cross_track_m",
        "route_departure_visual_lock",
        "route_departure_unverified_metric_progress",
        "route_visual_departure_floor_reconciled",
    ):
        pose.pop(stale_key, None)
    pose.update(
        {
            "instance_id": (
                f"visual_route_{int(frame_index):06d}"
                if frame_index is not None
                else "visual_route"
            ),
            "success": True,
            "time_sec": current_frame.get("time_sec"),
            "received_unix": current_frame.get("received_unix"),
            "image_name": current_frame.get("image_name"),
            "R": None,
            "t": None,
            "center": None,
            "uncalibrated_center": None,
            "output_center_bias": None,
            "rcenter": published_room_center,
            "rheading": heading,
            "rheading_source": heading_source,
            "route_visual_recorded_heading": recorded_heading,
            "route_visual_heading_preserved": previous_heading is not None,
            "held_pose": False,
            "output_rejected": False,
            "rejected_reason": None,
            "translation_allowed": bool(not reconciling and translation_safe),
            "rotation_position_locked": False,
            "rotation_raw_rcenter": target_room_center,
            "pose_source": "patrol_visual_route_recovery",
            "route_visual_verified": True,
            "route_visual_progress": float(observation["progress"]),
            "route_visual_matched_progress": observation.get("matched_progress"),
            "route_visual_inliers": int(observation["inliers"]),
            "route_visual_ratio_matches": int(observation["ratio_matches"]),
            "route_visual_anchor": observation.get("anchor_name"),
            "route_visual_source_frame": observation.get("source_frame"),
            "route_visual_acquisition_hits": int(observation["acquisition_hits"]),
            "route_visual_minimum_inliers": int(observation["minimum_inliers"]),
            "route_visual_map_id": observation.get("map_id"),
            "route_visual_patrol_id": observation.get("patrol_id"),
            "route_visual_baseline_replay_id": observation.get("baseline_replay_id"),
            "route_visual_target_center": target_room_center,
            "route_visual_reconciling": reconciling,
            "route_visual_reconciliation_remaining_m": remaining_horizontal,
            "route_visual_translation_safe": translation_safe,
            "route_visual_command_progress_ceiling": observation.get(
                "command_progress_ceiling"
            ),
            "route_visual_command_progress_guarded": bool(
                observation.get("command_progress_guarded")
            ),
            "route_visual_unbounded_progress": observation.get(
                "unbounded_progress"
            ),
            "route_visual_weak_endpoint_recovery": bool(
                observation.get("weak_endpoint_recovery")
            ),
            "route_visual_temporal_recovery": bool(
                observation.get("temporal_recovery")
            ),
            "route_visual_temporal_recovery_hits": int(
                observation.get("temporal_recovery_hits") or 0
            ),
            "route_visual_temporal_recovery_required_hits": int(
                observation.get("temporal_recovery_required_hits") or 0
            ),
            "route_visual_endpoint_guarded": bool(
                observation.get("endpoint_guarded")
            ),
            "route_visual_endpoint_guard_progress": observation.get(
                "endpoint_guard_progress"
            ),
            "route_visual_endpoint_safe_prearrival_progress": observation.get(
                "endpoint_safe_prearrival_progress"
            ),
            "route_visual_endpoint_checked": bool(
                observation.get("endpoint_checked")
            ),
            "route_visual_endpoint_verified": bool(
                observation.get("endpoint_verified")
            ),
            "route_visual_endpoint_match_consensus_verified": bool(
                observation.get("endpoint_match_consensus_verified")
            ),
            "route_visual_endpoint_view_geometry_verified": bool(
                observation.get("endpoint_view_geometry_verified")
            ),
            "route_visual_endpoint_view_scale_min": observation.get(
                "endpoint_view_scale_min"
            ),
            "route_visual_endpoint_view_scale_max": observation.get(
                "endpoint_view_scale_max"
            ),
            "route_visual_endpoint_hits": int(
                observation.get("endpoint_hits") or 0
            ),
            "route_visual_endpoint_required_hits": int(
                observation.get("endpoint_required_hits") or 0
            ),
            "route_visual_endpoint_minimum_inliers": int(
                observation.get("endpoint_minimum_inliers") or 0
            ),
            "route_visual_endpoint_candidate_progress": observation.get(
                "endpoint_candidate_progress"
            ),
            "route_visual_endpoint_best_progress": observation.get(
                "endpoint_best_progress"
            ),
            "route_visual_endpoint_best_inliers": int(
                observation.get("endpoint_best_inliers") or 0
            ),
            "route_visual_endpoint_best_anchor": observation.get(
                "endpoint_best_anchor"
            ),
            "route_visual_verified_rewind": bool(
                observation.get("verified_rewind")
            ),
            "route_visual_verified_rewind_progress": observation.get(
                "verified_rewind_progress"
            ),
            "route_visual_verified_rewind_inliers": int(
                observation.get("verified_rewind_inliers") or 0
            ),
            "route_visual_verified_rewind_hits": int(
                observation.get("verified_rewind_hits") or 0
            ),
            "route_visual_verified_rewind_required_hits": int(
                observation.get("verified_rewind_required_hits") or 0
            ),
            "total_ms": 0.0,
            "stages_ms": {},
            "colmap_reference": None,
        }
    )
    optical_heading = normalize_room_heading(rotation_heading)
    if optical_heading is not None and int(rotation_heading_tracks) >= 16:
        # Position/progress comes from the audited route matcher; the current
        # camera direction comes from consecutive live frames.  Keep both in
        # the published payload so the controller does not steer from an old
        # baseline frame while the physical drone has already rotated.
        pose.update(
            {
                "rotation_heading": optical_heading,
                "rotation_heading_source": "optical_flow_yaw",
                "rotation_heading_tracks": int(rotation_heading_tracks),
            }
        )
    return pose


def visual_route_position_authority_allowed(
    route_context: dict[str, Any] | None,
) -> bool:
    """Keep Point 1->2->3 position metric-led.

    Recorded route images may still validate departure heading and may rebuild
    a 2D-to-3D correspondence pool.  On legs 1 and 2, however, only a pose
    produced by TSolve is allowed to update room position or route progress.
    The weak tail retains its existing guarded visual recovery behavior.
    """
    return not metric_tsolve_position_authority_required(route_context)


def metric_tsolve_position_authority_required(
    route_context: dict[str, Any] | None,
) -> bool:
    """Return true for the established metric-led Point 1->2->3 sector."""
    if not isinstance(route_context, dict):
        return False
    return int(route_context.get("leg_index") or 0) in {1, 2}


def accepted_visual_route_recovery_pose(
    *,
    last_pose: dict[str, Any] | None,
    current_frame: dict[str, Any],
    observation: dict[str, Any],
    supervision: dict[str, Any],
    route_context: dict[str, Any],
    rotation_heading: list[float] | None,
    rotation_heading_tracks: int,
    rotation_position_stabilizer: RotationOnlyPositionStabilizer,
    route_gate: LivePatrolRouteGate,
    visual_recovery: PatrolVisualRouteRecovery,
    metric_route_observation: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Promote verified route vision where visual position authority is allowed.

    A background COLMAP attempt can return a pool that is too weak to form a
    stable TSolve case. The old frame loop then skipped the earlier ``pool is
    None`` fallback and copied another held pose, even when route vision had a
    fresh 190-inlier observation. Build and commit the same absolute visual
    pose from either failure path on the weak tail only. Point 1->2->3 holds
    the last metric pose until TSolve recovers.
    """
    if not visual_route_position_authority_allowed(route_context):
        return None
    reconcile_departure = getattr(
        route_gate,
        "reconcile_verified_departure_floor",
        None,
    )
    departure_floor_reconciled = bool(
        callable(reconcile_departure)
        and reconcile_departure(
            observation,
            route_context,
            metric_route_observation,
        )
    )
    effective_observation = dict(observation)
    if departure_floor_reconciled:
        try:
            departure_progress = float(observation.get("matched_progress"))
        except (TypeError, ValueError):
            departure_progress = float(observation.get("progress") or 0.0)
        start = finite_room_vector(route_context.get("start"))
        end = finite_room_vector(route_context.get("end"))
        if start is not None and end is not None:
            departure_limit = (
                0.12 if int(route_context.get("leg_index") or 0) == 4 else 0.04
            )
            departure_progress = max(
                0.0,
                min(departure_limit, departure_progress),
            )
            effective_observation["progress"] = departure_progress
            effective_observation["center"] = [
                start[index] + departure_progress * (end[index] - start[index])
                for index in range(3)
            ]
    try:
        command_progress_ceiling = float(
            route_context.get("route_progress_command_ceiling")
        )
        effective_progress = float(effective_observation.get("progress"))
    except (TypeError, ValueError):
        command_progress_ceiling = None
        effective_progress = None
    if (
        command_progress_ceiling is not None
        and effective_progress is not None
        and math.isfinite(command_progress_ceiling)
        and math.isfinite(effective_progress)
        and effective_progress > command_progress_ceiling
    ):
        start = finite_room_vector(route_context.get("start"))
        end = finite_room_vector(route_context.get("end"))
        if start is None or end is None:
            return None
        bounded_progress = max(0.0, min(1.0, command_progress_ceiling))
        effective_observation["unbounded_progress"] = effective_progress
        effective_observation["progress"] = bounded_progress
        effective_observation["center"] = [
            start[index] + bounded_progress * (end[index] - start[index])
            for index in range(3)
        ]
        effective_observation["command_progress_ceiling"] = bounded_progress
        effective_observation["command_progress_guarded"] = True
    visual_pose = visual_route_pose_from_last(
        last_pose=last_pose,
        current_frame=current_frame,
        observation=effective_observation,
        rotation_heading=rotation_heading,
        rotation_heading_tracks=rotation_heading_tracks,
    )
    if visual_pose is None:
        return None
    reconcile_endpoint = getattr(route_gate, "reconcile_verified_endpoint_floor", None)
    endpoint_floor_reconciled = bool(
        callable(reconcile_endpoint)
        and reconcile_endpoint(effective_observation, route_context)
    )
    rotation_position_stabilizer.accept_absolute_position()
    visual_pose.update(supervision)
    if endpoint_floor_reconciled:
        visual_pose["route_visual_endpoint_overrun_reconciled"] = True
    if departure_floor_reconciled:
        visual_pose["route_visual_departure_floor_reconciled"] = True
    visual_pose = apply_visual_route_heading_alignment(visual_pose, supervision)
    published_observation = published_visual_route_observation(
        visual_pose,
        route_context,
    )
    # Route vision can replace a TSolve pose that is slightly ahead of the
    # current matched frame.  Use the shared monotonic/rate publication gate
    # so accepting visual authority cannot create a backwards jump or snap.
    constrain = getattr(route_gate, "constrain_published_pose", None)
    if callable(constrain):
        visual_pose = constrain(visual_pose, published_observation)
    else:
        # Lightweight replay/test gates predating monotonic publication still
        # expose commit(). Production LivePatrolRouteGate always constrains.
        route_gate.commit(published_observation)
    published_observation = published_visual_route_observation(
        visual_pose,
        route_context,
    )
    if published_observation is not None:
        visual_recovery.commit_published_progress(published_observation["progress"])
    return visual_pose


def weak_patrol_leg_visual_primary_mode(
    *,
    route_context: dict[str, Any] | None,
    observation: dict[str, Any] | None,
    last_pose: dict[str, Any] | None,
) -> str | None:
    """Keep live metric tracking primary on every patrol leg.

    Recorded route frames remain useful as current-image supervision and as a
    bounded recovery source after a metric rejection.  They must not bypass
    TSolve during 3->4: doing so clears the live 2D->3D pool during the Point-3
    yaw and leaves no metric anchor for the subsequent translation or 4->1
    turn.  Returning no primary mode deliberately keeps optical flow and
    TSolve active while the rotation-position stabilizer independently locks
    published translation during yaw.
    """
    return None


def patrol_reference_frames_enabled(
    route_context: dict[str, Any] | None,
) -> bool:
    """Return whether the complete recorded loop covers the active route leg."""
    if not isinstance(route_context, dict):
        return False
    return int(route_context.get("leg_index") or 0) in {1, 2, 3, 4}


def visual_route_position_recovery_needed(
    route_context: dict[str, Any] | None,
    *,
    force_route_taught_recovery: bool = False,
) -> bool:
    """Run the full route-position matcher only when it can affect position.

    A controller-locked yaw cannot translate the aircraft. Its small
    departure-heading matcher is handled separately, so running the full-loop
    position matcher too only spends the live frame budget twice. The weak
    3->4 translation keeps continuous route supervision, and every leg may
    still invoke position matching during neutral recovery hover.
    """
    if not patrol_reference_frames_enabled(route_context):
        return False
    assert isinstance(route_context, dict)
    if route_context.get("recovery_hover") is True or force_route_taught_recovery:
        return True
    return bool(
        int(route_context.get("leg_index") or 0) == 3
        and route_context.get("controller_translation_locked") is not True
    )


def visual_route_temporal_recovery_minimum_inliers(
    configured_minimum_inliers: int,
    *,
    leg_index: int,
) -> int:
    """Return the audited neutral-hover recovery floor for one patrol leg.

    Points 1, 2 and 3 are the endpoints of legs 4, 1 and 2 respectively. The
    operator-selected recovery floor there is 50 ORB/homography inliers. Point
    4 (leg 3) keeps the established 90-inlier floor. This lower value is never
    a normal-flight translation gate: recovery still requires five strong
    frames inside a short bounded window, endpoint geometry, and the
    accumulated physical-command distance cap.
    """
    configured = max(16, int(configured_minimum_inliers))
    if int(leg_index) in {1, 2, 4}:
        return 50
    return max(90, configured)


def visual_recovery_supersedes_stalled_metric_pose(
    *,
    last_pose: dict[str, Any] | None,
    observation: dict[str, Any] | None,
    route_context: dict[str, Any] | None,
    output_rejection_reason: str | None,
    metric_route_observation: dict[str, Any] | None = None,
    departure_floor_repair_available: bool = True,
    minimum_progress_gain: float = 0.00075,
) -> bool:
    """Use verified full-loop vision only to recover a stalled metric pose.

    The recorded patrol matcher is the only position source audited over the
    complete weak legs, but its anchors are intentionally sparse. Continuous
    TSolve motion is the position authority whenever a current metric pose is
    accepted. Route vision may repair a rejected/missing metric result or a
    neutral recovery hover; disagreement alone must not replace a valid
    metric pose with prerecorded path progress.

    A real commanded yaw is different: physical translation is locked and the
    position must remain at the controller's turn anchor.  Visual route poses
    are never promoted in that state.  Recovery hover explicitly permits them
    because the command is neutral, not yaw.
    """
    if (
        not isinstance(last_pose, dict)
        or not isinstance(observation, dict)
        or not isinstance(route_context, dict)
        or observation.get("verified") is not True
    ):
        return False
    if not visual_route_position_authority_allowed(route_context):
        # Legs 1 and 2 are the already-established TSolve/optical-flow path.
        # Route frames can supervise yaw or reseed metric correspondences, but
        # they must never become the published position authority here.
        return False
    controller_locked = route_context.get("controller_translation_locked") is True
    recovery_hover = route_context.get("recovery_hover") is True
    if controller_locked and not recovery_hover:
        # Never change room position while the flight controller is issuing a
        # yaw-only command.  The rotation stabilizer owns position here.
        return False
    weak_endpoint_recovery = bool(observation.get("weak_endpoint_recovery"))
    temporal_recovery = bool(observation.get("temporal_recovery"))
    translation_safe = observation.get("translation_safe") is not False
    try:
        visual_progress = float(observation.get("progress"))
        inliers = int(observation.get("inliers") or 0)
        minimum_inliers = int(observation.get("minimum_inliers") or 120)
        acquisition_hits = int(observation.get("acquisition_hits") or 0)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(visual_progress):
        return False
    if translation_safe:
        required_inliers = (
            visual_route_temporal_recovery_minimum_inliers(
                minimum_inliers,
                leg_index=int(route_context.get("leg_index") or 0),
            )
            if temporal_recovery
            else max(120, minimum_inliers)
        )
        required_hits = (
            max(
                5,
                int(observation.get("temporal_recovery_required_hits") or 0),
            )
            if temporal_recovery
            else 2
        )
        if inliers < required_inliers or acquisition_hits < required_hits:
            return False
    elif not (
        recovery_hover
        and controller_locked
        and weak_endpoint_recovery
        and visual_progress >= 0.90
        and inliers >= 60
    ):
        # recover() emits weak endpoint observations only after five
        # consecutive endpoint-supported frames. They may update the model
        # while the controller remains in neutral hover, but their
        # translation_safe=false flag continues to forbid another RC pulse.
        return False
    start = finite_room_vector(route_context.get("start"))
    end = finite_room_vector(route_context.get("end"))
    center = finite_room_vector(last_pose.get("rcenter"))
    if start is None or end is None or center is None:
        return False
    projected = route_segment_projection_xz(center, start, end)
    if projected is None:
        return False
    current_progress = float(projected[0])
    if (
        departure_floor_repair_available
        and not controller_locked
        and visual_progress <= 0.04
        and -0.02 <= current_progress <= 0.08
    ):
        # A verified two-frame departure image may repair a small false
        # forward floor before it becomes established route history. The
        # actual floor reset is independently bounded in
        # reconcile_verified_departure_floor().
        return True
    if (
        recovery_hover
        and controller_locked
        and observation.get("endpoint_verified") is True
        and visual_progress >= 0.90
        and current_progress > 1.0
    ):
        # Independent whole-leg endpoint consensus is allowed to repair a
        # poisoned beyond-target metric floor while the aircraft is neutrally
        # hovering. It cannot authorize translation and cannot apply mid-leg.
        return True
    if output_rejection_reason is not None:
        return visual_progress >= current_progress - 0.01
    if recovery_hover:
        return visual_progress >= current_progress + max(
            0.001,
            float(minimum_progress_gain),
        )
    if not isinstance(metric_route_observation, dict):
        # No metric route observation exists for this frame (TSolve/pool miss),
        # so verified current-image route vision is the only new position.
        return visual_progress >= current_progress - 0.01
    try:
        metric_progress = float(metric_route_observation.get("progress"))
    except (TypeError, ValueError):
        return visual_progress >= current_progress - 0.01
    if not math.isfinite(metric_progress):
        return visual_progress >= current_progress - 0.01
    # A valid current-frame metric observation owns translation. The visual
    # route remains attached as supervision metadata and can still recover on
    # a later metric rejection; it cannot make the model arrive at Point 4
    # before the physical drone merely because a prerecorded image matched a
    # later route frame.
    return False


def visual_route_heading_minimum_inliers(
    base_minimum_inliers: int,
    *,
    leg_index: int,
) -> int:
    """Use thresholds supported by each recorded departure-view sector.

    The whole-route matcher normally has 120+ inliers, but departure heading
    uses only five images at the leg start.  The returned Point-1 sector has a
    small position offset and the Point-4 window/wall sector is much narrower.
    The 10:04:06 live Point-1 turn produced repeatable aligned observations at
    48-52 inliers, while the previous 75-inlier floor rejected every frame.
    Keep the strong gates on the other two corners and require three distinct
    controller-side aligned frames before translation regardless of this
    per-frame threshold.
    """
    base = max(16, int(base_minimum_inliers))
    if int(leg_index) == 1:
        return max(48, int(round(base * 0.40)))
    if int(leg_index) == 4:
        return max(50, int(round(base * 0.25)))
    return base


def visual_route_heading_metadata(
    *,
    context: dict[str, Any] | None,
    observation: dict[str, Any] | None,
    diagnostic: dict[str, Any] | None,
    minimum_inliers: int,
    map_id: str = "",
    patrol_id: str = "",
    baseline_replay_id: str = "",
) -> dict[str, Any]:
    """Publish absolute recorded-view heading during audited waypoint turns.

    Route identity belongs to the already-validated recovery bank, not to the
    success state of a single image match.  Publish it on both verified and
    rejected observations so the controller can report and react to the real
    rejection reason instead of misclassifying weak evidence as an identity
    mismatch.
    """
    if not isinstance(context, dict):
        return {}
    if not patrol_reference_frames_enabled(context):
        return {}
    if context.get("controller_translation_locked") is not True:
        return {}
    diagnostic = diagnostic if isinstance(diagnostic, dict) else {}
    verified = bool(
        isinstance(observation, dict) and observation.get("verified") is True
    )
    recorded_heading = (
        normalize_room_heading(observation.get("heading")) if verified else None
    )
    correction_deg = (
        float(observation["correction_deg"])
        if verified and observation.get("correction_deg") is not None
        else None
    )
    current_heading = (
        rotate_room_heading(recorded_heading, -math.radians(correction_deg))
        if recorded_heading is not None and correction_deg is not None
        else None
    )
    observation_identity = observation if isinstance(observation, dict) else {}
    trusted_map_id = str(map_id or observation_identity.get("map_id") or "")
    trusted_patrol_id = str(
        patrol_id or observation_identity.get("patrol_id") or ""
    )
    trusted_baseline_replay_id = str(
        baseline_replay_id
        or observation_identity.get("baseline_replay_id")
        or ""
    )
    return {
        "route_visual_heading_required": True,
        "route_visual_heading_leg_index": int(context.get("leg_index") or 0),
        "route_visual_heading_verified": verified,
        "route_visual_heading_reason": str(diagnostic.get("reason") or ""),
        "route_visual_heading_correction_deg": correction_deg,
        "route_visual_heading_current": current_heading,
        "route_visual_heading_recorded": recorded_heading,
        "route_visual_heading_inliers": int(
            observation.get("inliers")
            if verified
            else (diagnostic.get("best_inliers") or 0)
        ),
        "route_visual_heading_minimum_inliers": max(16, int(minimum_inliers)),
        "route_visual_heading_ratio_matches": (
            int(observation.get("ratio_matches") or 0) if verified else 0
        ),
        "route_visual_heading_anchor": (
            observation.get("anchor_name") if verified else None
        ),
        "route_visual_heading_source_frame": (
            observation.get("source_frame") if verified else None
        ),
        "route_visual_heading_map_id": trusted_map_id or None,
        "route_visual_heading_patrol_id": trusted_patrol_id or None,
        "route_visual_heading_baseline_replay_id": (
            trusted_baseline_replay_id or None
        ),
    }


def apply_visual_route_heading_alignment(
    pose: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Make the rendered heading follow the absolute departure-image match."""
    if not isinstance(pose, dict) or not isinstance(metadata, dict):
        return pose
    try:
        route_leg_index = int(metadata.get("route_visual_heading_leg_index") or 0)
    except (TypeError, ValueError):
        route_leg_index = 0
    if route_leg_index in {1, 2}:
        # The saved best-one-lap route is metric-led through Point 3. A
        # recorded departure match may validate/reseed tracking here, but it
        # must not replace the current TSolve/optical heading consumed by the
        # flight bridge. The captured 2026-08-24 failure showed the two sources
        # fighting: TSolve observed the physical turn while the bank held an
        # older heading, causing repeated yaw and then diagonal travel.
        pose["route_visual_heading_diagnostic_only"] = True
        pose["metric_heading_authority"] = "tsolve_with_optical_yaw_feedback"
        pose["route_visual_heading_authority"] = False
        return pose
    # A single ORB image can be a repeated-texture false positive.  Rendering
    # it immediately made the model snap by ~52 degrees and then snap back on
    # the next weak frame.  Rendering therefore accepts only the stabilized
    # consensus produced below: three aligned frames at turn completion, or
    # two strong observations that consistently track an unfinished turn.
    if metadata.get("route_visual_heading_render_consensus_verified") is not True:
        return pose
    current_heading = normalize_room_heading(
        metadata.get("route_visual_heading_render_current")
    )
    if current_heading is None:
        return pose
    solver_heading = normalize_room_heading(pose.get("rheading"))
    if solver_heading is not None and "rheading_raw" not in pose:
        pose["rheading_raw"] = solver_heading
    pose["rheading"] = current_heading
    pose["rheading_source"] = str(
        metadata.get("route_visual_heading_render_source")
        or "recorded_departure_image_alignment"
    )
    return pose


def stabilize_visual_route_heading_for_render(
    state: dict[str, Any],
    metadata: dict[str, Any] | None,
    *,
    frame_index: int,
    required_hits: int = 3,
    maximum_correction_deg: float = 4.0,
    latch_frames: int = 3,
) -> dict[str, Any]:
    """Latch a current-frame ORB heading without accepting single-image snaps.

    The three-frame, <=4 degree aligned consensus remains the flight turn-
    completion proof.  Rendering also needs a continuous heading while the
    aircraft is still turning: skipped TSolve frames otherwise leave optical
    yaw frozen even though consecutive route-image observations see the turn.
    Two strong observations may therefore carry the *current* camera heading
    when their correction moves consistently toward the departure view.  This
    state affects heading only; position remains locked by the controller.
    """
    out = dict(metadata) if isinstance(metadata, dict) else {}
    out["route_visual_heading_render_consensus_verified"] = False
    if out.get("route_visual_heading_required") is not True:
        state.clear()
        return out
    key = (
        int(out.get("route_visual_heading_leg_index") or 0),
        str(out.get("route_visual_heading_map_id") or ""),
        str(out.get("route_visual_heading_patrol_id") or ""),
        str(out.get("route_visual_heading_baseline_replay_id") or ""),
    )
    if state.get("key") != key:
        state.clear()
        state["key"] = key
    try:
        correction_deg = float(out.get("route_visual_heading_correction_deg"))
        inliers = int(out.get("route_visual_heading_inliers") or 0)
        minimum_inliers = int(
            out.get("route_visual_heading_minimum_inliers") or 0
        )
    except (TypeError, ValueError):
        correction_deg = float("inf")
        inliers = 0
        minimum_inliers = 1
    current_heading = normalize_room_heading(
        out.get("route_visual_heading_current")
    )
    strongly_verified = bool(
        out.get("route_visual_heading_verified") is True
        and math.isfinite(correction_deg)
        and abs(correction_deg) <= 120.0
        and inliers >= max(1, minimum_inliers)
        and current_heading is not None
    )
    aligned = bool(
        strongly_verified
        and abs(correction_deg) <= max(0.1, float(maximum_correction_deg))
    )
    previous_correction = state.get("tracking_correction_deg")
    previous_tracking_heading = normalize_room_heading(
        state.get("tracking_current")
    )
    tracking_consistent = False
    if strongly_verified:
        try:
            previous_correction_value = float(previous_correction)
        except (TypeError, ValueError):
            previous_correction_value = None
        correction_consistent = bool(
            previous_correction_value is not None
            and math.isfinite(previous_correction_value)
            and (
                abs(correction_deg)
                <= abs(previous_correction_value) + 3.0
            )
            and (
                correction_deg * previous_correction_value >= 0.0
                or max(abs(correction_deg), abs(previous_correction_value)) <= 6.0
            )
        )
        heading_separation = room_heading_separation_degrees(
            previous_tracking_heading,
            current_heading,
        )
        tracking_consistent = bool(
            correction_consistent
            and heading_separation is not None
            and heading_separation <= 35.0
        )
        state["tracking_hits"] = (
            int(state.get("tracking_hits") or 0) + 1
            if tracking_consistent
            else 1
        )
        state["tracking_correction_deg"] = correction_deg
        state["tracking_current"] = list(current_heading)
        if int(state["tracking_hits"]) >= 2:
            state["tracking_latched_frame"] = int(frame_index)
            state["tracking_latched_hits"] = int(state["tracking_hits"])
            state["tracking_latched_current"] = list(current_heading)
    else:
        state["tracking_hits"] = 0
        state.pop("tracking_correction_deg", None)
        state.pop("tracking_current", None)
    previous_frame = int(state.get("last_verified_frame") or -2)
    if aligned:
        state["hits"] = (
            int(state.get("hits") or 0) + 1
            if int(frame_index) == previous_frame + 1
            else 1
        )
        state["last_verified_frame"] = int(frame_index)
        if int(state["hits"]) >= max(3, int(required_hits)):
            state["latched_frame"] = int(frame_index)
            state["latched_hits"] = int(state["hits"])
            state["latched_current"] = list(
                normalize_room_heading(out["route_visual_heading_current"])
            )
    else:
        state["hits"] = 0
        state["last_verified_frame"] = -2
    latched_frame = int(state.get("latched_frame") or -10_000)
    render_current = normalize_room_heading(state.get("latched_current"))
    render_source = "recorded_departure_image_alignment"
    render_hits = int(state.get("latched_hits") or 0)
    tracking_latched_frame = int(
        state.get("tracking_latched_frame") or -10_000
    )
    if (
        render_current is None
        or int(frame_index) - latched_frame > max(0, int(latch_frames))
    ):
        render_current = normalize_room_heading(
            state.get("tracking_latched_current")
        )
        latched_frame = tracking_latched_frame
        render_source = "recorded_departure_image_tracking_consensus"
        render_hits = int(state.get("tracking_latched_hits") or 0)
    if (
        render_current is not None
        and int(frame_index) - latched_frame <= max(0, int(latch_frames))
    ):
        out["route_visual_heading_render_consensus_verified"] = True
        out["route_visual_heading_render_consensus_count"] = render_hits
        out["route_visual_heading_render_current"] = render_current
        out["route_visual_heading_render_source"] = render_source
    return out


def visual_route_supervision_metadata(
    *,
    context: dict[str, Any] | None,
    observation: dict[str, Any] | None,
    diagnostic: dict[str, Any] | None,
    progress_hint: float | None,
    minimum_inliers: int,
) -> dict[str, Any]:
    """Describe continuous recorded-route supervision for the command gate."""
    if not isinstance(context, dict):
        return {}
    leg_index = int(context.get("leg_index") or 0)
    if not patrol_reference_frames_enabled(context):
        return {}
    diagnostic = diagnostic if isinstance(diagnostic, dict) else {}
    endpoint_source = observation if isinstance(observation, dict) else diagnostic
    diagnostic_temporal_hits = int(diagnostic.get("acquisition_hits") or 0)
    diagnostic_temporal_required_hits = int(
        diagnostic.get("required_hits") or 0
    )
    temporal_recovery = bool(
        isinstance(observation, dict)
        and observation.get("temporal_recovery") is True
        or diagnostic.get("temporal_recovery_evidence_retained") is True
        or (
            str(diagnostic.get("reason") or "").startswith(
                "visual_route_temporal_recovery_"
            )
            and diagnostic_temporal_hits > 0
        )
    )
    observation_minimum_inliers = (
        int(observation.get("minimum_inliers") or minimum_inliers)
        if isinstance(observation, dict)
        else int(diagnostic.get("minimum_inliers") or minimum_inliers)
    )
    visual_progress = (
        float(observation["progress"])
        if isinstance(observation, dict) and observation.get("progress") is not None
        else None
    )
    disagreement_m = None
    if visual_progress is not None and progress_hint is not None:
        start = finite_room_vector(context.get("start"))
        end = finite_room_vector(context.get("end"))
        if start is not None and end is not None:
            leg_length = math.hypot(end[0] - start[0], end[2] - start[2])
            disagreement_m = abs(visual_progress - float(progress_hint)) * leg_length
    return {
        "route_visual_monitor_required": True,
        "route_visual_monitor_verified": bool(
            isinstance(observation, dict) and observation.get("verified") is True
        ),
        "route_visual_monitor_reason": str(diagnostic.get("reason") or ""),
        "route_visual_monitor_inliers": int(
            observation.get("inliers")
            if isinstance(observation, dict)
            else (diagnostic.get("best_inliers") or 0)
        ),
        "route_visual_monitor_minimum_inliers": (
            visual_route_temporal_recovery_minimum_inliers(
                observation_minimum_inliers,
                leg_index=leg_index,
            )
            if temporal_recovery
            else max(120, int(minimum_inliers))
        ),
        "route_visual_monitor_temporal_recovery": temporal_recovery,
        "route_visual_monitor_temporal_recovery_hits": int(
            observation.get("temporal_recovery_hits") or 0
            if isinstance(observation, dict)
            else diagnostic_temporal_hits
        ),
        "route_visual_monitor_temporal_recovery_required_hits": int(
            observation.get("temporal_recovery_required_hits") or 0
            if isinstance(observation, dict)
            else diagnostic_temporal_required_hits
        ),
        "route_visual_monitor_progress": visual_progress,
        "route_visual_monitor_tsolve_progress": (
            float(progress_hint) if progress_hint is not None else None
        ),
        "route_visual_monitor_disagreement_m": disagreement_m,
        "route_visual_monitor_leg_index": leg_index,
        "route_visual_monitor_translation_locked": bool(
            context.get("translation_locked")
        ),
        # This is deliberately produced by a whole-leg search that does not
        # consume ``progress_hint``. It is the only visual authority allowed to
        # unlock a taught waypoint arrival.
        "route_visual_endpoint_checked": bool(
            endpoint_source.get("endpoint_checked")
        ),
        "route_visual_endpoint_verified": bool(
            endpoint_source.get("endpoint_verified")
        ),
        "route_visual_endpoint_match_consensus_verified": bool(
            endpoint_source.get("endpoint_match_consensus_verified")
        ),
        "route_visual_endpoint_view_geometry_verified": bool(
            endpoint_source.get("endpoint_view_geometry_verified")
        ),
        "route_visual_endpoint_view_scale_min": endpoint_source.get(
            "endpoint_view_scale_min"
        ),
        "route_visual_endpoint_view_scale_max": endpoint_source.get(
            "endpoint_view_scale_max"
        ),
        "route_visual_endpoint_guarded": bool(
            endpoint_source.get("endpoint_guarded")
        ),
        "route_visual_endpoint_guard_progress": endpoint_source.get(
            "endpoint_guard_progress"
        ),
        "route_visual_endpoint_safe_prearrival_progress": endpoint_source.get(
            "endpoint_safe_prearrival_progress"
        ),
        "route_visual_endpoint_hits": int(
            endpoint_source.get("endpoint_hits") or 0
        ),
        "route_visual_endpoint_required_hits": int(
            endpoint_source.get("endpoint_required_hits") or 0
        ),
        "route_visual_endpoint_minimum_inliers": int(
            endpoint_source.get("endpoint_minimum_inliers") or 0
        ),
        "route_visual_endpoint_candidate_progress": endpoint_source.get(
            "endpoint_candidate_progress"
        ),
        "route_visual_endpoint_best_progress": endpoint_source.get(
            "endpoint_best_progress"
        ),
        "route_visual_endpoint_best_inliers": int(
            endpoint_source.get("endpoint_best_inliers") or 0
        ),
        "route_visual_endpoint_best_anchor": endpoint_source.get(
            "endpoint_best_anchor"
        ),
    }


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


def continuity_max_step(
    previous_time: float | None,
    current_time: float | None,
    *,
    hard_cap: float = 0.55,
    max_speed: float = 0.85,
) -> float:
    if previous_time is None or current_time is None:
        return 0.30
    dt = max(0.0, float(current_time) - float(previous_time))
    # Ten-FPS catch-up can intentionally skip queued images. Allow physically
    # continuous travel across that short capture gap, but retain a hard cap so
    # elapsed time can never make a meter-scale wrong root acceptable.
    return max(0.30, min(max(0.30, float(hard_cap)), 0.18 + max(0.0, float(max_speed)) * dt))


OUTPUT_OBJECTIVE_REJECTION_THRESHOLD = 26.0
TRACKED_OBJECTIVE_HARD_MULTIPLIER = 2.0
# The output gate may be configured more tightly than this (the lab currently
# uses 0.30 m), but that publication threshold is not proof that the underlying
# 2D-to-3D optical correspondences are corrupt.  Keep tracking through bounded
# catch-up roots and reserve destructive pool reset for repeated jumps beyond
# this independent absolute limit.  Rejected poses remain held either way.
TRACKING_RESET_HARD_MOTION_CAP = 0.55


def tracking_reset_hard_motion_cap(configured_output_cap: float) -> float:
    """Separate pose-publication strictness from destructive tracker reset."""
    return max(float(configured_output_cap), TRACKING_RESET_HARD_MOTION_CAP)


def output_objective_rejection(
    result: dict[str, Any],
    threshold: float = OUTPUT_OBJECTIVE_REJECTION_THRESHOLD,
) -> str | None:
    try:
        objective = float(result.get("objective"))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(objective):
        return "nonfinite_objective"
    # Good live poses in our indoor runs are normally single-digit residuals.
    # Keep this as a broad sanity gate so only very poor roots are held back.
    if objective > threshold:
        return (
            f"objective_{objective:.3f}_gt_"
            f"{threshold:.3f}"
        )
    return None


def output_continuity_rejection(
    *,
    case: dict[str, Any],
    result: dict[str, Any],
    previous_center: np.ndarray | None,
    previous_time: float | None,
    max_step: float = 0.55,
    max_speed: float = 0.85,
    objective_threshold: float = OUTPUT_OBJECTIVE_REJECTION_THRESHOLD,
    allow_startup_vertical_motion: bool = False,
    room_transform=None,
    startup_vertical_max_step: float = 0.75,
) -> tuple[str | None, np.ndarray | None, float | None]:
    if not result.get("success"):
        return None, previous_center, previous_time

    center = result_center_from_rt(result)
    if center is None:
        return "missing_tsolve_center", previous_center, previous_time

    try:
        current_time = float(case.get("time_sec"))
    except (TypeError, ValueError):
        current_time = None

    objective_reason = output_objective_rejection(result, objective_threshold)
    if previous_center is None:
        # Bootstrap has no independent temporal geometry to corroborate a
        # marginal root, so retain the normal objective gate here.
        if objective_reason is not None:
            return objective_reason, previous_center, previous_time
        return None, center, current_time

    motion_center = center
    motion_previous_center = previous_center
    if allow_startup_vertical_motion and room_transform is not None:
        try:
            motion_center = np.asarray(room_transform(center.astype(float).tolist()), dtype=float).reshape(3)
            motion_previous_center = np.asarray(
                room_transform(previous_center.astype(float).tolist()),
                dtype=float,
            ).reshape(3)
        except (TypeError, ValueError):
            motion_center = center
            motion_previous_center = previous_center
    motion_delta = motion_center - motion_previous_center
    if allow_startup_vertical_motion:
        # Takeoff is deliberately separated from patrol translation.  The
        # 2026-08-11 startup rose 0.523 m while moving only 0.262 m across the
        # room floor.  Treating the full 3D 0.585 m ascent as a horizontal
        # teleport rejected the healthy optical chain before the mission even
        # began.  Permit only a bounded vertical takeoff step while keeping the
        # ordinary horizontal continuity cap fully active.
        vertical_step = abs(float(motion_delta[1]))
        vertical_cap = max(float(max_step), float(startup_vertical_max_step))
        if vertical_step > vertical_cap:
            return (
                f"vertical_jump_{vertical_step:.3f}m_gt_{vertical_cap:.3f}m",
                previous_center,
                previous_time,
            )
        step = float(np.linalg.norm(motion_delta[[0, 2]]))
    else:
        step = float(np.linalg.norm(motion_delta))
    # Consensus/taught recovery improves *which* root is proposed; it does not
    # alter what the aircraft can physically do.  The former 0.85 m trusted
    # exception accepted the 0.815 m false point-3 turn jump and poisoned the
    # optical tracking chain. Every candidate now shares one physical hard cap.
    hard_cap = float(max_step)
    allowed_step = continuity_max_step(
        previous_time,
        current_time,
        hard_cap=hard_cap,
        max_speed=max_speed,
    )
    if step > allowed_step:
        return f"motion_jump_{step:.3f}m_gt_{allowed_step:.3f}m", previous_center, previous_time

    # Once optical tracking is anchored, a small objective-threshold crossing
    # is a quality warning, not proof that the camera pose is wrong.  Frames
    # 1308-1310 in the live regression score 31.127-33.062 yet move only
    # 2.4-5.1 cm each.  Accept that geometrically corroborated continuation;
    # keep a doubled hard objective ceiling for truly poor roots.
    hard_objective_reason = output_objective_rejection(
        result,
        max(float(objective_threshold), float(objective_threshold) * TRACKED_OBJECTIVE_HARD_MULTIPLIER),
    )
    if hard_objective_reason is not None:
        return hard_objective_reason, previous_center, previous_time

    return None, center, current_time


def result_room_center(
    result: dict[str, Any],
    *,
    room_transform,
    output_center_bias: np.ndarray | None = None,
) -> list[float] | None:
    center = result_center_from_rt(result)
    if center is None:
        return None
    if output_center_bias is not None:
        center = center + np.asarray(output_center_bias, dtype=float).reshape(3)
    if room_transform is None:
        return center.astype(float).tolist()
    try:
        return finite_room_vector(room_transform(center.astype(float).tolist()))
    except (TypeError, ValueError):
        return None


def route_guarded_output_rejection(
    *,
    case: dict[str, Any],
    result: dict[str, Any],
    previous_center: np.ndarray | None,
    previous_time: float | None,
    previous_pose: dict[str, Any] | None,
    route_gate: LivePatrolRouteGate | None,
    room_transform,
    output_center_bias: np.ndarray | None,
    max_step: float,
    max_speed: float,
    objective_threshold: float,
    metric_route_room_bias: Any = None,
    post_yaw_reanchor_cap: float = 0.0,
    lap_start_metric_rebootstrap: bool = False,
    stopped_metric_rebootstrap: bool = False,
) -> tuple[str | None, np.ndarray | None, float | None, dict[str, Any] | None]:
    active_route_context = (
        route_gate.active_context()
        if route_gate is not None and route_gate.enabled
        else None
    )
    absolute_metric_rebootstrap = bool(
        lap_start_metric_rebootstrap or stopped_metric_rebootstrap
    )
    if absolute_metric_rebootstrap:
        # The aircraft has just completed 4->1 while the public pose may have
        # remained route-anchored. A strong current-frame full-map measurement
        # may reveal that accumulated discrepancy, but only during neutral
        # hover. The bridge separately checks its Point-1 error before motion.
        candidate_center = result_center_from_rt(result)
        try:
            candidate_time = float(case.get("time_sec"))
        except (TypeError, ValueError):
            candidate_time = previous_time
        reason = output_objective_rejection(result, objective_threshold)
        if candidate_center is None:
            reason = "lap_checkpoint_missing_tsolve_center"
            candidate_center = previous_center
            candidate_time = previous_time
    else:
        reason, candidate_center, candidate_time = output_continuity_rejection(
            case=case,
            result=result,
            previous_center=previous_center,
            previous_time=previous_time,
            max_step=max_step,
            max_speed=max_speed,
            objective_threshold=objective_threshold,
            allow_startup_vertical_motion=active_route_context is None,
            room_transform=room_transform,
        )
    if route_gate is None or not route_gate.enabled:
        return reason, candidate_center, candidate_time, None
    candidate_room = result_room_center(
        result,
        room_transform=room_transform,
        output_center_bias=output_center_bias,
    )
    route_context = route_gate.active_context()
    if absolute_metric_rebootstrap:
        if stopped_metric_rebootstrap:
            if route_context is None or candidate_room is None:
                return (
                    "stopped_rebootstrap_route_context_missing",
                    previous_center,
                    previous_time,
                    {"context": route_context},
                )
            projection = route_segment_projection_xz(
                candidate_room,
                route_context.get("start"),
                route_context.get("end"),
            )
            if projection is None:
                return (
                    "stopped_rebootstrap_route_projection_invalid",
                    previous_center,
                    previous_time,
                    {"context": route_context},
                )
            progress, cross_track = projection
            if cross_track > min(0.30, float(route_gate.max_cross_track)):
                return (
                    f"stopped_rebootstrap_cross_track_{cross_track:.3f}m",
                    previous_center,
                    previous_time,
                    {
                        "context": route_context,
                        "progress": progress,
                        "cross_track": cross_track,
                    },
                )
            if progress < -0.08 or progress > 1.20:
                return (
                    f"stopped_rebootstrap_progress_{progress:.3f}_outside_leg",
                    previous_center,
                    previous_time,
                    {
                        "context": route_context,
                        "progress": progress,
                        "cross_track": cross_track,
                    },
                )
            return (
                reason,
                candidate_center if reason is None else previous_center,
                candidate_time if reason is None else previous_time,
                {
                    "key": route_gate._key(route_context),
                    "context": route_context,
                    "progress": float(progress),
                    "unbiased_progress": float(progress),
                    "cross_track": float(cross_track),
                    "stopped_global_metric_measurement": True,
                    "route_continuity_override": True,
                    "route_continuity_override_source": (
                        "stopped_current_frame_full_map_measurement"
                    ),
                },
            )
        return (
            reason,
            candidate_center if reason is None else previous_center,
            candidate_time if reason is None else previous_time,
            {
                "context": route_context,
                "lap_start_metric_measurement": True,
                "route_continuity_override": True,
                "route_continuity_override_source": (
                    "stopped_lap_start_full_map_measurement"
                ),
            },
        )
    metric_led_leg = bool(
        metric_tsolve_position_authority_required(route_context)
        and route_context.get("position_guard_locked") is not True
    )
    route_room_bias = finite_room_vector(metric_route_room_bias)
    if metric_led_leg and candidate_room is not None and route_room_bias is not None:
        # Restore the proven best-lap Point-1->2->3 handoff. A commanded yaw
        # cannot translate the aircraft, so the constant post-yaw room bias is
        # part of the TSolve position used by the publisher. Validate that same
        # corrected metric center here; validating the raw monocular center
        # instead pins the route at the waypoint and creates a recovery loop.
        candidate_room = [
            float(candidate_room[index]) + float(route_room_bias[index])
            for index in range(3)
        ]
    previous_room = previous_pose.get("rcenter") if isinstance(previous_pose, dict) else None
    route_reason, observation = route_gate.rejection(candidate_room, previous_room)
    if (
        metric_led_leg
        and route_room_bias is not None
        and isinstance(observation, dict)
    ):
        observation["metric_route_room_bias_applied"] = route_room_bias
        observation["metric_route_position_source"] = "tsolve_post_yaw_room_bias"
    if route_reason is not None:
        # The candidate never becomes a temporal output anchor and therefore
        # cannot become the next optical-flow/global-search reference.
        return route_reason, previous_center, previous_time, observation
    context = observation.get("context") if isinstance(observation, dict) else None
    controller_position_locked = bool(
        isinstance(context, dict)
        and context.get("position_guard_locked") is True
    )
    post_yaw_controller_reanchor = bool(
        not controller_position_locked
        and isinstance(previous_pose, dict)
        and previous_pose.get("rotation_position_locked") is True
        and previous_pose.get("rotation_anchor_commanded") is True
        and float(post_yaw_reanchor_cap) > 0.0
    )
    if reason is None and controller_position_locked:
        # A numerically continuous raw center is still not a translation
        # measurement while the controller physically forbids translation.
        # Advance image correspondences/orientation, but keep the last trusted
        # metric center for continuity and future map search.
        observation["route_continuity_override"] = True
        observation["route_continuity_override_source"] = (
            "controller_position_lock"
        )
        observation["route_continuity_preserved_tracking_center"] = True
        return None, previous_center, previous_time, observation
    if reason is not None:
        # The active patrol segment is a genuine motion constraint: lateral
        # input is zero. A raw 3D correction can therefore exceed the general
        # continuity cap mostly because of monocular lateral/vertical noise
        # while its along-route motion remains small. Keep only strongly
        # tracked, route-consistent corrections as raw tracking state; the
        # published pose is projected/rate-limited below.
        tracked_points = int(case.get("tracked_pool_points") or 0)
        key = observation.get("key") if isinstance(observation, dict) else None
        try:
            raw_progress = float(observation.get("progress"))
        except (AttributeError, TypeError, ValueError):
            raw_progress = float("nan")
        floor = (
            float(route_gate.last_progress)
            if key == route_gate.last_key and route_gate.last_progress is not None
            else None
        )
        if floor is None and isinstance(context, dict):
            # A new leg must use the same controller-anchor floor as the route
            # gate above. Falling back to the previous raw room center here
            # would reintroduce the Point-2 yaw poisoning through the
            # strong-tracking continuity override.
            floor = route_gate._anchor_progress(context)
        segment_length = 0.0
        if isinstance(context, dict):
            start = finite_room_vector(context.get("start"))
            end = finite_room_vector(context.get("end"))
            if start is not None and end is not None:
                segment_length = math.hypot(end[0] - start[0], end[2] - start[2])
        along_step = (
            abs(raw_progress - floor) * segment_length
            if floor is not None and math.isfinite(raw_progress)
            else float("inf")
        )
        # While the controller is explicitly hovering with translation locked,
        # a post-yaw TSolve center is not allowed to move the public pose.  It
        # still has to reach RotationPositionStabilizer.apply(), however, so
        # that the stabilizer can convert the raw post-turn offset into a room
        # bias and clear its release anchor.  Applying the ordinary 0.30 m
        # adjacent-frame cap before that correction created a permanent loop
        # in Live ATLAS 15:41: every stable 0.31-0.33 m candidate was held
        # against the same stale anchor.  Admit only route-consistent locked
        # candidates inside the existing 0.55 m absolute safety cap; below we
        # preserve the trusted tracking center, and the published position
        # remains pinned to the controller anchor.
        post_translation_progress_recovery = bool(
            isinstance(context, dict)
            and context.get("post_translation_progress_recovery") is True
        )
        try:
            command_ceiling = float(
                context.get("route_progress_command_ceiling")
            )
            command_sequence = int(
                context.get("route_progress_command_sequence") or 0
            )
        except (AttributeError, TypeError, ValueError):
            command_ceiling = float("nan")
            command_sequence = 0
        # The bridge publishes the command envelope before sending RC and
        # intentionally retains it while waiting for the resulting camera
        # frame.  Tying catch-up authority only to the short-lived
        # ``physical_translation_active`` flag rejected the real delayed
        # Point-2->Point-3 pose after the command had already ended.  Preserve
        # that issued-motion proof until localization catches the advertised
        # ceiling; yaw and hover cannot increase the sequence or ceiling.
        pending_command_envelope = bool(
            command_sequence > 0
            and floor is not None
            and math.isfinite(command_ceiling)
            and command_ceiling > float(floor) + 1.0e-6
        )
        pending_command_envelope_only = bool(
            pending_command_envelope
            and not post_translation_progress_recovery
            and context.get("physical_translation_active") is not True
        )
        command_bounded_translation = bool(
            isinstance(context, dict)
            and (
                post_translation_progress_recovery
                or context.get("physical_translation_active") is True
                or pending_command_envelope
            )
        )
        commanded_recovery_cap = 0.0
        commanded_recovery_progress_ok = True
        if command_bounded_translation and floor is not None:
            if math.isfinite(command_ceiling) and segment_length > 1e-9:
                # The ceiling is advanced only when the bridge actually sends
                # a horizontal command.  A small numeric tolerance permits a
                # solve on the boundary, but a pose beyond issued motion
                # remains rejected.
                commanded_recovery_cap = max(
                    0.0,
                    (command_ceiling - float(floor)) * segment_length + 0.02,
                )
                commanded_recovery_progress_ok = bool(
                    math.isfinite(raw_progress)
                    and raw_progress <= command_ceiling + 0.008
                )
            else:
                commanded_recovery_progress_ok = False
        route_motion_cap = (
            tracking_reset_hard_motion_cap(max_step)
            if controller_position_locked
            else max(
                0.10,
                float(max_step),
                min(
                    tracking_reset_hard_motion_cap(max_step),
                    commanded_recovery_cap,
                ),
                float(post_yaw_reanchor_cap) if post_yaw_controller_reanchor else 0.0,
            )
        )
        candidate_step = (
            float(np.linalg.norm(np.asarray(candidate_center) - np.asarray(previous_center)))
            if candidate_center is not None and previous_center is not None
            else float("inf")
        )
        route_consistent_motion = (
            str(reason).startswith("motion_jump_")
            and (
                controller_position_locked
                or tracked_points >= 120
                or (
                    command_bounded_translation
                    and (
                        not pending_command_envelope_only
                        or tracked_points >= 80
                    )
                )
                or (
                    post_yaw_controller_reanchor
                    and candidate_step <= float(post_yaw_reanchor_cap)
                )
            )
            and along_step <= route_motion_cap
            and commanded_recovery_progress_ok
        )
        if not route_consistent_motion:
            return reason, candidate_center, candidate_time, observation
        observation["route_continuity_override"] = True
        observation["route_continuity_override_source"] = (
            "controller_position_lock"
            if controller_position_locked
            else (
                "post_yaw_controller_reanchor"
                if post_yaw_controller_reanchor
                else (
                    (
                        "command_bounded_post_translation_recovery"
                        if post_translation_progress_recovery
                        else (
                            "command_bounded_pending_translation"
                            if pending_command_envelope_only
                            else "command_bounded_active_translation"
                        )
                    )
                    if command_bounded_translation
                    else "strong_route_tracking"
                )
            )
        )
        observation["route_along_step_m"] = along_step
        if controller_position_locked:
            # Accept the frame's optical correspondences and orientation, but
            # never let a monocular center solved during commanded yaw become
            # the next temporal/map-search center. The aircraft is physically
            # fixed at the controller anchor, so retain the last trusted metric
            # center until translation is unlocked.
            observation["route_continuity_preserved_tracking_center"] = True
            candidate_center = previous_center
            candidate_time = previous_time
        else:
            candidate_center = result_center_from_rt(result)
            try:
                candidate_time = float(case.get("time_sec"))
            except (TypeError, ValueError):
                candidate_time = previous_time
    return None, candidate_center, candidate_time, observation


def mark_global_recovery_pool(
    pool: dict[str, Any],
    *,
    last_center: np.ndarray | None,
    recovery_max_step: float,
) -> dict[str, Any]:
    """Carry a verified global recovery's bounded continuity allowance to TSolve.

    GlobalRelocalizer already checks the recovered camera center against the
    last trusted center.  Without these fields, the immediately following
    TSolve output is checked again as an ordinary adjacent frame at 0.55 and
    can reject a valid 0.55-0.85 recovery forever.
    """
    marked = dict(pool)
    if last_center is not None:
        marked["trusted_recovery"] = True
        marked["recovery_max_step"] = max(0.10, float(recovery_max_step))
    return marked


def output_rejection_requires_tracking_reset(
    rejection_reason: str | None,
    consecutive_rejections: int,
    reset_after_failures: int,
    hard_motion_cap: float = 0.55,
) -> bool:
    """Keep a healthy optical chain through isolated rejected TSolve roots.

    Drone Path 09:38:32's live-equivalent regression produced one marginal
    objective rejection (31.127 at a 30.0 gate) after 781 accepted poses.  The
    old one-strike policy erased a still-healthy 2D/3D pool and left every
    remaining frame held behind a slow global rematch.  A rejected pose must
    not advance the public position, but it also must not destroy the tracker
    until the configured consecutive-failure threshold is actually reached.
    """
    if rejection_reason is None:
        return False
    if str(rejection_reason).startswith("objective_"):
        # Objective-only failures do not prove that optical correspondences
        # are corrupt.  Preserve them so a later frame can recover locally;
        # the rejected pose itself is still held and never published.
        return False
    if str(rejection_reason).startswith("motion_jump_"):
        try:
            jump = float(
                str(rejection_reason)
                .split("motion_jump_", 1)[1]
                .split("m_gt_", 1)[0]
            )
        except (IndexError, ValueError):
            jump = float("inf")
        if jump <= max(0.0, float(hard_motion_cap)) + 1e-9:
            # A catch-up pose can initially exceed the time-scaled allowance
            # while remaining inside the absolute safety cap.  The live Point-1
            # exit produced the same 0.526 m candidate on frames 1451-1453; the
            # old three-strike reset discarded 137 healthy tracks one frame
            # before elapsed time could corroborate it.  Hold publication, but
            # retain the optical anchor until the temporal gate can decide.
            return False
    return int(consecutive_rejections) >= max(1, int(reset_after_failures))


def route_rejection_can_advance_flow_anchor(rejection_reason: str | None) -> bool:
    """Keep 2D/3D optical tracks without accepting a rejected room pose.

    A low-objective TSolve root can be geometrically consistent yet land on
    the wrong side of the active patrol segment during a turn.  The route gate
    must continue to hold that 3D position, but erasing the independently
    tracked 2D-to-map correspondences makes every later frame unrecoverable.
    Carry only the optical anchor across route-only rejections; the trusted
    room center, route progress, and published pose remain unchanged.
    """
    reason = str(rejection_reason or "")
    return reason.startswith(
        (
            "route_backward_",
            "route_cross_track_",
            "route_turn_drift_",
            "route_progress_outside_segment_",
        )
    )


def next_output_tracking_reset_streak(
    rejection_reason: str | None,
    current_streak: int,
    hard_motion_cap: float = 0.55,
) -> int:
    """Count only consecutive rejections that actually implicate tracking.

    Route-only holds deliberately preserve and advance valid optical 2D/3D
    correspondences.  They must not preload the destructive-reset counter for
    a later, isolated TSolve root.  Live ATLAS 16:30:28 accumulated 168 route
    holds during a neutral recovery; the next single 4.364 m algebraic root was
    therefore treated as a third tracking failure and erased the healthy pool.
    Once erased, all 1,271 later frames had no anchor from which to recover.
    """
    if rejection_reason is None or route_rejection_can_advance_flow_anchor(rejection_reason):
        return 0
    reset_worthy = output_rejection_requires_tracking_reset(
        rejection_reason,
        consecutive_rejections=1,
        reset_after_failures=1,
        hard_motion_cap=hard_motion_cap,
    )
    return max(0, int(current_streak)) + 1 if reset_worthy else 0


def rejected_output_can_advance_flow_anchor(
    rejection_reason: str | None,
    consecutive_rejections: int,
    reset_after_failures: int,
    hard_motion_cap: float,
) -> bool:
    """Keep frame-to-frame map tracks until rejection evidence is persistent.

    The solved 3D camera position and the optical 2D-to-map pool are separate
    outputs.  Refusing a position jump must not also skip the current camera
    image as an optical-flow anchor: during translation the next image may then
    be too far from the older anchor to track at all.  That exact chain froze
    Live ATLAS 11:01:14 at frame 912 after one 0.350 m position rejection.

    Continue the independently tracked pool while the public pose remains
    held.  Persistent reset-worthy roots still clear the pool at the configured
    failure limit and force a clean global rematch.
    """
    if rejection_reason is None:
        return False
    return not output_rejection_requires_tracking_reset(
        rejection_reason,
        consecutive_rejections,
        reset_after_failures,
        hard_motion_cap,
    )


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

    def pose_sort_key(
        pose: dict[str, Any],
    ) -> tuple[int, int, int, float, int, float, str]:
        match = re.search(
            r"(\d+)(?:\.[^.]+)?$",
            str(pose.get("image_name") or pose.get("instance_id") or ""),
        )
        frame_index = int(match.group(1)) if match is not None else -1
        try:
            frame_time = float(pose.get("time_sec"))
        except (TypeError, ValueError):
            frame_time = float("-inf")
        try:
            received_unix = float(pose.get("received_unix"))
        except (TypeError, ValueError):
            received_unix = float("-inf")
        return (
            1 if frame_index >= 0 else 0,
            frame_index,
            1 if math.isfinite(received_unix) else 0,
            received_unix,
            1 if math.isfinite(frame_time) else 0,
            frame_time,
            str(pose.get("instance_id") or ""),
        )

    event_path = path.with_name(f"{path.stem}_events.jsonl")
    state_key = str(path.resolve())
    state = POSE_STREAM_STATE.get(state_key)
    if state is None:
        accepted = held = failed = emitted_count = 0
        if event_path.exists():
            with event_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        existing_pose = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    emitted_count += 1
                    if existing_pose.get("held_pose"):
                        held += 1
                    elif bool(existing_pose.get("success")) and existing_pose.get("center"):
                        accepted += 1
                    else:
                        failed += 1
        if emitted_count > len(poses):
            event_path.write_text("", encoding="utf-8")
            accepted = held = failed = emitted_count = 0
        state = {
            "emitted_count": emitted_count,
            "accepted_count": accepted,
            "held_count": held,
            "failed_count": failed,
        }
        POSE_STREAM_STATE[state_key] = state

    emitted_count = int(state["emitted_count"])
    if len(poses) < emitted_count:
        event_path.write_text("", encoding="utf-8")
        state.update(
            emitted_count=0,
            accepted_count=0,
            held_count=0,
            failed_count=0,
        )
        emitted_count = 0

    new_poses = poses[emitted_count:]
    if new_poses:
        event_path.parent.mkdir(parents=True, exist_ok=True)
        with event_path.open("a", encoding="utf-8") as event_stream:
            for pose in new_poses:
                event_stream.write(
                    json.dumps(pose, separators=(",", ":"), default=str) + "\n"
                )
                if pose.get("held_pose"):
                    state["held_count"] += 1
                elif bool(pose.get("success")) and pose.get("center"):
                    state["accepted_count"] += 1
                else:
                    state["failed_count"] += 1
        state["emitted_count"] = len(poses)

    retained = poses if complete else poses[-POSE_STREAM_WINDOW:]
    pose_start_index = 0 if complete else max(0, len(poses) - len(retained))

    payload = {
        "mode": "simulated_live_tsolve_partial",
        "replay_id": replay_id,
        "frame_source": str(drone_video) if drone_video else None,
        "expected_count": int(expected_count),
        "processed_count": len(poses),
        "accepted_count": int(state["accepted_count"]),
        "held_count": int(state["held_count"]),
        "failed_count": int(state["failed_count"]),
        "complete": bool(complete),
        "updated_at": time.time(),
        "pose_start_index": pose_start_index,
        "pose_events_file": event_path.name,
        "poses": sorted(retained, key=pose_sort_key),
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
    end_frame_idx = int(getattr(args, "end_frame_index", -1))
    last_new_frame_time = time.perf_counter()
    while True:
        if end_frame_idx >= 0 and frame_idx > end_frame_idx:
            return
        frames = image_files(args.query_frames)
        if frame_idx < len(frames):
            if args.follow_dir and not args.follow_all_frames and len(frames) - frame_idx > 3:
                skipped = len(frames) - frame_idx - 1
                frame_idx = len(frames) - 1
                print(
                    f"LIVE CATCH-UP: dropped {skipped} stale queued frames; "
                    f"localizing newest frame {frame_idx}.",
                    flush=True,
                )
            while frame_idx < len(frames):
                if end_frame_idx >= 0 and frame_idx > end_frame_idx:
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
    def __init__(
        self,
        *,
        enabled: bool,
        scale: float,
        max_lag_seconds: float = 0.25,
    ) -> None:
        self.enabled = bool(enabled)
        self.scale = max(0.01, float(scale))
        self.max_lag_seconds = max(0.05, float(max_lag_seconds))
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
        if delay < -self.max_lag_seconds:
            # A slow first global solve must not make a finite live replay race
            # through tens of seconds of unseen frames. Rebase at the current
            # frame, then continue with the original recorded intervals.
            self.wall_start = now - video_elapsed * self.scale
            return 0.0
        if delay <= 0:
            return 0.0
        time.sleep(delay)
        return delay * 1000.0


def periodic_feature_refresh_due(
    *,
    frame_index: int,
    frame_time: float | None,
    last_frame_index: int | None,
    last_frame_time: float | None,
    max_frame_interval: int,
    max_time_interval_seconds: float,
) -> bool:
    """Use whichever feature-extraction limit is reached first."""
    if last_frame_index is None:
        return False
    frame_due = bool(
        max_frame_interval > 0
        and int(frame_index) - int(last_frame_index) >= int(max_frame_interval)
    )
    time_due = False
    if (
        max_time_interval_seconds > 0.0
        and frame_time is not None
        and last_frame_time is not None
    ):
        elapsed = float(frame_time) - float(last_frame_time)
        time_due = bool(
            math.isfinite(elapsed)
            and elapsed >= float(max_time_interval_seconds)
        )
    return frame_due or time_due


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


COLMAP_PAIR_ID_MAX = 2147483647


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """Convert a world-to-camera rotation matrix to COLMAP's [w,x,y,z]."""
    M = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(M))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [
                0.25 * scale,
                (M[2, 1] - M[1, 2]) / scale,
                (M[0, 2] - M[2, 0]) / scale,
                (M[1, 0] - M[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        axis = int(np.argmax(np.diag(M)))
        if axis == 0:
            scale = math.sqrt(max(0.0, 1.0 + M[0, 0] - M[1, 1] - M[2, 2])) * 2.0
            q = np.array(
                [
                    (M[2, 1] - M[1, 2]) / scale,
                    0.25 * scale,
                    (M[0, 1] + M[1, 0]) / scale,
                    (M[0, 2] + M[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        elif axis == 1:
            scale = math.sqrt(max(0.0, 1.0 + M[1, 1] - M[0, 0] - M[2, 2])) * 2.0
            q = np.array(
                [
                    (M[0, 2] - M[2, 0]) / scale,
                    (M[0, 1] + M[1, 0]) / scale,
                    0.25 * scale,
                    (M[1, 2] + M[2, 1]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = math.sqrt(max(0.0, 1.0 + M[2, 2] - M[0, 0] - M[1, 1])) * 2.0
            q = np.array(
                [
                    (M[1, 0] - M[0, 1]) / scale,
                    (M[0, 2] + M[2, 0]) / scale,
                    (M[1, 2] + M[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=np.float64,
            )
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        raise ValueError("Invalid zero quaternion from rotation matrix")
    q /= norm
    return -q if q[0] < 0.0 else q


def camera_distortion(camera: Camera) -> np.ndarray:
    """Return OpenCV distortion coefficients for the COLMAP camera models used here."""
    model = camera.model.upper()
    params = np.asarray(camera.params, dtype=np.float64)
    if model == "SIMPLE_RADIAL" and len(params) >= 4:
        return np.array([params[3], 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if model == "RADIAL" and len(params) >= 5:
        return np.array([params[3], params[4], 0.0, 0.0, 0.0], dtype=np.float64)
    if model == "OPENCV" and len(params) >= 8:
        return np.array([params[4], params[5], params[6], params[7], 0.0], dtype=np.float64)
    if model == "FULL_OPENCV" and len(params) >= 12:
        return np.asarray(params[4:12], dtype=np.float64)
    return np.zeros((5,), dtype=np.float64)


def fixed_center_orientation_consensus(
    *,
    world_rays: np.ndarray,
    camera_rays: np.ndarray,
    image_xy: np.ndarray,
    width: int,
    height: int,
    min_inliers: int,
    max_angle_degrees: float = 1.5,
    sample_count: int = 48000,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Estimate camera rotation while an external controller locks position.

    Repeated room structure can make unconstrained PnP choose a high-inlier
    camera center several metres away. During a controller-confirmed yaw/hover
    recovery the camera center is already known, so solve only the rotation
    that aligns map-space rays with image bearings. Broad image coverage and a
    strict angular consensus remain mandatory.
    """
    world = np.asarray(world_rays, dtype=np.float64).reshape(-1, 3)
    camera = np.asarray(camera_rays, dtype=np.float64).reshape(-1, 3)
    pixels = np.asarray(image_xy, dtype=np.float64).reshape(-1, 2)
    if len(world) != len(camera) or len(world) != len(pixels) or len(world) < 3:
        return None, {"reason": "fixed_center_invalid_correspondences", "inliers": 0}
    world_norm = np.linalg.norm(world, axis=1)
    camera_norm = np.linalg.norm(camera, axis=1)
    finite = (
        np.all(np.isfinite(world), axis=1)
        & np.all(np.isfinite(camera), axis=1)
        & np.all(np.isfinite(pixels), axis=1)
        & (world_norm > 1e-9)
        & (camera_norm > 1e-9)
    )
    world = world[finite] / world_norm[finite, None]
    camera = camera[finite] / camera_norm[finite, None]
    pixels = pixels[finite]
    required = max(18, int(min_inliers))
    if len(world) < required:
        return None, {
            "reason": "fixed_center_too_few_correspondences",
            "inliers": 0,
            "candidates": int(len(world)),
        }

    def fit_rotation(indices: np.ndarray) -> np.ndarray | None:
        covariance = world[indices].T @ camera[indices]
        try:
            left, _singular, right_t = np.linalg.svd(covariance)
        except np.linalg.LinAlgError:
            return None
        rotation = right_t.T @ left.T
        if float(np.linalg.det(rotation)) < 0.0:
            right_t[-1, :] *= -1.0
            rotation = right_t.T @ left.T
        return rotation if np.all(np.isfinite(rotation)) else None

    cosine_threshold = math.cos(math.radians(max(0.1, float(max_angle_degrees))))
    rng = np.random.default_rng(20260810)
    best_indices = np.empty((0,), dtype=np.int64)
    best_rotation: np.ndarray | None = None
    remaining = max(1, int(sample_count))
    batch_size = 256
    while remaining > 0:
        count = min(batch_size, remaining)
        remaining -= count
        # Argpartitioning random scores produces three distinct indices per
        # hypothesis without a Python loop over tens of thousands of samples.
        samples = np.argpartition(
            rng.random((count, len(world))),
            kth=2,
            axis=1,
        )[:, :3]
        covariance = np.einsum(
            "bni,bnj->bij",
            world[samples],
            camera[samples],
        )
        try:
            left, _singular, right_t = np.linalg.svd(covariance)
        except np.linalg.LinAlgError:
            continue
        rotations = np.matmul(right_t.transpose(0, 2, 1), left.transpose(0, 2, 1))
        reflected = np.linalg.det(rotations) < 0.0
        if np.any(reflected):
            right_t[reflected, -1, :] *= -1.0
            rotations[reflected] = np.matmul(
                right_t[reflected].transpose(0, 2, 1),
                left[reflected].transpose(0, 2, 1),
            )
        predicted = np.einsum("bij,nj->bni", rotations, world)
        agreement = np.einsum("bni,ni->bn", predicted, camera)
        counts = np.sum(agreement >= cosine_threshold, axis=1)
        winner = int(np.argmax(counts))
        if int(counts[winner]) > len(best_indices):
            best_rotation = rotations[winner].copy()
            best_indices = np.flatnonzero(
                agreement[winner] >= cosine_threshold
            ).astype(np.int64)

    if best_rotation is None or len(best_indices) < required:
        return None, {
            "reason": "fixed_center_no_orientation_consensus",
            "inliers": int(len(best_indices)),
            "candidates": int(len(world)),
        }
    for _ in range(3):
        refined = fit_rotation(best_indices)
        if refined is None:
            break
        best_rotation = refined
        agreement = np.sum((world @ best_rotation.T) * camera, axis=1)
        updated = np.flatnonzero(agreement >= cosine_threshold).astype(np.int64)
        if np.array_equal(updated, best_indices):
            break
        best_indices = updated
    agreement = np.sum((world @ best_rotation.T) * camera, axis=1)
    angular_errors = np.degrees(
        np.arccos(np.clip(agreement[best_indices], -1.0, 1.0))
    )
    spread = correspondence_spread_metrics(
        pixels[best_indices],
        width=int(width),
        height=int(height),
    )
    median_error = float(np.median(angular_errors)) if len(angular_errors) else float("inf")
    diagnostic = {
        "reason": "",
        "inliers": int(len(best_indices)),
        "candidates": int(len(world)),
        "median_angle_degrees": median_error,
        "correspondence_spread": spread,
    }
    if len(best_indices) < required:
        diagnostic["reason"] = "fixed_center_refined_below_minimum"
        return None, diagnostic
    if not spread["ok"]:
        diagnostic["reason"] = "fixed_center_low_spatial_concentration"
        return None, diagnostic
    if not math.isfinite(median_error) or median_error > 1.0:
        diagnostic["reason"] = "fixed_center_angular_error_too_high"
        return None, diagnostic
    return {
        "R": best_rotation,
        "indices": best_indices,
    }, diagnostic


def direct_pnp_correspondence_pool(
    *,
    database_path: Path,
    map_images: dict[int, Image],
    map_points: dict[int, Any],
    image_name: str,
    min_points: int,
    expected_center: np.ndarray | None = None,
    position_locked: bool = False,
) -> dict[str, Any]:
    """Recover a query pose directly from COLMAP's already-computed pair matches.

    `image_registrator` rewrites the complete fixed reference model even though a
    recovery needs only one query pose.  This routine joins the matched reference
    keypoints to their fixed map point IDs and runs PnP without loading or writing
    a second sparse model.
    """
    try:
        with sqlite3.connect(str(database_path)) as conn:
            image_row = conn.execute(
                "SELECT image_id, camera_id FROM images WHERE name = ?",
                (image_name,),
            ).fetchone()
            if image_row is None:
                return {"accepted": False, "reason": "direct_pnp_query_missing", "image_name": image_name}
            query_id, camera_id = (int(image_row[0]), int(image_row[1]))
            keypoint_row = conn.execute(
                "SELECT rows, cols, data FROM keypoints WHERE image_id = ?",
                (query_id,),
            ).fetchone()
            camera_row = conn.execute(
                "SELECT model, width, height, params FROM cameras WHERE camera_id = ?",
                (camera_id,),
            ).fetchone()
            if keypoint_row is None or camera_row is None:
                return {"accepted": False, "reason": "direct_pnp_query_features_missing", "image_name": image_name}
            kp_rows, kp_cols, kp_blob = keypoint_row
            keypoints = np.frombuffer(kp_blob, dtype=np.float32).reshape(int(kp_rows), int(kp_cols))
            model_id, width, height, params_blob = camera_row
            model_name, _ = CAMERA_MODEL_BY_ID.get(int(model_id), (f"MODEL_{model_id}", 0))
            camera = Camera(
                camera_id=camera_id,
                model=model_name,
                width=int(width),
                height=int(height),
                params=np.frombuffer(params_blob, dtype=np.float64).tolist(),
            )
            # Exact indexed pair-ID lookups avoid a modulo scan over the map's
            # multi-gigabyte two_view_geometries table. SQLite's common bind
            # limit is 999, so query the fixed reference IDs in small batches.
            geometry_rows: list[tuple[Any, ...]] = []
            pair_ids = [
                min(query_id, int(reference_id)) * COLMAP_PAIR_ID_MAX
                + max(query_id, int(reference_id))
                for reference_id in map_images
                if int(reference_id) != query_id
            ]
            for offset in range(0, len(pair_ids), 800):
                batch = pair_ids[offset : offset + 800]
                placeholders = ",".join("?" for _ in batch)
                geometry_rows.extend(
                    conn.execute(
                        f"""
                        SELECT pair_id, rows, cols, data
                        FROM two_view_geometries
                        WHERE rows > 0 AND pair_id IN ({placeholders})
                        """,
                        batch,
                    ).fetchall()
                )
    except (sqlite3.Error, ValueError) as exc:
        return {
            "accepted": False,
            "reason": f"direct_pnp_database_error:{exc}",
            "image_name": image_name,
        }

    votes: dict[int, dict[int, int]] = {}
    matched_references = 0
    for pair_id, match_rows, match_cols, match_blob in geometry_rows:
        left_id = int(pair_id) // COLMAP_PAIR_ID_MAX
        right_id = int(pair_id) % COLMAP_PAIR_ID_MAX
        if query_id == left_id:
            reference_id, query_column, reference_column = right_id, 0, 1
        elif query_id == right_id:
            reference_id, query_column, reference_column = left_id, 1, 0
        else:
            continue
        reference = map_images.get(reference_id)
        if reference is None:
            continue
        matches = np.frombuffer(match_blob, dtype=np.uint32).reshape(int(match_rows), int(match_cols))
        matched_references += 1
        for match in matches:
            query_keypoint = int(match[query_column])
            reference_keypoint = int(match[reference_column])
            if query_keypoint >= len(keypoints) or reference_keypoint >= len(reference.point3d_ids):
                continue
            point_id = int(reference.point3d_ids[reference_keypoint])
            if point_id < 0 or point_id not in map_points:
                continue
            point_votes = votes.setdefault(query_keypoint, {})
            point_votes[point_id] = point_votes.get(point_id, 0) + 1

    candidates: list[tuple[int, int, int]] = []
    for query_keypoint, point_votes in votes.items():
        point_id, count = max(point_votes.items(), key=lambda item: (item[1], -item[0]))
        candidates.append((int(count), int(query_keypoint), int(point_id)))
    candidates.sort(reverse=True)
    used_points: set[int] = set()
    unique: list[tuple[int, int]] = []
    for _, query_keypoint, point_id in candidates:
        if point_id in used_points:
            continue
        used_points.add(point_id)
        unique.append((query_keypoint, point_id))

    required = max(6, int(min_points))
    if len(unique) < required:
        return {
            "accepted": False,
            "reason": "direct_pnp_too_few_correspondences",
            "image_name": image_name,
            "valid_2d3d": int(len(unique)),
            "matched_references": int(matched_references),
        }
    xy = np.asarray([keypoints[idx, :2] for idx, _ in unique], dtype=np.float64)
    pids = np.asarray([point_id for _, point_id in unique], dtype=np.int64)
    p3d = np.asarray([map_points[int(point_id)].xyz for point_id in pids], dtype=np.float64)
    K = camera.K()
    distortion = camera_distortion(camera)
    expected = None
    if expected_center is not None:
        candidate_expected = np.asarray(expected_center, dtype=np.float64).reshape(3)
        if np.all(np.isfinite(candidate_expected)):
            expected = candidate_expected
    fixed_center_diagnostic: dict[str, Any] | None = None
    if bool(position_locked) and expected is not None:
        # Highest-vote pairs come first. Cap the repeated-room single-vote tail
        # so valid fixed-center consensus is not drowned by arbitrary aliases.
        fixed_limit = min(500, len(xy))
        fixed_xy = xy[:fixed_limit]
        fixed_p3d = p3d[:fixed_limit]
        fixed_pids = pids[:fixed_limit]
        normalized_xy = cv2.undistortPoints(
            fixed_xy.reshape(-1, 1, 2),
            K,
            distortion,
        ).reshape(-1, 2)
        camera_rays = np.column_stack(
            [normalized_xy, np.ones(len(normalized_xy), dtype=np.float64)]
        )
        fixed_solution, fixed_center_diagnostic = fixed_center_orientation_consensus(
            world_rays=fixed_p3d - expected.reshape(1, 3),
            camera_rays=camera_rays,
            image_xy=fixed_xy,
            width=camera.width,
            height=camera.height,
            min_inliers=required,
        )
        if fixed_solution is not None:
            fixed_indices = np.asarray(fixed_solution["indices"], dtype=np.int64)
            fixed_R = np.asarray(fixed_solution["R"], dtype=np.float64).reshape(3, 3)
            fixed_t = -fixed_R @ expected
            return {
                "accepted": True,
                "image_name": image_name,
                "xy": fixed_xy[fixed_indices].astype(np.float32),
                "p3d": fixed_p3d[fixed_indices],
                "point3d_ids": fixed_pids[fixed_indices],
                "K": K,
                "colmap_image_id": query_id,
                "colmap_camera_id": camera_id,
                "colmap_registered_points": int(len(fixed_indices)),
                "valid_2d3d": int(len(fixed_indices)),
                "direct_pnp_candidates": int(len(unique)),
                "matched_references": int(matched_references),
                "direct_pnp_center_step": 0.0,
                "direct_pnp_hypotheses": 0,
                "fixed_center_hypotheses": 48000,
                "correspondence_spread": fixed_center_diagnostic["correspondence_spread"],
                "fixed_center_position_lock": True,
                "fixed_center_median_angle_degrees": fixed_center_diagnostic[
                    "median_angle_degrees"
                ],
                "colmap_qvec_world_to_camera": rotmat_to_qvec(fixed_R).tolist(),
                "colmap_tvec_world_to_camera": fixed_t.tolist(),
            }
    solutions: list[dict[str, Any]] = []
    spatially_weak_solutions: list[dict[str, Any]] = []
    most_inliers = 0
    # Repeated indoor structure can give RANSAC more than one plausible pose.
    # Sample deterministic hypotheses, retain broadly distributed solutions,
    # then use the already-trusted preceding center to disambiguate them.
    for seed in range(8):
        cv2.setRNGSeed(1701 + seed)
        success, rvec, tvec, initial_inliers = cv2.solvePnPRansac(
            p3d,
            xy,
            K,
            distortion,
            iterationsCount=1000,
            reprojectionError=6.0,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not success or initial_inliers is None or len(initial_inliers) < required:
            most_inliers = max(most_inliers, 0 if initial_inliers is None else len(initial_inliers))
            continue
        refine_indices = np.asarray(initial_inliers, dtype=np.int64).reshape(-1)
        cv2.solvePnP(
            p3d[refine_indices],
            xy[refine_indices],
            K,
            distortion,
            rvec,
            tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        projected, _ = cv2.projectPoints(p3d, rvec, tvec, K, distortion)
        errors = np.linalg.norm(projected.reshape(-1, 2) - xy, axis=1)
        inliers = np.where(errors <= 6.0)[0].astype(np.int64)
        most_inliers = max(most_inliers, len(inliers))
        if len(inliers) < required:
            continue
        spread = correspondence_spread_metrics(
            xy[inliers], width=camera.width, height=camera.height
        )
        R, _ = cv2.Rodrigues(rvec)
        tvec_flat = np.asarray(tvec, dtype=np.float64).reshape(3)
        center = -R.T @ tvec_flat
        center_step = (
            float(np.linalg.norm(center - expected)) if expected is not None else 0.0
        )
        if len(inliers) < 40 and not spread["ok"]:
            spatially_weak_solutions.append(
                {
                    "inliers": int(len(inliers)),
                    "center_step": center_step,
                    "spread": spread,
                }
            )
            continue
        solutions.append(
            {
                "rvec": np.asarray(rvec, dtype=np.float64).copy(),
                "tvec": tvec_flat,
                "R": R,
                "center": center,
                "inliers": inliers,
                "spread": spread,
                "median_error": float(np.median(errors[inliers])),
                "center_step": center_step,
            }
        )
    if not solutions:
        return {
            "accepted": False,
            "reason": "direct_pnp_insufficient_inliers",
            "image_name": image_name,
            "valid_2d3d": int(len(unique)),
            "pnp_inliers": int(most_inliers),
            "matched_references": int(matched_references),
            "spatially_weak_hypotheses": spatially_weak_solutions,
            "fixed_center_diagnostic": fixed_center_diagnostic,
        }
    best_inlier_count = max(len(solution["inliers"]) for solution in solutions)
    credible_minimum = max(required, int(math.ceil(best_inlier_count * 0.75)))
    credible = [
        solution for solution in solutions if len(solution["inliers"]) >= credible_minimum
    ]
    if expected is not None:
        solution = min(
            credible,
            key=lambda item: (
                item["center_step"],
                -len(item["inliers"]),
                item["median_error"],
            ),
        )
    else:
        solution = min(
            credible,
            key=lambda item: (-len(item["inliers"]), item["median_error"]),
        )
    inliers = solution["inliers"]
    inlier_xy = xy[inliers].astype(np.float32)
    spread = solution["spread"]
    R = solution["R"]
    qvec = rotmat_to_qvec(R)
    tvec_flat = solution["tvec"]
    return {
        "accepted": True,
        "image_name": image_name,
        "xy": inlier_xy,
        "p3d": p3d[inliers],
        "point3d_ids": pids[inliers],
        "K": K,
        "colmap_image_id": query_id,
        "colmap_camera_id": camera_id,
        "colmap_registered_points": int(len(inliers)),
        "valid_2d3d": int(len(inliers)),
        "direct_pnp_candidates": int(len(unique)),
        "matched_references": int(matched_references),
        "direct_pnp_center_step": float(solution["center_step"]),
        "direct_pnp_hypotheses": int(len(solutions)),
        "correspondence_spread": spread,
        "colmap_qvec_world_to_camera": qvec.tolist(),
        "colmap_tvec_world_to_camera": tvec_flat.tolist(),
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


def merge_verified_tracking_pool(
    tracked: dict[str, Any],
    recovered: dict[str, Any],
    *,
    reprojection_error: float = 10.0,
) -> dict[str, Any]:
    """Filter drifting LK tracks with a verified local pose, then replenish.

    The online recovery bank recognizes map points that were observed earlier
    in this same flight.  Its PnP pose is used only to reject inconsistent LK
    correspondences.  Recovered current-frame matches are then merged by 3D
    point id so a refresh both removes poisoned tracks and restores points LK
    dropped during a turn.
    """
    K = np.asarray(recovered.get("K", tracked.get("K")), dtype=float).reshape(3, 3)
    R = np.asarray(recovered.get("pose_prior_R"), dtype=float).reshape(3, 3)
    t = np.asarray(recovered.get("colmap_tvec_world_to_camera"), dtype=float).reshape(3)
    tracked_xy = np.asarray(tracked.get("xy", []), dtype=np.float32).reshape(-1, 2)
    tracked_xyz = np.asarray(tracked.get("p3d", []), dtype=np.float64).reshape(-1, 3)
    tracked_ids = np.asarray(tracked.get("point3d_ids", []), dtype=np.int64).reshape(-1)
    camera_xyz = (R @ tracked_xyz.T).T + t
    valid = np.isfinite(camera_xyz).all(axis=1) & (camera_xyz[:, 2] > 1e-6)
    projected = np.full_like(tracked_xy, np.nan, dtype=np.float64)
    projected_h = (K @ camera_xyz.T).T
    projected[valid] = projected_h[valid, :2] / projected_h[valid, 2:3]
    error = np.linalg.norm(projected - tracked_xy, axis=1)
    valid &= np.isfinite(error) & (error <= float(reprojection_error))

    rows: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for xy, xyz, point_id in zip(tracked_xy[valid], tracked_xyz[valid], tracked_ids[valid]):
        rows[int(point_id)] = (xy, xyz)
    for xy, xyz, point_id in zip(
        np.asarray(recovered.get("xy", []), dtype=np.float32).reshape(-1, 2),
        np.asarray(recovered.get("p3d", []), dtype=np.float64).reshape(-1, 3),
        np.asarray(recovered.get("point3d_ids", []), dtype=np.int64).reshape(-1),
    ):
        rows[int(point_id)] = (xy, xyz)

    out = dict(recovered)
    out["xy"] = np.asarray([row[0] for row in rows.values()], dtype=np.float32)
    out["p3d"] = np.asarray([row[1] for row in rows.values()], dtype=np.float64)
    out["point3d_ids"] = np.asarray(list(rows), dtype=np.int64)
    out["verified_lk_input_points"] = int(len(tracked_xy))
    out["verified_lk_inlier_points"] = int(np.sum(valid))
    out["verified_recovered_points"] = int(len(np.asarray(recovered.get("xy", []))))
    out["trusted_recovery"] = True
    return out


def stable_case_indices(
    pool: dict[str, Any],
    *,
    max_points: int,
    preferred_point3d_ids: np.ndarray | None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Choose a geometrically healthy TSolve subset from the live flow pool.

    Point IDs alone are not a quality test.  A correspondence can survive LK
    tracking while its pixel location has drifted far enough to poison the
    algebraic pose.  The old implementation therefore kept the same forty
    IDs until one disappeared, even when hundreds of other, better mapped
    tracks were available.

    Use a full-pool robust PnP only as a correspondence-consensus test (TSolve
    remains the pose producer).  Retain the preferred IDs while most agree
    with that consensus; otherwise select a new spatially distributed subset
    from the inliers.  If OpenCV consensus is unavailable or temporarily
    underconstrained, preserve the previous deterministic behavior.
    """
    xy = np.asarray(pool["xy"], dtype=np.float64)
    p3d = np.asarray(pool["p3d"], dtype=np.float64)
    pids = np.asarray(pool.get("point3d_ids", np.arange(len(xy))), dtype=np.int64)
    consensus_indices: np.ndarray | None = None
    consensus_reason = "consensus_unavailable"
    if (
        len(xy) >= 8
        and len(p3d) == len(xy)
        and hasattr(cv2, "solvePnPRansac")
        and hasattr(cv2, "SOLVEPNP_EPNP")
    ):
        try:
            K = np.asarray(pool["K"], dtype=np.float64).reshape(3, 3)
            solved, _rvec, _tvec, inliers = cv2.solvePnPRansac(
                p3d.reshape(-1, 1, 3),
                xy.reshape(-1, 1, 2),
                K,
                None,
                iterationsCount=80,
                reprojectionError=7.0,
                confidence=0.995,
                flags=cv2.SOLVEPNP_EPNP,
            )
            if solved and inliers is not None:
                candidate = np.unique(np.asarray(inliers, dtype=int).reshape(-1))
                if len(candidate) >= min(len(xy), max(8, min(max_points, 12))):
                    consensus_indices = candidate
                    consensus_reason = "full_pool_ransac"
                else:
                    consensus_reason = "too_few_consensus_inliers"
            else:
                consensus_reason = "ransac_failed"
        except (KeyError, TypeError, ValueError, cv2.error):
            consensus_reason = "ransac_error"

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

        preferred = np.asarray(chosen[:max_points], dtype=int)
        preferred_healthy = not missing and len(preferred) >= min(max_points, len(xy))
        preferred_inlier_ratio = None
        if consensus_indices is not None and len(preferred):
            consensus_mask = np.zeros(len(xy), dtype=bool)
            consensus_mask[consensus_indices] = True
            preferred_inlier_ratio = float(np.mean(consensus_mask[preferred]))
            preferred_healthy = preferred_healthy and preferred_inlier_ratio >= 0.72

        if preferred_healthy:
            return preferred, {
                "accepted": True,
                "stable_solve_set": True,
                "solve_set_reselected": False,
                "missing_stable_points": 0,
                "preferred_inlier_ratio": preferred_inlier_ratio,
                "selection_consensus": consensus_reason,
                "selection_consensus_inliers": (
                    int(len(consensus_indices)) if consensus_indices is not None else None
                ),
            }

        # Missing or geometrically degraded preferred tracks are not a reason
        # to throw away the full optical pool.  Reselect immediately from the
        # current frame's robust consensus and continue the same local track.
        candidates = consensus_indices
        if candidates is None:
            candidates = np.arange(len(xy), dtype=int)
        if len(candidates) == 0:
            return None, {
                "accepted": False,
                "reason": "no_healthy_solve_correspondences",
                "missing_stable_points": len(missing),
            }
        if len(candidates) > max_points:
            local = farthest_spread_indices(xy[candidates], max_points)
            reselected = candidates[local]
        else:
            reselected = candidates
        return np.asarray(reselected, dtype=int), {
            "accepted": True,
            "stable_solve_set": False,
            "solve_set_reselected": True,
            "missing_stable_points": len(missing),
            "preferred_inlier_ratio": preferred_inlier_ratio,
            "selection_consensus": consensus_reason,
            "selection_consensus_inliers": (
                int(len(consensus_indices)) if consensus_indices is not None else None
            ),
        }

    candidates = consensus_indices
    if candidates is None:
        candidates = np.arange(len(xy), dtype=int)
    if len(candidates) > max_points:
        local = farthest_spread_indices(xy[candidates], max_points)
        chosen = candidates[local]
    else:
        chosen = candidates
    return chosen, {
        "accepted": True,
        "stable_solve_set": False,
        "solve_set_reselected": preferred_point3d_ids is not None,
        "missing_stable_points": 0,
        "preferred_inlier_ratio": None,
        "selection_consensus": consensus_reason,
        "selection_consensus_inliers": (
            int(len(consensus_indices)) if consensus_indices is not None else None
        ),
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
        return None, {
            "optical_flow_ms": 0.0,
            "flow_input_points": 0,
            "tracked_points": 0,
            "pruned_features": 0,
            "reason": "empty_track_pool",
        }

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
            "flow_input_points": int(len(p0)),
            "tracked_points": 0,
            "pruned_features": int(len(p0)),
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
        return None, {
            "optical_flow_ms": elapsed_ms,
            "flow_input_points": int(len(p0)),
            "tracked_points": 0,
            "pruned_features": int(len(p0)),
            "reason": "no_good_tracks",
        }

    out = dict(pool)
    out["xy"] = xy1[idx].astype(np.float32)
    out["p3d"] = np.asarray(pool["p3d"], dtype=np.float64)[idx]
    out["point3d_ids"] = np.asarray(pool.get("point3d_ids", np.arange(len(p0))), dtype=np.int64)[idx]
    out.pop("colmap_qvec_world_to_camera", None)
    out.pop("colmap_tvec_world_to_camera", None)
    return out, {
        "optical_flow_ms": elapsed_ms,
        "flow_input_points": int(len(p0)),
        "tracked_points": int(len(idx)),
        "pruned_features": int(len(p0) - len(idx)),
        "reason": "",
    }


def required_tracking_points(
    pool: dict[str, Any] | None,
    *,
    normal_minimum: int,
    solver_minimum: int,
) -> int:
    """Keep a consensus-recovered track alive down to the solver-safe floor.

    A taught recovery already required several independent 2D->3D anchors to
    agree.  Its merged pool can legitimately contain only 8-14 points during
    an in-place turn, so applying the ordinary 15-point refresh threshold
    immediately throws away the recovery on the next frame.
    """
    if pool is not None and bool(pool.get("trusted_recovery")):
        return max(8, int(solver_minimum))
    return max(1, int(normal_minimum))


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
        "solve_set_reselected": bool(selection_meta.get("solve_set_reselected", False)),
        "missing_stable_points": int(selection_meta.get("missing_stable_points", 0)),
        "preferred_inlier_ratio": selection_meta.get("preferred_inlier_ratio"),
        "selection_consensus": selection_meta.get("selection_consensus"),
        "selection_consensus_inliers": selection_meta.get("selection_consensus_inliers"),
        "input_sha256": sha256_case(K, p3d, p2d),
        "localization_method": method,
        "tracking_parent": tracking_parent,
        "colmap_image_id": pool.get("colmap_image_id"),
        "colmap_camera_id": pool.get("colmap_camera_id"),
        "colmap_registered_points": pool.get("colmap_registered_points"),
        "colmap_qvec_world_to_camera": pool.get("colmap_qvec_world_to_camera"),
        "colmap_tvec_world_to_camera": pool.get("colmap_tvec_world_to_camera"),
        "pose_prior_center": pool.get("pose_prior_center"),
        "pose_prior_R": pool.get("pose_prior_R"),
        "trusted_recovery": bool(pool.get("trusted_recovery", False)),
        "recovery_max_step": pool.get("recovery_max_step"),
        "taught_anchor_name": pool.get("taught_anchor_name"),
        "taught_consensus_count": pool.get("taught_consensus_count"),
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
        "solve_set_reselected": bool(selection_meta.get("solve_set_reselected", False)),
        "preferred_inlier_ratio": selection_meta.get("preferred_inlier_ratio"),
        "selection_consensus": selection_meta.get("selection_consensus"),
        "selection_consensus_inliers": selection_meta.get("selection_consensus_inliers"),
        "image_name": frame_name,
        "time_sec": meta["time_sec"],
        "method": method,
        "trusted_recovery": bool(pool.get("trusted_recovery", False)),
        "recovery_max_step": pool.get("recovery_max_step"),
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
        sift_max_num_features: int,
        query_camera_model: str,
        query_camera_params: str,
        max_reference_images: int,
        tracking_reference_images: int,
        min_points: int,
        recovery_max_step: float,
        matching_threads: int,
        direct_pnp_recovery: bool = False,
        faiss_index_dir: Path | None = None,
        faiss_nprobe: int = 32,
        faiss_top_k: int = 32,
        faiss_ratio: float = 0.80,
        faiss_min_points: int | None = None,
        faiss_reprojection_error: float = 6.0,
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
        self.sift_max_num_features = max(64, int(sift_max_num_features))
        self.query_camera_model = query_camera_model
        self.query_camera_params = str(query_camera_params or "").strip()
        self.references = ReferenceSelector(
            map_sparse_text,
            bootstrap_count=max_reference_images,
            tracking_count=tracking_reference_images,
        )
        self.min_points = min_points
        self.recovery_max_step = max(0.10, float(recovery_max_step))
        self.matching_threads = max(1, int(matching_threads))
        self.direct_pnp_recovery = bool(direct_pnp_recovery)
        # Faiss owns a shared in-memory index and COLMAP feature extraction
        # writes a temporary SQLite database.  A stopped checkpoint and a
        # background refresh must never enter that recovery path together.
        # Unique work directories below provide a second line of defence, but
        # serializing here also prevents two expensive SIFT jobs from starving
        # the live optical-flow loop.
        self.faiss_current_frame_lock = threading.Lock()
        self.map_model_images = (
            read_images_model(map_sparse_model) if self.direct_pnp_recovery else {}
        )
        self.faiss_relocalizer = None
        if faiss_index_dir is not None:
            self.faiss_relocalizer = FaissIVF3DRelocalizer(
                faiss_index_dir,
                nprobe=faiss_nprobe,
                top_k=faiss_top_k,
                ratio=faiss_ratio,
                min_points=(
                    self.min_points
                    if faiss_min_points is None or int(faiss_min_points) <= 0
                    else int(faiss_min_points)
                ),
                reprojection_error=faiss_reprojection_error,
            )
            print(
                "Loaded Faiss global 2D-3D map index:",
                json.dumps(
                    {
                        "path": str(faiss_index_dir),
                        "vectors": int(self.faiss_relocalizer.index.ntotal),
                        "nprobe": int(self.faiss_relocalizer.index.nprobe),
                        "top_k": int(self.faiss_relocalizer.top_k),
                    }
                ),
                flush=True,
            )

    def localize_faiss_current_frame(
        self,
        *,
        frame: Path,
        frame_idx: int,
        query_name: str,
        map_points: dict[int, Any],
        expected_center: np.ndarray | None,
        max_duration_seconds: float | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Run exactly one isolated newest-frame Faiss/TSolve recovery."""
        wait_started = time.perf_counter()
        with self.faiss_current_frame_lock:
            lock_wait_ms = 1000.0 * (time.perf_counter() - wait_started)
            pool, stage = self._localize_faiss_current_frame_locked(
                frame=frame,
                frame_idx=frame_idx,
                query_name=query_name,
                map_points=map_points,
                expected_center=expected_center,
                max_duration_seconds=max_duration_seconds,
            )
        stage["faiss_recovery_lock_wait_ms"] = lock_wait_ms
        return pool, stage

    def _localize_faiss_current_frame_locked(
        self,
        *,
        frame: Path,
        frame_idx: int,
        query_name: str,
        map_points: dict[int, Any],
        expected_center: np.ndarray | None,
        max_duration_seconds: float | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Measure one newest frame without copying the multi-GB map DB.

        The persistent Faiss index already owns all mapped COLMAP SIFT
        descriptors and their 3D identities. Only the current query's COLMAP
        SIFT rows are needed here, so an isolated query database is sufficient.
        The same bounded path is used for stopped lap-boundary checkpoints and
        ordinary live background recovery; copying the complete map database
        made a nominal 20-second recovery remain pending for several minutes.
        """
        started = time.perf_counter()
        stage: dict[str, Any] = {
            "feature_extract_ms": 0.0,
            "match_ms": 0.0,
            "register_ms": 0.0,
            "reason": "",
            "registration_profile": "faiss_ivf_current_frame_checkpoint",
        }
        if self.faiss_relocalizer is None:
            stage["reason"] = "faiss_checkpoint_index_unavailable"
            return None, stage

        recovery_frame_dir = self.work_dir / "recovery_frames"
        recovery_frame_dir.mkdir(parents=True, exist_ok=True)
        preserved_frame = recovery_frame_dir / Path(query_name).name
        if frame.resolve() != preserved_frame.resolve():
            shutil.copy2(frame, preserved_frame)
        stage["preserved_recovery_frame"] = str(preserved_frame)
        shutil.copy2(frame, self.query_root / frame.name)

        # Frame numbers repeat across synchronous and background callers.  A
        # frame-indexed filename therefore allowed one worker to unlink the
        # other worker's SQLite database, producing ``no such table: images``.
        # Every call now owns a private directory for its complete lifetime.
        call_id = f"{frame_idx:06d}_{uuid.uuid4().hex}"
        call_dir = self.work_dir / "faiss_recovery_calls" / call_id
        call_dir.mkdir(parents=True, exist_ok=False)
        database = call_dir / "query.db"
        image_list = call_dir / "image.txt"
        image_list.write_text(query_name + "\n", encoding="utf-8")
        stage["faiss_recovery_call_id"] = call_id
        feature_command: list[object] = [
            self.colmap,
            "feature_extractor",
            "--database_path",
            database,
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
            "--SiftExtraction.max_num_features",
            self.sift_max_num_features,
            "--SiftExtraction.use_gpu",
            "0",
        ]
        if self.query_camera_params:
            feature_command.extend(
                ["--ImageReader.camera_params", self.query_camera_params]
            )
        try:
            timeout = (
                float(max_duration_seconds)
                if max_duration_seconds is not None
                and float(max_duration_seconds) > 0.0
                else None
            )
            stage["feature_extract_ms"] = 1000.0 * run_timed(
                feature_command,
                timeout=timeout,
            )
            pool, diagnostic = self.faiss_relocalizer.localize(
                database_path=database,
                image_name=query_name,
                map_points=map_points,
                expected_center=expected_center,
            )
            stage["match_ms"] = sum(
                float(diagnostic.get(key) or 0.0)
                for key in ("feature_read_ms", "search_ms", "match_filter_ms")
            )
            stage["register_ms"] = float(diagnostic.get("pnp_ms") or 0.0)
            stage["faiss"] = diagnostic
            stage["extracted_features"] = int(
                diagnostic.get("query_descriptors") or 0
            )
            stage["matched_features"] = int(
                diagnostic.get("unique_2d3d_candidates") or 0
            )
            stage["pnp_inliers"] = int(
                (pool or {}).get("faiss_pnp_inliers")
                or diagnostic.get("faiss_pnp_inliers")
                or 0
            )
            if pool is None:
                stage["reason"] = str(
                    diagnostic.get("reason")
                    or "faiss_checkpoint_relocalization_failed"
                )
                return None, stage
            pool["localization_method"] = "faiss_ivf_current_frame_checkpoint"
            pool["trusted_recovery"] = True
            pool["recovery_max_step"] = self.recovery_max_step
            stage["reason"] = ""
            return pool, stage
        finally:
            stage["total_ms"] = 1000.0 * (time.perf_counter() - started)
            shutil.rmtree(call_dir, ignore_errors=True)

    def localize(
        self,
        *,
        frame: Path,
        frame_idx: int,
        query_name: str,
        map_points: dict[int, Any],
        last_center: np.ndarray | None,
        recovery_max_step: float | None = None,
        max_duration_seconds: float | None = None,
        position_locked: bool = False,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        localize_started = time.perf_counter()

        def command_timeout() -> float | None:
            if max_duration_seconds is None or float(max_duration_seconds) <= 0.0:
                return None
            remaining = float(max_duration_seconds) - (time.perf_counter() - localize_started)
            if remaining <= 0.0:
                raise subprocess.TimeoutExpired("global_relocalization", float(max_duration_seconds))
            return remaining

        step_limit = (
            self.recovery_max_step
            if recovery_max_step is None
            else max(self.recovery_max_step, float(recovery_max_step))
        )
        stage = {
            "feature_extract_ms": 0.0,
            "match_ms": 0.0,
            "register_ms": 0.0,
            "reason": "",
            "recovery_max_step": step_limit,
        }
        # Live query-frame directories are intentionally short-lived and are
        # removed when a DJI session stops.  Keep the exact encoded pixels for
        # every expensive global recovery attempt inside the persistent run
        # directory so a failed SIFT/Faiss/PnP decision can be reproduced
        # byte-for-byte after the flight.
        recovery_frame_dir = self.work_dir / "recovery_frames"
        recovery_frame_dir.mkdir(parents=True, exist_ok=True)
        preserved_frame = recovery_frame_dir / Path(query_name).name
        if frame.resolve() != preserved_frame.resolve():
            shutil.copy2(frame, preserved_frame)
        stage["preserved_recovery_frame"] = str(preserved_frame)
        shutil.copy2(frame, self.query_root / frame.name)
        image_list = self.work_dir / f"global_{frame_idx:06d}_image.txt"
        pair_list = self.work_dir / f"global_{frame_idx:06d}_pairs.txt"
        localized = self.work_dir / f"global_{frame_idx:06d}_localized"
        db = self.work_dir / f"global_{frame_idx:06d}.db"
        if db.exists():
            db.unlink()
        shutil.copy2(self.map_database, db)
        image_list.write_text(query_name + "\n", encoding="utf-8")

        feature_command: list[object] = [
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
            "--SiftExtraction.max_num_features",
            self.sift_max_num_features,
            "--SiftExtraction.use_gpu",
            "0",
        ]
        if self.query_camera_params:
            feature_command.extend(
                ["--ImageReader.camera_params", self.query_camera_params]
            )
        stage["feature_extract_ms"] = 1000.0 * run_timed(
            feature_command,
            timeout=command_timeout(),
        )
        try:
            with sqlite3.connect(str(db)) as feature_db:
                extracted_row = feature_db.execute(
                    """
                    SELECT descriptors.rows
                    FROM descriptors
                    JOIN images USING(image_id)
                    WHERE images.name = ?
                    """,
                    (query_name,),
                ).fetchone()
            stage["extracted_features"] = (
                int(extracted_row[0]) if extracted_row is not None else 0
            )
        except (sqlite3.Error, TypeError, ValueError):
            stage["extracted_features"] = None

        if self.faiss_relocalizer is not None:
            command_timeout()
            faiss_pool, faiss_diagnostic = self.faiss_relocalizer.localize(
                database_path=db,
                image_name=query_name,
                map_points=map_points,
                expected_center=last_center,
            )
            command_timeout()
            stage["match_ms"] += float(faiss_diagnostic.get("feature_read_ms") or 0.0)
            stage["match_ms"] += float(faiss_diagnostic.get("search_ms") or 0.0)
            stage["match_ms"] += float(faiss_diagnostic.get("match_filter_ms") or 0.0)
            stage["register_ms"] += float(faiss_diagnostic.get("pnp_ms") or 0.0)
            stage["registration_profile"] = "faiss_ivf_global_pnp"
            stage["faiss"] = faiss_diagnostic
            stage["extracted_features"] = int(
                faiss_diagnostic.get("query_descriptors")
                or stage.get("extracted_features")
                or 0
            )
            stage["matched_features"] = int(
                faiss_diagnostic.get("unique_2d3d_candidates") or 0
            )
            stage["pnp_inliers"] = int(
                faiss_diagnostic.get("faiss_pnp_inliers") or 0
            )
            if faiss_pool is not None:
                stage["pnp_inliers"] = int(
                    faiss_pool.get("faiss_pnp_inliers")
                    or faiss_pool.get("valid_2d3d")
                    or 0
                )
                continuity_reason = global_recovery_continuity_rejection(
                    pool=faiss_pool,
                    last_center=last_center,
                    max_step=step_limit,
                )
                if continuity_reason is None:
                    faiss_pool["localization_method"] = "faiss_ivf_global_pnp"
                    faiss_pool = mark_global_recovery_pool(
                        faiss_pool,
                        last_center=last_center,
                        recovery_max_step=step_limit,
                    )
                    stage["reason"] = ""
                    db.unlink(missing_ok=True)
                    image_list.unlink(missing_ok=True)
                    pair_list.unlink(missing_ok=True)
                    shutil.rmtree(localized, ignore_errors=True)
                    print(
                        "Faiss global PnP recovery accepted:",
                        json.dumps(
                            {
                                "image_name": query_name,
                                "inliers": faiss_pool.get("faiss_pnp_inliers"),
                                "unique_2d3d": faiss_pool.get("faiss_unique_matches"),
                                "source_images": faiss_pool.get("faiss_source_images"),
                                "center": pool_reference_center(faiss_pool).tolist(),
                                "timing_ms": faiss_diagnostic,
                            }
                        ),
                        flush=True,
                    )
                    return faiss_pool, stage
                faiss_diagnostic["reason"] = continuity_reason
            stage["reason"] = str(
                faiss_diagnostic.get("reason") or "faiss_global_relocalization_failed"
            )
            print(
                "Faiss global PnP recovery rejected:",
                json.dumps(faiss_diagnostic, default=str),
                flush=True,
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
                ],
                timeout=command_timeout(),
            )
            if self.direct_pnp_recovery and last_center is not None:
                direct_started = time.perf_counter()
                pool = direct_pnp_correspondence_pool(
                    database_path=db,
                    map_images=self.map_model_images,
                    map_points=map_points,
                    image_name=query_name,
                    min_points=self.min_points,
                    expected_center=last_center,
                    position_locked=position_locked,
                )
                stage["register_ms"] += 1000.0 * (time.perf_counter() - direct_started)
                stage["registration_profile"] = "direct_fixed_map_pnp"
                stage["matched_features"] = max(
                    int(stage.get("matched_features") or 0),
                    int(
                        pool.get("direct_pnp_candidates")
                        or pool.get("valid_2d3d")
                        or 0
                    ),
                )
                stage["pnp_inliers"] = int(
                    pool.get("valid_2d3d")
                    if pool.get("accepted")
                    else pool.get("direct_pnp_inliers") or 0
                )
                if pool.get("accepted"):
                    continuity_reason = global_recovery_continuity_rejection(
                        pool=pool,
                        last_center=last_center,
                        max_step=step_limit,
                    )
                    if continuity_reason is None:
                        pool["localization_method"] = f"{attempt_name}_direct_pnp"
                        pool = mark_global_recovery_pool(
                            pool,
                            last_center=last_center,
                            recovery_max_step=step_limit,
                        )
                        stage["reason"] = ""
                        db.unlink(missing_ok=True)
                        image_list.unlink(missing_ok=True)
                        pair_list.unlink(missing_ok=True)
                        shutil.rmtree(localized, ignore_errors=True)
                        print(
                            "direct PnP recovery accepted:",
                            json.dumps(
                                {
                                    "image_name": query_name,
                                    "attempt": attempt_name,
                                    "inliers": pool.get("valid_2d3d"),
                                    "candidates": pool.get("direct_pnp_candidates"),
                                    "matched_references": pool.get("matched_references"),
                                    "center": pool_reference_center(pool).tolist(),
                                }
                            ),
                            flush=True,
                        )
                        return pool, stage
                    pool = {**pool, "accepted": False, "reason": continuity_reason}
                stage["reason"] = str(pool.get("reason") or "direct_pnp_recovery_failed")
                print("direct PnP recovery rejected:", json.dumps(pool, default=str), flush=True)
                # A second attempt can add the wider bootstrap reference set.
                # Do not fall through to image_registrator: doing so would
                # reintroduce the long model-rewrite pause this mode avoids.
                continue
            if localized.exists():
                shutil.rmtree(localized)
            localized.mkdir(parents=True, exist_ok=True)
            register_cmd: list[object] = [
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
            if last_center is None:
                # The first frame is the global anchor for the complete path.
                # Keep COLMAP's original full bootstrap refinement here: the
                # repeated lab walls can produce a high-inlier PnP solution at
                # the opposite end of the room when every reference camera is
                # fixed.  Later recoveries already have a trusted center and
                # can safely use the fast fixed-reference profile below.
                stage["registration_profile"] = "robust_initial_anchor"
            else:
                stage["registration_profile"] = "fast_fixed_map_recovery"
                register_cmd += [
                    # The map is a fixed localization reference. Optimizing
                    # all existing cameras after registering one recovery
                    # query was the source of the 30-80 second live stalls.
                    "--Mapper.fix_existing_images",
                    "1",
                    "--Mapper.extract_colors",
                    "0",
                    "--Mapper.ba_refine_focal_length",
                    "0",
                    "--Mapper.ba_refine_extra_params",
                    "0",
                    "--Mapper.ba_local_max_num_iterations",
                    "5",
                    "--Mapper.ba_local_max_refinements",
                    "1",
                    "--Mapper.ba_global_max_num_iterations",
                    "5",
                    "--Mapper.ba_global_max_refinements",
                    "1",
                ]
            print(
                f"COLMAP registration profile: {stage['registration_profile']}",
                flush=True,
            )
            stage["register_ms"] += 1000.0 * run_timed(
                register_cmd,
                timeout=command_timeout(),
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
                    max_step=step_limit,
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
                pool = mark_global_recovery_pool(
                    pool,
                    last_center=last_center,
                    recovery_max_step=step_limit,
                )
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
    ap.add_argument(
        "--sift-max-num-features",
        type=int,
        default=1024,
        help=(
            "Maximum SIFT features extracted from each live query used for global "
            "relocalization. The fixed COLMAP map remains unchanged."
        ),
    )
    ap.add_argument("--query-camera-model", default="SIMPLE_RADIAL")
    ap.add_argument(
        "--query-camera-params",
        default="",
        help=(
            "Calibrated COLMAP camera parameters for live query frames, for example "
            "f,cx,cy,k for SIMPLE_RADIAL. Empty preserves COLMAP's automatic initialization."
        ),
    )
    ap.add_argument(
        "--tsolve-root-profile",
        choices=("full", "live_fast"),
        default="full",
        help="TSolve root policy; live_fast scores the preferred separator first and retains full fallback.",
    )
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
    ap.add_argument(
        "--relocalize-every",
        type=int,
        default=0,
        help="Refresh global image features after at most this many processed frames; 0 disables the frame limit.",
    )
    ap.add_argument(
        "--relocalize-every-seconds",
        type=float,
        default=0.0,
        help="Refresh global image features after at most this much frame time; 0 disables the time limit.",
    )
    ap.add_argument("--flow-max-error", type=float, default=34.0)
    ap.add_argument("--flow-backtrack-error", type=float, default=2.5)
    ap.add_argument("--flow-window", type=int, default=21)
    ap.add_argument("--flow-levels", type=int, default=3)
    ap.add_argument("--flow-iterations", type=int, default=18)
    ap.add_argument("--min-track-points", type=int, default=0, help="Minimum tracked 2D/3D correspondences before local TSolve. 0 chooses a safe automatic value.")
    ap.add_argument("--min-track-ratio", type=float, default=0.10, help="Minimum fraction of the previous track pool that must survive LK tracking.")
    ap.add_argument("--proactive-relocalize-points", type=int, default=500, help="Start a non-blocking map-point refresh while the local optical pool is still geometrically useful. 0 disables proactive refresh.")
    ap.add_argument("--proactive-relocalize-cooldown-frames", type=int, default=60, help="Minimum processed frames between proactive correspondence refresh attempts.")
    ap.add_argument(
        "--pose-recovery-global-cooldown-frames",
        type=int,
        default=15,
        help=(
            "Minimum live frames between newest-frame global relocalization attempts "
            "while a post-motion or endpoint recovery hover is active."
        ),
    )
    ap.add_argument("--global-recovery-after-failures", type=int, default=2, help="Run COLMAP recovery only after this many consecutive local failures.")
    ap.add_argument(
        "--background-recovery-timeout-seconds",
        type=float,
        default=0.0,
        help=(
            "Maximum wall time for one asynchronous live map rematch. "
            "Zero leaves finite/offline recovery unbounded."
        ),
    )
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
    ap.add_argument(
        "--global-recovery-max-speed",
        type=float,
        default=0.0,
        help=(
            "Optional speed allowance for finite-video recovery after a multi-frame gap. "
            "Zero preserves the fixed live-flight limit."
        ),
    )
    ap.add_argument(
        "--global-recovery-max-total-step",
        type=float,
        default=5.0,
        help="Hard ceiling for a time-scaled finite-video global recovery.",
    )
    ap.add_argument(
        "--output-max-step",
        type=float,
        default=0.55,
        help="Hard map-space step cap for consecutive published poses.",
    )
    ap.add_argument(
        "--output-max-speed",
        type=float,
        default=0.85,
        help="Time-scaled maximum camera speed used by the published-pose continuity gate.",
    )
    ap.add_argument(
        "--output-objective-threshold",
        type=float,
        default=OUTPUT_OBJECTIVE_REJECTION_THRESHOLD,
        help="Reject successful TSolve roots whose objective exceeds this broad residual sanity limit.",
    )
    ap.add_argument("--blocking-global-recovery", action="store_true", help="Debug mode: block the frame loop while COLMAP recovery runs.")
    ap.add_argument(
        "--blocking-global-retry-interval",
        type=int,
        default=1,
        help=(
            "For finite blocking recovery, run expensive COLMAP only every Nth failed frame; "
            "intermediate frames keep a held pose. Live/background behavior is unchanged."
        ),
    )
    ap.add_argument(
        "--direct-pnp-recovery",
        action="store_true",
        help="Recover later frames directly from fixed-map pair matches without rewriting the sparse model.",
    )
    ap.add_argument(
        "--faiss-index-dir",
        type=Path,
        help="Persistent Faiss IVF index over all COLMAP SIFT observations linked to 3D points.",
    )
    ap.add_argument("--faiss-nprobe", type=int, default=32)
    ap.add_argument("--faiss-top-k", type=int, default=32)
    ap.add_argument("--faiss-ratio", type=float, default=0.80)
    ap.add_argument(
        "--faiss-min-points",
        type=int,
        default=0,
        help="Minimum geometrically verified Faiss 2D-3D matches; zero uses --min-points.",
    )
    ap.add_argument("--faiss-reprojection-error", type=float, default=6.0)
    ap.add_argument(
        "--calibrate-output-to-first-global-anchor",
        action="store_true",
        help=(
            "Apply the small first-frame TSolve center bias to published poses, using "
            "the full COLMAP registration as the absolute anchor. Tracking remains in "
            "the unmodified map frame. Intended for the Camera Path display only."
        ),
    )
    ap.add_argument(
        "--wait-for-background-recovery",
        action="store_true",
        help=(
            "When local tracking is lost, stop consuming newer frames until the "
            "active background COLMAP recovery can be applied to the next frame. "
            "Use this for finite uploaded-video streams where every frame must stay "
            "synchronized; real flight should continue publishing held poses instead."
        ),
    )
    ap.add_argument(
        "--wait-for-metric-checkpoint-recovery",
        action="store_true",
        help=(
            "Hold only while a controller-announced metric lap checkpoint has "
            "a background recovery pending. Ordinary live/replay tracking loss "
            "continues to publish fresh held frames without blocking ingestion."
        ),
    )
    ap.add_argument("--disable-background-recovery", action="store_true", help="Live mode: do not start asynchronous COLMAP recovery after local tracking drops.")
    ap.add_argument("--strict-stable-solve-set", action="store_true", help="Force the original 40 solve correspondences to survive; slower and mostly for debugging.")
    ap.add_argument(
        "--taught-patrol-recovery-bank",
        action="append",
        default=[],
        type=Path,
        help="Compact taught-path 2D->3D anchor bank used before repeated-room global recovery.",
    )
    ap.add_argument(
        "--online-recovery-learn-interval",
        type=int,
        default=4,
        help="Learn an in-memory visual anchor this many frames apart while a live track is healthy.",
    )
    ap.add_argument(
        "--online-recovery-learn-max-points",
        type=int,
        default=80,
        help="Learn live recovery anchors only after the optical pool falls to this size or smaller.",
    )
    ap.add_argument(
        "--rotation-recovery-cooldown-frames",
        type=int,
        default=30,
        help=(
            "While the controller proves yaw-only motion, retry the expensive taught "
            "2D->3D matcher at most once per this many frames. Fresh optical yaw and "
            "the locked position continue publishing between attempts. Translation "
            "retries immediately when the yaw lock ends."
        ),
    )
    ap.add_argument(
        "--route-rejection-recovery-cooldown-frames",
        type=int,
        default=15,
        help=(
            "After repeated route-only pose rejections, retry a fresh recorded/current "
            "2D->3D anchor match at most once per this many frames. This quarantines a "
            "geometrically consistent but wrong repeated-room optical-flow pool."
        ),
    )
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
        "--end-frame-index",
        type=int,
        default=-1,
        help="Stop a finite diagnostic replay after this zero-based frame index.",
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
    ap.add_argument(
        "--follow-all-frames",
        action="store_true",
        help="When following a finite producer, drain every published frame instead of dropping queued frames.",
    )
    ap.add_argument("--stop-file", type=Path, default=None)
    ap.add_argument("--follow-idle-timeout", type=float, default=0.0, help="0 means wait indefinitely while following.")
    ap.add_argument(
        "--rotation-position-stabilizer-profile",
        choices=("off", "live-atlas-141441", "live-atlas-141750"),
        default="off",
        help=(
            "Optional rotation-only room-position profile. live-atlas-141441 "
            "uses the Point-4-reaching 2026-08-10 14:14:41 anchor behavior; the "
            "flight bridge must still gate every translation command."
        ),
    )
    ap.add_argument(
        "--rotation-command-status-json",
        type=Path,
        default=None,
        help="Fresh bridge control-status JSON proving that the current live command is yaw-only.",
    )
    ap.add_argument(
        "--patrol-route-baseline",
        type=Path,
        default=None,
        help=(
            "Audited full-loop reference_candidate.json. During its matching live patrol, "
            "reject off-route/backward poses before updating localization state."
        ),
    )
    ap.add_argument(
        "--patrol-visual-recovery-bank",
        type=Path,
        default=None,
        help=(
            "Pre-audited ORB bank for the exact locked patrol baseline. It is a "
            "two-hit, leg-constrained fallback and never performs free global recovery."
        ),
    )
    ap.add_argument("--patrol-route-max-cross-track", type=float, default=0.55)
    ap.add_argument("--patrol-route-backward-tolerance", type=float, default=0.08)
    ap.add_argument(
        "--patrol-status-max-age",
        type=float,
        default=5.0,
        help="Freshness limit for live route context. Increase only for finite offline replay tests.",
    )
    ap.add_argument(
        "--patrol-turn-max-position-drift",
        type=float,
        default=0.75,
        help=(
            "Maximum raw monocular center drift accepted during a proven yaw-only command. "
            "Published route position remains fixed at the controller anchor."
        ),
    )
    ap.add_argument("--pace-replay", action="store_true", help="For finite uploaded-video replay, process frames on their video timeline instead of as fast as possible.")
    ap.add_argument("--pace-scale", type=float, default=1.0, help="Timeline scale for --pace-replay. 1.0 means real video time; 0.5 means 2x faster.")
    args = ap.parse_args()
    if args.min_track_points <= 0:
        args.min_track_points = max(int(args.min_points), int(args.max_points) * 2)
    args.proactive_relocalize_points = max(0, int(args.proactive_relocalize_points))
    if 0 < args.proactive_relocalize_points <= args.min_track_points:
        args.proactive_relocalize_points = args.min_track_points + 1
    args.proactive_relocalize_cooldown_frames = max(1, int(args.proactive_relocalize_cooldown_frames))
    args.relocalize_every = max(0, int(args.relocalize_every))
    args.relocalize_every_seconds = max(0.0, float(args.relocalize_every_seconds))
    args.pose_recovery_global_cooldown_frames = max(
        1,
        int(args.pose_recovery_global_cooldown_frames),
    )
    args.rotation_recovery_cooldown_frames = max(1, int(args.rotation_recovery_cooldown_frames))
    args.route_rejection_recovery_cooldown_frames = max(
        1, int(args.route_rejection_recovery_cooldown_frames)
    )
    args.min_track_ratio = max(0.0, min(1.0, float(args.min_track_ratio)))
    args.flow_window = max(7, int(args.flow_window) | 1)
    args.flow_levels = max(0, int(args.flow_levels))
    args.flow_iterations = max(5, int(args.flow_iterations))
    args.global_recovery_after_failures = max(1, int(args.global_recovery_after_failures))
    args.background_recovery_timeout_seconds = max(
        0.0,
        float(args.background_recovery_timeout_seconds),
    )

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
    args.taught_patrol_recovery_bank = [path.resolve() for path in args.taught_patrol_recovery_bank]

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

    resuming = args.resume_pose_stream is not None and args.resume_case_dir is not None
    for path in (args.out_dir, args.inputs_out_dir, args.work_dir):
        # Resume case files live under inputs_out_dir.  Validate and load them
        # later, but never erase them before that can happen.  The transient
        # COLMAP work directory is safe to rebuild on every invocation.
        preserve = resuming and path in (args.out_dir, args.inputs_out_dir)
        if path.exists() and not preserve:
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
        sift_max_num_features=args.sift_max_num_features,
        query_camera_model=args.query_camera_model,
        query_camera_params=args.query_camera_params,
        max_reference_images=args.max_reference_images,
        tracking_reference_images=args.tracking_reference_images,
        min_points=args.min_points,
        recovery_max_step=args.global_recovery_max_step,
        matching_threads=args.sift_matching_threads,
        direct_pnp_recovery=args.direct_pnp_recovery,
        faiss_index_dir=args.faiss_index_dir,
        faiss_nprobe=args.faiss_nprobe,
        faiss_top_k=args.faiss_top_k,
        faiss_ratio=args.faiss_ratio,
        faiss_min_points=args.faiss_min_points,
        faiss_reprojection_error=args.faiss_reprojection_error,
    )
    # Always maintain a small sliding bank of correspondences learned from the
    # current live flight.  It is independent of any taught/prerecorded path
    # and gives a draining optical pool a local, pose-constrained refresh
    # before repeated-room global localization becomes necessary.
    online_recovery = TaughtPatrolRecovery.empty_online(max_anchors=60)
    taught_recoveries = [online_recovery] + [
        TaughtPatrolRecovery(path)
        for path in args.taught_patrol_recovery_bank
        if path.exists()
    ]
    if args.taught_patrol_recovery_bank:
        print(
            f"Loaded {len(taught_recoveries) - 1} taught-patrol recovery bank(s) "
            "plus the current-flight online bank.",
            flush=True,
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
    output_center_bias: np.ndarray | None = None
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
    rotation_prev_frame_time: float | None = None
    rotation_heading_frame_gap_seconds: float | None = None
    rotation_heading_timing_valid = False
    rotation_heading_timing_policy_name = "uninitialized"
    rotation_heading_max_frame_gap_seconds = 0.30
    active_route_pose_epoch_key: tuple[int, float] | None = None
    # Resolution-specific fallback from the live DJI/COLMAP calibration
    # matrix (1200 x 675): K[0, 0] = K[1, 1] = 882.4866783165957 px.
    # A registered frame's current K still replaces this fallback below.
    rotation_focal_px = 882.4866783165957
    rotation_last_center: list[float] | None = None
    rotation_last_received_unix: float | None = None
    rotation_stabilizer_profile = str(args.rotation_position_stabilizer_profile or "off")
    rotation_position_stabilizer = RotationOnlyPositionStabilizer(
        enabled=rotation_stabilizer_profile in {"live-atlas-141441", "live-atlas-141750"},
        max_reanchor_step=0.75,
        bias_decay=0.92,
    )
    patrol_route_gate = LivePatrolRouteGate(
        args.patrol_route_baseline,
        args.rotation_command_status_json,
        max_cross_track=args.patrol_route_max_cross_track,
        backward_tolerance=args.patrol_route_backward_tolerance,
        turn_max_drift=args.patrol_turn_max_position_drift,
        max_status_age=args.patrol_status_max_age,
    )
    visual_route_heading_render_state: dict[str, Any] = {}
    if args.patrol_route_baseline and not patrol_route_gate.enabled:
        raise RuntimeError(
            f"Patrol route baseline is invalid or incomplete: {args.patrol_route_baseline}"
        )
    visual_route_recovery = (
        PatrolVisualRouteRecovery(args.patrol_visual_recovery_bank)
        if args.patrol_visual_recovery_bank
        else None
    )
    if visual_route_recovery is not None:
        if not patrol_route_gate.enabled:
            raise RuntimeError("Patrol visual recovery requires the matching live route gate")
        if (
            visual_route_recovery.map_id != patrol_route_gate.map_id
            or visual_route_recovery.patrol_id != patrol_route_gate.patrol_id
            or visual_route_recovery.baseline_replay_id
            != patrol_route_gate.baseline_replay_id
        ):
            raise RuntimeError("Patrol visual recovery bank does not match the locked route baseline")
    prev_case_id: str | None = None
    last_center: np.ndarray | None = None
    last_output_center: np.ndarray | None = None
    last_output_time: float | None = None
    # The first metric pose at Point 1 is a live loop-closure anchor. Keep its
    # map-space center and camera intrinsics in memory so a later lap can
    # reacquire the same place from compact SIFT/2D->3D recovery banks instead
    # of repeatedly timing out in the full COLMAP map.
    lap_start_metric_center: np.ndarray | None = None
    lap_start_metric_K: np.ndarray | None = None
    stable_solve_point3d_ids: np.ndarray | None = None
    consecutive_local_failures = 0
    consecutive_tracking_reset_rejections = 0
    global_relocalization_count = 0
    local_tracking_count = 0
    background_recovery_count = 0
    background_recovery_success_count = 0
    background_recovery_stale_count = 0
    proactive_relocalization_count = 0
    proactive_relocalization_success_count = 0
    proactive_relocalization_fallback_count = 0
    online_recovery_anchor_count = 0
    visual_route_recovery_count = 0
    visual_route_acquisition_hold_count = 0
    route_visual_recovery_window_key: tuple[Any, ...] | None = None
    route_visual_recovery_window_start_frame: int | None = None
    route_visual_recovery_grace_frames = 30
    route_rejection_recovery_attempt_count = 0
    route_rejection_recovery_success_count = 0
    lap_checkpoint_metric_recovery_count = 0
    periodic_feature_refresh_count = 0
    last_feature_extraction_frame: int | None = None
    last_feature_extraction_time: float | None = None
    last_online_recovery_anchor_frame = -max(1, args.online_recovery_learn_interval)
    last_rotation_taught_recovery_attempt_frame = -args.rotation_recovery_cooldown_frames
    last_route_rejection_recovery_attempt_frame = (
        -args.route_rejection_recovery_cooldown_frames
    )
    last_proactive_relocalize_frame = -args.proactive_relocalize_cooldown_frames
    last_pose_recovery_global_frame = -args.pose_recovery_global_cooldown_frames
    last_blocking_global_attempt_frame = -max(1, args.blocking_global_retry_interval)
    pending_global: dict[str, Any] | None = None
    force_route_taught_recovery = False
    pacer = ReplayPacer(enabled=bool(args.pace_replay and not args.follow_dir), scale=args.pace_scale)
    # A paced uploaded video is an interactive stream from the viewer's point
    # of view: frames must keep advancing while a slow COLMAP rematch runs.
    # Treat it like the follow-directory live feed for recovery scheduling,
    # stale-result rejection, and bounded optical-flow catch-up.  Without this
    # shared gate, an uploaded video falls back to a synchronous 40-50 second
    # global rematch on every weak frame even when background recovery is
    # enabled.
    interactive_recovery = bool(args.follow_dir or pacer.enabled)

    if (args.resume_pose_stream is None) != (args.resume_case_dir is None):
        raise ValueError("--resume-pose-stream and --resume-case-dir must be provided together")
    if args.resume_pose_stream is not None and args.resume_case_dir is not None:
        resume_doc = json.loads(args.resume_pose_stream.read_text(encoding="utf-8"))
        resume_poses = []
        for pose in list(resume_doc.get("poses") or []):
            stem = Path(str(pose.get("image_name") or "")).stem
            digits = stem.rsplit("_", 1)[-1]
            if digits.isdigit() and int(digits) >= int(args.start_frame_index):
                continue
            resume_poses.append(pose)
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
        rotation_prev_frame_time = (
            float(trusted_pose["time_sec"])
            if trusted_pose.get("time_sec") is not None
            else None
        )
        stable_solve_point3d_ids = resume_ids.copy()
        last_output_pose = trusted_pose
        last_center = np.asarray(trusted_pose["center"], dtype=np.float64)
        last_output_center = last_center.copy()
        saved_output_bias = trusted_pose.get("output_center_bias")
        if saved_output_bias is not None:
            candidate_bias = np.asarray(saved_output_bias, dtype=np.float64).reshape(-1)
            if candidate_bias.size == 3 and np.all(np.isfinite(candidate_bias)):
                output_center_bias = candidate_bias.copy()
        last_output_time = (
            float(trusted_pose["time_sec"])
            if trusted_pose.get("time_sec") is not None
            else None
        )
        last_feature_extraction_frame = max(0, int(args.start_frame_index) - 1)
        last_feature_extraction_time = last_output_time
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
                    "output_center_bias": (
                        output_center_bias.tolist()
                        if output_center_bias is not None
                        else None
                    ),
                }
            ),
            flush=True,
        )

    def attach_rotation_only_hint(pose: dict[str, Any] | None) -> dict[str, Any] | None:
        """Attach independent optical yaw; it never changes pose validity."""
        if pose is None:
            return pose
        route_context = patrol_route_gate.active_context()
        if metric_tsolve_position_authority_required(route_context):
            # Preserve the proven best-one-lap ownership boundary. On
            # Point 1 -> 2 and Point 2 -> 3, current-frame TSolve is the only
            # source allowed to advance room position/route progress. Optical
            # flow may carry yaw between metric solves, while recorded patrol
            # images remain supervision/recovery evidence only.
            pose["metric_position_authority"] = "tsolve"
            pose["metric_heading_authority"] = "tsolve_with_optical_yaw_feedback"
            pose["route_visual_position_authority"] = False
            pose["route_visual_heading_authority"] = False
        # Held poses are copies of the previous publication.  Explicitly clear
        # its yaw fields so a discontinuous current frame cannot inherit an old
        # optical observation and masquerade as fresh feedback.
        for stale_key in (
            "rotation_heading",
            "rotation_heading_source",
            "rotation_heading_tracks",
            "rotation_heading_delta_deg",
        ):
            pose.pop(stale_key, None)
        pose.update(
            {
                "rotation_heading_timing_valid": bool(
                    rotation_heading_timing_valid
                ),
                "rotation_heading_frame_gap_seconds": (
                    float(rotation_heading_frame_gap_seconds)
                    if rotation_heading_frame_gap_seconds is not None
                    else None
                ),
                "rotation_heading_max_frame_gap_seconds": float(
                    rotation_heading_max_frame_gap_seconds
                ),
                "rotation_heading_timing_policy": str(
                    rotation_heading_timing_policy_name
                ),
            }
        )
        heading = normalize_room_heading(rotation_heading)
        if (
            rotation_heading_timing_valid
            and heading is not None
            and rotation_heading_tracks >= 16
        ):
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
        if (
            pose is None
            or not pose.get("success")
            or pose.get("held_pose")
            or pose.get("rotation_position_locked")
        ):
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

    def record_feature_extraction(frame_idx: int, frame_time: float | None) -> None:
        nonlocal last_feature_extraction_frame, last_feature_extraction_time
        last_feature_extraction_frame = int(frame_idx)
        last_feature_extraction_time = (
            float(frame_time)
            if frame_time is not None and math.isfinite(float(frame_time))
            else None
        )

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
            pending_age = time.perf_counter() - float(
                pending_global.get("started_at", time.perf_counter())
            )
            pending_limit = max(
                2.0,
                float(args.background_recovery_timeout_seconds or 0.0) + 1.0,
            )
            if pending_global.get("done"):
                apply_background_global_recovery(
                    current_frame_idx=frame_idx,
                    current_gray=curr_gray,
                )
            elif pending_age > pending_limit:
                # A Python thread cannot be safely cancelled.  Releasing the
                # state owner here used to launch a second COLMAP/SQLite job
                # while the first still existed.  Keep the owner until its
                # bounded subprocess returns; stale-view checks decide whether
                # its result can still be consumed.
                if not pending_global.get("deadline_exceeded"):
                    pending_global["deadline_exceeded"] = True
                    pending_global["deadline_exceeded_at"] = time.perf_counter()
                    background_recovery_stale_count += 1
                    print(
                        "BACKGROUND RECOVERY DEADLINE EXCEEDED; KEEPING SINGLE OWNER:",
                        json.dumps(
                            {
                                "frame_index": int(
                                    pending_global.get("frame_idx", -1)
                                ),
                                "age_seconds": pending_age,
                                "deadline_seconds": pending_limit,
                            }
                        ),
                        flush=True,
                    )
                return False
            else:
                return False
        center = None if last_center is None else np.asarray(last_center, dtype=float).copy()
        recovery_route_context = patrol_route_gate.active_context()
        recovery_route_key = (
            LivePatrolRouteGate._key(recovery_route_context)
            if recovery_route_context is not None
            else None
        )
        controller_translation_locked = bool(
            recovery_route_context is not None
            and recovery_route_context.get("controller_translation_locked") is True
        )
        position_locked_recovery = bool(
            recovery_route_context is not None
            and recovery_route_context.get("position_guard_locked") is True
        )
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
            "position_locked_recovery": position_locked_recovery,
            # A correspondence pool belongs to both a camera view and the
            # controller motion phase that produced it. A yaw-only recovery
            # must never replace a forward-translation pool after the turn.
            "route_key": recovery_route_key,
            "controller_translation_locked": controller_translation_locked,
        }

        def worker() -> None:
            try:
                if interactive_recovery and relocalizer.faiss_relocalizer is not None:
                    pool, stage = relocalizer.localize_faiss_current_frame(
                        frame=frame,
                        frame_idx=frame_idx,
                        query_name=query_name,
                        map_points=map_points,
                        expected_center=center,
                        max_duration_seconds=(
                            args.background_recovery_timeout_seconds
                        ),
                    )
                    stage["registration_profile"] = (
                        "faiss_ivf_current_frame_live_recovery"
                    )
                    if pool is not None:
                        pool["localization_method"] = (
                            "faiss_ivf_current_frame_live_recovery"
                        )
                else:
                    pool, stage = relocalizer.localize(
                        frame=frame,
                        frame_idx=frame_idx,
                        query_name=query_name,
                        map_points=map_points,
                        last_center=center,
                        max_duration_seconds=(
                            args.background_recovery_timeout_seconds
                            if interactive_recovery
                            else None
                        ),
                        position_locked=position_locked_recovery,
                    )
                recovery["pool"] = pool
                recovery["stage"] = stage
            except Exception as exc:  # pragma: no cover - defensive live path
                recovery["error"] = repr(exc)
                recovery["stage"]["reason"] = repr(exc)
            finally:
                # A timed-out worker owns frame-indexed disposable artifacts.
                # Remove them before a fresh current-view recovery is allowed.
                (relocalizer.work_dir / f"global_{frame_idx:06d}.db").unlink(missing_ok=True)
                (relocalizer.work_dir / f"global_{frame_idx:06d}.db-wal").unlink(missing_ok=True)
                (relocalizer.work_dir / f"global_{frame_idx:06d}.db-shm").unlink(missing_ok=True)
                (relocalizer.work_dir / f"global_{frame_idx:06d}_image.txt").unlink(missing_ok=True)
                (relocalizer.work_dir / f"global_{frame_idx:06d}_pairs.txt").unlink(missing_ok=True)
                shutil.rmtree(
                    relocalizer.work_dir / f"global_{frame_idx:06d}_localized",
                    ignore_errors=True,
                )
                recovery["done"] = True

        thread = threading.Thread(target=worker, name=f"atlas-global-recovery-{frame_idx:06d}", daemon=True)
        recovery["thread"] = thread
        pending_global = recovery
        background_recovery_count += 1
        frame_time_raw = frame_times.get(frame.name, {}).get("time_sec")
        try:
            feature_frame_time = (
                float(frame_time_raw)
                if frame_time_raw not in (None, "")
                else None
            )
        except (TypeError, ValueError):
            feature_frame_time = None
        record_feature_extraction(frame_idx, feature_frame_time)
        thread.start()
        return True

    def append_global_recovery_pose(
        *,
        recovery: dict[str, Any],
        pool: dict[str, Any],
        stage: dict[str, Any],
        elapsed_ms: float,
    ) -> bool:
        foreground_started = time.perf_counter()
        nonlocal current_pool, prev_gray, prev_case_id
        nonlocal last_center, last_output_center, last_output_time, last_output_pose
        nonlocal output_center_bias
        nonlocal stable_solve_point3d_ids, consecutive_local_failures
        nonlocal consecutive_tracking_reset_rejections
        nonlocal last_online_recovery_anchor_frame, online_recovery_anchor_count
        nonlocal force_route_taught_recovery

        # ``elapsed_ms`` is the age/wall-clock latency of work completed by a
        # background thread.  It is useful recovery telemetry, but it is not
        # foreground frame processing and must not be folded into "Other".
        stage["background_worker_ms"] = float(elapsed_ms)

        frame_idx = int(recovery["frame_idx"])
        frame = Path(recovery["frame"])
        query_name = str(recovery["query_name"])
        curr_gray = np.asarray(recovery["gray"])
        # Frame-stable case IDs make interrupted finite-video runs safely
        # resumable. Held/rejected frames no longer shift later IDs or cause a
        # resumed run to overwrite an earlier correspondence case.
        case_id = f"instance_{frame_idx:06d}"
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
        case_build_started = time.perf_counter()
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
        stage["case_build_ms"] = float(stage.get("case_build_ms") or 0.0) + (
            1000.0 * (time.perf_counter() - case_build_started)
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
            case_build_started = time.perf_counter()
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
            stage["case_build_ms"] = float(stage.get("case_build_ms") or 0.0) + (
                1000.0 * (time.perf_counter() - case_build_started)
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
                    "total_frame_ms": (
                        time.perf_counter() - foreground_started
                    ) * 1000.0,
                    "reason": reason,
                },
                stage,
            )
            print("BACKGROUND RECOVERY CASE SKIPPED:", json.dumps({"frame_index": frame_idx, "reason": reason}), flush=True)
            return False

        if stable_solve_point3d_ids is None or stable_solve_reset:
            stable_solve_point3d_ids = np.asarray(case["selected_point3d_ids"], dtype=np.int64)
            print(
                f"locked stable TSolve solve set: {len(stable_solve_point3d_ids)} 3D points",
                flush=True,
            )

        case_output_started = time.perf_counter()
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
        append_manifest_row(args.inputs_out_dir, manifest_rows[-1])
        instance_dir = instances_dir / case_id
        copy_case_to_instance(Path(case["case_dir"]), instance_dir)
        stage["case_output_ms"] = float(stage.get("case_output_ms") or 0.0) + (
            1000.0 * (time.perf_counter() - case_output_started)
        )

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
            fork_on_miss=not interactive_recovery,
            root_candidate_profile=args.tsolve_root_profile,
        )
        solve_ms = (time.perf_counter() - solve_t0) * 1000.0
        pose_update_started = time.perf_counter()
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
        route_observation = None
        if result.get("success"):
            output_rejection_reason, last_output_center, last_output_time, route_observation = route_guarded_output_rejection(
                case=case,
                result=result,
                previous_center=last_output_center,
                previous_time=last_output_time,
                previous_pose=last_output_pose,
                route_gate=patrol_route_gate,
                room_transform=room_transform,
                output_center_bias=output_center_bias,
                metric_route_room_bias=(
                    rotation_position_stabilizer.metric_route_room_bias()
                ),
                max_step=args.output_max_step,
                max_speed=args.output_max_speed,
                objective_threshold=args.output_objective_threshold,
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
            if case.get("solve_set_reselected"):
                stable_solve_point3d_ids = np.asarray(
                    case["selected_point3d_ids"], dtype=np.int64
                )
                stable_solve_reset = True
            consecutive_tracking_reset_rejections = 0
            # Only a publicly accepted pose may advance the inherited
            # optical-flow chain and its map-reference center.
            current_pool = cap_tracking_pool(
                pool,
                args.track_pool_size,
                keep_point3d_ids=stable_solve_point3d_ids,
            )
            prev_gray = curr_gray
            prev_case_id = case_id
            if (
                isinstance(route_observation, dict)
                and route_observation.get(
                    "route_continuity_preserved_tracking_center"
                )
                is True
            ):
                reference_update_reason = "controller_locked_turn_flow_anchor_only"
            else:
                last_center, reference_update_reason = update_tracking_reference_center(
                    case=case,
                    result=result,
                    previous_center=last_center,
                )
            live_pool_count = int(len(np.asarray(current_pool.get("xy", []))))
            learn_interval = max(1, int(args.online_recovery_learn_interval))
            if (
                taught_recoveries
                and live_pool_count >= int(args.min_track_points)
                and frame_idx - last_online_recovery_anchor_frame >= learn_interval
            ):
                learned = 0
                for taught_recovery in taught_recoveries:
                    learned += taught_recovery.learn_anchor(
                        gray=curr_gray,
                        xy=np.asarray(current_pool.get("xy", [])),
                        p3d=np.asarray(current_pool.get("p3d", [])),
                        point3d_ids=np.asarray(current_pool.get("point3d_ids", [])),
                        anchor_name=f"online/{args.replay_id}/{query_name}",
                    )
                if learned > 0:
                    last_online_recovery_anchor_frame = frame_idx
                    online_recovery_anchor_count += 1
                    print(
                        "ONLINE RECOVERY ANCHOR LEARNED:",
                        json.dumps(
                            {
                                "frame_index": frame_idx,
                                "tracked_points": live_pool_count,
                                "descriptor_points": learned,
                                "anchor_count": online_recovery_anchor_count,
                            }
                        ),
                        flush=True,
                    )
        elif result.get("success"):
            consecutive_tracking_reset_rejections = next_output_tracking_reset_streak(
                output_rejection_reason,
                consecutive_tracking_reset_rejections,
                tracking_reset_hard_motion_cap(args.output_max_step),
            )
            reference_update_reason = "held_rejected_pose_not_trusted"
            if route_rejection_can_advance_flow_anchor(output_rejection_reason):
                # The TSolve room pose stays rejected, but its input pool is a
                # separate set of optical 2D-to-map correspondences. Advance
                # that flow anchor so the next frame does not have to track
                # across an ever-growing gap after a harmless turn ambiguity.
                current_pool = cap_tracking_pool(
                    pool,
                    args.track_pool_size,
                    keep_point3d_ids=stable_solve_point3d_ids,
                )
                prev_gray = curr_gray
                prev_case_id = case_id
                reference_update_reason = "route_rejected_pose_flow_anchor_only"
                if consecutive_local_failures >= max(1, int(args.global_recovery_after_failures)):
                    force_route_taught_recovery = True
                    refresh_scheduled = schedule_background_global_recovery(
                        frame_idx=frame_idx,
                        frame=frame,
                        query_name=query_name,
                        curr_gray=curr_gray,
                        reason=f"route_guard_rejection_{output_rejection_reason}",
                    )
                    if refresh_scheduled:
                        print(
                            "ROUTE REJECTION BACKGROUND REFRESH:",
                            json.dumps(
                                {
                                    "frame_index": frame_idx,
                                    "consecutive_rejections": consecutive_local_failures,
                                    "reason": output_rejection_reason,
                                    "tracked_points": int(len(np.asarray(current_pool.get("xy", [])))),
                                }
                            ),
                            flush=True,
                        )
            elif rejected_output_can_advance_flow_anchor(
                output_rejection_reason,
                consecutive_tracking_reset_rejections,
                args.global_recovery_after_failures,
                tracking_reset_hard_motion_cap(args.output_max_step),
            ):
                # Hold the suspect room position, but do not make the next
                # image track across this moving frame.  The independently
                # valid optical pool can carry the chain through an isolated
                # algebraic jump; repeated bad roots still reach the reset
                # branch below.
                current_pool = cap_tracking_pool(
                    pool,
                    args.track_pool_size,
                    keep_point3d_ids=stable_solve_point3d_ids,
                )
                prev_gray = curr_gray
                prev_case_id = case_id
                reference_update_reason = "output_rejected_pose_flow_anchor_only"
            else:
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
                            "consecutive_rejections": consecutive_tracking_reset_rejections,
                            "reason": output_rejection_reason,
                        }
                    ),
                    flush=True,
                )

        if output_rejection_reason is not None and last_output_pose is not None:
            pose_payload = held_pose_from_last(
                # ``last_output_pose`` is the last solver-trusted pose.  The
                # route and rotation guards may have published a newer safe
                # position while intentionally leaving that solver anchor
                # untouched.  Holding the stale solver pose made the model
                # jump back to Point 4 for one frame at the lap seam.  Hold
                # the latest published pose here while retaining
                # ``last_output_pose`` for tracking/recovery continuity.
                last_pose=latest_published_pose(partial_poses, last_output_pose),
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
                output_center_bias=output_center_bias,
            )
        if pose_payload is None:
            pose_payload = partial_pose_from_result(
                case,
                result,
                room_transform=room_transform,
                output_rejection_reason=output_rejection_reason or "no_previous_pose_to_hold",
                output_center_bias=output_center_bias,
            )
        pose_payload = attach_rotation_only_hint(pose_payload)
        pose_payload = rotation_position_stabilizer.apply(pose_payload)
        pose_payload = patrol_route_gate.constrain_published_pose(
            pose_payload,
            route_observation,
        )
        partial_poses.append(pose_payload)
        if pose_payload.get("success") and pose_payload.get("center"):
            last_output_pose = pose_payload
        stage["pose_update_ms"] = float(stage.get("pose_update_ms") or 0.0) + (
            1000.0 * (time.perf_counter() - pose_update_started)
        )
        stream_publish_started = time.perf_counter()
        write_partial_pose_stream(
            path=args.partial_pose_out,
            replay_id=args.replay_id,
            drone_video=args.drone_video,
            expected_count=expected_count_for_stream(args),
            poses=partial_poses,
            complete=False,
            current_frame=current_frame_meta,
        )
        stage["stream_publish_ms"] = float(
            stage.get("stream_publish_ms") or 0.0
        ) + 1000.0 * (time.perf_counter() - stream_publish_started)

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
                "total_frame_ms": (
                    time.perf_counter() - foreground_started
                ) * 1000.0,
                "reason": output_rejection_reason or ("" if result.get("success") else "tsolve_failed"),
            },
            stage,
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
        max_sequential_catchup_frames = 12 if interactive_recovery else catchup_span
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
        foreground_started = time.perf_counter()
        nonlocal pending_global, current_pool, prev_gray, prev_case_id
        nonlocal last_center, consecutive_local_failures, global_relocalization_count
        nonlocal background_recovery_success_count, background_recovery_stale_count
        nonlocal stable_solve_point3d_ids
        if pending_global is None or not pending_global.get("done"):
            return
        # A live recovery result is consumed only when the loop has loaded a
        # current image.  Publishing the worker's old pose is never allowed.
        if interactive_recovery and (current_frame_idx is None or current_gray is None):
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
        stage["background_worker_ms"] = elapsed_ms
        accepted = pool is not None
        global_relocalization_count += 1
        recovery_age_frames = max(
            0,
            int(current_frame_idx if current_frame_idx is not None else processed_frames)
            - int(recovery.get("frame_idx", processed_frames)),
        )
        if interactive_recovery and accepted:
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
            current_route_context = patrol_route_gate.active_context()
            current_route_key = (
                LivePatrolRouteGate._key(current_route_context)
                if current_route_context is not None
                else None
            )
            source_route_key = recovery.get("route_key")
            source_translation_locked = bool(
                recovery.get("controller_translation_locked")
            )
            current_translation_locked = bool(
                current_route_context is not None
                and current_route_context.get("controller_translation_locked") is True
            )
            if source_route_key is not None and (
                current_route_key != source_route_key
                or current_translation_locked != source_translation_locked
            ):
                # The background worker was launched under a different patrol
                # leg or a different yaw/translation contract. Even with a
                # similar image heading, installing it here can replace a good
                # forward LK chain with correspondences from the preceding
                # turn—the exact frame-2921 Point-3 departure freeze.
                background_recovery_stale_count += 1
                stage["reason"] = "background_recovery_controller_phase_changed"
                print(
                    "BACKGROUND RECOVERY DISCARDED AFTER CONTROLLER PHASE CHANGE:",
                    json.dumps(
                        {
                            "source_frame": int(recovery["frame_idx"]),
                            "current_frame": int(current_frame_idx),
                            "source_route_key": source_route_key,
                            "current_route_key": current_route_key,
                            "source_translation_locked": source_translation_locked,
                            "current_translation_locked": current_translation_locked,
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
                        "method": "global_colmap_background_recovery_phase_discard",
                        "accepted": False,
                        "tracked_points": int(len(np.asarray(pool.get("xy", [])))),
                        "selected_points": 0,
                        "feature_extract_ms": stage.get("feature_extract_ms", 0.0),
                        "match_ms": stage.get("match_ms", 0.0),
                        "register_ms": stage.get("register_ms", 0.0),
                        "optical_flow_ms": 0.0,
                        "tsolve_ms": 0.0,
                        "total_frame_ms": (
                            time.perf_counter() - foreground_started
                        ) * 1000.0,
                        "reason": stage["reason"],
                    },
                    stage,
                )
                return
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
                        "total_frame_ms": (
                            time.perf_counter() - foreground_started
                        ) * 1000.0,
                        "reason": stage["reason"],
                    },
                    stage,
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
                        "total_frame_ms": (
                            time.perf_counter() - foreground_started
                        ) * 1000.0,
                        "reason": stage["reason"],
                    },
                    stage,
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
                    "total_frame_ms": (
                        time.perf_counter() - foreground_started
                    ) * 1000.0,
                    "reason": "",
                },
                stage,
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
                "total_frame_ms": (
                    time.perf_counter() - foreground_started
                ) * 1000.0,
                "reason": stage.get("reason", ""),
            },
            stage,
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
        # A deterministic frame-derived ID prevents correspondence cases from
        # colliding when an interrupted finite-video run resumes.
        case_id = f"instance_{frame_idx:06d}"
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
        # Publish once after this frame has a pose (or a held result).  The old
        # pre-solve publication rewrote the full pose history a second time but
        # contained no new localization result for the renderer.
        pace_wait_ms = pacer.wait_until(current_frame_time)
        if pace_wait_ms > 0:
            apply_background_global_recovery()
        frame_load_started = time.perf_counter()
        curr_gray = load_gray(frame)
        frame_load_ms = 1000.0 * (time.perf_counter() - frame_load_started)
        if current_pool is not None:
            try:
                focal_candidate = float(np.asarray(current_pool.get("K"), dtype=float)[0, 0])
                if math.isfinite(focal_candidate) and focal_candidate >= 100.0:
                    rotation_focal_px = focal_candidate
            except (TypeError, ValueError, IndexError):
                pass
        # Resolve the active patrol leg before optical yaw.  The validated
        # 09:30:17 route used continuous integration on legs 1 and 2, while
        # the later weak tail needs the stricter non-adjacent-frame reseed.
        route_context = patrol_route_gate.active_context()
        route_leg_index = (
            int(route_context.get("leg_index") or 0)
            if isinstance(route_context, dict)
            else 0
        )
        route_pose_epoch_key = None
        if isinstance(route_context, dict):
            try:
                candidate_epoch = int(route_context.get("route_pose_epoch") or 0)
                candidate_epoch_unix = float(
                    route_context.get("route_pose_epoch_unix")
                )
            except (TypeError, ValueError):
                candidate_epoch = 0
                candidate_epoch_unix = float("nan")
            if candidate_epoch > 0 and math.isfinite(candidate_epoch_unix):
                route_pose_epoch_key = (
                    candidate_epoch,
                    candidate_epoch_unix,
                )
        route_pose_epoch_transition = bool(
            route_pose_epoch_key is not None
            and route_pose_epoch_key != active_route_pose_epoch_key
        )
        route_frame_predates_epoch = (
            LivePatrolRouteGate.frame_predates_route_pose_epoch(
                route_context,
                current_frame_meta.get("received_unix"),
            )
        )
        if route_pose_epoch_transition:
            active_route_pose_epoch_key = route_pose_epoch_key
            # Do not integrate a yaw delta across an exact waypoint ownership
            # boundary. The route/stabilizer key below independently resets
            # visual progress and position bias to the controller's anchor.
            rotation_prev_gray = None
            rotation_prev_frame_time = None
        (
            rotation_heading_timing_valid,
            rotation_heading_frame_gap_seconds,
            rotation_heading_timing_policy_name,
        ) = rotation_heading_timing_policy(
            rotation_prev_frame_time,
            current_frame_time,
            route_leg_index=route_leg_index,
            max_gap_seconds=rotation_heading_max_frame_gap_seconds,
        )
        rotation_heading_timing_valid = bool(
            rotation_heading_timing_valid
            and not route_frame_predates_epoch
        )
        heading_flow_started = time.perf_counter()
        if rotation_heading_timing_valid:
            rotation_delta, rotation_tracks = optical_flow_yaw_delta(
                rotation_prev_gray,
                curr_gray,
                rotation_focal_px,
            )
        else:
            # Re-seed on the newest image without integrating a non-adjacent
            # frame pair.  This is the exact lag mode that made the displayed
            # Point-3 turn trail the physical DJI turn.
            rotation_delta, rotation_tracks = None, 0
        heading_flow_ms = 1000.0 * (time.perf_counter() - heading_flow_started)
        if route_frame_predates_epoch:
            # Buffered frames captured before the verified endpoint must not
            # seed optical yaw or recorded-route consensus for the new phase.
            rotation_prev_gray = None
            rotation_prev_frame_time = None
        else:
            rotation_prev_gray = curr_gray
            rotation_prev_frame_time = current_frame_time
        rotation_heading_tracks = rotation_tracks
        rotation_heading_delta_deg = math.degrees(rotation_delta) if rotation_delta is not None else None
        if rotation_delta is not None and rotation_heading is not None:
            rotation_heading = rotate_room_heading(rotation_heading, rotation_delta)
        if rotation_stabilizer_profile in {"live-atlas-141441", "live-atlas-141750"}:
            commanded_yaw_only, commanded_position_anchor = live_rotation_command_state(
                args.rotation_command_status_json
            )
        else:
            commanded_yaw_only, commanded_position_anchor = None, None
        # The route gate is a metric safety input even when prerecorded visual
        # recovery is disabled.  Keeping it conditional on the visual matcher
        # previously removed leg/anchor ownership from TSolve-only patrols.
        route_key = (
            patrol_route_gate._key(route_context)
            if route_context is not None
            else None
        )
        rotation_position_stabilizer.observe(
            delta_degrees=rotation_heading_delta_deg,
            tracks=rotation_heading_tracks,
            last_published_center=(
                last_output_pose.get("rcenter")
                if isinstance(last_output_pose, dict)
                else None
            ),
            commanded_yaw_only=commanded_yaw_only,
            commanded_position_anchor=commanded_position_anchor,
            route_key=route_key,
        )
        method = "optical_flow"
        stage = {
            "frame_load_ms": frame_load_ms,
            "heading_flow_ms": heading_flow_ms,
            "feature_extract_ms": 0.0,
            "match_ms": 0.0,
            "register_ms": 0.0,
            "optical_flow_ms": 0.0,
            "case_build_ms": 0.0,
            "case_output_ms": 0.0,
            "visual_route_ms": 0.0,
            "visual_heading_ms": 0.0,
            "route_logic_ms": 0.0,
            "local_recovery_ms": 0.0,
            "background_apply_ms": 0.0,
            "background_worker_ms": 0.0,
            "pose_update_ms": 0.0,
            "stream_publish_ms": 0.0,
            "pace_wait_ms": pace_wait_ms,
            "reason": "",
        }
        visual_observation: dict[str, Any] | None = None
        visual_stage: dict[str, Any] = {"reason": "visual_route_unavailable"}
        visual_progress_hint: float | None = None
        visual_attempted = False
        visual_supervision: dict[str, Any] = {}
        visual_heading_observation: dict[str, Any] | None = None
        visual_heading_stage: dict[str, Any] = {
            "reason": "visual_heading_unavailable"
        }
        # Keep the established 1->2->3 metric path unchanged.  The validated
        # weak leg 3->4 is continuously supervised by current-frame matches.
        # The complete-loop bank also contains the real Point-4 departure and
        # 4->1 frames.  On that leg route vision is a guarded fallback only:
        # it supplies absolute heading during the locked turn, and position
        # only during neutral recovery or after a rejected metric pose. TSolve
        # and optical flow continue to run on every frame.
        route_supervision_started = time.perf_counter()
        reference_frames_enabled = bool(
            patrol_reference_frames_enabled(route_context)
            and not route_frame_predates_epoch
        )
        controller_translation_locked = bool(
            isinstance(route_context, dict)
            and route_context.get("controller_translation_locked") is True
        )
        recovery_hover = bool(
            isinstance(route_context, dict)
            and route_context.get("recovery_hover") is True
        )
        should_match_route_position = bool(
            reference_frames_enabled
            and visual_route_position_recovery_needed(
                route_context,
                force_route_taught_recovery=force_route_taught_recovery,
            )
        )
        if (
            visual_route_recovery is not None
            and route_context is not None
            and reference_frames_enabled
            and should_match_route_position
        ):
            route_key = LivePatrolRouteGate._key(route_context)
            visual_progress_hint = (
                patrol_route_gate.last_progress
                if patrol_route_gate.last_key == route_key
                else None
            )
            visual_route_started = time.perf_counter()
            visual_observation, visual_stage = visual_route_recovery.recover(
                gray=curr_gray,
                segment_start=route_context["start"],
                segment_end=route_context["end"],
                segment_key=route_key,
                translation_locked=bool(route_context["translation_locked"]),
                progress_hint=visual_progress_hint,
                progress_ceiling=route_context.get(
                    "route_progress_command_ceiling"
                ),
                recovery_hover=bool(route_context.get("recovery_hover")),
                recovery_minimum_inliers=(
                    visual_route_temporal_recovery_minimum_inliers(
                        50 if route_leg_index == 4 else 90,
                        leg_index=route_leg_index,
                    )
                ),
                independent_progress=bool(
                    int(route_context.get("leg_index") or 0) in {3, 4}
                ),
                # Only Point 4->1 may use a progress-independent endpoint
                # proof when TSolve remains stale behind the physical drone.
                # Other legs retain their established route-window behavior.
                allow_endpoint_only_recovery=bool(route_leg_index == 4),
                sequence_index=frame_idx,
            )
            stage["visual_route_ms"] += 1000.0 * (
                time.perf_counter() - visual_route_started
            )
            visual_attempted = True
            if visual_stage.get("reason") == "visual_route_acquiring":
                visual_route_acquisition_hold_count += 1
            visual_supervision = visual_route_supervision_metadata(
                context=route_context,
                observation=visual_observation,
                diagnostic=visual_stage,
                progress_hint=visual_progress_hint,
                minimum_inliers=visual_route_recovery.minimum_inliers,
            )
        # During a commanded yaw, run only the small departure-heading bank.
        # Full-route position matching on the same locked frame cannot reveal
        # physical translation and used to duplicate ORB work at every turn.
        # Neutral recovery hover still takes the full position path above.
        heading_leg_index = int(route_context.get("leg_index") or 0) if route_context else 0
        if (
            visual_route_recovery is not None
            and route_context is not None
            and reference_frames_enabled
            and route_context.get("controller_translation_locked") is True
            and heading_leg_index in {1, 2, 3, 4}
        ):
            heading_minimum_inliers = visual_route_heading_minimum_inliers(
                visual_route_recovery.minimum_inliers,
                leg_index=heading_leg_index,
            )
            visual_heading_started = time.perf_counter()
            visual_heading_observation, visual_heading_stage = (
                visual_route_recovery.departure_heading_alignment(
                    gray=curr_gray,
                    segment_start=route_context["start"],
                    segment_end=route_context["end"],
                    focal_px=rotation_focal_px,
                    minimum_inliers=heading_minimum_inliers,
                )
            )
            stage["visual_heading_ms"] += 1000.0 * (
                time.perf_counter() - visual_heading_started
            )
            visual_supervision.update(
                visual_route_heading_metadata(
                    context=route_context,
                    observation=visual_heading_observation,
                    diagnostic=visual_heading_stage,
                    minimum_inliers=heading_minimum_inliers,
                    map_id=visual_route_recovery.map_id,
                    patrol_id=visual_route_recovery.patrol_id,
                    baseline_replay_id=visual_route_recovery.baseline_replay_id,
                )
            )
        # Give the small route bank first access to a stopped Point-4 recovery
        # view. It is current-frame, command-bounded, and typically finishes
        # in a few frames; launching full-map COLMAP at the same instant made
        # both paths compete for CPU and delayed the route evidence that
        # eventually recovered the latest flight. This grace is finite: after
        # 30 incoming frames, or 15 frames after the last strong route match,
        # the existing full-map fallback is available unchanged.
        route_visual_recovery_window_active = False
        if (
            route_leg_index == 4
            and recovery_hover
            and visual_attempted
            and isinstance(route_context, dict)
        ):
            current_recovery_key = LivePatrolRouteGate._key(route_context)
            if route_visual_recovery_window_key != current_recovery_key:
                route_visual_recovery_window_key = current_recovery_key
                route_visual_recovery_window_start_frame = frame_idx
            retained_route_hits = int(
                visual_supervision.get(
                    "route_visual_monitor_temporal_recovery_hits"
                )
                or 0
            )
            route_visual_recovery_window_active = bool(
                retained_route_hits > 0
                or (
                    route_visual_recovery_window_start_frame is not None
                    and frame_idx - route_visual_recovery_window_start_frame
                    < route_visual_recovery_grace_frames
                )
            )
        else:
            route_visual_recovery_window_key = None
            route_visual_recovery_window_start_frame = None
        visual_supervision = stabilize_visual_route_heading_for_render(
            visual_route_heading_render_state,
            visual_supervision,
            frame_index=frame_idx,
        )
        visual_primary_mode = weak_patrol_leg_visual_primary_mode(
            route_context=route_context,
            observation=visual_observation,
            last_pose=last_output_pose,
        )
        stable_solve_reset = False
        metric_checkpoint_wait = bool(
            args.wait_for_metric_checkpoint_recovery
            and route_context is not None
            and route_context.get("require_metric_pose") is True
        )
        lap_start_metric_rebootstrap = bool(
            metric_checkpoint_wait
            and route_context.get("lap_start_metric_rebootstrap") is True
        )
        stopped_metric_rebootstrap = bool(
            metric_checkpoint_wait
            and route_context.get("stopped_metric_rebootstrap") is True
        )
        absolute_metric_rebootstrap = bool(
            lap_start_metric_rebootstrap or stopped_metric_rebootstrap
        )
        route_supervision_total_ms = 1000.0 * (
            time.perf_counter() - route_supervision_started
        )
        stage["route_logic_ms"] += max(
            0.0,
            route_supervision_total_ms
            - stage["visual_route_ms"]
            - stage["visual_heading_ms"],
        )
        if metric_checkpoint_wait and pending_global is not None:
            checkpoint_route_key = LivePatrolRouteGate._key(route_context)
            if (
                absolute_metric_rebootstrap
                or pending_global.get("route_key") != checkpoint_route_key
            ):
                # A rematch launched on the previous 4->1 leg cannot satisfy
                # the new Point-1->2 metric checkpoint. Do not make the drone
                # hover until that obsolete full-map worker times out; discard
                # it when it finishes while current-frame compact recovery runs.
                pending_global["ignored"] = True
                pending_global["ignore_reason"] = (
                    "superseded_by_stopped_metric_rebootstrap"
                    if stopped_metric_rebootstrap
                    else "superseded_by_lap_start_metric_rebootstrap"
                    if lap_start_metric_rebootstrap
                    else "superseded_by_new_lap_metric_checkpoint"
                )
        if (
            visual_primary_mode is None
            and pending_global is not None
            and not pending_global.get("ignored")
            and (
                metric_checkpoint_wait
                or (
                    args.wait_for_background_recovery
                    and consecutive_local_failures
                    >= args.global_recovery_after_failures
                )
            )
        ):
            # Camera Path is a finite, reproducible simulation of incoming
            # frames.  Once the local 2D/3D pool is gone, consuming hundreds of
            # newer frames while a 30-80 second COLMAP rematch is pending makes
            # that result stale and leaves the visible camera frozen.  Hold the
            # frame consumer here instead.  The browser independently pauses at
            # its last trusted pose, so video, pose, and recovery remain on the
            # same frame without changing real-flight behaviour.
            recovery_wait_started = time.perf_counter()
            while (
                pending_global is not None
                and not pending_global.get("done")
            ):
                time.sleep(0.05)
            stage["recovery_wait_ms"] = (
                time.perf_counter() - recovery_wait_started
            ) * 1000.0
        background_apply_started = time.perf_counter()
        apply_background_global_recovery(
            current_frame_idx=frame_idx,
            current_gray=curr_gray,
        )
        stage["background_apply_ms"] += 1000.0 * (
            time.perf_counter() - background_apply_started
        )

        # A healthy-looking LK pool is not proof that the published metric
        # position caught up with a command. In both captured failures the
        # localizer retained 400-500 tracks while repeating the old turn
        # anchor for tens of seconds. When the bridge is neutrally hovering
        # specifically because post-command progress or endpoint geometry is
        # unresolved, launch a newest-frame global measurement immediately.
        # Optical flow continues to publish while this worker runs, so the
        # camera stream never blocks on COLMAP/SIFT.
        recovery_global_due = bool(
            interactive_recovery
            and isinstance(route_context, dict)
            and route_context.get("recovery_hover") is True
            and (
                route_context.get("post_translation_progress_recovery") is True
                or route_context.get("endpoint_position_recovery") is True
            )
            and last_center is not None
            and frame_idx - last_pose_recovery_global_frame
            >= args.pose_recovery_global_cooldown_frames
            and not route_visual_recovery_window_active
        )
        if recovery_global_due and pending_global is None:
            recovery_reason = (
                "endpoint_pose_recovery"
                if route_context.get("endpoint_position_recovery") is True
                else "post_translation_pose_stasis"
            )
            if schedule_background_global_recovery(
                frame_idx=frame_idx,
                frame=frame,
                query_name=query_name,
                curr_gray=curr_gray,
                reason=recovery_reason,
            ):
                last_pose_recovery_global_frame = frame_idx
                print(
                    "POSE-RECOVERY NEWEST-FRAME GLOBAL SCHEDULED:",
                    json.dumps(
                        {
                            "frame_index": frame_idx,
                            "reason": recovery_reason,
                            "cooldown_frames": (
                                args.pose_recovery_global_cooldown_frames
                            ),
                            "sift_max_num_features": (
                                args.sift_max_num_features
                            ),
                        }
                    ),
                    flush=True,
                )

        must_global = current_pool is None or prev_gray is None
        global_reason = "bootstrap" if must_global else ""
        if absolute_metric_rebootstrap:
            # A verified Point-1 endpoint changes route ownership, but the
            # previous raw optical pool may still belong to Point 4. It is not
            # eligible to seed lap 2: recover 2D->3D correspondences from the
            # current Point-1 image and the saved first-lap metric anchor.
            must_global = True
            if stopped_metric_rebootstrap:
                global_reason = "stopped_metric_rebootstrap"
            else:
                # Keep the explicit historical lap branch visible for the
                # source-level safety audit as well as for runtime logging.
                global_reason = "lap_start_metric_rebootstrap"
        periodic_refresh_due = periodic_feature_refresh_due(
            frame_index=frame_idx,
            frame_time=current_frame_time,
            last_frame_index=last_feature_extraction_frame,
            last_frame_time=last_feature_extraction_time,
            max_frame_interval=args.relocalize_every,
            max_time_interval_seconds=args.relocalize_every_seconds,
        )
        if (
            visual_primary_mode is None
            and frame_idx > 0
            and periodic_refresh_due
            and current_pool is not None
            and prev_gray is not None
        ):
            if (
                interactive_recovery
                and not args.blocking_global_recovery
                and not args.disable_background_recovery
            ):
                periodic_scheduled = schedule_background_global_recovery(
                    frame_idx=frame_idx,
                    frame=frame,
                    query_name=query_name,
                    curr_gray=curr_gray,
                    reason="periodic_feature_refresh",
                )
                if periodic_scheduled:
                    periodic_feature_refresh_count += 1
                    method = "optical_flow_periodic_feature_refresh_scheduled"
                    stage["reason"] = "periodic_feature_refresh_background"
                    print(
                        "PERIODIC FEATURE REFRESH SCHEDULED:",
                        json.dumps(
                            {
                                "frame_index": frame_idx,
                                "frame_time": current_frame_time,
                                "max_frame_interval": args.relocalize_every,
                                "max_time_interval_seconds": (
                                    args.relocalize_every_seconds
                                ),
                            }
                        ),
                        flush=True,
                    )
                elif pending_global is not None:
                    method = "optical_flow_periodic_feature_refresh_pending"
                    stage["reason"] = (
                        "periodic_feature_refresh_background_recovery_pending"
                    )
            elif pending_global is not None and not args.blocking_global_recovery:
                method = "periodic_global_recovery_pending"
                stage["reason"] = "periodic_relocalization_background_recovery_pending"
            else:
                must_global = True
                global_reason = "periodic_feature_refresh"
                periodic_feature_refresh_count += 1

        pool: dict[str, Any] | None = None
        proactive_fallback_pool: dict[str, Any] | None = None
        if (
            absolute_metric_rebootstrap
            and must_global
            and pending_global is None
            and (
                stopped_metric_rebootstrap
                or lap_start_metric_center is not None
            )
        ):
            checkpoint_stage: dict[str, Any] = {
                "reason": "lap_checkpoint_faiss_recovery_unavailable"
            }
            pool, checkpoint_stage = relocalizer.localize_faiss_current_frame(
                frame=frame,
                frame_idx=frame_idx,
                query_name=query_name,
                map_points=map_points,
                expected_center=(
                    None
                    if stopped_metric_rebootstrap
                    else lap_start_metric_center
                ),
            )
            stage.update(checkpoint_stage)
            global_relocalization_count += 1
            if pool is not None:
                pool[
                    "stopped_metric_rebootstrap"
                    if stopped_metric_rebootstrap
                    else "lap_start_metric_rebootstrap"
                ] = True
                must_global = False
                stable_solve_reset = True
                stable_solve_point3d_ids = None
                method = (
                    "stopped_rebootstrap_faiss_current_frame_tsolve"
                    if stopped_metric_rebootstrap
                    else "lap_checkpoint_faiss_current_frame_tsolve"
                )
                stage["reason"] = ""
                lap_checkpoint_metric_recovery_count += 1
                print(
                    "LAP CHECKPOINT CURRENT-FRAME METRIC RECOVERY:",
                    json.dumps(
                        {
                            "frame_index": frame_idx,
                            "source": "colmap_sift_faiss_current_frame_to_tsolve",
                            "inliers": pool.get("faiss_pnp_inliers"),
                            "unique_2d3d": pool.get("faiss_unique_matches"),
                            "source_images": pool.get("faiss_source_images"),
                            "center_step": pool.get("faiss_pnp_center_step"),
                            "recovery_ms": checkpoint_stage.get("total_ms"),
                        }
                    ),
                    flush=True,
                )
            else:
                # Stay at neutral hover and try the next newest image. Never
                # fall back to Point-4 LK or a global search constrained by its
                # stale center at this explicit loop-closure checkpoint.
                must_global = False
                if stopped_metric_rebootstrap:
                    method = "stopped_metric_rebootstrap_retry"
                else:
                    method = "lap_start_metric_rebootstrap_retry"
                stage["reason"] = str(
                    checkpoint_stage.get("reason")
                    or "lap_checkpoint_faiss_recovery_failed"
                )
        if absolute_metric_rebootstrap and pool is None and must_global:
            must_global = False
            method = (
                "stopped_metric_rebootstrap_waiting_for_anchor"
                if stopped_metric_rebootstrap
                else "lap_start_metric_rebootstrap_waiting_for_anchor"
            )
            stage["reason"] = (
                "stopped_metric_rebootstrap_anchor_unavailable"
                if stopped_metric_rebootstrap
                else "lap_start_metric_rebootstrap_anchor_unavailable"
            )
        if (
            visual_primary_mode is None
            and not must_global
            and pool is None
            and not absolute_metric_rebootstrap
            and current_pool is not None
            and prev_gray is not None
        ):
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
            required_track_points = required_tracking_points(
                current_pool,
                normal_minimum=args.min_track_points,
                solver_minimum=args.min_points,
            )
            if (
                tracked is not None
                and tracked_count >= required_track_points
                and tracked_ratio >= args.min_track_ratio
            ):
                pool = tracked
                method = "optical_flow"
                local_tracking_count += 1
                route_recovery_due = bool(
                    force_route_taught_recovery
                    and reference_frames_enabled
                    and route_context is not None
                    and route_context.get("controller_translation_locked") is not True
                    and frame_idx - last_route_rejection_recovery_attempt_frame
                    >= args.route_rejection_recovery_cooldown_frames
                )
                if route_recovery_due and last_center is not None:
                    last_route_rejection_recovery_attempt_frame = frame_idx
                    route_rejection_recovery_attempt_count += 1
                    route_recovered_pool: dict[str, Any] | None = None
                    route_recovery_stage: dict[str, Any] = {
                        "reason": "route_anchor_recovery_unavailable"
                    }
                    # The in-memory bank may already contain descriptors from
                    # the aliased optical track. Prefer the immutable recorded
                    # patrol banks when escaping a repeated route rejection,
                    # then use current-flight anchors only as a fallback.
                    full_loop_route_banks = [
                        recovery_bank
                        for recovery_bank in taught_recoveries[1:]
                        if "full_loop" in recovery_bank.bank_path.name
                    ]
                    route_recovery_banks = [
                        *(full_loop_route_banks or taught_recoveries[1:]),
                        online_recovery,
                    ]
                    for route_recovery_bank in route_recovery_banks:
                        local_recovery_started = time.perf_counter()
                        route_recovered_pool, route_recovery_stage = (
                            route_recovery_bank.recover(
                                gray=curr_gray,
                                K=np.asarray(current_pool["K"], dtype=float),
                                last_center=np.asarray(last_center, dtype=float),
                                max_step=args.global_recovery_max_step,
                            )
                        )
                        stage["local_recovery_ms"] += 1000.0 * (
                            time.perf_counter() - local_recovery_started
                        )
                        if route_recovered_pool is not None:
                            break
                    if route_recovered_pool is not None:
                        # Replace, rather than merge with, the rejected LK
                        # pool. Merging preserved the repeated-room alias and
                        # let it dominate the newly verified correspondences.
                        pool = route_recovered_pool
                        stable_solve_reset = True
                        stable_solve_point3d_ids = None
                        force_route_taught_recovery = False
                        consecutive_local_failures = 0
                        route_rejection_recovery_success_count += 1
                        method = "route_rejection_taught_consensus_recovery"
                        stage["reason"] = ""
                        if pending_global is not None:
                            pending_global["ignored"] = True
                            pending_global["ignore_reason"] = (
                                "superseded_by_current_frame_route_anchor_recovery"
                            )
                        print(
                            "ROUTE REJECTION CURRENT-FRAME ANCHOR RECOVERY:",
                            json.dumps(
                                {
                                    "frame_index": frame_idx,
                                    "anchor": route_recovery_stage.get("anchor_name"),
                                    "inliers": route_recovery_stage.get("inliers"),
                                    "consensus": route_recovery_stage.get("consensus_count"),
                                    "center_step": route_recovery_stage.get("center_step"),
                                    "replacement_points": int(
                                        len(np.asarray(pool.get("xy", [])))
                                    ),
                                }
                            ),
                            flush=True,
                        )
                    else:
                        print(
                            "ROUTE REJECTION CURRENT-FRAME RECOVERY RETRY NEEDED:",
                            json.dumps(
                                {
                                    "frame_index": frame_idx,
                                    "reason": route_recovery_stage.get("reason"),
                                    "next_retry_frame": frame_idx
                                    + args.route_rejection_recovery_cooldown_frames,
                                }
                            ),
                            flush=True,
                        )
                proactive_cooldown = int(args.proactive_relocalize_cooldown_frames)
                if tracked_count <= 40:
                    proactive_cooldown = min(proactive_cooldown, 4)
                elif tracked_count <= 120:
                    proactive_cooldown = min(proactive_cooldown, 8)
                proactive_due = (
                    args.proactive_relocalize_points > 0
                    and tracked_count <= args.proactive_relocalize_points
                    and frame_idx - last_proactive_relocalize_frame
                    >= proactive_cooldown
                )
                if proactive_due:
                    global_reason = f"proactive_pool_refresh_{tracked_count}"
                    # Rebuild correspondences from anchors learned during this
                    # very flight while LK still has a valid local pool.  If
                    # we wait for the hard tracking minimum, repeated room
                    # textures can force an expensive global search after the
                    # local evidence has already disappeared.
                    proactive_pool: dict[str, Any] | None = None
                    proactive_stage: dict[str, Any] = {
                        "reason": "online_anchor_refresh_unavailable"
                    }
                    if last_center is not None and len(online_recovery.anchor_names) >= 3:
                        local_recovery_started = time.perf_counter()
                        proactive_pool, proactive_stage = online_recovery.recover(
                            gray=curr_gray,
                            K=np.asarray(current_pool["K"], dtype=float),
                            last_center=np.asarray(last_center, dtype=float),
                            max_step=args.global_recovery_max_step,
                        )
                        stage["local_recovery_ms"] += 1000.0 * (
                            time.perf_counter() - local_recovery_started
                        )
                    local_anchor_verified = proactive_pool is not None
                    if local_anchor_verified:
                        pool = merge_verified_tracking_pool(tracked, proactive_pool)
                        stable_solve_reset = True
                        consecutive_local_failures = 0
                        print(
                            "PROACTIVE LOCAL MAP ANCHOR VERIFIED:",
                            json.dumps(
                                {
                                    "frame_index": frame_idx,
                                    "tracked_points": tracked_count,
                                    "trigger_points": args.proactive_relocalize_points,
                                    "anchor": proactive_stage.get("anchor_name"),
                                    "inliers": proactive_stage.get("inliers"),
                                    "consensus": proactive_stage.get("consensus_count"),
                                    "center_step": proactive_stage.get("center_step"),
                                    "lk_input": pool.get("verified_lk_input_points"),
                                    "lk_inliers": pool.get("verified_lk_inlier_points"),
                                    "merged_points": len(np.asarray(pool.get("xy", []))),
                                }
                            ),
                            flush=True,
                        )
                        method = "online_live_map_anchor_proactive_refresh"
                        stage["reason"] = ""
                        proactive_relocalization_count += 1
                        last_proactive_relocalize_frame = frame_idx
                    elif not args.blocking_global_recovery and schedule_background_global_recovery(
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
                        # One pending rematch owns this refresh window.  Do not
                        # repeat SIFT anchor verification on every intervening
                        # video frame while that worker is still running.
                        proactive_relocalization_count += 1
                        last_proactive_relocalize_frame = frame_idx
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
                taught_stage: dict[str, Any] = {"reason": "local_anchor_recovery_unavailable"}
                rotation_recovery_cooling_down = bool(
                    route_context is not None
                    and route_context.get("controller_translation_locked") is True
                    and frame_idx - last_rotation_taught_recovery_attempt_frame
                    < args.rotation_recovery_cooldown_frames
                )
                if rotation_recovery_cooling_down:
                    taught_stage = {"reason": "rotation_recovery_cooldown"}
                elif taught_recoveries and last_center is not None:
                    if (
                        route_context is not None
                        and route_context.get("controller_translation_locked") is True
                    ):
                        last_rotation_taught_recovery_attempt_frame = frame_idx
                    recovery_banks = (
                        taught_recoveries
                        if reference_frames_enabled
                        else [online_recovery]
                    )
                    for taught_recovery in recovery_banks:
                        local_recovery_started = time.perf_counter()
                        recovered_pool, taught_stage = taught_recovery.recover(
                            gray=curr_gray,
                            K=np.asarray(current_pool["K"], dtype=float),
                            last_center=np.asarray(last_center, dtype=float),
                            max_step=args.global_recovery_max_step,
                        )
                        stage["local_recovery_ms"] += 1000.0 * (
                            time.perf_counter() - local_recovery_started
                        )
                        if recovered_pool is not None:
                            pool = recovered_pool
                            break
                if pool is not None:
                    method = (
                        "online_live_map_anchor_recovery"
                        if taught_recovery is online_recovery
                        else "taught_patrol_consensus_recovery"
                    )
                    stable_solve_reset = True
                    consecutive_local_failures = 0
                    stage["reason"] = ""
                    print(
                        "LOCAL MAP ANCHOR RECOVERY ACCEPTED:",
                        json.dumps(
                            {
                                "frame_index": frame_idx,
                                "anchor": taught_stage.get("anchor_name"),
                                "inliers": taught_stage.get("inliers"),
                                "consensus": taught_stage.get("consensus_count"),
                                "center_step": taught_stage.get("center_step"),
                            }
                        ),
                        flush=True,
                    )
                else:
                    consecutive_local_failures += 1
                    stage["reason"] = str(flow_stage.get("reason") or "too_few_tracked_points")
                    if taught_recoveries:
                        stage["reason"] += f"_{taught_stage.get('reason') or 'taught_recovery_failed'}"
                    if tracked_count < required_track_points:
                        stage["reason"] = f"{stage['reason']}_tracked_{tracked_count}_lt_{required_track_points}"
                    elif tracked_ratio < args.min_track_ratio:
                        stage["reason"] = f"{stage['reason']}_ratio_{tracked_ratio:.3f}_lt_{args.min_track_ratio:.3f}"
                    if consecutive_local_failures >= args.global_recovery_after_failures:
                        global_reason = f"recovery_after_{consecutive_local_failures}_local_failures"
                        if args.disable_background_recovery and interactive_recovery:
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
            and interactive_recovery
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

        if (
            must_global
            and args.blocking_global_recovery
            and last_output_pose is not None
            and args.blocking_global_retry_interval > 1
            and frame_idx - last_blocking_global_attempt_frame
            < args.blocking_global_retry_interval
        ):
            must_global = False
            method = "blocking_global_retry_interval_hold"
            stage["reason"] = (
                f"blocking_global_retry_interval_{args.blocking_global_retry_interval}_hold"
            )

        if must_global:
            last_blocking_global_attempt_frame = frame_idx
            method = f"global_colmap_{global_reason or 'recovery'}"
            recovery_step_limit = float(args.global_recovery_max_step)
            if (
                args.global_recovery_max_speed > 0.0
                and current_frame_time is not None
                and last_output_time is not None
            ):
                elapsed = max(0.0, current_frame_time - last_output_time)
                recovery_step_limit = min(
                    max(args.global_recovery_max_step, args.global_recovery_max_total_step),
                    max(
                        args.global_recovery_max_step,
                        0.18 + args.global_recovery_max_speed * elapsed,
                    ),
                )
            refreshed_pool, global_stage = relocalizer.localize(
                frame=frame,
                frame_idx=frame_idx,
                query_name=query_name,
                map_points=map_points,
                last_center=last_center,
                recovery_max_step=recovery_step_limit,
            )
            record_feature_extraction(frame_idx, current_frame_time)
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
            if (
                not visual_attempted
                and visual_route_recovery is not None
                and reference_frames_enabled
                and route_context is not None
            ):
                route_key = LivePatrolRouteGate._key(route_context)
                visual_progress_hint = (
                    patrol_route_gate.last_progress
                    if patrol_route_gate.last_key == route_key
                    else None
                )
                visual_route_started = time.perf_counter()
                visual_observation, visual_stage = visual_route_recovery.recover(
                    gray=curr_gray,
                    segment_start=route_context["start"],
                    segment_end=route_context["end"],
                    segment_key=route_key,
                    translation_locked=bool(route_context["translation_locked"]),
                    progress_hint=visual_progress_hint,
                    progress_ceiling=route_context.get(
                        "route_progress_command_ceiling"
                    ),
                    recovery_hover=bool(route_context.get("recovery_hover")),
                    recovery_minimum_inliers=(
                        visual_route_temporal_recovery_minimum_inliers(
                            50
                            if int(route_context.get("leg_index") or 0) == 4
                            else 90,
                            leg_index=int(
                                route_context.get("leg_index") or 0
                            ),
                        )
                    ),
                    independent_progress=bool(
                        int(route_context.get("leg_index") or 0) in {3, 4}
                    ),
                    sequence_index=frame_idx,
                )
                stage["visual_route_ms"] += 1000.0 * (
                    time.perf_counter() - visual_route_started
                )
                visual_attempted = True
                if visual_stage.get("reason") == "visual_route_acquiring":
                    visual_route_acquisition_hold_count += 1
                visual_supervision = visual_route_supervision_metadata(
                    context=route_context,
                    observation=visual_observation,
                    diagnostic=visual_stage,
                    progress_hint=visual_progress_hint,
                    minimum_inliers=visual_route_recovery.minimum_inliers,
                )
            if (
                visual_observation is not None
                and not (
                    route_context is not None
                    and route_context.get("require_metric_pose") is True
                )
            ):
                visual_pose = accepted_visual_route_recovery_pose(
                    last_pose=last_output_pose,
                    current_frame=current_frame_meta,
                    observation=visual_observation,
                    supervision=visual_supervision,
                    route_context=route_context,
                    rotation_heading=rotation_heading,
                    rotation_heading_tracks=rotation_heading_tracks,
                    rotation_position_stabilizer=rotation_position_stabilizer,
                    route_gate=patrol_route_gate,
                    visual_recovery=visual_route_recovery,
                )
                if visual_pose is not None:
                    visual_route_recovery_count += 1
                    last_output_pose = visual_pose
                    partial_poses.append(visual_pose)
                    append_stage(
                        stage_csv,
                        {
                            "frame_index": frame_idx,
                            "case_id": visual_pose["instance_id"],
                            "image_name": query_name,
                            "time_sec": current_frame_time,
                            "method": "patrol_visual_route_recovery",
                            "accepted": True,
                            "tracked_points": int(visual_observation["inliers"]),
                            "selected_points": int(visual_observation["inliers"]),
                            "feature_extract_ms": visual_stage.get("total_ms", 0.0),
                            "match_ms": 0.0,
                            "register_ms": 0.0,
                            "optical_flow_ms": stage.get("optical_flow_ms", 0.0),
                            "tsolve_ms": 0.0,
                            "total_frame_ms": (time.perf_counter() - frame_t0) * 1000.0,
                            "reason": "",
                        },
                        {
                            **stage,
                            "extracted_features": visual_stage.get("query_features"),
                            "matched_features": visual_observation.get("ratio_matches"),
                            "pnp_inliers": visual_observation.get("inliers"),
                        },
                    )
                    print(
                        "PATROL VISUAL ROUTE RECOVERY ACCEPTED:",
                        json.dumps(
                            {
                                "frame_index": frame_idx,
                                "progress": visual_observation["progress"],
                                "inliers": visual_observation["inliers"],
                                "anchor": visual_observation["anchor_name"],
                            }
                        ),
                        flush=True,
                    )
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
                stage,
            )
            print("FRAME SKIPPED:", json.dumps(rejected_row), flush=True)
            held = held_pose_from_last(
                last_pose=latest_published_pose(partial_poses, last_output_pose),
                current_frame=current_frame_meta,
                reason=str(rejected_row["reason"]),
                rotation_heading=rotation_heading,
                rotation_heading_tracks=rotation_heading_tracks,
                rotation_heading_delta_deg=rotation_heading_delta_deg,
            )
            if held is not None:
                held = attach_rotation_only_hint(held)
                held.update(visual_supervision)
                held = rotation_position_stabilizer.apply(held)
                held = apply_visual_route_heading_alignment(
                    held, visual_supervision
                )
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
        case_build_started = time.perf_counter()
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
        stage["case_build_ms"] += 1000.0 * (
            time.perf_counter() - case_build_started
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
            record_feature_extraction(frame_idx, current_frame_time)
            stage["feature_extract_ms"] += float(global_stage.get("feature_extract_ms", 0.0))
            stage["match_ms"] += float(global_stage.get("match_ms", 0.0))
            stage["register_ms"] += float(global_stage.get("register_ms", 0.0))
            for count_key in (
                "extracted_features",
                "matched_features",
                "pnp_inliers",
            ):
                if global_stage.get(count_key) is not None:
                    stage[count_key] = global_stage[count_key]
            method = "global_colmap"
            if pool is not None:
                pool["K"] = np.asarray(pool["K"], dtype=np.float64)
                case_build_started = time.perf_counter()
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
                stage["case_build_ms"] += 1000.0 * (
                    time.perf_counter() - case_build_started
                )
                if not case.get("accepted") and stable_solve_point3d_ids is not None:
                    print(
                        "global relocalization did not contain the locked solve set; resetting solve set for this track segment",
                        flush=True,
                    )
                    case_build_started = time.perf_counter()
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
                    stage["case_build_ms"] += 1000.0 * (
                        time.perf_counter() - case_build_started
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
            case_build_started = time.perf_counter()
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
            stage["case_build_ms"] += 1000.0 * (
                time.perf_counter() - case_build_started
            )
            stable_solve_reset = bool(case.get("accepted"))
        if not case.get("accepted"):
            if (
                visual_observation is not None
                and visual_route_recovery is not None
                and route_context is not None
            ):
                visual_pose = accepted_visual_route_recovery_pose(
                    last_pose=last_output_pose,
                    current_frame=current_frame_meta,
                    observation=visual_observation,
                    supervision=visual_supervision,
                    route_context=route_context,
                    rotation_heading=rotation_heading,
                    rotation_heading_tracks=rotation_heading_tracks,
                    rotation_position_stabilizer=rotation_position_stabilizer,
                    route_gate=patrol_route_gate,
                    visual_recovery=visual_route_recovery,
                )
                if visual_pose is not None:
                    visual_route_recovery_count += 1
                    last_output_pose = visual_pose
                    partial_poses.append(visual_pose)
                    append_stage(
                        stage_csv,
                        {
                            "frame_index": frame_idx,
                            "case_id": visual_pose["instance_id"],
                            "image_name": query_name,
                            "time_sec": current_frame_time,
                            "method": "patrol_visual_route_recovery_case_fallback",
                            "accepted": True,
                            "tracked_points": int(visual_observation["inliers"]),
                            "selected_points": int(visual_observation["inliers"]),
                            "feature_extract_ms": visual_stage.get("total_ms", 0.0),
                            "match_ms": 0.0,
                            "register_ms": 0.0,
                            "optical_flow_ms": stage.get("optical_flow_ms", 0.0),
                            "tsolve_ms": 0.0,
                            "total_frame_ms": (time.perf_counter() - frame_t0) * 1000.0,
                            "reason": "",
                        },
                        {
                            **stage,
                            "extracted_features": visual_stage.get("query_features"),
                            "matched_features": visual_observation.get("ratio_matches"),
                            "pnp_inliers": visual_observation.get("inliers"),
                        },
                    )
                    print(
                        "PATROL VISUAL ROUTE RECOVERY ACCEPTED AFTER CASE FAILURE:",
                        json.dumps(
                            {
                                "frame_index": frame_idx,
                                "case_reason": str(
                                    case.get("reason") or "case_selection_failed"
                                ),
                                "progress": visual_observation["progress"],
                                "inliers": visual_observation["inliers"],
                                "anchor": visual_observation["anchor_name"],
                            }
                        ),
                        flush=True,
                    )
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
                stage,
            )
            print("FRAME SKIPPED:", json.dumps(rejected_row), flush=True)
            held = held_pose_from_last(
                last_pose=latest_published_pose(partial_poses, last_output_pose),
                current_frame=current_frame_meta,
                reason=str(rejected_row["reason"]),
                rotation_heading=rotation_heading,
                rotation_heading_tracks=rotation_heading_tracks,
                rotation_heading_delta_deg=rotation_heading_delta_deg,
            )
            if held is not None:
                held = attach_rotation_only_hint(held)
                partial_poses.append(rotation_position_stabilizer.apply(held))
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
        case_output_started = time.perf_counter()
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
        append_manifest_row(args.inputs_out_dir, manifest_rows[-1])
        instance_dir = instances_dir / case_id
        copy_case_to_instance(Path(case["case_dir"]), instance_dir)
        stage["case_output_ms"] += 1000.0 * (
            time.perf_counter() - case_output_started
        )

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
            fork_on_miss=not interactive_recovery,
            root_candidate_profile=args.tsolve_root_profile,
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
        route_observation = None
        if result.get("success"):
            output_rejection_reason, last_output_center, last_output_time, route_observation = route_guarded_output_rejection(
                case=case,
                result=result,
                previous_center=last_output_center,
                previous_time=last_output_time,
                previous_pose=last_output_pose,
                route_gate=patrol_route_gate,
                room_transform=room_transform,
                output_center_bias=output_center_bias,
                metric_route_room_bias=(
                    rotation_position_stabilizer.metric_route_room_bias()
                ),
                max_step=args.output_max_step,
                max_speed=args.output_max_speed,
                objective_threshold=args.output_objective_threshold,
                post_yaw_reanchor_cap=args.patrol_turn_max_position_drift,
                lap_start_metric_rebootstrap=bool(
                    pool.get("lap_start_metric_rebootstrap")
                ),
                stopped_metric_rebootstrap=bool(
                    pool.get("stopped_metric_rebootstrap")
                ),
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
        if (
            output_accepted
            and args.calibrate_output_to_first_global_anchor
            and output_center_bias is None
        ):
            reference_center = pool_reference_center(pool)
            if reference_center is None:
                # write_case_from_pool preserves COLMAP's full registration
                # in input.json even when the reduced optical/TSolve pool no
                # longer carries qvec/tvec.  Camera Path must still publish
                # from that stronger absolute first-frame anchor.
                reference_center = pose_reference_center_from_case(case)
            solved_center = result_center_from_rt(result)
            if reference_center is not None and solved_center is not None:
                candidate_bias = reference_center - solved_center
                bias_length = float(np.linalg.norm(candidate_bias))
                # The full COLMAP bootstrap uses every accepted map match and
                # is the stronger absolute anchor. Correct only the small
                # bounded bias introduced by reducing that pool to the TSolve
                # case; a larger difference is evidence of a bad root and must
                # continue through the normal rejection/recovery path.
                if bias_length <= 0.35:
                    output_center_bias = candidate_bias
                    print(
                        "OUTPUT ANCHOR CALIBRATED:",
                        json.dumps(
                            {
                                "frame_index": frame_idx,
                                "bias_world": candidate_bias.tolist(),
                                "bias_m": bias_length,
                                "registered_points": pool.get("colmap_registered_points"),
                            }
                        ),
                        flush=True,
                    )
        if output_accepted:
            if case.get("solve_set_reselected"):
                stable_solve_point3d_ids = np.asarray(
                    case["selected_point3d_ids"], dtype=np.int64
                )
                stable_solve_reset = True
            consecutive_tracking_reset_rejections = 0
            # A rejected TSolve root must not poison later optical-flow
            # correspondences or move the map-search reference center.
            current_pool = cap_tracking_pool(
                pool,
                args.track_pool_size,
                keep_point3d_ids=stable_solve_point3d_ids,
            )
            prev_gray = curr_gray
            prev_case_id = case_id
            if (
                isinstance(route_observation, dict)
                and route_observation.get(
                    "route_continuity_preserved_tracking_center"
                )
                is True
            ):
                reference_update_reason = "controller_locked_turn_flow_anchor_only"
            else:
                last_center, reference_update_reason = update_tracking_reference_center(
                    case=case,
                    result=result,
                    previous_center=last_center,
                )
            if (
                lap_start_metric_center is None
                and isinstance(route_context, dict)
                and int(route_context.get("lap") or 0) == 1
                and int(route_context.get("leg_index") or 0) == 1
            ):
                candidate_loop_center = result_center_from_rt(result)
                candidate_loop_K = np.asarray(pool.get("K", []), dtype=float)
                if (
                    candidate_loop_center is not None
                    and candidate_loop_K.size == 9
                    and np.all(np.isfinite(candidate_loop_K))
                ):
                    lap_start_metric_center = np.asarray(
                        candidate_loop_center,
                        dtype=float,
                    ).reshape(3)
                    lap_start_metric_K = candidate_loop_K.reshape(3, 3).copy()
                    print(
                        "LAP START METRIC LOOP-CLOSURE ANCHOR SAVED:",
                        json.dumps(
                            {
                                "frame_index": frame_idx,
                                "map_center": lap_start_metric_center.tolist(),
                            }
                        ),
                        flush=True,
                    )
            # The saved patrol can be edited after its original taught run.
            # Learn sparse visual anchors from the route that is actually being
            # flown, before the current optical pool disappears in a turn.
            live_pool_count = int(len(np.asarray(current_pool.get("xy", []))))
            learn_interval = max(1, int(args.online_recovery_learn_interval))
            learn_minimum = required_tracking_points(
                current_pool,
                normal_minimum=args.min_track_points,
                solver_minimum=args.min_points,
            )
            if (
                taught_recoveries
                and live_pool_count >= learn_minimum
                and frame_idx - last_online_recovery_anchor_frame >= learn_interval
            ):
                learned = 0
                for taught_recovery in taught_recoveries:
                    learned += taught_recovery.learn_anchor(
                        gray=curr_gray,
                        xy=np.asarray(current_pool.get("xy", [])),
                        p3d=np.asarray(current_pool.get("p3d", [])),
                        point3d_ids=np.asarray(current_pool.get("point3d_ids", [])),
                        anchor_name=f"online/{args.replay_id}/{query_name}",
                    )
                if learned > 0:
                    last_online_recovery_anchor_frame = frame_idx
                    online_recovery_anchor_count += 1
                    print(
                        "ONLINE RECOVERY ANCHOR LEARNED:",
                        json.dumps(
                            {
                                "frame_index": frame_idx,
                                "tracked_points": live_pool_count,
                                "descriptor_points": learned,
                                "anchor_count": online_recovery_anchor_count,
                            }
                        ),
                        flush=True,
                    )
        elif result.get("success"):
            consecutive_tracking_reset_rejections = next_output_tracking_reset_streak(
                output_rejection_reason,
                consecutive_tracking_reset_rejections,
                tracking_reset_hard_motion_cap(args.output_max_step),
            )
            reference_update_reason = "held_rejected_pose_not_trusted"
            if route_rejection_can_advance_flow_anchor(output_rejection_reason):
                # Preserve the verified optical correspondences while keeping
                # the false room pose and route progress fully rejected.
                current_pool = cap_tracking_pool(
                    pool,
                    args.track_pool_size,
                    keep_point3d_ids=stable_solve_point3d_ids,
                )
                prev_gray = curr_gray
                prev_case_id = case_id
                reference_update_reason = "route_rejected_pose_flow_anchor_only"
                if consecutive_local_failures >= max(1, int(args.global_recovery_after_failures)):
                    force_route_taught_recovery = True
                    refresh_scheduled = schedule_background_global_recovery(
                        frame_idx=frame_idx,
                        frame=frame,
                        query_name=query_name,
                        curr_gray=curr_gray,
                        reason=f"route_guard_rejection_{output_rejection_reason}",
                    )
                    if refresh_scheduled:
                        print(
                            "ROUTE REJECTION BACKGROUND REFRESH:",
                            json.dumps(
                                {
                                    "frame_index": frame_idx,
                                    "consecutive_rejections": consecutive_local_failures,
                                    "reason": output_rejection_reason,
                                    "tracked_points": int(len(np.asarray(current_pool.get("xy", [])))),
                                }
                            ),
                            flush=True,
                        )
            elif rejected_output_can_advance_flow_anchor(
                output_rejection_reason,
                consecutive_tracking_reset_rejections,
                args.global_recovery_after_failures,
                tracking_reset_hard_motion_cap(args.output_max_step),
            ):
                # Keep consecutive optical frame anchors even while a suspect
                # 3D position is held.  Otherwise one rejected moving frame
                # creates a larger optical gap and turns a safe hold into a
                # permanent localization freeze.
                current_pool = cap_tracking_pool(
                    pool,
                    args.track_pool_size,
                    keep_point3d_ids=stable_solve_point3d_ids,
                )
                prev_gray = curr_gray
                prev_case_id = case_id
                reference_update_reason = "output_rejected_pose_flow_anchor_only"
            else:
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
                            "consecutive_rejections": consecutive_tracking_reset_rejections,
                            "reason": output_rejection_reason,
                        }
                    ),
                    flush=True,
                )
        else:
            # An algebraic TSolve miss does not invalidate the optical 2D->3D
            # correspondences that already passed tracking and robust PnP
            # consensus.  In live mode the former `fork_on_miss` retry could
            # block one frame for 6-27 seconds; then leaving prev_gray stale
            # made the next frame harder and repeated the freeze.  Advance the
            # optical image anchor, hold only the public pose, and force a
            # freshly selected TSolve subset on the next frame.
            current_pool = cap_tracking_pool(
                pool,
                args.track_pool_size,
                keep_point3d_ids=None,
            )
            prev_gray = curr_gray
            prev_case_id = case_id
            stable_solve_reset = True
            stable_solve_point3d_ids = None
            reference_update_reason = "tsolve_miss_flow_anchor_reselect"

        if (
            visual_route_recovery is not None
            and visual_recovery_supersedes_stalled_metric_pose(
                last_pose=last_output_pose,
                observation=visual_observation,
                route_context=route_context,
                output_rejection_reason=output_rejection_reason,
                metric_route_observation=route_observation,
                departure_floor_repair_available=bool(
                    route_context is not None
                    and patrol_route_gate.departure_floor_reconciled_key
                    != LivePatrolRouteGate._key(route_context)
                ),
            )
        ):
            visual_pose = accepted_visual_route_recovery_pose(
                last_pose=last_output_pose,
                current_frame=current_frame_meta,
                observation=visual_observation,
                supervision=visual_supervision,
                route_context=route_context,
                rotation_heading=rotation_heading,
                rotation_heading_tracks=rotation_heading_tracks,
                rotation_position_stabilizer=rotation_position_stabilizer,
                route_gate=patrol_route_gate,
                visual_recovery=visual_route_recovery,
                metric_route_observation=route_observation,
            )
            if visual_pose is not None:
                visual_pose["route_visual_superseded_stalled_metric_pose"] = True
                visual_pose["route_visual_primary_authority"] = True
                visual_pose["route_visual_metric_rejection_reason"] = (
                    output_rejection_reason
                )
                visual_route_recovery_count += 1
                consecutive_local_failures = 0
                last_output_pose = visual_pose
                partial_poses.append(visual_pose)
                append_stage(
                    stage_csv,
                    {
                        "frame_index": frame_idx,
                        "case_id": visual_pose["instance_id"],
                        "image_name": query_name,
                        "time_sec": current_frame_time,
                        "method": "patrol_visual_route_recovery_stalled_metric",
                        "accepted": True,
                        "tracked_points": int(visual_observation["inliers"]),
                        "selected_points": int(visual_observation["inliers"]),
                        "feature_extract_ms": visual_stage.get("total_ms", 0.0),
                        "match_ms": 0.0,
                        "register_ms": 0.0,
                        "optical_flow_ms": stage.get("optical_flow_ms", 0.0),
                        "tsolve_ms": solve_ms,
                        "total_frame_ms": (time.perf_counter() - frame_t0) * 1000.0,
                        "reason": "",
                    },
                    {
                        **stage,
                        "extracted_features": visual_stage.get("query_features"),
                        "matched_features": visual_observation.get("ratio_matches"),
                        "pnp_inliers": visual_observation.get("inliers"),
                    },
                )
                print(
                    "PATROL VISUAL ROUTE RECOVERY SUPERSEDED STALLED METRIC POSE:",
                    json.dumps(
                        {
                            "frame_index": frame_idx,
                            "progress": visual_observation["progress"],
                            "inliers": visual_observation["inliers"],
                            "metric_rejection": output_rejection_reason,
                            "published_center": visual_pose["rcenter"],
                            "reconciling": visual_pose.get(
                                "route_visual_reconciling"
                            ),
                        }
                    ),
                    flush=True,
                )
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

        pose_update_started = time.perf_counter()
        stopped_absolute_position_reset = bool(
            isinstance(pool, dict)
            and pool.get("stopped_metric_rebootstrap") is True
        )
        lap_start_absolute_position_reset = (
            accept_lap_start_absolute_metric_position(
                rotation_position_stabilizer,
                pool=pool,
                output_accepted=output_accepted,
            )
        )
        if lap_start_absolute_position_reset and isinstance(current_pool, dict):
            current_pool.pop("lap_start_metric_rebootstrap", None)
            current_pool.pop("stopped_metric_rebootstrap", None)
        if output_rejection_reason is not None and last_output_pose is not None:
            pose_payload = held_pose_from_last(
                last_pose=latest_published_pose(partial_poses, last_output_pose),
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
                output_center_bias=output_center_bias,
            )
        if pose_payload is None:
            pose_payload = partial_pose_from_result(
                case,
                result,
                room_transform=room_transform,
                output_rejection_reason=output_rejection_reason or "no_previous_pose_to_hold",
                output_center_bias=output_center_bias,
            )
        pose_payload = attach_rotation_only_hint(pose_payload)
        pose_payload = rotation_position_stabilizer.apply(pose_payload)
        if lap_start_absolute_position_reset:
            if stopped_absolute_position_reset:
                pose_payload["stopped_global_absolute_metric_position"] = True
                pose_payload["stopped_global_turn_bias_cleared"] = True
            else:
                pose_payload["lap_start_absolute_metric_position"] = True
                pose_payload["lap_start_turn_bias_cleared"] = True
        pose_payload = patrol_route_gate.constrain_published_pose(
            pose_payload,
            route_observation,
        )
        pose_payload.update(visual_supervision)
        pose_payload = apply_visual_route_heading_alignment(
            pose_payload, visual_supervision
        )
        # TSolve is the live position authority. Verified route observations
        # remain supervision metadata, or a bounded fallback when the metric
        # result was rejected/missing.
        partial_poses.append(pose_payload)
        if pose_payload.get("success") and (
            pose_payload.get("center") or pose_payload.get("rcenter")
        ):
            last_output_pose = pose_payload
            # Keep the optical heading anchor independent from the map-pose
            # publication path.  A TSolve candidate can pass this process's
            # broad output gate yet still be rejected later by the DJI bridge
            # as a physical motion jump.  Such a candidate must not replace
            # the heading used to safely finish an in-place turn.
            if pose_payload.get("center"):
                update_rotation_reference_from_accepted_pose(pose_payload)
        stage["pose_update_ms"] += 1000.0 * (
            time.perf_counter() - pose_update_started
        )
        stream_publish_started = time.perf_counter()
        write_partial_pose_stream(
            path=args.partial_pose_out,
            replay_id=args.replay_id,
            drone_video=args.drone_video,
            expected_count=expected_count_for_stream(args),
            poses=partial_poses,
            complete=False,
            current_frame=current_frame_meta,
        )
        stage["stream_publish_ms"] += 1000.0 * (
            time.perf_counter() - stream_publish_started
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
            stage,
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
        final_recovery_wait = 4.0 if interactive_recovery else 15.0
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
        "sift_max_num_features": args.sift_max_num_features,
        "relocalize_every": args.relocalize_every,
        "relocalize_every_seconds": args.relocalize_every_seconds,
        "periodic_feature_refresh_count": periodic_feature_refresh_count,
        "min_track_points": args.min_track_points,
        "min_track_ratio": args.min_track_ratio,
        "flow_window": args.flow_window,
        "flow_levels": args.flow_levels,
        "flow_iterations": args.flow_iterations,
        "global_recovery_after_failures": args.global_recovery_after_failures,
        "background_recovery_timeout_seconds": args.background_recovery_timeout_seconds,
        "global_recovery_max_step": float(args.global_recovery_max_step),
        "global_recovery_max_speed": float(args.global_recovery_max_speed),
        "global_recovery_max_total_step": float(args.global_recovery_max_total_step),
        "output_max_step": float(args.output_max_step),
        "output_max_speed": float(args.output_max_speed),
        "output_objective_threshold": float(args.output_objective_threshold),
        "blocking_global_recovery": bool(args.blocking_global_recovery),
        "blocking_global_retry_interval": int(args.blocking_global_retry_interval),
        "wait_for_background_recovery": bool(args.wait_for_background_recovery),
        "disable_background_recovery": bool(args.disable_background_recovery),
        "pace_replay": bool(args.pace_replay and not args.follow_dir),
        "pace_scale": float(args.pace_scale),
        "global_relocalization_count": global_relocalization_count,
        "background_recovery_count": background_recovery_count,
        "background_recovery_success_count": background_recovery_success_count,
        "background_recovery_stale_count": background_recovery_stale_count,
        "proactive_relocalize_points": args.proactive_relocalize_points,
        "proactive_relocalize_cooldown_frames": args.proactive_relocalize_cooldown_frames,
        "pose_recovery_global_cooldown_frames": (
            args.pose_recovery_global_cooldown_frames
        ),
        "proactive_relocalization_count": proactive_relocalization_count,
        "proactive_relocalization_success_count": proactive_relocalization_success_count,
        "proactive_relocalization_fallback_count": proactive_relocalization_fallback_count,
        "background_recovery_pending_at_finish": bool(pending_global is not None),
        "local_tracking_count": local_tracking_count,
        "online_recovery_anchor_count": online_recovery_anchor_count,
        "visual_route_recovery_count": visual_route_recovery_count,
        "visual_route_acquisition_hold_count": visual_route_acquisition_hold_count,
        "route_rejection_recovery_cooldown_frames": (
            args.route_rejection_recovery_cooldown_frames
        ),
        "route_rejection_recovery_attempt_count": (
            route_rejection_recovery_attempt_count
        ),
        "route_rejection_recovery_success_count": (
            route_rejection_recovery_success_count
        ),
        "lap_checkpoint_metric_recovery_count": (
            lap_checkpoint_metric_recovery_count
        ),
        # Backward-compatible summary field for older viewer/audit exports.
        "lap_checkpoint_taught_recovery_count": (
            lap_checkpoint_metric_recovery_count
        ),
        "rotation_position_locked_frames": rotation_position_stabilizer.locked_frames,
        "rotation_position_reanchor_count": rotation_position_stabilizer.reanchor_count,
        "rotation_position_stabilizer_profile": rotation_stabilizer_profile,
        "rotation_command_status_json": (
            str(args.rotation_command_status_json) if args.rotation_command_status_json else None
        ),
        "patrol_route_baseline": (
            str(args.patrol_route_baseline) if args.patrol_route_baseline else None
        ),
        "patrol_route_gate_enabled": patrol_route_gate.enabled,
        "patrol_route_accepted_count": patrol_route_gate.accepted_count,
        "patrol_route_rejected_count": patrol_route_gate.rejected_count,
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
