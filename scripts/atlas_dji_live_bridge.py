#!/usr/bin/env python3
"""Bridge DJI MSDK Remote live video into ATLAS frame streams.

This script connects to the Android MSDKRemote/OpenDJI app, receives decoded
camera frames, and writes them in the
same frame-bank format used by the ATLAS TSolve replay pipeline:

    data/dji_live/<session>/query_frames/query_000000.jpg
    data/dji_live/<session>/query_frames/frames.csv

It also writes a browser-visible preview:

    viewer/public/live_dji/latest.jpg
    viewer/public/live_dji/status.json

It also accepts explicit takeoff/land/hover/mission command packets from the
local ATLAS app.  Mission movement is guarded by fresh TSolve pose updates and
uses tiny RC pulses; missing or stale localization causes bounded hover/recovery
before motion continues.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import queue
import re
import signal
import socket
import sys
import threading
import time
import traceback
import types
import uuid
from pathlib import Path
from typing import Any, Callable

import cv2


ROOT = Path(__file__).resolve().parents[1]
WORKSTATION_OPENDJI_ROOT = Path("/Users/yamromano/Desktop/DJI-MSDK-to-PC-main")
VENDORED_OPENDJI_ROOT = ROOT / "vendor" / "opendji"
DEFAULT_OPENDJI_ROOT = (
    WORKSTATION_OPENDJI_ROOT
    if (WORKSTATION_OPENDJI_ROOT / "OpenDJI.py").is_file()
    else VENDORED_OPENDJI_ROOT
)
IMAGE_EXTS = {".jpg", ".jpeg"}
TAKEOFF_VERTICAL_SPEED = 0.03
ENDPOINT_UNDERSHOOT_CORRECTION_THRESHOLD_M = 0.08
TAKEOFF_STEP_SECONDS = 0.50
TAKEOFF_MAX_ASSIST_SECONDS = 16.0
PRE_LAND_STABILIZE_SECONDS = 1.50
GUIDED_DEFAULT_POSE_MAX_AGE_SECONDS = 2.5
GUIDED_DEFAULT_PULSE_SECONDS = 0.28
GUIDED_DEFAULT_MAX_FORWARD_RC = 0.055
GUIDED_DEFAULT_MAX_YAW_RC = 0.04
GUIDED_DEFAULT_MAX_VERTICAL_RC = 0.025
GUIDED_DEFAULT_MAX_STEP_SECONDS = 3.0
GUIDED_DEFAULT_POSE_RECOVERY_SECONDS = 45.0
GUIDED_RECOVERY_HOVER_SECONDS = 0.20
GUIDED_YAW_SIGN_CONFIRMATION_PULSES = 3
GUIDED_BASELINE_CATCHUP_DISAGREEMENT = 0.55
CONTROL_NEUTRAL_ACK_TIMEOUT_SECONDS = 0.50
CONTROL_RECONNECT_TIMEOUT_SECONDS = 0.45
CONTROL_RECONNECT_ATTEMPTS = 3


class ControlLinkSafetyError(RuntimeError):
    """A motion command had an uncertain control-link outcome.

    Non-zero RC commands must never be retried after a socket error because the
    phone may already have applied the first command.  The attached fields let
    the UI distinguish a confirmed recovered hover from a failure to confirm
    neutral sticks at all.
    """

    def __init__(
        self,
        message: str,
        *,
        neutral_confirmed: bool,
        control_link_recovered: bool,
    ) -> None:
        super().__init__(message)
        self.neutral_confirmed = bool(neutral_confirmed)
        self.control_link_recovered = bool(control_link_recovered)
        self.requires_relocalization = True


class StopFlag:
    def __init__(self) -> None:
        self.stop = False

    def handler(self, signum, frame) -> None:  # noqa: ANN001
        self.stop = True


class DecodedFrameListener:
    """Track actual decoder callbacks instead of repeatedly reading a cached frame.

    OpenDJI.getFrame() returns the most recently decoded ndarray forever.  When
    the Android H264 stream stalls, polling that method at 10 FPS therefore
    writes hundreds of byte-identical JPEGs and makes a frozen camera look like
    a healthy live source.  OpenDJI's listener is invoked only when its decoder
    produces a new frame, which gives the bridge the freshness signal it needs.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sequence = 0
        self._frame: Any = None
        self._received_unix: float | None = None

    def onValue(self, value: Any) -> None:  # OpenDJI EventListener interface
        with self._lock:
            self._sequence += 1
            self._frame = value
            self._received_unix = time.time()

    def latest(self) -> tuple[int, Any, float | None]:
        with self._lock:
            return self._sequence, self._frame, self._received_unix

    def age_seconds(self, now: float | None = None) -> float | None:
        with self._lock:
            received_unix = self._received_unix
        if received_unix is None:
            return None
        return max(0.0, float(now if now is not None else time.time()) - received_unix)


def live_video_motion_safety_issue(
    frame_age_seconds: float | None,
    *,
    maximum_age_seconds: float = 1.25,
) -> str | None:
    """Return a blocking reason when live video is not genuinely updating."""
    if frame_age_seconds is None:
        return "no fresh decoded DJI video frame is available; refusing flight movement"
    if not math.isfinite(float(frame_age_seconds)):
        return "DJI video freshness is invalid; refusing flight movement"
    if float(frame_age_seconds) > max(0.1, float(maximum_age_seconds)):
        return (
            f"DJI video stream is frozen ({float(frame_age_seconds):.1f}s without a new "
            "decoded frame); refusing flight movement"
        )
    return None


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_bytes(data)
        tmp.replace(path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def file_sha256(path: Path) -> str | None:
    """Return a reproducible runtime fingerprint without mutating the file."""
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def enemy_detection_control_enabled(path: Path) -> bool:
    """Fail closed: live inference runs only after an explicit operator opt-in."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return False
    return payload.get("enabled") is True


def read_altitude(drone: Any, OpenDJI: Any) -> str | None:
    try:
        return drone.getValue(OpenDJI.MODULE_FLIGHTCONTROLLER, "AircraftLocation3D")
    except Exception:
        return None


def parse_altitude_m(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        value = raw.get("altitude")
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    text = str(raw).strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict) and "altitude" in value:
            return float(value["altitude"])
    except Exception:
        pass
    match = re.search(r'"altitude"\s*:\s*([-+]?\d+(?:\.\d+)?)', text)
    if match:
        return float(match.group(1))
    return None


def climb_to_requested_height(
    drone: Any,
    OpenDJI: Any,
    altitude_before: Any,
    target_height_m: Any,
) -> dict[str, Any]:
    if target_height_m is None:
        return {"enabled": False, "reason": "no requested height"}
    target_height_m = max(0.1, min(2.0, float(target_height_m)))
    ground_alt = parse_altitude_m(altitude_before)
    if ground_alt is None:
        return {
            "enabled": False,
            "reason": "altitude telemetry unavailable before takeoff",
        }

    target_abs = ground_alt + target_height_m
    time.sleep(3.0)
    current_raw = read_altitude(drone, OpenDJI)
    current_alt = parse_altitude_m(current_raw)
    if current_alt is None:
        return {
            "enabled": False,
            "reason": "altitude telemetry unavailable after takeoff",
            "target_height_m": target_height_m,
            "ground_altitude_m": ground_alt,
            "target_altitude_m": target_abs,
        }
    if current_alt >= target_abs - 0.15:
        return {
            "enabled": True,
            "reached": True,
            "reason": "takeoff altitude already reached requested guard height",
            "target_height_m": target_height_m,
            "ground_altitude_m": ground_alt,
            "target_altitude_m": target_abs,
            "final_altitude_m": current_alt,
            "move_steps": 0,
        }

    steps = 0
    control_enabled = False
    last_alt = current_alt
    started = time.time()
    try:
        drone.enableControl(True)
        control_enabled = True
        while time.time() - started < TAKEOFF_MAX_ASSIST_SECONDS:
            current_raw = read_altitude(drone, OpenDJI)
            current_alt = parse_altitude_m(current_raw)
            if current_alt is not None:
                last_alt = current_alt
                if current_alt >= target_abs - 0.15:
                    break
            execute_rc_pulse(
                drone,
                du=TAKEOFF_VERTICAL_SPEED,
                seconds=TAKEOFF_STEP_SECONDS,
            )
            steps += 1
    finally:
        try:
            neutral_hover(drone, 0.0)
        except Exception:
            pass
        if control_enabled:
            try:
                drone.disableControl(True)
            except Exception:
                pass

    return {
        "enabled": True,
        "reached": last_alt >= target_abs - 0.15,
        "target_height_m": target_height_m,
        "ground_altitude_m": ground_alt,
        "target_altitude_m": target_abs,
        "final_altitude_m": last_alt,
        "move_steps": steps,
        "elapsed_seconds": time.time() - started,
    }


def clamp_float(value: Any, default: float, lo: float, hi: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = default
    return max(lo, min(hi, out))


def mission_arrival_radii(
    arrival_radius: float,
    arrival_deadband: float,
    *,
    patrol_stage: str,
    strict_target: bool = False,
) -> tuple[float, float]:
    """Return hard/soft radii without letting entry stop outside the loop gate.

    A loop may use the indoor soft deadband at its intermediate patrol points,
    but the Point-1 entry leg must leave enough localization margin for the
    repeatable loop. Connected patrol packets tag that one-time cruise as the
    entry stage while later loop cruises retain the normal indoor deadband.
    """
    hard_radius = max(0.0, float(arrival_radius))
    if strict_target:
        precision_radius = min(hard_radius, 0.15)
        return precision_radius, precision_radius
    if str(patrol_stage or "").strip().lower() == "entry":
        entry_radius = min(hard_radius, 0.20)
        return entry_radius, entry_radius
    return hard_radius, hard_radius + max(0.0, float(arrival_deadband))


def dji_control_response_ok(response: Any) -> bool:
    if response is None:
        return True
    text = str(response).lower()
    return "error" not in text and "fail" not in text and "denied" not in text


def dji_control_response_reason(response: Any) -> str:
    text = str(response)
    if "CONTROL_AUTH_RC_NOT_P_MODE" in text or "VirtualStickEnabled:-36872" in text:
        return (
            "DJI refused virtual-stick control because the RC/drone is not in P/Normal positioning mode. "
            "Set the controller flight-mode switch to Normal/P mode, not Sport/Cine, then start Live ATLAS again."
        )
    return f"DJI refused virtual-stick control: {text}"


def stabilized_pose_safety_issue(
    pose: dict[str, Any] | None,
    *,
    max_horizontal_disagreement: float = 0.35,
) -> str | None:
    """Reject a synthetic/frozen position before it can guide translation."""
    if not isinstance(pose, dict):
        return None
    if pose.get("rotation_position_locked") or pose.get("translation_allowed") is False:
        return "localizer room position is rotation-frozen; refusing patrol translation"
    if (
        pose.get("route_visual_monitor_required") is True
        and str(pose.get("pose_source") or "") == "patrol_visual_route_recovery"
    ):
        try:
            monitor_leg = int(pose.get("route_visual_monitor_leg_index") or 0)
        except (TypeError, ValueError):
            monitor_leg = 0
        monitor_label = f"patrol leg {monitor_leg}" if monitor_leg > 0 else "patrol leg"
        try:
            monitor_inliers = int(pose.get("route_visual_monitor_inliers") or 0)
            temporal_monitor = bool(
                pose.get("route_visual_monitor_temporal_recovery") is True
                or pose.get("route_visual_temporal_recovery") is True
            )
            monitor_minimum = max(
                (
                    50
                    if temporal_monitor and monitor_leg == 4
                    else (90 if temporal_monitor else 120)
                ),
                int(pose.get("route_visual_monitor_minimum_inliers") or 0),
            )
        except (TypeError, ValueError):
            return f"{monitor_label} baseline supervision metadata is invalid; refusing translation"
        if pose.get("route_visual_monitor_verified") is not True:
            reason = str(pose.get("route_visual_monitor_reason") or "not verified")
            return f"{monitor_label} baseline supervision is not verified ({reason}); refusing translation"
        if monitor_inliers < monitor_minimum:
            return (
                f"{monitor_label} baseline supervision has {monitor_inliers} inliers, below "
                f"the {monitor_minimum} gate; refusing translation"
            )
        # Appearance progress is intentionally conservative and can lag
        # metric motion on a long, visually repetitive wall.  It is valid for
        # checking the active leg and for explicit recovery, but must never
        # override or veto an otherwise continuous route-gated TSolve center.
    if str(pose.get("pose_source") or "") == "patrol_visual_route_recovery":
        try:
            inliers = int(pose.get("route_visual_inliers") or 0)
            pose_leg = int(pose.get("route_visual_monitor_leg_index") or 0)
            temporal_recovery = bool(
                pose.get("route_visual_temporal_recovery") is True
            )
            minimum = max(
                (
                    50
                    if temporal_recovery and pose_leg == 4
                    else (90 if temporal_recovery else 120)
                ),
                int(pose.get("route_visual_minimum_inliers") or 0),
            )
            hits = int(pose.get("route_visual_acquisition_hits") or 0)
            required_hits = (
                max(
                    5,
                    int(
                        pose.get(
                            "route_visual_temporal_recovery_required_hits"
                        )
                        or 0
                    ),
                )
                if temporal_recovery
                else 2
            )
            progress = float(pose.get("route_visual_progress"))
        except (TypeError, ValueError):
            return "visual patrol recovery metadata is invalid; refusing translation"
        if pose.get("route_visual_verified") is not True:
            return "visual patrol recovery is not verified; refusing translation"
        if inliers < minimum:
            return (
                f"visual patrol recovery has {inliers} inliers, below the {minimum} gate; "
                "refusing translation"
            )
        if hits < required_hits:
            required_label = "two" if required_hits == 2 else str(required_hits)
            return (
                "visual patrol recovery needs "
                f"{required_label} consistent frames; refusing translation"
            )
        if not math.isfinite(progress) or progress < 0.0 or progress > 1.0:
            return "visual patrol recovery progress is outside the active leg"

    def finite_center(value: Any) -> list[float] | None:
        if not isinstance(value, (list, tuple)) or len(value) < 3:
            return None
        try:
            center = [float(value[0]), float(value[1]), float(value[2])]
        except (TypeError, ValueError):
            return None
        return center if all(math.isfinite(item) for item in center) else None

    published = finite_center(pose.get("rcenter"))
    raw = finite_center(pose.get("rotation_raw_rcenter"))
    if published is None or raw is None:
        return None
    disagreement = math.hypot(published[0] - raw[0], published[2] - raw[2])
    if disagreement > max(0.0, float(max_horizontal_disagreement)):
        if (
            pose.get("rotation_anchor_is_position_truth") is True
            and pose.get("rotation_anchor_commanded") is True
            and pose.get("rotation_position_source") == "post_yaw_room_bias"
        ):
            # A controller-confirmed yaw-only command cannot translate the
            # aircraft. TSolve's monocular center may change coordinate roots
            # during that turn, so the saved command anchor and its constant
            # room bias are the valid position track. This exception never
            # applies while the pose is frozen above, nor to an optical-only
            # inferred turn.
            return None
        return (
            "published/raw localizer positions disagree by "
            f"{disagreement:.3f} map units; refusing patrol translation"
        )
    return None


def baseline_supervised_pose_jump_issue(
    pose: dict[str, Any] | None,
    *,
    step: float,
    base_step_limit: float,
    disagreement_limit: float = GUIDED_BASELINE_CATCHUP_DISAGREEMENT,
) -> str | None:
    """Reject a skipped-frame catch-up only when the recorded route contradicts it.

    Ordinary per-frame TSolve motion remains authoritative.  This additional
    gate is deliberately narrower: it applies only when a candidate exceeds
    the normal 0.30 m step, and the independently matched patrol baseline is
    both strong and more than 0.55 m away.  That is the exact combination in
    Live ATLAS 16:42:09's false Point-1-to-2 jump.
    """
    if not isinstance(pose, dict) or float(step) <= max(0.0, float(base_step_limit)):
        return None
    if pose.get("route_visual_monitor_required") is not True:
        return None
    if pose.get("route_visual_monitor_verified") is not True:
        return None
    try:
        inliers = int(pose.get("route_visual_monitor_inliers") or 0)
        minimum = max(120, int(pose.get("route_visual_monitor_minimum_inliers") or 0))
        disagreement = float(pose.get("route_visual_monitor_disagreement_m"))
    except (TypeError, ValueError):
        return None
    if inliers < minimum or not math.isfinite(disagreement):
        return None
    if disagreement <= max(0.0, float(disagreement_limit)):
        return None
    try:
        leg = int(pose.get("route_visual_monitor_leg_index") or 0)
    except (TypeError, ValueError):
        leg = 0
    label = f"patrol leg {leg}" if leg > 0 else "patrol leg"
    return (
        f"{label} baseline contradicts a {float(step):.3f} map-unit catch-up "
        f"({disagreement:.3f} map-unit disagreement); refusing translation"
    )


def guided_command_pose_safety_issue(
    gate: dict[str, Any] | None,
    *,
    yaw: float = 0.0,
    lr: float = 0.0,
    bf: float = 0.0,
    du: float = 0.0,
) -> str | None:
    """Allow a stabilized pose for yaw/hover, never for room translation.

    Live ATLAS 14:17:50 deliberately anchored room position during in-place
    rotations.  That is useful for heading control, but a later regression
    let the same synthetic position authorize forward pulses.  Keep the
    reference turn behavior while enforcing the safety rule at the final RC
    command boundary where the intended motion is known.
    """
    del yaw, du  # These channels do not translate in the room X/Z plane.
    if abs(float(lr or 0.0)) <= 1e-6 and abs(float(bf or 0.0)) <= 1e-6:
        return None
    pose = gate.get("pose") if isinstance(gate, dict) else None
    return stabilized_pose_safety_issue(pose)


def pose_observation_order_key(pose: dict[str, Any] | None) -> tuple:
    """Return the capture order of a pose, independent of append order.

    Background recovery can finish after newer live frames and append its old
    held pose at the end of ``poses_partial.json``.  List order is therefore
    not camera order.  Patrol heading and translation gates must follow the
    frame captured most recently, with timestamps used only to distinguish
    multiple observations published for the same frame.
    """
    if not isinstance(pose, dict):
        return (0, -1, 0, float("-inf"), 0, float("-inf"))

    match = re.search(r"(\d+)(?:\.[^.]+)?$", str(pose.get("image_name") or ""))
    frame_index = int(match.group(1)) if match is not None else -1

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


def ordered_pose_observations(poses: object) -> list[dict[str, Any]]:
    """Sort valid pose records by captured-frame time, preserving ties."""
    if not isinstance(poses, list):
        return []
    return sorted(
        (pose for pose in poses if isinstance(pose, dict)),
        key=pose_observation_order_key,
    )


def latest_tsolve_pose_gate(
    pose_stream_path: Path | None,
    max_age_seconds: float,
    max_recent_hold_frames: int = 2,
    *,
    allow_rotation_frozen: bool = False,
) -> dict[str, Any]:
    if pose_stream_path is None:
        return {"ok": False, "reason": "mission has no TSolve pose stream path"}
    if not pose_stream_path.exists():
        return {"ok": False, "reason": f"TSolve pose stream not found: {pose_stream_path}"}
    try:
        payload = json.loads(pose_stream_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "reason": f"TSolve pose stream unreadable: {exc}"}
    updated = payload.get("updated_at")
    try:
        updated_f = float(updated)
    except (TypeError, ValueError):
        updated_f = 0.0
    age = time.time() - updated_f if updated_f > 0 else float("inf")
    if age > max_age_seconds:
        return {
            "ok": False,
            "reason": f"latest TSolve pose is stale ({age:.2f}s > {max_age_seconds:.2f}s)",
            "age_seconds": age,
        }
    poses = ordered_pose_observations(payload.get("poses"))
    latest_pose = poses[-1] if poses else None
    processed_count = payload.get("processed_count")
    if latest_pose is None:
        return {"ok": False, "reason": "no accepted TSolve pose is available yet", "age_seconds": age}

    def pose_room_center(pose: dict[str, Any] | None) -> list[float] | None:
        if not isinstance(pose, dict):
            return None
        value = pose.get("rcenter")
        if not isinstance(value, (list, tuple)) or len(value) < 3:
            return None
        try:
            out = [float(value[0]), float(value[1]), float(value[2])]
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(v) and abs(v) < 1e9 for v in out):
            return None
        return out

    def pose_is_real(pose: dict[str, Any] | None) -> bool:
        return bool(
            isinstance(pose, dict)
            and pose.get("success")
            and not pose.get("held_pose")
            and pose_room_center(pose) is not None
        )

    def motion_heading_for_pose(pose: dict[str, Any], index: int | None) -> list[float] | None:
        center = pose_room_center(pose)
        if center is None:
            return None
        if index is None:
            try:
                index = poses.index(pose)
            except ValueError:
                index = len(poses) - 1
        for prev in reversed(poses[max(0, index - 16):index]):
            if not pose_is_real(prev):
                continue
            prev_center = pose_room_center(prev)
            if prev_center is None:
                continue
            dx = center[0] - prev_center[0]
            dz = center[2] - prev_center[2]
            dist = math.hypot(dx, dz)
            if math.isfinite(dist) and dist >= 0.16:
                return [dx / dist, 0.0, dz / dist]
        return None

    def pose_payload(pose: dict[str, Any], *, fallback: bool = False, hold_count: int = 0) -> dict[str, Any]:
        try:
            index = poses.index(pose)
        except ValueError:
            index = len(poses) - 1
        raw_heading = pose.get("rheading")
        return {
            "instance_id": pose.get("instance_id"),
            "time_sec": pose.get("time_sec"),
            "received_unix": pose.get("received_unix"),
            "rcenter": pose.get("rcenter"),
            # Heading must describe the localized camera/drone orientation.
            # Motion direction is not heading: using it creates a feedback loop
            # where a sideways drift is interpreted as the new forward axis.
            "rheading": raw_heading,
            "rheading_raw": pose.get("rheading_raw") or raw_heading,
            "rheading_source": pose.get("rheading_source") or "tsolve_rotation",
            "center": pose.get("center"),
            "R": pose.get("R"),
            "t": pose.get("t"),
            "rotation_raw_rcenter": pose.get("rotation_raw_rcenter"),
            "rotation_position_anchor": pose.get("rotation_position_anchor"),
            "rotation_position_locked": bool(pose.get("rotation_position_locked")),
            "translation_allowed": pose.get("translation_allowed"),
            "rotation_position_bias": pose.get("rotation_position_bias"),
            "rotation_position_source": pose.get("rotation_position_source"),
            "rotation_reanchor_rejected": bool(pose.get("rotation_reanchor_rejected")),
            "rotation_reanchored_after_turn": bool(
                pose.get("rotation_reanchored_after_turn")
            ),
            "rotation_anchor_is_position_truth": bool(
                pose.get("rotation_anchor_is_position_truth")
            ),
            "rotation_anchor_commanded": bool(pose.get("rotation_anchor_commanded")),
            # Optical yaw is carried independently from route position.  The
            # patrol control loop needs these fields when a visual-route pose
            # is primary; omitting them silently reverted steering to the
            # older baseline frame heading even with hundreds of fresh tracks.
            "rotation_heading": pose.get("rotation_heading"),
            "rotation_heading_source": pose.get("rotation_heading_source"),
            "rotation_heading_tracks": pose.get("rotation_heading_tracks"),
            "rotation_heading_delta_deg": pose.get("rotation_heading_delta_deg"),
            "pose_source": pose.get("pose_source"),
            "route_visual_verified": pose.get("route_visual_verified"),
            "route_visual_progress": pose.get("route_visual_progress"),
            "route_visual_published_progress": pose.get(
                "route_visual_published_progress"
            ),
            "route_visual_progress_lag": pose.get("route_visual_progress_lag"),
            "route_visual_translation_safe": pose.get(
                "route_visual_translation_safe"
            ),
            "route_visual_weak_endpoint_recovery": pose.get(
                "route_visual_weak_endpoint_recovery"
            ),
            "route_visual_temporal_recovery": pose.get(
                "route_visual_temporal_recovery"
            ),
            "route_visual_temporal_recovery_hits": pose.get(
                "route_visual_temporal_recovery_hits"
            ),
            "route_visual_temporal_recovery_required_hits": pose.get(
                "route_visual_temporal_recovery_required_hits"
            ),
            "route_visual_inliers": pose.get("route_visual_inliers"),
            "route_visual_ratio_matches": pose.get("route_visual_ratio_matches"),
            "route_visual_anchor": pose.get("route_visual_anchor"),
            "route_visual_source_frame": pose.get("route_visual_source_frame"),
            "route_visual_acquisition_hits": pose.get("route_visual_acquisition_hits"),
            "route_visual_minimum_inliers": pose.get("route_visual_minimum_inliers"),
            "route_visual_map_id": pose.get("route_visual_map_id"),
            "route_visual_patrol_id": pose.get("route_visual_patrol_id"),
            "route_visual_baseline_replay_id": pose.get(
                "route_visual_baseline_replay_id"
            ),
            "route_visual_monitor_required": pose.get("route_visual_monitor_required"),
            "route_visual_monitor_verified": pose.get("route_visual_monitor_verified"),
            "route_visual_monitor_reason": pose.get("route_visual_monitor_reason"),
            "route_visual_monitor_inliers": pose.get("route_visual_monitor_inliers"),
            "route_visual_monitor_minimum_inliers": pose.get(
                "route_visual_monitor_minimum_inliers"
            ),
            "route_visual_monitor_temporal_recovery": pose.get(
                "route_visual_monitor_temporal_recovery"
            ),
            "route_visual_monitor_temporal_recovery_hits": pose.get(
                "route_visual_monitor_temporal_recovery_hits"
            ),
            "route_visual_monitor_temporal_recovery_required_hits": pose.get(
                "route_visual_monitor_temporal_recovery_required_hits"
            ),
            "route_visual_monitor_progress": pose.get("route_visual_monitor_progress"),
            "route_visual_monitor_tsolve_progress": pose.get(
                "route_visual_monitor_tsolve_progress"
            ),
            "route_visual_monitor_disagreement_m": pose.get(
                "route_visual_monitor_disagreement_m"
            ),
            "route_visual_monitor_leg_index": pose.get("route_visual_monitor_leg_index"),
            "route_visual_monitor_translation_locked": pose.get(
                "route_visual_monitor_translation_locked"
            ),
            # These fields are produced by the progress-independent whole-leg
            # endpoint verifier.  The patrol controller consumes the compact
            # pose payload returned here, not the raw pose stream.  Omitting
            # them made taught_endpoint_arrival_verified() impossible to
            # satisfy even while the raw stream reported hundreds of verified
            # endpoint observations.
            "route_visual_endpoint_guarded": pose.get(
                "route_visual_endpoint_guarded"
            ),
            "route_visual_endpoint_guard_progress": pose.get(
                "route_visual_endpoint_guard_progress"
            ),
            "route_visual_endpoint_safe_prearrival_progress": pose.get(
                "route_visual_endpoint_safe_prearrival_progress"
            ),
            "route_visual_endpoint_checked": pose.get(
                "route_visual_endpoint_checked"
            ),
            "route_visual_endpoint_verified": pose.get(
                "route_visual_endpoint_verified"
            ),
            "route_visual_endpoint_match_consensus_verified": pose.get(
                "route_visual_endpoint_match_consensus_verified"
            ),
            "route_visual_endpoint_view_geometry_verified": pose.get(
                "route_visual_endpoint_view_geometry_verified"
            ),
            "route_visual_endpoint_view_scale_min": pose.get(
                "route_visual_endpoint_view_scale_min"
            ),
            "route_visual_endpoint_view_scale_max": pose.get(
                "route_visual_endpoint_view_scale_max"
            ),
            "route_visual_endpoint_hits": pose.get("route_visual_endpoint_hits"),
            "route_visual_endpoint_required_hits": pose.get(
                "route_visual_endpoint_required_hits"
            ),
            "route_visual_endpoint_minimum_inliers": pose.get(
                "route_visual_endpoint_minimum_inliers"
            ),
            "route_visual_endpoint_candidate_progress": pose.get(
                "route_visual_endpoint_candidate_progress"
            ),
            "route_visual_endpoint_best_progress": pose.get(
                "route_visual_endpoint_best_progress"
            ),
            "route_visual_endpoint_best_inliers": pose.get(
                "route_visual_endpoint_best_inliers"
            ),
            "route_visual_endpoint_best_anchor": pose.get(
                "route_visual_endpoint_best_anchor"
            ),
            "route_visual_verified_rewind": pose.get(
                "route_visual_verified_rewind"
            ),
            "route_visual_verified_rewind_progress": pose.get(
                "route_visual_verified_rewind_progress"
            ),
            "route_visual_verified_rewind_inliers": pose.get(
                "route_visual_verified_rewind_inliers"
            ),
            "route_visual_verified_rewind_hits": pose.get(
                "route_visual_verified_rewind_hits"
            ),
            "route_visual_verified_rewind_required_hits": pose.get(
                "route_visual_verified_rewind_required_hits"
            ),
            "route_verified_visual_rewind_applied": pose.get(
                "route_verified_visual_rewind_applied"
            ),
            **({"recent_hold_fallback_pose": True, "trailing_hold_frames": hold_count} if fallback else {}),
        }

    if not (latest_pose.get("success") and not latest_pose.get("held_pose")):
        trailing_holds = 0
        fallback_pose: dict[str, Any] | None = None
        for pose in reversed(poses):
            if not isinstance(pose, dict):
                continue
            if pose.get("success") and not pose.get("held_pose"):
                fallback_pose = pose
                break
            trailing_holds += 1

        # At a shared waypoint the old visual leg ends before the new leg has
        # acquired enough image consensus.  The localizer deliberately emits
        # the exact trusted waypoint position plus independently tracked
        # optical yaw during that handoff.  Treating the third such frame as a
        # total pose failure created a circular Point-2->3 deadlock: the bridge
        # refused to yaw until localization recovered, while the new leg could
        # not acquire until the camera yawed toward it.
        #
        # This gate is yaw/hover-only.  It requires the localizer's explicit
        # translation lock, a fresh strong optical heading, and the narrowly
        # identified trusted-position hold reason.  The final RC command gate
        # still rejects every forward/lateral command from this pose.
        try:
            held_heading_tracks = int(latest_pose.get("rotation_heading_tracks") or 0)
        except (TypeError, ValueError):
            held_heading_tracks = 0
        held_reason = str(
            latest_pose.get("hold_reason") or latest_pose.get("rejected_reason") or ""
        )
        rotation_handoff_hold = bool(
            allow_rotation_frozen
            and latest_pose.get("held_pose") is True
            and latest_pose.get("rotation_position_locked") is True
            and latest_pose.get("translation_allowed") is False
            and pose_room_center(latest_pose) is not None
            and held_heading_tracks >= 16
            and "holding_trusted_position_with_fresh_rotation_heading" in held_reason
        )
        if rotation_handoff_hold:
            held_result = {
                "ok": True,
                "pose": pose_payload(
                    latest_pose,
                    fallback=True,
                    hold_count=trailing_holds,
                ),
                "age_seconds": age,
                "processed_count": processed_count,
                "latest_instance_id": latest_pose.get("instance_id"),
                "rotation_handoff_hold": True,
                "trailing_hold_frames": trailing_holds,
                "hold_reason": held_reason,
            }
            received_unix = held_result["pose"].get("received_unix")
            try:
                observation_age = time.time() - float(received_unix)
            except (TypeError, ValueError):
                observation_age = None
            if observation_age is not None:
                held_result["observation_age_seconds"] = observation_age
                if observation_age > max_age_seconds:
                    held_result["ok"] = False
                    held_result["reason"] = (
                        "rotation-handoff observation is stale "
                        f"({observation_age:.2f}s > {max_age_seconds:.2f}s)"
                    )
            return held_result

        if (
            fallback_pose is not None
            and trailing_holds <= max(0, int(max_recent_hold_frames))
            and (fallback_pose.get("rcenter") or fallback_pose.get("center"))
        ):
            safety_issue = stabilized_pose_safety_issue(fallback_pose)
            if allow_rotation_frozen:
                safety_issue = None
            fallback_result = {
                "ok": safety_issue is None,
                "pose": pose_payload(fallback_pose, fallback=True, hold_count=trailing_holds),
                "age_seconds": age,
                "processed_count": processed_count,
                "latest_instance_id": latest_pose.get("instance_id"),
                "recent_hold_fallback": True,
                "trailing_hold_frames": trailing_holds,
                "hold_reason": latest_pose.get("hold_reason") or latest_pose.get("rejected_reason"),
            }
            if safety_issue is not None:
                fallback_result["reason"] = safety_issue
            received_unix = fallback_result["pose"].get("received_unix")
            try:
                observation_age = time.time() - float(received_unix)
            except (TypeError, ValueError):
                observation_age = None
            if observation_age is not None:
                fallback_result["observation_age_seconds"] = observation_age
                if observation_age > max_age_seconds:
                    fallback_result["ok"] = False
                    fallback_result["reason"] = (
                        f"fallback TSolve observation is stale "
                        f"({observation_age:.2f}s > {max_age_seconds:.2f}s)"
                    )
            return fallback_result

        return {
            "ok": False,
            "reason": (
                "latest TSolve frame is held/rejected and the hold streak is too long "
                f"({latest_pose.get('instance_id')}, streak={trailing_holds})"
            ),
            "age_seconds": age,
            "processed_count": processed_count,
            "latest_instance_id": latest_pose.get("instance_id"),
            "trailing_hold_frames": trailing_holds,
            "hold_reason": latest_pose.get("hold_reason") or latest_pose.get("rejected_reason"),
        }

    if not (latest_pose.get("rcenter") or latest_pose.get("center")):
        return {
            "ok": False,
            "reason": f"latest TSolve pose has no center ({latest_pose.get('instance_id')})",
            "age_seconds": age,
            "processed_count": processed_count,
            "latest_instance_id": latest_pose.get("instance_id"),
        }

    safety_issue = stabilized_pose_safety_issue(latest_pose)
    if allow_rotation_frozen:
        safety_issue = None
    if safety_issue is not None:
        return {
            "ok": False,
            "reason": safety_issue,
            "age_seconds": age,
            "processed_count": processed_count,
            "latest_instance_id": latest_pose.get("instance_id"),
            "position_safety_rejected": True,
        }

    result = {
        "ok": True,
        "pose": pose_payload(latest_pose),
        "age_seconds": age,
        "processed_count": processed_count,
    }
    received_unix = result["pose"].get("received_unix")
    try:
        observation_age = time.time() - float(received_unix)
    except (TypeError, ValueError):
        observation_age = None
    if observation_age is not None:
        result["observation_age_seconds"] = observation_age
        if observation_age > max_age_seconds:
            return {
                **result,
                "ok": False,
                "reason": (
                    f"latest TSolve observation is stale "
                    f"({observation_age:.2f}s > {max_age_seconds:.2f}s)"
                ),
            }
    return result


def latest_rotation_only_heading(
    pose_stream_path: Path | None,
    max_age_seconds: float,
) -> dict[str, Any]:
    """Read a fresh optical-flow heading without ever treating it as position.

    The stream writer keeps this independent of the TSolve position candidate.
    It is suitable for a known in-place patrol turn, never for forward/lateral
    motion.  In particular, a position candidate may have passed the
    localizer's broad gate but still be rejected here as an implausible jump.
    """
    if pose_stream_path is None or not pose_stream_path.exists():
        return {"ok": False, "reason": "rotation-only pose stream is unavailable"}
    try:
        payload = json.loads(pose_stream_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "reason": f"rotation-only pose stream unreadable: {exc}"}
    poses = ordered_pose_observations(payload.get("poses"))
    latest = poses[-1] if poses else None
    if latest is None:
        return {"ok": False, "reason": "no rotation-only observation is available"}
    timing_valid = latest.get("rotation_heading_timing_valid")
    frame_gap_seconds = latest.get("rotation_heading_frame_gap_seconds")
    max_frame_gap_seconds = latest.get("rotation_heading_max_frame_gap_seconds")
    if timing_valid is False:
        try:
            gap_label = f"{float(frame_gap_seconds):.3f}s"
        except (TypeError, ValueError):
            gap_label = "unknown"
        return {
            "ok": False,
            "reason": (
                "rotation-only heading frame timing is discontinuous "
                f"(gap {gap_label}); waiting for an adjacent frame pair"
            ),
            "unsafe_for_yaw": True,
            "frame_gap_seconds": frame_gap_seconds,
            "max_frame_gap_seconds": max_frame_gap_seconds,
            "instance_id": latest.get("instance_id"),
        }
    if latest.get("rotation_heading_source") != "optical_flow_yaw":
        return {"ok": False, "reason": "latest frame has no trusted optical-flow yaw"}
    heading = normalize_xz(vector3(latest.get("rotation_heading")))
    try:
        tracks = int(latest.get("rotation_heading_tracks") or 0)
        received_unix = float(latest.get("received_unix"))
    except (TypeError, ValueError):
        return {"ok": False, "reason": "rotation-only observation has no usable timestamp"}
    age = time.time() - received_unix
    if heading is None or tracks < 16:
        return {"ok": False, "reason": "rotation-only optical flow is too weak"}
    if not math.isfinite(age) or age > max_age_seconds:
        return {
            "ok": False,
            "reason": f"rotation-only observation is stale ({age:.2f}s > {max_age_seconds:.2f}s)",
            "age_seconds": age,
        }
    return {
        "ok": True,
        "heading": heading,
        "tracks": tracks,
        "age_seconds": age,
        "received_unix": received_unix,
        "instance_id": latest.get("instance_id"),
        "delta_deg": latest.get("rotation_heading_delta_deg"),
        "frame_gap_seconds": frame_gap_seconds,
        "max_frame_gap_seconds": max_frame_gap_seconds,
    }


def latest_recorded_departure_heading(
    pose_stream_path: Path | None,
    max_age_seconds: float,
    *,
    map_id: str,
    patrol_id: str,
    baseline_replay_id: str,
    expected_leg_index: int | None = None,
) -> dict[str, Any]:
    """Read the absolute recorded-view heading correction for an audited turn.

    Unlike accumulated optical flow, this observation compares the current
    camera image directly with the recorded departure images for the audited
    patrol leg. It is heading-only evidence and never authorizes position or
    translation.
    """
    if pose_stream_path is None or not pose_stream_path.exists():
        return {"ok": False, "reason": "recorded heading pose stream is unavailable"}
    try:
        payload = json.loads(pose_stream_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "reason": f"recorded heading pose stream unreadable: {exc}"}
    poses = ordered_pose_observations(payload.get("poses"))
    newest = poses[-1] if poses else None
    if newest is None:
        return {"ok": False, "reason": "no recorded heading observation is available"}
    heading_poses = [
        pose
        for pose in poses
        if pose.get("route_visual_heading_required") is True
    ]
    if not heading_poses:
        return {"ok": False, "reason": "latest frame has no recorded departure heading gate"}

    leg_poses: list[dict[str, Any]] = []
    for pose in heading_poses:
        try:
            observed_leg_index = int(
                pose.get("route_visual_heading_leg_index") or 0
            )
        except (TypeError, ValueError):
            observed_leg_index = 0
        if (
            expected_leg_index is None
            or observed_leg_index == int(expected_leg_index)
        ):
            leg_poses.append(pose)
    if not leg_poses:
        return {
            "ok": False,
            "reason": "recorded departure heading patrol-leg mismatch",
        }
    expected_identity = (str(map_id), str(patrol_id), str(baseline_replay_id))
    identity_poses = [
        pose
        for pose in leg_poses
        if (
            str(pose.get("route_visual_heading_map_id") or ""),
            str(pose.get("route_visual_heading_patrol_id") or ""),
            str(pose.get("route_visual_heading_baseline_replay_id") or ""),
        )
        == expected_identity
    ]
    if not all(expected_identity) or not identity_poses:
        return {"ok": False, "reason": "recorded departure heading identity mismatch"}

    def validated_heading_observation(
        pose: dict[str, Any],
    ) -> dict[str, Any] | None:
        if pose.get("route_visual_heading_verified") is not True:
            return None
        try:
            correction = float(pose.get("route_visual_heading_correction_deg"))
            observed_inliers = int(pose.get("route_visual_heading_inliers") or 0)
            expected_floor = (
                50
                if expected_leg_index == 4
                else (48 if expected_leg_index == 1 else 120)
            )
            observed_floor = max(
                expected_floor,
                int(pose.get("route_visual_heading_minimum_inliers") or 0),
            )
            observed_unix = float(pose.get("received_unix"))
        except (TypeError, ValueError):
            return None
        observed_current = normalize_xz(
            vector3(pose.get("route_visual_heading_current"))
        )
        observed_recorded = normalize_xz(
            vector3(pose.get("route_visual_heading_recorded"))
        )
        observed_age = time.time() - observed_unix
        if (
            not math.isfinite(correction)
            or observed_inliers < observed_floor
            or observed_current is None
            or observed_recorded is None
            or not math.isfinite(observed_age)
            or observed_age > max_age_seconds
        ):
            return None
        return {
            "ok": True,
            "correction_deg": correction,
            "inliers": observed_inliers,
            "minimum_inliers": observed_floor,
            "anchor": pose.get("route_visual_heading_anchor"),
            "source_frame": pose.get("route_visual_heading_source_frame"),
            "age_seconds": observed_age,
            "received_unix": observed_unix,
            "instance_id": pose.get("instance_id"),
            "current_heading": observed_current,
            "recorded_heading": observed_recorded,
        }

    # The localizer can publish three valid adjacent ORB observations faster
    # than the bridge's polling interval.  Previously the next weak frame
    # replaced that complete triplet before the controller could count it,
    # so a physically correct Point-4 turn remained locked forever.  Latch a
    # completed, fresh three-frame consensus for this exact map/patrol/leg.
    consensus_run: list[dict[str, Any]] = []
    completed_consensus: list[dict[str, Any]] | None = None
    for pose in identity_poses:
        candidate = validated_heading_observation(pose)
        if candidate is None or abs(float(candidate["correction_deg"])) > 4.0:
            consensus_run = []
            continue
        instance_id = str(candidate.get("instance_id") or "")
        if not instance_id:
            consensus_run = []
            continue
        if consensus_run and str(consensus_run[-1].get("instance_id") or "") == instance_id:
            continue
        consensus_run.append(candidate)
        if len(consensus_run) >= 3:
            completed_consensus = list(consensus_run)
    if completed_consensus is not None:
        consensus = dict(completed_consensus[-1])
        consensus["localizer_heading_consensus_verified"] = True
        consensus["localizer_heading_consensus_count"] = len(completed_consensus)
        consensus["localizer_heading_consensus_instance_ids"] = [
            item.get("instance_id") for item in completed_consensus[-3:]
        ]
        return consensus
    # A normal solver/hold publication may immediately follow a verified ORB
    # heading frame without carrying route-heading metadata. Do not let that
    # unrelated publication erase fresh, leg-specific absolute evidence. An
    # explicit newer heading result for this same identity still supersedes an
    # older one, including when the newer result is unverified.
    latest = identity_poses[-1]
    if latest.get("route_visual_heading_verified") is not True:
        return {
            "ok": False,
            "reason": str(
                latest.get("route_visual_heading_reason")
                or "recorded departure heading is not verified"
            ),
        }
    try:
        correction_deg = float(latest.get("route_visual_heading_correction_deg"))
        inliers = int(latest.get("route_visual_heading_inliers") or 0)
        expected_floor = (
            50
            if expected_leg_index == 4
            else (48 if expected_leg_index == 1 else 120)
        )
        minimum_inliers = max(
            expected_floor,
            int(latest.get("route_visual_heading_minimum_inliers") or 0),
        )
        received_unix = float(latest.get("received_unix"))
    except (TypeError, ValueError):
        return {"ok": False, "reason": "recorded departure heading metadata is invalid"}
    current_heading = normalize_xz(
        vector3(latest.get("route_visual_heading_current"))
    )
    recorded_heading = normalize_xz(
        vector3(latest.get("route_visual_heading_recorded"))
    )
    if current_heading is None or recorded_heading is None:
        return {
            "ok": False,
            "reason": "recorded departure heading vectors are unavailable",
        }
    age = time.time() - received_unix
    if not math.isfinite(correction_deg):
        return {"ok": False, "reason": "recorded departure heading correction is invalid"}
    if inliers < minimum_inliers:
        return {
            "ok": False,
            "reason": (
                f"recorded departure heading has {inliers} inliers, below "
                f"the {minimum_inliers} gate"
            ),
        }
    if not math.isfinite(age) or age > max_age_seconds:
        return {
            "ok": False,
            "reason": (
                f"recorded departure heading is stale "
                f"({age:.2f}s > {max_age_seconds:.2f}s)"
            ),
            "age_seconds": age,
        }
    return {
        "ok": True,
        "correction_deg": correction_deg,
        "inliers": inliers,
        "minimum_inliers": minimum_inliers,
        "anchor": latest.get("route_visual_heading_anchor"),
        "source_frame": latest.get("route_visual_heading_source_frame"),
        "age_seconds": age,
        "received_unix": received_unix,
        "instance_id": latest.get("instance_id"),
        "current_heading": current_heading,
        "recorded_heading": recorded_heading,
    }


def verified_endpoint_turn_departure_gate(
    source_gate: dict[str, Any] | None,
    *,
    position_anchor: list[float] | None,
    heading_observation: dict[str, Any] | None,
    expected_leg_index: int | None,
    endpoint_handoff_verified: bool,
    endpoint_handoff_source: str | None,
    stable_heading_frames: int,
    required_stable_heading_frames: int = 3,
) -> dict[str, Any] | None:
    """Authorize one bounded departure from a verified stationary endpoint.

    Point 4 and the repeated-lap Point 1 checkpoint are independently verified
    *before* the aircraft begins its in-place yaw.  Yaw cannot change the
    aircraft's room position, so demanding a second translation solution after
    the turn creates a circular wait in the exact weak view where the first
    small forward movement is needed to reacquire normal route tracking.  This
    gate combines the already-verified endpoint position with three current,
    absolute recorded-departure heading matches.

    It is deliberately a controller-local one-command prior, not a synthetic
    metric localization.  The cruise loop consumes it for one ordinary 0.30 s
    low-stick command and then requires a new observed pose/progress before any
    additional translation.
    """
    if not endpoint_handoff_verified or expected_leg_index not in {1, 4}:
        return None
    if endpoint_handoff_source not in {"metric_tsolve", "verified_visual_endpoint"}:
        return None
    anchor = vector3(position_anchor)
    observation = heading_observation if isinstance(heading_observation, dict) else {}
    if anchor is None or observation.get("ok") is not True:
        return None
    try:
        stable = int(stable_heading_frames)
        required = max(3, int(required_stable_heading_frames))
        correction_deg = float(observation.get("correction_deg"))
        inliers = int(observation.get("inliers") or 0)
        minimum_inliers = max(
            48 if expected_leg_index == 1 else 50,
            int(observation.get("minimum_inliers") or 0),
        )
        received_unix = float(observation.get("received_unix"))
    except (TypeError, ValueError):
        return None
    if observation.get("optical_fine_handoff") is True:
        try:
            optical_tracks = int(observation.get("optical_tracks") or 0)
            absolute_correction_deg = float(
                observation.get("absolute_correction_deg")
            )
            optical_delta_deg = float(observation.get("optical_delta_deg"))
            observation_gap_seconds = float(
                observation.get("observation_gap_seconds")
            )
        except (TypeError, ValueError):
            return None
        if (
            optical_tracks < 60
            or abs(absolute_correction_deg) > 6.0
            or abs(optical_delta_deg) > 3.5
            or observation_gap_seconds < -0.05
            or observation_gap_seconds > 2.5
        ):
            return None
    current_heading = normalize_xz(vector3(observation.get("current_heading")))
    if (
        stable < required
        or not math.isfinite(correction_deg)
        or abs(correction_deg) > 4.0
        or inliers < minimum_inliers
        or not math.isfinite(received_unix)
        or current_heading is None
    ):
        return None

    gate = dict(source_gate) if isinstance(source_gate, dict) else {}
    source_pose = gate.get("pose") if isinstance(gate.get("pose"), dict) else {}
    pose = dict(source_pose)
    pose.update(
        {
            "received_unix": received_unix,
            "rcenter": anchor,
            "rheading": current_heading,
            "rheading_source": "recorded_patrol_departure_view",
            # This is no longer the localizer's rotation-frozen publication.
            # It is the unchanged, independently verified endpoint position.
            "rotation_raw_rcenter": None,
            "rotation_position_anchor": anchor,
            "rotation_position_locked": False,
            "translation_allowed": True,
            "rotation_position_source": (
                "verified_lap_point1_turn_handoff"
                if expected_leg_index == 1
                else "verified_endpoint_turn_handoff"
            ),
            "pose_source": "verified_endpoint_turn_departure",
            "verified_endpoint_turn_departure": True,
            "verified_endpoint_turn_leg_index": expected_leg_index,
            "verified_endpoint_turn_heading_inliers": inliers,
            "verified_endpoint_turn_heading_correction_deg": correction_deg,
            "verified_endpoint_turn_handoff_source": endpoint_handoff_source,
        }
    )
    gate.update(
        {
            "ok": True,
            "pose": pose,
            # ``position_anchor`` is already in ATLAS room coordinates.
            "pose_offset_room": [0.0, 0.0, 0.0],
            "recent_hold_fallback": False,
            "rotation_handoff_hold": False,
            "verified_endpoint_turn_departure": True,
        }
    )
    return gate


def verified_recorded_point_three_departure_gate(
    source_gate: dict[str, Any] | None,
    *,
    position_anchor: list[float] | None,
    heading_observation: dict[str, Any] | None,
    endpoint_handoff_verified: bool,
    endpoint_handoff_source: str | None,
    stable_heading_frames: int,
    required_stable_heading_frames: int = 3,
) -> dict[str, Any] | None:
    """Authorize Point 3->4 only from its endpoint and recorded target view.

    Optical flow remains the fast steering signal.  This gate is used only as
    a stop/hand-off authority when the current frame already matches the
    recorded Point-3 departure view.  It therefore prevents a lagging optical
    heading from commanding another quarter turn without turning prerecorded
    route position into localization truth.
    """
    gate = verified_endpoint_turn_departure_gate(
        source_gate,
        position_anchor=position_anchor,
        heading_observation=heading_observation,
        expected_leg_index=4,
        endpoint_handoff_verified=endpoint_handoff_verified,
        endpoint_handoff_source=endpoint_handoff_source,
        stable_heading_frames=stable_heading_frames,
        required_stable_heading_frames=required_stable_heading_frames,
    )
    if gate is None:
        return None
    pose = gate.get("pose") if isinstance(gate.get("pose"), dict) else None
    if pose is None:
        return None
    pose["pose_source"] = "verified_recorded_point3_turn_departure"
    pose["rotation_position_source"] = "verified_point3_recorded_turn_handoff"
    pose["verified_endpoint_turn_leg_index"] = 3
    return gate


def verified_optical_endpoint_turn_departure_gate(
    source_gate: dict[str, Any] | None,
    *,
    position_anchor: list[float] | None,
    heading_observation: dict[str, Any] | None,
    heading_error_rad: float | None,
    expected_leg_index: int | None,
    endpoint_handoff_verified: bool,
    stable_heading_frames: int,
    required_stable_heading_frames: int = 3,
    endpoint_handoff_source: str | None = None,
) -> dict[str, Any] | None:
    """Authorize one bounded pulse after a verified Point-3/Point-4 turn.

    The endpoint was independently verified at the end of the preceding leg.
    The following command is yaw-only, so that position remains physically
    valid even when the localizer publishes a rotation-locked hold. Three
    fresh, strong optical-heading observations may therefore combine with the
    saved endpoint for exactly one low-stick departure pulse. The normal
    post-command progress gate still blocks a second pulse until a new
    translation observation arrives.

    This deliberately does not authorize arbitrary optical-flow position or
    weaken either endpoint's arrival checks. Point 4->1 is held to a tighter
    heading error because no recorded departure image is authoritative there.
    """
    if not endpoint_handoff_verified or expected_leg_index not in {3, 4}:
        return None
    handoff_source = str(
        endpoint_handoff_source
        or ("verified_visual_endpoint" if expected_leg_index == 3 else "")
    )
    allowed_handoff_sources = {"verified_visual_endpoint", "metric_tsolve"}
    if handoff_source not in allowed_handoff_sources:
        return None
    anchor = vector3(position_anchor)
    observation = heading_observation if isinstance(heading_observation, dict) else {}
    if anchor is None or observation.get("ok") is not True:
        return None
    try:
        stable = int(stable_heading_frames)
        required = max(3, int(required_stable_heading_frames))
        heading_error = float(heading_error_rad)
        tracks = int(observation.get("tracks") or 0)
        received_unix = float(observation.get("received_unix"))
    except (TypeError, ValueError):
        return None
    current_heading = normalize_xz(vector3(observation.get("heading")))
    maximum_heading_error = math.radians(
        2.0 if expected_leg_index == 4 else 4.0
    )
    if (
        stable < required
        or not math.isfinite(heading_error)
        # Departure is body-forward only, with no lateral correction. Point 3
        # uses the proven four-degree entry bound; Point 4 uses two degrees so
        # the unreferenced 4->1 turn cannot stop several degrees early.
        or abs(heading_error) > maximum_heading_error
        or tracks < 60
        or not math.isfinite(received_unix)
        or current_heading is None
    ):
        return None

    gate = dict(source_gate) if isinstance(source_gate, dict) else {}
    source_pose = gate.get("pose") if isinstance(gate.get("pose"), dict) else {}
    pose = dict(source_pose)
    pose.update(
        {
            "received_unix": received_unix,
            "rcenter": anchor,
            "rheading": current_heading,
            "rheading_source": "optical_flow_yaw",
            "rotation_raw_rcenter": None,
            "rotation_position_anchor": anchor,
            "rotation_position_locked": False,
            "translation_allowed": True,
            "rotation_position_source": (
                f"verified_point{expected_leg_index}_turn_handoff"
            ),
            "pose_source": "verified_optical_endpoint_turn_departure",
            "verified_endpoint_turn_departure": True,
            "verified_endpoint_turn_leg_index": expected_leg_index,
            "verified_endpoint_turn_heading_tracks": tracks,
            "verified_endpoint_turn_heading_error_deg": math.degrees(
                heading_error
            ),
            "verified_endpoint_turn_handoff_source": handoff_source,
        }
    )
    gate.update(
        {
            "ok": True,
            "pose": pose,
            "pose_offset_room": [0.0, 0.0, 0.0],
            "recent_hold_fallback": False,
            "rotation_handoff_hold": False,
            "verified_endpoint_turn_departure": True,
        }
    )
    return gate


def recorded_heading_optical_fine_handoff(
    absolute_observation: dict[str, Any] | None,
    optical_observation: dict[str, Any] | None,
    *,
    optical_heading_bias_rad: float | None,
    max_absolute_error_deg: float = 6.0,
    max_optical_delta_deg: float = 3.5,
    max_observation_gap_seconds: float = 2.5,
    minimum_optical_tracks: int = 60,
) -> dict[str, Any] | None:
    """Bridge a brief Point-4 image-match dropout with rebased optical yaw.

    This is not an optical-only turn.  A current recorded-departure image must
    first place the camera within six degrees of the absolute taught heading.
    Its measured optical-to-room offset is then held for at most 2.5 seconds
    and 3.5 degrees of additional yaw.  Strong current optical flow may finish
    those last few degrees, but position remains locked and the caller still
    requires three distinct stable frames before the one-command departure
    handoff can be created.
    """
    absolute = (
        absolute_observation
        if isinstance(absolute_observation, dict)
        else {}
    )
    optical = optical_observation if isinstance(optical_observation, dict) else {}
    if absolute.get("ok") is not True or optical.get("ok") is not True:
        return None
    if optical_heading_bias_rad is None:
        return None
    try:
        absolute_error_deg = float(absolute.get("correction_deg"))
        absolute_inliers = int(absolute.get("inliers") or 0)
        absolute_minimum = max(
            30,
            int(absolute.get("minimum_inliers") or 0),
        )
        absolute_received = float(absolute.get("received_unix"))
        optical_received = float(optical.get("received_unix"))
        optical_tracks = int(optical.get("tracks") or 0)
    except (TypeError, ValueError):
        return None
    if (
        not math.isfinite(absolute_error_deg)
        or abs(absolute_error_deg) > max(0.0, float(max_absolute_error_deg))
        or absolute_inliers < absolute_minimum
        or optical_tracks < max(16, int(minimum_optical_tracks))
    ):
        return None
    observation_gap = optical_received - absolute_received
    if (
        not math.isfinite(observation_gap)
        or observation_gap < -0.05
        or observation_gap > max(0.1, float(max_observation_gap_seconds))
    ):
        return None
    absolute_current = normalize_xz(vector3(absolute.get("current_heading")))
    recorded_target = normalize_xz(vector3(absolute.get("recorded_heading")))
    optical_current = normalize_xz(vector3(optical.get("heading")))
    if (
        absolute_current is None
        or recorded_target is None
        or optical_current is None
    ):
        return None
    rebased_current = rotate_xz(
        optical_current,
        float(optical_heading_bias_rad),
    )
    correction = signed_angle_xz(rebased_current, recorded_target)
    optical_delta = signed_angle_xz(absolute_current, rebased_current)
    if correction is None or optical_delta is None:
        return None
    correction_deg = math.degrees(correction)
    optical_delta_deg = math.degrees(optical_delta)
    if (
        not math.isfinite(correction_deg)
        or not math.isfinite(optical_delta_deg)
        or abs(optical_delta_deg) > max(0.1, float(max_optical_delta_deg))
        # Optical continuation may reduce the last absolute error; it may not
        # invent a larger correction or reverse into a new search direction.
        or abs(correction_deg) > abs(absolute_error_deg) + 0.75
        or (
            abs(correction_deg) > 4.0
            and correction_deg * absolute_error_deg < 0.0
        )
    ):
        return None
    instance_id = str(optical.get("instance_id") or "")
    if not instance_id:
        return None
    return {
        "ok": True,
        "correction_deg": correction_deg,
        "inliers": absolute_inliers,
        "minimum_inliers": absolute_minimum,
        "anchor": absolute.get("anchor"),
        "source_frame": absolute.get("source_frame"),
        "received_unix": optical_received,
        "instance_id": instance_id,
        "current_heading": rebased_current,
        "recorded_heading": recorded_target,
        "optical_fine_handoff": True,
        "optical_tracks": optical_tracks,
        "absolute_correction_deg": absolute_error_deg,
        "optical_delta_deg": optical_delta_deg,
        "absolute_received_unix": absolute_received,
        "observation_gap_seconds": observation_gap,
    }


def load_taught_patrol_reference(mission: dict[str, Any]) -> dict[str, Any] | None:
    """Load the optional visual turn reference for this exact map and patrol."""
    map_id = str(mission.get("map_id") or "")
    patrol_id = str(mission.get("patrol_id") or "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", map_id):
        return None
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", patrol_id):
        return None
    baseline_replay_id = str(mission.get("baseline_replay_id") or "")
    if baseline_replay_id and re.fullmatch(r"[A-Za-z0-9_-]{1,160}", baseline_replay_id):
        path = (
            ROOT
            / "viewer"
            / "public"
            / "maps"
            / map_id
            / "replays"
            / baseline_replay_id
            / "reference_candidate.json"
        )
    else:
        path = ROOT / "viewer" / "public" / "maps" / map_id / "taught_patrols" / patrol_id / "reference.json"
    try:
        reference = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(reference, dict):
        return None
    if reference.get("map_id") != map_id or reference.get("patrol_id") != patrol_id:
        return None
    enabled = bool(
        reference.get("enabled_for_turn_recovery")
        or (
            baseline_replay_id
            and reference.get("enabled_for_live_route_gate") is True
        )
    )
    if not reference.get("complete_loop") or not enabled:
        return None
    if not isinstance(reference.get("legs"), list) or len(reference["legs"]) < 4:
        return None
    return reference


def load_verified_route_follow_lock(mission: dict[str, Any]) -> dict[str, Any] | None:
    """Load the pinned heading lock for the complete recorded patrol.

    Legs 1->2->3->4 retain the successful 11:57 live headings.  The missing
    4->1 leg comes from the separately audited complete-loop baseline, whose
    held-out frames are also the live position/progress authority.
    """
    map_id = str(mission.get("map_id") or "")
    patrol_id = str(mission.get("patrol_id") or "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", map_id):
        return None
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", patrol_id):
        return None
    path = (
        ROOT
        / "viewer"
        / "public"
        / "maps"
        / map_id
        / "taught_patrols"
        / patrol_id
        / "route_follow_lock.json"
    )
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(lock, dict):
        return None
    if lock.get("map_id") != map_id or lock.get("patrol_id") != patrol_id:
        return None
    if not isinstance(lock.get("legs"), list):
        return None
    return lock


def verified_route_follow_leg(
    route_lock: dict[str, Any] | None,
    taught_leg: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return a verified golden leg only when its point identity matches."""
    if not isinstance(route_lock, dict) or not isinstance(taught_leg, dict):
        return None
    try:
        expected = (
            int(taught_leg.get("from_point")),
            int(taught_leg.get("to_point")),
        )
    except (TypeError, ValueError):
        return None
    for leg in route_lock.get("legs") or []:
        if not isinstance(leg, dict) or leg.get("verified") is not True:
            continue
        try:
            identity = (int(leg.get("from_point")), int(leg.get("to_point")))
            heading = float(leg.get("golden_camera_heading_deg"))
        except (TypeError, ValueError):
            continue
        if identity == expected and math.isfinite(heading):
            return leg
    return None


def pose_gate_camera_heading(
    gate: dict[str, Any] | None,
    *,
    optical_heading_bias_rad: float | None = None,
) -> list[float] | None:
    """Return the raw camera heading used by the recorded 11:57 reference."""
    pose = gate.get("pose") if isinstance(gate, dict) else None
    if not isinstance(pose, dict):
        return None
    # A fused live/ORB heading is shared by control and rendering. Route-only
    # position recovery must not make the bridge independently switch back to
    # a stale optical basis: the 2026-08-23 4->1 flight physically completed
    # the turn while the model/controller pose carried a different heading.
    # Legacy bank-only payloads still use optical yaw as a compatibility
    # fallback, but every new live/TSolve/ORB source keeps its published
    # rheading authoritative.
    if pose.get("pose_source") == "patrol_visual_route_recovery":
        heading_source = str(pose.get("rheading_source") or "")
        fused_heading = normalize_xz(vector3(pose.get("rheading")))
        bank_only_fallback = heading_source in {
            "recorded_patrol_leg_heading",
            "recorded_patrol_leg_heading_fallback",
        }
        if fused_heading is not None and not bank_only_fallback:
            return fused_heading
        try:
            optical_tracks = int(pose.get("rotation_heading_tracks") or 0)
        except (TypeError, ValueError):
            optical_tracks = 0
        optical_heading = normalize_xz(vector3(pose.get("rotation_heading")))
        if optical_heading is not None and optical_tracks >= 16:
            return rotate_xz(
                optical_heading,
                float(optical_heading_bias_rad or 0.0),
            )
    # A recorded-image alignment has already corrected a weak/stale algebraic
    # rotation. Preserve that verified absolute view. Ordinary metric TSolve
    # poses use the unmodified camera heading so the golden comparison remains
    # independent of the body-forward calibration offset.
    recorded_heading = bool(
        pose.get("rheading_source")
        in {
            "recorded_departure_image_alignment",
            "recorded_departure_image_tracking_consensus",
            "recorded_patrol_leg_heading",
            "recorded_patrol_leg_heading_fallback",
        }
        or pose.get("pose_source") == "patrol_visual_route_recovery"
    )
    value = (
        pose.get("rheading")
        if recorded_heading
        else (pose.get("rheading_raw") or pose.get("rheading"))
    )
    return normalize_xz(vector3(value))


def verified_route_desired_camera_heading(
    route_leg: dict[str, Any] | None,
    *,
    current_position: list[float] | None,
    segment_start: list[float] | None,
    segment_end: list[float] | None,
    default_lookahead_m: float = 0.35,
    default_max_correction_deg: float = 7.0,
) -> list[float] | None:
    """Follow the successful camera heading with bounded route look-ahead.

    The golden heading reproduces the 11:57 view. A small pure-pursuit
    correction steers an offset aircraft back to the same segment without
    replacing that heading with a run-dependent direct waypoint bearing.
    """
    if not isinstance(route_leg, dict):
        return None
    try:
        golden_degrees = float(route_leg.get("golden_camera_heading_deg"))
        lookahead_m = max(
            0.10,
            float(route_leg.get("lookahead_m") or default_lookahead_m),
        )
        max_correction_deg = max(
            0.0,
            min(
                15.0,
                float(
                    route_leg.get("max_route_correction_deg")
                    or default_max_correction_deg
                ),
            ),
        )
    except (TypeError, ValueError):
        return None
    if not math.isfinite(golden_degrees):
        return None
    golden = [
        math.cos(math.radians(golden_degrees)),
        0.0,
        math.sin(math.radians(golden_degrees)),
    ]
    current = vector3(current_position)
    start = vector3(segment_start)
    end = vector3(segment_end)
    tangent = normalize_xz(target_direction_xz(start, end))
    if current is None or start is None or end is None or tangent is None:
        return normalize_xz(golden)
    leg_length = horizontal_xz_distance(start, end)
    progress = route_segment_progress_xz(current, start, end)
    if leg_length is None or leg_length <= 1e-9 or progress is None:
        return normalize_xz(golden)
    lookahead_progress = max(
        0.0,
        min(1.0, float(progress) + lookahead_m / leg_length),
    )
    lookahead = [
        start[0] + (end[0] - start[0]) * lookahead_progress,
        current[1],
        start[2] + (end[2] - start[2]) * lookahead_progress,
    ]
    recovery_direction = normalize_xz(target_direction_xz(current, lookahead))
    correction = signed_angle_xz(tangent, recovery_direction)
    if correction is None:
        return normalize_xz(golden)
    correction_limit = math.radians(max_correction_deg)
    correction = max(-correction_limit, min(correction_limit, correction))
    return rotate_xz(golden, correction)


def taught_leg_for_step(reference: dict[str, Any] | None, step: dict[str, Any]) -> dict[str, Any] | None:
    """Return the taught leg matching this patrol cruise.

    Taught camera headings are valid only for the geometry they were recorded
    against. Command order alone is not a safe fallback after a point edit: it
    can apply a stale turn direction to a newly shaped route.  The point-3 to
    point-4 weak-visual-sector guard is the one exception: it may be reused
    after a coordinate correction only when the new leg retains the taught
    heading.  That guard consumes neither taught endpoints nor taught distance.
    """
    if not isinstance(reference, dict):
        return None
    start = vector3(step.get("from"))
    end = vector3(step.get("to"))
    if start is None or end is None:
        return None
    for leg in reference.get("legs", []):
        if not isinstance(leg, dict):
            continue
        ref_start = vector3(leg.get("from"))
        ref_end = vector3(leg.get("to"))
        if (
            horizontal_xz_distance(start, ref_start) is not None
            and horizontal_xz_distance(end, ref_end) is not None
            and horizontal_xz_distance(start, ref_start) <= 0.06
            and horizontal_xz_distance(end, ref_end) <= 0.06
        ):
            return leg

    # Cruise 1 is the entry to point 1, then cruises 2/3/4 are the taught
    # 1->2, 2->3 and 3->4 loop legs.  Restore only the known weak-sector
    # point-3->4 turn when its room-frame direction is still compatible.  This
    # intentionally does not restore the former command-order fallback for
    # any other leg (especially the reshaped point-4->1 return).
    if not re.fullmatch(
        r"\s*Patrol cruise\s+4\s*",
        str(step.get("title") or ""),
        re.IGNORECASE,
    ):
        return None
    step_direction = normalize_xz(target_direction_xz(start, end))
    if step_direction is None:
        return None
    step_heading_deg = math.degrees(math.atan2(step_direction[2], step_direction[0]))
    for leg in reference.get("legs", []):
        if not isinstance(leg, dict):
            continue
        try:
            from_point = int(leg.get("from_point"))
            to_point = int(leg.get("to_point"))
            taught_heading_deg = float(leg.get("expected_heading_deg"))
        except (TypeError, ValueError):
            continue
        heading_error_deg = abs(
            ((step_heading_deg - taught_heading_deg + 180.0) % 360.0) - 180.0
        )
        if from_point == 3 and to_point == 4 and heading_error_deg <= 8.0:
            return leg
    return None


def taught_turn_direction_override(leg: dict[str, Any] | None) -> str | None:
    """Return an operator-requested turn direction for one taught leg."""
    if not isinstance(leg, dict):
        return None
    configured = str(leg.get("turn_direction_override") or "").strip().lower()
    if configured in {"left", "right"}:
        return configured
    try:
        from_point = int(leg.get("from_point"))
        to_point = int(leg.get("to_point"))
    except (TypeError, ValueError):
        return None
    # Point 3→4 has a repeatable weak visual sector.  Follow the
    # operator-requested clockwise/right route while keeping translation
    # locked and using the independent optical heading through that sector.
    if from_point == 3 and to_point == 4:
        return "right"
    return None


def is_guarded_point_three_to_four_turn(leg: dict[str, Any] | None) -> bool:
    """Identify the lab corner that must remain rotation-only until aligned."""
    if not isinstance(leg, dict):
        return False
    try:
        return int(leg.get("from_point")) == 3 and int(leg.get("to_point")) == 4
    except (TypeError, ValueError):
        return False


def is_point_two_to_three_leg(leg: dict[str, Any] | None) -> bool:
    """Identify the turn whose optical yaw over-count/under-count can oscillate."""
    if not isinstance(leg, dict):
        return False
    try:
        return int(leg.get("from_point")) == 2 and int(leg.get("to_point")) == 3
    except (TypeError, ValueError):
        return False


def is_point_four_to_one_leg(leg: dict[str, Any] | None) -> bool:
    """Identify the return leg whose turn must start from measured Point 4."""
    if not isinstance(leg, dict):
        return False
    try:
        return int(leg.get("from_point")) == 4 and int(leg.get("to_point")) == 1
    except (TypeError, ValueError):
        return False


def is_point_one_to_two_leg(leg: dict[str, Any] | None) -> bool:
    """Identify the next-lap departure now covered by the audited tail bank."""
    if not isinstance(leg, dict):
        return False
    try:
        return int(leg.get("from_point")) == 1 and int(leg.get("to_point")) == 2
    except (TypeError, ValueError):
        return False


def prior_verified_endpoint_arrival_record(
    executed: list[dict[str, Any]] | None,
    *,
    segment_start: list[float] | None,
    expected_leg_index: int,
    maximum_anchor_error: float = 0.02,
) -> dict[str, Any] | None:
    """Return the immediately preceding cruise's durable endpoint proof.

    A verified endpoint remains physically valid through subsequent neutral
    hovers and an in-place yaw.  The live route context changes at the next
    leg and may no longer publish the preceding leg's endpoint fields, so the
    controller must retain its own accepted-arrival decision instead of
    asking localization to prove the same position again before yaw.
    """
    anchor = vector3(segment_start)
    if anchor is None or not isinstance(executed, list):
        return None
    verified_modes = {
        "strict_radius_endpoint_verified",
        "visual_checkpoint_endpoint_verified",
        "visual_checkpoint_endpoint_verified_timeout",
    }
    for item in reversed(executed):
        if not isinstance(item, dict) or item.get("type") != "cruise":
            continue
        # Only the immediately preceding physical cruise can establish the
        # shared waypoint for this new leg.
        if (
            item.get("closed_loop") is not True
            or item.get("reached") is not True
            or item.get("arrival_mode") not in verified_modes
        ):
            return None
        try:
            arrival_leg_index = int(item.get("endpoint_leg_index"))
        except (TypeError, ValueError):
            return None
        target = vector3(item.get("target"))
        anchor_error = horizontal_xz_distance(target, anchor)
        if (
            arrival_leg_index != int(expected_leg_index)
            or anchor_error is None
            or anchor_error > max(0.001, float(maximum_anchor_error))
        ):
            return None
        return dict(item)
    return None


def taught_turn_requires_recorded_departure_view(
    leg: dict[str, Any] | None,
) -> bool:
    """Return whether a turn may use an absolute recorded departure image.

    Preserve the successful live/optical Point-3 -> Point-4 turn. The audited
    manual tail provides absolute departure views for Point 4 -> Point 1 and
    for the following Point 1 -> Point 2 turn. Position remains locked until
    the normal three-frame heading gate passes.
    """
    return is_point_four_to_one_leg(leg) or is_point_one_to_two_leg(leg)


def taught_leg_requires_precise_arrival(leg: dict[str, Any] | None) -> bool:
    """Require independent endpoint evidence on every closed-loop patrol leg.

    The Point-4->1 endpoint now has an audited live Point-1 view.  Making that
    leg precise lets the existing stale-translation arrival path stop at the
    physical window view and hand the exact shared Point-1 anchor to lap 2,
    without granting recorded imagery ordinary mid-leg position authority.
    """
    if not isinstance(leg, dict):
        return False
    try:
        return (
            int(leg.get("from_point")) in {1, 2, 3, 4}
            and int(leg.get("to_point")) in {1, 2, 3, 4}
        )
    except (TypeError, ValueError):
        return False


def taught_endpoint_arrival_verified(
    pose: dict[str, Any] | None,
    *,
    expected_leg_index: int | None,
) -> bool:
    """Require progress-independent baseline evidence at a taught waypoint."""
    if (
        not isinstance(pose, dict)
        or pose.get("route_visual_endpoint_verified") is not True
        or pose.get("route_visual_endpoint_view_geometry_verified") is not True
    ):
        return False
    try:
        observed_leg = int(pose.get("route_visual_monitor_leg_index") or 0)
        expected_leg = int(expected_leg_index or 0)
        hits = int(pose.get("route_visual_endpoint_hits") or 0)
        required_hits = max(
            2, int(pose.get("route_visual_endpoint_required_hits") or 0)
        )
        best_progress = float(pose.get("route_visual_endpoint_best_progress"))
        best_inliers = int(pose.get("route_visual_endpoint_best_inliers") or 0)
        minimum_inliers = max(
            72, int(pose.get("route_visual_endpoint_minimum_inliers") or 0)
        )
    except (TypeError, ValueError):
        return False
    return bool(
        expected_leg in {1, 2, 3, 4}
        and observed_leg == expected_leg
        and hits >= required_hits
        and best_progress >= 0.90
        and best_inliers >= minimum_inliers
    )


def taught_endpoint_stale_translation_arrival_verified(
    pose: dict[str, Any] | None,
    *,
    expected_leg_index: int | None,
) -> bool:
    """Accept a taught endpoint when metric translation is visibly stale.

    This is deliberately stricter than the ordinary endpoint gate.  It is
    used only to *stop* an active translation leg; it never authorizes another
    movement pulse.  Requiring a strong current endpoint candidate as well as
    the repeated whole-leg consensus prevents an old best match from skipping
    a waypoint when TSolve and the independently pinned patrol imagery differ.
    """
    if not taught_endpoint_arrival_verified(
        pose,
        expected_leg_index=expected_leg_index,
    ):
        return False
    if not isinstance(pose, dict):
        return False
    if pose.get("route_visual_monitor_verified") is not True:
        return False
    if pose.get("route_visual_endpoint_checked") is not True:
        return False
    if pose.get("rotation_position_locked") is True:
        return False
    if pose.get("translation_allowed") is False:
        return False
    try:
        candidate_progress = float(
            pose.get("route_visual_endpoint_candidate_progress")
        )
        best_progress = float(pose.get("route_visual_endpoint_best_progress"))
        best_inliers = int(pose.get("route_visual_endpoint_best_inliers") or 0)
        minimum_inliers = max(
            90,
            int(pose.get("route_visual_endpoint_minimum_inliers") or 0),
        )
    except (TypeError, ValueError):
        return False
    return bool(
        math.isfinite(candidate_progress)
        and math.isfinite(best_progress)
        and candidate_progress >= 0.90
        and best_progress >= 0.95
        and best_inliers >= minimum_inliers
    )


def tight_metric_visual_endpoint_arrival_candidate(
    gate: dict[str, Any] | None,
    *,
    target: list[float] | None,
    expected_leg_index: int | None,
    command_progress_ceiling: float | None,
    maximum_metric_error: float = 0.15,
) -> bool:
    """Use fresh metric position as Point-3 authority with ORB as a monitor.

    The recorded Point-3 endpoint images contain repeated room structure and
    can alternate between several progress values even while TSolve remains
    stable inside the strict arrival radius.  That ambiguity must not freeze
    the aircraft after the controller has consumed the complete 2->3 command
    budget.  A fresh metric pose may therefore finish only leg 2, and only
    when the ordinary route monitor still identifies that same leg strongly.

    Endpoint-image *progress* is deliberately not consumed here.  Its strong
    descriptor support is secondary evidence that the camera remains on the
    taught route, not the metric arrival authority.
    """
    if int(expected_leg_index or 0) != 2:
        return False
    if not pose_gate_has_fresh_metric_position(gate):
        return False
    if gate.get("recent_hold_fallback") or gate.get("rotation_handoff_hold"):
        return False
    pose = gate.get("pose") if isinstance(gate, dict) else None
    if not isinstance(pose, dict):
        return False
    if pose.get("rotation_position_locked") is True:
        return False
    if pose.get("translation_allowed") is False:
        return False
    try:
        ceiling = float(command_progress_ceiling)
        observed_leg = int(pose.get("route_visual_monitor_leg_index") or 0)
        route_inliers = int(pose.get("route_visual_monitor_inliers") or 0)
        route_progress = float(pose.get("route_visual_monitor_progress"))
        tsolve_progress = float(
            pose.get("route_visual_monitor_tsolve_progress")
        )
        route_disagreement = float(
            pose.get("route_visual_monitor_disagreement_m")
        )
    except (TypeError, ValueError):
        return False
    position = pose_gate_position(gate)
    error = horizontal_xz_distance(position, vector3(target))
    return bool(
        math.isfinite(ceiling)
        and ceiling >= 1.0 - 1e-6
        and error is not None
        and error <= max(0.02, min(0.15, float(maximum_metric_error)))
        and pose.get("route_visual_monitor_verified") is True
        and observed_leg == 2
        and route_inliers >= 90
        and math.isfinite(route_progress)
        and route_progress >= 0.88
        and math.isfinite(tsolve_progress)
        and tsolve_progress >= 0.88
        and math.isfinite(route_disagreement)
        and route_disagreement <= 0.10
    )


def tight_metric_point_two_endpoint_arrival_candidate(
    gate: dict[str, Any] | None,
    *,
    target: list[float] | None,
    segment_start: list[float] | None,
    expected_leg_index: int | None,
    maximum_metric_error: float = 0.03,
    maximum_cross_track: float = 0.08,
) -> bool:
    """Allow fresh, stable TSolve to finish Point 1 -> Point 2.

    The Point-2 endpoint image bank can temporarily have no ORB anchor even
    when a current metric solve is already at the saved waypoint. Keeping the
    aircraft in endpoint recovery in that state cannot improve its physical
    position and used to create an indefinite hover. This exception is kept
    deliberately narrow: it applies only to leg 1, requires a real current
    TSolve/COLMAP R,t, and accepts at most a three-centimetre endpoint error
    inside the active route corridor.
    """
    if int(expected_leg_index or 0) != 1:
        return False
    if not pose_gate_has_fresh_metric_position(gate):
        return False
    if gate.get("recent_hold_fallback") or gate.get("rotation_handoff_hold"):
        return False
    pose = gate.get("pose") if isinstance(gate, dict) else None
    if not isinstance(pose, dict):
        return False
    if pose.get("rotation_position_locked") is True:
        return False
    if pose.get("translation_allowed") is False:
        return False
    position = pose_gate_position(gate)
    endpoint = vector3(target)
    start = vector3(segment_start)
    error = horizontal_xz_distance(position, endpoint)
    progress = route_segment_progress_xz(position, start, endpoint)
    cross_track = horizontal_xz_segment_distance(position, start, endpoint)
    return bool(
        error is not None
        and error <= max(0.01, min(0.03, float(maximum_metric_error)))
        and progress is not None
        and 0.90 <= float(progress) <= 1.10
        and cross_track is not None
        and cross_track <= max(0.03, min(0.08, float(maximum_cross_track)))
    )


def update_stable_metric_endpoint_consensus(
    state: dict[str, Any],
    gate: dict[str, Any] | None,
    *,
    candidate: bool,
    required_hits: int = 3,
    maximum_step_m: float = 0.035,
) -> bool:
    """Require distinct, spatially stable metric frames at an endpoint."""
    if not candidate:
        state.clear()
        return False
    pose = gate.get("pose") if isinstance(gate, dict) else None
    instance_id = str((pose or {}).get("instance_id") or "")
    position = pose_gate_position(gate)
    if not instance_id or position is None:
        state.clear()
        return False
    if state.get("last_instance_id") != instance_id:
        previous_position = vector3(state.get("last_position"))
        position_step = horizontal_xz_distance(previous_position, position)
        if (
            previous_position is not None
            and position_step is not None
            and position_step > max(0.01, float(maximum_step_m))
        ):
            state["hits"] = 1
        else:
            state["hits"] = int(state.get("hits") or 0) + 1
        state["last_instance_id"] = instance_id
        state["last_position"] = list(position)
    return int(state.get("hits") or 0) >= max(3, int(required_hits))


def update_tight_endpoint_consensus(
    state: dict[str, Any],
    gate: dict[str, Any] | None,
    *,
    candidate: bool,
    required_hits: int = 3,
) -> bool:
    """Require distinct localized frames before accepting metric Point 3."""
    if not candidate:
        state.clear()
        return False
    pose = gate.get("pose") if isinstance(gate, dict) else None
    instance_id = str((pose or {}).get("instance_id") or "")
    if not instance_id:
        state.clear()
        return False
    if state.get("last_instance_id") != instance_id:
        state["last_instance_id"] = instance_id
        state["hits"] = int(state.get("hits") or 0) + 1
    return int(state.get("hits") or 0) >= max(3, int(required_hits))


def yaw_direction_for_angle(angle: float, override: str | None = None) -> float:
    """Choose yaw direction; negative is the bridge's map-frame left turn."""
    if override == "left":
        return -1.0
    if override == "right":
        return 1.0
    return 1.0 if angle > 0.0 else -1.0


def normalized_yaw_command_effort(
    yaw_rc: float,
    max_yaw_rc: float,
    seconds: float,
) -> float:
    """Return full-stick-equivalent yaw seconds for one RC command."""
    try:
        command = abs(float(yaw_rc))
        maximum = abs(float(max_yaw_rc))
        duration = max(0.0, float(seconds))
    except (TypeError, ValueError):
        return 0.0
    if (
        not math.isfinite(command)
        or not math.isfinite(maximum)
        or not math.isfinite(duration)
        or maximum <= 1e-9
    ):
        return 0.0
    return duration * min(1.0, command / maximum)


def point_three_recorded_recovery_effort_limit_seconds(
    correction_degrees: float,
    *,
    stop_tolerance_degrees: float = 6.0,
    maximum_recovery_degrees: float = 30.0,
    margin_degrees: float = 4.0,
    conservative_yaw_rate_deg_per_effort_second: float = 8.0,
    maximum_effort_seconds: float = 3.0,
) -> float:
    """Bound fine Point-3 yaw driven by an absolute recorded-view match.

    A correction inside the stop tolerance needs no command.  A correction
    beyond the recovery window is too large to trust as a fine handoff.  The
    remaining window gets a deliberately conservative, independently capped
    command budget; translation stays locked throughout.
    """
    try:
        correction = abs(float(correction_degrees))
        stop_tolerance = max(0.0, float(stop_tolerance_degrees))
        maximum_recovery = max(stop_tolerance, float(maximum_recovery_degrees))
        margin = max(0.0, float(margin_degrees))
        yaw_rate = float(conservative_yaw_rate_deg_per_effort_second)
        maximum_effort = max(0.0, float(maximum_effort_seconds))
    except (TypeError, ValueError):
        return 0.0
    values = (
        correction,
        stop_tolerance,
        maximum_recovery,
        margin,
        yaw_rate,
        maximum_effort,
    )
    if not all(math.isfinite(value) for value in values) or yaw_rate <= 1e-9:
        return 0.0
    if correction <= stop_tolerance or correction > maximum_recovery:
        return 0.0
    return min(maximum_effort, (correction + margin) / yaw_rate)


def vector3(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        out = [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None
    if not all(abs(v) < 1e9 for v in out):
        return None
    return out


def pose_gate_position(gate: dict[str, Any] | None) -> list[float] | None:
    pose = gate.get("pose") if isinstance(gate, dict) else None
    if not isinstance(pose, dict):
        return None
    # Patrol targets are sent in ATLAS room coordinates.  Never fall back to
    # raw COLMAP centers for physical motion; that mixes coordinate frames.
    position = vector3(pose.get("rcenter"))
    offset = vector3(gate.get("pose_offset_room")) if isinstance(gate, dict) else None
    if position is not None and offset is not None:
        return [position[index] + offset[index] for index in range(3)]
    return position


def pose_gate_predates_route_epoch(
    gate: dict[str, Any] | None,
    epoch_unix: float | None,
) -> bool:
    """Whether a pose observation was captured before a controller handoff.

    The pose stream can finish an older frame after the controller has already
    accepted a waypoint. Stream/file freshness is not enough in that case:
    only the camera capture timestamp proves that the observation belongs to
    the new motion phase.
    """
    if epoch_unix is None:
        return False
    pose = gate.get("pose") if isinstance(gate, dict) else None
    if not isinstance(pose, dict):
        return True
    try:
        received_unix = float(pose.get("received_unix"))
        cutoff = float(epoch_unix)
    except (TypeError, ValueError):
        return True
    if not math.isfinite(received_unix) or not math.isfinite(cutoff):
        return True
    return received_unix < cutoff


def verified_route_anchor_pose_gate(
    gate: dict[str, Any] | None,
    anchor: Any,
    *,
    epoch: int,
    epoch_unix: float,
    reason: str,
) -> dict[str, Any] | None:
    """Rebase yaw/hover continuity to a physically verified waypoint.

    This does not manufacture a new metric measurement and never authorizes
    translation. It records the controller-owned position truth at the exact
    boundary where the aircraft is known to be stationary, while retaining
    the observed heading and solver payload for the following yaw gate.
    """
    position = vector3(anchor)
    if not isinstance(gate, dict) or position is None:
        return gate
    anchored_gate = dict(gate)
    pose = dict(gate.get("pose") or {})
    prior_room_center = vector3(pose.get("rcenter"))
    prior_raw_room_center = vector3(pose.get("rotation_raw_rcenter"))
    if prior_room_center is not None:
        pose["route_pose_epoch_prior_rcenter"] = prior_room_center
    if prior_raw_room_center is not None:
        pose["route_pose_epoch_prior_raw_rcenter"] = prior_raw_room_center
    pose.update(
        {
            "rcenter": list(position),
            "rotation_raw_rcenter": list(position),
            "rotation_position_anchor": list(position),
            "rotation_position_locked": True,
            "translation_allowed": False,
            "rotation_position_source": "verified_route_pose_epoch_anchor",
            "rotation_anchor_is_position_truth": True,
            "rotation_anchor_commanded": True,
            "route_pose_epoch": int(epoch),
            "route_pose_epoch_unix": float(epoch_unix),
            "route_pose_epoch_reason": str(reason),
        }
    )
    anchored_gate.update(
        {
            "pose": pose,
            "route_pose_epoch": int(epoch),
            "route_pose_epoch_unix": float(epoch_unix),
            "route_pose_epoch_reason": str(reason),
            "verified_route_anchor_gate": True,
        }
    )
    return anchored_gate


def verified_route_endpoint_pose_gate(
    gate: dict[str, Any] | None,
    endpoint: Any,
    *,
    epoch: int,
    epoch_unix: float,
    reason: str,
) -> dict[str, Any] | None:
    """Commit a visually verified stationary endpoint to route pose state.

    The endpoint matcher and the metric pose stream are independent.  When the
    former has repeatedly proved that the aircraft reached a taught waypoint,
    merely stopping the cruise leaves ``last_pose_gate`` and rendered route
    progress behind the real aircraft.  This wrapper creates the same
    translation-locked controller-owned anchor used by the Point-4 handoff and
    explicitly completes the old leg.  It still cannot authorize translation;
    a post-epoch camera observation and the next leg's heading gates are
    required before another horizontal command.
    """
    anchored_gate = verified_route_anchor_pose_gate(
        gate,
        endpoint,
        epoch=epoch,
        epoch_unix=epoch_unix,
        reason=reason,
    )
    if not isinstance(anchored_gate, dict):
        return anchored_gate
    pose = dict(anchored_gate.get("pose") or {})
    pose.update(
        {
            "route_verified_endpoint_committed": True,
            "route_verified_endpoint_progress": 1.0,
        }
    )
    anchored_gate.update(
        {
            "pose": pose,
            "route_progress": 1.0,
            "verified_route_endpoint_gate": True,
        }
    )
    return anchored_gate


def pose_gate_has_fresh_metric_position(gate: dict[str, Any] | None) -> bool:
    """Return whether the gate contains a current metric TSolve/COLMAP pose.

    A verified patrol-image match is useful for bounded route recovery, but it
    intentionally has no map-space ``R``/``t``.  It must therefore not carry a
    completed lap straight into another lap as if metric tracking were still
    healthy.  Likewise, a recent fallback to an older accepted pose is safe for
    hover/yaw but is not a fresh position measurement.
    """
    if not isinstance(gate, dict) or gate.get("ok") is not True:
        return False
    if gate.get("recent_hold_fallback") or gate.get("rotation_handoff_hold"):
        return False
    pose = gate.get("pose")
    if not isinstance(pose, dict):
        return False
    if pose.get("pose_source") in {
        "patrol_visual_route_recovery",
        "verified_endpoint_turn_departure",
    }:
        return False
    if (
        pose_gate_position(gate) is None
        or vector3(pose.get("center")) is None
        or vector3(pose.get("t")) is None
    ):
        return False
    rotation = pose.get("R")
    if not isinstance(rotation, (list, tuple)) or len(rotation) != 3:
        return False
    try:
        rotation_values = [
            float(value)
            for row in rotation
            for value in (row if isinstance(row, (list, tuple)) else [])
        ]
    except (TypeError, ValueError):
        return False
    return len(rotation_values) == 9 and all(math.isfinite(value) for value in rotation_values)


def pose_gate_rotation_locked(gate: dict[str, Any] | None) -> bool:
    """Whether a fresh pose is safe for yaw/hover but not translation."""
    pose = gate.get("pose") if isinstance(gate, dict) else None
    if not isinstance(pose, dict):
        return False
    return bool(
        pose.get("rotation_position_locked")
        or pose.get("translation_allowed") is False
    )


def step_target_position(step: dict[str, Any]) -> list[float] | None:
    return vector3(step.get("to")) or vector3(step.get("at"))


def horizontal_xz_distance(a: list[float] | None, b: list[float] | None) -> float | None:
    if not a or not b:
        return None
    return ((a[0] - b[0]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def published_position_advanced_toward_target(
    before: list[float] | None,
    after: list[float] | None,
    target: list[float] | None,
    *,
    minimum_improvement: float = 0.015,
) -> bool:
    """Require visible room-position progress after a translation pulse."""
    before_distance = horizontal_xz_distance(before, target)
    after_distance = horizontal_xz_distance(after, target)
    return bool(
        before_distance is not None
        and after_distance is not None
        and after_distance <= before_distance - max(0.005, float(minimum_improvement))
    )


def patrol_translation_pulse_progress_issue(
    before: list[float] | None,
    after: list[float] | None,
    target: list[float] | None,
    *,
    got_new_pose: bool,
    maximum_pose_step: float,
    minimum_improvement: float = 0.015,
) -> str | None:
    """Require one trustworthy room-position result for every patrol pulse.

    The DJI bridge is sequential: execute_rc_pulse() always neutralizes the
    aircraft before this check runs.  Returning an issue therefore keeps the
    aircraft in neutral-hover recovery and, critically, prevents a second
    translation pulse from being issued against a frozen displayed pose.
    """
    if not got_new_pose:
        return "no newly processed localization result followed the translation pulse"
    if before is None or after is None or target is None:
        return "translation pulse has no complete before/after room position"
    pose_step = horizontal_xz_distance(before, after)
    if pose_step is None or not math.isfinite(pose_step):
        return "translation pulse produced an invalid room-position step"
    if pose_step > max(0.05, float(maximum_pose_step)):
        return (
            f"translation pulse produced a {pose_step:.3f}m pose jump "
            f"(limit {maximum_pose_step:.3f}m)"
        )
    if not published_position_advanced_toward_target(
        before,
        after,
        target,
        minimum_improvement=minimum_improvement,
    ):
        return (
            "new localization did not advance the published room position "
            "after one translation pulse"
        )
    return None


def horizontal_xz_segment_distance(
    point: list[float] | None,
    start: list[float] | None,
    end: list[float] | None,
) -> float | None:
    """Shortest horizontal distance from a pose to a finite route segment."""
    if point is None or start is None or end is None:
        return None
    sx, sz = float(start[0]), float(start[2])
    ex, ez = float(end[0]), float(end[2])
    px, pz = float(point[0]), float(point[2])
    dx, dz = ex - sx, ez - sz
    length_sq = dx * dx + dz * dz
    if length_sq <= 1e-12:
        return math.hypot(px - sx, pz - sz)
    projection = max(0.0, min(1.0, ((px - sx) * dx + (pz - sz) * dz) / length_sq))
    return math.hypot(px - (sx + projection * dx), pz - (sz + projection * dz))


def route_segment_progress_xz(
    point: list[float] | None,
    start: list[float] | None,
    end: list[float] | None,
) -> float | None:
    if point is None or start is None or end is None:
        return None
    dx = float(end[0]) - float(start[0])
    dz = float(end[2]) - float(start[2])
    length_sq = dx * dx + dz * dz
    if length_sq <= 1e-12:
        return None
    return (
        (float(point[0]) - float(start[0])) * dx
        + (float(point[2]) - float(start[2])) * dz
    ) / length_sq


def route_line_cross_track_xz(
    point: list[float] | None,
    start: list[float] | None,
    end: list[float] | None,
) -> float | None:
    """Horizontal distance to the infinite line through one route leg.

    Unlike ``horizontal_xz_segment_distance``, this deliberately does not
    clamp the projection to the finite segment. It lets the lap-boundary
    guard distinguish an aircraft that is safely *behind* Point 1 on the
    outgoing Point-1->2 centerline from an aircraft that is laterally off the
    patrol route.
    """
    progress = route_segment_progress_xz(point, start, end)
    if progress is None or point is None or start is None or end is None:
        return None
    nearest_x = float(start[0]) + progress * (float(end[0]) - float(start[0]))
    nearest_z = float(start[2]) + progress * (float(end[2]) - float(start[2]))
    return math.hypot(float(point[0]) - nearest_x, float(point[2]) - nearest_z)


def patrol_endpoint_overshoot_distance(
    point: list[float] | None,
    segment_start: list[float] | None,
    segment_end: list[float] | None,
) -> float | None:
    """Return signed-route distance beyond the active endpoint, if any."""
    progress = route_segment_progress_xz(point, segment_start, segment_end)
    if progress is None or segment_start is None or segment_end is None:
        return None
    leg_length = horizontal_xz_distance(segment_start, segment_end)
    if leg_length is None or leg_length <= 1e-9:
        return None
    return max(0.0, (float(progress) - 1.0) * leg_length)


def patrol_endpoint_undershoot_distance(
    point: list[float] | None,
    segment_start: list[float] | None,
    segment_end: list[float] | None,
) -> float | None:
    """Return signed-route distance still remaining before the endpoint."""
    progress = route_segment_progress_xz(point, segment_start, segment_end)
    if progress is None or segment_start is None or segment_end is None:
        return None
    leg_length = horizontal_xz_distance(segment_start, segment_end)
    if leg_length is None or leg_length <= 1e-9:
        return None
    return max(0.0, (1.0 - float(progress)) * leg_length)


def patrol_endpoint_undershoot_correction_allowed(
    undershoot_distance: float | None,
    current_distance: float | None,
    arrival_radius: float,
    *,
    endpoint_arrived: bool,
    retry_used: bool,
) -> bool:
    """Allow one forward correction only beyond the intended 8 cm residual."""
    return bool(
        not endpoint_arrived
        and not retry_used
        and undershoot_distance is not None
        and float(undershoot_distance)
        > ENDPOINT_UNDERSHOOT_CORRECTION_THRESHOLD_M
        and current_distance is not None
        and float(current_distance) <= float(arrival_radius)
    )


def patrol_route_pose_rejection(
    candidate: list[float] | None,
    *,
    segment_start: list[float] | None,
    segment_end: list[float] | None,
    previous_progress: float | None,
    translation_locked: bool,
    position_anchor: list[float] | None,
    max_cross_track: float,
    backward_tolerance: float = 0.08,
    turn_max_drift: float = 0.16,
    endpoint_overshoot_correction: bool = False,
) -> tuple[str | None, float | None]:
    """Second, command-side defense against off-route/backward poses."""
    if candidate is None or segment_start is None or segment_end is None:
        return None, previous_progress
    guarded_candidate = candidate
    if translation_locked:
        anchor = position_anchor or segment_start
        drift = horizontal_xz_distance(candidate, anchor)
        if drift is not None and drift > max(0.05, float(turn_max_drift)):
            return (
                f"route turn position drift {drift:.3f}m exceeds {turn_max_drift:.3f}m",
                previous_progress,
            )
        # A yaw-only command cannot translate the drone.  The localizer-side
        # route gate already evaluates progress at this fixed command anchor;
        # the duplicate bridge gate must do the same.  Using the raw TSolve
        # candidate here turned ordinary monocular yaw drift into a false
        # backward-progress abort immediately after takeoff.
        guarded_candidate = anchor
    cross_track = horizontal_xz_segment_distance(
        guarded_candidate,
        segment_start,
        segment_end,
    )
    if cross_track is not None and cross_track > max(0.10, float(max_cross_track)):
        return (
            f"route cross-track error {cross_track:.3f}m exceeds {max_cross_track:.3f}m",
            previous_progress,
        )
    progress = route_segment_progress_xz(
        guarded_candidate,
        segment_start,
        segment_end,
    )
    if progress is None:
        return "route progress is unavailable", previous_progress
    endpoint_rollback = bool(
        endpoint_overshoot_correction
        and not translation_locked
        and previous_progress is not None
        and float(previous_progress) >= 0.90
        and 0.90 <= progress <= 1.20
        and progress
        <= float(previous_progress) + max(0.0, float(backward_tolerance))
    )
    if (
        previous_progress is not None
        and progress
        < previous_progress - max(0.0, float(backward_tolerance))
        and not endpoint_rollback
    ):
        return (
            f"route progress moved backward ({progress:.3f} < {previous_progress - backward_tolerance:.3f})",
            previous_progress,
        )
    if progress < -0.16 or progress > 1.20:
        return f"route progress {progress:.3f} is outside the active segment", previous_progress
    accepted_progress = (
        progress
        if previous_progress is None or endpoint_rollback
        else max(previous_progress, progress)
    )
    return None, accepted_progress


def patrol_visual_recovery_reconciliation_ready(
    pose: dict[str, Any] | None,
    *,
    previous_progress: float | None,
    candidate_progress: float | None,
    state: dict[str, Any],
    required_observations: int = 3,
    max_progress_rollback: float = 0.08,
    max_progress_jitter: float = 0.012,
) -> bool:
    """Confirm a small stale-progress correction from the pinned visual route.

    The monotonic gate normally refuses every backwards pose. During a neutral
    recovery hover that can deadlock when the controller is slightly ahead of
    a strong, repeated baseline match: the drone cannot move until the pose is
    accepted, and the pose cannot advance while the drone is held. Only the
    caller decides whether the aircraft is in that neutral recovery state.
    This helper verifies that the replacement itself is independently strong,
    repeated, and bounded; it never applies to an ordinary TSolve pose.
    """
    def reset() -> bool:
        state.clear()
        return False

    if not isinstance(pose, dict) or pose.get("pose_source") != "patrol_visual_route_recovery":
        return reset()
    if pose.get("route_visual_verified") is not True:
        return reset()
    if pose.get("route_visual_monitor_verified") is not True:
        return reset()
    if pose.get("rotation_position_locked") is True or pose.get("translation_allowed") is False:
        return reset()
    try:
        previous = float(previous_progress)
        candidate = float(candidate_progress)
        visual_progress = float(pose.get("route_visual_progress"))
        monitor_progress = float(pose.get("route_visual_monitor_progress"))
        inliers = int(pose.get("route_visual_inliers") or 0)
        monitor_inliers = int(pose.get("route_visual_monitor_inliers") or 0)
        minimum = max(
            120,
            int(pose.get("route_visual_minimum_inliers") or 0),
            int(pose.get("route_visual_monitor_minimum_inliers") or 0),
        )
        acquisition_hits = int(pose.get("route_visual_acquisition_hits") or 0)
    except (TypeError, ValueError):
        return reset()
    if not all(math.isfinite(value) for value in (previous, candidate, visual_progress, monitor_progress)):
        return reset()
    rollback = previous - candidate
    if rollback <= 0.045 or rollback > max(0.045, float(max_progress_rollback)):
        return reset()
    if abs(visual_progress - candidate) > 0.005 or abs(monitor_progress - candidate) > 0.005:
        return reset()
    if inliers < minimum or monitor_inliers < minimum or acquisition_hits < 2:
        return reset()

    instance_id = str(pose.get("instance_id") or "")
    if not instance_id or instance_id == state.get("last_instance_id"):
        return False
    prior_candidate = state.get("candidate_progress")
    try:
        stable = prior_candidate is not None and abs(float(prior_candidate) - candidate) <= max_progress_jitter
    except (TypeError, ValueError):
        stable = False
    state["streak"] = int(state.get("streak") or 0) + 1 if stable else 1
    state["candidate_progress"] = candidate
    state["last_instance_id"] = instance_id
    if state["streak"] < max(2, int(required_observations)):
        return False
    state.clear()
    return True


def patrol_metric_recovery_reconciliation_ready(
    gate: dict[str, Any] | None,
    *,
    previous_progress: float | None,
    candidate_progress: float | None,
    state: dict[str, Any],
    required_observations: int = 5,
    max_progress_rollback: float = 0.25,
    max_progress_jitter: float = 0.015,
) -> bool:
    """Confirm that a neutral-hover metric consensus disproves a false ratchet.

    A series of individually small TSolve errors can walk the trusted route
    progress forward without tripping the per-frame jump limit.  Once the
    solve returns to the real, stationary location the ordinary monotonic gate
    would reject it forever.  During neutral recovery only, accept a bounded
    rollback after five fresh metric poses agree on the same lower progress.
    """

    def reset() -> bool:
        state.clear()
        return False

    if not pose_gate_has_fresh_metric_position(gate):
        return reset()
    try:
        previous = float(previous_progress)
        candidate = float(candidate_progress)
    except (TypeError, ValueError):
        return reset()
    if not math.isfinite(previous) or not math.isfinite(candidate):
        return reset()
    rollback = previous - candidate
    if rollback <= 0.045 or rollback > max(0.045, float(max_progress_rollback)):
        return reset()

    pose = gate.get("pose") if isinstance(gate, dict) else None
    instance_id = str((pose or {}).get("instance_id") or "")
    if not instance_id or instance_id == state.get("last_instance_id"):
        return False
    values = [
        float(value)
        for value in state.get("candidate_progress_values", [])
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    trial = values + [candidate]
    if values and max(trial) - min(trial) > max(0.001, float(max_progress_jitter)):
        trial = [candidate]
    state["candidate_progress_values"] = trial
    state["last_instance_id"] = instance_id
    if len(trial) < max(3, int(required_observations)):
        return False
    state.clear()
    return True


def patrol_visual_translation_resume_ready(pose: dict[str, Any] | None) -> bool:
    """Authorize only a bounded next pulse from a strong pinned-route match.

    This does not declare metric relocalization or authorize continuous
    flight. It lets the cruise loop leave neutral recovery for at most its
    normal three tiny pulses, after which live route progress is checked again.
    """
    if not isinstance(pose, dict):
        return False
    if pose.get("pose_source") != "patrol_visual_route_recovery":
        return False
    if pose.get("route_visual_verified") is not True:
        return False
    if pose.get("route_visual_monitor_verified") is not True:
        return False
    if pose.get("route_visual_translation_safe") is not True:
        return False
    if pose.get("rotation_position_locked") is True or pose.get("translation_allowed") is False:
        return False
    # Weak endpoint consensus may declare arrival while hovering, but it must
    # never authorize departure from an intermediate location.
    if pose.get("route_visual_weak_endpoint_recovery") is True:
        return False
    try:
        progress = float(pose.get("route_visual_progress"))
        monitor_progress = float(pose.get("route_visual_monitor_progress"))
        inliers = int(pose.get("route_visual_inliers") or 0)
        monitor_inliers = int(pose.get("route_visual_monitor_inliers") or 0)
        temporal_recovery = bool(
            pose.get("route_visual_temporal_recovery") is True
        )
        leg_index = int(pose.get("route_visual_monitor_leg_index") or 0)
        minimum = max(
            (
                50
                if temporal_recovery and leg_index == 4
                else (90 if temporal_recovery else 120)
            ),
            int(pose.get("route_visual_minimum_inliers") or 0),
            int(pose.get("route_visual_monitor_minimum_inliers") or 0),
        )
        acquisition_hits = int(pose.get("route_visual_acquisition_hits") or 0)
        required_hits = (
            max(
                5,
                int(
                    pose.get("route_visual_temporal_recovery_required_hits")
                    or 0
                ),
            )
            if temporal_recovery
            else 2
        )
    except (TypeError, ValueError):
        return False
    return bool(
        math.isfinite(progress)
        and math.isfinite(monitor_progress)
        and 0.0 <= progress <= 1.0
        and abs(progress - monitor_progress) <= 0.01
        and inliers >= minimum
        and monitor_inliers >= minimum
        and acquisition_hits >= required_hits
    )


def patrol_visual_stationary_retry_ready(
    pose: dict[str, Any] | None,
    *,
    retry_available: bool,
    observed_translation_progress: bool,
    expected_leg_index: int | None = None,
) -> bool:
    """Permit one low-stick retry from strong live route-image consensus.

    This helper never authorizes an unbounded cruise. The caller consumes the
    retry before returning to physical control and does not replenish it until
    ATLAS observes at least 1.5 cm of progress toward the target. That breaks
    the Point-3 hover deadlock without allowing repeated commands from a frozen
    model position. Point 4 -> Point 1 is intentionally excluded: its dense
    ordered image bank must confirm each command before another command.
    """
    return bool(
        int(expected_leg_index or 0) != 4
        and retry_available
        and not observed_translation_progress
        and patrol_visual_translation_resume_ready(pose)
    )


def advance_route_command_progress_ceiling(
    current_ceiling: float | None,
    trusted_progress: float | None,
    command_progress_budget: float,
) -> float:
    """Accumulate every real horizontal command's unresolved route budget.

    The published pose can lag the aircraft by more than one RC command.  A
    new command must therefore extend the existing physical envelope, not
    replace it with ``published_progress + one_command``.  Replacing the
    envelope made three Point-2 -> Point-3 commands advertise only 0.287 m of
    possible travel instead of the commanded 0.54 m, excluding the real live
    frames from visual relocalization.

    This function is called only for a horizontal command that has passed the
    command-side pose and geofence gates.  Yaw and neutral hover never call it
    and therefore never gain translation budget.
    """
    bases = [
        float(value)
        for value in (current_ceiling, trusted_progress)
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    base = max(bases or [0.0])
    try:
        budget = float(command_progress_budget)
    except (TypeError, ValueError):
        budget = 0.0
    if not math.isfinite(budget):
        budget = 0.0
    return min(1.0, max(0.0, base) + max(0.0, budget))


def patrol_visual_yaw_anchor_ready(gate: dict[str, Any] | None) -> bool:
    """Whether one fresh route pose can safely anchor the next in-place yaw.

    A verified full-loop match is an absolute observation of the current
    recorded route location, so it does not need two additional TSolve samples
    before rotation.  Held/fallback or still-reconciling visual poses remain
    ineligible and continue through the normal settle loop.
    """
    if not isinstance(gate, dict) or gate.get("ok") is not True:
        return False
    if gate.get("recent_hold_fallback") is True:
        return False
    pose = gate.get("pose")
    return bool(
        patrol_visual_translation_resume_ready(pose)
        and pose_gate_position(gate) is not None
    )


def point_in_xz_polygon(point: list[float], polygon: list[list[float]]) -> bool:
    if len(polygon) < 3:
        return False
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, zi = polygon[i][0], polygon[i][2]
        xj, zj = polygon[j][0], polygon[j][2]
        crosses = (zi > point[2]) != (zj > point[2])
        if crosses and point[0] < (xj - xi) * (point[2] - zi) / ((zj - zi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def closed_wall_ring(barriers: list[dict[str, Any]]) -> list[list[float]] | None:
    """Return an ordered room footprint only when the saved walls form a ring."""
    if len(barriers) < 3:
        return None
    ring: list[list[float]] = []
    previous_b: list[float] | None = None
    for barrier in barriers:
        a = vector3(barrier.get("a"))
        b = vector3(barrier.get("b"))
        if a is None or b is None:
            return None
        if previous_b is not None and horizontal_xz_distance(previous_b, a) > 0.08:
            return None
        ring.append(a)
        previous_b = b
    if previous_b is None or horizontal_xz_distance(previous_b, ring[0]) > 0.08:
        return None
    return ring


def point_aabb_distance(point: list[float], bounds: dict[str, Any]) -> float | None:
    lower = vector3(bounds.get("min")) if isinstance(bounds, dict) else None
    upper = vector3(bounds.get("max")) if isinstance(bounds, dict) else None
    if lower is None or upper is None:
        return None
    total = 0.0
    for axis in range(3):
        lo, hi = min(lower[axis], upper[axis]), max(lower[axis], upper[axis])
        delta = lo - point[axis] if point[axis] < lo else point[axis] - hi if point[axis] > hi else 0.0
        total += delta * delta
    return math.sqrt(total)


def pursuit_geofence_issue(
    position: list[float] | None,
    barriers: list[dict[str, Any]],
    obstacles: list[dict[str, Any]],
    *,
    motion_buffer_m: float = 0.30,
) -> str | None:
    if position is None:
        return "flight has no room-frame position for geofence validation"
    ring = closed_wall_ring(barriers)
    if ring is None:
        return "selected map does not have a closed saved-wall geofence"
    if not point_in_xz_polygon(position, ring):
        return "localized drone pose is outside the saved-wall geofence"
    wall_floor = min(
        point[1]
        for barrier in barriers
        for point in (vector3(barrier.get("a")), vector3(barrier.get("b")))
        if point is not None
    )
    wall_tops = [
        point[1]
        for barrier in barriers
        for point in [*(vector3(corner) for corner in barrier.get("corners") or []), vector3(barrier.get("a")), vector3(barrier.get("b"))]
        if point is not None
    ]
    wall_top = max(wall_tops) if wall_tops else wall_floor + 3.0
    if position[1] <= wall_floor + motion_buffer_m or position[1] >= wall_top - motion_buffer_m:
        return "localized drone altitude is outside the saved room flight envelope"
    for barrier in barriers:
        a = vector3(barrier.get("a"))
        b = vector3(barrier.get("b"))
        distance = horizontal_xz_segment_distance(position, a, b)
        clearance = clamp_float(barrier.get("clearance_m"), 0.45, 0.15, 2.0) + motion_buffer_m
        if distance is not None and distance <= clearance:
            return (
                f"{str(barrier.get('label') or 'saved wall')} is {distance:.2f} m away; "
                f"flight requires {clearance:.2f} m including pulse buffer"
            )
    for obstacle in obstacles:
        distance = point_aabb_distance(position, obstacle.get("bounds") if isinstance(obstacle, dict) else {})
        clearance = clamp_float(obstacle.get("clearance_m"), 0.35, 0.10, 2.0) + motion_buffer_m
        if distance is not None and distance <= clearance:
            return (
                f"{str(obstacle.get('label') or 'saved obstacle')} is {distance:.2f} m away; "
                f"flight requires {clearance:.2f} m including pulse buffer"
            )
    return None


def mission_step_sequence(
    commands: list[Any],
    patrol_loop: bool,
    patrol_laps: int | None = None,
    loop_start_index: int = 0,
):
    """Yield the one-time entry followed by each complete patrol lap.

    Browser patrol plans include a current-position -> point-1 entry leg before
    the closed point-1 -> ... -> point-1 loop.  That entry runs only once.
    At every later lap, the first loop cruise preserves the verified 4 -> 1
    endpoint as a yaw-only anchor, turns toward Point 2, and performs the fresh
    metric relocalization only after the heading is aligned.  Horizontal RC
    remains locked until that post-turn metric checkpoint succeeds.
    """
    if not patrol_loop:
        return iter(enumerate(commands))
    start = max(0, min(len(commands), int(loop_start_index)))
    if start >= len(commands):
        start = 0
    prefix = tuple(enumerate(commands[:start]))
    loop_body = tuple((index, commands[index]) for index in range(start, len(commands)))

    def marked_loop(lap_number: int):
        for offset, (index, command) in enumerate(loop_body):
            if isinstance(command, dict):
                command = dict(command)
                command["_atlas_lap_number"] = lap_number
                if offset == 0:
                    command["_atlas_lap_start"] = True
            yield index, command

    def finite_sequence(laps: int):
        yield from prefix
        for lap_number in range(1, laps + 1):
            yield from marked_loop(lap_number)

    def continuous_sequence():
        yield from prefix
        lap_number = 1
        while True:
            yield from marked_loop(lap_number)
            lap_number += 1

    if patrol_laps is None or int(patrol_laps) <= 0:
        return continuous_sequence()
    return finite_sequence(int(patrol_laps))


def patrol_loop_start_command_index(
    commands: list[Any],
    taught_reference: dict[str, Any] | None,
) -> int:
    """Find the first actual circle leg (point 1 -> point 2)."""
    def include_leg_yaw(cruise_index: int) -> int:
        # A repeated lap must turn from the final 4->1 heading back toward
        # 1->2 before translating. Connected packets place that yaw directly
        # before the first loop cruise, after the one-time entry prefix.
        previous = cruise_index - 1
        if previous >= 0 and isinstance(commands[previous], dict):
            if str(commands[previous].get("type") or "").strip().lower() == "yaw":
                return previous
        return cruise_index

    cruise_steps = [
        step
        for step in commands
        if isinstance(step, dict) and str(step.get("type") or "").strip().lower() == "cruise"
    ]
    reference_leg_count = len(taught_reference.get("legs") or []) if isinstance(taught_reference, dict) else 0
    first_cruise_leg: dict[str, Any] | None = None
    if cruise_steps:
        first_cruise_leg = taught_leg_for_step(taught_reference, cruise_steps[0])
    if (
        first_cruise_leg is not None
        and reference_leg_count > 0
        and len(cruise_steps) <= reference_leg_count
    ):
        # The packet already begins on the closed circle (the fleet controller
        # can rotate the circle to the nearest waypoint). Repeat it as-is.
        return 0
    for index, step in enumerate(commands):
        if not isinstance(step, dict) or str(step.get("type") or "").strip().lower() != "cruise":
            continue
        leg = taught_leg_for_step(taught_reference, step)
        if not isinstance(leg, dict):
            continue
        try:
            if int(leg.get("from_point")) == 1 and int(leg.get("to_point")) == 2:
                return include_leg_yaw(index)
        except (TypeError, ValueError):
            continue
    # Current ATLAS route packets call the entry "Patrol cruise 1" and the
    # first closed-loop leg "Patrol cruise 2". Keep this only as a packet-shape
    # fallback; endpoint matching above remains authoritative.
    for index, step in enumerate(commands):
        if isinstance(step, dict) and re.fullmatch(
            r"\s*Patrol cruise\s+2\s*",
            str(step.get("title") or ""),
            re.IGNORECASE,
        ):
            return include_leg_yaw(index)
    return 0


def bounded_pose_step_limit(
    previous_received_unix: Any,
    current_received_unix: Any,
    *,
    base_limit: float = 0.30,
    hard_limit: float = 0.55,
) -> float:
    """Allow short skipped-frame travel without ever accepting a large jump."""
    try:
        dt = max(0.0, float(current_received_unix) - float(previous_received_unix))
    except (TypeError, ValueError):
        return base_limit
    return max(base_limit, min(hard_limit, 0.18 + 0.85 * dt))


def command_bounded_pose_catchup_ready(
    pose: dict[str, Any] | None,
    candidate: list[float] | None,
    trusted: list[float] | None,
    *,
    segment_start: list[float] | None,
    segment_end: list[float] | None,
    trusted_progress: float | None,
    command_progress_ceiling: float | None,
    command_sequence: int,
    step: float,
    hard_limit: float = 0.55,
    max_raw_cross_track: float = 0.20,
) -> bool:
    """Accept delayed forward evidence only inside issued physical motion.

    The ordinary adjacent-pose limit remains unchanged.  This narrow exception
    exists for the Point-2->Point-3 failure where several real RC commands had
    advanced the aircraft, but the first accepted camera result arrived 0.45-
    0.49 m beyond the frozen model and was rejected by the 0.30 m gate.  Yaw
    and hover cannot create this authority: the bridge increments ``sequence``
    and ``ceiling`` only after a horizontal command passes every safety gate.
    """
    if not isinstance(pose, dict) or candidate is None or trusted is None:
        return False
    if (
        pose.get("held_pose") is True
        or pose.get("output_rejected") is True
        or pose.get("rotation_position_locked") is True
        or pose.get("translation_allowed") is False
    ):
        return False
    try:
        sequence = int(command_sequence)
        ceiling = float(command_progress_ceiling)
        previous = float(trusted_progress)
        distance = float(step)
        limit = max(0.30, min(0.55, float(hard_limit)))
    except (TypeError, ValueError):
        return False
    if (
        sequence <= 0
        or not all(math.isfinite(value) for value in (ceiling, previous, distance))
        or distance <= 0.30
        or distance > limit
    ):
        return False
    candidate_progress = route_segment_progress_xz(
        candidate,
        segment_start,
        segment_end,
    )
    leg_length = horizontal_xz_distance(segment_start, segment_end)
    if candidate_progress is None or leg_length is None or leg_length <= 1e-9:
        return False
    raw_cross_track = pose.get("route_cross_track_m")
    try:
        raw_cross_track_value = float(raw_cross_track)
    except (TypeError, ValueError):
        raw_cross_track_value = horizontal_xz_segment_distance(
            candidate,
            segment_start,
            segment_end,
        )
    if (
        raw_cross_track_value is None
        or not math.isfinite(float(raw_cross_track_value))
        or float(raw_cross_track_value) > max(0.05, float(max_raw_cross_track))
    ):
        return False
    forward_distance = (float(candidate_progress) - previous) * leg_length
    commanded_distance = max(0.0, (ceiling - previous) * leg_length)
    return bool(
        candidate_progress >= previous - 0.008
        and candidate_progress <= ceiling + 0.008
        and forward_distance >= -0.01
        and forward_distance <= commanded_distance + 0.02
    )


def corridor_recovery_speed_scale(
    cross_track: float | None,
    *,
    recovery_start: float,
    hard_limit: float,
) -> float:
    """Reduce translation while steering back, without stalling recovery."""
    if cross_track is None or cross_track <= recovery_start:
        return 1.0
    span = max(1e-6, hard_limit - recovery_start)
    return max(0.30, min(1.0, 1.0 - (cross_track - recovery_start) / span))


def normalize_xz(vec: list[float] | None) -> list[float] | None:
    if not vec:
        return None
    try:
        x = float(vec[0])
        z = float(vec[2])
    except (TypeError, ValueError, IndexError):
        return None
    n = math.hypot(x, z)
    if n < 1e-9 or not math.isfinite(n):
        return None
    return [x / n, 0.0, z / n]


def dot_xz(a: list[float] | None, b: list[float] | None) -> float | None:
    aa = normalize_xz(a)
    bb = normalize_xz(b)
    if aa is None or bb is None:
        return None
    return aa[0] * bb[0] + aa[2] * bb[2]


def observed_motion_axis(
    before: list[float] | None,
    after: list[float] | None,
    *,
    min_delta: float = 0.018,
) -> tuple[list[float] | None, float | None]:
    if before is None or after is None:
        return None, None
    dx = float(after[0]) - float(before[0])
    dz = float(after[2]) - float(before[2])
    norm = math.hypot(dx, dz)
    # A probe may explicitly ask to observe sub-threshold motion with
    # ``min_delta=0``. A fresh localization frame can still contain the exact
    # same position while the aircraft has not begun moving. Never normalize
    # that zero vector; report that no body axis was measured instead.
    if not math.isfinite(norm) or norm <= max(1e-9, float(min_delta)):
        return None, norm
    return [dx / norm, 0.0, dz / norm], norm


def rotate_xz(vec: list[float] | None, radians: float) -> list[float] | None:
    base = normalize_xz(vec)
    if base is None:
        return None
    c = math.cos(radians)
    s = math.sin(radians)
    return normalize_xz([base[0] * c - base[2] * s, 0.0, base[0] * s + base[2] * c])


def pose_gate_heading(gate: dict[str, Any] | None, heading_trim_rad: float = 0.0) -> list[float] | None:
    pose = gate.get("pose") if isinstance(gate, dict) else None
    if not isinstance(pose, dict):
        return None
    return rotate_xz(vector3(pose.get("rheading")), heading_trim_rad)


def signed_angle_xz(from_heading: list[float] | None, to_direction: list[float] | None) -> float | None:
    a = normalize_xz(from_heading)
    b = normalize_xz(to_direction)
    if a is None or b is None:
        return None
    cross = a[0] * b[2] - a[2] * b[0]
    dot = a[0] * b[0] + a[2] * b[2]
    return math.atan2(cross, dot)


def heading_calibration_error_degrees(
    camera_heading: list[float] | None,
    measured_forward: list[float] | None,
    expected_offset_deg: float,
) -> float | None:
    """Return disagreement between a measured body axis and fixed camera mounting."""
    expected_forward = rotate_xz(camera_heading, math.radians(float(expected_offset_deg)))
    error = signed_angle_xz(expected_forward, measured_forward)
    if error is None or not math.isfinite(error):
        return None
    return abs(math.degrees(error))


def yaw_response_is_reversed(
    expected_heading_sign: float,
    before_heading: list[float] | None,
    after_heading: list[float] | None,
    *,
    minimum_delta_deg: float = 1.0,
) -> bool | None:
    """Classify a fresh yaw response without relying on target-distance changes."""
    observed_delta = signed_angle_xz(before_heading, after_heading)
    if (
        observed_delta is None
        or abs(float(expected_heading_sign)) <= 0.5
        or abs(observed_delta) < math.radians(max(0.1, float(minimum_delta_deg)))
    ):
        return None
    return observed_delta * float(expected_heading_sign) < 0.0


def yaw_target_error_response(
    before_error_rad: float | None,
    after_error_rad: float | None,
    *,
    improvement_deg: float = 0.75,
    regression_deg: float = 1.5,
) -> bool | None:
    """Classify yaw using comparable target-heading errors.

    ``True`` means the absolute target error clearly became worse, ``False``
    means it clearly improved, and ``None`` means the fresh observation is too
    small/noisy to vote.  Both errors must be computed in the same heading
    frame; mixing route-camera error with body-heading delta caused the false
    sign reversal seen at Point 3 in the 15:04 flight.
    """
    if before_error_rad is None or after_error_rad is None:
        return None
    before = abs(float(before_error_rad))
    after = abs(float(after_error_rad))
    if not math.isfinite(before) or not math.isfinite(after):
        return None
    change = after - before
    if change >= math.radians(max(0.1, float(regression_deg))):
        return True
    if change <= -math.radians(max(0.1, float(improvement_deg))):
        return False
    return None


def yaw_sign_recovery_action(
    *,
    yaw_sign_verified: bool,
    wrong_yaw_pulses: int,
    yaw_flip_count: int,
    confirmation_pulses: int = GUIDED_YAW_SIGN_CONFIRMATION_PULSES,
) -> str | None:
    """Choose a bounded response to repeated yaw-error regressions.

    The DJI yaw polarity is a hardware property.  Once proven during this
    mission it must never be inverted because of delayed visual feedback.
    """
    required = max(1, int(confirmation_pulses)) if yaw_flip_count == 0 else 5
    if int(wrong_yaw_pulses) < required:
        return None
    if yaw_sign_verified:
        return "recover"
    if yaw_flip_count == 0:
        return "flip"
    return "abort"


def target_direction_xz(current: list[float] | None, target: list[float] | None) -> list[float] | None:
    if current is None or target is None:
        return None
    return [target[0] - current[0], 0.0, target[2] - current[2]]


def patrol_navigation_direction_xz(
    current: list[float] | None,
    target: list[float] | None,
    *,
    endpoint_heading: list[float] | None = None,
) -> list[float] | None:
    """Keep yaw control usable when TSolve is already exactly at a waypoint.

    A zero target vector means that translation is complete, not that the pose
    is unusable.  During endpoint image alignment, fall back to the verified
    route heading so the controller can finish yaw with forward stick locked.
    """
    target_direction = normalize_xz(target_direction_xz(current, target))
    if target_direction is not None:
        return target_direction
    return normalize_xz(endpoint_heading)


def patrol_zero_direction_arrival_allowed(
    current_distance: float | None,
    arrival_radius: float,
    *,
    precise_arrival: bool,
    endpoint_ready: bool,
) -> bool:
    """Treat a zero target vector as arrival only with required evidence."""
    try:
        distance = float(current_distance)
        radius = float(arrival_radius)
    except (TypeError, ValueError):
        return False
    return bool(
        math.isfinite(distance)
        and math.isfinite(radius)
        and distance <= max(0.0, radius)
        and (not precise_arrival or endpoint_ready)
    )


def vertical_rc_toward(current: list[float] | None, target: list[float] | None, max_vertical_rc: float) -> float:
    if current is None or target is None:
        return 0.0
    error = float(target[1]) - float(current[1])
    if abs(error) < 0.08:
        return 0.0
    return max(-max_vertical_rc, min(max_vertical_rc, error * 0.028))


def _rc_command(rcw: float, du: float, lr: float, bf: float) -> str:
    return (
        f"rc {max(-1.0, min(1.0, float(rcw))):.4f} "
        f"{max(-1.0, min(1.0, float(du))):.2f} "
        f"{max(-1.0, min(1.0, float(lr))):.2f} "
        f"{max(-1.0, min(1.0, float(bf))):.2f}"
    )


def _control_ack_success(response: Any) -> bool:
    return str(response or "").strip().lower() == "success"


def _send_rc_with_bounded_ack(
    drone: Any,
    *,
    rcw: float,
    du: float,
    lr: float,
    bf: float,
    timeout_seconds: float = CONTROL_NEUTRAL_ACK_TIMEOUT_SECONDS,
) -> str | None:
    """Send one RC state and bound how long confirmation may block.

    OpenDJI's public ``move(..., True)`` waits on its response queue without a
    working timeout.  The live bridge owns the control worker, so it can safely
    use the same socket and receiver queue while applying a real timeout.  A
    minimal fallback keeps unit-test and alternate drone adapters compatible.
    """
    control_socket = getattr(drone, "_socket_control", None)
    background = getattr(drone, "_background_control_messages", None)
    response_queue = getattr(background, "_queue", None)
    if control_socket is None or response_queue is None:
        response = drone.move(float(rcw), float(du), float(lr), float(bf), True)
        if response is not None and not _control_ack_success(response):
            raise RuntimeError(f"DJI rejected RC command: {response}")
        return response

    command = _rc_command(rcw, du, lr, bf)
    control_socket.sendall(bytes(command + "\r\n", "utf-8"))
    try:
        response = response_queue.get(
            block=True,
            timeout=max(0.05, float(timeout_seconds)),
        )
    except queue.Empty as exc:
        raise TimeoutError(
            f"DJI control acknowledgement timed out after {float(timeout_seconds):.2f}s"
        ) from exc
    if not _control_ack_success(response):
        raise RuntimeError(f"DJI rejected RC command: {response}")
    return str(response)


def _recv_control_ack(sock: Any, *, maximum_bytes: int = 4096) -> str:
    received = bytearray()
    while b"\r\n" not in received and len(received) < maximum_bytes:
        chunk = sock.recv(min(1024, maximum_bytes - len(received)))
        if not chunk:
            raise ConnectionError("DJI control socket closed before neutral acknowledgement")
        received.extend(chunk)
    if b"\r\n" not in received:
        raise RuntimeError("DJI control acknowledgement exceeded its bounded response size")
    return bytes(received).split(b"\r\n", 1)[0].decode("utf-8", errors="replace")


def reconnect_control_and_confirm_neutral(
    drone: Any,
    *,
    attempts: int = CONTROL_RECONNECT_ATTEMPTS,
    timeout_seconds: float = CONTROL_RECONNECT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Replace only OpenDJI's failed control channel and confirm zero sticks.

    Video and telemetry use separate sockets and remain alive.  Neutral is sent
    directly before a new background reader is installed so its acknowledgement
    cannot be consumed by a stale response queue.
    """
    old_socket = getattr(drone, "_socket_control", None)
    old_background = getattr(drone, "_background_control_messages", None)
    background_type = type(old_background) if old_background is not None else None
    host = getattr(drone, "host_address", None)
    port = getattr(drone, "PORT_CONTROL", None)
    if not host or port is None or background_type is None:
        raise ControlLinkSafetyError(
            "DJI neutral command failed and this control adapter cannot reconnect safely.",
            neutral_confirmed=False,
            control_link_recovered=False,
        )

    if old_background is not None:
        try:
            old_background.stop(timeout=0.20)
        except Exception:
            try:
                if old_socket is not None:
                    old_socket.close()
            except Exception:
                pass
    elif old_socket is not None:
        try:
            old_socket.close()
        except Exception:
            pass

    last_error: Exception | None = None
    for attempt in range(1, max(1, int(attempts)) + 1):
        replacement = None
        try:
            replacement = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            replacement.settimeout(max(0.10, float(timeout_seconds)))
            replacement.connect((str(host), int(port)))
            replacement.sendall(b"rc 0.0000 0.00 0.00 0.00\r\n")
            acknowledgement = _recv_control_ack(replacement)
            if not _control_ack_success(acknowledgement):
                raise RuntimeError(f"DJI rejected recovered neutral command: {acknowledgement}")
            replacement.settimeout(None)
            new_background = background_type(replacement)
            drone._socket_control = replacement
            drone._background_control_messages = new_background
            return {
                "neutral_confirmed": True,
                "control_link_recovered": True,
                "reconnect_attempt": attempt,
                "acknowledgement": acknowledgement,
            }
        except Exception as exc:
            last_error = exc
            if replacement is not None:
                try:
                    replacement.close()
                except Exception:
                    pass
            if attempt < max(1, int(attempts)):
                time.sleep(0.08)

    raise ControlLinkSafetyError(
        "DJI control link failed and ATLAS could not confirm neutral sticks after "
        f"{max(1, int(attempts))} reconnect attempts: {last_error}",
        neutral_confirmed=False,
        control_link_recovered=False,
    ) from last_error


def neutral_hover(drone: Any, seconds: float = 0.10) -> dict[str, Any]:
    try:
        acknowledgement = _send_rc_with_bounded_ack(
            drone,
            rcw=0.0,
            du=0.0,
            lr=0.0,
            bf=0.0,
        )
        result = {
            "neutral_confirmed": True,
            "control_link_recovered": False,
            "acknowledgement": acknowledgement,
        }
    except Exception:
        result = reconnect_control_and_confirm_neutral(drone)
    if seconds > 0:
        time.sleep(seconds)
    return result


def execute_rc_pulse(
    drone: Any,
    *,
    lr: float = 0.0,
    du: float = 0.0,
    bf: float = 0.0,
    yaw: float = 0.0,
    seconds: float = GUIDED_DEFAULT_PULSE_SECONDS,
) -> dict[str, Any]:
    # OpenDJI.move is ordered as rcw, du, lr, bf.  Keep this wrapper named in
    # navigation terms so mission code cannot accidentally swap yaw/forward.
    sent = {
        "rcw_yaw": round(float(yaw), 4),
        "du_vertical": round(float(du), 2),
        "lr_lateral": round(float(lr), 2),
        "bf_forward": round(float(bf), 2),
        "seconds": round(max(0.05, float(seconds)), 3),
    }
    try:
        _send_rc_with_bounded_ack(
            drone,
            rcw=float(yaw),
            du=float(du),
            lr=float(lr),
            bf=float(bf),
        )
    except Exception as command_error:
        try:
            neutral_result = neutral_hover(drone, 0.03)
        except ControlLinkSafetyError:
            raise
        sent.update(
            {
                "motion_command_acknowledged": False,
                "motion_outcome_uncertain": True,
                "neutral_confirmed": True,
                "control_link_recovered": bool(
                    neutral_result.get("control_link_recovered")
                ),
                "requires_pose_recovery": True,
                "control_error": str(command_error),
            }
        )
        return sent
    time.sleep(max(0.05, float(seconds)))
    neutral_result = neutral_hover(drone, 0.03)
    if neutral_result.get("control_link_recovered"):
        sent.update(
            {
                "motion_command_acknowledged": True,
                "motion_outcome_uncertain": True,
                "neutral_confirmed": True,
                "control_link_recovered": True,
                "requires_pose_recovery": True,
            }
        )
    return sent


def enemy_box_measurement(detection: dict[str, Any]) -> dict[str, float] | None:
    box = detection.get("box") if isinstance(detection, dict) else None
    if not isinstance(box, dict):
        return None
    try:
        x1 = float(box.get("x1") or 0.0)
        y1 = float(box.get("y1") or 0.0)
        width = float(box.get("width") or (float(box.get("x2") or 0.0) - x1))
        height = float(box.get("height") or (float(box.get("y2") or 0.0) - y1))
    except (TypeError, ValueError):
        return None
    area = width * height
    if not all(math.isfinite(value) for value in (x1, y1, width, height, area)):
        return None
    if width < 0.001 or height < 0.001 or area < 1e-6 or width > 1.0 or height > 1.0:
        return None
    return {
        "center_x": max(0.0, min(1.0, x1 + width * 0.5)),
        "center_y": max(0.0, min(1.0, y1 + height * 0.5)),
        "width": width,
        "height": height,
        "area": area,
    }


def estimate_enemy_clearance(measurement: dict[str, float], range_model: dict[str, Any]) -> tuple[float, float]:
    model_type = str(range_model.get("type") or "")
    if model_type == "inverse_width":
        feature = float(measurement.get("width") or 0.0)
    elif model_type == "inverse_height":
        feature = float(measurement.get("height") or 0.0)
    elif model_type == "inverse_sqrt_area":
        feature = math.sqrt(max(0.0, float(measurement.get("area") or 0.0)))
    else:
        raise RuntimeError("unsupported enemy range model")
    scale = clamp_float(range_model.get("scale"), 0.0, 0.0, 100.0)
    margin = clamp_float(range_model.get("conservative_margin_m"), 0.20, 0.05, 1.0)
    if scale <= 0.0 or not math.isfinite(feature) or feature <= 1e-6:
        raise RuntimeError("enemy range calibration scale is invalid")
    estimate = scale / feature
    if not math.isfinite(estimate) or estimate <= 0.0:
        raise RuntimeError("enemy range estimate is invalid")
    return estimate, margin


def predict_enemy_clearance(history: list[dict[str, float]], horizon_seconds: float = 0.35) -> tuple[float, float]:
    """Return a smoothed range plus an approach-only short-horizon prediction.

    Increasing range never authorizes extra forward motion.  Only a negative
    range slope is projected, so an enemy moving toward the DJI aircraft makes
    the controller brake earlier while a noisy/receding estimate does not make
    it more aggressive.
    """
    if not history:
        raise RuntimeError("enemy range history is empty")
    recent = history[-5:]
    values = [float(item["clearance_m"]) for item in recent]
    smoothed = sorted(values[-3:])[len(values[-3:]) // 2]
    if len(recent) < 3:
        return smoothed, smoothed
    times = [float(item["updated_at"]) for item in recent]
    mean_t = sum(times) / len(times)
    mean_value = sum(values) / len(values)
    denom = sum((value - mean_t) ** 2 for value in times)
    if denom < 1e-6:
        return smoothed, smoothed
    slope = sum((t - mean_t) * (value - mean_value) for t, value in zip(times, values)) / denom
    approach_rate = min(0.0, max(-2.0, slope))
    predicted = max(0.0, smoothed + approach_rate * max(0.0, horizon_seconds))
    return smoothed, predicted


def predict_enemy_image_center(history: list[dict[str, float]], horizon_seconds: float = 0.30) -> tuple[float, float]:
    if not history:
        return 0.5, 0.5
    recent = history[-5:]
    latest = recent[-1]
    if len(recent) < 2:
        return float(latest["center_x"]), float(latest["center_y"])
    times = [float(item["updated_at"]) for item in recent]
    mean_t = sum(times) / len(times)
    denom = sum((value - mean_t) ** 2 for value in times)
    if denom < 1e-6:
        return float(latest["center_x"]), float(latest["center_y"])

    def projected(axis: str) -> float:
        values = [float(item[axis]) for item in recent]
        mean_value = sum(values) / len(values)
        slope = sum((t - mean_t) * (value - mean_value) for t, value in zip(times, values)) / denom
        slope = max(-0.8, min(0.8, slope))
        return max(0.0, min(1.0, float(latest[axis]) + slope * max(0.0, horizon_seconds)))

    return projected("center_x"), projected("center_y")


def select_tracked_enemy_detection(
    payload: dict[str, Any],
    target_class_name: str,
    previous_center: tuple[float, float] | None = None,
    minimum_confidence: float = 0.35,
) -> tuple[dict[str, Any], dict[str, float]] | None:
    candidates: list[tuple[float, dict[str, Any], dict[str, float]]] = []
    expected = str(target_class_name or "").strip().lower()
    for detection in payload.get("detections") or []:
        if not isinstance(detection, dict):
            continue
        class_name = str(detection.get("class_name") or "").strip().lower()
        if expected and class_name != expected:
            continue
        try:
            confidence = float(detection.get("confidence") or 0.0)
        except (TypeError, ValueError):
            continue
        measurement = enemy_box_measurement(detection)
        if confidence < minimum_confidence or measurement is None:
            continue
        score = confidence
        if previous_center is not None:
            displacement = math.hypot(
                measurement["center_x"] - previous_center[0],
                measurement["center_y"] - previous_center[1],
            )
            if displacement > 0.40:
                continue
            score -= displacement * 0.70
        candidates.append((score, detection, measurement))
    if not candidates:
        return None
    _score, detection, measurement = max(candidates, key=lambda item: item[0])
    return detection, measurement


def execute_guarded_enemy_pursuit(
    drone: Any,
    mission: dict[str, Any],
    *,
    pose_stream_path: Path | None,
    enemy_detection_path: Path | None,
    stop_flag: StopFlag | None = None,
    mission_stop_event: threading.Event | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    started = time.time()
    executed_pulses = 0
    hover_count = 0
    pulse_counts = {"yaw": 0, "forward": 0, "vertical": 0, "lateral": 0}
    abort_reason: str | None = None
    reached = False
    control_enabled = False
    closest_clearance: float | None = None
    final_clearance: float | None = None
    initial_clearance: float | None = None
    target_history: list[dict[str, float]] = []
    range_history: list[dict[str, float]] = []
    observed_frames: list[str] = []
    confirmation_events: list[bool] = []
    last_target_center: tuple[float, float] | None = None
    last_seen_at: float | None = None
    last_motion_frame = ""
    last_trusted_pose: dict[str, Any] | None = None

    if not bool(mission.get("guided_enabled")) or not bool(mission.get("operator_confirmed")):
        raise RuntimeError("enemy pursuit is not armed by the operator")
    if int(mission.get("client_safety_version") or 0) < 3:
        raise RuntimeError("enemy pursuit blocked: browser safety code is stale; reload ATLAS")
    if enemy_detection_path is None:
        raise RuntimeError("enemy pursuit has no live detection stream")
    range_model = mission.get("range_model") if isinstance(mission.get("range_model"), dict) else None
    if range_model is None:
        raise RuntimeError("enemy pursuit has no validated range calibration")

    safety_barriers = [item for item in mission.get("safety_barriers") or [] if isinstance(item, dict)]
    safety_obstacles = [item for item in mission.get("safety_obstacles") or [] if isinstance(item, dict)]
    if closed_wall_ring(safety_barriers) is None:
        raise RuntimeError("enemy pursuit requires a closed saved-wall geofence on the selected map")
    initial_pose_offset = vector3(mission.get("initial_pose_offset_room")) or [0.0, 0.0, 0.0]
    initial_pose_offset[1] = 0.0
    if horizontal_xz_distance(initial_pose_offset, [0.0, 0.0, 0.0]) > 1.0:
        raise RuntimeError("enemy pursuit initial pose correction exceeds the 1.0 m safety limit")
    safety_motion_buffer = clamp_float(mission.get("safety_motion_buffer_m"), 0.30, 0.30, 1.0)

    target_class_name = str(mission.get("target_class_name") or "").strip()
    stop_clearance = clamp_float(mission.get("stop_clearance_m"), 0.50, 0.50, 2.0)
    pose_max_age = clamp_float(mission.get("pose_max_age_seconds"), 1.2, 0.5, 1.8)
    pose_recovery_seconds = clamp_float(mission.get("pose_recovery_seconds"), 4.0, 1.0, 8.0)
    pulse_seconds = clamp_float(mission.get("pulse_seconds"), 0.14, 0.10, 0.20)
    max_forward_rc = clamp_float(mission.get("max_forward_rc"), 0.025, 0.01, 0.03)
    max_yaw_rc = clamp_float(mission.get("max_yaw_rc"), 0.028, 0.01, 0.035)
    max_vertical_rc = clamp_float(mission.get("max_vertical_rc"), 0.010, 0.005, 0.015)
    vertical_tracking = bool(mission.get("vertical_tracking_enabled"))
    detection_max_age = clamp_float(mission.get("detection_max_age_seconds"), 1.0, 0.60, 1.50)
    lost_target_abort = clamp_float(mission.get("lost_target_abort_seconds"), 4.0, 2.0, 8.0)
    max_pursuit_seconds = clamp_float(mission.get("max_pursuit_seconds"), 45.0, 5.0, 60.0)
    minimum_confidence = clamp_float(mission.get("minimum_confidence"), 0.40, 0.35, 0.80)
    yaw_sign = -1.0 if float(mission.get("pursuit_yaw_sign") or 1.0) < 0 else 1.0
    trained_min_clearance = clamp_float(range_model.get("trained_min_clearance_m"), 0.0, 0.0, 10.0)
    trained_max_clearance = clamp_float(range_model.get("trained_max_clearance_m"), 0.0, 0.0, 10.0)
    yaw_deadband = 0.065
    vertical_deadband = 0.10
    confirmation_hits = 3
    confirmation_window = 5

    def cancelled() -> str | None:
        if stop_flag is not None and stop_flag.stop:
            return "live localization stop requested"
        if mission_stop_event is not None and mission_stop_event.is_set():
            return "emergency hover requested"
        return None

    def publish(phase: str, message: str, **extra: Any) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(
                {
                    "phase": phase,
                    "message": message,
                    "enemy_pursuit": True,
                    "stop_clearance_m": stop_clearance,
                    "initial_clearance_m": initial_clearance,
                    "estimated_clearance_m": final_clearance,
                    "closest_clearance_m": closest_clearance,
                    "executed_pulses": executed_pulses,
                    "pulse_counts": dict(pulse_counts),
                    "elapsed_seconds": time.time() - started,
                    "updated_at": time.time(),
                    **extra,
                }
            )
        except Exception:
            pass

    def fresh_pose_gate() -> dict[str, Any] | None:
        nonlocal last_trusted_pose, abort_reason
        deadline = time.time() + pose_recovery_seconds
        last_reason = "waiting for a fresh TSolve pose"
        while time.time() < deadline:
            stop_reason = cancelled()
            if stop_reason:
                abort_reason = stop_reason
                return None
            gate = latest_tsolve_pose_gate(pose_stream_path, pose_max_age)
            if gate.get("ok") and pose_gate_position(gate) is not None:
                gate = {**gate, "pose_offset_room": initial_pose_offset}
                candidate = pose_gate_position(gate)
                trusted = pose_gate_position(last_trusted_pose)
                step = horizontal_xz_distance(candidate, trusted)
                if step is None or step <= 0.55:
                    last_trusted_pose = gate
                    return gate
                last_reason = f"rejected pursuit pose jump {step:.3f}m > 0.550m"
            else:
                last_reason = str(gate.get("reason") or last_reason)
            publish("pose_recovery", f"Hovering; {last_reason}.")
            neutral_hover(drone, 0.12)
        abort_reason = f"TSolve localization did not recover during pursuit ({last_reason})"
        return None

    try:
        enable_result = drone.enableControl(True)
        if not dji_control_response_ok(enable_result):
            abort_reason = dji_control_response_reason(enable_result)
        else:
            control_enabled = True
            publish("acquiring", "Pursuit armed; waiting for three fresh detections before movement.")
            while abort_reason is None and not reached and time.time() - started < max_pursuit_seconds:
                stop_reason = cancelled()
                if stop_reason:
                    abort_reason = stop_reason
                    break
                try:
                    payload = json.loads(enemy_detection_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    payload = {}
                    detection_error = str(exc)
                else:
                    detection_error = ""
                try:
                    detection_age = time.time() - float(payload.get("updated_at") or 0.0)
                except (TypeError, ValueError):
                    detection_age = float("inf")
                selected = None
                if 0.0 <= detection_age <= detection_max_age and str(payload.get("status") or "") == "detected":
                    selected = select_tracked_enemy_detection(
                        payload,
                        target_class_name,
                        last_target_center,
                        minimum_confidence,
                    )
                    if selected is None and last_seen_at is not None and time.time() - last_seen_at > 0.8:
                        # The target may move quickly between 2-FPS detector
                        # updates. Reacquire the same trained class after a
                        # short hover-only gap; forward motion remains locked
                        # until the 3/5 confirmation window is rebuilt.
                        selected = select_tracked_enemy_detection(
                            payload,
                            target_class_name,
                            None,
                            minimum_confidence,
                        )
                if selected is None:
                    clear_frame_key = str(payload.get("frame") or "")
                    if clear_frame_key and clear_frame_key not in observed_frames:
                        observed_frames.append(clear_frame_key)
                        observed_frames = observed_frames[-12:]
                        confirmation_events.append(False)
                        confirmation_events = confirmation_events[-confirmation_window:]
                    neutral_hover(drone, 0.10)
                    hover_count += 1
                    lost_for = float("inf") if last_seen_at is None else time.time() - last_seen_at
                    publish(
                        "target_recovery",
                        "Target is not fresh; hovering while searching the last-seen sector.",
                        detection_age_seconds=detection_age,
                        lost_seconds=lost_for,
                        detection_error=detection_error,
                    )
                    if last_seen_at is not None and lost_for > lost_target_abort:
                        abort_reason = f"enemy target was lost for {lost_for:.1f}s; pursuit stopped in hover"
                        break
                    if last_seen_at is None and time.time() - started > lost_target_abort:
                        abort_reason = "no confirmed enemy target appeared after pursuit was armed"
                        break
                    time.sleep(0.05)
                    continue

                detection, measurement = selected
                frame_key = str(payload.get("frame") or payload.get("updated_at") or "")
                is_new_frame = bool(frame_key and frame_key not in observed_frames)
                if is_new_frame:
                    observed_frames.append(frame_key)
                    observed_frames = observed_frames[-12:]
                    confirmation_events.append(True)
                    confirmation_events = confirmation_events[-confirmation_window:]
                    sample = {
                        **measurement,
                        "updated_at": float(payload.get("updated_at") or time.time()),
                    }
                    target_history.append(sample)
                    target_history = target_history[-8:]
                    last_seen_at = time.time()
                    last_target_center = (measurement["center_x"], measurement["center_y"])
                    estimate, uncertainty = estimate_enemy_clearance(measurement, range_model)
                    range_history.append(
                        {
                            "updated_at": float(payload.get("updated_at") or time.time()),
                            "clearance_m": estimate,
                        }
                    )
                    range_history = range_history[-5:]
                    smoothed_clearance, predicted_clearance = predict_enemy_clearance(range_history, 0.35)
                    final_clearance = smoothed_clearance
                    if initial_clearance is None:
                        initial_clearance = smoothed_clearance
                    closest_clearance = (
                        smoothed_clearance
                        if closest_clearance is None
                        else min(closest_clearance, smoothed_clearance)
                    )
                else:
                    estimate, uncertainty = estimate_enemy_clearance(measurement, range_model)
                    if range_history:
                        final_clearance, predicted_clearance = predict_enemy_clearance(range_history, 0.35)
                    else:
                        final_clearance = estimate
                        predicted_clearance = estimate

                confirmation_count = sum(1 for detected in confirmation_events if detected)
                confirmed = confirmation_count >= confirmation_hits
                conservative_clearance = max(0.0, min(float(final_clearance), predicted_clearance) - uncertainty)
                predicted_x, predicted_y = predict_enemy_image_center(target_history, 0.30)
                error_x = predicted_x - 0.5
                error_y = predicted_y - 0.5
                if conservative_clearance <= stop_clearance:
                    reached = True
                    neutral_hover(drone, 0.45)
                    hover_count += 1
                    publish(
                        "arrived",
                        f"Reached conservative {stop_clearance:.2f} m clearance; holding position.",
                        conservative_clearance_m=conservative_clearance,
                        uncertainty_m=uncertainty,
                    )
                    break
                if trained_min_clearance > 0.0 and float(final_clearance) < max(0.0, trained_min_clearance - uncertainty):
                    abort_reason = (
                        f"estimated target clearance {float(final_clearance):.2f} m is below the calibrated "
                        f"range envelope; pursuit stopped in hover"
                    )
                    publish("range_gate", abort_reason, conservative_clearance_m=conservative_clearance)
                    break
                if trained_max_clearance > 0.0 and float(final_clearance) > trained_max_clearance + uncertainty:
                    neutral_hover(drone, 0.10)
                    hover_count += 1
                    publish(
                        "range_gate",
                        (
                            f"Target is {float(final_clearance):.2f} m away, beyond the validated "
                            f"{trained_max_clearance:.2f} m range; hovering without forward motion."
                        ),
                        conservative_clearance_m=conservative_clearance,
                    )
                    time.sleep(0.05)
                    continue
                if not confirmed:
                    neutral_hover(drone, 0.10)
                    hover_count += 1
                    publish(
                        "confirming",
                        f"Confirming moving target {confirmation_count}/{confirmation_hits} within the last {confirmation_window} detector frames; translation remains locked.",
                        conservative_clearance_m=conservative_clearance,
                    )
                    time.sleep(0.05)
                    continue
                if not is_new_frame or frame_key == last_motion_frame:
                    time.sleep(0.04)
                    continue
                pose_gate = fresh_pose_gate()
                if pose_gate is None:
                    break
                geofence_issue = pursuit_geofence_issue(
                    pose_gate_position(pose_gate),
                    safety_barriers,
                    safety_obstacles,
                    motion_buffer_m=safety_motion_buffer,
                )
                if geofence_issue:
                    abort_reason = f"pursuit geofence blocked motion: {geofence_issue}"
                    publish("geofence", abort_reason)
                    break

                if abs(error_x) > yaw_deadband:
                    yaw_rc = yaw_sign * max(-max_yaw_rc, min(max_yaw_rc, error_x * 0.10))
                    if abs(yaw_rc) < 0.012:
                        yaw_rc = 0.012 if yaw_rc >= 0 else -0.012
                    sent = execute_rc_pulse(drone, yaw=yaw_rc, seconds=pulse_seconds)
                    pulse_counts["yaw"] += 1
                    executed_pulses += 1
                    motion = "yaw"
                elif vertical_tracking and abs(error_y) > vertical_deadband:
                    vertical_rc = max(-max_vertical_rc, min(max_vertical_rc, -error_y * 0.04))
                    sent = execute_rc_pulse(drone, du=vertical_rc, seconds=pulse_seconds)
                    pulse_counts["vertical"] += 1
                    executed_pulses += 1
                    motion = "vertical"
                else:
                    gap = max(0.0, conservative_clearance - stop_clearance)
                    speed_scale = max(0.35, min(1.0, gap / 1.50))
                    sent = execute_rc_pulse(
                        drone,
                        bf=max_forward_rc * speed_scale,
                        seconds=pulse_seconds,
                    )
                    pulse_counts["forward"] += 1
                    executed_pulses += 1
                    motion = "forward"
                last_motion_frame = frame_key
                publish(
                    "tracking",
                    f"Moving-target pursuit: {motion} pulse, estimated clearance {final_clearance:.2f} m.",
                    predicted_center={"x": predicted_x, "y": predicted_y},
                    image_velocity_prediction=True,
                    conservative_clearance_m=conservative_clearance,
                    predicted_clearance_m=predicted_clearance,
                    uncertainty_m=uncertainty,
                    last_command=sent,
                    confidence=float(detection.get("confidence") or 0.0),
                )

            if abort_reason is None and not reached:
                abort_reason = f"enemy pursuit timed out after {max_pursuit_seconds:.1f}s; holding position"
    finally:
        try:
            neutral_hover(drone, 0.20)
        except Exception:
            pass
        if control_enabled:
            try:
                drone.disableControl(True)
            except Exception:
                pass

    return {
        "ok": reached and abort_reason is None,
        "armed": True,
        "enemy_pursuit": True,
        "physical_motion_locked": False,
        "reached": reached,
        "aborted": abort_reason is not None,
        "abort_reason": abort_reason,
        "executed_hover_steps": hover_count,
        "executed_pulses": executed_pulses,
        "initial_clearance_m": initial_clearance,
        "final_clearance_m": final_clearance,
        "closest_clearance_m": closest_clearance,
        "stop_clearance_m": stop_clearance,
        "rc_summary": {
            "pulse_counts": pulse_counts,
            "mode": "moving_target_visual_servo_with_calibrated_range",
            "prediction_horizon_seconds": 0.30,
            "range_prediction_horizon_seconds": 0.35,
            "pursuit_yaw_sign": yaw_sign,
            "detection_confirmation": f"{confirmation_hits}/{confirmation_window}",
        },
        "guided_settings": {
            "pose_max_age_seconds": pose_max_age,
            "pose_recovery_seconds": pose_recovery_seconds,
            "pulse_seconds": pulse_seconds,
            "max_forward_rc": max_forward_rc,
            "max_yaw_rc": max_yaw_rc,
            "max_vertical_rc": max_vertical_rc,
            "vertical_tracking_enabled": vertical_tracking,
            "safety_motion_buffer_m": safety_motion_buffer,
            "detection_max_age_seconds": detection_max_age,
            "lost_target_abort_seconds": lost_target_abort,
            "max_pursuit_seconds": max_pursuit_seconds,
        },
        "executed": [],
        "skipped": [],
        "command_count": 1,
        "message": (
            f"Enemy pursuit reached the conservative {stop_clearance:.2f} m clearance and is hovering."
            if reached
            else f"Enemy pursuit stopped: {abort_reason}"
        ),
        "elapsed_seconds": time.time() - started,
    }


def execute_control_command(
    drone: Any,
    OpenDJI: Any,
    command: dict[str, Any],
    *,
    pose_stream_path: Path | None = None,
    enemy_detection_path: Path | None = None,
    stop_flag: StopFlag | None = None,
    mission_stop_event: threading.Event | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    name = str(command.get("command", "")).strip().lower()
    height_m = command.get("height_m")
    started = time.time()
    altitude_before = read_altitude(drone, OpenDJI)
    if name == "takeoff":
        result = drone.takeoff(True)
        height_guard = climb_to_requested_height(drone, OpenDJI, altitude_before, height_m)
    elif name == "land":
        time.sleep(PRE_LAND_STABILIZE_SECONDS)
        result = drone.land(True)
        height_guard = {
            "enabled": False,
            "reason": "native DJI landing command; OpenDJI does not expose landing-speed control",
            "pre_land_stabilize_seconds": PRE_LAND_STABILIZE_SECONDS,
        }
    elif name == "enable":
        result = drone.enableControl(True)
        height_guard = {"enabled": False, "reason": "not a takeoff command"}
    elif name == "disable":
        result = drone.disableControl(True)
        height_guard = {"enabled": False, "reason": "not a takeoff command"}
    elif name == "hover":
        result = neutral_hover(drone, 0.0)
        height_guard = {"enabled": False, "reason": "not a takeoff command"}
    elif name == "mission":
        result = execute_guarded_mission_packet(
            drone,
            command.get("mission") if isinstance(command.get("mission"), dict) else {},
            pose_stream_path=pose_stream_path,
            enemy_detection_path=enemy_detection_path,
            stop_flag=stop_flag,
            mission_stop_event=mission_stop_event,
            progress_callback=progress_callback,
        )
        height_guard = {
            "enabled": False,
            "reason": "mission execution uses guarded per-step policy, not takeoff height guard",
        }
    else:
        raise ValueError(f"Unsupported live DJI command: {name}")
    altitude_after = read_altitude(drone, OpenDJI)
    note = ""
    if name == "takeoff" and height_m is not None:
        note = (
            "Takeoff command sent. The requested height is enforced with a "
            "conservative telemetry-based upward-only guard when altitude "
            "telemetry is available."
        )
    ok = True
    error = None
    if name == "mission" and isinstance(result, dict):
        ok = bool(result.get("ok"))
        if not ok:
            error = result.get("abort_reason") or result.get("message") or "mission did not execute"

    return {
        "ok": ok,
        "id": command.get("id"),
        "command": name,
        "height_m": height_m,
        "result": result,
        "error": error,
        "altitude_before": altitude_before,
        "altitude_after": altitude_after,
        "height_guard": height_guard,
        "note": note,
        "elapsed_seconds": time.time() - started,
        "updated_at": time.time(),
    }


def execute_guarded_mission_packet(
    drone: Any,
    mission: dict[str, Any],
    *,
    pose_stream_path: Path | None = None,
    enemy_detection_path: Path | None = None,
    stop_flag: StopFlag | None = None,
    mission_stop_event: threading.Event | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if bool(mission.get("enemy_pursuit")):
        return execute_guarded_enemy_pursuit(
            drone,
            mission,
            pose_stream_path=pose_stream_path,
            enemy_detection_path=enemy_detection_path,
            stop_flag=stop_flag,
            mission_stop_event=mission_stop_event,
            progress_callback=progress_callback,
        )
    commands = mission.get("commands") if isinstance(mission.get("commands"), list) else []
    executed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    hover_count = 0
    started = time.time()
    guided_enabled = bool(mission.get("guided_enabled"))
    mission_has_cruise = any(
        isinstance(step, dict) and str(step.get("type", "")).strip().lower() == "cruise"
        for step in commands
    )
    safety_overrides: list[str] = []
    if guided_enabled and mission_has_cruise:
        try:
            client_safety_version = int(mission.get("client_safety_version") or 0)
        except (TypeError, ValueError):
            client_safety_version = 0
        if client_safety_version < 3 or "initial_pose_offset_room" not in mission:
            raise RuntimeError("patrol blocked: browser safety code is stale; reload ATLAS before flight")
        # Patrol travel is deliberately yaw-then-forward. Lateral/body-axis
        # mixing caused the aircraft to strafe because the two measured axes
        # could become nearly parallel. Force this safe behavior even for a
        # mission packet sent by an older cached browser page.
        if bool(mission.get("allow_axis_auto_calibration")):
            safety_overrides.append("forced allow_axis_auto_calibration=false for yaw-then-forward patrol")
        if bool(mission.get("allow_lateral_rc")):
            safety_overrides.append("forced allow_lateral_rc=false for yaw-then-forward patrol")
        try:
            old_pulse = float(mission.get("pulse_seconds") or GUIDED_DEFAULT_PULSE_SECONDS)
        except (TypeError, ValueError):
            old_pulse = GUIDED_DEFAULT_PULSE_SECONDS
        if old_pulse < 0.28:
            safety_overrides.append("raised pulse_seconds for visible slow cruise")
        mission = dict(mission)
        mission["allow_axis_auto_calibration"] = False
        mission["allow_lateral_rc"] = False
        # Preserve the server-owned heading verification policy.  The strong
        # 13:17 and 14:17 runs measured body-forward with a tiny physical probe;
        # overriding that request here caused later missions to trust an
        # unverified 0-degree seed and react to one bad yaw observation.
        mission["pulse_seconds"] = max(old_pulse, 0.30)
        mission["axis_probe_seconds"] = max(float(mission.get("axis_probe_seconds") or 0.0), 0.55)
        mission["max_cruise_seconds"] = max(float(mission.get("max_cruise_seconds") or 0.0), 120.0)
        if bool(mission.get("patrol")):
            # Server and command bridge both own this safety policy.  Do not
            # permit a cached browser mission to restore the former three-pulse
            # blind window or stationary metric test pulses.
            mission["one_pulse_pose_confirmation"] = True
            mission["smooth_continuous_cruise"] = True
            try:
                requested_cruise_window = float(
                    mission.get("cruise_window_seconds") or 0.55
                )
            except (TypeError, ValueError):
                requested_cruise_window = 0.55
            bounded_cruise_window = min(
                max(requested_cruise_window, 0.45),
                0.55,
            )
            if abs(bounded_cruise_window - requested_cruise_window) > 1e-6:
                safety_overrides.append(
                    "bounded cruise_window_seconds to a TSolve-verifiable forward increment"
                )
            mission["cruise_window_seconds"] = bounded_cruise_window
            mission["cruise_pose_watchdog_seconds"] = min(
                max(float(mission.get("cruise_pose_watchdog_seconds") or 0.0), 0.45),
                0.75,
            )
            # A smaller persistent stick produces visibly smooth indoor
            # travel and preserves optical overlap better than 0.035 bursts.
            mission["max_forward_rc"] = min(
                float(mission.get("max_forward_rc") or GUIDED_DEFAULT_MAX_FORWARD_RC),
                0.024,
            )
    pose_max_age = clamp_float(
        mission.get("pose_max_age_seconds"),
        GUIDED_DEFAULT_POSE_MAX_AGE_SECONDS,
        0.4,
        10.0,
    )
    pose_recovery_seconds = clamp_float(
        mission.get("pose_recovery_seconds"),
        8.0 if bool(mission.get("patrol")) else GUIDED_DEFAULT_POSE_RECOVERY_SECONDS,
        0.5,
        8.0 if bool(mission.get("patrol")) else 90.0,
    )
    pulse_seconds = clamp_float(
        mission.get("pulse_seconds"),
        GUIDED_DEFAULT_PULSE_SECONDS,
        0.08,
        0.75,
    )
    max_forward_rc = clamp_float(
        mission.get("max_forward_rc"),
        GUIDED_DEFAULT_MAX_FORWARD_RC,
        0.01,
        0.12,
    )
    max_lateral_rc = clamp_float(
        mission.get("max_lateral_rc"),
        max_forward_rc,
        0.01,
        0.12,
    )
    allow_lateral_rc = bool(mission.get("allow_lateral_rc"))
    allow_axis_auto_calibration = bool(mission.get("allow_axis_auto_calibration"))
    if not allow_lateral_rc:
        max_lateral_rc = 0.0
    max_yaw_rc = clamp_float(
        mission.get("max_yaw_rc"),
        GUIDED_DEFAULT_MAX_YAW_RC,
        0.01,
        0.10,
    )
    max_scan_yaw_rc = clamp_float(
        mission.get("max_scan_yaw_rc"),
        min(max_yaw_rc, 0.025),
        0.01,
        max_yaw_rc,
    )
    allow_patrol_scan_yaw = bool(mission.get("allow_patrol_scan_yaw"))
    alignment_grace_seconds = clamp_float(
        mission.get("alignment_grace_seconds"),
        35.0,
        10.0,
        60.0,
    )
    max_vertical_rc = clamp_float(
        mission.get("max_vertical_rc"),
        GUIDED_DEFAULT_MAX_VERTICAL_RC,
        0.005,
        0.06,
    )
    max_step_seconds = clamp_float(
        mission.get("max_step_seconds"),
        GUIDED_DEFAULT_MAX_STEP_SECONDS,
        0.25,
        6.0,
    )
    configured_arrival_radius = clamp_float(
        mission.get("arrival_radius_map_units"),
        0.22,
        0.05,
        0.80,
    )
    arrival_deadband = clamp_float(
        mission.get("arrival_deadband_map_units"),
        0.14,
        0.0,
        0.35,
    )
    patrol_stage = str(mission.get("patrol_stage") or "").strip().lower()
    arrival_radius, soft_arrival_radius = mission_arrival_radii(
        configured_arrival_radius,
        arrival_deadband,
        patrol_stage=patrol_stage,
    )
    strict_entry_arrival = patrol_stage in {"entry", "combined"}
    max_cruise_seconds = clamp_float(
        mission.get("max_cruise_seconds"),
        max(8.0, max_step_seconds * 3.0),
        2.0,
        180.0,
    )
    axis_probe_rc = clamp_float(
        mission.get("axis_probe_rc"),
        min(max_forward_rc, max_lateral_rc, 0.035),
        0.012,
        0.055,
    )
    axis_probe_seconds = clamp_float(
        mission.get("axis_probe_seconds"),
        max(0.16, min(pulse_seconds, 0.24)),
        0.12,
        0.65,
    )
    heading_trim_deg = clamp_float(mission.get("heading_trim_deg"), 0.0, -180.0, 180.0)
    heading_trim_rad = math.radians(heading_trim_deg)
    operator_heading_calibrated = bool(mission.get("operator_heading_calibrated"))
    initial_body_heading_offset_deg = clamp_float(
        mission.get("initial_body_heading_offset_deg"),
        0.0,
        -180.0,
        180.0,
    )
    require_physical_forward_probe = bool(mission.get("require_physical_forward_probe")) or not (
        operator_heading_calibrated
    )
    max_heading_calibration_error_deg = clamp_float(
        mission.get("max_heading_calibration_error_deg"),
        35.0,
        10.0,
        60.0,
    )
    initial_pose_offset_room = vector3(mission.get("initial_pose_offset_room")) or [0.0, 0.0, 0.0]
    initial_pose_offset_room[1] = 0.0
    if horizontal_xz_distance(initial_pose_offset_room, [0.0, 0.0, 0.0]) > 1.0:
        raise RuntimeError("manual initial pose correction exceeds the 1.0 map-unit safety limit")
    max_pose_step = clamp_float(
        mission.get("max_pose_step_map_units"),
        0.30,
        0.08,
        1.0,
    )
    max_pose_step_hard = clamp_float(
        mission.get("max_pose_step_hard_map_units"),
        0.55,
        max_pose_step,
        0.80,
    )
    max_cross_track = clamp_float(
        mission.get("max_cross_track_map_units"),
        0.80,
        0.30,
        2.0,
    )
    cross_track_recovery_start = clamp_float(
        mission.get("cross_track_recovery_start_map_units"),
        0.30,
        0.10,
        max_cross_track - 0.05,
    )
    is_patrol = bool(mission.get("patrol"))
    patrol_loop = is_patrol and bool(mission.get("loop"))
    continuous_relocalization = bool(
        is_patrol and mission.get("continuous_relocalization") is not False
    )
    one_pulse_pose_confirmation = bool(
        is_patrol and mission.get("one_pulse_pose_confirmation") is True
    )
    smooth_continuous_cruise = bool(
        is_patrol and mission.get("smooth_continuous_cruise") is True
    )
    cruise_window_seconds = clamp_float(
        mission.get("cruise_window_seconds"),
        0.55,
        0.45,
        0.55,
    )
    cruise_pose_watchdog_seconds = clamp_float(
        mission.get("cruise_pose_watchdog_seconds"),
        0.65,
        0.30,
        1.00,
    )
    max_unverified_translation_m = clamp_float(
        mission.get("max_unverified_translation_m"),
        0.18,
        0.05,
        0.30,
    )
    try:
        requested_patrol_laps = int(mission.get("patrol_laps") or 0)
    except (TypeError, ValueError):
        requested_patrol_laps = 0
    patrol_laps = max(0, min(20, requested_patrol_laps)) if patrol_loop else 1
    route_monotonic_gate = bool(is_patrol and mission.get("route_monotonic_gate"))
    route_gate_max_cross_track = min(max_cross_track, 0.55)
    route_gate_backward_tolerance = 0.08
    route_gate_turn_max_drift = 0.16
    patrol_safety_barriers = [item for item in mission.get("safety_barriers") or [] if isinstance(item, dict)]
    patrol_safety_obstacles = [item for item in mission.get("safety_obstacles") or [] if isinstance(item, dict)]
    safety_motion_buffer = clamp_float(mission.get("safety_motion_buffer_m"), 0.30, 0.30, 1.0)
    if is_patrol and closed_wall_ring(patrol_safety_barriers) is None:
        raise RuntimeError("patrol requires a closed saved-wall geofence on the selected map")
    taught_reference = load_taught_patrol_reference(mission) if patrol_loop else None
    verified_route_lock = (
        load_verified_route_follow_lock(mission) if patrol_loop else None
    )
    patrol_loop_start = (
        patrol_loop_start_command_index(commands, taught_reference)
        if patrol_loop
        else 0
    )

    if not guided_enabled:
        for idx, step in enumerate(commands):
            if not isinstance(step, dict):
                skipped.append({"index": idx, "type": "unknown", "reason": "invalid command record"})
                continue
            kind = str(step.get("type", "")).strip().lower()
            title = str(step.get("title", kind or "step"))
            if kind == "hover":
                duration = max(0.1, min(2.0, float(step.get("duration_s") or 0.5)))
                neutral_hover(drone, duration)
                hover_count += 1
                executed.append({"index": idx, "type": kind, "title": title, "duration_s": duration})
            elif kind == "gate":
                skipped.append({"index": idx, "type": kind, "title": title, "reason": "UI confirmation gate"})
            else:
                skipped.append(
                    {
                        "index": idx,
                        "type": kind or "unknown",
                        "title": title,
                        "reason": "guided_enabled=false; physical lateral/yaw/landing execution is locked",
                    }
                )
        return {
            "ok": True,
            "armed": False,
            "physical_motion_locked": True,
            "executed_hover_steps": hover_count,
            "executed": executed,
            "skipped": skipped,
            "command_count": len(commands),
            "message": "Mission packet received by the DJI bridge, but guided movement was not armed.",
            "elapsed_seconds": time.time() - started,
        }

    executed_pulses = 0
    abort_reason: str | None = None
    control_enabled = False
    last_pose_gate: dict[str, Any] | None = None
    active_route_segment_start: list[float] | None = None
    active_route_segment_end: list[float] | None = None
    total_pose_recovery_pause_seconds = 0.0

    def motion_clock() -> float:
        """Mission time excluding safe hover spent waiting for localization."""
        return time.monotonic() - total_pose_recovery_pause_seconds
    active_route_progress: float | None = None
    # Route-image matching is scale tolerant, so a stationary camera can still
    # resemble several later views on a long indoor leg.  Keep an explicit
    # command-side ceiling: vision may publish only distance that a verified
    # horizontal RC command could physically have produced.  Yaw and neutral
    # hover never increase this budget.
    active_route_command_progress_ceiling: float | None = None
    active_route_command_sequence = 0
    active_route_translation_locked = False
    active_route_position_anchor: list[float] | None = None
    # The leg toward Point 1 begins before Point 4 is fully verified. Without
    # a second ownership boundary, an older leg-4 pose/bias can survive after
    # the exact endpoint is accepted and compete with it during the turn.
    active_route_pose_epoch = 0
    active_route_pose_epoch_unix: float | None = None
    active_route_pose_epoch_reason: str | None = None
    active_endpoint_overshoot_correction = False
    active_endpoint_undershoot_correction = False
    pose_recovery_active = False
    route_visual_reconciliation_state: dict[str, Any] = {}
    route_metric_reconciliation_state: dict[str, Any] = {}
    current_lap_number = 0
    lap_metric_checkpoint_pending = False
    lap_reentry_metric_ready = False
    rc_summary: dict[str, Any] = {
        "open_dji_move_order": "rcw, du, lr, bf",
        "open_dji_rounding": "yaw to 4 decimals; vertical/lateral/forward to 2 decimals",
        "enable_control_result": None,
        "disable_control_result": None,
        "pulse_counts": {"yaw": 0, "forward": 0, "vertical": 0, "lateral": 0},
        "lap_metric_checkpoints": [],
        "pulse_samples": [],
        # Unlike the compact UI samples, this complete trace is retained in
        # the finished per-session command record so saved camera/pose frames
        # can be replayed against the exact physical command timeline.
        "pulse_trace": [],
            "adaptive_axis": {
            "yaw_sign": 1.0,
            "yaw_sign_verified": False,
            "forward_sign": 1.0,
            "yaw_flips": 0,
            "forward_flips": 0,
            "mode": "heading_yaw_forward_only",
            "forward_axis_xz": None,
            "lateral_axis_xz": None,
            "axis_probe_rc": axis_probe_rc,
            "axis_probe_seconds": axis_probe_seconds,
            "axis_recalibrations": 0,
            "allow_lateral_rc": allow_lateral_rc,
                "allow_axis_auto_calibration": allow_axis_auto_calibration,
            },
            "heading_trim_deg": heading_trim_deg,
            "operator_heading_calibrated": operator_heading_calibrated,
            "initial_body_heading_offset_deg": initial_body_heading_offset_deg,
            "require_physical_forward_probe": require_physical_forward_probe,
            "max_heading_calibration_error_deg": max_heading_calibration_error_deg,
            "initial_pose_offset_room": initial_pose_offset_room,
            "safety_overrides": safety_overrides,
            "taught_turn_reference": {
                "enabled": taught_reference is not None,
                "path": (
                    (
                        f"maps/{mission.get('map_id')}/replays/{mission.get('baseline_replay_id')}/reference_candidate.json"
                        if mission.get("baseline_replay_id")
                        else f"maps/{mission.get('map_id')}/taught_patrols/{mission.get('patrol_id')}/reference.json"
                    )
                    if taught_reference is not None
                    else None
                ),
            },
            "verified_route_follow": {
                "enabled": verified_route_lock is not None,
                "source_replay_id": (
                    verified_route_lock.get("source_replay_id")
                    if isinstance(verified_route_lock, dict)
                    else None
                ),
                "verified_through_point": (
                    verified_route_lock.get("verified_through_point")
                    if isinstance(verified_route_lock, dict)
                    else None
                ),
                "control_mode": (
                    verified_route_lock.get("control_mode")
                    if isinstance(verified_route_lock, dict)
                    else None
                ),
            },
        }
    body_axes: dict[str, list[float] | None] = {"forward": None, "lateral": None}
    # The camera-to-body mounting is fixed.  When the operator has aligned the
    # live model, use that COLMAP-derived heading without moving the aircraft
    # before the route starts.  A physical forward probe remains available as
    # a fallback only when no operator heading calibration exists.
    calibrated_heading_offset_rad: float | None = None
    if operator_heading_calibrated:
        rc_summary["adaptive_axis"]["mode"] = "operator_heading_seed_pending_yaw_verification"
        rc_summary["adaptive_axis"]["operator_heading_seed_deg"] = initial_body_heading_offset_deg
    yaw_sign = 1.0
    yaw_sign_verified = False
    forward_sign = 1.0
    yaw_flip_count = 0
    forward_flip_count = 0
    axis_recalibrations = 0

    def record_pulse(kind: str, sent: dict[str, Any]) -> None:
        counts = rc_summary["pulse_counts"]
        counts[kind] = int(counts.get(kind, 0)) + 1
        samples = rc_summary["pulse_samples"]
        if len(samples) < 16:
            samples.append({"kind": kind, **sent})
        rc_summary["pulse_trace"].append(
            {
                "sequence": len(rc_summary["pulse_trace"]) + 1,
                "recorded_unix": time.time(),
                "lap": current_lap_number or None,
                "kind": kind,
                **sent,
            }
        )

    def publish_progress(payload: dict[str, Any]) -> None:
        if progress_callback is None:
            return
        try:
            route_context = {
                "map_id": mission.get("map_id"),
                "patrol_id": mission.get("patrol_id"),
                "baseline_replay_id": mission.get("baseline_replay_id"),
                "lap": current_lap_number or None,
                "patrol_laps": patrol_laps or None,
            }
            if active_route_segment_start is not None and active_route_segment_end is not None:
                route_context.update(
                    {
                        "segment_start": active_route_segment_start,
                        "target": active_route_segment_end,
                        "translation_locked": active_route_translation_locked,
                        "position_anchor": active_route_position_anchor,
                        "route_progress_command_ceiling": (
                            active_route_command_progress_ceiling
                        ),
                        "route_progress_command_sequence": (
                            active_route_command_sequence
                        ),
                        "route_progress_command_budget_m": (
                            max_unverified_translation_m
                        ),
                        "route_pose_epoch": active_route_pose_epoch,
                        "route_pose_epoch_unix": active_route_pose_epoch_unix,
                        "route_pose_epoch_reason": active_route_pose_epoch_reason,
                    }
                )
            progress_callback(
                {
                    **route_context,
                    **payload,
                    "executed_pulses": executed_pulses,
                    "executed_steps": len(executed),
                    "skipped_steps": len(skipped),
                    "elapsed_seconds": time.time() - started,
                    "updated_at": time.time(),
                }
            )
        except Exception:
            pass

    def commit_verified_point_one_handoff(
        source_gate: dict[str, Any] | None,
        endpoint: list[float],
    ) -> dict[str, Any] | None:
        """Atomically make a proved 4->1 arrival the next lap's pose truth."""
        nonlocal last_pose_gate, active_route_progress
        nonlocal active_route_command_progress_ceiling
        nonlocal active_route_translation_locked, active_route_position_anchor
        nonlocal active_route_pose_epoch, active_route_pose_epoch_unix
        nonlocal active_route_pose_epoch_reason, yaw_position_anchor

        active_route_pose_epoch += 1
        active_route_pose_epoch_unix = time.time()
        active_route_pose_epoch_reason = "verified_point1_handoff"
        endpoint_gate = verified_route_endpoint_pose_gate(
            source_gate,
            endpoint,
            epoch=active_route_pose_epoch,
            epoch_unix=active_route_pose_epoch_unix,
            reason=active_route_pose_epoch_reason,
        )
        if not isinstance(endpoint_gate, dict):
            return endpoint_gate
        last_pose_gate = endpoint_gate
        active_route_progress = 1.0
        active_route_command_progress_ceiling = 1.0
        active_route_translation_locked = True
        active_route_position_anchor = list(endpoint)
        yaw_position_anchor = list(endpoint)
        return endpoint_gate

    def enforce_patrol_geofence(gate: dict[str, Any] | None, phase: str) -> bool:
        nonlocal abort_reason
        if not is_patrol:
            return True
        issue = pursuit_geofence_issue(
            pose_gate_position(gate),
            patrol_safety_barriers,
            patrol_safety_obstacles,
            # Reserve the conservative distance of the next unverified pulse.
            # With one-pulse confirmation this is the entire unresolved physical
            # motion budget; it can never accumulate across several commands.
            motion_buffer_m=(
                safety_motion_buffer + max_unverified_translation_m
                if one_pulse_pose_confirmation
                else safety_motion_buffer
            ),
        )
        if issue is None:
            return True
        abort_reason = f"patrol geofence blocked {phase}: {issue}"
        publish_progress(
            {
                "phase": "patrol_geofence",
                "message": abort_reason,
                "safety_motion_buffer_m": safety_motion_buffer,
                "max_unverified_translation_m": max_unverified_translation_m,
            }
        )
        try:
            neutral_hover(drone, 0.20)
        except Exception:
            pass
        return False

    def wait_for_pose_recovery(
        phase: str,
        reason: str | None = None,
        *,
        timeout: float | None = None,
        require_translation_safe: bool = False,
        translation_target: list[float] | None = None,
        translation_reference_distance: float | None = None,
        translation_arrival_radius: float | None = None,
        require_endpoint_verified: bool = False,
        endpoint_leg_index: int | None = None,
        require_observed_translation_progress: bool = False,
        allow_visual_stationary_retry: bool = False,
        require_metric_pose: bool = False,
        lap_start_metric_rebootstrap: bool = False,
    ) -> dict[str, Any] | None:
        nonlocal abort_reason, last_pose_gate, active_route_progress
        nonlocal total_pose_recovery_pause_seconds
        nonlocal active_route_translation_locked, active_route_position_anchor
        nonlocal yaw_position_anchor
        nonlocal pose_recovery_active
        wait_seconds = pose_recovery_seconds if timeout is None else max(0.1, float(timeout))
        recovery_started = time.monotonic()
        deadline = recovery_started + wait_seconds
        last_reason = reason or "waiting for a fresh TSolve pose"
        last_recovery_gate: dict[str, Any] | None = None
        recovery_display_anchor = pose_gate_position(last_pose_gate)
        endpoint_metric_correction_hits = 0
        endpoint_metric_correction_instance: str | None = None
        endpoint_metric_correction_progress: float | None = None
        point_two_metric_endpoint_consensus: dict[str, Any] = {}

        # This function issues neutral hover only.  A yaw pulse may have left
        # both route-lock variables and the cruise loop's navigation anchor
        # populated, but neither remains valid once online recovery is allowed
        # to replace the held pose.  Keeping either stale anchor makes the
        # command-side route gate classify a legitimate recovered position as
        # impossible turn drift forever.
        #
        # Clearing these navigation variables does not authorize aircraft
        # translation.  Progress continues to advertise an explicit physical
        # movement lock, and the final RC command gate still requires verified
        # baseline supervision before any forward/lateral pulse.
        active_route_translation_locked = False
        active_route_position_anchor = None
        yaw_position_anchor = None
        pose_recovery_active = True
        # Continuous patrol recovery normally hovers until localization
        # returns or the operator stops it.  An explicit timeout is different:
        # callers use it for a bounded checkpoint/turn handoff and must regain
        # control of the state machine when that deadline expires.
        bounded_recovery = timeout is not None
        try:
            while (
                (continuous_relocalization and not bounded_recovery)
                or time.monotonic() < deadline
            ):
                if stop_flag is not None and stop_flag.stop:
                    abort_reason = "live localization stop requested"
                    return None
                if mission_stop_event is not None and mission_stop_event.is_set():
                    abort_reason = "emergency hover requested"
                    return None
                elapsed = time.monotonic() - recovery_started
                publish_progress(
                    {
                        "phase": "pose_recovery",
                        "recovery_phase": phase,
                        "pose_gate": last_recovery_gate,
                        "continuous_relocalization": continuous_relocalization,
                        "require_metric_pose": require_metric_pose,
                        "lap_start_metric_rebootstrap": bool(
                            lap_start_metric_rebootstrap
                        ),
                        # A metric checkpoint is a neutral hover, not a yaw
                        # command.  Permit the localizer to publish the newly
                        # measured raw room position instead of projecting it
                        # back onto the old route/yaw anchor. Physical RC
                        # translation remains locked at zero below.
                        "metric_position_recovery_allowed": bool(
                            require_metric_pose
                        ),
                        # Endpoint imagery can disprove a model pose that was
                        # clipped to progress 1.0. While the aircraft hovers,
                        # allow a newest-frame metric solve to expose up to the
                        # guarded 20% endpoint neighborhood so the outer loop
                        # can select a bounded forward/reverse correction.
                        "endpoint_position_recovery_allowed": bool(
                            require_endpoint_verified
                        ),
                        # A failed post-command observation is different from
                        # an ordinary yaw/hover recovery: the aircraft may
                        # already have translated inside the accumulated
                        # command budget.  Tell the localizer it may expose a
                        # fresh, route-constrained metric center instead of
                        # pinning every later solve to the pre-command anchor.
                        # The localizer still caps that recovery at the
                        # command-progress ceiling and the bridge still
                        # requires at least 15 mm of observed progress before
                        # another horizontal command can be issued.
                        "post_translation_progress_recovery": bool(
                            require_observed_translation_progress
                            and translation_target is not None
                            and translation_reference_distance is not None
                        ),
                        "recovery_elapsed_seconds": elapsed,
                        "translation_locked": True,
                        # The aircraft is receiving neutral hover commands,
                        # not yaw commands. Permit the leg-constrained visual
                        # baseline (and a fresh COLMAP solve) to replace the
                        # rejected pose while physical translation stays off.
                        "route_visual_recovery_allowed": True,
                        "position_anchor": recovery_display_anchor,
                        "rotation_release_requested": True,
                        "body_forward_gain": 0.0,
                        "body_lateral_gain": 0.0,
                        "message": (
                            "Hovering with physical movement locked while ATLAS releases "
                            f"the stale turn anchor and relocalizes online ({elapsed:.1f}s): "
                            f"{last_reason}"
                        ),
                    }
                )
                neutral_hover(drone, GUIDED_RECOVERY_HOVER_SECONDS)
                gate = continuity_guarded_pose_gate()
                last_recovery_gate = gate
                candidate_position = pose_gate_position(gate)
                candidate_distance = horizontal_xz_distance(
                    candidate_position,
                    translation_target,
                )
                target_arrived = bool(
                    candidate_distance is not None
                    and translation_arrival_radius is not None
                    and candidate_distance <= float(translation_arrival_radius)
                )
                endpoint_ready = bool(
                    not require_endpoint_verified
                    or taught_endpoint_arrival_verified(
                        gate.get("pose") if isinstance(gate, dict) else None,
                        expected_leg_index=endpoint_leg_index,
                    )
                )
                visual_checkpoint_arrived = bool(
                    endpoint_leg_index is not None
                    and taught_endpoint_stale_translation_arrival_verified(
                        gate.get("pose") if isinstance(gate, dict) else None,
                        expected_leg_index=endpoint_leg_index,
                    )
                )
                metric_pose_ready = pose_gate_has_fresh_metric_position(gate)
                endpoint_metric_progress = route_segment_progress_xz(
                    candidate_position,
                    active_route_segment_start,
                    translation_target,
                )
                point_two_metric_endpoint_candidate = bool(
                    require_endpoint_verified
                    and tight_metric_point_two_endpoint_arrival_candidate(
                        gate,
                        target=translation_target,
                        segment_start=active_route_segment_start,
                        expected_leg_index=endpoint_leg_index,
                    )
                )
                point_two_metric_endpoint_ready = (
                    update_stable_metric_endpoint_consensus(
                        point_two_metric_endpoint_consensus,
                        gate,
                        candidate=point_two_metric_endpoint_candidate,
                    )
                )
                endpoint_ready = bool(
                    endpoint_ready or point_two_metric_endpoint_ready
                )
                if point_two_metric_endpoint_ready:
                    gate = {
                        **gate,
                        "metric_point_two_arrival_verified": True,
                        "metric_point_two_arrival_hits": int(
                            point_two_metric_endpoint_consensus.get("hits") or 0
                        ),
                    }
                endpoint_metric_offset_m = None
                if (
                    endpoint_metric_progress is not None
                    and active_route_segment_start is not None
                    and translation_target is not None
                ):
                    endpoint_leg_length = horizontal_xz_distance(
                        active_route_segment_start,
                        translation_target,
                    )
                    if endpoint_leg_length is not None:
                        endpoint_metric_offset_m = (
                            float(endpoint_metric_progress) - 1.0
                        ) * float(endpoint_leg_length)
                endpoint_metric_correction_candidate = bool(
                    require_endpoint_verified
                    and gate.get("ok")
                    and metric_pose_ready
                    and endpoint_metric_progress is not None
                    and endpoint_metric_offset_m is not None
                    and abs(float(endpoint_metric_offset_m)) > 0.03
                    and -0.08 <= float(endpoint_metric_progress) <= 1.20
                )
                candidate_pose = gate.get("pose") if isinstance(gate, dict) else None
                candidate_instance = str(
                    candidate_pose.get("instance_id")
                    if isinstance(candidate_pose, dict)
                    else ""
                )
                if endpoint_metric_correction_candidate:
                    if (
                        candidate_instance
                        and candidate_instance != endpoint_metric_correction_instance
                    ):
                        if (
                            endpoint_metric_correction_progress is not None
                            and abs(
                                float(endpoint_metric_progress)
                                - endpoint_metric_correction_progress
                            )
                            <= 0.03
                        ):
                            endpoint_metric_correction_hits += 1
                        else:
                            endpoint_metric_correction_hits = 1
                        endpoint_metric_correction_instance = candidate_instance
                        endpoint_metric_correction_progress = float(
                            endpoint_metric_progress
                        )
                else:
                    endpoint_metric_correction_hits = 0
                    endpoint_metric_correction_instance = None
                    endpoint_metric_correction_progress = None
                translation_progress_ready = True
                visual_stationary_retry = False
                observed_translation_delta: float | None = None
                if (
                    translation_target is not None
                    and translation_reference_distance is not None
                ):
                    candidate_pose = gate.get("pose") if isinstance(gate, dict) else None
                    metric_pose = bool(
                        isinstance(candidate_pose, dict)
                        and candidate_pose.get("pose_source")
                        != "patrol_visual_route_recovery"
                    )
                    visual_bounded_resume = patrol_visual_translation_resume_ready(
                        candidate_pose
                    )
                    observed_translation_progress = bool(
                        candidate_distance is not None
                        and candidate_distance
                        <= float(translation_reference_distance) - 0.015
                    )
                    if candidate_distance is not None:
                        observed_translation_delta = (
                            float(translation_reference_distance)
                            - float(candidate_distance)
                        )
                    visual_stationary_retry = (
                        require_observed_translation_progress
                        and patrol_visual_stationary_retry_ready(
                            candidate_pose,
                            retry_available=allow_visual_stationary_retry,
                            observed_translation_progress=observed_translation_progress,
                            expected_leg_index=endpoint_leg_index,
                        )
                    )
                    translation_progress_ready = bool(
                        target_arrived
                        or observed_translation_progress
                        or visual_checkpoint_arrived
                        or visual_stationary_retry
                        or (
                            visual_bounded_resume
                            and not require_observed_translation_progress
                        )
                    )
                    if visual_bounded_resume:
                        gate = {
                            **gate,
                            "visual_route_bounded_resume": True,
                            "visual_route_bounded_resume_progress": candidate_pose.get(
                                "route_visual_progress"
                            ),
                        }
                    if visual_stationary_retry:
                        gate = {
                            **gate,
                            "visual_route_stationary_retry": True,
                            "visual_route_stationary_retry_progress_delta_m": (
                                observed_translation_delta
                            ),
                        }
                pose_ready = bool(
                    gate.get("ok")
                    and candidate_position is not None
                    and translation_progress_ready
                    and endpoint_ready
                    and (not require_metric_pose or metric_pose_ready)
                    and (
                        not require_translation_safe
                        or not pose_gate_rotation_locked(gate)
                        or (target_arrived and not require_metric_pose)
                    )
                )
                if gate.get("ok") and candidate_position is not None:
                    # Recovery hover physically locks the aircraft, so a
                    # continuity-checked intermediate model pose can safely
                    # become the next publication anchor even when it is not
                    # yet allowed to resume translation. Without this, every
                    # bounded 2 cm catch-up remained measured from the old
                    # pose and eventually exceeded the 30 cm step gate before
                    # reaching the waypoint.
                    last_pose_gate = gate
                    recovery_display_anchor = candidate_position
                    if gate.get("route_progress") is not None:
                        active_route_progress = float(gate["route_progress"])
                if endpoint_metric_correction_hits >= 3:
                    return {
                        **gate,
                        "endpoint_metric_correction_ready": True,
                        "endpoint_metric_correction_progress": (
                            endpoint_metric_progress
                        ),
                        "endpoint_metric_correction_offset_m": (
                            endpoint_metric_offset_m
                        ),
                        "endpoint_metric_correction_hits": (
                            endpoint_metric_correction_hits
                        ),
                    }
                if pose_ready:
                    return (
                        {
                            **gate,
                            "visual_checkpoint_arrival": True,
                        }
                        if visual_checkpoint_arrived
                        else gate
                    )
                if require_metric_pose and not metric_pose_ready:
                    last_reason = (
                        "waiting for a fresh metric TSolve R,t; route-only or held "
                        "patrol poses cannot start another lap"
                    )
                elif require_translation_safe and gate.get("ok") and pose_gate_rotation_locked(gate):
                    last_reason = "waiting for the rotation-only position lock to release"
                elif require_endpoint_verified and not endpoint_ready:
                    if point_two_metric_endpoint_candidate:
                        last_reason = (
                            "waiting for three distinct stable TSolve poses inside "
                            "the three-centimetre Point-2 endpoint gate"
                        )
                    else:
                        last_reason = (
                            "waiting for progress-independent taught-endpoint image consensus"
                        )
                elif (
                    require_observed_translation_progress
                    and visual_bounded_resume
                    and not translation_progress_ready
                ):
                    measured_progress = max(0.0, float(observed_translation_delta or 0.0))
                    last_reason = (
                        "fresh visual route pose is verified, but observed forward "
                        f"progress is {measured_progress:.3f} m < 0.015 m; the single "
                        "bounded retry was already used, so movement remains locked"
                    )
                elif (
                    require_observed_translation_progress
                    and not translation_progress_ready
                ):
                    measured_progress = max(
                        0.0,
                        float(observed_translation_delta or 0.0),
                    )
                    last_reason = (
                        "fresh poses are arriving, but observed forward progress is "
                        f"{measured_progress:.3f} m < 0.015 m; requesting a newest-frame "
                        "global recovery while movement remains locked"
                    )
                else:
                    last_reason = str(gate.get("reason") or last_reason)
            abort_reason = f"TSolve localization did not recover within {wait_seconds:.1f}s ({last_reason})"
            return None
        finally:
            pose_recovery_active = False
            total_pose_recovery_pause_seconds += max(
                0.0,
                time.monotonic() - recovery_started,
            )

    def continuity_guarded_pose_gate() -> dict[str, Any]:
        """Reject a localization jump without poisoning the trusted patrol pose."""
        gate = latest_tsolve_pose_gate(
            pose_stream_path,
            pose_max_age,
            allow_rotation_frozen=is_patrol,
        )
        if not gate.get("ok"):
            route_visual_reconciliation_state.clear()
            return gate
        if pose_gate_predates_route_epoch(gate, active_route_pose_epoch_unix):
            route_visual_reconciliation_state.clear()
            pose = gate.get("pose") if isinstance(gate.get("pose"), dict) else {}
            return {
                "ok": False,
                "reason": (
                    "latest localization observation predates the verified "
                    "waypoint handoff; waiting for a post-handoff frame"
                ),
                "processed_count": gate.get("processed_count"),
                "latest_instance_id": pose.get("instance_id"),
                "route_pose_epoch": active_route_pose_epoch,
                "route_pose_epoch_unix": active_route_pose_epoch_unix,
                "pose_received_unix": pose.get("received_unix"),
                "route_pose_epoch_rejected": True,
            }
        gate = {**gate, "pose_offset_room": initial_pose_offset_room}
        candidate = pose_gate_position(gate)
        trusted = pose_gate_position(last_pose_gate)
        if candidate is None or trusted is None:
            return gate
        step = horizontal_xz_distance(candidate, trusted)
        if step is None:
            return gate
        try:
            old_count = int(last_pose_gate.get("processed_count") or 0)
            new_count = int(gate.get("processed_count") or old_count)
        except (TypeError, ValueError):
            old_count = new_count = 0
        frame_gap = max(1, new_count - old_count)
        # A rejected candidate must not become acceptable merely because more
        # bad frames followed it. Recovery is measured from the last trusted
        # pose until localization returns to the same neighborhood.
        trusted_pose = last_pose_gate.get("pose") if isinstance(last_pose_gate, dict) else {}
        candidate_pose = gate.get("pose") if isinstance(gate, dict) else {}
        if (
            isinstance(candidate_pose, dict)
            and candidate_pose.get("pose_source") == "patrol_visual_route_recovery"
        ):
            expected_identity = (
                str(mission.get("map_id") or ""),
                str(mission.get("patrol_id") or ""),
                str(mission.get("baseline_replay_id") or ""),
            )
            observed_identity = (
                str(candidate_pose.get("route_visual_map_id") or ""),
                str(candidate_pose.get("route_visual_patrol_id") or ""),
                str(candidate_pose.get("route_visual_baseline_replay_id") or ""),
            )
            if not all(expected_identity) or observed_identity != expected_identity:
                route_visual_reconciliation_state.clear()
                return {
                    "ok": False,
                    "reason": (
                        "visual patrol recovery identity does not match the active "
                        "map/patrol/baseline"
                    ),
                    "processed_count": gate.get("processed_count"),
                    "latest_instance_id": candidate_pose.get("instance_id"),
                    "visual_route_identity_rejected": True,
                }
        allowed_step = bounded_pose_step_limit(
            trusted_pose.get("received_unix") if isinstance(trusted_pose, dict) else None,
            candidate_pose.get("received_unix") if isinstance(candidate_pose, dict) else None,
            base_limit=max_pose_step,
            hard_limit=max_pose_step_hard,
        )
        command_bounded_catchup = command_bounded_pose_catchup_ready(
            candidate_pose,
            candidate,
            trusted,
            segment_start=active_route_segment_start,
            segment_end=active_route_segment_end,
            trusted_progress=active_route_progress,
            command_progress_ceiling=active_route_command_progress_ceiling,
            command_sequence=active_route_command_sequence,
            step=step,
            # Keep the global 0.30 m gate intact. Issued horizontal commands,
            # route direction and raw cross-track agreement jointly authorize
            # at most the existing independent 0.55 m tracking safety cap.
            hard_limit=0.55,
        )
        baseline_issue = baseline_supervised_pose_jump_issue(
            candidate_pose,
            step=step,
            base_step_limit=max_pose_step,
        )
        if baseline_issue is not None:
            route_visual_reconciliation_state.clear()
            return {
                "ok": False,
                "reason": baseline_issue,
                "age_seconds": gate.get("age_seconds"),
                "processed_count": gate.get("processed_count"),
                "latest_instance_id": candidate_pose.get("instance_id"),
                "baseline_supervised_jump_rejected": True,
                "pose_jump_map_units": step,
                "pose_jump_limit_map_units": max_pose_step,
                "route_visual_monitor_disagreement_m": candidate_pose.get(
                    "route_visual_monitor_disagreement_m"
                ),
            }
        if step > allowed_step and not command_bounded_catchup:
            route_visual_reconciliation_state.clear()
            return {
                "ok": False,
                "reason": (
                    f"rejected patrol pose jump {step:.3f} map units "
                    f"(limit {allowed_step:.3f}, frame gap {frame_gap}); keeping last trusted pose"
                ),
                "age_seconds": gate.get("age_seconds"),
                "processed_count": gate.get("processed_count"),
                "latest_instance_id": (gate.get("pose") or {}).get("instance_id"),
                "pose_jump_rejected": True,
                "pose_jump_map_units": step,
                "pose_jump_limit_map_units": allowed_step,
            }
        if command_bounded_catchup:
            gate = {
                **gate,
                "command_bounded_pose_catchup": True,
                "pose_jump_map_units": step,
                "pose_jump_limit_map_units": allowed_step,
                "route_progress_command_ceiling": (
                    active_route_command_progress_ceiling
                ),
                "route_progress_command_sequence": (
                    active_route_command_sequence
                ),
            }
        if route_monotonic_gate:
            route_reason, candidate_progress = patrol_route_pose_rejection(
                candidate,
                segment_start=active_route_segment_start,
                segment_end=active_route_segment_end,
                previous_progress=active_route_progress,
                translation_locked=active_route_translation_locked,
                position_anchor=active_route_position_anchor,
                max_cross_track=route_gate_max_cross_track,
                backward_tolerance=route_gate_backward_tolerance,
                turn_max_drift=route_gate_turn_max_drift,
                endpoint_overshoot_correction=(
                    active_endpoint_overshoot_correction
                ),
            )
            if route_reason is not None:
                if not route_reason.startswith("route progress moved backward"):
                    route_visual_reconciliation_state.clear()
                raw_candidate_progress = route_segment_progress_xz(
                    candidate,
                    active_route_segment_start,
                    active_route_segment_end,
                )
                if (
                    pose_recovery_active
                    and route_reason.startswith("route progress moved backward")
                    and patrol_visual_recovery_reconciliation_ready(
                        candidate_pose,
                        previous_progress=active_route_progress,
                        candidate_progress=raw_candidate_progress,
                        state=route_visual_reconciliation_state,
                    )
                ):
                    return {
                        **gate,
                        "route_progress": raw_candidate_progress,
                        "route_progress_reconciled": True,
                        "route_progress_previous": active_route_progress,
                        "route_progress_reconciliation_source": "pinned_visual_route_consensus",
                    }
                if (
                    pose_recovery_active
                    and route_reason.startswith("route progress moved backward")
                    and patrol_metric_recovery_reconciliation_ready(
                        gate,
                        previous_progress=active_route_progress,
                        candidate_progress=raw_candidate_progress,
                        state=route_metric_reconciliation_state,
                    )
                ):
                    route_visual_reconciliation_state.clear()
                    return {
                        **gate,
                        "route_progress": raw_candidate_progress,
                        "route_progress_reconciled": True,
                        "route_progress_previous": active_route_progress,
                        "route_progress_reconciliation_source": "neutral_metric_pose_consensus",
                    }
                return {
                    "ok": False,
                    "reason": f"rejected patrol pose: {route_reason}; keeping last trusted pose",
                    "age_seconds": gate.get("age_seconds"),
                    "processed_count": gate.get("processed_count"),
                    "latest_instance_id": (gate.get("pose") or {}).get("instance_id"),
                    "route_pose_rejected": True,
                    "route_progress": active_route_progress,
                }
            route_visual_reconciliation_state.clear()
            route_metric_reconciliation_state.clear()
            if candidate_progress is not None:
                gate = {**gate, "route_progress": candidate_progress}
        return gate

    def pose_gate_or_abort() -> dict[str, Any] | None:
        nonlocal abort_reason, last_pose_gate, active_route_progress
        if stop_flag is not None and stop_flag.stop:
            abort_reason = "live localization stop requested"
            return None
        if mission_stop_event is not None and mission_stop_event.is_set():
            abort_reason = "emergency hover requested"
            return None
        gate = continuity_guarded_pose_gate()
        if not gate.get("ok"):
            return wait_for_pose_recovery("motion_gate", str(gate.get("reason") or "TSolve pose gate failed"))
        if pose_gate_position(gate) is None:
            abort_reason = "latest TSolve pose has no ATLAS room-frame rcenter"
            return None
        last_pose_gate = gate
        if gate.get("route_progress") is not None:
            active_route_progress = float(gate["route_progress"])
        return gate

    def wait_for_pose_after(previous_gate: dict[str, Any] | None, timeout: float = 1.4) -> dict[str, Any] | None:
        previous_count = previous_gate.get("processed_count") if isinstance(previous_gate, dict) else None
        deadline = time.time() + max(0.15, timeout)
        last_gate: dict[str, Any] | None = None
        while time.time() < deadline:
            gate = pose_gate_or_abort()
            if gate is None:
                return None
            last_gate = gate
            if previous_count is None or gate.get("processed_count") != previous_count:
                return gate
            time.sleep(0.05)
        return last_gate

    def wait_for_pose_captured_after(
        capture_cutoff_unix: float,
        previous_gate: dict[str, Any] | None,
        timeout: float = 5.0,
        *,
        abort_on_timeout: bool = True,
    ) -> dict[str, Any] | None:
        """Wait for a camera observation captured after an RC pulse completed."""
        nonlocal abort_reason
        deadline = time.time() + max(0.2, timeout)
        last_gate: dict[str, Any] | None = None
        saw_capture_timestamp = False
        last_progress_at = 0.0
        while time.time() < deadline:
            gate = pose_gate_or_abort()
            if gate is None:
                return None
            last_gate = gate
            pose = gate.get("pose") if isinstance(gate, dict) else None
            try:
                received_unix = float(pose.get("received_unix")) if isinstance(pose, dict) else None
            except (TypeError, ValueError):
                received_unix = None
            saw_capture_timestamp = saw_capture_timestamp or received_unix is not None
            if received_unix is not None and received_unix >= capture_cutoff_unix:
                return gate
            if time.time() - last_progress_at >= 0.5:
                publish_progress(
                    {
                        "phase": "post_pulse_pose_wait",
                        "message": "Holding neutral; frames are arriving but ATLAS is waiting for an accepted observation captured after the last RC pulse.",
                        "capture_cutoff_unix": capture_cutoff_unix,
                        "latest_received_unix": received_unix,
                    }
                )
                last_progress_at = time.time()
            time.sleep(0.05)
        if saw_capture_timestamp:
            if abort_on_timeout:
                abort_reason = "no fresh camera observation arrived after the RC pulse; hovering"
            return None
        # Older replay streams have no received_unix. Preserve compatibility.
        return wait_for_pose_after(previous_gate, timeout=0.4)

    def execute_guarded_cruise_window(
        *,
        current_gate: dict[str, Any],
        current_position: list[float],
        target: list[float],
        bf: float,
        lr: float,
        du: float,
        seconds: float,
        arrival_radius: float,
    ) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
        """Fly one low-stick window while current-frame poses supervise it.

        The former controller neutralized after every 0.30 s forward command
        and only then waited for localization.  That made the aircraft and the
        viewer visibly pulse even when ten fresh poses per second were
        available.  This window refreshes DJI's virtual stick at 10 Hz while
        polling the independent localizer.  It sends neutral immediately if a
        pose becomes unsafe or the pose stream misses its short watchdog.
        """
        nonlocal abort_reason, last_pose_gate, active_route_progress

        # A complete smooth window is one bounded physical command. The outer
        # controller must observe published room-position progress from this
        # window before it can call us again. Retain the proven 450-550 ms
        # window from the best one-lap run: the low DJI stick needs enough
        # time to produce more than localization noise, and the accompanying
        # watchdog still neutralizes the aircraft when poses stop arriving.
        command_seconds = max(0.45, min(0.55, float(seconds)))
        sent = {
            "rcw_yaw": 0.0,
            "du_vertical": round(float(du), 2),
            "lr_lateral": round(float(lr), 2),
            "bf_forward": round(float(bf), 3),
            "seconds": round(command_seconds, 3),
            "smooth_continuous_cruise": True,
        }
        try:
            previous_count = int(current_gate.get("processed_count") or 0)
        except (TypeError, ValueError):
            previous_count = 0
        latest_count = previous_count
        latest_gate: dict[str, Any] | None = None
        latest_position = list(current_position)
        best_target_distance = horizontal_xz_distance(current_position, target)
        command_started_unix = time.time()
        command_started = time.monotonic()
        last_safe_observation = command_started
        last_progress_observation = command_started
        next_stick_refresh = command_started
        issue: str | None = None
        fresh_observations = 0

        try:
            while time.monotonic() - command_started < command_seconds:
                if stop_flag is not None and stop_flag.stop:
                    abort_reason = "live localization stop requested"
                    issue = abort_reason
                    break
                if mission_stop_event is not None and mission_stop_event.is_set():
                    abort_reason = "emergency hover requested"
                    issue = abort_reason
                    break

                now = time.monotonic()
                if now >= next_stick_refresh:
                    try:
                        _send_rc_with_bounded_ack(
                            drone,
                            rcw=0.0,
                            du=float(du),
                            lr=float(lr),
                            bf=float(bf),
                        )
                    except Exception as command_error:
                        try:
                            neutral_result = neutral_hover(drone, 0.0)
                        except ControlLinkSafetyError:
                            raise
                        sent.update(
                            {
                                "motion_command_acknowledged": False,
                                "motion_outcome_uncertain": True,
                                "neutral_confirmed": True,
                                "control_link_recovered": bool(
                                    neutral_result.get("control_link_recovered")
                                ),
                                "requires_pose_recovery": True,
                                "control_error": str(command_error),
                            }
                        )
                        issue = (
                            "DJI smooth-cruise acknowledgement was lost; neutral sticks "
                            "were confirmed and ATLAS must relocalize before resuming"
                        )
                        break
                    next_stick_refresh = now + 0.10

                candidate_gate = continuity_guarded_pose_gate()
                try:
                    candidate_count = int(candidate_gate.get("processed_count") or 0)
                except (TypeError, ValueError):
                    candidate_count = latest_count
                if candidate_count > latest_count:
                    latest_count = candidate_count
                    if not candidate_gate.get("ok"):
                        issue = str(
                            candidate_gate.get("reason")
                            or "localization became unsafe during smooth cruise"
                        )
                        break
                    candidate_issue = guided_command_pose_safety_issue(
                        candidate_gate,
                        yaw=0.0,
                        lr=lr,
                        bf=bf,
                        du=du,
                    )
                    if candidate_issue is not None:
                        issue = candidate_issue
                        break
                    candidate_position = pose_gate_position(candidate_gate)
                    if candidate_position is None:
                        issue = "smooth cruise pose has no ATLAS room position"
                        break
                    pose_step = horizontal_xz_distance(
                        latest_position,
                        candidate_position,
                    )
                    if pose_step is None or pose_step > max_pose_step:
                        issue = (
                            "smooth cruise rejected a current-frame pose step "
                            f"of {pose_step if pose_step is not None else float('nan'):.3f} m"
                        )
                        break
                    if not enforce_patrol_geofence(
                        candidate_gate,
                        "smooth continuous cruise",
                    ):
                        issue = abort_reason or "patrol geofence blocked smooth cruise"
                        break
                    latest_gate = candidate_gate
                    latest_position = candidate_position
                    last_pose_gate = candidate_gate
                    if candidate_gate.get("route_progress") is not None:
                        active_route_progress = float(candidate_gate["route_progress"])
                    fresh_observations += 1
                    last_safe_observation = now
                    candidate_distance = horizontal_xz_distance(
                        candidate_position,
                        target,
                    )
                    if candidate_distance is not None and (
                        best_target_distance is None
                        or candidate_distance <= best_target_distance - 0.005
                    ):
                        best_target_distance = candidate_distance
                        last_progress_observation = now
                    if (
                        candidate_distance is not None
                        and candidate_distance <= float(arrival_radius)
                    ):
                        break
                    if now - last_progress_observation > cruise_pose_watchdog_seconds:
                        issue = (
                            "fresh frames arrived but the published ATLAS position "
                            f"did not advance for {cruise_pose_watchdog_seconds:.2f}s"
                        )
                        break
                elif now - last_safe_observation > cruise_pose_watchdog_seconds:
                    issue = (
                        "no fresh translation-safe localization arrived within "
                        f"{cruise_pose_watchdog_seconds:.2f}s of smooth cruise"
                    )
                    break
                time.sleep(0.04)
        finally:
            # Zero the stick before the outer controller re-evaluates target
            # distance and heading. With no inserted sleep this is a watchdog
            # refresh, not the former visible stop between every tiny pulse.
            neutral_result = neutral_hover(drone, 0.0)
            if neutral_result.get("control_link_recovered"):
                sent.update(
                    {
                        "motion_command_acknowledged": True,
                        "motion_outcome_uncertain": True,
                        "neutral_confirmed": True,
                        "control_link_recovered": True,
                        "requires_pose_recovery": True,
                    }
                )
                if issue is None:
                    issue = (
                        "DJI control link closed while ending smooth cruise; neutral sticks "
                        "were confirmed on a fresh connection and ATLAS must relocalize "
                        "before resuming"
                    )

        latest_distance = horizontal_xz_distance(latest_position, target)
        arrived = bool(
            latest_distance is not None
            and latest_distance <= float(arrival_radius)
        )
        observed_model_progress = published_position_advanced_toward_target(
            current_position,
            pose_gate_position(latest_gate),
            target,
            minimum_improvement=0.010,
        )
        if issue is None and not arrived:
            issue = patrol_translation_pulse_progress_issue(
                current_position,
                pose_gate_position(latest_gate),
                target,
                got_new_pose=fresh_observations > 0,
                maximum_pose_step=max_pose_step,
                minimum_improvement=0.010,
            )
        sent["fresh_pose_observations"] = fresh_observations
        sent["observed_model_progress"] = bool(observed_model_progress)
        sent["progress_required_before_next_window"] = True
        sent["actual_seconds"] = round(time.monotonic() - command_started, 3)
        sent["command_started_unix"] = command_started_unix
        sent["arrival_detected"] = arrived
        return sent, latest_gate, issue

    def observe_motion_after_probe(
        previous_gate: dict[str, Any] | None,
        before: list[float] | None,
        *,
        timeout: float = 2.8,
    ) -> tuple[dict[str, Any] | None, list[float] | None, float | None]:
        previous_count = previous_gate.get("processed_count") if isinstance(previous_gate, dict) else None
        deadline = time.time() + max(0.6, timeout)
        best_gate: dict[str, Any] | None = None
        best_axis: list[float] | None = None
        best_delta: float | None = None
        last_gate: dict[str, Any] | None = None
        while time.time() < deadline:
            gate = pose_gate_or_abort()
            if gate is None:
                return None, None, None
            last_gate = gate
            if previous_count is not None and gate.get("processed_count") == previous_count:
                time.sleep(0.05)
                continue
            after = pose_gate_position(gate)
            axis, delta = observed_motion_axis(before, after, min_delta=0.0)
            if delta is not None and (best_delta is None or delta > best_delta):
                best_gate = gate
                best_axis = axis
                best_delta = delta
            if best_delta is not None and best_delta >= 0.045:
                break
            time.sleep(0.05)
        if best_axis is None or best_delta is None or best_delta < 0.018:
            return last_gate or best_gate, None, best_delta
        return best_gate or last_gate, best_axis, best_delta

    def calibrate_body_axis(
        axis_name: str,
        *,
        lr: float = 0.0,
        bf: float = 0.0,
        reference_gate: dict[str, Any] | None = None,
    ) -> bool:
        nonlocal executed_pulses
        nonlocal active_route_translation_locked, active_route_position_anchor
        gate_before = reference_gate or pose_gate_or_abort()
        before = pose_gate_position(gate_before)
        if before is None or not enforce_patrol_geofence(gate_before, f"{axis_name} calibration"):
            return False
        translation_issue = guided_command_pose_safety_issue(gate_before, lr=lr, bf=bf)
        if translation_issue is not None:
            publish_progress(
                {
                    "phase": "translation_pose_gate",
                    "axis": axis_name,
                    "translation_locked": True,
                    "message": f"Calibration held: {translation_issue}",
                }
            )
            neutral_hover(drone, 0.20)
            return False
        # This is a real translation pulse, not a yaw-only command.  Publish
        # that fact before moving so the live rotation stabilizer releases any
        # position anchor.  The previous implementation advertised
        # translation_locked=true with zero gains, causing the localizer to
        # freeze the forward probe and later report its accumulated motion as
        # an impossible sideways heading calibration.
        active_route_translation_locked = False
        active_route_position_anchor = None
        publish_progress(
            {
                "phase": "axis_calibration",
                "axis": axis_name,
                "translation_locked": False,
                "position_anchor": None,
                "body_forward_gain": bf / axis_probe_rc if abs(axis_probe_rc) > 1e-9 else 0.0,
                "body_lateral_gain": lr / axis_probe_rc if abs(axis_probe_rc) > 1e-9 else 0.0,
                "message": f"Calibrating safe {axis_name} response from TSolve pose feedback.",
            }
        )
        sent = execute_rc_pulse(
            drone,
            lr=lr,
            bf=bf,
            seconds=axis_probe_seconds,
        )
        record_pulse(axis_name, sent)
        executed_pulses += 1
        _gate_after, axis, delta = observe_motion_after_probe(gate_before, before, timeout=2.8)
        rc_summary["adaptive_axis"][f"{axis_name}_probe_delta"] = delta
        if axis is None:
            return False
        body_axes[axis_name] = axis
        rc_summary["adaptive_axis"][f"{axis_name}_axis_xz"] = [round(axis[0], 6), 0.0, round(axis[2], 6)]
        return True

    def calibrate_body_axes(reference_gate: dict[str, Any] | None = None) -> bool:
        nonlocal axis_recalibrations
        axis_recalibrations += 1
        rc_summary["adaptive_axis"]["axis_recalibrations"] = axis_recalibrations
        body_axes["forward"] = None
        body_axes["lateral"] = None
        ok_forward = calibrate_body_axis("forward", bf=axis_probe_rc, reference_gate=reference_gate)
        gate = pose_gate_or_abort()
        ok_lateral = calibrate_body_axis("lateral", lr=axis_probe_rc, reference_gate=gate)
        if not ok_forward:
            rc_summary["adaptive_axis"]["forward_axis_fallback"] = "disabled_for_safety"
        if not ok_lateral and body_axes["forward"] is not None:
            rc_summary["adaptive_axis"]["lateral_axis_fallback"] = "disabled_for_safety"
        return ok_forward and (ok_lateral or not allow_lateral_rc)

    def calibrate_forward_heading(reference_gate: dict[str, Any] | None = None) -> bool:
        """Measure DJI body-forward relative to the localized camera heading."""
        nonlocal calibrated_heading_offset_rad
        gate_before = reference_gate or pose_gate_or_abort()
        raw_heading = pose_gate_heading(gate_before, heading_trim_rad)
        if raw_heading is None:
            return False
        body_axes["forward"] = None
        if not calibrate_body_axis("forward", bf=axis_probe_rc, reference_gate=gate_before):
            return False
        measured_forward = normalize_xz(body_axes["forward"])
        offset = signed_angle_xz(raw_heading, measured_forward)
        if offset is None or not math.isfinite(offset):
            return False
        if operator_heading_calibrated:
            calibration_error_deg = heading_calibration_error_degrees(
                raw_heading,
                measured_forward,
                initial_body_heading_offset_deg,
            )
            rc_summary["adaptive_axis"]["heading_calibration_error_deg"] = (
                round(calibration_error_deg, 3) if calibration_error_deg is not None else None
            )
            if (
                calibration_error_deg is None
                or calibration_error_deg > max_heading_calibration_error_deg
            ):
                body_axes["forward"] = None
                publish_progress(
                    {
                        "phase": "heading_calibration_rejected",
                        "translation_locked": False,
                        "position_anchor": None,
                        "body_forward_gain": 0.0,
                        "body_lateral_gain": 0.0,
                        "heading_calibration_error_deg": calibration_error_deg,
                        "message": (
                            "Rejected an implausible body-forward measurement; "
                            "hovering without yaw or route translation."
                        ),
                    }
                )
                return False
        calibrated_heading_offset_rad = offset
        rc_summary["adaptive_axis"]["mode"] = "calibrated_heading_yaw_then_forward"
        rc_summary["adaptive_axis"]["camera_to_body_heading_offset_deg"] = round(math.degrees(offset), 3)
        publish_progress(
            {
                "phase": "heading_calibration",
                "message": (
                    "Measured DJI body-forward heading offset "
                    f"{math.degrees(offset):+.1f} degrees; patrol will yaw, then fly forward."
                ),
                "camera_to_body_heading_offset_deg": math.degrees(offset),
            }
        )
        return True

    first_pose = wait_for_pose_recovery("mission_start", "waiting for first fresh TSolve pose")
    if first_pose is None:
        neutral_hover(drone, 0.25)
        return {
            "ok": False,
            "armed": True,
            "physical_motion_locked": False,
            "aborted": True,
            "abort_reason": abort_reason,
            "executed": executed,
            "skipped": skipped,
            "command_count": len(commands),
            "message": f"Mission refused before motion: {abort_reason}",
            "elapsed_seconds": time.time() - started,
        }
    if pose_gate_position(first_pose) is None:
        neutral_hover(drone, 0.25)
        abort_reason = "latest TSolve pose has no ATLAS room-frame rcenter; refusing guided patrol motion"
        return {
            "ok": False,
            "armed": True,
            "physical_motion_locked": False,
            "aborted": True,
            "abort_reason": abort_reason,
            "executed": executed,
            "skipped": skipped,
            "command_count": len(commands),
            "last_pose_gate": first_pose,
            "message": "Mission refused before motion: latest TSolve pose is not in the ATLAS room frame.",
            "elapsed_seconds": time.time() - started,
        }
    if not enforce_patrol_geofence(first_pose, "mission start"):
        return {
            "ok": False,
            "armed": True,
            "physical_motion_locked": False,
            "aborted": True,
            "abort_reason": abort_reason,
            "executed": executed,
            "skipped": skipped,
            "command_count": len(commands),
            "last_pose_gate": first_pose,
            "message": f"Mission refused before motion: {abort_reason}",
            "elapsed_seconds": time.time() - started,
        }
    if patrol_stage == "loop":
        route = mission.get("route") if isinstance(mission.get("route"), list) else []
        loop_start = vector3(route[0]) if route else None
        current_start = pose_gate_position(first_pose)
        start_error = horizontal_xz_distance(current_start, loop_start)
        if start_error is None or start_error > 0.24:
            neutral_hover(drone, 0.25)
            abort_reason = (
                "two-circle patrol requires a fresh pose within 0.24 map units of "
                f"Point 1 (current error {start_error:.3f})"
                if start_error is not None
                else "two-circle patrol has no valid Point 1 start pose"
            )
            return {
                "ok": False,
                "armed": True,
                "physical_motion_locked": False,
                "aborted": True,
                "abort_reason": abort_reason,
                "executed": executed,
                "skipped": skipped,
                "command_count": len(commands),
                "last_pose_gate": first_pose,
                "message": f"Mission refused before motion: {abort_reason}",
                "elapsed_seconds": time.time() - started,
            }

    try:
        rc_summary["enable_control_result"] = drone.enableControl(True)
        if not dji_control_response_ok(rc_summary["enable_control_result"]):
            abort_reason = dji_control_response_reason(rc_summary["enable_control_result"])
            return {
                "ok": False,
                "armed": True,
                "physical_motion_locked": False,
                "aborted": True,
                "abort_reason": abort_reason,
                "executed_hover_steps": hover_count,
                "executed_pulses": executed_pulses,
                "last_pose_gate": last_pose_gate,
                "guided_settings": {
                    "pose_max_age_seconds": pose_max_age,
                    "pose_recovery_seconds": pose_recovery_seconds,
                    "continuous_relocalization": continuous_relocalization,
                    "pulse_seconds": pulse_seconds,
                    "max_forward_rc": max_forward_rc,
                    "max_lateral_rc": max_lateral_rc,
                    "allow_lateral_rc": allow_lateral_rc,
                    "allow_axis_auto_calibration": allow_axis_auto_calibration,
                    "max_yaw_rc": max_yaw_rc,
                    "max_vertical_rc": max_vertical_rc,
                    "max_step_seconds": max_step_seconds,
                    "arrival_radius_map_units": arrival_radius,
                    "arrival_deadband_map_units": arrival_deadband,
                    "strict_entry_arrival": strict_entry_arrival,
                    "max_cruise_seconds": max_cruise_seconds,
                    "max_pose_step_map_units": max_pose_step,
                    "safety_motion_buffer_m": safety_motion_buffer,
                    "heading_trim_deg": heading_trim_deg,
                },
                "rc_summary": rc_summary,
                "executed": executed,
                "skipped": skipped,
                "command_count": len(commands),
                "message": f"Guided mission refused before motion: {abort_reason}",
                "elapsed_seconds": time.time() - started,
            }
        control_enabled = True

        step_source = mission_step_sequence(
            commands,
            patrol_loop,
            patrol_laps if patrol_laps > 0 else None,
            patrol_loop_start,
        )
        for _execution_index, (idx, step) in enumerate(step_source):
            if not isinstance(step, dict):
                skipped.append({"index": idx, "type": "unknown", "reason": "invalid command record"})
                continue
            kind = str(step.get("type", "")).strip().lower()
            title = str(step.get("title", kind or "step"))
            explicit_lap_start = step.get("_atlas_lap_start") is True
            if patrol_loop and (
                explicit_lap_start
                or (idx == patrol_loop_start and current_lap_number <= 0)
            ):
                try:
                    requested_lap_number = int(step.get("_atlas_lap_number") or 0)
                except (TypeError, ValueError):
                    requested_lap_number = 0
                current_lap_number = (
                    requested_lap_number
                    if requested_lap_number > 0
                    else current_lap_number + 1
                )
                lap_metric_checkpoint_pending = current_lap_number > 1
                lap_reentry_metric_ready = False
                # The loop marker precedes the first cruise command.  Do not
                # publish the completed 4->1 segment under the new lap number.
                # The cruise preserves that verified Point-1 endpoint only as
                # a yaw anchor, turns toward Point 2, and then obtains a fresh
                # metric pose before it can issue horizontal RC.
                active_route_segment_start = None
                active_route_segment_end = None
                active_route_progress = None
                active_route_translation_locked = True
                active_route_position_anchor = None
                if len(executed) > 500:
                    del executed[:-250]
                if len(skipped) > 500:
                    del skipped[:-250]
                publish_progress(
                    {
                        "phase": "patrol_lap",
                        "lap": current_lap_number,
                        "translation_locked": True,
                        "message": (
                            f"Starting patrol lap {current_lap_number}; ordered route begins at patrol point 1."
                            if current_lap_number == 1
                            else (
                                f"Starting patrol lap {current_lap_number}; turning from Point 1 toward "
                                "Point 2 before the fresh metric relocalization. Translation stays locked."
                            )
                        ),
                    }
                )
            if kind == "gate":
                skipped.append({"index": idx, "type": kind, "title": title, "reason": "UI confirmation gate"})
                continue

            # Navigation yaw is performed inside the following closed-loop
            # cruise. Do not require a fresh pose merely to skip this marker:
            # the next cruise can perform bounded yaw-only recovery from the
            # last trusted pose if the weak view began during the point hold.
            if kind == "yaw" and str(step.get("safety") or "").strip().lower() == "slow-yaw":
                skipped.append(
                    {
                        "index": idx,
                        "type": kind,
                        "title": title,
                        "reason": "navigation yaw is handled by the closed-loop cruise controller",
                    }
                )
                continue

            if kind == "lap_relocalize_entry":
                # Older saved mission packets may still contain the former
                # pre-turn relocalization marker.  It is deliberately ignored:
                # the Point-1 -> Point-2 cruise performs yaw first and requests
                # the metric rebootstrap only after the heading is aligned.
                skipped.append(
                    {
                        "index": idx,
                        "type": kind,
                        "title": title,
                        "reason": "obsolete pre-turn lap relocalization marker",
                    }
                )
                continue

            gate_attempt = continuity_guarded_pose_gate()
            if kind == "cruise" and not gate_attempt.get("ok") and isinstance(last_pose_gate, dict):
                gate = last_pose_gate
            else:
                gate = pose_gate_or_abort()
            if gate is None:
                break
            if not enforce_patrol_geofence(gate, f"step {idx} {kind or 'unknown'}"):
                break

            if kind == "hover":
                duration = max(0.1, min(2.0, float(step.get("duration_s") or 0.5)))
                neutral_hover(drone, duration)
                hover_count += 1
                executed.append({"index": idx, "type": kind, "title": title, "duration_s": duration})
                continue

            if kind == "yaw":
                if (
                    patrol_loop
                    and str(step.get("safety") or "").strip().lower() == "scan-yaw"
                    and not allow_patrol_scan_yaw
                ):
                    skipped.append(
                        {
                            "index": idx,
                            "type": kind,
                            "title": title,
                            "reason": "patrol scan yaw disabled to preserve TSolve continuity; forward hover scan only",
                        }
                    )
                    neutral_hover(drone, 0.35)
                    hover_count += 1
                    continue
                yaw_delta = clamp_float(step.get("yaw_delta_deg"), 0.0, -180.0, 180.0)
                if abs(yaw_delta) < 2.0:
                    skipped.append({"index": idx, "type": kind, "title": title, "reason": "initial or negligible yaw delta"})
                    continue
                yaw_rc = yaw_sign * (max_scan_yaw_rc if yaw_delta > 0 else -max_scan_yaw_rc)
                seconds = min(max_step_seconds, max(pulse_seconds, abs(yaw_delta) / 90.0 * 0.8))
                pulses = max(1, int(round(seconds / pulse_seconds)))
                for _ in range(pulses):
                    pulse_gate = pose_gate_or_abort()
                    if pulse_gate is None or not enforce_patrol_geofence(pulse_gate, f"scan yaw step {idx}"):
                        break
                    publish_progress(
                        {
                            "phase": "scan_yaw",
                            "step_index": idx,
                            "step_title": title,
                            "message": f"Scan yaw: {title}",
                        }
                    )
                    sent = execute_rc_pulse(drone, yaw=yaw_rc, seconds=pulse_seconds)
                    record_pulse("yaw", sent)
                    executed_pulses += 1
                if abort_reason:
                    break
                if allow_axis_auto_calibration:
                    body_axes["forward"] = None
                    body_axes["lateral"] = None
                    rc_summary["adaptive_axis"]["forward_axis_xz"] = None
                    rc_summary["adaptive_axis"]["lateral_axis_xz"] = None
                    publish_progress(
                        {
                            "phase": "axis_calibration_reset",
                            "step_index": idx,
                            "step_title": title,
                            "message": "Yaw changed the DJI body frame; ATLAS will recalibrate before the next cruise segment.",
                        }
                    )
                executed.append(
                    {
                        "index": idx,
                        "type": kind,
                        "title": title,
                        "yaw_delta_deg": yaw_delta,
                        "pulse_count": pulses,
                    }
                )
                continue

            if kind == "cruise":
                cruise_stage = str(step.get("patrol_stage") or patrol_stage).strip().lower()
                taught_leg = taught_leg_for_step(taught_reference, step)
                route_follow_leg = verified_route_follow_leg(
                    verified_route_lock,
                    taught_leg,
                )
                try:
                    route_follow_alignment_enter_deg = max(
                        2.0,
                        min(
                            6.0,
                            float(
                                (verified_route_lock or {}).get(
                                    "alignment_enter_deg", 4.0
                                )
                            ),
                        ),
                    )
                    route_follow_alignment_exit_deg = max(
                        route_follow_alignment_enter_deg + 1.0,
                        min(
                            8.0,
                            float(
                                (verified_route_lock or {}).get(
                                    "alignment_exit_deg", 6.0
                                )
                            ),
                        ),
                    )
                    route_follow_lookahead_m = max(
                        0.10,
                        min(
                            0.60,
                            float(
                                (verified_route_lock or {}).get(
                                    "lookahead_m", 0.35
                                )
                            ),
                        ),
                    )
                    route_follow_max_correction_deg = max(
                        0.0,
                        min(
                            12.0,
                            float(
                                (verified_route_lock or {}).get(
                                    "max_route_correction_deg", 7.0
                                )
                            ),
                        ),
                    )
                except (TypeError, ValueError):
                    route_follow_alignment_enter_deg = 4.0
                    route_follow_alignment_exit_deg = 6.0
                    route_follow_lookahead_m = 0.35
                    route_follow_max_correction_deg = 7.0
                precise_arrival = taught_leg_requires_precise_arrival(taught_leg)
                try:
                    endpoint_leg_index = (
                        int(taught_leg.get("from_point"))
                        if precise_arrival and isinstance(taught_leg, dict)
                        else None
                    )
                except (TypeError, ValueError):
                    endpoint_leg_index = None
                cruise_arrival_radius, cruise_soft_arrival_radius = mission_arrival_radii(
                    configured_arrival_radius,
                    arrival_deadband,
                    patrol_stage=cruise_stage,
                    strict_target=precise_arrival,
                )
                dynamic_lap_reentry = step.get("_atlas_lap_reentry") is True
                target = step_target_position(step)
                if target is None:
                    abort_reason = "cruise target is missing; refusing open-loop patrol travel"
                    break
                raw_start_position = pose_gate_position(gate)
                if dynamic_lap_reentry:
                    if not lap_reentry_metric_ready:
                        abort_reason = (
                            "repeated-lap Point-1 entry has no fresh global metric "
                            "localization; hovering before motion"
                        )
                        break
                    if raw_start_position is None:
                        abort_reason = (
                            "repeated-lap Point-1 entry has no room-frame start pose"
                        )
                        break
                planned_duration = clamp_float(step.get("duration_s"), pulse_seconds, pulse_seconds, 120.0)
                distance = clamp_float(step.get("distance"), 0.0, 0.0, 1000.0)
                if dynamic_lap_reentry:
                    distance = float(
                        horizontal_xz_distance(raw_start_position, target) or 0.0
                    )
                    entry_speed_mps = clamp_float(
                        step.get("speed_mps"),
                        0.10,
                        0.04,
                        0.20,
                    )
                    planned_duration = max(
                        pulse_seconds,
                        min(120.0, distance / max(0.04, entry_speed_mps)),
                    )
                if distance <= 1e-4:
                    if dynamic_lap_reentry:
                        lap_metric_checkpoint_pending = False
                        lap_reentry_metric_ready = False
                        publish_progress(
                            {
                                "phase": "lap_point1_entry_verified",
                                "lap": current_lap_number,
                                "translation_locked": True,
                                "position_anchor": target,
                                "saved_point1": target,
                                "distance_to_target": distance,
                                "message": (
                                    f"Lap {current_lap_number}: global localization is "
                                    "already at saved Point 1; Point 1->2 may begin."
                                ),
                            }
                        )
                    skipped.append({"index": idx, "type": kind, "title": title, "reason": "zero distance segment"})
                    continue

                # A 90-degree corner is alignment work, not failed translation.
                # Give it a separate bounded allowance before the normal travel
                # budget, while retaining the overall mission step ceiling.
                alignment_deadline = motion_clock() + min(
                    max_cruise_seconds,
                    alignment_grace_seconds + max(4.0, planned_duration * 2.2),
                )
                # Fresh-pose waits make a real DJI corner consume much more
                # wall time than its RC command time. Permit extra wall time
                # only while the measured heading error keeps improving.
                segment_safety_deadline = motion_clock() + max_cruise_seconds
                absolute_segment_deadline = motion_clock() + min(240.0, max_cruise_seconds * 2.0)
                alignment_progress_deadline = alignment_deadline
                best_alignment_error_rad = float("inf")
                forward_budget_seconds = min(
                    max_cruise_seconds,
                    max(4.0, planned_duration * 2.2),
                )
                yaw_correction_budget_seconds = min(
                    max_cruise_seconds,
                    max(8.0, alignment_grace_seconds),
                )
                travel_started = False
                forward_command_seconds = 0.0
                travel_yaw_command_seconds = 0.0
                segment_start = (
                    list(raw_start_position)
                    if dynamic_lap_reentry and raw_start_position is not None
                    else vector3(step.get("from"))
                )
                # A verified taught waypoint is the exact shared boundary of
                # two legs. Start every lap/leg from that recorded boundary,
                # not from a post-turn monocular center. The latter gave the
                # failed 14:08 run a false +14.4% head start on Point 2→3.
                start_position = (
                    list(segment_start)
                    if precise_arrival and segment_start is not None
                    else raw_start_position
                )
                initial_distance = horizontal_xz_distance(start_position, target)
                final_distance = initial_distance
                closest_distance = initial_distance if initial_distance is not None else float("inf")
                diverging_pulses = 0
                pulse_count = 0
                yaw_pulse_count = 0
                forward_pulse_count = 0
                lateral_pulse_count = 0
                reached = False
                arrival_mode = None
                stale_motion_count = 0
                stale_motion_reference_position: list[float] | None = None
                visual_stationary_retry_reference_distance: float | None = None
                wrong_yaw_pulses = 0
                correct_yaw_pulses = 0
                yaw_feedback_recovery_count = 0
                last_navigation_mode = "unknown"
                last_processed_count = None
                last_pose_age = None
                # Every patrol leg begins with an in-place heading alignment.
                # A yaw command cannot translate the aircraft, so navigation
                # must keep the leg's starting position fixed until the first
                # real forward/lateral pulse.  This prevents monocular center
                # drift during any waypoint turn, not only point 3 -> point 4.
                yaw_position_anchor = start_position
                forward_alignment_locked = False
                last_alignment_yaw_rc = 0.0
                last_yaw_command_observation_id: str | None = None
                last_measured_alignment_error_rad: float | None = None
                blind_yaw_seconds = 0.0
                max_blind_yaw_seconds = 12.0
                active_route_segment_start = segment_start or start_position
                active_route_segment_end = target
                active_route_progress = (
                    0.0
                    if precise_arrival and segment_start is not None
                    else route_segment_progress_xz(
                        start_position,
                        active_route_segment_start,
                        active_route_segment_end,
                    )
                )
                # The new leg begins with yaw-only alignment.  Until a real
                # horizontal command is issued, its complete translation
                # budget is the trusted start progress (normally exactly 0).
                active_route_command_progress_ceiling = active_route_progress
                active_route_command_sequence = 0
                route_visual_reconciliation_state.clear()
                route_metric_reconciliation_state.clear()
                tight_metric_endpoint_consensus_state: dict[str, Any] = {}
                point_two_metric_endpoint_consensus_state: dict[str, Any] = {}
                active_endpoint_overshoot_correction = False
                active_endpoint_undershoot_correction = False
                endpoint_undershoot_retry_used = False
                # The rotation-only heading is accumulated from an arbitrary
                # optical anchor. At Point 3 the direct recorded departure
                # match supplies an absolute room heading; retain their
                # measured offset for the complete 3->4 leg instead of
                # switching back to the raw optical basis after one pulse.
                route_optical_heading_bias_rad: float | None = None
                active_route_translation_locked = True
                active_route_position_anchor = start_position
                point_three_handoff = is_guarded_point_three_to_four_turn(
                    taught_leg
                )
                lap_point_one_handoff = False
                point_three_planned_turn_degrees: float | None = None
                point_three_yaw_effort_seconds = 0.0
                point_three_recorded_recovery_effort_seconds = 0.0
                point_three_recorded_recovery_effort_limit = math.inf
                point_four_handoff = is_point_four_to_one_leg(taught_leg)
                verified_endpoint_turn_source_gate: dict[str, Any] | None = None
                verified_endpoint_turn_anchor: list[float] | None = None
                verified_endpoint_turn_source: str | None = None
                if point_three_handoff:
                    # Capture the independently verified Point-3 endpoint
                    # before publishing the new 3->4 route context.  The
                    # localizer intentionally freezes this position during
                    # the following yaw, so the proof would otherwise be
                    # hidden by the new leg's rotation-locked hold frames.
                    point_three_handoff_pose = (
                        gate.get("pose") if isinstance(gate, dict) else None
                    )
                    point_three_handoff_position = pose_gate_position(gate)
                    point_three_metric_ready = bool(
                        pose_gate_has_fresh_metric_position(gate)
                        and not pose_gate_rotation_locked(gate)
                        and point_three_handoff_position is not None
                    )
                    point_three_visual_ready = taught_endpoint_arrival_verified(
                        point_three_handoff_pose,
                        expected_leg_index=2,
                    )
                    point_three_handoff_error = horizontal_xz_distance(
                        point_three_handoff_position,
                        segment_start,
                    )
                    point_three_handoff_limit = min(0.30, max_pose_step)
                    if (
                        (point_three_metric_ready or point_three_visual_ready)
                        and segment_start is not None
                        and point_three_handoff_error is not None
                        and point_three_handoff_error
                        <= point_three_handoff_limit
                    ):
                        verified_endpoint_turn_source_gate = gate
                        verified_endpoint_turn_anchor = list(segment_start)
                        verified_endpoint_turn_source = (
                            "metric_tsolve"
                            if point_three_metric_ready
                            else "verified_visual_endpoint"
                        )
                        rc_summary.setdefault("point3_handoffs", []).append(
                            {
                                "lap": current_lap_number,
                                "instance_id": point_three_handoff_pose.get(
                                    "instance_id"
                                ),
                                "source": verified_endpoint_turn_source,
                                "position": list(segment_start),
                                "point3_error_map_units": (
                                    point_three_handoff_error
                                ),
                            }
                        )
                        publish_progress(
                            {
                                "phase": "point3_handoff_verified",
                                "step_index": idx,
                                "step_title": title,
                                "translation_locked": True,
                                "position_anchor": segment_start,
                                "point3_handoff_source": (
                                    verified_endpoint_turn_source
                                ),
                                "point3_position_error_map_units": (
                                    point_three_handoff_error
                                ),
                                "message": (
                                    "Point-3 metric endpoint verified before the "
                                    "rotation-only 3->4 alignment."
                                ),
                            }
                        )
                if point_four_handoff:
                    # Route-image progress is deliberately constrained to the
                    # taught centerline.  It can prove that the camera reached
                    # the Point-4 visual neighborhood, but it cannot measure a
                    # small real left/right offset. Prefer a fresh metric pose
                    # when one already exists, but do not require one before a
                    # yaw-only command after independent endpoint consensus.
                    # Live ATLAS 14:41:16 supplied three verified Point-4
                    # endpoint frames and then waited 30 seconds only because
                    # this handoff rejected route vision. The three-frame
                    # live optical departure gate below still verifies heading
                    # before the first 4->1 translation pulse.
                    point_four_metric_gate = gate
                    point_four_handoff_pose = (
                        point_four_metric_gate.get("pose")
                        if (
                            isinstance(point_four_metric_gate, dict)
                            and isinstance(point_four_metric_gate.get("pose"), dict)
                        )
                        else {}
                    )
                    point_four_metric_position = pose_gate_position(
                        point_four_metric_gate
                    )
                    point_four_metric_ready = (
                        pose_gate_has_fresh_metric_position(point_four_metric_gate)
                        and point_four_metric_position is not None
                    )
                    prior_point_four_arrival = (
                        prior_verified_endpoint_arrival_record(
                            executed,
                            segment_start=segment_start,
                            expected_leg_index=3,
                        )
                    )
                    prior_point_four_visual_ready = bool(
                        prior_point_four_arrival is not None
                    )
                    point_four_visual_ready = taught_endpoint_arrival_verified(
                        point_four_handoff_pose,
                        expected_leg_index=3,
                    ) or prior_point_four_visual_ready
                    if not point_four_metric_ready and not point_four_visual_ready:
                        publish_progress(
                            {
                                "phase": "point4_endpoint_handoff",
                                "step_index": idx,
                                "step_title": title,
                                "translation_locked": True,
                                "require_metric_pose": False,
                                "position_anchor": start_position,
                                "body_forward_gain": 0.0,
                                "body_lateral_gain": 0.0,
                                "message": (
                                    "Holding position briefly while ATLAS verifies the "
                                    "Point-4 endpoint before the rotation-only 4->1 turn."
                                ),
                            }
                        )
                        point_four_metric_gate = wait_for_pose_recovery(
                            "point4_endpoint_handoff",
                            "waiting for Point-4 endpoint image consensus before the 4->1 turn",
                            timeout=8.0,
                            # The next physical command is yaw-only. A verified
                            # rotation-locked endpoint is therefore sufficient;
                            # translation is released only after the new 4->1
                            # heading is independently stable for three frames.
                            require_translation_safe=False,
                            require_endpoint_verified=True,
                            endpoint_leg_index=3,
                        )
                        if point_four_metric_gate is None:
                            break
                        point_four_metric_position = pose_gate_position(
                            point_four_metric_gate
                        )
                        point_four_handoff_pose = (
                            point_four_metric_gate.get("pose") or {}
                        )
                        point_four_metric_ready = bool(
                            pose_gate_has_fresh_metric_position(
                                point_four_metric_gate
                            )
                            and point_four_metric_position is not None
                        )
                        point_four_visual_ready = taught_endpoint_arrival_verified(
                            point_four_handoff_pose,
                            expected_leg_index=3,
                        ) or prior_point_four_visual_ready
                    point_four_metric_error = (
                        horizontal_xz_distance(
                            point_four_metric_position,
                            segment_start,
                        )
                        if point_four_metric_ready
                        else None
                    )
                    point_four_handoff_limit = min(0.30, max_pose_step)
                    if point_four_metric_ready and (
                        point_four_metric_error is None
                        or point_four_metric_error > point_four_handoff_limit
                    ):
                        abort_reason = (
                            "fresh metric Point-4 pose is outside the safe handoff "
                            f"radius ({point_four_metric_error!s} > "
                            f"{point_four_handoff_limit:.2f}m); hovering before turn"
                        )
                        publish_progress(
                            {
                                "phase": "point4_position_correction_required",
                                "step_index": idx,
                                "step_title": title,
                                "translation_locked": True,
                                "require_metric_pose": True,
                                "position_anchor": point_four_metric_position,
                                "point4_position_error_map_units": point_four_metric_error,
                                "point4_position_limit_map_units": point_four_handoff_limit,
                                "body_forward_gain": 0.0,
                                "body_lateral_gain": 0.0,
                                "message": abort_reason,
                            }
                        )
                        break
                    if point_four_metric_ready:
                        point_four_handoff_position = list(point_four_metric_position)
                        point_four_handoff_source = "metric_tsolve"
                    elif point_four_visual_ready and segment_start is not None:
                        point_four_handoff_position = list(segment_start)
                        point_four_handoff_source = "verified_visual_endpoint"
                    else:
                        abort_reason = (
                            "Point-4 handoff lost both metric and verified visual "
                            "localization; hovering before turn"
                        )
                        break
                    active_route_pose_epoch += 1
                    active_route_pose_epoch_unix = time.time()
                    active_route_pose_epoch_reason = "verified_point4_handoff"
                    point_four_anchor_gate = verified_route_anchor_pose_gate(
                        point_four_metric_gate,
                        point_four_handoff_position,
                        epoch=active_route_pose_epoch,
                        epoch_unix=active_route_pose_epoch_unix,
                        reason=active_route_pose_epoch_reason,
                    )
                    gate = point_four_anchor_gate
                    last_pose_gate = point_four_anchor_gate
                    start_position = list(point_four_handoff_position)
                    initial_distance = horizontal_xz_distance(start_position, target)
                    final_distance = initial_distance
                    closest_distance = (
                        initial_distance
                        if initial_distance is not None
                        else float("inf")
                    )
                    yaw_position_anchor = list(point_four_handoff_position)
                    active_route_progress = 0.0
                    active_route_translation_locked = True
                    active_route_position_anchor = list(point_four_handoff_position)
                    # Preserve the endpoint proof across the following yaw.
                    # The live localizer is expected to freeze or reject its
                    # monocular center during this weak-view rotation, but the
                    # aircraft cannot physically translate while only yaw RC
                    # is sent.  A later three-frame absolute departure-heading
                    # check may consume this anchor for one bounded command.
                    verified_endpoint_turn_source_gate = point_four_anchor_gate
                    verified_endpoint_turn_anchor = list(point_four_handoff_position)
                    verified_endpoint_turn_source = point_four_handoff_source
                    rc_summary.setdefault("point4_handoffs", []).append(
                        {
                            "lap": current_lap_number,
                            "instance_id": (
                                point_four_handoff_pose.get("instance_id")
                            ),
                            "source": point_four_handoff_source,
                            "position": list(point_four_handoff_position),
                            "point4_error_map_units": point_four_metric_error,
                        }
                    )
                    publish_progress(
                        {
                            "phase": "point4_handoff_verified",
                            "step_index": idx,
                            "step_title": title,
                            "translation_locked": True,
                            "require_metric_pose": False,
                            "position_anchor": point_four_handoff_position,
                            "point4_handoff_source": point_four_handoff_source,
                            "route_pose_epoch": active_route_pose_epoch,
                            "route_pose_epoch_unix": active_route_pose_epoch_unix,
                            "route_pose_epoch_reason": active_route_pose_epoch_reason,
                            "prior_endpoint_arrival_preserved": (
                                prior_point_four_visual_ready
                            ),
                            "point4_position_error_map_units": point_four_metric_error,
                            "message": (
                                "Point-4 endpoint verified with translation locked; "
                                "the rotation-only 4->1 alignment may begin."
                            ),
                        }
                    )
                if lap_metric_checkpoint_pending and not dynamic_lap_reentry:
                    prior_point_one_arrival = prior_verified_endpoint_arrival_record(
                        executed,
                        segment_start=active_route_segment_start,
                        expected_leg_index=4,
                    )
                    prior_point_one_visual_ready = bool(
                        prior_point_one_arrival is not None
                    )
                    checkpoint_gate = continuity_guarded_pose_gate()
                    checkpoint_pose = (
                        checkpoint_gate.get("pose")
                        if isinstance(checkpoint_gate, dict)
                        else None
                    )
                    checkpoint_position = pose_gate_position(checkpoint_gate)
                    checkpoint_error = horizontal_xz_distance(
                        checkpoint_position,
                        active_route_segment_start,
                    )
                    checkpoint_metric_ready = bool(
                        pose_gate_has_fresh_metric_position(checkpoint_gate)
                        and not pose_gate_rotation_locked(checkpoint_gate)
                        and checkpoint_error is not None
                        and checkpoint_error <= 0.24
                    )
                    checkpoint_visual_ready = taught_endpoint_arrival_verified(
                        checkpoint_pose,
                        expected_leg_index=4,
                    ) or prior_point_one_visual_ready
                    waited_for_checkpoint = not (
                        checkpoint_metric_ready or checkpoint_visual_ready
                    )
                    if waited_for_checkpoint:
                        publish_progress(
                            {
                                "phase": "lap_endpoint_checkpoint",
                                "lap": current_lap_number,
                                "translation_locked": True,
                                "position_anchor": start_position,
                                "metric_pose_ready": checkpoint_metric_ready,
                                "require_metric_pose": False,
                                "point1_error_map_units": checkpoint_error,
                                "message": (
                                    f"Lap {current_lap_number} is holding at Point 1 while ATLAS "
                                    "verifies the previous 4->1 endpoint before the next "
                                    "rotation-only alignment."
                                ),
                            }
                        )
                        checkpoint_gate = wait_for_pose_recovery(
                            "lap_start_endpoint_checkpoint",
                            "waiting for verified Point-1 endpoint localization before the next circle",
                            timeout=8.0,
                            require_translation_safe=True,
                            require_endpoint_verified=True,
                            endpoint_leg_index=4,
                        )
                        if checkpoint_gate is None:
                            break
                        checkpoint_pose = checkpoint_gate.get("pose") or {}
                        checkpoint_position = pose_gate_position(checkpoint_gate)
                        checkpoint_error = horizontal_xz_distance(
                            checkpoint_position,
                            active_route_segment_start,
                        )
                        checkpoint_metric_ready = bool(
                            pose_gate_has_fresh_metric_position(checkpoint_gate)
                            and not pose_gate_rotation_locked(checkpoint_gate)
                            and checkpoint_error is not None
                            and checkpoint_error <= 0.24
                        )
                        checkpoint_visual_ready = taught_endpoint_arrival_verified(
                            checkpoint_pose,
                            expected_leg_index=4,
                        ) or prior_point_one_visual_ready
                    if checkpoint_metric_ready:
                        checkpoint_source = "metric_tsolve"
                    elif checkpoint_visual_ready:
                        # Endpoint vision is constrained to the taught route,
                        # so use the exact shared Point-1 boundary as the yaw
                        # anchor. No translation is authorized by this choice.
                        checkpoint_source = "verified_visual_endpoint"
                        checkpoint_position = list(active_route_segment_start)
                        checkpoint_error = 0.0
                    else:
                        abort_reason = (
                            "lap boundary lost both metric and verified visual "
                            "Point-1 localization; hovering before lap start"
                        )
                        break
                    checkpoint_source_gate = (
                        checkpoint_gate
                        if pose_gate_position(checkpoint_gate) is not None
                        else last_pose_gate
                    )
                    if active_route_pose_epoch_reason != "verified_point1_handoff":
                        active_route_pose_epoch += 1
                        active_route_pose_epoch_unix = time.time()
                        active_route_pose_epoch_reason = "verified_point1_handoff"
                    checkpoint_anchor_gate = verified_route_anchor_pose_gate(
                        checkpoint_source_gate,
                        checkpoint_position,
                        epoch=active_route_pose_epoch,
                        epoch_unix=float(active_route_pose_epoch_unix),
                        reason=active_route_pose_epoch_reason,
                    )
                    gate = checkpoint_anchor_gate
                    last_pose_gate = checkpoint_anchor_gate
                    start_position = list(checkpoint_position)
                    yaw_position_anchor = list(checkpoint_position)
                    # Preserve the independently verified Point-1 checkpoint
                    # across the following yaw. The physical aircraft cannot
                    # translate during yaw-only RC, so the aligned departure
                    # view may consume this exact anchor for one bounded first
                    # command of the new lap. A second command still requires
                    # fresh observed progress through the ordinary pulse gate.
                    verified_endpoint_turn_source_gate = checkpoint_anchor_gate
                    verified_endpoint_turn_anchor = list(checkpoint_position)
                    verified_endpoint_turn_source = checkpoint_source
                    lap_point_one_handoff = True
                    active_route_progress = 0.0
                    active_route_translation_locked = True
                    active_route_position_anchor = list(checkpoint_position)
                    lap_metric_checkpoint_pending = False
                    checkpoint_record = {
                        "lap": current_lap_number,
                        "waited_for_checkpoint": waited_for_checkpoint,
                        "source": checkpoint_source,
                        "point1_error_map_units": checkpoint_error,
                        "instance_id": (checkpoint_gate.get("pose") or {}).get(
                            "instance_id"
                        ),
                        "prior_endpoint_arrival_preserved": (
                            prior_point_one_visual_ready
                        ),
                    }
                    rc_summary["lap_metric_checkpoints"].append(checkpoint_record)
                    publish_progress(
                        {
                            "phase": "lap_checkpoint_verified",
                            "lap": current_lap_number,
                            "translation_locked": True,
                            "position_anchor": checkpoint_position,
                            "checkpoint_source": checkpoint_source,
                            "metric_pose_ready": checkpoint_metric_ready,
                            "require_metric_pose": False,
                            "point1_error_map_units": checkpoint_error,
                            "prior_endpoint_arrival_preserved": (
                                prior_point_one_visual_ready
                            ),
                            "message": (
                                f"Lap {current_lap_number} Point-1 endpoint verified; "
                                "rotation-only heading alignment may begin."
                            ),
                        }
                    )
                turn_direction_override = taught_turn_direction_override(taught_leg)
                guarded_taught_rotation = (
                    taught_turn_requires_recorded_departure_view(taught_leg)
                )
                taught_rotation_yaw_seconds = 0.0
                max_taught_rotation_yaw_seconds = (
                    45.0 if turn_direction_override == "left" else 24.0
                )
                if turn_direction_override == "left":
                    # A forced left turn may be the long route to the same
                    # target heading.  The first live measurement covered
                    # about 110 degrees in 60 seconds, so the full ~258-degree
                    # turn needs roughly 140 seconds at the guarded pulse
                    # rate.  Keep a bounded margin for pose waits and enough
                    # time afterward to travel to point 4. Translation remains
                    # locked until aligned.
                    alignment_progress_deadline = max(
                        alignment_progress_deadline,
                        motion_clock() + 190.0,
                    )
                    segment_safety_deadline = max(
                        segment_safety_deadline,
                        motion_clock() + 220.0,
                    )
                rotation_alignment_ready_at: float | None = None
                rotation_heading_stable_frames = 0
                rotation_heading_last_instance_id: str | None = None
                required_rotation_heading_stable_frames = 3
                # Once an absolute departure image has been acquired, retain
                # that observation and its optical-to-room rebase. Optical
                # flow may bridge only the last six degrees for 2.5 seconds;
                # it never becomes an independent absolute turn source. This
                # removes the yaw/neutral fighting seen at Point 4 while still
                # preventing the old optical-vs-TSolve direction oscillation.
                recorded_heading_acquired = False
                last_recorded_heading_observation: dict[str, Any] | None = None
                max_rotation_position_recovery_seconds = 25.0
                if (
                    calibrated_heading_offset_rad is None
                    and pose_gate_rotation_locked(gate)
                ):
                    # The first connected patrol can inherit a rotation-only
                    # pose from the startup/model-heading phase. Release that
                    # stale turn anchor and wait online for a translation-safe
                    # observation before establishing the initial heading.
                    active_route_translation_locked = False
                    active_route_position_anchor = None
                    publish_progress(
                        {
                            "phase": "initial_heading_release_rotation_lock",
                            "step_index": idx,
                            "step_title": title,
                            "translation_locked": False,
                            "position_anchor": None,
                            "rotation_release_requested": True,
                            "message": (
                                "Initial pose is still rotation-locked; hovering while "
                                "ATLAS releases it and relocalizes before heading verification."
                            ),
                        }
                    )
                    unlocked_gate = wait_for_pose_recovery(
                        "initial_heading_translation_unlock",
                        "waiting for a translation-safe pose before heading verification",
                        require_translation_safe=True,
                    )
                    if unlocked_gate is None:
                        break
                    gate = unlocked_gate
                    unlocked_position = pose_gate_position(gate)
                    start_position = (
                        list(segment_start)
                        if precise_arrival and segment_start is not None
                        else unlocked_position
                    )
                    yaw_position_anchor = start_position
                    active_route_progress = (
                        0.0
                        if precise_arrival and segment_start is not None
                        else route_segment_progress_xz(
                            start_position,
                            active_route_segment_start,
                            active_route_segment_end,
                        )
                    )
                if allow_axis_auto_calibration and body_axes["forward"] is None and body_axes["lateral"] is None:
                    rc_summary["adaptive_axis"]["mode"] = "calibrated_body_axes"
                    if not calibrate_body_axes(gate):
                        abort_reason = "could not calibrate DJI body motion axes from TSolve pose feedback"
                        break
                if calibrated_heading_offset_rad is None:
                    if operator_heading_calibrated and not require_physical_forward_probe:
                        calibrated_heading_offset_rad = math.radians(initial_body_heading_offset_deg)
                        rc_summary["adaptive_axis"]["mode"] = "operator_heading_seed_yaw_verified"
                        rc_summary["adaptive_axis"]["camera_to_body_heading_offset_deg"] = round(
                            initial_body_heading_offset_deg,
                            3,
                        )
                        publish_progress(
                            {
                                "phase": "heading_calibration",
                                "translation_locked": True,
                                "position_anchor": yaw_position_anchor,
                                "body_forward_gain": 0.0,
                                "body_lateral_gain": 0.0,
                                "message": (
                                    "Using the operator-aligned COLMAP heading without a pre-route "
                                    "translation probe; the first yaw pulse will verify DJI direction."
                                ),
                                "camera_to_body_heading_offset_deg": initial_body_heading_offset_deg,
                            }
                        )
                    elif calibrate_forward_heading(gate):
                        pass
                    elif abs(heading_trim_deg) > 1e-9:
                        calibrated_heading_offset_rad = 0.0
                        rc_summary["adaptive_axis"]["mode"] = "manual_heading_trim_fallback"
                        publish_progress(
                            {
                                "phase": "heading_calibration",
                                "message": (
                                    "Automatic body-forward calibration was unavailable; "
                                    f"using operator heading trim {heading_trim_deg:+.1f} degrees."
                                ),
                                "operator_heading_trim_deg": heading_trim_deg,
                            }
                        )
                    else:
                        abort_reason = (
                            "could not measure DJI body-forward heading from fresh TSolve poses; "
                            "refusing patrol translation"
                        )
                        break
                active_route_translation_locked = True
                active_route_position_anchor = yaw_position_anchor

                while True:
                    now = motion_clock()
                    if now >= absolute_segment_deadline:
                        break
                    rotation_wait_active = (
                        rotation_alignment_ready_at is not None
                        and now - rotation_alignment_ready_at < max_rotation_position_recovery_seconds
                    )
                    blind_turn_active = (
                        not travel_started
                        and (
                            (blind_yaw_seconds > 0.0 and blind_yaw_seconds < max_blind_yaw_seconds)
                            or rotation_wait_active
                        )
                    )
                    if (
                        rotation_alignment_ready_at is not None
                        and not rotation_wait_active
                        and not travel_started
                    ):
                        abort_reason = (
                            "visual turn reached its taught heading but TSolve position did not recover; "
                            "hovering without forward movement"
                        )
                        break
                    if now >= segment_safety_deadline and (
                        travel_started
                        or (now >= alignment_progress_deadline and not blind_turn_active)
                    ):
                        break
                    if (
                        not travel_started
                        and now >= alignment_progress_deadline
                        and not blind_turn_active
                    ):
                        break
                    if travel_started and (
                        forward_command_seconds >= forward_budget_seconds
                        or travel_yaw_command_seconds >= yaw_correction_budget_seconds
                    ):
                        break
                    # During an established in-place alignment turn, a weak
                    # visual sector may temporarily provide no accepted pose.
                    # Continue only the already-established yaw direction for
                    # a short bounded interval so the camera can reach a
                    # feature-rich view. Translation remains locked.
                    gate_attempt = continuity_guarded_pose_gate()
                    # A recent-held fallback contains an old trusted position,
                    # not a new position observation.  It is useful for
                    # freezing the aircraft's navigation origin, but it must
                    # not take a taught in-place turn out of rotation-only
                    # mode.  Otherwise the following post-pulse pose wait can
                    # enter the generic blocking recovery loop even while a
                    # fresh optical heading remains available.
                    rotation_pose_locked = pose_gate_rotation_locked(gate_attempt)
                    rotation_position_untrusted = (
                        not gate_attempt.get("ok")
                        or bool(gate_attempt.get("recent_hold_fallback"))
                        or rotation_pose_locked
                    )
                    confirmed_rotation_angle: float | None = None
                    confirmed_rotation_gate: dict[str, Any] | None = None
                    rotation_observation = (
                        latest_rotation_only_heading(pose_stream_path, pose_max_age)
                        if (
                            taught_leg is not None
                            and not travel_started
                            and (
                                point_three_handoff
                                or guarded_taught_rotation
                                or rotation_position_untrusted
                            )
                        )
                        else {"ok": False}
                    )
                    recorded_heading_observation = (
                        latest_recorded_departure_heading(
                            pose_stream_path,
                            pose_max_age,
                            map_id=str(mission.get("map_id") or ""),
                            patrol_id=str(mission.get("patrol_id") or ""),
                            baseline_replay_id=str(
                                mission.get("baseline_replay_id") or ""
                            ),
                            expected_leg_index=endpoint_leg_index,
                        )
                        if (
                            (guarded_taught_rotation or point_three_handoff)
                            and not travel_started
                        )
                        else {"ok": False}
                    )
                    point_three_recorded_stop_ready = False
                    point_three_recorded_recovery_ready = False
                    if point_three_handoff and recorded_heading_observation.get("ok"):
                        try:
                            point_three_recorded_correction_deg = float(
                                recorded_heading_observation.get("correction_deg")
                            )
                            point_three_recorded_stop_ready = (
                                abs(point_three_recorded_correction_deg) <= 6.0
                            )
                            # A fresh absolute match to this exact departure
                            # view may steer only a small final remainder. This
                            # heading-only evidence never authorizes position
                            # or forward movement by itself. The coarse turn no
                            # longer has a fixed rate-derived budget: the live
                            # DJI response proved too variable for that value.
                            point_three_recorded_recovery_budget = (
                                point_three_recorded_recovery_effort_limit_seconds(
                                    point_three_recorded_correction_deg
                                )
                            )
                            point_three_recorded_recovery_ready = bool(
                                point_three_recorded_recovery_budget > 0.0
                            )
                        except (TypeError, ValueError):
                            point_three_recorded_stop_ready = False
                            point_three_recorded_recovery_ready = False
                    if (
                        point_three_handoff
                        and rotation_observation.get("unsafe_for_yaw")
                        and not point_three_recorded_stop_ready
                        and not point_three_recorded_recovery_ready
                    ):
                        # A capture/catch-up gap invalidates only incremental
                        # optical yaw.  Never convert the last command into a
                        # blind continuation.  A current absolute ORB match may
                        # still stop the turn because it does not integrate the
                        # missing frames.
                        last_alignment_yaw_rc = 0.0
                        publish_progress(
                            {
                                "phase": "point3_yaw_feedback_timing_hold",
                                "step_index": idx,
                                "step_title": title,
                                "taught_leg": [3, 4],
                                "translation_locked": True,
                                "position_anchor": yaw_position_anchor,
                                "frame_gap_seconds": rotation_observation.get(
                                    "frame_gap_seconds"
                                ),
                                "max_frame_gap_seconds": rotation_observation.get(
                                    "max_frame_gap_seconds"
                                ),
                                "message": (
                                    "Point-3 yaw feedback crossed a camera-frame gap; "
                                    "hovering until an adjacent frame pair or the "
                                    "absolute recorded departure view is available."
                                ),
                            }
                        )
                        neutral_hover(drone, 0.12)
                        continue
                    if (
                        rotation_observation.get("ok")
                        or recorded_heading_observation.get("ok")
                    ):
                        frozen_position = (
                            yaw_position_anchor
                            if yaw_position_anchor is not None
                            else pose_gate_position(last_pose_gate)
                        )
                        visual_direction = normalize_xz(target_direction_xz(frozen_position, target))
                        optical_angle = (
                            signed_angle_xz(
                                rotation_observation.get("heading"), visual_direction
                            )
                            if rotation_observation.get("ok")
                            else None
                        )
                        alignment_angle = optical_angle
                        alignment_tolerance = math.radians(
                            2.0
                            if point_four_handoff
                            else (4.0 if point_three_handoff else 10.0)
                        )
                        alignment_source = "optical_flow_yaw"
                        alignment_tracks = rotation_observation.get("tracks")
                        observation_instance_id = str(
                            rotation_observation.get("instance_id") or ""
                        )
                        point_three_recorded_stop = bool(
                            point_three_recorded_stop_ready
                        )
                        if (
                            point_three_recorded_stop
                            or point_three_recorded_recovery_ready
                        ):
                            # Optical flow remains the steering source through
                            # the turn.  The high-inlier current-frame match is
                            # only an absolute stop authority near the recorded
                            # Point-3 departure view, preventing delayed optical
                            # yaw from commanding another quarter turn.
                            alignment_angle = math.radians(
                                float(
                                    recorded_heading_observation.get(
                                        "correction_deg"
                                    )
                                )
                            )
                            alignment_tolerance = math.radians(4.0)
                            alignment_source = (
                                "recorded_point3_departure_stop"
                                if point_three_recorded_stop
                                else "recorded_point3_departure_bounded_recovery"
                            )
                            alignment_tracks = recorded_heading_observation.get(
                                "inliers"
                            )
                            observation_instance_id = str(
                                recorded_heading_observation.get("instance_id")
                                or ""
                            )
                            rotation_position_untrusted = True
                        if guarded_taught_rotation:
                            # The optical chain is only a coarse turn guide at
                            # the two weak patrol corners. A direct match
                            # against this exact leg's recorded departure
                            # images is its absolute final-heading reference.
                            if recorded_heading_observation.get("ok"):
                                recorded_heading_acquired = True
                                optical_rebase = (
                                    signed_angle_xz(
                                        rotation_observation.get("heading"),
                                        recorded_heading_observation.get(
                                            "current_heading"
                                        ),
                                    )
                                    if rotation_observation.get("ok")
                                    else None
                                )
                                optical_rebase_ready = bool(
                                    optical_rebase is not None
                                    and abs(optical_rebase) <= math.radians(45.0)
                                )
                                if optical_rebase_ready:
                                    route_optical_heading_bias_rad = float(
                                        optical_rebase
                                    )
                                    last_recorded_heading_observation = dict(
                                        recorded_heading_observation
                                    )
                                alignment_angle = math.radians(
                                    float(
                                        recorded_heading_observation.get(
                                            "correction_deg"
                                        )
                                    )
                                )
                                alignment_tolerance = math.radians(4.0)
                                alignment_source = "recorded_patrol_departure_view"
                                alignment_tracks = recorded_heading_observation.get(
                                    "inliers"
                                )
                                observation_instance_id = str(
                                    recorded_heading_observation.get("instance_id")
                                    or ""
                                )
                                # This is a current, absolute, leg-specific ORB
                                # observation, so it may steer the remaining
                                # in-place yaw even when incremental optical
                                # flow is missing or uses a disagreeing basis.
                                # Optical rebasing is needed only to propagate
                                # the heading across a later ORB gap. Forward
                                # movement remains locked until three distinct
                                # verified ORB observations are within the
                                # four-degree departure gate below.
                            elif recorded_heading_acquired:
                                optical_fine_handoff = (
                                    recorded_heading_optical_fine_handoff(
                                        last_recorded_heading_observation,
                                        rotation_observation,
                                        optical_heading_bias_rad=(
                                            route_optical_heading_bias_rad
                                        ),
                                    )
                                )
                                if optical_fine_handoff is not None:
                                    # Keep the absolute recorded target; only
                                    # propagate the current camera heading with
                                    # short, strongly tracked optical yaw. This
                                    # avoids alternating one yaw pulse with a
                                    # long neutral wait when inliers fluctuate
                                    # around the threshold near the target.
                                    recorded_heading_observation = (
                                        optical_fine_handoff
                                    )
                                    alignment_angle = math.radians(
                                        float(
                                            optical_fine_handoff[
                                                "correction_deg"
                                            ]
                                        )
                                    )
                                    alignment_tolerance = math.radians(4.0)
                                    alignment_source = (
                                        "recorded_view_optical_fine_handoff"
                                    )
                                    alignment_tracks = optical_fine_handoff.get(
                                        "optical_tracks"
                                    )
                                    observation_instance_id = str(
                                        optical_fine_handoff.get("instance_id")
                                        or ""
                                    )
                                else:
                                    # The absolute anchor is too old, too far
                                    # from target, or the optical bridge is no
                                    # longer bounded. Hold neutral and reacquire
                                    # rather than reversing or overshooting.
                                    alignment_angle = None
                                    alignment_tolerance = None
                                    alignment_source = (
                                        "recorded_view_reacquisition_hold"
                                    )
                            else:
                                # Do not let an optical estimate inside its
                                # old +/-10 degree window complete this guarded
                                # turn. Keep making small search pulses in the
                                # planned turn direction until the recorded
                                # departure view is visible and can provide an
                                # absolute error.
                                alignment_tolerance = None
                                if alignment_angle is not None and abs(
                                    alignment_angle
                                ) <= math.radians(10.0):
                                    alignment_angle = math.radians(10.1)
                                alignment_source = "optical_flow_coarse_search"
                        heading_aligned = bool(
                            alignment_angle is not None
                            and alignment_tolerance is not None
                            and abs(alignment_angle) <= alignment_tolerance
                        )
                        if heading_aligned:
                            forward_alignment_locked = True
                            last_alignment_yaw_rc = 0.0
                            localizer_consensus_count = int(
                                recorded_heading_observation.get(
                                    "localizer_heading_consensus_count"
                                )
                                or 0
                            )
                            if (
                                recorded_heading_observation.get(
                                    "localizer_heading_consensus_verified"
                                )
                                is True
                                and localizer_consensus_count >= 3
                            ):
                                rotation_heading_stable_frames = max(
                                    rotation_heading_stable_frames,
                                    localizer_consensus_count,
                                )
                                rotation_heading_last_instance_id = (
                                    observation_instance_id
                                    or rotation_heading_last_instance_id
                                )
                            elif (
                                observation_instance_id
                                and observation_instance_id != rotation_heading_last_instance_id
                            ):
                                rotation_heading_last_instance_id = observation_instance_id
                                rotation_heading_stable_frames += 1
                            if (
                                rotation_heading_stable_frames
                                >= required_rotation_heading_stable_frames
                                and rotation_alignment_ready_at is None
                            ):
                                rotation_alignment_ready_at = now
                            publish_progress(
                                {
                                    "phase": "taught_turn_heading_ready",
                                    "step_index": idx,
                                    "step_title": title,
                                    "taught_leg": [taught_leg.get("from_point"), taught_leg.get("to_point")],
                                    "heading_error_deg": math.degrees(alignment_angle),
                                    "heading_source": alignment_source,
                                    "rotation_tracks": alignment_tracks,
                                    "recorded_heading_anchor": recorded_heading_observation.get("anchor"),
                                    "optical_heading_bias_deg": (
                                        math.degrees(route_optical_heading_bias_rad)
                                        if route_optical_heading_bias_rad is not None
                                        else None
                                    ),
                                    "translation_locked": True,
                                    "position_anchor": yaw_position_anchor,
                                    "stable_heading_frames": rotation_heading_stable_frames,
                                    "required_stable_heading_frames": required_rotation_heading_stable_frames,
                                    "message": (
                                        (
                                            "Recorded departure view is aligned with the taught patrol leg; "
                                            if guarded_taught_rotation
                                            else "Visual turn is aligned with the taught patrol leg; "
                                        )
                                        +
                                        f"confirming steady heading "
                                        f"({rotation_heading_stable_frames}/"
                                        f"{required_rotation_heading_stable_frames}) before "
                                        "waiting for a fresh TSolve position."
                                    ),
                                }
                            )
                            if rotation_alignment_ready_at is None:
                                neutral_hover(drone, 0.12)
                                continue
                            # The visual Point-1 checkpoint can authorize the
                            # yaw-only alignment, but it cannot prove that the
                            # metric tracker has left its Point-4 state. Force
                            # the repeated-lap departure through a newest-frame
                            # Point-1 metric rebootstrap before horizontal RC.
                            if lap_point_one_handoff:
                                rotation_position_untrusted = True
                            if rotation_position_untrusted:
                                if lap_point_one_handoff:
                                    endpoint_departure_gate = None
                                elif point_three_recorded_stop:
                                    endpoint_departure_gate = (
                                        verified_recorded_point_three_departure_gate(
                                            verified_endpoint_turn_source_gate,
                                            position_anchor=(
                                                verified_endpoint_turn_anchor
                                            ),
                                            heading_observation=(
                                                recorded_heading_observation
                                            ),
                                            endpoint_handoff_verified=bool(
                                                verified_endpoint_turn_anchor
                                                is not None
                                            ),
                                            endpoint_handoff_source=(
                                                verified_endpoint_turn_source
                                            ),
                                            stable_heading_frames=(
                                                rotation_heading_stable_frames
                                            ),
                                            required_stable_heading_frames=(
                                                required_rotation_heading_stable_frames
                                            ),
                                        )
                                    )
                                elif guarded_taught_rotation:
                                    endpoint_departure_gate = (
                                        verified_endpoint_turn_departure_gate(
                                            verified_endpoint_turn_source_gate,
                                            position_anchor=(
                                                verified_endpoint_turn_anchor
                                            ),
                                            heading_observation=(
                                                recorded_heading_observation
                                            ),
                                            expected_leg_index=(
                                                endpoint_leg_index
                                            ),
                                            endpoint_handoff_verified=bool(
                                                (
                                                    point_four_handoff
                                                    or lap_point_one_handoff
                                                )
                                                and verified_endpoint_turn_anchor
                                                is not None
                                            ),
                                            endpoint_handoff_source=(
                                                verified_endpoint_turn_source
                                            ),
                                            stable_heading_frames=(
                                                rotation_heading_stable_frames
                                            ),
                                            required_stable_heading_frames=(
                                                required_rotation_heading_stable_frames
                                            ),
                                        )
                                    )
                                elif point_three_handoff or point_four_handoff:
                                    departure_leg_index = (
                                        4 if point_four_handoff else 3
                                    )
                                    allowed_handoff_sources = (
                                        {"verified_visual_endpoint", "metric_tsolve"}
                                        if point_four_handoff
                                        else {"verified_visual_endpoint", "metric_tsolve"}
                                    )
                                    endpoint_departure_gate = (
                                        verified_optical_endpoint_turn_departure_gate(
                                            verified_endpoint_turn_source_gate,
                                            position_anchor=(
                                                verified_endpoint_turn_anchor
                                            ),
                                            heading_observation=(
                                                rotation_observation
                                            ),
                                            heading_error_rad=alignment_angle,
                                            expected_leg_index=departure_leg_index,
                                            endpoint_handoff_verified=bool(
                                                verified_endpoint_turn_anchor
                                                is not None
                                                and verified_endpoint_turn_source
                                                in allowed_handoff_sources
                                            ),
                                            stable_heading_frames=(
                                                rotation_heading_stable_frames
                                            ),
                                            required_stable_heading_frames=(
                                                required_rotation_heading_stable_frames
                                            ),
                                            endpoint_handoff_source=(
                                                verified_endpoint_turn_source
                                            ),
                                        )
                                    )
                                else:
                                    endpoint_departure_gate = None
                                if endpoint_departure_gate is not None:
                                    # The waypoint was verified before the yaw
                                    # and the new leg's heading is now steady.
                                    # Requiring another translation solution
                                    # here deadlocks in a weak view.  Permit one
                                    # normal low-stick pulse, then require fresh
                                    # observed progress before any next command.
                                    confirmed_rotation_gate = endpoint_departure_gate
                                    publish_progress(
                                        {
                                            "phase": "verified_endpoint_turn_departure_ready",
                                            "step_index": idx,
                                            "step_title": title,
                                            "heading_error_deg": math.degrees(
                                                alignment_angle
                                            ),
                                            "heading_source": alignment_source,
                                            "rotation_tracks": alignment_tracks,
                                            "translation_locked": True,
                                            "position_anchor": (
                                                verified_endpoint_turn_anchor
                                            ),
                                            "bounded_departure_commands": 1,
                                            "require_observed_progress_after_command": True,
                                            "message": (
                                                "Waypoint position and the current departure "
                                                "heading are independently verified; "
                                                "one bounded low-stick departure is ready."
                                            ),
                                        }
                                    )
                                else:
                                    # The turn is complete and the aircraft is
                                    # neutrally hovering. Open the route-aware
                                    # visual recovery path even when TSolve is
                                    # currently rejected or held. Requiring a
                                    # healthy non-held TSolve pose before
                                    # setting this flag creates a circular wait:
                                    # the baseline matcher stays disabled while
                                    # the bridge waits for that matcher to
                                    # recover the position.
                                    publish_progress(
                                        {
                                            "phase": "taught_turn_wait_fresh_position",
                                            "step_index": idx,
                                            "step_title": title,
                                            "heading_error_deg": math.degrees(
                                                alignment_angle
                                            ),
                                            "heading_source": alignment_source,
                                            "rotation_tracks": alignment_tracks,
                                            "translation_locked": True,
                                            "route_visual_recovery_allowed": True,
                                            "position_anchor": yaw_position_anchor,
                                            "rotation_release_requested": True,
                                            "body_forward_gain": 0.0,
                                            "body_lateral_gain": 0.0,
                                            "message": (
                                                "Turn heading is steady; hovering at the "
                                                "waypoint while the recorded patrol baseline "
                                                "recovers a translation-safe position. No "
                                                "forward or lateral command is allowed."
                                            ),
                                        }
                                    )
                                    # Release the command-side route anchor while
                                    # issuing neutral hover only. Keeping
                                    # `active_route_translation_locked` true
                                    # made the controller reject the first valid
                                    # visual recovery (0.18 m) against the 0.16 m
                                    # turn drift limit forever. The recovery
                                    # helper clears only localization anchors;
                                    # physical forward/lateral RC remains locked
                                    # until the verified translation-safe gate
                                    # returns.
                                    recovered_gate = wait_for_pose_recovery(
                                        (
                                            "taught_turn_recorded_heading_recovery"
                                            if guarded_taught_rotation
                                            else "taught_turn_position_recovery"
                                        ),
                                        "waiting for the verified recorded route position after heading alignment",
                                        timeout=max_rotation_position_recovery_seconds,
                                        require_translation_safe=True,
                                        require_metric_pose=lap_point_one_handoff,
                                        lap_start_metric_rebootstrap=(
                                            lap_point_one_handoff
                                        ),
                                    )
                                    if recovered_gate is None:
                                        break
                                    if lap_point_one_handoff:
                                        recovered_position = pose_gate_position(
                                            recovered_gate
                                        )
                                        saved_point_one = (
                                            list(active_route_segment_start)
                                            if active_route_segment_start is not None
                                            else None
                                        )
                                        recovered_point_one_error = (
                                            horizontal_xz_distance(
                                                recovered_position,
                                                saved_point_one,
                                            )
                                        )
                                        recovered_route_progress = (
                                            route_segment_progress_xz(
                                                recovered_position,
                                                saved_point_one,
                                                target,
                                            )
                                        )
                                        recovered_cross_track = (
                                            route_line_cross_track_xz(
                                                recovered_position,
                                                saved_point_one,
                                                target,
                                            )
                                        )
                                        saved_first_leg_length = (
                                            horizontal_xz_distance(
                                                saved_point_one,
                                                target,
                                            )
                                        )
                                        recovered_behind_distance = (
                                            max(
                                                0.0,
                                                -float(recovered_route_progress)
                                                * float(saved_first_leg_length),
                                            )
                                            if (
                                                recovered_route_progress is not None
                                                and saved_first_leg_length is not None
                                            )
                                            else None
                                        )
                                        near_saved_point_one = bool(
                                            recovered_point_one_error is not None
                                            and recovered_point_one_error <= 0.24
                                        )
                                        bounded_behind_point_one = bool(
                                            recovered_position is not None
                                            and recovered_route_progress is not None
                                            and -0.60 <= recovered_route_progress < 0.0
                                            and recovered_cross_track is not None
                                            and recovered_cross_track <= 0.30
                                            and recovered_behind_distance is not None
                                            and recovered_behind_distance <= 1.55
                                        )
                                        if not (
                                            near_saved_point_one
                                            or bounded_behind_point_one
                                        ):
                                            abort_reason = (
                                                "fresh full-map localization places the "
                                                "aircraft outside the safe Point-1 re-entry "
                                                "corridor "
                                                f"({recovered_point_one_error:.3f} m); "
                                                "hovering instead of starting the next lap"
                                                if recovered_point_one_error is not None
                                                else (
                                                    "fresh full-map localization has no "
                                                    "valid Point-1 position; hovering before "
                                                    "the next lap"
                                                )
                                            )
                                            publish_progress(
                                                {
                                                    "phase": "lap_start_metric_mismatch",
                                                    "translation_locked": True,
                                                    "require_metric_pose": True,
                                                    "position_anchor": recovered_position,
                                                    "point1_error_map_units": (
                                                        recovered_point_one_error
                                                    ),
                                                    "route_progress": (
                                                        recovered_route_progress
                                                    ),
                                                    "route_cross_track_map_units": (
                                                        recovered_cross_track
                                                    ),
                                                    "behind_point1_map_units": (
                                                        recovered_behind_distance
                                                    ),
                                                    "message": abort_reason,
                                                }
                                            )
                                            neutral_hover(drone, 0.25)
                                            break
                                        if bounded_behind_point_one:
                                            # The full-map TSolve result says the
                                            # aircraft physically returned behind
                                            # saved Point 1, toward takeoff, rather
                                            # than reaching the old route marker.
                                            # Extend only this lap's first leg to
                                            # the measured position. The ordinary
                                            # closed-loop Point-1->2 controller will
                                            # now fly through Point 1 smoothly; no
                                            # pose snap and no open-loop correction
                                            # pulse is introduced.
                                            if not enforce_patrol_geofence(
                                                recovered_gate,
                                                "lap-start metric re-entry",
                                            ):
                                                break
                                            start_position = list(recovered_position)
                                            segment_start = list(recovered_position)
                                            yaw_position_anchor = list(
                                                recovered_position
                                            )
                                            active_route_segment_start = list(
                                                recovered_position
                                            )
                                            active_route_progress = 0.0
                                            active_route_command_progress_ceiling = 0.0
                                            active_route_command_sequence = 0
                                            active_route_translation_locked = True
                                            active_route_position_anchor = list(
                                                recovered_position
                                            )
                                            initial_distance = horizontal_xz_distance(
                                                start_position,
                                                target,
                                            )
                                            final_distance = initial_distance
                                            closest_distance = (
                                                initial_distance
                                                if initial_distance is not None
                                                else float("inf")
                                            )
                                            route_visual_reconciliation_state.clear()
                                            route_metric_reconciliation_state.clear()
                                            rc_summary.setdefault(
                                                "lap_start_metric_reentries", []
                                            ).append(
                                                {
                                                    "lap": current_lap_number,
                                                    "position": list(
                                                        recovered_position
                                                    ),
                                                    "saved_point1": saved_point_one,
                                                    "point1_error_map_units": (
                                                        recovered_point_one_error
                                                    ),
                                                    "route_progress": (
                                                        recovered_route_progress
                                                    ),
                                                    "route_cross_track_map_units": (
                                                        recovered_cross_track
                                                    ),
                                                    "behind_point1_map_units": (
                                                        recovered_behind_distance
                                                    ),
                                                }
                                            )
                                            publish_progress(
                                                {
                                                    "phase": "lap_start_metric_reentry",
                                                    "lap": current_lap_number,
                                                    "translation_locked": True,
                                                    "require_metric_pose": True,
                                                    "position_anchor": list(
                                                        recovered_position
                                                    ),
                                                    "saved_point1": saved_point_one,
                                                    "point1_error_map_units": (
                                                        recovered_point_one_error
                                                    ),
                                                    "route_progress": (
                                                        recovered_route_progress
                                                    ),
                                                    "route_cross_track_map_units": (
                                                        recovered_cross_track
                                                    ),
                                                    "behind_point1_map_units": (
                                                        recovered_behind_distance
                                                    ),
                                                    "message": (
                                                        "Fresh full-map TSolve places the "
                                                        "aircraft behind Point 1 on the "
                                                        "outgoing centerline; the next lap "
                                                        "will re-enter smoothly through Point "
                                                        "1 toward Point 2 without snapping."
                                                    ),
                                                }
                                            )
                                    confirmed_rotation_gate = recovered_gate
                            else:
                                confirmed_rotation_gate = gate_attempt
                            confirmed_rotation_angle = alignment_angle
                        elif alignment_angle is not None and taught_rotation_yaw_seconds < max_taught_rotation_yaw_seconds:
                            rotation_heading_stable_frames = 0
                            rotation_heading_last_instance_id = None
                            rotation_alignment_ready_at = None
                            last_measured_alignment_error_rad = alignment_angle
                            if (
                                observation_instance_id
                                and observation_instance_id
                                == last_yaw_command_observation_id
                            ):
                                publish_progress(
                                    {
                                        "phase": "taught_rotation_wait_new_heading",
                                        "step_index": idx,
                                        "step_title": title,
                                        "heading_error_deg": math.degrees(
                                            alignment_angle
                                        ),
                                        "heading_source": alignment_source,
                                        "rotation_tracks": alignment_tracks,
                                        "translation_locked": True,
                                        "position_anchor": yaw_position_anchor,
                                        "message": (
                                            "Holding neutral until a new heading frame arrives; "
                                            "the same optical observation cannot authorize "
                                            "another yaw pulse."
                                        ),
                                    }
                                )
                                neutral_hover(drone, 0.12)
                                continue
                            yaw_scale = max(0.45, min(1.0, abs(alignment_angle) / math.radians(70.0)))
                            fine_recorded_alignment = alignment_source in {
                                "recorded_patrol_departure_view",
                                "recorded_point3_departure_bounded_recovery",
                            }
                            fine_point_three_alignment = bool(
                                point_three_handoff
                                and abs(alignment_angle) <= math.radians(15.0)
                            )
                            taught_yaw_rc = (
                                yaw_sign
                                * max_yaw_rc
                                * yaw_direction_for_angle(
                                    alignment_angle,
                                    (
                                        None
                                        if fine_recorded_alignment
                                        or fine_point_three_alignment
                                        else turn_direction_override
                                    ),
                                )
                                * yaw_scale
                            )
                            last_alignment_yaw_rc = taught_yaw_rc
                            if point_three_handoff:
                                if point_three_planned_turn_degrees is None:
                                    point_three_planned_turn_degrees = abs(
                                        math.degrees(alignment_angle)
                                    )
                                using_recorded_recovery = bool(
                                    alignment_source
                                    == "recorded_point3_departure_bounded_recovery"
                                )
                                if using_recorded_recovery:
                                    if math.isinf(
                                        point_three_recorded_recovery_effort_limit
                                    ):
                                        point_three_recorded_recovery_effort_limit = (
                                            point_three_recorded_recovery_effort_limit_seconds(
                                                math.degrees(alignment_angle)
                                            )
                                        )
                                    projected_recorded_recovery_effort = (
                                        point_three_recorded_recovery_effort_seconds
                                        + normalized_yaw_command_effort(
                                            taught_yaw_rc,
                                            max_yaw_rc,
                                            pulse_seconds,
                                        )
                                    )
                                else:
                                    projected_recorded_recovery_effort = 0.0
                                recorded_recovery_limit_blocks = bool(
                                    using_recorded_recovery
                                    and projected_recorded_recovery_effort
                                    > point_three_recorded_recovery_effort_limit
                                )
                                if recorded_recovery_limit_blocks:
                                    last_alignment_yaw_rc = 0.0
                                    publish_progress(
                                        {
                                            "phase": "point3_recorded_recovery_limit_hold",
                                            "step_index": idx,
                                            "step_title": title,
                                            "taught_leg": [3, 4],
                                            "heading_error_deg": math.degrees(
                                                alignment_angle
                                            ),
                                            "heading_source": alignment_source,
                                            "translation_locked": True,
                                            "position_anchor": yaw_position_anchor,
                                            "planned_turn_degrees": (
                                                point_three_planned_turn_degrees
                                            ),
                                            "yaw_effort_seconds": (
                                                point_three_yaw_effort_seconds
                                            ),
                                            "recorded_recovery_effort_seconds": (
                                                point_three_recorded_recovery_effort_seconds
                                            ),
                                            "recorded_recovery_effort_limit_seconds": (
                                                point_three_recorded_recovery_effort_limit
                                                if not math.isinf(
                                                    point_three_recorded_recovery_effort_limit
                                                )
                                                else None
                                            ),
                                            "message": (
                                                "Point-3 absolute recorded-view recovery "
                                                "reached its bounded fine-yaw limit; "
                                                "hovering without further rotation until "
                                                "a current absolute/contiguous heading confirms "
                                                "the Point-4 direction."
                                            ),
                                        }
                                    )
                                    neutral_hover(drone, 0.12)
                                    continue
                            publish_progress(
                                {
                                    "phase": "taught_rotation_only_recovery",
                                    "step_index": idx,
                                    "step_title": title,
                                    "taught_leg": [taught_leg.get("from_point"), taught_leg.get("to_point")],
                                    "heading_error_deg": math.degrees(alignment_angle),
                                    "heading_source": alignment_source,
                                    "rotation_tracks": alignment_tracks,
                                    "recorded_heading_reason": recorded_heading_observation.get("reason"),
                                    "translation_locked": True,
                                    "position_anchor": yaw_position_anchor,
                                    "rotation_yaw_seconds": taught_rotation_yaw_seconds,
                                    "max_rotation_yaw_seconds": max_taught_rotation_yaw_seconds,
                                    "turn_direction_override": turn_direction_override,
                                    "message": (
                                        "Continuing the guarded turn until the live camera matches "
                                        "the recorded waypoint departure view. Forward and lateral are locked."
                                    ),
                                }
                            )
                            if not enforce_patrol_geofence(last_pose_gate, "taught rotation recovery"):
                                break
                            sent = execute_rc_pulse(
                                drone,
                                yaw=taught_yaw_rc,
                                lr=0.0,
                                bf=0.0,
                                du=0.0,
                                seconds=pulse_seconds,
                            )
                            record_pulse("yaw", sent)
                            executed_pulses += 1
                            pulse_count += 1
                            yaw_pulse_count += 1
                            taught_rotation_yaw_seconds += pulse_seconds
                            if point_three_handoff:
                                actual_yaw_effort = normalized_yaw_command_effort(
                                    taught_yaw_rc,
                                    max_yaw_rc,
                                    float(
                                        sent.get("actual_seconds")
                                        or sent.get("seconds")
                                        or pulse_seconds
                                    ),
                                )
                                point_three_yaw_effort_seconds += actual_yaw_effort
                                if using_recorded_recovery:
                                    point_three_recorded_recovery_effort_seconds += (
                                        actual_yaw_effort
                                    )
                            last_yaw_command_observation_id = (
                                observation_instance_id or None
                            )
                            time.sleep(0.08)
                            continue
                        if taught_rotation_yaw_seconds >= max_taught_rotation_yaw_seconds:
                            abort_reason = (
                                "taught rotation-only recovery reached its 24 second safety limit; "
                                "hovering without forward movement"
                            )
                            break
                    if (
                        guarded_taught_rotation
                        and not travel_started
                        and confirmed_rotation_gate is None
                    ):
                        # Audited patrol turns may translate only after their
                        # recorded departure view has been observed steadily.
                        # If both the absolute image matcher and the coarse
                        # optical guide are temporarily unavailable, fail
                        # closed here.  Falling through to the generic pose
                        # heading gate previously allowed a false Point-3
                        # optical turn to begin translation.
                        publish_progress(
                            {
                                "phase": "taught_turn_wait_recorded_departure_view",
                                "step_index": idx,
                                "step_title": title,
                                "taught_leg": [
                                    taught_leg.get("from_point"),
                                    taught_leg.get("to_point"),
                                ],
                                "translation_locked": True,
                                "position_anchor": yaw_position_anchor,
                                "recorded_heading_reason": recorded_heading_observation.get(
                                    "reason"
                                ),
                                "message": (
                                    "Waiting online for the recorded waypoint departure view. "
                                    "Forward and lateral movement remain locked."
                                ),
                            }
                        )
                        neutral_hover(drone, 0.12)
                        continue
                    if (
                        not gate_attempt.get("ok")
                        and confirmed_rotation_gate is None
                        and not travel_started
                        and abs(last_alignment_yaw_rc) <= 1e-6
                        and isinstance(last_pose_gate, dict)
                    ):
                        frozen_position = pose_gate_position(last_pose_gate)
                        frozen_heading = pose_gate_heading(
                            last_pose_gate,
                            heading_trim_rad + float(calibrated_heading_offset_rad or 0.0),
                        )
                        frozen_direction = normalize_xz(target_direction_xz(frozen_position, target))
                        frozen_angle = signed_angle_xz(frozen_heading, frozen_direction)
                        if frozen_angle is not None and abs(frozen_angle) > math.radians(10.0):
                            frozen_scale = max(
                                0.65,
                                min(1.0, abs(frozen_angle) / math.radians(70.0)),
                            )
                            last_alignment_yaw_rc = (
                                yaw_sign
                                * max_yaw_rc
                                * yaw_direction_for_angle(frozen_angle, turn_direction_override)
                                * frozen_scale
                            )
                    if (
                        not gate_attempt.get("ok")
                        and confirmed_rotation_gate is None
                        and not travel_started
                        and abs(last_alignment_yaw_rc) > 1e-6
                        and not point_three_handoff
                        and blind_yaw_seconds < max_blind_yaw_seconds
                        and not (
                            point_three_handoff
                            and last_measured_alignment_error_rad is not None
                            and abs(last_measured_alignment_error_rad)
                            <= math.radians(12.0)
                        )
                    ):
                        if stop_flag is not None and stop_flag.stop:
                            abort_reason = "live localization stop requested"
                            break
                        if mission_stop_event is not None and mission_stop_event.is_set():
                            abort_reason = "emergency hover requested"
                            break
                        recovery_yaw_rc = math.copysign(
                            min(abs(last_alignment_yaw_rc), max_yaw_rc * 0.65),
                            last_alignment_yaw_rc,
                        )
                        publish_progress(
                            {
                                "phase": "bounded_blind_yaw_recovery",
                                "step_index": idx,
                                "step_title": title,
                                "pose_gate": gate_attempt,
                                "translation_locked": True,
                                "position_anchor": yaw_position_anchor,
                                "blind_yaw_seconds": blind_yaw_seconds,
                                "max_blind_yaw_seconds": max_blind_yaw_seconds,
                                "message": (
                                    "Pose temporarily invalid during an established turn; "
                                    "continuing bounded yaw only. Forward and lateral movement are locked."
                                ),
                            }
                        )
                        if not enforce_patrol_geofence(last_pose_gate, "bounded yaw recovery"):
                            break
                        sent = execute_rc_pulse(
                            drone,
                            yaw=recovery_yaw_rc,
                            lr=0.0,
                            bf=0.0,
                            du=0.0,
                            seconds=pulse_seconds,
                        )
                        record_pulse("yaw_recovery", sent)
                        executed_pulses += 1
                        pulse_count += 1
                        yaw_pulse_count += 1
                        blind_yaw_seconds += pulse_seconds
                        time.sleep(0.08)
                        continue
                    if confirmed_rotation_gate is not None:
                        # Reuse the fresh, continuity-checked position that
                        # accompanied the third stable optical-heading frame.
                        # Reading again here creates a race where the next
                        # frame can be rejected and enter the generic 45 s
                        # recovery even though the turn already converged.
                        current_gate = confirmed_rotation_gate
                        last_pose_gate = current_gate
                    else:
                        current_gate = pose_gate_or_abort()
                        if current_gate is None:
                            break
                    if rotation_alignment_ready_at is not None and current_gate.get("recent_hold_fallback"):
                        publish_progress(
                            {
                                    "phase": "taught_turn_wait_fresh_position",
                                    "step_index": idx,
                                    "step_title": title,
                                    "translation_locked": True,
                                    "route_visual_recovery_allowed": True,
                                    "position_anchor": yaw_position_anchor,
                                    "message": "Turn is visually aligned; holding until a non-held TSolve position arrives before translation.",
                                }
                        )
                        neutral_hover(drone, 0.12)
                        continue
                    if rotation_alignment_ready_at is not None:
                        rotation_alignment_ready_at = None
                    blind_yaw_seconds = 0.0
                    raw_current_position = pose_gate_position(current_gate)
                    # A real in-place yaw does not translate the aircraft.
                    # Monocular pose can still move the reported camera center
                    # slightly during rotation, so keep navigation geometry
                    # anchored until a real forward pulse is issued.
                    current_position = (
                        yaw_position_anchor
                        if yaw_position_anchor is not None
                        else raw_current_position
                    )
                    current_distance = horizontal_xz_distance(current_position, target)
                    if (
                        visual_stationary_retry_reference_distance is not None
                        and current_distance is not None
                        and current_distance
                        <= visual_stationary_retry_reference_distance - 0.015
                    ):
                        # Replenish the one-retry allowance only after the
                        # localizer has actually observed the movement that
                        # the previous retry was meant to reveal.
                        visual_stationary_retry_reference_distance = None
                    endpoint_overshoot_distance = (
                        patrol_endpoint_overshoot_distance(
                            current_position,
                            segment_start,
                            target,
                        )
                        if precise_arrival
                        else None
                    )
                    endpoint_undershoot_distance = (
                        patrol_endpoint_undershoot_distance(
                            current_position,
                            segment_start,
                            target,
                        )
                        if precise_arrival
                        else None
                    )
                    if (
                        endpoint_overshoot_distance is not None
                        and endpoint_overshoot_distance > 0.40
                    ):
                        abort_reason = (
                            "endpoint overshoot exceeds the 0.40 m bounded reverse "
                            f"limit ({endpoint_overshoot_distance:.2f} m); hovering"
                        )
                        break
                    active_endpoint_overshoot_correction = bool(
                        endpoint_overshoot_distance is not None
                        and endpoint_overshoot_distance > 0.03
                    )
                    # Endpoint evidence is evaluated below before this can be
                    # enabled.  A three-centimetre residual is already a valid
                    # indoor arrival, not permission for repeated forward RC.
                    active_endpoint_undershoot_correction = False
                    cross_track = horizontal_xz_segment_distance(current_position, segment_start, target)
                    if cross_track is not None and cross_track > max_cross_track:
                        abort_reason = (
                            "patrol left the planned route corridor "
                            f"({cross_track:.2f} map units > {max_cross_track:.2f})"
                        )
                        break
                    endpoint_alignment_pending = False
                    endpoint_alignment_error: float | None = None
                    endpoint_desired_heading: list[float] | None = None
                    endpoint_ready = not precise_arrival
                    tight_point_three_ready = False
                    tight_point_two_ready = False
                    if current_distance is not None:
                        final_distance = current_distance
                        tight_point_three_candidate = bool(
                            precise_arrival
                            and endpoint_leg_index == 2
                            and tight_metric_visual_endpoint_arrival_candidate(
                                current_gate,
                                target=target,
                                expected_leg_index=endpoint_leg_index,
                                command_progress_ceiling=(
                                    active_route_command_progress_ceiling
                                ),
                                maximum_metric_error=cruise_arrival_radius,
                            )
                        )
                        tight_point_three_ready = update_tight_endpoint_consensus(
                            tight_metric_endpoint_consensus_state,
                            current_gate,
                            candidate=tight_point_three_candidate,
                        )
                        tight_point_two_candidate = bool(
                            precise_arrival
                            and tight_metric_point_two_endpoint_arrival_candidate(
                                current_gate,
                                target=target,
                                segment_start=segment_start,
                                expected_leg_index=endpoint_leg_index,
                            )
                        )
                        tight_point_two_ready = update_stable_metric_endpoint_consensus(
                            point_two_metric_endpoint_consensus_state,
                            current_gate,
                            candidate=tight_point_two_candidate,
                        )
                        endpoint_ready = bool(
                            not precise_arrival
                            or tight_point_three_ready
                            or tight_point_two_ready
                            or taught_endpoint_arrival_verified(
                                current_gate.get("pose")
                                if isinstance(current_gate, dict)
                                else None,
                                expected_leg_index=endpoint_leg_index,
                            )
                        )
                        visual_checkpoint_arrival = bool(
                            precise_arrival
                            and taught_endpoint_stale_translation_arrival_verified(
                                current_gate.get("pose")
                                if isinstance(current_gate, dict)
                                else None,
                                expected_leg_index=endpoint_leg_index,
                            )
                        )
                        if visual_checkpoint_arrival:
                            # The aircraft has already reached the independently
                            # recognized endpoint while TSolve's published
                            # translation is stale.  End this leg without one
                            # more forward pulse and hand the exact shared
                            # waypoint to the next turn as its position anchor.
                            reached = True
                            arrival_mode = "visual_checkpoint_endpoint_verified"
                            active_route_translation_locked = True
                            active_route_position_anchor = list(target)
                            yaw_position_anchor = list(target)
                            if endpoint_leg_index == 4:
                                # The endpoint matcher proved the physical
                                # Point-1 arrival even though TSolve's ordinary
                                # publication was still behind.  Commit that
                                # fact to the trusted/rendered route state
                                # before lap 2 changes the active segment.
                                current_gate = commit_verified_point_one_handoff(
                                    current_gate,
                                    target,
                                ) or current_gate
                            publish_progress(
                                {
                                    "phase": "cruise_arrival",
                                    "step_index": idx,
                                    "step_title": title,
                                    "target": target,
                                    "distance_to_target": current_distance,
                                    "arrival_radius": cruise_arrival_radius,
                                    "soft_arrival_radius": cruise_soft_arrival_radius,
                                    "translation_locked": True,
                                    "position_anchor": target,
                                    "endpoint_leg_index": endpoint_leg_index,
                                    "visual_checkpoint_arrival": True,
                                    "verified_endpoint_pose_committed": bool(
                                        endpoint_leg_index == 4
                                    ),
                                    "message": (
                                        "Repeated progress-independent endpoint images "
                                        "verified the waypoint while TSolve translation "
                                        "was stale; stopping at the checkpoint without "
                                        "another forward pulse."
                                    ),
                                }
                            )
                            neutral_hover(drone, 0.25)
                            break
                        if (
                            tight_point_three_candidate
                            and not tight_point_three_ready
                            and current_distance <= cruise_arrival_radius
                        ):
                            publish_progress(
                                {
                                    "phase": "point3_metric_endpoint_consensus",
                                    "step_index": idx,
                                    "step_title": title,
                                    "translation_locked": True,
                                    "position_anchor": current_position,
                                    "distance_to_target": current_distance,
                                    "endpoint_leg_index": endpoint_leg_index,
                                    "metric_endpoint_hits": int(
                                        tight_metric_endpoint_consensus_state.get(
                                            "hits"
                                        )
                                        or 0
                                    ),
                                    "metric_endpoint_required_hits": 3,
                                    "message": (
                                        "Fresh TSolve is inside Point 3 and ORB confirms "
                                        "the active leg; hovering for three distinct metric "
                                        "poses without requiring ambiguous endpoint progress."
                                    ),
                                }
                            )
                            neutral_hover(drone, 0.12)
                            continue
                        if (
                            tight_point_two_candidate
                            and not tight_point_two_ready
                            and current_distance <= cruise_arrival_radius
                        ):
                            publish_progress(
                                {
                                    "phase": "point2_metric_endpoint_consensus",
                                    "step_index": idx,
                                    "step_title": title,
                                    "translation_locked": True,
                                    "position_anchor": current_position,
                                    "distance_to_target": current_distance,
                                    "endpoint_leg_index": endpoint_leg_index,
                                    "metric_endpoint_hits": int(
                                        point_two_metric_endpoint_consensus_state.get(
                                            "hits"
                                        )
                                        or 0
                                    ),
                                    "metric_endpoint_required_hits": 3,
                                    "message": (
                                        "Fresh TSolve is within three centimetres of Point 2; "
                                        "hovering for three distinct stable metric poses."
                                    ),
                                }
                            )
                            neutral_hover(drone, 0.12)
                            continue
                        # Arrival always wins over correction.  In particular,
                        # an independently verified endpoint about 3 cm from
                        # the saved coordinate must stop here instead of
                        # entering the undershoot command path.
                        if (
                            current_distance is not None
                            and current_distance <= cruise_arrival_radius
                            and endpoint_ready
                        ):
                            reached = True
                            arrival_mode = (
                                "strict_radius_metric_tsolve"
                                if tight_point_three_ready or tight_point_two_ready
                                else "strict_radius_endpoint_verified"
                                if precise_arrival
                                else "strict_radius"
                            )
                            break
                        if (
                            current_distance is not None
                            and current_distance <= cruise_soft_arrival_radius
                            and endpoint_ready
                        ):
                            reached = True
                            arrival_mode = "soft_deadband"
                            publish_progress(
                                {
                                    "phase": "cruise_arrival",
                                    "step_index": idx,
                                    "step_title": title,
                                    "target": target,
                                    "distance_to_target": current_distance,
                                    "arrival_radius": cruise_arrival_radius,
                                    "soft_arrival_radius": cruise_soft_arrival_radius,
                                    "message": (
                                        f"Patrol target reached within soft indoor deadband "
                                        f"({current_distance:.2f} <= {cruise_soft_arrival_radius:.2f}); hovering."
                                    ),
                                }
                            )
                            break
                        active_endpoint_undershoot_correction = bool(
                            precise_arrival
                            and patrol_endpoint_undershoot_correction_allowed(
                                endpoint_undershoot_distance,
                                current_distance,
                                cruise_soft_arrival_radius,
                                endpoint_arrived=endpoint_ready,
                                retry_used=endpoint_undershoot_retry_used,
                            )
                        )
                        if (
                            precise_arrival
                            and current_distance <= cruise_soft_arrival_radius
                            and not endpoint_ready
                            and route_follow_leg is not None
                        ):
                            endpoint_desired_heading = (
                                verified_route_desired_camera_heading(
                                    route_follow_leg,
                                    current_position=current_position,
                                    segment_start=segment_start,
                                    segment_end=target,
                                    default_lookahead_m=route_follow_lookahead_m,
                                    default_max_correction_deg=(
                                        route_follow_max_correction_deg
                                    ),
                                )
                            )
                            endpoint_camera_heading = pose_gate_camera_heading(
                                current_gate,
                                optical_heading_bias_rad=route_optical_heading_bias_rad,
                            )
                            endpoint_alignment_error = signed_angle_xz(
                                endpoint_camera_heading,
                                endpoint_desired_heading,
                            )
                            endpoint_alignment_pending = bool(
                                endpoint_alignment_error is not None
                                and abs(endpoint_alignment_error)
                                > math.radians(route_follow_alignment_enter_deg)
                            )
                            if endpoint_alignment_pending:
                                # Do not enter the neutral-only recovery deadlock
                                # while the current image is visibly misaligned
                                # with the successful route. Reuse the normal
                                # yaw-only controller below; translation remains
                                # locked until the camera is back on the 11:57
                                # heading and endpoint imagery can verify arrival.
                                forward_alignment_locked = False
                                publish_progress(
                                    {
                                        "phase": "taught_endpoint_yaw_realign",
                                        "step_index": idx,
                                        "step_title": title,
                                        "translation_locked": True,
                                        "position_anchor": current_position,
                                        "distance_to_target": current_distance,
                                        "heading_error_deg": math.degrees(
                                            endpoint_alignment_error
                                        ),
                                        "golden_source_replay_id": (
                                            (verified_route_lock or {}).get(
                                                "source_replay_id"
                                            )
                                        ),
                                        "message": (
                                            "Inside the waypoint radius but the camera is not "
                                            "aligned with the verified 11:57 endpoint view; "
                                            "correcting yaw only before image verification."
                                        ),
                                    }
                                )
                        if (
                            precise_arrival
                            and current_distance <= cruise_soft_arrival_radius
                            and not endpoint_ready
                            and not endpoint_alignment_pending
                            and not active_endpoint_overshoot_correction
                            and not active_endpoint_undershoot_correction
                        ):
                            publish_progress(
                                {
                                    "phase": "taught_endpoint_verification",
                                    "step_index": idx,
                                    "step_title": title,
                                    "translation_locked": True,
                                    "route_visual_recovery_allowed": True,
                                    "position_anchor": current_position,
                                    "distance_to_target": current_distance,
                                    "endpoint_leg_index": endpoint_leg_index,
                                    "message": (
                                        "Inside the waypoint radius; hovering until three "
                                        "progress-independent whole-leg image matches verify "
                                        "the taught endpoint."
                                    ),
                                }
                            )
                            verified_gate = wait_for_pose_recovery(
                                "taught_endpoint_verification",
                                "waypoint geometry reached without independent endpoint evidence",
                                translation_target=target,
                                translation_arrival_radius=cruise_soft_arrival_radius,
                                require_endpoint_verified=True,
                                endpoint_leg_index=endpoint_leg_index,
                            )
                            if verified_gate is None:
                                break
                            current_gate = verified_gate
                            current_position = pose_gate_position(current_gate)
                            current_distance = horizontal_xz_distance(
                                current_position, target
                            )
                            final_distance = current_distance
                            endpoint_ready = bool(
                                taught_endpoint_arrival_verified(
                                    current_gate.get("pose"),
                                    expected_leg_index=endpoint_leg_index,
                                )
                                or current_gate.get(
                                    "metric_point_two_arrival_verified"
                                )
                            )
                            if (
                                endpoint_ready
                                and current_distance is not None
                                and current_distance <= cruise_arrival_radius
                            ):
                                reached = True
                                arrival_mode = (
                                    "strict_radius_metric_tsolve"
                                    if current_gate.get(
                                        "metric_point_two_arrival_verified"
                                    )
                                    else "strict_radius_endpoint_verified"
                                )
                                break
                        if current_distance < closest_distance - 0.03:
                            closest_distance = current_distance
                            diverging_pulses = 0
                        elif closest_distance < float("inf") and current_distance > closest_distance + 0.14:
                            diverging_pulses += 1
                        else:
                            diverging_pulses = max(0, diverging_pulses - 1)
                        if diverging_pulses >= 8:
                            abort_reason = (
                                "closed-loop cruise is moving away from the patrol target "
                                f"(closest {closest_distance:.2f}, now {current_distance:.2f})"
                            )
                            break

                    desired_unit = patrol_navigation_direction_xz(
                        current_position,
                        target,
                        endpoint_heading=(
                            endpoint_desired_heading
                            if endpoint_alignment_pending
                            else None
                        ),
                    )
                    if desired_unit is None:
                        if patrol_zero_direction_arrival_allowed(
                            current_distance,
                            cruise_arrival_radius,
                            precise_arrival=precise_arrival,
                            endpoint_ready=endpoint_ready,
                        ):
                            # TSolve may publish the exact saved coordinate
                            # after endpoint consensus. There is then no
                            # direction left to normalize: that is successful
                            # arrival, not a navigation failure.
                            reached = True
                            arrival_mode = (
                                "exact_position_endpoint_verified"
                                if precise_arrival
                                else "exact_position"
                            )
                            publish_progress(
                                {
                                    "phase": "cruise_arrival",
                                    "step_index": idx,
                                    "step_title": title,
                                    "target": target,
                                    "distance_to_target": current_distance,
                                    "arrival_radius": cruise_arrival_radius,
                                    "endpoint_leg_index": endpoint_leg_index,
                                    "translation_locked": True,
                                    "position_anchor": target,
                                    "message": (
                                        "TSolve is already at the verified waypoint; "
                                        "accepting the zero remaining direction as arrival."
                                    ),
                                }
                            )
                            break
                        abort_reason = "could not compute patrol target direction from latest TSolve pose"
                        break
                    # Horizontal patrol holds the takeoff altitude through DJI.
                    # Monocular map Y is not a safe altitude-control signal.
                    du_rc = 0.0
                    heading = pose_gate_heading(
                        current_gate,
                        heading_trim_rad + float(calibrated_heading_offset_rad or 0.0),
                    )
                    route_follow_desired_heading = (
                        verified_route_desired_camera_heading(
                            route_follow_leg,
                            current_position=current_position,
                            segment_start=segment_start,
                            segment_end=target,
                            default_lookahead_m=route_follow_lookahead_m,
                            default_max_correction_deg=(
                                route_follow_max_correction_deg
                            ),
                        )
                        if route_follow_leg is not None
                        else None
                    )
                    route_follow_camera_heading = (
                        pose_gate_camera_heading(
                            current_gate,
                            optical_heading_bias_rad=route_optical_heading_bias_rad,
                        )
                        if route_follow_desired_heading is not None
                        else None
                    )
                    route_follow_angle = signed_angle_xz(
                        route_follow_camera_heading,
                        route_follow_desired_heading,
                    )
                    # A probe measures the body axis in the room frame only at
                    # the instant it is made.  It becomes stale as soon as the
                    # aircraft yaws and must never be used as a persistent
                    # world-frame steering vector.  The probe-derived
                    # camera-to-body offset is invariant across yaw, so apply
                    # that offset to every fresh localized camera heading.
                    angle = (
                        confirmed_rotation_angle
                        if (
                            (
                                guarded_taught_rotation
                                or point_three_handoff
                                or point_four_handoff
                            )
                            and not travel_started
                            and confirmed_rotation_angle is not None
                        )
                        else (
                            route_follow_angle
                            if route_follow_angle is not None
                            else signed_angle_xz(heading, desired_unit)
                        )
                    )
                    if angle is None:
                        abort_reason = "latest TSolve pose has no usable heading or calibrated body axis; refusing patrol translation"
                        break
                    speed_scale = max(0.32, min(1.0, (current_distance or 0.0) / max(cruise_arrival_radius * 5.0, 0.55)))
                    speed_scale *= corridor_recovery_speed_scale(
                        cross_track,
                        recovery_start=cross_track_recovery_start,
                        hard_limit=max_cross_track,
                    )
                    yaw_rc = 0.0
                    bf_rc = 0.0
                    lr_rc = 0.0
                    expected_yaw_heading_sign = 0.0
                    navigation_mode = "camera_heading"
                    if heading is None and route_follow_camera_heading is None:
                        abort_reason = "latest TSolve pose has no usable calibrated body heading; refusing patrol translation"
                        break
                    else:
                        navigation_mode = (
                            "verified_1157_route_heading_yaw_then_forward"
                            if route_follow_angle is not None
                            else "fresh_heading_yaw_then_forward"
                        )
                        angle_abs = abs(angle)
                        if (
                            not travel_started
                            and angle_abs + math.radians(1.5) < best_alignment_error_rad
                        ):
                            best_alignment_error_rad = angle_abs
                            alignment_progress_deadline = min(
                                absolute_segment_deadline,
                                max(alignment_progress_deadline, now + 20.0),
                            )
                        # Hysteresis prevents a fresh but slightly noisy heading
                        # from alternating forever between yaw and forward.
                        if route_follow_angle is not None:
                            # The successful run is the primary steering
                            # reference through Point 4. Enter translation only
                            # inside four degrees and resume yaw correction as
                            # soon as drift exceeds six degrees.
                            alignment_limit_deg = (
                                route_follow_alignment_exit_deg
                                if forward_alignment_locked
                                else route_follow_alignment_enter_deg
                            )
                        elif guarded_taught_rotation:
                            # The recorded 3->4 corridor passes close to the
                            # partition. Enter forward flight only below six
                            # degrees and re-align as soon as error exceeds
                            # eight; the former 10/14-degree window produced
                            # the diagonal wall approach in the failed run.
                            alignment_limit_deg = 8.0 if forward_alignment_locked else 6.0
                        else:
                            alignment_limit_deg = 14.0 if forward_alignment_locked else 10.0
                        if angle_abs > math.radians(alignment_limit_deg):
                            forward_alignment_locked = False
                            yaw_scale = max(0.65, min(1.0, angle_abs / math.radians(70.0)))
                            expected_yaw_heading_sign = yaw_direction_for_angle(
                                angle,
                                turn_direction_override if not travel_started else None,
                            )
                            yaw_rc = (
                                yaw_sign
                                * max_yaw_rc
                                * expected_yaw_heading_sign
                                * yaw_scale
                            )
                            last_alignment_yaw_rc = yaw_rc
                        else:
                            # Do not mix turning and translation. Once aligned,
                            # fly only on DJI's body-forward channel.
                            forward_alignment_locked = True
                            last_alignment_yaw_rc = 0.0
                            if active_endpoint_overshoot_correction:
                                # Braking/localization delay can put the drone
                                # a short distance beyond a checkpoint while
                                # it is still facing along the route. Reverse
                                # slowly on the same body axis; never turn 180
                                # degrees or keep flying farther forward.
                                reverse_scale = min(
                                    0.35,
                                    max(
                                        0.18,
                                        float(endpoint_overshoot_distance or 0.0)
                                        / 0.80,
                                    ),
                                )
                                bf_rc = -max_forward_rc * reverse_scale
                                navigation_mode = (
                                    "bounded_endpoint_reverse_correction"
                                )
                            elif active_endpoint_undershoot_correction:
                                # A fresh recovery solve can also reveal that
                                # the model declared arrival slightly early.
                                # Use the same slow, one-pulse-confirmed policy
                                # in the forward direction; endpoint imagery is
                                # still mandatory before the turn begins.
                                forward_scale = min(
                                    0.30,
                                    max(
                                        0.18,
                                        float(endpoint_undershoot_distance or 0.0)
                                        / 0.80,
                                    ),
                                )
                                bf_rc = max_forward_rc * forward_scale
                                navigation_mode = (
                                    "bounded_endpoint_forward_correction"
                                )
                            else:
                                bf_rc = max_forward_rc * speed_scale
                            if guarded_taught_rotation and bf_rc > 0.0:
                                # Smaller pulses keep camera overlap high for
                                # continuous baseline verification and make
                                # every correction visible before the next
                                # physical movement.
                                bf_rc = min(bf_rc, max_forward_rc * 0.70)
                            departure_pose = current_gate.get("pose")
                            if (
                                bf_rc > 0.0
                                and current_gate.get(
                                    "verified_endpoint_turn_departure"
                                )
                                and isinstance(departure_pose, dict)
                                and departure_pose.get(
                                    "verified_endpoint_turn_leg_index"
                                )
                                == 4
                            ):
                                # Point 4->1 has no recorded-frame authority.
                                # Its verified endpoint plus live optical yaw may
                                # authorize exactly one deliberately small probe;
                                # another command still requires observed metric
                                # progress through the standard one-pulse gate.
                                bf_rc = min(bf_rc, max_forward_rc * 0.45)
                            # Mark travel started only after the final
                            # command-aware pose gate accepts this pulse.

                    forward_gain = bf_rc / max_forward_rc if max_forward_rc > 1e-9 else 0.0
                    lateral_gain = lr_rc / max_lateral_rc if max_lateral_rc > 1e-9 else 0.0
                    rotation_only_command = (
                        abs(yaw_rc) > 1e-6
                        and abs(bf_rc) <= 1e-6
                        and abs(lr_rc) <= 1e-6
                    )
                    if (
                        point_three_handoff
                        and not travel_started
                        and rotation_only_command
                    ):
                        if point_three_planned_turn_degrees is None:
                            point_three_planned_turn_degrees = abs(
                                math.degrees(angle)
                            )
                    if rotation_only_command and travel_started and yaw_position_anchor is None:
                        # DJI can retain a small amount of forward motion after
                        # a translation pulse.  Do not declare the pre-yaw
                        # position immutable until two fresh pose samples show
                        # that this residual motion has stopped.  Previously we
                        # anchored immediately, then rejected the legitimate
                        # 21.8 cm braking distance as rotation drift forever.
                        active_route_translation_locked = False
                        active_route_position_anchor = None
                        publish_progress(
                            {
                                "phase": "pre_yaw_translation_settle",
                                "step_index": idx,
                                "step_title": title,
                                "translation_locked": False,
                                "position_anchor": None,
                                "message": (
                                    "Hovering until forward motion settles before the next yaw correction."
                                ),
                            }
                        )
                        neutral_hover(drone, 0.35)
                        # A global/recovery TSolve can legitimately leave the
                        # controller without a newly published pose for several
                        # seconds. Keep the aircraft neutral while that solve
                        # finishes instead of treating a short publication gap
                        # as residual physical motion. The 15:20 live run had
                        # a ~5.9 s gap here and published a stable pose again
                        # immediately afterwards; the former fixed 3 s timeout
                        # aborted the entry to Point 1 unnecessarily.
                        settle_wait_seconds = max(8.0, pose_recovery_seconds)
                        settle_deadline = time.time() + settle_wait_seconds
                        settle_gate = current_gate
                        settle_position = raw_current_position
                        try:
                            settle_count = int(current_gate.get("processed_count") or 0)
                        except (TypeError, ValueError):
                            settle_count = 0
                        # Recovery may already have returned a fresh,
                        # two-source verified full-loop pose for this exact
                        # current frame. That absolute route observation is
                        # sufficient to anchor an in-place yaw immediately.
                        # Waiting for another such pose created the 10:18
                        # Point-2->3 deadlock: strong route matches arrived
                        # about every 9.2 s while the settle window was 8 s,
                        # so every retry missed the next match by ~1 s.
                        stable_settle_samples = (
                            2
                            if patrol_visual_yaw_anchor_ready(current_gate)
                            else 0
                        )
                        while time.time() < settle_deadline and stable_settle_samples < 2:
                            candidate_gate = continuity_guarded_pose_gate()
                            try:
                                candidate_count = int(candidate_gate.get("processed_count") or 0)
                            except (TypeError, ValueError):
                                candidate_count = settle_count
                            if not candidate_gate.get("ok") or candidate_count <= settle_count:
                                neutral_hover(drone, 0.08)
                                continue
                            candidate_position = pose_gate_position(candidate_gate)
                            settle_step = horizontal_xz_distance(candidate_position, settle_position)
                            settle_gate = candidate_gate
                            settle_count = candidate_count
                            last_pose_gate = candidate_gate
                            if candidate_gate.get("route_progress") is not None:
                                active_route_progress = float(candidate_gate["route_progress"])
                            if candidate_position is not None:
                                settle_position = candidate_position
                            if patrol_visual_yaw_anchor_ready(candidate_gate):
                                # The full-loop matcher directly observes the
                                # current recorded route location.  Waiting for
                                # two more TSolve publications here caused the
                                # otherwise healthy patrol to freeze before a
                                # turn whenever TSolve was recovering from a
                                # repeated-room alias.
                                stable_settle_samples = 2
                                break
                            if settle_step is not None and settle_step <= 0.06:
                                stable_settle_samples += 1
                            else:
                                stable_settle_samples = 0
                            time.sleep(0.06)
                        if stable_settle_samples < 2 or settle_position is None:
                            # A localization timeout is not a flight failure.
                            # Keep the aircraft in neutral hover, release any
                            # stale yaw anchor, and let the online localizer
                            # continue processing the same active leg.  The
                            # next cruise-loop iteration retries from fresh
                            # data; only an explicit stop/emergency or a real
                            # safety barrier may terminate the mission.
                            yaw_position_anchor = None
                            active_route_translation_locked = False
                            active_route_position_anchor = None
                            publish_progress(
                                {
                                    "phase": "pre_yaw_online_recovery",
                                    "step_index": idx,
                                    "step_title": title,
                                    "translation_locked": True,
                                    "route_visual_recovery_allowed": True,
                                    "rotation_release_requested": True,
                                    "position_anchor": settle_position,
                                    "body_forward_gain": 0.0,
                                    "body_lateral_gain": 0.0,
                                    "message": (
                                        "Position did not settle before yaw; hovering with "
                                        "movement locked while ATLAS relocalizes online."
                                    ),
                                }
                            )
                            neutral_hover(drone, GUIDED_RECOVERY_HOVER_SECONDS)
                            continue
                        current_gate = settle_gate
                        yaw_position_anchor = settle_position
                        active_route_translation_locked = True
                        active_route_position_anchor = yaw_position_anchor
                        publish_progress(
                            {
                                "phase": "pre_yaw_translation_settled",
                                "step_index": idx,
                                "step_title": title,
                                "translation_locked": True,
                                "position_anchor": yaw_position_anchor,
                                "message": (
                                    "Forward motion settled; rotation position is now anchored."
                                ),
                            }
                        )
                        # Re-evaluate the desired heading from the settled pose
                        # before issuing any yaw command.
                        continue
                    if rotation_only_command and yaw_position_anchor is None:
                        yaw_position_anchor = current_position
                    # Keep the room position fixed only while an actual
                    # yaw-only command is selected.  As soon as heading is
                    # aligned and the controller intends to move forward,
                    # announce hover/translation intent first.  The localizer
                    # can then release its rotation anchor; the safety gate
                    # below still prevents any RC translation until a fresh
                    # unlocked pose is observed.
                    active_route_translation_locked = rotation_only_command
                    active_route_position_anchor = (
                        yaw_position_anchor if active_route_translation_locked else None
                    )
                    progress = {
                        "phase": "cruise",
                        "step_index": idx,
                        "step_title": title,
                        "target": target,
                        "distance_to_target": current_distance,
                        "cross_track_error": cross_track,
                        "cross_track_recovery": bool(
                            cross_track is not None and cross_track > cross_track_recovery_start
                        ),
                        "heading_error_deg": math.degrees(angle),
                        "body_forward_gain": forward_gain,
                        "body_lateral_gain": lateral_gain,
                        "translation_locked": active_route_translation_locked,
                        "position_anchor": active_route_position_anchor,
                        "navigation_mode": navigation_mode,
                        "endpoint_overshoot_correction": (
                            active_endpoint_overshoot_correction
                        ),
                        "endpoint_overshoot_distance_m": (
                            endpoint_overshoot_distance
                        ),
                        "endpoint_undershoot_correction": (
                            active_endpoint_undershoot_correction
                        ),
                        "endpoint_undershoot_distance_m": (
                            endpoint_undershoot_distance
                        ),
                        "turn_direction_override": (
                            turn_direction_override if not travel_started else None
                        ),
                        "pose_age_seconds": current_gate.get("age_seconds"),
                        "processed_count": current_gate.get("processed_count"),
                        "message": (
                            f"Closed-loop patrol: {title}, "
                            f"distance {current_distance:.2f} map units, "
                            f"heading error {math.degrees(angle):+.1f} deg, "
                            f"forward {forward_gain:+.2f}, lateral {lateral_gain:+.2f}"
                            if current_distance is not None
                            else f"Closed-loop patrol: {title}"
                        ),
                    }
                    last_navigation_mode = navigation_mode
                    last_processed_count = current_gate.get("processed_count")
                    last_pose_age = current_gate.get("age_seconds")
                    publish_progress(progress)

                    if current_distance is None:
                        abort_reason = "could not measure distance to patrol target"
                        break
                    if not enforce_patrol_geofence(current_gate, f"closed-loop cruise step {idx}"):
                        break
                    translation_issue = guided_command_pose_safety_issue(
                        current_gate,
                        yaw=yaw_rc,
                        lr=lr_rc,
                        bf=bf_rc,
                        du=du_rc,
                    )
                    if translation_issue is not None:
                        release_rotation_lock = bool(
                            pose_gate_rotation_locked(current_gate)
                            and (abs(bf_rc) > 1e-6 or abs(lr_rc) > 1e-6)
                        )
                        if release_rotation_lock:
                            active_route_translation_locked = False
                            active_route_position_anchor = None
                        publish_progress(
                            {
                                "phase": "translation_pose_gate",
                                "step_index": idx,
                                "step_title": title,
                                "translation_locked": (
                                    False if release_rotation_lock else active_route_translation_locked
                                ),
                                "position_anchor": (
                                    None if release_rotation_lock else active_route_position_anchor
                                ),
                                # A completed yaw has released the display/
                                # metric position anchor, but the aircraft is
                                # still receiving neutral RC here. Open the
                                # visual recovery gate so a verified ORB route
                                # pose can become the first translation-safe
                                # pose instead of being rejected forever as
                                # ``visual_route_translation_locked``.
                                "route_visual_recovery_allowed": (
                                    release_rotation_lock
                                ),
                                "physical_translation_active": False,
                                "body_forward_gain": (
                                    0.0 if release_rotation_lock else forward_gain
                                ),
                                "body_lateral_gain": (
                                    0.0 if release_rotation_lock else lateral_gain
                                ),
                                "rotation_release_requested": release_rotation_lock,
                                "pose_gate": current_gate,
                                "message": (
                                    (
                                        "Holding neutral while releasing the completed turn's "
                                        "position lock; waiting for a fresh translation-safe pose."
                                    )
                                    if release_rotation_lock
                                    else (
                                        f"Holding before forward motion: {translation_issue}. "
                                        "Yaw-only control remains available while position recovers."
                                    )
                                ),
                            }
                        )
                        neutral_hover(drone, 0.12)
                        continue
                    horizontal_command = abs(bf_rc) > 1e-6 or abs(lr_rc) > 1e-6
                    if horizontal_command:
                        command_start_progress = route_segment_progress_xz(
                            current_position,
                            active_route_segment_start,
                            active_route_segment_end,
                        )
                        progress_candidates = [
                            float(value)
                            for value in (
                                active_route_progress,
                                command_start_progress,
                            )
                            if isinstance(value, (int, float))
                            and math.isfinite(float(value))
                        ]
                        command_progress_base = max(progress_candidates or [0.0])
                        command_leg_length = horizontal_xz_distance(
                            active_route_segment_start,
                            active_route_segment_end,
                        )
                        command_progress_budget = (
                            max_unverified_translation_m / command_leg_length
                            if command_leg_length is not None
                            and command_leg_length > 1e-9
                            else 0.0
                        )
                        active_route_command_progress_ceiling = (
                            advance_route_command_progress_ceiling(
                                active_route_command_progress_ceiling,
                                command_progress_base,
                                command_progress_budget,
                            )
                        )
                        active_route_command_sequence += 1
                        # This is the single publication that authorizes the
                        # visual route to advance. It occurs only after every
                        # command-side pose/geofence gate has passed and stays
                        # active while the real RC command and its captured
                        # result frame are processed.
                        publish_progress(
                            {
                                **progress,
                                "phase": "translation_command_active",
                                "translation_locked": False,
                                "position_anchor": None,
                                "physical_translation_active": True,
                                "message": (
                                    f"Executing verified horizontal patrol motion: {title}."
                                ),
                            }
                        )
                    used_smooth_cruise = bool(
                        smooth_continuous_cruise
                        and horizontal_command
                        and abs(yaw_rc) <= 1e-6
                        and not active_endpoint_undershoot_correction
                        and not current_gate.get(
                            "verified_endpoint_turn_departure"
                        )
                    )
                    smooth_cruise_issue: str | None = None
                    smooth_after_gate: dict[str, Any] | None = None
                    if used_smooth_cruise:
                        sent, smooth_after_gate, smooth_cruise_issue = (
                            execute_guarded_cruise_window(
                                current_gate=current_gate,
                                current_position=current_position,
                                target=target,
                                bf=bf_rc,
                                lr=lr_rc,
                                du=du_rc,
                                seconds=cruise_window_seconds,
                                arrival_radius=cruise_soft_arrival_radius,
                            )
                        )
                    else:
                        # Turns remain position-locked pulses. Only horizontal
                        # travel becomes a continuous, pose-supervised cruise.
                        sent = execute_rc_pulse(
                            drone,
                            yaw=yaw_rc,
                            lr=lr_rc,
                            bf=bf_rc,
                            du=du_rc,
                            seconds=pulse_seconds,
                        )
                    pulse_completed_unix = time.time()
                    if (
                        active_endpoint_undershoot_correction
                        and abs(bf_rc) > 1e-6
                    ):
                        # Consume the allowance when the command is attempted,
                        # even if its acknowledgement later becomes uncertain.
                        # No second correction may be sent from the same stale
                        # endpoint estimate.
                        endpoint_undershoot_retry_used = True
                    command_seconds = float(
                        sent.get("actual_seconds")
                        or sent.get("seconds")
                        or pulse_seconds
                    )
                    if abs(yaw_rc) > 1e-6:
                        if yaw_position_anchor is None:
                            yaw_position_anchor = current_position
                        record_pulse("yaw", sent)
                        yaw_pulse_count += 1
                        if point_three_handoff and not travel_started:
                            point_three_yaw_effort_seconds += (
                                normalized_yaw_command_effort(
                                    yaw_rc,
                                    max_yaw_rc,
                                    command_seconds,
                                )
                            )
                        if travel_started:
                            travel_yaw_command_seconds += pulse_seconds
                    if abs(bf_rc) > 1e-6:
                        travel_started = True
                        active_route_translation_locked = False
                        active_route_position_anchor = None
                        record_pulse("forward", sent)
                        forward_pulse_count += 1
                        forward_command_seconds += command_seconds
                    if abs(lr_rc) > 1e-6:
                        travel_started = True
                        record_pulse("lateral", sent)
                        lateral_pulse_count += 1
                    if abs(du_rc) > 1e-6:
                        record_pulse("vertical", sent)
                    executed_pulses += 1
                    pulse_count += 1
                    if smooth_cruise_issue is not None:
                        if abort_reason:
                            break
                        visual_stationary_retry_available = (
                            visual_stationary_retry_reference_distance is None
                        )
                        recovered_progress_gate = wait_for_pose_recovery(
                            "smooth_continuous_cruise",
                            smooth_cruise_issue,
                            require_translation_safe=True,
                            translation_target=target,
                            translation_reference_distance=current_distance,
                            translation_arrival_radius=cruise_soft_arrival_radius,
                            endpoint_leg_index=endpoint_leg_index,
                            require_observed_translation_progress=True,
                            allow_visual_stationary_retry=(
                                visual_stationary_retry_available
                            ),
                        )
                        if recovered_progress_gate is None:
                            break
                        if recovered_progress_gate.get(
                            "visual_route_stationary_retry"
                        ):
                            visual_stationary_retry_reference_distance = (
                                current_distance
                            )
                            publish_progress(
                                {
                                    "phase": "visual_stationary_bounded_retry_ready",
                                    "step_index": idx,
                                    "step_title": title,
                                    "translation_locked": True,
                                    "position_anchor": pose_gate_position(
                                        recovered_progress_gate
                                    ),
                                    "visual_stationary_retry_consumed": True,
                                    "message": (
                                        "Strong current route-image consensus permits one "
                                        "bounded low-stick retry. Another retry remains "
                                        "locked until ATLAS observes at least 0.015 m of "
                                        "forward progress."
                                    ),
                                }
                            )
                        stale_motion_count = 0
                        stale_motion_reference_position = None
                        diverging_pulses = 0
                        # Recovery held zero stick. Recompute navigation from
                        # its fresh authoritative pose before resuming cruise.
                        continue
                    if (
                        guarded_taught_rotation
                        and not travel_started
                        and abs(yaw_rc) > 1e-6
                    ):
                        # Never call the generic accepted-position waiter
                        # after a point-3→4 yaw pulse.  A rejection can arrive
                        # immediately after a perfectly valid turn command;
                        # blocking here prevented the next loop iteration from
                        # consuming the still-fresh optical heading.  Return
                        # directly to the rotation-only controller instead.
                        publish_progress(
                            {
                                "phase": "taught_turn_post_pulse_optical_return",
                                "step_index": idx,
                                "step_title": title,
                                "taught_leg": [
                                    taught_leg.get("from_point"),
                                    taught_leg.get("to_point"),
                                ],
                                "translation_locked": True,
                                "position_anchor": yaw_position_anchor,
                                "message": (
                                    "Right-turn pulse completed; keeping point-3 position "
                                    "frozen and returning directly to optical heading."
                                ),
                            }
                        )
                        time.sleep(0.08)
                        continue
                    after_gate = (
                        smooth_after_gate
                        if used_smooth_cruise
                        else wait_for_pose_captured_after(
                            pulse_completed_unix,
                            current_gate,
                            abort_on_timeout=not one_pulse_pose_confirmation,
                        )
                    )
                    if after_gate is None and abort_reason:
                        break
                    raw_after_position = pose_gate_position(after_gate)
                    if abs(yaw_rc) > 1e-6 and yaw_position_anchor is not None:
                        after_position = yaw_position_anchor
                    else:
                        after_position = raw_after_position
                        if abs(bf_rc) > 1e-6 or abs(lr_rc) > 1e-6:
                            yaw_position_anchor = None
                    after_distance = horizontal_xz_distance(after_position, target)
                    if abs(yaw_rc) > 1e-6 and isinstance(after_gate, dict):
                        if route_follow_desired_heading is not None:
                            # Compare camera-route error before and after the
                            # pulse.  The old code compared a route-camera
                            # error before the pulse with a body-heading error
                            # afterwards, so an asynchronous valid turn could
                            # be classified as reversed.
                            after_heading = pose_gate_camera_heading(
                                after_gate,
                                optical_heading_bias_rad=route_optical_heading_bias_rad,
                            )
                            after_angle = signed_angle_xz(
                                after_heading,
                                route_follow_desired_heading,
                            )
                        else:
                            after_heading = pose_gate_heading(
                                after_gate,
                                heading_trim_rad + float(calibrated_heading_offset_rad or 0.0),
                            )
                            after_direction = normalize_xz(
                                target_direction_xz(after_position, target)
                            )
                            after_angle = signed_angle_xz(after_heading, after_direction)
                        if after_angle is not None:
                            yaw_response_reversed = yaw_target_error_response(
                                angle,
                                after_angle,
                            )
                            if yaw_response_reversed is not None:
                                if yaw_response_reversed:
                                    wrong_yaw_pulses += 1
                                    correct_yaw_pulses = 0
                                else:
                                    wrong_yaw_pulses = 0
                                    correct_yaw_pulses += 1
                                    if correct_yaw_pulses >= 3 and not yaw_sign_verified:
                                        yaw_sign_verified = True
                                        rc_summary["adaptive_axis"]["yaw_sign_verified"] = True
                            sign_action = yaw_sign_recovery_action(
                                yaw_sign_verified=yaw_sign_verified,
                                wrong_yaw_pulses=wrong_yaw_pulses,
                                yaw_flip_count=yaw_flip_count,
                            )
                            if sign_action == "flip":
                                yaw_sign *= -1.0
                                yaw_flip_count += 1
                                wrong_yaw_pulses = 0
                                correct_yaw_pulses = 0
                                rc_summary["adaptive_axis"]["yaw_sign"] = yaw_sign
                                rc_summary["adaptive_axis"]["yaw_flips"] = yaw_flip_count
                                publish_progress(
                                    {
                                        "phase": "yaw_sign_correction",
                                        "step_index": idx,
                                        "step_title": title,
                                        "message": (
                                            "Initial yaw-polarity verification observed three "
                                            "comparable target-error regressions; reversing DJI "
                                            "yaw once before horizontal motion."
                                        ),
                                    }
                                )
                            elif sign_action == "recover":
                                # Delayed localization must not change a
                                # polarity already proved earlier in flight.
                                # Hold zero stick and let the normal loop
                                # reacquire a comparable fresh heading.
                                wrong_yaw_pulses = 0
                                correct_yaw_pulses = 0
                                yaw_feedback_recovery_count += 1
                                neutral_hover(drone, 0.45)
                                publish_progress(
                                    {
                                        "phase": "yaw_feedback_recovery",
                                        "step_index": idx,
                                        "step_title": title,
                                        "translation_locked": True,
                                        "position_anchor": yaw_position_anchor,
                                        "yaw_sign_preserved": yaw_sign,
                                        "recovery_count": yaw_feedback_recovery_count,
                                        "message": (
                                            "Yaw polarity was already verified; preserving it and "
                                            "hovering for fresh heading feedback instead of reversing."
                                        ),
                                    }
                                )
                                continue
                            elif sign_action == "abort":
                                abort_reason = (
                                    "yaw alignment did not converge after initial polarity verification; "
                                    "hovering without forward movement"
                                )
                                break
                    horizontal_pulse = abs(bf_rc) > 1e-6 or abs(lr_rc) > 1e-6
                    got_new_pose = (
                        isinstance(after_gate, dict)
                        and after_gate.get("processed_count") != current_gate.get("processed_count")
                    )
                    strict_progress_issue = (
                        patrol_translation_pulse_progress_issue(
                            current_position,
                            after_position,
                            target,
                            got_new_pose=got_new_pose,
                            maximum_pose_step=max_pose_step,
                        )
                        if (
                            horizontal_pulse
                            and one_pulse_pose_confirmation
                            and not used_smooth_cruise
                        )
                        else None
                    )
                    if strict_progress_issue is not None:
                        recovered_progress_gate = wait_for_pose_recovery(
                            "translation_progress",
                            strict_progress_issue,
                            require_translation_safe=True,
                            translation_target=target,
                            translation_reference_distance=current_distance,
                            translation_arrival_radius=cruise_soft_arrival_radius,
                            endpoint_leg_index=endpoint_leg_index,
                            require_observed_translation_progress=True,
                        )
                        if recovered_progress_gate is None:
                            break
                        stale_motion_count = 0
                        stale_motion_reference_position = None
                        diverging_pulses = 0
                        # The recovery loop remained neutral and observed delayed
                        # progress from this same pulse. Recompute the complete
                        # navigation state before any next command.
                        continue
                    if horizontal_pulse and one_pulse_pose_confirmation:
                        stale_motion_count = 0
                        stale_motion_reference_position = None
                        diverging_pulses = 0
                        if after_distance is not None:
                            closest_distance = min(closest_distance, after_distance)
                    elif not horizontal_pulse:
                        stale_motion_count = 0
                        stale_motion_reference_position = None
                    elif not got_new_pose:
                        if stale_motion_count == 0:
                            stale_motion_reference_position = list(current_position)
                        stale_motion_count += 1
                    elif after_distance is None:
                        if stale_motion_count == 0:
                            stale_motion_reference_position = list(current_position)
                        stale_motion_count += 1
                    elif published_position_advanced_toward_target(
                        current_position,
                        after_position,
                        target,
                    ):
                        stale_motion_count = 0
                        stale_motion_reference_position = None
                        diverging_pulses = 0
                        closest_distance = min(closest_distance, after_distance)
                    elif after_distance > current_distance + 0.08:
                        diverging_pulses += 1
                        if allow_axis_auto_calibration and diverging_pulses >= 3 and forward_flip_count < 2:
                            forward_flip_count += 1
                            diverging_pulses = 0
                            calibrate_body_axes(after_gate)
                            rc_summary["adaptive_axis"]["forward_flips"] = forward_flip_count
                            publish_progress(
                                {
                                    "phase": "adaptive_axis_recalibration",
                                    "step_index": idx,
                                    "step_title": title,
                                    "message": "Body-axis motion moved away from target; recalibrating with fresh TSolve feedback.",
                                }
                            )
                        elif diverging_pulses >= 3:
                            abort_reason = (
                                "heading-guided cruise repeatedly increased distance to patrol target "
                                f"(before {current_distance:.2f}, after {after_distance:.2f})"
                            )
                            break
                    else:
                        # Fresh frame IDs are not movement evidence. The
                        # landed 10:37:20 run kept sending forward pulses while
                        # the rendered room position lagged behind the real
                        # drone all the way to Point 3. Count unchanged model
                        # positions as stale and lock physical motion quickly.
                        if stale_motion_count == 0:
                            stale_motion_reference_position = list(current_position)
                        stale_motion_count += 1
                    if stale_motion_count >= 3:
                        # TSolve publishes asynchronously with respect to DJI
                        # RC pulses.  Three individual post-command samples can
                        # each move less than 1.5 cm even while their combined
                        # window clearly advances.  Judge the whole window
                        # before declaring localization stale.  The 16:44:59
                        # run advanced about 0.65 m toward Point 1 but entered
                        # recovery because this cumulative check was missing.
                        if published_position_advanced_toward_target(
                            stale_motion_reference_position,
                            after_position,
                            target,
                        ):
                            stale_motion_count = 0
                            stale_motion_reference_position = None
                            diverging_pulses = 0
                            closest_distance = min(closest_distance, after_distance)
                            continue
                        recovered_progress_gate = wait_for_pose_recovery(
                            "translation_progress",
                            (
                                "fresh camera frames did not advance the published "
                                "ATLAS room position after three translation pulses"
                            ),
                            require_translation_safe=True,
                            translation_target=target,
                            translation_reference_distance=current_distance,
                            translation_arrival_radius=cruise_soft_arrival_radius,
                            endpoint_leg_index=endpoint_leg_index,
                            require_observed_translation_progress=True,
                        )
                        if recovered_progress_gate is None:
                            break
                        stale_motion_count = 0
                        stale_motion_reference_position = None
                        # Recompute distance, heading, and arrival from the
                        # recovered authoritative pose before any new command.
                        continue
                if abort_reason:
                    break
                if not reached:
                    final_endpoint_ready = bool(
                        not precise_arrival
                        or taught_endpoint_arrival_verified(
                            (last_pose_gate or {}).get("pose"),
                            expected_leg_index=endpoint_leg_index,
                        )
                    )
                    final_visual_checkpoint_arrival = bool(
                        precise_arrival
                        and taught_endpoint_stale_translation_arrival_verified(
                            (last_pose_gate or {}).get("pose"),
                            expected_leg_index=endpoint_leg_index,
                        )
                    )
                    if final_visual_checkpoint_arrival:
                        reached = True
                        arrival_mode = "visual_checkpoint_endpoint_verified_timeout"
                        active_route_translation_locked = True
                        active_route_position_anchor = list(target)
                        yaw_position_anchor = list(target)
                        if endpoint_leg_index == 4:
                            last_pose_gate = commit_verified_point_one_handoff(
                                last_pose_gate,
                                target,
                            ) or last_pose_gate
                        publish_progress(
                            {
                                "phase": "cruise_arrival",
                                "step_index": idx,
                                "step_title": title,
                                "target": target,
                                "distance_to_target": final_distance,
                                "translation_locked": True,
                                "position_anchor": target,
                                "endpoint_leg_index": endpoint_leg_index,
                                "visual_checkpoint_arrival": True,
                                "verified_endpoint_pose_committed": bool(
                                    endpoint_leg_index == 4
                                ),
                                "message": (
                                    "Cruise ended with repeated independent endpoint "
                                    "verification; stopping at the checkpoint without "
                                    "another forward pulse."
                                ),
                            }
                        )
                        neutral_hover(drone, 0.25)
                    elif (
                        isinstance(final_distance, (int, float))
                        and final_distance <= cruise_soft_arrival_radius
                        and final_endpoint_ready
                    ):
                        reached = True
                        arrival_mode = "timeout_soft_deadband"
                        publish_progress(
                            {
                                "phase": "cruise_arrival",
                                "step_index": idx,
                                "step_title": title,
                                "target": target,
                                "distance_to_target": final_distance,
                                "arrival_radius": cruise_arrival_radius,
                                "soft_arrival_radius": cruise_soft_arrival_radius,
                                "message": (
                                    f"Patrol cruise timed out near target but inside soft deadband "
                                    f"({final_distance:.2f} <= {cruise_soft_arrival_radius:.2f}); accepting and hovering."
                                ),
                            }
                        )
                        neutral_hover(drone, 0.25)
                    elif (
                        closest_distance < float("inf")
                        and closest_distance <= cruise_soft_arrival_radius
                        and final_endpoint_ready
                    ):
                        reached = True
                        arrival_mode = "closest_soft_deadband"
                        publish_progress(
                            {
                                "phase": "cruise_arrival",
                                "step_index": idx,
                                "step_title": title,
                                "target": target,
                                "distance_to_target": closest_distance,
                                "arrival_radius": cruise_arrival_radius,
                                "soft_arrival_radius": cruise_soft_arrival_radius,
                                "message": (
                                    f"Patrol cruise passed within soft deadband "
                                    f"({closest_distance:.2f} <= {cruise_soft_arrival_radius:.2f}); accepting and hovering."
                                ),
                            }
                        )
                        neutral_hover(drone, 0.25)
                if not reached:
                    final_text = f"{final_distance:.2f}" if isinstance(final_distance, (int, float)) else "unknown"
                    initial_text = f"{initial_distance:.2f}" if isinstance(initial_distance, (int, float)) else "unknown"
                    closest_text = f"{closest_distance:.2f}" if closest_distance < float("inf") else "unknown"
                    abort_reason = (
                        "closed-loop cruise timed out before reaching patrol target "
                        f"(target radius {cruise_arrival_radius:.2f}, soft radius {cruise_soft_arrival_radius:.2f}, "
                        f"initial distance {initial_text}, closest distance {closest_text}, final distance {final_text}, "
                        f"pulses {pulse_count}, forward {forward_pulse_count}, lateral {lateral_pulse_count}, "
                        f"mode {last_navigation_mode}, processed {last_processed_count}, pose age {last_pose_age})"
                    )
                    skipped.append(
                        {
                            "index": idx,
                            "type": kind,
                            "title": title,
                            "reason": abort_reason,
                            "target": target,
                            "initial_distance": initial_distance,
                            "closest_distance": closest_distance if closest_distance < float("inf") else None,
                            "final_distance": final_distance,
                            "pulse_count": pulse_count,
                            "forward_pulse_count": forward_pulse_count,
                            "lateral_pulse_count": lateral_pulse_count,
                            "navigation_mode": last_navigation_mode,
                            "processed_count": last_processed_count,
                            "pose_age_seconds": last_pose_age,
                        }
                    )
                    break
                if reached and yaw_pulse_count > 0 and not yaw_sign_verified:
                    # Completing a closed-loop segment after yaw control is
                    # sufficient proof of the fixed DJI yaw polarity.  Never
                    # let a later delayed frame reverse it mid-mission.
                    yaw_sign_verified = True
                    rc_summary["adaptive_axis"]["yaw_sign_verified"] = True
                if reached and dynamic_lap_reentry:
                    # The repeated lap has now independently localized and
                    # physically entered the saved Point 1.  Do not run the
                    # former implicit checkpoint that stretched Point 1->2
                    # from the previous lap's different endpoint.
                    lap_metric_checkpoint_pending = False
                    lap_reentry_metric_ready = False
                    active_route_translation_locked = True
                    active_route_position_anchor = list(target)
                    publish_progress(
                        {
                            "phase": "lap_point1_entry_verified",
                            "lap": current_lap_number,
                            "translation_locked": True,
                            "position_anchor": target,
                            "saved_point1": target,
                            "distance_to_target": final_distance,
                            "arrival_mode": arrival_mode,
                            "message": (
                                f"Lap {current_lap_number}: fresh global localization "
                                "and guarded Point-1 entry completed; Point 1->2 may begin."
                            ),
                        }
                    )
                executed.append(
                    {
                        "index": idx,
                        "type": kind,
                        "title": title,
                        "closed_loop": True,
                        "target": target,
                        "distance": distance,
                        "planned_duration_s": planned_duration,
                        "arrival_radius": cruise_arrival_radius,
                        "soft_arrival_radius": cruise_soft_arrival_radius,
                        "endpoint_leg_index": endpoint_leg_index,
                        "patrol_stage": cruise_stage,
                        "arrival_mode": arrival_mode,
                        "initial_distance": initial_distance,
                        "final_distance": final_distance,
                        "closest_distance": closest_distance if closest_distance < float("inf") else None,
                        "reached": reached,
                        "pulse_count": pulse_count,
                        "yaw_pulse_count": yaw_pulse_count,
                        "forward_pulse_count": forward_pulse_count,
                        "lateral_pulse_count": lateral_pulse_count,
                    }
                )
                continue

            if kind == "descend":
                duration = clamp_float(step.get("duration_s"), pulse_seconds, pulse_seconds, max_step_seconds)
                pulses = max(1, int(round(duration / pulse_seconds)))
                for _ in range(pulses):
                    pulse_gate = pose_gate_or_abort()
                    if pulse_gate is None or not enforce_patrol_geofence(pulse_gate, f"descent step {idx}"):
                        break
                    sent = execute_rc_pulse(drone, du=-max_vertical_rc, seconds=pulse_seconds)
                    record_pulse("vertical", sent)
                    executed_pulses += 1
                if abort_reason:
                    break
                executed.append({"index": idx, "type": kind, "title": title, "duration_s": duration, "pulse_count": pulses})
                continue

            if kind == "land":
                neutral_hover(drone, 0.4)
                if pose_gate_or_abort() is None:
                    break
                result = drone.land(True)
                executed.append({"index": idx, "type": kind, "title": title, "result": result})
                continue

            skipped.append({"index": idx, "type": kind or "unknown", "title": title, "reason": "unsupported mission step"})
        if abort_reason is None and patrol_loop and patrol_laps > 0:
            active_route_translation_locked = True
            active_route_position_anchor = pose_gate_position(last_pose_gate)
            publish_progress(
                {
                    "phase": "patrol_complete",
                    "lap": patrol_laps,
                    "completed_laps": patrol_laps,
                    "message": (
                        f"Completed exactly {patrol_laps} patrol circles; holding position in hover."
                    ),
                }
            )
            neutral_hover(drone, 0.30)
    finally:
        try:
            neutral_hover(drone, 0.08)
        except Exception:
            pass
        if control_enabled:
            try:
                rc_summary["disable_control_result"] = drone.disableControl(True)
            except Exception:
                pass

    return {
        "ok": abort_reason is None,
        "armed": True,
        "physical_motion_locked": False,
        "aborted": abort_reason is not None,
        "abort_reason": abort_reason,
        "executed_hover_steps": hover_count,
        "executed_pulses": executed_pulses,
        "last_pose_gate": last_pose_gate,
        "guided_settings": {
            "pose_max_age_seconds": pose_max_age,
            "pose_recovery_seconds": pose_recovery_seconds,
            "continuous_relocalization": continuous_relocalization,
            "one_pulse_pose_confirmation": one_pulse_pose_confirmation,
            "smooth_continuous_cruise": smooth_continuous_cruise,
            "cruise_window_seconds": cruise_window_seconds,
            "cruise_pose_watchdog_seconds": cruise_pose_watchdog_seconds,
            "max_unverified_translation_m": max_unverified_translation_m,
            "localization_recovery_hover_seconds": total_pose_recovery_pause_seconds,
            "pulse_seconds": pulse_seconds,
            "max_forward_rc": max_forward_rc,
            "max_lateral_rc": max_lateral_rc,
            "allow_lateral_rc": allow_lateral_rc,
            "allow_axis_auto_calibration": allow_axis_auto_calibration,
            "max_yaw_rc": max_yaw_rc,
            "max_scan_yaw_rc": max_scan_yaw_rc,
            "allow_patrol_scan_yaw": allow_patrol_scan_yaw,
            "alignment_grace_seconds": alignment_grace_seconds,
            "max_vertical_rc": max_vertical_rc,
            "max_step_seconds": max_step_seconds,
            "arrival_radius_map_units": arrival_radius,
            "arrival_deadband_map_units": arrival_deadband,
            "strict_entry_arrival": strict_entry_arrival,
            "max_cruise_seconds": max_cruise_seconds,
            "max_pose_step_map_units": max_pose_step,
            "max_pose_step_hard_map_units": max_pose_step_hard,
            "max_cross_track_map_units": max_cross_track,
            "cross_track_recovery_start_map_units": cross_track_recovery_start,
            "patrol_loop": patrol_loop,
            "safety_motion_buffer_m": safety_motion_buffer,
            "heading_trim_deg": heading_trim_deg,
            "operator_heading_calibrated": operator_heading_calibrated,
            "initial_body_heading_offset_deg": initial_body_heading_offset_deg,
            "require_physical_forward_probe": require_physical_forward_probe,
            "max_heading_calibration_error_deg": max_heading_calibration_error_deg,
        },
        "rc_summary": rc_summary,
        "executed": executed,
        "skipped": skipped,
        "command_count": len(commands),
        "message": "Guided mission executed with TSolve pose gating." if abort_reason is None else f"Guided mission aborted: {abort_reason}",
        "elapsed_seconds": time.time() - started,
    }


def resize_keep_aspect(frame, max_size: int):  # noqa: ANN001
    if max_size <= 0:
        return frame
    h, w = frame.shape[:2]
    long_side = max(h, w)
    if long_side <= max_size:
        return frame
    scale = max_size / float(long_side)
    return cv2.resize(
        frame,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def load_enemy_detector(model_path: Path):
    config_dir = ROOT / "viewer" / "public" / "live_dji" / "ultralytics_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("Ultralytics is not installed; train/install YOLO before live enemy detection.") from exc
    return YOLO(str(model_path))


def detect_enemy_drones(detector, image, confidence: float) -> list[dict[str, Any]]:  # noqa: ANN001
    results = detector.predict(source=image, conf=confidence, verbose=False)
    names = getattr(detector, "names", {}) or {}
    detections: list[dict[str, Any]] = []
    if not results:
        return detections
    boxes = getattr(results[0], "boxes", None)
    if boxes is None:
        return detections
    xyxy = boxes.xyxy.cpu().numpy() if getattr(boxes, "xyxy", None) is not None else []
    confs = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else []
    classes = boxes.cls.cpu().numpy() if getattr(boxes, "cls", None) is not None else []
    h, w = image.shape[:2]
    for raw_box, conf, cls in zip(xyxy, confs, classes):
        x1, y1, x2, y2 = [float(v) for v in raw_box]
        class_id = int(cls)
        if isinstance(names, dict):
            class_name = str(names.get(class_id, f"enemy_{class_id}"))
        elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            class_name = str(names[class_id])
        else:
            class_name = f"enemy_{class_id}"
        detections.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": float(conf),
                "box": {
                    "x1": x1 / max(1, w),
                    "y1": y1 / max(1, h),
                    "x2": x2 / max(1, w),
                    "y2": y2 / max(1, h),
                    "width": max(0.0, x2 - x1) / max(1, w),
                    "height": max(0.0, y2 - y1) / max(1, h),
                },
            }
        )
    return detections


def open_frame_csv(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    handle = path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "image_name",
            "source_frame",
            "time_sec",
            "width",
            "height",
            "received_unix",
        ],
    )
    if not exists:
        writer.writeheader()
        handle.flush()
    return handle, writer


def load_opendji(opendji_root: Path):
    opendji_path = opendji_root / "OpenDJI.py"
    if not opendji_path.exists():
        raise FileNotFoundError(
            f"OpenDJI.py was not found in {opendji_root}. "
            "Pass --opendji-root /path/to/DJI-MSDK-to-PC-main."
        )
    sys.path.insert(0, str(opendji_root))
    try:
        from OpenDJI import OpenDJI  # type: ignore

        return OpenDJI
    except TypeError as exc:
        if "unsupported operand type(s) for |" not in str(exc):
            raise

    # The public MSDK-to-PC OpenDJI.py uses Python 3.10 union annotations
    # such as `str | None`.  Our bundled ATLAS venv may be Python 3.9, so load
    # the file with postponed annotation evaluation without editing the DJI repo.
    sys.modules.pop("OpenDJI", None)
    module = types.ModuleType("OpenDJI")
    module.__file__ = str(opendji_path)
    module.__package__ = ""
    source = opendji_path.read_text(encoding="utf-8")
    code = compile("from __future__ import annotations\n" + source, str(opendji_path), "exec")
    exec(code, module.__dict__)
    sys.modules["OpenDJI"] = module
    return module.OpenDJI


def existing_frame_count(query_dir: Path) -> int:
    if not query_dir.exists():
        return 0
    return len([p for p in query_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Receive DJI Mini live frames from MSDKRemote and expose them to ATLAS."
    )
    ap.add_argument("--phone-ip", required=True, help="IP shown inside the Android MSDKRemote app.")
    ap.add_argument("--opendji-root", type=Path, default=DEFAULT_OPENDJI_ROOT)
    ap.add_argument("--session", default="", help="Optional stable session id. Default: timestamp.")
    ap.add_argument("--fps", type=float, default=10.0, help="Frame sampling rate written to ATLAS.")
    ap.add_argument("--max-size", type=int, default=1200, help="Resize longest side before saving.")
    ap.add_argument("--jpeg-quality", type=int, default=88)
    ap.add_argument("--max-frames", type=int, default=0, help="0 means run until Ctrl-C.")
    ap.add_argument("--show", action="store_true", help="Open a local OpenCV preview window.")
    ap.add_argument("--no-history", action="store_true", help="Only update latest.jpg/status.json.")
    ap.add_argument("--out-root", type=Path, default=ROOT / "data" / "dji_live")
    ap.add_argument("--public-root", type=Path, default=ROOT / "viewer" / "public" / "live_dji")
    ap.add_argument("--pose-stream", type=Path, default=None, help="Live poses_partial.json used as the guided-flight freshness gate.")
    ap.add_argument(
        "--view-only",
        action="store_true",
        help="Receive video and publish localization, but never consume DJI flight commands.",
    )
    ap.add_argument("--enemy-model", type=Path, default=None, help="Optional trained YOLO model used for live enemy-drone detection.")
    ap.add_argument("--enemy-output", type=Path, default=None, help="Browser-visible enemy detection JSON output.")
    ap.add_argument("--enemy-control", type=Path, default=None, help="Runtime JSON gate for enabling enemy inference.")
    ap.add_argument("--enemy-detect-fps", type=float, default=5.0, help="Maximum detector rate for moving-target visual servoing.")
    ap.add_argument("--enemy-conf", type=float, default=0.35, help="YOLO confidence threshold for enemy-drone detections.")
    args = ap.parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    args.jpeg_quality = max(1, min(100, int(args.jpeg_quality)))
    session = args.session.strip() or time.strftime("dji_live_%Y%m%d_%H%M%S")
    session_root = args.out_root / session
    query_dir = session_root / "query_frames"
    latest_path = args.public_root / "latest.jpg"
    public_status_path = args.public_root / "status.json"
    session_status_path = session_root / "status.json"
    enemy_output_path = args.enemy_output or (args.public_root / "enemy_detections.json")
    enemy_control_path = args.enemy_control or (args.public_root / "enemy_detection_control.json")
    control_command_path = args.public_root / "control_command.json"
    control_status_path = args.public_root / "control_status.json"
    control_history_path = args.public_root / "control_status_history.jsonl"
    session_control_trace_path = session_root / "control_trace.jsonl"
    frames_csv = query_dir / "frames.csv"
    metadata_path = session_root / "metadata.json"

    session_root.mkdir(parents=True, exist_ok=True)
    query_dir.mkdir(parents=True, exist_ok=True)
    args.public_root.mkdir(parents=True, exist_ok=True)

    OpenDJI = load_opendji(args.opendji_root.resolve())
    stop = StopFlag()
    signal.signal(signal.SIGINT, stop.handler)
    signal.signal(signal.SIGTERM, stop.handler)

    metadata = {
        "mode": "dji_msdk_live_video_bridge",
        "session": session,
        "phone_ip": args.phone_ip,
        "fps": args.fps,
        "max_size": args.max_size,
        "jpeg_quality": args.jpeg_quality,
        "query_frames": str(query_dir),
        "public_latest": str(latest_path),
        "started_at": time.time(),
        "control_enabled": not args.view_only,
        "view_only": bool(args.view_only),
        "control_command_path": str(control_command_path),
        "control_status_path": str(control_status_path),
        "control_history_path": str(control_history_path),
        "session_control_trace_path": str(session_control_trace_path),
        "pose_stream_path": str(args.pose_stream) if args.pose_stream else None,
        "enemy_model": str(args.enemy_model) if args.enemy_model else None,
        "enemy_output": str(enemy_output_path),
        "enemy_control": str(enemy_control_path),
        "enemy_detect_fps": args.enemy_detect_fps,
        "enemy_confidence": args.enemy_conf,
        "runtime_fingerprints": {
            "atlas_dji_live_bridge.py": file_sha256(Path(__file__).resolve()),
            "atlas_app_server.py": file_sha256(ROOT / "scripts" / "atlas_app_server.py"),
            "run_bounded_tsolve_video_stream.py": file_sha256(
                ROOT / "scripts" / "run_bounded_tsolve_video_stream.py"
            ),
            "patrol_visual_route_recovery.py": file_sha256(
                ROOT / "scripts" / "patrol_visual_route_recovery.py"
            ),
            "config.json": file_sha256(ROOT / "config.json"),
        },
        "note": (
            "Live Check mode: video and localization only; DJI flight commands are disabled."
            if args.view_only
            else "This bridge receives frames and accepts explicit takeoff/land/hover commands."
        ),
    }
    atomic_write_json(metadata_path, metadata)

    status = {
        **metadata,
        "status": "connecting",
        "frames_saved": existing_frame_count(query_dir),
        "latest_frame": None,
        "latest_public_url": "public/live_dji/latest.jpg",
        "message": "Connecting to Android MSDKRemote.",
        "updated_at": time.time(),
    }
    atomic_write_json(public_status_path, status)
    atomic_write_json(session_status_path, status)

    frame_handle, frame_writer = open_frame_csv(frames_csv)
    next_capture_time = 0.0
    next_enemy_detection_time = 0.0
    frame_index = existing_frame_count(query_dir)
    first_frame_time: float | None = None
    last_shape: tuple[int, int] | None = None
    last_progress_print = 0.0
    last_stale_status_publish = 0.0
    last_seen_decoded_sequence = 0
    last_control_id: str | None = None
    enemy_detector = None
    enemy_detection_error = ""
    enemy_detection_was_enabled = False
    if args.enemy_model:
        try:
            enemy_detector = load_enemy_detector(args.enemy_model)
            atomic_write_json(
                enemy_output_path,
                {
                    "status": "ready",
                    "message": "Enemy-drone detector loaded.",
                    "model": str(args.enemy_model),
                    "detections": [],
                    "updated_at": time.time(),
                },
            )
        except Exception as exc:
            enemy_detection_error = str(exc)
            atomic_write_json(
                enemy_output_path,
                {
                    "status": "error",
                    "message": enemy_detection_error,
                    "model": str(args.enemy_model),
                    "detections": [],
                    "updated_at": time.time(),
                },
            )
    else:
        atomic_write_json(
            enemy_output_path,
            {
                "status": "disabled",
                "message": "No trained enemy-drone detector selected.",
                "detections": [],
                "updated_at": time.time(),
            },
        )
    if control_command_path.exists():
        try:
            last_control_id = str(json.loads(control_command_path.read_text(encoding="utf-8")).get("id") or "") or None
        except Exception:
            last_control_id = None

    print("ATLAS DJI live bridge")
    print(f"  phone IP:      {args.phone_ip}")
    print(f"  OpenDJI root:  {args.opendji_root.resolve()}")
    print(f"  session root:  {session_root}")
    print(f"  query frames:  {query_dir}")
    print(f"  public status: {public_status_path}")
    print(f"  control:       {control_command_path}")
    print(f"  pose stream:   {args.pose_stream or 'disabled'}")
    print(f"  enemy model:   {args.enemy_model if args.enemy_model else 'disabled'}")
    if enemy_detection_error:
        print(f"  enemy error:   {enemy_detection_error}")
    print("Press Ctrl-C to stop.\n")

    try:
        with OpenDJI(args.phone_ip) as drone:
            decoded_frames = DecodedFrameListener()
            drone.frameListener(decoded_frames)
            status.update(
                {
                    "status": "waiting_for_video",
                    "message": "Connected. Waiting for decoded DJI video frames.",
                    "updated_at": time.time(),
                }
            )
            atomic_write_json(public_status_path, status)
            atomic_write_json(session_status_path, status)
            control_lock = threading.Lock()
            control_busy = False
            control_thread: threading.Thread | None = None
            mission_cancel = threading.Event()

            def control_worker(command_payload: dict[str, Any]) -> None:
                nonlocal control_busy
                try:
                    command_name = str(command_payload.get("command") or "").strip().lower()
                    command_id = command_payload.get("id")
                    control_started = {
                        "ok": True,
                        "id": command_id,
                        "command": command_name,
                        "status": "running",
                        "message": f"{command_name or 'command'} accepted by DJI bridge and is now executing.",
                        "started_at": time.time(),
                        "updated_at": time.time(),
                    }
                    atomic_write_json(control_status_path, control_started)
                    append_jsonl(control_history_path, {**control_started, "event": "started"})
                    append_jsonl(
                        session_control_trace_path,
                        {
                            **control_started,
                            "event": "started",
                            "command_payload": command_payload,
                        },
                    )
                    with control_lock:
                        status["last_control"] = control_started
                        status["updated_at"] = time.time()
                        atomic_write_json(public_status_path, status)
                        atomic_write_json(session_status_path, status)
                    if command_name == "mission":
                        mission_cancel.clear()
                    if command_name in {"takeoff", "mission"}:
                        video_issue = live_video_motion_safety_issue(
                            decoded_frames.age_seconds()
                        )
                        if video_issue is not None:
                            raise RuntimeError(video_issue)

                    def publish_progress(progress: dict[str, Any]) -> None:
                        control_progress = {
                            **control_started,
                            "status": "running",
                            "progress": progress,
                            "message": progress.get("message") or control_started["message"],
                            "updated_at": time.time(),
                        }
                        atomic_write_json(control_status_path, control_progress)
                        append_jsonl(
                            session_control_trace_path,
                            {**control_progress, "event": "progress"},
                        )
                        with control_lock:
                            status["last_control"] = control_progress
                            status["updated_at"] = time.time()
                            atomic_write_json(public_status_path, status)
                            atomic_write_json(session_status_path, status)

                    control_result = execute_control_command(
                        drone,
                        OpenDJI,
                        command_payload,
                        pose_stream_path=args.pose_stream,
                        enemy_detection_path=enemy_output_path,
                        stop_flag=stop,
                        mission_stop_event=mission_cancel,
                        progress_callback=publish_progress,
                    )
                    atomic_write_json(control_status_path, control_result)
                    append_jsonl(control_history_path, {**control_result, "event": "finished"})
                    append_jsonl(
                        session_control_trace_path,
                        {**control_result, "event": "finished"},
                    )
                    with control_lock:
                        status["last_control"] = control_result
                        status["updated_at"] = time.time()
                        atomic_write_json(public_status_path, status)
                        atomic_write_json(session_status_path, status)
                    print(f"control={control_result['command']} result={control_result.get('result')}")
                except ControlLinkSafetyError as exc:
                    traceback.print_exc()
                    control_result = {
                        "ok": False,
                        "id": command_payload.get("id"),
                        "command": command_payload.get("command"),
                        "status": "safety_abort",
                        "safety_critical": True,
                        "neutral_confirmed": exc.neutral_confirmed,
                        "control_link_recovered": exc.control_link_recovered,
                        "requires_relocalization": exc.requires_relocalization,
                        "message": str(exc),
                        "error": str(exc),
                        "updated_at": time.time(),
                    }
                    atomic_write_json(control_status_path, control_result)
                    append_jsonl(control_history_path, {**control_result, "event": "safety_abort"})
                    append_jsonl(
                        session_control_trace_path,
                        {**control_result, "event": "safety_abort"},
                    )
                    with control_lock:
                        status["last_control"] = control_result
                        status["updated_at"] = time.time()
                        atomic_write_json(public_status_path, status)
                        atomic_write_json(session_status_path, status)
                except Exception as exc:
                    traceback.print_exc()
                    control_result = {
                        "ok": False,
                        "id": command_payload.get("id"),
                        "command": command_payload.get("command"),
                        "error": str(exc),
                        "updated_at": time.time(),
                    }
                    atomic_write_json(control_status_path, control_result)
                    append_jsonl(control_history_path, {**control_result, "event": "error"})
                    append_jsonl(
                        session_control_trace_path,
                        {**control_result, "event": "error"},
                    )
                    with control_lock:
                        status["last_control"] = control_result
                        status["updated_at"] = time.time()
                        atomic_write_json(public_status_path, status)
                        atomic_write_json(session_status_path, status)
                finally:
                    with control_lock:
                        control_busy = False

            while not stop.stop:
                if not args.view_only and control_command_path.exists():
                    try:
                        command_payload = json.loads(control_command_path.read_text(encoding="utf-8"))
                        command_id = str(command_payload.get("id") or "")
                        if command_id and command_id != last_control_id:
                            last_control_id = command_id
                            with control_lock:
                                is_busy = control_busy
                                if not control_busy:
                                    control_busy = True
                            if is_busy:
                                command_name = str(command_payload.get("command") or "").strip().lower()
                                if command_name == "hover" and command_payload.get("emergency_stop"):
                                    mission_cancel.set()
                                    control_result = {
                                        "ok": True,
                                        "id": command_id,
                                        "command": command_name,
                                        "emergency_stop": True,
                                        "message": (
                                            "Emergency hover requested; cancellation is latched and the active "
                                            "pulse will neutralize within its 0.30 second bound."
                                        ),
                                        "error": None,
                                        "updated_at": time.time(),
                                    }
                                else:
                                    control_result = {
                                        "ok": False,
                                        "id": command_id,
                                        "command": command_payload.get("command"),
                                        "error": "another DJI command is already running",
                                        "updated_at": time.time(),
                                    }
                                atomic_write_json(control_status_path, control_result)
                                append_jsonl(
                                    control_history_path,
                                    {
                                        **control_result,
                                        "event": (
                                            "emergency_stop"
                                            if control_result.get("emergency_stop")
                                            else "rejected"
                                        ),
                                    },
                                )
                                append_jsonl(
                                    session_control_trace_path,
                                    {
                                        **control_result,
                                        "event": (
                                            "emergency_stop"
                                            if control_result.get("emergency_stop")
                                            else "rejected"
                                        ),
                                        "command_payload": command_payload,
                                    },
                                )
                                with control_lock:
                                    status["last_control"] = control_result
                                    status["updated_at"] = time.time()
                                    atomic_write_json(public_status_path, status)
                                    atomic_write_json(session_status_path, status)
                            else:
                                control_thread = threading.Thread(
                                    target=control_worker,
                                    args=(command_payload,),
                                    daemon=True,
                                )
                                control_thread.start()
                    except Exception as exc:
                        control_result = {
                            "ok": False,
                            "id": command_payload.get("id") if "command_payload" in locals() else None,
                            "error": str(exc),
                            "updated_at": time.time(),
                        }
                        atomic_write_json(control_status_path, control_result)
                        append_jsonl(control_history_path, {**control_result, "event": "error"})
                        append_jsonl(
                            session_control_trace_path,
                            {**control_result, "event": "error"},
                        )
                        status["last_control"] = control_result
                        atomic_write_json(public_status_path, status)
                        atomic_write_json(session_status_path, status)
                now = time.perf_counter()
                decoded_sequence, frame, decoded_unix = decoded_frames.latest()
                if frame is None or decoded_sequence <= 0:
                    if time.perf_counter() - last_progress_print > 2.0:
                        print("waiting for first frame...")
                        last_progress_print = time.perf_counter()
                    time.sleep(0.03)
                    continue

                if decoded_sequence == last_seen_decoded_sequence:
                    stale_seconds = decoded_frames.age_seconds()
                    if (
                        stale_seconds is not None
                        and stale_seconds >= 1.0
                        and time.perf_counter() - last_stale_status_publish >= 1.0
                    ):
                        status.update(
                            {
                                "status": "video_stale",
                                "message": (
                                    "DJI connection is open, but the Android video decoder "
                                    f"has produced no new frame for {stale_seconds:.1f}s. "
                                    "Cached frames are not being sent to TSolve."
                                ),
                                "decoded_frame_sequence": decoded_sequence,
                                "last_new_frame_unix": decoded_unix,
                                "video_stale_seconds": stale_seconds,
                                "updated_at": time.time(),
                            }
                        )
                        atomic_write_json(public_status_path, status)
                        atomic_write_json(session_status_path, status)
                        last_stale_status_publish = time.perf_counter()
                        print(
                            f"video stale: no newly decoded DJI frame for {stale_seconds:.1f}s",
                            flush=True,
                        )
                    time.sleep(0.01)
                    continue

                # This sequence changes only on an actual H264 decoder callback.
                # It is safe to sample or skip this fresh frame according to the
                # requested ATLAS FPS, but never publish it more than once.
                last_seen_decoded_sequence = decoded_sequence

                if first_frame_time is None:
                    first_frame_time = time.perf_counter()
                    status["first_frame_latency_sec"] = first_frame_time - now

                if now < next_capture_time:
                    if args.show:
                        preview = resize_keep_aspect(frame, 900)
                        cv2.imshow("ATLAS DJI live stream", preview)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    time.sleep(0.005)
                    continue
                next_capture_time = now + (1.0 / args.fps)

                image = resize_keep_aspect(frame, args.max_size)
                h, w = image.shape[:2]
                last_shape = (w, h)
                ok, encoded = cv2.imencode(
                    ".jpg",
                    image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality],
                )
                if not ok:
                    raise RuntimeError("OpenCV failed to encode DJI frame as JPEG")

                jpg = encoded.tobytes()
                elapsed_sec = 0.0 if first_frame_time is None else time.perf_counter() - first_frame_time
                image_name = f"query_{frame_index:06d}.jpg"

                enemy_detection_summary = None
                enemy_detection_is_enabled = enemy_detection_control_enabled(enemy_control_path)
                if enemy_detection_was_enabled and not enemy_detection_is_enabled:
                    atomic_write_json(
                        enemy_output_path,
                        {
                            "status": "disabled",
                            "message": "Enemy detection is disabled by the operator.",
                            "model": str(args.enemy_model) if args.enemy_model else None,
                            "detections": [],
                            "updated_at": time.time(),
                        },
                    )
                elif enemy_detection_is_enabled and not enemy_detection_was_enabled:
                    next_enemy_detection_time = 0.0
                enemy_detection_was_enabled = enemy_detection_is_enabled
                if enemy_detector is not None and enemy_detection_is_enabled and now >= next_enemy_detection_time:
                    next_enemy_detection_time = now + (1.0 / max(0.05, float(args.enemy_detect_fps)))
                    try:
                        detections = detect_enemy_drones(enemy_detector, image, float(args.enemy_conf))
                        enemy_detection_summary = {
                            "status": "detected" if detections else "clear",
                            "message": f"{len(detections)} enemy drone candidate(s)" if detections else "No enemy drone detected.",
                            "model": str(args.enemy_model),
                            "frame": image_name,
                            "time_sec": elapsed_sec,
                            "width": w,
                            "height": h,
                            "detections": detections,
                            "updated_at": time.time(),
                        }
                        atomic_write_json(enemy_output_path, enemy_detection_summary)
                    except Exception as exc:
                        enemy_detection_summary = {
                            "status": "error",
                            "message": str(exc),
                            "model": str(args.enemy_model),
                            "frame": image_name,
                            "time_sec": elapsed_sec,
                            "detections": [],
                            "updated_at": time.time(),
                        }
                        atomic_write_json(enemy_output_path, enemy_detection_summary)

                if not args.no_history:
                    frame_path = query_dir / image_name
                    atomic_write_bytes(frame_path, jpg)
                    frame_writer.writerow(
                        {
                            "image_name": image_name,
                            "source_frame": frame_index,
                            "time_sec": f"{elapsed_sec:.6f}",
                            "width": w,
                            "height": h,
                            "received_unix": f"{time.time():.6f}",
                        }
                    )
                    frame_handle.flush()

                atomic_write_bytes(latest_path, jpg)
                frame_index += 1
                status.update(
                    {
                        "status": "streaming",
                        "message": "Receiving DJI live frames. Ready for ATLAS live localization consumer.",
                        "frames_saved": frame_index,
                        "latest_frame": image_name,
                        "latest_width": w,
                        "latest_height": h,
                        "latest_time_sec": elapsed_sec,
                        "decoded_frame_sequence": decoded_sequence,
                        "last_new_frame_unix": decoded_unix,
                        "video_stale_seconds": 0.0,
                        "query_frames": str(query_dir),
                        "frames_csv": str(frames_csv),
                        "enemy_detection": enemy_detection_summary,
                        "updated_at": time.time(),
                    }
                )
                atomic_write_json(public_status_path, status)
                atomic_write_json(session_status_path, status)

                if time.perf_counter() - last_progress_print > 1.0:
                    print(
                        f"frames={frame_index:05d} "
                        f"latest={image_name} "
                        f"t={elapsed_sec:7.2f}s "
                        f"size={w}x{h}"
                    )
                    last_progress_print = time.perf_counter()

                if args.show:
                    cv2.imshow("ATLAS DJI live stream", image)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if args.max_frames > 0 and frame_index >= args.max_frames:
                    break

            # Keep DJI sockets open until an active mission observes
            # cancellation, sends neutral hover, and disables virtual-stick
            # control. Closing the OpenDJI context first causes Errno 9 in the
            # worker and leaves the UI with a stale bridge heartbeat.
            mission_cancel.set()
            if control_thread is not None and control_thread.is_alive():
                control_thread.join(timeout=2.5)
    except Exception as exc:
        status.update(
            {
                "status": "error",
                "message": str(exc),
                "frames_saved": frame_index,
                "last_frame_shape": last_shape,
                "updated_at": time.time(),
            }
        )
        atomic_write_json(public_status_path, status)
        atomic_write_json(session_status_path, status)
        raise
    finally:
        frame_handle.close()
        if args.show:
            cv2.destroyAllWindows()
        final_status = {
            **status,
            "status": "stopped" if status.get("status") != "error" else "error",
            "message": (
                "DJI live bridge stopped."
                if status.get("status") != "error"
                else status.get("message")
            ),
            "frames_saved": frame_index,
            "stopped_at": time.time(),
            "updated_at": time.time(),
        }
        atomic_write_json(public_status_path, final_status)
        atomic_write_json(session_status_path, final_status)
        print(f"\nStopped. Frames saved: {frame_index}")
        print(f"ATLAS query frame bank: {query_dir}")


if __name__ == "__main__":
    main()
