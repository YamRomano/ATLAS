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
import json
import math
import os
import re
import signal
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any, Callable

import cv2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPENDJI_ROOT = Path("/Users/yamromano/Desktop/DJI-MSDK-to-PC-main")
IMAGE_EXTS = {".jpg", ".jpeg"}
TAKEOFF_VERTICAL_SPEED = 0.03
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


class StopFlag:
    def __init__(self) -> None:
        self.stop = False

    def handler(self, signum, frame) -> None:  # noqa: ANN001
        self.stop = True


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


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
            drone.move(0.0, TAKEOFF_VERTICAL_SPEED, 0.0, 0.0, False)
            steps += 1
            time.sleep(TAKEOFF_STEP_SECONDS)
    finally:
        try:
            drone.move(0.0, 0.0, 0.0, 0.0, True)
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


def latest_tsolve_pose_gate(
    pose_stream_path: Path | None,
    max_age_seconds: float,
    max_recent_hold_frames: int = 2,
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
    poses = payload.get("poses") if isinstance(payload.get("poses"), list) else []
    latest_pose = next((pose for pose in reversed(poses) if isinstance(pose, dict)), None)
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
        auto_heading = motion_heading_for_pose(pose, index)
        raw_heading = pose.get("rheading")
        return {
            "instance_id": pose.get("instance_id"),
            "time_sec": pose.get("time_sec"),
            "rcenter": pose.get("rcenter"),
            "rheading": auto_heading or raw_heading,
            "rheading_raw": raw_heading,
            "rheading_source": "recent_motion" if auto_heading else "tsolve_rotation",
            "center": pose.get("center"),
            "R": pose.get("R"),
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

        if (
            fallback_pose is not None
            and trailing_holds <= max(0, int(max_recent_hold_frames))
            and (fallback_pose.get("rcenter") or fallback_pose.get("center"))
        ):
            return {
                "ok": True,
                "pose": pose_payload(fallback_pose, fallback=True, hold_count=trailing_holds),
                "age_seconds": age,
                "processed_count": processed_count,
                "latest_instance_id": latest_pose.get("instance_id"),
                "recent_hold_fallback": True,
                "trailing_hold_frames": trailing_holds,
                "hold_reason": latest_pose.get("hold_reason") or latest_pose.get("rejected_reason"),
            }

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

    return {
        "ok": True,
        "pose": pose_payload(latest_pose),
        "age_seconds": age,
        "processed_count": processed_count,
    }


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
    return vector3(pose.get("rcenter"))


def step_target_position(step: dict[str, Any]) -> list[float] | None:
    return vector3(step.get("to")) or vector3(step.get("at"))


def horizontal_xz_distance(a: list[float] | None, b: list[float] | None) -> float | None:
    if not a or not b:
        return None
    return ((a[0] - b[0]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


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
    if not math.isfinite(norm) or norm < min_delta:
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


def target_direction_xz(current: list[float] | None, target: list[float] | None) -> list[float] | None:
    if current is None or target is None:
        return None
    return [target[0] - current[0], 0.0, target[2] - current[2]]


def vertical_rc_toward(current: list[float] | None, target: list[float] | None, max_vertical_rc: float) -> float:
    if current is None or target is None:
        return 0.0
    error = float(target[1]) - float(current[1])
    if abs(error) < 0.08:
        return 0.0
    return max(-max_vertical_rc, min(max_vertical_rc, error * 0.028))


def neutral_hover(drone: Any, seconds: float = 0.10) -> None:
    drone.move(0.0, 0.0, 0.0, 0.0, True)
    if seconds > 0:
        time.sleep(seconds)


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
    drone.move(float(yaw), float(du), float(lr), float(bf), False)
    time.sleep(max(0.05, float(seconds)))
    neutral_hover(drone, 0.03)
    return sent


def execute_control_command(
    drone: Any,
    OpenDJI: Any,
    command: dict[str, Any],
    *,
    pose_stream_path: Path | None = None,
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
        result = drone.move(0, 0, 0, 0, True)
        height_guard = {"enabled": False, "reason": "not a takeoff command"}
    elif name == "mission":
        result = execute_guarded_mission_packet(
            drone,
            command.get("mission") if isinstance(command.get("mission"), dict) else {},
            pose_stream_path=pose_stream_path,
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
    stop_flag: StopFlag | None = None,
    mission_stop_event: threading.Event | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
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
        # Indoor cruise must not depend on camera/gimbal heading or a stale UI
        # trim.  Calibrate the actual DJI body axes from TSolve feedback before
        # translating, even if an older cached page sends the old payload.
        if not bool(mission.get("allow_axis_auto_calibration")):
            safety_overrides.append("forced allow_axis_auto_calibration=true for cruise")
        if not bool(mission.get("allow_lateral_rc")):
            safety_overrides.append("forced allow_lateral_rc=true for cruise")
        if abs(float(mission.get("heading_trim_deg") or 0.0)) > 1e-9:
            safety_overrides.append("forced heading_trim_deg=0 for cruise")
        try:
            old_pulse = float(mission.get("pulse_seconds") or GUIDED_DEFAULT_PULSE_SECONDS)
        except (TypeError, ValueError):
            old_pulse = GUIDED_DEFAULT_PULSE_SECONDS
        if old_pulse < 0.28:
            safety_overrides.append("raised pulse_seconds for visible slow cruise")
        mission = dict(mission)
        mission["allow_axis_auto_calibration"] = True
        mission["allow_lateral_rc"] = True
        mission["heading_trim_deg"] = 0.0
        mission["pulse_seconds"] = max(old_pulse, 0.30)
        mission["axis_probe_seconds"] = max(float(mission.get("axis_probe_seconds") or 0.0), 0.55)
        mission["max_cruise_seconds"] = max(float(mission.get("max_cruise_seconds") or 0.0), 120.0)
    pose_max_age = clamp_float(
        mission.get("pose_max_age_seconds"),
        GUIDED_DEFAULT_POSE_MAX_AGE_SECONDS,
        0.4,
        10.0,
    )
    pose_recovery_seconds = clamp_float(
        mission.get("pose_recovery_seconds"),
        GUIDED_DEFAULT_POSE_RECOVERY_SECONDS,
        0.5,
        90.0,
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
    arrival_radius = clamp_float(
        mission.get("arrival_radius_map_units"),
        0.22,
        0.05,
        0.80,
    )
    arrival_deadband = clamp_float(
        mission.get("arrival_deadband_map_units"),
        0.10,
        0.0,
        0.35,
    )
    soft_arrival_radius = arrival_radius + arrival_deadband
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
    rc_summary: dict[str, Any] = {
        "open_dji_move_order": "rcw, du, lr, bf",
        "open_dji_rounding": "yaw to 4 decimals; vertical/lateral/forward to 2 decimals",
        "enable_control_result": None,
        "disable_control_result": None,
        "pulse_counts": {"yaw": 0, "forward": 0, "vertical": 0, "lateral": 0},
        "pulse_samples": [],
            "adaptive_axis": {
            "yaw_sign": 1.0,
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
            "safety_overrides": safety_overrides,
        }
    body_axes: dict[str, list[float] | None] = {"forward": None, "lateral": None}
    yaw_sign = 1.0
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

    def publish_progress(payload: dict[str, Any]) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(
                {
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

    def wait_for_pose_recovery(
        phase: str,
        reason: str | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any] | None:
        nonlocal abort_reason, last_pose_gate
        wait_seconds = pose_recovery_seconds if timeout is None else max(0.1, float(timeout))
        deadline = time.time() + wait_seconds
        last_reason = reason or "waiting for a fresh TSolve pose"
        while time.time() < deadline:
            if stop_flag is not None and stop_flag.stop:
                abort_reason = "live localization stop requested"
                return None
            if mission_stop_event is not None and mission_stop_event.is_set():
                abort_reason = "emergency hover requested"
                return None
            gate = latest_tsolve_pose_gate(pose_stream_path, pose_max_age)
            if gate.get("ok") and pose_gate_position(gate) is not None:
                last_pose_gate = gate
                return gate
            last_reason = str(gate.get("reason") or last_reason)
            publish_progress(
                {
                    "phase": "pose_recovery",
                    "recovery_phase": phase,
                    "pose_gate": gate,
                    "message": f"Hovering; waiting for TSolve recovery: {last_reason}",
                }
            )
            neutral_hover(drone, GUIDED_RECOVERY_HOVER_SECONDS)
        abort_reason = f"TSolve localization did not recover within {wait_seconds:.1f}s ({last_reason})"
        return None

    def pose_gate_or_abort() -> dict[str, Any] | None:
        nonlocal abort_reason, last_pose_gate
        if stop_flag is not None and stop_flag.stop:
            abort_reason = "live localization stop requested"
            return None
        if mission_stop_event is not None and mission_stop_event.is_set():
            abort_reason = "emergency hover requested"
            return None
        gate = latest_tsolve_pose_gate(pose_stream_path, pose_max_age)
        if not gate.get("ok"):
            return wait_for_pose_recovery("motion_gate", str(gate.get("reason") or "TSolve pose gate failed"))
        if pose_gate_position(gate) is None:
            abort_reason = "latest TSolve pose has no ATLAS room-frame rcenter"
            return None
        last_pose_gate = gate
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
        gate_before = reference_gate or pose_gate_or_abort()
        before = pose_gate_position(gate_before)
        if before is None:
            return False
        publish_progress(
            {
                "phase": "axis_calibration",
                "axis": axis_name,
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
                    "max_cruise_seconds": max_cruise_seconds,
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

        for idx, step in enumerate(commands):
            if not isinstance(step, dict):
                skipped.append({"index": idx, "type": "unknown", "reason": "invalid command record"})
                continue
            kind = str(step.get("type", "")).strip().lower()
            title = str(step.get("title", kind or "step"))
            if kind == "gate":
                skipped.append({"index": idx, "type": kind, "title": title, "reason": "UI confirmation gate"})
                continue

            gate = pose_gate_or_abort()
            if gate is None:
                break

            if kind == "hover":
                duration = max(0.1, min(2.0, float(step.get("duration_s") or 0.5)))
                neutral_hover(drone, duration)
                hover_count += 1
                executed.append({"index": idx, "type": kind, "title": title, "duration_s": duration})
                continue

            if kind == "yaw":
                if str(step.get("safety") or "").strip().lower() == "slow-yaw":
                    skipped.append(
                        {
                            "index": idx,
                            "type": kind,
                            "title": title,
                            "reason": "navigation yaw is handled by the closed-loop cruise controller",
                        }
                    )
                    continue
                yaw_delta = clamp_float(step.get("yaw_delta_deg"), 0.0, -180.0, 180.0)
                if abs(yaw_delta) < 2.0:
                    skipped.append({"index": idx, "type": kind, "title": title, "reason": "initial or negligible yaw delta"})
                    continue
                yaw_rc = yaw_sign * (max_yaw_rc if yaw_delta > 0 else -max_yaw_rc)
                seconds = min(max_step_seconds, max(pulse_seconds, abs(yaw_delta) / 90.0 * 0.8))
                pulses = max(1, int(round(seconds / pulse_seconds)))
                for _ in range(pulses):
                    if pose_gate_or_abort() is None:
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
                planned_duration = clamp_float(step.get("duration_s"), pulse_seconds, pulse_seconds, 120.0)
                distance = clamp_float(step.get("distance"), 0.0, 0.0, 1000.0)
                if distance <= 1e-4:
                    skipped.append({"index": idx, "type": kind, "title": title, "reason": "zero distance segment"})
                    continue
                target = step_target_position(step)
                if target is None:
                    abort_reason = "cruise target is missing; refusing open-loop patrol travel"
                    break

                deadline = time.time() + min(max_cruise_seconds, max(4.0, planned_duration * 2.2))
                start_position = pose_gate_position(gate)
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
                last_navigation_mode = "unknown"
                last_processed_count = None
                last_pose_age = None
                if allow_axis_auto_calibration and body_axes["forward"] is None and body_axes["lateral"] is None:
                    rc_summary["adaptive_axis"]["mode"] = "calibrated_body_axes"
                    if not calibrate_body_axes(gate):
                        abort_reason = "could not calibrate DJI body motion axes from TSolve pose feedback"
                        break

                while time.time() < deadline:
                    current_gate = pose_gate_or_abort()
                    if current_gate is None:
                        break
                    current_position = pose_gate_position(current_gate)
                    current_distance = horizontal_xz_distance(current_position, target)
                    if current_distance is not None:
                        final_distance = current_distance
                        vertical_error = abs(float(target[1]) - float(current_position[1])) if current_position else 0.0
                        if current_distance <= arrival_radius and vertical_error <= 0.35:
                            reached = True
                            arrival_mode = "strict_radius"
                            break
                        if current_distance <= soft_arrival_radius and vertical_error <= 0.45:
                            reached = True
                            arrival_mode = "soft_deadband"
                            publish_progress(
                                {
                                    "phase": "cruise_arrival",
                                    "step_index": idx,
                                    "step_title": title,
                                    "target": target,
                                    "distance_to_target": current_distance,
                                    "arrival_radius": arrival_radius,
                                    "soft_arrival_radius": soft_arrival_radius,
                                    "message": (
                                        f"Patrol target reached within soft indoor deadband "
                                        f"({current_distance:.2f} <= {soft_arrival_radius:.2f}); hovering."
                                    ),
                                }
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

                    desired_direction = target_direction_xz(current_position, target)
                    desired_unit = normalize_xz(desired_direction)
                    if desired_unit is None:
                        abort_reason = "could not compute patrol target direction from latest TSolve pose"
                        break
                    du_rc = vertical_rc_toward(current_position, target, max_vertical_rc)
                    heading = pose_gate_heading(current_gate, heading_trim_rad)
                    calibrated_forward = normalize_xz(body_axes["forward"]) if allow_axis_auto_calibration else None
                    calibrated_lateral = normalize_xz(body_axes["lateral"]) if allow_axis_auto_calibration and allow_lateral_rc else None
                    angle = signed_angle_xz(calibrated_forward or heading, desired_unit)
                    if angle is None:
                        abort_reason = "latest TSolve pose has no usable heading or calibrated body axis; refusing patrol translation"
                        break
                    speed_scale = max(0.32, min(1.0, (current_distance or 0.0) / max(arrival_radius * 5.0, 0.55)))
                    yaw_rc = 0.0
                    bf_rc = 0.0
                    lr_rc = 0.0
                    navigation_mode = "camera_heading"
                    if calibrated_forward is not None:
                        # Use the measured DJI body axes, not the camera/gimbal
                        # heading.  Positive bf/lr pulses were calibrated in
                        # ATLAS room coordinates just before this cruise.
                        navigation_mode = "calibrated_body_axes"
                        forward_gain = desired_unit[0] * calibrated_forward[0] + desired_unit[2] * calibrated_forward[2]
                        lateral_gain = 0.0
                        if calibrated_lateral is not None:
                            lateral_gain = desired_unit[0] * calibrated_lateral[0] + desired_unit[2] * calibrated_lateral[2]
                        forward_gain = max(-1.0, min(1.0, forward_gain))
                        lateral_gain = max(-1.0, min(1.0, lateral_gain))
                        bf_rc = max_forward_rc * forward_gain * speed_scale
                        lr_rc = max_lateral_rc * lateral_gain * speed_scale
                    else:
                        angle_abs = abs(angle)
                        if angle_abs > math.radians(10.0):
                            yaw_scale = max(0.35, min(1.0, angle_abs / math.radians(70.0)))
                            yaw_rc = yaw_sign * (max_yaw_rc if angle > 0.0 else -max_yaw_rc) * yaw_scale
                            if angle_abs < math.radians(32.0):
                                bf_rc = max_forward_rc * max(0.25, math.cos(angle)) * speed_scale * 0.45
                        else:
                            bf_rc = max_forward_rc * max(0.25, math.cos(angle)) * speed_scale

                    forward_gain = bf_rc / max_forward_rc if max_forward_rc > 1e-9 else 0.0
                    lateral_gain = lr_rc / max_lateral_rc if max_lateral_rc > 1e-9 else 0.0
                    progress = {
                        "phase": "cruise",
                        "step_index": idx,
                        "step_title": title,
                        "target": target,
                        "distance_to_target": current_distance,
                        "heading_error_deg": math.degrees(angle),
                        "body_forward_gain": forward_gain,
                        "body_lateral_gain": lateral_gain,
                        "navigation_mode": navigation_mode,
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
                    sent = execute_rc_pulse(drone, yaw=yaw_rc, lr=lr_rc, bf=bf_rc, du=du_rc, seconds=pulse_seconds)
                    if abs(yaw_rc) > 1e-6:
                        record_pulse("yaw", sent)
                        yaw_pulse_count += 1
                    if abs(bf_rc) > 1e-6:
                        record_pulse("forward", sent)
                        forward_pulse_count += 1
                    if abs(lr_rc) > 1e-6:
                        record_pulse("lateral", sent)
                        lateral_pulse_count += 1
                    if abs(du_rc) > 1e-6:
                        record_pulse("vertical", sent)
                    executed_pulses += 1
                    pulse_count += 1
                    after_gate = wait_for_pose_after(current_gate)
                    after_position = pose_gate_position(after_gate)
                    after_distance = horizontal_xz_distance(after_position, target)
                    horizontal_pulse = abs(bf_rc) > 1e-6 or abs(lr_rc) > 1e-6
                    got_new_pose = (
                        isinstance(after_gate, dict)
                        and after_gate.get("processed_count") != current_gate.get("processed_count")
                    )
                    if not horizontal_pulse:
                        stale_motion_count = 0
                    elif not got_new_pose:
                        stale_motion_count += 1
                    elif after_distance is None:
                        stale_motion_count += 1
                    elif after_distance < current_distance - 0.015:
                        stale_motion_count = 0
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
                        # A fresh TSolve pose arrived, but the tiny RC pulse did
                        # not change horizontal distance enough to classify as
                        # progress yet.  Keep watching instead of aborting.
                        stale_motion_count = max(0, stale_motion_count - 1)
                    if stale_motion_count >= 12:
                        abort_reason = "patrol cruise did not receive enough useful TSolve pose progress"
                        break
                if abort_reason:
                    break
                if not reached:
                    if isinstance(final_distance, (int, float)) and final_distance <= soft_arrival_radius:
                        reached = True
                        arrival_mode = "timeout_soft_deadband"
                        publish_progress(
                            {
                                "phase": "cruise_arrival",
                                "step_index": idx,
                                "step_title": title,
                                "target": target,
                                "distance_to_target": final_distance,
                                "arrival_radius": arrival_radius,
                                "soft_arrival_radius": soft_arrival_radius,
                                "message": (
                                    f"Patrol cruise timed out near target but inside soft deadband "
                                    f"({final_distance:.2f} <= {soft_arrival_radius:.2f}); accepting and hovering."
                                ),
                            }
                        )
                        neutral_hover(drone, 0.25)
                    elif closest_distance < float("inf") and closest_distance <= soft_arrival_radius:
                        reached = True
                        arrival_mode = "closest_soft_deadband"
                        publish_progress(
                            {
                                "phase": "cruise_arrival",
                                "step_index": idx,
                                "step_title": title,
                                "target": target,
                                "distance_to_target": closest_distance,
                                "arrival_radius": arrival_radius,
                                "soft_arrival_radius": soft_arrival_radius,
                                "message": (
                                    f"Patrol cruise passed within soft deadband "
                                    f"({closest_distance:.2f} <= {soft_arrival_radius:.2f}); accepting and hovering."
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
                        f"(target radius {arrival_radius:.2f}, soft radius {soft_arrival_radius:.2f}, "
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
                executed.append(
                    {
                        "index": idx,
                        "type": kind,
                        "title": title,
                        "closed_loop": True,
                        "target": target,
                        "distance": distance,
                        "planned_duration_s": planned_duration,
                        "arrival_radius": arrival_radius,
                        "soft_arrival_radius": soft_arrival_radius,
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
                    if pose_gate_or_abort() is None:
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
            "max_cruise_seconds": max_cruise_seconds,
            "heading_trim_deg": heading_trim_deg,
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
    ap.add_argument("--fps", type=float, default=2.0, help="Frame sampling rate written to ATLAS.")
    ap.add_argument("--max-size", type=int, default=1200, help="Resize longest side before saving.")
    ap.add_argument("--jpeg-quality", type=int, default=88)
    ap.add_argument("--max-frames", type=int, default=0, help="0 means run until Ctrl-C.")
    ap.add_argument("--show", action="store_true", help="Open a local OpenCV preview window.")
    ap.add_argument("--no-history", action="store_true", help="Only update latest.jpg/status.json.")
    ap.add_argument("--out-root", type=Path, default=ROOT / "data" / "dji_live")
    ap.add_argument("--public-root", type=Path, default=ROOT / "viewer" / "public" / "live_dji")
    ap.add_argument("--pose-stream", type=Path, default=None, help="Live poses_partial.json used as the guided-flight freshness gate.")
    ap.add_argument("--enemy-model", type=Path, default=None, help="Optional trained YOLO model used for live enemy-drone detection.")
    ap.add_argument("--enemy-output", type=Path, default=None, help="Browser-visible enemy detection JSON output.")
    ap.add_argument("--enemy-detect-fps", type=float, default=1.0, help="Maximum detector rate. Keeps TSolve/localization responsive.")
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
    control_command_path = args.public_root / "control_command.json"
    control_status_path = args.public_root / "control_status.json"
    control_history_path = args.public_root / "control_status_history.jsonl"
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
        "control_enabled": True,
        "control_command_path": str(control_command_path),
        "control_status_path": str(control_status_path),
        "control_history_path": str(control_history_path),
        "pose_stream_path": str(args.pose_stream) if args.pose_stream else None,
        "enemy_model": str(args.enemy_model) if args.enemy_model else None,
        "enemy_output": str(enemy_output_path),
        "enemy_detect_fps": args.enemy_detect_fps,
        "enemy_confidence": args.enemy_conf,
        "note": "This bridge receives frames and accepts explicit takeoff/land/hover commands.",
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
    last_control_id: str | None = None
    enemy_detector = None
    enemy_detection_error = ""
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
                    with control_lock:
                        status["last_control"] = control_started
                        status["updated_at"] = time.time()
                        atomic_write_json(public_status_path, status)
                        atomic_write_json(session_status_path, status)
                    if command_name == "mission":
                        mission_cancel.clear()

                    def publish_progress(progress: dict[str, Any]) -> None:
                        control_progress = {
                            **control_started,
                            "status": "running",
                            "progress": progress,
                            "message": progress.get("message") or control_started["message"],
                            "updated_at": time.time(),
                        }
                        atomic_write_json(control_status_path, control_progress)
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
                        stop_flag=stop,
                        mission_stop_event=mission_cancel,
                        progress_callback=publish_progress,
                    )
                    atomic_write_json(control_status_path, control_result)
                    append_jsonl(control_history_path, {**control_result, "event": "finished"})
                    with control_lock:
                        status["last_control"] = control_result
                        status["updated_at"] = time.time()
                        atomic_write_json(public_status_path, status)
                        atomic_write_json(session_status_path, status)
                    print(f"control={control_result['command']} result={control_result.get('result')}")
                except Exception as exc:
                    control_result = {
                        "ok": False,
                        "id": command_payload.get("id"),
                        "command": command_payload.get("command"),
                        "error": str(exc),
                        "updated_at": time.time(),
                    }
                    atomic_write_json(control_status_path, control_result)
                    append_jsonl(control_history_path, {**control_result, "event": "error"})
                    with control_lock:
                        status["last_control"] = control_result
                        status["updated_at"] = time.time()
                        atomic_write_json(public_status_path, status)
                        atomic_write_json(session_status_path, status)
                finally:
                    with control_lock:
                        control_busy = False

            while not stop.stop:
                if control_command_path.exists():
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
                                    try:
                                        neutral_hover(drone, 0.05)
                                        error_text = ""
                                        ok = True
                                    except Exception as exc:
                                        error_text = str(exc)
                                        ok = False
                                    control_result = {
                                        "ok": ok,
                                        "id": command_id,
                                        "command": command_name,
                                        "emergency_stop": True,
                                        "message": "Emergency hover requested; active guided mission cancellation signaled.",
                                        "error": error_text or None,
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
                                append_jsonl(control_history_path, {**control_result, "event": "rejected"})
                                with control_lock:
                                    status["last_control"] = control_result
                                    status["updated_at"] = time.time()
                                    atomic_write_json(public_status_path, status)
                                    atomic_write_json(session_status_path, status)
                            else:
                                threading.Thread(
                                    target=control_worker,
                                    args=(command_payload,),
                                    daemon=True,
                                ).start()
                    except Exception as exc:
                        control_result = {
                            "ok": False,
                            "id": command_payload.get("id") if "command_payload" in locals() else None,
                            "error": str(exc),
                            "updated_at": time.time(),
                        }
                        atomic_write_json(control_status_path, control_result)
                        append_jsonl(control_history_path, {**control_result, "event": "error"})
                        status["last_control"] = control_result
                        atomic_write_json(public_status_path, status)
                        atomic_write_json(session_status_path, status)
                now = time.perf_counter()
                frame = drone.getFrame()
                if frame is None:
                    if time.perf_counter() - last_progress_print > 2.0:
                        print("waiting for first frame...")
                        last_progress_print = time.perf_counter()
                    time.sleep(0.03)
                    continue

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
                if enemy_detector is not None and now >= next_enemy_detection_time:
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
