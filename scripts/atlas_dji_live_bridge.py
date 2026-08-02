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
import itertools
import json
import math
import os
import re
import signal
import sys
import threading
import time
import types
import uuid
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
            "rheading_raw": raw_heading,
            "rheading_source": "tsolve_rotation",
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
            fallback_result = {
                "ok": True,
                "pose": pose_payload(fallback_pose, fallback=True, hold_count=trailing_holds),
                "age_seconds": age,
                "processed_count": processed_count,
                "latest_instance_id": latest_pose.get("instance_id"),
                "recent_hold_fallback": True,
                "trailing_hold_frames": trailing_holds,
                "hold_reason": latest_pose.get("hold_reason") or latest_pose.get("rejected_reason"),
            }
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
    poses = payload.get("poses") if isinstance(payload.get("poses"), list) else []
    latest = next((pose for pose in reversed(poses) if isinstance(pose, dict)), None)
    if latest is None:
        return {"ok": False, "reason": "no rotation-only observation is available"}
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
    }


def load_taught_patrol_reference(mission: dict[str, Any]) -> dict[str, Any] | None:
    """Load the optional visual turn reference for this exact map and patrol."""
    map_id = str(mission.get("map_id") or "")
    patrol_id = str(mission.get("patrol_id") or "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", map_id):
        return None
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", patrol_id):
        return None
    path = ROOT / "viewer" / "public" / "maps" / map_id / "taught_patrols" / patrol_id / "reference.json"
    try:
        reference = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(reference, dict):
        return None
    if reference.get("map_id") != map_id or reference.get("patrol_id") != patrol_id:
        return None
    if not reference.get("complete_loop") or not reference.get("enabled_for_turn_recovery"):
        return None
    if not isinstance(reference.get("legs"), list) or len(reference["legs"]) < 4:
        return None
    return reference


def taught_leg_for_step(reference: dict[str, Any] | None, step: dict[str, Any]) -> dict[str, Any] | None:
    """Return the taught leg matching this patrol cruise.

    Prefer exact endpoints.  A user may adjust patrol points after teaching,
    however, so the stable ``Patrol cruise N`` command order is a safe
    fallback: cruise 2 is point 1→2, cruise 3 is point 2→3, and so on.
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

    title_match = re.fullmatch(
        r"\s*Patrol cruise\s+(\d+)\s*",
        str(step.get("title") or ""),
        re.IGNORECASE,
    )
    if title_match:
        # Cruise 1 is the entry leg to point 1 and has no taught loop leg.
        taught_leg_index = int(title_match.group(1)) - 2
        legs = reference.get("legs", [])
        if 0 <= taught_leg_index < len(legs) and isinstance(legs[taught_leg_index], dict):
            return legs[taught_leg_index]
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


def yaw_direction_for_angle(angle: float, override: str | None = None) -> float:
    """Choose yaw direction; negative is the bridge's map-frame left turn."""
    if override == "left":
        return -1.0
    if override == "right":
        return 1.0
    return 1.0 if angle > 0.0 else -1.0


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


def step_target_position(step: dict[str, Any]) -> list[float] | None:
    return vector3(step.get("to")) or vector3(step.get("at"))


def horizontal_xz_distance(a: list[float] | None, b: list[float] | None) -> float | None:
    if not a or not b:
        return None
    return ((a[0] - b[0]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


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


def mission_step_sequence(commands: list[Any], patrol_loop: bool):
    """Yield stable command indexes once, or continuously for a patrol loop."""
    indexed = enumerate(commands)
    return itertools.cycle(indexed) if patrol_loop else indexed


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
        result = drone.move(0, 0, 0, 0, True)
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
    arrival_radius = clamp_float(
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
    operator_heading_calibrated = bool(mission.get("operator_heading_calibrated"))
    initial_body_heading_offset_deg = clamp_float(
        mission.get("initial_body_heading_offset_deg"),
        0.0,
        -180.0,
        180.0,
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
    patrol_safety_barriers = [item for item in mission.get("safety_barriers") or [] if isinstance(item, dict)]
    patrol_safety_obstacles = [item for item in mission.get("safety_obstacles") or [] if isinstance(item, dict)]
    safety_motion_buffer = clamp_float(mission.get("safety_motion_buffer_m"), 0.30, 0.30, 1.0)
    if is_patrol and closed_wall_ring(patrol_safety_barriers) is None:
        raise RuntimeError("patrol requires a closed saved-wall geofence on the selected map")
    taught_reference = load_taught_patrol_reference(mission) if patrol_loop else None

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
            "operator_heading_calibrated": operator_heading_calibrated,
            "initial_body_heading_offset_deg": initial_body_heading_offset_deg,
            "initial_pose_offset_room": initial_pose_offset_room,
            "safety_overrides": safety_overrides,
            "taught_turn_reference": {
                "enabled": taught_reference is not None,
                "path": (
                    f"maps/{mission.get('map_id')}/taught_patrols/{mission.get('patrol_id')}/reference.json"
                    if taught_reference is not None
                    else None
                ),
            },
        }
    body_axes: dict[str, list[float] | None] = {"forward": None, "lateral": None}
    # Visual model alignment is a useful initial estimate, but it is not proof
    # of DJI's physical body-forward RC axis. Always verify body-forward from
    # a small TSolve-observed probe before the first patrol translation.
    calibrated_heading_offset_rad: float | None = None
    if operator_heading_calibrated:
        rc_summary["adaptive_axis"]["mode"] = "operator_heading_seed_pending_physical_verification"
        rc_summary["adaptive_axis"]["operator_heading_seed_deg"] = initial_body_heading_offset_deg
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

    def enforce_patrol_geofence(gate: dict[str, Any] | None, phase: str) -> bool:
        nonlocal abort_reason
        if not is_patrol:
            return True
        issue = pursuit_geofence_issue(
            pose_gate_position(gate),
            patrol_safety_barriers,
            patrol_safety_obstacles,
            motion_buffer_m=safety_motion_buffer,
        )
        if issue is None:
            return True
        abort_reason = f"patrol geofence blocked {phase}: {issue}"
        publish_progress(
            {
                "phase": "patrol_geofence",
                "message": abort_reason,
                "safety_motion_buffer_m": safety_motion_buffer,
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
            gate = continuity_guarded_pose_gate()
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

    def continuity_guarded_pose_gate() -> dict[str, Any]:
        """Reject a localization jump without poisoning the trusted patrol pose."""
        gate = latest_tsolve_pose_gate(pose_stream_path, pose_max_age)
        if not gate.get("ok"):
            return gate
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
        allowed_step = bounded_pose_step_limit(
            trusted_pose.get("received_unix") if isinstance(trusted_pose, dict) else None,
            candidate_pose.get("received_unix") if isinstance(candidate_pose, dict) else None,
            base_limit=max_pose_step,
            hard_limit=max_pose_step_hard,
        )
        if step <= allowed_step:
            return gate
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

    def pose_gate_or_abort() -> dict[str, Any] | None:
        nonlocal abort_reason, last_pose_gate
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
            abort_reason = "no fresh camera observation arrived after the RC pulse; hovering"
            return None
        # Older replay streams have no received_unix. Preserve compatibility.
        return wait_for_pose_after(previous_gate, timeout=0.4)

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
        if before is None or not enforce_patrol_geofence(gate_before, f"{axis_name} calibration"):
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

        step_source = mission_step_sequence(commands, patrol_loop)
        for execution_index, (idx, step) in enumerate(step_source):
            lap_index = execution_index // max(1, len(commands))
            if not isinstance(step, dict):
                skipped.append({"index": idx, "type": "unknown", "reason": "invalid command record"})
                continue
            kind = str(step.get("type", "")).strip().lower()
            title = str(step.get("title", kind or "step"))
            if patrol_loop and idx == 0:
                if len(executed) > 500:
                    del executed[:-250]
                if len(skipped) > 500:
                    del skipped[:-250]
                publish_progress(
                    {
                        "phase": "patrol_lap",
                        "lap": lap_index + 1,
                        "message": f"Starting patrol lap {lap_index + 1}; ordered route begins at patrol point 1.",
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
                planned_duration = clamp_float(step.get("duration_s"), pulse_seconds, pulse_seconds, 120.0)
                distance = clamp_float(step.get("distance"), 0.0, 0.0, 1000.0)
                if distance <= 1e-4:
                    skipped.append({"index": idx, "type": kind, "title": title, "reason": "zero distance segment"})
                    continue
                target = step_target_position(step)
                if target is None:
                    abort_reason = "cruise target is missing; refusing open-loop patrol travel"
                    break

                # A 90-degree corner is alignment work, not failed translation.
                # Give it a separate bounded allowance before the normal travel
                # budget, while retaining the overall mission step ceiling.
                alignment_deadline = time.time() + min(
                    max_cruise_seconds,
                    alignment_grace_seconds + max(4.0, planned_duration * 2.2),
                )
                # Fresh-pose waits make a real DJI corner consume much more
                # wall time than its RC command time. Permit extra wall time
                # only while the measured heading error keeps improving.
                segment_safety_deadline = time.time() + max_cruise_seconds
                absolute_segment_deadline = time.time() + min(240.0, max_cruise_seconds * 2.0)
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
                wrong_yaw_pulses = 0
                last_navigation_mode = "unknown"
                last_processed_count = None
                last_pose_age = None
                yaw_position_anchor = None
                forward_alignment_locked = False
                last_alignment_yaw_rc = 0.0
                blind_yaw_seconds = 0.0
                max_blind_yaw_seconds = 12.0
                segment_start = vector3(step.get("from"))
                taught_leg = taught_leg_for_step(taught_reference, step)
                turn_direction_override = taught_turn_direction_override(taught_leg)
                guarded_taught_rotation = is_guarded_point_three_to_four_turn(taught_leg)
                if guarded_taught_rotation:
                    # A pure yaw cannot translate the aircraft.  Keep point 3
                    # as the navigation anchor throughout the turn so gradual
                    # monocular position drift cannot move the apparent
                    # origin toward point 4.
                    yaw_position_anchor = start_position
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
                        time.time() + 190.0,
                    )
                    segment_safety_deadline = max(
                        segment_safety_deadline,
                        time.time() + 220.0,
                    )
                rotation_alignment_ready_at: float | None = None
                rotation_heading_stable_frames = 0
                rotation_heading_last_instance_id: str | None = None
                required_rotation_heading_stable_frames = 3
                max_rotation_position_recovery_seconds = 25.0
                rotation_reacquisition_pulses = 0
                max_rotation_reacquisition_pulses = 3
                rotation_reacquisition_forward_rc = max(
                    0.01,
                    min(0.02, max_forward_rc * 0.60),
                )
                if allow_axis_auto_calibration and body_axes["forward"] is None and body_axes["lateral"] is None:
                    rc_summary["adaptive_axis"]["mode"] = "calibrated_body_axes"
                    if not calibrate_body_axes(gate):
                        abort_reason = "could not calibrate DJI body motion axes from TSolve pose feedback"
                        break
                if calibrated_heading_offset_rad is None:
                    if calibrate_forward_heading(gate):
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

                while True:
                    now = time.time()
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
                    rotation_position_untrusted = (
                        not gate_attempt.get("ok")
                        or bool(gate_attempt.get("recent_hold_fallback"))
                    )
                    confirmed_rotation_angle: float | None = None
                    confirmed_rotation_gate: dict[str, Any] | None = None
                    rotation_observation = (
                        latest_rotation_only_heading(pose_stream_path, pose_max_age)
                        if (
                            taught_leg is not None
                            and not travel_started
                            and (guarded_taught_rotation or rotation_position_untrusted)
                        )
                        else {"ok": False}
                    )
                    if rotation_observation.get("ok"):
                        frozen_position = (
                            yaw_position_anchor
                            if yaw_position_anchor is not None
                            else pose_gate_position(last_pose_gate)
                        )
                        visual_direction = normalize_xz(target_direction_xz(frozen_position, target))
                        visual_angle = signed_angle_xz(rotation_observation.get("heading"), visual_direction)
                        if visual_angle is not None and abs(visual_angle) <= math.radians(10.0):
                            forward_alignment_locked = True
                            last_alignment_yaw_rc = 0.0
                            observation_instance_id = str(
                                rotation_observation.get("instance_id") or ""
                            )
                            if (
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
                                    "heading_error_deg": math.degrees(visual_angle),
                                    "rotation_tracks": rotation_observation.get("tracks"),
                                    "stable_heading_frames": rotation_heading_stable_frames,
                                    "required_stable_heading_frames": required_rotation_heading_stable_frames,
                                    "message": (
                                        "Visual turn is aligned with the taught patrol leg; "
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
                            if rotation_position_untrusted:
                                if (
                                    guarded_taught_rotation
                                    and rotation_reacquisition_pulses
                                    < max_rotation_reacquisition_pulses
                                ):
                                    if stop_flag is not None and stop_flag.stop:
                                        abort_reason = "live localization stop requested"
                                        break
                                    if (
                                        mission_stop_event is not None
                                        and mission_stop_event.is_set()
                                    ):
                                        abort_reason = "emergency hover requested"
                                        break
                                    # The right turn has converged on three
                                    # independent optical frames, but the
                                    # camera is still looking through the
                                    # point-3 weak-map sector.  Make only a
                                    # very small straight movement along the
                                    # known 3→4 corridor to expose new
                                    # features.  Yaw and lateral channels stay
                                    # locked, and no further pulse is allowed
                                    # after the bounded count.
                                    publish_progress(
                                        {
                                            "phase": "taught_turn_forward_reacquisition",
                                            "step_index": idx,
                                            "step_title": title,
                                            "taught_leg": [
                                                taught_leg.get("from_point"),
                                                taught_leg.get("to_point"),
                                            ],
                                            "heading_error_deg": math.degrees(
                                                visual_angle
                                            ),
                                            "rotation_tracks": rotation_observation.get(
                                                "tracks"
                                            ),
                                            "reacquisition_pulse": (
                                                rotation_reacquisition_pulses + 1
                                            ),
                                            "max_reacquisition_pulses": (
                                                max_rotation_reacquisition_pulses
                                            ),
                                            "message": (
                                                "Point-4 heading is steady; issuing one "
                                                "bounded low-speed forward pulse to "
                                                "reacquire map features."
                                            ),
                                        }
                                    )
                                    if not enforce_patrol_geofence(last_pose_gate, "point-4 feature reacquisition"):
                                        break
                                    sent = execute_rc_pulse(
                                        drone,
                                        yaw=0.0,
                                        lr=0.0,
                                        bf=rotation_reacquisition_forward_rc,
                                        du=0.0,
                                        seconds=pulse_seconds,
                                    )
                                    record_pulse("forward", sent)
                                    executed_pulses += 1
                                    pulse_count += 1
                                    forward_pulse_count += 1
                                    forward_command_seconds += pulse_seconds
                                    rotation_reacquisition_pulses += 1
                                    time.sleep(0.45)
                                    continue
                                neutral_hover(drone, 0.12)
                                continue
                            # Three distinct optical frames agree and the
                            # current position is both fresh and close to the
                            # frozen point-3 anchor.  Preserve the optical
                            # angle through the normal translation gate so a
                            # disagreeing monocular rotation cannot restart
                            # the completed turn.
                            confirmed_rotation_angle = visual_angle
                            confirmed_rotation_gate = gate_attempt
                        elif visual_angle is not None and taught_rotation_yaw_seconds < max_taught_rotation_yaw_seconds:
                            rotation_heading_stable_frames = 0
                            rotation_heading_last_instance_id = None
                            rotation_alignment_ready_at = None
                            yaw_scale = max(0.65, min(1.0, abs(visual_angle) / math.radians(70.0)))
                            taught_yaw_rc = (
                                yaw_sign
                                * max_yaw_rc
                                * yaw_direction_for_angle(visual_angle, turn_direction_override)
                                * yaw_scale
                            )
                            last_alignment_yaw_rc = taught_yaw_rc
                            publish_progress(
                                {
                                    "phase": "taught_rotation_only_recovery",
                                    "step_index": idx,
                                    "step_title": title,
                                    "taught_leg": [taught_leg.get("from_point"), taught_leg.get("to_point")],
                                    "heading_error_deg": math.degrees(visual_angle),
                                    "rotation_tracks": rotation_observation.get("tracks"),
                                    "rotation_yaw_seconds": taught_rotation_yaw_seconds,
                                    "max_rotation_yaw_seconds": max_taught_rotation_yaw_seconds,
                                    "turn_direction_override": turn_direction_override,
                                    "message": (
                                        "TSolve position is rejected during a taught corner; "
                                        "continuing yaw from fresh optical heading only. Forward and lateral are locked."
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
                            time.sleep(0.08)
                            continue
                        if taught_rotation_yaw_seconds >= max_taught_rotation_yaw_seconds:
                            abort_reason = (
                                "taught rotation-only recovery reached its 24 second safety limit; "
                                "hovering without forward movement"
                            )
                            break
                    if (
                        not gate_attempt.get("ok")
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
                        and not travel_started
                        and abs(last_alignment_yaw_rc) > 1e-6
                        and blind_yaw_seconds < max_blind_yaw_seconds
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
                        if rotation_reacquisition_pulses > 0:
                            # A real translation occurred during feature
                            # reacquisition, so switch from the pure-yaw
                            # anchor to the newly recovered map position.
                            yaw_position_anchor = None
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
                    cross_track = horizontal_xz_segment_distance(current_position, segment_start, target)
                    if cross_track is not None and cross_track > max_cross_track:
                        abort_reason = (
                            "patrol left the planned route corridor "
                            f"({cross_track:.2f} map units > {max_cross_track:.2f})"
                        )
                        break
                    if current_distance is not None:
                        final_distance = current_distance
                        if current_distance <= arrival_radius:
                            reached = True
                            arrival_mode = "strict_radius"
                            break
                        if current_distance <= soft_arrival_radius:
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
                    # Horizontal patrol holds the takeoff altitude through DJI.
                    # Monocular map Y is not a safe altitude-control signal.
                    du_rc = 0.0
                    heading = pose_gate_heading(
                        current_gate,
                        heading_trim_rad + float(calibrated_heading_offset_rad or 0.0),
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
                            guarded_taught_rotation
                            and not travel_started
                            and confirmed_rotation_angle is not None
                        )
                        else signed_angle_xz(heading, desired_unit)
                    )
                    if angle is None:
                        abort_reason = "latest TSolve pose has no usable heading or calibrated body axis; refusing patrol translation"
                        break
                    speed_scale = max(0.32, min(1.0, (current_distance or 0.0) / max(arrival_radius * 5.0, 0.55)))
                    speed_scale *= corridor_recovery_speed_scale(
                        cross_track,
                        recovery_start=cross_track_recovery_start,
                        hard_limit=max_cross_track,
                    )
                    yaw_rc = 0.0
                    bf_rc = 0.0
                    lr_rc = 0.0
                    navigation_mode = "camera_heading"
                    if heading is None:
                        abort_reason = "latest TSolve pose has no usable calibrated body heading; refusing patrol translation"
                        break
                    else:
                        navigation_mode = "fresh_heading_yaw_then_forward"
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
                        alignment_limit_deg = 14.0 if forward_alignment_locked else 10.0
                        if angle_abs > math.radians(alignment_limit_deg):
                            forward_alignment_locked = False
                            yaw_scale = max(0.65, min(1.0, angle_abs / math.radians(70.0)))
                            yaw_rc = (
                                yaw_sign
                                * max_yaw_rc
                                * yaw_direction_for_angle(
                                    angle,
                                    turn_direction_override if not travel_started else None,
                                )
                                * yaw_scale
                            )
                            last_alignment_yaw_rc = yaw_rc
                        else:
                            # Do not mix turning and translation. Once aligned,
                            # fly only on DJI's body-forward channel.
                            forward_alignment_locked = True
                            last_alignment_yaw_rc = 0.0
                            bf_rc = max_forward_rc * speed_scale
                            # Wall time spent hovering, relocalizing, or yawing
                            # must not consume the forward-travel budget.
                            travel_started = True

                    forward_gain = bf_rc / max_forward_rc if max_forward_rc > 1e-9 else 0.0
                    lateral_gain = lr_rc / max_lateral_rc if max_lateral_rc > 1e-9 else 0.0
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
                        "navigation_mode": navigation_mode,
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
                    sent = execute_rc_pulse(drone, yaw=yaw_rc, lr=lr_rc, bf=bf_rc, du=du_rc, seconds=pulse_seconds)
                    pulse_completed_unix = time.time()
                    if abs(yaw_rc) > 1e-6:
                        if yaw_position_anchor is None:
                            yaw_position_anchor = current_position
                        record_pulse("yaw", sent)
                        yaw_pulse_count += 1
                        if travel_started:
                            travel_yaw_command_seconds += pulse_seconds
                    if abs(bf_rc) > 1e-6:
                        record_pulse("forward", sent)
                        forward_pulse_count += 1
                        forward_command_seconds += pulse_seconds
                    if abs(lr_rc) > 1e-6:
                        record_pulse("lateral", sent)
                        lateral_pulse_count += 1
                    if abs(du_rc) > 1e-6:
                        record_pulse("vertical", sent)
                    executed_pulses += 1
                    pulse_count += 1
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
                                "message": (
                                    "Right-turn pulse completed; keeping point-3 position "
                                    "frozen and returning directly to optical heading."
                                ),
                            }
                        )
                        time.sleep(0.08)
                        continue
                    after_gate = wait_for_pose_captured_after(pulse_completed_unix, current_gate)
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
                        after_heading = pose_gate_heading(
                            after_gate,
                            heading_trim_rad + float(calibrated_heading_offset_rad or 0.0),
                        )
                        after_direction = normalize_xz(target_direction_xz(after_position, target))
                        after_angle = signed_angle_xz(after_heading, after_direction)
                        if after_angle is not None:
                            if turn_direction_override is not None and not travel_started:
                                # A forced long turn initially increases the
                                # shortest-angle error by design.  Do not
                                # mistake that for a reversed DJI yaw channel.
                                wrong_yaw_pulses = 0
                                continue
                            before_error = abs(angle)
                            after_error = abs(after_angle)
                            if after_error > before_error + math.radians(2.0):
                                wrong_yaw_pulses += 1
                            elif after_error < before_error - math.radians(1.0):
                                wrong_yaw_pulses = 0
                            required_wrong_yaw_pulses = 3 if yaw_flip_count == 0 else 5
                            if wrong_yaw_pulses >= required_wrong_yaw_pulses:
                                if yaw_flip_count == 0:
                                    yaw_sign *= -1.0
                                    yaw_flip_count += 1
                                    wrong_yaw_pulses = 0
                                    rc_summary["adaptive_axis"]["yaw_sign"] = yaw_sign
                                    rc_summary["adaptive_axis"]["yaw_flips"] = yaw_flip_count
                                    publish_progress(
                                        {
                                            "phase": "yaw_sign_correction",
                                            "step_index": idx,
                                            "step_title": title,
                                            "message": "Yaw moved away from the route twice; reversing yaw sign before forward motion.",
                                        }
                                    )
                                else:
                                    abort_reason = (
                                        "yaw alignment did not converge after automatic sign correction; "
                                        "hovering without forward movement"
                                    )
                                    break
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
            "max_scan_yaw_rc": max_scan_yaw_rc,
            "allow_patrol_scan_yaw": allow_patrol_scan_yaw,
            "alignment_grace_seconds": alignment_grace_seconds,
            "max_vertical_rc": max_vertical_rc,
            "max_step_seconds": max_step_seconds,
            "arrival_radius_map_units": arrival_radius,
            "arrival_deadband_map_units": arrival_deadband,
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
    ap.add_argument("--enemy-model", type=Path, default=None, help="Optional trained YOLO model used for live enemy-drone detection.")
    ap.add_argument("--enemy-output", type=Path, default=None, help="Browser-visible enemy detection JSON output.")
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
                        enemy_detection_path=enemy_output_path,
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
                                append_jsonl(control_history_path, {**control_result, "event": "rejected"})
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
