#!/usr/bin/env python3
from __future__ import annotations

import cgi
import argparse
import csv
import hashlib
import json
import math
import mimetypes
import os
import re
import select
import shutil
import signal
import subprocess
import threading
import time
import urllib.parse
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    from colmap_io import camera_center as colmap_camera_center
    from colmap_io import qvec_to_rotmat, read_images_text
except Exception:
    colmap_camera_center = None
    qvec_to_rotmat = None
    read_images_text = None


ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "viewer"
PUBLIC = VIEWER / "public"
MAPS_DIR = PUBLIC / "maps"
MAP_MANIFEST = MAPS_DIR / "manifest.json"
ENEMY_DIR = PUBLIC / "enemy_drones"
ENEMY_MANIFEST = ENEMY_DIR / "manifest.json"
LIVE_ENEMY_DETECTION_CONTROL = PUBLIC / "live_dji" / "enemy_detection_control.json"
LIVE_ENEMY_DETECTION_OUTPUT = PUBLIC / "live_dji" / "enemy_detections.json"
FLEET_DIR = PUBLIC / "fleet"
FLEET_MANIFEST = FLEET_DIR / "manifest.json"
CAMERA_PATH_LAB_DIR = PUBLIC / "camera_path_lab"
CAMERA_PATH_PREVIEW_CALIBRATIONS = CAMERA_PATH_LAB_DIR / "preview_calibrations.json"
CONFIG = ROOT / "config.json"
DEFAULT_PORT = 8765
VIDEO_REVIEW_MEDIA = ROOT / "outputs" / "spy_demo" / "ATLAS_CINEMATIC_PRODUCT_STORY_2MIN.mp4"
VIDEO_REVIEW_DIR = ROOT / "outputs" / "spy_demo" / "reviews"
VIDEO_REVIEW_JSON = VIDEO_REVIEW_DIR / "ATLAS_CINEMATIC_PRODUCT_STORY_review.json"
VIDEO_REVIEW_BRIEF = VIDEO_REVIEW_DIR / "ATLAS_CINEMATIC_PRODUCT_STORY_adjustment_brief.md"
VIDEO_REVIEW_LOCK = threading.RLock()

STATE_LOCK = threading.Lock()
STATE = {
    "map": {
        "status": "idle",
        "message": "No generated map yet.",
        "log": [],
        "updated_at": None,
        "live_preview": None,
        "frames_saved": 0,
        "capture_started_at": None,
    },
    "drone": {
        "status": "idle",
        "message": "No drone replay yet.",
        "log": [],
        "updated_at": None,
        "live_stream": None,
    },
    "enemy": {
        "status": "idle",
        "message": "Enemy detector idle.",
        "log": [],
        "updated_at": None,
    },
    "current_map_frames": None,
    "selected_map_id": "default_demo",
}
MAP_STOP_EVENT = threading.Event()
DRONE_STOP_EVENT = threading.Event()
DRONE_JOB_ACTIVE = threading.Event()
DRONE_JOB_LIFECYCLE_LOCK = threading.Lock()
DRONE_JOB_WORKER_IDENT: int | None = None
ACTIVE_JOB_STATES = {"queued", "running", "stopping"}
COLMAP_QUERY_POSE_CACHE: dict[str, tuple[float, dict[str, dict]]] = {}
ACTIVE_PROCS_LOCK = threading.Lock()
ACTIVE_PROCS: dict[str, set[subprocess.Popen]] = {"map": set(), "drone": set(), "enemy": set()}
FLEET_LOCK = threading.RLock()
FLEET_SESSIONS: dict[str, dict] = {}
LIBRARY_LOCK = threading.RLock()
CAMERA_PATH_LAB_LOCK = threading.RLock()
CAMERA_PATH_LAB_STATE = {
    "status": "idle",
    "message": "Choose a phone video to begin camera tracking.",
    "updated_at": None,
    "stream": None,
}


def empty_video_review() -> dict:
    return {
        "version": 1,
        "video": VIDEO_REVIEW_MEDIA.name,
        "updated_at": None,
        "annotations": [],
    }


def normalize_video_review(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise RuntimeError("Review payload must be a JSON object.")
    raw_annotations = payload.get("annotations")
    if raw_annotations is None:
        raw_annotations = []
    if not isinstance(raw_annotations, list):
        raise RuntimeError("annotations must be a list.")
    if len(raw_annotations) > 500:
        raise RuntimeError("A review can contain at most 500 annotations.")

    categories = {"visual", "edit", "music", "text", "technical", "other"}
    priorities = {"P1", "P2", "P3"}
    normalized = []
    for index, item in enumerate(raw_annotations):
        if not isinstance(item, dict):
            raise RuntimeError(f"Annotation {index + 1} must be an object.")
        note = str(item.get("note") or "").strip()
        if not note:
            raise RuntimeError(f"Annotation {index + 1} is missing its note.")
        if len(note) > 2000:
            raise RuntimeError(f"Annotation {index + 1} is longer than 2,000 characters.")
        try:
            start = max(0.0, float(item.get("start", 0)))
        except (TypeError, ValueError):
            raise RuntimeError(f"Annotation {index + 1} has an invalid start time.")
        raw_end = item.get("end")
        try:
            end = None if raw_end in (None, "") else max(start, float(raw_end))
        except (TypeError, ValueError):
            raise RuntimeError(f"Annotation {index + 1} has an invalid end time.")
        category = str(item.get("category") or "visual").lower()
        priority = str(item.get("priority") or "P2").upper()
        if category not in categories:
            category = "other"
        if priority not in priorities:
            priority = "P2"
        point = item.get("point")
        normalized_point = None
        if isinstance(point, dict):
            try:
                x = min(1.0, max(0.0, float(point.get("x"))))
                y = min(1.0, max(0.0, float(point.get("y"))))
                normalized_point = {"x": round(x, 5), "y": round(y, 5)}
            except (TypeError, ValueError):
                normalized_point = None
        normalized.append(
            {
                "id": re.sub(r"[^A-Za-z0-9_-]", "", str(item.get("id") or uuid.uuid4().hex))[:80],
                "start": round(start, 3),
                "end": None if end is None else round(end, 3),
                "category": category,
                "priority": priority,
                "note": note,
                "point": normalized_point,
                "created_at": str(item.get("created_at") or "")[:64] or None,
            }
        )
    normalized.sort(key=lambda item: (item["start"], item["priority"], item["id"]))
    return {
        "version": 1,
        "video": VIDEO_REVIEW_MEDIA.name,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "annotations": normalized,
    }


def video_review_timecode(seconds: float) -> str:
    total_ms = max(0, round(float(seconds) * 1000))
    minutes, remainder = divmod(total_ms, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def video_review_markdown(review: dict) -> str:
    lines = [
        "# ATLAS Cinematic Product Story — Adjustment Brief",
        "",
        f"Video: `{review.get('video') or VIDEO_REVIEW_MEDIA.name}`  ",
        f"Updated: {review.get('updated_at') or 'not saved'}  ",
        f"Annotations: {len(review.get('annotations') or [])}",
        "",
    ]
    annotations = review.get("annotations") or []
    if not annotations:
        lines.append("No annotations yet.")
    for index, item in enumerate(annotations, start=1):
        start = video_review_timecode(item["start"])
        end = item.get("end")
        timing = start if end is None else f"{start}–{video_review_timecode(end)}"
        point = item.get("point")
        location = ""
        if point:
            location = f" · frame position {point['x'] * 100:.1f}% x, {point['y'] * 100:.1f}% y"
        lines.extend(
            [
                f"## {index}. {timing} · {item['priority']} · {item['category'].title()}",
                "",
                f"{item['note']}{location}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def load_video_review() -> dict:
    with VIDEO_REVIEW_LOCK:
        if not VIDEO_REVIEW_JSON.is_file():
            return empty_video_review()
        try:
            return normalize_video_review(json.loads(VIDEO_REVIEW_JSON.read_text(encoding="utf-8")))
        except Exception:
            return empty_video_review()


def save_video_review(payload: object) -> dict:
    review = normalize_video_review(payload)
    brief = video_review_markdown(review)
    with VIDEO_REVIEW_LOCK:
        VIDEO_REVIEW_DIR.mkdir(parents=True, exist_ok=True)
        json_tmp = VIDEO_REVIEW_JSON.with_suffix(".json.tmp")
        brief_tmp = VIDEO_REVIEW_BRIEF.with_suffix(".md.tmp")
        json_tmp.write_text(json.dumps(review, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        brief_tmp.write_text(brief, encoding="utf-8")
        json_tmp.replace(VIDEO_REVIEW_JSON)
        brief_tmp.replace(VIDEO_REVIEW_BRIEF)
    return review


def parse_http_byte_range(value: str, size: int) -> tuple[int, int]:
    """Parse one RFC 7233 byte range into an inclusive (start, end)."""
    if size <= 0 or not value.startswith("bytes=") or "," in value:
        raise ValueError("Unsupported byte range")
    left, separator, right = value[6:].strip().partition("-")
    if not separator:
        raise ValueError("Malformed byte range")
    if not left:
        suffix = int(right)
        if suffix <= 0:
            raise ValueError("Invalid suffix range")
        return max(0, size - suffix), size - 1
    start = int(left)
    end = int(right) if right else size - 1
    if start < 0 or start >= size or end < start:
        raise ValueError("Unsatisfiable byte range")
    return start, min(end, size - 1)


def register_active_proc(kind: str, proc: subprocess.Popen) -> None:
    with ACTIVE_PROCS_LOCK:
        ACTIVE_PROCS.setdefault(kind, set()).add(proc)


def unregister_active_proc(kind: str, proc: subprocess.Popen) -> None:
    with ACTIVE_PROCS_LOCK:
        ACTIVE_PROCS.setdefault(kind, set()).discard(proc)


def active_proc_count(kind: str) -> int:
    with ACTIVE_PROCS_LOCK:
        procs = list(ACTIVE_PROCS.get(kind, set()))
    return sum(1 for proc in procs if proc.poll() is None)


def terminate_active_procs(kind: str) -> int:
    with ACTIVE_PROCS_LOCK:
        procs = list(ACTIVE_PROCS.get(kind, set()))
    terminated = 0
    for proc in procs:
        if proc.poll() is not None:
            continue
        append_log(kind, "Cancellation requested; terminating active subprocess immediately.")
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        terminated += 1
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            append_log(kind, "Subprocess did not exit after SIGTERM; killing it.")
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                append_log(kind, "Subprocess still did not exit after SIGKILL.")
    return terminated


def terminate_orphan_live_drone_procs() -> int:
    """Stop live-drone children that survived a server restart.

    The normal path tracks subprocess handles in ACTIVE_PROCS. If the server is
    restarted while a live run is active, those handles are lost but the child
    processes can keep receiving frames and running localization. This fallback
    is intentionally narrow: it only touches this project root and the two live
    scripts used by Start Live ATLAS.
    """
    needles = ("atlas_dji_live_bridge.py", "run_bounded_tsolve_video_stream.py")
    try:
        proc = subprocess.run(
            ["ps", "-ax", "-o", "pid=,command="],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError as exc:
        append_log("drone", f"Could not inspect live drone processes: {exc}")
        return 0

    killed = 0
    root_text = str(ROOT)
    own_pid = os.getpid()
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        first, _, command = line.partition(" ")
        try:
            pid = int(first)
        except ValueError:
            continue
        if pid == own_pid:
            continue
        if root_text not in command:
            continue
        if not any(needle in command for needle in needles):
            continue
        try:
            os.killpg(pid, signal.SIGTERM)
            killed += 1
            append_log("drone", f"Stopped orphan live process {pid}: {command[:180]}")
        except ProcessLookupError:
            continue
        except OSError as exc:
            try:
                os.kill(pid, signal.SIGTERM)
                killed += 1
                append_log("drone", f"Stopped orphan live process {pid}: {command[:180]}")
            except ProcessLookupError:
                continue
            except OSError as fallback_exc:
                append_log("drone", f"Could not stop orphan live process {pid}: {exc}; fallback failed: {fallback_exc}")
    return killed


def touch_stop_file(path: str | Path | None, reason: str = "stop requested") -> None:
    if not path:
        return
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"{reason}\n{time.time():.6f}\n", encoding="utf-8")
    except OSError as exc:
        append_log("drone", f"Could not write live stop file {path}: {exc}")


def mark_live_dji_status_stopped(message: str) -> None:
    def read_status(path: Path) -> dict:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return {}
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            # A legacy non-atomic writer could leave bytes after an otherwise
            # complete JSON object when Stop overlapped the bridge heartbeat.
            # Recover that first object so Stop can still terminate the worker.
            try:
                payload, _end = json.JSONDecoder().raw_decode(text.lstrip())
            except json.JSONDecodeError:
                return {}
        return payload if isinstance(payload, dict) else {}

    def cancel_running_control(control: dict) -> dict:
        if not isinstance(control, dict) or control.get("status") != "running":
            return control
        return {
            **control,
            "ok": False,
            "status": "cancelled",
            "message": message,
            "error": message,
            "aborted": True,
            "cancelled_at": time.time(),
            "updated_at": time.time(),
        }

    status_paths = [PUBLIC / "live_dji" / "status.json"]
    latest = read_status(status_paths[0])
    session = latest.get("session")
    if session:
        status_paths.append(PUBLIC / "live_dji_sessions" / str(session) / "status.json")
    for path in status_paths:
        try:
            payload = read_status(path) if path.exists() else dict(latest)
            payload.update(
                {
                    "status": "stopped",
                    "message": message,
                    "stopped_at": time.time(),
                    "updated_at": time.time(),
                }
            )
            if isinstance(payload.get("last_control"), dict):
                payload["last_control"] = cancel_running_control(payload["last_control"])
            atomic_write_json(path, payload)
        except OSError as exc:
            append_log("drone", f"Could not mark DJI live status stopped at {path}: {exc}")
    control_path = PUBLIC / "live_dji" / "control_status.json"
    control = read_status(control_path)
    if control.get("status") == "running":
        atomic_write_json(control_path, cancel_running_control(control))


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def make_map_id(prefix: str) -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def public_rel(path: Path) -> str:
    return str(path.relative_to(VIEWER)).replace(os.sep, "/")


def now_label() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def slugify_label(value: str, fallback: str = "enemy_drone") -> str:
    cleaned = []
    last_was_sep = False
    for ch in str(value or "").lower():
        if ch.isalnum():
            cleaned.append(ch)
            last_was_sep = False
        elif not last_was_sep:
            cleaned.append("_")
            last_was_sep = True
    slug = "".join(cleaned).strip("_")
    return slug or fallback


def pose_stream_counts(poses: list[dict] | object) -> dict:
    if not isinstance(poses, list):
        poses = []
    accepted = 0
    held = 0
    failed = 0
    for pose in poses:
        if not isinstance(pose, dict):
            continue
        if pose.get("held_pose"):
            held += 1
        elif bool(pose.get("success")) and (pose.get("center") or pose.get("rcenter")):
            accepted += 1
        else:
            failed += 1
    return {
        "poses": accepted,
        "accepted": accepted,
        "frames": len(poses),
        "held": held,
        "failed": failed,
    }


def read_counts(asset_dir: Path) -> dict:
    scene_path = asset_dir / "scene.json"
    pose_path = asset_dir / "poses.json"
    counts = {"points": 0, "dense_points": 0, "cameras": 0, "poses": 0, "frames": 0, "held": 0}
    if scene_path.exists():
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
        counts["points"] = len(scene.get("points3D", []))
        counts["dense_points"] = len(scene.get("dense_points3D", []))
        counts["cameras"] = len(scene.get("map_cameras", []))
    if pose_path.exists():
        poses = json.loads(pose_path.read_text(encoding="utf-8"))
        counts.update(pose_stream_counts(poses.get("poses", [])))
    return counts


def normalize_replays(entry: dict) -> dict:
    counts = entry.get("counts") or {}
    replays = list(entry.get("replays") or [])
    if not replays and (entry.get("has_drone_demo") or int(counts.get("poses") or 0) > 0):
        replays = [
            {
                "id": "base",
                "title": "Built-in Drone Path",
                "asset_base": entry.get("asset_base", "public"),
                "created_at": entry.get("created_at"),
                "source_video": "bundled replay",
                "built_in": True,
                "counts": {"poses": int(counts.get("poses") or 0)},
            }
        ]
    if replays:
        entry["replays"] = replays
        active = entry.get("active_replay_id")
        if active not in {r.get("id") for r in replays}:
            entry["active_replay_id"] = replays[-1].get("id")
    else:
        entry.pop("replays", None)
        entry.pop("active_replay_id", None)
    entry["has_drone_demo"] = bool(replays)
    return entry


def default_library() -> dict:
    asset_base = PUBLIC
    default_asset = MAPS_DIR / "default_demo"
    if (default_asset / "scene.json").exists():
        asset_base = default_asset
    counts = read_counts(asset_base)
    entry = normalize_replays(
        {
            "id": "default_demo",
            "title": "Indoor Patrol Map",
            "description": "Original TSolve drone replay map with DJI Mini 3 Pro route.",
            "asset_base": public_rel(asset_base),
            "frames_path": str(ROOT / "data" / "map_frames"),
            "deletable": False,
            "kind": "demo",
            "created_at": None,
            "counts": counts,
            "has_drone_demo": counts["poses"] > 0,
        }
    )
    return {
        "selected_map_id": "default_demo",
        "maps": [entry],
    }


def load_library() -> dict:
    if not MAP_MANIFEST.exists():
        return default_library()
    lib = json.loads(MAP_MANIFEST.read_text(encoding="utf-8"))
    by_id = {m["id"]: m for m in lib.get("maps", [])}
    hidden_builtin_ids = set(lib.get("hidden_builtin_ids") or [])
    if "default_demo" not in by_id and "default_demo" not in hidden_builtin_ids:
        base = default_library()
        lib["maps"] = base["maps"] + lib.get("maps", [])
    map_ids = {m["id"] for m in lib.get("maps", [])}
    if not lib.get("selected_map_id") or lib["selected_map_id"] not in map_ids:
        lib["selected_map_id"] = next(iter(map_ids), "")
    lib["maps"] = [normalize_replays(m) for m in lib.get("maps", [])]
    return lib


def save_library(lib: dict) -> None:
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    # ThreadingHTTPServer can save patrol/map metadata from overlapping
    # requests. Replace the complete file atomically so readers never observe
    # the zero-length interval created by Path.write_text().
    atomic_write_json(MAP_MANIFEST, lib)


def resolved_live_patrol_lock(lib: dict) -> dict | None:
    """Resolve the optional map/patrol pair reserved for live patrol flights."""
    raw = lib.get("live_patrol_lock") if isinstance(lib, dict) else None
    if not isinstance(raw, dict) or raw.get("enabled") is not True:
        return None
    map_id = str(raw.get("map_id") or "").strip()
    patrol_id = str(raw.get("patrol_id") or "").strip()
    map_entry = next(
        (item for item in lib.get("maps", []) if isinstance(item, dict) and item.get("id") == map_id),
        None,
    )
    if map_entry is None:
        raise RuntimeError(f"Pinned live patrol map no longer exists: {map_id or 'missing map id'}")
    patrol = next(
        (item for item in map_entry.get("patrols", []) if isinstance(item, dict) and item.get("id") == patrol_id),
        None,
    )
    if patrol is None:
        raise RuntimeError(f"Pinned live patrol no longer exists on {map_entry.get('title') or map_id}.")
    baseline_replay_id = str(raw.get("baseline_replay_id") or "").strip()
    if baseline_replay_id:
        replay = next(
            (
                item
                for item in map_entry.get("replays", [])
                if isinstance(item, dict) and str(item.get("id") or "") == baseline_replay_id
            ),
            None,
        )
        if replay is None or str(replay.get("kind") or "") != "route_constrained_taught_baseline":
            raise RuntimeError(
                f"Pinned live patrol baseline is unavailable or invalid: {baseline_replay_id}"
            )
        baseline_reference = (
            map_asset_dir(map_entry)
            / "replays"
            / baseline_replay_id
            / "reference_candidate.json"
        )
        try:
            baseline = json.loads(baseline_reference.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Pinned live patrol baseline cannot be read: {exc}") from exc
        if (
            baseline.get("map_id") != map_id
            or baseline.get("patrol_id") != patrol_id
            or baseline.get("complete_loop") is not True
            or baseline.get("enabled_for_live_route_gate") is not True
        ):
            raise RuntimeError("Pinned live patrol baseline failed its route-gate metadata check.")
        visual_recovery_path = ""
        if baseline.get("enabled_for_visual_route_recovery") is True:
            bank_name = str(baseline.get("visual_route_recovery_bank") or "").strip()
            audit_name = str(baseline.get("visual_route_recovery_audit") or "").strip()
            if (
                not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", bank_name)
                or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", audit_name)
            ):
                raise RuntimeError("Pinned patrol visual recovery metadata is invalid.")
            visual_bank = baseline_reference.parent / bank_name
            visual_audit_path = baseline_reference.parent / audit_name
            try:
                visual_audit = json.loads(visual_audit_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Pinned patrol visual recovery audit cannot be read: {exc}") from exc
            if not visual_bank.is_file() or visual_audit.get("passed") is not True:
                raise RuntimeError("Pinned patrol visual recovery bank has not passed its offline audit.")
            visual_recovery_path = str(visual_bank)
    else:
        baseline_reference = None
        visual_recovery_path = ""
    try:
        patrol_laps = int(raw.get("patrol_laps") or 0)
    except (TypeError, ValueError):
        patrol_laps = 0
    return {
        "map_id": map_id,
        "map_title": str(map_entry.get("title") or map_id),
        "patrol_id": patrol_id,
        "patrol_title": str(patrol.get("title") or patrol_id),
        "flight_enabled": raw.get("flight_enabled") is not False,
        "suspension_reason": str(raw.get("suspension_reason") or "").strip(),
        "reference_profile": (
            "live-atlas-141441"
            if str(raw.get("reference_profile") or "").strip().lower() == "live-atlas-141441"
            else "off"
        ),
        "baseline_replay_id": baseline_replay_id,
        "baseline_reference_path": str(baseline_reference) if baseline_reference else "",
        "visual_recovery_path": visual_recovery_path,
        "patrol_laps": max(0, min(20, patrol_laps)),
    }


def default_enemy_library() -> dict:
    return {
        "version": 1,
        "updated_at": now_label(),
        "selected_model": None,
        "model_status": "not_trained",
        "enemies": [],
    }


def normalize_enemy_range_calibration(raw: object) -> dict:
    calibration = raw if isinstance(raw, dict) else {}
    samples = []
    for item in calibration.get("samples") or []:
        if not isinstance(item, dict):
            continue
        try:
            measured = float(item.get("measured_clearance_m"))
            width = float(item.get("box_width"))
            height = float(item.get("box_height"))
            area = float(item.get("box_area") or width * height)
            confidence = float(item.get("confidence") or 0.0)
        except (TypeError, ValueError):
            continue
        if not (
            math.isfinite(measured)
            and math.isfinite(width)
            and math.isfinite(height)
            and math.isfinite(area)
            and 0.20 <= measured <= 10.0
            and 0.001 <= width <= 1.0
            and 0.001 <= height <= 1.0
            and 1e-6 <= area <= 1.0
        ):
            continue
        samples.append(
            {
                "id": str(item.get("id") or f"range_{uuid.uuid4().hex[:10]}"),
                "measured_clearance_m": measured,
                "box_width": width,
                "box_height": height,
                "box_area": area,
                "confidence": max(0.0, min(1.0, confidence)),
                "frame": str(item.get("frame") or ""),
                "captured_at": str(item.get("captured_at") or now_label()),
            }
        )
    samples = samples[-240:]
    model = calibration.get("model") if isinstance(calibration.get("model"), dict) else None
    if model is not None:
        try:
            scale = float(model.get("scale"))
            margin = float(model.get("conservative_margin_m"))
        except (TypeError, ValueError):
            model = None
        else:
            model_type = str(model.get("type") or "inverse_sqrt_area")
            if (
                model_type not in {"inverse_width", "inverse_sqrt_area", "inverse_height"}
                or not (math.isfinite(scale) and scale > 0 and math.isfinite(margin) and margin >= 0)
            ):
                model = None
            else:
                model = {
                    **model,
                    "type": model_type,
                    "scale": scale,
                    "conservative_margin_m": margin,
                }
    status = str(calibration.get("status") or ("needs_validation" if samples else "needs_samples"))
    if status not in {"needs_samples", "needs_validation", "validated", "rejected"}:
        status = "needs_validation" if samples else "needs_samples"
    if status == "validated" and model is None:
        status = "needs_validation"
    return {
        "status": status,
        "samples": samples,
        "sample_count": len(samples),
        "model": model,
        "validation": calibration.get("validation") if isinstance(calibration.get("validation"), dict) else None,
        "updated_at": str(calibration.get("updated_at") or ""),
    }


def normalize_enemy_profile(profile: dict) -> dict:
    now = now_label()
    name = " ".join(str(profile.get("name") or "Enemy Drone").split())[:80] or "Enemy Drone"
    enemy_id = str(profile.get("id") or f"enemy_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}")
    class_name = slugify_label(str(profile.get("class_name") or name), "enemy_drone")
    videos = []
    for raw in profile.get("videos") or []:
        if not isinstance(raw, dict):
            continue
        videos.append(
            {
                "id": str(raw.get("id") or f"clip_{uuid.uuid4().hex[:8]}"),
                "filename": str(raw.get("filename") or "calibration_video.mp4"),
                "url": str(raw.get("url") or ""),
                "size_bytes": int(raw.get("size_bytes") or 0),
                "uploaded_at": str(raw.get("uploaded_at") or now),
            }
        )
    frames = []
    for raw in profile.get("frames") or []:
        if not isinstance(raw, dict):
            continue
        status = str(raw.get("status") or "unlabeled")
        if status not in {"unlabeled", "labeled", "review", "skipped", "negative"}:
            status = "unlabeled"
        box = raw.get("box") if isinstance(raw.get("box"), dict) else None
        if box:
            box = {
                "x_center": float(box.get("x_center") or 0.0),
                "y_center": float(box.get("y_center") or 0.0),
                "width": float(box.get("width") or 0.0),
                "height": float(box.get("height") or 0.0),
            }
        frames.append(
            {
                "id": str(raw.get("id") or f"frame_{uuid.uuid4().hex[:10]}"),
                "filename": str(raw.get("filename") or "frame.jpg"),
                "url": str(raw.get("url") or ""),
                "label_url": str(raw.get("label_url") or ""),
                "source_video_id": str(raw.get("source_video_id") or ""),
                "source_filename": str(raw.get("source_filename") or ""),
                "time_sec": float(raw.get("time_sec") or 0.0),
                "width": int(raw.get("width") or 0),
                "height": int(raw.get("height") or 0),
                "status": status,
                "box": box,
            }
        )
    training_status = str(profile.get("training_status") or ("needs_labels" if videos else "needs_videos"))
    labeled_count = sum(1 for f in frames if f.get("status") == "labeled")
    return {
        "id": enemy_id,
        "name": name,
        "class_name": class_name,
        "created_at": str(profile.get("created_at") or now),
        "updated_at": str(profile.get("updated_at") or now),
        "videos": videos,
        "frames": frames,
        "video_count": len(videos),
        "frame_count": len(frames),
        "labeled_frame_count": labeled_count,
        "review_frame_count": sum(1 for f in frames if f.get("status") == "review"),
        "skipped_frame_count": sum(1 for f in frames if f.get("status") == "skipped"),
        "negative_frame_count": sum(1 for f in frames if f.get("status") == "negative"),
        "training_status": training_status,
        "model_status": str(profile.get("model_status") or "not_trained"),
        "dataset_manifest": str(profile.get("dataset_manifest") or ""),
        "range_calibration": normalize_enemy_range_calibration(profile.get("range_calibration")),
        "notes": str(profile.get("notes") or ""),
    }


def load_enemy_library() -> dict:
    if not ENEMY_MANIFEST.exists():
        return default_enemy_library()
    try:
        lib = json.loads(ENEMY_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        lib = default_enemy_library()
    enemies = [normalize_enemy_profile(e) for e in lib.get("enemies", []) if isinstance(e, dict)]
    enemies.sort(key=lambda e: (e.get("created_at") or "", e.get("name") or ""))
    lib["version"] = int(lib.get("version") or 1)
    lib["updated_at"] = str(lib.get("updated_at") or now_label())
    lib["selected_model"] = lib.get("selected_model")
    lib["model_status"] = str(lib.get("model_status") or "not_trained")
    lib["enemies"] = enemies
    lib["total_videos"] = sum(len(e.get("videos", [])) for e in enemies)
    lib["total_frames"] = sum(len(e.get("frames", [])) for e in enemies)
    lib["total_labeled_frames"] = sum(e.get("labeled_frame_count", 0) for e in enemies)
    lib["class_count"] = len(enemies)
    return lib


def save_enemy_library(lib: dict) -> None:
    ENEMY_DIR.mkdir(parents=True, exist_ok=True)
    lib["updated_at"] = now_label()
    temporary = ENEMY_MANIFEST.with_name(f".{ENEMY_MANIFEST.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(lib, indent=2), encoding="utf-8")
    temporary.replace(ENEMY_MANIFEST)


def get_enemy_profile(enemy_id: str) -> tuple[dict, dict]:
    lib = load_enemy_library()
    for profile in lib.get("enemies", []):
        if profile.get("id") == enemy_id:
            return lib, profile
    raise RuntimeError(f"Unknown enemy drone profile: {enemy_id}")


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return float("nan")
    position = max(0.0, min(1.0, float(quantile))) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def live_enemy_detection_payload() -> dict:
    path = LIVE_ENEMY_DETECTION_OUTPUT
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError("Start Live ATLAS and wait for a current NEO detection before saving a range sample.") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"The live enemy-detection result is unavailable: {exc}") from exc
    try:
        age = time.time() - float(payload.get("updated_at") or 0.0)
    except (TypeError, ValueError):
        age = float("inf")
    if age < 0 or age > 2.0:
        raise RuntimeError(f"The live enemy detection is stale ({age:.1f}s old). Hold the NEO in view and try again.")
    return payload


def set_live_enemy_detection_enabled(
    enabled: bool,
    control_path: Path | None = None,
    detection_path: Path | None = None,
) -> dict:
    """Set the live detector gate without issuing any DJI flight command."""
    control = control_path or LIVE_ENEMY_DETECTION_CONTROL
    detections = detection_path or LIVE_ENEMY_DETECTION_OUTPUT
    now = time.time()
    payload = {
        "enabled": bool(enabled),
        "updated_at": now,
        "source": "operator_ui",
    }
    atomic_write_json(control, payload)
    if not enabled:
        atomic_write_json(
            detections,
            {
                "status": "disabled",
                "message": "Enemy detection is disabled by the operator.",
                "detections": [],
                "updated_at": now,
            },
        )
    return payload


def capture_enemy_range_sample(
    enemy_id: str,
    measured_clearance_m: float,
    detection_payload: dict | None = None,
) -> dict:
    lib, profile = get_enemy_profile(enemy_id)
    measured = float(measured_clearance_m)
    if not math.isfinite(measured) or not 0.20 <= measured <= 10.0:
        raise RuntimeError("Measured body-to-body clearance must be between 0.20 m and 10.0 m.")
    payload = detection_payload if isinstance(detection_payload, dict) else live_enemy_detection_payload()
    detections = payload.get("detections") if isinstance(payload.get("detections"), list) else []
    target_class = str(profile.get("class_name") or "").strip().lower()
    candidates = []
    for detection in detections:
        if not isinstance(detection, dict) or not isinstance(detection.get("box"), dict):
            continue
        class_name = str(detection.get("class_name") or "").strip().lower()
        if target_class and class_name and class_name != target_class:
            continue
        try:
            confidence = float(detection.get("confidence") or 0.0)
            width = float(detection["box"].get("width") or 0.0)
            height = float(detection["box"].get("height") or 0.0)
        except (TypeError, ValueError):
            continue
        area = width * height
        if confidence >= 0.35 and 0.001 <= width <= 1.0 and 0.001 <= height <= 1.0 and area >= 1e-6:
            candidates.append((confidence, width, height, area))
    if not candidates:
        raise RuntimeError("No fresh matching NEO detection above 35% confidence is available for this range sample.")
    confidence, width, height, area = max(candidates, key=lambda item: item[0])
    calibration = normalize_enemy_range_calibration(profile.get("range_calibration"))
    calibration["samples"].append(
        {
            "id": f"range_{uuid.uuid4().hex[:10]}",
            "measured_clearance_m": measured,
            "box_width": width,
            "box_height": height,
            "box_area": area,
            "confidence": confidence,
            "frame": str(payload.get("frame") or ""),
            "captured_at": now_label(),
        }
    )
    calibration["samples"] = calibration["samples"][-240:]
    calibration["sample_count"] = len(calibration["samples"])
    calibration["status"] = "needs_validation"
    calibration["model"] = None
    calibration["validation"] = None
    calibration["updated_at"] = now_label()
    profile["range_calibration"] = calibration
    profile["updated_at"] = now_label()
    save_enemy_library(lib)
    return {"enemy": normalize_enemy_profile(profile), "library": load_enemy_library()}


def fit_enemy_range_calibration(enemy_id: str, stop_clearance_m: float = 0.50) -> dict:
    lib, profile = get_enemy_profile(enemy_id)
    stop_clearance = max(0.50, min(2.0, float(stop_clearance_m)))
    calibration = normalize_enemy_range_calibration(profile.get("range_calibration"))
    samples = calibration.get("samples") or []
    if len(samples) < 8:
        raise RuntimeError("Record at least 8 live range samples before validation.")
    distances = [float(sample["measured_clearance_m"]) for sample in samples]
    distance_bins = {round(distance / 0.10) for distance in distances}
    distance_span = max(distances) - min(distances)
    if len(distance_bins) < 4 or distance_span < 0.75:
        raise RuntimeError("Range calibration needs at least 4 distinct distances spanning 0.75 m or more.")
    if min(distances) > stop_clearance + 0.25:
        raise RuntimeError(
            f"Record at least one sample at or below {stop_clearance + 0.25:.2f} m before enabling a {stop_clearance:.2f} m stop clearance."
        )
    feature_models = [
        ("inverse_width", [max(1e-9, float(sample["box_width"])) for sample in samples]),
        ("inverse_sqrt_area", [math.sqrt(max(1e-9, float(sample["box_area"]))) for sample in samples]),
        ("inverse_height", [max(1e-9, float(sample["box_height"])) for sample in samples]),
    ]
    fitted_models = []
    for model_type, features in feature_models:
        candidate_scale = percentile(
            [distance * feature for distance, feature in zip(distances, features)],
            0.50,
        )
        candidate_predictions = [candidate_scale / feature for feature in features]
        candidate_errors = [abs(predicted - measured) for predicted, measured in zip(candidate_predictions, distances)]
        fitted_models.append(
            (
                percentile(candidate_errors, 0.90),
                max(candidate_errors),
                sum(candidate_errors) / len(candidate_errors),
                model_type,
                candidate_scale,
                candidate_predictions,
            )
        )
    _candidate_p90, _candidate_max, _candidate_mae, model_type, scale, predictions = min(fitted_models)
    errors = [abs(predicted - measured) for predicted, measured in zip(predictions, distances)]
    signed_errors = [predicted - measured for predicted, measured in zip(predictions, distances)]
    mae = sum(errors) / len(errors)
    rmse = math.sqrt(sum(error * error for error in signed_errors) / len(signed_errors))
    p90 = percentile(errors, 0.90)
    max_error = max(errors)
    accepted = bool(mae <= 0.18 and p90 <= 0.30 and max_error <= 0.45)
    validation = {
        "accepted": accepted,
        "sample_count": len(samples),
        "distance_count": len(distance_bins),
        "min_clearance_m": min(distances),
        "max_clearance_m": max(distances),
        "distance_span_m": distance_span,
        "mean_absolute_error_m": mae,
        "rmse_m": rmse,
        "p90_absolute_error_m": p90,
        "max_absolute_error_m": max_error,
        "required": {"mae_m": 0.18, "p90_m": 0.30, "max_error_m": 0.45},
        "validated_at": now_label(),
    }
    calibration["status"] = "validated" if accepted else "rejected"
    calibration["validation"] = validation
    calibration["updated_at"] = now_label()
    calibration["model"] = (
        {
            "type": model_type,
            "scale": scale,
            "conservative_margin_m": max(0.12, p90),
            "trained_min_clearance_m": min(distances),
            "trained_max_clearance_m": max(distances),
            "stop_clearance_m": stop_clearance,
            "validated_at": now_label(),
        }
        if accepted
        else None
    )
    profile["range_calibration"] = calibration
    profile["updated_at"] = now_label()
    save_enemy_library(lib)
    return {"enemy": normalize_enemy_profile(profile), "library": load_enemy_library(), "validation": validation}


def reset_enemy_range_calibration(enemy_id: str) -> dict:
    lib, profile = get_enemy_profile(enemy_id)
    profile["range_calibration"] = normalize_enemy_range_calibration(None)
    profile["updated_at"] = now_label()
    save_enemy_library(lib)
    return {"enemy": normalize_enemy_profile(profile), "library": load_enemy_library()}


def upsert_enemy_profile(profile: dict) -> dict:
    lib = load_enemy_library()
    profile = normalize_enemy_profile(profile)
    enemies = [e for e in lib.get("enemies", []) if e.get("id") != profile["id"]]
    enemies.append(profile)
    lib["enemies"] = enemies
    lib["model_status"] = "needs_training"
    save_enemy_library(lib)
    return profile


def upload_enemy_videos(enemy_id: str | None, name: str, fields: list[cgi.FieldStorage]) -> dict:
    if not fields:
        raise RuntimeError("Upload at least one enemy-drone calibration video.")
    now = now_label()
    if enemy_id:
        lib, profile = get_enemy_profile(enemy_id)
        profile["name"] = " ".join(str(name or profile.get("name") or "Enemy Drone").split())[:80] or "Enemy Drone"
        profile["class_name"] = slugify_label(str(profile.get("class_name") or profile["name"]), "enemy_drone")
    else:
        profile = normalize_enemy_profile(
            {
                "name": name or "Enemy Drone",
                "created_at": now,
                "updated_at": now,
                "videos": [],
            }
        )

    profile_dir = ENEMY_DIR / profile["id"]
    saved = save_uploaded_videos(fields, profile_dir / "videos", "enemy_calib")
    videos = list(profile.get("videos") or [])
    for path, original_name in saved:
        videos.append(
            {
                "id": f"clip_{uuid.uuid4().hex[:8]}",
                "filename": original_name,
                "url": public_rel(path),
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "uploaded_at": now,
            }
        )
    profile["videos"] = videos
    profile["updated_at"] = now
    profile["training_status"] = "needs_labels"
    profile["model_status"] = "not_trained"
    return upsert_enemy_profile(profile)


def enemy_class_index(lib: dict, enemy_id: str) -> int:
    for idx, profile in enumerate(lib.get("enemies", [])):
        if profile.get("id") == enemy_id:
            return idx
    raise RuntimeError(f"Unknown enemy drone profile: {enemy_id}")


def enemy_public_path(url: str) -> Path:
    rel = str(url or "").strip().lstrip("/")
    if not rel:
        raise RuntimeError("Missing enemy media path.")
    path = (VIEWER / rel).resolve()
    if not path.is_relative_to(VIEWER.resolve()):
        raise RuntimeError("Enemy media path escaped viewer root.")
    return path


def extract_enemy_frames(
    enemy_id: str,
    fps: float = 2.0,
    max_frames_per_video: int = 180,
    max_size: int = 960,
    force: bool = False,
) -> dict:
    lib, profile = get_enemy_profile(enemy_id)
    videos = list(profile.get("videos") or [])
    if not videos:
        raise RuntimeError("Upload calibration videos before extracting frames.")
    fps = max(0.2, min(float(fps or 2.0), 10.0))
    max_frames_per_video = max(1, min(int(max_frames_per_video or 180), 2000))
    max_size = max(320, min(int(max_size or 960), 2400))

    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"OpenCV is required to extract training frames: {exc}") from exc

    profile_dir = ENEMY_DIR / profile["id"]
    frames_dir = profile_dir / "frames"
    labels_dir = profile_dir / "labels"
    frames_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    frames = list(profile.get("frames") or [])
    if force:
        for old in frames_dir.glob("*.jpg"):
            old.unlink()
        for old in labels_dir.glob("*.txt"):
            old.unlink()
        frames = []
    extracted_sources = {f.get("source_video_id") for f in frames}
    added = 0
    now = now_label()

    for video in videos:
        source_video_id = str(video.get("id") or "")
        if not force and source_video_id in extracted_sources:
            continue
        video_path = enemy_public_path(str(video.get("url") or ""))
        if not video_path.exists():
            continue
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        step = max(1, int(round(source_fps / fps)))
        frame_idx = 0
        kept = 0
        while kept < max_frames_per_video:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step != 0:
                frame_idx += 1
                continue
            h, w = frame.shape[:2]
            scale = min(1.0, max_size / max(w, h))
            if scale < 1.0:
                frame = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
                h, w = frame.shape[:2]
            frame_id = f"{source_video_id}_{frame_idx:06d}"
            filename = f"{frame_id}.jpg"
            out_path = frames_dir / filename
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            label_path = labels_dir / f"{frame_id}.txt"
            frames.append(
                {
                    "id": frame_id,
                    "filename": filename,
                    "url": public_rel(out_path),
                    "label_url": public_rel(label_path),
                    "source_video_id": source_video_id,
                    "source_filename": str(video.get("filename") or ""),
                    "time_sec": frame_idx / source_fps if source_fps > 0 else 0.0,
                    "width": int(w),
                    "height": int(h),
                    "status": "unlabeled",
                    "box": None,
                }
            )
            kept += 1
            added += 1
            frame_idx += 1
        cap.release()

    profile["frames"] = frames
    profile["updated_at"] = now
    profile["training_status"] = "needs_labels"
    profile["model_status"] = "not_trained"
    save_enemy_library(lib)
    return {
        "added": added,
        "enemy": normalize_enemy_profile(profile),
        "library": load_enemy_library(),
    }


def normalize_enemy_box(box: dict | None) -> dict:
    if not isinstance(box, dict):
        raise RuntimeError("A bounding box is required.")
    x = float(box.get("x_center"))
    y = float(box.get("y_center"))
    w = float(box.get("width"))
    h = float(box.get("height"))
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
        raise RuntimeError("YOLO box must be normalized to [0,1] and have positive width/height.")
    return {"x_center": x, "y_center": y, "width": w, "height": h}


def write_enemy_frame_label(lib: dict, profile: dict, frame: dict, box: dict | None, status: str) -> None:
    status = str(status or "labeled")
    if status not in {"labeled", "review", "skipped", "negative", "unlabeled"}:
        raise RuntimeError("Frame label status must be labeled, review, negative, skipped, or unlabeled.")
    label_path = enemy_public_path(frame.get("label_url") or "")
    label_path.parent.mkdir(parents=True, exist_ok=True)

    if status in {"labeled", "review"}:
        normalized = normalize_enemy_box(box)
        frame["box"] = normalized
        frame["status"] = status
        if status == "labeled":
            # This sidecar is only an annotation preview. Dataset preparation
            # rewrites labels using the immutable class map stored with that
            # dataset version.
            class_id = enemy_class_index(lib, profile["id"])
            label_path.write_text(
                (
                    f"{class_id} {normalized['x_center']:.8f} {normalized['y_center']:.8f} "
                    f"{normalized['width']:.8f} {normalized['height']:.8f}\n"
                ),
                encoding="utf-8",
            )
        elif label_path.exists():
            label_path.unlink()
        return

    if label_path.exists():
        label_path.unlink()
    frame["box"] = None
    frame["status"] = status


def update_enemy_training_state(profile: dict) -> None:
    frames = list(profile.get("frames") or [])
    labeled = sum(1 for f in frames if f.get("status") == "labeled")
    profile["training_status"] = "labels_ready" if labeled else "needs_labels"
    profile["model_status"] = "not_trained"
    profile["updated_at"] = now_label()


def save_enemy_frame_label(enemy_id: str, frame_id: str, box: dict | None, status: str) -> dict:
    lib, profile = get_enemy_profile(enemy_id)
    frames = list(profile.get("frames") or [])
    frame = next((f for f in frames if f.get("id") == frame_id), None)
    if not frame:
        raise RuntimeError(f"Unknown enemy frame: {frame_id}")
    write_enemy_frame_label(lib, profile, frame, box, status)

    profile["frames"] = frames
    update_enemy_training_state(profile)
    lib["model_status"] = "needs_training"
    save_enemy_library(lib)
    return {"enemy": normalize_enemy_profile(profile), "library": load_enemy_library()}


def enemy_box_to_pixels(frame: dict, box: dict) -> tuple[int, int, int, int]:
    width = max(1, int(frame.get("width") or 1))
    height = max(1, int(frame.get("height") or 1))
    x1 = int(round((float(box["x_center"]) - float(box["width"]) / 2.0) * width))
    y1 = int(round((float(box["y_center"]) - float(box["height"]) / 2.0) * height))
    x2 = int(round((float(box["x_center"]) + float(box["width"]) / 2.0) * width))
    y2 = int(round((float(box["y_center"]) + float(box["height"]) / 2.0) * height))
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 2, min(width, x2))
    y2 = max(y1 + 2, min(height, y2))
    return x1, y1, x2, y2


def enemy_pixels_to_box(frame: dict, x1: int, y1: int, x2: int, y2: int) -> dict:
    width = max(1, int(frame.get("width") or 1))
    height = max(1, int(frame.get("height") or 1))
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(x1 + 2, min(width, int(x2)))
    y2 = max(y1 + 2, min(height, int(y2)))
    return {
        "x_center": ((x1 + x2) / 2.0) / width,
        "y_center": ((y1 + y2) / 2.0) / height,
        "width": (x2 - x1) / width,
        "height": (y2 - y1) / height,
    }


def track_enemy_labels(
    enemy_id: str,
    frame_id: str,
    box: dict,
    direction: str = "both",
    accept_threshold: float = 0.72,
    review_threshold: float = 0.50,
    search_scale: float = 3.0,
    max_frames: int = 160,
    overwrite: bool = False,
) -> dict:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"OpenCV is required for auto-tracking enemy labels: {exc}") from exc

    lib, profile = get_enemy_profile(enemy_id)
    frames = list(profile.get("frames") or [])
    start_frame = next((f for f in frames if f.get("id") == frame_id), None)
    if not start_frame:
        raise RuntimeError(f"Unknown enemy frame: {frame_id}")
    source_video_id = str(start_frame.get("source_video_id") or "")
    source_frames = [
        f for f in frames
        if str(f.get("source_video_id") or "") == source_video_id
    ]
    source_frames.sort(key=lambda f: (float(f.get("time_sec") or 0.0), str(f.get("id") or "")))
    start_index = next((i for i, f in enumerate(source_frames) if f.get("id") == frame_id), -1)
    if start_index < 0:
        raise RuntimeError("Could not locate the starting frame inside its source video.")

    start_box = normalize_enemy_box(box)
    start_image = cv2.imread(str(enemy_public_path(start_frame.get("url") or "")), cv2.IMREAD_GRAYSCALE)
    if start_image is None:
        raise RuntimeError("Could not read the starting calibration frame.")
    sx1, sy1, sx2, sy2 = enemy_box_to_pixels(start_frame, start_box)
    template = start_image[sy1:sy2, sx1:sx2]
    if template.size == 0 or template.shape[0] < 8 or template.shape[1] < 8:
        raise RuntimeError("The starting box is too small for reliable auto-tracking.")

    direction = str(direction or "both").lower()
    accept_threshold = max(0.05, min(0.99, float(accept_threshold)))
    review_threshold = max(0.0, min(accept_threshold, float(review_threshold)))
    search_scale = max(1.2, min(8.0, float(search_scale)))
    max_frames = max(1, min(2000, int(max_frames)))

    write_enemy_frame_label(lib, profile, start_frame, start_box, "labeled")
    counts = {"labeled": 1, "review": 0, "stopped": 0, "processed": 1}
    track_reports = []

    def run_one_way(indices: list[int]) -> None:
        nonlocal template
        previous_box = dict(start_box)
        processed = 0
        current_template = template.copy()
        for idx in indices:
            if processed >= max_frames:
                break
            frame = source_frames[idx]
            if not overwrite and frame.get("id") != frame_id and frame.get("status") in {"labeled", "skipped"}:
                break
            image = cv2.imread(str(enemy_public_path(frame.get("url") or "")), cv2.IMREAD_GRAYSCALE)
            if image is None:
                break
            height, width = image.shape[:2]
            t_h, t_w = current_template.shape[:2]
            center_x = float(previous_box["x_center"]) * width
            center_y = float(previous_box["y_center"]) * height
            prev_w = max(t_w, int(round(float(previous_box["width"]) * width)))
            prev_h = max(t_h, int(round(float(previous_box["height"]) * height)))
            search_w = max(t_w + 6, int(round(prev_w * search_scale)))
            search_h = max(t_h + 6, int(round(prev_h * search_scale)))
            x0 = max(0, int(round(center_x - search_w / 2.0)))
            y0 = max(0, int(round(center_y - search_h / 2.0)))
            x1 = min(width, x0 + search_w)
            y1 = min(height, y0 + search_h)
            if x1 - x0 < t_w or y1 - y0 < t_h:
                break
            search = image[y0:y1, x0:x1]
            result = cv2.matchTemplate(search, current_template, cv2.TM_CCOEFF_NORMED)
            _, max_score, _, max_loc = cv2.minMaxLoc(result)
            px = x0 + int(max_loc[0])
            py = y0 + int(max_loc[1])
            tracked_box = enemy_pixels_to_box(frame, px, py, px + t_w, py + t_h)
            if max_score >= accept_threshold:
                # Template matching is an annotation assistant, not a trusted
                # detector. Every propagated box must be reviewed by a person.
                write_enemy_frame_label(lib, profile, frame, tracked_box, "review")
                current_template = image[py:py + t_h, px:px + t_w].copy()
                previous_box = tracked_box
                counts["review"] += 1
                counts["processed"] += 1
                processed += 1
                track_reports.append({"frame_id": frame.get("id"), "status": "review", "score": float(max_score)})
                continue
            if max_score >= review_threshold:
                write_enemy_frame_label(lib, profile, frame, tracked_box, "review")
                counts["review"] += 1
                counts["processed"] += 1
                track_reports.append({"frame_id": frame.get("id"), "status": "review", "score": float(max_score)})
            else:
                track_reports.append({"frame_id": frame.get("id"), "status": "stopped", "score": float(max_score)})
            counts["stopped"] += 1
            break

    if direction in {"both", "forward"}:
        run_one_way(list(range(start_index + 1, len(source_frames))))
    if direction in {"both", "backward"}:
        run_one_way(list(range(start_index - 1, -1, -1)))

    profile["frames"] = frames
    update_enemy_training_state(profile)
    lib["model_status"] = "needs_training"
    save_enemy_library(lib)
    return {
        **counts,
        "reports": track_reports[-40:],
        "enemy": normalize_enemy_profile(profile),
        "library": load_enemy_library(),
    }


def rename_enemy_profile(enemy_id: str, name: str) -> dict:
    lib, profile = get_enemy_profile(enemy_id)
    cleaned = " ".join(str(name or "").split())
    if not cleaned:
        raise RuntimeError("Enemy drone name cannot be empty.")
    profile["name"] = cleaned[:80]
    profile["class_name"] = slugify_label(profile["name"], "enemy_drone")
    profile["updated_at"] = now_label()
    save_enemy_library(lib)
    return normalize_enemy_profile(profile)


def delete_enemy_profile(enemy_id: str) -> None:
    lib = load_enemy_library()
    enemies = list(lib.get("enemies") or [])
    if enemy_id not in {e.get("id") for e in enemies}:
        raise RuntimeError(f"Unknown enemy drone profile: {enemy_id}")
    lib["enemies"] = [e for e in enemies if e.get("id") != enemy_id]
    profile_dir = ENEMY_DIR / enemy_id
    if profile_dir.exists():
        shutil.rmtree(profile_dir)
    lib["model_status"] = "needs_training" if lib["enemies"] else "not_trained"
    save_enemy_library(lib)


def prepare_enemy_yolo_dataset(enemy_id: str | None = None) -> dict:
    lib = load_enemy_library()
    targets = [
        profile
        for profile in lib.get("enemies", [])
        if not enemy_id or profile.get("id") == enemy_id
    ]
    if not targets:
        raise RuntimeError("No enemy drone profiles selected for YOLO dataset preparation.")
    dataset_id = f"dataset_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    dataset_dir = ENEMY_DIR / "datasets" / dataset_id
    for split in ("train", "val", "test"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    labeled_items: list[dict] = []
    negative_items: list[dict] = []
    target_ids = {profile["id"] for profile in targets}
    class_map = {
        profile["id"]: {
            "index": index,
            "name": profile["name"],
            "class_name": profile["class_name"],
        }
        for index, profile in enumerate(targets)
    }
    source_ids = sorted(
        {
            str(frame.get("source_video_id") or "")
            for profile in targets
            for frame in profile.get("frames", [])
            if frame.get("status") in {"labeled", "negative"}
        }
    )
    if not source_ids:
        raise RuntimeError("No reviewed labels or confirmed-negative frames are available.")
    if len(source_ids) < 2:
        raise RuntimeError(
            "Use reviewed frames from at least two different calibration videos so training and validation stay independent."
        )
    ranked_sources = sorted(
        source_ids,
        key=lambda value: hashlib.sha256(f"{dataset_id}:{value}".encode("utf-8")).hexdigest(),
    )
    source_split: dict[str, str] = {}
    for index, source_id in enumerate(ranked_sources):
        if len(ranked_sources) == 1:
            source_split[source_id] = "train"
        elif index == 0:
            source_split[source_id] = "val"
        elif len(ranked_sources) >= 3 and index == 1:
            source_split[source_id] = "test"
        else:
            source_split[source_id] = "train"

    for profile in targets:
        for frame in profile.get("frames", []):
            status = str(frame.get("status") or "")
            if status not in {"labeled", "negative"}:
                continue
            image_path = enemy_public_path(frame.get("url") or "")
            box = frame.get("box") if isinstance(frame.get("box"), dict) else None
            if not image_path.exists() or (status == "labeled" and not box):
                continue
            source_id = str(frame.get("source_video_id") or "")
            split = source_split.get(source_id, "train")
            out_name = f"{profile['id']}_{frame['filename']}"
            shutil.copy2(image_path, dataset_dir / "images" / split / out_name)
            label_path = dataset_dir / "labels" / split / f"{Path(out_name).stem}.txt"
            if status == "labeled":
                class_id = int(class_map[profile["id"]]["index"])
                label_path.write_text(
                    (
                        f"{class_id} {float(box['x_center']):.8f} {float(box['y_center']):.8f} "
                        f"{float(box['width']):.8f} {float(box['height']):.8f}\n"
                    ),
                    encoding="utf-8",
                )
                labeled_items.append(
                    {"enemy_id": profile["id"], "frame_id": frame["id"], "image": out_name, "split": split}
                )
            else:
                label_path.write_text("", encoding="utf-8")
                negative_items.append(
                    {"enemy_id": profile["id"], "frame_id": frame["id"], "image": out_name, "split": split}
                )
    if not labeled_items:
        raise RuntimeError("Extract frames and save at least one bounding-box label before preparing the YOLO dataset.")

    class_names = [profile["class_name"] for profile in targets]
    split_counts = {
        split: sum(1 for item in labeled_items + negative_items if item["split"] == split)
        for split in ("train", "val", "test")
    }
    if split_counts["train"] == 0 or split_counts["val"] == 0:
        raise RuntimeError("The prepared dataset needs both training and validation images.")
    yaml_path = dataset_dir / "data.yaml"
    yaml_path.write_text(
        f"path: {dataset_dir}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(class_names)}\n"
        "names:\n"
        + "".join(f"  {idx}: {name}\n" for idx, name in enumerate(class_names)),
        encoding="utf-8",
    )
    dataset_manifest = {
        "version": 2,
        "dataset_id": dataset_id,
        "prepared_at": now_label(),
        "status": "ready_for_training",
        "data_yaml": public_rel(yaml_path),
        "labeled_frame_count": len(labeled_items),
        "negative_frame_count": len(negative_items),
        "split_counts": split_counts,
        "source_split": source_split,
        "warnings": [],
        "items": labeled_items,
        "negative_items": negative_items,
        "class_map": class_map,
        "classes": [
            {
                "id": profile["id"],
                "name": profile["name"],
                "class_name": profile["class_name"],
                "video_count": len(profile.get("videos", [])),
                "frame_count": len(profile.get("frames", [])),
                "labeled_frame_count": profile.get("labeled_frame_count", 0),
            }
            for profile in targets
        ],
    }
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(dataset_manifest, indent=2), encoding="utf-8")
    for profile in lib.get("enemies", []):
        if profile["id"] in target_ids:
            profile["training_status"] = "dataset_ready"
            profile["dataset_manifest"] = public_rel(manifest_path)
            profile["updated_at"] = now_label()
    lib["model_status"] = "dataset_ready"
    lib["training"] = {
        "status": "dataset_ready",
        "data_yaml": public_rel(yaml_path),
        "dataset_manifest": public_rel(manifest_path),
        "labeled_frame_count": len(labeled_items),
        "negative_frame_count": len(negative_items),
        "split_counts": split_counts,
        "dataset_id": dataset_id,
        "prepared_at": dataset_manifest["prepared_at"],
    }
    save_enemy_library(lib)
    return load_enemy_library()


def selected_enemy_model_path() -> Path | None:
    lib = load_enemy_library()
    rel = str(lib.get("selected_model") or "").strip().lstrip("/")
    if not rel:
        return None
    path = (VIEWER / rel).resolve()
    try:
        if not path.is_relative_to(VIEWER.resolve()):
            return None
    except AttributeError:
        if not str(path).startswith(str(VIEWER.resolve())):
            return None
    return path if path.exists() else None


def set_enemy_library_training_state(
    *,
    status: str,
    message: str,
    run_id: str | None = None,
    log_url: str | None = None,
    summary_url: str | None = None,
    selected_model: str | None = None,
    training: dict | None = None,
) -> dict:
    lib = load_enemy_library()
    lib["model_status"] = status
    lib["training_message"] = message
    if run_id is not None:
        lib["training_run_id"] = run_id
    if log_url is not None:
        lib["training_log"] = log_url
    if summary_url is not None:
        lib["training_summary"] = summary_url
    if selected_model is not None:
        lib["selected_model"] = selected_model
    if training is not None:
        lib["training"] = training
    save_enemy_library(lib)
    return load_enemy_library()


def enemy_yolo_training_job(
    *,
    run_id: str,
    base_model: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
) -> None:
    run_dir = ENEMY_DIR / "training_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    summary_path = run_dir / "training_summary.json"
    try:
        cfg = load_config()
        py = Path(cfg["python"])
        scripts = ROOT / "scripts"
        set_job("enemy", "running", "Preparing YOLO dataset from accepted enemy-drone labels.")
        lib = prepare_enemy_yolo_dataset(None)
        dataset_rel = str(lib.get("training", {}).get("data_yaml") or "")
        dataset_yaml = (VIEWER / dataset_rel).resolve() if dataset_rel else (ENEMY_DIR / "yolo_dataset" / "data.yaml")
        if not dataset_yaml.exists():
            dataset_yaml = ENEMY_DIR / "yolo_dataset" / "data.yaml"
        set_enemy_library_training_state(
            status="training",
            message=f"Fine-tuning {base_model} on {lib.get('total_labeled_frames', 0)} labeled frames.",
            run_id=run_id,
            log_url=public_rel(log_path),
            training={
                "status": "running",
                "run_id": run_id,
                "base_model": base_model,
                "epochs": epochs,
                "imgsz": imgsz,
                "batch": batch,
                "device": device,
                "data_yaml": public_rel(dataset_yaml),
                "started_at": now_label(),
                "log": public_rel(log_path),
            },
        )
        cmd = [
            str(py),
            str(scripts / "train_enemy_yolo.py"),
            "--dataset-yaml",
            str(dataset_yaml),
            "--output-dir",
            str(run_dir),
            "--project-dir",
            str(run_dir / "ultralytics_runs"),
            "--model",
            base_model,
            "--epochs",
            str(epochs),
            "--imgsz",
            str(imgsz),
            "--batch",
            str(batch),
            "--device",
            device,
            "--run-name",
            "enemy_drone_detector",
        ]
        append_log("enemy", "+ " + " ".join(cmd))
        with log_path.open("w", encoding="utf-8") as log_handle:
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            register_active_proc("enemy", proc)
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    log_handle.write(line)
                    log_handle.flush()
                    append_log("enemy", line.rstrip())
                rc = proc.wait()
            finally:
                unregister_active_proc("enemy", proc)
        if rc != 0:
            raise RuntimeError(f"YOLO fine-tuning failed with exit code {rc}. See {public_rel(log_path)}")
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        model_path = run_dir / "best.pt"
        if not model_path.exists():
            raise RuntimeError("Training finished but best.pt was not produced.")
        model_rel = public_rel(model_path)
        set_enemy_library_training_state(
            status="trained",
            message="Enemy-drone detector trained and enabled for live patrol.",
            run_id=run_id,
            log_url=public_rel(log_path),
            summary_url=public_rel(summary_path),
            selected_model=model_rel,
            training={
                **summary,
                "status": "trained",
                "run_id": run_id,
                "base_model": base_model,
                "epochs": epochs,
                "imgsz": imgsz,
                "batch": batch,
                "device": device,
                "log": public_rel(log_path),
                "best_model": model_rel,
                "finished_at": now_label(),
            },
        )
        set_job("enemy", "done", "Enemy-drone YOLO model trained and enabled for live patrol.")
    except Exception as exc:
        append_log("enemy", f"ERROR: {exc}")
        set_enemy_library_training_state(
            status="training_failed",
            message=str(exc),
            run_id=run_id,
            log_url=public_rel(log_path),
            summary_url=public_rel(summary_path) if summary_path.exists() else None,
            training={
                "status": "failed",
                "run_id": run_id,
                "base_model": base_model,
                "epochs": epochs,
                "imgsz": imgsz,
                "batch": batch,
                "device": device,
                "log": public_rel(log_path),
                "error": str(exc),
                "finished_at": now_label(),
            },
        )
        set_job("enemy", "failed", f"Enemy detector training failed: {exc}")


def get_map_entry(map_id: str | None = None) -> dict:
    lib = load_library()
    target = map_id or lib.get("selected_map_id") or "default_demo"
    for entry in lib.get("maps", []):
        if entry["id"] == target:
            return entry
    raise RuntimeError(f"Unknown map id: {target}")


def selected_map_entry() -> dict:
    with STATE_LOCK:
        map_id = STATE.get("selected_map_id")
    return get_map_entry(str(map_id) if map_id else None)


def set_selected_map(map_id: str) -> dict:
    lib = load_library()
    entry = None
    for candidate in lib.get("maps", []):
        if candidate["id"] == map_id:
            entry = candidate
            break
    if entry is None:
        raise RuntimeError(f"Unknown map id: {map_id}")
    lib["selected_map_id"] = map_id
    save_library(lib)
    frames = Path(entry["frames_path"]) if entry.get("frames_path") else None
    with STATE_LOCK:
        STATE["selected_map_id"] = map_id
        STATE["current_map_frames"] = str(frames) if frames else None
    return entry


def add_or_update_map(entry: dict, select: bool = True) -> dict:
    lib = load_library()
    entry = normalize_replays(entry)
    maps = [m for m in lib.get("maps", []) if m["id"] != entry["id"]]
    maps.append(entry)
    maps.sort(key=lambda m: (0 if m["id"] == "default_demo" else 1, m.get("created_at") or ""))
    lib["maps"] = maps
    if select:
        lib["selected_map_id"] = entry["id"]
    save_library(lib)
    if select:
        set_selected_map(entry["id"])
    return entry


def add_replay_to_map(map_id: str, replay: dict, select: bool = True) -> dict:
    lib = load_library()
    entry = None
    for candidate in lib.get("maps", []):
        if candidate["id"] == map_id:
            entry = candidate
            break
    if entry is None:
        raise RuntimeError(f"Unknown map id: {map_id}")
    replays = [r for r in entry.get("replays", []) if r.get("id") != replay.get("id")]
    replays.append(replay)
    entry["replays"] = replays
    entry["active_replay_id"] = replay["id"]
    entry["has_drone_demo"] = True
    entry["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    add_or_update_map(entry, select=select)
    return entry


def set_active_replay(map_id: str, replay_id: str) -> dict:
    lib = load_library()
    entry = None
    for candidate in lib.get("maps", []):
        if candidate["id"] == map_id:
            entry = candidate
            break
    if entry is None:
        raise RuntimeError(f"Unknown map id: {map_id}")
    replays = entry.get("replays", [])
    if replay_id not in {r.get("id") for r in replays}:
        raise RuntimeError(f"Unknown replay id for {map_id}: {replay_id}")
    entry["active_replay_id"] = replay_id
    add_or_update_map(entry, select=True)
    return entry


def delete_replay_from_map(map_id: str, replay_id: str) -> dict:
    lib = load_library()
    entry = None
    for candidate in lib.get("maps", []):
        if candidate["id"] == map_id:
            entry = candidate
            break
    if entry is None:
        raise RuntimeError(f"Unknown map id: {map_id}")

    replays = list(entry.get("replays", []))
    replay = next((r for r in replays if r.get("id") == replay_id), None)
    if replay is None:
        raise RuntimeError(f"Unknown replay id for {map_id}: {replay_id}")

    asset_base = replay.get("asset_base")
    if asset_base:
        asset_dir = VIEWER / asset_base
        try:
            is_child_replay = asset_dir.is_relative_to(VIEWER / entry["asset_base"] / "replays")
        except ValueError:
            is_child_replay = False
        if is_child_replay and asset_dir.exists():
            shutil.rmtree(asset_dir)

    replays = [r for r in replays if r.get("id") != replay_id]
    entry["replays"] = replays
    entry["has_drone_demo"] = bool(replays)
    if replays:
        if entry.get("active_replay_id") == replay_id:
            entry["active_replay_id"] = replays[-1].get("id")
    else:
        entry.pop("active_replay_id", None)
    entry["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    add_or_update_map(entry, select=True)
    return entry


def rename_map_entry(map_id: str, title: str) -> dict:
    title = " ".join(str(title or "").split())
    if not title:
        raise RuntimeError("Map name cannot be empty.")
    lib = load_library()
    entry = None
    for candidate in lib.get("maps", []):
        if candidate["id"] == map_id:
            entry = candidate
            break
    if entry is None:
        raise RuntimeError(f"Unknown map id: {map_id}")
    entry["title"] = title[:80]
    entry["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_library(lib)
    return entry


def unique_map_title(base_title: str, maps: list[dict]) -> str:
    used = {str(m.get("title") or "") for m in maps}
    candidate = f"{base_title} Copy"
    if candidate not in used:
        return candidate
    index = 2
    while f"{candidate} {index}" in used:
        index += 1
    return f"{candidate} {index}"


def map_asset_dir(entry: dict) -> Path:
    asset_base = entry.get("asset_base")
    if asset_base:
        asset_dir = VIEWER / asset_base
        if asset_dir.exists():
            return asset_dir
    fallback = MAPS_DIR / str(entry.get("id") or "")
    if fallback.exists():
        return fallback
    raise RuntimeError(f"Map assets not found for {entry.get('title') or entry.get('id')}.")


def active_taught_recovery_banks(base_asset_dir: Path) -> list[Path]:
    """Return active patrol recovery banks, excluding timestamped backups."""
    taught_root = Path(base_asset_dir) / "taught_patrols"
    banks = list(taught_root.glob("*/recovery_bank.npz"))
    banks.extend(taught_root.glob("*/recovery_bank_full_loop_*.npz"))
    return sorted({path.resolve() for path in banks if path.is_file()})


def patrol_geometry_points(patrol: dict) -> list[list[float]]:
    """Return a patrol's ordered point coordinates, or an empty list if invalid."""
    if not isinstance(patrol, dict):
        return []
    points: list[list[float]] = []
    for raw_point in patrol.get("points", []):
        raw_xyz = raw_point.get("rxyz") if isinstance(raw_point, dict) else raw_point
        xyz = _vec3(raw_xyz)
        if xyz is None:
            return []
        points.append(xyz)
    return points


def taught_patrol_geometry_lock_path(entry: dict, patrol_id: str) -> Path:
    """Resolve a patrol lock inside its map asset directory without path traversal."""
    patrol_id = str(patrol_id or "").strip()
    root = (map_asset_dir(entry) / "taught_patrols").resolve()
    lock_path = (root / patrol_id / "geometry_lock.json").resolve()
    if not patrol_id or not lock_path.is_relative_to(root):
        raise RuntimeError("Invalid patrol id for taught geometry lock.")
    return lock_path


def load_taught_patrol_geometry_lock(entry: dict, patrol_id: str) -> dict | None:
    lock_path = taught_patrol_geometry_lock_path(entry, patrol_id)
    if not lock_path.exists():
        return None
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Taught patrol geometry lock is unreadable: {lock_path.name}."
        ) from exc
    if not isinstance(lock, dict) or len(patrol_geometry_points(lock)) < 2:
        raise RuntimeError(
            "Taught patrol geometry lock is invalid; refusing to change the patrol."
        )
    return lock


def patrol_geometry_matches(first: dict, second: dict, tolerance: float = 1e-6) -> bool:
    first_points = patrol_geometry_points(first)
    second_points = patrol_geometry_points(second)
    if len(first_points) != len(second_points) or len(first_points) < 2:
        return False
    return all(
        math.isclose(a, b, rel_tol=0.0, abs_tol=tolerance)
        for first_point, second_point in zip(first_points, second_points)
        for a, b in zip(first_point, second_point)
    )


def copy_map_without_replays(src_asset_dir: Path, dst_asset_dir: Path) -> None:
    if dst_asset_dir.exists():
        shutil.rmtree(dst_asset_dir)
    dst_asset_dir.mkdir(parents=True, exist_ok=True)
    for item in src_asset_dir.iterdir():
        if item.name == "replays":
            continue
        target = dst_asset_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    (dst_asset_dir / "poses.json").write_text(json.dumps({"poses": []}, indent=2), encoding="utf-8")


def duplicate_map_entry(map_id: str) -> dict:
    lib = load_library()
    source = next((m for m in lib.get("maps", []) if m["id"] == map_id), None)
    if source is None:
        raise RuntimeError(f"Unknown map id: {map_id}")

    new_id = make_map_id("map_copy")
    source_asset_dir = map_asset_dir(source)
    target_asset_dir = MAPS_DIR / new_id
    copy_map_without_replays(source_asset_dir, target_asset_dir)

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    title = unique_map_title(str(source.get("title") or "Untitled Map"), lib.get("maps", []))
    entry = dict(source)
    entry.update(
        {
            "id": new_id,
            "title": title,
            "description": f"3D-only duplicate of {source.get('title') or 'map'}; drone paths removed.",
            "asset_base": public_rel(target_asset_dir),
            "deletable": True,
            "kind": "map_copy",
            "source_map_id": source.get("id"),
            "localization_map_id": source.get("localization_map_id") or source.get("source_map_id") or source.get("id"),
            "created_at": now,
            "updated_at": now,
            "counts": read_counts(target_asset_dir),
            "has_drone_demo": False,
            "replays": [],
            "active_replay_id": None,
            "patrols": [],
        }
    )
    add_or_update_map(entry, select=True)
    append_log("map", f'Duplicated 3D map "{source.get("title")}" as "{title}" without drone paths.')
    set_job("map", "done", f'3D map duplicated without paths: {title}.')
    return entry


def set_map_display_z_sign(map_id: str, display_z_sign: int | float | str) -> dict:
    lib = load_library()
    entry = None
    for candidate in lib.get("maps", []):
        if candidate["id"] == map_id:
            entry = candidate
            break
    if entry is None:
        raise RuntimeError(f"Unknown map id: {map_id}")
    try:
        sign = 1 if float(display_z_sign) >= 0 else -1
    except (TypeError, ValueError):
        raise RuntimeError("display_z_sign must be positive or negative.")
    entry["display_z_sign"] = sign
    entry["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_library(lib)
    return entry


def rename_replay_in_map(map_id: str, replay_id: str, title: str) -> dict:
    title = " ".join(str(title or "").split())
    if not title:
        raise RuntimeError("Drone path name cannot be empty.")
    lib = load_library()
    entry = None
    for candidate in lib.get("maps", []):
        if candidate["id"] == map_id:
            entry = candidate
            break
    if entry is None:
        raise RuntimeError(f"Unknown map id: {map_id}")
    replay = next((r for r in entry.get("replays", []) if r.get("id") == replay_id), None)
    if replay is None:
        raise RuntimeError(f"Unknown replay id for {map_id}: {replay_id}")
    replay["title"] = title[:80]
    entry["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_library(lib)
    return entry


def _vec3(value) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    out: list[float] = []
    for raw in value[:3]:
        try:
            number = float(raw)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        out.append(number)
    return out


def _color(value, fallback: str) -> str:
    text = str(value or "").strip()
    if len(text) == 7 and text[0] == "#":
        try:
            int(text[1:], 16)
            return text.lower()
        except ValueError:
            pass
    return fallback


def _opacity(value, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = fallback
    if not math.isfinite(number):
        number = fallback
    return max(0.05, min(0.95, number))


def sanitize_safety_barriers(raw_barriers) -> list[dict]:
    barriers: list[dict] = []
    if not isinstance(raw_barriers, list):
        return barriers
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    for index, raw in enumerate(raw_barriers[:200]):
        if not isinstance(raw, dict):
            continue
        corners = []
        if isinstance(raw.get("corners"), list):
            corners = [p for p in (_vec3(v) for v in raw.get("corners", [])[:4]) if p]
        a = _vec3(raw.get("a") or raw.get("start") or (corners[0] if corners else None))
        b = _vec3(raw.get("b") or raw.get("end") or (corners[1] if len(corners) > 1 else None))
        if not a or not b:
            continue
        if math.hypot(b[0] - a[0], b[2] - a[2]) < 1e-4:
            continue
        try:
            clearance = float(raw.get("clearance_m", 0.45))
        except (TypeError, ValueError):
            clearance = 0.45
        try:
            height = float(raw.get("height_m", 1.8))
        except (TypeError, ValueError):
            height = 1.8
        clearance = max(0.05, min(5.0, clearance if math.isfinite(clearance) else 0.45))
        height = max(0.25, min(8.0, height if math.isfinite(height) else 1.8))
        if len(corners) < 4:
            floor_y = min(a[1], b[1])
            corners = [
                [a[0], floor_y, a[2]],
                [b[0], floor_y, b[2]],
                [b[0], floor_y + height, b[2]],
                [a[0], floor_y + height, a[2]],
            ]
        ys = [p[1] for p in corners[:4]]
        height = max(0.25, min(8.0, max(ys) - min(ys) or height))
        label = " ".join(str(raw.get("label") or f"Wall {len(barriers) + 1}").split())[:64]
        barrier_id = " ".join(str(raw.get("id") or f"barrier_{index}").split())[:80]
        barriers.append(
            {
                "id": barrier_id or f"barrier_{index}",
                "label": label or f"Wall {len(barriers) + 1}",
                "a": a,
                "b": b,
                "corners": corners[:4],
                "height_m": height,
                "clearance_m": clearance,
                "color": _color(raw.get("color"), "#cfd8df"),
                "opacity": _opacity(raw.get("opacity"), 0.24),
                "created_at": str(raw.get("created_at") or now),
                "updated_at": now,
            }
        )
    return barriers


def sanitize_safety_obstacles(raw_obstacles) -> list[dict]:
    obstacles: list[dict] = []
    if not isinstance(raw_obstacles, list):
        return obstacles
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    for index, raw in enumerate(raw_obstacles[:200]):
        if not isinstance(raw, dict):
            continue
        points = []
        if isinstance(raw.get("points"), list):
            points = [p for p in (_vec3(v) for v in raw.get("points", [])[:80]) if p]
        if len(points) < 2:
            continue
        try:
            clearance = float(raw.get("clearance_m", 0.35))
        except (TypeError, ValueError):
            clearance = 0.35
        clearance = max(0.05, min(3.0, clearance if math.isfinite(clearance) else 0.35))
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]
        bounds = {
            "min": [min(xs), min(ys), min(zs)],
            "max": [max(xs), max(ys), max(zs)],
        }
        raw_bounds = raw.get("bounds")
        if isinstance(raw_bounds, dict):
            raw_min = _vec3(raw_bounds.get("min"))
            raw_max = _vec3(raw_bounds.get("max"))
            if raw_min and raw_max:
                bounds = {
                    "min": [min(raw_min[i], raw_max[i]) for i in range(3)],
                    "max": [max(raw_min[i], raw_max[i]) for i in range(3)],
                }
        # Keep very thin picked structures visible and useful for route checks.
        for axis in range(3):
            if bounds["max"][axis] - bounds["min"][axis] < 0.05:
                pad = 0.025
                bounds["min"][axis] -= pad
                bounds["max"][axis] += pad
        label = " ".join(str(raw.get("label") or f"Obstacle {len(obstacles) + 1}").split())[:64]
        obstacle_id = " ".join(str(raw.get("id") or f"obstacle_{index}").split())[:80]
        obstacles.append(
            {
                "id": obstacle_id or f"obstacle_{index}",
                "label": label or f"Obstacle {len(obstacles) + 1}",
                "points": points,
                "bounds": bounds,
                "clearance_m": clearance,
                "color": _color(raw.get("color"), "#86dfff"),
                "opacity": _opacity(raw.get("opacity"), 0.24),
                "created_at": str(raw.get("created_at") or now),
                "updated_at": now,
            }
        )
    return obstacles


def set_map_safety_barriers(map_id: str, barriers, obstacles=None) -> dict:
    lib = load_library()
    entry = None
    for candidate in lib.get("maps", []):
        if candidate["id"] == map_id:
            entry = candidate
            break
    if entry is None:
        raise RuntimeError(f"Unknown map id: {map_id}")
    entry["safety_barriers"] = sanitize_safety_barriers(barriers)
    if obstacles is not None:
        entry["safety_obstacles"] = sanitize_safety_obstacles(obstacles)
    entry["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_library(lib)
    return entry


def sanitize_patrols(raw_patrols) -> list[dict]:
    patrols: list[dict] = []
    if not isinstance(raw_patrols, list):
        return patrols
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    for index, raw in enumerate(raw_patrols[:100]):
        if not isinstance(raw, dict):
            continue
        points: list[dict] = []
        for raw_point in raw.get("points", [])[:200]:
            if isinstance(raw_point, dict):
                rxyz = _vec3(raw_point.get("rxyz"))
                rgb_raw = raw_point.get("rgb")
            else:
                rxyz = _vec3(raw_point)
                rgb_raw = None
            if not rxyz:
                continue
            rgb = None
            if isinstance(rgb_raw, (list, tuple)) and len(rgb_raw) >= 3:
                try:
                    rgb = [max(0, min(255, int(float(v)))) for v in rgb_raw[:3]]
                except (TypeError, ValueError):
                    rgb = None
            points.append({"rxyz": rxyz, "rgb": rgb})
        if len(points) < 2:
            continue
        patrol_id = " ".join(str(raw.get("id") or f"patrol_{index}").split())[:96] or f"patrol_{index}"
        title = " ".join(str(raw.get("title") or f"Patrol {len(patrols) + 1}").split())[:80]
        try:
            speed = float(raw.get("speed", 0.10))
        except (TypeError, ValueError):
            speed = 0.10
        try:
            altitude_m = float(raw.get("altitude_m", 1.0))
        except (TypeError, ValueError):
            altitude_m = 1.0
        try:
            dwell_s = float(raw.get("dwell_s", 2.0))
        except (TypeError, ValueError):
            dwell_s = 2.0
        scan_mode = str(raw.get("scan_mode") or "yaw-sweep")
        if scan_mode not in {"yaw-sweep", "forward"}:
            scan_mode = "yaw-sweep"
        patrol_mode = str(raw.get("patrol_mode") or raw.get("mode") or "").strip().lower()
        if patrol_mode in {"back_and_forth", "back-forth", "pingpong", "ping-pong", "bounce"}:
            patrol_mode = "back-and-forth"
        elif patrol_mode not in {"circle", "back-and-forth"}:
            patrol_mode = "circle" if bool(raw.get("loop", True)) else "back-and-forth"
        patrols.append(
            {
                "id": patrol_id,
                "title": title or f"Patrol {len(patrols) + 1}",
                "points": points,
                "speed": max(0.04, min(0.20, speed if math.isfinite(speed) else 0.10)),
                "altitude_m": max(0.3, min(2.0, altitude_m if math.isfinite(altitude_m) else 1.0)),
                "dwell_s": max(0.8, min(8.0, dwell_s if math.isfinite(dwell_s) else 2.0)),
                "scan_mode": scan_mode,
                "patrol_mode": patrol_mode,
                "loop": patrol_mode == "circle",
                "created_at": str(raw.get("created_at") or now),
                "updated_at": str(raw.get("updated_at") or now),
            }
        )
    return patrols


def set_map_patrols(map_id: str, patrols) -> dict:
    lib = load_library()
    live_patrol_lock = resolved_live_patrol_lock(lib)
    entry = None
    for candidate in lib.get("maps", []):
        if candidate["id"] == map_id:
            entry = candidate
            break
    if entry is None:
        raise RuntimeError(f"Unknown map id: {map_id}")
    sanitized_patrols = sanitize_patrols(patrols)
    if (
        live_patrol_lock is not None
        and map_id == live_patrol_lock["map_id"]
        and not any(
            str(patrol.get("id") or "") == live_patrol_lock["patrol_id"]
            for patrol in sanitized_patrols
        )
    ):
        raise RuntimeError(
            f"{live_patrol_lock['patrol_title']} is pinned for live patrol and cannot be deleted "
            "until a replacement live patrol is explicitly selected."
        )
    for patrol in sanitized_patrols:
        patrol_id = str(patrol.get("id") or "").strip()
        geometry_lock = load_taught_patrol_geometry_lock(entry, patrol_id)
        if geometry_lock is not None and not patrol_geometry_matches(patrol, geometry_lock):
            raise RuntimeError(
                "Patrol geometry is locked to its taught recovery data. "
                "Create a new patrol or explicitly reset teaching before moving its points."
            )
    entry["patrols"] = sanitized_patrols
    entry["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_library(lib)
    return entry


def map_coordinate_lineage(entry: dict, maps_by_id: dict[str, dict]) -> set[str]:
    """Return map ids known to share this map's fixed coordinate frame."""
    lineage: set[str] = set()
    pending = [str(entry.get("id") or "").strip()]
    while pending and len(lineage) < 32:
        map_id = pending.pop()
        if not map_id or map_id in lineage:
            continue
        lineage.add(map_id)
        current = maps_by_id.get(map_id)
        if current is None:
            continue
        for key in ("source_map_id", "localization_map_id"):
            parent_id = str(current.get(key) or "").strip()
            if parent_id and parent_id != map_id and parent_id not in lineage:
                pending.append(parent_id)
    return lineage


def import_map_patrol(target_map_id: str, source_map_id: str, patrol_id: str) -> tuple[dict, dict]:
    """Copy one patrol between maps that use the same coordinate frame."""
    lib = load_library()
    maps_by_id = {
        str(entry.get("id") or ""): entry
        for entry in lib.get("maps", [])
        if isinstance(entry, dict) and entry.get("id")
    }
    target = maps_by_id.get(str(target_map_id or "").strip())
    source = maps_by_id.get(str(source_map_id or "").strip())
    if target is None:
        raise RuntimeError(f"Unknown target map id: {target_map_id}")
    if source is None:
        raise RuntimeError(f"Unknown source map id: {source_map_id}")
    if target is source:
        raise RuntimeError("Choose a patrol from another map.")

    target_lineage = map_coordinate_lineage(target, maps_by_id)
    source_lineage = map_coordinate_lineage(source, maps_by_id)
    if not target_lineage.intersection(source_lineage):
        raise RuntimeError(
            "This patrol belongs to a different map coordinate frame and cannot be imported safely."
        )

    source_patrol = next(
        (
            patrol
            for patrol in source.get("patrols", [])
            if isinstance(patrol, dict) and str(patrol.get("id") or "") == str(patrol_id or "")
        ),
        None,
    )
    if source_patrol is None:
        raise RuntimeError(f"Unknown patrol id for {source_map_id}: {patrol_id}")

    target_patrols = list(target.get("patrols") or [])
    if any(
        isinstance(patrol, dict)
        and str(patrol.get("id") or "") == str(source_patrol.get("id") or "")
        for patrol in target_patrols
    ):
        raise RuntimeError(
            f'"{source_patrol.get("title") or "Patrol"}" is already present on this map.'
        )
    if len(target_patrols) >= 100:
        raise RuntimeError("This map already has the maximum of 100 saved patrols.")

    sanitized = sanitize_patrols([source_patrol])
    if not sanitized:
        raise RuntimeError("The selected patrol has fewer than two valid points.")
    imported = sanitized[0]
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    imported["created_at"] = now
    imported["updated_at"] = now
    target["patrols"] = sanitize_patrols([*target_patrols, imported])
    target["updated_at"] = now
    save_library(lib)
    return target, imported


def delete_map_entry(map_id: str) -> None:
    lib = load_library()
    live_patrol_lock = resolved_live_patrol_lock(lib)
    if live_patrol_lock is not None and map_id == live_patrol_lock["map_id"]:
        raise RuntimeError(
            f"{live_patrol_lock['map_title']} is pinned for live patrol and cannot be deleted "
            "until a replacement live patrol map is explicitly selected."
        )
    entry = next((m for m in lib.get("maps", []) if m["id"] == map_id), None)
    if entry is None:
        raise RuntimeError(f"Unknown map id: {map_id}")
    asset_base = entry.get("asset_base")
    should_remove_assets = map_id != "default_demo" and bool(entry.get("deletable", True))
    if asset_base and should_remove_assets:
        asset_dir = VIEWER / asset_base
        if asset_dir.is_relative_to(MAPS_DIR) and asset_dir.exists():
            shutil.rmtree(asset_dir)
    if map_id == "default_demo":
        hidden = set(lib.get("hidden_builtin_ids") or [])
        hidden.add("default_demo")
        lib["hidden_builtin_ids"] = sorted(hidden)
    lib["maps"] = [m for m in lib.get("maps", []) if m["id"] != map_id]
    if lib.get("selected_map_id") == map_id:
        lib["selected_map_id"] = lib["maps"][0]["id"] if lib["maps"] else ""
    save_library(lib)
    if lib["selected_map_id"]:
        set_selected_map(lib["selected_map_id"])
    else:
        with STATE_LOCK:
            STATE["selected_map_id"] = ""
            STATE["current_map_frames"] = None


def set_job(kind: str, status: str, message: str | None = None) -> None:
    with STATE_LOCK:
        STATE[kind]["status"] = status
        if message is not None:
            STATE[kind]["message"] = message
        STATE[kind]["updated_at"] = time.time()


def queue_job(kind: str, message: str) -> None:
    with STATE_LOCK:
        STATE[kind]["status"] = "queued"
        STATE[kind]["message"] = message
        STATE[kind]["log"] = []
        if kind == "drone":
            STATE[kind]["live_stream"] = None
        STATE[kind]["updated_at"] = time.time()


def job_is_active(kind: str) -> bool:
    with STATE_LOCK:
        return STATE[kind].get("status") in ACTIVE_JOB_STATES


def reserve_and_queue_drone_job(message: str) -> bool:
    """Atomically reserve the single drone worker before its thread starts."""
    with DRONE_JOB_LIFECYCLE_LOCK:
        if DRONE_JOB_ACTIVE.is_set() or job_is_active("drone"):
            return False
        DRONE_STOP_EVENT.clear()
        DRONE_JOB_ACTIVE.set()
        queue_job("drone", message)
        return True


def mark_drone_job_worker_started() -> None:
    """Record the thread that owns the reserved drone job.

    The active Event is set by the request handler before the worker starts so
    two requests cannot race.  Recording the eventual worker separately lets
    Stop distinguish that short queued interval from a stale Event left behind
    by a worker thread that has already disappeared.
    """
    global DRONE_JOB_WORKER_IDENT
    with DRONE_JOB_LIFECYCLE_LOCK:
        DRONE_JOB_ACTIVE.set()
        DRONE_JOB_WORKER_IDENT = threading.get_ident()


def drone_job_worker_liveness() -> bool | None:
    """Return True/False for a known worker, or None before it has started."""
    with DRONE_JOB_LIFECYCLE_LOCK:
        worker_ident = DRONE_JOB_WORKER_IDENT
    if worker_ident is None:
        return None
    return any(
        thread.ident == worker_ident and thread.is_alive()
        for thread in threading.enumerate()
    )


def release_drone_job() -> None:
    """Acknowledge cancellation only after the worker has fully cleaned up."""
    global DRONE_JOB_WORKER_IDENT
    with DRONE_JOB_LIFECYCLE_LOCK:
        DRONE_JOB_WORKER_IDENT = None
        DRONE_JOB_ACTIVE.clear()
        DRONE_STOP_EVENT.clear()


def set_camera_path_lab_job(status: str, message: str | None = None) -> None:
    with CAMERA_PATH_LAB_LOCK:
        CAMERA_PATH_LAB_STATE["status"] = status
        if message is not None:
            CAMERA_PATH_LAB_STATE["message"] = message
        CAMERA_PATH_LAB_STATE["updated_at"] = time.time()


def set_camera_path_lab_stream(stream: dict | None) -> None:
    with CAMERA_PATH_LAB_LOCK:
        CAMERA_PATH_LAB_STATE["stream"] = json.loads(json.dumps(stream)) if stream else None
        CAMERA_PATH_LAB_STATE["updated_at"] = time.time()


def update_camera_path_lab_stream(**fields) -> None:
    with CAMERA_PATH_LAB_LOCK:
        stream = CAMERA_PATH_LAB_STATE.get("stream")
        if not isinstance(stream, dict):
            stream = {}
        stream.update(fields)
        CAMERA_PATH_LAB_STATE["stream"] = stream
        CAMERA_PATH_LAB_STATE["updated_at"] = time.time()


def load_camera_path_preview_calibration(replay_id: str) -> dict | None:
    replay_id = str(replay_id or "").strip()
    if not replay_id or not CAMERA_PATH_PREVIEW_CALIBRATIONS.is_file():
        return None
    try:
        payload = json.loads(CAMERA_PATH_PREVIEW_CALIBRATIONS.read_text(encoding="utf-8"))
        calibration = (payload.get("calibrations") or {}).get(replay_id)
        return json.loads(json.dumps(calibration)) if isinstance(calibration, dict) else None
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def attach_camera_path_preview_calibration(snapshot: dict) -> dict:
    stream = snapshot.get("stream")
    if not isinstance(stream, dict) or not stream.get("validation_preview"):
        return snapshot
    calibration = load_camera_path_preview_calibration(str(stream.get("replay_id") or ""))
    if calibration:
        stream["preview_calibration"] = calibration
    else:
        stream.pop("preview_calibration", None)
    return snapshot


def camera_path_lab_snapshot() -> dict:
    with CAMERA_PATH_LAB_LOCK:
        snapshot = json.loads(json.dumps(CAMERA_PATH_LAB_STATE))
    # A validated offline replay is produced by a separate worker so it can
    # run without reserving or mutating the live ATLAS map.  Surface it in the
    # same viewer after it is newer than the last live/cancelled UI state.
    latest_offline = CAMERA_PATH_LAB_DIR / "offline_latest.json"
    if snapshot.get("status") not in ACTIVE_JOB_STATES and latest_offline.is_file():
        try:
            offline = json.loads(latest_offline.read_text(encoding="utf-8"))
            generated_at = float(offline.get("generated_at") or 0.0)
            state_updated_at = float(snapshot.get("updated_at") or 0.0)
            if generated_at >= state_updated_at and isinstance(offline.get("stream"), dict):
                return attach_camera_path_preview_calibration({
                    "status": str(offline.get("status") or "done"),
                    "message": str(offline.get("message") or "Validated offline camera path ready."),
                    "updated_at": generated_at,
                    "stream": offline["stream"],
                })
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    return attach_camera_path_preview_calibration(snapshot)


def set_map_capture_state(**fields) -> None:
    with STATE_LOCK:
        STATE["map"].update(fields)
        STATE["map"]["updated_at"] = time.time()


def append_log(kind: str, line: str) -> None:
    line = line.rstrip()
    if not line:
        return
    if kind.startswith("fleet:"):
        fleet_event(kind.split(":", 1)[1], line)
        return
    with STATE_LOCK:
        log = STATE[kind]["log"]
        log.append(line)
        del log[:-240]
        STATE[kind]["updated_at"] = time.time()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def save_camera_path_preview_calibration(payload: dict) -> dict:
    replay_id = str(payload.get("replay_id") or "").strip()
    if not replay_id or len(replay_id) > 160:
        raise ValueError("A valid preview replay id is required.")
    snapshot = camera_path_lab_snapshot()
    stream = snapshot.get("stream")
    if (
        not isinstance(stream, dict)
        or not stream.get("validation_preview")
        or str(stream.get("replay_id") or "") != replay_id
    ):
        raise ValueError("Only the active validation preview can be calibrated.")
    target = payload.get("target_start")
    if not isinstance(target, list) or len(target) < 3:
        raise ValueError("target_start must contain X, Y, and Z.")
    target_start = [float(value) for value in target[:3]]
    if not all(math.isfinite(value) and abs(value) <= 100.0 for value in target_start):
        raise ValueError("The requested path position is outside the calibration workspace.")
    movement_scale = float(payload.get("movement_scale"))
    if not math.isfinite(movement_scale) or not 0.05 <= movement_scale <= 2.0:
        raise ValueError("Path scale must be between 5% and 200%.")
    movement_scale_x = float(payload.get("movement_scale_x", movement_scale))
    movement_scale_y = float(payload.get("movement_scale_y", movement_scale))
    if not all(
        math.isfinite(value) and 0.05 <= value <= 2.0
        for value in (movement_scale_x, movement_scale_y)
    ):
        raise ValueError("Path X and Y sizes must be between 5% and 200%.")
    rotation_deg = float(payload.get("rotation_deg") or 0.0)
    if not math.isfinite(rotation_deg) or not -180.0 <= rotation_deg <= 180.0:
        raise ValueError("Path rotation must be between -180 and +180 degrees.")
    timing_offset_sec = float(payload.get("timing_offset_sec") or 0.0)
    if not math.isfinite(timing_offset_sec) or not -8.0 <= timing_offset_sec <= 8.0:
        raise ValueError("Path timing offset must be between -8 and +8 seconds.")
    default_timing_boundaries = ((0, 333), (334, 599), (600, 749), (750, None))
    requested_segments = payload.get("timing_segments")
    if requested_segments is None:
        timing_segments = [
            {"start_frame": start, "end_frame": end, "offset_sec": timing_offset_sec}
            for start, end in default_timing_boundaries
        ]
    else:
        if not isinstance(requested_segments, list) or not 1 <= len(requested_segments) <= 12:
            raise ValueError("Path timing must contain between 1 and 12 frame sections.")
        timing_segments = []
        expected_start = 0
        for index, requested in enumerate(requested_segments):
            if not isinstance(requested, dict):
                raise ValueError("Each path timing section must be an object.")
            start = int(requested.get("start_frame"))
            requested_end = requested.get("end_frame")
            end = None if requested_end is None else int(requested_end)
            offset_sec = float(requested.get("offset_sec"))
            if not math.isfinite(offset_sec) or not -8.0 <= offset_sec <= 8.0:
                raise ValueError("Each path timing offset must be between -8 and +8 seconds.")
            if start != expected_start:
                raise ValueError("Path timing sections must be ordered, contiguous, and start at frame 0.")
            is_last = index == len(requested_segments) - 1
            if is_last:
                if end is not None:
                    raise ValueError("The final path timing section must continue to the end of the video.")
            else:
                if end is None or end < start + 40:
                    raise ValueError("Each bounded path timing section must contain at least 41 frames for smooth blending.")
                expected_start = end + 1
            timing_segments.append(
                {"start_frame": start, "end_frame": end, "offset_sec": offset_sec}
            )
        timing_offset_sec = timing_segments[0]["offset_sec"]
    calibration = {
        "replay_id": replay_id,
        "target_start": target_start,
        "movement_scale": movement_scale,
        "movement_scale_x": movement_scale_x,
        "movement_scale_y": movement_scale_y,
        "rotation_deg": rotation_deg,
        "timing_offset_sec": timing_offset_sec,
        "timing_segments": timing_segments,
        "locked": True,
        "updated_at": time.time(),
    }
    with CAMERA_PATH_LAB_LOCK:
        try:
            saved = json.loads(CAMERA_PATH_PREVIEW_CALIBRATIONS.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            saved = {}
        calibrations = saved.get("calibrations")
        if not isinstance(calibrations, dict):
            calibrations = {}
        calibrations[replay_id] = calibration
        atomic_write_json(
            CAMERA_PATH_PREVIEW_CALIBRATIONS,
            {"version": 1, "updated_at": calibration["updated_at"], "calibrations": calibrations},
        )
    return json.loads(json.dumps(calibration))


def default_fleet_manifest() -> dict:
    return {"version": 1, "updated_at": now_label(), "drones": []}


def load_fleet_manifest() -> dict:
    try:
        payload = json.loads(FLEET_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = default_fleet_manifest()
    drones: list[dict] = []
    seen: set[str] = set()
    colors = ["#56ddff", "#a78bfa", "#5ee6a8", "#ffb45c", "#ff6f91", "#66a3ff"]
    for index, raw in enumerate(payload.get("drones") or []):
        if not isinstance(raw, dict):
            continue
        drone_id = slugify_label(raw.get("id") or raw.get("name"), f"drone_{index + 1}")[:64]
        if drone_id in seen:
            continue
        name = " ".join(str(raw.get("name") or f"Drone {index + 1}").split())[:64]
        phone_ip = str(raw.get("phone_ip") or "").strip()[:128]
        color = str(raw.get("color") or colors[index % len(colors)])
        if len(color) != 7 or not color.startswith("#"):
            color = colors[index % len(colors)]
        drones.append(
            {
                "id": drone_id,
                "name": name or f"Drone {index + 1}",
                "phone_ip": phone_ip,
                "color": color,
                "created_at": raw.get("created_at") or now_label(),
                "updated_at": raw.get("updated_at") or now_label(),
            }
        )
        seen.add(drone_id)
    return {"version": 1, "updated_at": payload.get("updated_at") or now_label(), "drones": drones}


def save_fleet_manifest(payload: dict) -> None:
    FLEET_DIR.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = now_label()
    atomic_write_json(FLEET_MANIFEST, payload)


def upsert_fleet_drone(payload: dict) -> dict:
    name = " ".join(str(payload.get("name") or "").split())[:64]
    phone_ip = str(payload.get("phone_ip") or "").strip()[:128]
    if not name:
        raise RuntimeError("Enter a name for this drone endpoint.")
    if not phone_ip:
        raise RuntimeError("Enter the Android phone IP for this drone.")
    requested_id = str(payload.get("drone_id") or "").strip()
    drone_id = slugify_label(requested_id or name, "drone")[:64]
    with FLEET_LOCK:
        manifest = load_fleet_manifest()
        duplicate = next(
            (
                item for item in manifest["drones"]
                if item["phone_ip"] == phone_ip and item["id"] != drone_id
            ),
            None,
        )
        if duplicate:
            raise RuntimeError(
                f"Android endpoint {phone_ip} is already assigned to {duplicate['name']}. "
                "Each simultaneous drone needs its own phone/controller endpoint."
            )
        existing = next((item for item in manifest["drones"] if item["id"] == drone_id), None)
        if existing is None:
            palette = ["#56ddff", "#a78bfa", "#5ee6a8", "#ffb45c", "#ff6f91", "#66a3ff"]
            existing = {
                "id": drone_id,
                "created_at": now_label(),
                "color": palette[len(manifest["drones"]) % len(palette)],
            }
            manifest["drones"].append(existing)
        existing.update({"name": name, "phone_ip": phone_ip, "updated_at": now_label()})
        save_fleet_manifest(manifest)
        return json.loads(json.dumps(existing))


def delete_fleet_drone(drone_id: str) -> None:
    drone_id = slugify_label(drone_id, "")
    with FLEET_LOCK:
        session = FLEET_SESSIONS.get(drone_id)
        if session and session.get("status") in ACTIVE_JOB_STATES:
            raise RuntimeError("Stop this drone session before removing its endpoint.")
        manifest = load_fleet_manifest()
        before = len(manifest["drones"])
        manifest["drones"] = [item for item in manifest["drones"] if item["id"] != drone_id]
        if len(manifest["drones"]) == before:
            raise RuntimeError(f"Unknown fleet drone: {drone_id}")
        save_fleet_manifest(manifest)


def fleet_session_public_root(drone_id: str) -> Path:
    return FLEET_DIR / "drones" / slugify_label(drone_id, "drone") / "live"


def fleet_event(drone_id: str, message: str, level: str = "info") -> None:
    message = " ".join(str(message or "").split())
    if not message:
        return
    with FLEET_LOCK:
        session = FLEET_SESSIONS.get(drone_id)
        if not session:
            return
        events = session.setdefault("events", [])
        if events and events[-1].get("message") == message and events[-1].get("level") == level:
            events[-1]["updated_at"] = time.time()
        else:
            events.append(
                {
                    "id": uuid.uuid4().hex[:12],
                    "message": message,
                    "level": level,
                    "created_at": time.time(),
                }
            )
            del events[:-160]
        session["message"] = message
        session["updated_at"] = time.time()


def fleet_update(drone_id: str, **fields) -> None:
    with FLEET_LOCK:
        session = FLEET_SESSIONS.get(drone_id)
        if not session:
            return
        session.update(fields)
        session["updated_at"] = time.time()


def fleet_session_snapshot(session: dict) -> dict:
    ignored = {"stop_event", "thread"}
    payload = {key: value for key, value in session.items() if key not in ignored}
    return json.loads(json.dumps(payload))


def fleet_snapshot() -> dict:
    manifest = load_fleet_manifest()
    for drone in manifest["drones"]:
        drone_id = drone["id"]
        bridge_path = fleet_session_public_root(drone_id) / "status.json"
        control_path = fleet_session_public_root(drone_id) / "control_status.json"
        try:
            bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            bridge = {}
        try:
            control = json.loads(control_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            control = {}
        if bridge or control:
            control_status = control.get("status")
            if not control_status and control:
                control_status = "ok" if bool(control.get("ok")) else "error"
            fields = {
                "bridge_status": bridge.get("status"),
                "bridge_message": bridge.get("message"),
                "bridge_updated_at": bridge.get("updated_at"),
                "frames_saved": bridge.get("frames_saved", 0),
            }
            with FLEET_LOCK:
                session = FLEET_SESSIONS.get(drone_id) or {}
                command_matches = bool(control.get("id")) and control.get("id") == session.get("last_command_id")
            if command_matches:
                fields.update(
                    last_control_status=control_status,
                    last_control_message=control.get("message") or control.get("error"),
                    last_control_result=control.get("result"),
                )
                command = str(control.get("command") or "").strip().lower()
                pending = control_status in {"queued", "running"}
                fields["control_pending"] = pending
                if command == "takeoff":
                    fields["takeoff_pending"] = pending
                    if control_status == "ok":
                        fields["airborne"] = True
                    elif control_status == "error":
                        fields["airborne"] = False
                elif command == "land":
                    fields["land_pending"] = pending
                    if control_status == "ok":
                        fields["airborne"] = False
                        fields["patrol_running"] = False
                elif command == "mission":
                    fields["patrol_pending"] = control_status == "queued"
                    fields["patrol_running"] = control_status == "running"
                elif command == "hover" and control_status in {"ok", "running"}:
                    fields["patrol_running"] = False
            fleet_update(drone_id, **fields)
    with FLEET_LOCK:
        sessions = {drone_id: fleet_session_snapshot(item) for drone_id, item in FLEET_SESSIONS.items()}
    drones = []
    for drone in manifest["drones"]:
        item = dict(drone)
        item["session"] = sessions.get(drone["id"])
        drones.append(item)
    active = [item for item in drones if (item.get("session") or {}).get("status") in ACTIVE_JOB_STATES]
    airborne = [item for item in drones if bool((item.get("session") or {}).get("airborne"))]
    attention = [
        item for item in drones
        if (item.get("session") or {}).get("status") in {"error", "attention"}
        or (item.get("session") or {}).get("last_control_status") == "error"
    ]
    return {
        "version": 1,
        "drones": drones,
        "summary": {
            "registered": len(drones),
            "active": len(active),
            "airborne": len(airborne),
            "attention": len(attention),
        },
        "hardware": {
            "architecture": "one_android_endpoint_per_drone",
            "message": "Each physical drone requires its own Android phone/controller running MSDKRemote.",
        },
        "updated_at": time.time(),
    }


def manual_patrol_recording_state_path() -> Path:
    return PUBLIC / "live_dji" / "manual_patrol_recording.json"


def start_manual_patrol_recording(payload: dict) -> dict:
    state_path = manual_patrol_recording_state_path()
    if state_path.exists():
        existing = json.loads(state_path.read_text(encoding="utf-8"))
        if existing.get("status") == "recording":
            raise RuntimeError("A manual patrol recording is already active.")
    live_path = PUBLIC / "live_dji" / "status.json"
    if not live_path.exists():
        raise RuntimeError("Start Live Localization before recording a manual patrol.")
    live = json.loads(live_path.read_text(encoding="utf-8"))
    if str(live.get("status") or "").lower() != "streaming":
        raise RuntimeError("The DJI live bridge is not streaming.")
    pose_stream = Path(str(live.get("pose_stream_path") or ""))
    frames_csv = Path(str(live.get("frames_csv") or ""))
    if not pose_stream.is_file() or not frames_csv.is_file():
        raise RuntimeError("Live poses and synchronized camera frames are not ready yet.")
    map_id = str(payload.get("map_id") or "").strip()
    patrol_id = str(payload.get("patrol_id") or "").strip()
    if not map_id or not patrol_id:
        raise RuntimeError("Select a saved patrol before recording.")
    entry = get_map_entry(map_id)
    recording_mode = str(payload.get("recording_mode") or "full_loop").strip().lower()
    route_by_mode = {
        "full_loop": {
            "route_points": [1, 2, 3, 4, 1],
            "route_label": "1 -> 2 -> 3 -> 4 -> 1",
            "start_rule": "operator begins at patrol point 1",
        },
        "continuation_3_4_1": {
            "route_points": [3, 4, 1],
            "route_label": "3 -> 4 -> 1",
            "start_rule": "operator begins at patrol point 3 to complete an existing 1 -> 2 -> 3 teach run",
        },
        "continuation_4_1": {
            "route_points": [4, 1],
            "route_label": "4 -> 1",
            "start_rule": "operator begins at patrol point 4 to complete an existing 1 -> 2 -> 3 -> 4 teach run",
        },
    }
    route = route_by_mode.get(recording_mode)
    if route is None:
        raise RuntimeError("Unknown manual patrol recording mode.")
    recording_id = f"manual_patrol_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    replay_id = pose_stream.parent.name if pose_stream.parent.name.startswith("dji_live_") else None
    state = {
        "status": "recording",
        "recording_id": recording_id,
        "map_id": map_id,
        "map_title": entry.get("title"),
        "patrol_id": patrol_id,
        "patrol_title": str(payload.get("patrol_title") or patrol_id),
        "recording_mode": recording_mode,
        **route,
        "source_replay_id": replay_id,
        "started_at": time.time(),
        "pose_stream_path": str(pose_stream.resolve()),
        "frames_csv_path": str(frames_csv.resolve()),
        "control_history_path": str((PUBLIC / "live_dji" / "control_status_history.jsonl").resolve()),
        "note": (
            "Manual controller stick telemetry is not exposed by the current DJI bridge; "
            "replay controls must be derived from this synchronized map trajectory."
        ),
    }
    atomic_write_json(state_path, state)
    return state


def finish_manual_patrol_recording(payload: dict) -> dict:
    state_path = manual_patrol_recording_state_path()
    if not state_path.exists():
        raise RuntimeError("No manual patrol recording is active.")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "recording":
        raise RuntimeError("No manual patrol recording is active.")
    requested_id = str(payload.get("recording_id") or "").strip()
    if requested_id and requested_id != state.get("recording_id"):
        raise RuntimeError("The active manual patrol recording changed; reload the page.")
    finished_at = time.time()
    started_at = float(state["started_at"])
    pose_path = Path(state["pose_stream_path"])
    if pose_path.exists():
        pose_payload = json.loads(pose_path.read_text(encoding="utf-8"))
    else:
        # Stopping Live Localization may finalize/remove the partial viewer
        # stream before the operator can press Finish. The synchronized camera
        # frames are still a valid teach recording and can be localized later.
        pose_payload = {"poses": []}
    recorded_poses = [
        pose for pose in pose_payload.get("poses", [])
        if isinstance(pose, dict)
        and float(pose.get("received_unix") or 0.0) >= started_at
        and float(pose.get("received_unix") or 0.0) <= finished_at
    ]
    accepted_poses = [
        pose for pose in recorded_poses
        if pose.get("success") is not False
        and not pose.get("held_pose")
        and not pose.get("output_rejected")
        and isinstance(pose.get("rcenter"), list)
        and isinstance(pose.get("rheading"), list)
    ]
    entry = get_map_entry(str(state["map_id"]))
    out_dir = map_asset_dir(entry) / "manual_patrol_recordings" / str(state["recording_id"])
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_rows: list[dict] = []
    with Path(state["frames_csv_path"]).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                received = float(row.get("received_unix") or 0.0)
            except (TypeError, ValueError):
                continue
            if started_at <= received <= finished_at:
                frame_rows.append(row)
    if frame_rows:
        with (out_dir / "frames.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0].keys()))
            writer.writeheader()
            writer.writerows(frame_rows)
    if not recorded_poses and not frame_rows:
        raise RuntimeError("No poses or synchronized camera frames were captured.")

    command_events: list[dict] = []
    history_path = Path(state["control_history_path"])
    if history_path.exists():
        for line in history_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
                event_time = float(event.get("updated_at") or event.get("created_at") or 0.0)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if started_at <= event_time <= finished_at:
                command_events.append(event)

    recording = {
        **state,
        "status": "completed",
        "finished_at": finished_at,
        "duration_seconds": finished_at - started_at,
        "pose_count": len(recorded_poses),
        "accepted_pose_count": len(accepted_poses),
        "frame_count": len(frame_rows),
        "command_event_count": len(command_events),
        "trajectory_frame": "atlas_room",
        "requires_offline_pose_recovery": not bool(accepted_poses),
        "start_rule": state.get("start_rule"),
        "poses": recorded_poses,
    }
    trajectory = {
        "recording_id": state["recording_id"],
        "map_id": state["map_id"],
        "patrol_id": state["patrol_id"],
        "frame": "atlas_room",
        "samples": [
            {
                "received_unix": pose.get("received_unix"),
                "time_sec": pose.get("time_sec"),
                "image_name": pose.get("image_name"),
                "rcenter": pose.get("rcenter"),
                "rheading": pose.get("rheading"),
                "objective": pose.get("objective"),
                "instance_id": pose.get("instance_id"),
            }
            for pose in accepted_poses
        ],
    }
    atomic_write_json(out_dir / "recording.json", recording)
    atomic_write_json(out_dir / "trajectory.json", trajectory)
    atomic_write_json(out_dir / "command_events.json", {"events": command_events})
    completed_state = {
        **state,
        "status": "completed",
        "finished_at": finished_at,
        "output_dir": str(out_dir.resolve()),
        "pose_count": len(recorded_poses),
        "accepted_pose_count": len(accepted_poses),
        "frame_count": len(frame_rows),
    }
    atomic_write_json(state_path, completed_state)
    return completed_state


def camera_center_from_rt(R: list, t: list) -> list[float] | None:
    if not isinstance(R, list) or len(R) != 3 or not isinstance(t, list) or len(t) != 3:
        return None
    try:
        return [
            -sum(float(R[row][col]) * float(t[row]) for row in range(3))
            for col in range(3)
        ]
    except (TypeError, ValueError, IndexError):
        return None


def room_dot(a: list[float], b: list[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def room_sub(a: list[float], b: list[float]) -> list[float]:
    return [float(x) - float(y) for x, y in zip(a, b)]


def room_mul(a: list[float], s: float) -> list[float]:
    return [float(x) * float(s) for x in a]


def room_cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def room_normalize(a: list[float]) -> list[float]:
    n = max(float(sum(x * x for x in a)) ** 0.5, 1e-12)
    return [float(x) / n for x in a]


def room_mat_vec(C: list[list[float]], v: list[float]) -> list[float]:
    return [room_dot(row, v) for row in C]


def room_power_eigen(C: list[list[float]], seed: list[float]) -> dict:
    v = room_normalize(seed)
    for _ in range(64):
        v = room_normalize(room_mat_vec(C, v))
    Cv = room_mat_vec(C, v)
    return {"v": v, "lambda": room_dot(v, Cv)}


def room_deflate(C: list[list[float]], eig: dict) -> list[list[float]]:
    v = eig["v"]
    lam = float(eig["lambda"])
    return [[C[r][c] - lam * v[r] * v[c] for c in range(3)] for r in range(3)]


def room_covariance(points: list[list[float]], center: list[float]) -> list[list[float]]:
    C = [[0.0, 0.0, 0.0] for _ in range(3)]
    inv = 1.0 / max(1, len(points))
    for p in points:
        d = room_sub(p, center)
        for r in range(3):
            for c in range(3):
                C[r][c] += d[r] * d[c] * inv
    return C


def room_median(values: list[float]) -> float:
    if not values:
        return 0.0
    arr = sorted(float(v) for v in values)
    mid = len(arr) // 2
    if len(arr) % 2:
        return arr[mid]
    return 0.5 * (arr[mid - 1] + arr[mid])


def explicit_room_transform(room_alignment):
    matrix = room_alignment.get("matrix") if isinstance(room_alignment, dict) else None
    if not isinstance(matrix, list) or len(matrix) != 3:
        return None
    try:
        rows = [[float(value) for value in row] for row in matrix]
    except (TypeError, ValueError):
        return None
    if any(len(row) != 4 or not all(math.isfinite(value) for value in row) for row in rows):
        return None

    def transform(xyz):
        if xyz is None:
            return None
        try:
            p = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
        except (TypeError, ValueError, IndexError):
            return None
        return [sum(row[i] * p[i] for i in range(3)) + row[3] for row in rows]

    def transform_direction(xyz):
        if xyz is None:
            return None
        try:
            p = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
        except (TypeError, ValueError, IndexError):
            return None
        return [sum(row[i] * p[i] for i in range(3)) for row in rows]

    transform.direction = transform_direction
    return transform


def build_room_transform_from_scene(
    scene_json: Path | None,
    display_z_sign: float = -1.0,
    room_alignment=None,
):
    """Match viewer/app.js buildRoomFrame() so patrol targets and TSolve poses share one frame."""
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
    cloud: list[list[float]] = []
    for row in visual_rows:
        if not isinstance(row, dict) or not row.get("xyz"):
            continue
        try:
            cloud.append([float(row["xyz"][0]), float(row["xyz"][1]), float(row["xyz"][2])])
        except (TypeError, ValueError, IndexError):
            continue
    cameras: list[list[float]] = []
    for row in scene.get("map_cameras", []):
        if not isinstance(row, dict) or not row.get("center"):
            continue
        try:
            cameras.append([float(row["center"][0]), float(row["center"][1]), float(row["center"][2])])
        except (TypeError, ValueError, IndexError):
            continue
    if not cloud:
        return None
    sample_stride = max(1, int(math.ceil(len(cloud) / 7000.0)))
    sample = cloud[::sample_stride] + cameras
    center = [sum(p[i] for p in sample) / max(1, len(sample)) for i in range(3)]
    C = room_covariance(sample, center)
    e0 = room_power_eigen(C, [1.0, 0.2, 0.1])
    e1 = room_power_eigen(room_deflate(C, e0), [0.1, 1.0, 0.2])
    axis_x = room_normalize(e0["v"])
    axis_z = room_normalize(room_sub(e1["v"], room_mul(axis_x, room_dot(e1["v"], axis_x))))
    if float(display_z_sign) < 0:
        axis_z = room_mul(axis_z, -1.0)
    axis_y = room_normalize(room_cross(axis_z, axis_x))

    def raw_transform(xyz: list[float]) -> list[float]:
        d = room_sub(xyz, center)
        return [room_dot(d, axis_x), room_dot(d, axis_y), room_dot(d, axis_z)]

    point_y = [raw_transform(p)[1] for p in cloud[:5000]]
    cam_y = [raw_transform(p)[1] for p in cameras]
    if cam_y and room_median(cam_y) < room_median(point_y):
        axis_y = room_mul(axis_y, -1.0)

    def transform(xyz):
        if xyz is None:
            return None
        try:
            p = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
        except (TypeError, ValueError, IndexError):
            return None
        d = room_sub(p, center)
        return [room_dot(d, axis_x), room_dot(d, axis_y), room_dot(d, axis_z)]

    def transform_direction(xyz):
        if xyz is None:
            return None
        try:
            p = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
        except (TypeError, ValueError, IndexError):
            return None
        return [room_dot(p, axis_x), room_dot(p, axis_y), room_dot(p, axis_z)]

    transform.direction = transform_direction
    return transform


def room_heading_from_R(R, room_transform):
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
    n = math.sqrt(sum(v * v for v in room_forward))
    if n < 1e-9:
        return None
    return [room_forward[0] / n, 0.0, room_forward[2] / n]


def load_instance_meta(instances_dir: Path, instance_id: str) -> dict:
    p = instances_dir / instance_id / "input.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def vector_list(v) -> list[float]:
    if hasattr(v, "tolist"):
        v = v.tolist()
    return [float(x) for x in v]


def matrix_list(M) -> list[list[float]]:
    if hasattr(M, "tolist"):
        M = M.tolist()
    return [[float(x) for x in row] for row in M]


def load_colmap_query_pose_by_name(localized_model_text: Path | None) -> dict[str, dict]:
    if (
        localized_model_text is None
        or read_images_text is None
        or qvec_to_rotmat is None
        or colmap_camera_center is None
        or not (localized_model_text / "images.txt").exists()
    ):
        return {}
    images_path = localized_model_text / "images.txt"
    try:
        cache_key = str(images_path.resolve())
        mtime = images_path.stat().st_mtime
        cached = COLMAP_QUERY_POSE_CACHE.get(cache_key)
        if cached and cached[0] == mtime:
            return cached[1]
    except OSError:
        return {}
    try:
        images = read_images_text(images_path)
    except Exception:
        return {}
    out = {}
    for im in images.values():
        if not str(im.name).startswith("query/"):
            continue
        try:
            R = qvec_to_rotmat(im.qvec)
            C = colmap_camera_center(im)
            out[str(im.name)] = {
                "image_name": im.name,
                "R": matrix_list(R),
                "t": vector_list(im.tvec),
                "center": vector_list(C),
                "registered_points": int(sum(1 for pid in im.point3d_ids if int(pid) >= 0)),
            }
        except Exception:
            continue
    try:
        COLMAP_QUERY_POSE_CACHE[cache_key] = (mtime, out)
    except UnboundLocalError:
        pass
    return out


def colmap_reference_from_meta(meta: dict) -> dict | None:
    if qvec_to_rotmat is None:
        return None
    qvec = meta.get("colmap_qvec_world_to_camera")
    tvec = meta.get("colmap_tvec_world_to_camera")
    if qvec is None or tvec is None:
        return None
    try:
        R = qvec_to_rotmat(qvec)
        t = [float(x) for x in tvec]
        center = [
            -sum(float(R[row][col]) * float(t[row]) for row in range(3))
            for col in range(3)
        ]
        return {
            "image_name": meta.get("image_name"),
            "R": matrix_list(R),
            "t": t,
            "center": center,
            "registered_points": int(meta.get("colmap_registered_points") or meta.get("points") or 0),
        }
    except Exception:
        return None


def build_partial_pose_payload(
    tsolve_runtime: Path,
    drone_video: Path,
    replay_id: str,
    expected_count: int = 0,
    localized_model_text: Path | None = None,
    room_transform=None,
) -> dict:
    static_dir = tsolve_runtime / "persistent_static_json"
    instances_dir = tsolve_runtime / "instances_all"
    colmap_query_pose_by_name = load_colmap_query_pose_by_name(localized_model_text)
    poses = []
    if static_dir.exists():
        for result_path in sorted(static_dir.glob("*.json")):
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            instance_id = result_path.stem
            meta = load_instance_meta(instances_dir, instance_id)
            R = result.get("R")
            t = result.get("t")
            if isinstance(t, list) and len(t) == 1 and isinstance(t[0], list):
                t = t[0]
            center = camera_center_from_rt(R, t)
            success = bool(result.get("success"))
            rcenter = room_transform(center) if success and room_transform is not None else None
            rheading = room_heading_from_R(R, room_transform) if success and room_transform is not None else None
            poses.append(
                {
                    "instance_id": instance_id,
                    "success": success,
                    "time_sec": meta.get("time_sec"),
                    "image_name": meta.get("image_name"),
                    "R": R,
                    "t": t,
                    "center": center if success else None,
                    "rcenter": rcenter if success else None,
                    "rheading": rheading if success else None,
                    "raw_center": center if not success else None,
                    "objective": result.get("objective"),
                    "total_ms": result.get("total_ms"),
                    "stages_ms": result.get("stages_ms", {}),
                    "colmap_reference": colmap_query_pose_by_name.get(str(meta.get("image_name")))
                    or colmap_reference_from_meta(meta),
                }
            )
    def pose_sort_key(pose: dict) -> tuple[float, str]:
        try:
            frame_time = float(pose["time_sec"])
        except (TypeError, ValueError, KeyError):
            frame_time = float("inf")
        return frame_time, str(pose.get("instance_id") or "")

    poses.sort(key=pose_sort_key)
    return {
        "mode": "simulated_live_tsolve_partial",
        "replay_id": replay_id,
        "frame_source": str(drone_video),
        "expected_count": int(expected_count or 0),
        "processed_count": len(poses),
        "complete": bool(expected_count and len(poses) >= expected_count),
        "updated_at": time.time(),
        "poses": poses,
    }


def set_live_stream(stream: dict | None) -> None:
    with STATE_LOCK:
        STATE["drone"]["live_stream"] = stream
        STATE["drone"]["updated_at"] = time.time()


def update_live_stream(**fields) -> None:
    with STATE_LOCK:
        stream = STATE["drone"].get("live_stream")
        if not isinstance(stream, dict):
            stream = {}
        stream.update(fields)
        STATE["drone"]["live_stream"] = stream
        STATE["drone"]["updated_at"] = time.time()


def current_live_stream() -> dict | None:
    with STATE_LOCK:
        stream = STATE["drone"].get("live_stream")
        return json.loads(json.dumps(stream)) if isinstance(stream, dict) else None


def dji_live_bridge_readiness(live_status: dict, command: str) -> tuple[bool, str]:
    state = str(live_status.get("status") or "").strip().lower()
    try:
        updated_at = float(live_status.get("updated_at") or 0.0)
    except (TypeError, ValueError):
        updated_at = 0.0
    age = time.time() - updated_at if updated_at > 0 else float("inf")
    age_text = f"{age:.1f}s old" if math.isfinite(age) else "not available"
    if command == "mission":
        if state != "streaming":
            return False, f"DJI bridge is {state or 'offline'}, not streaming."
        if age > 4.0:
            return False, f"DJI bridge heartbeat is stale ({age_text})."
        if not bool(live_status.get("control_enabled")):
            return False, "DJI bridge is running in Live Check view-only mode; flight commands are disabled."
        return True, "DJI bridge is streaming and control-enabled."
    if command in {"takeoff", "land", "hover", "enable", "disable"}:
        if state not in {"streaming", "waiting_for_video"}:
            return False, f"DJI bridge is {state or 'offline'}."
        if age > 8.0:
            return False, f"DJI bridge heartbeat is stale ({age_text})."
        if not bool(live_status.get("control_enabled")):
            return False, "DJI bridge is running in Live Check view-only mode; flight commands are disabled."
        return True, "DJI bridge is connected."
    return False, f"Unsupported DJI command: {command}"


def validated_enemy_pursuit_mission(mission: dict) -> dict:
    if not bool(mission.get("operator_confirmed")):
        raise RuntimeError("Enemy pursuit requires an explicit operator confirmation.")
    enemy_id = str(mission.get("enemy_id") or "").strip()
    lib, profile = get_enemy_profile(enemy_id)
    if str(lib.get("model_status") or "") != "trained" or not selected_enemy_model_path():
        raise RuntimeError("Enemy pursuit is locked until the trained detector is selected and available.")
    calibration = normalize_enemy_range_calibration(profile.get("range_calibration"))
    if calibration.get("status") != "validated" or not isinstance(calibration.get("model"), dict):
        raise RuntimeError("Enemy pursuit is locked until live range calibration passes validation.")
    try:
        requested_stop = float(mission.get("stop_clearance_m") or 0.50)
    except (TypeError, ValueError):
        requested_stop = 0.50
    stop_clearance = max(0.50, min(2.0, requested_stop))
    model = dict(calibration["model"])
    calibrated_stop = max(0.50, float(model.get("stop_clearance_m") or 0.50))
    if stop_clearance < calibrated_stop:
        raise RuntimeError(
            f"Requested {stop_clearance:.2f} m stop clearance is below the validated {calibrated_stop:.2f} m calibration."
        )
    map_id = str(mission.get("map_id") or "").strip()
    map_entry = next((item for item in load_library().get("maps", []) if item.get("id") == map_id), None)
    if not isinstance(map_entry, dict):
        raise RuntimeError("Enemy pursuit requires the selected saved map.")
    safety_barriers = [item for item in map_entry.get("safety_barriers") or [] if isinstance(item, dict)]
    safety_obstacles = [item for item in map_entry.get("safety_obstacles") or [] if isinstance(item, dict)]
    if len(safety_barriers) < 3:
        raise RuntimeError("Enemy pursuit requires a closed saved-wall geofence on the selected map.")
    initial_pose_offset = mission.get("initial_pose_offset_room")
    if not isinstance(initial_pose_offset, (list, tuple)) or len(initial_pose_offset) < 3:
        initial_pose_offset = [0.0, 0.0, 0.0]
    initial_pose_offset = [float(initial_pose_offset[0]), 0.0, float(initial_pose_offset[2])]
    if math.hypot(initial_pose_offset[0], initial_pose_offset[2]) > 1.0:
        raise RuntimeError("Enemy pursuit initial pose correction exceeds the 1.0 m safety limit.")
    safe = dict(mission)
    pursuit_yaw_sign = -1.0 if float(mission.get("pursuit_yaw_sign") or 1.0) < 0 else 1.0
    safe.update(
        {
            "client_safety_version": 3,
            "guided_enabled": True,
            "enemy_pursuit": True,
            "enemy_id": profile["id"],
            "target_class_name": profile["class_name"],
            "range_model": model,
            "safety_barriers": safety_barriers,
            "safety_obstacles": safety_obstacles,
            "safety_motion_buffer_m": max(0.30, min(1.0, float(mission.get("safety_motion_buffer_m") or 0.30))),
            "initial_pose_offset_room": initial_pose_offset,
            "stop_clearance_m": stop_clearance,
            "pose_max_age_seconds": max(0.5, min(1.8, float(mission.get("pose_max_age_seconds") or 1.2))),
            "pose_recovery_seconds": max(1.0, min(8.0, float(mission.get("pose_recovery_seconds") or 4.0))),
            "pulse_seconds": max(0.10, min(0.20, float(mission.get("pulse_seconds") or 0.14))),
            "max_forward_rc": max(0.01, min(0.03, float(mission.get("max_forward_rc") or 0.025))),
            "max_yaw_rc": max(0.01, min(0.035, float(mission.get("max_yaw_rc") or 0.028))),
            "max_vertical_rc": max(0.005, min(0.015, float(mission.get("max_vertical_rc") or 0.010))),
            "vertical_tracking_enabled": bool(mission.get("vertical_tracking_enabled")),
            "pursuit_yaw_sign": pursuit_yaw_sign,
            "detection_max_age_seconds": max(0.60, min(1.50, float(mission.get("detection_max_age_seconds") or 1.0))),
            "lost_target_abort_seconds": max(2.0, min(8.0, float(mission.get("lost_target_abort_seconds") or 4.0))),
            "max_pursuit_seconds": max(5.0, min(60.0, float(mission.get("max_pursuit_seconds") or 45.0))),
        }
    )
    return safe


def validated_guarded_patrol_mission(mission: dict) -> dict:
    try:
        client_safety_version = int(mission.get("client_safety_version") or 0)
    except (TypeError, ValueError):
        client_safety_version = 0
    if client_safety_version < 3:
        raise RuntimeError("Patrol requires the current wall/obstacle safety code. Reload ATLAS before flight.")
    map_id = str(mission.get("map_id") or "").strip()
    patrol_id = str(mission.get("patrol_id") or "").strip()
    library = load_library()
    live_patrol_lock = resolved_live_patrol_lock(library)
    if live_patrol_lock is not None and not live_patrol_lock["flight_enabled"]:
        reason = live_patrol_lock["suspension_reason"] or "offline safety validation is required"
        raise RuntimeError(f"Live Patrol is suspended: {reason}")
    if live_patrol_lock is not None and (
        map_id != live_patrol_lock["map_id"]
        or patrol_id != live_patrol_lock["patrol_id"]
    ):
        raise RuntimeError(
            "Live patrol is pinned to "
            f"{live_patrol_lock['map_title']} / {live_patrol_lock['patrol_title']}. "
            "Reload ATLAS and use the pinned patrol."
        )
    map_entry = next((item for item in library.get("maps", []) if item.get("id") == map_id), None)
    if not isinstance(map_entry, dict):
        raise RuntimeError("Patrol requires the selected saved map.")
    saved_patrol = next(
        (item for item in map_entry.get("patrols", []) if isinstance(item, dict) and item.get("id") == patrol_id),
        None,
    )
    if not isinstance(saved_patrol, dict):
        raise RuntimeError("Patrol requires a saved patrol belonging to the selected map.")
    patrol_stage = str(mission.get("patrol_stage") or "combined").strip().lower()
    if patrol_stage not in {"entry", "loop", "combined"}:
        raise RuntimeError("Patrol stage must be entry, loop, or combined.")

    def route_point(value: object) -> list[float] | None:
        if not isinstance(value, (list, tuple)) or len(value) < 3:
            return None
        try:
            point = [float(value[0]), float(value[1]), float(value[2])]
        except (TypeError, ValueError):
            return None
        return point if all(math.isfinite(item) for item in point) else None

    saved_points = [
        route_point(item.get("rxyz"))
        for item in saved_patrol.get("points") or []
        if isinstance(item, dict)
    ]
    saved_points = [point for point in saved_points if point is not None]
    mission_route = [
        route_point(item)
        for item in mission.get("route") or []
    ]
    mission_route = [point for point in mission_route if point is not None]

    def same_room_point(left: list[float], right: list[float], tolerance: float = 0.08) -> bool:
        return math.hypot(left[0] - right[0], left[2] - right[2]) <= tolerance

    if live_patrol_lock is not None and live_patrol_lock.get("baseline_replay_id"):
        if len(saved_points) < 2 or not mission_route:
            raise RuntimeError("Pinned patrol route geometry is missing.")
        if patrol_stage == "entry":
            if bool(mission.get("loop")) or not same_room_point(mission_route[-1], saved_points[0]):
                raise RuntimeError("Go to Start must end at saved patrol Point 1 without looping.")
        elif patrol_stage == "loop":
            expected_loop = saved_points + [saved_points[0]]
            if (
                not bool(mission.get("loop"))
                or len(mission_route) != len(expected_loop)
                or any(
                    not same_room_point(actual, expected)
                    for actual, expected in zip(mission_route, expected_loop)
                )
            ):
                raise RuntimeError(
                    "Start 2 Circles must use the exact saved Point 1→2→3→4→1 geometry."
                )
        else:
            expected_loop = saved_points + [saved_points[0]]
            loop_tail = mission_route[-len(expected_loop):]
            if (
                not bool(mission.get("loop"))
                or len(mission_route) <= len(expected_loop)
                or len(loop_tail) != len(expected_loop)
                or any(
                    not same_room_point(actual, expected)
                    for actual, expected in zip(loop_tail, expected_loop)
                )
            ):
                raise RuntimeError(
                    "Connected patrol must enter Point 1, then use the exact saved "
                    "Point 1→2→3→4→1 loop geometry."
                )
    safety_barriers = [item for item in map_entry.get("safety_barriers") or [] if isinstance(item, dict)]
    safety_obstacles = [item for item in map_entry.get("safety_obstacles") or [] if isinstance(item, dict)]
    if len(safety_barriers) < 3:
        raise RuntimeError("Patrol requires a closed saved-wall geofence on the selected map.")
    try:
        requested_buffer = float(mission.get("safety_motion_buffer_m") or 0.30)
    except (TypeError, ValueError):
        requested_buffer = 0.30
    safe = dict(mission)
    safe.update(
        {
            "client_safety_version": 3,
            "safety_barriers": safety_barriers,
            "safety_obstacles": safety_obstacles,
            "safety_motion_buffer_m": max(0.30, min(1.0, requested_buffer)),
            "patrol_stage": patrol_stage,
        }
    )
    if live_patrol_lock is not None and live_patrol_lock.get("baseline_replay_id"):
        safe.update(
            {
                "baseline_replay_id": live_patrol_lock["baseline_replay_id"],
                "patrol_laps": (
                    live_patrol_lock.get("patrol_laps") or 2
                    if patrol_stage in {"loop", "combined"}
                    else 1
                ),
                "route_monotonic_gate": True,
                "continuous_relocalization": patrol_stage in {"entry", "loop", "combined"},
            }
        )
    if live_patrol_lock is not None and live_patrol_lock["reference_profile"] == "live-atlas-141441":
        # Server-owned profile based on the observed behavior of Live ATLAS
        # 14:14:41, which physically reached Point 4 and turned toward Point 1.
        # The browser cannot silently drift these values.
        safe.update(
            {
                "reference_profile": "live-atlas-141441",
                "pose_max_age_seconds": 2.5,
                "pose_recovery_seconds": 8.0,
                "continuous_relocalization": True,
                # Horizontal travel is a low-stick cruise refreshed at 10 Hz.
                # It may continue only while new translation-safe room poses
                # arrive; loss of that stream immediately sends zero stick.
                "one_pulse_pose_confirmation": True,
                "smooth_continuous_cruise": True,
                "cruise_window_seconds": 0.55,
                "cruise_pose_watchdog_seconds": 0.65,
                "max_unverified_translation_m": 0.18,
                "pulse_seconds": 0.30,
                "max_forward_rc": 0.022,
                "max_lateral_rc": 0.0,
                "allow_lateral_rc": False,
                "allow_axis_auto_calibration": False,
                # Start from the operator-aligned COLMAP/model heading. A
                # short translation probe produced a false ~43-degree offset
                # in the failed 2026-08-11 run and made the drone yaw past the
                # correct route direction. The first bounded yaw pulse still
                # verifies DJI yaw sign before translation.
                "operator_heading_calibrated": True,
                "require_physical_forward_probe": False,
                "max_heading_calibration_error_deg": 25.0,
                "max_yaw_rc": 0.050,
                "max_scan_yaw_rc": 0.025,
                "allow_patrol_scan_yaw": False,
                "alignment_grace_seconds": 35.0,
                "max_vertical_rc": 0.018,
                "max_step_seconds": 2.0,
                "max_cruise_seconds": 120.0,
                "max_pose_step_map_units": 0.30,
                "max_pose_step_hard_map_units": 0.30,
                "cross_track_recovery_start_map_units": 0.30,
                "max_cross_track_map_units": 0.80,
                "arrival_radius_map_units": 0.24,
                "arrival_deadband_map_units": 0.14,
            }
        )
    return safe


def send_dji_flight_command(payload: dict) -> dict:
    command = str(payload.get("command", "")).strip().lower()
    if command not in {"takeoff", "land", "enable", "disable", "hover", "mission"}:
        raise ValueError(f"Unsupported DJI flight command: {command}")
    phone_ip = str(payload.get("phone_ip", "")).strip()
    height_m = payload.get("height_m")
    if height_m is not None:
        height_m = max(0.1, min(2.0, float(height_m)))
    mission = payload.get("mission") if isinstance(payload.get("mission"), dict) else None
    if command == "mission" and isinstance(mission, dict) and bool(mission.get("enemy_pursuit")):
        mission = validated_enemy_pursuit_mission(mission)
    elif command == "mission" and isinstance(mission, dict) and bool(mission.get("patrol")):
        mission = validated_guarded_patrol_mission(mission)

    stream = current_live_stream() or recover_live_stream_from_disk() or {}
    live_status_path = PUBLIC / "live_dji" / "status.json"
    live_status = {}
    if live_status_path.exists():
        try:
            live_status = json.loads(live_status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            live_status = {}
    live_state = str(live_status.get("status") or "").strip().lower()
    bridge_ready, bridge_reason = dji_live_bridge_readiness(live_status, command)
    use_live_bridge = bool(stream.get("live_atlas") and live_status_path.exists() and bridge_ready)
    if command == "mission" and not use_live_bridge:
        raise ValueError(f"Start Live ATLAS before confirming a mission, so the DJI bridge can receive the mission packet. {bridge_reason}")
    if command in {"takeoff", "land", "hover"} and stream.get("live_atlas") and live_status_path.exists() and not bridge_ready:
        raise ValueError(f"Live DJI bridge is not ready for {command}. {bridge_reason}")
    command_id = uuid.uuid4().hex
    command_payload = {
        "id": command_id,
        "command": command,
        "phone_ip": phone_ip or stream.get("phone_ip"),
        "height_m": height_m,
        "created_at": time.time(),
    }
    if command == "hover" and bool(payload.get("emergency_stop")):
        # Preserve the browser's safety intent.  Dropping this field made the
        # live bridge reject Hover Now as a competing command while a patrol
        # was active.
        command_payload["emergency_stop"] = True
    if command == "mission":
        command_payload["mission"] = mission or {}
    if use_live_bridge:
        command_path = PUBLIC / "live_dji" / "control_command.json"
        atomic_write_json(command_path, command_payload)
        append_log("drone", f"Queued DJI {command} command through live bridge.")
        return {
            "ok": True,
            "queued": True,
            "via": "live_bridge",
            "command": command,
            "height_m": height_m,
            "command_id": command_id,
            "message": f"DJI {command} command queued on the live bridge.",
        }

    if not phone_ip:
        raise ValueError("Missing phone_ip. Start Live ATLAS or enter the Android phone IP.")
    cfg = load_config()
    py = Path(cfg["python"])
    cmd = [
        str(py),
        str(ROOT / "scripts" / "atlas_dji_command.py"),
        "--phone-ip",
        phone_ip,
        "--command",
        command,
    ]
    if height_m is not None:
        cmd += ["--height-m", f"{height_m:.3f}"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or f"DJI command failed: {command}").strip())
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        result = {"ok": True, "stdout": proc.stdout.strip()}
    append_log("drone", f"Sent DJI {command} command directly.")
    return {
        "ok": True,
        "queued": False,
        "via": "direct_opendji",
        "command": command,
        "height_m": height_m,
        "result": result,
        "message": f"DJI {command} command sent.",
    }


def recover_live_stream_from_disk() -> dict | None:
    """Reattach the UI to a live session after a browser/server restart."""
    if current_live_stream():
        return current_live_stream()

    status_path = PUBLIC / "live_dji" / "status.json"
    try:
        live_status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if str(live_status.get("status") or "").lower() not in {"streaming", "starting"}:
        return None

    candidates: list[tuple[float, Path]] = []
    for partial in MAPS_DIR.glob("*/replays/dji_live_*/poses_partial.json"):
        try:
            payload = json.loads(partial.read_text(encoding="utf-8"))
            if payload.get("complete"):
                continue
            updated = float(payload.get("updated_at") or partial.stat().st_mtime)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        candidates.append((updated, partial))
    if not candidates:
        return None

    _, partial_pose_path = max(candidates, key=lambda item: item[0])
    replay_dir = partial_pose_path.parent
    replay_id = replay_dir.name
    map_dir = replay_dir.parent.parent
    map_id = map_dir.name
    session = str(live_status.get("session") or "")
    query_frames = PUBLIC / "live_dji_sessions" / session / "query_frames" if session else None
    stream = {
        "live_atlas": True,
        "recovered": True,
        "map_id": map_id,
        "replay_id": replay_id,
        "title": f"Live ATLAS {replay_id[-6:]}",
        "asset_base": public_rel(replay_dir),
        "partial_pose_url": public_rel(partial_pose_path),
        "stop_file": str(replay_dir / "STOP_LIVE_ATLAS"),
        "live_preview_url": "public/live_dji/latest.jpg",
        "query_frame_base_url": public_rel(query_frames) if query_frames else None,
        "pose_count": 0,
        "expected_count": 0,
        "complete": False,
        "started_at": live_status.get("started_at"),
        "message": "Recovered active DJI live stream after app/server restart.",
    }
    try:
        payload = json.loads(partial_pose_path.read_text(encoding="utf-8"))
        poses = payload.get("poses") if isinstance(payload.get("poses"), list) else []
        counts = pose_stream_counts(poses)
        stream["pose_count"] = int(payload.get("processed_count") or len(poses))
        stream["accepted_pose_count"] = counts["poses"]
        stream["held_pose_count"] = counts["held"]
        stream["failed_pose_count"] = counts["failed"]
        stream["expected_count"] = int(payload.get("expected_count") or 0)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    set_live_stream(stream)
    set_job("drone", "running", "Recovered active DJI live stream. Waiting for TSolve poses.")
    return stream


def stream_partial_poses(
    tsolve_runtime: Path,
    drone_video: Path,
    partial_path: Path,
    replay_id: str,
    stop_event: threading.Event,
    expected_count: int,
    localized_model_text: Path | None = None,
    scene_json: Path | None = None,
    display_z_sign: float = -1.0,
    room_alignment=None,
    started_at: float | None = None,
    stream_update_callback=None,
    status_callback=None,
) -> None:
    publish_stream = stream_update_callback or update_live_stream
    publish_status = status_callback or (lambda message: set_job("drone", "running", message))
    room_transform = build_room_transform_from_scene(scene_json, display_z_sign, room_alignment)
    last_count = -1
    last_write = 0.0
    while not stop_event.is_set():
        payload = build_partial_pose_payload(
            tsolve_runtime,
            drone_video,
            replay_id,
            expected_count,
            localized_model_text,
            room_transform,
        )
        try:
            existing = json.loads(partial_path.read_text(encoding="utf-8")) if partial_path.exists() else {}
        except (OSError, json.JSONDecodeError):
            existing = {}
        if existing.get("current_frame"):
            payload["current_frame"] = existing["current_frame"]
            payload["current_frame_time_sec"] = existing.get("current_frame_time_sec")
        poses = payload.get("poses") if isinstance(payload.get("poses"), list) else []
        counts = pose_stream_counts(poses)
        count = int(payload.get("processed_count") or len(poses))
        now = time.time()
        if count != last_count or now - last_write > 2.5:
            atomic_write_json(partial_path, payload)
            stream_update = {
                "pose_count": count,
                "accepted_pose_count": counts["poses"],
                "held_pose_count": counts["held"],
                "failed_pose_count": counts["failed"],
                "expected_count": expected_count,
                "partial_pose_url": public_rel(partial_path),
            }
            if count > 0 and started_at is not None and last_count <= 0:
                stream_update["first_pose_at"] = now
                stream_update["first_pose_latency_seconds"] = now - started_at
            publish_stream(**stream_update)
            if count != last_count and count > 0:
                publish_status(
                    f"Live TSolve self-localization: {counts['poses']}/{count}/{expected_count or '?'} accepted/processed/target.",
                )
            last_count = count
            last_write = now
        time.sleep(0.35)

    payload = build_partial_pose_payload(
        tsolve_runtime,
        drone_video,
        replay_id,
        expected_count,
        localized_model_text,
        room_transform,
    )
    try:
        existing = json.loads(partial_path.read_text(encoding="utf-8")) if partial_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        existing = {}
    if existing.get("current_frame"):
        payload["current_frame"] = existing["current_frame"]
        payload["current_frame_time_sec"] = existing.get("current_frame_time_sec")
    atomic_write_json(partial_path, payload)
    poses = payload.get("poses") if isinstance(payload.get("poses"), list) else []
    counts = pose_stream_counts(poses)
    final_count = int(payload.get("processed_count") or len(poses))
    stream_update = {
        "pose_count": final_count,
        "accepted_pose_count": counts["poses"],
        "held_pose_count": counts["held"],
        "failed_pose_count": counts["failed"],
        "expected_count": expected_count,
        "partial_pose_url": public_rel(partial_path),
        "complete": True,
    }
    if final_count > 0 and started_at is not None and last_count <= 0:
        now = time.time()
        stream_update["first_pose_at"] = now
        stream_update["first_pose_latency_seconds"] = now - started_at
    publish_stream(
        **stream_update,
    )


def snapshot_state() -> dict:
    recover_live_stream_from_disk()
    with STATE_LOCK:
        state = json.loads(json.dumps(STATE))
    state["library"] = load_library()
    recording_path = manual_patrol_recording_state_path()
    if recording_path.exists():
        try:
            state["manual_patrol_recording"] = json.loads(recording_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state["manual_patrol_recording"] = None
    else:
        state["manual_patrol_recording"] = None
    return state


def run_cmd(kind: str, cmd: list[object], stop_event: threading.Event | None = None) -> None:
    cmd = [str(x) for x in cmd]
    append_log(kind, "+ " + " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "minimal")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    register_active_proc(kind, proc)
    try:
        assert proc.stdout is not None
        while proc.poll() is None:
            if stop_event is not None and stop_event.is_set():
                append_log(kind, "Cancellation requested; terminating active subprocess.")
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    append_log(kind, "Subprocess did not exit after SIGTERM; killing it.")
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait(timeout=4)
                raise RuntimeError(f"{kind.capitalize()} job cancelled.")
            ready, _, _ = select.select([proc.stdout], [], [], 0.2)
            if ready:
                line = proc.stdout.readline()
                if line:
                    append_log(kind, line)
        for line in proc.stdout:
            append_log(kind, line)
        rc = proc.wait()
    finally:
        unregister_active_proc(kind, proc)
    if stop_event is not None and stop_event.is_set():
        raise RuntimeError(f"{kind.capitalize()} job cancelled.")
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def start_logged_background_cmd(kind: str, cmd: list[object], stop_event: threading.Event | None = None) -> threading.Thread:
    def target() -> None:
        try:
            run_cmd(kind, cmd, stop_event)
        except Exception as exc:
            if stop_event is None or not stop_event.is_set():
                append_log(kind, f"BACKGROUND ERROR: {exc}")

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


def latest_live_stage_row(stage_times_path: Path | None) -> dict:
    if not stage_times_path or not stage_times_path.exists():
        return {}
    try:
        with stage_times_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except (OSError, csv.Error):
        return {}
    return rows[-1] if rows else {}


def monitor_partial_pose_file(
    partial_path: Path,
    stop_event: threading.Event,
    started_at: float | None = None,
    stage_times_path: Path | None = None,
    stream_update_callback=None,
    status_callback=None,
) -> None:
    publish_stream = stream_update_callback or update_live_stream
    publish_status = status_callback or (lambda message: set_job("drone", "running", message))
    last_count = -1
    last_stage_key = ""
    while not stop_event.is_set():
        try:
            payload = json.loads(partial_path.read_text(encoding="utf-8")) if partial_path.exists() else {}
        except (OSError, json.JSONDecodeError):
            payload = {}
        poses = payload.get("poses") if isinstance(payload.get("poses"), list) else []
        counts = pose_stream_counts(poses)
        count = int(payload.get("processed_count") or len(poses))
        if count != last_count:
            stream_update = {
                "pose_count": count,
                "accepted_pose_count": counts["poses"],
                "held_pose_count": counts["held"],
                "failed_pose_count": counts["failed"],
                "expected_count": int(payload.get("expected_count") or 0),
                "partial_pose_url": public_rel(partial_path),
            }
            if count > 0 and started_at is not None and last_count <= 0:
                now = time.time()
                stream_update["first_pose_at"] = now
                stream_update["first_pose_latency_seconds"] = now - started_at
            publish_stream(**stream_update)
            last_count = count
        latest_stage = latest_live_stage_row(stage_times_path)
        if latest_stage:
            frame_index = str(latest_stage.get("frame_index") or "")
            accepted = str(latest_stage.get("accepted") or "")
            reason = str(latest_stage.get("reason") or "").strip() or "processing"
            total_ms = str(latest_stage.get("total_frame_ms") or "").strip()
            stage_key = f"{frame_index}:{accepted}:{reason}"
            should_report = stage_key != last_stage_key and (
                accepted.lower() == "true"
                or reason != "processing"
                or (frame_index.isdigit() and int(frame_index) % 10 == 0)
            )
            if should_report:
                msg = f"Live localization frame {frame_index}: {reason}"
                if total_ms:
                    try:
                        msg += f" ({float(total_ms):.0f} ms)"
                    except ValueError:
                        pass
                publish_stream(
                    latest_frame_index=frame_index,
                    latest_localization_reason=reason,
                    latest_total_frame_ms=total_ms,
                    message=msg,
                )
                publish_status(msg)
                last_stage_key = stage_key
        time.sleep(0.35)


def finalize_partial_replay(
    *,
    selected: dict,
    replay_id: str,
    replay_title: str,
    out_asset_dir: Path,
    partial_pose_path: Path,
    source_video: str,
    query_frame_base_url: str | None = None,
) -> int:
    payload = json.loads(partial_pose_path.read_text(encoding="utf-8")) if partial_pose_path.exists() else {"poses": []}
    poses = payload.get("poses") if isinstance(payload.get("poses"), list) else []
    counts = pose_stream_counts(poses)
    final_payload = {
        **payload,
        "mode": "dji_live_tsolve_replay" if payload.get("mode") else "atlas_live_tsolve_replay",
        "description": "ATLAS TSolve R,t estimates produced from DJI MSDK live frames.",
        "complete": True,
        "processed_count": len(poses),
        "accepted_count": counts["poses"],
        "held_count": counts["held"],
        "failed_count": counts["failed"],
        "updated_at": time.time(),
        "poses": poses,
    }
    if query_frame_base_url:
        final_payload["query_frame_base_url"] = query_frame_base_url
        final_payload["frame_source"] = query_frame_base_url
    final_pose_path = out_asset_dir / "poses.json"
    atomic_write_json(final_pose_path, final_payload)
    replay = {
        "id": replay_id,
        "title": replay_title,
        "asset_base": public_rel(out_asset_dir),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_video": source_video,
        "counts": counts,
    }
    if query_frame_base_url:
        replay["query_frame_base_url"] = query_frame_base_url
    update_live_stream(
        pose_count=len(poses),
        accepted_pose_count=counts["poses"],
        held_pose_count=counts["held"],
        failed_pose_count=counts["failed"],
        expected_count=int(payload.get("expected_count") or 0),
        partial_pose_url=public_rel(final_pose_path),
        final_pose_url=public_rel(final_pose_path),
        complete=True,
    )
    add_replay_to_map(selected["id"], replay, select=True)
    return counts["poses"]


def save_upload(field, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as f:
        shutil.copyfileobj(field.file, f)


def uploaded_video_fields(form: cgi.FieldStorage) -> list[cgi.FieldStorage]:
    if "video" not in form:
        return []
    raw = form["video"]
    fields = raw if isinstance(raw, list) else [raw]
    return [field for field in fields if getattr(field, "filename", "")]


def save_uploaded_videos(fields: list[cgi.FieldStorage], uploads: Path, prefix: str) -> list[tuple[Path, str]]:
    saved: list[tuple[Path, str]] = []
    for field in fields:
        original_name = str(getattr(field, "filename", "") or "video.mp4")
        suffix = Path(original_name).suffix or ".mp4"
        dst = uploads / f"{prefix}_{uuid.uuid4().hex[:8]}{suffix}"
        save_upload(field, dst)
        saved.append((dst, original_name))
    return saved


def copy_video_to_public(video: Path, asset_dir: Path) -> None:
    media = asset_dir / "media"
    media.mkdir(parents=True, exist_ok=True)
    dst = media / "drone_query.mp4"
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    source_video = video.resolve()
    try:
        os.symlink(source_video, dst)
    except OSError:
        shutil.copy2(source_video, dst)


def set_current_map_frames(frames: Path) -> None:
    with STATE_LOCK:
        STATE["current_map_frames"] = str(frames)


def current_map_frames() -> Path:
    with STATE_LOCK:
        value = STATE.get("current_map_frames")
    if not value:
        fallback = ROOT / "data" / "map_frames"
        if fallback.exists():
            return fallback
        raise RuntimeError("Create or upload a map before uploading a drone video.")
    frames = Path(value)
    if not frames.exists():
        raise RuntimeError(f"Current map frames are missing: {frames}")
    return frames


def manifest_count(inputs_dir: Path) -> int:
    manifest = inputs_dir / "manifest.csv"
    if not manifest.exists():
        return 0
    with manifest.open(encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


def frames_for_entry(entry: dict) -> Path:
    frames_value = entry.get("frames_path")
    if not frames_value:
        raise RuntimeError(f"Map has no stored reconstruction frames: {entry.get('title', entry.get('id'))}")
    frames = Path(frames_value)
    if not frames.exists():
        raise RuntimeError(f"Map frames are missing: {frames}")
    return frames


def choose_sparse_model(sparse_root: Path) -> Path:
    candidates = [p for p in sorted(sparse_root.iterdir()) if p.is_dir() and (p / "images.bin").exists()]
    if not candidates:
        raise RuntimeError(f"No COLMAP sparse model found under {sparse_root}")
    return max(candidates, key=lambda p: (p / "points3D.bin").stat().st_size if (p / "points3D.bin").exists() else 0)


def colmap_artifacts_for_entry(entry: dict) -> dict[str, Path]:
    lib = load_library()
    maps_by_id = {m.get("id"): m for m in lib.get("maps", [])}
    candidate_ids: list[str] = []

    def add_candidate(value: object) -> None:
        map_id = str(value or "").strip()
        if map_id:
            candidate_ids.append(map_id)

    add_candidate(entry.get("localization_map_id"))
    add_candidate(entry.get("source_map_id"))
    add_candidate(entry.get("id"))

    current = entry
    visited = {str(entry.get("id") or "")}
    for _ in range(8):
        parent_id = str(current.get("localization_map_id") or current.get("source_map_id") or "").strip()
        if not parent_id or parent_id in visited:
            break
        visited.add(parent_id)
        add_candidate(parent_id)
        parent = maps_by_id.get(parent_id)
        if parent is None:
            break
        current = parent

    try:
        if Path(str(entry.get("frames_path") or "")).resolve() == (ROOT / "data" / "map_frames").resolve():
            candidate_ids.append("default_demo")
    except OSError:
        pass
    candidate_ids.append("default_demo")

    seen = set()
    for map_id in candidate_ids:
        if not map_id or map_id in seen:
            continue
        seen.add(map_id)
        map_root = ROOT / "results" / "maps" / map_id
        roots = [map_root / "colmap"]
        roots.extend(sorted(map_root.glob("colmap_backup_*"), reverse=True))
        for root in roots:
            database = root / "database.db"
            images = root / "images"
            sparse_text = root / "sparse_text"
            sparse_root = root / "sparse"
            if not (
                database.exists()
                and images.exists()
                and (sparse_text / "points3D.txt").exists()
                and sparse_root.exists()
            ):
                continue
            return {
                "root": root,
                "database": database,
                "images": images,
                "sparse_model": choose_sparse_model(sparse_root),
                "sparse_text": sparse_text,
            }
    raise RuntimeError(
        f"No reusable COLMAP reference map was found for {entry.get('title', entry.get('id'))}. "
        "Create/rebuild the map before uploading a drone video."
    )


def copy_frame_bank(src: Path, dst: Path, prefix: str = "") -> None:
    dst.mkdir(parents=True, exist_ok=True)
    try:
        if not prefix and src.resolve() == dst.resolve():
            return
    except FileNotFoundError:
        pass
    for item in src.iterdir():
        if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            target = dst / (f"{prefix}_{item.name}" if prefix else item.name)
            if not target.exists():
                shutil.copy2(item, target)


def backup_frame_bank(frames: Path) -> Path | None:
    if not frames.exists():
        return None
    backup = frames.parent / f"{frames.name}_backup_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copytree(frames, backup)
    return backup


def restore_frame_bank(backup: Path | None, frames: Path) -> None:
    if backup is None or not backup.exists():
        return
    if frames.exists():
        shutil.rmtree(frames)
    shutil.copytree(backup, frames)


def make_frame_subset(src: Path, dst: Path, max_frames: int) -> Path:
    images = sorted(
        [p for p in src.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if max_frames <= 0 or len(images) <= max_frames:
        return src
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    if max_frames <= 1:
        selected = [images[len(images) // 2]]
    else:
        selected = []
        last_idx = -1
        for i in range(max_frames):
            idx = round(i * (len(images) - 1) / (max_frames - 1))
            if idx != last_idx:
                selected.append(images[idx])
                last_idx = idx
    for src_img in selected:
        shutil.copy2(src_img, dst / src_img.name)
    return dst


def count_frame_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def run_map_from_videos(videos: list[Path], kind_label: str) -> None:
    if not videos:
        raise RuntimeError("No map videos were provided.")
    cfg = load_config()
    py = Path(cfg["python"])
    scripts = ROOT / "scripts"
    map_id = make_map_id("video_map")
    frames = ROOT / "data" / "maps" / map_id / "frames"
    colmap_out = ROOT / "results" / "maps" / map_id / "colmap"
    asset_dir = MAPS_DIR / map_id

    MAP_STOP_EVENT.clear()
    set_map_capture_state(live_preview=None, frames_saved=0, capture_started_at=None)
    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir(parents=True, exist_ok=True)
    set_job("map", "running", f"Extracting map frames from {kind_label}.")
    for idx, video in enumerate(videos):
        prefix = f"map{idx:02d}_{uuid.uuid4().hex[:6]}"
        temp_frames = ROOT / "data" / "maps" / map_id / f"{prefix}_frames"
        set_job("map", "running", f"Extracting map video {idx + 1}/{len(videos)}: {video.name}")
        run_cmd(
            "map",
            [
                py,
                scripts / "extract_frames.py",
                "--video",
                video,
                "--out-dir",
                temp_frames,
                "--fps",
                cfg["map_frame_fps"],
                "--max-size",
                cfg["max_image_size"],
                "--prefix",
                prefix,
            ],
            MAP_STOP_EVENT,
        )
        copy_frame_bank(temp_frames, frames)
        append_log("map", f"Combined new-map frame bank: {count_frame_images(frames)} images.")
    run_map_from_frames(frames, colmap_out, asset_dir, map_id, f"Video Map {time.strftime('%H:%M:%S')}", kind_label)


def run_map_from_video(video: Path, kind_label: str) -> None:
    run_map_from_videos([video], kind_label)


def run_map_from_frames(frames: Path, colmap_out: Path, asset_dir: Path, map_id: str, title: str, source_label: str) -> None:
    cfg = load_config()
    py = Path(cfg["python"])
    scripts = ROOT / "scripts"
    previous = None
    try:
        previous = get_map_entry(map_id)
    except RuntimeError:
        previous = None
    previous_counts = previous.get("counts", {}) if previous else {}
    asset_backup = None
    colmap_backup = None
    if previous and asset_dir.exists():
        asset_backup = asset_dir.parent / f"{asset_dir.name}_asset_backup_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.copytree(asset_dir, asset_backup)
    if previous and colmap_out.exists():
        colmap_backup = colmap_out.parent / f"{colmap_out.name}_backup_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.copytree(colmap_out, colmap_backup)

    dense_enabled = bool(cfg.get("dense_view_map", True))
    dense_max_image_size = int(cfg.get("dense_max_image_size", cfg["max_image_size"]))
    dense_point_limit = int(cfg.get("dense_point_limit", 180000))
    map_matcher = str(cfg.get("map_matcher", "exhaustive")).strip().lower()
    if map_matcher not in {"exhaustive", "sequential"}:
        map_matcher = "exhaustive"
    colmap_cmd = [
        py,
        scripts / "run_colmap_map_only.py",
        "--colmap",
        cfg["colmap_bin"],
        "--frames",
        frames,
        "--out-dir",
        colmap_out,
        "--max-image-size",
        cfg["max_image_size"],
        "--camera-model",
        cfg["map_camera_model"],
        "--matcher",
        map_matcher,
    ]
    if dense_enabled:
        colmap_cmd.extend(["--dense", "--dense-max-image-size", dense_max_image_size])

    set_job(
        "map",
        "running",
        f"Running COLMAP sparse localization map with {map_matcher} matching plus dense viewer reconstruction.",
    )
    run_cmd("map", colmap_cmd, MAP_STOP_EVENT)
    set_job("map", "running", "Exporting map to ATLAS viewer.")
    build_cmd = [
        py,
        scripts / "build_map_only_viewer_data.py",
        "--model-text",
        colmap_out / "sparse_text",
        "--out-public",
        asset_dir,
        "--preserve-media",
    ]
    dense_ply = colmap_out / "dense" / "fused.ply"
    if dense_ply.exists():
        build_cmd.extend(["--dense-points", dense_ply, "--dense-point-limit", dense_point_limit])
    run_cmd("map", build_cmd, MAP_STOP_EVENT)
    validation_path = asset_dir / "map_validation.json"
    set_job("map", "running", "Validating frame bank and COLMAP reconstruction.")
    run_cmd(
        "map",
        [
            py,
            scripts / "atlas_map_validation.py",
            "--frames",
            frames,
            "--colmap-root",
            colmap_out,
            "--asset-dir",
            asset_dir,
            "--matcher",
            map_matcher,
            "--out",
            validation_path,
        ],
        MAP_STOP_EVENT,
    )
    validation = {}
    if validation_path.exists():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        for note in validation.get("notes", []):
            append_log("map", f"validation: {note}")
    counts = read_counts(asset_dir)
    old_points = int(previous_counts.get("points") or 0)
    old_cameras = int(previous_counts.get("cameras") or 0)
    if previous and counts["points"] < old_points and counts["cameras"] < old_cameras:
        append_log(
            "map",
            (
                "New reconstruction was weaker "
                f"({counts['points']} points/{counts['cameras']} cameras) than current map "
                f"({old_points} points/{old_cameras} cameras); keeping the current viewer map."
            ),
        )
        if asset_backup and asset_backup.exists():
            if asset_dir.exists():
                shutil.rmtree(asset_dir)
            shutil.copytree(asset_backup, asset_dir)
        if colmap_backup and colmap_backup.exists():
            if colmap_out.exists():
                shutil.rmtree(colmap_out)
            shutil.copytree(colmap_backup, colmap_out)
        counts = read_counts(asset_dir)
        set_job(
            "map",
            "done",
            (
                f"Enhancement frames were saved, but COLMAP produced a smaller map. "
                f"Kept existing {title}: {counts['points']} points, {counts['cameras']} cameras."
            ),
        )
        return
    add_or_update_map(
        {
            "id": map_id,
            "title": title,
            "description": f"COLMAP point-cloud map from {source_label}.",
            "asset_base": public_rel(asset_dir),
            "frames_path": str(frames),
            "deletable": bool(previous.get("deletable", True)) if previous else True,
            "kind": "map",
            "created_at": previous.get("created_at") if previous else time.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "counts": counts,
            "validation": validation,
            "display_z_sign": int(previous.get("display_z_sign", -1)) if previous else -1,
            "source_map_id": previous.get("source_map_id") if previous else None,
            "localization_map_id": previous.get("localization_map_id") if previous else None,
            "has_drone_demo": counts["poses"] > 0,
            "replays": previous.get("replays", []) if previous else [],
            "active_replay_id": previous.get("active_replay_id") if previous else None,
        },
        select=True,
    )
    set_job("map", "done", f"3D map rebuilt: {title}. Upload a drone video for this map to compute TSolve replay poses.")


def resize_frame(frame, max_size: int):
    h, w = frame.shape[:2]
    scale = min(1.0, max_size / max(h, w))
    if scale >= 1.0:
        return frame
    import cv2

    return cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)


def capture_live_map_frames(frames: Path, duration: float, fps: float, camera_index: int) -> int:
    import cv2

    cfg = load_config()
    frames.mkdir(parents=True, exist_ok=True)
    for old in frames.glob("webcam_map_*.jpg"):
        old.unlink()

    live_dir = VIEWER / "public" / "live_mapping"
    live_dir.mkdir(parents=True, exist_ok=True)
    latest = live_dir / "latest.jpg"
    if latest.exists():
        latest.unlink()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}.")

    interval = 1.0 / max(float(fps), 0.1)
    start = time.perf_counter()
    next_capture = start
    count = 0
    MAP_STOP_EVENT.clear()
    set_map_capture_state(
        live_preview="public/live_mapping/latest.jpg",
        frames_saved=0,
        capture_started_at=time.time(),
    )
    append_log("map", f"Live camera opened. Capturing at {fps:.2f} fps; press Stop Mapping to reconstruct early.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.03)
                if time.perf_counter() - start > 2.0 and count == 0:
                    raise RuntimeError("Camera opened but no frames were received.")
                continue

            now = time.perf_counter()
            should_save = now >= next_capture
            preview = resize_frame(frame, min(int(cfg["max_image_size"]), 900))
            cv2.imwrite(str(latest), preview, [int(cv2.IMWRITE_JPEG_QUALITY), 82])

            if should_save:
                image = resize_frame(frame, int(cfg["max_image_size"]))
                out = frames / f"webcam_map_{count:05d}.jpg"
                cv2.imwrite(str(out), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                count += 1
                set_map_capture_state(frames_saved=count)
                if count == 1 or count % 5 == 0:
                    append_log("map", f"captured {count} mapping frames")
                next_capture = now + interval

            if MAP_STOP_EVENT.is_set():
                append_log("map", "Stop Mapping pressed. Closing camera and reconstructing from saved frames.")
                break
            if now - start >= duration:
                append_log("map", f"Live capture duration reached ({duration:.1f}s).")
                break
            time.sleep(0.01)
    finally:
        cap.release()

    set_map_capture_state(frames_saved=count)
    return count


def live_map_job(duration: float, fps: float, camera_index: int) -> None:
    try:
        map_id = make_map_id("live_map")
        frames = ROOT / "data" / "maps" / map_id / "frames"
        colmap_out = ROOT / "results" / "maps" / map_id / "colmap"
        asset_dir = MAPS_DIR / map_id
        set_job("map", "running", "Capturing live camera frames.")
        frame_count = capture_live_map_frames(frames, duration, fps, camera_index)
        if frame_count < 5:
            raise RuntimeError(f"Only {frame_count} frames captured. Move the camera longer before stopping.")
        set_job("map", "running", f"Captured {frame_count} frames. Running COLMAP sparse reconstruction.")
        run_map_from_frames(frames, colmap_out, asset_dir, map_id, f"Live Camera Map {time.strftime('%H:%M:%S')}", "live camera capture")
    except Exception as exc:
        append_log("map", f"ERROR: {exc}")
        if MAP_STOP_EVENT.is_set():
            set_job("map", "cancelled", "Map creation stopped by user.")
            MAP_STOP_EVENT.clear()
        else:
            set_job("map", "error", str(exc))


def upload_map_job(videos: list[Path]) -> None:
    try:
        label = ", ".join(video.name for video in videos[:3])
        if len(videos) > 3:
            label += f", +{len(videos) - 3} more"
        run_map_from_videos(videos, label)
    except Exception as exc:
        append_log("map", f"ERROR: {exc}")
        if MAP_STOP_EVENT.is_set():
            set_job("map", "cancelled", "Map creation stopped by user.")
            MAP_STOP_EVENT.clear()
        else:
            set_job("map", "error", str(exc))


def room_alignment_matrix_from_scene(scene_json: Path, display_z_sign: float) -> dict | None:
    transform = build_room_transform_from_scene(scene_json, display_z_sign)
    if transform is None:
        return None
    origin = transform([0.0, 0.0, 0.0])
    if origin is None:
        return None
    columns = []
    for axis in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]):
        value = transform(axis)
        if value is None:
            return None
        columns.append([value[index] - origin[index] for index in range(3)])
    matrix = [
        [columns[0][row], columns[1][row], columns[2][row], origin[row]]
        for row in range(3)
    ]
    return {
        "matrix": matrix,
        "method": "fixed-reference-room-frame",
    }


def enhance_fixed_reference_map_job(selected: dict, videos: list[Path]) -> None:
    cfg = load_config()
    py = Path(cfg["python"])
    scripts = ROOT / "scripts"
    out_map_id = str(selected["id"])
    source_map_id = str(selected.get("localization_map_id") or selected.get("source_map_id") or "")
    if not source_map_id or source_map_id == out_map_id:
        raise RuntimeError("Fixed-reference enhancement requires a duplicate linked to its original map.")
    source = get_map_entry(source_map_id)
    reference_artifacts = colmap_artifacts_for_entry(source)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    work_root = ROOT / "data" / "maps" / out_map_id / f"fixed_enhance_{timestamp}_{uuid.uuid4().hex[:6]}"
    extracted = work_root / "new_frames"
    staging_colmap = ROOT / "results" / "maps" / out_map_id / f"colmap_fixed_staging_{timestamp}_{uuid.uuid4().hex[:6]}"
    staging_asset = MAPS_DIR / f".{out_map_id}.fixed_staging_{uuid.uuid4().hex[:6]}"
    final_colmap = ROOT / "results" / "maps" / out_map_id / "colmap"
    final_asset = VIEWER / str(selected["asset_base"])
    extracted.mkdir(parents=True, exist_ok=True)
    MAP_STOP_EVENT.clear()

    set_job(
        "map",
        "running",
        f"Extracting {len(videos)} enhancement video{'' if len(videos) == 1 else 's'} for fixed registration into {source['title']}.",
    )
    for index, video in enumerate(videos):
        prefix = f"fixed_{time.strftime('%H%M%S')}_{index:02d}_{uuid.uuid4().hex[:6]}"
        video_frames = work_root / f"{prefix}_frames"
        run_cmd(
            "map",
            [
                py,
                scripts / "extract_frames.py",
                "--video",
                video,
                "--out-dir",
                video_frames,
                "--fps",
                cfg["map_frame_fps"],
                "--max-size",
                cfg["max_image_size"],
                "--prefix",
                prefix,
            ],
            MAP_STOP_EVENT,
        )
        copy_frame_bank(video_frames, extracted)
    new_frame_count = count_frame_images(extracted)
    if new_frame_count < 8:
        raise RuntimeError(f"Only {new_frame_count} enhancement frames were extracted.")

    set_job(
        "map",
        "running",
        f"Registering {new_frame_count} new frames into fixed {source['title']} coordinates. Existing cameras are locked.",
    )
    run_cmd(
        "map",
        [
            py,
            scripts / "enhance_colmap_fixed_reference.py",
            "--colmap",
            cfg["colmap_bin"],
            "--reference-colmap",
            reference_artifacts["root"],
            "--new-frames",
            extracted,
            "--out-dir",
            staging_colmap,
            "--max-image-size",
            cfg["max_image_size"],
        ],
        MAP_STOP_EVENT,
    )
    summary_path = staging_colmap / "fixed_reference_summary.json"
    if not summary_path.exists():
        raise RuntimeError("Fixed-reference enhancement finished without a validation summary.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("coordinate_frame_preserved"):
        raise RuntimeError("Fixed-reference enhancement did not confirm coordinate preservation.")

    set_job("map", "running", "Building and validating the fixed-reference ATLAS viewer map.")
    run_cmd(
        "map",
        [
            py,
            scripts / "build_map_only_viewer_data.py",
            "--model-text",
            staging_colmap / "sparse_text",
            "--out-public",
            staging_asset,
        ],
        MAP_STOP_EVENT,
    )
    validation_path = staging_asset / "map_validation.json"
    run_cmd(
        "map",
        [
            py,
            scripts / "atlas_map_validation.py",
            "--frames",
            staging_colmap / "images",
            "--colmap-root",
            staging_colmap,
            "--asset-dir",
            staging_asset,
            "--matcher",
            "fixed_reference",
            "--out",
            validation_path,
        ],
        MAP_STOP_EVENT,
    )
    counts = read_counts(staging_asset)
    reference_points = int(summary.get("reference_points") or source.get("counts", {}).get("points") or 0)
    reference_cameras = int(summary.get("reference_registered_images") or source.get("counts", {}).get("cameras") or 0)
    if counts["points"] < reference_points or counts["cameras"] < reference_cameras:
        raise RuntimeError(
            "Fixed-reference output was weaker than the source map; duplicate was not replaced "
            f"({counts['points']} points/{counts['cameras']} cameras)."
        )

    result_backup = None
    asset_backup = None
    try:
        if final_colmap.exists():
            result_backup = final_colmap.with_name(f"colmap_before_fixed_{timestamp}")
            shutil.move(str(final_colmap), str(result_backup))
        shutil.move(str(staging_colmap), str(final_colmap))
        if final_asset.exists():
            asset_backup = final_asset.with_name(f"{final_asset.name}_before_fixed_{timestamp}")
            shutil.move(str(final_asset), str(asset_backup))
        shutil.move(str(staging_asset), str(final_asset))
    except Exception:
        if not final_colmap.exists() and result_backup and result_backup.exists():
            shutil.move(str(result_backup), str(final_colmap))
        if not final_asset.exists() and asset_backup and asset_backup.exists():
            shutil.move(str(asset_backup), str(final_asset))
        raise

    validation = json.loads((final_asset / "map_validation.json").read_text(encoding="utf-8"))
    reference_asset = map_asset_dir(source)
    room_alignment = source.get("room_alignment")
    if not room_alignment:
        room_alignment = room_alignment_matrix_from_scene(
            reference_asset / "scene.json",
            float(source.get("display_z_sign", -1)),
        )
    updated = dict(selected)
    updated.update(
        {
            "description": (
                f"Fixed-reference enhancement of {source['title']}: "
                f"{summary.get('registered_new_images', 0)}/{summary.get('new_frame_count', 0)} new frames registered, "
                f"{summary.get('added_points', 0)} points added."
            ),
            "frames_path": str(final_colmap / "images"),
            "kind": "fixed_reference_map",
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "counts": counts,
            "validation": validation,
            "source_map_id": source["id"],
            "localization_map_id": out_map_id,
            "fixed_reference_summary": summary,
            "room_alignment": room_alignment,
        }
    )
    add_or_update_map(updated, select=True)
    set_job(
        "map",
        "done",
        (
            f"Fixed-reference enhancement complete: {counts['points']} points, {counts['cameras']} cameras. "
            f"Original {source['title']} was not modified."
        ),
    )


def enhance_map_job(map_id: str, videos: list[Path]) -> None:
    try:
        if not videos:
            raise RuntimeError("No enhancement videos were provided.")
        cfg = load_config()
        py = Path(cfg["python"])
        scripts = ROOT / "scripts"
        selected = get_map_entry(map_id)
        if (
            selected.get("kind") == "map_copy"
            and (selected.get("localization_map_id") or selected.get("source_map_id"))
        ):
            enhance_fixed_reference_map_job(selected, videos)
            return
        frames = frames_for_entry(selected)
        out_map_id = selected["id"]
        out_title = selected["title"]
        asset_dir = VIEWER / selected["asset_base"]
        colmap_out = ROOT / "results" / "maps" / out_map_id / "colmap"
        MAP_STOP_EVENT.clear()
        set_selected_map(selected["id"])
        set_job("map", "running", f"Adding {len(videos)} mapping video{'' if len(videos) == 1 else 's'} to {selected['title']} and rerunning COLMAP on the combined frame bank.")
        backup = backup_frame_bank(frames)
        temp_dirs: list[Path] = []
        for idx, video in enumerate(videos):
            prefix = f"enhance_{time.strftime('%H%M%S')}_{idx:02d}_{uuid.uuid4().hex[:6]}"
            temp_frames = ROOT / "data" / "maps" / out_map_id / f"{prefix}_frames"
            temp_dirs.append(temp_frames)
            set_job("map", "running", f"Extracting enhancement video {idx + 1}/{len(videos)}: {video.name}")
            run_cmd(
                "map",
                [
                    py,
                    scripts / "extract_frames.py",
                    "--video",
                    video,
                    "--out-dir",
                    temp_frames,
                    "--fps",
                    cfg["map_frame_fps"],
                    "--max-size",
                    cfg["max_image_size"],
                    "--prefix",
                    prefix,
                ],
                MAP_STOP_EVENT,
            )
        restore_frame_bank(backup, frames)
        for temp_frames in temp_dirs:
            copy_frame_bank(temp_frames, frames)
        append_log("map", f"Combined frame bank now has {count_frame_images(frames)} images.")
        source_names = ", ".join(video.name for video in videos[:3])
        if len(videos) > 3:
            source_names += f", +{len(videos) - 3} more"
        run_map_from_frames(
            frames,
            colmap_out,
            asset_dir,
            out_map_id,
            out_title,
            f"{selected['title']} frame bank plus {source_names}",
        )
    except Exception as exc:
        append_log("map", f"ERROR: {exc}")
        if MAP_STOP_EVENT.is_set():
            set_job("map", "cancelled", "Map enhancement stopped by user.")
            MAP_STOP_EVENT.clear()
        else:
            set_job("map", "error", str(exc))


def replay_query_frame_dir(replay: dict) -> Path | None:
    raw = str(replay.get("query_frame_base_url") or replay.get("frame_source") or "").strip()
    if not raw:
        return None
    if raw.startswith("public/") or raw.startswith("maps/"):
        candidate = VIEWER / raw
    else:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = VIEWER / raw
    return candidate if candidate.exists() and candidate.is_dir() else None


def enhance_map_from_replay_job(map_id: str, replay_id: str) -> None:
    try:
        selected = get_map_entry(map_id)
        replay = next((r for r in selected.get("replays", []) if r.get("id") == replay_id), None)
        if replay is None:
            raise RuntimeError(f"Unknown replay id for {selected.get('title') or map_id}: {replay_id}")

        replay_title = str(replay.get("title") or replay_id)
        replay_asset = VIEWER / str(replay.get("asset_base") or "")
        replay_video = replay_asset / "media" / "drone_query.mp4"
        if replay_video.exists():
            set_job("map", "running", f"Enhancing {selected['title']} with source video from {replay_title}.")
            enhance_map_job(map_id, [replay_video])
            return

        src_frames = replay_query_frame_dir(replay)
        if src_frames is None or count_frame_images(src_frames) < 5:
            raise RuntimeError("This path has no saved video or enough saved query frames to enhance the 3D map.")

        frames = frames_for_entry(selected)
        asset_dir = VIEWER / selected["asset_base"]
        colmap_out = ROOT / "results" / "maps" / selected["id"] / "colmap"
        safe_prefix = f"replay_{replay_id.replace('-', '_')[:24]}"
        MAP_STOP_EVENT.clear()
        set_selected_map(selected["id"])
        set_job(
            "map",
            "running",
            f"Adding {count_frame_images(src_frames)} saved DJI frames from {replay_title} to {selected['title']} and rerunning COLMAP.",
        )
        backup = backup_frame_bank(frames)
        restore_frame_bank(backup, frames)
        copy_frame_bank(src_frames, frames, prefix=safe_prefix)
        append_log("map", f"Combined frame bank now has {count_frame_images(frames)} images.")
        run_map_from_frames(
            frames,
            colmap_out,
            asset_dir,
            selected["id"],
            selected["title"],
            f"{selected['title']} frame bank plus saved DJI frames from {replay_title}",
        )
    except Exception as exc:
        append_log("map", f"ERROR: {exc}")
        if MAP_STOP_EVENT.is_set():
            set_job("map", "cancelled", "Map enhancement from drone path stopped by user.")
            MAP_STOP_EVENT.clear()
        else:
            set_job("map", "error", str(exc))


def dji_live_atlas_job(
    *,
    map_id: str | None = None,
    phone_ip: str = "",
    fps: float = 10.0,
    max_size: int = 1200,
    view_only: bool = False,
) -> None:
    # The request handler reserves this worker before starting the thread.
    # Do not clear DRONE_STOP_EVENT here: Stop may have been pressed during
    # the small interval between reservation and thread startup.
    mark_drone_job_worker_started()
    bridge_thread: threading.Thread | None = None
    monitor_thread: threading.Thread | None = None
    monitor_stop = threading.Event()
    try:
        if not phone_ip.strip():
            raise RuntimeError("Missing Android phone IP for DJI live ATLAS.")
        cfg = load_config()
        py = Path(cfg["python"])
        scripts = ROOT / "scripts"
        library = load_library()
        live_patrol_lock = resolved_live_patrol_lock(library)
        view_only = bool(
            view_only
            or (live_patrol_lock is not None and not live_patrol_lock["flight_enabled"])
        )
        effective_map_id = live_patrol_lock["map_id"] if live_patrol_lock is not None else map_id
        selected = set_selected_map(effective_map_id) if effective_map_id else selected_map_entry()
        map_artifacts = colmap_artifacts_for_entry(selected)
        replay_id = make_map_id("dji_live")
        session = f"atlas_{replay_id}"
        replay_title = f"Live ATLAS {time.strftime('%H:%M:%S')}"
        base_asset_dir = VIEWER / selected["asset_base"]
        if not base_asset_dir.exists():
            base_asset_dir = MAPS_DIR / selected["id"]
        out_asset_dir = base_asset_dir / "replays" / replay_id
        out_asset_dir.mkdir(parents=True, exist_ok=True)
        partial_pose_path = out_asset_dir / "poses_partial.json"
        stop_file = out_asset_dir / "STOP_LIVE_ATLAS"
        run_root = ROOT / "results" / "dji_live_runs" / selected["id"] / replay_id
        runtime_dir = run_root / "tsolve_runtime_code"
        tsolve_runtime = run_root / "tsolve_runtime"
        tsolve_inputs = run_root / "tsolve_inputs"
        stream_work = run_root / "live_existing_map_stream"
        public_live_root = PUBLIC / "live_dji_sessions"
        query_frames = public_live_root / session / "query_frames"
        live_latest = PUBLIC / "live_dji" / "latest.jpg"
        live_started_at = time.time()

        atomic_write_json(
            partial_pose_path,
            {
                "mode": "dji_live_tsolve_partial",
                "replay_id": replay_id,
                "frame_source": str(query_frames),
                "query_frame_base_url": public_rel(query_frames),
                "expected_count": 0,
                "processed_count": 0,
                "complete": False,
                "updated_at": time.time(),
                "poses": [],
            },
        )
        set_live_stream(
            {
                "live_atlas": True,
                "map_id": selected["id"],
                "replay_id": replay_id,
                "title": replay_title,
                "asset_base": public_rel(out_asset_dir),
                "partial_pose_url": public_rel(partial_pose_path),
                "stop_file": str(stop_file),
                "live_preview_url": "public/live_dji/latest.jpg",
                "query_frame_base_url": public_rel(query_frames),
                "pose_count": 0,
                "expected_count": 0,
                "complete": False,
                "started_at": live_started_at,
                "first_pose_at": None,
                "first_pose_latency_seconds": None,
                "phone_ip": phone_ip.strip(),
                "session": session,
                "view_only": view_only,
            }
        )

        set_job("drone", "running", f"Starting DJI live bridge from Android phone {phone_ip.strip()}.")
        bridge_cmd = [
            py,
            scripts / "atlas_dji_live_bridge.py",
            "--phone-ip",
            phone_ip.strip(),
            "--fps",
            fps,
            "--max-size",
            max_size,
            "--session",
            session,
            "--out-root",
            public_live_root,
            "--public-root",
            PUBLIC / "live_dji",
            "--pose-stream",
            partial_pose_path,
        ]
        if view_only:
            bridge_cmd.append("--view-only")
        enemy_model = selected_enemy_model_path()
        if enemy_model:
            bridge_cmd.extend(
                [
                    "--enemy-model",
                    enemy_model,
                    "--enemy-output",
                    PUBLIC / "live_dji" / "enemy_detections.json",
                    "--enemy-detect-fps",
                    cfg.get("enemy_live_detect_fps", min(float(fps), 5.0)),
                    "--enemy-conf",
                    cfg.get("enemy_live_confidence", 0.35),
                ]
            )
            append_log("drone", f"Enemy-drone detector enabled: {enemy_model.name}")
        else:
            append_log("drone", "Enemy-drone detector disabled: no trained YOLO model selected.")
        bridge_thread = start_logged_background_cmd("drone", bridge_cmd, DRONE_STOP_EVENT)

        set_job("drone", "running", "Preparing TSolve runtime while DJI frames start streaming.")
        run_cmd(
            "drone",
            [
                py,
                scripts / "setup_tsolve_runtime.py",
                "--base-yam-code-dir",
                cfg["base_yam_code_dir"],
                "--dropin-patch-dir",
                cfg["dropin_patch_dir"],
                "--base-harness-dir",
                str(ROOT.parent / "pnp-symbolic-research/Yam/exact_ff_ysolve_pnp/harness"),
                "--out-dir",
                runtime_dir,
            ],
            DRONE_STOP_EVENT,
        )

        # A single OpenDJI frame is not proof of a live video stream: getFrame()
        # retains its last ndarray after the Android decoder stalls.  The bridge
        # now saves decoder callbacks only, and this warm-up requires enough
        # distinct callbacks to prove that frames are genuinely advancing before
        # TSolve or any flight command can use them.
        required_fresh_live_frames = 8
        wait_started = time.time()
        while (
            not DRONE_STOP_EVENT.is_set()
            and count_frame_images(query_frames) < required_fresh_live_frames
        ):
            if live_latest.exists():
                update_live_stream(live_preview_url="public/live_dji/latest.jpg")
            if time.time() - wait_started > 60:
                raise RuntimeError(
                    "DJI bridge connected, but the Android video stream did not produce "
                    f"{required_fresh_live_frames} fresh decoded frames within 60 seconds."
                )
            set_job(
                "drone",
                "running",
                (
                    "Waiting for a continuously updating DJI video stream "
                    f"({count_frame_images(query_frames)}/{required_fresh_live_frames} fresh frames)."
                ),
            )
            time.sleep(0.5)

        if DRONE_STOP_EVENT.is_set():
            raise RuntimeError("Live ATLAS localization stopped before first frame.")

        set_job(
            "drone",
            "running",
            "Fresh DJI video confirmed. Running TSolve live self-localization.",
        )
        monitor_thread = threading.Thread(
            target=monitor_partial_pose_file,
            args=(partial_pose_path, monitor_stop, live_started_at, tsolve_runtime / "live_stage_times.csv"),
            daemon=True,
        )
        monitor_thread.start()

        live_stream_cmd = [
            py,
            scripts / "run_bounded_tsolve_video_stream.py",
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
            runtime_dir,
            "--solver-dir",
            cfg["solver_dir"],
            "--inputs-out-dir",
            tsolve_inputs,
            "--out-dir",
            tsolve_runtime,
            "--work-dir",
            stream_work,
            "--max-image-size",
            cfg["max_image_size"],
            "--query-camera-model",
            cfg["query_camera_model"],
            "--min-points",
            cfg["min_query_correspondences"],
            "--max-points",
            cfg["max_query_correspondences"],
            "--max-reference-images",
            cfg.get("live_reference_image_cap", 24),
            "--tracking-reference-images",
            cfg.get("live_tracking_reference_image_cap", 10),
            "--track-pool-size",
            cfg.get("live_tracking_pool_size", 900),
            "--relocalize-every",
            cfg.get("live_relocalize_every", 0),
            "--flow-max-error",
            cfg.get("live_flow_max_error", 34.0),
            "--flow-backtrack-error",
            cfg.get("live_flow_backtrack_error", 2.5),
            "--flow-window",
            cfg.get("live_flow_window", 21),
            "--flow-levels",
            cfg.get("live_flow_levels", 3),
            "--flow-iterations",
            cfg.get("live_flow_iterations", 18),
            "--min-track-points",
            cfg.get("live_min_track_points", 80),
            "--min-track-ratio",
            cfg.get("live_min_track_ratio", 0.10),
            "--proactive-relocalize-points",
            cfg.get("live_proactive_relocalize_points", 500),
            "--proactive-relocalize-cooldown-frames",
            cfg.get("live_proactive_relocalize_cooldown_frames", 60),
            "--global-recovery-after-failures",
            cfg.get("live_global_recovery_after_failures", 2),
            "--background-recovery-timeout-seconds",
            cfg.get("live_background_recovery_timeout_seconds", 20.0),
            "--global-recovery-max-step",
            cfg.get("live_global_recovery_max_step", 0.55),
            "--output-max-step",
            cfg.get("live_output_max_step", 0.55),
            "--output-objective-threshold",
            cfg.get("live_output_objective_threshold", 30.0),
            "--prime",
            cfg["tsolve_prime"],
            "--degree",
            cfg["tsolve_degree"],
            "--action-weights",
            cfg["tsolve_action_weights"],
            "--tsolve-root-profile",
            cfg.get("tsolve_root_profile", "full"),
            "--fallback-action-weights",
            cfg["tsolve_fallback_action_weights"],
            "--partial-pose-out",
            partial_pose_path,
            "--replay-id",
            replay_id,
            "--expected-count",
            0,
            "--scene-json",
            base_asset_dir / "scene.json",
            "--display-z-sign",
            selected.get("display_z_sign", -1),
            "--room-alignment-json",
            json.dumps(selected.get("room_alignment") or {}),
            "--follow-dir",
            "--stop-file",
            stop_file,
        ]
        if live_patrol_lock is not None and live_patrol_lock["reference_profile"] != "off":
            live_stream_cmd.extend(
                [
                    "--rotation-position-stabilizer-profile",
                    live_patrol_lock["reference_profile"],
                    "--rotation-command-status-json",
                    PUBLIC / "live_dji" / "control_status.json",
                ]
            )
        if live_patrol_lock is not None and live_patrol_lock.get("baseline_reference_path"):
            live_stream_cmd.extend(
                [
                    "--patrol-route-baseline",
                    live_patrol_lock["baseline_reference_path"],
                    "--patrol-route-max-cross-track",
                    cfg.get("live_patrol_route_max_cross_track", 0.55),
                    "--patrol-route-backward-tolerance",
                    cfg.get("live_patrol_route_backward_tolerance", 0.045),
                    "--patrol-status-max-age",
                    cfg.get("live_patrol_status_max_age", 5.0),
                    "--patrol-turn-max-position-drift",
                    cfg.get("live_patrol_turn_max_position_drift", 0.75),
                ]
            )
            if live_patrol_lock.get("visual_recovery_path"):
                live_stream_cmd.extend(
                    [
                        "--patrol-visual-recovery-bank",
                        live_patrol_lock["visual_recovery_path"],
                    ]
                )
        for recovery_bank in active_taught_recovery_banks(base_asset_dir):
            live_stream_cmd.extend(["--taught-patrol-recovery-bank", recovery_bank])
        if cfg.get("live_blocking_global_recovery", True):
            live_stream_cmd.append("--blocking-global-recovery")
        if not cfg.get("live_background_recovery", False):
            live_stream_cmd.append("--disable-background-recovery")
        if cfg.get("live_direct_pnp_recovery", True):
            live_stream_cmd.append("--direct-pnp-recovery")
        run_cmd("drone", live_stream_cmd, None)

        set_job("drone", "running", "Finalizing stopped DJI live path.")
        pose_count = finalize_partial_replay(
            selected=selected,
            replay_id=replay_id,
            replay_title=replay_title,
            out_asset_dir=out_asset_dir,
            partial_pose_path=partial_pose_path,
            source_video="DJI MSDK live stream",
            query_frame_base_url=public_rel(query_frames),
        )
        set_job("drone", "done", f"Live ATLAS stopped and saved: {replay_title} ({pose_count} poses).")
    except Exception as exc:
        append_log("drone", f"ERROR: {exc}")
        stream = current_live_stream() or {}
        if DRONE_STOP_EVENT.is_set() and stream.get("live_atlas"):
            try:
                selected = set_selected_map(stream.get("map_id") or map_id or "")
                replay_id = str(stream.get("replay_id") or make_map_id("dji_live"))
                replay_title = str(stream.get("title") or f"Live ATLAS {time.strftime('%H:%M:%S')}")
                out_asset_dir = VIEWER / str(stream.get("asset_base") or "")
                partial_pose_path = VIEWER / str(stream.get("partial_pose_url") or "")
                if out_asset_dir.exists() and partial_pose_path.exists():
                    pose_count = finalize_partial_replay(
                        selected=selected,
                        replay_id=replay_id,
                        replay_title=replay_title,
                        out_asset_dir=out_asset_dir,
                        partial_pose_path=partial_pose_path,
                        source_video="DJI MSDK live stream",
                        query_frame_base_url=stream.get("query_frame_base_url"),
                    )
                    set_job("drone", "done", f"Live ATLAS stopped and saved: {replay_title} ({pose_count} poses).")
                else:
                    update_live_stream(complete=True, cancelled=True)
                    set_job("drone", "cancelled", "Live ATLAS stopped before any saved pose stream existed.")
            except Exception as final_exc:
                append_log("drone", f"FINALIZE ERROR: {final_exc}")
                update_live_stream(complete=True, cancelled=True)
                set_job("drone", "cancelled", "Live ATLAS stopped before a path could be saved.")
        else:
            update_live_stream(
                complete=True,
                cancelled=False,
                failed=True,
                stopping=False,
                live_preview_url=None,
                error=str(exc),
            )
            mark_live_dji_status_stopped(f"ATLAS live localization failed: {exc}")
            set_job("drone", "error", str(exc))
    finally:
        monitor_stop.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=2.0)
        # The bridge is a background subprocess.  It must also be closed after
        # a startup/video error, not only after an explicit Stop request.
        terminate_active_procs("drone")
        if bridge_thread is not None:
            bridge_thread.join(timeout=1.0)
        release_drone_job()


def simulated_patrol_baseline_job(
    *,
    map_id: str,
    baseline_replay_id: str,
    fps: float = 10.0,
    start_source_frame: int | None = None,
    end_source_frame: int | None = None,
    validation_only: bool = False,
    recorded_timing: bool = False,
) -> None:
    """Publish a validated recorded patrol incrementally through the live stream API."""
    mark_drone_job_worker_started()
    try:
        library = load_library()
        lock = resolved_live_patrol_lock(library)
        if lock is None:
            raise RuntimeError("The locked live patrol baseline is not available.")
        if map_id != lock["map_id"] or baseline_replay_id != lock["baseline_replay_id"]:
            raise RuntimeError("Live simulation must use the locked map and full patrol baseline.")

        selected = set_selected_map(map_id)
        baseline = next(
            (item for item in selected.get("replays", []) if item.get("id") == baseline_replay_id),
            None,
        )
        if not isinstance(baseline, dict) or baseline.get("kind") != "route_constrained_taught_baseline":
            raise RuntimeError("The selected replay is not the validated full patrol baseline.")

        baseline_asset = VIEWER / str(baseline.get("asset_base") or "")
        baseline_pose_path = baseline_asset / "poses.json"
        source_payload = json.loads(baseline_pose_path.read_text(encoding="utf-8"))
        source_poses = source_payload.get("poses")
        if not isinstance(source_poses, list) or len(source_poses) < 2:
            raise RuntimeError("The full patrol baseline has no usable recorded-frame poses.")

        if start_source_frame is not None or end_source_frame is not None:
            filtered_poses: list[dict] = []
            for pose_index, source_pose in enumerate(source_poses):
                if not isinstance(source_pose, dict):
                    continue
                try:
                    source_frame = int(source_pose.get("source_frame", source_pose.get("frame_index", pose_index)))
                except (TypeError, ValueError):
                    source_frame = pose_index
                if start_source_frame is not None and source_frame < start_source_frame:
                    continue
                if end_source_frame is not None and source_frame > end_source_frame:
                    continue
                filtered_poses.append(source_pose)
            source_poses = filtered_poses
            if len(source_poses) < 2:
                raise RuntimeError("The requested source-frame validation range has fewer than two poses.")

        fps = max(0.5, min(30.0, float(fps)))
        replay_id = make_map_id("sim_validate" if validation_only else "sim_live")
        replay_title = (
            f"Live Render Validation {time.strftime('%H:%M:%S')}"
            if validation_only
            else f"New Live Path Simulation {time.strftime('%H:%M:%S')}"
        )
        base_asset_dir = VIEWER / selected["asset_base"]
        out_asset_dir = base_asset_dir / "replays" / replay_id
        out_asset_dir.mkdir(parents=True, exist_ok=True)
        partial_pose_path = out_asset_dir / "poses_partial.json"
        final_pose_path = out_asset_dir / "poses.json"
        query_frame_base_url = str(
            baseline.get("query_frame_base_url")
            or source_payload.get("query_frame_base_url")
            or source_payload.get("frame_source")
            or ""
        ).strip()
        expected_count = len(source_poses)
        started_at = time.time()
        published: list[dict] = []

        def publish(complete: bool = False) -> None:
            latest = published[-1] if published else None
            current_frame = None
            if latest is not None:
                current_frame = {
                    "frame_index": int(latest.get("source_frame") or len(published) - 1),
                    "image_name": latest.get("image_name"),
                    "time_sec": latest.get("time_sec"),
                    "received_unix": latest.get("received_unix"),
                }
            payload = {
                "mode": "simulated_live_patrol_partial",
                "replay_id": replay_id,
                "source_replay_id": baseline_replay_id,
                "frame_source": query_frame_base_url,
                "query_frame_base_url": query_frame_base_url,
                "expected_count": expected_count,
                "start_source_frame": start_source_frame,
                "end_source_frame": end_source_frame,
                "validation_only": validation_only,
                "timing_mode": "recorded" if recorded_timing else "fixed_fps",
                "processed_count": len(published),
                "accepted_count": len(published),
                "held_count": 0,
                "failed_count": 0,
                "complete": complete,
                "updated_at": time.time(),
                "current_frame": current_frame,
                "current_frame_time_sec": current_frame.get("time_sec") if current_frame else None,
                "poses": published,
            }
            atomic_write_json(partial_pose_path, payload)
            update_live_stream(
                pose_count=len(published),
                accepted_pose_count=len(published),
                expected_count=expected_count,
                latest_frame_index=current_frame.get("frame_index") if current_frame else None,
                complete=complete,
            )

        publish(False)
        set_live_stream(
            {
                "simulated_live": True,
                "live_atlas": False,
                "map_id": selected["id"],
                "replay_id": replay_id,
                "source_replay_id": baseline_replay_id,
                "title": replay_title,
                "asset_base": public_rel(out_asset_dir),
                "partial_pose_url": public_rel(partial_pose_path),
                "query_frame_base_url": query_frame_base_url,
                "pose_count": 0,
                "accepted_pose_count": 0,
                "expected_count": expected_count,
                "complete": False,
                "started_at": started_at,
                "fps": fps,
                "view_only": True,
                "start_source_frame": start_source_frame,
                "end_source_frame": end_source_frame,
                "validation_only": validation_only,
                "timing_mode": "recorded" if recorded_timing else "fixed_fps",
            }
        )
        set_job(
            "drone",
            "running",
            (
                f"Validating the live renderer from source frame {start_source_frame} "
                f"through {end_source_frame} at {fps:g} FPS."
                if validation_only
                else (
                    "Creating a new live path at the original recorded frame timing."
                    if recorded_timing
                    else f"Creating a new live path from the validated baseline at {fps:g} FPS."
                )
            ),
        )

        batch_size = max(1, int(round(fps * 0.25)))
        clock_started = time.monotonic()
        first_recorded_time = None
        if recorded_timing:
            try:
                first_recorded_time = float(source_poses[0].get("time_sec"))
            except (AttributeError, TypeError, ValueError):
                first_recorded_time = None
        for index, source_pose in enumerate(source_poses):
            if DRONE_STOP_EVENT.is_set():
                break
            pose = dict(source_pose)
            pose["instance_id"] = f"sim_live_{index + 1:06d}"
            pose["received_unix"] = time.time()
            pose["pose_source"] = "simulated_live_recorded_baseline"
            pose["simulated_live"] = True
            published.append(pose)
            if len(published) == 1 or len(published) % batch_size == 0 or len(published) == expected_count:
                publish(False)
                set_job(
                    "drone",
                    "running",
                    f"New live path: {len(published)}/{expected_count} recorded frames localized.",
                )
            target_elapsed = (index + 1) / fps
            if first_recorded_time is not None:
                try:
                    next_pose = source_poses[min(index + 1, expected_count - 1)]
                    target_elapsed = max(0.0, float(next_pose.get("time_sec")) - first_recorded_time)
                except (AttributeError, TypeError, ValueError):
                    pass
            wait_seconds = target_elapsed - (time.monotonic() - clock_started)
            if wait_seconds > 0 and DRONE_STOP_EVENT.wait(wait_seconds):
                break

        publish(True)
        final_payload = json.loads(partial_pose_path.read_text(encoding="utf-8"))
        final_payload.update(
            {
                "mode": "simulated_live_patrol_replay",
                "description": "Recorded DJI frames published incrementally with the validated patrol baseline poses.",
                "complete": True,
                "cancelled": DRONE_STOP_EVENT.is_set(),
                "updated_at": time.time(),
            }
        )
        atomic_write_json(final_pose_path, final_payload)
        replay = {
            "id": replay_id,
            "title": replay_title,
            "asset_base": public_rel(out_asset_dir),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_video": "Validated Full Patrol Baseline 15:47:14 live simulation",
            "source_replay_id": baseline_replay_id,
            "kind": "simulated_live_patrol_baseline",
            "counts": {
                "poses": len(published),
                "accepted": len(published),
                "frames": len(published),
                "held": 0,
                "failed": 0,
            },
            "query_frame_base_url": query_frame_base_url,
        }
        if not validation_only:
            add_replay_to_map(selected["id"], replay, select=True)
        update_live_stream(
            pose_count=len(published),
            accepted_pose_count=len(published),
            partial_pose_url=public_rel(final_pose_path),
            final_pose_url=public_rel(final_pose_path),
            complete=True,
            cancelled=DRONE_STOP_EVENT.is_set(),
        )
        if DRONE_STOP_EVENT.is_set():
            set_job("drone", "cancelled", f"Live path simulation stopped after {len(published)} frames.")
        elif validation_only:
            set_job("drone", "done", f"Live renderer validation complete: {len(published)} frames.")
        else:
            set_job("drone", "done", f"New live path simulation complete: {len(published)} frames.")
    except Exception as exc:
        append_log("drone", f"SIMULATION ERROR: {exc}")
        update_live_stream(complete=True, failed=True, error=str(exc))
        set_job("drone", "error", str(exc))
    finally:
        release_drone_job()


def build_recorded_patrol_lap_sequence(
    *,
    source_frame_dir: Path,
    output_frame_dir: Path,
    start_frame: int,
    loop_return_frame: int,
    next_departure_frame: int,
    laps: int,
    phase_boundaries: dict | None = None,
    source_replay_id: str = "",
    second_lap_segments: list[dict] | None = None,
) -> list[dict]:
    """Create one continuous query bank for a two-lap live simulation.

    Non-final laps include the recorded Point-1 turn toward Point 2.  The next
    lap may either repeat the original recording or use audited segments from
    other recordings. Hard links avoid duplicating JPEG data while giving the
    localizer a monotonic frame index and one continuous clock.
    """
    laps = max(1, min(2, int(laps)))
    if not (start_frame < loop_return_frame <= next_departure_frame):
        raise RuntimeError("Recorded two-lap boundaries are not ordered.")

    baseline_boundaries = dict(phase_boundaries or {})
    baseline_boundaries.setdefault("point1", start_frame)
    baseline_boundaries.setdefault("point1_return", loop_return_frame)
    baseline_boundaries.setdefault("point1_next_departure", next_departure_frame)

    segments: list[dict] = [
        {
            "lap": 1,
            "source_frame_dir": source_frame_dir,
            "source_replay_id": source_replay_id,
            "start_frame": start_frame,
            "end_frame": next_departure_frame if laps > 1 else loop_return_frame,
            "phase_boundaries": baseline_boundaries,
            "label": "validated baseline lap 1",
        }
    ]
    if laps > 1:
        if second_lap_segments:
            for raw_segment in second_lap_segments:
                segment = dict(raw_segment)
                segment["lap"] = 2
                segments.append(segment)
        else:
            segments.append(
                {
                    "lap": 2,
                    "source_frame_dir": source_frame_dir,
                    "source_replay_id": source_replay_id,
                    "start_frame": start_frame,
                    "end_frame": loop_return_frame,
                    "phase_boundaries": baseline_boundaries,
                    "label": "repeated validated baseline lap 2",
                }
            )

    def route_state(frame_index: int, boundaries: dict) -> tuple[int, bool, str]:
        p2_arrival = int(boundaries.get("point2_arrival") or start_frame)
        p2_departure = int(boundaries.get("point2_departure") or p2_arrival)
        p3_arrival = int(boundaries.get("point3_arrival") or p2_departure)
        p3_departure = int(boundaries.get("point3_departure") or p3_arrival)
        p4_arrival = int(boundaries.get("point4_arrival") or p3_departure)
        p4_departure = int(boundaries.get("point4_departure") or p4_arrival)
        p1_return = int(boundaries.get("point1_return") or loop_return_frame)
        if frame_index < p2_arrival:
            return 0, False, "point1_to_point2"
        if frame_index < p2_departure:
            return 1, True, "turn_at_point2"
        if frame_index < p3_arrival:
            return 1, False, "point2_to_point3"
        if frame_index < p3_departure:
            return 2, True, "turn_at_point3"
        if frame_index < p4_arrival:
            return 2, False, "point3_to_point4"
        if frame_index < p4_departure:
            return 3, True, "turn_at_point4"
        if frame_index < p1_return:
            return 3, False, "point4_to_point1"
        return 0, True, "turn_at_point1"

    csv_rows_by_dir: dict[str, dict[int, dict]] = {}

    def rows_for(frame_dir: Path) -> dict[int, dict]:
        key = str(frame_dir.resolve())
        if key in csv_rows_by_dir:
            return csv_rows_by_dir[key]
        rows: dict[int, dict] = {}
        with (frame_dir / "frames.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    rows[int(row["source_frame"])] = row
                except (KeyError, TypeError, ValueError):
                    continue
        csv_rows_by_dir[key] = rows
        return rows

    output_frame_dir.mkdir(parents=True, exist_ok=False)
    plan: list[dict] = []
    sequence_time = 0.0
    default_delta = 0.1
    csv_path = output_frame_dir / "frames.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "image_name",
            "source_frame",
            "time_sec",
            "width",
            "height",
            "received_unix",
            "recorded_source_frame",
            "recorded_source_replay_id",
            "lap",
            "route_leg_index",
            "translation_locked",
            "route_phase",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for segment_index, segment in enumerate(segments):
            frame_dir = Path(segment["source_frame_dir"])
            rows = rows_for(frame_dir)
            segment_start = int(segment["start_frame"])
            segment_end = int(segment["end_frame"])
            if segment_start > segment_end:
                raise RuntimeError("Composite patrol segment boundaries are reversed.")
            segment_boundaries = dict(segment.get("phase_boundaries") or baseline_boundaries)
            segment_source_id = str(segment.get("source_replay_id") or "")
            lap_number = int(segment.get("lap") or 1)
            previous_source_time = None
            for recorded_index in range(segment_start, segment_end + 1):
                row = rows.get(recorded_index)
                source_path = frame_dir / f"query_{recorded_index:06d}.jpg"
                if row is None or not source_path.is_file():
                    raise RuntimeError(
                        f"Composite patrol source frame {recorded_index} is missing from {frame_dir}."
                    )
                try:
                    source_time = float(row.get("time_sec"))
                except (TypeError, ValueError):
                    source_time = None
                if plan:
                    delta = default_delta
                    if (
                        source_time is not None
                        and previous_source_time is not None
                        and 0.02 <= source_time - previous_source_time <= 0.5
                    ):
                        delta = source_time - previous_source_time
                    sequence_time += delta
                sequence_index = len(plan)
                target_name = f"query_{sequence_index:06d}.jpg"
                target_path = output_frame_dir / target_name
                try:
                    os.link(source_path, target_path)
                except OSError:
                    # Symlinks are still zero-copy and SimpleHTTPRequestHandler
                    # resolves them inside this app-owned replay directory.
                    target_path.symlink_to(source_path)
                leg_index, translation_locked, route_phase = route_state(
                    recorded_index,
                    segment_boundaries,
                )
                plan.append(
                    {
                        "sequence_frame": sequence_index,
                        "recorded_source_frame": recorded_index,
                        "recorded_source_replay_id": segment_source_id,
                        "lap": lap_number,
                        "segment_index": segment_index,
                        "route_leg_index": leg_index,
                        "translation_locked": translation_locked,
                        "route_phase": route_phase,
                    }
                )
                writer.writerow(
                    {
                        "image_name": target_name,
                        "source_frame": sequence_index,
                        "time_sec": f"{sequence_time:.6f}",
                        "width": row.get("width") or "1200",
                        "height": row.get("height") or "675",
                        "received_unix": f"{time.time() + sequence_time:.6f}",
                        "recorded_source_frame": recorded_index,
                        "recorded_source_replay_id": segment_source_id,
                        "lap": lap_number,
                        "route_leg_index": leg_index,
                        "translation_locked": str(translation_locked).lower(),
                        "route_phase": route_phase,
                    }
                )
                previous_source_time = source_time
    atomic_write_json(
        output_frame_dir / "frame_plan.json",
        {
            "kind": "atlas_recorded_two_lap_frame_plan",
            "laps": laps,
            "start_frame": start_frame,
            "loop_return_frame": loop_return_frame,
            "next_departure_frame": next_departure_frame,
            "composite_second_lap": bool(second_lap_segments),
            "source_replay_ids": list(
                dict.fromkeys(str(segment.get("source_replay_id") or "") for segment in segments)
            ),
            "frames": plan,
        },
    )
    return plan


def localized_recorded_patrol_job(
    *,
    map_id: str,
    baseline_replay_id: str,
    fps: float = 10.0,
    start_source_frame: int | None = None,
    end_source_frame: int | None = None,
    validation_only: bool = False,
    recorded_timing: bool = False,
    laps: int = 1,
) -> None:
    """Localize the full recorded patrol frames through the live pipeline.

    Unlike ``simulated_patrol_baseline_job``, this never copies a baseline
    pose.  The baseline supplies only the source-frame interval and camera
    images; COLMAP bootstrap, optical flow and TSolve create a new pose stream.
    """
    mark_drone_job_worker_started()
    monitor_stop = threading.Event()
    monitor_thread: threading.Thread | None = None
    controller_status_stop = threading.Event()
    controller_status_thread: threading.Thread | None = None
    try:
        cfg = load_config()
        library = load_library()
        lock = resolved_live_patrol_lock(library)
        if lock is None:
            raise RuntimeError("The locked live patrol baseline is not available.")
        if map_id != lock["map_id"] or baseline_replay_id != lock["baseline_replay_id"]:
            raise RuntimeError("Live simulation must use the locked map and full patrol baseline.")

        selected = set_selected_map(map_id)
        baseline = next(
            (item for item in selected.get("replays", []) if item.get("id") == baseline_replay_id),
            None,
        )
        if not isinstance(baseline, dict) or baseline.get("kind") != "route_constrained_taught_baseline":
            raise RuntimeError("The selected replay is not the validated full patrol baseline.")

        baseline_asset = VIEWER / str(baseline.get("asset_base") or "")
        baseline_pose_path = baseline_asset / "poses.json"
        source_payload = json.loads(baseline_pose_path.read_text(encoding="utf-8"))
        source_poses = [pose for pose in source_payload.get("poses", []) if isinstance(pose, dict)]
        source_frames = sorted(
            {
                int(pose.get("source_frame", pose.get("frame_index")))
                for pose in source_poses
                if pose.get("source_frame", pose.get("frame_index")) is not None
            }
        )
        if len(source_frames) < 2:
            raise RuntimeError("The full patrol baseline has no usable recorded-frame interval.")

        query_frame_base_url = str(
            baseline.get("query_frame_base_url")
            or source_payload.get("query_frame_base_url")
            or source_payload.get("frame_source")
            or ""
        ).strip()
        query_frames = VIEWER / query_frame_base_url
        if not query_frames.is_dir():
            raise RuntimeError(f"Recorded patrol frame directory is missing: {query_frames}")
        recorded_frame_indices: list[int] = []
        for frame_path in query_frames.glob("query_*.jpg"):
            try:
                recorded_frame_indices.append(int(frame_path.stem.rsplit("_", 1)[-1]))
            except (TypeError, ValueError):
                continue
        if not recorded_frame_indices:
            raise RuntimeError("The recorded patrol frame directory has no numbered frames.")
        recorded_frame_indices.sort()
        # The validated route pose ends exactly at the returned Point 1, but
        # the same uninterrupted camera recording continues through the
        # in-place Point-1-to-Point-2 turn. Permit an explicitly requested
        # validation range to include those real continuation frames. They are
        # localized online; no pose is copied or extrapolated from the route.
        start_frame = max(
            source_frames[0],
            recorded_frame_indices[0],
            int(start_source_frame) if start_source_frame is not None else source_frames[0],
        )
        requested_end_frame = (
            int(end_source_frame) if end_source_frame is not None else source_frames[-1]
        )
        end_frame = min(recorded_frame_indices[-1], requested_end_frame)
        if start_frame >= end_frame:
            raise RuntimeError("The requested recorded-frame interval is empty.")
        expected_count = end_frame - start_frame + 1

        replay_id = make_map_id("live_frames")
        replay_title = (
            f"Live Frame Validation {time.strftime('%H:%M:%S')}"
            if validation_only
            else f"New Localized Live Path {time.strftime('%H:%M:%S')}"
        )
        base_asset_dir = map_asset_dir(selected)
        out_asset_dir = base_asset_dir / "replays" / replay_id
        out_asset_dir.mkdir(parents=True, exist_ok=True)
        partial_pose_path = out_asset_dir / "poses_partial.json"
        final_pose_path = out_asset_dir / "poses.json"
        run_root = ROOT / "results" / "recorded_live_runs" / selected["id"] / replay_id
        runtime_out = run_root / "tsolve_runtime"
        inputs_out = run_root / "tsolve_inputs"
        work_dir = run_root / "live_existing_map_stream"
        map_artifacts = colmap_artifacts_for_entry(selected)
        baseline_reference_path = Path(str(lock.get("baseline_reference_path") or ""))
        baseline_reference = json.loads(baseline_reference_path.read_text(encoding="utf-8"))
        baseline_legs = list(baseline_reference.get("legs") or [])
        phase_boundaries = dict(
            baseline_reference.get("phase_boundaries")
            or source_payload.get("phase_boundaries")
            or {}
        )
        if len(baseline_legs) < 4:
            raise RuntimeError("The locked baseline has no complete four-leg route trace.")
        baseline_cruise_y = float(
            baseline.get("cruise_y")
            if baseline.get("cruise_y") is not None
            else source_payload.get("cruise_y", 0.0)
        )
        simulated_control_status = run_root / "simulated_control_status.json"
        frame_plan: list[dict] = []
        composite_source_replay_ids: list[str] = [baseline_replay_id]
        # A concatenated validation stream has an artificial camera cut at
        # the start of lap 2.  Requiring a fresh map-wide metric solve at that
        # exact cut tests COLMAP's cold-start latency, not the continuity a
        # real two-lap flight has.  An alternate recording may bypass that
        # *simulation-only* checkpoint only after its Point-1 seam has passed
        # the explicit feature audit below.  Physical patrol execution never
        # enters this recorded-frame job and retains the metric-pose gate.
        composite_point1_seam_verified = False
        laps = max(1, min(2, int(laps)))
        if laps > 1:
            if start_source_frame is not None or end_source_frame is not None:
                raise RuntimeError(
                    "A two-lap simulation uses the complete audited loop; "
                    "custom source-frame limits are not allowed."
                )
            start_frame = int(phase_boundaries.get("point1") or source_frames[0])
            loop_return_frame = int(
                phase_boundaries.get("point1_return") or source_frames[-1]
            )
            next_departure_frame = int(
                phase_boundaries.get("point1_next_departure") or loop_return_frame
            )
            second_lap_segments: list[dict] | None = None
            composite_plan_path = baseline_asset / "two_lap_composite_plan.json"
            if composite_plan_path.is_file():
                composite_plan = json.loads(composite_plan_path.read_text(encoding="utf-8"))
                if (
                    composite_plan.get("enabled") is not True
                    or composite_plan.get("complete_second_lap") is not True
                    or composite_plan.get("map_id") != selected["id"]
                    or composite_plan.get("baseline_replay_id") != baseline_replay_id
                ):
                    raise RuntimeError("The alternate second-lap plan failed its identity/completeness check.")
                seam_audits = list(composite_plan.get("seam_audits") or [])
                if not seam_audits or any(
                    int(seam.get("orb_inliers") or 0) < 40
                    for seam in seam_audits
                    if isinstance(seam, dict)
                ):
                    raise RuntimeError("The alternate second-lap recording has an unaudited visual seam.")
                point1_seams = [
                    seam
                    for seam in seam_audits
                    if isinstance(seam, dict)
                    and str(seam.get("at") or seam.get("name") or "").lower()
                    in {"point1", "lap1_to_lap2", "lap_1_to_lap_2"}
                ]
                # Older plans did not name their seams.  In that schema the
                # first audit is the documented lap-1 -> lap-2 Point-1 cut.
                point1_seam = point1_seams[0] if point1_seams else seam_audits[0]
                composite_point1_seam_verified = bool(
                    isinstance(point1_seam, dict)
                    and int(point1_seam.get("orb_inliers") or 0) >= 40
                    and float(point1_seam.get("inlier_ratio") or 0.0) >= 0.35
                )
                if not composite_point1_seam_verified:
                    raise RuntimeError(
                        "The alternate second-lap Point-1 seam is not strong enough for a simulation handoff."
                    )
                second_lap_segments = []
                for raw_segment in composite_plan.get("second_lap_segments") or []:
                    if not isinstance(raw_segment, dict):
                        continue
                    segment_replay_id = str(raw_segment.get("source_replay_id") or "")
                    source_replay = next(
                        (
                            item
                            for item in selected.get("replays", [])
                            if isinstance(item, dict) and item.get("id") == segment_replay_id
                        ),
                        None,
                    )
                    if not isinstance(source_replay, dict):
                        raise RuntimeError(
                            f"Alternate second-lap source replay is missing: {segment_replay_id}"
                        )
                    source_frame_url = str(
                        source_replay.get("query_frame_base_url")
                        or (
                            query_frame_base_url
                            if segment_replay_id == baseline_replay_id
                            else ""
                        )
                    ).strip()
                    source_frame_path = (VIEWER / source_frame_url).resolve()
                    if VIEWER.resolve() not in source_frame_path.parents:
                        raise RuntimeError("Alternate second-lap frame path leaves the viewer asset root.")
                    second_lap_segments.append(
                        {
                            "source_frame_dir": source_frame_path,
                            "source_replay_id": segment_replay_id,
                            "start_frame": int(raw_segment["start_frame"]),
                            "end_frame": int(raw_segment["end_frame"]),
                            "phase_boundaries": dict(raw_segment.get("phase_boundaries") or {}),
                            "label": str(raw_segment.get("label") or segment_replay_id),
                        }
                    )
                    if segment_replay_id not in composite_source_replay_ids:
                        composite_source_replay_ids.append(segment_replay_id)
                if len(second_lap_segments) < 2:
                    raise RuntimeError("The alternate second lap must contain independent and final-leg segments.")
            query_frames = out_asset_dir / "two_lap_query_frames"
            frame_plan = build_recorded_patrol_lap_sequence(
                source_frame_dir=VIEWER / str(
                    baseline.get("query_frame_base_url")
                    or source_payload.get("query_frame_base_url")
                    or source_payload.get("frame_source")
                    or ""
                ).strip(),
                output_frame_dir=query_frames,
                start_frame=start_frame,
                loop_return_frame=loop_return_frame,
                next_departure_frame=next_departure_frame,
                laps=laps,
                phase_boundaries=phase_boundaries,
                source_replay_id=baseline_replay_id,
                second_lap_segments=second_lap_segments,
            )
            query_frame_base_url = public_rel(query_frames)
            expected_count = len(frame_plan)

        source_session = str(
            source_payload.get("source_session")
            or baseline.get("source_replay_id")
            or source_payload.get("source_replay_id")
            or ""
        )
        source_run_id = source_session[6:] if source_session.startswith("atlas_") else source_session
        runtime_dir = ROOT / "results" / "dji_live_runs" / selected["id"] / source_run_id / "tsolve_runtime_code"
        if not runtime_dir.is_dir():
            raise RuntimeError(f"Recorded patrol TSolve runtime is missing: {runtime_dir}")

        started_at = time.time()
        atomic_write_json(
            partial_pose_path,
            {
                "mode": "recorded_frames_live_tsolve_partial",
                "replay_id": replay_id,
                "source_replay_id": baseline_replay_id,
                "source_replay_ids": composite_source_replay_ids,
                "frame_source": query_frame_base_url,
                "query_frame_base_url": query_frame_base_url,
                "expected_count": expected_count,
                "processed_count": 0,
                "complete": False,
                "poses": [],
            },
        )
        set_live_stream(
            {
                "simulated_live": True,
                "genuine_localization": True,
                "live_atlas": False,
                "map_id": selected["id"],
                "replay_id": replay_id,
                "source_replay_id": baseline_replay_id,
                "source_replay_ids": composite_source_replay_ids,
                "title": replay_title,
                "asset_base": public_rel(out_asset_dir),
                "partial_pose_url": public_rel(partial_pose_path),
                "query_frame_base_url": query_frame_base_url,
                "pose_count": 0,
                "expected_count": expected_count,
                "complete": False,
                "started_at": started_at,
                "fps": max(0.5, min(10.0, float(fps))),
                "view_only": True,
                "start_source_frame": start_frame,
                "end_source_frame": end_frame,
                "validation_only": validation_only,
                "timing_mode": "recorded" if recorded_timing else "fixed_fps",
                "laps": laps,
                "frame_plan_url": (
                    public_rel(query_frames / "frame_plan.json")
                    if frame_plan
                    else None
                ),
            }
        )
        set_job(
            "drone",
            "running",
            f"Genuine live simulation: localizing recorded frame {start_frame}/{end_frame} with COLMAP, optical flow and TSolve.",
        )

        monitor_thread = threading.Thread(
            target=monitor_partial_pose_file,
            args=(partial_pose_path, monitor_stop, started_at, runtime_out / "live_stage_times.csv"),
            daemon=True,
        )
        monitor_thread.start()

        def simulated_route_state(frame_index: int) -> tuple[int, bool, int, int]:
            lap_number = 1
            route_frame_index = frame_index
            if frame_plan:
                plan_index = max(0, min(len(frame_plan) - 1, int(frame_index)))
                plan_row = frame_plan[plan_index]
                route_frame_index = int(plan_row["recorded_source_frame"])
                lap_number = int(plan_row["lap"])
                if plan_row.get("route_leg_index") is not None:
                    return (
                        int(plan_row["route_leg_index"]),
                        bool(plan_row.get("translation_locked")),
                        route_frame_index,
                        lap_number,
                    )
            p2_arrival = int(phase_boundaries.get("point2_arrival") or start_frame)
            p2_departure = int(phase_boundaries.get("point2_departure") or p2_arrival)
            p3_arrival = int(phase_boundaries.get("point3_arrival") or p2_departure)
            p3_departure = int(phase_boundaries.get("point3_departure") or p3_arrival)
            p4_arrival = int(phase_boundaries.get("point4_arrival") or p3_departure)
            p4_departure = int(phase_boundaries.get("point4_departure") or p4_arrival)
            p1_return = int(phase_boundaries.get("point1_return") or source_frames[-1])
            if route_frame_index < p2_arrival:
                return 0, False, route_frame_index, lap_number
            if route_frame_index < p2_departure:
                return 1, True, route_frame_index, lap_number
            if route_frame_index < p3_arrival:
                return 1, False, route_frame_index, lap_number
            if route_frame_index < p3_departure:
                return 2, True, route_frame_index, lap_number
            if route_frame_index < p4_arrival:
                return 2, False, route_frame_index, lap_number
            if route_frame_index < p4_departure:
                return 3, True, route_frame_index, lap_number
            if route_frame_index < p1_return:
                return 3, False, route_frame_index, lap_number
            # The manual recording continues after the final 4->1 arrival.
            # Those frames are the physical in-place turn toward Point 2 for
            # the next lap: hold the Point 1 anchor and change heading only.
            return 0, True, route_frame_index, lap_number

        def publish_simulated_controller_status() -> None:
            # A concatenated replay is addressed by sequence index, not by the
            # source recording's frame number.  Starting at ``start_frame``
            # would incorrectly place the controller part-way through lap 1
            # until the first streamed pose arrived.
            last_frame = 0 if frame_plan else start_frame
            lap_two_metric_ready = laps < 2
            lap_two_sequence_start = next(
                (
                    index
                    for index, row in enumerate(frame_plan)
                    if int(row.get("lap") or 1) == 2
                ),
                None,
            )
            while not controller_status_stop.is_set():
                try:
                    partial = json.loads(partial_pose_path.read_text(encoding="utf-8"))
                    current = partial.get("current_frame") if isinstance(partial, dict) else None
                    if isinstance(current, dict) and current.get("frame_index") is not None:
                        last_frame = int(current["frame_index"]) + 1
                    partial_poses = (
                        partial.get("poses")
                        if isinstance(partial, dict)
                        and isinstance(partial.get("poses"), list)
                        else []
                    )
                    latest_pose = next(
                        (pose for pose in reversed(partial_poses) if isinstance(pose, dict)),
                        None,
                    )
                    if (
                        not lap_two_metric_ready
                        and lap_two_sequence_start is not None
                        and isinstance(latest_pose, dict)
                    ):
                        match = re.search(
                            r"(\d{6})(?:\.[^.]+)?$",
                            str(latest_pose.get("image_name") or ""),
                        )
                        pose_sequence = int(match.group(1)) if match else -1
                        lap_two_metric_ready = bool(
                            pose_sequence >= lap_two_sequence_start
                            and latest_pose.get("success") is True
                            and latest_pose.get("held_pose") is not True
                            and latest_pose.get("pose_source")
                            != "patrol_visual_route_recovery"
                            and latest_pose.get("R") is not None
                            and latest_pose.get("t") is not None
                            and latest_pose.get("center") is not None
                        )
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    pass
                leg_index, translation_locked, recorded_frame, lap_number = (
                    simulated_route_state(last_frame)
                )
                metric_checkpoint_active = bool(
                    lap_number > 1
                    and not lap_two_metric_ready
                    and not composite_point1_seam_verified
                )
                if metric_checkpoint_active:
                    # A real aircraft would hover at Point 1 here. Keep the
                    # simulator on the same Point-1->2 route contract until a
                    # current frame produces genuine metric R,t as well.
                    leg_index = 0
                    translation_locked = True
                leg = baseline_legs[leg_index]
                anchor = list(leg.get("from") or [])
                target = list(leg.get("to") or [])
                if len(anchor) >= 3 and len(target) >= 3:
                    # Match the real patrol planner: map points provide X/Z,
                    # while the controller owns one fixed flight altitude.
                    anchor[1] = baseline_cruise_y
                    target[1] = baseline_cruise_y
                atomic_write_json(
                    simulated_control_status,
                    {
                        "status": "running",
                        "command": "mission",
                        "updated_at": time.time(),
                        "progress": {
                            "map_id": selected["id"],
                            "patrol_id": lock["patrol_id"],
                            "baseline_replay_id": baseline_replay_id,
                            "lap": lap_number,
                            "leg_index": leg_index + 1,
                            "segment_start": anchor,
                            "target": target,
                            "position_anchor": anchor,
                            "translation_locked": translation_locked,
                            "physical_translation_active": not translation_locked,
                            "body_forward_gain": 0.0 if translation_locked else 1.0,
                            "body_lateral_gain": 0.0,
                            "phase": (
                                "pose_recovery"
                                if metric_checkpoint_active
                                else (
                                    "patrol_yaw"
                                    if translation_locked
                                    else "patrol_translation"
                                )
                            ),
                            "route_visual_recovery_allowed": bool(
                                metric_checkpoint_active
                                or (lap_number > 1 and composite_point1_seam_verified)
                            ),
                            "require_metric_pose": metric_checkpoint_active,
                            "metric_pose_ready": lap_two_metric_ready,
                            "simulation_point1_seam_verified": bool(
                                lap_number > 1 and composite_point1_seam_verified
                            ),
                            "simulation_metric_checkpoint_bypassed": bool(
                                lap_number > 1
                                and composite_point1_seam_verified
                                and not lap_two_metric_ready
                            ),
                            "simulated_source_frame": recorded_frame,
                            "simulated_sequence_frame": last_frame,
                        },
                    },
                )
                controller_status_stop.wait(0.05)

        controller_status_thread = threading.Thread(
            target=publish_simulated_controller_status,
            name=f"recorded-controller-trace-{replay_id}",
            daemon=True,
        )
        controller_status_thread.start()

        source_fps = float(cfg.get("simulated_live_query_frame_fps", cfg.get("query_frame_fps", 10.0)))
        pace_scale = 1.0 if recorded_timing else source_fps / max(0.5, min(10.0, float(fps)))
        cmd: list[object] = [
            cfg["python"],
            ROOT / "scripts" / "run_bounded_tsolve_video_stream.py",
            "--colmap", cfg["colmap_bin"],
            "--map-database", map_artifacts["database"],
            "--map-images", map_artifacts["images"],
            "--map-sparse-model", map_artifacts["sparse_model"],
            "--map-sparse-text", map_artifacts["sparse_text"],
            "--query-frames", query_frames,
            "--runtime-dir", runtime_dir,
            "--solver-dir", cfg["solver_dir"],
            "--inputs-out-dir", inputs_out,
            "--out-dir", runtime_out,
            "--work-dir", work_dir,
            "--max-image-size", cfg["max_image_size"],
            "--query-camera-model", cfg["query_camera_model"],
            "--min-points", cfg["min_query_correspondences"],
            "--max-points", cfg["max_query_correspondences"],
            "--max-reference-images", cfg.get("live_reference_image_cap", 120),
            "--tracking-reference-images", cfg.get("live_tracking_reference_image_cap", 18),
            "--track-pool-size", cfg.get("live_tracking_pool_size", 900),
            "--relocalize-every", cfg.get("live_relocalize_every", 0),
            "--flow-max-error", cfg.get("live_flow_max_error", 34.0),
            "--flow-backtrack-error", cfg.get("live_flow_backtrack_error", 2.5),
            "--flow-window", cfg.get("live_flow_window", 31),
            "--flow-levels", cfg.get("live_flow_levels", 4),
            "--flow-iterations", cfg.get("live_flow_iterations", 24),
            "--min-track-points", cfg.get("live_min_track_points", 15),
            "--min-track-ratio", cfg.get("live_min_track_ratio", 0.10),
            "--proactive-relocalize-points", cfg.get("live_proactive_relocalize_points", 500),
            "--proactive-relocalize-cooldown-frames", cfg.get("live_proactive_relocalize_cooldown_frames", 30),
            "--global-recovery-after-failures", cfg.get("live_global_recovery_after_failures", 3),
            "--background-recovery-timeout-seconds", cfg.get("live_background_recovery_timeout_seconds", 20.0),
            "--global-recovery-max-step", cfg.get("live_global_recovery_max_step", 0.55),
            "--output-max-step", cfg.get("live_output_max_step", 0.30),
            "--output-objective-threshold", cfg.get("live_output_objective_threshold", 30.0),
            "--prime", cfg["tsolve_prime"],
            "--degree", cfg["tsolve_degree"],
            "--action-weights", cfg["tsolve_action_weights"],
            "--tsolve-root-profile", cfg.get("tsolve_root_profile", "full"),
            "--fallback-action-weights", cfg["tsolve_fallback_action_weights"],
            "--partial-pose-out", partial_pose_path,
            "--replay-id", replay_id,
            "--expected-count", expected_count,
            "--start-frame-index", 0 if frame_plan else start_frame,
            "--end-frame-index", (expected_count - 1) if frame_plan else end_frame,
            "--scene-json", base_asset_dir / "scene.json",
            "--display-z-sign", selected.get("display_z_sign", -1),
            "--room-alignment-json", json.dumps(selected.get("room_alignment") or {}),
            "--rotation-position-stabilizer-profile", lock.get("reference_profile", "live-atlas-141441"),
            "--rotation-command-status-json", simulated_control_status,
            "--patrol-route-baseline", baseline_reference_path,
            "--patrol-route-max-cross-track", cfg.get("live_patrol_route_max_cross_track", 0.55),
            "--patrol-route-backward-tolerance", cfg.get("live_patrol_route_backward_tolerance", 0.045),
            "--patrol-status-max-age", 2.0,
            "--patrol-turn-max-position-drift", cfg.get("live_patrol_turn_max_position_drift", 0.75),
            "--direct-pnp-recovery",
            "--pace-replay",
            "--pace-scale", pace_scale,
        ]
        # Match the real Patrol 1 self-localization inputs.  These are compact
        # 2D->3D map correspondences learned from the taught full loop; they
        # feed a fresh TSolve case and do not copy any prerecorded pose.
        for recovery_bank in active_taught_recovery_banks(base_asset_dir):
            cmd.extend(["--taught-patrol-recovery-bank", recovery_bank])
        if lock.get("visual_recovery_path"):
            cmd.extend(["--patrol-visual-recovery-bank", lock["visual_recovery_path"]])
        if laps > 1:
            # The recorded two-lap stream emulates the live Point-1 hover:
            # never consume an advancing second-lap camera stream while the
            # metric rematch for the current frame is still pending.
            cmd.append("--wait-for-metric-checkpoint-recovery")
        run_cmd("drone", cmd, DRONE_STOP_EVENT)
        monitor_stop.set()
        monitor_thread.join(timeout=3.0)

        payload = json.loads(partial_pose_path.read_text(encoding="utf-8"))
        poses = payload.get("poses") if isinstance(payload.get("poses"), list) else []
        if not poses:
            raise RuntimeError("Recorded-frame live localization produced no poses.")
        counts = pose_stream_counts(poses)
        if frame_plan:
            for pose in poses:
                match = re.search(r"(\d{6})(?:\.[^.]+)?$", str(pose.get("image_name") or ""))
                if not match:
                    continue
                sequence_index = int(match.group(1))
                if 0 <= sequence_index < len(frame_plan):
                    plan_row = frame_plan[sequence_index]
                    pose["simulation_lap"] = int(plan_row["lap"])
                    pose["recorded_source_frame"] = int(
                        plan_row["recorded_source_frame"]
                    )
                    pose["recorded_source_replay_id"] = str(
                        plan_row.get("recorded_source_replay_id") or ""
                    )
        payload.update(
            {
                "mode": "recorded_frames_live_tsolve_replay",
                "description": "New poses computed online from the recorded DJI frames with the live COLMAP/optical-flow/TSolve pipeline.",
                "source_replay_id": baseline_replay_id,
                "source_replay_ids": composite_source_replay_ids,
                "query_frame_base_url": query_frame_base_url,
                "complete": True,
                "cancelled": False,
                "updated_at": time.time(),
                "laps": laps,
            }
        )
        atomic_write_json(final_pose_path, payload)
        replay = {
            "id": replay_id,
            "title": replay_title,
            "asset_base": public_rel(out_asset_dir),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_video": (
                "Mixed 15:47:14 and independent 11:57:36 patrol frames localized as one two-lap live stream"
                if len(composite_source_replay_ids) > 1
                else "Full patrol recorded DJI frames localized as a new live stream"
            ),
            "source_replay_id": baseline_replay_id,
            "source_replay_ids": composite_source_replay_ids,
            "kind": "simulated_live_tsolve_recorded_frames",
            "counts": {
                "poses": counts["poses"],
                "accepted": counts["poses"],
                "frames": len(poses),
                "held": counts["held"],
                "failed": counts["failed"],
            },
            "query_frame_base_url": query_frame_base_url,
            "laps": laps,
        }
        if not validation_only:
            add_replay_to_map(selected["id"], replay, select=True)
        update_live_stream(
            pose_count=len(poses),
            accepted_pose_count=counts["poses"],
            held_pose_count=counts["held"],
            failed_pose_count=counts["failed"],
            partial_pose_url=public_rel(final_pose_path),
            final_pose_url=public_rel(final_pose_path),
            complete=True,
            cancelled=False,
        )
        set_job(
            "drone",
            "done",
            f"Genuine recorded-frame live localization complete: {counts['poses']} accepted, {counts['held']} held.",
        )
    except Exception as exc:
        append_log("drone", f"RECORDED LIVE LOCALIZATION ERROR: {exc}")
        update_live_stream(complete=True, failed=True, error=str(exc))
        set_job("drone", "error", str(exc))
    finally:
        monitor_stop.set()
        controller_status_stop.set()
        if monitor_thread is not None and monitor_thread.is_alive():
            monitor_thread.join(timeout=2.0)
        if controller_status_thread is not None and controller_status_thread.is_alive():
            controller_status_thread.join(timeout=2.0)
        release_drone_job()


def fleet_run_cmd(drone_id: str, cmd: list[object], stop_event: threading.Event | None = None) -> None:
    """Run one fleet subprocess without sharing the legacy single-drone job state."""
    cmd = [str(item) for item in cmd]
    kind = f"fleet:{drone_id}"
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "minimal")
    log_path = fleet_session_public_root(drone_id) / "worker.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    register_active_proc(kind, proc)
    fleet_update(drone_id, active_processes=active_proc_count(kind))
    try:
        assert proc.stdout is not None
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write("+ " + " ".join(cmd) + "\n")
            log_handle.flush()
            while proc.poll() is None:
                if stop_event is not None and stop_event.is_set():
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        proc.wait(timeout=4)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        proc.wait(timeout=4)
                    raise RuntimeError("Fleet session cancelled.")
                ready, _, _ = select.select([proc.stdout], [], [], 0.2)
                if ready:
                    line = proc.stdout.readline()
                    if line:
                        log_handle.write(line)
                        log_handle.flush()
            for line in proc.stdout:
                log_handle.write(line)
            rc = proc.wait()
    finally:
        unregister_active_proc(kind, proc)
        fleet_update(drone_id, active_processes=active_proc_count(kind))
    if stop_event is not None and stop_event.is_set():
        raise RuntimeError("Fleet session cancelled.")
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def fleet_start_background_cmd(drone_id: str, cmd: list[object], stop_event: threading.Event) -> threading.Thread:
    def target() -> None:
        try:
            fleet_run_cmd(drone_id, cmd, stop_event)
        except Exception as exc:
            if not stop_event.is_set():
                fleet_event(drone_id, f"DJI bridge error: {exc}", "error")
                fleet_update(drone_id, bridge_error=str(exc))

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


def monitor_fleet_partial_pose(
    drone_id: str,
    partial_path: Path,
    monitor_stop: threading.Event,
    started_at: float,
    stage_times_path: Path,
) -> None:
    last_count = -1
    last_stage = ""
    while not monitor_stop.is_set():
        try:
            payload = json.loads(partial_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        poses = payload.get("poses") if isinstance(payload.get("poses"), list) else []
        counts = pose_stream_counts(poses)
        count = int(payload.get("processed_count") or len(poses))
        fields = {
            "pose_count": count,
            "accepted_pose_count": counts["poses"],
            "held_pose_count": counts["held"],
            "failed_pose_count": counts["failed"],
        }
        if count > 0:
            fields["localization_ready"] = counts["poses"] > 0
            with FLEET_LOCK:
                session = FLEET_SESSIONS.get(drone_id) or {}
                if not session.get("first_pose_at"):
                    fields["first_pose_at"] = time.time()
                    fields["first_pose_latency_seconds"] = time.time() - started_at
        fleet_update(drone_id, **fields)
        if count != last_count and count > 0 and (count <= 3 or count % 10 == 0):
            fleet_event(
                drone_id,
                f"Localization healthy: {counts['poses']} accepted, {counts['held']} held, {counts['failed']} rejected.",
                "ok" if counts["poses"] else "warning",
            )
            last_count = count
        row = latest_live_stage_row(stage_times_path)
        if row:
            frame_index = str(row.get("frame_index") or "")
            reason = str(row.get("reason") or "processing").strip()
            key = f"{frame_index}:{reason}"
            if key != last_stage:
                fleet_update(
                    drone_id,
                    latest_frame_index=frame_index,
                    latest_localization_reason=reason,
                    latest_total_frame_ms=row.get("total_frame_ms"),
                )
                if reason not in {"processing", "accepted", "fresh_pose"}:
                    fleet_event(drone_id, f"Localization frame {frame_index}: {reason}", "warning")
                last_stage = key
        time.sleep(0.35)


def finalize_fleet_replay(
    *,
    drone_id: str,
    selected: dict,
    replay_id: str,
    replay_title: str,
    out_asset_dir: Path,
    partial_pose_path: Path,
    query_frames: Path,
) -> int:
    payload = json.loads(partial_pose_path.read_text(encoding="utf-8")) if partial_pose_path.exists() else {"poses": []}
    poses = payload.get("poses") if isinstance(payload.get("poses"), list) else []
    counts = pose_stream_counts(poses)
    final_payload = {
        **payload,
        "mode": "dji_fleet_live_tsolve_replay",
        "description": f"ATLAS fleet TSolve R,t estimates for {drone_id}.",
        "complete": True,
        "processed_count": len(poses),
        "accepted_count": counts["poses"],
        "held_count": counts["held"],
        "failed_count": counts["failed"],
        "query_frame_base_url": public_rel(query_frames),
        "frame_source": public_rel(query_frames),
        "updated_at": time.time(),
        "poses": poses,
    }
    final_path = out_asset_dir / "poses.json"
    atomic_write_json(final_path, final_payload)
    replay = {
        "id": replay_id,
        "title": replay_title,
        "asset_base": public_rel(out_asset_dir),
        "created_at": now_label(),
        "source_video": f"DJI fleet live stream ({drone_id})",
        "query_frame_base_url": public_rel(query_frames),
        "counts": counts,
    }
    with LIBRARY_LOCK:
        add_replay_to_map(selected["id"], replay, select=False)
    fleet_update(
        drone_id,
        pose_count=len(poses),
        accepted_pose_count=counts["poses"],
        held_pose_count=counts["held"],
        failed_pose_count=counts["failed"],
        final_pose_url=public_rel(final_path),
        complete=True,
    )
    return counts["poses"]


def fleet_live_atlas_job(drone_id: str) -> None:
    bridge_thread: threading.Thread | None = None
    monitor_thread: threading.Thread | None = None
    monitor_stop = threading.Event()
    with FLEET_LOCK:
        session = FLEET_SESSIONS.get(drone_id)
        if not session:
            return
        stop_event = session["stop_event"]
        map_id = str(session["map_id"])
        patrol_id = str(session["patrol_id"])
        phone_ip = str(session["phone_ip"])
        fps = float(session["fps"])
        max_size = int(session["max_size"])
    try:
        cfg = load_config()
        py = Path(cfg["python"])
        scripts = ROOT / "scripts"
        library = load_library()
        selected = next((item for item in library.get("maps", []) if item.get("id") == map_id), None)
        if not selected:
            raise RuntimeError(f"Unknown map id: {map_id}")
        patrol = next((item for item in selected.get("patrols") or [] if item.get("id") == patrol_id), None)
        if not patrol:
            raise RuntimeError(f"Unknown patrol {patrol_id} on {selected.get('title') or map_id}.")
        live_patrol_lock = resolved_live_patrol_lock(library)
        if live_patrol_lock is not None and (
            live_patrol_lock["map_id"] != map_id
            or live_patrol_lock["patrol_id"] != patrol_id
        ):
            live_patrol_lock = None
        map_artifacts = colmap_artifacts_for_entry(selected)
        replay_id = make_map_id(f"fleet_{drone_id}")
        session_id = f"atlas_{replay_id}"
        replay_title = f"Fleet {drone_id} {time.strftime('%H:%M:%S')}"
        base_asset_dir = VIEWER / selected["asset_base"]
        if not base_asset_dir.exists():
            base_asset_dir = MAPS_DIR / selected["id"]
        out_asset_dir = base_asset_dir / "replays" / replay_id
        out_asset_dir.mkdir(parents=True, exist_ok=True)
        partial_pose_path = out_asset_dir / "poses_partial.json"
        stop_file = out_asset_dir / "STOP_LIVE_ATLAS"
        run_root = ROOT / "results" / "fleet_live_runs" / drone_id / replay_id
        runtime_dir = run_root / "tsolve_runtime_code"
        tsolve_runtime = run_root / "tsolve_runtime"
        tsolve_inputs = run_root / "tsolve_inputs"
        stream_work = run_root / "live_existing_map_stream"
        public_sessions_root = PUBLIC / "live_dji_sessions"
        query_frames = public_sessions_root / session_id / "query_frames"
        public_root = fleet_session_public_root(drone_id)
        latest_path = public_root / "latest.jpg"
        started_at = time.time()
        atomic_write_json(
            partial_pose_path,
            {
                "mode": "dji_fleet_live_tsolve_partial",
                "drone_id": drone_id,
                "replay_id": replay_id,
                "frame_source": str(query_frames),
                "query_frame_base_url": public_rel(query_frames),
                "expected_count": 0,
                "processed_count": 0,
                "complete": False,
                "updated_at": time.time(),
                "poses": [],
            },
        )
        fleet_update(
            drone_id,
            status="running",
            stage="connecting",
            replay_id=replay_id,
            replay_title=replay_title,
            replay_asset_base=public_rel(out_asset_dir),
            partial_pose_url=public_rel(partial_pose_path),
            stop_file=str(stop_file),
            live_preview_url=public_rel(latest_path),
            query_frame_base_url=public_rel(query_frames),
            public_root=public_rel(public_root),
            started_at=started_at,
        )
        fleet_event(drone_id, f"Connecting to Android endpoint {phone_ip}.")
        bridge_cmd: list[object] = [
            py,
            scripts / "atlas_dji_live_bridge.py",
            "--phone-ip",
            phone_ip,
            "--fps",
            fps,
            "--max-size",
            max_size,
            "--session",
            session_id,
            "--out-root",
            public_sessions_root,
            "--public-root",
            public_root,
            "--pose-stream",
            partial_pose_path,
        ]
        enemy_model = selected_enemy_model_path()
        if enemy_model:
            bridge_cmd.extend(
                [
                    "--enemy-model",
                    enemy_model,
                    "--enemy-output",
                    public_root / "enemy_detections.json",
                    "--enemy-detect-fps",
                    cfg.get("enemy_live_detect_fps", min(fps, 5.0)),
                    "--enemy-conf",
                    cfg.get("enemy_live_confidence", 0.35),
                ]
            )
        bridge_thread = fleet_start_background_cmd(drone_id, bridge_cmd, stop_event)
        fleet_update(drone_id, stage="preparing_localizer")
        fleet_event(drone_id, "Preparing an isolated TSolve localization runtime.")
        fleet_run_cmd(
            drone_id,
            [
                py,
                scripts / "setup_tsolve_runtime.py",
                "--base-yam-code-dir",
                cfg["base_yam_code_dir"],
                "--dropin-patch-dir",
                cfg["dropin_patch_dir"],
                "--base-harness-dir",
                str(ROOT.parent / "pnp-symbolic-research/Yam/exact_ff_ysolve_pnp/harness"),
                "--out-dir",
                runtime_dir,
            ],
            stop_event,
        )
        wait_started = time.time()
        while not stop_event.is_set() and count_frame_images(query_frames) < 1:
            if time.time() - wait_started > 60:
                raise RuntimeError("DJI bridge connected, but no live frames arrived within 60 seconds.")
            time.sleep(0.5)
        if stop_event.is_set():
            raise RuntimeError("Fleet session stopped before the first DJI frame.")
        fleet_update(drone_id, stage="localizing")
        fleet_event(drone_id, "First DJI frame received; live map localization is running.", "ok")
        monitor_thread = threading.Thread(
            target=monitor_fleet_partial_pose,
            args=(drone_id, partial_pose_path, monitor_stop, started_at, tsolve_runtime / "live_stage_times.csv"),
            daemon=True,
        )
        monitor_thread.start()
        live_cmd: list[object] = [
            py,
            scripts / "run_bounded_tsolve_video_stream.py",
            "--colmap", cfg["colmap_bin"],
            "--map-database", map_artifacts["database"],
            "--map-images", map_artifacts["images"],
            "--map-sparse-model", map_artifacts["sparse_model"],
            "--map-sparse-text", map_artifacts["sparse_text"],
            "--query-frames", query_frames,
            "--runtime-dir", runtime_dir,
            "--solver-dir", cfg["solver_dir"],
            "--inputs-out-dir", tsolve_inputs,
            "--out-dir", tsolve_runtime,
            "--work-dir", stream_work,
            "--max-image-size", cfg["max_image_size"],
            "--query-camera-model", cfg["query_camera_model"],
            "--min-points", cfg["min_query_correspondences"],
            "--max-points", cfg["max_query_correspondences"],
            "--max-reference-images", cfg.get("live_reference_image_cap", 24),
            "--tracking-reference-images", cfg.get("live_tracking_reference_image_cap", 10),
            "--track-pool-size", cfg.get("live_tracking_pool_size", 900),
            "--relocalize-every", cfg.get("live_relocalize_every", 0),
            "--flow-max-error", cfg.get("live_flow_max_error", 34.0),
            "--flow-backtrack-error", cfg.get("live_flow_backtrack_error", 2.5),
            "--flow-window", cfg.get("live_flow_window", 21),
            "--flow-levels", cfg.get("live_flow_levels", 3),
            "--flow-iterations", cfg.get("live_flow_iterations", 18),
            "--min-track-points", cfg.get("live_min_track_points", 80),
            "--min-track-ratio", cfg.get("live_min_track_ratio", 0.10),
            "--proactive-relocalize-points", cfg.get("live_proactive_relocalize_points", 500),
            "--proactive-relocalize-cooldown-frames", cfg.get("live_proactive_relocalize_cooldown_frames", 60),
            "--global-recovery-after-failures", cfg.get("live_global_recovery_after_failures", 2),
            "--background-recovery-timeout-seconds", cfg.get("live_background_recovery_timeout_seconds", 20.0),
            "--global-recovery-max-step", cfg.get("live_global_recovery_max_step", 0.55),
            "--output-max-step", cfg.get("live_output_max_step", 0.55),
            "--output-objective-threshold", cfg.get("live_output_objective_threshold", 30.0),
            "--prime", cfg["tsolve_prime"],
            "--degree", cfg["tsolve_degree"],
            "--action-weights", cfg["tsolve_action_weights"],
            "--tsolve-root-profile", cfg.get("tsolve_root_profile", "full"),
            "--fallback-action-weights", cfg["tsolve_fallback_action_weights"],
            "--partial-pose-out", partial_pose_path,
            "--replay-id", replay_id,
            "--expected-count", 0,
            "--scene-json", base_asset_dir / "scene.json",
            "--display-z-sign", selected.get("display_z_sign", -1),
            "--room-alignment-json", json.dumps(selected.get("room_alignment") or {}),
            "--follow-dir",
            "--stop-file", stop_file,
        ]
        if live_patrol_lock is not None and live_patrol_lock["reference_profile"] != "off":
            live_cmd.extend(
                [
                    "--rotation-position-stabilizer-profile",
                    live_patrol_lock["reference_profile"],
                    "--rotation-command-status-json",
                    public_root / "control_status.json",
                ]
            )
        if live_patrol_lock is not None and live_patrol_lock.get("baseline_reference_path"):
            live_cmd.extend(
                [
                    "--patrol-route-baseline",
                    live_patrol_lock["baseline_reference_path"],
                    "--patrol-route-max-cross-track",
                    cfg.get("live_patrol_route_max_cross_track", 0.55),
                    "--patrol-route-backward-tolerance",
                    cfg.get("live_patrol_route_backward_tolerance", 0.045),
                    "--patrol-status-max-age",
                    cfg.get("live_patrol_status_max_age", 5.0),
                    "--patrol-turn-max-position-drift",
                    cfg.get("live_patrol_turn_max_position_drift", 0.75),
                ]
            )
            if live_patrol_lock.get("visual_recovery_path"):
                live_cmd.extend(
                    [
                        "--patrol-visual-recovery-bank",
                        live_patrol_lock["visual_recovery_path"],
                    ]
                )
        for recovery_bank in active_taught_recovery_banks(base_asset_dir):
            live_cmd.extend(["--taught-patrol-recovery-bank", recovery_bank])
        if cfg.get("live_blocking_global_recovery", True):
            live_cmd.append("--blocking-global-recovery")
        if not cfg.get("live_background_recovery", False):
            live_cmd.append("--disable-background-recovery")
        if cfg.get("live_direct_pnp_recovery", True):
            live_cmd.append("--direct-pnp-recovery")
        fleet_run_cmd(drone_id, live_cmd, None)
        fleet_update(drone_id, stage="finalizing")
        fleet_event(drone_id, "Localization stopped; saving the fleet replay.")
        accepted = finalize_fleet_replay(
            drone_id=drone_id,
            selected=selected,
            replay_id=replay_id,
            replay_title=replay_title,
            out_asset_dir=out_asset_dir,
            partial_pose_path=partial_pose_path,
            query_frames=query_frames,
        )
        fleet_update(drone_id, status="done", stage="stopped", airborne=False, patrol_running=False)
        fleet_event(drone_id, f"Session saved with {accepted} accepted poses.", "ok")
    except Exception as exc:
        stopping = stop_event.is_set() or (FLEET_SESSIONS.get(drone_id) or {}).get("status") == "stopping"
        fleet_update(
            drone_id,
            status="done" if stopping else "error",
            stage="stopped" if stopping else "error",
            airborne=False,
            patrol_running=False,
            error=None if stopping else str(exc),
        )
        fleet_event(drone_id, "Session stopped by operator." if stopping else f"Session failed: {exc}", "info" if stopping else "error")
    finally:
        monitor_stop.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=2.0)
        stop_event.set()
        terminate_active_procs(f"fleet:{drone_id}")
        if bridge_thread is not None:
            bridge_thread.join(timeout=2.0)
        fleet_update(drone_id, active_processes=0)


def dispatch_fleet_drone(payload: dict) -> dict:
    drone_id = slugify_label(payload.get("drone_id"), "")
    map_id = str(payload.get("map_id") or "").strip()
    patrol_id = str(payload.get("patrol_id") or "").strip()
    if not drone_id or not map_id or not patrol_id:
        raise RuntimeError("Choose a drone, map, and saved patrol before dispatching.")
    manifest = load_fleet_manifest()
    drone = next((item for item in manifest["drones"] if item["id"] == drone_id), None)
    if not drone:
        raise RuntimeError(f"Unknown fleet drone: {drone_id}")
    map_entry = next((item for item in load_library().get("maps", []) if item.get("id") == map_id), None)
    if not map_entry:
        raise RuntimeError(f"Unknown map id: {map_id}")
    patrol = next((item for item in map_entry.get("patrols") or [] if item.get("id") == patrol_id), None)
    if not patrol:
        raise RuntimeError("The selected map does not contain that patrol.")
    if len(map_entry.get("safety_barriers") or []) < 3:
        raise RuntimeError("Dispatch requires a saved closed-wall geofence on the selected map.")
    fps = max(0.5, min(10.0, float(payload.get("fps") or 5.0)))
    max_size = max(640, min(1600, int(payload.get("max_size") or 1200)))
    with FLEET_LOCK:
        existing = FLEET_SESSIONS.get(drone_id)
        if existing and existing.get("status") in ACTIVE_JOB_STATES:
            raise RuntimeError(f"{drone['name']} already has an active session.")
        for other_id, other in FLEET_SESSIONS.items():
            if (
                other_id != drone_id
                and other.get("status") in ACTIVE_JOB_STATES
                and other.get("phone_ip") == drone["phone_ip"]
            ):
                raise RuntimeError(
                    f"Android endpoint {drone['phone_ip']} is already active for another drone. "
                    "Each simultaneous drone needs a separate phone/controller."
                )
        stop_event = threading.Event()
        session = {
            "drone_id": drone_id,
            "drone_name": drone["name"],
            "phone_ip": drone["phone_ip"],
            "map_id": map_id,
            "map_title": map_entry.get("title") or map_id,
            "patrol_id": patrol_id,
            "patrol_title": patrol.get("title") or patrol_id,
            "fps": fps,
            "max_size": max_size,
            "status": "queued",
            "stage": "queued",
            "message": "Fleet dispatch queued.",
            "events": [],
            "localization_ready": False,
            "patrol_running": False,
            "patrol_pending": False,
            "airborne": False,
            "control_pending": False,
            "takeoff_pending": False,
            "land_pending": False,
            "active_processes": 0,
            "created_at": time.time(),
            "updated_at": time.time(),
            "stop_event": stop_event,
            "thread": None,
        }
        FLEET_SESSIONS[drone_id] = session
        thread = threading.Thread(target=fleet_live_atlas_job, args=(drone_id,), daemon=True)
        session["thread"] = thread
        fleet_event(drone_id, f"Assigned to {session['map_title']} · {session['patrol_title']}.")
        thread.start()
        return fleet_session_snapshot(session)


def fleet_bridge_status(drone_id: str) -> dict:
    path = fleet_session_public_root(drone_id) / "status.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def fleet_write_control(drone_id: str, command: str, **fields) -> dict:
    command = str(command or "").strip().lower()
    if command not in {"takeoff", "land", "hover", "enable", "disable", "mission"}:
        raise RuntimeError(f"Unsupported fleet command: {command}")
    with FLEET_LOCK:
        session = FLEET_SESSIONS.get(drone_id)
        if not session or session.get("status") not in ACTIVE_JOB_STATES:
            raise RuntimeError("This drone does not have an active fleet session.")
        phone_ip = session["phone_ip"]
    status = fleet_bridge_status(drone_id)
    ready, reason = dji_live_bridge_readiness(status, command)
    if not ready:
        raise RuntimeError(f"{command.title()} is unavailable: {reason}")
    command_id = uuid.uuid4().hex
    payload = {
        "id": command_id,
        "command": command,
        "phone_ip": phone_ip,
        "created_at": time.time(),
        **fields,
    }
    atomic_write_json(fleet_session_public_root(drone_id) / "control_command.json", payload)
    fleet_update(
        drone_id,
        last_command=command,
        last_command_id=command_id,
        last_control_status="queued",
        last_control_message=f"{command.title()} command is waiting for the DJI bridge.",
        control_pending=True,
    )
    fleet_event(drone_id, f"{command.title()} command queued to the dedicated DJI bridge.", "ok")
    return {"queued": True, "command": command, "command_id": command_id}


def fleet_current_room_center(drone_id: str) -> list[float] | None:
    with FLEET_LOCK:
        rel = (FLEET_SESSIONS.get(drone_id) or {}).get("partial_pose_url")
    if not rel:
        return None
    try:
        payload = json.loads((VIEWER / rel).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for pose in reversed(payload.get("poses") or []):
        center = _vec3(pose.get("rcenter") if isinstance(pose, dict) else None)
        if center and bool(pose.get("success")):
            return center
    return None


def build_fleet_patrol_mission(drone_id: str) -> dict:
    with FLEET_LOCK:
        session = FLEET_SESSIONS.get(drone_id)
        if not session:
            raise RuntimeError("Fleet session no longer exists.")
        map_id = session["map_id"]
        patrol_id = session["patrol_id"]
    map_entry = next((item for item in load_library().get("maps", []) if item.get("id") == map_id), None)
    if not map_entry:
        raise RuntimeError("Assigned fleet map no longer exists.")
    patrol = next((item for item in map_entry.get("patrols") or [] if item.get("id") == patrol_id), None)
    if not patrol:
        raise RuntimeError("Assigned fleet patrol no longer exists.")
    raw_points = [_vec3(item.get("rxyz") if isinstance(item, dict) else item) for item in patrol.get("points") or []]
    raw_points = [item for item in raw_points if item]
    if len(raw_points) < 2:
        raise RuntimeError("The assigned patrol needs at least two valid points.")
    base_asset_dir = VIEWER / map_entry["asset_base"]
    try:
        scene = json.loads((base_asset_dir / "scene.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        scene = {}
    room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
    bounds = room.get("bounds") if isinstance(room.get("bounds"), dict) else {}
    floor_y = float(room.get("floorY") if room.get("floorY") is not None else (bounds.get("min") or [0, 0, 0])[1])
    altitude_m = max(0.3, min(2.0, float(patrol.get("altitude_m") or 1.0)))
    points = [[point[0], floor_y + altitude_m, point[2]] for point in raw_points]
    current = fleet_current_room_center(drone_id)
    entry_index = 0
    if current:
        entry_index = min(
            range(len(points)),
            key=lambda idx: math.hypot(points[idx][0] - current[0], points[idx][2] - current[2]),
        )
    mode = str(patrol.get("patrol_mode") or "circle")
    if mode == "back-and-forth":
        if entry_index < len(points) - 1:
            sequence = points[entry_index:] + list(reversed(points[:-1])) + points[1:entry_index + 1]
        else:
            sequence = list(reversed(points[:entry_index + 1])) + points[1:]
    else:
        sequence = [points[(entry_index + offset) % len(points)] for offset in range(len(points))]
        sequence.append(points[entry_index])
        mode = "circle"
    speed = max(0.04, min(0.20, float(patrol.get("speed") or 0.10)))
    dwell = max(0.8, min(8.0, float(patrol.get("dwell_s") or 2.0)))
    commands: list[dict] = [
        {
            "type": "gate",
            "title": "Fleet operator gate",
            "detail": "Operator armed this patrol from Fleet Monitor after live localization became valid.",
            "safety": "operator-confirmed",
        }
    ]
    arrival_indices: list[int] = []
    previous_heading: float | None = None
    for index in range(1, len(sequence)):
        start, end = sequence[index - 1], sequence[index]
        distance = math.sqrt(sum((end[axis] - start[axis]) ** 2 for axis in range(3)))
        if distance <= 1e-5:
            continue
        heading = math.degrees(math.atan2(end[2] - start[2], end[0] - start[0]))
        yaw_delta = 0.0 if previous_heading is None else ((heading - previous_heading + 180.0) % 360.0) - 180.0
        commands.extend(
            [
                {
                    "type": "yaw",
                    "title": f"Patrol yaw {index}",
                    "from": start,
                    "to": end,
                    "heading_deg": heading,
                    "reference_heading_deg": previous_heading,
                    "yaw_delta_deg": yaw_delta,
                    "duration_s": 1.2,
                    "safety": "slow-yaw",
                },
                {
                    "type": "cruise",
                    "title": f"Patrol cruise {index}",
                    "from": start,
                    "to": end,
                    "heading_deg": heading,
                    "speed_mps": speed,
                    "distance": distance,
                    "duration_s": distance / speed,
                    "safety": "patrol-speed",
                },
                {
                    "type": "hover",
                    "title": f"Patrol point {index}",
                    "point_index": index,
                    "at": end,
                    "duration_s": dwell,
                    "safety": "pose-check",
                },
            ]
        )
        arrival_indices.append(index)
        previous_heading = heading
    mission = {
        "client_safety_version": 3,
        "guided_enabled": True,
        "patrol": True,
        "operator_confirmed": True,
        "pose_max_age_seconds": 2.5,
        "pose_recovery_seconds": 45.0,
        "pulse_seconds": 0.30,
        "smooth_continuous_cruise": True,
        "cruise_window_seconds": 0.55,
        "cruise_pose_watchdog_seconds": 0.65,
        "max_forward_rc": 0.022,
        "max_lateral_rc": 0.010,
        "allow_lateral_rc": False,
        "allow_axis_auto_calibration": False,
        "axis_probe_rc": 0.018,
        "axis_probe_seconds": 0.55,
        "max_yaw_rc": 0.050,
        "max_scan_yaw_rc": 0.025,
        "allow_patrol_scan_yaw": False,
        "alignment_grace_seconds": 35.0,
        "max_vertical_rc": 0.018,
        "max_step_seconds": 2.0,
        "max_cruise_seconds": 120.0,
        "max_pose_step_map_units": 0.30,
        "max_pose_step_hard_map_units": 0.55,
        "cross_track_recovery_start_map_units": 0.30,
        "max_cross_track_map_units": 0.80,
        "arrival_radius_map_units": 0.24,
        "arrival_deadband_map_units": 0.14,
        "target_frame": "atlas_room",
        "map_id": map_id,
        "map_title": map_entry.get("title"),
        "patrol_id": patrol_id,
        "patrol_title": patrol.get("title"),
        "entry_index": entry_index,
        "points": points,
        "route": sequence,
        "route_segments": [{"from": sequence[i - 1], "to": sequence[i]} for i in range(1, len(sequence))],
        "arrival_indices": arrival_indices,
        "patrol_mode": mode,
        "loop": mode == "circle",
        "speed": speed,
        "altitude_y": floor_y + altitude_m,
        "altitude_m": altitude_m,
        "dwell_s": dwell,
        "scan_mode": patrol.get("scan_mode") or "forward",
        "commands": commands,
        "safety_barriers": map_entry.get("safety_barriers") or [],
        "safety_obstacles": map_entry.get("safety_obstacles") or [],
        "safety_motion_buffer_m": 0.30,
        "initial_pose_offset_room": [0.0, 0.0, 0.0],
        "confirmed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return validated_guarded_patrol_mission(mission)


def control_fleet_drone(payload: dict) -> dict:
    drone_id = slugify_label(payload.get("drone_id"), "")
    action = str(payload.get("action") or "").strip().lower()
    if not drone_id:
        raise RuntimeError("Choose a fleet drone.")
    if action == "takeoff":
        height = max(0.3, min(2.0, float(payload.get("height_m") or 1.0)))
        result = fleet_write_control(drone_id, "takeoff", height_m=height)
        fleet_update(drone_id, takeoff_pending=True)
        return result
    if action == "start_patrol":
        with FLEET_LOCK:
            session = FLEET_SESSIONS.get(drone_id) or {}
            if not session.get("localization_ready"):
                raise RuntimeError("Wait for at least one accepted live pose before starting this patrol.")
            if not session.get("airborne"):
                raise RuntimeError("Take off and confirm a stable live pose before starting this patrol.")
        mission = build_fleet_patrol_mission(drone_id)
        result = fleet_write_control(drone_id, "mission", mission=mission)
        fleet_update(drone_id, patrol_pending=True)
        fleet_event(
            drone_id,
            f"Patrol {mission['patrol_title']} queued from nearest point {mission['entry_index'] + 1}.",
            "ok",
        )
        return result
    if action in {"hover", "stop_patrol"}:
        result = fleet_write_control(drone_id, "hover", emergency_stop=True, fleet_stop=True)
        fleet_update(drone_id, patrol_running=False)
        fleet_event(drone_id, "Patrol stopped; the drone is holding position.", "warning")
        return result
    if action == "land":
        result = fleet_write_control(drone_id, "land")
        fleet_update(drone_id, patrol_running=False, land_pending=True)
        return result
    raise RuntimeError(f"Unsupported fleet action: {action}")


def stop_fleet_session(drone_id: str) -> bool:
    drone_id = slugify_label(drone_id, "")
    with FLEET_LOCK:
        session = FLEET_SESSIONS.get(drone_id)
        if not session or session.get("status") not in ACTIVE_JOB_STATES:
            return False
        session["status"] = "stopping"
        session["stage"] = "stopping"
        session["patrol_running"] = False
        stop_file = session.get("stop_file")
    try:
        atomic_write_json(
            fleet_session_public_root(drone_id) / "control_command.json",
            {
                "id": uuid.uuid4().hex,
                "command": "hover",
                "emergency_stop": True,
                "fleet_stop": True,
                "created_at": time.time(),
            },
        )
    except OSError:
        pass
    touch_stop_file(stop_file, "ATLAS Fleet Monitor stop requested")
    fleet_event(drone_id, "Stop requested; neutral hover is latched while the live path is saved.", "warning")

    def force_stop() -> None:
        time.sleep(4.0)
        with FLEET_LOCK:
            current = FLEET_SESSIONS.get(drone_id) or {}
            if current.get("status") != "stopping":
                return
            current.get("stop_event", threading.Event()).set()
        terminate_active_procs(f"fleet:{drone_id}")

    threading.Thread(target=force_stop, daemon=True).start()
    return True


def validate_extracted_video_coverage(
    query_frames: Path,
    *,
    min_temporal_coverage: float,
) -> dict:
    metadata_path = query_frames / "metadata.json"
    frames_path = query_frames / "frames.csv"
    if not metadata_path.is_file() or not frames_path.is_file():
        raise RuntimeError("Uploaded-video frame extraction did not produce complete metadata.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frame_count = int(metadata.get("frame_count") or 0)
    step_frames = max(1, int(metadata.get("step_frames") or 1))
    saved_frames = int(metadata.get("saved_frames") or 0)
    expected_full = ((frame_count - 1) // step_frames) + 1 if frame_count > 0 else 0
    with frames_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if saved_frames != len(rows):
        raise RuntimeError(
            f"Uploaded-video frame index mismatch: metadata says {saved_frames}, CSV has {len(rows)}."
        )
    if not rows or expected_full <= 0:
        raise RuntimeError("Uploaded video yielded no usable localization frames.")

    source_frames = [int(row.get("source_frame") or 0) for row in rows]
    if any(current <= previous for previous, current in zip(source_frames, source_frames[1:])):
        raise RuntimeError("Uploaded-video frame index is not strictly increasing.")
    internal_gaps = [
        current - previous
        for previous, current in zip(source_frames, source_frames[1:])
        if current - previous > step_frames
    ]
    if internal_gaps:
        raise RuntimeError(
            "Uploaded-video extraction has an internal frame gap: "
            f"largest source-frame step {max(internal_gaps)} > expected {step_frames}."
        )

    last_source_frame = source_frames[-1]
    temporal_coverage = min(1.0, (last_source_frame + step_frames) / frame_count)
    frame_coverage = min(1.0, saved_frames / expected_full)
    if saved_frames > expected_full:
        raise RuntimeError(
            "Uploaded-video extraction count exceeds the container estimate: "
            f"{saved_frames}/{expected_full} frames."
        )
    failed_requirements = []
    if frame_coverage < min_temporal_coverage:
        failed_requirements.append(
            f"frame coverage {frame_coverage:.3f} < {min_temporal_coverage:.3f}"
        )
    if temporal_coverage < min_temporal_coverage:
        failed_requirements.append(
            f"temporal coverage {temporal_coverage:.3f} < {min_temporal_coverage:.3f}"
        )
    if failed_requirements:
        raise RuntimeError(
            "Uploaded-video extraction is incomplete: "
            f"{saved_frames}/{expected_full} frames; " + "; ".join(failed_requirements) + "."
        )
    return {
        "frame_count": frame_count,
        "step_frames": step_frames,
        "saved_frames": saved_frames,
        "expected_full": expected_full,
        "duration_sec": float(metadata.get("duration_sec") or 0.0),
        "frame_coverage": frame_coverage,
        "temporal_coverage": temporal_coverage,
        "missing_tail_frames": max(0, expected_full - saved_frames),
    }


def validate_simulated_live_localization(
    summary_path: Path,
    pose_path: Path,
    *,
    expected_count: int,
    video_duration_sec: float,
    min_temporal_coverage: float,
    min_acceptance_ratio: float,
) -> dict:
    if not summary_path.is_file():
        raise RuntimeError("Uploaded-video localization summary is missing.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    query_count = int(summary.get("query_frames") or 0)
    processed_count = int(summary.get("processed_frames") or 0)
    solved_count = int(summary.get("accepted_cases") or 0)
    output_rejected_count = int(summary.get("output_rejected_cases") or 0)
    accepted_count = solved_count - output_rejected_count
    if query_count != expected_count or processed_count != expected_count:
        raise RuntimeError(
            "Uploaded-video localization stopped early: "
            f"processed {processed_count}/{expected_count} frames "
            f"(visible query frames: {query_count})."
        )
    acceptance_ratio = accepted_count / max(1, processed_count)
    if acceptance_ratio < min_acceptance_ratio:
        raise RuntimeError(
            "Uploaded-video localization acceptance is too low: "
            f"{accepted_count}/{processed_count} = {acceptance_ratio:.3f} "
            f"< {min_acceptance_ratio:.3f}."
        )

    if not pose_path.is_file():
        raise RuntimeError("Uploaded-video localization did not produce a final pose stream.")
    payload = json.loads(pose_path.read_text(encoding="utf-8"))
    poses = payload.get("poses", []) if isinstance(payload, dict) else payload
    if len(poses) != accepted_count:
        raise RuntimeError(
            "Uploaded-video final pose count does not match the guarded live output: "
            f"{len(poses)} exported, {accepted_count} accepted."
        )
    times = [
        float(pose["time_sec"])
        for pose in poses
        if isinstance(pose, dict)
        and pose.get("success") is not False
        and pose.get("time_sec") is not None
        and math.isfinite(float(pose["time_sec"]))
    ]
    if not times or video_duration_sec <= 0:
        raise RuntimeError("Uploaded-video localization has no timestamped successful poses.")
    if any(current <= previous for previous, current in zip(times, times[1:])):
        raise RuntimeError(
            "Uploaded-video pose stream is not in strictly increasing timestamp order."
        )
    centers = [pose.get("center") for pose in poses if isinstance(pose, dict)]
    if any(not isinstance(center, list) or len(center) < 3 for center in centers):
        raise RuntimeError("Uploaded-video final pose stream contains a missing camera center.")
    max_center_step = max(
        (
            math.sqrt(sum((float(right[index]) - float(left[index])) ** 2 for index in range(3)))
            for left, right in zip(centers, centers[1:])
        ),
        default=0.0,
    )
    if max_center_step > 0.55:
        raise RuntimeError(
            "Uploaded-video final pose stream contains an unguarded position jump: "
            f"{max_center_step:.3f}m > 0.550m."
        )
    temporal_coverage = max(0.0, min(1.0, (times[-1] - times[0]) / video_duration_sec))
    if temporal_coverage < min_temporal_coverage:
        raise RuntimeError(
            "Uploaded-video pose stream is temporally incomplete: "
            f"{temporal_coverage:.3f} < {min_temporal_coverage:.3f} "
            f"({times[0]:.3f}s to {times[-1]:.3f}s of {video_duration_sec:.3f}s)."
        )
    return {
        "query_frames": query_count,
        "processed_frames": processed_count,
        "accepted_cases": accepted_count,
        "solved_cases": solved_count,
        "output_rejected_cases": output_rejected_count,
        "acceptance_ratio": acceptance_ratio,
        "first_pose_time_sec": times[0],
        "last_pose_time_sec": times[-1],
        "temporal_coverage": temporal_coverage,
        "max_consecutive_center_step": max_center_step,
    }


def drone_video_job(
    video: Path,
    map_id: str | None = None,
    *,
    publish_to_map: bool = True,
) -> None:
    # The endpoint initializes the cancellation token before reserving this
    # worker; only final cleanup may acknowledge and clear a Stop request.
    mark_drone_job_worker_started()
    try:
        cfg = load_config()
        py = Path(cfg["python"])
        scripts = ROOT / "scripts"
        if publish_to_map:
            selected = set_selected_map(map_id) if map_id else selected_map_entry()
        else:
            selected = get_map_entry(map_id) if map_id else selected_map_entry()
        full_map_frames = frames_for_entry(selected)
        replay_id = make_map_id("replay" if publish_to_map else "camera_lab")
        base_asset_dir = map_asset_dir(selected)
        if publish_to_map:
            out_asset_dir = base_asset_dir / "replays" / replay_id
            run_root = ROOT / "results" / "drone_runs" / selected["id"] / replay_id
            query_frames = ROOT / "data" / "drone_runs" / selected["id"] / replay_id / "query_frames"
            replay_title = f"Drone Path {time.strftime('%H:%M:%S')}"
        else:
            out_asset_dir = CAMERA_PATH_LAB_DIR / "runs" / replay_id
            run_root = ROOT / "results" / "camera_path_lab_runs" / selected["id"] / replay_id
            query_frames = ROOT / "data" / "camera_path_lab_runs" / selected["id"] / replay_id / "query_frames"
            replay_title = f"Camera Track {time.strftime('%H:%M:%S')}"
        map_frames = make_frame_subset(
            full_map_frames,
            run_root / "map_frame_subset",
            int(cfg.get("live_demo_map_frame_cap", 0) or 0),
        )
        map_artifacts = colmap_artifacts_for_entry(selected)
        append_log("drone", f"Using COLMAP reference artifacts: {map_artifacts['root']}")
        tsolve_inputs = run_root / "tsolve_inputs"
        runtime_dir = run_root / "tsolve_runtime_code"
        tsolve_runtime = run_root / "tsolve_runtime"
        stream_work = run_root / "live_existing_map_stream"
        partial_pose_path = out_asset_dir / "poses_partial.json"
        partial_stop = threading.Event()
        partial_thread: threading.Thread | None = None
        live_started_at = time.time()

        def report(status: str, message: str) -> None:
            set_job("drone", status, message)
            if not publish_to_map:
                set_camera_path_lab_job(status, message)

        def start_stream(stream: dict) -> None:
            if publish_to_map:
                set_live_stream(stream)
            else:
                set_camera_path_lab_stream(stream)

        def publish_stream(**fields) -> None:
            if publish_to_map:
                update_live_stream(**fields)
            else:
                update_camera_path_lab_stream(**fields)

        report(
            "running",
            (
                "Reading uploaded drone video as a simulated live camera stream."
                if publish_to_map
                else "Starting fresh frame-by-frame localization from the uploaded phone video."
            ),
        )
        out_asset_dir.mkdir(parents=True, exist_ok=True)
        copy_video_to_public(video, out_asset_dir)
        start_stream(
            {
                "map_id": selected["id"],
                "replay_id": replay_id,
                "title": replay_title,
                "asset_base": public_rel(out_asset_dir),
                "partial_pose_url": public_rel(partial_pose_path),
                "media_url": public_rel(out_asset_dir / "media" / "drone_query.mp4"),
                "pose_count": 0,
                "expected_count": 0,
                "complete": False,
                "side_project": not publish_to_map,
                "fresh_localization": not publish_to_map,
                "source_map_title": selected.get("title") or selected["id"],
                "started_at": live_started_at,
                "first_pose_at": None,
                "first_pose_latency_seconds": None,
            }
        )
        simulated_query_fps = float(
            cfg.get("simulated_live_query_frame_fps", cfg["query_frame_fps"])
        )
        simulated_query_cap = int(cfg.get("simulated_live_query_frame_cap", 0) or 0)
        minimum_temporal_coverage = float(
            cfg.get("simulated_live_min_temporal_coverage", 0.98)
        )
        minimum_acceptance_ratio = float(
            cfg.get("simulated_live_min_acceptance_ratio", 0.75)
        )
        extract_cmd = [
            py,
            scripts / "extract_frames.py",
            "--video",
            video,
            "--out-dir",
            query_frames,
            "--fps",
            simulated_query_fps,
            "--max-size",
            cfg["max_image_size"],
            "--prefix",
            "query",
        ]
        if simulated_query_cap > 0:
            extract_cmd += ["--max-frames", simulated_query_cap]
        extraction_thread: threading.Thread | None = None
        extraction_errors: list[Exception] = []
        extraction_done_file = query_frames.parent / "extraction_complete.signal"
        extraction_done_file.unlink(missing_ok=True)
        extraction_validation: dict | None = None

        if publish_to_map:
            run_cmd("drone", extract_cmd, DRONE_STOP_EVENT)
            expected_count = count_frame_images(query_frames)
            extraction_validation = validate_extracted_video_coverage(
                query_frames,
                min_temporal_coverage=minimum_temporal_coverage,
            )
        else:
            # Camera Path Lab is a live producer/consumer demonstration: the
            # decoder publishes one complete JPEG and CSV row at a time while
            # the localizer follows the directory. Do not front-load several
            # minutes of 4K HEVC extraction before the first camera pose.
            def extract_live_frames() -> None:
                try:
                    run_cmd("drone", extract_cmd, DRONE_STOP_EVENT)
                except Exception as exc:
                    extraction_errors.append(exc)
                finally:
                    touch_stop_file(extraction_done_file, "uploaded-video extraction finished")

            extraction_thread = threading.Thread(
                target=extract_live_frames,
                name=f"camera-path-extractor-{replay_id}",
                daemon=True,
            )
            extraction_thread.start()
            metadata_path = query_frames / "metadata.json"
            metadata_deadline = time.time() + 30.0
            while not metadata_path.is_file() and not extraction_errors:
                if DRONE_STOP_EVENT.is_set():
                    raise RuntimeError("Camera Path live extraction stopped before metadata was ready.")
                if time.time() >= metadata_deadline:
                    raise RuntimeError("Camera Path live extraction did not publish video metadata within 30 seconds.")
                time.sleep(0.05)
            if extraction_errors:
                raise extraction_errors[0]
            extraction_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected_count = int(extraction_metadata.get("expected_frames") or 0)
            if simulated_query_cap > 0:
                expected_count = min(expected_count, simulated_query_cap)
            if expected_count <= 0:
                raise RuntimeError("Uploaded video contains no frames for live localization.")

        report(
            "running",
            (
                "Live camera localization: decoding and localizing incoming frames concurrently; "
                f"{expected_count} expected frames against the existing "
                f"{count_frame_images(map_frames)}/{count_frame_images(full_map_frames)}-frame map."
            ),
        )

        report("running", "Preparing TSolve online runtime; first frame initializes the branch/template.")
        run_cmd(
            "drone",
            [
                py,
                scripts / "setup_tsolve_runtime.py",
                "--base-yam-code-dir",
                cfg["base_yam_code_dir"],
                "--dropin-patch-dir",
                cfg["dropin_patch_dir"],
                "--base-harness-dir",
                str(ROOT.parent / "pnp-symbolic-research/Yam/exact_ff_ysolve_pnp/harness"),
                "--out-dir",
                runtime_dir,
            ],
            DRONE_STOP_EVENT,
        )

        atomic_write_json(
            partial_pose_path,
            {
                "mode": "simulated_live_tsolve_partial",
                "replay_id": replay_id,
                "frame_source": str(video),
                "expected_count": expected_count,
                "processed_count": 0,
                "complete": False,
                "updated_at": time.time(),
                "poses": [],
            },
        )
        publish_stream(
            partial_pose_url=public_rel(partial_pose_path),
            expected_count=expected_count,
            pose_count=0,
            complete=False,
        )

        localizer_mode = str(cfg.get("live_localizer_mode") or "bounded_tracking")
        stream_tracking_reference_images = cfg.get("live_tracking_reference_image_cap", 10)
        if localizer_mode == "colmap_per_frame":
            stream_script = scripts / "run_live_tsolve_existing_map_stream.py"
            stream_mode_message = "Running TSolve online R,t updates with COLMAP registration on each frame."
            extra_stream_args: list[object] = []
        else:
            # Uploaded-video replay is our finite, reproducible live simulation.
            # Keep it on the proven stable settings that produced the reference
            # 13:09 path: enough local tracks, and immediate recovery when the
            # drone lands or the image content changes abruptly.
            stream_tracking_reference_images = cfg.get("simulated_live_tracking_reference_image_cap", 10)
            stream_script = scripts / "run_bounded_tsolve_video_stream.py"
            replay_pace_scale = float(
                cfg.get("camera_path_lab_pace_scale", 0.25)
                if not publish_to_map
                else cfg.get("simulated_live_pace_scale", 1.0)
            )
            blocking_recovery = bool(
                cfg.get("camera_path_lab_blocking_global_recovery", False)
                if not publish_to_map
                else cfg.get("simulated_live_blocking_global_recovery", True)
            )
            stream_mode_message = (
                "Running buffered Camera Path localization ahead of video playback: "
                f"first-frame COLMAP bootstrap, optical-flow tracking at up to {1.0 / max(0.01, replay_pace_scale):.1f}x, "
                "with trusted-pose holds while map recovery runs at weak frames."
                if not publish_to_map
                else "Running bounded simulated-live TSolve: first-frame COLMAP bootstrap, then optical-flow tracking."
            )
            extra_stream_args = [
                "--track-pool-size",
                cfg.get("simulated_live_tracking_pool_size", 900),
                "--relocalize-every",
                cfg.get("live_relocalize_every", 0),
                "--flow-max-error",
                cfg.get("live_flow_max_error", 34.0),
                "--flow-backtrack-error",
                cfg.get("live_flow_backtrack_error", 2.5),
                "--flow-window",
                cfg.get("live_flow_window", 21),
                "--flow-levels",
                cfg.get("live_flow_levels", 3),
                "--flow-iterations",
                cfg.get("live_flow_iterations", 18),
                "--min-track-points",
                (
                    cfg.get("camera_path_lab_min_track_points", 6)
                    if not publish_to_map
                    else cfg.get("simulated_live_min_track_points", 60)
                ),
                "--min-track-ratio",
                cfg.get("simulated_live_min_track_ratio", cfg.get("live_min_track_ratio", 0.10)),
                "--global-recovery-after-failures",
                cfg.get("simulated_live_global_recovery_after_failures", 2),
                "--partial-pose-out",
                partial_pose_path,
                "--replay-id",
                replay_id,
                "--drone-video",
                video,
                "--expected-count",
                expected_count,
            ]
            if publish_to_map:
                extra_stream_args += ["--pace-replay", "--pace-scale", replay_pace_scale]
            else:
                extra_stream_args += [
                    "--global-recovery-max-step",
                    cfg.get("camera_path_lab_global_recovery_max_step", 1.6),
                    "--output-max-step",
                    cfg.get("camera_path_lab_output_max_step", 1.6),
                    "--output-max-speed",
                    cfg.get("camera_path_lab_output_max_speed", 2.2),
                    "--follow-dir",
                    "--follow-all-frames",
                    "--direct-pnp-recovery",
                    "--wait-for-background-recovery",
                    "--calibrate-output-to-first-global-anchor",
                    "--stop-file",
                    extraction_done_file,
                ]
            if blocking_recovery:
                extra_stream_args.append("--blocking-global-recovery")

        report("running", stream_mode_message)
        if publish_to_map:
            # The simulated-live replay still derives its viewer payload from
            # TSolve output files, so this legacy publisher owns the partial
            # JSON it writes.
            partial_thread = threading.Thread(
                target=stream_partial_poses,
                args=(
                    tsolve_runtime,
                    video,
                    partial_pose_path,
                    replay_id,
                    partial_stop,
                    expected_count,
                    map_artifacts["sparse_text"],
                    base_asset_dir / "scene.json",
                    selected.get("display_z_sign", -1),
                    selected.get("room_alignment"),
                    live_started_at,
                ),
                daemon=True,
            )
        else:
            # Camera Path localization writes accepted and held observations
            # directly to poses_partial.json.  Only observe that file here.
            # Rebuilding it concurrently from accepted-only TSolve outputs
            # discarded held rotations and made the browser appear frozen.
            partial_thread = threading.Thread(
                target=monitor_partial_pose_file,
                args=(
                    partial_pose_path,
                    partial_stop,
                    live_started_at,
                    tsolve_runtime / "live_stage_times.csv",
                    update_camera_path_lab_stream,
                    lambda message: report("running", message),
                ),
                daemon=True,
            )
        partial_thread.start()
        try:
            run_cmd(
                "drone",
                [
                    py,
                    stream_script,
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
                    runtime_dir,
                    "--solver-dir",
                    cfg["solver_dir"],
                    "--inputs-out-dir",
                    tsolve_inputs,
                    "--out-dir",
                    tsolve_runtime,
                    "--work-dir",
                    stream_work,
                    "--max-image-size",
                    cfg["max_image_size"],
                    "--query-camera-model",
                    cfg["query_camera_model"],
                    "--min-points",
                    cfg["min_query_correspondences"],
                    "--max-points",
                    cfg["max_query_correspondences"],
                    "--max-reference-images",
                    (
                        cfg.get("camera_path_lab_reference_image_cap", 36)
                        if not publish_to_map
                        else cfg.get("live_reference_image_cap", 24)
                    ),
                    "--tracking-reference-images",
                    stream_tracking_reference_images,
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
                    selected.get("display_z_sign", -1),
                    "--room-alignment-json",
                    json.dumps(selected.get("room_alignment") or {}),
                ]
                + extra_stream_args,
                DRONE_STOP_EVENT,
            )
        finally:
            partial_stop.set()
            if partial_thread is not None:
                partial_thread.join(timeout=4.0)

        if DRONE_STOP_EVENT.is_set():
            raise RuntimeError("Live TSolve path creation cancelled.")

        if extraction_thread is not None:
            extraction_thread.join(timeout=10.0)
            if extraction_thread.is_alive():
                raise RuntimeError("Camera Path extraction did not finish after the live localizer drained its frame queue.")
            if extraction_errors:
                raise extraction_errors[0]
            extraction_validation = validate_extracted_video_coverage(
                query_frames,
                min_temporal_coverage=minimum_temporal_coverage,
            )
        if extraction_validation is None:
            raise RuntimeError("Uploaded-video extraction validation was not completed.")

        report(
            "running",
            (
                "Exporting timestamped live R,t stream to the ATLAS viewer."
                if publish_to_map
                else "Exporting timestamped live R,t stream to the Camera Path Lab viewer."
            ),
        )
        run_cmd(
            "drone",
            [
                py,
                scripts / "build_viewer_data.py",
                "--localized-model-text",
                map_artifacts["sparse_text"],
                "--tsolve-runtime-dir",
                tsolve_runtime,
                "--drone-video",
                video,
                "--out-public",
                out_asset_dir,
            ],
            DRONE_STOP_EVENT,
        )
        counts = read_counts(out_asset_dir)
        localization_validation = validate_simulated_live_localization(
            tsolve_runtime / "live_stream_summary.json",
            out_asset_dir / "poses.json",
            expected_count=expected_count,
            video_duration_sec=extraction_validation["duration_sec"],
            min_temporal_coverage=minimum_temporal_coverage,
            min_acceptance_ratio=minimum_acceptance_ratio,
        )
        replay = {
            "id": replay_id,
            "title": replay_title,
            "asset_base": public_rel(out_asset_dir),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_video": video.name,
            "counts": {
                "poses": counts["poses"],
                "processed": localization_validation["processed_frames"],
                "accepted": localization_validation["accepted_cases"],
                "frames": extraction_validation["saved_frames"],
                "temporal_coverage": localization_validation["temporal_coverage"],
            },
        }
        publish_stream(
            pose_count=counts["poses"],
            expected_count=expected_count,
            partial_pose_url=public_rel(out_asset_dir / "poses.json"),
            final_pose_url=public_rel(out_asset_dir / "poses.json"),
            complete=True,
        )
        if publish_to_map:
            add_replay_to_map(selected["id"], replay, select=True)
            report("done", f"Live TSolve path ready: {replay_title} on {selected['title']}. Press Play Live Replay to play it.")
        else:
            report(
                "done",
                f"Camera path complete: {localization_validation['accepted_cases']} accepted frames on the read-only {selected['title']} mesh.",
            )
    except Exception as exc:
        append_log("drone", f"ERROR: {exc}")
        if DRONE_STOP_EVENT.is_set():
            if publish_to_map:
                update_live_stream(complete=True, cancelled=True)
            else:
                update_camera_path_lab_stream(complete=True, cancelled=True)
                set_camera_path_lab_job("cancelled", "Camera path creation cancelled.")
            set_job("drone", "cancelled", "Live TSolve path creation cancelled.")
        else:
            if publish_to_map:
                update_live_stream(
                    complete=True,
                    cancelled=False,
                    failed=True,
                    stopping=False,
                    error=str(exc),
                )
            else:
                failure_stream = {
                    "complete": True,
                    "cancelled": False,
                    "failed": True,
                    "stopping": False,
                    "error": str(exc),
                }
                failure_asset_dir = locals().get("out_asset_dir")
                if isinstance(failure_asset_dir, Path):
                    failure_pose_path = failure_asset_dir / "poses.json"
                    if failure_pose_path.is_file():
                        failure_counts = read_counts(failure_asset_dir)
                        failure_stream.update(
                            pose_count=failure_counts["poses"],
                            accepted_pose_count=failure_counts["poses"],
                            partial_pose_url=public_rel(failure_pose_path),
                            final_pose_url=public_rel(failure_pose_path),
                        )
                update_camera_path_lab_stream(
                    **failure_stream,
                )
                set_camera_path_lab_job("error", str(exc))
            set_job("drone", "error", str(exc))
    finally:
        release_drone_job()


def camera_path_lab_video_job(video: Path, map_id: str) -> None:
    drone_video_job(video, map_id, publish_to_map=False)


def start_thread(target, *args) -> None:
    thread = threading.Thread(target=target, args=args, daemon=True)
    thread.start()


class AtlasHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(VIEWER), **kwargs)

    def end_headers(self) -> None:
        # ATLAS is an actively edited local control application. Never let an
        # old HTML/JS bundle hide new flight-safety controls after a restart.
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def send_head(self):
        range_header = self.headers.get("Range")
        if not range_header:
            self._response_byte_range = None
            return super().send_head()
        path = Path(self.translate_path(self.path))
        if not path.is_file():
            self._response_byte_range = None
            return super().send_head()
        size = path.stat().st_size
        try:
            start, end = parse_http_byte_range(range_header, size)
        except (TypeError, ValueError):
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        handle = path.open("rb")
        self._response_byte_range = (start, end)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Last-Modified", self.date_time_string(path.stat().st_mtime))
        self.end_headers()
        return handle

    def copyfile(self, source, outputfile) -> None:
        selected_range = getattr(self, "_response_byte_range", None)
        if not selected_range:
            super().copyfile(source, outputfile)
            return
        start, end = selected_range
        source.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            block = source.read(min(128 * 1024, remaining))
            if not block:
                break
            outputfile.write(block)
            remaining -= len(block)

    def send_local_media(self, path: Path, *, head_only: bool = False) -> None:
        if not path.is_file():
            self.send_error(404, "Review video is not available.")
            return
        size = path.stat().st_size
        start = 0
        end = max(0, size - 1)
        status = 200
        range_header = self.headers.get("Range")
        if range_header:
            try:
                start, end = parse_http_byte_range(range_header, size)
                status = 206
            except (TypeError, ValueError):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
        self.send_response(status)
        self.send_header("Content-Type", self.guess_type(str(path)) or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Last-Modified", self.date_time_string(path.stat().st_mtime))
        self.end_headers()
        if head_only:
            return
        try:
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    block = handle.read(min(256 * 1024, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_HEAD(self) -> None:
        url = urllib.parse.urlparse(self.path)
        if url.path == "/api/video-review/media":
            self.send_local_media(VIDEO_REVIEW_MEDIA, head_only=True)
            return
        super().do_HEAD()

    def do_GET(self) -> None:
        url = urllib.parse.urlparse(self.path)
        if url.path == "/api/video-review/media":
            self.send_local_media(VIDEO_REVIEW_MEDIA)
            return
        if url.path == "/api/video-review":
            self.send_json({"ok": True, "review": load_video_review()})
            return
        if url.path == "/api/video-review/brief":
            review = load_video_review()
            content = video_review_markdown(review).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/markdown; charset=utf-8")
            self.send_header("Content-Disposition", f'inline; filename="{VIDEO_REVIEW_BRIEF.name}"')
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return
        if url.path == "/api/status":
            state = snapshot_state()
            query = urllib.parse.parse_qs(url.query)
            if (query.get("compact") or [""])[0] in {"1", "true", "yes"}:
                state.pop("library", None)
                for job_name in ("map", "drone", "enemy"):
                    job = state.get(job_name)
                    if isinstance(job, dict) and isinstance(job.get("log"), list):
                        job["log"] = job["log"][-4:]
            self.send_json(state)
            return
        if url.path == "/api/camera-path-lab/status":
            self.send_json({"ok": True, **camera_path_lab_snapshot()})
            return
        if url.path == "/api/live-replay":
            stream = current_live_stream()
            if not stream:
                with STATE_LOCK:
                    drone_state = json.loads(json.dumps(STATE["drone"]))
                if drone_state.get("status") in ACTIVE_JOB_STATES:
                    self.send_json(
                        {
                            "ok": True,
                            "mode": "simulated_live_tsolve_partial",
                            "processed_count": 0,
                            "expected_count": 0,
                            "complete": False,
                            "message": drone_state.get("message") or "Preparing TSolve stream.",
                            "poses": [],
                            "stream": None,
                        }
                    )
                    return
                self.send_json({"ok": False, "error": "No live TSolve replay stream is active.", "poses": []}, 404)
                return
            rel = stream.get("partial_pose_url") or stream.get("final_pose_url")
            try:
                pose_path = (VIEWER / str(rel)).resolve()
                if not pose_path.is_relative_to(VIEWER.resolve()):
                    raise RuntimeError("Invalid live stream path.")
                payload = json.loads(pose_path.read_text(encoding="utf-8")) if pose_path.exists() else {"poses": []}
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc), "stream": stream, "poses": []}, 200)
                return
            query = urllib.parse.parse_qs(url.query)
            after_values = query.get("after") or []
            if after_values:
                try:
                    requested_after = max(0, int(after_values[0]))
                except (TypeError, ValueError):
                    requested_after = 0
                all_poses = payload.get("poses") if isinstance(payload.get("poses"), list) else []
                delta_reset = requested_after > len(all_poses)
                delta_start = 0 if delta_reset else requested_after
                payload["poses"] = all_poses[delta_start:]
                payload["delta"] = True
                payload["delta_start"] = delta_start
                payload["delta_reset"] = delta_reset
                payload["returned_count"] = len(payload["poses"])
            payload["ok"] = True
            payload["stream"] = stream
            self.send_json(payload)
            return
        if url.path == "/api/maps":
            self.send_json(load_library())
            return
        if url.path == "/api/enemy-drones":
            payload = load_enemy_library()
            with STATE_LOCK:
                payload["job"] = json.loads(json.dumps(STATE["enemy"]))
            self.send_json(payload)
            return
        if url.path == "/api/fleet":
            payload = fleet_snapshot()
            payload["maps"] = [
                {
                    "id": item.get("id"),
                    "title": item.get("title") or item.get("id"),
                    "patrols": [
                        {
                            "id": patrol.get("id"),
                            "title": patrol.get("title") or patrol.get("id"),
                            "points": len(patrol.get("points") or []),
                            "speed": patrol.get("speed"),
                            "patrol_mode": patrol.get("patrol_mode"),
                        }
                        for patrol in item.get("patrols") or []
                    ],
                    "has_geofence": len(item.get("safety_barriers") or []) >= 3,
                }
                for item in load_library().get("maps", [])
                if item.get("patrols")
            ]
            self.send_json(payload)
            return
        super().do_GET()

    def do_POST(self) -> None:
        url = urllib.parse.urlparse(self.path)
        if url.path == "/api/video-review":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 2 * 1024 * 1024:
                    raise RuntimeError("Review payload is too large.")
                body = self.rfile.read(length).decode("utf-8") if length else "{}"
                review = save_video_review(json.loads(body or "{}"))
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
                return
            self.send_json(
                {
                    "ok": True,
                    "review": review,
                    "review_file": str(VIDEO_REVIEW_JSON.relative_to(ROOT)),
                    "brief_file": str(VIDEO_REVIEW_BRIEF.relative_to(ROOT)),
                    "brief_url": "/api/video-review/brief",
                }
            )
            return
        if url.path == "/api/enemy-drone/live-detection":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            try:
                request = json.loads(body or "{}")
                if not isinstance(request.get("enabled"), bool):
                    raise RuntimeError("enabled must be true or false")
                control = set_live_enemy_detection_enabled(request["enabled"])
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
                return
            self.send_json({"ok": True, "control": control})
            return

        if url.path == "/api/camera-path-lab/preview-calibration":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            try:
                calibration = save_camera_path_preview_calibration(json.loads(body or "{}"))
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
                return
            self.send_json({"ok": True, "calibration": calibration})
            return

        if url.path == "/api/fleet/drone":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            try:
                drone = upsert_fleet_drone(json.loads(body or "{}"))
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
                return
            self.send_json({"ok": True, "drone": drone, "fleet": fleet_snapshot()})
            return

        if url.path == "/api/fleet/drone/delete":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            try:
                payload = json.loads(body or "{}")
                delete_fleet_drone(str(payload.get("drone_id") or ""))
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
                return
            self.send_json({"ok": True, "fleet": fleet_snapshot()})
            return

        if url.path == "/api/fleet/dispatch":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            try:
                session = dispatch_fleet_drone(json.loads(body or "{}"))
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
                return
            self.send_json({"ok": True, "session": session, "fleet": fleet_snapshot()})
            return

        if url.path == "/api/fleet/control":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            try:
                result = control_fleet_drone(json.loads(body or "{}"))
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
                return
            self.send_json({"ok": True, **result, "fleet": fleet_snapshot()})
            return

        if url.path == "/api/fleet/stop":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            try:
                payload = json.loads(body or "{}")
                requested = str(payload.get("drone_id") or "").strip()
                stopped = []
                if requested:
                    if stop_fleet_session(requested):
                        stopped.append(requested)
                else:
                    for item in load_fleet_manifest()["drones"]:
                        if stop_fleet_session(item["id"]):
                            stopped.append(item["id"])
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
                return
            self.send_json({"ok": True, "stopped": stopped, "fleet": fleet_snapshot()})
            return

        if url.path == "/api/enemy-drone/upload":
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"})
            file_fields = uploaded_video_fields(form)
            try:
                profile = upload_enemy_videos(
                    str(form.getfirst("enemy_id", "")).strip() or None,
                    str(form.getfirst("name", "")).strip(),
                    file_fields,
                )
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
                return
            self.send_json({"ok": True, "enemy": profile, "library": load_enemy_library()})
            return

        if url.path == "/api/enemy-drone/rename":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body or "{}")
            try:
                profile = rename_enemy_profile(
                    str(payload.get("enemy_id", "")).strip(),
                    str(payload.get("name", "")).strip(),
                )
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
                return
            self.send_json({"ok": True, "enemy": profile, "library": load_enemy_library()})
            return

        if url.path == "/api/enemy-drone/delete":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body or "{}")
            try:
                delete_enemy_profile(str(payload.get("enemy_id", "")).strip())
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
                return
            self.send_json({"ok": True, "library": load_enemy_library()})
            return

        if url.path == "/api/enemy-drone/extract-frames":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body or "{}")
            try:
                result = extract_enemy_frames(
                    str(payload.get("enemy_id", "")).strip(),
                    float(payload.get("fps") or 2.0),
                    int(payload.get("max_frames_per_video") or 180),
                    int(payload.get("max_size") or 960),
                    bool(payload.get("force") or False),
                )
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
                return
            self.send_json({"ok": True, **result})
            return

        if url.path == "/api/enemy-drone/label-frame":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body or "{}")
            try:
                result = save_enemy_frame_label(
                    str(payload.get("enemy_id", "")).strip(),
                    str(payload.get("frame_id", "")).strip(),
                    payload.get("box") if isinstance(payload.get("box"), dict) else None,
                    str(payload.get("status") or "labeled"),
                )
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
                return
            self.send_json({"ok": True, **result})
            return

        if url.path == "/api/enemy-drone/track-labels":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body or "{}")
            try:
                result = track_enemy_labels(
                    str(payload.get("enemy_id", "")).strip(),
                    str(payload.get("frame_id", "")).strip(),
                    payload.get("box") if isinstance(payload.get("box"), dict) else None,
                    str(payload.get("direction") or "both"),
                    float(payload.get("accept_threshold") or 0.72),
                    float(payload.get("review_threshold") or 0.50),
                    float(payload.get("search_scale") or 3.0),
                    int(payload.get("max_frames") or 160),
                    bool(payload.get("overwrite") or False),
                )
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
                return
            self.send_json({"ok": True, **result})
            return

        if url.path == "/api/enemy-drone/range-sample":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body or "{}")
            try:
                result = capture_enemy_range_sample(
                    str(payload.get("enemy_id", "")).strip(),
                    float(payload.get("measured_clearance_m")),
                )
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
                return
            self.send_json({"ok": True, **result})
            return

        if url.path == "/api/enemy-drone/range-validate":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body or "{}")
            try:
                result = fit_enemy_range_calibration(
                    str(payload.get("enemy_id", "")).strip(),
                    float(payload.get("stop_clearance_m") or 0.50),
                )
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
                return
            self.send_json({"ok": True, **result})
            return

        if url.path == "/api/enemy-drone/range-reset":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body or "{}")
            try:
                result = reset_enemy_range_calibration(str(payload.get("enemy_id", "")).strip())
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
                return
            self.send_json({"ok": True, **result})
            return

        if url.path == "/api/enemy-drone/prepare-yolo":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body or "{}")
            try:
                lib = prepare_enemy_yolo_dataset(str(payload.get("enemy_id", "")).strip() or None)
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
                return
            self.send_json({"ok": True, "library": lib})
            return

        if url.path == "/api/enemy-drone/train-yolo":
            if job_is_active("enemy"):
                self.send_json({"ok": False, "error": "Enemy detector training is already running."}, 409)
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body or "{}")
            run_id = f"enemy_yolo_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            base_model = str(payload.get("base_model") or "yolov8n.pt").strip() or "yolov8n.pt"
            epochs = max(1, min(300, int(payload.get("epochs") or 50)))
            imgsz = max(160, min(1600, int(payload.get("imgsz") or 640)))
            batch = max(1, min(128, int(payload.get("batch") or 8)))
            device = str(payload.get("device") or "auto").strip() or "auto"
            queue_job("enemy", "Enemy-drone YOLO fine-tuning queued.")
            set_enemy_library_training_state(
                status="queued",
                message="Enemy detector fine-tuning queued.",
                run_id=run_id,
                training={
                    "status": "queued",
                    "run_id": run_id,
                    "base_model": base_model,
                    "epochs": epochs,
                    "imgsz": imgsz,
                    "batch": batch,
                    "device": device,
                    "queued_at": now_label(),
                },
            )
            thread = threading.Thread(
                target=enemy_yolo_training_job,
                kwargs={
                    "run_id": run_id,
                    "base_model": base_model,
                    "epochs": epochs,
                    "imgsz": imgsz,
                    "batch": batch,
                    "device": device,
                },
                daemon=True,
            )
            thread.start()
            payload = load_enemy_library()
            with STATE_LOCK:
                payload["job"] = json.loads(json.dumps(STATE["enemy"]))
            self.send_json({"ok": True, "library": payload})
            return

        if url.path == "/api/map/live":
            if job_is_active("map"):
                self.send_json({"ok": False, "error": "A map build is already running. Stop it or wait for it to finish."}, 409)
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            try:
                payload = json.loads(body or "{}")
            except json.JSONDecodeError:
                payload = {}
            duration = float(payload.get("duration", 75))
            fps = float(payload.get("fps", 1.5))
            camera_index = int(payload.get("camera_index", 0))
            queue_job("map", "Live camera map queued.")
            start_thread(live_map_job, duration, fps, camera_index)
            self.send_json({"ok": True, "state": snapshot_state()})
            return

        if url.path == "/api/map/stop":
            MAP_STOP_EVENT.set()
            set_job("map", "stopping", "Stopping map creation and terminating active COLMAP subprocesses.")
            terminated = terminate_active_procs("map")
            if terminated == 0:
                set_job("map", "cancelled", "Map creation stopped; no active map subprocess remained.")
                MAP_STOP_EVENT.clear()
            self.send_json({"ok": True, "state": snapshot_state()})
            return

        if url.path == "/api/drone/stop":
            camera_lab_active = camera_path_lab_snapshot().get("status") in ACTIVE_JOB_STATES
            if camera_lab_active:
                set_camera_path_lab_job("stopping", "Stopping camera path creation.")
                update_camera_path_lab_stream(stopping=True)
                set_job("drone", "stopping", "Cancelling Camera Path Lab localization.")
                DRONE_STOP_EVENT.set()
                terminated = terminate_active_procs("drone")
                if not DRONE_JOB_ACTIVE.is_set():
                    update_camera_path_lab_stream(complete=True, cancelled=True, stopping=False)
                    set_camera_path_lab_job(
                        "cancelled",
                        (
                            "Camera path creation stopped; the active localization process was terminated."
                            if terminated
                            else "Camera path creation stopped; no active localization process remained."
                        ),
                    )
                    set_job("drone", "cancelled", "Camera Path Lab localization stopped.")
                    release_drone_job()
                self.send_json({"ok": True, "state": snapshot_state()})
                return
            stream = current_live_stream() or {}
            if not stream:
                stream = recover_live_stream_from_disk() or {}
            worker_active = DRONE_JOB_ACTIVE.is_set()
            if worker_active:
                if stream.get("live_atlas"):
                    set_job("drone", "stopping", "Stopping DJI live localization and saving the current ATLAS path.")
                    touch_stop_file(stream.get("stop_file"), "ATLAS Stop Live Localization pressed")
                    update_live_stream(stopping=True, live_preview_url=None)
                    mark_live_dji_status_stopped("ATLAS live localization stop requested; saving current path.")
                else:
                    set_job("drone", "stopping", "Cancelling live TSolve path creation.")
                    update_live_stream(stopping=True)
            # The worker owns clearing this event after it has left every wait
            # loop and joined its background processes.
            DRONE_STOP_EVENT.set()
            terminated = terminate_active_procs("drone")
            orphan_count = terminate_orphan_live_drone_procs()
            if orphan_count:
                append_log("drone", f"Stopped {orphan_count} orphan live subprocess(es).")
            mark_live_dji_status_stopped("ATLAS live localization stopped by user.")
            stale_worker = (
                DRONE_JOB_ACTIVE.is_set()
                and drone_job_worker_liveness() is False
                and active_proc_count("drone") == 0
            )
            if stale_worker:
                append_log(
                    "drone",
                    "Recovered stale live-localization ownership after its worker thread exited.",
                )
                release_drone_job()
            if not DRONE_JOB_ACTIVE.is_set():
                update_live_stream(complete=True, cancelled=True, stopping=False, live_preview_url=None)
                if stale_worker:
                    set_job("drone", "cancelled", "Live localization stopped; stale worker state was cleared safely.")
                elif terminated or orphan_count:
                    set_job("drone", "cancelled", "Live localization stopped; active live subprocesses were terminated.")
                else:
                    set_job("drone", "cancelled", "Live localization stopped; no active drone worker remained.")
                release_drone_job()
            self.send_json({"ok": True, "state": snapshot_state()})
            return

        if url.path == "/api/drone/simulate-patrol-baseline":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            try:
                payload = json.loads(body or "{}")
                map_id = str(payload.get("map_id") or "").strip()
                baseline_replay_id = str(payload.get("baseline_replay_id") or "").strip()
                fps = max(0.5, min(30.0, float(payload.get("fps") or 10.0)))
                start_source_frame = payload.get("start_source_frame")
                end_source_frame = payload.get("end_source_frame")
                start_source_frame = int(start_source_frame) if start_source_frame is not None else None
                end_source_frame = int(end_source_frame) if end_source_frame is not None else None
                validation_only = payload.get("validation_only") is True
                recorded_timing = payload.get("recorded_timing") is True
                laps = max(1, min(2, int(payload.get("laps") or 1)))
                if not map_id or not baseline_replay_id:
                    raise ValueError("map_id and baseline_replay_id are required.")
                if (
                    start_source_frame is not None
                    and end_source_frame is not None
                    and start_source_frame > end_source_frame
                ):
                    raise ValueError("start_source_frame must be at or before end_source_frame.")
                if not reserve_and_queue_drone_job("Preparing a new simulated live patrol path."):
                    self.send_json({"ok": False, "error": "Another localization worker is active. Stop it before starting the simulation."}, 409)
                    return
                start_thread(
                    lambda: localized_recorded_patrol_job(
                        map_id=map_id,
                        baseline_replay_id=baseline_replay_id,
                        fps=fps,
                        start_source_frame=start_source_frame,
                        end_source_frame=end_source_frame,
                        validation_only=validation_only,
                        recorded_timing=recorded_timing,
                        laps=laps,
                    )
                )
            except Exception as exc:
                if DRONE_JOB_ACTIVE.is_set() and not job_is_active("drone"):
                    release_drone_job()
                self.send_json({"ok": False, "error": str(exc)}, 400)
                return
            self.send_json({"ok": True, "job": "simulated_patrol_baseline", "status": "queued"})
            return

        if url.path == "/api/drone/flight-command":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            try:
                payload = json.loads(body or "{}")
                result = send_dji_flight_command(payload)
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
                return
            self.send_json({"ok": True, **result, "state": snapshot_state()})
            return

        if url.path == "/api/drone/live-atlas":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            try:
                payload = json.loads(body or "{}")
            except json.JSONDecodeError:
                payload = {}
            map_id = str(payload.get("map_id", "")).strip() or None
            phone_ip = str(payload.get("phone_ip", "")).strip()
            fps = max(0.5, min(10.0, float(payload.get("fps", 10.0))))
            max_size = int(payload.get("max_size", 1200))
            view_only = bool(payload.get("view_only"))
            if not phone_ip:
                self.send_json({"ok": False, "error": "Missing phone_ip. Enter the Android MSDK phone IP first."}, 400)
                return
            if not reserve_and_queue_drone_job(f"Starting live ATLAS from Android phone {phone_ip}."):
                self.send_json({"ok": False, "error": "Drone live localization is already running. Stop it before starting another live ATLAS session."}, 409)
                return
            try:
                start_thread(
                    lambda: dji_live_atlas_job(
                        map_id=map_id,
                        phone_ip=phone_ip,
                        fps=fps,
                        max_size=max_size,
                        view_only=view_only,
                    )
                )
            except Exception as exc:
                release_drone_job()
                set_job("drone", "error", f"Could not start live localization worker: {exc}")
                self.send_json({"ok": False, "error": str(exc)}, 500)
                return
            self.send_json({"ok": True, "state": snapshot_state()})
            return

        if url.path == "/api/map/select":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body or "{}")
            entry = set_selected_map(str(payload.get("map_id", "")))
            self.send_json({"ok": True, "map": entry, "state": snapshot_state()})
            return

        if url.path == "/api/map/rename":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body or "{}")
            try:
                entry = rename_map_entry(str(payload.get("map_id", "")), str(payload.get("title", "")))
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
                return
            self.send_json({"ok": True, "map": entry, "state": snapshot_state()})
            return

        if url.path == "/api/map/duplicate":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body or "{}")
            try:
                entry = duplicate_map_entry(str(payload.get("map_id", "")))
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
                return
            self.send_json({"ok": True, "map": entry, "state": snapshot_state()})
            return

        if url.path == "/api/map/display-z":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body or "{}")
            try:
                entry = set_map_display_z_sign(
                    str(payload.get("map_id", "")),
                    payload.get("display_z_sign", 1),
                )
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
                return
            self.send_json({"ok": True, "map": entry, "state": snapshot_state()})
            return

        if url.path == "/api/map/barriers":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body or "{}")
            try:
                entry = set_map_safety_barriers(
                    str(payload.get("map_id", "")),
                    payload.get("barriers", []),
                    payload.get("obstacles", None),
                )
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
                return
            self.send_json({"ok": True, "map": entry, "state": snapshot_state()})
            return

        if url.path == "/api/map/patrols":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body or "{}")
            try:
                entry = set_map_patrols(
                    str(payload.get("map_id", "")),
                    payload.get("patrols", []),
                )
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
                return
            self.send_json({"ok": True, "map": entry, "state": snapshot_state()})
            return

        if url.path == "/api/map/patrol/import":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body or "{}")
            try:
                entry, patrol = import_map_patrol(
                    str(payload.get("target_map_id", "")),
                    str(payload.get("source_map_id", "")),
                    str(payload.get("patrol_id", "")),
                )
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
                return
            self.send_json(
                {
                    "ok": True,
                    "map": entry,
                    "patrol": patrol,
                    "state": snapshot_state(),
                }
            )
            return

        if url.path == "/api/manual-patrol/start":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body or "{}")
            try:
                recording = start_manual_patrol_recording(payload)
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
                return
            self.send_json({"ok": True, "recording": recording})
            return

        if url.path == "/api/manual-patrol/finish":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body or "{}")
            try:
                recording = finish_manual_patrol_recording(payload)
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
                return
            self.send_json({"ok": True, "recording": recording})
            return

        if url.path == "/api/replay/select":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body or "{}")
            try:
                entry = set_active_replay(str(payload.get("map_id", "")), str(payload.get("replay_id", "")))
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
                return
            self.send_json({"ok": True, "map": entry, "state": snapshot_state()})
            return

        if url.path == "/api/replay/rename":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body or "{}")
            try:
                entry = rename_replay_in_map(
                    str(payload.get("map_id", "")),
                    str(payload.get("replay_id", "")),
                    str(payload.get("title", "")),
                )
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
                return
            self.send_json({"ok": True, "map": entry, "state": snapshot_state()})
            return

        if url.path == "/api/replay/enhance-map":
            if job_is_active("map"):
                self.send_json({"ok": False, "error": "A map build is already running. Wait for it to finish before enhancing this map from a drone path."}, 409)
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body or "{}")
            map_id = str(payload.get("map_id", "")).strip()
            replay_id = str(payload.get("replay_id", "")).strip()
            if not map_id or not replay_id:
                self.send_json({"ok": False, "error": "Missing map_id or replay_id."}, 400)
                return
            try:
                entry = set_selected_map(map_id)
                replay = next((r for r in entry.get("replays", []) if r.get("id") == replay_id), None)
                replay_title = str(replay.get("title") if replay else replay_id)
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
                return
            queue_job("map", f"Enhancing {entry.get('title') or map_id} from drone path {replay_title}.")
            start_thread(enhance_map_from_replay_job, map_id, replay_id)
            self.send_json({"ok": True, "state": snapshot_state()})
            return

        if url.path == "/api/replay/delete":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body or "{}")
            try:
                entry = delete_replay_from_map(str(payload.get("map_id", "")), str(payload.get("replay_id", "")))
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
                return
            self.send_json({"ok": True, "map": entry, "state": snapshot_state()})
            return

        if url.path == "/api/map/delete":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body or "{}")
            try:
                delete_map_entry(str(payload.get("map_id", "")))
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
                return
            self.send_json({"ok": True, "state": snapshot_state()})
            return

        if url.path in {"/api/map/upload", "/api/map/enhance", "/api/drone/upload", "/api/camera-path-lab/upload"}:
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"})
            file_fields = uploaded_video_fields(form)
            if not file_fields:
                self.send_json({"ok": False, "error": "Missing video upload field."}, 400)
                return
            uploads = ROOT / "data" / "app_uploads"
            if url.path == "/api/map/upload":
                if job_is_active("map"):
                    self.send_json({"ok": False, "error": "A map build is already running. Wait for it to finish before uploading another map video."}, 409)
                    return
                saved = save_uploaded_videos(file_fields, uploads, "map_upload")
                names = ", ".join(name for _, name in saved[:3])
                if len(saved) > 3:
                    names += f", +{len(saved) - 3} more"
                queue_job("map", f"Uploaded {len(saved)} map video{'' if len(saved) == 1 else 's'}: {names}")
                start_thread(upload_map_job, [path for path, _ in saved])
            elif url.path == "/api/map/enhance":
                if job_is_active("map"):
                    self.send_json({"ok": False, "error": "A map build is already running. Wait for it to finish before enhancing this map again."}, 409)
                    return
                map_id = str(form.getfirst("map_id", "")).strip()
                if not map_id:
                    self.send_json({"ok": False, "error": "Missing map_id for map enhancement."}, 400)
                    return
                saved = save_uploaded_videos(file_fields, uploads, "map_enhance")
                names = ", ".join(name for _, name in saved[:3])
                if len(saved) > 3:
                    names += f", +{len(saved) - 3} more"
                queue_job("map", f"Enhancing map with {len(saved)} video{'' if len(saved) == 1 else 's'}: {names}")
                start_thread(enhance_map_job, map_id, [path for path, _ in saved])
            elif url.path == "/api/camera-path-lab/upload":
                file_field = file_fields[0]
                map_id = str(form.getfirst("map_id", "")).strip()
                if not map_id:
                    self.send_json({"ok": False, "error": "Missing reference map for Camera Path Lab."}, 400)
                    return
                try:
                    reference = get_map_entry(map_id)
                    map_asset_dir(reference)
                    colmap_artifacts_for_entry(reference)
                except Exception as exc:
                    self.send_json({"ok": False, "error": str(exc)}, 400)
                    return
                suffix = Path(file_field.filename).suffix or ".mp4"
                dst = uploads / f"camera_path_lab_{uuid.uuid4().hex[:8]}{suffix}"
                if not reserve_and_queue_drone_job(f"Camera Path Lab video: {file_field.filename}"):
                    self.send_json({"ok": False, "error": "Another localization worker is active. Wait for it to finish or stop it first."}, 409)
                    return
                set_camera_path_lab_stream(None)
                set_camera_path_lab_job("queued", f"Uploading {file_field.filename} against {reference.get('title') or map_id}.")
                try:
                    save_upload(file_field, dst)
                    start_thread(camera_path_lab_video_job, dst, map_id)
                except Exception as exc:
                    release_drone_job()
                    set_camera_path_lab_job("error", f"Could not start Camera Path Lab: {exc}")
                    set_job("drone", "error", f"Could not start Camera Path Lab: {exc}")
                    self.send_json({"ok": False, "error": str(exc)}, 500)
                    return
            else:
                file_field = file_fields[0]
                map_id = str(form.getfirst("map_id", "")).strip() or None
                suffix = Path(file_field.filename).suffix or ".mp4"
                dst = uploads / f"drone_upload_{uuid.uuid4().hex[:8]}{suffix}"
                if not reserve_and_queue_drone_job(f"Uploaded drone video: {file_field.filename}"):
                    self.send_json({"ok": False, "error": "Drone live localization is already running. Wait for it to finish before uploading another drone video."}, 409)
                    return
                try:
                    save_upload(file_field, dst)
                    start_thread(drone_video_job, dst, map_id)
                except Exception as exc:
                    release_drone_job()
                    set_job("drone", "error", f"Could not start drone path worker: {exc}")
                    self.send_json({"ok": False, "error": str(exc)}, 500)
                    return
            self.send_json({"ok": True, "state": snapshot_state()})
            return

        self.send_json({"ok": False, "error": f"Unknown endpoint: {url.path}"}, 404)


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve the local ATLAS app with map/drone pipeline API endpoints.")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("model/gltf-binary", ".glb")
    lib = load_library()
    selected_map_id = lib.get("selected_map_id")
    if selected_map_id:
        set_selected_map(selected_map_id)
    else:
        with STATE_LOCK:
            STATE["selected_map_id"] = ""
            STATE["current_map_frames"] = None
    server = ThreadingHTTPServer(("127.0.0.1", args.port), AtlasHandler)
    print(f"ATLAS app serving {VIEWER}")
    print(f"http://127.0.0.1:{args.port}")
    print("Use the start screen to create a map and upload a drone replay.")
    server.serve_forever()


if __name__ == "__main__":
    main()
