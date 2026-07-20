#!/usr/bin/env python3
from __future__ import annotations

import cgi
import argparse
import csv
import json
import math
import mimetypes
import os
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
CONFIG = ROOT / "config.json"
DEFAULT_PORT = 8765

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
ACTIVE_JOB_STATES = {"queued", "running", "stopping"}
COLMAP_QUERY_POSE_CACHE: dict[str, tuple[float, dict[str, dict]]] = {}
ACTIVE_PROCS_LOCK = threading.Lock()
ACTIVE_PROCS: dict[str, set[subprocess.Popen]] = {"map": set(), "drone": set(), "enemy": set()}


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
    status_paths = [PUBLIC / "live_dji" / "status.json"]
    try:
        latest = json.loads(status_paths[0].read_text(encoding="utf-8"))
        session = latest.get("session")
        if session:
            status_paths.append(PUBLIC / "live_dji_sessions" / str(session) / "status.json")
    except (OSError, json.JSONDecodeError):
        latest = {}
    for path in status_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else dict(latest)
            payload.update(
                {
                    "status": "stopped",
                    "message": message,
                    "stopped_at": time.time(),
                    "updated_at": time.time(),
                }
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            append_log("drone", f"Could not mark DJI live status stopped at {path}: {exc}")


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
        elif bool(pose.get("success")) and pose.get("center"):
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
    MAP_MANIFEST.write_text(json.dumps(lib, indent=2), encoding="utf-8")


def default_enemy_library() -> dict:
    return {
        "version": 1,
        "updated_at": now_label(),
        "selected_model": None,
        "model_status": "not_trained",
        "enemies": [],
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
        if status not in {"unlabeled", "labeled", "review", "skipped"}:
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
        "training_status": training_status,
        "model_status": str(profile.get("model_status") or "not_trained"),
        "dataset_manifest": str(profile.get("dataset_manifest") or ""),
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
    ENEMY_MANIFEST.write_text(json.dumps(lib, indent=2), encoding="utf-8")


def get_enemy_profile(enemy_id: str) -> tuple[dict, dict]:
    lib = load_enemy_library()
    for profile in lib.get("enemies", []):
        if profile.get("id") == enemy_id:
            return lib, profile
    raise RuntimeError(f"Unknown enemy drone profile: {enemy_id}")


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
    if status not in {"labeled", "review", "skipped", "unlabeled"}:
        raise RuntimeError("Frame label status must be labeled, review, skipped, or unlabeled.")
    label_path = enemy_public_path(frame.get("label_url") or "")
    label_path.parent.mkdir(parents=True, exist_ok=True)

    if status in {"labeled", "review"}:
        normalized = normalize_enemy_box(box)
        frame["box"] = normalized
        frame["status"] = status
        if status == "labeled":
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
                write_enemy_frame_label(lib, profile, frame, tracked_box, "labeled")
                current_template = image[py:py + t_h, px:px + t_w].copy()
                previous_box = tracked_box
                counts["labeled"] += 1
                counts["processed"] += 1
                processed += 1
                track_reports.append({"frame_id": frame.get("id"), "status": "labeled", "score": float(max_score)})
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
    dataset_dir = ENEMY_DIR / "yolo_dataset"
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    images_dir = dataset_dir / "images" / "train"
    labels_dir = dataset_dir / "labels" / "train"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    labeled_items = []
    target_ids = {profile["id"] for profile in targets}
    for profile in targets:
        for frame in profile.get("frames", []):
            if frame.get("status") != "labeled":
                continue
            image_path = enemy_public_path(frame.get("url") or "")
            box = frame.get("box") if isinstance(frame.get("box"), dict) else None
            if not image_path.exists() or not box:
                continue
            out_name = f"{profile['id']}_{frame['filename']}"
            shutil.copy2(image_path, images_dir / out_name)
            class_id = enemy_class_index(lib, profile["id"])
            labels_dir.joinpath(f"{Path(out_name).stem}.txt").write_text(
                (
                    f"{class_id} {float(box['x_center']):.8f} {float(box['y_center']):.8f} "
                    f"{float(box['width']):.8f} {float(box['height']):.8f}\n"
                ),
                encoding="utf-8",
            )
            labeled_items.append({"enemy_id": profile["id"], "frame_id": frame["id"], "image": out_name})
    if not labeled_items:
        raise RuntimeError("Extract frames and save at least one bounding-box label before preparing the YOLO dataset.")

    class_names = [profile["class_name"] for profile in lib.get("enemies", [])]
    yaml_path = dataset_dir / "data.yaml"
    yaml_path.write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/train\n"
        f"nc: {len(class_names)}\n"
        "names:\n"
        + "".join(f"  {idx}: {name}\n" for idx, name in enumerate(class_names)),
        encoding="utf-8",
    )
    dataset_manifest = {
        "version": 1,
        "prepared_at": now_label(),
        "status": "ready_for_training",
        "data_yaml": public_rel(yaml_path),
        "labeled_frame_count": len(labeled_items),
        "items": labeled_items,
        "classes": [
            {
                "id": profile["id"],
                "name": profile["name"],
                "class_name": profile["class_name"],
                "video_count": len(profile.get("videos", [])),
                "frame_count": len(profile.get("frames", [])),
                "labeled_frame_count": profile.get("labeled_frame_count", 0),
            }
            for profile in lib.get("enemies", [])
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
    lib["selected_model"] = None
    lib["training"] = {
        "status": "dataset_ready",
        "data_yaml": public_rel(yaml_path),
        "dataset_manifest": public_rel(manifest_path),
        "labeled_frame_count": len(labeled_items),
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
    entry = None
    for candidate in lib.get("maps", []):
        if candidate["id"] == map_id:
            entry = candidate
            break
    if entry is None:
        raise RuntimeError(f"Unknown map id: {map_id}")
    entry["patrols"] = sanitize_patrols(patrols)
    entry["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_library(lib)
    return entry


def delete_map_entry(map_id: str) -> None:
    lib = load_library()
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


def set_map_capture_state(**fields) -> None:
    with STATE_LOCK:
        STATE["map"].update(fields)
        STATE["map"]["updated_at"] = time.time()


def append_log(kind: str, line: str) -> None:
    line = line.rstrip()
    if not line:
        return
    with STATE_LOCK:
        log = STATE[kind]["log"]
        log.append(line)
        del log[:-240]
        STATE[kind]["updated_at"] = time.time()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


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


def build_room_transform_from_scene(scene_json: Path | None, display_z_sign: float = -1.0):
    """Match viewer/app.js buildRoomFrame() so patrol targets and TSolve poses share one frame."""
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
            return False, "DJI bridge was started without control enabled."
        return True, "DJI bridge is streaming and control-enabled."
    if command in {"takeoff", "land", "hover", "enable", "disable"}:
        if state not in {"streaming", "waiting_for_video"}:
            return False, f"DJI bridge is {state or 'offline'}."
        if age > 8.0:
            return False, f"DJI bridge heartbeat is stale ({age_text})."
        return True, "DJI bridge is connected."
    return False, f"Unsupported DJI command: {command}"


def send_dji_flight_command(payload: dict) -> dict:
    command = str(payload.get("command", "")).strip().lower()
    if command not in {"takeoff", "land", "enable", "disable", "hover", "mission"}:
        raise ValueError(f"Unsupported DJI flight command: {command}")
    phone_ip = str(payload.get("phone_ip", "")).strip()
    height_m = payload.get("height_m")
    if height_m is not None:
        height_m = max(0.1, min(2.0, float(height_m)))
    mission = payload.get("mission") if isinstance(payload.get("mission"), dict) else None

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
    started_at: float | None = None,
) -> None:
    room_transform = build_room_transform_from_scene(scene_json, display_z_sign)
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
            current = current_live_stream() or {}
            if count > 0 and started_at is not None and not current.get("first_pose_at"):
                stream_update["first_pose_at"] = now
                stream_update["first_pose_latency_seconds"] = now - started_at
            update_live_stream(**stream_update)
            if count != last_count and count > 0:
                set_job(
                    "drone",
                    "running",
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
    current = current_live_stream() or {}
    if final_count > 0 and started_at is not None and not current.get("first_pose_at"):
        now = time.time()
        stream_update["first_pose_at"] = now
        stream_update["first_pose_latency_seconds"] = now - started_at
    update_live_stream(
        **stream_update,
    )


def snapshot_state() -> dict:
    recover_live_stream_from_disk()
    with STATE_LOCK:
        state = json.loads(json.dumps(STATE))
    state["library"] = load_library()
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
) -> None:
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
            current = current_live_stream() or {}
            if count > 0 and started_at is not None and not current.get("first_pose_at"):
                now = time.time()
                stream_update["first_pose_at"] = now
                stream_update["first_pose_latency_seconds"] = now - started_at
            update_live_stream(**stream_update)
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
                update_live_stream(
                    latest_frame_index=frame_index,
                    latest_localization_reason=reason,
                    latest_total_frame_ms=total_ms,
                    message=msg,
                )
                set_job("drone", "running", msg)
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


def enhance_map_job(map_id: str, videos: list[Path]) -> None:
    try:
        if not videos:
            raise RuntimeError("No enhancement videos were provided.")
        cfg = load_config()
        py = Path(cfg["python"])
        scripts = ROOT / "scripts"
        selected = get_map_entry(map_id)
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
    fps: float = 2.0,
    max_size: int = 1200,
) -> None:
    DRONE_STOP_EVENT.clear()
    bridge_thread: threading.Thread | None = None
    monitor_thread: threading.Thread | None = None
    monitor_stop = threading.Event()
    try:
        if not phone_ip.strip():
            raise RuntimeError("Missing Android phone IP for DJI live ATLAS.")
        cfg = load_config()
        py = Path(cfg["python"])
        scripts = ROOT / "scripts"
        selected = set_selected_map(map_id) if map_id else selected_map_entry()
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
        enemy_model = selected_enemy_model_path()
        if enemy_model:
            bridge_cmd.extend(
                [
                    "--enemy-model",
                    enemy_model,
                    "--enemy-output",
                    PUBLIC / "live_dji" / "enemy_detections.json",
                    "--enemy-detect-fps",
                    cfg.get("enemy_live_detect_fps", min(float(fps), 2.0)),
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

        wait_started = time.time()
        while not DRONE_STOP_EVENT.is_set() and count_frame_images(query_frames) < 1:
            if live_latest.exists():
                update_live_stream(live_preview_url="public/live_dji/latest.jpg")
            if time.time() - wait_started > 60:
                raise RuntimeError("DJI bridge connected, but no live frames arrived within 60 seconds.")
            set_job("drone", "running", "Waiting for first DJI frame from Android MSDK stream.")
            time.sleep(0.5)

        if DRONE_STOP_EVENT.is_set():
            raise RuntimeError("Live ATLAS localization stopped before first frame.")

        set_job("drone", "running", "First DJI frame received. Running TSolve live self-localization.")
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
            "--global-recovery-after-failures",
            cfg.get("live_global_recovery_after_failures", 2),
            "--prime",
            cfg["tsolve_prime"],
            "--degree",
            cfg["tsolve_degree"],
            "--action-weights",
            cfg["tsolve_action_weights"],
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
            "--follow-dir",
            "--stop-file",
            stop_file,
        ]
        if cfg.get("live_blocking_global_recovery", True):
            live_stream_cmd.append("--blocking-global-recovery")
        if not cfg.get("live_background_recovery", False):
            live_stream_cmd.append("--disable-background-recovery")
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
            DRONE_STOP_EVENT.clear()
        else:
            set_job("drone", "error", str(exc))
    finally:
        monitor_stop.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=2.0)
        if DRONE_STOP_EVENT.is_set():
            terminate_active_procs("drone")
            DRONE_STOP_EVENT.clear()
        if bridge_thread is not None:
            bridge_thread.join(timeout=1.0)


def drone_video_job(video: Path, map_id: str | None = None) -> None:
    DRONE_STOP_EVENT.clear()
    try:
        cfg = load_config()
        py = Path(cfg["python"])
        scripts = ROOT / "scripts"
        selected = set_selected_map(map_id) if map_id else selected_map_entry()
        full_map_frames = frames_for_entry(selected)
        replay_id = make_map_id("replay")
        base_asset_dir = VIEWER / selected["asset_base"]
        if not base_asset_dir.exists():
            base_asset_dir = MAPS_DIR / selected["id"]
        out_asset_dir = base_asset_dir / "replays" / replay_id
        run_root = ROOT / "results" / "drone_runs" / selected["id"] / replay_id
        map_frames = make_frame_subset(
            full_map_frames,
            run_root / "map_frame_subset",
            int(cfg.get("live_demo_map_frame_cap", 0) or 0),
        )
        map_artifacts = colmap_artifacts_for_entry(selected)
        append_log("drone", f"Using COLMAP reference artifacts: {map_artifacts['root']}")
        query_frames = ROOT / "data" / "drone_runs" / selected["id"] / replay_id / "query_frames"
        tsolve_inputs = run_root / "tsolve_inputs"
        runtime_dir = run_root / "tsolve_runtime_code"
        tsolve_runtime = run_root / "tsolve_runtime"
        stream_work = run_root / "live_existing_map_stream"
        replay_title = f"Drone Path {time.strftime('%H:%M:%S')}"
        partial_pose_path = out_asset_dir / "poses_partial.json"
        partial_stop = threading.Event()
        partial_thread: threading.Thread | None = None
        live_started_at = time.time()

        set_job("drone", "running", "Reading uploaded drone video as a simulated live camera stream.")
        out_asset_dir.mkdir(parents=True, exist_ok=True)
        copy_video_to_public(video, out_asset_dir)
        set_live_stream(
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
                "started_at": live_started_at,
                "first_pose_at": None,
                "first_pose_latency_seconds": None,
            }
        )
        extract_cmd = [
            py,
            scripts / "extract_frames.py",
            "--video",
            video,
            "--out-dir",
            query_frames,
            "--fps",
            cfg["query_frame_fps"],
            "--max-size",
            cfg["max_image_size"],
            "--prefix",
            "query",
        ]
        query_cap = int(cfg.get("live_demo_query_frame_cap", 0) or 0)
        if query_cap > 0:
            extract_cmd += ["--max-frames", query_cap]
        run_cmd("drone", extract_cmd, DRONE_STOP_EVENT)

        expected_count = count_frame_images(query_frames)
        set_job(
            "drone",
            "running",
            (
                "Live replay localization: "
                f"{expected_count} incoming drone frames against the existing "
                f"{count_frame_images(map_frames)}/{count_frame_images(full_map_frames)}-frame map."
            ),
        )

        set_job("drone", "running", "Preparing TSolve online runtime; first frame initializes the branch/template.")
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
        update_live_stream(
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
            stream_mode_message = "Running bounded simulated-live TSolve: first-frame COLMAP bootstrap, then optical-flow tracking."
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
                cfg.get("simulated_live_min_track_points", 60),
                "--min-track-ratio",
                cfg.get("simulated_live_min_track_ratio", cfg.get("live_min_track_ratio", 0.10)),
                "--global-recovery-after-failures",
                cfg.get("simulated_live_global_recovery_after_failures", 2),
                "--pace-replay",
                "--pace-scale",
                cfg.get("simulated_live_pace_scale", 1.0),
                "--partial-pose-out",
                partial_pose_path,
                "--replay-id",
                replay_id,
                "--drone-video",
                video,
                "--expected-count",
                expected_count,
            ]
            if cfg.get("simulated_live_blocking_global_recovery", True):
                extra_stream_args.append("--blocking-global-recovery")

        set_job("drone", "running", stream_mode_message)
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
                live_started_at,
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
                    cfg.get("live_reference_image_cap", 24),
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

        set_job("drone", "running", "Exporting timestamped live R,t stream to the ATLAS viewer.")
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
        replay = {
            "id": replay_id,
            "title": replay_title,
            "asset_base": public_rel(out_asset_dir),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_video": video.name,
            "counts": {"poses": counts["poses"]},
        }
        update_live_stream(
            pose_count=counts["poses"],
            expected_count=expected_count,
            partial_pose_url=public_rel(out_asset_dir / "poses.json"),
            final_pose_url=public_rel(out_asset_dir / "poses.json"),
            complete=True,
        )
        add_replay_to_map(selected["id"], replay, select=True)
        set_job("drone", "done", f"Live TSolve path ready: {replay_title} on {selected['title']}. Press Play Live Replay to play it.")
    except Exception as exc:
        append_log("drone", f"ERROR: {exc}")
        if DRONE_STOP_EVENT.is_set():
            update_live_stream(complete=True, cancelled=True)
            set_job("drone", "cancelled", "Live TSolve path creation cancelled.")
            DRONE_STOP_EVENT.clear()
        else:
            set_job("drone", "error", str(exc))


def start_thread(target, *args) -> None:
    thread = threading.Thread(target=target, args=args, daemon=True)
    thread.start()


class AtlasHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(VIEWER), **kwargs)

    def send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        url = urllib.parse.urlparse(self.path)
        if url.path == "/api/status":
            self.send_json(snapshot_state())
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
        super().do_GET()

    def do_POST(self) -> None:
        url = urllib.parse.urlparse(self.path)
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
            DRONE_STOP_EVENT.set()
            stream = current_live_stream() or {}
            if not stream:
                stream = recover_live_stream_from_disk() or {}
            if stream.get("live_atlas"):
                set_job("drone", "stopping", "Stopping DJI live localization and saving the current ATLAS path.")
                touch_stop_file(stream.get("stop_file"), "ATLAS Stop Live Localization pressed")
                update_live_stream(stopping=True, live_preview_url=None)
                mark_live_dji_status_stopped("ATLAS live localization stop requested; saving current path.")
            else:
                set_job("drone", "stopping", "Cancelling live TSolve path creation.")
                update_live_stream(complete=True, cancelled=True)
            terminated = terminate_active_procs("drone")
            orphan_count = terminate_orphan_live_drone_procs()
            if orphan_count:
                append_log("drone", f"Stopped {orphan_count} orphan live subprocess(es).")
            mark_live_dji_status_stopped("ATLAS live localization stopped by user.")
            if terminated or orphan_count:
                update_live_stream(complete=True, cancelled=True, stopping=False, live_preview_url=None)
                set_job("drone", "cancelled", "Live localization stopped; active live subprocesses were terminated.")
                DRONE_STOP_EVENT.clear()
            else:
                update_live_stream(complete=True, cancelled=True, stopping=False, live_preview_url=None)
                set_job("drone", "cancelled", "Live localization stopped; no active drone subprocess remained.")
                DRONE_STOP_EVENT.clear()
            self.send_json({"ok": True, "state": snapshot_state()})
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
            if job_is_active("drone"):
                self.send_json({"ok": False, "error": "Drone live localization is already running. Stop it before starting another live ATLAS session."}, 409)
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            try:
                payload = json.loads(body or "{}")
            except json.JSONDecodeError:
                payload = {}
            map_id = str(payload.get("map_id", "")).strip() or None
            phone_ip = str(payload.get("phone_ip", "")).strip()
            fps = float(payload.get("fps", 2.0))
            max_size = int(payload.get("max_size", 1200))
            if not phone_ip:
                self.send_json({"ok": False, "error": "Missing phone_ip. Enter the Android MSDK phone IP first."}, 400)
                return
            queue_job("drone", f"Starting live ATLAS from Android phone {phone_ip}.")
            start_thread(
                lambda: dji_live_atlas_job(
                    map_id=map_id,
                    phone_ip=phone_ip,
                    fps=fps,
                    max_size=max_size,
                )
            )
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

        if url.path in {"/api/map/upload", "/api/map/enhance", "/api/drone/upload"}:
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
            else:
                if job_is_active("drone"):
                    self.send_json({"ok": False, "error": "Drone live localization is already running. Wait for it to finish before uploading another drone video."}, 409)
                    return
                file_field = file_fields[0]
                map_id = str(form.getfirst("map_id", "")).strip() or None
                suffix = Path(file_field.filename).suffix or ".mp4"
                dst = uploads / f"drone_upload_{uuid.uuid4().hex[:8]}{suffix}"
                save_upload(file_field, dst)
                queue_job("drone", f"Uploaded drone video: {file_field.filename}")
                start_thread(drone_video_job, dst, map_id)
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
