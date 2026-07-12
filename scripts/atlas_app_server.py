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
    "current_map_frames": None,
    "selected_map_id": "default_demo",
}
MAP_STOP_EVENT = threading.Event()
DRONE_STOP_EVENT = threading.Event()
ACTIVE_JOB_STATES = {"queued", "running", "stopping"}
COLMAP_QUERY_POSE_CACHE: dict[str, tuple[float, dict[str, dict]]] = {}
ACTIVE_PROCS_LOCK = threading.Lock()
ACTIVE_PROCS: dict[str, set[subprocess.Popen]] = {"map": set(), "drone": set()}


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
            append_log("drone", f"Could not stop orphan live process {pid}: {exc}")
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


def read_counts(asset_dir: Path) -> dict:
    scene_path = asset_dir / "scene.json"
    pose_path = asset_dir / "poses.json"
    counts = {"points": 0, "dense_points": 0, "cameras": 0, "poses": 0}
    if scene_path.exists():
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
        counts["points"] = len(scene.get("points3D", []))
        counts["dense_points"] = len(scene.get("dense_points3D", []))
        counts["cameras"] = len(scene.get("map_cameras", []))
    if pose_path.exists():
        poses = json.loads(pose_path.read_text(encoding="utf-8"))
        counts["poses"] = len(poses.get("poses", []))
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
                "created_at": str(raw.get("created_at") or now),
                "updated_at": now,
            }
        )
    return barriers


def set_map_safety_barriers(map_id: str, barriers) -> dict:
    lib = load_library()
    entry = None
    for candidate in lib.get("maps", []):
        if candidate["id"] == map_id:
            entry = candidate
            break
    if entry is None:
        raise RuntimeError(f"Unknown map id: {map_id}")
    entry["safety_barriers"] = sanitize_safety_barriers(barriers)
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
            poses.append(
                {
                    "instance_id": instance_id,
                    "success": bool(result.get("success")),
                    "time_sec": meta.get("time_sec"),
                    "image_name": meta.get("image_name"),
                    "R": R,
                    "t": t,
                    "center": center,
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


def send_dji_flight_command(payload: dict) -> dict:
    command = str(payload.get("command", "")).strip().lower()
    if command not in {"takeoff", "land", "enable", "disable", "hover"}:
        raise ValueError(f"Unsupported DJI flight command: {command}")
    phone_ip = str(payload.get("phone_ip", "")).strip()
    height_m = payload.get("height_m")
    if height_m is not None:
        height_m = max(0.1, min(2.0, float(height_m)))

    stream = current_live_stream() or recover_live_stream_from_disk() or {}
    live_status_path = PUBLIC / "live_dji" / "status.json"
    use_live_bridge = bool(stream.get("live_atlas") and live_status_path.exists())
    command_id = uuid.uuid4().hex
    command_payload = {
        "id": command_id,
        "command": command,
        "phone_ip": phone_ip or stream.get("phone_ip"),
        "height_m": height_m,
        "created_at": time.time(),
    }
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
        stream["pose_count"] = int(payload.get("processed_count") or len(poses))
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
    started_at: float | None = None,
) -> None:
    last_count = -1
    last_write = 0.0
    while not stop_event.is_set():
        payload = build_partial_pose_payload(
            tsolve_runtime,
            drone_video,
            replay_id,
            expected_count,
            localized_model_text,
        )
        try:
            existing = json.loads(partial_path.read_text(encoding="utf-8")) if partial_path.exists() else {}
        except (OSError, json.JSONDecodeError):
            existing = {}
        if existing.get("current_frame"):
            payload["current_frame"] = existing["current_frame"]
            payload["current_frame_time_sec"] = existing.get("current_frame_time_sec")
        count = int(payload.get("processed_count") or 0)
        now = time.time()
        if count != last_count or now - last_write > 2.5:
            atomic_write_json(partial_path, payload)
            stream_update = {
                "pose_count": count,
                "expected_count": expected_count,
                "partial_pose_url": public_rel(partial_path),
            }
            current = current_live_stream() or {}
            if count > 0 and started_at is not None and not current.get("first_pose_at"):
                stream_update["first_pose_at"] = now
                stream_update["first_pose_latency_seconds"] = now - started_at
            update_live_stream(**stream_update)
            if count != last_count and count > 0:
                set_job("drone", "running", f"Live TSolve self-localization: streamed {count}/{expected_count or '?'} R,t updates.")
            last_count = count
            last_write = now
        time.sleep(0.35)

    payload = build_partial_pose_payload(
        tsolve_runtime,
        drone_video,
        replay_id,
        expected_count,
        localized_model_text,
    )
    try:
        existing = json.loads(partial_path.read_text(encoding="utf-8")) if partial_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        existing = {}
    if existing.get("current_frame"):
        payload["current_frame"] = existing["current_frame"]
        payload["current_frame_time_sec"] = existing.get("current_frame_time_sec")
    atomic_write_json(partial_path, payload)
    final_count = int(payload.get("processed_count") or 0)
    stream_update = {
        "pose_count": final_count,
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
        count = int(payload.get("processed_count") or len(poses))
        if count != last_count:
            stream_update = {
                "pose_count": count,
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
    final_payload = {
        **payload,
        "mode": "dji_live_tsolve_replay" if payload.get("mode") else "atlas_live_tsolve_replay",
        "description": "ATLAS TSolve R,t estimates produced from DJI MSDK live frames.",
        "complete": True,
        "processed_count": len(poses),
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
        "counts": {"poses": len(poses)},
    }
    if query_frame_base_url:
        replay["query_frame_base_url"] = query_frame_base_url
    update_live_stream(
        pose_count=len(poses),
        expected_count=int(payload.get("expected_count") or 0),
        partial_pose_url=public_rel(final_pose_path),
        final_pose_url=public_rel(final_pose_path),
        complete=True,
    )
    add_replay_to_map(selected["id"], replay, select=True)
    return len(poses)


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
        root = ROOT / "results" / "maps" / map_id / "colmap"
        database = root / "database.db"
        images = root / "images"
        sparse_text = root / "sparse_text"
        sparse_root = root / "sparse"
        if (
            database.exists()
            and images.exists()
            and (sparse_text / "points3D.txt").exists()
            and sparse_root.exists()
        ):
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


def copy_frame_bank(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    try:
        if src.resolve() == dst.resolve():
            return
    except FileNotFoundError:
        pass
    for item in src.iterdir():
        if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            target = dst / item.name
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
        ]
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

        run_cmd(
            "drone",
            [
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
                "--follow-dir",
                "--stop-file",
                stop_file,
            ],
            DRONE_STOP_EVENT,
        )

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
        if localizer_mode == "colmap_per_frame":
            stream_script = scripts / "run_live_tsolve_existing_map_stream.py"
            stream_mode_message = "Running TSolve online R,t updates with COLMAP registration on each frame."
            extra_stream_args: list[object] = []
        else:
            stream_script = scripts / "run_bounded_tsolve_video_stream.py"
            stream_mode_message = "Running bounded simulated-live TSolve: first-frame COLMAP bootstrap, then optical-flow tracking."
            extra_stream_args = [
                "--track-pool-size",
                cfg.get("live_tracking_pool_size", 900),
                "--relocalize-every",
                cfg.get("live_relocalize_every", 0),
                "--flow-max-error",
                cfg.get("live_flow_max_error", 34.0),
                "--flow-backtrack-error",
                cfg.get("live_flow_backtrack_error", 2.5),
                "--partial-pose-out",
                partial_pose_path,
                "--replay-id",
                replay_id,
                "--drone-video",
                video,
                "--expected-count",
                expected_count,
            ]

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
                    cfg.get("live_tracking_reference_image_cap", 10),
                    "--prime",
                    cfg["tsolve_prime"],
                    "--degree",
                    cfg["tsolve_degree"],
                    "--action-weights",
                    cfg["tsolve_action_weights"],
                    "--fallback-action-weights",
                    cfg["tsolve_fallback_action_weights"],
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
        super().do_GET()

    def do_POST(self) -> None:
        url = urllib.parse.urlparse(self.path)
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
            else:
                set_job("drone", "stopping", "Cancelling live TSolve path creation.")
                update_live_stream(complete=True, cancelled=True)
            terminated = terminate_active_procs("drone")
            orphan_count = terminate_orphan_live_drone_procs()
            if orphan_count:
                append_log("drone", f"Stopped {orphan_count} orphan live subprocess(es).")
            mark_live_dji_status_stopped("ATLAS live localization stopped by user.")
            if terminated == 0 and orphan_count == 0:
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
