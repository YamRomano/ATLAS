#!/usr/bin/env python3
"""Bridge DJI MSDK Remote live video into ATLAS frame streams.

This script is intentionally video-only.  It connects to the Android
MSDKRemote/OpenDJI app, receives decoded camera frames, and writes them in the
same frame-bank format used by the ATLAS TSolve replay pipeline:

    data/dji_live/<session>/query_frames/query_000000.jpg
    data/dji_live/<session>/query_frames/frames.csv

It also writes a browser-visible preview:

    viewer/public/live_dji/latest.jpg
    viewer/public/live_dji/status.json

No control commands are sent here.  Control must stay a separate, explicit
phase after live localization is stable.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import signal
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import cv2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPENDJI_ROOT = Path("/Users/yamromano/Desktop/DJI-MSDK-to-PC-main")
IMAGE_EXTS = {".jpg", ".jpeg"}
TAKEOFF_VERTICAL_SPEED = 0.03
TAKEOFF_STEP_SECONDS = 0.50
TAKEOFF_MAX_ASSIST_SECONDS = 16.0
PRE_LAND_STABILIZE_SECONDS = 1.50


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


def execute_control_command(drone: Any, OpenDJI: Any, command: dict[str, Any]) -> dict[str, Any]:
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
    return {
        "ok": True,
        "id": command.get("id"),
        "command": name,
        "height_m": height_m,
        "result": result,
        "altitude_before": altitude_before,
        "altitude_after": altitude_after,
        "height_guard": height_guard,
        "note": note,
        "elapsed_seconds": time.time() - started,
        "updated_at": time.time(),
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
    control_command_path = args.public_root / "control_command.json"
    control_status_path = args.public_root / "control_status.json"
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
    frame_index = existing_frame_count(query_dir)
    first_frame_time: float | None = None
    last_shape: tuple[int, int] | None = None
    last_progress_print = 0.0
    last_control_id: str | None = None
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

            def control_worker(command_payload: dict[str, Any]) -> None:
                nonlocal control_busy
                try:
                    control_result = execute_control_command(drone, OpenDJI, command_payload)
                    atomic_write_json(control_status_path, control_result)
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
                                control_result = {
                                    "ok": False,
                                    "id": command_id,
                                    "command": command_payload.get("command"),
                                    "error": "another DJI command is already running",
                                    "updated_at": time.time(),
                                }
                                atomic_write_json(control_status_path, control_result)
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
