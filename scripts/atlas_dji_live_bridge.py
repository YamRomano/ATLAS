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
import signal
import sys
import time
import types
from pathlib import Path
from typing import Any

import cv2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPENDJI_ROOT = Path("/Users/yamromano/Desktop/DJI-MSDK-to-PC-main")
IMAGE_EXTS = {".jpg", ".jpeg"}


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
        "control_enabled": False,
        "note": "This bridge receives frames only. It does not send movement commands.",
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

    print("ATLAS DJI live bridge")
    print(f"  phone IP:      {args.phone_ip}")
    print(f"  OpenDJI root:  {args.opendji_root.resolve()}")
    print(f"  session root:  {session_root}")
    print(f"  query frames:  {query_dir}")
    print(f"  public status: {public_status_path}")
    print("  control:       disabled")
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

            while not stop.stop:
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
