#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2


def resize_frame(frame, max_size: int):
    if max_size <= 0:
        return frame
    h, w = frame.shape[:2]
    scale = min(1.0, max_size / max(w, h))
    if scale >= 0.999:
        return frame
    return cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)


def main() -> None:
    ap = argparse.ArgumentParser(description="Capture sparse mapping frames from a live PC/Mac webcam.")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--duration", type=float, default=60.0, help="Capture duration in seconds.")
    ap.add_argument("--fps", type=float, default=1.5, help="Saved frame rate.")
    ap.add_argument("--max-size", type=int, default=1200)
    ap.add_argument("--prefix", default="webcam_map")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for old in args.out_dir.glob(f"{args.prefix}_*.jpg"):
        old.unlink()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {args.camera_index}. "
            "On macOS, approve camera access for the terminal/Python app and retry."
        )

    rows = []
    start = time.perf_counter()
    next_save = start
    interval = 1.0 / max(args.fps, 1e-6)
    frame_idx = 0
    saved_idx = 0

    print("Capture started. Move slowly around the room with overlap; press q to stop early.")
    try:
        while True:
            now = time.perf_counter()
            if now - start >= args.duration:
                break

            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera frame read failed.")
            frame_idx += 1

            if now >= next_save:
                image = resize_frame(frame, args.max_size)
                name = f"{args.prefix}_{saved_idx:06d}.jpg"
                out = args.out_dir / name
                cv2.imwrite(str(out), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                rows.append({"frame": saved_idx, "time_sec": now - start, "image_name": name})
                saved_idx += 1
                next_save += interval
                print(f"saved {name}", flush=True)

            if not args.no_preview:
                preview = resize_frame(frame, 900)
                cv2.putText(
                    preview,
                    f"captured {saved_idx} frames - press q to stop",
                    (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (50, 255, 180),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("ATLAS webcam mapping capture", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if not args.no_preview:
            cv2.destroyAllWindows()

    with (args.out_dir / "frames.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "time_sec", "image_name"])
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "camera_index": args.camera_index,
        "duration_requested_sec": args.duration,
        "fps_requested": args.fps,
        "max_size": args.max_size,
        "frames_read": frame_idx,
        "frames_saved": saved_idx,
        "out_dir": str(args.out_dir),
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
