#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import cv2


def resize_keep_aspect(frame, max_size: int):
    if max_size <= 0:
        return frame
    h, w = frame.shape[:2]
    scale = min(1.0, float(max_size) / float(max(h, w)))
    if scale >= 0.999:
        return frame
    return cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max-size", type=int, default=1600)
    ap.add_argument("--prefix", default="frame")
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--end-sec", type=float, default=0.0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()

    if not args.video.exists():
        raise FileNotFoundError(args.video)
    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / native_fps if native_fps > 0 else 0.0
    step = max(1, int(round(native_fps / args.fps)))
    start_frame = max(0, int(math.floor(args.start_sec * native_fps)))
    end_frame = frame_count if args.end_sec <= 0 else min(frame_count, int(math.ceil(args.end_sec * native_fps)))

    rows = []
    saved = 0
    frame_idx = start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if (frame_idx - start_frame) % step == 0:
            out_name = f"{args.prefix}_{saved:06d}.jpg"
            out_path = args.out_dir / out_name
            frame_small = resize_keep_aspect(frame, args.max_size)
            cv2.imwrite(str(out_path), frame_small, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.quality)])
            rows.append(
                {
                    "image_name": out_name,
                    "source_frame": frame_idx,
                    "time_sec": frame_idx / native_fps,
                    "width": frame_small.shape[1],
                    "height": frame_small.shape[0],
                }
            )
            saved += 1
            if args.max_frames > 0 and saved >= args.max_frames:
                break
        frame_idx += 1

    cap.release()

    with (args.out_dir / "frames.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "source_frame", "time_sec", "width", "height"])
        writer.writeheader()
        writer.writerows(rows)

    (args.out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "video": str(args.video),
                "native_fps": native_fps,
                "frame_count": frame_count,
                "duration_sec": duration,
                "requested_fps": args.fps,
                "step_frames": step,
                "saved_frames": saved,
                "max_size": args.max_size,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"video": str(args.video), "out_dir": str(args.out_dir), "saved_frames": saved}, indent=2))


if __name__ == "__main__":
    main()
