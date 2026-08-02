#!/usr/bin/env python3
"""Export a recorded ATLAS live-frame sequence as an uploadable MP4."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("frame_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--codec",
        default="avc1",
        help="FourCC codec to request (default: avc1/H.264).",
    )
    return parser.parse_args()


def load_sequence(frame_dir: Path) -> tuple[list[Path], float, float]:
    csv_path = frame_dir / "frames.csv"
    if not csv_path.is_file():
        raise SystemExit(f"Missing frame index: {csv_path}")

    rows: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows.extend(csv.DictReader(handle))
    if len(rows) < 2:
        raise SystemExit("At least two indexed frames are required")

    frames = [frame_dir / row["image_name"] for row in rows]
    missing = [path for path in frames if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing {len(missing)} indexed frames; first: {missing[0]}")

    first_time = float(rows[0]["time_sec"])
    last_time = float(rows[-1]["time_sec"])
    duration = last_time - first_time
    if duration <= 0:
        raise SystemExit(f"Invalid capture duration: {duration}")
    fps = (len(frames) - 1) / duration
    return frames, fps, duration


def main() -> None:
    args = parse_args()
    frames, fps, source_duration = load_sequence(args.frame_dir)

    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise SystemExit(f"Could not decode first frame: {frames[0]}")
    source_height, source_width = first.shape[:2]
    output_width = source_width + (source_width % 2)
    output_height = source_height + (source_height % 2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*args.codec),
        fps,
        (output_width, output_height),
    )
    codec = args.codec
    if not writer.isOpened() and args.codec != "mp4v":
        writer.release()
        codec = "mp4v"
        writer = cv2.VideoWriter(
            str(args.output),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (output_width, output_height),
        )
    if not writer.isOpened():
        raise SystemExit(f"Could not open video writer for {args.output}")

    try:
        for index, path in enumerate(frames):
            frame = first if index == 0 else cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise SystemExit(f"Could not decode frame {index}: {path}")
            if frame.shape[:2] != (source_height, source_width):
                raise SystemExit(
                    f"Frame {index} has dimensions {frame.shape[1]}x{frame.shape[0]}, "
                    f"expected {source_width}x{source_height}: {path}"
                )
            if (output_height, output_width) != frame.shape[:2]:
                padded = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                padded[:source_height, :source_width] = frame
                frame = padded
            writer.write(frame)
            if (index + 1) % 500 == 0 or index + 1 == len(frames):
                print(f"encoded {index + 1}/{len(frames)}", flush=True)
    finally:
        writer.release()

    print(
        f"output={args.output}\n"
        f"codec={codec}\n"
        f"frames={len(frames)}\n"
        f"fps={fps:.9f}\n"
        f"source_duration_sec={source_duration:.6f}\n"
        f"dimensions={output_width}x{output_height}"
    )


if __name__ == "__main__":
    main()
