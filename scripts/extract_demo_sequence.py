#!/usr/bin/env python3
"""Extract a rendered ATLAS master to a numbered JPEG sequence for H.264 encoding."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--quality", type=int, default=91)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {args.video}")
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        path = args.output_dir / f"frame_{index:05d}.jpg"
        if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality]):
            raise RuntimeError(f"Could not write {path}")
        index += 1
        if index % 240 == 0:
            print(f"extracted {index}", flush=True)
    capture.release()
    print(f"frames={index}")


if __name__ == "__main__":
    main()
