#!/usr/bin/env python3
"""Create compact visual contact sheets for ATLAS demo source selection."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def fit_cover(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = frame.shape[:2]
    scale = max(width / src_w, height / src_h)
    resized = cv2.resize(frame, (round(src_w * scale), round(src_h * scale)))
    y0 = max(0, (resized.shape[0] - height) // 2)
    x0 = max(0, (resized.shape[1] - width) // 2)
    return resized[y0 : y0 + height, x0 : x0 + width]


def make_sheet(video: Path, output: Path, samples: int = 12) -> None:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video}")
    frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    cols, rows = 4, (samples + 3) // 4
    tile_w, tile_h = 480, 270
    header_h = 54
    sheet = np.full((header_h + rows * tile_h, cols * tile_w, 3), (5, 16, 28), np.uint8)
    cv2.putText(
        sheet,
        f"{video.name}  |  {frame_count / fps:.1f}s  |  {fps:.1f} fps",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (205, 240, 255),
        2,
        cv2.LINE_AA,
    )
    for index in range(samples):
        fraction = (index + 0.5) / samples
        frame_number = min(frame_count - 1, round(fraction * (frame_count - 1)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        if not ok:
            continue
        tile = fit_cover(frame, tile_w, tile_h)
        seconds = frame_number / fps
        cv2.rectangle(tile, (0, tile_h - 34), (tile_w, tile_h), (2, 9, 16), -1)
        cv2.putText(
            tile,
            f"{seconds:06.1f}s",
            (14, tile_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (126, 236, 255),
            1,
            cv2.LINE_AA,
        )
        row, col = divmod(index, cols)
        sheet[header_h + row * tile_h : header_h + (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w] = tile
    capture.release()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"Could not write {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("videos", nargs="+", type=Path)
    args = parser.parse_args()
    for video in args.videos:
        output = args.output_dir / f"{video.stem}_contact.jpg"
        make_sheet(video, output)
        print(output)


if __name__ == "__main__":
    main()
