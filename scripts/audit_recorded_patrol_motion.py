#!/usr/bin/env python3
"""Measure motion-phase boundaries in the recorded full-patrol frame bank.

This is a read-only diagnostic.  It estimates short-baseline image expansion
(forward/backward translation), horizontal flow (yaw), and homography residual
(parallax), then ranks the final Point-1 turn frames against the first-lap
Point-1 departure view.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def read_gray(frame_dir: Path, index: int) -> np.ndarray:
    path = frame_dir / f"query_{index:06d}.jpg"
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def tracked_points(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = cv2.goodFeaturesToTrack(a, maxCorners=1200, qualityLevel=0.01, minDistance=8)
    if points is None:
        return np.empty((0, 2)), np.empty((0, 2))
    moved, ok, _ = cv2.calcOpticalFlowPyrLK(
        a,
        b,
        points,
        None,
        winSize=(31, 31),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    back, back_ok, _ = cv2.calcOpticalFlowPyrLK(
        b,
        a,
        moved,
        None,
        winSize=(31, 31),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    valid = ok.reshape(-1).astype(bool) & back_ok.reshape(-1).astype(bool)
    p = points.reshape(-1, 2)
    q = moved.reshape(-1, 2)
    valid &= np.linalg.norm(back.reshape(-1, 2) - p, axis=1) <= 2.0
    return p[valid], q[valid]


def motion_metrics(a: np.ndarray, b: np.ndarray) -> tuple[int, float, float, float, float]:
    p, q = tracked_points(a, b)
    if len(p) < 20:
        return len(p), float("nan"), float("nan"), float("nan"), float("nan")
    h, w = a.shape
    x = np.column_stack(
        [
            np.ones(len(p)),
            (p[:, 0] - 0.5 * w) / w,
            (p[:, 1] - 0.5 * h) / h,
        ]
    )
    flow = q - p
    coef_x = np.linalg.lstsq(x, flow[:, 0], rcond=None)[0]
    coef_y = np.linalg.lstsq(x, flow[:, 1], rcond=None)[0]
    # A forward camera translation produces radial expansion: positive dX/dx
    # and dY/dy.  Divide by image size so both terms are dimensionless.
    expansion = 0.5 * (coef_x[1] / w + coef_y[2] / h)
    yaw_px = float(np.median(flow[:, 0]))
    flow_px = float(np.median(np.linalg.norm(flow, axis=1)))
    homography, mask = cv2.findHomography(p, q, cv2.RANSAC, 2.5)
    residual = float("nan")
    if homography is not None and mask is not None:
        projected = cv2.perspectiveTransform(p.reshape(-1, 1, 2), homography).reshape(-1, 2)
        residual = float(np.median(np.linalg.norm(projected - q, axis=1)))
    return len(p), expansion, yaw_px, flow_px, residual


def orb_match(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    orb = cv2.ORB_create(nfeatures=2500, fastThreshold=8)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None:
        return 0, 0, 0.0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m for m, n in pairs if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return len(good), 0, 0.0
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    _, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
    inliers = int(mask.sum()) if mask is not None else 0
    return len(good), inliers, inliers / max(1, len(good))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("frame_dir", type=Path)
    parser.add_argument("--segments", default="2700:2960,4000:4300")
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--seam-reference", type=int, default=991)
    parser.add_argument("--seam-range", default="4400:4600")
    args = parser.parse_args()

    for segment in args.segments.split(","):
        start, end = (int(value) for value in segment.split(":"))
        print(f"segment {start}:{end}, delta={args.step}")
        for index in range(start, end - args.step + 1, args.step):
            metrics = motion_metrics(
                read_gray(args.frame_dir, index),
                read_gray(args.frame_dir, index + args.step),
            )
            tracks, expansion, yaw_px, flow_px, residual = metrics
            print(
                f"{index:04d}->{index + args.step:04d} tracks={tracks:4d} "
                f"expansion={expansion:+.6f} yaw_px={yaw_px:+7.2f} "
                f"flow_px={flow_px:7.2f} Hres={residual:5.2f}"
            )

    seam_start, seam_end = (int(value) for value in args.seam_range.split(":"))
    reference = read_gray(args.frame_dir, args.seam_reference)
    ranked: list[tuple[int, int, float, int]] = []
    for index in range(seam_start, seam_end + 1):
        matches, inliers, ratio = orb_match(reference, read_gray(args.frame_dir, index))
        ranked.append((inliers, matches, ratio, index))
    print(f"seam candidates matching frame {args.seam_reference}")
    for inliers, matches, ratio, index in sorted(ranked, reverse=True)[:20]:
        print(f"frame={index} inliers={inliers} matches={matches} ratio={ratio:.3f}")


if __name__ == "__main__":
    main()
