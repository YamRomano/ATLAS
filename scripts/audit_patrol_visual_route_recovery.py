#!/usr/bin/env python3
"""Replay held-out patrol frames through the leg-constrained visual bank."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from patrol_visual_route_recovery import PatrolVisualRouteRecovery


def transformed(gray: np.ndarray, variant: int) -> np.ndarray:
    if variant == 0:
        return np.clip(gray.astype(np.float32) * 0.72, 0, 255).astype(np.uint8)
    if variant == 1:
        return np.clip(gray.astype(np.float32) * 1.22 + 8.0, 0, 255).astype(np.uint8)
    if variant == 2:
        return cv2.GaussianBlur(gray, (5, 5), 1.1)
    matrix = np.float32([[1.0, 0.0, 8.0], [0.0, 1.0, -5.0]])
    return cv2.warpAffine(
        gray,
        matrix,
        (gray.shape[1], gray.shape[0]),
        borderMode=cv2.BORDER_REFLECT,
    )


def load_gray(frame_dir: Path, frame_index: int) -> np.ndarray:
    path = frame_dir / f"query_{frame_index:06d}.jpg"
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(path)
    return gray


def percentile(values: list[float], amount: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), amount)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--frame-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    reference = json.loads(args.baseline.read_text(encoding="utf-8"))
    pose_document = json.loads(args.baseline.with_name("poses.json").read_text(encoding="utf-8"))
    pose_by_frame = {
        int(pose["source_frame"]): pose
        for pose in pose_document.get("poses") or []
        if pose.get("source_frame") is not None
    }
    recovery = PatrolVisualRouteRecovery(args.bank)
    per_leg: list[dict[str, Any]] = []
    timings: list[float] = []
    all_errors: list[float] = []
    expected_total = 0
    accepted_total = 0

    for lap in (1, 2):
        for leg_index, leg in enumerate(reference["legs"]):
            sample_frames = [int(item["source_frame"]) for item in leg.get("samples") or []]
            query_frames = list(range(min(sample_frames) + 5, max(sample_frames), 10))
            # Strict waypoint transitions depend on the visual route reaching
            # the actual recorded endpoint.  Include that final frame instead
            # of validating only the interior of each leg.
            if not query_frames or query_frames[-1] != max(sample_frames):
                query_frames.append(max(sample_frames))
            expected = max(0, len(query_frames) - 1)  # first frame acquires; it never translates
            accepted = 0
            errors: list[float] = []
            leg_timings: list[float] = []
            reasons: dict[str, int] = {}
            for query_index, frame_index in enumerate(query_frames):
                gray = transformed(load_gray(args.frame_dir, frame_index), query_index % 4)
                started = time.perf_counter()
                observation, diagnostic = recovery.recover(
                    gray=gray,
                    segment_start=leg["from"],
                    segment_end=leg["to"],
                    segment_key=(lap, leg_index),
                    translation_locked=False,
                    independent_progress=leg_index in {2, 3},
                    sequence_index=frame_index,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                timings.append(elapsed_ms)
                leg_timings.append(elapsed_ms)
                if observation is None:
                    reason = str(diagnostic.get("reason") or "unknown")
                    reasons[reason] = reasons.get(reason, 0) + 1
                    continue
                truth = float(pose_by_frame[frame_index]["route_progress"])
                errors.append(abs(float(observation["progress"]) - truth))
                accepted += 1
                # Live publication commits the bounded position actually sent
                # to the bridge.  The matcher intentionally does not advance
                # its own route clock merely because it detected a farther
                # frame.  Mirror that handshake here; otherwise every query
                # after the first 50% of a leg is incorrectly audited against
                # a permanent zero-progress window.
                recovery.commit_published_progress(observation["progress"])
            expected_total += expected
            accepted_total += min(accepted, expected)
            all_errors.extend(errors)
            per_leg.append(
                {
                    "lap": lap,
                    "leg": leg_index + 1,
                    "from_point": leg.get("from_point"),
                    "to_point": leg.get("to_point"),
                    "query_count": len(query_frames),
                    "expected_after_acquisition": expected,
                    "accepted": accepted,
                    "acceptance_ratio": accepted / max(1, expected),
                    "max_progress_error": max(errors, default=0.0),
                    "median_runtime_ms": percentile(leg_timings, 50),
                    "p95_runtime_ms": percentile(leg_timings, 95),
                    "rejections": reasons,
                }
            )

    negative_accepts: list[dict[str, Any]] = []
    for target_index, target_leg in enumerate(reference["legs"]):
        for source_index, source_leg in enumerate(reference["legs"]):
            if source_index == target_index:
                continue
            source_start = min(int(item["source_frame"]) for item in source_leg["samples"])
            negative_recovery = PatrolVisualRouteRecovery(args.bank)
            for offset in (5, 6, 7):
                observation, diagnostic = negative_recovery.recover(
                    gray=load_gray(args.frame_dir, source_start + offset),
                    segment_start=target_leg["from"],
                    segment_end=target_leg["to"],
                    segment_key=("negative", target_index, source_index),
                    translation_locked=False,
                )
                if observation is not None:
                    negative_accepts.append(
                        {
                            "target_leg": target_index + 1,
                            "source_leg": source_index + 1,
                            "frame": source_start + offset,
                            "inliers": observation.get("inliers"),
                            "progress": observation.get("progress"),
                        }
                    )

    overall_ratio = accepted_total / max(1, expected_total)
    return_leg_rows = [item for item in per_leg if item["leg"] == 4]
    passed = bool(
        overall_ratio >= 0.98
        and max(all_errors, default=0.0) <= 0.20
        and all(item["acceptance_ratio"] >= 0.95 for item in return_leg_rows)
        and not negative_accepts
        and percentile(timings, 95) <= 120.0
    )
    summary = {
        "passed": passed,
        "baseline": str(args.baseline.resolve()),
        "bank": str(args.bank.resolve()),
        "laps": 2,
        "held_out_query_count": sum(item["query_count"] for item in per_leg),
        "expected_after_acquisition": expected_total,
        "accepted_after_acquisition": accepted_total,
        "acceptance_ratio": overall_ratio,
        "max_progress_error": max(all_errors, default=0.0),
        "mean_progress_error": float(np.mean(all_errors)) if all_errors else 0.0,
        "median_runtime_ms": percentile(timings, 50),
        "p95_runtime_ms": percentile(timings, 95),
        "max_runtime_ms": max(timings, default=0.0),
        "wrong_leg_accepts": negative_accepts,
        "per_leg": per_leg,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
