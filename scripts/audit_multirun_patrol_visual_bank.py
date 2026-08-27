#!/usr/bin/env python3
"""Audit independent patrol recordings against a multi-run route bank.

The audit deliberately queries frames between the stored anchors and applies
small photometric/geometric perturbations.  Passing therefore proves that the
route tracker can reacquire a nearby recorded flight; it does not merely look
up the exact images used to build the bank.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from build_multirun_patrol_visual_bank import translation_ranges
from patrol_visual_route_recovery import (
    PatrolVisualRouteRecovery,
    _forward_motion_profile,
)


def load_gray(frame_dir: Path, frame_index: int) -> np.ndarray:
    image_path = frame_dir / f"query_{frame_index:06d}.jpg"
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(image_path)
    return gray


def perturb(gray: np.ndarray, variant: int) -> np.ndarray:
    """Represent modest exposure, blur, and camera-offset differences."""
    if variant == 0:
        return np.clip(gray.astype(np.float32) * 0.78, 0, 255).astype(np.uint8)
    if variant == 1:
        return np.clip(gray.astype(np.float32) * 1.15 + 6.0, 0, 255).astype(np.uint8)
    if variant == 2:
        return cv2.GaussianBlur(gray, (5, 5), 0.9)
    height, width = gray.shape
    matrix = cv2.getRotationMatrix2D((0.5 * width, 0.5 * height), 1.1, 1.008)
    matrix[:, 2] += np.asarray([5.0, -3.0])
    return cv2.warpAffine(
        gray,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def percentile(values: list[float], amount: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), amount))


def held_out_frames(first: int, last: int, stride: int) -> list[int]:
    """Choose intervening frames plus one perturbed endpoint check."""
    step = max(2, int(stride))
    frames = list(range(int(first) + 1, int(last), step))
    # The endpoint is deliberately stored because it is safety-critical. Test
    # that anchor under a perturbation after the held-out temporal sequence.
    if not frames or frames[-1] != int(last):
        frames.append(int(last))
    return frames


def resolve_recorded_segments(
    *,
    plan: dict[str, Any],
    manifest: dict[str, Any],
    viewer_root: Path,
) -> list[dict[str, Any]]:
    map_id = str(plan.get("map_id") or "")
    baseline_replay_id = str(plan.get("baseline_replay_id") or "")
    map_entry = next(
        (entry for entry in manifest.get("maps") or [] if entry.get("id") == map_id),
        None,
    )
    if not isinstance(map_entry, dict):
        raise RuntimeError(f"Composite map is missing from the manifest: {map_id}")
    replays = {
        str(replay.get("id") or ""): replay
        for replay in map_entry.get("replays") or []
        if isinstance(replay, dict)
    }
    root = viewer_root.resolve()
    resolved: list[dict[str, Any]] = []
    for source in plan.get("second_lap_segments") or []:
        source_id = str(source.get("source_replay_id") or "")
        if not source_id or source_id == baseline_replay_id:
            continue
        replay = replays.get(source_id)
        if not isinstance(replay, dict):
            raise RuntimeError(f"Recorded source replay is missing: {source_id}")
        frame_dir = (root / str(replay.get("query_frame_base_url") or "").strip("/")).resolve()
        if root not in frame_dir.parents or not frame_dir.is_dir():
            raise RuntimeError(f"Unsafe or missing frame directory: {frame_dir}")
        for item in translation_ranges(
            dict(source.get("phase_boundaries") or {}),
            int(source["start_frame"]),
            int(source["end_frame"]),
        ):
            resolved.append(
                {
                    **item,
                    "source_replay_id": source_id,
                    "frame_dir": frame_dir,
                }
            )
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--composite-plan", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--viewer-root", required=True, type=Path)
    parser.add_argument("--baseline-audit", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--query-stride", type=int, default=2)
    args = parser.parse_args()

    reference = json.loads(args.baseline.read_text(encoding="utf-8"))
    plan = json.loads(args.composite_plan.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    baseline_audit = json.loads(args.baseline_audit.read_text(encoding="utf-8"))
    legs = list(reference.get("legs") or [])
    segments = resolve_recorded_segments(
        plan=plan,
        manifest=manifest,
        viewer_root=args.viewer_root,
    )
    if not segments:
        raise RuntimeError("No independent recorded route segments were found")

    all_timings: list[float] = []
    all_errors: list[float] = []
    expected_total = 0
    accepted_total = 0
    source_mismatches: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    per_segment: list[dict[str, Any]] = []

    for segment_number, segment in enumerate(segments):
        leg_index = int(segment["leg_index"])
        leg = legs[leg_index]
        first = int(segment["start_frame"])
        last = int(segment["end_frame"])
        source_id = str(segment["source_replay_id"])
        frame_dir = Path(segment["frame_dir"])
        truth = _forward_motion_profile(frame_dir, first, last)
        frames = held_out_frames(first, last, args.query_stride)
        recovery = PatrolVisualRouteRecovery(args.bank)
        published = 0.0
        accepted = 0
        reasons: dict[str, int] = {}
        rejection_samples: list[dict[str, Any]] = []
        errors: list[float] = []
        error_samples: list[dict[str, Any]] = []
        timings: list[float] = []
        last_published = 0.0

        for query_number, frame_index in enumerate(frames):
            gray = perturb(load_gray(frame_dir, frame_index), query_number % 4)
            started = time.perf_counter()
            observation, diagnostic = recovery.recover(
                gray=gray,
                segment_start=leg["from"],
                segment_end=leg["to"],
                segment_key=("independent", segment_number, leg_index),
                translation_locked=False,
                progress_hint=published,
                independent_progress=True,
                sequence_index=frame_index,
            )
            elapsed_ms = 1000.0 * (time.perf_counter() - started)
            timings.append(elapsed_ms)
            all_timings.append(elapsed_ms)
            if observation is None:
                reason = str(diagnostic.get("reason") or "unknown")
                reasons[reason] = reasons.get(reason, 0) + 1
                rejection_samples.append(
                    {
                        "frame": frame_index,
                        "variant": query_number % 4,
                        "reason": reason,
                        "best_inliers": diagnostic.get("best_inliers"),
                    }
                )
                continue
            accepted += 1
            observed_source = str(observation.get("source_replay_id") or "")
            if observed_source != source_id:
                source_mismatches.append(
                    {
                        "leg": leg_index + 1,
                        "frame": frame_index,
                        "expected": source_id,
                        "observed": observed_source,
                    }
                )
            published = float(observation["progress"])
            recovery.commit_published_progress(published)
            if published + 1.0e-9 < last_published:
                regressions.append(
                    {
                        "leg": leg_index + 1,
                        "frame": frame_index,
                        "previous": last_published,
                        "current": published,
                    }
                )
            last_published = max(last_published, published)
            error = abs(published - float(truth[frame_index]))
            errors.append(error)
            all_errors.append(error)
            error_samples.append(
                {
                    "frame": frame_index,
                    "variant": query_number % 4,
                    "truth": float(truth[frame_index]),
                    "published": published,
                    "error": error,
                    "inliers": int(observation.get("inliers") or 0),
                    "source_frame": int(observation.get("source_frame") or 0),
                }
            )

        expected = max(0, len(frames) - 1)
        expected_total += expected
        accepted_total += min(accepted, expected)
        per_segment.append(
            {
                "source_replay_id": source_id,
                "leg": leg_index + 1,
                "from_point": leg.get("from_point"),
                "to_point": leg.get("to_point"),
                "frame_range": [first, last],
                "held_out_query_count": len(frames),
                "expected_after_acquisition": expected,
                "accepted": accepted,
                "acceptance_ratio": accepted / max(1, expected),
                "max_progress_error": max(errors, default=0.0),
                "median_runtime_ms": percentile(timings, 50),
                "p95_runtime_ms": percentile(timings, 95),
                "rejections": reasons,
                "rejection_samples": rejection_samples,
                "worst_progress_errors": sorted(
                    error_samples,
                    key=lambda item: float(item["error"]),
                    reverse=True,
                )[:5],
            }
        )

    overall_ratio = accepted_total / max(1, expected_total)
    independent_passed = bool(
        overall_ratio >= 0.95
        and max(all_errors, default=0.0) <= 0.15
        and percentile(all_timings, 95) <= 120.0
        and not source_mismatches
        and not regressions
        and all(row["acceptance_ratio"] >= 0.90 for row in per_segment)
    )
    passed = bool(baseline_audit.get("passed") is True and independent_passed)
    summary = {
        "passed": passed,
        "kind": "atlas_multirun_patrol_visual_recovery_audit",
        "baseline": str(args.baseline.resolve()),
        "bank": str(args.bank.resolve()),
        "variation_model": [
            "22 percent darker",
            "15 percent brighter plus offset",
            "mild Gaussian blur",
            "1.1 degree rotation, 0.8 percent scale, and 5/-3 pixel shift",
        ],
        "stored_anchor_stride": 2,
        "interior_queries_use_intervening_frames": True,
        "endpoint_queries_are_stored_anchors_with_perturbation": True,
        # Preserve the original audit contract consumed by the pinned-patrol
        # loader/tests while adding the detailed per-source reports below.
        "expected_after_acquisition": int(
            baseline_audit.get("expected_after_acquisition") or 0
        )
        + expected_total,
        "accepted_after_acquisition": int(
            baseline_audit.get("accepted_after_acquisition") or 0
        )
        + accepted_total,
        "acceptance_ratio": (
            (
                int(baseline_audit.get("accepted_after_acquisition") or 0)
                + accepted_total
            )
            / max(
                1,
                int(baseline_audit.get("expected_after_acquisition") or 0)
                + expected_total,
            )
        ),
        "max_progress_error": max(
            float(baseline_audit.get("max_progress_error") or 0.0),
            max(all_errors, default=0.0),
        ),
        "wrong_leg_accepts": list(baseline_audit.get("wrong_leg_accepts") or []),
        "baseline_regression_audit": baseline_audit,
        "independent_recording_audit": {
            "passed": independent_passed,
            "expected_after_acquisition": expected_total,
            "accepted_after_acquisition": accepted_total,
            "acceptance_ratio": overall_ratio,
            "max_progress_error": max(all_errors, default=0.0),
            "mean_progress_error": float(np.mean(all_errors)) if all_errors else 0.0,
            "median_runtime_ms": percentile(all_timings, 50),
            "p95_runtime_ms": percentile(all_timings, 95),
            "max_runtime_ms": max(all_timings, default=0.0),
            "source_mismatches": source_mismatches,
            "progress_regressions": regressions,
            "per_segment": per_segment,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
