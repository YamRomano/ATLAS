#!/usr/bin/env python3
"""Audit the post-abort Point-4 tail before enabling it for live patrol."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from patrol_visual_route_recovery import PatrolVisualRouteRecovery, horizontal_distance


def percentile(values: list[float], amount: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), amount))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--base-bank", required=True, type=Path)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--frame-dir", required=True, type=Path)
    parser.add_argument("--source-replay-id", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    reference = json.loads(args.baseline.read_text(encoding="utf-8"))
    legs = list(reference.get("legs") or [])
    if len(legs) < 4:
        raise RuntimeError("Manual-tail audit requires a four-leg patrol")

    def gray(frame_index: int) -> np.ndarray:
        image_path = args.frame_dir / f"query_{frame_index:06d}.jpg"
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_path)
        return image

    recovery = PatrolVisualRouteRecovery(args.bank)
    heading_checks: list[dict[str, Any]] = []
    for leg_index, minimum_inliers, frames in (
        (3, 50, (2878, 2882, 2886, 2890)),
        (0, 72, (3035, 3036, 3037, 3038, 3039)),
    ):
        leg = legs[leg_index]
        for frame_index in frames:
            observation, diagnostic = recovery.departure_heading_alignment(
                gray=gray(frame_index),
                segment_start=leg["from"],
                segment_end=leg["to"],
                focal_px=882.4866783165957,
                minimum_inliers=minimum_inliers,
            )
            heading_checks.append(
                {
                    "leg": leg_index + 1,
                    "frame": frame_index,
                    "verified": observation is not None,
                    "correction_deg": (
                        float(observation["correction_deg"])
                        if observation is not None
                        else None
                    ),
                    "inliers": (
                        int(observation["inliers"])
                        if observation is not None
                        else int(diagnostic.get("best_inliers") or 0)
                    ),
                    "anchor": (
                        observation.get("anchor_name")
                        if observation is not None
                        else None
                    ),
                }
            )

    preturn, _ = recovery.departure_heading_alignment(
        gray=gray(2740),
        segment_start=legs[3]["from"],
        segment_end=legs[3]["to"],
        focal_px=882.4866783165957,
        minimum_inliers=50,
    )

    route_recovery = PatrolVisualRouteRecovery(args.bank)
    published = 0.0
    accepted: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    runtimes_ms: list[float] = []
    endpoint_verified = False
    for frame_index in range(2891, 2966):
        started = time.perf_counter()
        observation, diagnostic = route_recovery.recover(
            gray=gray(frame_index),
            segment_start=legs[3]["from"],
            segment_end=legs[3]["to"],
            segment_key=("manual-tail-audit", 4),
            translation_locked=False,
            progress_hint=published,
            independent_progress=True,
            sequence_index=frame_index,
        )
        runtimes_ms.append((time.perf_counter() - started) * 1000.0)
        if observation is None:
            rejections.append(
                {
                    "frame": frame_index,
                    "reason": str(diagnostic.get("reason") or "unknown"),
                }
            )
            continue
        progress = float(observation["progress"])
        if progress + 1.0e-9 < published:
            raise RuntimeError("Manual-tail route progress moved backwards")
        published = progress
        endpoint_verified = bool(observation.get("endpoint_verified"))
        route_recovery.commit_published_progress(progress)
        accepted.append(
            {
                "frame": frame_index,
                "progress": progress,
                "inliers": int(observation["inliers"]),
                "source_replay_id": str(observation["source_replay_id"]),
            }
        )

    with np.load(args.base_bank, allow_pickle=False) as base, np.load(
        args.bank, allow_pickle=False
    ) as bank:
        base_leg4 = np.asarray(
            [
                horizontal_distance(start, legs[3]["from"]) <= 0.08
                and horizontal_distance(end, legs[3]["to"]) <= 0.08
                for start, end in zip(base["anchor_from"], base["anchor_to"])
            ],
            dtype=bool,
        )
        bank_leg4 = np.asarray(
            [
                horizontal_distance(start, legs[3]["from"]) <= 0.08
                and horizontal_distance(end, legs[3]["to"]) <= 0.08
                for start, end in zip(bank["anchor_from"], bank["anchor_to"])
            ],
            dtype=bool,
        )
        source_ids = [
            str(value) for value in bank["anchor_source_replay_ids"].tolist()
        ]
        leg4_source_ids = {
            source_ids[index]
            for index, is_leg4 in enumerate(bank_leg4)
            if is_leg4
        }
        heading_priority = np.asarray(bank["anchor_heading_priority"], dtype=int)
        source_frames = np.asarray(bank["source_frames"], dtype=int)
        latest_source_indices = [
            index
            for index, source_id in enumerate(source_ids)
            if source_id == args.source_replay_id
        ]
        preserved_anchor_count = int(np.count_nonzero(~base_leg4))
        base_prefix_preserved = bool(
            np.array_equal(
                bank["anchor_names"][:preserved_anchor_count],
                base["anchor_names"][~base_leg4],
            )
            and np.allclose(
                bank["anchor_progress"][:preserved_anchor_count],
                base["anchor_progress"][~base_leg4],
            )
        )
        priority_frames = sorted(
            int(source_frames[index])
            for index in latest_source_indices
            if int(heading_priority[index]) == 100
        )
        maximum_latest_source_frame = max(
            int(source_frames[index]) for index in latest_source_indices
        )

    aligned_ok = all(
        item["verified"]
        and item["inliers"] >= (50 if item["leg"] == 4 else 72)
        and abs(float(item["correction_deg"])) <= 1.0
        and str(item["anchor"] or "").startswith(args.source_replay_id + "/")
        for item in heading_checks
    )
    passed = bool(
        aligned_ok
        and preturn is not None
        and float(preturn["correction_deg"]) <= -15.0
        and len(rejections) == 1
        and rejections[0]["reason"] == "visual_route_acquiring"
        and len(accepted) == 74
        and published >= 0.99
        and endpoint_verified
        and all(
            item["source_replay_id"] == args.source_replay_id for item in accepted
        )
        and percentile(runtimes_ms, 95) <= 120.0
        and leg4_source_ids == {args.source_replay_id}
        and base_prefix_preserved
        and priority_frames
        == [2878, 2880, 2882, 2884, 2886, 2888, 2890, 3035, 3036, 3037, 3038, 3039]
        and maximum_latest_source_frame == 3039
    )
    payload = {
        "passed": passed,
        "kind": "atlas_manual_patrol_tail_visual_bank_audit",
        "baseline": str(args.baseline.resolve()),
        "base_bank": str(args.base_bank.resolve()),
        "bank": str(args.bank.resolve()),
        "source_replay_id": args.source_replay_id,
        "cuts": {
            "point4_heading": [2878, 2890],
            "leg4_translation": [2891, 2965],
            "point1_heading": [3035, 3039],
            "landing_frames_excluded_after": 3039,
        },
        "heading_checks": heading_checks,
        "preturn_frame_2740_correction_deg": (
            float(preturn["correction_deg"]) if preturn is not None else None
        ),
        "route": {
            "frames": [2891, 2965],
            "accepted": len(accepted),
            "rejections": rejections,
            "final_progress": published,
            "endpoint_verified": endpoint_verified,
            "p95_runtime_ms": percentile(runtimes_ms, 95),
            "max_runtime_ms": max(runtimes_ms),
        },
        "preservation": {
            "old_leg4_anchors_removed": int(np.count_nonzero(base_leg4)),
            "preserved_other_anchor_count": preserved_anchor_count,
            "base_prefix_preserved": base_prefix_preserved,
            "leg4_source_ids": sorted(leg4_source_ids),
            "maximum_latest_source_frame": maximum_latest_source_frame,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
