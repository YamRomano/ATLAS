#!/usr/bin/env python3
"""Audit a Point-1 endpoint extension from a stale 4->1 progress state."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from patrol_visual_route_recovery import PatrolVisualRouteRecovery


def frame_range(value: str) -> tuple[int, int]:
    try:
        first, last = (int(item) for item in value.split(":", 1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("frame range must be FIRST:LAST") from exc
    if first < 0 or last <= first:
        raise argparse.ArgumentTypeError("frame range must increase")
    return first, last


def selected_frames(first: int, last: int, stride: int) -> list[int]:
    frames = list(range(first, last + 1, max(1, int(stride))))
    if frames[-1] != last:
        frames.append(last)
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--base-bank", required=True, type=Path)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--frame-dir", required=True, type=Path)
    parser.add_argument("--source-replay-id", required=True)
    parser.add_argument("--endpoint-frames", required=True, type=frame_range)
    parser.add_argument("--negative-frame", required=True, type=int)
    parser.add_argument("--preserved-tail-frame-dir", required=True, type=Path)
    parser.add_argument("--preserved-tail-source-replay-id", required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    reference = json.loads(args.baseline.read_text(encoding="utf-8"))
    leg = list(reference.get("legs") or [])[3]
    first, last = args.endpoint_frames
    bank_frames = selected_frames(first, last, args.stride)
    # Validate on interleaved live images that were not copied into the bank.
    # Exact self-matches trivially produce 1,200 inliers and are not evidence
    # that a neighboring online frame can be recognized.
    frames = selected_frames(first + 2, last - 3, args.stride)

    def gray(frame_index: int) -> np.ndarray:
        image_path = args.frame_dir / f"query_{frame_index:06d}.jpg"
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_path)
        return image

    with np.load(args.base_bank, allow_pickle=False) as base, np.load(
        args.bank, allow_pickle=False
    ) as bank:
        base_anchor_count = int(len(base["anchor_names"]))
        total_anchor_count = int(len(bank["anchor_names"]))
        base_prefix_preserved = all(
            np.array_equal(bank[key][:base_anchor_count], base[key])
            for key in (
                "anchor_names",
                "anchor_progress",
                "anchor_centers",
                "anchor_headings",
                "anchor_from",
                "anchor_to",
                "source_frames",
            )
        )
        source_ids = [
            str(value) for value in bank["anchor_source_replay_ids"].tolist()
        ]
        endpoint_indices = [
            index
            for index in range(base_anchor_count, total_anchor_count)
            if source_ids[index] == args.source_replay_id
        ]
        endpoint_progress = [
            float(bank["anchor_progress"][index]) for index in endpoint_indices
        ]
        endpoint_source_frames = [
            int(bank["source_frames"][index]) for index in endpoint_indices
        ]

    recovery = PatrolVisualRouteRecovery(args.bank)
    observations: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    runtimes_ms: list[float] = []
    # Recreate the failed live state: the published model is still near 8%,
    # while five bounded commands allow only 62% of route progress. Endpoint
    # proof must still be discoverable, but ordinary position publication must
    # remain capped until the controller latches the verified waypoint.
    published_progress = 0.080452
    for sequence_index, frame_index in enumerate(frames, start=1):
        started = time.perf_counter()
        observation, diagnostic = recovery.recover(
            gray=gray(frame_index),
            segment_start=leg["from"],
            segment_end=leg["to"],
            segment_key=("point1-endpoint-extension-audit", 1, 4),
            translation_locked=False,
            progress_hint=published_progress,
            progress_ceiling=0.619611,
            recovery_hover=True,
            recovery_minimum_inliers=50,
            independent_progress=True,
            allow_endpoint_only_recovery=True,
            sequence_index=sequence_index,
        )
        runtimes_ms.append((time.perf_counter() - started) * 1000.0)
        if observation is None:
            rejections.append(
                {
                    "frame": frame_index,
                    "reason": str(diagnostic.get("reason") or "unknown"),
                    "best_inliers": int(diagnostic.get("best_inliers") or 0),
                    "endpoint_only_hits": int(
                        diagnostic.get("endpoint_only_hits") or 0
                    ),
                    "endpoint_only_verified": bool(
                        diagnostic.get("endpoint_only_verified")
                    ),
                    "endpoint_only_best_progress": diagnostic.get(
                        "endpoint_only_best_progress"
                    ),
                    "endpoint_only_best_inliers": int(
                        diagnostic.get("endpoint_only_best_inliers") or 0
                    ),
                    "endpoint_only_best_anchor": diagnostic.get(
                        "endpoint_only_best_anchor"
                    ),
                }
            )
            continue
        published_progress = float(observation["progress"])
        recovery.commit_published_progress(published_progress)
        observations.append(
            {
                "frame": frame_index,
                "published_progress": published_progress,
                "inliers": int(observation.get("inliers") or 0),
                "source_replay_id": observation.get("source_replay_id"),
                "endpoint_verified": bool(observation.get("endpoint_verified")),
                "endpoint_hits": int(observation.get("endpoint_hits") or 0),
                "endpoint_best_progress": observation.get(
                    "endpoint_best_progress"
                ),
                "endpoint_best_inliers": int(
                    observation.get("endpoint_best_inliers") or 0
                ),
                "endpoint_best_anchor": observation.get("endpoint_best_anchor"),
                "command_progress_guarded": bool(
                    observation.get("command_progress_guarded")
                ),
            }
        )

    negative_recovery = PatrolVisualRouteRecovery(args.bank)
    negative_observation, negative_diagnostic = negative_recovery.recover(
        gray=gray(args.negative_frame),
        segment_start=leg["from"],
        segment_end=leg["to"],
        segment_key=("point1-endpoint-extension-negative", 1, 4),
        translation_locked=False,
        progress_hint=0.080452,
        progress_ceiling=0.619611,
        recovery_hover=True,
        recovery_minimum_inliers=50,
        independent_progress=True,
        allow_endpoint_only_recovery=True,
        sequence_index=1,
    )
    negative_verified = bool(
        negative_observation is not None
        and negative_observation.get("endpoint_verified") is True
    )

    # Replay the already-working 4->1 recording twice through the extended
    # bank.  This is intentionally separate from the stale-endpoint test: it
    # proves that appending the new Point-1 views did not change the normal
    # route matches, their progress, or the lap-boundary reset behavior.
    preserved_tail_laps: list[dict[str, Any]] = []
    preserved_tail_runtimes_ms: list[float] = []
    preserved_recovery = PatrolVisualRouteRecovery(args.bank)
    for lap_number in (1, 2):
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        published = 0.0
        for sequence_index, frame_index in enumerate(range(2891, 2966), start=1):
            image_path = (
                args.preserved_tail_frame_dir / f"query_{frame_index:06d}.jpg"
            )
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(image_path)
            started = time.perf_counter()
            observation, diagnostic = preserved_recovery.recover(
                gray=image,
                segment_start=leg["from"],
                segment_end=leg["to"],
                segment_key=("point1-endpoint-preservation", lap_number, 4),
                translation_locked=False,
                progress_hint=published,
                independent_progress=True,
                sequence_index=sequence_index,
            )
            preserved_tail_runtimes_ms.append(
                (time.perf_counter() - started) * 1000.0
            )
            if observation is None:
                rejected.append(
                    {
                        "frame": frame_index,
                        "reason": str(diagnostic.get("reason") or "unknown"),
                    }
                )
                continue
            next_progress = float(observation["progress"])
            if next_progress + 1.0e-9 < published:
                raise RuntimeError("Preserved tail route progress moved backwards")
            published = next_progress
            preserved_recovery.commit_published_progress(published)
            accepted.append(
                {
                    "frame": frame_index,
                    "progress": published,
                    "source_replay_id": observation.get("source_replay_id"),
                    "endpoint_verified": bool(
                        observation.get("endpoint_verified")
                    ),
                }
            )
        preserved_tail_laps.append(
            {
                "lap": lap_number,
                "accepted": len(accepted),
                "rejections": rejected,
                "final_progress": published,
                "endpoint_verified": bool(
                    accepted and accepted[-1]["endpoint_verified"]
                ),
                "all_matches_use_preserved_source": bool(
                    accepted
                    and all(
                        item["source_replay_id"]
                        == args.preserved_tail_source_replay_id
                        for item in accepted
                    )
                ),
            }
        )

    preserved_tail_ok = bool(
        len(preserved_tail_laps) == 2
        and all(
            item["accepted"] == 74
            and item["rejections"]
            == [{"frame": 2891, "reason": "visual_route_acquiring"}]
            and float(item["final_progress"]) >= 0.99
            and item["endpoint_verified"] is True
            and item["all_matches_use_preserved_source"] is True
            for item in preserved_tail_laps
        )
        and float(
            np.percentile(np.asarray(preserved_tail_runtimes_ms), 95)
        )
        <= 120.0
    )

    final = observations[-1] if observations else {}
    passed = bool(
        base_prefix_preserved
        and len(endpoint_indices) >= 6
        and endpoint_progress
        and all(abs(value - 1.0) <= 1.0e-12 for value in endpoint_progress)
        and endpoint_source_frames == bank_frames
        and final.get("endpoint_verified") is True
        and int(final.get("endpoint_hits") or 0) >= 3
        and float(final.get("endpoint_best_progress") or 0.0) >= 0.99
        and int(final.get("endpoint_best_inliers") or 0) >= 90
        and str(final.get("endpoint_best_anchor") or "").startswith(
            args.source_replay_id + "/"
        )
        and float(final.get("published_progress") or 0.0) <= 0.619611 + 1.0e-6
        and final.get("command_progress_guarded") is True
        and not negative_verified
        and preserved_tail_ok
        and float(np.percentile(np.asarray(runtimes_ms), 95)) <= 150.0
    )
    payload = {
        "passed": passed,
        "kind": "atlas_point1_endpoint_extension_audit",
        "baseline": str(args.baseline.resolve()),
        "base_bank": str(args.base_bank.resolve()),
        "bank": str(args.bank.resolve()),
        "source_replay_id": args.source_replay_id,
        "endpoint_frames": [first, last],
        "bank_endpoint_frames": bank_frames,
        "selected_endpoint_frames": frames,
        "negative_frame": args.negative_frame,
        "preservation": {
            "base_anchor_count": base_anchor_count,
            "total_anchor_count": total_anchor_count,
            "base_prefix_preserved": base_prefix_preserved,
            "route_anchors_replaced": False,
            "added_endpoint_anchor_count": len(endpoint_indices),
        },
        "stale_progress_recovery": {
            "initial_progress": 0.080452,
            "command_progress_ceiling": 0.619611,
            "observations": observations,
            "rejections": rejections,
            "p95_runtime_ms": float(np.percentile(np.asarray(runtimes_ms), 95)),
            "max_runtime_ms": max(runtimes_ms),
        },
        "negative_endpoint_verified": negative_verified,
        "negative_reason": str(negative_diagnostic.get("reason") or ""),
        "preserved_two_lap_tail": {
            "source_replay_id": args.preserved_tail_source_replay_id,
            "frame_range": [2891, 2965],
            "laps": preserved_tail_laps,
            "p95_runtime_ms": float(
                np.percentile(np.asarray(preserved_tail_runtimes_ms), 95)
            ),
            "max_runtime_ms": max(preserved_tail_runtimes_ms),
            "passed": preserved_tail_ok,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
