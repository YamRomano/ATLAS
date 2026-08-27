#!/usr/bin/env python3
"""Audit a single-recording patrol bank over two complete visual laps."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from build_single_run_patrol_visual_bank import DEFAULT_PHASES
from patrol_visual_route_recovery import PatrolVisualRouteRecovery


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--frame-dir", required=True, type=Path)
    parser.add_argument("--source-replay-id", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sample-stride", type=int, default=5)
    args = parser.parse_args()

    reference = json.loads(args.baseline.read_text(encoding="utf-8"))
    legs = reference["legs"]
    expected_source = str(args.source_replay_id)
    with np.load(args.bank, allow_pickle=False) as contents:
        anchor_sources = [
            str(value)
            for value in contents["anchor_source_replay_ids"].tolist()
        ]
        advertised_sources = [
            str(value) for value in contents["source_replay_ids"].tolist()
        ]
        anchor_progress = np.asarray(contents["anchor_progress"], dtype=float)
        anchor_from = np.asarray(contents["anchor_from"], dtype=float)
        anchor_to = np.asarray(contents["anchor_to"], dtype=float)
        anchor_count = int(len(contents["anchor_names"]))

    source_is_single = (
        set(anchor_sources) == {expected_source}
        and advertised_sources == [expected_source]
    )
    leg_inventory = []
    for leg_index, leg in enumerate(legs):
        start = np.asarray(leg["from"], dtype=float)
        end = np.asarray(leg["to"], dtype=float)
        mask = (
            np.linalg.norm(anchor_from[:, [0, 2]] - start[[0, 2]], axis=1)
            <= 0.08
        ) & (
            np.linalg.norm(anchor_to[:, [0, 2]] - end[[0, 2]], axis=1)
            <= 0.08
        )
        progress = anchor_progress[mask]
        leg_inventory.append(
            {
                "leg_index": leg_index,
                "anchors": int(np.count_nonzero(mask)),
                "minimum_progress": float(np.min(progress)) if len(progress) else None,
                "maximum_progress": float(np.max(progress)) if len(progress) else None,
                "covers_complete_leg": bool(
                    len(progress)
                    and float(np.min(progress)) <= 0.001
                    and float(np.max(progress)) >= 0.999
                ),
            }
        )

    recovery = PatrolVisualRouteRecovery(args.bank)
    runtime_ms: list[float] = []
    laps: list[dict[str, object]] = []
    sample_stride = max(1, int(args.sample_stride))
    for lap in (1, 2):
        lap_result: dict[str, object] = {"lap": lap, "legs": []}
        for leg_index, heading, translation, endpoint in DEFAULT_PHASES:
            leg = legs[leg_index]
            aligned_gray = cv2.imread(
                str(args.frame_dir / f"query_{heading[1]:06d}.jpg"),
                cv2.IMREAD_GRAYSCALE,
            )
            if aligned_gray is None:
                raise FileNotFoundError(heading[1])
            aligned, aligned_diagnostic = recovery.departure_heading_alignment(
                gray=aligned_gray,
                segment_start=leg["from"],
                segment_end=leg["to"],
                focal_px=882.4866783165957,
                minimum_inliers=50,
            )

            selected = list(range(translation[0], translation[1] + 1, sample_stride))
            selected.extend(range(endpoint[0], endpoint[1] + 1, sample_stride))
            selected = sorted(set([*selected, translation[1], endpoint[1]]))
            published = 0.0
            observations: list[dict[str, object]] = []
            rejections: list[dict[str, object]] = []
            for frame_index in selected:
                image_path = args.frame_dir / f"query_{frame_index:06d}.jpg"
                gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    raise FileNotFoundError(image_path)
                started = time.perf_counter()
                observation, diagnostic = recovery.recover(
                    gray=gray,
                    segment_start=leg["from"],
                    segment_end=leg["to"],
                    segment_key=("single-source-audit", lap, leg_index),
                    translation_locked=False,
                    progress_hint=published,
                    independent_progress=True,
                    # Live ATLAS supplies the source/query frame sequence, not
                    # the audit loop ordinal. Preserve real skipped-frame
                    # distance so the temporal matcher may advance equally.
                    sequence_index=frame_index,
                )
                runtime_ms.append((time.perf_counter() - started) * 1000.0)
                if observation is None:
                    rejections.append(
                        {
                            "frame": frame_index,
                            "reason": diagnostic.get("reason"),
                        }
                    )
                    continue
                progress = float(observation["progress"])
                if progress + 1.0e-9 < published:
                    raise RuntimeError(
                        f"Leg {leg_index + 1} regressed {published:.6f}->{progress:.6f}"
                    )
                published = progress
                recovery.commit_published_progress(progress)
                observations.append(
                    {
                        "frame": frame_index,
                        "progress": progress,
                        "inliers": int(observation["inliers"]),
                        "source_replay_id": observation.get("source_replay_id"),
                        "endpoint_verified": bool(
                            observation.get("endpoint_verified")
                        ),
                    }
                )
            leg_passed = bool(
                aligned is not None
                and abs(float(aligned["correction_deg"])) <= 5.0
                and len(observations) >= max(3, len(selected) - 3)
                and published >= 0.99
                and observations[-1]["endpoint_verified"]
                and all(
                    item["source_replay_id"] == expected_source
                    for item in observations
                )
            )
            lap_result["legs"].append(
                {
                    "leg_index": leg_index,
                    "heading_frame": heading[1],
                    "heading_correction_deg": (
                        float(aligned["correction_deg"]) if aligned else None
                    ),
                    "heading_inliers": (
                        int(aligned["inliers"])
                        if aligned and aligned.get("inliers") is not None
                        else 0
                    ),
                    "heading_reason": aligned_diagnostic.get("reason"),
                    "selected_frames": len(selected),
                    "accepted_frames": len(observations),
                    "rejections": rejections,
                    "final_progress": published,
                    "endpoint_verified": bool(
                        observations and observations[-1]["endpoint_verified"]
                    ),
                    "passed": leg_passed,
                }
            )
        lap_result["passed"] = all(
            bool(leg["passed"]) for leg in lap_result["legs"]
        )
        laps.append(lap_result)

    passed = bool(
        source_is_single
        and all(item["covers_complete_leg"] for item in leg_inventory)
        and all(bool(lap["passed"]) for lap in laps)
    )
    result = {
        "passed": passed,
        "kind": "atlas_single_run_patrol_visual_bank_audit",
        "baseline": str(args.baseline.resolve()),
        "bank": str(args.bank.resolve()),
        "source_replay_id": expected_source,
        "source_is_single": source_is_single,
        "advertised_sources": advertised_sources,
        "anchor_count": anchor_count,
        "leg_inventory": leg_inventory,
        "laps": laps,
        "runtime_ms": {
            "mean": statistics.fmean(runtime_ms) if runtime_ms else 0.0,
            "p95": percentile(runtime_ms, 0.95),
            "maximum": max(runtime_ms) if runtime_ms else 0.0,
        },
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
