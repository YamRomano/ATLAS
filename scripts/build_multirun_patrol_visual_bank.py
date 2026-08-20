#!/usr/bin/env python3
"""Extend the patrol visual-recovery bank with independent recorded flights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from patrol_visual_route_recovery import extend_bank_with_recorded_segments


def translation_ranges(boundaries: dict, first: int, last: int) -> list[dict]:
    candidates = [
        (0, boundaries.get("point1"), boundaries.get("point2_arrival")),
        (1, boundaries.get("point2_departure"), boundaries.get("point3_arrival")),
        (2, boundaries.get("point3_departure"), boundaries.get("point4_arrival")),
        (3, boundaries.get("point4_departure"), boundaries.get("point1_return")),
    ]
    ranges: list[dict] = []
    for leg_index, raw_start, raw_end in candidates:
        if raw_start is None or raw_end is None:
            continue
        start = max(first, int(raw_start))
        end = min(last, int(raw_end))
        if end > start:
            ranges.append(
                {
                    "leg_index": leg_index,
                    "start_frame": start,
                    "end_frame": end,
                }
            )
    return ranges


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--base-bank", required=True, type=Path)
    parser.add_argument("--composite-plan", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--viewer-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=1200)
    args = parser.parse_args()

    plan = json.loads(args.composite_plan.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    map_id = str(plan.get("map_id") or "")
    baseline_replay_id = str(plan.get("baseline_replay_id") or "")
    map_entry = next(
        (entry for entry in manifest.get("maps") or [] if entry.get("id") == map_id),
        None,
    )
    if not isinstance(map_entry, dict):
        raise RuntimeError(f"Composite map is missing from the manifest: {map_id}")
    replay_by_id = {
        str(replay.get("id") or ""): replay
        for replay in map_entry.get("replays") or []
        if isinstance(replay, dict)
    }
    viewer_root = args.viewer_root.resolve()
    segments: list[dict] = []
    for source in plan.get("second_lap_segments") or []:
        source_replay_id = str(source.get("source_replay_id") or "")
        if not source_replay_id or source_replay_id == baseline_replay_id:
            continue
        replay = replay_by_id.get(source_replay_id)
        if not isinstance(replay, dict):
            raise RuntimeError(f"Recorded source replay is missing: {source_replay_id}")
        frame_url = str(replay.get("query_frame_base_url") or "").strip()
        frame_dir = (viewer_root / frame_url).resolve()
        if viewer_root not in frame_dir.parents or not frame_dir.is_dir():
            raise RuntimeError(f"Unsafe or missing frame directory: {frame_dir}")
        for route_range in translation_ranges(
            dict(source.get("phase_boundaries") or {}),
            int(source["start_frame"]),
            int(source["end_frame"]),
        ):
            segments.append(
                {
                    **route_range,
                    "source_replay_id": source_replay_id,
                    "frame_dir": frame_dir,
                }
            )
    if not segments:
        raise RuntimeError("The composite plan contains no independent translation segments")
    result = extend_bank_with_recorded_segments(
        args.base_bank,
        args.out,
        reference_path=args.baseline,
        segments=segments,
        anchor_stride=args.stride,
        max_features=args.max_features,
    )
    result["segments"] = [
        {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in segment.items()
        }
        for segment in segments
    ]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
