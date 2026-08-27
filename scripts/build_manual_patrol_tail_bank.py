#!/usr/bin/env python3
"""Build a patrol ORB bank with an audited manual Point-4 tail."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from patrol_visual_route_recovery import extend_bank_with_recorded_segments


def frame_range(value: str) -> tuple[int, int]:
    try:
        first, last = (int(item) for item in value.split(":", 1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("frame range must be FIRST:LAST") from exc
    if first < 0 or last <= first:
        raise argparse.ArgumentTypeError("frame range must increase")
    return first, last


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--base-bank", required=True, type=Path)
    parser.add_argument("--frame-dir", required=True, type=Path)
    parser.add_argument("--source-replay-id", required=True)
    parser.add_argument("--point4-heading", type=frame_range, default=(2878, 2890))
    parser.add_argument("--leg4-translation", type=frame_range, default=(2891, 2965))
    parser.add_argument("--point1-heading", type=frame_range, default=(3035, 3039))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    common = {
        "source_replay_id": str(args.source_replay_id),
        "frame_dir": args.frame_dir,
    }
    p4_heading_first, p4_heading_last = args.point4_heading
    leg4_first, leg4_last = args.leg4_translation
    p1_heading_first, p1_heading_last = args.point1_heading
    segments = [
        {
            **common,
            "leg_index": 3,
            "start_frame": p4_heading_first,
            "end_frame": p4_heading_last,
            "progress_mode": "start",
            "heading_priority": 100,
            "stride": 2,
        },
        {
            **common,
            "leg_index": 3,
            "start_frame": leg4_first,
            "end_frame": leg4_last,
            "progress_mode": "motion",
            "heading_priority": 0,
            "stride": 1,
        },
        {
            **common,
            "leg_index": 0,
            "start_frame": p1_heading_first,
            "end_frame": p1_heading_last,
            "progress_mode": "start",
            "heading_priority": 100,
            "stride": 1,
        },
    ]
    result = extend_bank_with_recorded_segments(
        args.base_bank,
        args.out,
        reference_path=args.baseline,
        segments=segments,
        anchor_stride=1,
        replace_leg_indices={3},
    )
    result.update(
        {
            "point4_heading_frames": [p4_heading_first, p4_heading_last],
            "leg4_translation_frames": [leg4_first, leg4_last],
            "point1_heading_frames": [p1_heading_first, p1_heading_last],
            "landing_frames_excluded_after": p1_heading_last,
        }
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
