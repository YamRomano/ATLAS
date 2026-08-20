#!/usr/bin/env python3
"""Add a verified live Point-1 endpoint view without replacing route anchors."""

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
    parser.add_argument("--endpoint-frames", required=True, type=frame_range)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    first, last = args.endpoint_frames
    result = extend_bank_with_recorded_segments(
        args.base_bank,
        args.out,
        reference_path=args.baseline,
        segments=[
            {
                "source_replay_id": str(args.source_replay_id),
                "frame_dir": args.frame_dir,
                "leg_index": 3,
                "start_frame": first,
                "end_frame": last,
                "progress_mode": "end",
                "heading_priority": 0,
                "stride": max(1, int(args.stride)),
            }
        ],
        anchor_stride=max(1, int(args.stride)),
    )
    result.update(
        {
            "kind": "atlas_point1_endpoint_bank_extension",
            "endpoint_frames": [first, last],
            "endpoint_stride": max(1, int(args.stride)),
            "route_anchors_replaced": False,
        }
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
