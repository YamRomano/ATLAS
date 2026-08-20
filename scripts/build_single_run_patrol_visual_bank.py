#!/usr/bin/env python3
"""Build all four patrol legs from one verified live DJI recording."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from patrol_visual_route_recovery import extend_bank_with_recorded_segments


DEFAULT_PHASES = (
    # leg, aligned departure, forward translation, endpoint hold
    (0, (654, 660), (662, 747), (747, 770)),
    (1, (958, 966), (967, 1020), (1020, 1045)),
    (2, (1214, 1221), (1222, 1368), (1368, 1383)),
    (3, (1538, 1559), (1560, 1650), (1650, 1700)),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--base-bank", required=True, type=Path)
    parser.add_argument("--frame-dir", required=True, type=Path)
    parser.add_argument("--source-replay-id", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--translation-stride", type=int, default=2)
    parser.add_argument("--endpoint-stride", type=int, default=2)
    args = parser.parse_args()

    common = {
        "source_replay_id": str(args.source_replay_id),
        "frame_dir": args.frame_dir,
    }
    segments: list[dict[str, object]] = []
    for leg_index, heading, translation, endpoint in DEFAULT_PHASES:
        segments.extend(
            [
                {
                    **common,
                    "leg_index": leg_index,
                    "start_frame": heading[0],
                    "end_frame": heading[1],
                    "progress_mode": "start",
                    "heading_priority": 100,
                    "stride": 1,
                },
                {
                    **common,
                    "leg_index": leg_index,
                    "start_frame": translation[0],
                    "end_frame": translation[1],
                    "progress_mode": "linear",
                    "heading_priority": 0,
                    "stride": max(1, int(args.translation_stride)),
                },
                {
                    **common,
                    "leg_index": leg_index,
                    "start_frame": endpoint[0],
                    "end_frame": endpoint[1],
                    "progress_mode": "end",
                    "heading_priority": 0,
                    "stride": max(1, int(args.endpoint_stride)),
                },
            ]
        )

    result = extend_bank_with_recorded_segments(
        args.base_bank,
        args.out,
        reference_path=args.baseline,
        segments=segments,
        anchor_stride=1,
        replace_leg_indices={0, 1, 2, 3},
    )
    result.update(
        {
            "kind": "atlas_single_run_patrol_visual_bank",
            "all_route_legs_replaced": True,
            "phase_frames": [
                {
                    "leg_index": leg_index,
                    "heading": list(heading),
                    "translation": list(translation),
                    "endpoint": list(endpoint),
                }
                for leg_index, heading, translation, endpoint in DEFAULT_PHASES
            ],
        }
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
