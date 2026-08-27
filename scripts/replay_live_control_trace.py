#!/usr/bin/env python3
"""Replay a recorded DJI mission's controller context beside finite live frames.

The localization process reads one fresh JSON status file.  A saved live run
contains that same state as JSONL progress events, so this utility publishes
the event nearest to the frame currently being processed.  It is intended for
diagnostic replays only; it never sends a flight command.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
import re
import tempfile
import time
from pathlib import Path


FRAME_RE = re.compile(r"(\d{6})(?:\.[^.]+)?$")
PROGRESS_FIELDS = {
    "map_id",
    "patrol_id",
    "baseline_replay_id",
    "lap",
    "patrol_laps",
    "leg_index",
    "segment_start",
    "target",
    "translation_locked",
    "position_anchor",
    "phase",
    "recovery_phase",
    "route_visual_recovery_allowed",
    "require_metric_pose",
    "lap_start_metric_rebootstrap",
    "metric_position_recovery_allowed",
    "post_translation_progress_recovery",
    "physical_translation_active",
    "body_forward_gain",
    "body_lateral_gain",
    "route_pose_epoch",
    "route_pose_epoch_unix",
    "route_pose_epoch_reason",
}


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load_frame_clock(path: Path) -> tuple[list[int], list[float]]:
    indices: list[int] = []
    received: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            indices.append(int(row["source_frame"]))
            received.append(float(row["received_unix"]))
    if not indices:
        raise RuntimeError("The recorded frames CSV is empty.")
    return indices, received


def load_progress_events(
    trace_path: Path,
    frame_indices: list[int],
    received_unix: list[float],
) -> list[tuple[int, dict]]:
    events: list[tuple[int, dict]] = []
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            progress = record.get("progress")
            if record.get("event") != "progress" or not isinstance(progress, dict):
                continue
            updated = float(progress.get("updated_at") or record.get("updated_at") or 0.0)
            clock_index = bisect.bisect_right(received_unix, updated) - 1
            clock_index = max(0, min(clock_index, len(frame_indices) - 1))
            compact = {key: progress[key] for key in PROGRESS_FIELDS if key in progress}
            compact["recorded_trace_updated_at"] = updated
            events.append((frame_indices[clock_index], compact))
    if not events:
        raise RuntimeError("The control trace contains no mission progress events.")
    events.sort(key=lambda item: item[0])
    return events


def current_processed_frame(partial_path: Path, default: int) -> int:
    try:
        payload = json.loads(partial_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    current = payload.get("current_frame")
    if isinstance(current, dict) and current.get("frame_index") is not None:
        return int(current["frame_index"])
    poses = payload.get("poses")
    if isinstance(poses, list) and poses:
        image_name = str((poses[-1] or {}).get("image_name") or "")
        match = FRAME_RE.search(image_name)
        if match:
            return int(match.group(1))
    return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--frames-csv", type=Path, required=True)
    parser.add_argument("--partial-pose", type=Path, required=True)
    parser.add_argument("--status-out", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--stop-file", type=Path)
    args = parser.parse_args()

    frame_indices, received_unix = load_frame_clock(args.frames_csv)
    events = load_progress_events(args.trace, frame_indices, received_unix)
    event_frames = [item[0] for item in events]
    frame_index = args.start_frame
    last_published: tuple[int, int] | None = None

    while frame_index <= args.end_frame:
        if args.stop_file is not None and args.stop_file.exists():
            break
        frame_index = current_processed_frame(args.partial_pose, frame_index)
        event_index = bisect.bisect_right(event_frames, frame_index) - 1
        # Frames before the mission's first progress event use its initial
        # route contract, rather than publishing a non-mission status.
        event_index = max(0, min(event_index, len(events) - 1))
        marker = (frame_index, event_index)
        if marker != last_published:
            progress = dict(events[event_index][1])
            progress["simulated_source_frame"] = frame_index
            progress["updated_at"] = time.time()
            atomic_write_json(
                args.status_out,
                {
                    "status": "running",
                    "command": "mission",
                    "updated_at": time.time(),
                    "progress": progress,
                },
            )
            last_published = marker
        if frame_index >= args.end_frame:
            break
        time.sleep(0.02)


if __name__ == "__main__":
    main()
