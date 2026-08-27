#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import atlas_app_server as atlas


def write_status(path: Path, state: str, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"state": state, "message": message, "updated_at": time.time()},
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a new ATLAS map directly from a local video file."
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()

    video = args.video.expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Map video not found: {video}")

    log_path = args.log.expanduser().resolve()
    status_path = args.status.expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_append_log = atlas.append_log

    def persistent_append_log(kind: str, line: str) -> None:
        original_append_log(kind, line)
        if kind == "map" and line.rstrip():
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line.rstrip()}\n")

    atlas.append_log = persistent_append_log
    write_status(status_path, "running", f"Building a new map from {video}")
    persistent_append_log("map", f"Persistent direct map build started from {video}.")
    try:
        atlas.run_map_from_video(video, f"direct local video {video.name}")
    except BaseException as exc:
        persistent_append_log("map", f"FATAL: {type(exc).__name__}: {exc}")
        write_status(status_path, "failed", f"{type(exc).__name__}: {exc}")
        raise
    persistent_append_log("map", "Persistent direct map build completed successfully.")
    write_status(status_path, "completed", "New map build completed successfully.")


if __name__ == "__main__":
    main()
