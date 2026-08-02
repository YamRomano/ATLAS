#!/usr/bin/env python3
"""Prepare an isolated, ordered frame slice for anchor-constrained localization.

The first frame is a previously trusted anchor.  Its accepted pose and exact
TSolve correspondence case are rewritten to a local alias, while subsequent
aliases walk either forward or backward through the original 10-FPS video.
No source frames or production map artifacts are modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def basename(value: object) -> str:
    return Path(str(value or "")).name


def frame_name(index: int) -> str:
    return f"manual_patrol_{index:06d}.jpg"


def load_frame_rows(source_dir: Path) -> dict[str, dict[str, str]]:
    csv_path = source_dir / "frames.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_name = {basename(row.get("image_name")): row for row in rows}
    if len(by_name) != len(rows):
        raise RuntimeError(f"Duplicate frame names in {csv_path}")
    return by_name


def trusted_anchor_pose(
    pose_stream: Path,
    anchor_name: str,
    anchor_index: int,
    alias_mapping: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    document = json.loads(pose_stream.read_text(encoding="utf-8"))

    def matches_anchor(pose: dict[str, Any]) -> bool:
        name = basename(pose.get("image_name"))
        if name == anchor_name:
            return True
        item = alias_mapping.get(name)
        return item is not None and int(item["raw_10fps_index"]) == anchor_index

    matches = [
        pose
        for pose in document.get("poses") or []
        if matches_anchor(pose)
        and pose.get("success")
        and not pose.get("held_pose")
        and not pose.get("output_rejected")
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one trusted pose for {anchor_name} in {pose_stream}, found {len(matches)}"
        )
    return dict(matches[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-frames", required=True, type=Path)
    parser.add_argument("--pose-stream", required=True, type=Path)
    parser.add_argument("--resume-case", required=True, type=Path)
    parser.add_argument(
        "--source-alias-manifest",
        type=Path,
        default=None,
        help="Manifest that maps aliases in a recovered pose/case back to raw 10-FPS indices.",
    )
    parser.add_argument("--anchor-index", required=True, type=int)
    parser.add_argument("--end-index", required=True, type=int)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--alias-prefix", default="gap")
    args = parser.parse_args()

    source_dir = args.source_frames.resolve()
    pose_stream = args.pose_stream.resolve()
    resume_case = args.resume_case.resolve()
    out_dir = args.out_dir.resolve()
    alias_mapping: dict[str, dict[str, Any]] = {}
    if args.source_alias_manifest is not None:
        alias_document = json.loads(
            args.source_alias_manifest.resolve().read_text(encoding="utf-8")
        )
        alias_mapping = {
            str(item["alias"]): item for item in alias_document.get("mapping") or []
        }

    step = 1 if args.end_index >= args.anchor_index else -1
    indices = list(range(args.anchor_index, args.end_index + step, step))
    if len(indices) < 2:
        raise ValueError("The query must include an anchor and at least one subsequent frame")

    frame_rows = load_frame_rows(source_dir)
    anchor_source_name = frame_name(args.anchor_index)
    anchor_pose = trusted_anchor_pose(
        pose_stream, anchor_source_name, args.anchor_index, alias_mapping
    )

    source_case_meta = json.loads((resume_case / "input.json").read_text(encoding="utf-8"))
    case_source_name = basename(source_case_meta.get("image_name"))
    case_alias_item = alias_mapping.get(case_source_name)
    case_matches_anchor = case_source_name == anchor_source_name or (
        case_alias_item is not None
        and int(case_alias_item["raw_10fps_index"]) == args.anchor_index
    )
    if not case_matches_anchor:
        raise RuntimeError(
            f"Resume case is for {case_source_name}, expected anchor {anchor_source_name}"
        )

    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing recovery input: {out_dir}")
    query_dir = out_dir / "query"
    case_dir = out_dir / "resume_case"
    query_dir.mkdir(parents=True)
    case_dir.mkdir(parents=True)

    mapping: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for sequence_index, source_index in enumerate(indices):
        source_name = frame_name(source_index)
        source_path = source_dir / source_name
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        row = frame_rows.get(source_name)
        if row is None:
            raise RuntimeError(f"No frames.csv row for {source_name}")
        alias = f"{args.alias_prefix}_{sequence_index:06d}.jpg"
        (query_dir / alias).symlink_to(source_path)
        item = {
            "sequence_index": sequence_index,
            "alias": alias,
            "source_image_name": source_name,
            "raw_10fps_index": source_index,
            "source_frame": int(row["source_frame"]),
            "time_sec": float(row["time_sec"]),
            "direction": "forward" if step > 0 else "backward",
        }
        mapping.append(item)
        csv_rows.append(
            {
                "image_name": alias,
                "source_frame": row["source_frame"],
                "time_sec": row["time_sec"],
                "width": row.get("width", ""),
                "height": row.get("height", ""),
                "original_image_name": source_name,
                "raw_10fps_index": source_index,
            }
        )

    with (query_dir / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    alias_anchor = mapping[0]["alias"]
    anchor_pose["image_name"] = f"query/{alias_anchor}"
    anchor_pose["time_sec"] = mapping[0]["time_sec"]
    seed_document = {
        "mode": "anchor_gap_seed",
        "replay_id": out_dir.name,
        "frame_source": str(source_dir),
        "expected_count": len(mapping),
        "processed_count": 1,
        "accepted_count": 1,
        "held_count": 0,
        "failed_count": 0,
        "complete": False,
        "current_frame": anchor_pose,
        "current_frame_time_sec": anchor_pose["time_sec"],
        "poses": [anchor_pose],
    }
    atomic_write_json(out_dir / "seed_pose.json", seed_document)

    for filename in ("p2d.csv", "p3d.csv"):
        shutil.copy2(resume_case / filename, case_dir / filename)
    case_meta = dict(source_case_meta)
    case_meta["image_name"] = f"query/{alias_anchor}"
    case_meta["time_sec"] = mapping[0]["time_sec"]
    atomic_write_json(case_dir / "input.json", case_meta)

    manifest = {
        "source_frames": str(source_dir),
        "source_pose_stream": str(pose_stream),
        "source_resume_case": str(resume_case),
        "source_alias_manifest": (
            str(args.source_alias_manifest.resolve())
            if args.source_alias_manifest is not None
            else None
        ),
        "anchor_source_name": anchor_source_name,
        "anchor_index": args.anchor_index,
        "end_index": args.end_index,
        "direction": "forward" if step > 0 else "backward",
        "query_frame_count": len(mapping),
        "query_frames_to_process": len(mapping) - 1,
        "query_dir": str(query_dir),
        "seed_pose": str(out_dir / "seed_pose.json"),
        "resume_case": str(case_dir),
        "mapping": mapping,
    }
    atomic_write_json(out_dir / "manifest.json", manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key != "mapping"}, indent=2))


if __name__ == "__main__":
    main()
