#!/usr/bin/env python3
"""Merge trusted poses without replacing any pose already represented by a reference model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from colmap_io import read_images_model


def basename(pose: dict[str, Any]) -> str:
    return Path(str(pose.get("image_name", ""))).name


def accepted(pose: object) -> bool:
    if not isinstance(pose, dict):
        return False
    return bool(
        pose.get("success") is True
        and not pose.get("held_pose")
        and not pose.get("output_rejected")
        and isinstance(pose.get("R"), list)
        and isinstance(pose.get("t"), list)
        and isinstance(pose.get("center"), list)
        and isinstance(pose.get("time_sec"), (int, float))
        and math.isfinite(float(pose["time_sec"]))
    )


def load(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    poses = payload.get("poses", payload) if isinstance(payload, dict) else payload
    if not isinstance(poses, list):
        raise RuntimeError(f"Invalid pose stream: {path}")
    return [dict(pose) for pose in poses if accepted(pose)]


def frame_index(pose: dict[str, Any]) -> int:
    name = Path(basename(pose)).stem
    return int(name.rsplit("_", 1)[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-poses", type=Path, required=True)
    parser.add_argument("--recovered-poses", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--new-prefix", default="enhancement/")
    args = parser.parse_args()

    reference_names = {
        image.name for image in read_images_model(args.reference_model.resolve()).values()
    }
    base = load(args.base_poses)
    recovered = load(args.recovered_poses)
    base_by_name = {basename(pose): pose for pose in base}
    if len(base_by_name) != len(base):
        raise RuntimeError("The base pose stream contains duplicate target image names.")

    additions: dict[str, dict[str, Any]] = {}
    skipped_reference: list[str] = []
    skipped_base: list[str] = []
    for pose in recovered:
        name = basename(pose)
        if name in base_by_name:
            skipped_base.append(name)
            continue
        if f"{args.new_prefix}{name}" in reference_names:
            skipped_reference.append(name)
            continue
        current = additions.get(name)
        if current is not None:
            raise RuntimeError(f"Duplicate recovered target pose: {name}")
        additions[name] = pose

    merged = sorted(
        [*base_by_name.values(), *additions.values()],
        key=frame_index,
    )
    output = {
        "mode": "reference_preserving_trusted_pose_merge",
        "complete": True,
        "processed_count": len(merged),
        "accepted_count": len(merged),
        "held_count": 0,
        "failed_count": 0,
        "poses": merged,
    }
    summary = {
        "mode": output["mode"],
        "base_pose_count": len(base_by_name),
        "recovered_pose_count": len(recovered),
        "reference_registered_images": len(reference_names),
        "skipped_existing_base": len(skipped_base),
        "skipped_existing_reference": len(skipped_reference),
        "new_pose_count": len(additions),
        "merged_pose_count": len(merged),
        "new_pose_names": sorted(additions),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    summary_path = args.summary_out or args.out.with_name(f"{args.out.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
