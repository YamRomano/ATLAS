#!/usr/bin/env python3
"""Audit and normalize an anchor-constrained temporal localization result."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def basename(value: object) -> str:
    return Path(str(value or "")).name


def frame_name(index: int, prefix: str = "manual_patrol_", suffix: str = ".jpg") -> str:
    """Return a source frame name using the recording's filename convention."""
    return f"{prefix}{index:06d}{suffix}"


def trusted_pose(path: Path, image_name: str) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        pose
        for pose in document.get("poses") or []
        if basename(pose.get("image_name")) == image_name
        and pose.get("success")
        and not pose.get("held_pose")
        and not pose.get("output_rejected")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one trusted pose for {image_name}, found {len(matches)}")
    return matches[0]


def center_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    return float(
        np.linalg.norm(
            np.asarray(left["center"], dtype=float) - np.asarray(right["center"], dtype=float)
        )
    )


def rotation_distance_deg(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_r = np.asarray(left["R"], dtype=float)
    right_r = np.asarray(right["R"], dtype=float)
    cosine = float(np.clip((np.trace(right_r @ left_r.T) - 1.0) / 2.0, -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--right-pose-stream", required=True, type=Path)
    parser.add_argument("--right-anchor-index", required=True, type=int)
    parser.add_argument(
        "--right-frame-prefix",
        default=None,
        help="Right-anchor frame prefix. Defaults to the prefix stored in --manifest.",
    )
    parser.add_argument(
        "--right-frame-suffix",
        default=None,
        help="Right-anchor frame suffix. Defaults to the suffix stored in --manifest.",
    )
    parser.add_argument("--out-summary", required=True, type=Path)
    parser.add_argument("--out-normalized-poses", required=True, type=Path)
    parser.add_argument("--max-center-step", type=float, default=0.22)
    parser.add_argument("--max-rotation-step-deg", type=float, default=20.0)
    parser.add_argument("--max-endpoint-center-delta", type=float, default=0.18)
    parser.add_argument("--max-endpoint-rotation-deg", type=float, default=12.0)
    args = parser.parse_args()

    pose_document = json.loads(args.poses.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    mapping = {item["alias"]: item for item in manifest["mapping"]}

    normalized: list[dict[str, Any]] = []
    rejected_aliases: list[str] = []
    unknown_aliases: list[str] = []
    for pose in pose_document.get("poses") or []:
        alias = basename(pose.get("image_name"))
        item = mapping.get(alias)
        if item is None:
            unknown_aliases.append(alias)
            continue
        if (
            not pose.get("success")
            or pose.get("held_pose")
            or pose.get("output_rejected")
            or not pose.get("center")
            or not pose.get("R")
        ):
            rejected_aliases.append(alias)
            continue
        output = dict(pose)
        output["image_name"] = f"query/{item['source_image_name']}"
        output["time_sec"] = float(item["time_sec"])
        output["raw_10fps_index"] = int(item["raw_10fps_index"])
        output["anchor_gap_alias"] = alias
        output["anchor_gap_direction"] = item["direction"]
        normalized.append(output)

    normalized.sort(key=lambda pose: int(pose["raw_10fps_index"]))
    expected_indices = sorted(int(item["raw_10fps_index"]) for item in manifest["mapping"])
    accepted_indices = [int(pose["raw_10fps_index"]) for pose in normalized]
    missing_indices = sorted(set(expected_indices) - set(accepted_indices))

    center_steps: list[dict[str, Any]] = []
    rotation_steps: list[dict[str, Any]] = []
    for left, right in zip(normalized, normalized[1:]):
        edge = {
            "from": int(left["raw_10fps_index"]),
            "to": int(right["raw_10fps_index"]),
        }
        center_steps.append({**edge, "distance": center_distance(left, right)})
        rotation_steps.append({**edge, "degrees": rotation_distance_deg(left, right)})

    right_prefix = (
        args.right_frame_prefix
        if args.right_frame_prefix is not None
        else str(manifest.get("frame_prefix") or "manual_patrol_")
    )
    right_suffix = (
        args.right_frame_suffix
        if args.right_frame_suffix is not None
        else str(manifest.get("frame_suffix") or ".jpg")
    )
    right_name = frame_name(args.right_anchor_index, right_prefix, right_suffix)
    recovered_right = next(
        (
            pose
            for pose in normalized
            if int(pose["raw_10fps_index"]) == args.right_anchor_index
        ),
        None,
    )
    reference_right = trusted_pose(args.right_pose_stream, right_name)
    endpoint_center_delta = (
        center_distance(recovered_right, reference_right) if recovered_right is not None else None
    )
    endpoint_rotation_deg = (
        rotation_distance_deg(recovered_right, reference_right)
        if recovered_right is not None
        else None
    )
    max_center_step = max((edge["distance"] for edge in center_steps), default=0.0)
    max_rotation_step = max((edge["degrees"] for edge in rotation_steps), default=0.0)
    largest_center_edges = sorted(
        center_steps, key=lambda edge: edge["distance"], reverse=True
    )[:10]
    largest_rotation_edges = sorted(
        rotation_steps, key=lambda edge: edge["degrees"], reverse=True
    )[:10]

    passed = bool(
        pose_document.get("complete")
        and not missing_indices
        and not rejected_aliases
        and not unknown_aliases
        and max_center_step <= args.max_center_step
        and max_rotation_step <= args.max_rotation_step_deg
        and endpoint_center_delta is not None
        and endpoint_center_delta <= args.max_endpoint_center_delta
        and endpoint_rotation_deg is not None
        and endpoint_rotation_deg <= args.max_endpoint_rotation_deg
    )
    summary = {
        "passed": passed,
        "complete": bool(pose_document.get("complete")),
        "direction": manifest["direction"],
        "anchor_index": int(manifest["anchor_index"]),
        "right_anchor_index": args.right_anchor_index,
        "expected_count": len(expected_indices),
        "accepted_count": len(accepted_indices),
        "acceptance_ratio": len(accepted_indices) / max(1, len(expected_indices)),
        "missing_indices": missing_indices,
        "rejected_aliases": rejected_aliases,
        "unknown_aliases": unknown_aliases,
        "max_center_step": max_center_step,
        "max_center_step_limit": args.max_center_step,
        "largest_center_edges": largest_center_edges,
        "max_rotation_step_deg": max_rotation_step,
        "max_rotation_step_limit_deg": args.max_rotation_step_deg,
        "largest_rotation_edges": largest_rotation_edges,
        "endpoint_center_delta": endpoint_center_delta,
        "endpoint_center_delta_limit": args.max_endpoint_center_delta,
        "endpoint_rotation_deg": endpoint_rotation_deg,
        "endpoint_rotation_limit_deg": args.max_endpoint_rotation_deg,
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    normalized_document = {
        "mode": "anchor_gap_recovery_normalized",
        "source_poses": str(args.poses.resolve()),
        "source_manifest": str(args.manifest.resolve()),
        "complete": bool(pose_document.get("complete")),
        "passed_audit": passed,
        "processed_count": len(expected_indices),
        "accepted_count": len(normalized),
        "held_count": len(rejected_aliases),
        "failed_count": len(missing_indices),
        "poses": normalized,
    }
    args.out_normalized_poses.parent.mkdir(parents=True, exist_ok=True)
    args.out_normalized_poses.write_text(
        json.dumps(normalized_document, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
