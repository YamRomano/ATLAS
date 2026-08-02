#!/usr/bin/env python3
"""Select only temporally connected enhancement cameras from a COLMAP model.

Trusted cameras are always retained.  Cameras inside a gap are retained only
when they form a physically plausible chain to the trusted camera on the left,
the trusted camera on the right, or both.  Disconnected middle components and
frames outside the trusted temporal span are written to an image-deletion list
for COLMAP's `image_deleter`; this script does not modify a model itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from colmap_io import camera_center, qvec_to_rotmat, read_images_model


def rotation_distance_degrees(left: Any, right: Any) -> float:
    relative = qvec_to_rotmat(right.qvec) @ qvec_to_rotmat(left.qvec).T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def trusted_center(pose: dict[str, Any]) -> np.ndarray:
    rotation = np.asarray(pose["R"], dtype=float).reshape(3, 3)
    translation = np.asarray(pose["t"], dtype=float).reshape(3)
    return -rotation.T @ translation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--selected-frames", type=Path, required=True)
    parser.add_argument("--trusted-poses", type=Path, required=True)
    parser.add_argument("--new-prefix", default="enhancement/")
    parser.add_argument("--max-center-step", type=float, default=0.55)
    parser.add_argument("--max-rotation-step", type=float, default=60.0)
    parser.add_argument("--max-trusted-center-delta", type=float, default=1e-7)
    parser.add_argument("--delete-names-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()

    candidate = read_images_model(args.candidate_model.resolve())
    by_name = {image.name: image for image in candidate.values()}
    selected_sources = sorted(args.selected_frames.glob("*.jpg"))
    selected_names = [f"{args.new_prefix}{path.name}" for path in selected_sources]
    if not selected_names:
        raise RuntimeError(f"No selected frames found in {args.selected_frames}")

    payload = json.loads(args.trusted_poses.read_text(encoding="utf-8"))
    poses = payload.get("poses", payload) if isinstance(payload, dict) else payload
    if not isinstance(poses, list):
        raise RuntimeError(f"Invalid trusted pose stream: {args.trusted_poses}")
    trusted_by_name = {
        f"{args.new_prefix}{Path(str(pose.get('image_name', ''))).name}": pose
        for pose in poses
        if isinstance(pose, dict)
        and pose.get("success") is True
        and not pose.get("held_pose")
        and not pose.get("output_rejected")
        and pose.get("R") is not None
        and pose.get("t") is not None
    }
    selected_index = {name: index for index, name in enumerate(selected_names)}
    trusted_indices = sorted(
        selected_index[name]
        for name in trusted_by_name
        if name in selected_index and name in by_name
    )
    if len(trusted_indices) < 2:
        raise RuntimeError("Need at least two trusted cameras in the candidate timeline.")

    trusted_deltas: list[float] = []
    for name, pose in trusted_by_name.items():
        image = by_name.get(name)
        if image is None or name not in selected_index:
            continue
        trusted_deltas.append(
            float(np.linalg.norm(camera_center(image) - trusted_center(pose)))
        )
    max_trusted_delta = max(trusted_deltas, default=float("inf"))
    if (
        len(trusted_deltas) != len(trusted_indices)
        or max_trusted_delta > args.max_trusted_center_delta
    ):
        raise RuntimeError(
            "Candidate moved or dropped trusted cameras: "
            f"{len(trusted_deltas)}/{len(trusted_indices)} verified, "
            f"max center delta {max_trusted_delta:.6g}"
        )

    def edge(index: int) -> dict[str, Any]:
        left_name = selected_names[index]
        right_name = selected_names[index + 1]
        left = by_name.get(left_name)
        right = by_name.get(right_name)
        if left is None or right is None:
            return {
                "left": left_name,
                "right": right_name,
                "registered": False,
                "center_step": float("inf"),
                "rotation_deg": float("inf"),
                "valid": False,
            }
        center_step = float(np.linalg.norm(camera_center(right) - camera_center(left)))
        rotation_deg = rotation_distance_degrees(left, right)
        return {
            "left": left_name,
            "right": right_name,
            "registered": True,
            "center_step": center_step,
            "rotation_deg": rotation_deg,
            "valid": (
                center_step <= args.max_center_step
                and rotation_deg <= args.max_rotation_step
            ),
        }

    edge_cache = {
        index: edge(index)
        for index in range(trusted_indices[0], trusted_indices[-1])
    }
    keep_indices = set(trusted_indices)
    gap_audits: list[dict[str, Any]] = []
    for left, right in zip(trusted_indices, trusted_indices[1:]):
        if right == left + 1:
            continue
        prefix_last = left
        while prefix_last < right and edge_cache[prefix_last]["valid"]:
            prefix_last += 1
            keep_indices.add(prefix_last)
        suffix_first = right
        while suffix_first > left and edge_cache[suffix_first - 1]["valid"]:
            suffix_first -= 1
            keep_indices.add(suffix_first)
        # If the two connected components touch across one invalid edge, leave
        # a one-frame hole so downstream temporal validation cannot mistake
        # the disconnected poses for a valid consecutive pair.
        if (
            prefix_last + 1 == suffix_first
            and not edge_cache[prefix_last]["valid"]
        ):
            if suffix_first != right:
                keep_indices.discard(suffix_first)
                suffix_first += 1
            elif prefix_last != left:
                keep_indices.discard(prefix_last)
                prefix_last -= 1
            else:
                raise RuntimeError(
                    "Adjacent trusted cameras violate the temporal motion gate: "
                    f"{selected_names[left]} -> {selected_names[right]}"
                )
        gap_audits.append(
            {
                "left_trusted_index": left,
                "right_trusted_index": right,
                "gap_frames": right - left - 1,
                "left_connected_through": prefix_last,
                "right_connected_from": suffix_first,
                "kept_gap_frames": sum(
                    1 for index in range(left + 1, right) if index in keep_indices
                ),
                "removed_gap_frames": sum(
                    1 for index in range(left + 1, right) if index not in keep_indices
                ),
            }
        )

    kept_names = {
        selected_names[index]
        for index in keep_indices
        if selected_names[index] in by_name
    }
    registered_selected_names = {name for name in selected_names if name in by_name}
    delete_names = sorted(registered_selected_names - kept_names)

    kept_edges = [
        edge_cache[index]
        for index in range(trusted_indices[0], trusted_indices[-1])
        if index in keep_indices and index + 1 in keep_indices
    ]
    summary = {
        "mode": "trusted_temporal_chain_filter",
        "candidate_model": str(args.candidate_model.resolve()),
        "selected_frame_count": len(selected_names),
        "candidate_registered_selected_frames": len(registered_selected_names),
        "trusted_frame_count": len(trusted_indices),
        "first_trusted_index": trusted_indices[0],
        "last_trusted_index": trusted_indices[-1],
        "kept_selected_frames": len(kept_names),
        "deleted_selected_frames": len(delete_names),
        "coverage_within_trusted_span": (
            len(kept_names) / (trusted_indices[-1] - trusted_indices[0] + 1)
        ),
        "max_trusted_center_delta": max_trusted_delta,
        "max_kept_consecutive_center_step": max(
            (float(item["center_step"]) for item in kept_edges), default=None
        ),
        "max_kept_consecutive_rotation_deg": max(
            (float(item["rotation_deg"]) for item in kept_edges), default=None
        ),
        "gap_audits": gap_audits,
        "invalid_edges": [
            item for item in edge_cache.values() if not item["valid"]
        ],
    }
    args.delete_names_out.parent.mkdir(parents=True, exist_ok=True)
    args.delete_names_out.write_text(
        "".join(f"{name}\n" for name in delete_names),
        encoding="utf-8",
    )
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
