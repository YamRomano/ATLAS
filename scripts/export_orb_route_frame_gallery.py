#!/usr/bin/env python3
"""Export the exact ORB patrol-bank anchors as a browser-review manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP_ID = "map_copy_20260730_114851_cfefdc"
DEFAULT_REPLAY_ID = "patrol_baseline_precision_20260813"


def finite_rows(values: np.ndarray) -> list[list[float]]:
    return [[float(component) for component in row] for row in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-id", default=DEFAULT_MAP_ID)
    parser.add_argument("--baseline-replay-id", default=DEFAULT_REPLAY_ID)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "viewer" / "public" / "orb-route-frame-index.json",
    )
    args = parser.parse_args()

    replay_root = (
        ROOT
        / "viewer"
        / "public"
        / "maps"
        / args.map_id
        / "replays"
        / args.baseline_replay_id
    )
    reference_path = replay_root / "reference_candidate.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    bank_name = str(reference.get("visual_route_recovery_bank") or "").strip()
    if not bank_name:
        raise RuntimeError("The selected baseline does not declare an ORB route bank")
    bank_path = replay_root / bank_name

    manifest_path = ROOT / "viewer" / "public" / "maps" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_map = next(
        item for item in manifest.get("maps", []) if item.get("id") == args.map_id
    )
    replay_by_id = {
        str(item.get("id")): item for item in selected_map.get("replays", [])
    }

    with np.load(bank_path, allow_pickle=False) as bank:
        names = [str(value) for value in bank["anchor_names"].tolist()]
        source_frames = np.asarray(bank["source_frames"], dtype=np.int32)
        progress = np.asarray(bank["anchor_progress"], dtype=np.float64)
        centers = np.asarray(bank["anchor_centers"], dtype=np.float64)
        headings = np.asarray(bank["anchor_headings"], dtype=np.float64)
        starts = np.asarray(bank["anchor_from"], dtype=np.float64)
        ends = np.asarray(bank["anchor_to"], dtype=np.float64)
        anchor_ids = np.asarray(bank["anchor_ids"], dtype=np.int32)
        descriptor_counts = np.bincount(anchor_ids, minlength=len(names))
        if "anchor_source_replay_ids" in bank.files:
            source_ids = [
                str(value) for value in bank["anchor_source_replay_ids"].tolist()
            ]
        else:
            source_ids = [args.baseline_replay_id] * len(names)
        heading_priorities = (
            np.asarray(bank["anchor_heading_priority"], dtype=np.int16)
            if "anchor_heading_priority" in bank.files
            else np.zeros(len(names), dtype=np.int16)
        )
        descriptor_total = int(len(bank["descriptors"]))

    reference_legs = list(reference.get("legs") or [])

    def leg_number(start: np.ndarray, end: np.ndarray) -> int:
        for index, leg in enumerate(reference_legs, start=1):
            leg_start = np.asarray(leg.get("from"), dtype=np.float64)
            leg_end = np.asarray(leg.get("to"), dtype=np.float64)
            if np.allclose(start, leg_start) and np.allclose(end, leg_end):
                return index
        return 0

    source_metadata: dict[str, dict[str, object]] = {}
    anchors: list[dict[str, object]] = []
    for index, name in enumerate(names):
        source_id = source_ids[index]
        replay = replay_by_id.get(source_id)
        if replay is None:
            raise RuntimeError(f"ORB source replay is absent from manifest: {source_id}")
        frame_base = str(replay.get("query_frame_base_url") or "").strip("/")
        if not frame_base:
            raise RuntimeError(f"ORB source replay has no frame URL: {source_id}")
        image_name = Path(name).name
        source_metadata.setdefault(
            source_id,
            {
                "id": source_id,
                "title": replay.get("title") or source_id,
                "query_frame_base_url": frame_base,
            },
        )
        leg = leg_number(starts[index], ends[index])
        anchors.append(
            {
                "anchor_index": index,
                "source_replay_id": source_id,
                "source_title": replay.get("title") or source_id,
                "source_frame": int(source_frames[index]),
                "image_name": image_name,
                "image_url": f"/{frame_base}/{image_name}",
                "leg": leg,
                "from_point": leg if leg else None,
                "to_point": (leg % 4) + 1 if leg else None,
                "live_reference_enabled": leg in {1, 2, 3, 4},
                "heading_reference": bool(heading_priorities[index] > 0),
                "heading_priority": int(heading_priorities[index]),
                "progress": float(progress[index]),
                "descriptor_count": int(descriptor_counts[index]),
                "center": finite_rows(centers[index : index + 1])[0],
                "heading": finite_rows(headings[index : index + 1])[0],
            }
        )

    per_leg = {
        str(leg): sum(1 for anchor in anchors if anchor["leg"] == leg)
        for leg in range(1, 5)
    }
    per_source = {
        source_id: sum(
            1 for anchor in anchors if anchor["source_replay_id"] == source_id
        )
        for source_id in source_metadata
    }
    payload = {
        "kind": "atlas_orb_route_frame_gallery",
        "map_id": args.map_id,
        "patrol_id": reference.get("patrol_id"),
        "baseline_replay_id": args.baseline_replay_id,
        "bank": str(bank_path.relative_to(ROOT)),
        "matching": {
            "detector": "ORB",
            "features_per_frame": 1200,
            "distance": "Hamming",
            "geometry": "homography + patrol-leg constraints",
        },
        "anchor_count": len(anchors),
        "live_reference_anchor_count": sum(
            1 for anchor in anchors if anchor["live_reference_enabled"]
        ),
        "stored_audit_only_anchor_count": sum(
            1 for anchor in anchors if not anchor["live_reference_enabled"]
        ),
        "descriptor_count": descriptor_total,
        "per_leg": per_leg,
        "per_source": per_source,
        "sources": list(source_metadata.values()),
        "anchors": anchors,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("bank", "anchor_count", "descriptor_count", "per_leg", "per_source")}, indent=2))


if __name__ == "__main__":
    main()
