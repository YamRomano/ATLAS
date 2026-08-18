#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


def normalize_xz(value: list[float]) -> np.ndarray:
    out = np.asarray([float(value[0]), float(value[2])], dtype=float)
    norm = float(np.linalg.norm(out))
    return out / norm if norm > 1e-9 else np.asarray([1.0, 0.0])


def rotate_xz(heading: np.ndarray, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray([c * heading[0] - s * heading[1], s * heading[0] + c * heading[1]])


def median_position(samples: list[dict], index: int, radius: int = 2) -> list[float]:
    window = samples[max(0, index - radius): min(len(samples), index + radius + 1)]
    xyz = np.asarray([sample["rcenter"] for sample in window], dtype=float)
    return np.median(xyz, axis=0).tolist()


def optical_yaw_delta(previous: np.ndarray, current: np.ndarray, focal_px: float) -> tuple[float, int]:
    points = cv2.goodFeaturesToTrack(previous, maxCorners=500, qualityLevel=0.01, minDistance=12)
    if points is None or len(points) < 20:
        return 0.0, 0
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        previous,
        current,
        points,
        None,
        winSize=(31, 31),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 24, 0.01),
    )
    if tracked is None or status is None:
        return 0.0, 0
    good = status.reshape(-1).astype(bool)
    if int(np.sum(good)) < 20:
        return 0.0, int(np.sum(good))
    displacement = tracked.reshape(-1, 2)[good] - points.reshape(-1, 2)[good]
    dx = displacement[:, 0]
    median = float(np.median(dx))
    mad = float(np.median(np.abs(dx - median)))
    stable = np.abs(dx - median) <= max(2.5, 3.0 * mad)
    if int(np.sum(stable)) < 16:
        return 0.0, int(np.sum(stable))
    median = float(np.median(dx[stable]))
    return -math.atan2(median, focal_px), int(np.sum(stable))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording-dir", required=True, type=Path)
    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--focal-px",
        type=float,
        default=882.4866783165957,
        help="Horizontal focal length in pixels (default: live DJI/COLMAP 1200x675 calibration).",
    )
    parser.add_argument("--frame-step", type=int, default=3)
    args = parser.parse_args()

    recording = json.loads((args.recording_dir / "recording.json").read_text(encoding="utf-8"))
    trusted = [
        pose for pose in recording.get("poses", [])
        if pose.get("success") is not False
        and not pose.get("held_pose")
        and not pose.get("output_rejected")
        and isinstance(pose.get("rcenter"), list)
        and isinstance(pose.get("rheading"), list)
    ]
    if not trusted:
        raise RuntimeError("Recording has no trusted pose prefix.")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    patrol = next(
        patrol
        for map_entry in manifest.get("maps", [])
        if map_entry.get("id") == recording.get("map_id")
        for patrol in map_entry.get("patrols", [])
        if patrol.get("id") == recording.get("patrol_id")
    )
    patrol_points = [point["rxyz"] for point in patrol["points"]]

    # Smooth and decimate the trusted prefix to about 5 Hz.
    cleaned: list[dict] = []
    for index in range(0, len(trusted), 2):
        pose = trusted[index]
        cleaned.append(
            {
                "time_sec": pose.get("time_sec"),
                "received_unix": pose.get("received_unix"),
                "image_name": pose.get("image_name"),
                "rcenter": median_position(trusted, index),
                "rheading": pose.get("rheading"),
                "source": "trusted_tsolve",
                "translation_allowed": True,
            }
        )

    with (args.recording_dir / "frames.csv").open("r", encoding="utf-8", newline="") as handle:
        frames = list(csv.DictReader(handle))
    last_trusted_unix = float(trusted[-1]["received_unix"])
    tail = [row for row in frames if float(row["received_unix"]) > last_trusted_unix]
    tail = tail[::max(1, int(args.frame_step))]

    anchor = list(trusted[-1]["rcenter"])
    heading = normalize_xz(trusted[-1]["rheading"])
    previous_gray = None
    recovered_rotation: list[dict] = []
    accumulated_yaw = 0.0
    for row in tail:
        image_path = args.frames_dir / row["image_name"]
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        if previous_gray is not None:
            yaw_delta, tracks = optical_yaw_delta(previous_gray, gray, args.focal_px)
            # Reject single-frame optical-flow spikes while retaining slow yaw.
            if abs(yaw_delta) <= math.radians(6.0) and tracks >= 16:
                accumulated_yaw += yaw_delta
                heading = rotate_xz(heading, yaw_delta)
            else:
                tracks = 0
            recovered_rotation.append(
                {
                    "time_sec": float(row["time_sec"]),
                    "received_unix": float(row["received_unix"]),
                    "image_name": row["image_name"],
                    "rcenter": anchor,
                    "rheading": [float(heading[0]), 0.0, float(heading[1])],
                    "source": "rotation_only_optical_flow",
                    "translation_allowed": False,
                    "tracked_features": tracks,
                    "accumulated_yaw_deg": math.degrees(accumulated_yaw),
                }
            )
        previous_gray = gray

    all_samples = cleaned + recovered_rotation
    coverage = []
    for point_index, point in enumerate(patrol_points, 1):
        nearest = min(
            (
            (
                math.hypot(
                    float(sample["rcenter"][0]) - float(point[0]),
                    float(sample["rcenter"][2]) - float(point[2]),
                ),
                sample,
            )
            for sample in cleaned
            ),
            key=lambda item: item[0],
        )
        coverage.append(
            {
                "point_index": point_index,
                "nearest_distance_map_units": nearest[0],
                "nearest_time_sec": nearest[1]["time_sec"],
                "observed": nearest[0] <= 0.55,
            }
        )
    complete_loop = all(item["observed"] for item in coverage)
    output = {
        "version": 1,
        "recording_id": recording["recording_id"],
        "map_id": recording["map_id"],
        "patrol_id": recording["patrol_id"],
        "frame": "atlas_room",
        "safe_for_online_replay": complete_loop,
        "complete_loop": complete_loop,
        "trusted_sample_count": len(cleaned),
        "rotation_only_sample_count": len(recovered_rotation),
        "rotation_only_yaw_deg": math.degrees(accumulated_yaw),
        "coverage": coverage,
        "warning": (
            ""
            if complete_loop
            else "Incomplete teach lap. Rotation-only samples lock position and must never command translation."
        ),
        "samples": all_samples,
    }
    (args.recording_dir / "processed_trajectory.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in output.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
