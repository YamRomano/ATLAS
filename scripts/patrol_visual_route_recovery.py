#!/usr/bin/env python3
"""Build and query a leg-constrained visual patrol recovery bank.

This is deliberately not a free global localizer.  It can only recover progress
on the exact patrol leg currently published by the flight controller, advances
monotonically through nearby recorded views, and requires two consistent
observations before it can authorize translation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def finite_vector(value: Any) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vector.size < 3 or not np.all(np.isfinite(vector[:3])):
        return None
    return vector[:3].copy()


def horizontal_distance(left: Any, right: Any) -> float:
    a = finite_vector(left)
    b = finite_vector(right)
    if a is None or b is None:
        return float("inf")
    return float(math.hypot(a[0] - b[0], a[2] - b[2]))


def segment_progress(point: Any, start: Any, end: Any) -> float | None:
    p = finite_vector(point)
    a = finite_vector(start)
    b = finite_vector(end)
    if p is None or a is None or b is None:
        return None
    direction = b[[0, 2]] - a[[0, 2]]
    length_sq = float(direction @ direction)
    if length_sq <= 1e-12:
        return None
    return float(((p[[0, 2]] - a[[0, 2]]) @ direction) / length_sq)


def geometric_candidate_rank(item: dict[str, Any]) -> tuple[float, ...]:
    """Rank equal-inlier ORB candidates by independent geometric quality.

    Inlier count deliberately remains the primary key so the established live
    thresholds and route behavior do not change.  Indoor repeated patterns
    frequently tie on count, however; for those ties prefer a larger inlier
    fraction and broader image support, then the lower reprojection residual.
    """
    error = item.get("median_reprojection_error_px")
    try:
        error_value = float(error)
    except (TypeError, ValueError):
        error_value = float("inf")
    if not math.isfinite(error_value):
        error_value = float("inf")
    return (
        float(int(item.get("inliers") or 0)),
        float(item.get("inlier_ratio") or 0.0),
        min(
            float(item.get("source_coverage") or 0.0),
            float(item.get("query_coverage") or 0.0),
        ),
        -error_value,
    )


def conservative_candidate_progress(
    candidates: list[dict[str, Any]],
    *,
    previous: float,
    score_fraction: float = 0.80,
    radius: float = 0.18,
) -> float | None:
    """Return the earliest strongly supported nearby progress observation."""
    if not candidates:
        return None
    winner = max(candidates, key=lambda item: int(item["inliers"]))
    threshold = max(1, int(math.ceil(int(winner["inliers"]) * score_fraction)))
    support = [
        item
        for item in candidates
        if int(item["inliers"]) >= threshold
        and abs(float(item["progress"]) - float(winner["progress"])) <= radius
    ]
    if not support:
        return None
    return max(float(previous), min(float(item["progress"]) for item in support))


def sequence_candidate_progress(
    candidates: list[dict[str, Any]],
    *,
    previous: float,
    minimum_inliers: int = 120,
    score_fraction: float = 0.95,
    backward_window: float = 0.045,
    forward_window: float = 0.12,
    progress_quantile: float = 0.70,
) -> float | None:
    """Follow nearby ordered baseline views without sticking to an old view.

    Indoor patrol images keep matching several nearby recorded views because
    the same wall remains visible while the camera approaches it.  A weighted
    upper quantile of all those matches is unsafe: on the Point 4 -> Point 1
    departure it repeatedly selected future anchors even though the exact
    current anchor had almost the same score.  Restrict candidates to the
    local monotonic window and choose its strongest individual anchor.  The
    exact current saved frame becomes the clear winner as the camera reaches
    it; the old upper-quantile rule instead accumulated many merely similar
    future views and overruled that exact match.  The caller's monotonic room-
    distance publication cap remains the final movement bound.
    ``score_fraction`` and ``progress_quantile`` remain accepted for API
    compatibility.
    """
    try:
        floor = float(previous) - max(0.0, float(backward_window))
        ceiling = float(previous) + max(0.01, float(forward_window))
    except (TypeError, ValueError):
        return None
    local = [
        item
        for item in candidates
        if floor <= float(item.get("progress", float("inf"))) <= ceiling
    ]
    if not local:
        return None
    winner = max(
        local,
        key=lambda item: (
            int(item.get("inliers") or 0),
            -abs(float(item["progress"]) - float(previous)),
            -float(item["progress"]),
        ),
    )
    if int(winner.get("inliers") or 0) < max(1, int(minimum_inliers)):
        return None
    selected = winner["progress"]
    return max(float(previous), float(selected))


def weak_endpoint_candidate_progress(
    candidates: list[dict[str, Any]],
    *,
    minimum_inliers: int = 60,
    endpoint_floor: float = 0.90,
    score_fraction: float = 0.72,
) -> float | None:
    """Return an endpoint only when several endpoint views support the winner.

    This is used exclusively during a controller-locked recovery hover. It can
    update the model/declare arrival, but is explicitly forbidden from
    authorizing more translation. Requiring the best match and multiple
    neighboring anchors to be at the end of the active leg avoids treating a
    repeated mid-room view as arrival.
    """
    if not candidates:
        return None
    winner = max(candidates, key=lambda item: int(item.get("inliers") or 0))
    winner_inliers = int(winner.get("inliers") or 0)
    winner_progress = float(winner.get("progress") or 0.0)
    if winner_inliers < max(1, int(minimum_inliers)) or winner_progress < endpoint_floor:
        return None
    support_threshold = max(
        max(1, int(minimum_inliers)),
        int(math.ceil(winner_inliers * max(0.50, min(1.0, float(score_fraction))))),
    )
    support = [
        float(item["progress"])
        for item in candidates
        if int(item.get("inliers") or 0) >= support_threshold
        and float(item.get("progress") or 0.0) >= endpoint_floor
    ]
    if len(support) < 2:
        return None
    return min(support)


def independent_endpoint_candidate_progress(
    candidates: list[dict[str, Any]],
    *,
    minimum_inliers: int = 120,
    endpoint_floor: float = 0.90,
    score_fraction: float = 0.78,
    minimum_supporters: int = 3,
    nonendpoint_margin_fraction: float = 0.03,
) -> float | None:
    """Verify an endpoint without using the previously published progress.

    ``recover`` normally searches only a bounded window around committed route
    progress.  That prior is useful for smooth tracking but cannot also prove
    arrival: repeated indoor views can make the window walk itself to 100%.
    This helper receives matches from the *entire active leg*.  Its global
    winner must itself be an endpoint view and several neighboring endpoint
    anchors must independently support it.
    """
    if not candidates:
        return None
    winner = max(candidates, key=lambda item: int(item.get("inliers") or 0))
    winner_inliers = int(winner.get("inliers") or 0)
    winner_progress = float(winner.get("progress") or 0.0)
    threshold = max(1, int(minimum_inliers))
    if winner_inliers < threshold or winner_progress < float(endpoint_floor):
        return None
    nonendpoint_best = max(
        (
            int(item.get("inliers") or 0)
            for item in candidates
            if float(item.get("progress") or 0.0) < float(endpoint_floor)
        ),
        default=0,
    )
    required_margin = max(
        3,
        int(
            math.ceil(
                nonendpoint_best
                * max(0.0, min(0.25, float(nonendpoint_margin_fraction)))
            )
        ),
    )
    if nonendpoint_best > 0 and winner_inliers < nonendpoint_best + required_margin:
        return None
    support_threshold = max(
        threshold,
        int(
            math.ceil(
                winner_inliers
                * max(0.50, min(1.0, float(score_fraction)))
            )
        ),
    )
    support = sorted(
        float(item["progress"])
        for item in candidates
        if int(item.get("inliers") or 0) >= support_threshold
        and float(item.get("progress") or 0.0) >= float(endpoint_floor)
    )
    if len(support) < max(2, int(minimum_supporters)):
        return None
    return support[0]


def verified_recovery_rewind_candidate(
    candidates: list[dict[str, Any]],
    *,
    previous: float,
    minimum_inliers: int = 120,
    minimum_rollback: float = 0.12,
    maximum_rollback: float = 0.45,
    score_fraction: float = 0.90,
) -> dict[str, Any] | None:
    """Find a same-depth earlier view after a false endpoint walk-ahead.

    This is used only while the controller is physically hovering in recovery.
    Repeated shelves can make a far-away endpoint view win by descriptor count
    even though its homography scale proves the camera is not at that depth.
    A rewind candidate must independently have endpoint-grade inliers,
    same-depth homography geometry, and nearly the score of the global winner.
    The caller still requires five consecutive frames before publishing any
    bounded backwards model correction.
    """
    if not candidates:
        return None
    try:
        prior = float(previous)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(prior):
        return None
    winner = max(candidates, key=lambda item: int(item.get("inliers") or 0))
    winner_inliers = int(winner.get("inliers") or 0)
    threshold = max(
        max(1, int(minimum_inliers)),
        int(math.ceil(winner_inliers * max(0.75, min(1.0, score_fraction)))),
    )
    floor = max(0.0, prior - max(0.12, float(maximum_rollback)))
    ceiling = min(0.89, prior - max(0.08, float(minimum_rollback)))
    if ceiling < floor:
        return None
    eligible = [
        item
        for item in candidates
        if floor <= float(item.get("progress") or 0.0) <= ceiling
        and int(item.get("inliers") or 0) >= threshold
        and item.get("endpoint_view_geometry_verified") is True
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda item: (
            int(item.get("inliers") or 0),
            -abs(float(item.get("progress") or 0.0) - prior),
        ),
    )


def _atomic_save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build_bank(
    baseline_path: Path,
    out_path: Path,
    *,
    frame_dir: Path | None = None,
) -> dict[str, Any]:
    baseline_path = Path(baseline_path).resolve()
    reference = json.loads(baseline_path.read_text(encoding="utf-8"))
    if (
        reference.get("complete_loop") is not True
        or reference.get("enabled_for_live_route_gate") is not True
        or not isinstance(reference.get("legs"), list)
        or len(reference["legs"]) < 4
    ):
        raise RuntimeError("Visual recovery requires an audited complete-loop route baseline")

    pose_document_path = baseline_path.with_name("poses.json")
    pose_document = json.loads(pose_document_path.read_text(encoding="utf-8"))
    if frame_dir is None:
        frame_source = str(pose_document.get("frame_source") or "").strip("/")
        if not frame_source or ".." in Path(frame_source).parts:
            raise RuntimeError("Baseline poses.json has no safe frame_source")
        frame_dir = ROOT / "viewer" / frame_source
    frame_dir = Path(frame_dir).resolve()
    if not frame_dir.is_dir():
        raise FileNotFoundError(frame_dir)

    detector = cv2.ORB_create(nfeatures=1200, fastThreshold=10)
    descriptors: list[np.ndarray] = []
    xy_rows: list[np.ndarray] = []
    anchor_ids: list[int] = []
    names: list[str] = []
    progress_rows: list[float] = []
    center_rows: list[np.ndarray] = []
    heading_rows: list[np.ndarray] = []
    from_rows: list[np.ndarray] = []
    to_rows: list[np.ndarray] = []
    source_frames: list[int] = []
    seen_leg_frames: set[tuple[int, int]] = set()

    def add_anchor(
        *,
        leg_index: int,
        start: np.ndarray,
        end: np.ndarray,
        image_name: str,
        center: np.ndarray,
        heading: np.ndarray,
        source_frame: int,
    ) -> None:
        key = (int(leg_index), int(source_frame))
        if key in seen_leg_frames:
            return
        progress = segment_progress(center, start, end)
        image_path = frame_dir / image_name
        if not image_name or progress is None:
            return
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(image_path)
        keypoints, image_descriptors = detector.detectAndCompute(gray, None)
        if image_descriptors is None or len(keypoints) < 120:
            raise RuntimeError(f"Visual baseline frame is too weak: {image_path}")
        anchor_id = len(names)
        descriptors.append(np.asarray(image_descriptors, dtype=np.uint8))
        xy_rows.append(np.float32([point.pt for point in keypoints]))
        anchor_ids.extend([anchor_id] * len(image_descriptors))
        names.append(image_name)
        progress_rows.append(max(0.0, min(1.0, float(progress))))
        center_rows.append(center)
        heading_rows.append(heading)
        from_rows.append(start)
        to_rows.append(end)
        source_frames.append(int(source_frame))
        seen_leg_frames.add(key)

    for leg_index, leg in enumerate(reference["legs"]):
        start = finite_vector(leg.get("from"))
        end = finite_vector(leg.get("to"))
        if start is None or end is None:
            continue
        for sample in leg.get("samples") or []:
            image_name = Path(str(sample.get("image_name") or "")).name
            center = finite_vector(sample.get("rcenter"))
            heading = finite_vector(sample.get("rheading"))
            if not image_name or center is None or heading is None:
                continue
            add_anchor(
                leg_index=leg_index,
                start=start,
                end=end,
                image_name=image_name,
                center=center,
                heading=heading,
                source_frame=int(sample.get("source_frame") or 0),
            )

    # The weak repeated-room legs need a current-image position for every
    # incoming frame, not a ten-frame staircase.  Retain sparse anchors on the
    # already-stable first two legs, and add every recorded frame only for
    # Point 3->4 and Point 4->1. Runtime matching remains bounded below to a
    # small temporal neighborhood, so density improves smoothness without a
    # whole-leg descriptor scan on every live frame.
    boundaries = dict(pose_document.get("phase_boundaries") or {})
    dense_ranges = {
        2: (
            int(boundaries.get("point3_departure") or 0),
            int(boundaries.get("point4_arrival") or -1),
        ),
        3: (
            int(boundaries.get("point4_departure") or 0),
            int(boundaries.get("point1_return") or -1),
        ),
    }
    poses_by_frame = {
        int(pose.get("source_frame")): pose
        for pose in pose_document.get("poses") or []
        if isinstance(pose, dict) and pose.get("source_frame") is not None
    }
    for leg_index, (first_frame, last_frame) in dense_ranges.items():
        if first_frame <= 0 or last_frame < first_frame:
            continue
        leg = reference["legs"][leg_index]
        start = finite_vector(leg.get("from"))
        end = finite_vector(leg.get("to"))
        if start is None or end is None:
            continue
        for source_frame in range(first_frame, last_frame + 1):
            pose = poses_by_frame.get(source_frame)
            if not isinstance(pose, dict):
                continue
            center = finite_vector(pose.get("rcenter"))
            heading = finite_vector(pose.get("rheading"))
            image_name = Path(str(pose.get("image_name") or "")).name
            if center is None or heading is None or not image_name:
                continue
            add_anchor(
                leg_index=leg_index,
                start=start,
                end=end,
                image_name=image_name,
                center=center,
                heading=heading,
                source_frame=source_frame,
            )

    if len(names) < 16:
        raise RuntimeError("Visual route bank did not contain enough patrol anchors")
    _atomic_save_npz(
        Path(out_path),
        descriptors=np.concatenate(descriptors, axis=0),
        xy=np.concatenate(xy_rows, axis=0),
        anchor_ids=np.asarray(anchor_ids, dtype=np.int32),
        anchor_names=np.asarray(names),
        anchor_progress=np.asarray(progress_rows, dtype=np.float64),
        anchor_centers=np.asarray(center_rows, dtype=np.float64),
        anchor_headings=np.asarray(heading_rows, dtype=np.float64),
        anchor_from=np.asarray(from_rows, dtype=np.float64),
        anchor_to=np.asarray(to_rows, dtype=np.float64),
        source_frames=np.asarray(source_frames, dtype=np.int32),
        map_id=np.asarray([str(reference.get("map_id") or "")]),
        patrol_id=np.asarray([str(reference.get("patrol_id") or "")]),
        baseline_replay_id=np.asarray([baseline_path.parent.name]),
        baseline_sha256=np.asarray([hashlib.sha256(baseline_path.read_bytes()).hexdigest()]),
    )
    return {
        "out": str(Path(out_path).resolve()),
        "baseline": str(baseline_path),
        "frame_dir": str(frame_dir),
        "anchors": len(names),
        "descriptors": int(sum(len(item) for item in descriptors)),
        "map_id": str(reference.get("map_id") or ""),
        "patrol_id": str(reference.get("patrol_id") or ""),
    }


def _forward_motion_profile(
    frame_dir: Path,
    start_frame: int,
    end_frame: int,
) -> dict[int, float]:
    """Estimate monotonic route progress without borrowing a saved pose.

    The DJI controller moves in short forward pulses.  Mapping progress only
    from elapsed frame number makes the model drift while the physical video
    is stationary.  Positive radial image expansion is a useful, scale-free
    indicator of those pulses.  Its cumulative sum preserves the observed
    stop/move timing; the patrol endpoints provide only the final metric scale.
    """
    first = int(start_frame)
    last = int(end_frame)
    if last <= first:
        return {first: 0.0}
    previous = cv2.imread(
        str(Path(frame_dir) / f"query_{first:06d}.jpg"),
        cv2.IMREAD_GRAYSCALE,
    )
    if previous is None:
        raise FileNotFoundError(Path(frame_dir) / f"query_{first:06d}.jpg")
    weights: list[float] = []
    for frame_index in range(first + 1, last + 1):
        current = cv2.imread(
            str(Path(frame_dir) / f"query_{frame_index:06d}.jpg"),
            cv2.IMREAD_GRAYSCALE,
        )
        if current is None:
            raise FileNotFoundError(
                Path(frame_dir) / f"query_{frame_index:06d}.jpg"
            )
        points = cv2.goodFeaturesToTrack(
            previous,
            maxCorners=700,
            qualityLevel=0.01,
            minDistance=8,
        )
        expansion = 0.0
        if points is not None and len(points) >= 20:
            moved, status, _ = cv2.calcOpticalFlowPyrLK(
                previous,
                current,
                points,
                None,
                winSize=(31, 31),
                maxLevel=4,
                criteria=(
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    30,
                    0.01,
                ),
            )
            if moved is not None and status is not None:
                valid = status.reshape(-1).astype(bool)
                source_xy = points.reshape(-1, 2)[valid]
                target_xy = moved.reshape(-1, 2)[valid]
                if len(source_xy) >= 20:
                    height, width = previous.shape
                    design = np.column_stack(
                        [
                            np.ones(len(source_xy)),
                            (source_xy[:, 0] - 0.5 * width) / width,
                            (source_xy[:, 1] - 0.5 * height) / height,
                        ]
                    )
                    flow = target_xy - source_xy
                    coefficient_x = np.linalg.lstsq(
                        design, flow[:, 0], rcond=None
                    )[0]
                    coefficient_y = np.linalg.lstsq(
                        design, flow[:, 1], rcond=None
                    )[0]
                    expansion = 0.5 * (
                        coefficient_x[1] / width
                        + coefficient_y[2] / height
                    )
        # Ignore sub-pixel breathing and negative/backward estimates.  A tiny
        # floor keeps long, low-texture forward pulses represented without
        # turning stationary frames into motion.
        weights.append(max(0.0, float(expansion) - 1.0e-4))
        previous = current

    cumulative = np.concatenate(
        [np.zeros(1, dtype=np.float64), np.cumsum(weights, dtype=np.float64)]
    )
    total = float(cumulative[-1])
    if not np.isfinite(total) or total <= 1.0e-8:
        cumulative = np.linspace(0.0, 1.0, last - first + 1)
    else:
        cumulative /= total
    return {
        frame_index: float(cumulative[frame_index - first])
        for frame_index in range(first, last + 1)
    }


def extend_bank_with_recorded_segments(
    base_bank_path: Path,
    out_path: Path,
    *,
    reference_path: Path,
    segments: list[dict[str, Any]],
    anchor_stride: int = 2,
) -> dict[str, Any]:
    """Add route views from independent flights to an existing safe bank.

    Every added view remains constrained to one active patrol leg.  The bank
    therefore tolerates lighting, start-position, and pulse-timing variation
    without becoming a room-wide nearest-image controller.
    """
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    legs = list(reference.get("legs") or [])
    if len(legs) < 4:
        raise RuntimeError("Multi-run visual recovery requires four route legs")
    with np.load(Path(base_bank_path), allow_pickle=False) as bank:
        arrays = {name: np.asarray(bank[name]).copy() for name in bank.files}

    required = {
        "descriptors",
        "xy",
        "anchor_ids",
        "anchor_names",
        "anchor_progress",
        "anchor_centers",
        "anchor_headings",
        "anchor_from",
        "anchor_to",
        "source_frames",
    }
    missing = sorted(required - arrays.keys())
    if missing:
        raise RuntimeError(f"Visual recovery bank is missing: {', '.join(missing)}")

    detector = cv2.ORB_create(nfeatures=1200, fastThreshold=10)
    descriptor_rows: list[np.ndarray] = []
    xy_rows: list[np.ndarray] = []
    descriptor_anchor_ids: list[int] = []
    names: list[str] = []
    progress_rows: list[float] = []
    centers: list[np.ndarray] = []
    headings: list[np.ndarray] = []
    from_rows: list[np.ndarray] = []
    to_rows: list[np.ndarray] = []
    source_frames: list[int] = []
    source_ids: list[str] = []
    next_anchor_id = int(len(arrays["anchor_names"]))
    stride = max(1, int(anchor_stride))

    for segment in segments:
        leg_index = int(segment.get("leg_index", -1))
        if not 0 <= leg_index < len(legs):
            raise RuntimeError(f"Invalid multi-run leg index: {leg_index}")
        frame_dir = Path(segment["frame_dir"]).resolve()
        source_replay_id = str(segment.get("source_replay_id") or "").strip()
        first = int(segment["start_frame"])
        last = int(segment["end_frame"])
        if not source_replay_id or not frame_dir.is_dir() or last <= first:
            raise RuntimeError("Invalid multi-run visual segment")
        start = finite_vector(legs[leg_index].get("from"))
        end = finite_vector(legs[leg_index].get("to"))
        if start is None or end is None:
            raise RuntimeError(f"Patrol leg {leg_index + 1} has invalid endpoints")
        direction = end - start
        heading = np.asarray([direction[0], 0.0, direction[2]], dtype=float)
        heading_norm = float(np.linalg.norm(heading))
        if heading_norm <= 1.0e-9:
            raise RuntimeError(f"Patrol leg {leg_index + 1} has zero length")
        heading /= heading_norm
        progress_by_frame = _forward_motion_profile(frame_dir, first, last)
        selected_frames = list(range(first, last + 1, stride))
        if selected_frames[-1] != last:
            selected_frames.append(last)
        for frame_index in selected_frames:
            image_path = frame_dir / f"query_{frame_index:06d}.jpg"
            gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise FileNotFoundError(image_path)
            keypoints, descriptors = detector.detectAndCompute(gray, None)
            if descriptors is None or len(keypoints) < 120:
                continue
            progress = max(0.0, min(1.0, progress_by_frame[frame_index]))
            center = start + direction * progress
            descriptor_rows.append(np.asarray(descriptors, dtype=np.uint8))
            xy_rows.append(np.float32([point.pt for point in keypoints]))
            descriptor_anchor_ids.extend([next_anchor_id] * len(descriptors))
            names.append(f"{source_replay_id}/query_{frame_index:06d}.jpg")
            progress_rows.append(progress)
            centers.append(center)
            headings.append(heading.copy())
            from_rows.append(start.copy())
            to_rows.append(end.copy())
            source_frames.append(frame_index)
            source_ids.append(source_replay_id)
            next_anchor_id += 1

    if not names:
        raise RuntimeError("No usable anchors were added from the independent runs")
    base_anchor_count = int(len(arrays["anchor_names"]))
    base_replay_id = str(np.asarray(arrays["baseline_replay_id"]).tolist()[0])
    base_source_ids = np.asarray(
        [base_replay_id] * base_anchor_count,
        dtype=f"<U{max(1, len(base_replay_id))}",
    )
    arrays.update(
        {
            "descriptors": np.concatenate(
                [arrays["descriptors"], *descriptor_rows], axis=0
            ),
            "xy": np.concatenate([arrays["xy"], *xy_rows], axis=0),
            "anchor_ids": np.concatenate(
                [arrays["anchor_ids"], np.asarray(descriptor_anchor_ids, dtype=np.int32)]
            ),
            "anchor_names": np.concatenate(
                [arrays["anchor_names"], np.asarray(names)]
            ),
            "anchor_progress": np.concatenate(
                [arrays["anchor_progress"], np.asarray(progress_rows, dtype=np.float64)]
            ),
            "anchor_centers": np.concatenate(
                [arrays["anchor_centers"], np.asarray(centers, dtype=np.float64)], axis=0
            ),
            "anchor_headings": np.concatenate(
                [arrays["anchor_headings"], np.asarray(headings, dtype=np.float64)], axis=0
            ),
            "anchor_from": np.concatenate(
                [arrays["anchor_from"], np.asarray(from_rows, dtype=np.float64)], axis=0
            ),
            "anchor_to": np.concatenate(
                [arrays["anchor_to"], np.asarray(to_rows, dtype=np.float64)], axis=0
            ),
            "source_frames": np.concatenate(
                [arrays["source_frames"], np.asarray(source_frames, dtype=np.int32)]
            ),
            "anchor_source_replay_ids": np.concatenate(
                [base_source_ids, np.asarray(source_ids)]
            ),
            "source_replay_ids": np.asarray(
                list(dict.fromkeys([base_replay_id, *source_ids]))
            ),
        }
    )
    _atomic_save_npz(Path(out_path), **arrays)
    return {
        "out": str(Path(out_path).resolve()),
        "base_bank": str(Path(base_bank_path).resolve()),
        "base_anchors": base_anchor_count,
        "added_anchors": len(names),
        "total_anchors": int(len(arrays["anchor_names"])),
        "source_replay_ids": list(dict.fromkeys([base_replay_id, *source_ids])),
    }


class PatrolVisualRouteRecovery:
    """Recover conservative patrol progress from the commanded leg only."""

    def __init__(
        self,
        bank_path: Path,
        *,
        minimum_inliers: int = 120,
        recovery_minimum_inliers: int = 90,
        forward_window: float = 0.24,
        acquisition_hits: int = 2,
        recovery_acquisition_hits: int = 5,
        max_position_step: float = 0.18,
        endpoint_guard_progress: float = 0.84,
        endpoint_required_hits: int = 3,
        matching_profile: str = "hierarchical",
    ) -> None:
        self.bank_path = Path(bank_path)
        with np.load(self.bank_path, allow_pickle=False) as bank:
            self.descriptors = np.asarray(bank["descriptors"], dtype=np.uint8)
            self.xy = np.asarray(bank["xy"], dtype=np.float32)
            self.anchor_ids = np.asarray(bank["anchor_ids"], dtype=np.int32)
            self.anchor_names = [str(value) for value in bank["anchor_names"].tolist()]
            self.anchor_progress = np.asarray(bank["anchor_progress"], dtype=np.float64)
            self.anchor_centers = np.asarray(bank["anchor_centers"], dtype=np.float64)
            self.anchor_headings = np.asarray(bank["anchor_headings"], dtype=np.float64)
            self.anchor_from = np.asarray(bank["anchor_from"], dtype=np.float64)
            self.anchor_to = np.asarray(bank["anchor_to"], dtype=np.float64)
            self.source_frames = np.asarray(bank["source_frames"], dtype=np.int32)
            self.map_id = str(bank["map_id"].tolist()[0])
            self.patrol_id = str(bank["patrol_id"].tolist()[0])
            self.baseline_replay_id = str(bank["baseline_replay_id"].tolist()[0])
            if "anchor_source_replay_ids" in bank.files:
                self.anchor_source_replay_ids = [
                    str(value)
                    for value in bank["anchor_source_replay_ids"].tolist()
                ]
            else:
                self.anchor_source_replay_ids = [
                    self.baseline_replay_id
                ] * len(self.anchor_names)
        self.detector = cv2.ORB_create(nfeatures=1200, fastThreshold=10)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        normalized_matching_profile = str(matching_profile).strip().lower()
        if normalized_matching_profile not in {"exact", "hierarchical"}:
            raise ValueError(
                "matching_profile must be 'exact' or 'hierarchical'"
            )
        self.matching_profile = normalized_matching_profile
        self.last_match_diagnostic: dict[str, Any] = {}
        # The bank is immutable for the lifetime of this matcher.  Building
        # descriptor row lists inside every frame/anchor loop used
        # ``np.flatnonzero`` thousands of times and repeatedly crossed the
        # Python -> OpenCV boundary.  Cache the rows once and batch all
        # selected source descriptors into one exact BF query below.  The
        # distance search and ratio test are unchanged; only their scheduling
        # is different.
        anchor_order = np.argsort(self.anchor_ids, kind="stable")
        anchor_counts = np.bincount(
            self.anchor_ids,
            minlength=len(self.anchor_names),
        )
        anchor_offsets = np.concatenate(
            ([0], np.cumsum(anchor_counts, dtype=np.int64))
        )
        self._anchor_rows = tuple(
            anchor_order[anchor_offsets[anchor_id] : anchor_offsets[anchor_id + 1]]
            for anchor_id in range(len(self.anchor_names))
        )
        self.minimum_inliers = max(16, int(minimum_inliers))
        # Normal flight retains the strict 120-inlier gate.  During a neutral
        # recovery hover, five consecutive command-bounded observations may
        # use this lower floor.  This is deliberately separate from heading
        # and normal-cruise acquisition: the 13:22:38 Point-2 -> Point-3 live
        # frames carried 90-115 correct inliers after real motion, but could
        # never produce two adjacent >=120 frames and therefore froze forever.
        self.recovery_minimum_inliers = max(
            60,
            min(self.minimum_inliers, int(recovery_minimum_inliers)),
        )
        # Normal flight can publish at most one 0.18 m visual catch-up step,
        # and sequence consensus already rejects candidates beyond +0.12
        # progress.  Searching half of a long leg every frame only duplicated
        # work and pushed the 10-FPS localizer over budget.  Recovery hover
        # still searches the entire remaining leg below.
        self.forward_window = max(0.08, min(0.30, float(forward_window)))
        self.acquisition_hits = max(2, int(acquisition_hits))
        self.recovery_acquisition_hits = max(
            self.acquisition_hits,
            int(recovery_acquisition_hits),
        )
        self.max_position_step = max(0.05, min(0.22, float(max_position_step)))
        self.endpoint_guard_progress = max(
            0.75, min(0.90, float(endpoint_guard_progress))
        )
        self.endpoint_required_hits = max(2, int(endpoint_required_hits))
        self.active_key: tuple[Any, ...] | None = None
        # Image-sequence progress is deliberately independent from the pose
        # currently published by TSolve.  Feeding the metric floor back into
        # the match window made a false repeated-room root move the visual
        # search forward with it, so both sources appeared to agree while the
        # physical frames were still behind.
        self.last_matched_progress: float | None = None
        self.last_matched_source_frame: int | None = None
        self.last_sequence_index: int | None = None
        self.last_progress: float | None = None
        self.pending_progress: float | None = None
        self.pending_source_replay_id: str | None = None
        self.pending_hits = 0
        self.active_source_replay_id: str | None = None
        self.needs_acquisition = True
        self.weak_endpoint_progress: float | None = None
        self.weak_endpoint_hits = 0
        self.temporal_recovery_progress: float | None = None
        self.temporal_recovery_source_replay_id: str | None = None
        self.temporal_recovery_hits = 0
        self.endpoint_candidate_progress: float | None = None
        self.endpoint_hits = 0
        self.endpoint_verified = False
        self.endpoint_view_geometry_verified = False
        self.endpoint_view_scale_min: float | None = None
        self.endpoint_view_scale_max: float | None = None
        self.rewind_candidate_progress: float | None = None
        self.rewind_candidate_source_replay_id: str | None = None
        self.rewind_candidate_hits = 0

    def _matching_anchors(self, start: Any, end: Any) -> list[int]:
        return [
            index
            for index in range(len(self.anchor_names))
            if horizontal_distance(start, self.anchor_from[index]) <= 0.08
            and horizontal_distance(end, self.anchor_to[index]) <= 0.08
        ]

    def _endpoint_audit_anchors(self, segment_anchors: list[int]) -> list[int]:
        """Return dense endpoint support plus sparse whole-leg alias sentinels.

        Endpoint verification must compare against earlier repeated views, but
        matching every descriptor from a dense leg again on every endpoint
        frame made the 10-FPS loop miss its deadline.  Evenly cover the
        endpoint with up to 32 anchors and the non-endpoint leg with up to 24
        sentinels.
        Indoor aliases persist across neighboring recorded frames, so this
        retains whole-leg negative evidence without the duplicate 100-anchor
        pass.
        """
        ordered = sorted(
            (int(index) for index in segment_anchors),
            key=lambda index: float(self.anchor_progress[index]),
        )
        endpoint = [
            index for index in ordered if float(self.anchor_progress[index]) >= 0.90
        ]
        nonendpoint = [
            index for index in ordered if float(self.anchor_progress[index]) < 0.90
        ]
        if len(nonendpoint) > 24:
            positions = np.linspace(0, len(nonendpoint) - 1, 24)
            nonendpoint = sorted({nonendpoint[int(round(value))] for value in positions})
        if len(endpoint) > 32:
            positions = np.linspace(0, len(endpoint) - 1, 32)
            endpoint = sorted({endpoint[int(round(value))] for value in positions})
        return nonendpoint + endpoint

    def _reset_for_key(self, key: tuple[Any, ...]) -> None:
        if key == self.active_key:
            return
        self.active_key = key
        self.last_matched_progress = None
        self.last_matched_source_frame = None
        self.last_sequence_index = None
        self.last_progress = None
        self.pending_progress = None
        self.pending_source_replay_id = None
        self.pending_hits = 0
        self.active_source_replay_id = None
        self.needs_acquisition = True
        self.weak_endpoint_progress = None
        self.weak_endpoint_hits = 0
        self.temporal_recovery_progress = None
        self.temporal_recovery_source_replay_id = None
        self.temporal_recovery_hits = 0
        self.endpoint_candidate_progress = None
        self.endpoint_hits = 0
        self.endpoint_verified = False
        self.endpoint_view_geometry_verified = False
        self.endpoint_view_scale_min = None
        self.endpoint_view_scale_max = None
        self.rewind_candidate_progress = None
        self.rewind_candidate_source_replay_id = None
        self.rewind_candidate_hits = 0

    def _mark_unverified(self) -> None:
        """Require a fresh multi-frame lock before translation can resume."""
        self.pending_progress = None
        self.pending_source_replay_id = None
        self.pending_hits = 0
        self.needs_acquisition = True
        self.weak_endpoint_progress = None
        self.weak_endpoint_hits = 0
        self.temporal_recovery_progress = None
        self.temporal_recovery_source_replay_id = None
        self.temporal_recovery_hits = 0
        self.rewind_candidate_progress = None
        self.rewind_candidate_source_replay_id = None
        self.rewind_candidate_hits = 0

    def _match_candidates(
        self,
        *,
        query_keypoints: list[Any],
        query_descriptors: np.ndarray,
        anchors: list[int],
        minimum_geometric_inliers: int = 8,
        force_exact: bool = False,
    ) -> list[dict[str, Any]]:
        if not anchors:
            return []
        valid_anchors = list(
            dict.fromkeys(
                int(anchor_id)
                for anchor_id in anchors
                if 0 <= int(anchor_id) < len(self._anchor_rows)
                and len(self._anchor_rows[int(anchor_id)]) > 0
            )
        )
        if not valid_anchors:
            return []

        # BFMatcher treats every query descriptor independently, so matching
        # one concatenated source matrix against the same current-frame train
        # matrix produces the exact same two nearest neighbors as one call per
        # anchor.  Keep a parallel anchor/local-row lookup to split the result
        # back into the existing per-anchor homography checks.
        row_blocks = [self._anchor_rows[anchor_id] for anchor_id in valid_anchors]
        batched_rows = np.concatenate(row_blocks)
        batched_anchor_ids = np.repeat(
            np.asarray(valid_anchors, dtype=np.int32),
            [len(rows) for rows in row_blocks],
        )
        pairs = self.matcher.knnMatch(
            self.descriptors[batched_rows], query_descriptors, k=2
        )
        accepted_by_anchor: dict[int, list[Any]] = {
            anchor_id: [] for anchor_id in valid_anchors
        }
        for batched_index, pair in enumerate(pairs):
            if (
                len(pair) == 2
                and pair[0].distance < 0.78 * pair[1].distance
            ):
                accepted_by_anchor[int(batched_anchor_ids[batched_index])].append(
                    pair[0]
                )

        # Homography RANSAC, rather than Hamming distance, dominates a wide
        # endpoint audit.  Its inlier count can never exceed the number of
        # ratio-test matches supplied to it, so anchors below the caller's
        # minimum useful inlier count are mathematically unable to affect a
        # successful decision.  Skip only those impossible fits; this is an
        # exact upper-bound prune, not an approximate visual shortlist.
        hierarchy_enabled = bool(
            self.matching_profile == "hierarchical"
            and not force_exact
            and len(valid_anchors) > 16
        )
        homography_support_floor = (
            8
            if self.matching_profile == "exact" or force_exact
            else max(8, int(minimum_geometric_inliers))
        )

        query_points = np.float32([keypoint.pt for keypoint in query_keypoints])
        candidates: list[dict[str, Any]] = []
        original_order = {
            anchor_id: position for position, anchor_id in enumerate(valid_anchors)
        }
        block_offsets: dict[int, int] = {}
        block_offset = 0
        for anchor_id, rows in zip(valid_anchors, row_blocks):
            block_offsets[anchor_id] = block_offset
            block_offset += len(rows)
        anchor_blocks = list(zip(valid_anchors, row_blocks))
        unselected_anchor_ids: set[int] = set()
        if hierarchy_enabled:
            # Build small temporal keyframe blocks independently for each
            # recorded source run. Descriptor votes select two promising
            # blocks, and their immediate neighbors form the exact local
            # homography search. No excluded block is trusted blindly: the
            # upper-bound check below falls back to the full exact search if
            # an excluded anchor could still tie or beat the local winner.
            selected_anchor_ids: set[int] = set()
            source_groups: dict[str, list[int]] = {}
            for anchor_id in valid_anchors:
                source_groups.setdefault(
                    self.anchor_source_replay_ids[anchor_id], []
                ).append(anchor_id)
            for source_anchors in source_groups.values():
                source_anchors.sort(
                    key=lambda anchor_id: (
                        int(self.source_frames[anchor_id]),
                        float(self.anchor_progress[anchor_id]),
                    )
                )
                keyframe_blocks = [
                    source_anchors[offset : offset + 4]
                    for offset in range(0, len(source_anchors), 4)
                ]
                block_scores = [
                    max(
                        (
                            len(accepted_by_anchor[anchor_id])
                            for anchor_id in block
                        ),
                        default=0,
                    )
                    for block in keyframe_blocks
                ]
                selected_blocks = sorted(
                    range(len(keyframe_blocks)),
                    key=lambda block_index: block_scores[block_index],
                    reverse=True,
                )[:2]
                for block_index in selected_blocks:
                    for neighbor_index in range(
                        max(0, block_index - 1),
                        min(len(keyframe_blocks), block_index + 2),
                    ):
                        selected_anchor_ids.update(
                            keyframe_blocks[neighbor_index]
                        )
            unselected_anchor_ids = set(valid_anchors) - selected_anchor_ids
            anchor_blocks = [
                item for item in anchor_blocks if item[0] in selected_anchor_ids
            ]

        if self.matching_profile == "hierarchical" and not force_exact:
            # Descriptor support is an upper bound on homography inliers.
            # Evaluate the strongest upper bounds first so a verified view can
            # tighten the bound for the remaining dense neighboring anchors.
            anchor_blocks.sort(
                key=lambda item: len(accepted_by_anchor[item[0]]),
                reverse=True,
            )
        best_geometric_inliers = 0
        homography_evaluated = 0
        upper_bound_pruned = 0
        maximum_pruned_support = 0
        for anchor_id, rows in anchor_blocks:
            # ``queryIdx`` refers to the concatenated source matrix. Convert
            # it back to this anchor's local descriptor index so all geometric
            # calculations remain byte-for-byte equivalent to the old loop.
            matches = accepted_by_anchor[anchor_id]
            decision_support_floor = homography_support_floor
            if (
                self.matching_profile == "hierarchical"
                and not force_exact
                and best_geometric_inliers >= homography_support_floor
            ):
                # Every current consumer requires either the absolute success
                # floor or at least 50% of its geometric winner.  Existing
                # route rules are stricter (72-95%); 50% is the shared lower
                # bound in their defensive clamps.  If raw matches cannot
                # reach this number, RANSAC inliers cannot reach it either.
                decision_support_floor = max(
                    homography_support_floor,
                    int(math.ceil(best_geometric_inliers * 0.50)),
                )
            if len(matches) < decision_support_floor:
                upper_bound_pruned += 1
                maximum_pruned_support = max(
                    maximum_pruned_support,
                    len(matches),
                )
                continue
            homography_evaluated += 1
            block_offset = block_offsets[anchor_id]
            source_xy = np.float32(
                [self.xy[rows[match.queryIdx - block_offset]] for match in matches]
            )
            query_xy = query_points[[match.trainIdx for match in matches]]
            homography, mask = cv2.findHomography(
                source_xy,
                query_xy,
                cv2.RANSAC,
                4.0,
            )
            inlier_mask = (
                mask.reshape(-1).astype(bool)
                if mask is not None
                else np.zeros(len(matches), dtype=bool)
            )
            best_geometric_inliers = max(
                best_geometric_inliers,
                int(inlier_mask.sum()),
            )
            view_scale_min = None
            view_scale_max = None
            endpoint_view_geometry_verified = False
            inlier_ratio = float(inlier_mask.mean()) if len(inlier_mask) else 0.0
            median_reprojection_error_px = None
            horizontal_shift_px = None
            source_coverage = 0.0
            query_coverage = 0.0
            if homography is not None and int(inlier_mask.sum()) >= 8:
                inlier_source_xy = source_xy[inlier_mask]
                inlier_query_xy = query_xy[inlier_mask]
                horizontal_shift_px = float(
                    np.median(inlier_query_xy[:, 0] - inlier_source_xy[:, 0])
                )
                try:
                    projected_inliers = cv2.perspectiveTransform(
                        inlier_source_xy.reshape(1, -1, 2), homography
                    )[0]
                    residuals = np.linalg.norm(
                        projected_inliers - inlier_query_xy, axis=1
                    )
                    if len(residuals) and np.all(np.isfinite(residuals)):
                        median_reprojection_error_px = float(np.median(residuals))
                except (cv2.error, ValueError):
                    median_reprojection_error_px = None

                # Repeated wall/floor texture can produce many concentrated
                # inliers.  Record how much of each image the verified matches
                # cover so equally strong candidates can prefer a spatially
                # supported view instead of a small repeated patch.
                source_span = np.ptp(inlier_source_xy, axis=0)
                query_span = np.ptp(inlier_query_xy, axis=0)
                source_extent = np.ptp(self.xy[rows], axis=0)
                query_extent = np.ptp(query_points, axis=0)
                source_coverage = float(
                    np.prod(source_span / np.maximum(source_extent, 1.0))
                )
                query_coverage = float(
                    np.prod(query_span / np.maximum(query_extent, 1.0))
                )
                # A repeated indoor wall can retain hundreds of ORB/homography
                # inliers even when the aircraft is still well before the
                # waypoint.  Match count therefore proves scene identity, not
                # arrival depth.  Evaluate the local homography scale at the
                # median inlier location: a true endpoint repeat remains close
                # to unit scale, while the 09:49 Point-4 false arrival was only
                # 0.65-0.70 because the cabinet wall was still much farther
                # away.  This check is endpoint evidence only; ordinary route
                # tracking remains tolerant of perspective change.
                center = np.median(source_xy[inlier_mask], axis=0)
                delta = 24.0
                probes = np.float32(
                    [[
                        center,
                        center + np.asarray([delta, 0.0], dtype=np.float32),
                        center + np.asarray([0.0, delta], dtype=np.float32),
                    ]]
                )
                try:
                    projected = cv2.perspectiveTransform(probes, homography)[0]
                    jacobian = np.column_stack(
                        (
                            (projected[1] - projected[0]) / delta,
                            (projected[2] - projected[0]) / delta,
                        )
                    )
                    singular_values = np.linalg.svd(
                        jacobian,
                        compute_uv=False,
                    )
                    if (
                        singular_values.shape == (2,)
                        and np.all(np.isfinite(singular_values))
                    ):
                        view_scale_min = float(np.min(singular_values))
                        view_scale_max = float(np.max(singular_values))
                        endpoint_view_geometry_verified = bool(
                            view_scale_min >= 0.82
                            and view_scale_max <= 1.30
                            and view_scale_max
                            <= max(1.0e-9, view_scale_min) * 1.30
                        )
                except (cv2.error, np.linalg.LinAlgError, ValueError):
                    endpoint_view_geometry_verified = False
            candidates.append(
                {
                    "anchor_id": anchor_id,
                    "anchor_name": self.anchor_names[anchor_id],
                    "source_frame": int(self.source_frames[anchor_id]),
                    "source_replay_id": self.anchor_source_replay_ids[anchor_id],
                    "progress": float(self.anchor_progress[anchor_id]),
                    "inliers": int(inlier_mask.sum()),
                    "ratio_matches": len(matches),
                    "inlier_ratio": inlier_ratio,
                    "median_reprojection_error_px": median_reprojection_error_px,
                    "horizontal_shift_px": horizontal_shift_px,
                    "source_coverage": source_coverage,
                    "query_coverage": query_coverage,
                    "endpoint_view_scale_min": view_scale_min,
                    "endpoint_view_scale_max": view_scale_max,
                    "endpoint_view_geometry_verified": (
                        endpoint_view_geometry_verified
                    ),
                }
            )
        # Preserve bank/route ordering for helper functions whose conservative
        # tie behavior predates the hierarchy.  Candidate scores themselves
        # are independent of the evaluation order.
        candidates.sort(key=lambda item: original_order[int(item["anchor_id"])])
        best_candidate = (
            max(candidates, key=geometric_candidate_rank)
            if candidates
            else None
        )
        best_candidate_inliers = int(
            best_candidate.get("inliers") or 0
        ) if best_candidate is not None else 0
        maximum_unselected_support = max(
            (
                len(accepted_by_anchor[anchor_id])
                for anchor_id in unselected_anchor_ids
            ),
            default=0,
        )
        required_inliers = max(8, int(minimum_geometric_inliers))
        local_winner_accepted = best_candidate_inliers >= required_inliers
        excluded_anchor_can_succeed = (
            maximum_unselected_support >= required_inliers
        )
        excluded_anchor_can_tie_or_beat = bool(
            local_winner_accepted
            and maximum_unselected_support >= best_candidate_inliers
        )
        if hierarchy_enabled and (
            (not local_winner_accepted and excluded_anchor_can_succeed)
            or excluded_anchor_can_tie_or_beat
        ):
            fallback_reason = (
                "hierarchy_no_accepted_local_winner"
                if not local_winner_accepted
                else "hierarchy_excluded_anchor_can_tie_or_beat_winner"
            )
            exact_candidates = self._match_candidates(
                query_keypoints=query_keypoints,
                query_descriptors=query_descriptors,
                anchors=valid_anchors,
                minimum_geometric_inliers=minimum_geometric_inliers,
                force_exact=True,
            )
            exact_diagnostic = dict(self.last_match_diagnostic)
            exact_diagnostic.update(
                {
                    "matching_profile": self.matching_profile,
                    "hierarchy_attempted": True,
                    "hierarchy_fallback": True,
                    "hierarchy_fallback_reason": fallback_reason,
                    "hierarchy_selected_anchor_count": len(anchor_blocks),
                    "hierarchy_unselected_anchor_count": len(
                        unselected_anchor_ids
                    ),
                    "hierarchy_local_winner_inliers": best_candidate_inliers,
                    "hierarchy_maximum_unselected_ratio_matches": (
                        maximum_unselected_support
                    ),
                }
            )
            self.last_match_diagnostic = exact_diagnostic
            return exact_candidates
        final_relevance_floor = max(
            homography_support_floor,
            int(math.ceil(best_geometric_inliers * 0.50)),
        )
        self.last_match_diagnostic = {
            "matching_profile": self.matching_profile,
            "anchor_count": len(valid_anchors),
            "ratio_supported_anchor_count": sum(
                len(accepted_by_anchor[anchor_id]) >= homography_support_floor
                for anchor_id in valid_anchors
            ),
            "homography_evaluated": homography_evaluated,
            "upper_bound_pruned": upper_bound_pruned,
            "maximum_pruned_ratio_matches": maximum_pruned_support,
            "best_geometric_inliers": best_geometric_inliers,
            "final_relevance_floor": final_relevance_floor,
            "upper_bound_verified": bool(
                maximum_pruned_support < final_relevance_floor
            ),
            "hierarchy_attempted": hierarchy_enabled,
            "hierarchy_fallback": False,
            "hierarchy_fallback_reason": "",
            "hierarchy_selected_anchor_count": len(anchor_blocks),
            "hierarchy_unselected_anchor_count": len(unselected_anchor_ids),
            "hierarchy_local_winner_inliers": best_candidate_inliers,
            "hierarchy_maximum_unselected_ratio_matches": (
                maximum_unselected_support
            ),
            "hierarchy_winner_proven": bool(
                not hierarchy_enabled
                or (
                    local_winner_accepted
                    and maximum_unselected_support < best_candidate_inliers
                )
                or (
                    not local_winner_accepted
                    and maximum_unselected_support < required_inliers
                )
            ),
        }
        return candidates

    def _update_endpoint_verification(
        self,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        winner = (
            max(candidates, key=geometric_candidate_rank)
            if candidates
            else None
        )
        # Endpoint confirmation runs while the aircraft is already inside the
        # strict geometric radius and physically hovering. Live Point 2 at
        # 14:38 produced a stable endpoint cluster at 97-99 inliers while the
        # normal route threshold remained 120. Keep translation at 120, but
        # allow this stricter whole-leg/three-frame endpoint consensus at 75.
        endpoint_minimum_inliers = max(
            72,
            int(round(self.minimum_inliers * 0.625)),
        )
        candidate = independent_endpoint_candidate_progress(
            candidates,
            minimum_inliers=endpoint_minimum_inliers,
        )
        # Endpoint consensus is a leg-scoped state transition. Once the
        # whole-leg audit has independently verified the endpoint over the
        # required consecutive frames, a single aliased/blurred frame must
        # not revoke it. The old edge-triggered behavior did exactly that:
        # the matcher verified Point 1, the next forward-pulse frame briefly
        # returned no independent winner, and progress was clamped backwards
        # to the endpoint guard. Keep the proof latched until _reset_for_key()
        # starts another leg/lap; current-frame best-match metadata remains
        # available to the flight gate as an additional arrival check.
        verification_latched = bool(self.endpoint_verified)
        if candidate is None and not verification_latched:
            self.endpoint_candidate_progress = None
            self.endpoint_hits = 0
            self.endpoint_verified = False
        elif (
            not verification_latched
            and candidate is not None
            and (
                self.endpoint_candidate_progress is None
                or abs(float(candidate) - self.endpoint_candidate_progress) > 0.08
            )
        ):
            self.endpoint_candidate_progress = float(candidate)
            self.endpoint_hits = 1
            self.endpoint_verified = False
        elif candidate is not None and not verification_latched:
            self.endpoint_candidate_progress = min(
                self.endpoint_candidate_progress, float(candidate)
            )
            self.endpoint_hits += 1
            self.endpoint_verified = (
                self.endpoint_hits >= self.endpoint_required_hits
            )
        current_view_geometry_verified = bool(
            winner is not None
            and candidate is not None
            and float(winner.get("progress") or 0.0) >= 0.90
            and winner.get("endpoint_view_geometry_verified") is not False
        )
        if current_view_geometry_verified:
            self.endpoint_view_geometry_verified = True
            self.endpoint_view_scale_min = winner.get("endpoint_view_scale_min")
            self.endpoint_view_scale_max = winner.get("endpoint_view_scale_max")
        endpoint_view_geometry_verified = bool(
            getattr(self, "endpoint_view_geometry_verified", False)
        )
        return {
            "endpoint_checked": True,
            # Descriptor consensus remains useful for command-bounded route
            # tracking, but it may declare physical waypoint arrival only
            # when the current view also has endpoint-like depth/scale.
            "endpoint_match_consensus_verified": bool(self.endpoint_verified),
            "endpoint_verified": bool(
                self.endpoint_verified and endpoint_view_geometry_verified
            ),
            "endpoint_view_geometry_verified": endpoint_view_geometry_verified,
            "endpoint_view_scale_min": (
                getattr(self, "endpoint_view_scale_min", None)
            ),
            "endpoint_view_scale_max": (
                getattr(self, "endpoint_view_scale_max", None)
            ),
            "endpoint_hits": int(self.endpoint_hits),
            "endpoint_required_hits": int(self.endpoint_required_hits),
            "endpoint_minimum_inliers": int(endpoint_minimum_inliers),
            "endpoint_candidate_progress": self.endpoint_candidate_progress,
            "endpoint_best_progress": (
                float(winner.get("progress") or 0.0) if winner is not None else None
            ),
            "endpoint_best_inliers": (
                int(winner.get("inliers") or 0) if winner is not None else 0
            ),
            "endpoint_best_anchor": (
                winner.get("anchor_name") if winner is not None else None
            ),
            "endpoint_candidate_count": len(candidates),
        }

    def observe_progress(self, progress: float | None) -> None:
        if progress is None or not math.isfinite(float(progress)):
            return
        value = max(0.0, min(1.0, float(progress)))
        self.last_progress = value if self.last_progress is None else max(self.last_progress, value)

    def commit_published_progress(self, progress: float | None) -> None:
        """Synchronize matching with the room position actually published."""
        if progress is None or not math.isfinite(float(progress)):
            return
        self.last_progress = max(0.0, min(1.0, float(progress)))

    def departure_heading_alignment(
        self,
        *,
        gray: np.ndarray,
        segment_start: Any,
        segment_end: Any,
        focal_px: float,
        departure_window: float = 0.05,
        minimum_inliers: int | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Measure camera yaw against recorded views at a patrol-leg start.

        Optical flow is useful for following an in-place turn through a weak
        map sector, but its small per-frame errors accumulate and cannot prove
        the final absolute heading.  The recorded departure images are an
        absolute visual reference at the same physical waypoint.  Restricting
        candidates to the first few percent of the commanded leg also keeps
        forward-motion parallax out of this rotation-only measurement.

        ``correction_deg`` is signed in the controller's convention: positive
        means the camera still needs a right yaw to match the recorded view.
        This method observes heading only and never publishes map position.
        """
        anchors = self._matching_anchors(segment_start, segment_end)
        if not anchors:
            return None, {"reason": "visual_heading_segment_not_in_locked_baseline"}
        try:
            focal = float(focal_px)
        except (TypeError, ValueError):
            focal = float("nan")
        if not math.isfinite(focal) or focal < 100.0:
            return None, {"reason": "visual_heading_focal_length_invalid"}

        first_progress = min(float(self.anchor_progress[index]) for index in anchors)
        ceiling = first_progress + max(0.01, min(0.10, float(departure_window)))
        anchors = [
            index
            for index in anchors
            if float(self.anchor_progress[index]) <= ceiling
        ]
        if not anchors:
            return None, {"reason": "visual_heading_departure_anchors_missing"}
        # Five adjacent recorded views are enough to absorb a small waypoint
        # position difference without adding a full route-matching pass to
        # every 10-FPS turn frame.
        anchors = anchors[:5]

        started = cv2.getTickCount()
        query_keypoints, query_descriptors = self.detector.detectAndCompute(
            np.asarray(gray), None
        )
        if query_descriptors is None or len(query_keypoints) < 120:
            return None, {"reason": "visual_heading_query_features_missing"}

        heading_minimum = max(
            16,
            int(self.minimum_inliers if minimum_inliers is None else minimum_inliers),
        )
        candidates = self._match_candidates(
            query_keypoints=query_keypoints,
            query_descriptors=query_descriptors,
            anchors=anchors,
            minimum_geometric_inliers=heading_minimum,
        )
        for candidate in candidates:
            horizontal_shift = candidate.get("horizontal_shift_px")
            if horizontal_shift is None:
                candidate["correction_deg"] = None
                continue
            # A camera that is still left of the taught heading sees recorded
            # features shifted right.  It therefore needs a positive/right
            # correction in the DJI bridge convention.
            correction_deg = math.degrees(math.atan2(horizontal_shift, focal))
            candidate["correction_deg"] = correction_deg
        candidates = [
            candidate
            for candidate in candidates
            if candidate.get("correction_deg") is not None
        ]

        elapsed_ms = 1000.0 * (cv2.getTickCount() - started) / cv2.getTickFrequency()
        if not candidates:
            return None, {"reason": "visual_heading_no_candidates", "total_ms": elapsed_ms}
        candidates.sort(key=geometric_candidate_rank, reverse=True)
        winner = candidates[0]
        if int(winner["inliers"]) < heading_minimum:
            return None, {
                "reason": "visual_heading_inliers_below_threshold",
                "best_inliers": int(winner["inliers"]),
                "minimum_inliers": heading_minimum,
                "total_ms": elapsed_ms,
            }
        return {
            "verified": True,
            "correction_deg": float(winner["correction_deg"]),
            "heading": self.anchor_headings[int(winner["anchor_id"])].astype(float).tolist(),
            "inliers": int(winner["inliers"]),
            "ratio_matches": int(winner["ratio_matches"]),
            "horizontal_shift_px": float(winner["horizontal_shift_px"]),
            "anchor_name": str(winner["anchor_name"]),
            "source_frame": int(winner["source_frame"]),
            "progress": float(winner["progress"]),
            "minimum_inliers": heading_minimum,
            "map_id": self.map_id,
            "patrol_id": self.patrol_id,
            "baseline_replay_id": self.baseline_replay_id,
        }, {
            "reason": "",
            "candidate_count": len(candidates),
            "best_inliers": int(winner["inliers"]),
            "correction_deg": float(winner["correction_deg"]),
            "total_ms": elapsed_ms,
        }

    def recover(
        self,
        *,
        gray: np.ndarray,
        segment_start: Any,
        segment_end: Any,
        segment_key: tuple[Any, ...],
        translation_locked: bool = False,
        progress_hint: float | None = None,
        progress_ceiling: float | None = None,
        recovery_hover: bool = False,
        independent_progress: bool = False,
        sequence_index: int | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        if translation_locked:
            self._mark_unverified()
            return None, {"reason": "visual_route_translation_locked"}
        self._reset_for_key(tuple(segment_key))
        try:
            command_progress_ceiling = float(progress_ceiling)
        except (TypeError, ValueError):
            command_progress_ceiling = None
        if command_progress_ceiling is not None:
            if not math.isfinite(command_progress_ceiling):
                command_progress_ceiling = None
            else:
                command_progress_ceiling = max(
                    0.0,
                    min(1.0, command_progress_ceiling),
                )
        segment_anchors = self._matching_anchors(segment_start, segment_end)
        # Keep an immutable whole-leg set for endpoint alias auditing and
        # recovery rewind.  The live search below is intentionally narrowed by
        # source/progress/time, but reusing that narrowed list for the alleged
        # "whole-leg" endpoint check hid the real mid-leg view in the 10:38
        # run and compared only six endpoint frames against one another.
        complete_segment_anchors = list(segment_anchors)
        selected_source_replay_id = (
            self.active_source_replay_id
            or (
                self.pending_source_replay_id
                if self.needs_acquisition
                else None
            )
        )
        if selected_source_replay_id is not None:
            segment_anchors = [
                index
                for index in segment_anchors
                if self.anchor_source_replay_ids[index]
                == selected_source_replay_id
            ]
        if not segment_anchors:
            self._mark_unverified()
            return None, {"reason": "visual_route_segment_not_in_locked_baseline"}
        if independent_progress:
            if self.last_matched_source_frame is None:
                if selected_source_replay_id is None:
                    first_by_source: dict[str, int] = {}
                    for index in segment_anchors:
                        source_id = self.anchor_source_replay_ids[index]
                        source_frame = int(self.source_frames[index])
                        first_by_source[source_id] = min(
                            source_frame,
                            first_by_source.get(source_id, source_frame),
                        )
                    segment_anchors = [
                        index
                        for index in segment_anchors
                        if int(self.source_frames[index])
                        <= first_by_source[self.anchor_source_replay_ids[index]] + 12
                    ]
                    temporal_floor = None
                    temporal_ceiling = None
                else:
                    first_source_frame = min(
                        int(self.source_frames[index]) for index in segment_anchors
                    )
                    temporal_floor = first_source_frame
                    temporal_ceiling = first_source_frame + 12
            else:
                frame_delta = 1
                if sequence_index is not None and self.last_sequence_index is not None:
                    frame_delta = max(1, int(sequence_index) - self.last_sequence_index)
                maximum_advance = max(3, int(math.ceil(2.0 * frame_delta)))
                if recovery_hover:
                    maximum_advance = max(60, maximum_advance)
                temporal_floor = self.last_matched_source_frame - 8
                temporal_ceiling = self.last_matched_source_frame + maximum_advance
            if temporal_floor is not None and temporal_ceiling is not None:
                segment_anchors = [
                    index
                    for index in segment_anchors
                    if temporal_floor
                    <= int(self.source_frames[index])
                    <= temporal_ceiling
                ]
            if not segment_anchors:
                self._mark_unverified()
                return None, {
                    "reason": "visual_route_temporal_anchor_window_empty",
                    "temporal_floor": temporal_floor,
                    "temporal_ceiling": temporal_ceiling,
                }
        committed_candidates = [0.0]
        if self.last_progress is not None:
            committed_candidates.append(float(self.last_progress))
        if progress_hint is not None:
            committed_candidates.append(float(progress_hint))
        base_progress = max(committed_candidates)
        base_progress = max(0.0, min(1.0, base_progress))
        matched_base_progress = (
            max(
                0.0,
                min(1.0, float(self.last_matched_progress or 0.0)),
            )
            if independent_progress
            else base_progress
        )
        floor = max(0.0, matched_base_progress - 0.04)
        ceiling = (
            1.0
            if recovery_hover
            else min(1.0, matched_base_progress + self.forward_window)
        )
        if command_progress_ceiling is not None:
            # Image recognition may look a few percent beyond the physical
            # command envelope so the departure/heading anchor remains
            # available.  It may not publish or commit that look-ahead as
            # translation.  Without this split, stationary Point-3 hover
            # frames walked through scale-invariant future ORB anchors.
            ceiling = min(ceiling, command_progress_ceiling + 0.045)
        anchors = [
            index
            for index in segment_anchors
            if floor <= self.anchor_progress[index] <= ceiling
        ]
        if not anchors:
            self._mark_unverified()
            return None, {"reason": "visual_route_no_anchor_in_progress_window"}

        started = cv2.getTickCount()
        query_keypoints, query_descriptors = self.detector.detectAndCompute(
            np.asarray(gray), None
        )
        if query_descriptors is None or len(query_keypoints) < 120:
            self._mark_unverified()
            return None, {"reason": "visual_route_query_features_missing"}
        candidates = self._match_candidates(
            query_keypoints=query_keypoints,
            query_descriptors=query_descriptors,
            anchors=anchors,
            # Every successful route path requires at least 60 geometric
            # inliers (the recovery-hover endpoint floor). Candidates below
            # that bound cannot influence publication or endpoint consensus.
            minimum_geometric_inliers=60,
            # Recovery hover may use multi-anchor weak endpoint consensus;
            # retain its complete candidate set rather than the normal-flight
            # winner-only hierarchy.
            force_exact=bool(recovery_hover),
        )
        elapsed_ms = 1000.0 * (cv2.getTickCount() - started) / cv2.getTickFrequency()
        if not candidates:
            self._mark_unverified()
            return None, {"reason": "visual_route_no_candidates", "total_ms": elapsed_ms}
        candidates.sort(key=geometric_candidate_rank, reverse=True)
        winner = candidates[0]
        winner_source_replay_id = str(winner.get("source_replay_id") or "")
        # Once a frame chooses the strongest recorded run, compute progress
        # only from neighboring anchors in that run.  Mixing equally valid
        # views from two flights with different pulse timing can otherwise
        # create a false 10-20% progress shift even though both recognize the
        # correct room location.
        source_candidates = [
            item
            for item in candidates
            if str(item.get("source_replay_id") or "")
            == winner_source_replay_id
        ]
        if source_candidates:
            candidates = source_candidates
            winner = max(candidates, key=geometric_candidate_rank)
        start = finite_vector(segment_start)
        end = finite_vector(segment_end)
        assert start is not None and end is not None
        leg_length = float(np.linalg.norm((end - start)[[0, 2]]))
        weak_endpoint_recovery = False
        temporal_recovery = False
        effective_minimum_inliers = self.minimum_inliers
        if int(winner["inliers"]) < self.minimum_inliers:
            weak_endpoint = (
                weak_endpoint_candidate_progress(candidates)
                if recovery_hover
                else None
            )
            if weak_endpoint is None and (
                not recovery_hover
                or int(winner["inliers"]) < self.recovery_minimum_inliers
            ):
                self._mark_unverified()
                return None, {
                    "reason": "visual_route_inliers_below_threshold",
                    "best_inliers": int(winner["inliers"]),
                    "minimum_inliers": self.minimum_inliers,
                    "total_ms": elapsed_ms,
                }
            if weak_endpoint is not None:
                self.temporal_recovery_progress = None
                self.temporal_recovery_source_replay_id = None
                self.temporal_recovery_hits = 0
                if (
                    self.weak_endpoint_progress is None
                    or abs(float(weak_endpoint) - self.weak_endpoint_progress) > 0.08
                ):
                    self.weak_endpoint_progress = float(weak_endpoint)
                    self.weak_endpoint_hits = 1
                else:
                    self.weak_endpoint_progress = min(
                        self.weak_endpoint_progress,
                        float(weak_endpoint),
                    )
                    self.weak_endpoint_hits += 1
                if self.weak_endpoint_hits < 5:
                    return None, {
                        "reason": "visual_route_weak_endpoint_acquiring",
                        "acquisition_hits": self.weak_endpoint_hits,
                        "required_hits": 5,
                        "best_inliers": int(winner["inliers"]),
                        "minimum_inliers": self.minimum_inliers,
                        "total_ms": elapsed_ms,
                    }
                proposed = max(
                    matched_base_progress,
                    float(self.weak_endpoint_progress),
                )
                weak_endpoint_recovery = True
            else:
                # A recovery hover is physically motionless.  The matcher may
                # therefore integrate several slightly weaker frames before
                # publishing, provided they agree on one source run and one
                # local route neighborhood.  The command-progress ceiling
                # below remains the hard upper bound, so this cannot invent
                # motion that was never commanded.
                self.weak_endpoint_progress = None
                self.weak_endpoint_hits = 0
                sequence_forward_window = max(
                    0.12,
                    min(
                        self.forward_window,
                        (
                            (self.max_position_step + 0.02) / leg_length
                            if leg_length > 1.0e-9
                            else 0.12
                        ),
                    ),
                )
                proposed = sequence_candidate_progress(
                    candidates,
                    previous=matched_base_progress,
                    minimum_inliers=self.recovery_minimum_inliers,
                    forward_window=sequence_forward_window,
                )
                if proposed is None:
                    self._mark_unverified()
                    return None, {
                        "reason": "visual_route_temporal_recovery_consensus_missing",
                        "best_inliers": int(winner["inliers"]),
                        "minimum_inliers": self.recovery_minimum_inliers,
                        "total_ms": elapsed_ms,
                    }
                if (
                    self.temporal_recovery_progress is None
                    or self.temporal_recovery_source_replay_id
                    != winner_source_replay_id
                    or abs(
                        float(proposed) - self.temporal_recovery_progress
                    )
                    > 0.12
                ):
                    self.temporal_recovery_progress = float(proposed)
                    self.temporal_recovery_source_replay_id = (
                        winner_source_replay_id
                    )
                    self.temporal_recovery_hits = 1
                else:
                    self.temporal_recovery_progress = min(
                        self.temporal_recovery_progress,
                        float(proposed),
                    )
                    self.temporal_recovery_hits += 1
                if self.temporal_recovery_hits < self.recovery_acquisition_hits:
                    return None, {
                        "reason": "visual_route_temporal_recovery_acquiring",
                        "acquisition_hits": self.temporal_recovery_hits,
                        "required_hits": self.recovery_acquisition_hits,
                        "best_inliers": int(winner["inliers"]),
                        "minimum_inliers": self.recovery_minimum_inliers,
                        "total_ms": elapsed_ms,
                    }
                proposed = max(
                    matched_base_progress,
                    float(self.temporal_recovery_progress),
                )
                temporal_recovery = True
                effective_minimum_inliers = self.recovery_minimum_inliers
                self.needs_acquisition = False
                self.pending_progress = float(proposed)
                self.pending_source_replay_id = winner_source_replay_id
                self.pending_hits = self.temporal_recovery_hits
        else:
            self.weak_endpoint_progress = None
            self.weak_endpoint_hits = 0
            self.temporal_recovery_progress = None
            self.temporal_recovery_source_replay_id = None
            self.temporal_recovery_hits = 0
            # A fixed 0.12 normalized window is smaller than one safe 18 cm
            # correction on the short Point-2 -> Point-3 leg.  A real forward
            # pulse could therefore move from progress 0.785 to 0.908 while
            # the matcher kept choosing the older 0.785 anchor forever. Size
            # the consensus window from the same metric correction bound used
            # below, plus 2 cm of image/route tolerance. It remains capped by
            # the route search window and cannot skip to another leg.
            sequence_forward_window = max(
                0.12,
                min(
                    self.forward_window,
                    (
                        (self.max_position_step + 0.02) / leg_length
                        if leg_length > 1.0e-9
                        else 0.12
                    ),
                ),
            )
            proposed = sequence_candidate_progress(
                candidates,
                previous=matched_base_progress,
                minimum_inliers=self.minimum_inliers,
                forward_window=sequence_forward_window,
            )
            if proposed is not None:
                # Report the anchor that actually supplied route progress.
                # Previously the UI displayed the global best (often a future
                # frame) while publication deliberately used an older local
                # anchor, making frames and model appear out of sync.
                winner = max(
                    candidates,
                    key=lambda item: (
                        -abs(float(item["progress"]) - float(proposed)),
                        *geometric_candidate_rank(item),
                    ),
                )
        if proposed is None:
            self._mark_unverified()
            return None, {"reason": "visual_route_progress_consensus_missing"}

        verified_rewind = False
        endpoint_metadata: dict[str, Any] = {
            "endpoint_checked": False,
            "endpoint_verified": False,
            "endpoint_match_consensus_verified": bool(self.endpoint_verified),
            "endpoint_view_geometry_verified": False,
            "endpoint_view_scale_min": None,
            "endpoint_view_scale_max": None,
            "endpoint_hits": int(self.endpoint_hits),
            "endpoint_required_hits": int(self.endpoint_required_hits),
            "endpoint_minimum_inliers": max(
                72, int(round(self.minimum_inliers * 0.625))
            ),
            "endpoint_candidate_progress": self.endpoint_candidate_progress,
            "endpoint_best_progress": None,
            "endpoint_best_inliers": 0,
            "endpoint_best_anchor": None,
            "endpoint_candidate_count": 0,
            "verified_rewind": False,
            "verified_rewind_progress": None,
            "verified_rewind_inliers": 0,
            "verified_rewind_hits": int(self.rewind_candidate_hits),
            "verified_rewind_required_hits": 5,
        }
        endpoint_check_needed = bool(
            recovery_hover
            or max(matched_base_progress, float(proposed))
            >= self.endpoint_guard_progress - 0.04
        )
        if endpoint_check_needed:
            matched_anchor_ids = {
                int(item["anchor_id"]) for item in candidates
            }
            endpoint_audit_anchors = self._endpoint_audit_anchors(
                complete_segment_anchors
            )
            missing_anchors = [
                anchor_id
                for anchor_id in endpoint_audit_anchors
                if anchor_id not in matched_anchor_ids
            ]
            full_candidates = candidates + self._match_candidates(
                query_keypoints=query_keypoints,
                query_descriptors=query_descriptors,
                anchors=missing_anchors,
                minimum_geometric_inliers=60,
                # Whole-leg endpoint auditing needs negative evidence from
                # both endpoint and non-endpoint sentinels, not only a proven
                # local winner.
                force_exact=True,
            )
            endpoint_metadata = self._update_endpoint_verification(full_candidates)
            dense_temporal_endpoint = bool(
                independent_progress
                and int(winner.get("inliers") or 0) >= self.minimum_inliers
                and matched_base_progress >= 0.88
                and float(winner.get("progress") or 0.0) >= 0.90
            )
            if dense_temporal_endpoint:
                # Dense weak-leg matching has already verified an ordered
                # current-image sequence from the departure through the
                # endpoint region. Neighboring dense frames intentionally tie
                # on descriptor score, so the sparse-bank rule requiring an
                # endpoint to beat every non-endpoint is no longer meaningful.
                # Reaching 90% from a previously verified >=88% frame is a
                # multi-frame arrival proof, not an isolated endpoint alias.
                self.endpoint_candidate_progress = float(winner["progress"])
                self.endpoint_hits = max(
                    self.endpoint_hits,
                    self.endpoint_required_hits,
                )
                self.endpoint_verified = True
                if winner.get("endpoint_view_geometry_verified") is not False:
                    self.endpoint_view_geometry_verified = True
                    self.endpoint_view_scale_min = winner.get(
                        "endpoint_view_scale_min"
                    )
                    self.endpoint_view_scale_max = winner.get(
                        "endpoint_view_scale_max"
                    )
                endpoint_metadata.update(
                    {
                        "endpoint_match_consensus_verified": True,
                        "endpoint_verified": bool(
                            self.endpoint_view_geometry_verified
                        ),
                        "endpoint_view_geometry_verified": bool(
                            self.endpoint_view_geometry_verified
                        ),
                        "endpoint_view_scale_min": self.endpoint_view_scale_min,
                        "endpoint_view_scale_max": self.endpoint_view_scale_max,
                        "endpoint_hits": int(self.endpoint_hits),
                        "endpoint_candidate_progress": float(winner["progress"]),
                        "endpoint_dense_temporal_sequence": True,
                    }
                )
            false_endpoint_depth = bool(
                recovery_hover
                and matched_base_progress >= 0.84
                and endpoint_metadata.get(
                    "endpoint_match_consensus_verified"
                )
                is True
                and endpoint_metadata.get(
                    "endpoint_view_geometry_verified"
                )
                is not True
            )
            rewind_latched = bool(
                recovery_hover
                and self.rewind_candidate_hits >= 5
                and self.rewind_candidate_progress is not None
                and self.rewind_candidate_source_replay_id
            )
            continuing_rewind = False
            if false_endpoint_depth:
                rewind_candidate = verified_recovery_rewind_candidate(
                    full_candidates,
                    previous=matched_base_progress,
                    minimum_inliers=self.minimum_inliers,
                )
            elif rewind_latched:
                latched_target = float(self.rewind_candidate_progress)
                latched_source = str(
                    self.rewind_candidate_source_replay_id or ""
                )
                latched_candidates = [
                    item
                    for item in full_candidates
                    if str(item.get("source_replay_id") or "")
                    == latched_source
                    and abs(
                        float(item.get("progress") or 0.0)
                        - latched_target
                    )
                    <= 0.12
                    and int(item.get("inliers") or 0)
                    >= self.minimum_inliers
                    and item.get("endpoint_view_geometry_verified") is True
                ]
                rewind_candidate = (
                    max(
                        latched_candidates,
                        key=lambda item: int(item.get("inliers") or 0),
                    )
                    if latched_candidates
                    else None
                )
                continuing_rewind = rewind_candidate is not None
            else:
                rewind_candidate = None
            if rewind_candidate is None:
                self.rewind_candidate_progress = None
                self.rewind_candidate_source_replay_id = None
                self.rewind_candidate_hits = 0
            else:
                rewind_progress = float(rewind_candidate["progress"])
                rewind_source = str(
                    rewind_candidate.get("source_replay_id") or ""
                )
                if not continuing_rewind:
                    stable_rewind = bool(
                        self.rewind_candidate_progress is not None
                        and self.rewind_candidate_source_replay_id
                        == rewind_source
                        and abs(
                            rewind_progress
                            - float(self.rewind_candidate_progress)
                        )
                        <= 0.10
                    )
                    if stable_rewind:
                        self.rewind_candidate_progress = min(
                            float(self.rewind_candidate_progress),
                            rewind_progress,
                        )
                        self.rewind_candidate_hits += 1
                    else:
                        self.rewind_candidate_progress = rewind_progress
                        self.rewind_candidate_source_replay_id = rewind_source
                        self.rewind_candidate_hits = 1
                if self.rewind_candidate_hits >= 5:
                    # The aircraft has been neutrally hovering for all five
                    # observations.  Correct only the model's over-advanced
                    # route clock; no physical reverse command is implied.
                    verified_rewind = True
                    proposed = float(self.rewind_candidate_progress)
                    winner = rewind_candidate
                    winner_source_replay_id = rewind_source
                    # A multi-run rewind may prove that the current camera is
                    # following another validated recording.  Keep that
                    # source and its frame clock after the correction.  The
                    # 12:20:39 run found the correct 57% frame from the 15:47
                    # baseline, but left the source locked to 11:57 at its
                    # Point-4 endpoint.  The very next query therefore jumped
                    # back to the false endpoint or returned no consensus.
                    self.active_source_replay_id = rewind_source
                    self.pending_source_replay_id = rewind_source
                    self.last_matched_source_frame = int(
                        rewind_candidate.get("source_frame") or 0
                    )
                    weak_endpoint_recovery = False
                    temporal_recovery = False
                    effective_minimum_inliers = self.minimum_inliers
                    self.needs_acquisition = False
                endpoint_metadata.update(
                    {
                        "verified_rewind": verified_rewind,
                        "verified_rewind_progress": (
                            float(self.rewind_candidate_progress)
                            if self.rewind_candidate_progress is not None
                            else None
                        ),
                        "verified_rewind_inliers": int(
                            rewind_candidate.get("inliers") or 0
                        ),
                        "verified_rewind_hits": int(
                            self.rewind_candidate_hits
                        ),
                        "verified_rewind_required_hits": 5,
                    }
                )
            elapsed_ms = (
                1000.0
                * (cv2.getTickCount() - started)
                / cv2.getTickFrequency()
            )

        if (
            endpoint_metadata.get("endpoint_match_consensus_verified") is True
            and not verified_rewind
        ):
            # Whole-leg endpoint consensus is independent, stronger evidence
            # than the local sequence prior. Let it become the target once it
            # is latched; the room-distance reconciliation and publication
            # gates below still make the rendered correction gradual. Invalid
            # endpoint depth remains outside the arrival radius and triggers
            # the persistent verified-rewind path above.
            endpoint_progress = endpoint_metadata.get("endpoint_candidate_progress")
            try:
                endpoint_progress = float(endpoint_progress)
            except (TypeError, ValueError):
                endpoint_progress = None
            if endpoint_progress is not None and math.isfinite(endpoint_progress):
                proposed = max(float(proposed), min(1.0, endpoint_progress))

        # Route progress is a tracking prior, never arrival evidence.  Until a
        # whole-leg comparison verifies the endpoint across several frames,
        # keep the visual pose outside every strict 0.15 m waypoint radius.
        # The controller can therefore continue issuing its normal bounded
        # forward pulses instead of accepting a self-advanced route prior.
        endpoint_guarded = bool(
            float(proposed) > self.endpoint_guard_progress
            and endpoint_metadata.get("endpoint_verified") is not True
        )
        # Keep an unverified visual pose at least 16 cm before the waypoint.
        # This is outside the controller's strict 0.15 m arrival radius, but
        # unlike a fixed 0.84 progress wall it lets the rendered pose follow
        # strong endpoint-region images smoothly while independent endpoint
        # consensus is still being collected. Cap at 0.90 for long legs so a
        # repeated-room alias can never self-advance close to the destination.
        safe_prearrival_progress = self.endpoint_guard_progress
        if leg_length > 1e-9:
            safe_prearrival_progress = max(
                self.endpoint_guard_progress,
                min(0.90, 1.0 - 0.16 / leg_length),
            )
        if endpoint_guarded:
            # The guard may stop an unverified visual observation from
            # advancing farther, but it must never rewind route truth that the
            # publication gate has already committed. Rewinding 0.865 -> 0.84
            # made the monotonic promotion gate reject every otherwise-good
            # 4->1 frame and froze the model until the final endpoint match.
            endpoint_best = endpoint_metadata.get("endpoint_best_progress")
            try:
                endpoint_best = float(endpoint_best)
            except (TypeError, ValueError):
                endpoint_best = self.endpoint_guard_progress
            if not math.isfinite(endpoint_best):
                endpoint_best = self.endpoint_guard_progress
            guard_floor = matched_base_progress
            if (
                endpoint_metadata.get("endpoint_match_consensus_verified") is True
                and endpoint_metadata.get("endpoint_view_geometry_verified")
                is not True
            ):
                # Descriptor sequence consensus may continue consuming the
                # controller's bounded command budget, but a wrong-depth wall
                # match must not carry its self-advanced matcher floor into the
                # published pose.  Keep only the already published/metric
                # progress until a same-depth endpoint view arrives.
                guard_floor = base_progress
            proposed = max(
                guard_floor,
                self.endpoint_guard_progress,
                min(endpoint_best, safe_prearrival_progress),
            )

        if self.needs_acquisition and not weak_endpoint_recovery:
            if (
                self.pending_progress is None
                or self.pending_source_replay_id != winner_source_replay_id
                or abs(proposed - self.pending_progress) > 0.12
            ):
                self.pending_progress = proposed
                self.pending_source_replay_id = winner_source_replay_id
                self.pending_hits = 1
            else:
                self.pending_progress = min(self.pending_progress, proposed)
                self.pending_hits += 1
            if self.pending_hits < self.acquisition_hits:
                return None, {
                    "reason": "visual_route_acquiring",
                    "acquisition_hits": self.pending_hits,
                    "required_hits": self.acquisition_hits,
                    "best_inliers": int(winner["inliers"]),
                    "total_ms": elapsed_ms,
                }
            proposed = max(
                matched_base_progress,
                float(self.pending_progress),
            )
            self.needs_acquisition = False
            self.active_source_replay_id = self.pending_source_replay_id

        unbounded_proposed_progress = float(proposed)
        command_progress_guarded = bool(
            command_progress_ceiling is not None
            and unbounded_proposed_progress > command_progress_ceiling + 1e-9
        )
        if command_progress_ceiling is not None:
            # This is the hard physical invariant.  A visual matcher may
            # identify a later-looking saved frame, but yaw/hover has a zero
            # translation budget and each horizontal command adds only its
            # bounded unresolved distance.  Do not advance either the model
            # or the matcher's monotonic clock beyond that budget.
            proposed = min(float(proposed), command_progress_ceiling)

        # Only a verified current-image sequence result advances the visual
        # search window.  Published metric progress remains a reconciliation
        # reference below, never a candidate-selection prior.
        self.last_matched_progress = (
            float(proposed)
            if verified_rewind
            else max(
                matched_base_progress,
                float(proposed),
            )
        )
        winner_source_frame = int(winner.get("source_frame") or 0)
        if self.last_matched_source_frame is None:
            self.last_matched_source_frame = winner_source_frame
        elif float(winner.get("progress") or 0.0) >= matched_base_progress - 1e-6:
            self.last_matched_source_frame = max(
                self.last_matched_source_frame,
                winner_source_frame,
            )
        if sequence_index is not None:
            self.last_sequence_index = int(sequence_index)

        # Acquisition can reveal that a smooth TSolve track is globally ahead
        # or behind the recorded route.  Reconcile that disagreement over
        # bounded position steps while the bridge hovers; never publish the
        # complete correction as a single model/controller jump.
        # ``progress_hint`` is the position the main TSolve route gate has
        # already published. It is newer than this monitor's last successful
        # visual match, so include it in the step-limit reference through
        # ``base_progress``. Using only stale ``self.last_progress`` made the
        # visual monitor remain near 0.085 while the healthy model was already
        # at 0.939 on Live ATLAS 13:23:59, creating a false disagreement hold.
        reference_progress = float(base_progress)
        if leg_length > 1e-9:
            proposed = min(
                float(proposed),
                reference_progress + self.max_position_step / leg_length,
            )

        proposed_progress = float(proposed)
        if endpoint_guarded:
            proposed_progress = min(
                proposed_progress,
                max(safe_prearrival_progress, reference_progress),
            )
        # Detection proposes a target; it does not commit route truth. The
        # localizer commits only the bounded room center it actually publishes
        # through ``commit_published_progress``. This keeps matching, model
        # rendering, and flight gating on one progress clock.
        center = start + (end - start) * proposed_progress
        anchor_id = int(winner["anchor_id"])
        return {
            "verified": True,
            "center": center.astype(float).tolist(),
            "heading": self.anchor_headings[anchor_id].astype(float).tolist(),
            "progress": proposed_progress,
            # Keep the current image match separate from the monotonic route
            # clock.  A just-started leg may already have a small false TSolve
            # floor; reporting only proposed_progress hides the correct
            # departure anchor and makes that floor impossible to repair.
            "matched_progress": float(self.anchor_progress[anchor_id]),
            "inliers": int(winner["inliers"]),
            "ratio_matches": int(winner["ratio_matches"]),
            "anchor_name": str(winner["anchor_name"]),
            "source_frame": int(winner["source_frame"]),
            "source_replay_id": winner_source_replay_id,
            "acquisition_hits": self.pending_hits,
            "minimum_inliers": effective_minimum_inliers,
            "map_id": self.map_id,
            "patrol_id": self.patrol_id,
            "baseline_replay_id": self.baseline_replay_id,
            "translation_safe": not weak_endpoint_recovery,
            "command_progress_ceiling": command_progress_ceiling,
            "command_progress_guarded": command_progress_guarded,
            "unbounded_progress": unbounded_proposed_progress,
            "weak_endpoint_recovery": weak_endpoint_recovery,
            "temporal_recovery": temporal_recovery,
            "temporal_recovery_hits": int(self.temporal_recovery_hits),
            "temporal_recovery_required_hits": int(
                self.recovery_acquisition_hits
            ),
            "endpoint_guarded": endpoint_guarded,
            "endpoint_guard_progress": self.endpoint_guard_progress,
            "endpoint_safe_prearrival_progress": safe_prearrival_progress,
            **endpoint_metadata,
        }, {
            "reason": "",
            "candidate_count": len(candidates),
            "best_inliers": int(winner["inliers"]),
            "progress": proposed_progress,
            "command_progress_ceiling": command_progress_ceiling,
            "command_progress_guarded": command_progress_guarded,
            "unbounded_progress": unbounded_proposed_progress,
            "weak_endpoint_recovery": weak_endpoint_recovery,
            "temporal_recovery": temporal_recovery,
            "temporal_recovery_hits": int(self.temporal_recovery_hits),
            "temporal_recovery_required_hits": int(
                self.recovery_acquisition_hits
            ),
            "endpoint_guarded": endpoint_guarded,
            "endpoint_guard_progress": self.endpoint_guard_progress,
            "endpoint_safe_prearrival_progress": safe_prearrival_progress,
            **endpoint_metadata,
            "total_ms": elapsed_ms,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--frame-dir", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build_bank(args.baseline, args.out, frame_dir=args.frame_dir),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
