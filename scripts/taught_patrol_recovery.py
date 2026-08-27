#!/usr/bin/env python3
"""Build and query compact 2D->3D recovery anchors from taught patrol runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _require_faiss():
    try:
        import faiss  # type: ignore
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError(
            "Faiss is required for taught-patrol SIFT anchor retrieval."
        ) from exc
    return faiss


def unique_ann_point_matches(
    *,
    distances: np.ndarray,
    neighbor_ids: np.ndarray,
    ann_rows: np.ndarray,
    point3d_ids: np.ndarray,
    anchor_ids: np.ndarray,
    ratio: float = 0.78,
) -> list[dict[str, Any]]:
    """Convert one SIFT ANN query into unique query-feature/3D-point pairs.

    Faiss reports squared L2 distance.  The ratio test therefore compares the
    squared ratio and, importantly, searches for the second *different 3D
    point*.  Multiple taught views often contain observations of the same map
    point and must not incorrectly become each other's ratio-test competitor.
    """
    ratio_squared = float(ratio) ** 2
    matches: list[dict[str, Any]] = []
    for query_index, (query_distances, query_neighbors) in enumerate(
        zip(np.asarray(distances), np.asarray(neighbor_ids))
    ):
        valid: list[tuple[float, int, int, int]] = []
        for distance, ann_id in zip(query_distances, query_neighbors):
            ann_id = int(ann_id)
            distance = float(distance)
            if ann_id < 0 or ann_id >= len(ann_rows) or not math.isfinite(distance):
                continue
            source_row = int(ann_rows[ann_id])
            valid.append(
                (
                    distance,
                    source_row,
                    int(point3d_ids[source_row]),
                    int(anchor_ids[source_row]),
                )
            )
        if not valid:
            continue
        best_distance, best_row, best_point, _best_anchor = valid[0]
        second_distance = next(
            (distance for distance, _row, point_id, _anchor in valid if point_id != best_point),
            None,
        )
        if second_distance is None:
            continue
        if best_distance > ratio_squared * max(float(second_distance), 1.0e-12):
            continue
        supporting_anchors = sorted(
            {
                anchor_id
                for distance, _row, point_id, anchor_id in valid
                if point_id == best_point and distance <= float(second_distance)
            }
        )
        matches.append(
            {
                "query_index": int(query_index),
                "point3d_id": int(best_point),
                "source_row": int(best_row),
                "distance": float(best_distance),
                "second_point_distance": float(second_distance),
                "anchor_ids": supporting_anchors,
            }
        )

    # A map point may be assigned to only one current-image feature.
    best_by_point: dict[int, dict[str, Any]] = {}
    for match in matches:
        point_id = int(match["point3d_id"])
        previous = best_by_point.get(point_id)
        if previous is None or (
            float(match["distance"]), int(match["query_index"])
        ) < (
            float(previous["distance"]), int(previous["query_index"])
        ):
            best_by_point[point_id] = match
    return sorted(best_by_point.values(), key=lambda item: int(item["query_index"]))


def select_anchor_match_window(
    matches: list[dict[str, Any]],
    *,
    anchor_names: list[str],
    radius: int = 4,
    minimum_points: int = 8,
    minimum_anchors: int = 3,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Select a locally coherent taught-view window without estimating pose.

    Descriptor agreement chooses a small temporal keyframe neighborhood.  It
    is deliberately only correspondence selection: TSolve receives the
    resulting 2D->3D set and is the sole component allowed to estimate R,t.
    """
    anchor_votes: dict[int, int] = {}
    for match in matches:
        for anchor_id in match.get("anchor_ids") or []:
            anchor_votes[int(anchor_id)] = anchor_votes.get(int(anchor_id), 0) + 1
    if not anchor_votes:
        return [], []

    best: tuple[tuple[int, int, int, float], list[dict[str, Any]], list[int]] | None = None
    for seed in anchor_votes:
        if seed < 0 or seed >= len(anchor_names):
            continue
        # Do not let an integer-neighbor window cross from one recorded replay
        # into another merely because their anchors are adjacent in the bank.
        seed_group = anchor_names[seed].split("/", 1)[0]
        window = [
            anchor_id
            for anchor_id in anchor_votes
            if abs(anchor_id - seed) <= max(0, int(radius))
            and 0 <= anchor_id < len(anchor_names)
            and anchor_names[anchor_id].split("/", 1)[0] == seed_group
        ]
        window_set = set(window)
        selected = [
            match
            for match in matches
            if window_set.intersection(int(value) for value in (match.get("anchor_ids") or []))
        ]
        contributing = sorted(
            {
                int(anchor_id)
                for match in selected
                for anchor_id in (match.get("anchor_ids") or [])
                if int(anchor_id) in window_set
            }
        )
        median_distance = (
            float(np.median([float(item["distance"]) for item in selected]))
            if selected
            else math.inf
        )
        rank = (
            len(selected),
            len(contributing),
            sum(anchor_votes.get(anchor_id, 0) for anchor_id in window),
            -median_distance,
        )
        if best is None or rank > best[0]:
            best = (rank, selected, contributing)
    if best is None:
        return [], []
    selected, contributing = best[1], best[2]
    if len(selected) < max(8, int(minimum_points)) or len(contributing) < max(1, int(minimum_anchors)):
        return [], []
    return selected, contributing


def rotmat_to_qvec(R: np.ndarray) -> list[float]:
    """Convert world-to-camera rotation to COLMAP's ``[w, x, y, z]``.

    This eigen decomposition yields the quaternion for the transpose
    convention.  Transpose the requested matrix up front so converting the
    returned qvec with ``colmap_io.qvec_to_rotmat`` round-trips to ``R``.
    Without this transpose, a successful taught recovery published a false
    reference camera center and made the immediately following recovery fail.
    """
    matrix = np.asarray(R, dtype=float).reshape(3, 3).T
    K = np.array(
        [
            [matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 0, 0, 0],
            [matrix[1, 0] + matrix[0, 1], matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 0, 0],
            [matrix[2, 0] + matrix[0, 2], matrix[2, 1] + matrix[1, 2], matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 0],
            [matrix[1, 2] - matrix[2, 1], matrix[2, 0] - matrix[0, 2], matrix[0, 1] - matrix[1, 0], matrix.trace()],
        ],
        dtype=float,
    ) / 3.0
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    q = eigenvectors[[3, 0, 1, 2], int(np.argmax(eigenvalues))]
    if q[0] < 0:
        q = -q
    return q.astype(float).tolist()


def consensus_candidate_cluster(
    candidates: list[dict[str, Any]],
    *,
    radius: float = 0.18,
    minimum: int = 3,
) -> list[dict[str, Any]]:
    """Return the strongest mutually consistent camera-center cluster."""
    if not candidates:
        return []
    best_inliers = max(int(item.get("inliers") or 0) for item in candidates)
    credible_minimum = max(8, int(math.ceil(best_inliers * 0.65)))
    credible = [item for item in candidates if int(item.get("inliers") or 0) >= credible_minimum]
    ranked: list[tuple[tuple[int, int, float], list[dict[str, Any]]]] = []
    for seed in credible:
        center = np.asarray(seed["center"], dtype=float).reshape(3)
        cluster = [
            item
            for item in credible
            if float(np.linalg.norm(np.asarray(item["center"], dtype=float).reshape(3) - center))
            <= float(radius)
        ]
        steps = [float(item.get("center_step") or 0.0) for item in cluster]
        rank = (len(cluster), sum(int(item.get("inliers") or 0) for item in cluster), -float(np.median(steps)))
        ranked.append((rank, cluster))
    if not ranked:
        return []
    cluster = max(ranked, key=lambda item: item[0])[1]
    return cluster if len(cluster) >= max(1, int(minimum)) else []


class TaughtPatrolRecovery:
    def __init__(self, bank_path: Path):
        self.bank_path = Path(bank_path)
        with np.load(self.bank_path, allow_pickle=False) as bank:
            self.descriptors = np.asarray(bank["descriptors"], dtype=np.float32)
            self.p3d = np.asarray(bank["p3d"], dtype=np.float64)
            self.point3d_ids = np.asarray(bank["point3d_ids"], dtype=np.int64)
            self.anchor_ids = np.asarray(bank["anchor_ids"], dtype=np.int32)
            self.anchor_names = [str(value) for value in bank["anchor_names"].tolist()]
        self.sift = cv2.SIFT_create(nfeatures=1024, contrastThreshold=0.015)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.max_anchors: int | None = None

    @classmethod
    def empty_online(cls, *, max_anchors: int = 60) -> "TaughtPatrolRecovery":
        """Create an in-memory 2D->3D bank learned only from this live run."""
        recovery = cls.__new__(cls)
        recovery.bank_path = Path("<online-live-map-anchors>")
        recovery.descriptors = np.empty((0, 128), dtype=np.float32)
        recovery.p3d = np.empty((0, 3), dtype=np.float64)
        recovery.point3d_ids = np.empty((0,), dtype=np.int64)
        recovery.anchor_ids = np.empty((0,), dtype=np.int32)
        recovery.anchor_names = []
        recovery.sift = cv2.SIFT_create(nfeatures=1024, contrastThreshold=0.015)
        recovery.matcher = cv2.BFMatcher(cv2.NORM_L2)
        recovery.max_anchors = max(4, int(max_anchors))
        return recovery

    def _prune_online_anchors(self) -> None:
        if self.max_anchors is None or len(self.anchor_names) <= self.max_anchors:
            return
        remove_count = len(self.anchor_names) - self.max_anchors
        keep_rows = self.anchor_ids >= remove_count
        self.descriptors = self.descriptors[keep_rows]
        self.p3d = self.p3d[keep_rows]
        self.point3d_ids = self.point3d_ids[keep_rows]
        self.anchor_ids = self.anchor_ids[keep_rows] - remove_count
        self.anchor_names = self.anchor_names[remove_count:]

    def save(self, out_path: Path) -> None:
        """Atomically persist the current static and learned anchor set."""
        destination = Path(out_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    descriptors=np.asarray(self.descriptors, dtype=np.float32),
                    p3d=np.asarray(self.p3d, dtype=np.float64),
                    point3d_ids=np.asarray(self.point3d_ids, dtype=np.int64),
                    anchor_ids=np.asarray(self.anchor_ids, dtype=np.int32),
                    anchor_names=np.asarray(self.anchor_names),
                )
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    def learn_anchor(
        self,
        *,
        gray: np.ndarray,
        xy: np.ndarray,
        p3d: np.ndarray,
        point3d_ids: np.ndarray,
        anchor_name: str,
        max_pixel_distance: float = 5.0,
    ) -> int:
        """Add one accepted live 2D->3D view to this process' recovery bank.

        The anchor stays in memory.  It lets a live patrol learn the actual
        route it is flying without rewriting the operator's saved taught bank.
        """
        name = str(anchor_name)
        if not name or name in self.anchor_names:
            return 0
        image_xy = np.atleast_2d(np.asarray(xy, dtype=np.float32))
        world_xyz = np.atleast_2d(np.asarray(p3d, dtype=np.float64))
        point_ids = np.asarray(point3d_ids, dtype=np.int64).reshape(-1)
        if len(image_xy) < 8 or len(image_xy) != len(world_xyz) or len(image_xy) != len(point_ids):
            return 0
        keypoints, image_descriptors = self.sift.detectAndCompute(np.asarray(gray), None)
        if image_descriptors is None or len(keypoints) < 8:
            return 0
        feature_xy = np.float32([point.pt for point in keypoints])
        distances = np.linalg.norm(feature_xy[:, None, :] - image_xy[None, :, :], axis=2)
        nearest = np.argmin(distances, axis=1)
        closest = np.min(distances, axis=1)
        best_by_point: dict[int, tuple[float, int]] = {}
        for feature_index in np.flatnonzero(closest <= float(max_pixel_distance)):
            point_index = int(nearest[feature_index])
            candidate = (float(closest[feature_index]), int(feature_index))
            old = best_by_point.get(point_index)
            if old is None or candidate < old:
                best_by_point[point_index] = candidate
        if len(best_by_point) < 8:
            return 0

        anchor_id = len(self.anchor_names)
        selected_points = sorted(best_by_point.items())
        descriptor_rows = np.asarray(
            [image_descriptors[feature_index] for _, (_, feature_index) in selected_points],
            dtype=np.float32,
        )
        point_rows = np.asarray([world_xyz[index] for index, _ in selected_points], dtype=np.float64)
        id_rows = np.asarray([point_ids[index] for index, _ in selected_points], dtype=np.int64)
        self.descriptors = np.concatenate([self.descriptors, descriptor_rows], axis=0)
        self.p3d = np.concatenate([self.p3d, point_rows], axis=0)
        self.point3d_ids = np.concatenate([self.point3d_ids, id_rows], axis=0)
        self.anchor_ids = np.concatenate(
            [self.anchor_ids, np.full(len(selected_points), anchor_id, dtype=np.int32)], axis=0
        )
        self.anchor_names.append(name)
        self._prune_online_anchors()
        return len(selected_points)

    def recover(
        self,
        *,
        gray: np.ndarray,
        K: np.ndarray,
        last_center: np.ndarray,
        max_step: float = 0.85,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        started = cv2.getTickCount()
        keypoints, query_descriptors = self.sift.detectAndCompute(np.asarray(gray), None)
        if query_descriptors is None or len(keypoints) < 8:
            return None, {"reason": "taught_recovery_query_features_missing"}

        expected = np.asarray(last_center, dtype=float).reshape(3)
        camera = np.asarray(K, dtype=float).reshape(3, 3)
        candidates: list[dict[str, Any]] = []

        for anchor_id, anchor_name in enumerate(self.anchor_names):
            rows = np.flatnonzero(self.anchor_ids == anchor_id)
            if len(rows) < 8:
                continue
            pairs = self.matcher.knnMatch(self.descriptors[rows], query_descriptors, k=2)
            matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.78 * pair[1].distance]
            selected: list[tuple[Any, int]] = []
            used_points: set[int] = set()
            used_query_features: set[int] = set()
            for match in sorted(matches, key=lambda item: float(item.distance)):
                source_row = int(rows[int(match.queryIdx)])
                point_id = int(self.point3d_ids[source_row])
                query_feature = int(match.trainIdx)
                if point_id in used_points or query_feature in used_query_features:
                    continue
                used_points.add(point_id)
                used_query_features.add(query_feature)
                selected.append((match, source_row))
            if len(selected) < 8:
                continue
            xyz = np.float32([self.p3d[row] for _, row in selected])
            xy = np.float32([keypoints[match.trainIdx].pt for match, _ in selected])
            ok, rvec, tvec, inlier_rows = cv2.solvePnPRansac(
                xyz,
                xy,
                camera,
                None,
                flags=cv2.SOLVEPNP_EPNP,
                iterationsCount=400,
                reprojectionError=8.0,
                confidence=0.999,
            )
            if not ok or inlier_rows is None or len(inlier_rows) < 8:
                continue
            rotation, _ = cv2.Rodrigues(rvec)
            translation = np.asarray(tvec, dtype=float).reshape(3)
            center = -rotation.T @ translation
            center_step = float(np.linalg.norm(center - expected))
            if not np.all(np.isfinite(center)) or center_step > float(max_step):
                continue
            inlier_indices = np.asarray(inlier_rows, dtype=int).reshape(-1)
            candidates.append(
                {
                    "anchor_name": anchor_name,
                    "inliers": int(len(inlier_indices)),
                    "center": center,
                    "center_step": center_step,
                    "R": rotation,
                    "t": translation,
                    "xy": xy[inlier_indices],
                    "p3d": xyz[inlier_indices],
                    "point3d_ids": np.asarray(
                        [self.point3d_ids[row] for _, row in selected], dtype=np.int64
                    )[inlier_indices],
                }
            )

        cluster = consensus_candidate_cluster(candidates)
        elapsed_ms = 1000.0 * (cv2.getTickCount() - started) / cv2.getTickFrequency()
        if not cluster:
            return None, {
                "reason": "taught_recovery_no_consensus",
                "candidate_count": len(candidates),
                "total_ms": elapsed_ms,
            }
        median_center = np.median(np.asarray([item["center"] for item in cluster]), axis=0)
        winner = min(
            cluster,
            key=lambda item: (
                -int(item["inliers"]),
                float(np.linalg.norm(np.asarray(item["center"]) - median_center)),
            ),
        )
        rotation = np.asarray(winner["R"], dtype=float)
        translation = np.asarray(winner["t"], dtype=float).reshape(3)
        # Consecutive taught views observe overlapping but not identical map
        # points. Merge all consensus inliers so TSolve receives a well-spread
        # 40-point case instead of a fragile 10-18 point minimal set.
        merged_xy: list[np.ndarray] = []
        merged_p3d: list[np.ndarray] = []
        merged_ids: list[int] = []
        used_points: set[int] = set()
        used_pixels: set[tuple[int, int]] = set()
        for candidate in sorted(cluster, key=lambda item: -int(item["inliers"])):
            for xy, xyz, point_id in zip(
                candidate["xy"], candidate["p3d"], candidate["point3d_ids"]
            ):
                pid = int(point_id)
                pixel = (int(round(float(xy[0]))), int(round(float(xy[1]))))
                if pid in used_points or pixel in used_pixels:
                    continue
                used_points.add(pid)
                used_pixels.add(pixel)
                merged_xy.append(np.asarray(xy, dtype=np.float32))
                merged_p3d.append(np.asarray(xyz, dtype=np.float64))
                merged_ids.append(pid)
        pool = {
            "accepted": True,
            "xy": np.asarray(merged_xy, dtype=np.float32),
            "p3d": np.asarray(merged_p3d, dtype=np.float64),
            "point3d_ids": np.asarray(merged_ids, dtype=np.int64),
            "K": camera,
            "colmap_registered_points": len(merged_ids),
            "colmap_qvec_world_to_camera": rotmat_to_qvec(rotation),
            "colmap_tvec_world_to_camera": translation.tolist(),
            "pose_prior_center": np.asarray(winner["center"], dtype=float).tolist(),
            "pose_prior_R": rotation.tolist(),
            "trusted_recovery": True,
            "recovery_max_step": float(max_step),
            "taught_anchor_name": winner["anchor_name"],
            "taught_consensus_count": len(cluster),
        }
        return pool, {
            "reason": "",
            "candidate_count": len(candidates),
            "consensus_count": len(cluster),
            "inliers": int(winner["inliers"]),
            "center_step": float(winner["center_step"]),
            "anchor_name": winner["anchor_name"],
            "total_ms": elapsed_ms,
        }


def build_bank(reference_path: Path, source_map_id: str, out_path: Path) -> dict[str, Any]:
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    replay_ids = sorted({str(leg["source_replay"]) for leg in reference.get("legs", [])})
    manifests: dict[str, dict[str, str]] = {}
    for replay_id in replay_ids:
        manifest = ROOT / "results" / "dji_live_runs" / source_map_id / replay_id / "tsolve_inputs" / "manifest.csv"
        with manifest.open(newline="", encoding="utf-8") as handle:
            manifests[replay_id] = {
                Path(row["image_name"]).name: row["case_id"] for row in csv.DictReader(handle)
            }

    sift = cv2.SIFT_create(nfeatures=3500, contrastThreshold=0.015)
    descriptors: list[np.ndarray] = []
    p3d_rows: list[np.ndarray] = []
    point_ids: list[int] = []
    anchor_ids: list[int] = []
    anchor_names: list[str] = []
    seen: set[tuple[str, str]] = set()
    for leg in reference.get("legs", []):
        replay_id = str(leg["source_replay"])
        for sample in leg.get("samples", []):
            image_name = Path(str(sample.get("image_name") or "")).name
            key = (replay_id, image_name)
            case_id = manifests.get(replay_id, {}).get(image_name)
            if not image_name or not case_id or key in seen:
                continue
            seen.add(key)
            image_path = ROOT / "viewer" / "public" / "live_dji_sessions" / f"atlas_{replay_id}" / "query_frames" / image_name
            case_dir = ROOT / "results" / "dji_live_runs" / source_map_id / replay_id / "tsolve_inputs" / "inputs" / case_id
            gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if gray is None or not case_dir.exists():
                continue
            xy = np.atleast_2d(np.loadtxt(case_dir / "p2d.csv", delimiter=","))
            xyz = np.atleast_2d(np.loadtxt(case_dir / "p3d.csv", delimiter=","))
            meta = json.loads((case_dir / "input.json").read_text(encoding="utf-8"))
            ids = np.asarray(meta.get("selected_point3d_ids") or [], dtype=np.int64)
            if len(xy) != len(xyz) or len(ids) != len(xy):
                continue
            keypoints, image_descriptors = sift.detectAndCompute(gray, None)
            if image_descriptors is None:
                continue
            feature_xy = np.float32([point.pt for point in keypoints])
            distances = np.linalg.norm(feature_xy[:, None, :] - xy[None, :, :], axis=2)
            nearest = np.argmin(distances, axis=1)
            closest = np.min(distances, axis=1)
            best_by_point: dict[int, tuple[float, int]] = {}
            for feature_index in np.flatnonzero(closest <= 5.0):
                point_index = int(nearest[feature_index])
                old = best_by_point.get(point_index)
                candidate = (float(closest[feature_index]), int(feature_index))
                if old is None or candidate < old:
                    best_by_point[point_index] = candidate
            if len(best_by_point) < 8:
                continue
            anchor_id = len(anchor_names)
            anchor_names.append(f"{replay_id}/{image_name}")
            for point_index, (_, feature_index) in sorted(best_by_point.items()):
                descriptors.append(np.asarray(image_descriptors[feature_index], dtype=np.float32))
                p3d_rows.append(np.asarray(xyz[point_index], dtype=np.float64))
                point_ids.append(int(ids[point_index]))
                anchor_ids.append(anchor_id)

    if len(anchor_names) < 3 or len(descriptors) < 24:
        raise RuntimeError("Taught patrol did not yield enough 2D->3D recovery anchors")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        descriptors=np.asarray(descriptors, dtype=np.float32),
        p3d=np.asarray(p3d_rows, dtype=np.float64),
        point3d_ids=np.asarray(point_ids, dtype=np.int64),
        anchor_ids=np.asarray(anchor_ids, dtype=np.int32),
        anchor_names=np.asarray(anchor_names),
    )
    return {
        "reference": str(reference_path),
        "source_map_id": source_map_id,
        "out": str(out_path),
        "anchors": len(anchor_names),
        "descriptors": len(descriptors),
    }


def append_run_to_bank(
    *,
    base_bank: Path,
    manifest_path: Path,
    frame_dir: Path,
    out_path: Path,
    start_index: int = 0,
    end_index: int = -1,
    stride: int = 4,
    anchor_prefix: str = "recorded",
) -> dict[str, Any]:
    """Append accepted TSolve cases from one recorded route to a bank."""
    recovery = TaughtPatrolRecovery(base_bank)
    original_anchors = len(recovery.anchor_names)
    original_descriptors = len(recovery.descriptors)
    added_anchors = 0
    added_descriptors = 0
    with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_name = Path(str(row.get("image_name") or "")).name
            digits = "".join(character for character in Path(image_name).stem if character.isdigit())
            if not image_name or not digits:
                continue
            frame_index = int(digits[-6:])
            if frame_index < max(0, int(start_index)):
                continue
            if int(end_index) >= 0 and frame_index > int(end_index):
                continue
            if frame_index % max(1, int(stride)) != 0:
                continue
            input_json = Path(manifest_path).parent / str(row.get("input_json") or "")
            case_dir = input_json.parent
            image_path = Path(frame_dir) / image_name
            if not input_json.is_file() or not image_path.is_file():
                continue
            gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            meta = json.loads(input_json.read_text(encoding="utf-8"))
            xy = np.atleast_2d(np.loadtxt(case_dir / "p2d.csv", delimiter=","))
            xyz = np.atleast_2d(np.loadtxt(case_dir / "p3d.csv", delimiter=","))
            point_ids = np.asarray(meta.get("selected_point3d_ids") or [], dtype=np.int64)
            learned = recovery.learn_anchor(
                gray=gray,
                xy=xy,
                p3d=xyz,
                point3d_ids=point_ids,
                anchor_name=f"{anchor_prefix}/{image_name}",
            )
            if learned > 0:
                added_anchors += 1
                added_descriptors += learned
    if added_anchors == 0:
        raise RuntimeError("Recorded run did not yield any new recovery anchors")
    recovery.save(out_path)
    return {
        "base_bank": str(base_bank),
        "manifest": str(manifest_path),
        "frames": str(frame_dir),
        "out": str(out_path),
        "original_anchors": original_anchors,
        "original_descriptors": original_descriptors,
        "added_anchors": added_anchors,
        "added_descriptors": added_descriptors,
        "anchors": len(recovery.anchor_names),
        "descriptors": len(recovery.descriptors),
        "start_index": int(start_index),
        "end_index": int(end_index),
        "stride": max(1, int(stride)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--source-map-id")
    parser.add_argument("--base-bank", type=Path)
    parser.add_argument("--append-manifest", type=Path)
    parser.add_argument("--append-frame-dir", type=Path)
    parser.add_argument("--append-start-index", type=int, default=0)
    parser.add_argument("--append-end-index", type=int, default=-1)
    parser.add_argument("--append-stride", type=int, default=4)
    parser.add_argument("--anchor-prefix", default="recorded")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.base_bank:
        if not args.append_manifest or not args.append_frame_dir:
            parser.error("--base-bank requires --append-manifest and --append-frame-dir")
        result = append_run_to_bank(
            base_bank=args.base_bank,
            manifest_path=args.append_manifest,
            frame_dir=args.append_frame_dir,
            out_path=args.out,
            start_index=args.append_start_index,
            end_index=args.append_end_index,
            stride=args.append_stride,
            anchor_prefix=args.anchor_prefix,
        )
    else:
        if not args.reference or not args.source_map_id:
            parser.error("building a new bank requires --reference and --source-map-id")
        result = build_bank(args.reference, args.source_map_id, args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
