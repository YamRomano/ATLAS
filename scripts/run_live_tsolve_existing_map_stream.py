#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from colmap_io import camera_center, qvec_to_rotmat, read_cameras_model, read_images_model, read_images_text, read_points3d_text


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def run(cmd: list[object]) -> None:
    cmd = [str(x) for x in cmd]
    print("+", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "minimal")
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        check=False,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.perf_counter() - t0
    print(f"  done in {elapsed:.2f}s", flush=True)
    if proc.returncode != 0:
        print(proc.stdout[-6000:], flush=True)
        proc.check_returncode()


def image_files(path: Path) -> list[Path]:
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_image_root(map_images: Path, all_images: Path) -> Path:
    all_images.mkdir(parents=True, exist_ok=True)
    for src in image_files(map_images):
        link_or_copy(src, all_images / src.name)
    query_root = all_images / "query"
    query_root.mkdir(exist_ok=True)
    return query_root


def read_frame_times(frames_csv: Path) -> dict[str, dict[str, str]]:
    if not frames_csv.exists():
        return {}
    with frames_csv.open(newline="", encoding="utf-8") as f:
        return {row["image_name"]: row for row in csv.DictReader(f)}


def farthest_spread_indices(xy: np.ndarray, count: int) -> np.ndarray:
    if len(xy) <= count:
        return np.arange(len(xy), dtype=int)
    center = xy.mean(axis=0)
    first = int(np.argmax(np.linalg.norm(xy - center, axis=1)))
    chosen = [first]
    dist = np.linalg.norm(xy - xy[first], axis=1)
    while len(chosen) < count:
        idx = int(np.argmax(dist))
        chosen.append(idx)
        dist = np.minimum(dist, np.linalg.norm(xy - xy[idx], axis=1))
    return np.array(chosen, dtype=int)


def sha256_case(K: np.ndarray, p3d: np.ndarray, p2d: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(K, dtype=np.float64).tobytes())
    h.update(np.asarray(p3d, dtype=np.float64).tobytes())
    h.update(np.asarray(p2d, dtype=np.float64).tobytes())
    return h.hexdigest()


def write_manifest(inputs_out: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "experiment",
        "case_id",
        "p3d_csv",
        "p2d_csv",
        "input_json",
        "points",
        "image_name",
        "time_sec",
        "localization_attempt",
    ]
    with (inputs_out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class ReferenceSelector:
    """Small ORB-SLAM-style keyframe selector for repeated query localization.

    The first query needs a global relocalization set.  After one frame is
    accepted, the next frame is usually nearby, so matching against nearby map
    cameras gives the same 2D/3D path into TSolve without redoing a broad
    COLMAP-style query against many reference images.
    """

    def __init__(self, map_sparse_text: Path, bootstrap_count: int, tracking_count: int):
        self.images_by_points = [
            im for im in read_images_text(map_sparse_text / "images.txt").values()
            if not im.name.startswith("query/")
        ]
        self.images_by_points.sort(key=lambda im: int(np.sum(im.point3d_ids >= 0)), reverse=True)
        self.bootstrap_count = max(1, int(bootstrap_count))
        self.tracking_count = max(1, int(tracking_count))
        self.centers: dict[str, np.ndarray] = {}
        self.heading_bins: dict[str, int] = {}
        for im in self.images_by_points:
            try:
                self.centers[im.name] = camera_center(im)
            except Exception:
                continue
            try:
                # COLMAP cameras look along +Z.  Keep only a coarse horizontal
                # orientation bin: global recovery needs nearby keyframes that
                # look in different directions around an in-place patrol turn,
                # not ten nearly identical cameras chosen by distance alone.
                forward = qvec_to_rotmat(im.qvec).T @ np.array(
                    [0.0, 0.0, 1.0],
                    dtype=float,
                )
                if float(np.linalg.norm(forward[[0, 2]])) > 1e-9:
                    angle = float(np.arctan2(forward[0], forward[2]))
                    self.heading_bins[im.name] = int(
                        np.floor(((angle + np.pi) % (2.0 * np.pi)) / (np.pi / 6.0))
                    )
            except Exception:
                continue
        self.images_by_spread = self._spatial_spread()

    def _spatial_spread(self) -> list[str]:
        if not self.centers:
            return []
        names = list(self.centers)
        xyz = np.asarray([self.centers[name] for name in names], dtype=float)
        xy = xyz[:, [0, 2]] if xyz.shape[1] >= 3 else xyz[:, :2]
        idx = farthest_spread_indices(xy, min(len(names), self.bootstrap_count))
        return [names[int(i)] for i in idx]

    def bootstrap(self) -> list[str]:
        selected: list[str] = []
        top_count = max(8, self.bootstrap_count // 2)
        for im in self.images_by_points[:top_count]:
            if im.name not in selected:
                selected.append(im.name)
        for name in self.images_by_spread:
            if name not in selected:
                selected.append(name)
            if len(selected) >= self.bootstrap_count:
                break
        for im in self.images_by_points:
            if len(selected) >= self.bootstrap_count:
                break
            if im.name not in selected:
                selected.append(im.name)
        return selected

    def near(self, center: np.ndarray | None) -> list[str]:
        if center is None or not self.centers:
            return self.bootstrap()
        ranked = sorted(
            self.centers.items(),
            key=lambda kv: float(np.linalg.norm(kv[1] - center)),
        )
        global_reserve = min(
            self.tracking_count - 1,
            max(3, self.tracking_count // 4),
        )
        local_budget = max(1, self.tracking_count - global_reserve)
        candidate_count = min(
            len(ranked),
            max(64, self.tracking_count * 8),
        )
        candidates = ranked[:candidate_count]

        # Preserve the closest cameras, then spend the rest of the local
        # budget on different viewing directions from the same neighborhood.
        # This is critical at point 3: position is unchanged while the camera
        # turns roughly 90 degrees toward point 4.
        nearest_count = min(local_budget, max(3, local_budget // 3))
        local = [name for name, _ in candidates[:nearest_count]]
        used_heading_bins = {
            self.heading_bins[name]
            for name in local
            if name in self.heading_bins
        }
        for name, _camera_center in candidates:
            if len(local) >= local_budget:
                break
            heading_bin = self.heading_bins.get(name)
            if (
                name not in local
                and heading_bin is not None
                and heading_bin not in used_heading_bins
            ):
                local.append(name)
                used_heading_bins.add(heading_bin)
        for name, _camera_center in candidates:
            if len(local) >= local_budget:
                break
            if name not in local:
                local.append(name)

        # Keep a few globally strong keyframes as a cheap recovery net while
        # respecting the configured total reference-image cap.
        for name in self.bootstrap():
            if len(local) >= self.tracking_count:
                break
            if name not in local:
                local.append(name)
        return local


def center_from_meta(meta: dict[str, Any]) -> np.ndarray | None:
    qvec = meta.get("colmap_qvec_world_to_camera")
    tvec = meta.get("colmap_tvec_world_to_camera")
    if qvec is None or tvec is None:
        return None
    try:
        R = qvec_to_rotmat(np.asarray(qvec, dtype=float))
        t = np.asarray(tvec, dtype=float).reshape(3)
        return -R.T @ t
    except Exception:
        return None


def export_one_case(
    *,
    localized_model: Path,
    map_points: dict[int, Any],
    frame_times: dict[str, dict[str, str]],
    image_name: str,
    case_id: str,
    inputs_out: Path,
    min_points: int,
    max_points: int,
) -> dict[str, Any]:
    cameras = read_cameras_model(localized_model)
    images = read_images_model(localized_model)
    image = next((im for im in images.values() if im.name == image_name), None)
    if image is None:
        return {"accepted": False, "reason": "query_not_registered", "image_name": image_name}

    valid = image.point3d_ids >= 0
    valid &= np.array([int(pid) in map_points for pid in image.point3d_ids], dtype=bool)
    valid_idx = np.where(valid)[0]
    if len(valid_idx) < min_points:
        return {
            "accepted": False,
            "reason": "too_few_correspondences",
            "image_name": image_name,
            "valid_2d3d": int(len(valid_idx)),
        }

    xy_all = image.xys[valid_idx]
    chosen_local = farthest_spread_indices(xy_all, min(max_points, len(valid_idx)))
    chosen_idx = valid_idx[chosen_local]

    p2d = image.xys[chosen_idx].astype(float)
    p3d = np.asarray([map_points[int(pid)].xyz for pid in image.point3d_ids[chosen_idx]], dtype=float)
    K = cameras[image.camera_id].K()

    raw_name = image.name.split("/", 1)[-1]
    frame_row = frame_times.get(raw_name, {})
    case_dir = inputs_out / "inputs" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(case_dir / "p3d.csv", p3d, delimiter=",", fmt="%.12g")
    np.savetxt(case_dir / "p2d.csv", p2d, delimiter=",", fmt="%.12g")

    meta = {
        "K": K.tolist(),
        "image_name": image.name,
        "source_frame": frame_row.get("source_frame"),
        "time_sec": float(frame_row["time_sec"]) if frame_row.get("time_sec") else None,
        "points": int(len(chosen_idx)),
        "input_sha256": sha256_case(K, p3d, p2d),
        "colmap_image_id": image.image_id,
        "colmap_camera_id": image.camera_id,
        "colmap_registered_points": int(np.sum(image.point3d_ids >= 0)),
        "colmap_qvec_world_to_camera": image.qvec.tolist(),
        "colmap_tvec_world_to_camera": image.tvec.tolist(),
    }
    (case_dir / "input.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {
        "accepted": True,
        "case_id": case_id,
        "case_dir": case_dir,
        "points": int(len(chosen_idx)),
        "image_name": image.name,
        "time_sec": meta["time_sec"],
    }


def copy_case_to_instance(case_dir: Path, instance_dir: Path) -> None:
    if instance_dir.exists():
        shutil.rmtree(instance_dir)
    instance_dir.mkdir(parents=True, exist_ok=True)
    for name in ("input.json", "p3d.csv", "p2d.csv"):
        shutil.copy2(case_dir / name, instance_dir / name)


def import_runtime(runtime_dir: Path, solver_dir: Path):
    harness = runtime_dir / "harness"
    yam_code = runtime_dir / "yam_code"
    if str(harness) not in sys.path:
        sys.path.insert(0, str(harness))
    if str(solver_dir) not in sys.path:
        sys.path.insert(0, str(solver_dir))

    from run_fares_static_c_persistent_batch import (  # type: ignore
        ensure_c_root_refiner,
        ensure_direct_coeff_builder,
        learn_one_static_branch,
        solve_static_persistent,
    )

    import ysolve_template_core as yc  # type: ignore

    yc.add_yam_code_dir(yam_code)
    from pnp_solver import PnPSolver  # type: ignore

    return {
        "yam_code": yam_code,
        "PnPSolver": PnPSolver,
        "ensure_c_root_refiner": ensure_c_root_refiner,
        "ensure_direct_coeff_builder": ensure_direct_coeff_builder,
        "learn_one_static_branch": learn_one_static_branch,
        "solve_static_persistent": solve_static_persistent,
    }


def solve_case(
    *,
    runtime_api: dict[str, Any],
    solver_dir: Path,
    out_dir: Path,
    branch_dir: Path,
    branches: list[Any],
    double_sos: dict[int, Path],
    root_refiner: Any,
    direct_coeff_builder: Any,
    instance_dir: Path,
    instance_id: str,
    prime: int,
    degree: int,
    action_weights: str,
    fallback_action_weights: str,
    fork_seed: int,
    fork_on_miss: bool = True,
    root_candidate_profile: str = "full",
) -> dict[str, Any]:
    solve_static_persistent = runtime_api["solve_static_persistent"]
    instance_meta = json.loads((instance_dir / "input.json").read_text(encoding="utf-8"))
    pose_prior_kwargs: dict[str, Any] = {}
    if instance_meta.get("pose_prior_center") is not None:
        pose_prior_kwargs = {
            "pose_prior_center": instance_meta.get("pose_prior_center"),
            "pose_prior_rotation": instance_meta.get("pose_prior_R"),
            "pose_prior_max_step": float(instance_meta.get("recovery_max_step") or 0.85),
        }
    result = solve_static_persistent(
        PnPSolver=runtime_api["PnPSolver"],
        solver_dir=solver_dir,
        branches=branches,
        double_sos=double_sos,
        branch_dir=branch_dir,
        prime=prime,
        degree=degree,
        fork_seed=fork_seed,
        root_refiner=root_refiner,
        instance_dir=instance_dir,
        action_weights=action_weights,
        root_residual_tol=1e-8,
        max_roots=80,
        fork_on_miss=bool(fork_on_miss),
        direct_coeff_builder=direct_coeff_builder,
        root_candidate_profile=root_candidate_profile,
        **pose_prior_kwargs,
    )
    fast_result = result if root_candidate_profile == "live_fast" else None
    fast_objective = fast_result.get("objective") if fast_result else None
    fast_acceptable = bool(
        fast_result
        and fast_result.get("success")
        and fast_objective is not None
        and float(fast_objective) <= 26.0
    )
    if root_candidate_profile == "live_fast" and not fast_acceptable:
        result = solve_static_persistent(
            PnPSolver=runtime_api["PnPSolver"],
            solver_dir=solver_dir,
            branches=branches,
            double_sos=double_sos,
            branch_dir=branch_dir,
            prime=prime,
            degree=degree,
            fork_seed=fork_seed,
            root_refiner=root_refiner,
            instance_dir=instance_dir,
            action_weights=action_weights,
            root_residual_tol=1e-8,
            max_roots=80,
            fork_on_miss=bool(fork_on_miss),
            direct_coeff_builder=direct_coeff_builder,
            root_candidate_profile="full",
            **pose_prior_kwargs,
        )
        result["live_fast_fallback_used"] = True
        result["live_fast_result"] = {
            "success": bool(fast_result.get("success")),
            "objective": fast_objective,
            "total_ms": fast_result.get("total_ms"),
        }
    elif root_candidate_profile == "live_fast":
        result["live_fast_fallback_used"] = False
    if not result.get("success") and fallback_action_weights:
        fallback = solve_static_persistent(
            PnPSolver=runtime_api["PnPSolver"],
            solver_dir=solver_dir,
            branches=branches,
            double_sos=double_sos,
            branch_dir=branch_dir,
            prime=prime,
            degree=degree,
            fork_seed=30300629 + fork_seed,
            root_refiner=root_refiner,
            instance_dir=instance_dir,
            action_weights=fallback_action_weights,
            root_residual_tol=1e-8,
            max_roots=80,
            fork_on_miss=bool(fork_on_miss),
            direct_coeff_builder=direct_coeff_builder,
            root_candidate_profile=(
                "full" if root_candidate_profile == "live_fast" else root_candidate_profile
            ),
            **pose_prior_kwargs,
        )
        fallback["fallback_used"] = True
        if fallback.get("success"):
            result = fallback
    static_dir = out_dir / "persistent_static_json"
    static_dir.mkdir(parents=True, exist_ok=True)
    (static_dir / f"{instance_id}.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--colmap", required=True, type=Path)
    ap.add_argument("--map-database", required=True, type=Path)
    ap.add_argument("--map-images", required=True, type=Path)
    ap.add_argument("--map-sparse-model", required=True, type=Path)
    ap.add_argument("--map-sparse-text", required=True, type=Path)
    ap.add_argument("--query-frames", required=True, type=Path)
    ap.add_argument("--runtime-dir", required=True, type=Path)
    ap.add_argument("--solver-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--inputs-out-dir", required=True, type=Path)
    ap.add_argument("--work-dir", required=True, type=Path)
    ap.add_argument("--max-image-size", type=int, default=1200)
    ap.add_argument("--query-camera-model", default="SIMPLE_RADIAL")
    ap.add_argument("--min-points", type=int, default=40)
    ap.add_argument("--max-points", type=int, default=40)
    ap.add_argument("--max-reference-images", type=int, default=80)
    ap.add_argument("--tracking-reference-images", type=int, default=10)
    ap.add_argument("--prime", type=int, default=2147483647)
    ap.add_argument("--degree", type=int, default=11)
    ap.add_argument("--action-weights", default="branch")
    ap.add_argument("--fallback-action-weights", default="")
    ap.add_argument("--scene-json", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--display-z-sign", type=float, default=-1.0, help=argparse.SUPPRESS)
    ap.add_argument("--room-alignment-json", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if not args.colmap.exists():
        raise FileNotFoundError(args.colmap)
    if not args.map_database.exists():
        raise FileNotFoundError(args.map_database)
    if not (args.map_sparse_model / "images.bin").exists():
        raise FileNotFoundError(args.map_sparse_model / "images.bin")
    if not (args.map_sparse_text / "points3D.txt").exists():
        raise FileNotFoundError(args.map_sparse_text / "points3D.txt")

    frames = image_files(args.query_frames)
    if not frames:
        raise RuntimeError(f"No query frames in {args.query_frames}")

    for path in (args.out_dir, args.inputs_out_dir, args.work_dir):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    frame_times = read_frame_times(args.query_frames / "frames.csv")
    map_points = read_points3d_text(args.map_sparse_text / "points3D.txt")
    references = ReferenceSelector(
        args.map_sparse_text,
        bootstrap_count=args.max_reference_images,
        tracking_count=args.tracking_reference_images,
    )
    if not references.bootstrap():
        raise RuntimeError("No reference map images are available for live localization.")
    print(
        "Reference selector:",
        json.dumps(
            {
                "bootstrap_reference_images": len(references.bootstrap()),
                "tracking_reference_images": args.tracking_reference_images,
                "mode": "first frame bootstrap, then nearest map cameras to last accepted pose",
            }
        ),
        flush=True,
    )
    all_images = args.work_dir / "all_images"
    query_root = prepare_image_root(args.map_images, all_images)
    db = args.work_dir / "live_incremental.db"
    shutil.copy2(args.map_database, db)

    runtime_api = import_runtime(args.runtime_dir, args.solver_dir.resolve())
    branch_dir = args.out_dir / "offline_branch"
    instances_dir = args.out_dir / "instances_all"
    branch_dir.mkdir(parents=True, exist_ok=True)
    instances_dir.mkdir(parents=True, exist_ok=True)

    print("=== Live TSolve runtime setup ===", flush=True)
    root_refiner = runtime_api["ensure_c_root_refiner"](
        yam_code_dir=runtime_api["yam_code"],
        out_dir=branch_dir,
        require_lapack=True,
    )
    direct_coeff_builder = runtime_api["ensure_direct_coeff_builder"](
        yam_code_dir=runtime_api["yam_code"],
        out_dir=branch_dir,
    )
    branches: list[Any] = []
    double_sos: dict[int, Path] = {}
    manifest_rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    last_center: np.ndarray | None = None

    localized = args.work_dir / "localized_model"
    image_list = args.work_dir / "current_query_image.txt"
    pair_list = args.work_dir / "current_query_pairs.txt"

    for frame_idx, frame in enumerate(frames):
        query_name = f"query/{frame.name}"
        print(f"\n=== STREAM FRAME {frame_idx + 1}/{len(frames)}: {query_name} ===", flush=True)
        shutil.copy2(frame, query_root / frame.name)
        image_list.write_text(query_name + "\n", encoding="utf-8")

        run(
            [
                args.colmap,
                "feature_extractor",
                "--database_path",
                db,
                "--image_path",
                all_images,
                "--image_list_path",
                image_list,
                "--ImageReader.camera_model",
                args.query_camera_model,
                "--ImageReader.single_camera_per_folder",
                "1",
                "--SiftExtraction.max_image_size",
                args.max_image_size,
                "--SiftExtraction.use_gpu",
                "0",
            ]
        )
        case_id = f"instance_{len(manifest_rows):03d}"
        attempts = [
            ("tracking", references.near(last_center)),
        ]
        bootstrap_names = references.bootstrap()
        if attempts[0][1] != bootstrap_names:
            attempts.append(("bootstrap", bootstrap_names))

        export: dict[str, Any] | None = None
        chosen_attempt = None
        for attempt_name, reference_names in attempts:
            print(
                f"matching {query_name} against {len(reference_names)} {attempt_name} reference images",
                flush=True,
            )
            pair_list.write_text(
                "".join(f"{query_name} {ref_name}\n" for ref_name in reference_names),
                encoding="utf-8",
            )
            run(
                [
                    args.colmap,
                    "matches_importer",
                    "--database_path",
                    db,
                    "--match_list_path",
                    pair_list,
                    "--match_type",
                    "pairs",
                    "--SiftMatching.guided_matching",
                    "1",
                    "--SiftMatching.use_gpu",
                    "0",
                ]
            )
            if localized.exists():
                shutil.rmtree(localized)
            localized.mkdir()
            run(
                [
                    args.colmap,
                    "image_registrator",
                    "--database_path",
                    db,
                    "--input_path",
                    args.map_sparse_model,
                    "--output_path",
                    localized,
                    "--Mapper.abs_pose_min_num_inliers",
                    "15",
                    "--Mapper.abs_pose_min_inlier_ratio",
                    "0.10",
                ]
            )

            export = export_one_case(
                localized_model=localized,
                map_points=map_points,
                frame_times=frame_times,
                image_name=query_name,
                case_id=case_id,
                inputs_out=args.inputs_out_dir,
                min_points=args.min_points,
                max_points=args.max_points,
            )
            chosen_attempt = attempt_name
            if export.get("accepted"):
                break
            print("localization attempt rejected:", json.dumps(export), flush=True)

        assert export is not None
        if not export.get("accepted"):
            rejected.append(export)
            print("FRAME SKIPPED:", json.dumps(export), flush=True)
            continue

        meta = json.loads((Path(export["case_dir"]) / "input.json").read_text(encoding="utf-8"))
        accepted_center = center_from_meta(meta)
        if accepted_center is not None:
            last_center = accepted_center

        manifest_rows.append(
            {
                "experiment": "live_existing_map_tsolve_stream",
                "case_id": case_id,
                "p3d_csv": f"inputs/{case_id}/p3d.csv",
                "p2d_csv": f"inputs/{case_id}/p2d.csv",
                "input_json": f"inputs/{case_id}/input.json",
                "points": int(export["points"]),
                "image_name": export["image_name"],
                "time_sec": export["time_sec"],
                "localization_attempt": chosen_attempt,
            }
        )
        write_manifest(args.inputs_out_dir, manifest_rows)
        instance_dir = instances_dir / case_id
        copy_case_to_instance(Path(export["case_dir"]), instance_dir)

        if not branches:
            print("Learning static-C branch from first accepted live frame.", flush=True)
            t0 = time.perf_counter()
            branch, branch_json, double_so = runtime_api["learn_one_static_branch"](
                solver_dir=args.solver_dir.resolve(),
                yam_code_dir=runtime_api["yam_code"],
                instance_dir=instance_dir,
                branch_dir=branch_dir,
                seed=20260707,
                prime=args.prime,
                degree=args.degree,
                action_weights=args.action_weights,
            )
            print(f"learned branch={branch.index} in {time.perf_counter() - t0:.3f}s", flush=True)
            print("branch json:", branch_json, flush=True)
            branches.append(branch)
            double_sos[branch.index] = double_so

        solve_t0 = time.perf_counter()
        result = solve_case(
            runtime_api=runtime_api,
            solver_dir=args.solver_dir.resolve(),
            out_dir=args.out_dir,
            branch_dir=branch_dir,
            branches=branches,
            double_sos=double_sos,
            root_refiner=root_refiner,
            direct_coeff_builder=direct_coeff_builder,
            instance_dir=instance_dir,
            instance_id=case_id,
            prime=args.prime,
            degree=args.degree,
            action_weights=args.action_weights,
            fallback_action_weights=args.fallback_action_weights,
            fork_seed=20260707 + frame_idx,
        )
        stages = result.get("stages_ms") or {}
        print(
            "POSE APPENDED:",
            json.dumps(
                {
                    "case_id": case_id,
                    "success": bool(result.get("success")),
                    "solve_ms": round((time.perf_counter() - solve_t0) * 1000.0, 2),
                    "total_ms": result.get("total_ms"),
                    "action_ms": stages.get("ysolve_static_action_double_ms"),
                    "root_ms": stages.get("ysolve_static_root_total_ms"),
                    "branch": result.get("branch_index"),
                    "new_branch": result.get("branch_new"),
                }
            ),
            flush=True,
        )

    summary = {
        "query_frames": len(frames),
        "accepted_cases": len(manifest_rows),
        "rejected_cases": len(rejected),
        "online_branch_count_final": len(branches),
        "inputs_out_dir": str(args.inputs_out_dir),
        "out_dir": str(args.out_dir),
        "rejected": rejected[:80],
    }
    (args.out_dir / "live_stream_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (args.inputs_out_dir / "export_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("\n=== LIVE STREAM SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
