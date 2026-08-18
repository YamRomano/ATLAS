#!/usr/bin/env python3
"""Persistent FARES + static-C YSolve batch experiment.

The previous drop-in experiment launched a fresh Python adapter for every
online input.  That was useful for isolation, but it hid the actual online
cost behind repeated imports, branch loading, shared-library lookup, and
process startup.

This runner keeps the FARES equation builder, the learned static-C branch, the
floating action kernel, and the C LAPACK root extractor loaded once, then loops
over all online inputs in one Python process.

Correctness is still checked against unchanged FARES/msolve, but that baseline
is timed separately and is not included in the persistent static-C online time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import ysolve_template_core as yc
from fares_static_c_replay import (
    DEFAULT_ACTION_WEIGHTS,
    direct_coefficients_c,
    eqs_from_coefficients_float,
    ensure_c_root_refiner,
    ensure_direct_coeff_builder,
    ensure_double_kernel,
    ensure_static_branch,
    float_eqs_for_runtime,
    format_action_weights,
    learn_static_branch,
    quaternions_from_numeric_action_matrices_with_fallback,
    rationalize_eqs,
    replay_fares_actions_static_c_double_coeffs,
    replay_fares_actions_static_c_double,
    select_separating_linear_form_for_actions,
)


def run(cmd: list[object]) -> None:
    cmd = [str(x) for x in cmd]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def read_csv_float(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=float)


def select_pose_by_reprojection(
    PnPSolver,
    solver,
    quaternions: list[Any],
    p2d: np.ndarray,
    *,
    pose_prior_center: Any = None,
    pose_prior_rotation: Any = None,
    pose_prior_max_step: float = 0.85,
) -> tuple[Any, Any, float, dict[str, Any]]:
    """Choose the PnP candidate using only the current 2D/3D correspondences.

    FARES' original score is a 3D ray-distance objective.  That is useful, but
    in live localization we ultimately need the pose whose projection explains
    the current image measurements.  This routine evaluates every quaternion
    candidate by:

      1. computing FARES' closed-form optimal translation for that rotation;
      2. projecting all 3D points with K[R|t];
      3. rejecting/penalizing negative depth;
      4. choosing the smallest pixel reprojection RMSE.

    This is not an oracle and does not compare against COLMAP/ground truth.  It
    only scores the same PnP input that TSolve receives online.
    """

    if quaternions is None:
        quaternions = []
    p2d = np.asarray(p2d, dtype=float)
    rays_full = solver._get_rays(p2d)
    p2d_valid = p2d[solver.valid_mask]
    rays = solver._get_rays(p2d_valid)

    W_sum = np.zeros((3, 3), dtype=float)
    for i in range(len(rays)):
        v = np.asarray(rays[i], dtype=float).reshape(3, 1)
        W_sum += solver.weights_valid[i] * (np.eye(3) - v @ v.T)
    Winv_full = np.linalg.inv(W_sum)

    best: dict[str, Any] | None = None
    candidate_scores: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    prior_center = None if pose_prior_center is None else np.asarray(pose_prior_center, dtype=float).reshape(3)
    prior_rotation = None if pose_prior_rotation is None else np.asarray(pose_prior_rotation, dtype=float).reshape(3, 3)
    for idx, q in enumerate(quaternions):
        try:
            q = np.asarray(q, dtype=float).reshape(-1)
            q_norm = float(np.linalg.norm(q))
            if not np.isfinite(q_norm) or q_norm < 1e-10:
                raise ValueError("near-zero quaternion")
            R_est = PnPSolver.quat_to_R(q / q_norm)

            rhs = np.zeros((3, 1), dtype=float)
            for i in range(len(solver.p3d_valid)):
                v = np.asarray(rays[i], dtype=float).reshape(3, 1)
                W_i = solver.weights_valid[i] * (np.eye(3) - v @ v.T)
                rhs += W_i @ (R_est @ solver.p3d_valid[i].reshape(3, 1))
            t_est = -(Winv_full @ rhs).reshape(3)

            pc = (R_est @ solver.p3d.T).T + t_est.reshape(1, 3)
            depth = pc[:, 2]
            positive = depth > 1e-8
            negative_depth_count = int(np.size(depth) - np.count_nonzero(positive))
            if not np.any(positive):
                reproj_rmse = float("inf")
                reproj_median = float("inf")
                score = float("inf")
            else:
                uvw = (solver.K @ pc[positive].T).T
                uv = uvw[:, :2] / uvw[:, 2:3]
                err = np.linalg.norm(uv - p2d[positive], axis=1)
                reproj_rmse = float(np.sqrt(np.mean(err * err)))
                reproj_median = float(np.median(err))
                # Depth violations are catastrophic for a physical camera pose.
                score = reproj_rmse + 1000.0 * negative_depth_count

            ray_objective = float(solver.calc_err(R_est, t_est, rays_full))
            camera_center = -R_est.T @ t_est
            temporal_step = (
                None if prior_center is None else float(np.linalg.norm(camera_center - prior_center))
            )
            temporal_rotation_deg = None
            if prior_rotation is not None:
                relative = R_est @ prior_rotation.T
                cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
                temporal_rotation_deg = float(np.degrees(np.arccos(cosine)))
            rec = {
                "index": idx,
                "score": score,
                "reprojection_rmse_px": reproj_rmse,
                "reprojection_median_px": reproj_median,
                "negative_depth_count": negative_depth_count,
                "positive_depth_count": int(np.count_nonzero(positive)),
                "ray_objective": ray_objective,
                "camera_center": camera_center,
                "temporal_step": temporal_step,
                "temporal_rotation_deg": temporal_rotation_deg,
                "R": R_est,
                "t": t_est,
            }
        except Exception as exc:
            rec = {
                "index": idx,
                "score": float("inf"),
                "error": repr(exc),
                "reprojection_rmse_px": float("inf"),
                "reprojection_median_px": float("inf"),
                "negative_depth_count": len(p2d),
                "positive_depth_count": 0,
                "ray_objective": float("inf"),
                "camera_center": None,
                "temporal_step": None,
                "temporal_rotation_deg": None,
                "R": None,
                "t": None,
            }
        candidate_records.append(rec)
        candidate_scores.append(
            {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in rec.items()
                if k not in {"R", "t"}
            }
        )
        if rec["R"] is not None and (best is None or rec["score"] < best["score"]):
            best = rec

    if best is None:
        R_est, t_est, ray_objective = solver._extract_best_solution(quaternions, p2d)
        return R_est, t_est, float(ray_objective), {
            "selection": "fallback_original_ray_objective",
            "candidate_scores": candidate_scores,
        }

    selection_name = "pixel_reprojection_with_positive_depth"
    reprojection_best = best
    if prior_center is not None and np.all(np.isfinite(prior_center)):
        best_rmse = float(reprojection_best["reprojection_rmse_px"])
        reprojection_limit = best_rmse + max(2.0, 0.15 * best_rmse)
        temporal_candidates = [
            rec
            for rec in candidate_records
            if rec.get("R") is not None
            and int(rec.get("negative_depth_count") or 0) == 0
            and float(rec.get("reprojection_rmse_px") or float("inf")) <= reprojection_limit
            and rec.get("temporal_step") is not None
            and float(rec["temporal_step"]) <= float(pose_prior_max_step)
        ]
        if temporal_candidates:
            best = min(
                temporal_candidates,
                key=lambda rec: (
                    float(rec["temporal_step"])
                    + 0.01 * float(rec.get("temporal_rotation_deg") or 0.0),
                    float(rec["reprojection_rmse_px"]),
                ),
            )
            selection_name = "near_best_reprojection_with_taught_pose_prior"

    return best["R"], best["t"], float(best["reprojection_rmse_px"]), {
        "selection": selection_name,
        "selected_candidate_index": int(best["index"]),
        "selected_reprojection_rmse_px": float(best["reprojection_rmse_px"]),
        "selected_reprojection_median_px": float(best["reprojection_median_px"]),
        "selected_negative_depth_count": int(best["negative_depth_count"]),
        "selected_ray_objective": float(best["ray_objective"]),
        "selected_temporal_step": best.get("temporal_step"),
        "selected_temporal_rotation_deg": best.get("temporal_rotation_deg"),
        "reprojection_best_candidate_index": int(reprojection_best["index"]),
        "candidate_scores": candidate_scores,
    }


def input_hash(input_json: Path, p3d: Path, p2d: Path) -> str:
    meta = json.loads(input_json.read_text(encoding="utf-8"))
    if meta.get("input_sha256"):
        return str(meta["input_sha256"])
    h = hashlib.sha256()
    h.update(p3d.read_bytes())
    h.update(p2d.read_bytes())
    h.update(json.dumps(meta.get("K"), sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def median(values: list[float]) -> float | None:
    return None if not values else float(statistics.median(values))


def rot_error_deg(Ra: Any, Rb: Any) -> float | None:
    if Ra is None or Rb is None:
        return None
    A = np.asarray(Ra, dtype=float)
    B = np.asarray(Rb, dtype=float)
    if A.shape != (3, 3) or B.shape != (3, 3):
        return None
    d = A @ B.T
    c = (float(np.trace(d)) - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def trans_l2(ta: Any, tb: Any) -> float | None:
    if ta is None or tb is None:
        return None
    a = np.asarray(ta, dtype=float).reshape(-1)
    b = np.asarray(tb, dtype=float).reshape(-1)
    if a.shape != b.shape:
        return None
    return float(np.linalg.norm(a - b))


def learn_one_static_branch(
    *,
    solver_dir: Path,
    yam_code_dir: Path,
    instance_dir: Path,
    branch_dir: Path,
    seed: int,
    prime: int,
    degree: int,
    action_weights: str = "branch",
):
    if str(solver_dir) not in sys.path:
        sys.path.insert(0, str(solver_dir))
    yc.add_yam_code_dir(yam_code_dir)
    from pnp_solver import PnPSolver

    meta = json.loads((instance_dir / "input.json").read_text(encoding="utf-8"))
    K = np.array(meta["K"], dtype=float)
    p3d = read_csv_float(instance_dir / "p3d.csv")
    p2d = read_csv_float(instance_dir / "p2d.csv")

    old_cwd = Path.cwd()
    try:
        os.chdir(solver_dir)
        solver = PnPSolver(K, p3d, np.ones(p3d.shape[0], dtype=float))
        equations = solver._build_equations(p2d)
    finally:
        os.chdir(old_cwd)

    dehom = rationalize_eqs(yc.dehomogenize_fares_equations(equations, mode="first3"))
    branch = learn_static_branch(
        dehom,
        seed=seed,
        p=prime,
        out_dir=branch_dir,
        index=0,
        degree=degree,
    )
    double_so = ensure_double_kernel(branch, out_dir=branch_dir)
    branch_json = branch_dir / f"fares_static_branch_{branch.index}_seed_{branch.seed}_p_{branch.p}.json"
    if not branch_json.exists():
        raise FileNotFoundError(branch_json)
    candidate_weights = DEFAULT_ACTION_WEIGHTS if str(action_weights).strip().lower() == "branch" else action_weights
    mats, _meta = replay_fares_actions_static_c_double(dehom, branch.tmpl, branch.coeff_terms, double_so)
    selection = select_separating_linear_form_for_actions(mats, candidate_weights)
    if selection.get("ok"):
        selected = selection["selected_weight"]
        selected_key = selection["selected_key"]
        all_weights = yc._yam().parse_action_weights(candidate_weights)
        ordered = [selected] + [
            w for w in all_weights
            if ",".join(f"{float(v):.17g}" for v in w) != selected_key
        ]
        branch.preferred_action_weights = format_action_weights(ordered)
    else:
        branch.preferred_action_weights = format_action_weights(candidate_weights)
    branch.learn_info["linear_form_selection_offline"] = selection
    branch.learn_info["preferred_action_weights"] = branch.preferred_action_weights
    branch_json.write_text(json.dumps(branch.to_manifest(), indent=2), encoding="utf-8")
    return branch, branch_json, double_so


def run_fares_baseline_subprocess(
    *,
    script_dir: Path,
    solver_dir: Path,
    msolve_bin: Path,
    instance_dir: Path,
    out_json: Path,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        script_dir / "optimalpnp_python_adapter.py",
        "--solver-dir",
        solver_dir,
        "--msolve-bin",
        msolve_bin,
        "--input",
        instance_dir / "input.json",
        "--p3d",
        instance_dir / "p3d.csv",
        "--p2d",
        instance_dir / "p2d.csv",
        "--out",
        out_json,
    ]
    proc = subprocess.run([str(x) for x in cmd], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(
                {
                    "success": False,
                    "error": proc.stdout[-4000:],
                    "command": [str(x) for x in cmd],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return json.loads(out_json.read_text(encoding="utf-8"))


def solve_static_persistent(
    *,
    PnPSolver,
    solver_dir: Path,
    branches: list[Any],
    double_sos: dict[int, Path],
    branch_dir: Path,
    prime: int,
    degree: int,
    fork_seed: int,
    root_refiner: Path | None,
    instance_dir: Path,
    action_weights: str,
    root_residual_tol: float,
    max_roots: int,
    fork_on_miss: bool = True,
    save_actions_path: Path | None = None,
    exact_runtime_coeffs: bool = False,
    direct_coeff_builder: Path | None = None,
    pose_prior_center: Any = None,
    pose_prior_rotation: Any = None,
    pose_prior_max_step: float = 0.85,
    root_candidate_profile: str = "full",
) -> dict[str, Any]:
    stages: dict[str, float] = {}
    t_total = time.perf_counter()
    old_cwd = Path.cwd()
    try:
        t = time.perf_counter()
        meta = json.loads((instance_dir / "input.json").read_text(encoding="utf-8"))
        K = np.array(meta["K"], dtype=float)
        p3d = read_csv_float(instance_dir / "p3d.csv")
        p2d = read_csv_float(instance_dir / "p2d.csv")
        stages["input_load_ms"] = 1000.0 * (time.perf_counter() - t)

        os.chdir(solver_dir)
        t = time.perf_counter()
        solver = PnPSolver(K, p3d, np.ones(p3d.shape[0], dtype=float))
        stages["solver_init_ms"] = 1000.0 * (time.perf_counter() - t)

        equations = None
        dehom = None
        stages["equation_build_ms"] = 0.0

        def build_python_dehom() -> list[dict[Any, Any]]:
            nonlocal equations
            if equations is None:
                t_eq = time.perf_counter()
                equations = solver._build_equations(p2d)
                stages["equation_build_ms"] += 1000.0 * (time.perf_counter() - t_eq)
            t_dehom = time.perf_counter()
            dehom_raw = yc.dehomogenize_fares_equations(equations, mode="first3")
            if exact_runtime_coeffs:
                out = rationalize_eqs(dehom_raw)
                stages["ysolve_dehomogenize_rationalize_ms"] = (
                    stages.get("ysolve_dehomogenize_rationalize_ms", 0.0)
                    + 1000.0 * (time.perf_counter() - t_dehom)
                )
                return out
            out = float_eqs_for_runtime(dehom_raw)
            stages["ysolve_dehomogenize_float_ms"] = (
                stages.get("ysolve_dehomogenize_float_ms", 0.0)
                + 1000.0 * (time.perf_counter() - t_dehom)
            )
            return out

        def coeffs_for_branch(candidate: Any) -> tuple[np.ndarray | None, list[dict[Any, Any]], dict[str, Any]]:
            if direct_coeff_builder is not None and not exact_runtime_coeffs:
                coeffs, coeff_meta = direct_coefficients_c(
                    K,
                    p3d,
                    p2d,
                    np.ones(p3d.shape[0], dtype=float),
                    candidate.coeff_terms,
                    direct_coeff_builder,
                )
                stages["ysolve_direct_c_coeff_ms"] = (
                    stages.get("ysolve_direct_c_coeff_ms", 0.0)
                    + 1000.0 * float(coeff_meta.get("direct_coeff_seconds", 0.0))
                )
                return coeffs, eqs_from_coefficients_float(coeffs, candidate.coeff_terms), coeff_meta
            nonlocal dehom
            if dehom is None:
                dehom = build_python_dehom()
            return None, dehom, {"backend": "python_fares_equations"}

        branch_errors: list[str] = []
        selected_branch: Any | None = None
        selected_double_so: Path | None = None
        mats: dict[str, np.ndarray] | None = None
        action_meta: dict[str, Any] = {}
        new_branch = False

        branch_t0 = time.perf_counter()
        for candidate in branches:
            try:
                candidate_double_so = double_sos.get(candidate.index)
                if candidate_double_so is None:
                    candidate_double_so = ensure_double_kernel(candidate, out_dir=branch_dir)
                    double_sos[candidate.index] = candidate_double_so
                coeffs, candidate_dehom, coeff_meta = coeffs_for_branch(candidate)
                if coeffs is not None:
                    mats, action_meta = replay_fares_actions_static_c_double_coeffs(
                        coeffs,
                        candidate.tmpl,
                        candidate_double_so,
                    )
                else:
                    mats, action_meta = replay_fares_actions_static_c_double(
                        candidate_dehom,
                        candidate.tmpl,
                        candidate.coeff_terms,
                        candidate_double_so,
                    )
                action_meta["coefficient_builder"] = coeff_meta
                dehom = candidate_dehom
                selected_branch = candidate
                selected_double_so = candidate_double_so
                break
            except Exception as exc:
                branch_errors.append(f"branch {candidate.index} seed {candidate.seed}: {exc}")

        if selected_branch is None:
            if not fork_on_miss:
                raise RuntimeError("no existing branch accepted this input: " + "; ".join(branch_errors))
            fork_t0 = time.perf_counter()
            dehom = build_python_dehom()
            branch_index, selected_branch, _mod_mats, fork_meta, fork_errors, new_branch = ensure_static_branch(
                dehom,
                branches=branches,
                p=prime,
                seed=fork_seed,
                out_dir=branch_dir,
                degree=degree,
                try_existing=False,
            )
            selected_double_so = ensure_double_kernel(selected_branch, out_dir=branch_dir)
            double_sos[selected_branch.index] = selected_double_so
            mats, action_meta = replay_fares_actions_static_c_double(
                dehom,
                selected_branch.tmpl,
                selected_branch.coeff_terms,
                selected_double_so,
            )
            action_meta["fork_seconds"] = time.perf_counter() - fork_t0
            action_meta["fork_branch_index"] = branch_index
            action_meta["fork_errors"] = fork_errors

        stages["ysolve_branch_select_ms"] = 1000.0 * (time.perf_counter() - branch_t0)
        stages["ysolve_static_action_double_ms"] = 1000.0 * float(action_meta.get("replay_seconds", 0.0))

        action_matrices_path = None
        if save_actions_path is not None:
            action_matrices_path = Path(save_actions_path)
            action_matrices_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                action_matrices_path,
                Ax=np.asarray(mats["x"], dtype=np.float64),
                Ay=np.asarray(mats["y"], dtype=np.float64),
                Az=np.asarray(mats["z"], dtype=np.float64),
                branch_index=np.asarray([selected_branch.index], dtype=np.int64),
                branch_seed=np.asarray([selected_branch.seed], dtype=np.int64),
            )

        if str(action_weights).strip().lower() == "branch":
            candidate_weights = (
                selected_branch.preferred_action_weights
                or selected_branch.learn_info.get("preferred_action_weights")
                or DEFAULT_ACTION_WEIGHTS
            )
        else:
            candidate_weights = action_weights
        select_t0 = time.perf_counter()
        selection = select_separating_linear_form_for_actions(mats, candidate_weights)
        stages["ysolve_linear_form_select_ms"] = 1000.0 * (time.perf_counter() - select_t0)
        if selection.get("ok"):
            selected_key = selection["selected_key"]
            all_weights = yc._yam().parse_action_weights(candidate_weights)
            ordered_weights = [selection["selected_weight"]] + [
                w for w in all_weights
                if ",".join(f"{float(v):.17g}" for v in w) != selected_key
            ]
            root_weights = format_action_weights(ordered_weights)
        else:
            root_weights = format_action_weights(candidate_weights)

        root_t0 = time.perf_counter()
        quats, roots_info = quaternions_from_numeric_action_matrices_with_fallback(
            dehom,
            mats,
            action_weights=root_weights,
            fallback_action_weights=DEFAULT_ACTION_WEIGHTS,
            residual_health_tol=root_residual_tol,
            min_root_count=len(selected_branch.tmpl.basis_cols),
            max_roots=max_roots,
            residual_tol=root_residual_tol,
            root_refine_backend="c" if root_refiner else "python",
            root_refine_lib=root_refiner,
            candidate_profile=root_candidate_profile,
        )
        stages["ysolve_static_root_total_ms"] = 1000.0 * (time.perf_counter() - root_t0)
        stages["ysolve_static_root_eig_ms"] = 1000.0 * float(roots_info.get("eig_sec") or 0.0)
        stages["ysolve_static_root_newton_ms"] = 1000.0 * float(roots_info.get("newton_sec") or 0.0)
        stages["ysolve_static_root_ctypes_overhead_ms"] = 1000.0 * float(roots_info.get("ctypes_overhead_sec") or 0.0)

        calls: list[dict[str, Any]] = []
        calls.append(
            {
                "branch_index": selected_branch.index,
                "branch_seed": selected_branch.seed,
                "branch_new": bool(new_branch),
                "branch_error_count": len(branch_errors),
                "branch_errors": branch_errors,
                "branch_count_after": len(branches),
                "action_meta": action_meta,
                "linear_form_selection": selection,
                "roots_info": roots_info,
            }
        )

        t = time.perf_counter()
        R, tv, objective, pose_selection = select_pose_by_reprojection(
            PnPSolver,
            solver,
            quats,
            p2d,
            pose_prior_center=pose_prior_center,
            pose_prior_rotation=pose_prior_rotation,
            pose_prior_max_step=pose_prior_max_step,
        )
        stages["pose_scoring_ms"] = 1000.0 * (time.perf_counter() - t)
        stages["candidate_count"] = float(len(quats))
        stages["polynomial_solver_total_ms"] = (
            stages.get("ysolve_dehomogenize_float_ms", stages.get("ysolve_dehomogenize_rationalize_ms", 0.0))
            + stages.get("ysolve_direct_c_coeff_ms", 0.0)
            + stages["ysolve_static_action_double_ms"]
            + stages["ysolve_static_root_total_ms"]
        )
        total_ms = 1000.0 * (time.perf_counter() - t_total)
        return {
            "success": R is not None and tv is not None,
            "R": None if R is None else np.asarray(R, dtype=float).tolist(),
            "t": None if tv is None else np.asarray(tv, dtype=float).reshape(-1).tolist(),
            "objective": None if objective is None else float(objective),
            "total_ms": total_ms,
            "stages_ms": stages,
            "ysolve_calls": calls,
            "roots_info": {k: v for k, v in roots_info.items() if k not in {"roots", "real_roots"}},
            "pose_selection": pose_selection,
            "quaternion_count": len(quats),
            "branch_index": selected_branch.index,
            "branch_new": bool(new_branch),
            "branch_count_after": len(branches),
            "action_matrices_path": None if action_matrices_path is None else str(action_matrices_path),
        }
    except Exception as exc:
        return {
            "success": False,
            "error": repr(exc),
            "total_ms": 1000.0 * (time.perf_counter() - t_total),
            "stages_ms": stages,
        }
    finally:
        os.chdir(old_cwd)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def summarize(rows: list[dict[str, Any]], *, train_count: int, offline_seconds: float, out: Path) -> dict[str, Any]:
    def vals(key: str) -> list[float]:
        out_vals: list[float] = []
        for row in rows:
            value = row.get(key)
            if value in (None, ""):
                continue
            try:
                out_vals.append(float(value))
            except Exception:
                pass
        return out_vals

    def count_true(key: str) -> int:
        return sum(1 for row in rows if str(row.get(key)).lower() == "true")

    summary = {
        "experiment": "fares_static_c_persistent_batch",
        "train_count": train_count,
        "eval_count": len(rows),
        "offline_template_seconds": float(offline_seconds),
        "fares_msolve_success_count": count_true("fares_msolve_success"),
        "static_persistent_success_count": count_true("static_persistent_success"),
        "strict_pose_pass_count": count_true("strict_pose_pass"),
        "strict_pose_all_pass": count_true("strict_pose_pass") == len(rows),
        "max_rotation_error_deg": max(vals("rotation_error_deg")) if vals("rotation_error_deg") else None,
        "max_translation_l2": max(vals("translation_l2")) if vals("translation_l2") else None,
        "median_fares_total_ms": median(vals("fares_msolve_total_ms")),
        "median_static_persistent_total_ms": median(vals("static_persistent_total_ms")),
        "median_fares_polynomial_ms": median(vals("fares_msolve_stage_polynomial_solver_total_ms")),
        "median_static_polynomial_ms": median(vals("static_stage_polynomial_solver_total_ms")),
        "median_fares_msolve_subprocess_ms": median(vals("fares_msolve_stage_msolve_subprocess_ms")),
        "median_static_dehom_ms": median(
            vals("static_stage_ysolve_dehomogenize_float_ms")
            or vals("static_stage_ysolve_dehomogenize_rationalize_ms")
        ),
        "median_static_action_ms": median(vals("static_stage_ysolve_static_action_double_ms")),
        "median_static_root_ms": median(vals("static_stage_ysolve_static_root_total_ms")),
        "median_static_equation_build_ms": median(vals("static_stage_equation_build_ms")),
        "speedup_fares_subprocess_over_static_action_median": (
            median(vals("fares_msolve_stage_msolve_subprocess_ms")) / median(vals("static_stage_ysolve_static_action_double_ms"))
            if median(vals("fares_msolve_stage_msolve_subprocess_ms")) is not None
            and median(vals("static_stage_ysolve_static_action_double_ms"))
            else None
        ),
        "speedup_fares_polynomial_over_static_polynomial_median": (
            median(vals("fares_msolve_stage_polynomial_solver_total_ms")) / median(vals("static_stage_polynomial_solver_total_ms"))
            if median(vals("fares_msolve_stage_polynomial_solver_total_ms")) is not None
            and median(vals("static_stage_polynomial_solver_total_ms"))
            else None
        ),
        "speedup_fares_total_over_static_persistent_total_median": (
            median(vals("fares_msolve_total_ms")) / median(vals("static_persistent_total_ms"))
            if median(vals("fares_msolve_total_ms")) is not None and median(vals("static_persistent_total_ms"))
            else None
        ),
        "total_fares_total_ms": float(sum(vals("fares_msolve_total_ms"))),
        "total_static_persistent_total_ms": float(sum(vals("static_persistent_total_ms"))),
        "total_fares_polynomial_ms": float(sum(vals("fares_msolve_stage_polynomial_solver_total_ms"))),
        "total_static_polynomial_ms": float(sum(vals("static_stage_polynomial_solver_total_ms"))),
    }
    (out / "persistent_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver-dir", required=True)
    ap.add_argument("--yam-code-dir", required=True)
    ap.add_argument("--msolve-bin", required=True)
    ap.add_argument("--out", default="results/fares_static_c_persistent_batch_train1_eval99")
    ap.add_argument("--count", type=int, default=11)
    ap.add_argument("--train-count", type=int, default=1)
    ap.add_argument("--points", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20240518)
    ap.add_argument("--noise-px", type=float, default=0.0)
    ap.add_argument("--prime", type=int, default=2147483647)
    ap.add_argument("--degree", type=int, default=11)
    ap.add_argument("--root-residual-tol", type=float, default=1e-8)
    ap.add_argument("--max-roots", type=int, default=80)
    ap.add_argument("--action-weights", default="branch")
    ap.add_argument(
        "--no-online-fork",
        action="store_true",
        help="Disable online branch creation. Existing branches are still tried in order.",
    )
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--rot-tol-deg", type=float, default=1e-5)
    ap.add_argument("--trans-tol", type=float, default=1e-5)
    args = ap.parse_args()

    if args.train_count < 1 or args.train_count >= args.count:
        raise ValueError("--train-count must be in [1, count-1]")

    script_dir = Path(__file__).resolve().parent
    solver_dir = Path(args.solver_dir).resolve()
    yam_code_dir = Path(args.yam_code_dir).resolve()
    msolve_bin = Path(args.msolve_bin).resolve()
    out = Path(args.out).resolve()
    all_instances = out / "instances_all"
    branch_dir = out / "offline_branch"
    baseline_dir = out / "baseline_fares_msolve_json"
    static_dir = out / "persistent_static_json"

    out.mkdir(parents=True, exist_ok=True)
    print("\n=== Generate FARES-format PnP inputs ===")
    run(
        [
            sys.executable,
            script_dir / "generate_live_pnp_instances.py",
            "--out",
            all_instances,
            "--count",
            args.count,
            "--points",
            args.points,
            "--seed",
            args.seed,
            "--noise-px",
            args.noise_px,
        ]
    )

    print("\n=== Offline: learn static-C branch once ===")
    t0 = time.perf_counter()
    branch, branch_json, double_so = learn_one_static_branch(
        solver_dir=solver_dir,
        yam_code_dir=yam_code_dir,
        instance_dir=all_instances / "instance_000",
        branch_dir=branch_dir,
        seed=args.seed,
        prime=args.prime,
        degree=args.degree,
        action_weights=args.action_weights,
    )
    offline_seconds = time.perf_counter() - t0
    learned_rank = getattr(branch.tmpl, "rank", None)
    if learned_rank is None:
        learned_rank = len(getattr(branch.tmpl, "pivot_cols", []))
    print(
        f"branch={branch.index} rank={learned_rank} quotient={len(branch.tmpl.basis_cols)} "
        f"offline={offline_seconds:.3f}s"
    )
    print("branch json:", branch_json)
    print("double so:", double_so)
    branches: list[Any] = [branch]
    double_sos: dict[int, Path] = {branch.index: double_so}

    print("\n=== Build/load C root extractor once ===")
    root_refiner = ensure_c_root_refiner(yam_code_dir=yam_code_dir, out_dir=branch_dir, require_lapack=True)
    print("root refiner:", root_refiner)

    print("\n=== Build/load direct C coefficient builder once ===")
    direct_coeff_builder = ensure_direct_coeff_builder(yam_code_dir=yam_code_dir, out_dir=branch_dir)
    print("direct coeff builder:", direct_coeff_builder)

    print("\n=== Import FARES solver once for persistent static path ===")
    if str(solver_dir) not in sys.path:
        sys.path.insert(0, str(solver_dir))
    yc.add_yam_code_dir(yam_code_dir)
    from pnp_solver import PnPSolver

    rows: list[dict[str, Any]] = []
    for idx in range(args.train_count, args.count):
        instance_id = f"instance_{idx:03d}"
        instance_dir = all_instances / instance_id
        print(f"\n=== Eval {instance_id} ===", flush=True)

        baseline: dict[str, Any] = {}
        if not args.skip_baseline:
            baseline = run_fares_baseline_subprocess(
                script_dir=script_dir,
                solver_dir=solver_dir,
                msolve_bin=msolve_bin,
                instance_dir=instance_dir,
                out_json=baseline_dir / f"{instance_id}.json",
            )

        static = solve_static_persistent(
            PnPSolver=PnPSolver,
            solver_dir=solver_dir,
            branches=branches,
            double_sos=double_sos,
            branch_dir=branch_dir,
            prime=args.prime,
            degree=args.degree,
            fork_seed=args.seed + idx,
            root_refiner=root_refiner,
            instance_dir=instance_dir,
            action_weights=args.action_weights,
            root_residual_tol=args.root_residual_tol,
            max_roots=args.max_roots,
            fork_on_miss=not args.no_online_fork,
            direct_coeff_builder=direct_coeff_builder,
        )
        static_dir.mkdir(parents=True, exist_ok=True)
        (static_dir / f"{instance_id}.json").write_text(json.dumps(static, indent=2, default=str), encoding="utf-8")

        rerr = rot_error_deg(static.get("R"), baseline.get("R")) if baseline else None
        terr = trans_l2(static.get("t"), baseline.get("t")) if baseline else None
        strict_pass = (
            bool(static.get("success"))
            and (not baseline or bool(baseline.get("success")))
            and (rerr is None or rerr <= args.rot_tol_deg)
            and (terr is None or terr <= args.trans_tol)
        )

        row: dict[str, Any] = {
            "instance_id": instance_id,
            "input_sha256": input_hash(instance_dir / "input.json", instance_dir / "p3d.csv", instance_dir / "p2d.csv"),
            "fares_msolve_success": baseline.get("success") if baseline else "",
            "static_persistent_success": static.get("success"),
            "strict_pose_pass": strict_pass,
            "rotation_error_deg": rerr,
            "translation_l2": terr,
            "objective_gap_static_minus_fares": (
                float(static["objective"]) - float(baseline["objective"])
                if static.get("objective") is not None and baseline.get("objective") is not None
                else None
            ),
            "fares_msolve_total_ms": baseline.get("total_ms", ""),
            "static_persistent_total_ms": static.get("total_ms", ""),
            "static_quaternion_count": static.get("quaternion_count", ""),
            "static_branch_index": static.get("branch_index", ""),
            "static_branch_new": static.get("branch_new", ""),
            "static_branch_count_after": static.get("branch_count_after", ""),
        }
        for k, v in (baseline.get("stages_ms") or {}).items():
            row[f"fares_msolve_stage_{k}"] = v
        for k, v in (static.get("stages_ms") or {}).items():
            row[f"static_stage_{k}"] = v
        rows.append(row)
        print(
            f"baseline={baseline.get('total_ms', '')} ms  static={static.get('total_ms', '')} ms  "
            f"action={(static.get('stages_ms') or {}).get('ysolve_static_action_double_ms')} ms  "
            f"root={(static.get('stages_ms') or {}).get('ysolve_static_root_total_ms')} ms  pass={strict_pass}",
            flush=True,
        )

    csv_path = out / "persistent_fares_msolve_vs_static_c.csv"
    write_csv(rows, csv_path)
    summary = summarize(rows, train_count=args.train_count, offline_seconds=offline_seconds, out=out)
    summary["online_branch_count_final"] = len(branches)
    summary["online_new_branch_count"] = sum(1 for row in rows if str(row.get("static_branch_new")).lower() == "true")
    (out / "persistent_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\nOutputs:")
    print("csv:", csv_path)
    print("summary:", out / "persistent_summary.json")
    print("baseline json dir:", baseline_dir)
    print("static json dir:", static_dir)


if __name__ == "__main__":
    main()
