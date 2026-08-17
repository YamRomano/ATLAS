#!/usr/bin/env python3
"""FARES end-to-end adapter using static-C Ysolve action matrices.

This keeps the downloaded OptimalPnP/FARES pipeline intact except for the
polynomial solver call.  Instead of calling msolve, it:

1. dehomogenizes the first three FARES quartics in the q0=1 chart,
2. replays the learned FARES action-template in a generated static-C kernel,
3. eigensolves/Newton-filters the action matrices into quaternion candidates,
4. lets the original FARES scoring code choose the final R,t.

The companion modular static-C kernel remains the exact proof path; this file is
the practical numeric bridge into the live R,t output pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import ysolve_template_core as yc
from fares_static_c_replay import (
    ensure_double_kernel,
    load_static_branch_manifest,
)
from ysolve_msolve_adapter import YSolveMsolveAdapter, add_stage


def read_csv_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=float)


def input_hash(input_json: Path, p3d: Path, p2d: Path) -> str:
    meta = json.loads(input_json.read_text(encoding="utf-8"))
    if meta.get("input_sha256"):
        return str(meta["input_sha256"])
    h = hashlib.sha256()
    h.update(p3d.read_bytes())
    h.update(p2d.read_bytes())
    h.update(json.dumps(meta.get("K"), sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def write_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")


def patch_pnp_stage_timers(PnPSolver, stages: dict[str, float]) -> None:
    original_build = PnPSolver._build_equations
    original_extract = PnPSolver._extract_best_solution

    def timed_build_equations(self, p2d):
        t0 = time.perf_counter()
        try:
            return original_build(self, p2d)
        finally:
            add_stage(stages, "equation_build_ms", 1000.0 * (time.perf_counter() - t0))

    def timed_extract_best_solution(self, quaternions, p2d):
        stages["candidate_count"] = float(len(quaternions) if quaternions is not None else 0)
        t0 = time.perf_counter()
        try:
            return original_extract(self, quaternions, p2d)
        finally:
            add_stage(stages, "pose_scoring_ms", 1000.0 * (time.perf_counter() - t0))

    PnPSolver._build_equations = timed_build_equations
    PnPSolver._extract_best_solution = timed_extract_best_solution


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver-dir", required=True, help="Folder containing pnp_solver.py and pnp_poly_solvers.py.")
    ap.add_argument("--yam-code-dir", required=True, help="Folder containing pnp_symbolic_tree_offline.py.")
    ap.add_argument("--branch-json", required=True, help="Learned FARES static branch manifest.")
    ap.add_argument("--double-so", help="Optional precompiled floating static-C action kernel.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--p3d", required=True)
    ap.add_argument("--p2d", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--equation-mode", default="first3", choices=["first3"])
    ap.add_argument("--root-residual-tol", type=float, default=1e-8)
    ap.add_argument("--root-refine-backend", default="auto", choices=["auto", "python", "c"])
    ap.add_argument("--root-refine-lib")
    ap.add_argument("--action-weights", default="branch")
    ap.add_argument("--max-roots", type=int, default=80)
    ap.add_argument("--verify-modular", action="store_true", help="Also replay exact modular static-C branch and check f(A)=0.")
    ap.add_argument("--prime", type=int, default=2147483647)
    args = ap.parse_args()

    solver_dir = Path(args.solver_dir).resolve()
    yam_code_dir = Path(args.yam_code_dir).resolve()
    branch_json = Path(args.branch_json).resolve()
    input_json = Path(args.input)
    p3d_path = Path(args.p3d)
    p2d_path = Path(args.p2d)
    out_path = Path(args.out)

    result = {
        "solver": "optimalpnp_fares_shell_ysolve_static_c_rt",
        "success": False,
        "input_sha256": input_hash(input_json, p3d_path, p2d_path),
        "solver_dir": str(solver_dir),
        "yam_code_dir": str(yam_code_dir),
        "branch_json": str(branch_json),
    }
    stages: dict[str, float] = {}
    ysolve_calls: list[dict] = []

    t0 = time.perf_counter()
    old_cwd = Path.cwd()
    try:
        import_t0 = time.perf_counter()
        if str(solver_dir) not in sys.path:
            sys.path.insert(0, str(solver_dir))
        yc.add_yam_code_dir(yam_code_dir)

        branch_load_t0 = time.perf_counter()
        branch = load_static_branch_manifest(branch_json)
        double_so = Path(args.double_so).resolve() if args.double_so else ensure_double_kernel(branch)
        stages["static_branch_load_compile_ms"] = 1000.0 * (time.perf_counter() - branch_load_t0)

        import pnp_poly_solvers

        ysolve_adapter = YSolveMsolveAdapter(
            branch=branch,
            double_so=double_so,
            root_refiner=args.root_refine_lib,
            equation_mode=args.equation_mode,
            action_weights=args.action_weights,
            root_residual_tol=args.root_residual_tol,
            root_refine_backend=args.root_refine_backend,
            max_roots=args.max_roots,
            verify_modular=args.verify_modular,
            prime=args.prime,
            stages=stages,
            calls=ysolve_calls,
        )

        pnp_poly_solvers.solve_with_msolve = ysolve_adapter.solve
        from pnp_solver import PnPSolver

        patch_pnp_stage_timers(PnPSolver, stages)
        stages["adapter_import_patch_ms"] = 1000.0 * (time.perf_counter() - import_t0)

        io_t0 = time.perf_counter()
        meta = json.loads(input_json.read_text(encoding="utf-8"))
        K = np.array(meta["K"], dtype=float)
        p3d = read_csv_matrix(p3d_path)
        p2d = read_csv_matrix(p2d_path)
        weights = np.ones(p3d.shape[0], dtype=float)
        stages["input_load_ms"] = 1000.0 * (time.perf_counter() - io_t0)

        os.chdir(solver_dir)
        init_t0 = time.perf_counter()
        solver = PnPSolver(K, p3d, weights)
        stages["solver_init_ms"] = 1000.0 * (time.perf_counter() - init_t0)
        solve_t0 = time.perf_counter()
        R, t, objective = solver.solve(p2d)
        solve_ms = 1000.0 * (time.perf_counter() - solve_t0)
        total_ms = 1000.0 * (time.perf_counter() - t0)
        stages["pnp_solve_ms"] = solve_ms
        stages["shared_non_solver_inside_pnp_ms"] = max(
            0.0, solve_ms - stages.get("polynomial_solver_total_ms", 0.0)
        )
        stages["adapter_non_pnp_overhead_ms"] = max(0.0, total_ms - solve_ms)

        result.update(
            {
                "success": R is not None and t is not None,
                "R": None if R is None else np.asarray(R, dtype=float).tolist(),
                "t": None if t is None else np.asarray(t, dtype=float).reshape(-1).tolist(),
                "objective": None if objective is None else float(objective),
                "total_ms": total_ms,
                "stages_ms": stages,
                "ysolve_calls": ysolve_calls,
                "quaternion_count": None if not ysolve_calls else ysolve_calls[-1].get("quaternion_count"),
            }
        )
    except Exception as exc:
        result.update(
            {
                "success": False,
                "total_ms": 1000.0 * (time.perf_counter() - t0),
                "error": repr(exc),
                "ysolve_calls": ysolve_calls,
                "stages_ms": stages,
            }
        )
    finally:
        os.chdir(old_cwd)

    write_result(out_path, result)
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
