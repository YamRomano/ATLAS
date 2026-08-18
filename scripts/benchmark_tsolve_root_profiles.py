#!/usr/bin/env python3
"""Compare the preserved full TSolve root policy with the opt-in live-fast policy."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np


def camera_center(result: dict) -> np.ndarray | None:
    if not result.get("success") or result.get("R") is None or result.get("t") is None:
        return None
    R = np.asarray(result["R"], dtype=float).reshape(3, 3)
    t = np.asarray(result["t"], dtype=float).reshape(3)
    return -R.T @ t


def rotation_error_degrees(a: dict, b: dict) -> float | None:
    if a.get("R") is None or b.get("R") is None:
        return None
    relative = np.asarray(a["R"], dtype=float) @ np.asarray(b["R"], dtype=float).T
    cosine = max(-1.0, min(1.0, (float(np.trace(relative)) - 1.0) * 0.5))
    return math.degrees(math.acos(cosine))


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--branch-dir", required=True, type=Path)
    parser.add_argument("--solver-dir", required=True, type=Path)
    parser.add_argument("--yam-code", required=True, type=Path)
    parser.add_argument("--harness", required=True, type=Path)
    parser.add_argument("--sample-count", type=int, default=80)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.cases = args.cases.resolve()
    args.branch_dir = args.branch_dir.resolve()
    args.solver_dir = args.solver_dir.resolve()
    args.yam_code = args.yam_code.resolve()
    args.harness = args.harness.resolve()
    if args.output:
        args.output = args.output.resolve()

    sys.path.insert(0, str(args.harness))
    sys.path.insert(0, str(args.solver_dir))
    sys.path.insert(0, str(args.yam_code))
    import ysolve_template_core as yc

    yc.add_yam_code_dir(args.yam_code)
    from fares_static_c_replay import load_static_branch_manifest
    from pnp_solver import PnPSolver
    from run_fares_static_c_persistent_batch import solve_static_persistent

    manifest = next(args.branch_dir.glob("fares_static_branch_*.json"))
    branch = load_static_branch_manifest(manifest)
    double_so = args.branch_dir / f"{Path(branch.so_path).stem}_f64.so"
    root_refiner = args.branch_dir / "pnp_root_refine_lapack.so"
    direct_coeff_builder = args.branch_dir / "fares_direct_coeffs.so"
    all_cases = sorted(p for p in args.cases.iterdir() if (p / "input.json").exists())
    stride = max(1, len(all_cases) // max(1, args.sample_count))
    cases = all_cases[::stride][: args.sample_count]
    rows = []
    for index, case in enumerate(cases):
        results = {}
        for profile in ("full", "live_fast"):
            started = time.perf_counter()
            results[profile] = solve_static_persistent(
                PnPSolver=PnPSolver,
                solver_dir=args.solver_dir,
                branches=[branch],
                double_sos={branch.index: double_so},
                branch_dir=args.branch_dir,
                prime=2147483647,
                degree=11,
                fork_seed=20260707 + index,
                root_refiner=root_refiner,
                instance_dir=case,
                action_weights="branch",
                root_residual_tol=1e-8,
                max_roots=80,
                fork_on_miss=False,
                direct_coeff_builder=direct_coeff_builder,
                root_candidate_profile=profile,
            )
            results[profile]["wall_ms"] = 1000.0 * (time.perf_counter() - started)
        full, fast = results["full"], results["live_fast"]
        cf, cq = camera_center(full), camera_center(fast)
        rows.append(
            {
                "case": case.name,
                "full_success": bool(full.get("success")),
                "fast_success": bool(fast.get("success")),
                "full_ms": full["wall_ms"],
                "fast_ms": fast["wall_ms"],
                "full_root_ms": (full.get("stages_ms") or {}).get("ysolve_static_root_total_ms"),
                "fast_root_ms": (fast.get("stages_ms") or {}).get("ysolve_static_root_total_ms"),
                "full_objective": full.get("objective"),
                "fast_objective": fast.get("objective"),
                "center_difference": None if cf is None or cq is None else float(np.linalg.norm(cf - cq)),
                "rotation_difference_deg": rotation_error_degrees(full, fast),
                "fast_candidates": int((fast.get("stages_ms") or {}).get("candidate_count") or 0),
            }
        )

    full_ms = [r["full_ms"] for r in rows]
    fast_ms = [r["fast_ms"] for r in rows]
    center = [r["center_difference"] for r in rows if r["center_difference"] is not None]
    rotation = [r["rotation_difference_deg"] for r in rows if r["rotation_difference_deg"] is not None]
    objective_delta = [
        abs(float(r["fast_objective"]) - float(r["full_objective"]))
        for r in rows
        if r["fast_objective"] is not None and r["full_objective"] is not None
    ]
    report = {
        "sample_count": len(rows),
        "full_successes": sum(r["full_success"] for r in rows),
        "fast_successes": sum(r["fast_success"] for r in rows),
        "full_median_ms": statistics.median(full_ms),
        "fast_median_ms": statistics.median(fast_ms),
        "speedup": statistics.median(full_ms) / statistics.median(fast_ms),
        "fast_p95_ms": percentile(fast_ms, 0.95),
        "max_center_difference": max(center, default=None),
        "max_rotation_difference_deg": max(rotation, default=None),
        "max_objective_difference": max(objective_delta, default=None),
        "rows": rows,
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
