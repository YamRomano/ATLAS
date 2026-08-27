#!/usr/bin/env python3
"""Run FARES/msolve vs FARES with static-C YSolve as a drop-in solver.

This experiment keeps the FARES/OptimalPnP pipeline intact.  The only changed
part is the algebraic solver entry point:

  baseline: FARES builds equations -> calls msolve -> scores candidates -> R,t
  static:   FARES builds equations -> calls static-C YSolve -> scores candidates -> R,t

Offline work is excluded from online timing.  Verification/proof can also be
run separately and is not included in online timing.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

import ysolve_template_core as yc
from fares_static_c_replay import ensure_c_root_refiner, ensure_double_kernel, learn_static_branch, rationalize_eqs


def run(cmd: list[object]) -> None:
    cmd = [str(x) for x in cmd]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def read_csv_float(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=float)


def median(values: list[float]) -> float | None:
    return None if not values else float(statistics.median(values))


def field_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def copy_eval_instances(all_dir: Path, eval_dir: Path, train_count: int, count: int) -> None:
    if eval_dir.exists():
        shutil.rmtree(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    for src_idx in range(train_count, count):
        src = all_dir / f"instance_{src_idx:03d}"
        dst = eval_dir / f"instance_{src_idx:03d}"
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copytree(src, dst)


def learn_one_static_branch(
    *,
    solver_dir: Path,
    yam_code_dir: Path,
    instance_dir: Path,
    branch_dir: Path,
    seed: int,
    prime: int,
    degree: int,
):
    """Learn one offline branch from one FARES input and compile static kernels."""

    if str(solver_dir) not in sys.path:
        sys.path.insert(0, str(solver_dir))
    yc.add_yam_code_dir(yam_code_dir)
    from pnp_solver import PnPSolver

    meta = json.loads((instance_dir / "input.json").read_text(encoding="utf-8"))
    K = np.array(meta["K"], dtype=float)
    p3d = read_csv_float(instance_dir / "p3d.csv")
    p2d = read_csv_float(instance_dir / "p2d.csv")

    solver = PnPSolver(K, p3d, np.ones(p3d.shape[0], dtype=float))
    equations = solver._build_equations(p2d)
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
    return branch, branch_json, double_so


def summarize_compare_csv(
    csv_path: Path,
    out_summary_json: Path,
    *,
    train_count: int,
    offline_seconds: float,
    branch_json: Path,
    double_so: Path,
    rot_tol_deg: float,
    trans_tol: float,
) -> dict:
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))

    def vals(key: str) -> list[float]:
        out: list[float] = []
        for row in rows:
            value = field_float(row, key)
            if value is not None:
                out.append(value)
        return out

    def count_true(key: str) -> int:
        return sum(1 for row in rows if str(row.get(key)).lower() == "true")

    msolve_subprocess = vals("fares_msolve_stage_msolve_subprocess_ms")
    msolve_poly = vals("fares_msolve_stage_polynomial_solver_total_ms")
    msolve_total = vals("fares_msolve_total_ms")

    static_action = vals("static_rt_stage_ysolve_static_action_double_ms")
    static_root = vals("static_rt_stage_ysolve_static_root_total_ms")
    static_poly = vals("static_rt_stage_polynomial_solver_total_ms")
    static_total = vals("static_rt_total_ms")

    rot_errors = vals("static_rt_vs_fares_msolve_rotation_error_deg")
    trans_errors = vals("static_rt_vs_fares_msolve_translation_l2")

    strict_pose_pass = 0
    for row in rows:
        rot = field_float(row, "static_rt_vs_fares_msolve_rotation_error_deg")
        trans = field_float(row, "static_rt_vs_fares_msolve_translation_l2")
        if rot is not None and trans is not None and rot <= rot_tol_deg and trans <= trans_tol:
            strict_pose_pass += 1

    summary = {
        "experiment": "fares_static_c_dropin_train_eval",
        "train_count": train_count,
        "eval_count": len(rows),
        "offline_template_seconds": float(offline_seconds),
        "branch_json": str(branch_json),
        "double_so": str(double_so),
        "fares_msolve_success_count": count_true("fares_msolve_success"),
        "static_rt_success_count": count_true("static_rt_success"),
        "strict_pose_pass_count": strict_pose_pass,
        "strict_pose_all_pass": strict_pose_pass == len(rows),
        "rot_tol_deg": rot_tol_deg,
        "trans_tol": trans_tol,
        "max_rotation_error_deg": max(rot_errors) if rot_errors else None,
        "max_translation_l2": max(trans_errors) if trans_errors else None,
        "median_fares_msolve_total_ms": median(msolve_total),
        "median_static_rt_total_ms": median(static_total),
        "median_fares_msolve_polynomial_ms": median(msolve_poly),
        "median_static_rt_polynomial_ms": median(static_poly),
        "median_fares_msolve_subprocess_ms": median(msolve_subprocess),
        "median_static_action_ms": median(static_action),
        "median_static_root_ms": median(static_root),
        "speedup_msolve_subprocess_over_static_action_median": (
            median(msolve_subprocess) / median(static_action)
            if median(msolve_subprocess) is not None and median(static_action)
            else None
        ),
        "speedup_msolve_polynomial_over_static_polynomial_median": (
            median(msolve_poly) / median(static_poly)
            if median(msolve_poly) is not None and median(static_poly)
            else None
        ),
        "speedup_msolve_total_over_static_total_median": (
            median(msolve_total) / median(static_total)
            if median(msolve_total) is not None and median(static_total)
            else None
        ),
        "total_fares_msolve_subprocess_ms": float(sum(msolve_subprocess)),
        "total_static_action_ms": float(sum(static_action)),
        "total_fares_msolve_polynomial_ms": float(sum(msolve_poly)),
        "total_static_polynomial_ms": float(sum(static_poly)),
        "total_fares_msolve_total_ms": float(sum(msolve_total)),
        "total_static_rt_total_ms": float(sum(static_total)),
    }
    out_summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def maybe_run_proof(
    *,
    script_dir: Path,
    solver_dir: Path,
    msolve_bin: Path,
    out_dir: Path,
    count: int,
    train_count: int,
    seed: int,
    branch_seed: int,
    proof_count: int,
) -> Path | None:
    proof_script = script_dir / "run_fares_static_c_vs_clean_msolve.py"
    if not proof_script.exists():
        print("Proof script missing; skipping exact Shape/RUR proof:", proof_script)
        return None

    actual_count = min(max(0, count - train_count), proof_count)
    if actual_count <= 0:
        return None
    proof_dir = out_dir / "untimed_exact_shape_rur_proof"
    run(
        [
            sys.executable,
            proof_script,
            "--solver-dir",
            solver_dir,
            "--msolve-bin",
            msolve_bin,
            "--count",
            actual_count,
            "--seed-start",
            seed + train_count,
            "--branch-seeds",
            branch_seed,
            "--out-dir",
            proof_dir,
        ]
    )
    return proof_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver-dir", required=True)
    ap.add_argument("--yam-code-dir", required=True)
    ap.add_argument("--msolve-bin", required=True)
    ap.add_argument("--out", default="results/fares_static_c_dropin_train1_eval99")
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--train-count", type=int, default=1)
    ap.add_argument("--points", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20240518)
    ap.add_argument("--noise-px", type=float, default=0.0)
    ap.add_argument("--prime", type=int, default=2147483647)
    ap.add_argument("--degree", type=int, default=11)
    ap.add_argument("--root-refine-backend", choices=["auto", "python", "c"], default="c")
    ap.add_argument("--root-refine-lib", default="")
    ap.add_argument("--rot-tol-deg", type=float, default=1e-5)
    ap.add_argument("--trans-tol", type=float, default=1e-5)
    ap.add_argument("--verify-modular", action="store_true", help="Slow per-input exact modular verification inside adapter.")
    ap.add_argument("--run-proof", action="store_true", help="Run untimed clean-msolve Shape/RUR proof after the timed comparison.")
    ap.add_argument("--proof-count", type=int, default=10, help="How many eval inputs to proof-check when --run-proof is set.")
    args = ap.parse_args()

    if args.train_count < 1 or args.train_count >= args.count:
        raise ValueError("--train-count must be in [1, count-1]")

    script_dir = Path(__file__).resolve().parent
    solver_dir = Path(args.solver_dir).resolve()
    yam_code_dir = Path(args.yam_code_dir).resolve()
    msolve_bin = Path(args.msolve_bin).resolve()
    out = Path(args.out).resolve()
    all_instances = out / "instances_all"
    eval_instances = out / f"instances_eval_from_{args.train_count:03d}"
    branch_dir = out / "offline_branch"
    runs_dir = out / "timed_dropin_compare"

    out.mkdir(parents=True, exist_ok=True)
    for required in [script_dir / "generate_live_pnp_instances.py", script_dir / "run_multi_solver_compare.py"]:
        if not required.exists():
            raise FileNotFoundError(required)

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

    print("\n=== Offline: learn static-C template from first training input ===")
    import time

    t0 = time.perf_counter()
    branch, branch_json, double_so = learn_one_static_branch(
        solver_dir=solver_dir,
        yam_code_dir=yam_code_dir,
        instance_dir=all_instances / "instance_000",
        branch_dir=branch_dir,
        seed=args.seed,
        prime=args.prime,
        degree=args.degree,
    )
    offline_seconds = time.perf_counter() - t0
    learned_rank = getattr(branch.tmpl, "rank", None)
    if learned_rank is None:
        learned_rank = len(getattr(branch.tmpl, "pivot_cols", []))
    print(
        f"learned branch index={branch.index} seed={branch.seed} p={branch.p} "
        f"rank={learned_rank} quotient={len(branch.tmpl.basis_cols)} in {offline_seconds:.3f}s"
    )

    print("\n=== Freeze template and prepare online eval set ===")
    copy_eval_instances(all_instances, eval_instances, args.train_count, args.count)
    print(f"train instances: instance_000..instance_{args.train_count - 1:03d}")
    print(f"eval instances : instance_{args.train_count:03d}..instance_{args.count - 1:03d}")

    print("\n=== Prepare optional C root refiner ===")
    root_refiner = None
    effective_root_backend = args.root_refine_backend
    if args.root_refine_backend in {"auto", "c"}:
        if args.root_refine_lib:
            root_refiner = Path(args.root_refine_lib).resolve()
        else:
            try:
                root_refiner = ensure_c_root_refiner(
                    yam_code_dir=yam_code_dir,
                    out_dir=branch_dir,
                    require_lapack=args.root_refine_backend == "c",
                )
            except Exception as exc:
                if args.root_refine_backend == "c":
                    print(f"C root refiner build/load failed; falling back to python root extraction: {exc}")
                    effective_root_backend = "python"
                    root_refiner = None
                else:
                    raise
    if root_refiner:
        print("root refiner:", root_refiner)
    print("effective root backend:", effective_root_backend)

    print("\n=== Online timed comparison: unchanged FARES/msolve vs FARES+static-C YSolve ===")
    static_cmd = (
        f"{sys.executable} {script_dir / 'optimalpnp_static_c_rt_adapter.py'} "
        f"--solver-dir {solver_dir} --yam-code-dir {yam_code_dir} "
        f"--branch-json {branch_json} --double-so {double_so} "
        f"--input {{input_json}} --p3d {{p3d_csv}} --p2d {{p2d_csv}} --out {{out_json}} "
        f"--root-refine-backend {effective_root_backend}"
    )
    if root_refiner:
        static_cmd += f" --root-refine-lib {root_refiner}"
    if args.verify_modular:
        static_cmd += f" --verify-modular --prime {args.prime}"

    msolve_cmd = (
        f"{sys.executable} {script_dir / 'optimalpnp_python_adapter.py'} "
        f"--solver-dir {solver_dir} --msolve-bin {msolve_bin} "
        f"--input {{input_json}} --p3d {{p3d_csv}} --p2d {{p2d_csv}} --out {{out_json}}"
    )

    merged_csv = runs_dir / "fares_msolve_vs_static_c_ysolve_rt.csv"
    run(
        [
            sys.executable,
            script_dir / "run_multi_solver_compare.py",
            "--instances",
            eval_instances,
            "--reference",
            "fares_msolve",
            "--out",
            merged_csv,
            "--solver",
            f"fares_msolve={msolve_cmd}",
            "--solver",
            f"static_rt={static_cmd}",
        ]
    )

    summary = summarize_compare_csv(
        merged_csv,
        out / "dropin_summary.json",
        train_count=args.train_count,
        offline_seconds=offline_seconds,
        branch_json=branch_json,
        double_so=double_so,
        rot_tol_deg=args.rot_tol_deg,
        trans_tol=args.trans_tol,
    )

    proof_dir = None
    if args.run_proof:
        print("\n=== Untimed proof: static-C actions / Shape-RUR vs clean msolve ===")
        proof_dir = maybe_run_proof(
            script_dir=script_dir,
            solver_dir=solver_dir,
            msolve_bin=msolve_bin,
            out_dir=out,
            count=args.count,
            train_count=args.train_count,
            seed=args.seed,
            branch_seed=args.seed,
            proof_count=args.proof_count,
        )
        summary["proof_dir"] = None if proof_dir is None else str(proof_dir)
        (out / "dropin_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\nOutputs:")
    print("merged csv:", merged_csv)
    print("summary json:", out / "dropin_summary.json")
    if proof_dir:
        print("proof dir:", proof_dir)


if __name__ == "__main__":
    main()
