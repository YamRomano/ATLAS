#!/usr/bin/env python3
"""Persistent static-C YSolve on a provided FARES-style input archive.

The regular persistent runner generates synthetic inputs internally.  This
variant consumes an existing zip/folder containing FARES-style cases:

  case/input.json
  case/p3d.csv
  case/p2d.csv

It learns one static-C branch offline from the first selected case, keeps FARES,
the static-C action kernel, and the C root refiner loaded once, then evaluates
the remaining cases in a single process.  The baseline FARES/msolve path is
still run independently for correctness and timing comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from run_fares_static_c_persistent_batch import (
    ensure_c_root_refiner,
    ensure_direct_coeff_builder,
    input_hash,
    learn_one_static_branch,
    rot_error_deg,
    run_fares_baseline_subprocess,
    solve_static_persistent,
    summarize,
    trans_l2,
    write_csv,
)


def _read_manifest(root: Path) -> list[dict[str, str]]:
    manifest = root / "manifest.csv"
    if manifest.exists():
        return list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))

    rows: list[dict[str, str]] = []
    for p3d in sorted(root.glob("inputs/**/p3d.csv")):
        case_dir = p3d.parent
        p2d = case_dir / "p2d.csv"
        meta = case_dir / "input.json"
        if p2d.exists() and meta.exists():
            rows.append(
                {
                    "experiment": case_dir.parent.name,
                    "case_id": case_dir.name,
                    "p3d_csv": str(p3d.relative_to(root)),
                    "p2d_csv": str(p2d.relative_to(root)),
                }
            )
    return rows


def _prepare_instances_from_source(
    *,
    source_root: Path,
    out_dir: Path,
    limit: int,
    experiment_filter: str,
    min_points: int,
) -> list[dict[str, Any]]:
    rows = _read_manifest(source_root)
    selected: list[dict[str, Any]] = []
    for row in rows:
        exp = str(row.get("experiment") or "")
        if experiment_filter and exp != experiment_filter:
            continue
        try:
            points = int(float(row.get("points") or 0))
        except Exception:
            points = 0
        if min_points and points and points < min_points:
            continue
        p3d_rel = row.get("p3d_csv")
        p2d_rel = row.get("p2d_csv")
        if not p3d_rel or not p2d_rel:
            continue
        p3d = source_root / p3d_rel
        p2d = source_root / p2d_rel
        meta = p3d.parent / "input.json"
        if not (p3d.exists() and p2d.exists() and meta.exists()):
            continue
        selected.append({"row": row, "case_dir": p3d.parent, "points": points})
        if limit and len(selected) >= limit:
            break

    if len(selected) < 2:
        raise RuntimeError("Need at least two selected cases: one train case and one eval case.")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, rec in enumerate(selected):
        dst = out_dir / f"instance_{i:03d}"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rec["case_dir"] / "input.json", dst / "input.json")
        shutil.copy2(rec["case_dir"] / "p3d.csv", dst / "p3d.csv")
        shutil.copy2(rec["case_dir"] / "p2d.csv", dst / "p2d.csv")
        enriched = json.loads((dst / "input.json").read_text(encoding="utf-8"))
        enriched["source_case_dir"] = str(rec["case_dir"])
        enriched["source_manifest_row"] = rec["row"]
        (dst / "input.json").write_text(json.dumps(enriched, indent=2), encoding="utf-8")
    return selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver-dir", required=True)
    ap.add_argument("--yam-code-dir", required=True)
    ap.add_argument("--msolve-bin", required=True)
    ap.add_argument("--inputs-zip", type=Path, default=None)
    ap.add_argument("--inputs-dir", type=Path, default=None)
    ap.add_argument("--out", default="results/fares_original_inputs_persistent_static_c")
    ap.add_argument("--count", type=int, default=11, help="Total selected cases including train cases.")
    ap.add_argument("--train-count", type=int, default=1)
    ap.add_argument("--experiment-filter", default="", help="Optional exact experiment name from manifest.csv.")
    ap.add_argument("--min-points", type=int, default=0)
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
    ap.add_argument(
        "--fallback-action-weights",
        default="",
        help="Optional robust action weights used only when baseline comparison shows a pose mismatch.",
    )
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--rot-tol-deg", type=float, default=1e-5)
    ap.add_argument("--trans-tol", type=float, default=1e-5)
    args = ap.parse_args()

    if args.train_count < 1 or args.train_count >= args.count:
        raise ValueError("--train-count must be in [1, count-1]")
    if not args.inputs_zip and not args.inputs_dir:
        raise ValueError("Pass --inputs-zip or --inputs-dir")

    script_dir = Path(__file__).resolve().parent
    solver_dir = Path(args.solver_dir).resolve()
    yam_code_dir = Path(args.yam_code_dir).resolve()
    msolve_bin = Path(args.msolve_bin).resolve()
    out = Path(args.out).resolve()
    source_root = out / "source_inputs"
    all_instances = out / "instances_all"
    branch_dir = out / "offline_branch"
    baseline_dir = out / "baseline_fares_msolve_json"
    static_dir = out / "persistent_static_json"
    out.mkdir(parents=True, exist_ok=True)

    if source_root.exists():
        shutil.rmtree(source_root)
    if args.inputs_zip:
        print("\n=== Extract provided FARES-style inputs zip ===")
        with zipfile.ZipFile(args.inputs_zip, "r") as z:
            z.extractall(source_root)
    else:
        source_root.mkdir(parents=True, exist_ok=True)
        src = Path(args.inputs_dir).resolve()
        # Copy the folder contents, not the folder wrapper.
        for child in src.iterdir():
            dst = source_root / child.name
            if child.is_dir():
                shutil.copytree(child, dst)
            else:
                shutil.copy2(child, dst)

    print("\n=== Select/copy input cases ===")
    selected = _prepare_instances_from_source(
        source_root=source_root,
        out_dir=all_instances,
        limit=args.count,
        experiment_filter=args.experiment_filter,
        min_points=args.min_points,
    )
    print(f"selected cases: {len(selected)}")
    print("first train case:", selected[0]["case_dir"])
    print("first eval case :", selected[args.train_count]["case_dir"])

    print("\n=== Offline: learn static-C branch once from selected input ===")
    t0 = time.perf_counter()
    branch, branch_json, double_so = learn_one_static_branch(
        solver_dir=solver_dir,
        yam_code_dir=yam_code_dir,
        instance_dir=all_instances / "instance_000",
        branch_dir=branch_dir,
        seed=20260629,
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
    import ysolve_template_core as yc

    yc.add_yam_code_dir(yam_code_dir)
    from pnp_solver import PnPSolver

    rows: list[dict[str, Any]] = []
    for idx in range(args.train_count, len(selected)):
        instance_id = f"instance_{idx:03d}"
        instance_dir = all_instances / instance_id
        source_case = selected[idx]["case_dir"]
        print(f"\n=== Eval {instance_id}: {source_case} ===", flush=True)

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
            fork_seed=20260629 + idx,
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
        fallback_used = False
        if baseline and not strict_pass and args.fallback_action_weights:
            print("  fast pose mismatch; retrying static path with fallback action weights...", flush=True)
            fallback_static = solve_static_persistent(
                PnPSolver=PnPSolver,
                solver_dir=solver_dir,
                branches=branches,
                double_sos=double_sos,
                branch_dir=branch_dir,
                prime=args.prime,
                degree=args.degree,
                fork_seed=30300629 + idx,
                root_refiner=root_refiner,
                instance_dir=instance_dir,
                action_weights=args.fallback_action_weights,
                root_residual_tol=args.root_residual_tol,
                max_roots=args.max_roots,
                fork_on_miss=not args.no_online_fork,
                direct_coeff_builder=direct_coeff_builder,
            )
            fallback_used = True
            fallback_rerr = rot_error_deg(fallback_static.get("R"), baseline.get("R"))
            fallback_terr = trans_l2(fallback_static.get("t"), baseline.get("t"))
            fallback_pass = (
                bool(fallback_static.get("success"))
                and bool(baseline.get("success"))
                and fallback_rerr is not None
                and fallback_terr is not None
                and fallback_rerr <= args.rot_tol_deg
                and fallback_terr <= args.trans_tol
            )
            if fallback_pass:
                static = fallback_static
                rerr = fallback_rerr
                terr = fallback_terr
                strict_pass = True
                print("  fallback fixed pose mismatch", flush=True)
            else:
                print(
                    f"  fallback did not fix mismatch: rot={fallback_rerr} trans={fallback_terr}",
                    flush=True,
                )

        row: dict[str, Any] = {
            "instance_id": instance_id,
            "source_case": str(source_case),
            "source_experiment": selected[idx]["row"].get("experiment", ""),
            "source_case_id": selected[idx]["row"].get("case_id", ""),
            "points": selected[idx].get("points", ""),
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
            "fallback_used": fallback_used,
            "fallback_action_weights": args.fallback_action_weights if fallback_used else "",
        }
        for k, v in (baseline.get("stages_ms") or {}).items():
            row[f"fares_msolve_stage_{k}"] = v
        for k, v in (static.get("stages_ms") or {}).items():
            row[f"static_stage_{k}"] = v
        rows.append(row)
        print(
            f"baseline={baseline.get('total_ms', '')} ms  static={static.get('total_ms', '')} ms  "
            f"action={(static.get('stages_ms') or {}).get('ysolve_static_action_double_ms')} ms  "
            f"root={(static.get('stages_ms') or {}).get('ysolve_static_root_total_ms')} ms  "
            f"fallback={fallback_used} pass={strict_pass}",
            flush=True,
        )

    csv_path = out / "persistent_original_inputs_fares_msolve_vs_static_c.csv"
    write_csv(rows, csv_path)
    summary = summarize(rows, train_count=args.train_count, offline_seconds=offline_seconds, out=out)
    summary["input_source"] = str(args.inputs_zip or args.inputs_dir)
    summary["selected_count"] = len(selected)
    summary["experiment_filter"] = args.experiment_filter
    summary["min_points"] = args.min_points
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

    try:
        df = pd.DataFrame(rows)
        print("\n=== First rows ===")
        print(df.head(20).to_string(index=False))
    except Exception:
        pass


if __name__ == "__main__":
    main()
