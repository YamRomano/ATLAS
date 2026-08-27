#!/usr/bin/env python3
"""FARES equations + Ysolve static-C replay vs clean upstream msolve.

This is the FARES-family companion to the fast wedge/static-C benchmark.  It
does not use wedge equations.  Each test input follows:

  FARES Python equation builder -> 3 dehomogenized quartics
  Full exact finite-field build -> action matrices
  FARES Ysolve static-C branch replay -> action matrices
  clean msolve over the same finite field -> Shape/RUR

The correctness checks are exact modulo p:
  * Ysolve action matrices equal Full exact action matrices.
  * f_i(Ax,Ay,Az)=0 and the action matrices commute.
  * Ysolve-derived Shape/RUR equals clean msolve Shape/RUR.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HARNESS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HARNESS))
sys.path.insert(0, str(ROOT / "wedge_replay"))

from fares_static_c_replay import (  # noqa: E402
    FaresStaticBranch,
    coeff_mod_fraction,
    ensure_static_branch,
    flatten_coefficients_mod,
    full_exact_fares_action_matrices,
    matrices_equal,
    patch_core_coeff_mod,
    rationalize_eqs,
    verify_fares_actions,
)
import ysolve_template_core as yc  # noqa: E402
from wedge_replay.phasef_wedge_replay import (  # noqa: E402
    compare_msolve_shape_to_actions,
    parse_msolve_parametrization,
    rur_solution_preview,
)


ACTION_PRIME = 2147483647
MSOLVE_PROOF_PRIME = 1073741827


def _median(vals):
    vals = [float(v) for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def _mean(vals):
    vals = [float(v) for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _parse_seed_list(text: str | None) -> list[int]:
    if not text:
        return []
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def locate_packaged_solver_dir(explicit: str | Path | None = None) -> Path:
    if explicit:
        p = Path(explicit).resolve()
        if (p / "pnp_solver.py").exists():
            return p
    candidates = [
        ROOT.parents[2] / "fares_python_solver/OptimalPnP-main-master/OptimalPnP-main-master",
        Path("/content/fares_python_solver/OptimalPnP-main-master/OptimalPnP-main-master"),
        Path("/content/pnp/OptimalPnP-main-master/OptimalPnP-main-master"),
        Path("/content/PnP_EndToEnd_StrongProof/Downloaded_OptimalPnP-main-master/OptimalPnP-main-master"),
    ]
    for p in candidates:
        if (p / "pnp_solver.py").exists():
            return p.resolve()
    raise FileNotFoundError("Could not find a FARES Python solver dir containing pnp_solver.py")


def make_instance(seed: int, points: int, noise_px: float) -> dict[str, np.ndarray]:
    from generate_live_pnp_instances import make_instance as _make

    K = np.array(
        [[647.81841563, 0.0, 335.8814632], [0.0, 645.9438274, 225.99776891], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    return _make(seed, points, noise_px, K)


def load_fares_dehom_eqs(
    *,
    seed: int,
    solver_dir: Path,
    points: int,
    noise_px: float,
) -> list[dict[tuple[int, int, int], Any]]:
    """Build the exact same first-three FARES quartics used by the pipeline."""

    if str(solver_dir) not in sys.path:
        sys.path.insert(0, str(solver_dir))
    from pnp_solver import PnPSolver

    data = make_instance(seed, points, noise_px)
    old_cwd = Path.cwd()
    try:
        os.chdir(solver_dir)
        solver = PnPSolver(data["K"], data["p3d"], np.ones(data["p3d"].shape[0], dtype=float))
        equations = solver._build_equations(data["p2d"])
    finally:
        os.chdir(old_cwd)
    return rationalize_eqs(yc.dehomogenize_fares_equations(equations, mode="first3"))


def write_msolve_input(eqs, out_path: Path, characteristic: int) -> Path:
    yam = yc._yam()
    text = ",".join(yam.FREE_VARS) + f"\n{int(characteristic)}\n" + ",\n".join(
        yam.poly_to_msolve(eq) for eq in eqs
    ) + "\n"
    out_path.write_text(text, encoding="utf-8")
    return out_path


def run_clean_msolve_parametrization(eqs, *, msolve_bin: Path, p: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        ms_path = td_path / f"fares_mod_{p}.ms"
        out_path = td_path / f"fares_mod_{p}.res"
        write_msolve_input(eqs, ms_path, p)
        cmd = [str(msolve_bin), "-P", "2", "-d", "0", "-f", str(ms_path), "-o", str(out_path)]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
        elapsed = time.perf_counter() - t0
        text = out_path.read_text(encoding="utf-8") if out_path.exists() else proc.stdout
    parsed = None
    parse_error = ""
    if proc.returncode == 0:
        try:
            parsed = parse_msolve_parametrization(text)
        except Exception as exc:
            parse_error = repr(exc)
    return {
        "available": True,
        "ok": proc.returncode == 0 and parsed is not None and bool(parsed.get("ok")),
        "seconds": elapsed,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "")[:1000],
        "stderr": (proc.stderr or "")[:1000],
        "parse_error": parse_error,
        "param": parsed,
        "cmd": " ".join(cmd),
    }


def compare_msolve_for_eqs(
    eqs,
    proof_branches: list[FaresStaticBranch],
    *,
    p: int,
    seed: int,
    out_dir: Path,
    msolve_bin: Path | None,
    degree: int,
) -> dict[str, Any]:
    if msolve_bin is None:
        return {
            "msolve_ok": False,
            "msolve_available": False,
            "same_solutions": False,
            "msolve_shape_full_rur_equal": False,
            "msolve_shape_reason": "msolve disabled/not found",
        }

    proof_idx, proof_branch, proof_mats, proof_meta, proof_errors, proof_new = ensure_static_branch(
        eqs,
        branches=proof_branches,
        p=p,
        seed=seed,
        out_dir=out_dir / "proof_branches",
        degree=degree,
    )
    msolve = run_clean_msolve_parametrization(eqs, msolve_bin=msolve_bin, p=p)
    shape_cmp = {}
    rur_preview = ""
    ysolve_rur_seconds = None
    if msolve.get("param"):
        t0 = time.perf_counter()
        shape_cmp = compare_msolve_shape_to_actions(msolve.get("param"), proof_mats, p)
        ysolve_rur_seconds = time.perf_counter() - t0
        rur_preview = rur_solution_preview(msolve.get("param"), shape_cmp)

    return {
        "proof_branch_index": proof_idx,
        "proof_branch_seed": proof_branch.seed,
        "proof_new_branch": bool(proof_new),
        "proof_branch_errors": " | ".join(proof_errors),
        "proof_ysolve_static_seconds": proof_meta.get("replay_seconds"),
        "msolve_available": bool(msolve.get("available")),
        "msolve_ok": bool(msolve.get("ok")),
        "msolve_seconds": msolve.get("seconds") if msolve.get("ok") else None,
        "msolve_returncode": msolve.get("returncode"),
        "msolve_stdout_preview": msolve.get("stdout", ""),
        "msolve_stderr_preview": msolve.get("stderr", ""),
        "msolve_parse_error": msolve.get("parse_error", ""),
        "msolve_shape_ok": bool(shape_cmp.get("ok")),
        "msolve_shape_elim_equal": bool(shape_cmp.get("elim_equal")),
        "msolve_shape_coord_equal": bool(shape_cmp.get("coord_equal")),
        "msolve_shape_full_rur_equal": bool(shape_cmp.get("full_rur_equal")),
        "msolve_shape_qdim": shape_cmp.get("qdim_msolve"),
        "msolve_shape_linear_form": json.dumps(shape_cmp.get("linear_form", [])),
        "msolve_shape_reason": shape_cmp.get("reason", shape_cmp.get("coord_match_reason", "")),
        "ysolve_rur_seconds": ysolve_rur_seconds,
        "same_solutions": bool(shape_cmp.get("full_rur_equal")),
        "rur_solution_preview": rur_preview,
    }


def _c_i64_array(vals: list[int]) -> str:
    return ", ".join(f"{int(v)}u" for v in vals)


def write_pure_c_dispatch_benchmark(
    *,
    out_c: Path,
    action_branches: list[FaresStaticBranch],
    row_specs: list[dict[str, Any]],
    repeats: int,
) -> Path:
    so_paths = ",\n    ".join(json.dumps(str(b.so_path.resolve())) for b in action_branches)
    seeds = [int(r["seed"]) for r in row_specs]
    branch_ids = [int(r["action_branch_index"]) for r in row_specs]
    n_coeffs = max(len(r["coeffs"]) for r in row_specs)
    coeff_rows = []
    coeff_lens = []
    for r in row_specs:
        vals = list(map(int, r["coeffs"]))
        coeff_lens.append(len(vals))
        vals = vals + [0] * (n_coeffs - len(vals))
        coeff_rows.append("{" + _c_i64_array(vals) + "}")
    qdim = int(row_specs[0]["qdim"])
    n_actions = 3 * qdim * qdim
    code = f"""// Generated FARES pure-C dispatch benchmark.
#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N_CASES {len(row_specs)}
#define N_BRANCHES {len(action_branches)}
#define MAX_COEFFS {n_coeffs}
#define N_ACTIONS {n_actions}
#define REPEATS {max(1, int(repeats))}

typedef int (*replay_fn_t)(const uint32_t *coeffs, int n_coeffs, uint32_t *actions_out);

static const char *SO_PATHS[N_BRANCHES] = {{
    {so_paths}
}};
static const int SEEDS[N_CASES] = {{{", ".join(map(str, seeds))}}};
static const int BRANCHES[N_CASES] = {{{", ".join(map(str, branch_ids))}}};
static const int COEFF_LENS[N_CASES] = {{{", ".join(map(str, coeff_lens))}}};
static const uint32_t COEFFS[N_CASES][MAX_COEFFS] = {{
    {", ".join(coeff_rows)}
}};

static double now_seconds(void) {{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}}

int main(void) {{
    void *handles[N_BRANCHES];
    replay_fn_t funcs[N_BRANCHES];
    for (int i = 0; i < N_BRANCHES; ++i) {{
        handles[i] = dlopen(SO_PATHS[i], RTLD_NOW);
        if (!handles[i]) {{ fprintf(stderr, "dlopen failed: %s\\n", dlerror()); return 10 + i; }}
        funcs[i] = (replay_fn_t)dlsym(handles[i], "fares_generated_replay_actions_u32");
        if (!funcs[i]) {{ fprintf(stderr, "dlsym failed: %s\\n", dlerror()); return 20 + i; }}
    }}
    uint32_t *actions = (uint32_t *)calloc((size_t)N_ACTIONS, sizeof(uint32_t));
    if (!actions) return 30;
    printf("seed,branch,pure_c_static_seconds,rc,checksum\\n");
    for (int i = 0; i < N_CASES; ++i) {{
        int b = BRANCHES[i];
        int rc = funcs[b](COEFFS[i], COEFF_LENS[i], actions);
        if (rc < 0) {{
            printf("%d,%d,0,%d,0\\n", SEEDS[i], b, rc);
            continue;
        }}
        uint64_t checksum = 0;
        double t0 = now_seconds();
        for (int r = 0; r < REPEATS; ++r) {{
            rc = funcs[b](COEFFS[i], COEFF_LENS[i], actions);
            if (rc < 0) break;
            checksum += actions[(r * 1315423911u) % N_ACTIONS];
            checksum += actions[(r + i * 17) % N_ACTIONS];
        }}
        double t1 = now_seconds();
        printf("%d,%d,%.12g,%d,%llu\\n", SEEDS[i], b, (t1 - t0) / (double)REPEATS, rc, (unsigned long long)checksum);
    }}
    free(actions);
    for (int i = 0; i < N_BRANCHES; ++i) dlclose(handles[i]);
    return 0;
}}
"""
    out_c.parent.mkdir(parents=True, exist_ok=True)
    out_c.write_text(code, encoding="utf-8")
    return out_c


def compile_and_run_pure_c_benchmark(c_path: Path, exe_path: Path) -> dict[int, dict[str, Any]]:
    cc = os.environ.get("CC", "cc")
    subprocess.run([cc, "-O3", "-std=c11", str(c_path), "-ldl", "-o", str(exe_path)], check=True)
    proc = subprocess.run([str(exe_path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    print(proc.stdout)
    reader = csv.DictReader(proc.stdout.splitlines())
    out: dict[int, dict[str, Any]] = {}
    for row in reader:
        seed = int(row["seed"])
        out[seed] = {
            "pure_c_static_branch": int(row["branch"]),
            "pure_c_static_seconds": float(row["pure_c_static_seconds"]),
            "pure_c_static_rc": int(row["rc"]),
            "pure_c_static_checksum": row["checksum"],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver-dir", type=Path, default=None)
    ap.add_argument("--msolve-bin", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/fares_static_c_vs_clean_msolve")
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--seed-start", type=int, default=20260620)
    ap.add_argument("--branch-seeds", default="20260616")
    ap.add_argument("--points", type=int, default=40)
    ap.add_argument("--noise-px", type=float, default=0.0)
    ap.add_argument("--degree", type=int, default=11)
    ap.add_argument("--action-prime", type=int, default=ACTION_PRIME)
    ap.add_argument("--msolve-prime", type=int, default=MSOLVE_PROOF_PRIME)
    ap.add_argument("--skip-msolve-proof", action="store_true")
    ap.add_argument("--pure-c-repeats", type=int, default=200)
    args = ap.parse_args()

    patch_core_coeff_mod()
    yc.add_yam_code_dir(ROOT / "yam_code")
    solver_dir = locate_packaged_solver_dir(args.solver_dir)
    msolve_bin = None if args.skip_msolve_proof else args.msolve_bin
    if msolve_bin is not None:
        msolve_bin = Path(msolve_bin).resolve()
        if not msolve_bin.exists():
            raise FileNotFoundError(f"missing msolve binary: {msolve_bin}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("FARES solver dir:", solver_dir)
    print("clean msolve:", msolve_bin or "disabled")
    print("out:", args.out_dir)

    action_branches: list[FaresStaticBranch] = []
    proof_branches: list[FaresStaticBranch] = []

    print("\n" + "=" * 78)
    print("OFFLINE: learn initial FARES branch bank")
    print("=" * 78)
    for bseed in _parse_seed_list(args.branch_seeds):
        eqs = load_fares_dehom_eqs(seed=bseed, solver_dir=solver_dir, points=args.points, noise_px=args.noise_px)
        if not action_branches:
            action_branches.append(
                ensure_static_branch(
                    eqs,
                    branches=action_branches,
                    p=args.action_prime,
                    seed=bseed,
                    out_dir=args.out_dir / "action_branches",
                    degree=args.degree,
                )[1]
            )
        if msolve_bin is not None and not proof_branches:
            proof_branches.append(
                ensure_static_branch(
                    eqs,
                    branches=proof_branches,
                    p=args.msolve_prime,
                    seed=bseed,
                    out_dir=args.out_dir / "proof_branches",
                    degree=args.degree,
                )[1]
            )

    print("\n" + "=" * 78)
    print("ONLINE/AUDIT: FARES static C vs Full exact and clean msolve")
    print("=" * 78)
    rows: list[dict[str, Any]] = []
    c_bench_rows: list[dict[str, Any]] = []
    for i in range(args.count):
        seed = args.seed_start + i
        print(f"\n[{i + 1}/{args.count}] seed={seed}")
        eqs = load_fares_dehom_eqs(seed=seed, solver_dir=solver_dir, points=args.points, noise_px=args.noise_px)

        _full_tmpl, full_mats, full_meta = full_exact_fares_action_matrices(eqs, p=args.action_prime, degree=args.degree)
        action_idx, action_branch, replay_mats, replay_meta, branch_errors, new_branch = ensure_static_branch(
            eqs,
            branches=action_branches,
            p=args.action_prime,
            seed=seed,
            out_dir=args.out_dir / "action_branches",
            degree=args.degree,
        )
        verify = verify_fares_actions(eqs, replay_mats, args.action_prime)
        actions_equal_full = matrices_equal(full_mats, replay_mats, args.action_prime)
        coeffs = flatten_coefficients_mod(eqs, action_branch.coeff_terms, args.action_prime).tolist()
        c_bench_rows.append(
            {
                "seed": seed,
                "action_branch_index": action_idx,
                "coeffs": coeffs,
                "qdim": replay_meta["N"],
            }
        )

        msolve_fields = compare_msolve_for_eqs(
            eqs,
            proof_branches,
            p=args.msolve_prime,
            seed=seed,
            out_dir=args.out_dir,
            msolve_bin=msolve_bin,
            degree=args.degree,
        )

        row = {
            "seed": seed,
            "status": "ok" if actions_equal_full and verify["ok"] and (args.skip_msolve_proof or msolve_fields.get("same_solutions")) else "mismatch",
            "action_branch_index": action_idx,
            "action_branch_seed": action_branch.seed,
            "action_new_branch": bool(new_branch),
            "branch_errors": " | ".join(branch_errors),
            "rank": replay_meta.get("rank"),
            "N": replay_meta.get("N"),
            "full_exact_build_seconds": full_meta.get("full_seconds"),
            "ysolve_static_action_seconds": replay_meta.get("replay_seconds"),
            "actions_equal_full": bool(actions_equal_full),
            "verify_ok": bool(verify["ok"]),
            "commutators": json.dumps(verify["commutators"]),
            "ideal_nonzeros": json.dumps(verify["ideal"]),
            "exact_rank_pass": bool(full_meta.get("exact_rank_pass")),
            "exact_quotient_pass": bool(full_meta.get("exact_quotient_pass")),
            **msolve_fields,
        }
        rows.append(row)
        print(
            f"  branch={action_idx} new={new_branch} "
            f"full={row['full_exact_build_seconds'] * 1000:.3f}ms "
            f"static={row['ysolve_static_action_seconds'] * 1000:.3f}ms "
            f"eq_full={row['actions_equal_full']} verify={row['verify_ok']}"
        )
        if row.get("msolve_available"):
            print(
                f"  msolve={row.get('msolve_seconds')} "
                f"same_solutions={row.get('same_solutions')} "
                f"rur_equal={row.get('msolve_shape_full_rur_equal')}"
            )

    print("\n" + "=" * 78)
    print("PURE C STATIC DISPATCH BENCHMARK")
    print("=" * 78)
    bench_c = args.out_dir / "static_c_dispatch/bench_fares_static_dispatch.c"
    bench_exe = args.out_dir / "static_c_dispatch/bench_fares_static_dispatch"
    write_pure_c_dispatch_benchmark(
        out_c=bench_c,
        action_branches=action_branches,
        row_specs=c_bench_rows,
        repeats=args.pure_c_repeats,
    )
    static_results = compile_and_run_pure_c_benchmark(bench_c, bench_exe)
    for row in rows:
        row.update(static_results.get(int(row["seed"]), {}))
        if row.get("msolve_seconds") and row.get("pure_c_static_seconds"):
            row["speedup_clean_msolve_over_pure_c_static"] = row["msolve_seconds"] / row["pure_c_static_seconds"]
        if row.get("full_exact_build_seconds") and row.get("pure_c_static_seconds"):
            row["speedup_full_exact_over_pure_c_static"] = row["full_exact_build_seconds"] / row["pure_c_static_seconds"]

    csv_path = args.out_dir / "fares_static_c_vs_clean_msolve.csv"
    _write_csv(csv_path, rows)

    summary = {
        "count": args.count,
        "action_prime": args.action_prime,
        "msolve_prime": args.msolve_prime,
        "final_action_branch_count": len(action_branches),
        "final_proof_branch_count": len(proof_branches),
        "ok_count": sum(1 for r in rows if r.get("status") == "ok"),
        "all_actions_equal_full": all(r.get("actions_equal_full") is True for r in rows),
        "all_verify_ok": all(r.get("verify_ok") is True for r in rows),
        "clean_msolve_success_count": sum(1 for r in rows if r.get("msolve_ok")),
        "same_exact_RUR_solution_count": sum(1 for r in rows if r.get("same_solutions")),
        "median_full_exact_build_ms": (_median([r.get("full_exact_build_seconds") for r in rows]) or 0) * 1000,
        "median_python_wrapped_static_action_ms": (_median([r.get("ysolve_static_action_seconds") for r in rows]) or 0) * 1000,
        "median_pure_c_static_action_ms": (_median([r.get("pure_c_static_seconds") for r in rows]) or 0) * 1000,
        "mean_pure_c_static_action_ms": (_mean([r.get("pure_c_static_seconds") for r in rows]) or 0) * 1000,
        "median_clean_msolve_ms": (_median([r.get("msolve_seconds") for r in rows]) or 0) * 1000,
        "median_speedup_clean_msolve_over_pure_c_static": _median(
            [r.get("speedup_clean_msolve_over_pure_c_static") for r in rows]
        ),
        "median_speedup_full_exact_over_pure_c_static": _median(
            [r.get("speedup_full_exact_over_pure_c_static") for r in rows]
        ),
        "csv": str(csv_path),
        "bench_c": str(bench_c),
        "bench_exe": str(bench_exe),
    }
    summary_path = args.out_dir / "fares_static_c_vs_clean_msolve_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(args.out_dir / "summary_table.csv", [{"metric": k, "value": v} for k, v in summary.items()])
    (args.out_dir / "branch_bank_manifest.json").write_text(
        json.dumps(
            {
                "action_branches": [b.to_manifest() for b in action_branches],
                "proof_branches": [b.to_manifest() for b in proof_branches],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(json.dumps(summary, indent=2))
    print("CSV:", csv_path)

    if not summary["all_actions_equal_full"]:
        raise SystemExit("FAIL: at least one FARES static replay differs from Full exact build")
    if not summary["all_verify_ok"]:
        raise SystemExit("FAIL: at least one FARES static replay failed exact equation/commutation checks")
    if not args.skip_msolve_proof and summary["same_exact_RUR_solution_count"] != args.count:
        raise SystemExit("FAIL: at least one clean msolve Shape/RUR comparison failed")


if __name__ == "__main__":
    main()
