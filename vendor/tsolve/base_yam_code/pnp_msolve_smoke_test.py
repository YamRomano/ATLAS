#!/usr/bin/env python3
"""Small msolve/Fares smoke test for the PnP wedge system.

Use this in Colab before the full benchmark to verify that the selected msolve
binary actually solves the generated PnP equations instead of segfaulting.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from pnp_symbolic_tree_offline import (
    generate_k_digit_matrices,
    residual_summary,
    root_relative_residuals,
    run_msolve_baseline,
    setup_msolve_from_zip,
    wedge_equations_3eq,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=52)
    ap.add_argument("--k-digits", type=int, default=2)
    ap.add_argument("--msolve-bin", default="")
    ap.add_argument("--msolve-zip", default="/content/task3_msolve.zip")
    ap.add_argument("--extract-dir", default="/content/task3_msolve_unzipped")
    ap.add_argument("--keep-input", action="store_true")
    ap.add_argument("--output-dir", default="", help="Optional directory for kept msolve input/output files.")
    ap.add_argument("--root-residual-tol", type=float, default=1e-4)
    ap.add_argument("--no-output-head", action="store_true", help="Do not print the first msolve output block.")
    args = ap.parse_args()

    if args.msolve_bin:
        msolve_bin = args.msolve_bin
        os.environ["MSOLVE_BIN"] = args.msolve_bin
    else:
        msolve_bin = setup_msolve_from_zip(args.msolve_zip, args.extract_dir, verbose=True)
        msolve_bin = msolve_bin or os.getenv("MSOLVE_BIN") or shutil.which("msolve")

    print("msolve:", msolve_bin or "not found")
    if not msolve_bin or not Path(msolve_bin).exists():
        raise SystemExit("No usable msolve path found.")

    A, B, C = generate_k_digit_matrices(args.k_digits, seed=args.seed)
    eqs = wedge_equations_3eq(A, B, C)
    out = run_msolve_baseline(
        eqs,
        msolve_bin=msolve_bin,
        keep_input=args.keep_input,
        output_dir=args.output_dir or None,
    )

    print("seed:", args.seed)
    print("ran:", out.get("ran"))
    print("returncode:", out.get("returncode"))
    print("elapsed_ms:", None if out.get("elapsed_sec") is None else 1000.0 * out["elapsed_sec"])
    print("cmd:", out.get("cmd"))
    print("parse_ok:", out.get("parse_ok"))
    print("root_box_count:", out.get("root_box_count"))
    root_residuals = root_relative_residuals(eqs, out.get("root_centers", []))
    root_summary = residual_summary(root_residuals, args.root_residual_tol)
    print("root_residual_count:", root_summary["root_residual_count"])
    print("root_max_relative_residual:", root_summary["root_max_relative_residual"])
    print("root_median_relative_residual:", root_summary["root_median_relative_residual"])
    print("root_residual_pass:", root_summary["root_residual_pass"])
    if out.get("input_path"):
        print("input_path:", out.get("input_path"))
    if out.get("output_path"):
        print("output_path:", out.get("output_path"))
    if out.get("output_head") and not args.no_output_head:
        print("\nOUTPUT")
        print(out["output_head"])
    if out.get("stdout_head"):
        print("\nSTDOUT")
        print(out["stdout_head"])
    if out.get("stderr_head"):
        print("\nSTDERR")
        print(out["stderr_head"])

    if out.get("returncode") != 0:
        raise SystemExit("msolve failed; do not use this binary for side-by-side timing.")


if __name__ == "__main__":
    main()
