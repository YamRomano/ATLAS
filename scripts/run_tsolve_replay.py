#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


def run(cmd: list[object]) -> None:
    cmd = [str(x) for x in cmd]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def manifest_count(inputs_dir: Path) -> int:
    with (inputs_dir / "manifest.csv").open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required=True, type=Path)
    ap.add_argument("--runtime-dir", required=True, type=Path)
    ap.add_argument("--solver-dir", required=True, type=Path)
    ap.add_argument("--inputs-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--count", type=int, default=0)
    ap.add_argument("--train-count", type=int, default=1)
    ap.add_argument("--prime", type=int, default=2147483647)
    ap.add_argument("--degree", type=int, default=11)
    ap.add_argument("--action-weights", default="branch")
    ap.add_argument("--fallback-action-weights", default="")
    ap.add_argument("--skip-baseline", action="store_true", default=True)
    args = ap.parse_args()

    count = args.count or manifest_count(args.inputs_dir)
    if count <= args.train_count:
        raise RuntimeError(f"Need more than train-count cases. count={count}, train={args.train_count}")

    harness = args.runtime_dir / "harness"
    yam_code = args.runtime_dir / "yam_code"
    script = harness / "run_fares_static_c_persistent_original_inputs.py"
    out = args.out_dir
    cmd = [
        args.python,
        script,
        "--solver-dir",
        args.solver_dir,
        "--yam-code-dir",
        yam_code,
        "--msolve-bin",
        "/usr/bin/false",
        "--inputs-dir",
        args.inputs_dir,
        "--out",
        out,
        "--count",
        count,
        "--train-count",
        args.train_count,
        "--prime",
        args.prime,
        "--degree",
        args.degree,
        "--action-weights",
        args.action_weights,
        "--fallback-action-weights",
        args.fallback_action_weights,
        "--skip-baseline",
    ]
    run(cmd)
    summary = json.loads((out / "persistent_summary.json").read_text(encoding="utf-8"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
