#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def copy_files(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.is_file():
            shutil.copy2(p, dst / p.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-yam-code-dir", required=True, type=Path)
    ap.add_argument("--dropin-patch-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--base-harness-dir", default="", type=Path)
    args = ap.parse_args()

    out = args.out_dir
    if out.exists():
        shutil.rmtree(out)
    yam_code = out / "yam_code"
    harness = out / "harness"
    copy_files(args.base_yam_code_dir, yam_code)

    base_harness = args.base_harness_dir
    if str(base_harness) and base_harness.exists():
        harness.mkdir(parents=True, exist_ok=True)
        for src in sorted(base_harness.glob("*.py")):
            shutil.copy2(src, harness / src.name)

    copy_files(args.dropin_patch_dir / "yam_code", yam_code)
    copy_files(args.dropin_patch_dir / "harness", harness)

    print("TSolve runtime:")
    print("  yam_code:", yam_code)
    print("  harness :", harness)


if __name__ == "__main__":
    main()
