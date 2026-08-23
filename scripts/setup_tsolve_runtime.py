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


def ensure_harness_dependencies(base_yam_code_dir: Path, harness: Path) -> None:
    """Complete the generated harness with its non-patch runtime modules.

    The drop-in harness imports ``ysolve_template_core`` but that module lives
    beside the base Yam ``yam_code`` directory, not in the drop-in patch.  A
    runtime assembled only from the reduced vendored harness therefore passed
    file-copy setup and then crashed on its first import, which made the app
    tear down an otherwise healthy DJI bridge.  Copy only the missing base
    dependency so patch files and explicit harness overrides keep precedence.
    """
    dependency_name = "ysolve_template_core.py"
    dependency = harness / dependency_name
    if dependency.exists():
        return
    source = base_yam_code_dir.parent / "harness" / dependency_name
    if not source.is_file():
        raise FileNotFoundError(
            "TSolve runtime requires ysolve_template_core.py; expected it at "
            f"{source} or in the configured base harness"
        )
    harness.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dependency)


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
    ensure_harness_dependencies(args.base_yam_code_dir, harness)

    print("TSolve runtime:")
    print("  yam_code:", yam_code)
    print("  harness :", harness)


if __name__ == "__main__":
    main()
