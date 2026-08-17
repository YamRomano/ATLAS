#!/usr/bin/env python3
"""Install a clean upstream msolve binary in Colab.

The task-3 zip used during development can contain instrumentation and may
segfault on normal msolve input.  This helper builds upstream msolve and keeps
the binary at a stable path for the benchmark notebooks.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def apt_install() -> None:
    run(["apt-get", "update", "-qq"])
    run([
        "apt-get",
        "install",
        "-y",
        "-qq",
        "build-essential",
        "git",
        "autoconf",
        "automake",
        "libtool",
        "pkg-config",
        "libflint-dev",
        "libgmp-dev",
        "libmpfr-dev",
    ])


def build_from_source(src_dir: Path, prefix: Path, ref: str, jobs: int) -> Path:
    if src_dir.exists():
        shutil.rmtree(src_dir)
    if prefix.exists():
        shutil.rmtree(prefix)

    run(["git", "clone", "https://github.com/algebraic-solving/msolve.git", str(src_dir)])
    run(["git", "fetch", "--tags", "--force"], cwd=src_dir)
    run(["git", "checkout", ref], cwd=src_dir)
    run(["git", "submodule", "update", "--init", "--recursive"], cwd=src_dir)

    env = os.environ.copy()
    env.setdefault("MAKEFLAGS", f"-j{jobs}")
    run(["bash", "autogen.sh"], cwd=src_dir, env=env)
    run(["bash", "configure", f"--prefix={prefix}"], cwd=src_dir, env=env)
    run(["make", f"-j{jobs}"], cwd=src_dir, env=env)
    run(["make", "install"], cwd=src_dir, env=env)

    bin_path = prefix / "bin" / "msolve"
    if not bin_path.exists():
        raise FileNotFoundError(f"msolve binary was not created: {bin_path}")
    return bin_path


def smoke_test(msolve_bin: Path) -> None:
    inp = Path("/tmp/msolve_smoke.ms")
    inp.write_text("x\n0\nx^2 - 2\n", encoding="utf-8")
    out = subprocess.run(
        [str(msolve_bin), "-f", str(inp), "-p", "53"],
        text=True,
        capture_output=True,
        timeout=20,
    )
    try:
        inp.unlink()
    except OSError:
        pass
    if out.returncode != 0:
        raise RuntimeError((out.stderr or out.stdout)[:2000])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["source"], default="source")
    ap.add_argument("--ref", default="v0.8.0")
    ap.add_argument("--fallback-ref", default="master")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--prefix", default="/content/msolve_clean")
    ap.add_argument("--fallback-prefix", default="/content/msolve_clean_fallback")
    ap.add_argument("--src-dir", default="/content/msolve_clean_src")
    ap.add_argument("--fallback-src-dir", default="/content/msolve_clean_fallback_src")
    ap.add_argument("--skip-apt", action="store_true")
    args = ap.parse_args()

    if not args.skip_apt:
        apt_install()

    attempts = [
        (Path(args.src_dir), Path(args.prefix), args.ref),
        (Path(args.fallback_src_dir), Path(args.fallback_prefix), args.fallback_ref),
    ]
    last_error: Exception | None = None
    for src, prefix, ref in attempts:
        try:
            bin_path = build_from_source(src, prefix, ref, args.jobs)
            smoke_test(bin_path)
            print()
            print("MSOLVE_BIN:", bin_path)
            return
        except Exception as exc:
            last_error = exc
            print()
            print(f"Build failed for {ref}: {exc}")
            print()

    raise SystemExit(f"No usable clean msolve build. Last error: {last_error}")


if __name__ == "__main__":
    main()
