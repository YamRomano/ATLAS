#!/usr/bin/env python3
"""Offline builder for the PnP symbolic-template tree.

Upload this file to Colab together with ``pnp_symbolic_tree_online.py`` and
``task3_msolve.zip``.  Run this file once to build a reusable symbolic branch
tree artifact, then run the online file repeatedly.

The tree stores the fixed PnP wedge-equation structure and a Macaulay pivot
branch.  It does not call msolve to build the template; msolve is only detected
so the online runner can compare against the Fares-style baseline.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

FREE_VARS: Tuple[str, str, str] = ("q2", "q3", "q4")
Exp4 = Tuple[int, int, int, int]
Exp3 = Tuple[int, int, int]
Poly4 = Dict[Exp4, Any]
Poly3 = Dict[Exp3, Any]
_C_ROOT_REFINE_LIB: Any = None
_C_ROOT_REFINE_PATH: Optional[str] = None


# ---------------------------------------------------------------------------
# Optional msolve setup
# ---------------------------------------------------------------------------


def setup_msolve_from_zip(
    zip_path: str | Path = "/content/task3_msolve.zip",
    extract_dir: str | Path = "/content/task3_msolve_unzipped",
    install_flint: bool = True,
    verbose: bool = True,
) -> Optional[str]:
    """Expose an uploaded task3_msolve.zip binary in Colab.

    Returns a usable msolve path when possible.  If dependencies are missing and
    ``install_flint`` is true, the function attempts to install ``libflint-dev``.
    """

    zip_path = Path(zip_path)
    extract_dir = Path(extract_dir)
    if not zip_path.exists():
        existing = os.getenv("MSOLVE_BIN") or shutil.which("msolve")
        if existing and Path(existing).exists():
            return str(existing)
        return None

    if not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    root = extract_dir / "task3_msolve"
    if not root.exists():
        root = extract_dir

    for p in root.rglob("msolve"):
        if p.is_file():
            os.chmod(p, 0o755)

    lib_dirs = sorted({str(p.parent) for p in root.rglob("*.so*")})
    if lib_dirs:
        os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")

    wrapper = root / "msolve" / "msolve"
    real = root / "msolve" / ".libs" / "msolve"
    candidate = wrapper if wrapper.exists() else real if real.exists() else None
    if candidate is None:
        return None

    os.chmod(candidate, 0o755)
    os.environ["MSOLVE_BIN"] = str(candidate)

    smoke = subprocess.run(
        [str(candidate), "-h"],
        capture_output=True,
        text=True,
        timeout=10,
        env=os.environ.copy(),
    )
    msg = smoke.stdout + smoke.stderr
    if install_flint and ("libflint" in msg.lower() or "shared libraries" in msg.lower()):
        if verbose:
            print("Installing FLINT runtime for msolve...")
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        subprocess.run(["apt-get", "install", "-y", "-qq", "libflint-dev"], check=True)
        smoke = subprocess.run(
            [str(candidate), "-h"],
            capture_output=True,
            text=True,
            timeout=10,
            env=os.environ.copy(),
        )

    if verbose and smoke.returncode not in (0, 1):
        print("msolve smoke-test return code:", smoke.returncode)
        print((smoke.stdout or smoke.stderr)[:1000])

    return str(candidate)


# ---------------------------------------------------------------------------
# Sparse polynomial arithmetic
# ---------------------------------------------------------------------------


def clean(poly: Mapping[Tuple[int, ...], Any]) -> Dict[Tuple[int, ...], Any]:
    return {tuple(k): v for k, v in poly.items() if v != 0}


def add(a: Mapping[Tuple[int, ...], Any], b: Mapping[Tuple[int, ...], Any], scale_b: Any = 1) -> Dict[Tuple[int, ...], Any]:
    out = dict(a)
    for exp, coeff in b.items():
        out[tuple(exp)] = out.get(tuple(exp), 0) + scale_b * coeff
    return clean(out)


def scale(a: Mapping[Tuple[int, ...], Any], s: Any) -> Dict[Tuple[int, ...], Any]:
    if s == 0:
        return {}
    return clean({tuple(exp): s * coeff for exp, coeff in a.items()})


def mul(a: Mapping[Tuple[int, ...], Any], b: Mapping[Tuple[int, ...], Any]) -> Dict[Tuple[int, ...], Any]:
    out: Dict[Tuple[int, ...], Any] = {}
    for ea, ca in a.items():
        for eb, cb in b.items():
            exp = tuple(x + y for x, y in zip(ea, eb))
            out[exp] = out.get(exp, 0) + ca * cb
    return clean(out)


def derivative(a: Mapping[Exp4, Any], var: int) -> Poly4:
    out: Poly4 = {}
    for exp, coeff in a.items():
        power = exp[var]
        if power == 0:
            continue
        e = list(exp)
        e[var] -= 1
        out[tuple(e)] = out.get(tuple(e), 0) + coeff * power
    return clean(out)


def var4(i: int) -> Poly4:
    exp = [0, 0, 0, 0]
    exp[i] = 1
    return {tuple(exp): 1}


def dehom_q1(poly: Mapping[Exp4, Any]) -> Poly3:
    out: Poly3 = {}
    for (_q1, q2, q3, q4), coeff in poly.items():
        exp = (q2, q3, q4)
        out[exp] = out.get(exp, 0) + coeff
    return clean(out)


def add_exp(a: Exp3, b: Exp3) -> Exp3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def eval_poly3(poly: Mapping[Exp3, Any], x: complex, y: complex, z: complex) -> complex:
    val = 0j
    for (a, b, c), coeff in poly.items():
        val += complex(coeff) * (x**a) * (y**b) * (z**c)
    return val


def derivative3(poly: Mapping[Exp3, Any], var: int) -> Poly3:
    out: Poly3 = {}
    for exp, coeff in poly.items():
        power = exp[var]
        if power == 0:
            continue
        e = list(exp)
        e[var] -= 1
        out[tuple(e)] = out.get(tuple(e), 0) + coeff * power
    return clean(out)


def equation_norms(eqs: Sequence[Mapping[Exp3, Any]]) -> list[float]:
    return [max(1.0, float(sum(abs(complex(c)) for c in eq.values()))) for eq in eqs]


def eval_equations_and_jacobian(
    eqs: Sequence[Mapping[Exp3, Any]],
    derivs: Sequence[Sequence[Mapping[Exp3, Any]]],
    root: Sequence[complex],
) -> Tuple[np.ndarray, np.ndarray]:
    x, y, z = complex(root[0]), complex(root[1]), complex(root[2])
    f = np.array([eval_poly3(eq, x, y, z) for eq in eqs], dtype=np.complex128)
    J = np.array(
        [[eval_poly3(derivs[i][j], x, y, z) for j in range(3)] for i in range(3)],
        dtype=np.complex128,
    )
    return f, J


def newton_refine_root(
    eqs: Sequence[Mapping[Exp3, Any]],
    derivs: Sequence[Sequence[Mapping[Exp3, Any]]],
    norms: Sequence[float],
    root0: Sequence[complex],
    max_iter: int = 40,
    target_rel_tol: float = 1e-10,
) -> Tuple[Tuple[complex, complex, complex], float, bool, int]:
    r = np.array([complex(root0[0]), complex(root0[1]), complex(root0[2])], dtype=np.complex128)
    best = r.copy()
    best_rel = float("inf")

    for it in range(1, max_iter + 1):
        f, J = eval_equations_and_jacobian(eqs, derivs, r)
        rel = max(abs(f[i]) / max(1.0, float(norms[i])) for i in range(3))
        if np.isfinite(rel) and float(rel) < best_rel:
            best_rel = float(rel)
            best = r.copy()
        if rel <= target_rel_tol:
            return (complex(r[0]), complex(r[1]), complex(r[2])), float(rel), True, it

        try:
            step = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            try:
                step = np.linalg.lstsq(J, -f, rcond=None)[0]
            except np.linalg.LinAlgError:
                break

        improved = False
        for damping in (1.0, 0.5, 0.25, 0.1, 0.05, 0.01):
            cand = r + damping * step
            fc, _ = eval_equations_and_jacobian(eqs, derivs, cand)
            cand_rel = max(abs(fc[i]) / max(1.0, float(norms[i])) for i in range(3))
            if np.isfinite(cand_rel) and cand_rel < rel:
                r = cand
                improved = True
                break
        if not improved:
            r = r + step

    return (complex(best[0]), complex(best[1]), complex(best[2])), float(best_rel), best_rel <= target_rel_tol, max_iter


def _candidate_c_root_refine_paths(path: Optional[str | Path] = None) -> list[Path]:
    paths: list[Path] = []
    for value in [path, os.getenv("PNP_ROOT_REFINE_LIB")]:
        if value:
            paths.append(Path(value))
    here = Path(__file__).resolve().parent
    paths.extend(
        [
            here / "pnp_root_refine.so",
            here / "pnp_root_refine.dylib",
            Path("/content/pnp_root_refine.so"),
            Path("/content/pnp_symbolic_tree_demo/pnp_root_refine.so"),
        ]
    )
    out: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def load_c_root_refiner(path: Optional[str | Path] = None):
    global _C_ROOT_REFINE_LIB, _C_ROOT_REFINE_PATH
    for candidate in _candidate_c_root_refine_paths(path):
        if not candidate.exists():
            continue
        candidate_str = str(candidate)
        if _C_ROOT_REFINE_LIB is not None and _C_ROOT_REFINE_PATH == candidate_str:
            return _C_ROOT_REFINE_LIB, candidate_str
        lib = ctypes.CDLL(candidate_str)
        lib.pnp_refine_roots.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.pnp_refine_roots.restype = ctypes.c_int
        if hasattr(lib, "pnp_project_actions"):
            lib.pnp_project_actions.argtypes = [
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_int),
            ]
            lib.pnp_project_actions.restype = ctypes.c_int
        if hasattr(lib, "pnp_project_uses_lapack"):
            lib.pnp_project_uses_lapack.argtypes = []
            lib.pnp_project_uses_lapack.restype = ctypes.c_int
        try:
            lib.pnp_refine_roots(
                ctypes.c_int(0),
                None,
                None,
                ctypes.c_int(0),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                ctypes.c_int(0),
                ctypes.c_double(0.0),
                ctypes.c_double(0.0),
                ctypes.c_double(0.0),
                None,
                None,
                None,
                None,
                None,
            )
        except Exception:
            pass
        _C_ROOT_REFINE_LIB = lib
        _C_ROOT_REFINE_PATH = candidate_str
        return lib, candidate_str
    return None, None


def _flatten_equation_terms(eqs: Sequence[Mapping[Exp3, Any]]):
    term_eq: list[int] = []
    exp_a: list[int] = []
    exp_b: list[int] = []
    exp_c: list[int] = []
    coeff_re: list[float] = []
    coeff_im: list[float] = []
    for eq_idx, eq in enumerate(eqs):
        for (a, b, c), coeff in eq.items():
            z = complex(coeff)
            term_eq.append(eq_idx)
            exp_a.append(int(a))
            exp_b.append(int(b))
            exp_c.append(int(c))
            coeff_re.append(float(z.real))
            coeff_im.append(float(z.imag))
    return (
        np.ascontiguousarray(term_eq, dtype=np.int32),
        np.ascontiguousarray(exp_a, dtype=np.int32),
        np.ascontiguousarray(exp_b, dtype=np.int32),
        np.ascontiguousarray(exp_c, dtype=np.int32),
        np.ascontiguousarray(coeff_re, dtype=np.float64),
        np.ascontiguousarray(coeff_im, dtype=np.float64),
    )


def c_refine_candidate_roots(
    eqs: Sequence[Mapping[Exp3, Any]],
    seeds: Sequence[Tuple[complex, complex, complex]],
    residual_tol: float,
    max_newton_iter: int,
    max_abs_root: float,
    root_refine_lib: Optional[str | Path] = None,
) -> Optional[dict[str, Any]]:
    if not seeds:
        return {
            "ok": True,
            "library": None,
            "roots": [],
            "residuals": [],
            "iters": [],
            "elapsed_sec": 0.0,
            "ok_count": 0,
        }
    lib, lib_path = load_c_root_refiner(root_refine_lib)
    if lib is None:
        return None

    t0 = time.perf_counter()
    n = len(seeds)
    seed_arr = np.ascontiguousarray([[complex(z) for z in root] for root in seeds], dtype=np.complex128)
    seed_re = np.ascontiguousarray(seed_arr.real.reshape(-1), dtype=np.float64)
    seed_im = np.ascontiguousarray(seed_arr.imag.reshape(-1), dtype=np.float64)
    term_eq, exp_a, exp_b, exp_c, coeff_re, coeff_im = _flatten_equation_terms(eqs)
    norms = np.ascontiguousarray(equation_norms(eqs), dtype=np.float64)
    out_re = np.zeros(n * 3, dtype=np.float64)
    out_im = np.zeros(n * 3, dtype=np.float64)
    out_res = np.zeros(n, dtype=np.float64)
    out_ok = np.zeros(n, dtype=np.int32)
    out_iters = np.zeros(n, dtype=np.int32)

    ok_count = lib.pnp_refine_roots(
        ctypes.c_int(n),
        seed_re.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        seed_im.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(len(term_eq)),
        term_eq.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        exp_a.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        exp_b.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        exp_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        coeff_re.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        coeff_im.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        norms.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(max_newton_iter),
        ctypes.c_double(min(1e-10, residual_tol * 0.01)),
        ctypes.c_double(residual_tol),
        ctypes.c_double(max_abs_root),
        out_re.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_im.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_res.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_ok.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_iters.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )

    roots: list[Tuple[Tuple[complex, complex, complex], float]] = []
    for i in range(n):
        if int(out_ok[i]):
            root = tuple(complex(out_re[3 * i + j], out_im[3 * i + j]) for j in range(3))
            roots.append((root, float(out_res[i])))

    return {
        "ok": True,
        "library": lib_path,
        "roots": roots,
        "residuals": [float(x) for x in out_res.tolist()],
        "iters": [int(x) for x in out_iters.tolist()],
        "elapsed_sec": time.perf_counter() - t0,
        "ok_count": int(ok_count),
    }


def refine_candidate_roots(
    eqs: Sequence[Mapping[Exp3, Any]],
    seeds: Sequence[Tuple[complex, complex, complex]],
    residual_tol: float,
    max_newton_iter: int,
    max_abs_root: float,
    backend: str = "auto",
    root_refine_lib: Optional[str | Path] = None,
) -> tuple[list[Tuple[Tuple[complex, complex, complex], float]], dict[str, Any]]:
    backend = str(backend or "auto").lower()
    if backend not in {"auto", "c", "python"}:
        raise ValueError(f"unknown root refinement backend: {backend}")

    if backend in {"auto", "c"}:
        c_out = c_refine_candidate_roots(
            eqs,
            seeds,
            residual_tol=residual_tol,
            max_newton_iter=max_newton_iter,
            max_abs_root=max_abs_root,
            root_refine_lib=root_refine_lib,
        )
        if c_out is not None:
            return c_out["roots"], {
                "backend": "c",
                "library": c_out.get("library"),
                "elapsed_sec": c_out.get("elapsed_sec", 0.0),
                "ok_count": c_out.get("ok_count", 0),
                "iters": c_out.get("iters", []),
            }
        if backend == "c":
            raise FileNotFoundError("C root refinement library was requested but not found")

    derivs = [[derivative3(eq, j) for j in range(3)] for eq in eqs]
    norms = equation_norms(eqs)
    roots: list[Tuple[Tuple[complex, complex, complex], float]] = []
    t0 = time.perf_counter()
    for seed_root in seeds:
        root, rel, _ok, _iters = newton_refine_root(
            eqs,
            derivs,
            norms,
            seed_root,
            max_iter=max_newton_iter,
            target_rel_tol=min(1e-10, residual_tol * 0.01),
        )
        if rel <= residual_tol and max(abs(z) for z in root) <= max_abs_root:
            roots.append((root, rel))
    return roots, {
        "backend": "python",
        "library": None,
        "elapsed_sec": time.perf_counter() - t0,
        "ok_count": len(roots),
        "iters": [],
    }


# ---------------------------------------------------------------------------
# PnP wedge equations
# ---------------------------------------------------------------------------


def rotation_columns_poly4() -> Tuple[list[Poly4], list[Poly4], list[Poly4]]:
    q1, q2, q3, q4 = [var4(i) for i in range(4)]
    q1q1, q2q2, q3q3, q4q4 = [mul(q, q) for q in [q1, q2, q3, q4]]
    q1q2, q1q3, q1q4 = mul(q1, q2), mul(q1, q3), mul(q1, q4)
    q2q3, q2q4, q3q4 = mul(q2, q3), mul(q2, q4), mul(q3, q4)

    def s(*parts: Tuple[Any, Poly4]) -> Poly4:
        out: Poly4 = {}
        for alpha, p in parts:
            out = add(out, scale(p, alpha))
        return out

    col_x = [
        s((1, q1q1), (1, q2q2), (-1, q3q3), (-1, q4q4)),
        s((2, q2q3), (2, q1q4)),
        s((2, q2q4), (-2, q1q3)),
    ]
    col_y = [
        s((2, q2q3), (-2, q1q4)),
        s((1, q1q1), (-1, q2q2), (1, q3q3), (-1, q4q4)),
        s((2, q3q4), (2, q1q2)),
    ]
    col_z = [
        s((2, q2q4), (2, q1q3)),
        s((2, q3q4), (-2, q1q2)),
        s((1, q1q1), (-1, q2q2), (-1, q3q3), (1, q4q4)),
    ]
    return col_x, col_y, col_z


ROT_COLS = rotation_columns_poly4()


def matrix3(M: Sequence[Sequence[Any]]) -> list[list[Any]]:
    rows = [list(r) for r in M]
    if len(rows) != 3 or any(len(r) != 3 for r in rows):
        raise ValueError("expected a 3x3 matrix")
    return rows


def pnp_objective_poly4(A: Sequence[Sequence[Any]], B: Sequence[Sequence[Any]], C: Sequence[Sequence[Any]]) -> Poly4:
    A, B, C = matrix3(A), matrix3(B), matrix3(C)
    col_x, col_y, col_z = ROT_COLS
    obj: Poly4 = {}
    for i in range(3):
        residual: Poly4 = {}
        for j in range(3):
            residual = add(residual, scale(col_x[j], A[i][j]))
            residual = add(residual, scale(col_y[j], B[i][j]))
            residual = add(residual, scale(col_z[j], C[i][j]))
        obj = add(obj, mul(residual, residual))
    return obj


def normalized_rotation_columns_from_root(root: Sequence[float | complex]) -> Tuple[list[complex], list[complex], list[complex]]:
    """Return normalized rotation columns for q=(1,q2,q3,q4)."""
    q2, q3, q4 = (complex(root[0]), complex(root[1]), complex(root[2]))
    n2 = 1.0 + q2 * q2 + q3 * q3 + q4 * q4
    if abs(n2) < 1e-300:
        raise ZeroDivisionError("quaternion chart norm is zero")
    col_x = [
        (1 + q2 * q2 - q3 * q3 - q4 * q4) / n2,
        (2 * (q2 * q3 + q4)) / n2,
        (2 * (q2 * q4 - q3)) / n2,
    ]
    col_y = [
        (2 * (q2 * q3 - q4)) / n2,
        (1 - q2 * q2 + q3 * q3 - q4 * q4) / n2,
        (2 * (q3 * q4 + q2)) / n2,
    ]
    col_z = [
        (2 * (q2 * q4 + q3)) / n2,
        (2 * (q3 * q4 - q2)) / n2,
        (1 - q2 * q2 - q3 * q3 + q4 * q4) / n2,
    ]
    return col_x, col_y, col_z


def pnp_score_root(
    A: Sequence[Sequence[Any]],
    B: Sequence[Sequence[Any]],
    C: Sequence[Sequence[Any]],
    root: Sequence[float | complex],
) -> float:
    """Score one dehomogenized q=(1,q2,q3,q4) root by normalized PnP loss."""
    A3, B3, C3 = matrix3(A), matrix3(B), matrix3(C)
    col_x, col_y, col_z = normalized_rotation_columns_from_root(root)
    score = 0j
    for i in range(3):
        residual = 0j
        for j in range(3):
            residual += complex(A3[i][j]) * col_x[j]
            residual += complex(B3[i][j]) * col_y[j]
            residual += complex(C3[i][j]) * col_z[j]
        score += residual * residual.conjugate()
    return float(score.real)


def best_root_by_score(
    A: Sequence[Sequence[Any]],
    B: Sequence[Sequence[Any]],
    C: Sequence[Sequence[Any]],
    roots: Sequence[Sequence[float | complex]],
) -> dict[str, Any]:
    """Return the lowest-scoring root and score from a list of real roots."""
    best_root = None
    best_score = None
    scored = 0
    for root in roots:
        if len(root) < 3:
            continue
        try:
            score = pnp_score_root(A, B, C, root)
        except Exception:
            continue
        if not np.isfinite(score):
            continue
        scored += 1
        if best_score is None or score < best_score:
            best_score = float(score)
            best_root = [float(complex(root[i]).real) for i in range(3)]
    return {
        "best_root": best_root,
        "best_score": best_score,
        "scored_count": scored,
    }


def wedge_equations_3eq(
    A: Sequence[Sequence[Any]],
    B: Sequence[Sequence[Any]],
    C: Sequence[Sequence[Any]],
    pairs: Sequence[Tuple[int, int]] = ((0, 1), (0, 2), (0, 3)),
) -> list[Poly3]:
    obj = pnp_objective_poly4(A, B, C)
    grad = [derivative(obj, i) for i in range(4)]
    q = [var4(i) for i in range(4)]
    eqs: list[Poly3] = []
    for i, j in pairs:
        eq4 = add(mul(q[i], grad[j]), mul(q[j], grad[i]), scale_b=-1)
        eqs.append(dehom_q1(eq4))
    return eqs


def monomials_upto_degree(D: int) -> Tuple[Exp3, ...]:
    mons: list[Exp3] = []
    for total in range(D, -1, -1):
        for a in range(total, -1, -1):
            for b in range(total - a, -1, -1):
                c = total - a - b
                mons.append((a, b, c))
    return tuple(mons)


def generate_k_digit_matrices(k: int = 2, seed: int = 0) -> Tuple[list[list[int]], list[list[int]], list[list[int]]]:
    rng = random.Random(seed)
    lo = 0 if k <= 1 else 10 ** (k - 1)
    hi = 10**k - 1

    def one() -> list[list[int]]:
        return [[rng.choice([-1, 1]) * rng.randint(lo, hi) for _ in range(3)] for _ in range(3)]

    return one(), one(), one()


def parse_seeds(text: str) -> list[int]:
    """Parse seeds from "42:52", "42:52:2", or "42,43,44"."""
    text = str(text).strip()
    if not text:
        return []
    if ":" in text and "," not in text:
        parts = [int(x) for x in text.split(":")]
        if len(parts) == 2:
            return list(range(parts[0], parts[1]))
        if len(parts) == 3:
            return list(range(parts[0], parts[1], parts[2]))
        raise ValueError(f"bad seed range: {text}")
    return [int(x.strip()) for x in text.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# msolve baseline formatting
# ---------------------------------------------------------------------------


def to_fraction(x: Any) -> Fraction:
    if isinstance(x, Fraction):
        return x
    return Fraction(x).limit_denominator(10**9)


def fmt_num(x: Any) -> str:
    q = to_fraction(x)
    return str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


def poly_to_msolve(poly: Mapping[Exp3, Any], var_names: Sequence[str] = FREE_VARS) -> str:
    terms = []
    for exp in sorted(poly.keys(), key=lambda e: (sum(e), e), reverse=True):
        coeff = to_fraction(poly[exp])
        if coeff == 0:
            continue
        sign = "-" if coeff < 0 else "+"
        coeff = abs(coeff)
        mon = "*".join(v if p == 1 else f"{v}^{p}" for v, p in zip(var_names, exp) if p > 0)
        body = (mon if coeff == 1 else f"{fmt_num(coeff)}*{mon}") if mon else fmt_num(coeff)
        terms.append((sign, body))
    if not terms:
        return "0"
    first_sign, first_body = terms[0]
    s = ("-" if first_sign == "-" else "") + first_body
    for sign, body in terms[1:]:
        s += f" {sign} {body}"
    return s


def msolve_input_text(eqs: Sequence[Mapping[Exp3, Any]]) -> str:
    return ",".join(FREE_VARS) + "\n0\n" + ",\n".join(poly_to_msolve(e) for e in eqs) + "\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_msolve_number(token: str) -> float:
    token = token.strip()
    if "/" not in token:
        return float(int(token))
    num_s, den_s = token.split("/", 1)
    num = int(num_s.strip())
    den_s = den_s.strip()
    if den_s.startswith("2^"):
        den = 2 ** int(den_s[2:])
    else:
        den = int(den_s)
    return float(Fraction(num, den))


def parse_msolve_nested_lists(text: str) -> Any:
    """Parse the list-style real-root output emitted by clean msolve.

    This supports integers and intervals written as ``a / 2^k``.  It is meant
    for accuracy summaries, not as a replacement for exact certified parsing.
    """

    import re

    toks = re.findall(r"-?\d+\s*/\s*2\^\d+|-?\d+\s*/\s*-?\d+|-?\d+|\[|\]|,|:", text)
    pos = 0

    def parse_value():
        nonlocal pos
        if pos >= len(toks):
            raise ValueError("unexpected end of msolve output")
        tok = toks[pos]
        if tok == "[":
            pos += 1
            arr = []
            while pos < len(toks) and toks[pos] != "]":
                if toks[pos] in {",", ":"}:
                    pos += 1
                    continue
                arr.append(parse_value())
            if pos >= len(toks):
                raise ValueError("unterminated msolve list")
            pos += 1
            return arr
        if tok in {"]", ",", ":"}:
            raise ValueError(f"unexpected token {tok!r}")
        pos += 1
        return parse_msolve_number(tok)

    while pos < len(toks) and toks[pos] != "[":
        pos += 1
    if pos >= len(toks):
        raise ValueError("no list found in msolve output")
    return parse_value()


def msolve_root_boxes(parsed: Any, nvars: int = 3) -> list[list[list[float]]]:
    def is_interval(v: Any) -> bool:
        return isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v)

    def is_box(v: Any) -> bool:
        return isinstance(v, list) and len(v) == nvars and all(is_interval(x) for x in v)

    out: list[list[list[float]]] = []

    def rec(v: Any) -> None:
        if is_box(v):
            out.append(v)
        elif isinstance(v, list):
            for x in v:
                rec(x)

    rec(parsed)
    return out


def root_centers_from_boxes(boxes: Sequence[Sequence[Sequence[float]]]) -> list[list[float]]:
    return [
        [0.5 * (float(interval[0]) + float(interval[1])) for interval in box]
        for box in boxes
    ]


def root_relative_residuals(
    eqs: Sequence[Mapping[Exp3, Any]],
    roots: Sequence[Sequence[float | complex]],
) -> list[float]:
    norms = [max(1.0, float(sum(abs(complex(c)) for c in eq.values()))) for eq in eqs]
    out: list[float] = []
    for root in roots:
        if len(root) < 3:
            continue
        x, y, z = complex(root[0]), complex(root[1]), complex(root[2])
        rel = max(abs(eval_poly3(eq, x, y, z)) / norm for eq, norm in zip(eqs, norms))
        if np.isfinite(rel):
            out.append(float(rel))
    return out


def residual_summary(values: Sequence[float], tol: float) -> dict[str, Any]:
    vals = [float(x) for x in values if np.isfinite(float(x))]
    if not vals:
        return {
            "root_residual_count": 0,
            "root_max_relative_residual": None,
            "root_median_relative_residual": None,
            "root_residual_pass": False,
        }
    return {
        "root_residual_count": len(vals),
        "root_max_relative_residual": float(max(vals)),
        "root_median_relative_residual": float(np.median(vals)),
        "root_residual_pass": bool(max(vals) <= tol),
    }


def deduplicate_roots(
    roots: Sequence[Tuple[Tuple[complex, complex, complex], float]],
    distance_tol: float = 1e-6,
) -> list[Tuple[Tuple[complex, complex, complex], float]]:
    out: list[Tuple[Tuple[complex, complex, complex], float]] = []
    for root, residual in sorted(roots, key=lambda x: float(x[1])):
        rv = np.array(root, dtype=np.complex128)
        if all(np.linalg.norm(rv - np.array(old, dtype=np.complex128)) > distance_tol for old, _ in out):
            out.append((root, float(residual)))
    return out


def complex_root_to_json(root: Sequence[complex]) -> list[list[float]]:
    return [[float(complex(z).real), float(complex(z).imag)] for z in root]


def real_root_centers_from_complex(
    roots: Sequence[Tuple[complex, complex, complex]],
    imag_tol: float = 1e-7,
) -> list[list[float]]:
    out = []
    for root in roots:
        if max(abs(complex(z).imag) for z in root) <= imag_tol:
            out.append([float(complex(z).real) for z in root])
    return out


def nearest_root_distances(
    source: Sequence[Sequence[float | complex]],
    target: Sequence[Sequence[float | complex]],
) -> list[float]:
    if not source or not target:
        return []
    target_arr = [np.array([complex(z) for z in r], dtype=np.complex128) for r in target]
    out = []
    for root in source:
        rv = np.array([complex(z) for z in root], dtype=np.complex128)
        out.append(float(min(np.linalg.norm(rv - tv) for tv in target_arr)))
    return out


def root_match_summary(
    tree_real_roots: Sequence[Sequence[float]],
    msolve_roots: Sequence[Sequence[float]],
    match_tol: float,
) -> dict[str, Any]:
    tree_to_ms = nearest_root_distances(tree_real_roots, msolve_roots)
    ms_to_tree = nearest_root_distances(msolve_roots, tree_real_roots)
    return {
        "tree_to_msolve_match_count": sum(1 for d in tree_to_ms if d <= match_tol),
        "tree_to_msolve_match_rate": 100.0 * sum(1 for d in tree_to_ms if d <= match_tol) / len(tree_to_ms) if tree_to_ms else 0.0,
        "msolve_to_tree_match_count": sum(1 for d in ms_to_tree if d <= match_tol),
        "msolve_to_tree_match_rate": 100.0 * sum(1 for d in ms_to_tree if d <= match_tol) / len(ms_to_tree) if ms_to_tree else 0.0,
        "tree_to_msolve_max_distance": max(tree_to_ms) if tree_to_ms else None,
        "msolve_to_tree_max_distance": max(ms_to_tree) if ms_to_tree else None,
        "root_match_pass": bool(tree_to_ms and ms_to_tree and all(d <= match_tol for d in tree_to_ms) and all(d <= match_tol for d in ms_to_tree)),
    }


def action_seed_roots_from_macaulay(
    eqs: Sequence[Mapping[Exp3, Any]],
    degree: int = 11,
    rank_tol: float = 1e-8,
    finite_rank_tol: float = 1e-8,
    random_actions: int = 12,
    random_seed: int = 0,
) -> Tuple[list[Tuple[complex, complex, complex]], dict[str, Any]]:
    """Build approximate root seeds from a numerical Macaulay nullspace.

    This is an experimental root-output layer.  The fast production path remains
    the stored pivot replay certificate; this helper is used to test whether the
    same Macaulay data can also produce candidate roots online.
    """

    t0 = time.perf_counter()
    columns, row_specs = build_macaulay_layout(degree)
    M = assemble_macaulay(eqs, columns, row_specs)
    svd_t0 = time.perf_counter()
    _U, S, Vh = np.linalg.svd(M, full_matrices=True)
    svd_sec = time.perf_counter() - svd_t0
    if len(S) == 0:
        return [], {"reason": "empty Macaulay matrix", "degree": degree}

    rank = int(np.sum(S > rank_tol * max(float(S[0]), 1.0)))
    Z = Vh.conj().T[:, rank:]
    if Z.shape[1] == 0:
        return [], {
            "reason": "no numerical nullspace",
            "degree": degree,
            "rank": rank,
            "svd_sec": svd_sec,
        }

    finite_rows = [i for i, mon in enumerate(columns) if sum(mon) <= degree - 1]
    _Uc, Sc, Vhc = np.linalg.svd(Z[finite_rows, :], full_matrices=False)
    finite_rank = int(np.sum(Sc > finite_rank_tol * max(float(Sc[0]), 1.0))) if len(Sc) else 0
    if finite_rank <= 0:
        return [], {
            "reason": "no finite-rank component",
            "degree": degree,
            "rank": rank,
            "nullity": int(Z.shape[1]),
            "svd_sec": svd_sec,
        }

    Zf = Z @ Vhc.conj().T[:, :finite_rank]
    R = Zf[finite_rows, :].copy()
    selected_local: list[int] = []
    selected_global: list[int] = []
    for _ in range(finite_rank):
        norms = np.linalg.norm(R, axis=1)
        for idx in selected_local:
            norms[idx] = -1.0
        j = int(np.argmax(norms))
        if norms[j] <= 1e-12:
            break
        selected_local.append(j)
        selected_global.append(finite_rows[j])
        q = R[j] / max(float(norms[j]), 1e-300)
        R -= np.outer(R @ q.conj(), q)

    if len(selected_global) != finite_rank:
        return [], {
            "reason": "could not select a full finite action basis",
            "degree": degree,
            "rank": rank,
            "nullity": int(Z.shape[1]),
            "finite_rank": finite_rank,
            "selected": len(selected_global),
            "svd_sec": svd_sec,
        }

    ZB = Zf[selected_global, :]
    try:
        inv_ZB = np.linalg.inv(ZB)
        basis_cond = float(np.linalg.cond(ZB))
    except np.linalg.LinAlgError:
        return [], {
            "reason": "finite action basis is singular",
            "degree": degree,
            "rank": rank,
            "nullity": int(Z.shape[1]),
            "finite_rank": finite_rank,
            "svd_sec": svd_sec,
        }

    col_index = {mon: i for i, mon in enumerate(columns)}
    actions = []
    for unit in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        targets = []
        for idx in selected_global:
            target = add_exp(columns[idx], unit)
            if target not in col_index:
                return [], {
                    "reason": "action target outside Macaulay columns",
                    "missing_target": target,
                    "degree": degree,
                    "rank": rank,
                    "finite_rank": finite_rank,
                    "svd_sec": svd_sec,
                }
            targets.append(col_index[target])
        actions.append(Zf[targets, :] @ inv_ZB)

    rng = random.Random(random_seed)
    action_weights = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    action_weights.extend((rng.random(), rng.random(), rng.random()) for _ in range(max(0, random_actions)))

    seeds: list[Tuple[complex, complex, complex]] = []
    eig_failures = 0
    eig_t0 = time.perf_counter()
    for weights in action_weights:
        T = weights[0] * actions[0] + weights[1] * actions[1] + weights[2] * actions[2]
        try:
            _vals, V = np.linalg.eig(T)
            try:
                Vinv = np.linalg.inv(V)
            except np.linalg.LinAlgError:
                Vinv = np.linalg.pinv(V)
            diag_actions = [Vinv @ A @ V for A in actions]
        except np.linalg.LinAlgError:
            eig_failures += 1
            continue
        for i in range(finite_rank):
            seeds.append(
                (
                    complex(diag_actions[0][i, i]),
                    complex(diag_actions[1][i, i]),
                    complex(diag_actions[2][i, i]),
                )
            )

    info = {
        "reason": "action seeds built",
        "degree": degree,
        "rank": rank,
        "nullity": int(Z.shape[1]),
        "finite_rank": finite_rank,
        "basis_condition": basis_cond,
        "seed_count": len(seeds),
        "eig_failures": eig_failures,
        "svd_sec": svd_sec,
        "eig_sec": time.perf_counter() - eig_t0,
        "seed_total_sec": time.perf_counter() - t0,
    }
    return seeds, info


def tree_roots_from_macaulay(
    eqs: Sequence[Mapping[Exp3, Any]],
    degree: int = 11,
    random_actions: int = 12,
    random_seed: int = 0,
    residual_tol: float = 1e-8,
    dedup_tol: float = 1e-6,
    real_imag_tol: float = 1e-7,
    max_newton_iter: int = 40,
    max_abs_root: float = 500.0,
    root_refine_backend: str = "auto",
    root_refine_lib: Optional[str | Path] = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    seeds, seed_info = action_seed_roots_from_macaulay(
        eqs,
        degree=degree,
        random_actions=random_actions,
        random_seed=random_seed,
    )
    newton_t0 = time.perf_counter()
    roots, refine_info = refine_candidate_roots(
        eqs,
        seeds,
        residual_tol=residual_tol,
        max_newton_iter=max_newton_iter,
        max_abs_root=max_abs_root,
        backend=root_refine_backend,
        root_refine_lib=root_refine_lib,
    )
    newton_sec = time.perf_counter() - newton_t0

    roots = deduplicate_roots(roots, distance_tol=dedup_tol)
    root_values = [root for root, _ in roots]
    residuals = [float(rel) for _, rel in roots]
    real_roots = real_root_centers_from_complex(root_values, imag_tol=real_imag_tol)
    return {
        "ok": bool(roots),
        "reason": "roots extracted" if roots else "no roots passed residual filter",
        "seed_info": seed_info,
        "root_count": len(roots),
        "real_root_count": len(real_roots),
        "roots": root_values,
        "real_roots": real_roots,
        "root_residuals": residuals,
        "max_relative_residual": max(residuals) if residuals else None,
        "median_relative_residual": float(np.median(residuals)) if residuals else None,
        "root_sample": [complex_root_to_json(root) for root in root_values[:8]],
        "root_refine_backend": refine_info.get("backend"),
        "root_refine_library": refine_info.get("library"),
        "root_refine_ok_count": refine_info.get("ok_count"),
        "newton_sec": newton_sec,
        "total_sec": time.perf_counter() - t0,
    }


def summarize_msolve_output(text: str, nvars: int = 3) -> dict[str, Any]:
    if not text.strip():
        return {"parse_ok": False, "root_box_count": 0, "parse_error": "empty output"}
    try:
        parsed = parse_msolve_nested_lists(text)
        boxes = msolve_root_boxes(parsed, nvars=nvars)
        centers = root_centers_from_boxes(boxes)
        return {
            "parse_ok": True,
            "root_box_count": len(boxes),
            "root_centers": centers,
            "root_centers_sample": centers[:8],
            "root_boxes": boxes,
        }
    except Exception as exc:
        return {"parse_ok": False, "root_box_count": 0, "parse_error": str(exc)[:240]}


def run_msolve_baseline(
    eqs: Sequence[Mapping[Exp3, Any]],
    msolve_bin: Optional[str] = None,
    timeout: int = 60,
    keep_input: bool = False,
    output_dir: Optional[str | Path] = None,
    nvars: int = 3,
) -> dict[str, Any]:
    msolve_bin = msolve_bin or os.getenv("MSOLVE_BIN") or shutil.which("msolve")
    if not msolve_bin:
        return {"ran": False, "reason": "msolve not found"}
    text = msolve_input_text(eqs)
    output_path = None
    tmp_dir = Path(output_dir) if output_dir else None
    if tmp_dir:
        tmp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".ms", delete=False, dir=str(tmp_dir) if tmp_dir else None) as f:
        f.write(text)
        path = f.name
    if keep_input or output_dir:
        output_path = str(Path(path).with_suffix(".out.ms"))
    cmd = [msolve_bin, "-f", path]
    if output_path:
        cmd += ["-o", output_path]
    cmd += ["-p", "53"]
    t0 = time.perf_counter()
    subprocess_t0 = time.perf_counter()
    try:
        p = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=os.environ.copy(),
        )
    finally:
        if not keep_input:
            os.unlink(path)
    subprocess_sec = time.perf_counter() - subprocess_t0

    read_t0 = time.perf_counter()
    output_text = ""
    if output_path and Path(output_path).exists():
        output_text = Path(output_path).read_text(encoding="utf-8", errors="replace")
    else:
        output_text = p.stdout
    output_read_sec = time.perf_counter() - read_t0

    parse_t0 = time.perf_counter()
    parsed = summarize_msolve_output(output_text, nvars=nvars) if p.returncode == 0 else {
        "parse_ok": False,
        "root_box_count": 0,
        "parse_error": "msolve did not return success",
    }
    parse_sec = time.perf_counter() - parse_t0
    elapsed_sec = time.perf_counter() - t0
    out = {
        "ran": True,
        "elapsed_sec": elapsed_sec,
        "subprocess_sec": subprocess_sec,
        "output_read_sec": output_read_sec,
        "parse_sec": parse_sec,
        "returncode": p.returncode,
        "stdout_head": p.stdout[:500],
        "stderr_head": p.stderr[:500],
        "output_head": output_text[:500],
        "input_sha256": sha256_text(text),
        "input_text": text if keep_input else "",
        "cmd": " ".join(cmd),
        "output_path": output_path,
        **parsed,
    }
    if keep_input:
        out["input_path"] = path
    elif output_path and not output_dir:
        try:
            os.unlink(output_path)
        except OSError:
            pass
    return out


# ---------------------------------------------------------------------------
# Macaulay checkpoint tree
# ---------------------------------------------------------------------------


def build_macaulay_layout(D: int) -> Tuple[Tuple[Exp3, ...], Tuple[Tuple[int, Exp3], ...]]:
    columns = monomials_upto_degree(D)
    multipliers = monomials_upto_degree(D - 4)
    rows = []
    for eq_idx in range(3):
        for m in multipliers:
            rows.append((eq_idx, m))
    return columns, tuple(rows)


def assemble_macaulay(eqs: Sequence[Mapping[Exp3, Any]], columns: Sequence[Exp3], row_specs: Sequence[Tuple[int, Exp3]]) -> np.ndarray:
    idx = {m: i for i, m in enumerate(columns)}
    M = np.zeros((len(row_specs), len(columns)), dtype=np.complex128)
    for r, (eq_idx, mult) in enumerate(row_specs):
        for exp, coeff in eqs[eq_idx].items():
            target = add_exp(mult, exp)
            c = idx.get(target)
            if c is not None:
                M[r, c] += complex(coeff)
    return M


def assemble_macaulay_selected_rows(
    eqs: Sequence[Mapping[Exp3, Any]],
    columns: Sequence[Exp3],
    row_specs: Sequence[Tuple[int, Exp3]],
    selected_rows: Sequence[int],
) -> np.ndarray:
    idx = {m: i for i, m in enumerate(columns)}
    M = np.zeros((len(selected_rows), len(columns)), dtype=np.complex128)
    for out_r, src_r in enumerate(selected_rows):
        eq_idx, mult = row_specs[int(src_r)]
        for exp, coeff in eqs[eq_idx].items():
            target = add_exp(mult, exp)
            c = idx.get(target)
            if c is not None:
                M[out_r, c] += complex(coeff)
    return M


def assemble_macaulay_mod(
    eqs: Sequence[Mapping[Exp3, Any]],
    columns: Sequence[Exp3],
    row_specs: Sequence[Tuple[int, Exp3]],
    prime: int = 2147483647,
) -> tuple[list[dict[int, int]], int]:
    """Assemble the Macaulay matrix sparsely over one prime field.

    This is an offline discovery helper.  Floating complete pivoting gives a
    stable replay block, but it does not preserve the monomial-order structure
    needed for quotient/action roots.  The modular scan below follows the fixed
    monomial order and recovers the low-degree standard monomials.
    """

    idx = {m: i for i, m in enumerate(columns)}
    rows: list[dict[int, int]] = []
    for eq_idx, mult in row_specs:
        row: dict[int, int] = {}
        for exp, coeff in eqs[eq_idx].items():
            target = add_exp(mult, exp)
            c = idx.get(target)
            if c is not None:
                row[c] = (row.get(c, 0) + int(coeff)) % prime
        rows.append(row)
    return rows, len(columns)


def pivot_scan_modular_columns(
    rows_in: Sequence[Mapping[int, int]],
    ncols: int,
    prime: int = 2147483647,
) -> tuple[Tuple[int, ...], Tuple[int, ...], int]:
    """Column-order Gaussian scan over F_p.

    Returns the original row indices and column indices selected as pivots.
    The column order is the Macaulay monomial order, so the non-pivot columns
    reveal the candidate standard monomial staircase.  This is deliberately an
    offline template-discovery step, not an online solve.
    """

    rows = [dict(r) for r in rows_in]
    row_perm = list(range(len(rows)))
    pivot_rows: list[int] = []
    pivot_cols: list[int] = []
    r = 0

    for c in range(ncols):
        pivrow = None
        for i in range(r, len(rows)):
            if rows[i].get(c, 0) % prime:
                pivrow = i
                break
        if pivrow is None:
            continue

        rows[r], rows[pivrow] = rows[pivrow], rows[r]
        row_perm[r], row_perm[pivrow] = row_perm[pivrow], row_perm[r]

        inv_piv = pow(rows[r][c] % prime, prime - 2, prime)
        rows[r] = {
            j: (v * inv_piv) % prime
            for j, v in rows[r].items()
            if (v * inv_piv) % prime
        }
        pivot_row = rows[r]

        for i in range(r + 1, len(rows)):
            fac = rows[i].get(c, 0) % prime
            if not fac:
                continue
            row = rows[i]
            for j, v in pivot_row.items():
                new_val = (row.get(j, 0) - fac * v) % prime
                if new_val:
                    row[j] = new_val
                elif j in row:
                    del row[j]

        pivot_rows.append(row_perm[r])
        pivot_cols.append(c)
        r += 1
        if r == len(rows):
            break

    return tuple(pivot_rows), tuple(pivot_cols), len(pivot_cols)


def low_degree_nonpivots(
    columns: Sequence[Exp3],
    pivot_cols: Sequence[int],
    max_degree: int,
) -> Tuple[int, ...]:
    pivot_set = set(pivot_cols)
    return tuple(
        i for i, mon in enumerate(columns)
        if i not in pivot_set and sum(mon) <= max_degree
    )


def quotient_action_target_cols(
    columns: Sequence[Exp3],
    quotient_basis_cols: Sequence[int],
) -> Tuple[int, ...]:
    exp_to_col = {m: i for i, m in enumerate(columns)}
    targets: list[int] = []
    for c in quotient_basis_cols:
        for unit in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            targets.append(int(exp_to_col.get(add_exp(columns[int(c)], unit), -1)))
    return tuple(targets)


def build_quotient_action_plan(
    columns: Sequence[Exp3],
    quotient_basis_cols: Sequence[int],
    quotient_projection_rows: Sequence[int],
    quotient_action_targets: Sequence[int],
) -> dict[str, Any]:
    """Precompute the structural scatter plan for quotient action matrices.

    The online coefficients change for every PnP instance, so we cannot cache
    the numeric action matrices themselves.  We can, however, cache everything
    structural: quotient basis order, multiplication targets, which targets are
    already quotient-basis monomials, and which targets are missing.  Online
    projection then only assembles the selected numeric Macaulay rows and fills
    a small cached normal system from this plan.
    """

    qbasis = [int(c) for c in quotient_basis_cols]
    targets = [int(c) for c in quotient_action_targets]
    qpos = {int(c): i for i, c in enumerate(qbasis)}
    target_basis_rows: list[int] = []
    target_basis_cols: list[int] = []
    target_variables: list[int] = []
    target_source_basis: list[int] = []
    missing = 0
    direct = 0
    for k, target_col in enumerate(targets):
        basis_idx, var_idx = divmod(k, 3)
        target_variables.append(int(var_idx))
        target_source_basis.append(int(basis_idx))
        if target_col < 0:
            target_basis_rows.append(-1)
            target_basis_cols.append(-1)
            missing += 1
            continue
        row = qpos.get(int(target_col), -1)
        target_basis_rows.append(int(row))
        target_basis_cols.append(int(target_col) if row >= 0 else -1)
        if row >= 0:
            direct += 1
    return {
        "version": 1,
        "method": "cached_action_projection_plan",
        "variable_order": ["x", "y", "z"],
        "quotient_basis_cols": qbasis,
        "quotient_basis_exponents": [list(columns[c]) for c in qbasis],
        "quotient_projection_rows": [int(r) for r in quotient_projection_rows],
        "target_cols": targets,
        "target_basis_rows": target_basis_rows,
        "target_basis_cols": target_basis_cols,
        "target_source_basis": target_source_basis,
        "target_variables": target_variables,
        "qdim": len(qbasis),
        "target_count": len(targets),
        "cached_row_count": len(quotient_projection_rows),
        "direct_target_count": direct,
        "missing_target_count": missing,
    }


def select_cached_projection_rows(
    M: np.ndarray,
    quotient_basis_cols: Sequence[int],
    tol: float = 1e-10,
) -> Tuple[int, ...]:
    """Select row-space columns for the cached quotient projection.

    Online quotient roots solve ``[M.T_selected | E_Q] x = target``.  The
    quotient identity columns are always included; this helper greedily adds
    independent columns of ``M.T`` on the offline instance.  The stored row
    indices are a fixed symbolic template for later numeric instances.
    """

    M = np.asarray(M, dtype=np.complex128)
    nrows, ncols = M.shape
    qbasis = [int(c) for c in quotient_basis_cols]
    scale = max(1.0, float(np.linalg.norm(M, ord=np.inf)))
    threshold = tol * scale
    q_vectors: list[np.ndarray] = []

    def add_vector(v: np.ndarray) -> bool:
        w = np.array(v, dtype=np.complex128, copy=True)
        for q in q_vectors:
            w -= q * np.vdot(q, w)
        for q in q_vectors:
            w -= q * np.vdot(q, w)
        norm = float(np.linalg.norm(w))
        if norm <= threshold:
            return False
        q_vectors.append(w / norm)
        return True

    for c in qbasis:
        v = np.zeros(ncols, dtype=np.complex128)
        v[c] = 1.0
        add_vector(v)

    selected: list[int] = []
    for row_idx in range(nrows):
        if add_vector(M[row_idx, :]):
            selected.append(row_idx)

    return tuple(selected)


def pivot_scan_columns(M: np.ndarray, tol: float = 1e-10) -> Tuple[Tuple[int, ...], int]:
    A = np.array(M, dtype=np.complex128, copy=True)
    m, n = A.shape
    pivots: list[int] = []
    r = 0
    scale0 = max(1.0, float(np.linalg.norm(A, ord=np.inf)))
    threshold = tol * scale0

    for c in range(n):
        if r >= m:
            break
        col = np.abs(A[r:, c])
        k = int(np.argmax(col))
        if float(col[k]) <= threshold:
            continue
        p = r + k
        if p != r:
            A[[r, p], :] = A[[p, r], :]
        A[r, :] /= A[r, c]
        if r + 1 < m:
            A[r + 1 :, :] -= A[r + 1 :, [c]] * A[r, :]
        pivots.append(c)
        r += 1

    return tuple(pivots), r


def complete_pivot_rows_cols(M: np.ndarray, tol: float = 1e-7) -> Tuple[Tuple[int, ...], Tuple[int, ...], int]:
    """Select a stable square submatrix using complete Gaussian pivoting.

    The previous column-first pivot scan could overestimate the numerical rank
    and later select singular square blocks.  Complete pivoting chooses rows and
    columns together, so the stored online replay block is the same block that
    proved independent offline.
    """

    A = np.array(M, dtype=np.complex128, copy=True)
    m, n = A.shape
    row_perm = list(range(m))
    col_perm = list(range(n))
    pivot_rows: list[int] = []
    pivot_cols: list[int] = []
    scale0 = max(1.0, float(np.max(np.abs(A))))
    threshold = tol * scale0
    limit = min(m, n)

    for k in range(limit):
        sub = np.abs(A[k:, k:])
        ii, jj = np.unravel_index(int(np.argmax(sub)), sub.shape)
        val = float(sub[ii, jj])
        if val <= threshold:
            break

        i = k + int(ii)
        j = k + int(jj)
        if i != k:
            A[[k, i], :] = A[[i, k], :]
            row_perm[k], row_perm[i] = row_perm[i], row_perm[k]
        if j != k:
            A[:, [k, j]] = A[:, [j, k]]
            col_perm[k], col_perm[j] = col_perm[j], col_perm[k]

        piv = A[k, k]
        if k + 1 < m and k + 1 < n:
            A[k + 1 :, k + 1 :] -= np.outer(A[k + 1 :, k], A[k, k + 1 :]) / piv

        pivot_rows.append(row_perm[k])
        pivot_cols.append(col_perm[k])

    return tuple(pivot_rows), tuple(pivot_cols), len(pivot_cols)


def select_pivot_rows_maxvol(Pfull: np.ndarray, tol: float = 1e-12) -> Tuple[int, ...]:
    """Greedy row-subset selection for a square pivot block."""
    Pfull = np.array(Pfull, dtype=np.complex128)
    m, r = Pfull.shape

    # First try plain Gaussian pivoting on Pfull.T.  Pivot columns of Pfull.T
    # are independent rows of Pfull, which gives a square block when rank is OK.
    row_pivots, rank = pivot_scan_columns(Pfull.T, tol=tol)
    if rank == r and len(row_pivots) == r:
        return tuple(int(i) for i in row_pivots)

    R = Pfull.copy()
    selected: list[int] = []
    scale0 = max(1.0, float(np.linalg.norm(R, ord=np.inf)))

    for _ in range(r):
        norms = np.linalg.norm(R, axis=1)
        for i in selected:
            norms[i] = -1.0
        i = int(np.argmax(norms))
        if norms[i] <= tol * scale0:
            break
        selected.append(i)
        q = R[i] / max(norms[i], 1e-300)
        coeffs = R @ q.conj()
        R = R - np.outer(coeffs, q)

    if len(selected) < r:
        norms = np.linalg.norm(Pfull, axis=1)
        for i in np.argsort(-norms):
            ii = int(i)
            if ii not in selected:
                selected.append(ii)
            if len(selected) == r:
                break

    return tuple(selected)


@dataclass
class TreeTemplate:
    template_id: int
    D: int
    columns: Tuple[Exp3, ...]
    row_specs: Tuple[Tuple[int, Exp3], ...]
    pivot_cols: Tuple[int, ...]
    pivot_rows: Tuple[int, ...]
    basis_cols: Tuple[int, ...]
    solve_mode: str = "square"
    template_kind: str = "replay"
    quotient_basis_cols: Tuple[int, ...] = ()
    quotient_degree: int = 0
    discovery_prime: int = 0
    quotient_projection_rows: Tuple[int, ...] = ()
    quotient_action_targets: Tuple[int, ...] = ()
    quotient_action_plan: dict[str, Any] = field(default_factory=dict)
    affine_chart: int = 0
    success: int = 0
    rejected: int = 0

    def label(self) -> str:
        q = f", quotient={len(self.quotient_basis_cols)}" if self.quotient_basis_cols else ""
        proj = f", proj_rows={len(self.quotient_projection_rows)}" if self.quotient_projection_rows else ""
        return (
            f"template #{self.template_id}: D={self.D}, rows={len(self.row_specs)}, "
            f"cols={len(self.columns)}, pivots={len(self.pivot_cols)}, "
            f"basis={len(self.basis_cols)}{q}{proj}, mode={self.solve_mode}, kind={self.template_kind}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "D": self.D,
            "columns": [list(x) for x in self.columns],
            "row_specs": [[eq, list(mult)] for eq, mult in self.row_specs],
            "pivot_cols": list(self.pivot_cols),
            "pivot_rows": list(self.pivot_rows),
            "basis_cols": list(self.basis_cols),
            "solve_mode": self.solve_mode,
            "template_kind": self.template_kind,
            "quotient_basis_cols": list(self.quotient_basis_cols),
            "quotient_degree": self.quotient_degree,
            "discovery_prime": self.discovery_prime,
            "quotient_projection_rows": list(self.quotient_projection_rows),
            "quotient_action_targets": list(self.quotient_action_targets),
            "quotient_action_plan": self.quotient_action_plan,
            "affine_chart": self.affine_chart,
            "success": self.success,
            "rejected": self.rejected,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "TreeTemplate":
        return TreeTemplate(
            template_id=int(d["template_id"]),
            D=int(d["D"]),
            columns=tuple(tuple(x) for x in d["columns"]),
            row_specs=tuple((int(eq), tuple(mult)) for eq, mult in d["row_specs"]),
            pivot_cols=tuple(int(x) for x in d["pivot_cols"]),
            pivot_rows=tuple(int(x) for x in d["pivot_rows"]),
            basis_cols=tuple(int(x) for x in d["basis_cols"]),
            solve_mode=str(d.get("solve_mode", "square")),
            template_kind=str(d.get("template_kind", "replay")),
            quotient_basis_cols=tuple(int(x) for x in d.get("quotient_basis_cols", [])),
            quotient_degree=int(d.get("quotient_degree", 0)),
            discovery_prime=int(d.get("discovery_prime", 0)),
            quotient_projection_rows=tuple(int(x) for x in d.get("quotient_projection_rows", [])),
            quotient_action_targets=tuple(int(x) for x in d.get("quotient_action_targets", [])),
            quotient_action_plan=dict(d.get("quotient_action_plan") or {}),
            affine_chart=int(d.get("affine_chart", 0)),
            success=int(d.get("success", 0)),
            rejected=int(d.get("rejected", 0)),
        )


@dataclass
class PnPSymbolicTree:
    D: int = 10
    templates: list[TreeTemplate] = field(default_factory=list)
    next_id: int = 1

    def add_template_from_instance(self, A: Sequence[Sequence[Any]], B: Sequence[Sequence[Any]], C: Sequence[Sequence[Any]]) -> Tuple[TreeTemplate, dict[str, Any]]:
        t0 = time.perf_counter()
        eqs = wedge_equations_3eq(A, B, C)
        columns, row_specs = build_macaulay_layout(self.D)
        M = assemble_macaulay(eqs, columns, row_specs)
        pivot_rows, pivot_cols, rank = complete_pivot_rows_cols(M)
        basis = tuple(i for i in range(len(columns)) if i not in set(pivot_cols))

        tmpl = TreeTemplate(
            self.next_id,
            self.D,
            tuple(columns),
            tuple(row_specs),
            tuple(pivot_cols),
            tuple(pivot_rows),
            basis,
        )
        self.next_id += 1
        self.templates.append(tmpl)

        return tmpl, {
            "offline_sec": time.perf_counter() - t0,
            "rank": rank,
            "basis_size": len(basis),
            "pivot_rows": len(pivot_rows),
            "equation_terms": [len(e) for e in eqs],
        }

    def add_monomial_order_action_template_from_instance(
        self,
        A: Sequence[Sequence[Any]],
        B: Sequence[Sequence[Any]],
        C: Sequence[Sequence[Any]],
        prime: int = 2147483647,
        quotient_degree: Optional[int] = None,
    ) -> Tuple[TreeTemplate, dict[str, Any]]:
        """Build the stronger action-template branch.

        This follows the monomial order over one prime field to discover the
        generic staircase.  For the current wedge system at D=11 this exposes
        40 low-degree standard monomials, matching the exact msolve/RUR degree.
        The full non-pivot set is still stored because the numeric online
        action replay uses those columns to form candidate action matrices.
        """

        t0 = time.perf_counter()
        eqs = wedge_equations_3eq(A, B, C)
        columns, row_specs = build_macaulay_layout(self.D)
        mod_rows, ncols = assemble_macaulay_mod(eqs, columns, row_specs, prime=prime)
        pivot_rows, pivot_cols, rank = pivot_scan_modular_columns(mod_rows, ncols, prime=prime)
        basis = tuple(i for i in range(len(columns)) if i not in set(pivot_cols))
        q_degree = self.D - 4 if quotient_degree is None else int(quotient_degree)
        quotient_basis = low_degree_nonpivots(columns, pivot_cols, q_degree)
        M = assemble_macaulay(eqs, columns, row_specs)
        projection_rows = select_cached_projection_rows(M, quotient_basis)
        action_targets = quotient_action_target_cols(columns, quotient_basis)
        action_plan = build_quotient_action_plan(columns, quotient_basis, projection_rows, action_targets)

        tmpl = TreeTemplate(
            self.next_id,
            self.D,
            tuple(columns),
            tuple(row_specs),
            tuple(pivot_cols),
            tuple(pivot_rows),
            basis,
            solve_mode="square",
            template_kind="monomial_order_action",
            quotient_basis_cols=quotient_basis,
            quotient_degree=q_degree,
            discovery_prime=prime,
            quotient_projection_rows=projection_rows,
            quotient_action_targets=action_targets,
            quotient_action_plan=action_plan,
        )
        self.next_id += 1
        self.templates.append(tmpl)

        degree_counts: dict[int, int] = {}
        for col in quotient_basis:
            d = sum(columns[col])
            degree_counts[d] = degree_counts.get(d, 0) + 1

        return tmpl, {
            "offline_sec": time.perf_counter() - t0,
            "rank": rank,
            "basis_size": len(basis),
            "quotient_basis_size": len(quotient_basis),
            "quotient_degree": q_degree,
            "quotient_degree_counts": degree_counts,
            "quotient_projection_rows": len(projection_rows),
            "quotient_action_targets": len(action_targets),
            "pivot_rows": len(pivot_rows),
            "discovery_prime": prime,
            "equation_terms": [len(e) for e in eqs],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "D": self.D,
            "next_id": self.next_id,
            "templates": [t.to_dict() for t in self.templates],
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "PnPSymbolicTree":
        tree = PnPSymbolicTree(D=int(d["D"]))
        tree.next_id = int(d.get("next_id", 1))
        tree.templates = [TreeTemplate.from_dict(x) for x in d.get("templates", [])]
        return tree


def replay_template(
    tmpl: TreeTemplate,
    eqs: Sequence[Mapping[Exp3, Any]],
    cond_max: float = 1e22,
    rel_tol: float = 1e-4,
    solve_mode: Optional[str] = None,
    check_cond: bool = True,
    return_rewrite: bool = False,
) -> Tuple[bool, dict[str, Any]]:
    M = assemble_macaulay(eqs, tmpl.columns, tmpl.row_specs)
    mode = solve_mode or tmpl.solve_mode

    if mode not in {"square", "lstsq"}:
        tmpl.rejected += 1
        return False, {
            "reason": f"unknown solve mode: {mode}",
            "condition": float("inf"),
            "relation_residual": float("nan"),
            "replay_sec": 0.0,
            "solve_mode": mode,
        }

    if mode == "square" and len(tmpl.pivot_rows) != len(tmpl.pivot_cols):
        tmpl.rejected += 1
        return False, {
            "reason": "not enough independent pivot rows",
            "condition": float("inf"),
            "relation_residual": float("nan"),
            "replay_sec": 0.0,
            "solve_mode": mode,
        }

    if mode == "square":
        P = M[np.ix_(list(tmpl.pivot_rows), list(tmpl.pivot_cols))]
        B = M[np.ix_(list(tmpl.pivot_rows), list(tmpl.basis_cols))]
    else:
        P = M[:, list(tmpl.pivot_cols)]
        B = M[:, list(tmpl.basis_cols)]

    t0 = time.perf_counter()
    if check_cond:
        try:
            cond = float(np.linalg.cond(P))
        except Exception:
            cond = float("inf")
    else:
        cond = None

    if check_cond and (not np.isfinite(cond) or cond > cond_max):
        tmpl.rejected += 1
        return False, {
            "reason": "square pivot block rejected by condition number",
            "condition": cond,
            "relation_residual": float("nan"),
            "replay_sec": time.perf_counter() - t0,
            "solve_mode": mode,
        }

    try:
        if mode == "square":
            X = np.linalg.solve(P, -B)
            rank = len(tmpl.pivot_cols)
        else:
            X, _residuals, rank, _svals = np.linalg.lstsq(P, -B, rcond=None)
    except np.linalg.LinAlgError as e:
        tmpl.rejected += 1
        return False, {
            "reason": f"{mode} pivot solve failed: {e}",
            "condition": cond,
            "relation_residual": float("nan"),
            "replay_sec": time.perf_counter() - t0,
            "solve_mode": mode,
        }

    rel = np.linalg.norm(P @ X + B) / max(1.0, np.linalg.norm(B))
    ok = rel <= rel_tol
    if ok:
        tmpl.success += 1
    else:
        tmpl.rejected += 1

    info: dict[str, Any] = {
        "reason": "accepted" if ok else "relation residual too high",
        "condition": cond,
        "relation_residual": float(rel),
        "pivot_rank": int(rank),
        "replay_sec": time.perf_counter() - t0,
        "solve_mode": mode,
    }
    if return_rewrite:
        info["_rewrite_table"] = X
    return ok, info


def parse_action_weights(text: str | Sequence[Sequence[float]] | None = None) -> list[Tuple[float, float, float]]:
    """Parse linear action forms such as ``"1,7,11;3,17,5"``."""
    if text is None or text == "":
        return [(1.0, 7.0, 11.0), (3.0, 17.0, 5.0)]
    if not isinstance(text, str):
        return [tuple(float(v) for v in w[:3]) for w in text]  # type: ignore[index]
    out: list[Tuple[float, float, float]] = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        vals = [float(x.strip()) for x in part.split(",") if x.strip()]
        if len(vals) != 3:
            raise ValueError(f"bad action weights {part!r}; expected a,b,c")
        out.append((vals[0], vals[1], vals[2]))
    return out or [(1.0, 7.0, 11.0), (3.0, 17.0, 5.0)]


def template_action_matrices(
    tmpl: TreeTemplate,
    rewrite_table: np.ndarray,
) -> Tuple[list[np.ndarray], dict[str, Any]]:
    """Build multiplication-by-q2/q3/q4 action matrices from a replay table.

    This is the production-style bridge between the stored symbolic branch and
    root extraction.  It uses the same rewrite table computed during online
    replay.  When the stored basis is not a true closed quotient staircase, some
    multiplication targets may be missing; the Newton residual filter below then
    decides which eigen-seeds are valid.
    """

    basis_cols = list(tmpl.basis_cols)
    pivot_cols = list(tmpl.pivot_cols)
    n = len(basis_cols)
    col_to_basis = {c: i for i, c in enumerate(basis_cols)}
    col_to_pivot = {c: i for i, c in enumerate(pivot_cols)}
    exp_to_col = {m: i for i, m in enumerate(tmpl.columns)}

    actions: list[np.ndarray] = []
    missing_by_var: list[int] = []
    direct_by_var: list[int] = []
    rewrite_by_var: list[int] = []

    for unit in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        A = np.zeros((n, n), dtype=np.complex128)
        missing = direct = rewritten = 0
        for j, col in enumerate(basis_cols):
            target_exp = add_exp(tmpl.columns[col], unit)
            target_col = exp_to_col.get(target_exp)
            if target_col is None:
                missing += 1
                continue
            basis_row = col_to_basis.get(target_col)
            if basis_row is not None:
                A[basis_row, j] = 1.0
                direct += 1
                continue
            pivot_row = col_to_pivot.get(target_col)
            if pivot_row is not None:
                A[:, j] = rewrite_table[pivot_row, :]
                rewritten += 1
                continue
            missing += 1

        actions.append(A)
        missing_by_var.append(missing)
        direct_by_var.append(direct)
        rewrite_by_var.append(rewritten)

    return actions, {
        "template_kind": tmpl.template_kind,
        "basis_size": n,
        "quotient_basis_size": len(tmpl.quotient_basis_cols),
        "quotient_degree": tmpl.quotient_degree,
        "missing_targets_by_var": missing_by_var,
        "direct_targets_by_var": direct_by_var,
        "rewritten_targets_by_var": rewrite_by_var,
    }


def tree_roots_from_template_replay(
    tmpl: TreeTemplate,
    eqs: Sequence[Mapping[Exp3, Any]],
    rewrite_table: Optional[np.ndarray] = None,
    action_weights: str | Sequence[Sequence[float]] | None = None,
    residual_tol: float = 1e-8,
    dedup_tol: float = 1e-6,
    real_imag_tol: float = 1e-7,
    max_newton_iter: int = 40,
    max_abs_root: float = 500.0,
    rel_tol: float = 1e-4,
    root_refine_backend: str = "auto",
    root_refine_lib: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Extract candidate roots using the replayed template action matrices.

    This avoids the previous SVD-of-a-new-Macaulay-matrix diagnostic path.  It
    still remains a v0 root layer: the current template basis is a replay basis,
    not yet a certified 40-dimensional quotient staircase, so completeness is
    measured against msolve rather than assumed.
    """

    t0 = time.perf_counter()
    replay_sec = 0.0
    replay_reason = "provided rewrite table"
    if rewrite_table is None:
        ok, replay_info = replay_template(tmpl, eqs, rel_tol=rel_tol, check_cond=False, return_rewrite=True)
        replay_sec = float(replay_info.get("replay_sec", 0.0) or 0.0)
        replay_reason = str(replay_info.get("reason"))
        if not ok or "_rewrite_table" not in replay_info:
            return {
                "ok": False,
                "reason": f"template replay failed before action extraction: {replay_reason}",
                "root_count": 0,
                "real_root_count": 0,
                "root_residuals": [],
                "real_roots": [],
                "total_sec": time.perf_counter() - t0,
                "replay_sec": replay_sec,
            }
        rewrite_table = replay_info["_rewrite_table"]

    action_t0 = time.perf_counter()
    actions, action_info = template_action_matrices(tmpl, rewrite_table)
    action_sec = time.perf_counter() - action_t0
    weights = parse_action_weights(action_weights)

    seeds: list[Tuple[complex, complex, complex]] = []
    eig_failures = 0
    eig_t0 = time.perf_counter()
    for w in weights:
        T = w[0] * actions[0] + w[1] * actions[1] + w[2] * actions[2]
        try:
            _vals, V = np.linalg.eig(T)
            try:
                Vinv = np.linalg.inv(V)
            except np.linalg.LinAlgError:
                Vinv = np.linalg.pinv(V)
            diag_actions = [Vinv @ A @ V for A in actions]
        except np.linalg.LinAlgError:
            eig_failures += 1
            continue
        for i in range(V.shape[1]):
            seeds.append(
                (
                    complex(diag_actions[0][i, i]),
                    complex(diag_actions[1][i, i]),
                    complex(diag_actions[2][i, i]),
                )
            )
    eig_sec = time.perf_counter() - eig_t0

    newton_t0 = time.perf_counter()
    roots, refine_info = refine_candidate_roots(
        eqs,
        seeds,
        residual_tol=residual_tol,
        max_newton_iter=max_newton_iter,
        max_abs_root=max_abs_root,
        backend=root_refine_backend,
        root_refine_lib=root_refine_lib,
    )
    newton_sec = time.perf_counter() - newton_t0

    roots = deduplicate_roots(roots, distance_tol=dedup_tol)
    root_values = [root for root, _ in roots]
    residuals = [float(rel) for _, rel in roots]
    real_roots = real_root_centers_from_complex(root_values, imag_tol=real_imag_tol)
    return {
        "ok": bool(roots),
        "reason": "template action roots extracted" if roots else "no template action roots passed residual filter",
        "method": "template_action",
        "action_info": action_info,
        "action_weights": weights,
        "seed_count": len(seeds),
        "eig_failures": eig_failures,
        "root_count": len(roots),
        "real_root_count": len(real_roots),
        "roots": root_values,
        "real_roots": real_roots,
        "root_residuals": residuals,
        "max_relative_residual": max(residuals) if residuals else None,
        "median_relative_residual": float(np.median(residuals)) if residuals else None,
        "root_sample": [complex_root_to_json(root) for root in root_values[:8]],
        "replay_sec": replay_sec,
        "action_sec": action_sec,
        "eig_sec": eig_sec,
        "root_refine_backend": refine_info.get("backend"),
        "root_refine_library": refine_info.get("library"),
        "root_refine_ok_count": refine_info.get("ok_count"),
        "newton_sec": newton_sec,
        "total_sec": time.perf_counter() - t0,
    }


def c_project_quotient_actions(
    tmpl: TreeTemplate,
    eqs: Sequence[Mapping[Exp3, Any]],
    root_refine_lib: Optional[str | Path] = None,
) -> Optional[tuple[list[np.ndarray], dict[str, Any]]]:
    qbasis = list(tmpl.quotient_basis_cols)
    cached_rows = list(tmpl.quotient_projection_rows)
    cached_targets = list(tmpl.quotient_action_targets)
    if not qbasis or not cached_rows or len(cached_targets) != 3 * len(qbasis):
        return None

    lib, lib_path = load_c_root_refiner(root_refine_lib)
    if lib is None or not hasattr(lib, "pnp_project_actions"):
        return None

    columns = list(tmpl.columns)
    row_specs = list(tmpl.row_specs)
    selected_specs = [row_specs[int(r)] for r in cached_rows]
    term_eq, term_a, term_b, term_c, coeff_re, coeff_im = _flatten_equation_terms(eqs)
    col_a = np.ascontiguousarray([m[0] for m in columns], dtype=np.int32)
    col_b = np.ascontiguousarray([m[1] for m in columns], dtype=np.int32)
    col_c = np.ascontiguousarray([m[2] for m in columns], dtype=np.int32)
    sel_eq = np.ascontiguousarray([s[0] for s in selected_specs], dtype=np.int32)
    sel_a = np.ascontiguousarray([s[1][0] for s in selected_specs], dtype=np.int32)
    sel_b = np.ascontiguousarray([s[1][1] for s in selected_specs], dtype=np.int32)
    sel_c = np.ascontiguousarray([s[1][2] for s in selected_specs], dtype=np.int32)
    qbasis_arr = np.ascontiguousarray(qbasis, dtype=np.int32)
    targets_arr = np.ascontiguousarray(cached_targets, dtype=np.int32)

    qdim = len(qbasis)
    actions_re = np.zeros(3 * qdim * qdim, dtype=np.float64)
    actions_im = np.zeros(3 * qdim * qdim, dtype=np.float64)
    projection_residual = np.zeros(1, dtype=np.float64)
    missing_targets = np.zeros(1, dtype=np.int32)

    t0 = time.perf_counter()
    ok = lib.pnp_project_actions(
        ctypes.c_int(len(term_eq)),
        term_eq.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        term_a.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        term_b.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        term_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        coeff_re.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        coeff_im.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(len(columns)),
        col_a.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        col_b.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        col_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(len(cached_rows)),
        sel_eq.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        sel_a.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        sel_b.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        sel_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(qdim),
        qbasis_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(len(cached_targets)),
        targets_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        actions_re.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        actions_im.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        projection_residual.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        missing_targets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )
    elapsed = time.perf_counter() - t0
    if not ok:
        return None

    actions: list[np.ndarray] = []
    for var in range(3):
        start = var * qdim * qdim
        stop = start + qdim * qdim
        arr = actions_re[start:stop].reshape((qdim, qdim)) + 1j * actions_im[start:stop].reshape((qdim, qdim))
        actions.append(np.ascontiguousarray(arr, dtype=np.complex128))

    return actions, {
        "template_kind": tmpl.template_kind,
        "basis_size": qdim,
        "quotient_basis_size": qdim,
        "quotient_degree": tmpl.quotient_degree,
        "cached_projection": True,
        "projection_solve_mode": "c_cached_normal",
        "projection_rank": len(cached_rows) + qdim,
        "projection_columns": len(cached_rows) + qdim,
        "cached_rowspace_columns": len(cached_rows),
        "projection_residual": float(projection_residual[0]),
        "missing_targets": int(missing_targets[0]),
        "action_project_backend": "c",
        "action_project_library": lib_path,
        "action_project_lapack": bool(getattr(lib, "pnp_project_uses_lapack", lambda: 0)()),
        "action_project_sec": elapsed,
    }


def action_plan_for_template(tmpl: TreeTemplate) -> dict[str, Any]:
    """Return a stored quotient action plan, deriving it for older trees."""

    plan = dict(tmpl.quotient_action_plan or {})
    qdim = len(tmpl.quotient_basis_cols)
    if (
        plan.get("version") == 1
        and int(plan.get("qdim", -1)) == qdim
        and int(plan.get("target_count", -1)) == 3 * qdim
    ):
        return plan
    return build_quotient_action_plan(
        tmpl.columns,
        tmpl.quotient_basis_cols,
        tmpl.quotient_projection_rows,
        tmpl.quotient_action_targets,
    )


def project_quotient_actions_with_plan(
    tmpl: TreeTemplate,
    eqs: Sequence[Mapping[Exp3, Any]],
    plan: Optional[Mapping[str, Any]] = None,
) -> tuple[list[np.ndarray], dict[str, Any], float, float]:
    """Project quotient action matrices using the cached structural plan.

    Compared with the older Python path, this avoids materializing the full
    ``Aproj`` and dense ``RHS`` matrices.  The offline tree already stores the
    quotient basis, selected projection rows, and multiplication targets; this
    helper uses that plan to assemble only the small normal system needed for
    the action coordinates.
    """

    t_assemble = time.perf_counter()
    plan = dict(plan or action_plan_for_template(tmpl))
    qbasis = [int(c) for c in plan.get("quotient_basis_cols", tmpl.quotient_basis_cols)]
    cached_rows = [int(r) for r in plan.get("quotient_projection_rows", tmpl.quotient_projection_rows)]
    targets = [int(c) for c in plan.get("target_cols", tmpl.quotient_action_targets)]
    target_basis_rows = [int(r) for r in plan.get("target_basis_rows", [])]
    qdim = len(qbasis)
    if not qbasis or not cached_rows or len(targets) != 3 * qdim:
        raise ValueError("quotient action plan is incomplete")
    if len(target_basis_rows) != len(targets):
        qpos = {int(c): i for i, c in enumerate(qbasis)}
        target_basis_rows = [qpos.get(int(c), -1) if int(c) >= 0 else -1 for c in targets]

    columns = list(tmpl.columns)
    row_specs = list(tmpl.row_specs)
    ncols = len(columns)
    Msel = assemble_macaulay_selected_rows(eqs, columns, row_specs, cached_rows)
    rowspace_cols = len(cached_rows)
    assemble_sec = time.perf_counter() - t_assemble

    t_project = time.perf_counter()
    # Normal-equation blocks for Aproj=[Msel.T | E_qbasis].
    G11 = Msel.conj() @ Msel.T
    G12 = Msel.conj()[:, qbasis]
    G = np.empty((rowspace_cols + qdim, rowspace_cols + qdim), dtype=np.complex128)
    G[:rowspace_cols, :rowspace_cols] = G11
    G[:rowspace_cols, rowspace_cols:] = G12
    G[rowspace_cols:, :rowspace_cols] = G12.conj().T
    G[rowspace_cols:, rowspace_cols:] = np.eye(qdim, dtype=np.complex128)

    H = np.zeros((rowspace_cols + qdim, len(targets)), dtype=np.complex128)
    missing_targets = 0
    for k, target_col in enumerate(targets):
        if target_col < 0:
            missing_targets += 1
            continue
        H[:rowspace_cols, k] = Msel[:, target_col].conj()
        basis_row = target_basis_rows[k]
        if basis_row >= 0:
            H[rowspace_cols + basis_row, k] = 1.0

    projection_rank: Optional[int] = None
    projection_residual: Optional[float] = None
    projection_solve_mode = "action_plan_cached_normal"
    try:
        X = np.linalg.solve(G, H)
        projection_rank = int(G.shape[0])
    except np.linalg.LinAlgError:
        X, _residuals, projection_rank, _svals = np.linalg.lstsq(G, H, rcond=None)
        projection_solve_mode = "action_plan_cached_lstsq"

    coords = X[rowspace_cols:, :]
    actions = [np.zeros((qdim, qdim), dtype=np.complex128) for _ in range(3)]
    for k in range(len(targets)):
        j, var = divmod(k, 3)
        actions[var][:, j] = coords[:, k]

    # Compute the projection residual for diagnostics without constructing the
    # old full Aproj/RHS matrices.
    Y = Msel.T @ X[:rowspace_cols, :]
    Y[qbasis, :] += coords
    for k, target_col in enumerate(targets):
        if target_col >= 0:
            Y[target_col, k] -= 1.0
    rhs_norm = max(1.0, float(np.sqrt(sum(1 for c in targets if c >= 0))))
    projection_residual = float(np.linalg.norm(Y) / rhs_norm)
    projection_sec = time.perf_counter() - t_project

    action_info = {
        "template_kind": tmpl.template_kind,
        "basis_size": qdim,
        "quotient_basis_size": qdim,
        "quotient_degree": tmpl.quotient_degree,
        "cached_projection": True,
        "action_plan_cached": True,
        "action_plan_version": int(plan.get("version", 0) or 0),
        "action_plan_direct_targets": int(plan.get("direct_target_count", 0) or 0),
        "projection_solve_mode": projection_solve_mode,
        "projection_rank": None if projection_rank is None else int(projection_rank),
        "projection_columns": int(G.shape[0]),
        "cached_rowspace_columns": rowspace_cols,
        "projection_residual": projection_residual,
        "missing_targets": missing_targets,
        "action_project_backend": "python_action_plan",
    }
    return actions, action_info, assemble_sec, projection_sec


def tree_roots_from_quotient_projection(
    tmpl: TreeTemplate,
    eqs: Sequence[Mapping[Exp3, Any]],
    action_weights: str | Sequence[Sequence[float]] | None = None,
    residual_tol: float = 1e-8,
    dedup_tol: float = 1e-6,
    real_imag_tol: float = 1e-7,
    max_newton_iter: int = 40,
    max_abs_root: float = 500.0,
    root_refine_backend: str = "auto",
    root_refine_lib: Optional[str | Path] = None,
    action_project_backend: str = "auto",
    action_project_lib: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Extract best-pose candidates from the 40D monomial-order quotient.

    This is a practical intermediate between the fast replay certificate and a
    full production C/MPFR action solver.  Offline discovery stores the
    low-degree standard monomials.  Online, this helper projects multiplication
    targets back to that quotient basis, builds action matrices, eigensolves,
    Newton-filters candidates, and lets the caller score the real roots.

    It is designed for the current global-pose benchmark, where matching the
    best objective matters more than enumerating every real root box returned by
    msolve.  Coverage is still reported by the online runner.
    """

    t0 = time.perf_counter()
    qbasis = list(tmpl.quotient_basis_cols)
    if not qbasis:
        return {
            "ok": False,
            "reason": "template has no quotient_basis_cols; rebuild offline with --template-kind monomial_order_action",
            "method": "quotient_projection",
            "root_count": 0,
            "real_root_count": 0,
            "root_residuals": [],
            "real_roots": [],
            "total_sec": time.perf_counter() - t0,
        }

    columns = list(tmpl.columns)
    row_specs = list(tmpl.row_specs)
    ncols = len(columns)
    nrows = len(row_specs)
    qdim = len(qbasis)
    exp_to_col = {m: i for i, m in enumerate(columns)}

    cached_rows = list(tmpl.quotient_projection_rows)
    cached_targets = list(tmpl.quotient_action_targets)
    use_cached_projection = bool(cached_rows and len(cached_targets) == 3 * qdim)

    action_project_backend = str(action_project_backend or "auto").lower()
    if action_project_backend not in {"auto", "c", "python"}:
        raise ValueError(f"unknown action projection backend: {action_project_backend}")
    if action_project_backend in {"auto", "c"} and use_cached_projection:
        c_project = c_project_quotient_actions(
            tmpl,
            eqs,
            root_refine_lib=action_project_lib or root_refine_lib,
        )
        if c_project is not None:
            actions, action_info = c_project
            assemble_sec = 0.0
            projection_sec = float(action_info.get("action_project_sec", 0.0) or 0.0)
            targets = [None if int(c) < 0 else int(c) for c in cached_targets]
            missing_targets = int(action_info.get("missing_targets", 0) or 0)
            goto_python_projection = False
        elif action_project_backend == "c":
            return {
                "ok": False,
                "reason": "C quotient projection requested but unavailable or failed",
                "method": "quotient_projection",
                "root_count": 0,
                "real_root_count": 0,
                "root_residuals": [],
                "real_roots": [],
                "total_sec": time.perf_counter() - t0,
            }
        else:
            goto_python_projection = True
    else:
        goto_python_projection = True

    if goto_python_projection and use_cached_projection:
        try:
            actions, action_info, assemble_sec, projection_sec = project_quotient_actions_with_plan(tmpl, eqs)
            goto_python_projection = False
        except Exception as exc:
            # Older fallback: build the full dense projection system.  The
            # final root residual remains the certificate, so a plan failure
            # should not reject an otherwise solvable instance.
            plan_failure_reason = repr(exc)
            goto_python_projection = True
    else:
        plan_failure_reason = None

    if goto_python_projection:
        assemble_t0 = time.perf_counter()
        if use_cached_projection:
            Msel = assemble_macaulay_selected_rows(eqs, columns, row_specs, cached_rows)
            rowspace_cols = len(cached_rows)
            Aproj = np.zeros((ncols, rowspace_cols + qdim), dtype=np.complex128)
            Aproj[:, :rowspace_cols] = Msel.T
            for j, c in enumerate(qbasis):
                Aproj[c, rowspace_cols + j] = 1.0
            targets = [None if int(c) < 0 else int(c) for c in cached_targets]
        else:
            M = assemble_macaulay(eqs, columns, row_specs)
            rowspace_cols = nrows
            Aproj = np.zeros((ncols, rowspace_cols + qdim), dtype=np.complex128)
            Aproj[:, :rowspace_cols] = M.T
            for j, c in enumerate(qbasis):
                Aproj[c, rowspace_cols + j] = 1.0
            targets = []
            for c in qbasis:
                for unit in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    targets.append(exp_to_col.get(add_exp(columns[c], unit)))
        assemble_sec = time.perf_counter() - assemble_t0

        target_info: list[tuple[int, int, Optional[int]]] = []
        missing_targets = 0
        for k, target_col in enumerate(targets):
            j, var = divmod(k, 3)
            if target_col is None:
                missing_targets += 1
            target_info.append((j, var, target_col))

        RHS = np.zeros((ncols, len(targets)), dtype=np.complex128)
        for k, target_col in enumerate(targets):
            if target_col is not None:
                RHS[target_col, k] = 1.0

        solve_t0 = time.perf_counter()
        projection_rank: Optional[int] = None
        projection_residual: Optional[float] = None
        projection_solve_mode = "full_lstsq"
        try:
            if use_cached_projection:
                AH = Aproj.conj().T
                G = AH @ Aproj
                H = AH @ RHS
                try:
                    X = np.linalg.solve(G, H)
                    projection_solve_mode = "cached_normal"
                    projection_rank = int(Aproj.shape[1])
                    projection_residual = float(np.linalg.norm(Aproj @ X - RHS) / max(1.0, np.linalg.norm(RHS)))
                    # The projection system is intentionally over-compressed; the
                    # final certificate is the Newton/root residual, not this fit.
                    if not np.isfinite(projection_residual):
                        X, _residuals, projection_rank, _svals = np.linalg.lstsq(Aproj, RHS, rcond=None)
                        projection_solve_mode = "cached_lstsq"
                        projection_residual = float(np.linalg.norm(Aproj @ X - RHS) / max(1.0, np.linalg.norm(RHS)))
                except np.linalg.LinAlgError:
                    X, _residuals, projection_rank, _svals = np.linalg.lstsq(Aproj, RHS, rcond=None)
                    projection_solve_mode = "cached_lstsq"
                    projection_residual = float(np.linalg.norm(Aproj @ X - RHS) / max(1.0, np.linalg.norm(RHS)))
            else:
                X, _residuals, projection_rank, _svals = np.linalg.lstsq(Aproj, RHS, rcond=None)
                projection_residual = float(np.linalg.norm(Aproj @ X - RHS) / max(1.0, np.linalg.norm(RHS)))
        except np.linalg.LinAlgError as exc:
            return {
                "ok": False,
                "reason": f"quotient projection solve failed: {exc}",
                "method": "quotient_projection",
                "root_count": 0,
                "real_root_count": 0,
                "root_residuals": [],
                "real_roots": [],
                "assemble_sec": assemble_sec,
                "total_sec": time.perf_counter() - t0,
            }
        projection_sec = time.perf_counter() - solve_t0

        coords = X[rowspace_cols:, :]
        actions = [np.zeros((qdim, qdim), dtype=np.complex128) for _ in range(3)]
        for k, (j, var, _target_col) in enumerate(target_info):
            actions[var][:, j] = coords[:, k]
        action_info = {
            "template_kind": tmpl.template_kind,
            "basis_size": qdim,
            "quotient_basis_size": qdim,
            "quotient_degree": tmpl.quotient_degree,
            "cached_projection": use_cached_projection,
            "projection_solve_mode": projection_solve_mode,
            "projection_rank": None if projection_rank is None else int(projection_rank),
            "projection_columns": int(Aproj.shape[1]),
            "cached_rowspace_columns": len(cached_rows),
            "projection_residual": projection_residual,
            "missing_targets": missing_targets,
            "action_project_backend": "python",
        }
        if plan_failure_reason is not None:
            action_info["action_plan_failure"] = plan_failure_reason

    weights = parse_action_weights(action_weights)
    eig_failures = 0
    seeds: list[Tuple[complex, complex, complex]] = []
    eig_t0 = time.perf_counter()
    for w in weights:
        T = w[0] * actions[0] + w[1] * actions[1] + w[2] * actions[2]
        try:
            _vals, V = np.linalg.eig(T)
            try:
                Vinv = np.linalg.inv(V)
            except np.linalg.LinAlgError:
                Vinv = np.linalg.pinv(V)
            diag_actions = [Vinv @ A @ V for A in actions]
        except np.linalg.LinAlgError:
            eig_failures += 1
            continue
        for i in range(V.shape[1]):
            seeds.append(
                (
                    complex(diag_actions[0][i, i]),
                    complex(diag_actions[1][i, i]),
                    complex(diag_actions[2][i, i]),
                )
            )
    eig_sec = time.perf_counter() - eig_t0

    newton_t0 = time.perf_counter()
    roots, refine_info = refine_candidate_roots(
        eqs,
        seeds,
        residual_tol=residual_tol,
        max_newton_iter=max_newton_iter,
        max_abs_root=max_abs_root,
        backend=root_refine_backend,
        root_refine_lib=root_refine_lib,
    )
    newton_sec = time.perf_counter() - newton_t0

    roots = deduplicate_roots(roots, distance_tol=dedup_tol)
    root_values = [root for root, _ in roots]
    residuals = [float(rel) for _, rel in roots]
    real_roots = real_root_centers_from_complex(root_values, imag_tol=real_imag_tol)
    return {
        "ok": bool(roots),
        "reason": "quotient projection roots extracted" if roots else "no quotient projection roots passed residual filter",
        "method": "quotient_projection",
        "action_info": action_info,
        "action_weights": weights,
        "seed_count": len(seeds),
        "eig_failures": eig_failures,
        "root_count": len(roots),
        "real_root_count": len(real_roots),
        "roots": root_values,
        "real_roots": real_roots,
        "root_residuals": residuals,
        "max_relative_residual": max(residuals) if residuals else None,
        "median_relative_residual": float(np.median(residuals)) if residuals else None,
        "root_sample": [complex_root_to_json(root) for root in root_values[:8]],
        "assemble_sec": assemble_sec,
        "projection_sec": projection_sec,
        "eig_sec": eig_sec,
        "root_refine_backend": refine_info.get("backend"),
        "root_refine_library": refine_info.get("library"),
        "root_refine_ok_count": refine_info.get("ok_count"),
        "newton_sec": newton_sec,
        "total_sec": time.perf_counter() - t0,
    }


def save_tree(tree: PnPSymbolicTree, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tree.to_dict()), encoding="utf-8")


def load_tree(path: str | Path) -> PnPSymbolicTree:
    return PnPSymbolicTree.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def build_offline_tree(
    degree: int = 10,
    seed: int = 42,
    k_digits: int = 2,
    template_kind: str = "replay",
    discovery_prime: int = 2147483647,
    quotient_degree: Optional[int] = None,
) -> Tuple[PnPSymbolicTree, TreeTemplate, dict[str, Any]]:
    tree = PnPSymbolicTree(D=degree)
    A0, B0, C0 = generate_k_digit_matrices(k_digits, seed=seed)
    if template_kind == "monomial_order_action":
        tmpl, info = tree.add_monomial_order_action_template_from_instance(
            A0,
            B0,
            C0,
            prime=discovery_prime,
            quotient_degree=quotient_degree,
        )
    elif template_kind == "replay":
        tmpl, info = tree.add_template_from_instance(A0, B0, C0)
    else:
        raise ValueError(f"unknown template kind: {template_kind}")
    return tree, tmpl, info


def calibrate_offline_tree(
    degree: int,
    seeds: Sequence[int],
    k_digits: int = 2,
    cond_max: float = 1e22,
    rel_tol: float = 1e-4,
    allow_lstsq_branches: bool = True,
    lstsq_cond_max: float = 1e24,
    lstsq_rel_tol: float = 2e-4,
    keep_failed_branches: bool = False,
) -> Tuple[PnPSymbolicTree, list[dict[str, Any]]]:
    tree = PnPSymbolicTree(D=degree)
    rows: list[dict[str, Any]] = []

    for seed in seeds:
        A, B, C = generate_k_digit_matrices(k_digits, seed=seed)
        eqs = wedge_equations_3eq(A, B, C)

        accepted = False
        template_id = None
        replay_info: dict[str, Any] = {"reason": "no templates available", "solve_mode": None}
        for tmpl in list(tree.templates):
            ok, info = replay_template(tmpl, eqs, cond_max=cond_max, rel_tol=rel_tol)
            replay_info = info
            if ok:
                accepted = True
                template_id = tmpl.template_id
                break

        new_branch_created = False
        branch_info = None
        if not accepted:
            new_tmpl, branch_info = tree.add_template_from_instance(A, B, C)
            ok, replay_info = replay_template(new_tmpl, eqs, cond_max=cond_max, rel_tol=rel_tol)
            if not ok and allow_lstsq_branches:
                new_tmpl.solve_mode = "lstsq"
                ok, replay_info = replay_template(new_tmpl, eqs, cond_max=lstsq_cond_max, rel_tol=lstsq_rel_tol)

            accepted = ok
            template_id = new_tmpl.template_id
            new_branch_created = True
            if not ok and not keep_failed_branches:
                tree.templates = [t for t in tree.templates if t.template_id != new_tmpl.template_id]

        rows.append(
            {
                "seed": seed,
                "accepted": bool(accepted),
                "template_id": template_id,
                "new_branch_created": bool(new_branch_created),
                "solve_mode": replay_info.get("solve_mode"),
                "reason": replay_info.get("reason"),
                "relation_residual": replay_info.get("relation_residual"),
                "branch_rank": None if branch_info is None else branch_info.get("rank"),
                "branch_basis_size": None if branch_info is None else branch_info.get("basis_size"),
                "templates_now": len(tree.templates),
            }
        )

    return tree, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/content/pnp_symbolic_tree_demo/pnp_tree.json")
    ap.add_argument("--degree", type=int, default=10)
    ap.add_argument("--offline-seed", type=int, default=42)
    ap.add_argument("--offline-seeds", default="")
    ap.add_argument("--k-digits", type=int, default=2)
    ap.add_argument("--template-kind", choices=["replay", "monomial_order_action"], default="replay")
    ap.add_argument("--discovery-prime", type=int, default=2147483647)
    ap.add_argument("--quotient-degree", type=int, default=None)
    ap.add_argument("--cond-max", type=float, default=1e22)
    ap.add_argument("--rel-tol", type=float, default=1e-4)
    ap.add_argument("--allow-lstsq-branches", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--lstsq-cond-max", type=float, default=1e24)
    ap.add_argument("--lstsq-rel-tol", type=float, default=2e-4)
    ap.add_argument("--keep-failed-branches", action="store_true")
    ap.add_argument("--msolve-zip", default="/content/task3_msolve.zip")
    ap.add_argument("--extract-dir", default="/content/task3_msolve_unzipped")
    ap.add_argument("--skip-msolve-setup", action="store_true")
    args = ap.parse_args()

    run_root = Path(args.out).parent
    run_root.mkdir(parents=True, exist_ok=True)

    msolve_bin = None
    if not args.skip_msolve_setup:
        msolve_bin = setup_msolve_from_zip(args.msolve_zip, args.extract_dir, verbose=True)

    calibration_rows: list[dict[str, Any]] = []
    if args.offline_seeds:
        if args.template_kind != "replay":
            raise ValueError("--offline-seeds calibration currently supports only --template-kind replay")
        seeds = parse_seeds(args.offline_seeds)
        tree, calibration_rows = calibrate_offline_tree(
            degree=args.degree,
            seeds=seeds,
            k_digits=args.k_digits,
            cond_max=args.cond_max,
            rel_tol=args.rel_tol,
            allow_lstsq_branches=args.allow_lstsq_branches,
            lstsq_cond_max=args.lstsq_cond_max,
            lstsq_rel_tol=args.lstsq_rel_tol,
            keep_failed_branches=args.keep_failed_branches,
        )
        tmpl = tree.templates[0]
        info = {
            "calibration_seeds": seeds,
            "calibration_rows": len(calibration_rows),
            "accepted": sum(1 for r in calibration_rows if r["accepted"]),
            "new_branches": sum(1 for r in calibration_rows if r["new_branch_created"]),
        }
    else:
        tree, tmpl, info = build_offline_tree(
            args.degree,
            args.offline_seed,
            args.k_digits,
            template_kind=args.template_kind,
            discovery_prime=args.discovery_prime,
            quotient_degree=args.quotient_degree,
        )
    save_tree(tree, args.out)

    print("Run root:", run_root)
    print("msolve:", msolve_bin or os.getenv("MSOLVE_BIN") or shutil.which("msolve") or "not found")
    print("Offline seed:", args.offline_seed)
    print("Macaulay degree:", args.degree)
    print("Template kind:", args.template_kind)
    print()
    print("OFFLINE TEMPLATE BUILT")
    for t in tree.templates:
        print(" ", t.label())
    for k, v in info.items():
        print(f"  {k}: {v}")
    if calibration_rows:
        print()
        print("Calibration rows:")
        for row in calibration_rows:
            print(
                f"  seed={row['seed']} accepted={row['accepted']} "
                f"template={row['template_id']} new_branch={row['new_branch_created']} "
                f"mode={row['solve_mode']} reason={row['reason']}"
            )
    print()
    print("Tree templates:", len(tree.templates))
    print("Saved tree:", args.out)


if __name__ == "__main__":
    main()
