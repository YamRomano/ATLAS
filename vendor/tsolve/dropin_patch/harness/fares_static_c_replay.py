#!/usr/bin/env python3
"""Static-C replay helpers for FARES dehomogenized PnP quartics.

This is the FARES analogue of the fast wedge static-C replay path.  The input
family is the three dehomogenized FARES quartics in three variables.  Offline we
learn a fixed Macaulay block/quotient basis.  Online we substitute only the
current coefficient vector, solve the fixed block over F_p in C, and emit
action matrices A_x,A_y,A_z.
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import sympy as sp

import ysolve_template_core as yc


Exp3 = tuple[int, int, int]
DEFAULT_ACTION_WEIGHTS = "0,0,1;1,7,11;3,17,5;5,11,19;1,0,0;0,1,0"


def format_action_weights(weights: Sequence[Sequence[float]] | str | None) -> str:
    if weights is None:
        return DEFAULT_ACTION_WEIGHTS
    if isinstance(weights, str):
        return weights
    return ";".join(",".join(f"{float(v):.17g}" for v in w) for w in weights)


def select_separating_linear_form_for_actions(
    mats: Mapping[str, np.ndarray],
    candidates: str | Sequence[Sequence[float]] | None = None,
    *,
    sep_tol: float = 1e-10,
    cond_max: float = 1e14,
) -> dict[str, Any]:
    """Return the first numerically safe separating linear form.

    This is intentionally first-hit rather than exhaustive: online we only need
    to guard the preferred branch weight and fall back if it is unsafe.
    """

    yam = yc._yam()
    weights = yam.parse_action_weights(candidates or DEFAULT_ACTION_WEIGHTS)
    actions = [
        np.asarray(mats["x"], dtype=np.float64),
        np.asarray(mats["y"], dtype=np.float64),
        np.asarray(mats["z"], dtype=np.float64),
    ]
    qdim = int(actions[0].shape[0])
    tested: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for weight in weights:
        w = np.asarray(weight, dtype=np.float64)
        t0 = time.perf_counter()
        rec: dict[str, Any] = {
            "weight": [float(v) for v in w],
            "key": ",".join(f"{float(v):.17g}" for v in w),
        }
        try:
            T = w[0] * actions[0] + w[1] * actions[1] + w[2] * actions[2]
            vals, vecs = np.linalg.eig(T)
            if len(vals) != qdim or not np.all(np.isfinite(vals)):
                raise np.linalg.LinAlgError("non-finite or incomplete spectrum")
            scale = max(1.0, float(np.max(np.abs(vals))))
            if qdim <= 1:
                min_sep = float("inf")
            else:
                diffs = np.abs(vals.reshape(-1, 1) - vals.reshape(1, -1))
                diffs += np.eye(qdim) * scale
                min_sep = float(np.min(diffs) / scale)
            cond = float(np.linalg.cond(vecs))
            accepted = bool(min_sep >= sep_tol and cond <= cond_max)
            rec.update(
                {
                    "ok": True,
                    "accepted": accepted,
                    "min_sep_rel": min_sep,
                    "eigvec_cond": cond,
                    "elapsed_ms": 1000.0 * (time.perf_counter() - t0),
                }
            )
            if best is None or (min_sep, -cond) > (
                float(best.get("min_sep_rel", -1.0)),
                -float(best.get("eigvec_cond", float("inf"))),
            ):
                best = dict(rec)
            tested.append(rec)
            if accepted:
                return {
                    "ok": True,
                    "selected_weight": rec["weight"],
                    "selected_key": rec["key"],
                    "accepted_weights": [rec["weight"]],
                    "accepted_keys": [rec["key"]],
                    "sep_tol": sep_tol,
                    "cond_max": cond_max,
                    "tested": tested,
                }
        except Exception as exc:
            rec.update(
                {
                    "ok": False,
                    "accepted": False,
                    "error": repr(exc),
                    "elapsed_ms": 1000.0 * (time.perf_counter() - t0),
                }
            )
            tested.append(rec)
    return {
        "ok": False,
        "selected_weight": None,
        "selected_key": None,
        "sep_tol": sep_tol,
        "cond_max": cond_max,
        "tested": tested,
        "best_effort": best,
        "reason": "no candidate linear form passed separation and conditioning checks",
    }


def rationalize_coeff(value: Any, max_denominator: int = 10**9) -> Fraction:
    """Use one deterministic rationalization for both Ysolve and msolve.

    The FARES Python equation builder receives floating point image/world data.
    Clean msolve input in the older Yam utilities uses ``Fraction(...).
    limit_denominator(1e9)``.  For exact comparison, Ysolve must reduce exactly
    the same rational coefficients modulo p.
    """

    if isinstance(value, Fraction):
        return value
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value), 1)
    if isinstance(value, sp.Rational) and not isinstance(value, sp.Float):
        return Fraction(int(value.p), int(value.q))
    return Fraction(float(value)).limit_denominator(max_denominator)


def rationalize_eqs(
    eqs: Sequence[Mapping[Exp3, Any]],
    *,
    max_denominator: int = 10**9,
) -> list[dict[Exp3, Fraction]]:
    out: list[dict[Exp3, Fraction]] = []
    for eq in eqs:
        dst: dict[Exp3, Fraction] = {}
        for exp, coeff in eq.items():
            q = rationalize_coeff(coeff, max_denominator=max_denominator)
            if q:
                dst[tuple(int(v) for v in exp)] = q
        out.append(dst)
    return out


def float_eqs_for_runtime(
    eqs: Sequence[Mapping[Exp3, Any]],
) -> list[dict[Exp3, float]]:
    """Convert equations to plain doubles for the timed numeric runtime.

    Exact/proof paths still call ``rationalize_eqs``.  The online RT path feeds
    a double static-C kernel, so routing every coefficient through
    ``Fraction(...).limit_denominator`` only adds Python overhead without
    changing the numeric computation.
    """

    out: list[dict[Exp3, float]] = []
    for eq in eqs:
        dst: dict[Exp3, float] = {}
        for exp, coeff in eq.items():
            try:
                value = float(coeff)
            except Exception:
                value = float(rationalize_coeff(coeff))
            if value != 0.0:
                dst[tuple(int(v) for v in exp)] = value
        out.append(dst)
    return out


def coeff_mod_fraction(value: Any, p: int) -> int:
    q = rationalize_coeff(value)
    den = q.denominator % p
    if den == 0:
        raise ZeroDivisionError(f"coefficient denominator is zero modulo p={p}")
    return (q.numerator % p) * pow(den, p - 2, p) % p


def patch_core_coeff_mod() -> None:
    """Make ysolve_template_core reduce the same rationalized coefficients."""

    yc.coeff_mod_prime = coeff_mod_fraction  # type: ignore[assignment]


def _c_array(name: str, values: Sequence[int], ctype: str = "int") -> str:
    vals = ", ".join(str(int(v)) for v in values)
    return f"static const {ctype} {name}[{len(values)}] = {{{vals}}};"


def coeff_terms_from_template(
    tmpl: Any,
    eqs: Sequence[Mapping[Exp3, Any]] | None = None,
) -> list[tuple[int, Exp3]]:
    """Return the coefficient vector support for a FARES branch.

    Prefer the support visible in the training equations.  This keeps the
    online vector compact while still checking that future inputs do not contain
    unsupported monomials.
    """

    if eqs is not None:
        terms: list[tuple[int, Exp3]] = []
        for eq_idx, eq in enumerate(eqs):
            for exp in sorted(eq.keys(), key=lambda e: (sum(e), e), reverse=True):
                terms.append((int(eq_idx), tuple(int(v) for v in exp)))
        return terms

    # Fallback: infer terms from row_specs and Macaulay columns.  Less compact,
    # but still deterministic.
    inferred: set[tuple[int, Exp3]] = set()
    col_set = set(tmpl.columns)
    for eq_idx, mult in tmpl.row_specs:
        for col_exp in col_set:
            exp = tuple(col_exp[i] - mult[i] for i in range(3))
            if min(exp) >= 0 and sum(exp) <= 4:
                inferred.add((int(eq_idx), tuple(int(v) for v in exp)))
    return sorted(inferred, key=lambda item: (item[0], -sum(item[1]), item[1]))


def flatten_coefficients_mod(
    eqs: Sequence[Mapping[Exp3, Any]],
    coeff_terms: Sequence[tuple[int, Exp3]],
    p: int,
) -> np.ndarray:
    vals: list[int] = []
    for eq_idx, exp in coeff_terms:
        vals.append(coeff_mod_fraction(eqs[int(eq_idx)].get(tuple(exp), 0), p))
    return np.asarray(vals, dtype=np.uint32)


def flatten_coefficients_float(
    eqs: Sequence[Mapping[Exp3, Any]],
    coeff_terms: Sequence[tuple[int, Exp3]],
) -> np.ndarray:
    vals: list[float] = []
    for eq_idx, exp in coeff_terms:
        value = eqs[int(eq_idx)].get(tuple(exp), 0.0)
        try:
            vals.append(float(value))
        except Exception:
            vals.append(float(rationalize_coeff(value)))
    return np.asarray(vals, dtype=np.float64)


def flatten_equation_terms_float(
    eqs: Sequence[Mapping[Exp3, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten polynomial terms for the C root refiner without Fraction work."""

    term_eq: list[int] = []
    exp_a: list[int] = []
    exp_b: list[int] = []
    exp_c: list[int] = []
    coeff_re: list[float] = []
    coeff_im: list[float] = []
    for eq_idx, eq in enumerate(eqs):
        for exp, coeff in eq.items():
            z = complex(coeff)
            if z.real == 0.0 and z.imag == 0.0:
                continue
            e = tuple(int(v) for v in exp)
            term_eq.append(int(eq_idx))
            exp_a.append(e[0])
            exp_b.append(e[1])
            exp_c.append(e[2])
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


def equation_norms_float(eqs: Sequence[Mapping[Exp3, Any]]) -> np.ndarray:
    """Return per-equation coefficient norms for residual scaling."""

    norms: list[float] = []
    for eq in eqs:
        norm = 0.0
        for coeff in eq.values():
            norm += abs(complex(coeff))
        norms.append(max(1.0, float(norm)))
    return np.ascontiguousarray(norms, dtype=np.float64)


def unsupported_terms(
    eqs: Sequence[Mapping[Exp3, Any]],
    coeff_terms: Sequence[tuple[int, Exp3]],
) -> list[tuple[int, Exp3]]:
    allowed = {(int(i), tuple(e)) for i, e in coeff_terms}
    bad: list[tuple[int, Exp3]] = []
    for i, eq in enumerate(eqs):
        for exp, coeff in eq.items():
            if coeff and (i, tuple(exp)) not in allowed:
                bad.append((i, tuple(exp)))
    return bad


def _static_plan(tmpl: Any, coeff_terms: Sequence[tuple[int, Exp3]]) -> dict[str, Any]:
    coeff_pos = {(int(i), tuple(e)): k for k, (i, e) in enumerate(coeff_terms)}
    exp_to_col = {tuple(m): i for i, m in enumerate(tmpl.columns)}
    pivot_pos = {int(c): i for i, c in enumerate(tmpl.pivot_cols)}
    basis_pos = {int(c): i for i, c in enumerate(tmpl.basis_cols)}

    a_rows: list[int] = []
    a_cols: list[int] = []
    a_coeffs: list[int] = []
    b_rows: list[int] = []
    b_cols: list[int] = []
    b_coeffs: list[int] = []

    for out_r, src_r in enumerate(tmpl.pivot_rows):
        eq_idx, mult = tmpl.row_specs[int(src_r)]
        for coeff_idx, (term_eq, exp) in enumerate(coeff_terms):
            if int(term_eq) != int(eq_idx):
                continue
            target = yc._yam().add_exp(tuple(mult), tuple(exp))
            col = exp_to_col.get(tuple(target))
            if col is None:
                continue
            if col in pivot_pos:
                a_rows.append(out_r)
                a_cols.append(pivot_pos[col])
                a_coeffs.append(coeff_idx)
            elif col in basis_pos:
                b_rows.append(out_r)
                b_cols.append(basis_pos[col])
                b_coeffs.append(coeff_idx)

    target_kind: list[int] = []
    target_index: list[int] = []
    for unit in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        for basis_col in tmpl.basis_cols:
            target_exp = yc._yam().add_exp(tmpl.columns[int(basis_col)], unit)
            target_col = exp_to_col.get(tuple(target_exp), -1)
            if target_col in basis_pos:
                target_kind.append(1)
                target_index.append(basis_pos[target_col])
            elif target_col in pivot_pos:
                target_kind.append(2)
                target_index.append(pivot_pos[target_col])
            else:
                target_kind.append(0)
                target_index.append(-1)

    return {
        "n_pivots": len(tmpl.pivot_cols),
        "n_basis": len(tmpl.basis_cols),
        "n_coeffs": len(coeff_terms),
        "a_rows": a_rows,
        "a_cols": a_cols,
        "a_coeffs": a_coeffs,
        "b_rows": b_rows,
        "b_cols": b_cols,
        "b_coeffs": b_coeffs,
        "target_kind": target_kind,
        "target_index": target_index,
    }


def generate_fares_static_action_kernel(
    tmpl: Any,
    coeff_terms: Sequence[tuple[int, Exp3]],
    c_path: str | Path,
    *,
    symbol: str = "fares_generated_replay_actions_u32",
) -> Path:
    """Generate one FARES branch-specialized C action replay kernel."""

    plan = _static_plan(tmpl, coeff_terms)
    p = int(tmpl.discovery_prime or 2147483647)
    n = int(plan["n_pivots"])
    nb = int(plan["n_basis"])
    nc = int(plan["n_coeffs"])
    c_path = Path(c_path)
    c_path.parent.mkdir(parents=True, exist_ok=True)

    code = f"""// Generated FARES static-C action replay kernel.
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define MODP {p}u
#define N {n}
#define NB {nb}
#define NCOEFFS {nc}
#define NFILLA {len(plan['a_rows'])}
#define NFILLB {len(plan['b_rows'])}
#define NTARGETS {3 * nb}

{_c_array("A_ROWS", plan["a_rows"])}
{_c_array("A_COLS", plan["a_cols"])}
{_c_array("A_COEFFS", plan["a_coeffs"])}
{_c_array("B_ROWS", plan["b_rows"])}
{_c_array("B_COLS", plan["b_cols"])}
{_c_array("B_COEFFS", plan["b_coeffs"])}
{_c_array("TARGET_KIND", plan["target_kind"])}
{_c_array("TARGET_INDEX", plan["target_index"])}

static uint32_t AUG[N * (N + NB)];

static inline uint32_t add_mod(uint32_t a, uint32_t b) {{
    uint32_t c = a + b;
    if (c >= MODP || c < a) c -= MODP;
    return c;
}}

static inline uint32_t neg_mod(uint32_t a) {{
    return a ? (MODP - a) : 0u;
}}

static inline uint32_t mul_mod(uint32_t a, uint32_t b) {{
    return (uint32_t)(((uint64_t)a * (uint64_t)b) % (uint64_t)MODP);
}}

static uint32_t pow_mod(uint32_t a, uint32_t e) {{
    uint64_t base = (uint64_t)(a % MODP);
    uint64_t out = 1u;
    while (e) {{
        if (e & 1u) out = (out * base) % MODP;
        base = (base * base) % MODP;
        e >>= 1u;
    }}
    return (uint32_t)out;
}}

static inline uint32_t submul_mod(uint32_t value, uint32_t factor, uint32_t rhs) {{
    uint32_t prod = mul_mod(factor, rhs);
    return value >= prod ? (value - prod) : (uint32_t)((uint64_t)value + MODP - prod);
}}

static void swap_rows(int a, int b) {{
    if (a == b) return;
    for (int c = 0; c < N + NB; ++c) {{
        uint32_t tmp = AUG[(size_t)a * (N + NB) + c];
        AUG[(size_t)a * (N + NB) + c] = AUG[(size_t)b * (N + NB) + c];
        AUG[(size_t)b * (N + NB) + c] = tmp;
    }}
}}

static int solve_block(void) {{
    for (int col = 0; col < N; ++col) {{
        int piv = -1;
        for (int r = col; r < N; ++r) {{
            if (AUG[(size_t)r * (N + NB) + col] != 0u) {{ piv = r; break; }}
        }}
        if (piv < 0) return -1;
        swap_rows(col, piv);
        uint32_t *prow = &AUG[(size_t)col * (N + NB)];
        uint32_t inv = pow_mod(prow[col], MODP - 2u);
        for (int j = col; j < N + NB; ++j) prow[j] = mul_mod(prow[j], inv);
        for (int r = 0; r < N; ++r) {{
            if (r == col) continue;
            uint32_t *row = &AUG[(size_t)r * (N + NB)];
            uint32_t fac = row[col];
            if (!fac) continue;
            row[col] = 0u;
            for (int j = col + 1; j < N + NB; ++j) {{
                if (prow[j]) row[j] = submul_mod(row[j], fac, prow[j]);
            }}
        }}
    }}
    return 0;
}}

int {symbol}(const uint32_t *coeffs, int n_coeffs, uint32_t *actions_out) {{
    if (!coeffs || !actions_out || n_coeffs != NCOEFFS) return -10;
    memset(AUG, 0, sizeof(AUG));
    for (int k = 0; k < NFILLA; ++k) {{
        uint32_t v = coeffs[A_COEFFS[k]] % MODP;
        uint32_t *slot = &AUG[(size_t)A_ROWS[k] * (N + NB) + A_COLS[k]];
        *slot = add_mod(*slot, v);
    }}
    for (int k = 0; k < NFILLB; ++k) {{
        uint32_t v = neg_mod(coeffs[B_COEFFS[k]] % MODP);
        uint32_t *slot = &AUG[(size_t)B_ROWS[k] * (N + NB) + N + B_COLS[k]];
        *slot = add_mod(*slot, v);
    }}
    int rc = solve_block();
    if (rc < 0) return rc;
    memset(actions_out, 0, (size_t)NTARGETS * (size_t)NB * sizeof(uint32_t));
    for (int a = 0; a < 3; ++a) {{
        for (int j = 0; j < NB; ++j) {{
            int t = a * NB + j;
            int kind = TARGET_KIND[t];
            int idx = TARGET_INDEX[t];
            if (kind == 1) {{
                actions_out[((size_t)a * NB + idx) * NB + j] = 1u;
            }} else if (kind == 2) {{
                uint32_t *xrow = &AUG[(size_t)idx * (N + NB) + N];
                for (int r = 0; r < NB; ++r) {{
                    actions_out[((size_t)a * NB + r) * NB + j] = xrow[r];
                }}
            }} else {{
                return -20 - t;
            }}
        }}
    }}
    return 0;
}}
"""
    c_path.write_text(code, encoding="utf-8")
    return c_path


def compile_fares_static_action_kernel(c_path: str | Path, so_path: str | Path) -> Path:
    c_path = Path(c_path)
    so_path = Path(so_path)
    so_path.parent.mkdir(parents=True, exist_ok=True)
    cc = os.environ.get("CC", "cc")
    subprocess.run([cc, "-O3", "-std=c11", "-shared", "-fPIC", str(c_path), "-o", str(so_path)], check=True)
    return so_path


_STATIC_CACHE: dict[str, Any] = {}
_STATIC_DOUBLE_CACHE: dict[str, Any] = {}
_ROOT_EXTRACT_CACHE: dict[str, Any] = {}
_DIRECT_COEFF_CACHE: dict[str, Any] = {}


def generate_fares_static_action_kernel_double(
    tmpl: Any,
    coeff_terms: Sequence[tuple[int, Exp3]],
    c_path: str | Path,
    *,
    symbol: str = "fares_generated_replay_actions_f64",
) -> Path:
    """Generate a branch-specialized floating action kernel for RT output.

    The modular kernel is the exact algebraic proof object.  This numeric
    kernel uses the same learned branch/template and emits floating action
    matrices that can be eigensolved/Newton-refined into quaternion candidates
    for the original FARES pose scoring layer.
    """

    plan = _static_plan(tmpl, coeff_terms)
    n = int(plan["n_pivots"])
    nb = int(plan["n_basis"])
    nc = int(plan["n_coeffs"])
    c_path = Path(c_path)
    c_path.parent.mkdir(parents=True, exist_ok=True)

    code = f"""// Generated FARES static-C floating action replay kernel.
#include <math.h>
#include <stddef.h>
#include <string.h>

#define N {n}
#define NB {nb}
#define NCOEFFS {nc}
#define NFILLA {len(plan['a_rows'])}
#define NFILLB {len(plan['b_rows'])}
#define NTARGETS {3 * nb}
#define PIVOT_EPS 1e-14

{_c_array("A_ROWS", plan["a_rows"])}
{_c_array("A_COLS", plan["a_cols"])}
{_c_array("A_COEFFS", plan["a_coeffs"])}
{_c_array("B_ROWS", plan["b_rows"])}
{_c_array("B_COLS", plan["b_cols"])}
{_c_array("B_COEFFS", plan["b_coeffs"])}
{_c_array("TARGET_KIND", plan["target_kind"])}
{_c_array("TARGET_INDEX", plan["target_index"])}

static double AUG[N * (N + NB)];

static void swap_rows(int a, int b) {{
    if (a == b) return;
    for (int c = 0; c < N + NB; ++c) {{
        double tmp = AUG[(size_t)a * (N + NB) + c];
        AUG[(size_t)a * (N + NB) + c] = AUG[(size_t)b * (N + NB) + c];
        AUG[(size_t)b * (N + NB) + c] = tmp;
    }}
}}

static int solve_block(void) {{
    for (int col = 0; col < N; ++col) {{
        int piv = -1;
        double best = 0.0;
        for (int r = col; r < N; ++r) {{
            double v = fabs(AUG[(size_t)r * (N + NB) + col]);
            if (v > best) {{ best = v; piv = r; }}
        }}
        if (piv < 0 || best <= PIVOT_EPS) return -1;
        swap_rows(col, piv);
        double *prow = &AUG[(size_t)col * (N + NB)];
        double inv = 1.0 / prow[col];
        for (int j = col; j < N + NB; ++j) prow[j] *= inv;
        prow[col] = 1.0;
        for (int r = 0; r < N; ++r) {{
            if (r == col) continue;
            double *row = &AUG[(size_t)r * (N + NB)];
            double fac = row[col];
            if (fabs(fac) <= PIVOT_EPS) continue;
            row[col] = 0.0;
            for (int j = col + 1; j < N + NB; ++j) row[j] -= fac * prow[j];
        }}
    }}
    return 0;
}}

int {symbol}(const double *coeffs, int n_coeffs, double *actions_out) {{
    if (!coeffs || !actions_out || n_coeffs != NCOEFFS) return -10;
    memset(AUG, 0, sizeof(AUG));
    for (int k = 0; k < NFILLA; ++k) {{
        AUG[(size_t)A_ROWS[k] * (N + NB) + A_COLS[k]] += coeffs[A_COEFFS[k]];
    }}
    for (int k = 0; k < NFILLB; ++k) {{
        AUG[(size_t)B_ROWS[k] * (N + NB) + N + B_COLS[k]] -= coeffs[B_COEFFS[k]];
    }}
    int rc = solve_block();
    if (rc < 0) return rc;
    memset(actions_out, 0, (size_t)NTARGETS * (size_t)NB * sizeof(double));
    for (int a = 0; a < 3; ++a) {{
        for (int j = 0; j < NB; ++j) {{
            int t = a * NB + j;
            int kind = TARGET_KIND[t];
            int idx = TARGET_INDEX[t];
            if (kind == 1) {{
                actions_out[((size_t)a * NB + idx) * NB + j] = 1.0;
            }} else if (kind == 2) {{
                double *xrow = &AUG[(size_t)idx * (N + NB) + N];
                for (int r = 0; r < NB; ++r) {{
                    actions_out[((size_t)a * NB + r) * NB + j] = xrow[r];
                }}
            }} else {{
                return -20 - t;
            }}
        }}
    }}
    return 0;
}}
"""
    c_path.write_text(code, encoding="utf-8")
    return c_path


def _load_static_fn(so_path: str | Path):
    key = str(Path(so_path).resolve())
    fn = _STATIC_CACHE.get(key)
    if fn is not None:
        return fn
    lib = ctypes.CDLL(key)
    fn = lib.fares_generated_replay_actions_u32
    fn.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS"),
    ]
    fn.restype = ctypes.c_int
    _STATIC_CACHE[key] = fn
    return fn


def _load_static_double_fn(so_path: str | Path):
    key = str(Path(so_path).resolve())
    fn = _STATIC_DOUBLE_CACHE.get(key)
    if fn is not None:
        return fn
    lib = ctypes.CDLL(key)
    fn = lib.fares_generated_replay_actions_f64
    fn.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    ]
    fn.restype = ctypes.c_int
    _STATIC_DOUBLE_CACHE[key] = fn
    return fn


def replay_fares_actions_static_c(
    eqs: Sequence[Mapping[Exp3, Any]],
    tmpl: Any,
    coeff_terms: Sequence[tuple[int, Exp3]],
    so_path: str | Path,
    *,
    p: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    p = int(p or tmpl.discovery_prime)
    bad_terms = unsupported_terms(eqs, coeff_terms)
    if bad_terms:
        raise ValueError(f"input has unsupported coefficient terms: {bad_terms[:5]}")
    coeffs = flatten_coefficients_mod(eqs, coeff_terms, p)
    qdim = len(tmpl.basis_cols)
    out = np.zeros((3 * qdim * qdim,), dtype=np.uint32)
    fn = _load_static_fn(so_path)
    t0 = time.perf_counter()
    rc = int(fn(coeffs, ctypes.c_int(coeffs.shape[0]), out))
    elapsed = time.perf_counter() - t0
    if rc < 0:
        raise ZeroDivisionError(f"FARES static-C replay failed with code {rc}")
    arr = out.reshape(3, qdim, qdim).astype(np.int64)
    return {"x": arr[0], "y": arr[1], "z": arr[2]}, {
        "replay_seconds": elapsed,
        "compiled_coefficients": int(coeffs.shape[0]),
        "N": qdim,
        "rank": len(tmpl.pivot_cols),
        "p": p,
        "backend": "fares_static_c",
    }


def replay_fares_actions_static_c_double(
    eqs: Sequence[Mapping[Exp3, Any]],
    tmpl: Any,
    coeff_terms: Sequence[tuple[int, Exp3]],
    so_path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    bad_terms = unsupported_terms(eqs, coeff_terms)
    if bad_terms:
        raise ValueError(f"input has unsupported coefficient terms: {bad_terms[:5]}")
    coeffs = flatten_coefficients_float(eqs, coeff_terms)
    qdim = len(tmpl.basis_cols)
    out = np.zeros((3 * qdim * qdim,), dtype=np.float64)
    fn = _load_static_double_fn(so_path)
    t0 = time.perf_counter()
    rc = int(fn(coeffs, ctypes.c_int(coeffs.shape[0]), out))
    elapsed = time.perf_counter() - t0
    if rc < 0:
        raise np.linalg.LinAlgError(f"FARES static-C double replay failed with code {rc}")
    arr = out.reshape(3, qdim, qdim).astype(np.float64)
    return {"x": arr[0], "y": arr[1], "z": arr[2]}, {
        "replay_seconds": elapsed,
        "compiled_coefficients": int(coeffs.shape[0]),
        "N": qdim,
        "rank": len(tmpl.pivot_cols),
        "backend": "fares_static_c_double",
    }


def replay_fares_actions_static_c_double_coeffs(
    coeffs: np.ndarray,
    tmpl: Any,
    so_path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    coeffs = np.ascontiguousarray(np.asarray(coeffs, dtype=np.float64))
    qdim = len(tmpl.basis_cols)
    out = np.zeros((3 * qdim * qdim,), dtype=np.float64)
    fn = _load_static_double_fn(so_path)
    t0 = time.perf_counter()
    rc = int(fn(coeffs, ctypes.c_int(coeffs.shape[0]), out))
    elapsed = time.perf_counter() - t0
    if rc < 0:
        raise np.linalg.LinAlgError(f"FARES static-C double replay failed with code {rc}")
    arr = out.reshape(3, qdim, qdim).astype(np.float64)
    return {"x": arr[0], "y": arr[1], "z": arr[2]}, {
        "replay_seconds": elapsed,
        "compiled_coefficients": int(coeffs.shape[0]),
        "N": qdim,
        "rank": len(tmpl.pivot_cols),
        "backend": "fares_static_c_double_coeff_vector",
    }


def ensure_double_kernel(
    branch: "FaresStaticBranch",
    *,
    out_dir: str | Path | None = None,
) -> Path:
    root = Path(out_dir) if out_dir is not None else branch.so_path.parent
    root.mkdir(parents=True, exist_ok=True)
    stem = f"fares_static_branch_{branch.index}_seed_{branch.seed}_p_{branch.p}_f64"
    c_path = root / f"{stem}.c"
    so_path = root / f"{stem}.so"
    if not so_path.exists():
        generate_fares_static_action_kernel_double(branch.tmpl, branch.coeff_terms, c_path)
        compile_fares_static_action_kernel(c_path, so_path)
    return so_path


def _dense_macaulay_matrix(
    rows: Sequence[Mapping[int, int]],
    ncols: int,
    p: int,
) -> np.ndarray:
    """Materialize sparse Macaulay rows as an exact object matrix over F_p."""

    M = np.zeros((len(rows), int(ncols)), dtype=object)
    for r, row in enumerate(rows):
        for c, value in row.items():
            M[r, int(c)] = int(value) % int(p)
    return M


def _rref_with_row_trace(
    M: np.ndarray,
    p: int,
    *,
    start_row: int = 0,
    start_col: int = 0,
) -> tuple[np.ndarray, list[int], list[int], int]:
    """Compute RREF over F_p and record the literal pivot columns/row swaps.

    ``pivot_swaps[k]`` is the row index selected as pivot at step k in the
    already-mutated matrix, before swapping it into row k.  Replaying these
    pairs on a new input is the literal prefix-reuse trace.
    """

    R = np.asarray(M, dtype=object).copy()
    nr, nc = R.shape
    row = int(start_row)
    pivot_cols: list[int] = []
    pivot_swaps: list[int] = []
    for col in range(int(start_col), int(nc)):
        if row >= nr:
            break
        pivot = None
        for rr in range(row, nr):
            if int(R[rr, col]) % p:
                pivot = rr
                break
        if pivot is None:
            continue
        if pivot != row:
            R[[row, pivot]] = R[[pivot, row]]
        inv = pow(int(R[row, col]) % p, p - 2, p)
        R[row, :] = [(int(v) * inv) % p for v in R[row, :]]
        for rr in range(nr):
            if rr == row:
                continue
            factor = int(R[rr, col]) % p
            if factor:
                R[rr, :] = [(int(R[rr, cc]) - factor * int(R[row, cc])) % p for cc in range(nc)]
        pivot_cols.append(int(col))
        pivot_swaps.append(int(pivot))
        row += 1
    return R, pivot_cols, pivot_swaps, row


def _trace_plan(tmpl: Any) -> dict[str, Any]:
    plan = dict(getattr(tmpl, "quotient_action_plan", {}) or {})
    if plan.get("trace_format") != "full_rref_row_trace":
        raise ValueError("template does not contain a full RREF row-operation trace")
    return plan


def _replay_trace_prefix_until_failure(
    M: np.ndarray,
    tmpl: Any,
    p: int,
) -> tuple[np.ndarray, int, dict[str, Any] | None]:
    """Replay the stored row-operation trace until it fails or completes."""

    plan = _trace_plan(tmpl)
    pivot_cols = [int(c) for c in plan["trace_pivot_cols"]]
    pivot_swaps = [int(r) for r in plan["trace_pivot_swaps"]]
    R = np.asarray(M, dtype=object).copy()
    nr, nc = R.shape
    for step, (col, pivot) in enumerate(zip(pivot_cols, pivot_swaps)):
        if step >= nr or col >= nc or pivot >= nr:
            fail = {
                "ok": False,
                "failed_step": int(step),
                "failed_col": int(col),
                "failed_swap": int(pivot),
                "reused_prefix_pivots": int(step),
                "reason": "stored trace index is outside the online matrix",
            }
            return R, step, fail
        if int(R[pivot, col]) % p == 0:
            fail = {
                "ok": False,
                "failed_step": int(step),
                "failed_col": int(col),
                "failed_swap": int(pivot),
                "reused_prefix_pivots": int(step),
                "reason": "stored trace pivot vanished",
            }
            return R, step, fail
        if pivot != step:
            R[[step, pivot]] = R[[pivot, step]]
        inv = pow(int(R[step, col]) % p, p - 2, p)
        R[step, :] = [(int(v) * inv) % p for v in R[step, :]]
        for rr in range(nr):
            if rr == step:
                continue
            factor = int(R[rr, col]) % p
            if factor:
                R[rr, :] = [(int(R[rr, cc]) - factor * int(R[step, cc])) % p for cc in range(nc)]
    return R, len(pivot_cols), None


def _trace_actions_from_reduced_matrix(
    R: np.ndarray,
    tmpl: Any,
    p: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    plan = _trace_plan(tmpl)
    pivot_cols = [int(c) for c in plan["trace_pivot_cols"]]
    basis_cols = [int(c) for c in plan["trace_basis_cols"]]
    qbasis = [int(c) for c in plan.get("trace_quotient_basis_cols", basis_cols)]
    if tuple(qbasis) != tuple(basis_cols):
        raise ValueError("trace action export expects quotient basis to equal all nonpivots")

    qdim = len(qbasis)
    exp_to_col = {tuple(m): i for i, m in enumerate(tmpl.columns)}
    basis_index = {c: i for i, c in enumerate(qbasis)}
    pivot_index = {c: i for i, c in enumerate(pivot_cols)}
    yam = yc._yam()

    actions: dict[str, np.ndarray] = {}
    missing: list[dict[str, Any]] = []
    for name, unit in zip(("x", "y", "z"), ((1, 0, 0), (0, 1, 0), (0, 0, 1))):
        A = np.zeros((qdim, qdim), dtype=np.uint32)
        for j, basis_col in enumerate(qbasis):
            target_exp = yam.add_exp(tmpl.columns[basis_col], unit)
            target_col = exp_to_col.get(tuple(target_exp))
            if target_col is None:
                missing.append({"variable": name, "basis_index": j, "target_exp": target_exp})
                continue
            if target_col in basis_index:
                A[basis_index[target_col], j] = 1
            elif target_col in pivot_index:
                prow = pivot_index[target_col]
                for r, bcol in enumerate(qbasis):
                    A[r, j] = (-int(R[prow, bcol])) % p
            else:
                missing.append({"variable": name, "basis_index": j, "target_col": int(target_col)})
        actions[name] = A
    return actions, {"missing_targets": missing, "basis_size": qdim, "trace_rank": len(pivot_cols)}


def learn_fares_trace_template(
    eqs: Sequence[Mapping[Exp3, Any]],
    *,
    seed: int,
    p: int,
    out_dir: str | Path,
    index: int,
    degree: int = 11,
) -> Any:
    """Offline exact template with a full RREF row-operation trace."""

    patch_core_coeff_mod()
    yam = yc._yam()
    columns, row_specs = yam.build_macaulay_layout(degree)
    rows, ncols = yc.assemble_macaulay_mod_exact(eqs, columns, row_specs, int(p))
    M = _dense_macaulay_matrix(rows, ncols, int(p))
    t0 = time.perf_counter()
    _R, pivot_cols, pivot_swaps, rank = _rref_with_row_trace(M, int(p))
    elapsed = time.perf_counter() - t0
    basis_cols = tuple(i for i in range(int(ncols)) if i not in set(pivot_cols))
    tmpl = yam.TreeTemplate(
        template_id=index + 1,
        D=degree,
        columns=tuple(columns),
        row_specs=tuple(row_specs),
        pivot_cols=tuple(int(c) for c in pivot_cols),
        pivot_rows=tuple(range(int(rank))),
        basis_cols=tuple(int(c) for c in basis_cols),
        solve_mode="trace_rref",
        template_kind="fares_full_rref_trace_action",
        quotient_basis_cols=tuple(int(c) for c in basis_cols),
        quotient_degree=-1,
        discovery_prime=int(p),
        quotient_projection_rows=tuple(),
        quotient_action_targets=tuple(),
        quotient_action_plan={
            "trace_format": "full_rref_row_trace",
            "trace_seed": int(seed),
            "trace_degree": int(degree),
            "trace_prime": int(p),
            "trace_rows": int(M.shape[0]),
            "trace_cols": int(M.shape[1]),
            "trace_rank": int(rank),
            "trace_pivot_cols": [int(c) for c in pivot_cols],
            "trace_pivot_swaps": [int(r) for r in pivot_swaps],
            "trace_basis_cols": [int(c) for c in basis_cols],
            "trace_quotient_basis_cols": [int(c) for c in basis_cols],
            "trace_offline_seconds": float(elapsed),
            "literal_prefix_reuse": True,
        },
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"fares_trace_branch_{index}_seed_{seed}_p_{p}.json"
    path.write_text(json.dumps(tmpl.to_dict(), indent=2), encoding="utf-8")
    return tmpl


def replay_fares_trace_actions(
    eqs: Sequence[Mapping[Exp3, Any]],
    tmpl: Any,
    *,
    p: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rows, ncols = yc.assemble_macaulay_mod_exact(eqs, tmpl.columns, tmpl.row_specs, int(p))
    M = _dense_macaulay_matrix(rows, ncols, int(p))
    t0 = time.perf_counter()
    R, reused, failure = _replay_trace_prefix_until_failure(M, tmpl, int(p))
    if failure is not None:
        raise ZeroDivisionError(json.dumps(failure, sort_keys=True))
    mats, meta = _trace_actions_from_reduced_matrix(R, tmpl, int(p))
    meta.update({"reused_prefix_pivots": int(reused), "trace_replay_seconds": time.perf_counter() - t0})
    return mats, meta


def trace_failure_info(
    eqs: Sequence[Mapping[Exp3, Any]],
    tmpl: Any,
    *,
    p: int,
) -> dict[str, Any]:
    rows, ncols = yc.assemble_macaulay_mod_exact(eqs, tmpl.columns, tmpl.row_specs, int(p))
    M = _dense_macaulay_matrix(rows, ncols, int(p))
    _R, reused, failure = _replay_trace_prefix_until_failure(M, tmpl, int(p))
    if failure is None:
        return {
            "ok": True,
            "failed_step": None,
            "failed_col": None,
            "failed_swap": None,
            "reused_prefix_pivots": int(reused),
            "branch_seed": _trace_plan(tmpl).get("trace_seed"),
        }
    failure["branch_seed"] = _trace_plan(tmpl).get("trace_seed")
    return failure


def learn_fares_trace_template_by_forking_prefix(
    eqs: Sequence[Mapping[Exp3, Any]],
    parent_tmpl: Any,
    *,
    seed: int,
    p: int,
    out_dir: str | Path,
    index: int,
    degree: int = 11,
) -> Any:
    """Fork a new trace branch by replaying the valid parent prefix first."""

    patch_core_coeff_mod()
    yam = yc._yam()
    columns, row_specs = yam.build_macaulay_layout(degree)
    rows, ncols = yc.assemble_macaulay_mod_exact(eqs, columns, row_specs, int(p))
    M = _dense_macaulay_matrix(rows, ncols, int(p))
    prefix_R, failed_step, failure = _replay_trace_prefix_until_failure(M, parent_tmpl, int(p))
    parent_plan = _trace_plan(parent_tmpl)
    parent_cols = [int(c) for c in parent_plan["trace_pivot_cols"]]
    parent_swaps = [int(r) for r in parent_plan["trace_pivot_swaps"]]
    if failure is None:
        failed_col = int(parent_cols[-1] + 1) if parent_cols else 0
    else:
        failed_col = int(failure["failed_col"])
    t0 = time.perf_counter()
    R, suffix_cols, suffix_swaps, rank = _rref_with_row_trace(
        prefix_R,
        int(p),
        start_row=int(failed_step),
        start_col=failed_col,
    )
    elapsed = time.perf_counter() - t0
    pivot_cols = parent_cols[:failed_step] + suffix_cols
    pivot_swaps = parent_swaps[:failed_step] + suffix_swaps
    basis_cols = tuple(i for i in range(int(ncols)) if i not in set(pivot_cols))
    tmpl = yam.TreeTemplate(
        template_id=index + 1,
        D=degree,
        columns=tuple(columns),
        row_specs=tuple(row_specs),
        pivot_cols=tuple(int(c) for c in pivot_cols),
        pivot_rows=tuple(range(len(pivot_cols))),
        basis_cols=tuple(int(c) for c in basis_cols),
        solve_mode="trace_rref_prefix_fork",
        template_kind="fares_full_rref_trace_action",
        quotient_basis_cols=tuple(int(c) for c in basis_cols),
        quotient_degree=-1,
        discovery_prime=int(p),
        quotient_projection_rows=tuple(),
        quotient_action_targets=tuple(),
        quotient_action_plan={
            "trace_format": "full_rref_row_trace",
            "trace_seed": int(seed),
            "trace_degree": int(degree),
            "trace_prime": int(p),
            "trace_rows": int(M.shape[0]),
            "trace_cols": int(M.shape[1]),
            "trace_rank": int(len(pivot_cols)),
            "trace_pivot_cols": [int(c) for c in pivot_cols],
            "trace_pivot_swaps": [int(r) for r in pivot_swaps],
            "trace_basis_cols": [int(c) for c in basis_cols],
            "trace_quotient_basis_cols": [int(c) for c in basis_cols],
            "trace_fork_seconds": float(elapsed),
            "literal_prefix_reuse": True,
            "fork_from_seed": parent_plan.get("trace_seed"),
            "fork_failed_step": None if failure is None else failure.get("failed_step"),
            "fork_failed_col": None if failure is None else failure.get("failed_col"),
            "fork_reused_prefix_pivots": int(failed_step),
            "fork_suffix_pivots": int(len(suffix_cols)),
        },
    )
    # Sanity: the newly forked trace must replay on the same input.
    _mats, _meta = _trace_actions_from_reduced_matrix(R, tmpl, int(p))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"fares_trace_branch_{index}_prefix_fork_seed_{seed}_p_{p}.json"
    path.write_text(json.dumps(tmpl.to_dict(), indent=2), encoding="utf-8")
    return tmpl


def square_block_failure_info(
    eqs: Sequence[Mapping[Exp3, Any]],
    branch: "FaresStaticBranch",
    *,
    p: int,
) -> dict[str, Any]:
    """Diagnose the fixed square block used by a FARES static branch.

    The current FARES action template is represented by a selected square
    pivot block, not by the full RREF trace used by the wedge prefix-fork
    module.  This function still gives the exact fork point for that block:
    the first online pivot that vanishes and the number of valid prefix pivots.
    """

    rows, _ncols = yc.assemble_macaulay_mod_exact(eqs, branch.tmpl.columns, branch.tmpl.row_specs, int(p))
    A = np.asarray(
        yc.dense_submatrix_mod(rows, branch.tmpl.pivot_rows, branch.tmpl.pivot_cols, int(p)),
        dtype=object,
    )
    n = int(A.shape[0])
    R = A.copy()
    for col in range(n):
        pivot = None
        for r in range(col, n):
            if int(R[r, col]) % p:
                pivot = r
                break
        if pivot is None:
            return {
                "ok": False,
                "failed_step": col,
                "failed_col": int(branch.tmpl.pivot_cols[col]) if col < len(branch.tmpl.pivot_cols) else None,
                "failed_square_col": col,
                "reused_prefix_pivots": col,
                "branch_index": branch.index,
                "branch_seed": branch.seed,
            }
        if pivot != col:
            R[[col, pivot]] = R[[pivot, col]]
        inv = pow(int(R[col, col]) % p, p - 2, p)
        R[col, :] = [(int(v) * inv) % p for v in R[col, :]]
        for rr in range(n):
            if rr == col:
                continue
            factor = int(R[rr, col]) % p
            if factor:
                R[rr, :] = [(int(R[rr, j]) - factor * int(R[col, j])) % p for j in range(n)]
    return {
        "ok": True,
        "failed_step": None,
        "failed_col": None,
        "failed_square_col": None,
        "reused_prefix_pivots": n,
        "branch_index": branch.index,
        "branch_seed": branch.seed,
    }


def best_square_block_fork_parent(
    eqs: Sequence[Mapping[Exp3, Any]],
    branches: Sequence["FaresStaticBranch"],
    *,
    p: int,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for branch in branches:
        try:
            info = square_block_failure_info(eqs, branch, p=p)
        except Exception as exc:
            info = {
                "ok": False,
                "failed_step": None,
                "failed_col": None,
                "failed_square_col": None,
                "reused_prefix_pivots": -1,
                "branch_index": branch.index,
                "branch_seed": branch.seed,
                "error": repr(exc),
            }
        if info.get("ok"):
            continue
        if best is None or int(info.get("reused_prefix_pivots") or -1) > int(best.get("reused_prefix_pivots") or -1):
            best = info
    return best


def ensure_c_root_refiner(
    *,
    yam_code_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    require_lapack: bool = True,
) -> Path | None:
    """Build the compiled eigensolve+Newton root extractor.

    The older path used NumPy for the action-matrix eigensolve and only called C
    for Newton refinement.  This helper builds the LAPACK-enabled C library that
    does both stages from the static-C action matrices.
    """

    yam = yc._yam()
    src = Path(yam_code_dir).resolve() / "pnp_root_refine.c" if yam_code_dir else Path(yam.__file__).resolve().with_name("pnp_root_refine.c")
    if not src.exists():
        if require_lapack:
            raise FileNotFoundError(f"missing C root refiner source: {src}")
        return None

    root = Path(out_dir).resolve() if out_dir is not None else src.parent
    root.mkdir(parents=True, exist_ok=True)
    so_path = root / "pnp_root_refine_lapack.so"
    if so_path.exists():
        return so_path

    cc = os.environ.get("CC", "cc")
    base_cmd = [
        cc,
        "-O3",
        "-std=c11",
        "-shared",
        "-fPIC",
        "-DPNP_USE_LAPACK",
        str(src),
        "-o",
        str(so_path),
        "-lm",
    ]
    attempts = [
        base_cmd + ["-llapack", "-lblas"],
        base_cmd + ["-lopenblas"],
        base_cmd + ["-framework", "Accelerate"],
    ]
    errors: list[str] = []
    for cmd in attempts:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if proc.returncode == 0:
            return so_path
        errors.append("+ " + " ".join(cmd) + "\n" + proc.stdout)

    if require_lapack:
        raise RuntimeError("failed to build LAPACK C root extractor:\n" + "\n".join(errors))
    return None


def ensure_direct_coeff_builder(
    *,
    yam_code_dir: str | Path,
    out_dir: str | Path,
) -> Path:
    src = Path(yam_code_dir) / "fares_direct_coeffs.c"
    if not src.exists():
        raise FileNotFoundError(f"missing direct coefficient builder source: {src}")
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    so_path = root / "fares_direct_coeffs.so"
    if so_path.exists() and so_path.stat().st_mtime >= src.stat().st_mtime:
        return so_path
    cc = os.environ.get("CC", "cc")
    subprocess.run(
        [cc, "-O3", "-std=c11", "-shared", "-fPIC", str(src), "-o", str(so_path), "-lm"],
        check=True,
    )
    return so_path


def _load_direct_coeff_fn(lib_path: str | Path):
    key = str(Path(lib_path).resolve())
    cached = _DIRECT_COEFF_CACHE.get(key)
    if cached is not None:
        return cached
    lib = ctypes.CDLL(key)
    fn = lib.fares_direct_coeffs
    fn.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_double,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
    ]
    fn.restype = ctypes.c_int
    _DIRECT_COEFF_CACHE[key] = fn
    return fn


def direct_coefficients_c(
    camera_matrix: np.ndarray,
    p3d: np.ndarray,
    p2d: np.ndarray,
    weights: np.ndarray,
    coeff_terms: Sequence[tuple[int, Exp3]],
    lib_path: str | Path,
    *,
    round_scale: float = 100.0,
    clear_scale: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    fn = _load_direct_coeff_fn(lib_path)
    K = np.ascontiguousarray(np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3))
    X = np.ascontiguousarray(np.asarray(p3d, dtype=np.float64).reshape(-1, 3))
    U = np.ascontiguousarray(np.asarray(p2d, dtype=np.float64).reshape(-1, 2))
    W = np.ascontiguousarray(np.asarray(weights, dtype=np.float64).reshape(-1))
    if X.shape[0] != U.shape[0] or X.shape[0] != W.shape[0]:
        raise ValueError("p3d, p2d and weights must have the same row count")
    term_eq = np.ascontiguousarray([int(i) for i, _e in coeff_terms], dtype=np.int32)
    exp_a = np.ascontiguousarray([int(e[0]) for _i, e in coeff_terms], dtype=np.int32)
    exp_b = np.ascontiguousarray([int(e[1]) for _i, e in coeff_terms], dtype=np.int32)
    exp_c = np.ascontiguousarray([int(e[2]) for _i, e in coeff_terms], dtype=np.int32)
    coeffs = np.zeros((len(coeff_terms),), dtype=np.float64)
    t0 = time.perf_counter()
    rc = int(
        fn(
            ctypes.c_int(X.shape[0]),
            K.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            U.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            W.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(len(coeff_terms)),
            term_eq.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            exp_a.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            exp_b.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            exp_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_double(round_scale),
            ctypes.c_double(clear_scale),
            coeffs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
    )
    elapsed = time.perf_counter() - t0
    if rc != 0:
        raise RuntimeError(f"direct FARES coefficient C builder failed with code {rc}")
    return coeffs, {
        "direct_coeff_seconds": elapsed,
        "compiled_coefficients": int(coeffs.shape[0]),
        "round_scale": float(round_scale),
        "clear_scale": float(clear_scale),
        "backend": "fares_direct_coeffs_c",
    }


def eqs_from_coefficients_float(
    coeffs: Sequence[float],
    coeff_terms: Sequence[tuple[int, Exp3]],
) -> list[dict[Exp3, float]]:
    eqs: list[dict[Exp3, float]] = [{}, {}, {}]
    for value, (eq_idx, exp) in zip(coeffs, coeff_terms):
        v = float(value)
        if v != 0.0:
            eqs[int(eq_idx)][tuple(exp)] = v
    return eqs


def load_static_branch_manifest(path: str | Path) -> FaresStaticBranch:
    manifest_path = Path(path).resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    tmpl = yc._yam().TreeTemplate.from_dict(data["template"])
    coeff_terms = [(int(i), tuple(int(v) for v in e)) for i, e in data["coeff_terms"]]

    def resolve_artifact(raw: str | None, default_name: str) -> Path:
        if raw:
            p = Path(raw)
            if p.exists():
                return p
            alt = manifest_path.parent / p.name
            if alt.exists():
                return alt
        return manifest_path.parent / default_name

    index = int(data.get("index", 0))
    seed = int(data.get("seed", 0))
    p = int(data.get("p", getattr(tmpl, "discovery_prime", 2147483647)))
    c_path = resolve_artifact(data.get("c_path"), f"fares_static_branch_{index}_seed_{seed}_p_{p}.c")
    so_path = resolve_artifact(data.get("so_path"), f"fares_static_branch_{index}_seed_{seed}_p_{p}.so")
    if not so_path.exists() and c_path.exists():
        compile_fares_static_action_kernel(c_path, so_path)
    learn_info = dict(data.get("learn_info") or {})
    preferred_action_weights = data.get("preferred_action_weights") or learn_info.get("preferred_action_weights")
    return FaresStaticBranch(
        index=index,
        seed=seed,
        p=p,
        tmpl=tmpl,
        coeff_terms=coeff_terms,
        c_path=c_path,
        so_path=so_path,
        learn_info=learn_info,
        preferred_action_weights=preferred_action_weights,
    )


def _load_root_extract_fn(root_refine_lib: str | Path | None):
    if not root_refine_lib:
        return None, None
    key = str(Path(root_refine_lib).resolve())
    cached = _ROOT_EXTRACT_CACHE.get(key)
    if cached is not None:
        return cached, key

    lib, lib_path = yc._yam().load_c_root_refiner(key)
    if lib is None or not hasattr(lib, "pnp_extract_roots_from_actions"):
        return None, lib_path

    fn = lib.pnp_extract_roots_from_actions
    fn.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
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
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    fn.restype = ctypes.c_int
    _ROOT_EXTRACT_CACHE[key] = fn
    return fn, key


def c_roots_from_numeric_action_matrices(
    eqs: Sequence[Mapping[Exp3, Any]],
    mats: Mapping[str, np.ndarray],
    *,
    action_weights: str | Sequence[Sequence[float]] | None,
    residual_tol: float,
    dedup_tol: float,
    real_imag_tol: float,
    max_newton_iter: int,
    max_abs_root: float,
    root_refine_lib: str | Path | None,
    target_valid_roots: int | None = None,
) -> dict[str, Any] | None:
    yam = yc._yam()
    fn, lib_path = _load_root_extract_fn(root_refine_lib)
    if fn is None:
        return None

    t0 = time.perf_counter()
    actions = [
        np.ascontiguousarray(np.asarray(mats["x"], dtype=np.float64)),
        np.ascontiguousarray(np.asarray(mats["y"], dtype=np.float64)),
        np.ascontiguousarray(np.asarray(mats["z"], dtype=np.float64)),
    ]
    qdim = int(actions[0].shape[0])
    weights_list = yam.parse_action_weights(action_weights)
    weights = np.ascontiguousarray(weights_list, dtype=np.float64).reshape(-1)
    max_seeds = max(1, qdim * len(weights_list))
    target_valid_roots = int(target_valid_roots or qdim)

    term_eq, exp_a, exp_b, exp_c, coeff_re, coeff_im = flatten_equation_terms_float(eqs)
    norms = equation_norms_float(eqs)
    out_re = np.zeros(max_seeds * 3, dtype=np.float64)
    out_im = np.zeros(max_seeds * 3, dtype=np.float64)
    out_res = np.zeros(max_seeds, dtype=np.float64)
    out_ok = np.zeros(max_seeds, dtype=np.int32)
    out_iters = np.zeros(max_seeds, dtype=np.int32)
    out_seed_count = np.zeros(1, dtype=np.int32)
    out_eig_seconds = np.zeros(1, dtype=np.float64)
    out_newton_seconds = np.zeros(1, dtype=np.float64)

    rc = int(
        fn(
            ctypes.c_int(qdim),
            actions[0].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            actions[1].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            actions[2].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(len(weights_list)),
            weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
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
            ctypes.c_int(target_valid_roots),
            ctypes.c_int(max_seeds),
            out_re.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out_im.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out_res.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out_ok.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            out_iters.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            out_seed_count.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            out_eig_seconds.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out_newton_seconds.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
    )
    if rc < 0:
        raise RuntimeError(f"C action-root extraction failed with code {rc}")

    seed_count = min(int(out_seed_count[0]), max_seeds)
    roots: list[tuple[tuple[complex, complex, complex], float]] = []
    for i in range(seed_count):
        if int(out_ok[i]):
            root = tuple(complex(out_re[3 * i + j], out_im[3 * i + j]) for j in range(3))
            roots.append((root, float(out_res[i])))

    roots = yam.deduplicate_roots(roots, distance_tol=dedup_tol)
    root_values = [root for root, _rel in roots]
    residuals = [float(rel) for _root, rel in roots]
    real_roots = yam.real_root_centers_from_complex(root_values, imag_tol=real_imag_tol)
    total_sec = time.perf_counter() - t0
    return {
        "ok": bool(real_roots),
        "reason": "static-C action roots extracted in C" if real_roots else "no C action roots passed filters",
        "method": "static_c_action_double_lapack_root",
        "action_weights": weights_list,
        "seed_count": seed_count,
        "target_valid_roots": target_valid_roots,
        "eig_failures": 0,
        "root_count": len(root_values),
        "real_root_count": len(real_roots),
        "roots": root_values,
        "real_roots": real_roots,
        "root_residuals": residuals,
        "max_relative_residual": max(residuals) if residuals else None,
        "median_relative_residual": float(np.median(residuals)) if residuals else None,
        "eig_sec": float(out_eig_seconds[0]),
        "root_refine_backend": "c_lapack",
        "root_refine_library": lib_path,
        "root_refine_ok_count": int(rc),
        "newton_sec": float(out_newton_seconds[0]),
        "total_sec": total_sec,
        "ctypes_overhead_sec": max(0.0, total_sec - float(out_eig_seconds[0]) - float(out_newton_seconds[0])),
    }


def roots_from_numeric_action_matrices(
    eqs: Sequence[Mapping[Exp3, Any]],
    mats: Mapping[str, np.ndarray],
    *,
    action_weights: str | Sequence[Sequence[float]] | None = "1,7,11;3,17,5;5,11,19;1,0,0;0,1,0;0,0,1",
    residual_tol: float = 1e-8,
    dedup_tol: float = 1e-6,
    real_imag_tol: float = 1e-7,
    max_newton_iter: int = 40,
    max_abs_root: float = 500.0,
    root_refine_backend: str = "auto",
    root_refine_lib: str | Path | None = None,
    target_valid_roots: int | None = None,
) -> dict[str, Any]:
    yam = yc._yam()
    t0 = time.perf_counter()
    backend = str(root_refine_backend or "auto").lower()
    if backend not in {"auto", "python", "c"}:
        raise ValueError(f"unknown root refinement backend: {root_refine_backend}")

    if backend in {"auto", "c"} and root_refine_lib:
        c_info = c_roots_from_numeric_action_matrices(
            eqs,
            mats,
            action_weights=action_weights,
            residual_tol=residual_tol,
            dedup_tol=dedup_tol,
            real_imag_tol=real_imag_tol,
            max_newton_iter=max_newton_iter,
            max_abs_root=max_abs_root,
            root_refine_lib=root_refine_lib,
            target_valid_roots=target_valid_roots,
        )
        if c_info is not None:
            return c_info
        if backend == "c":
            raise FileNotFoundError("C LAPACK root extractor was requested but not found")

    actions = [
        np.asarray(mats["x"], dtype=np.complex128),
        np.asarray(mats["y"], dtype=np.complex128),
        np.asarray(mats["z"], dtype=np.complex128),
    ]
    weights = yam.parse_action_weights(action_weights)
    seeds: list[tuple[complex, complex, complex]] = []
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
    roots, refine_info = yam.refine_candidate_roots(
        eqs,
        seeds,
        residual_tol=residual_tol,
        max_newton_iter=max_newton_iter,
        max_abs_root=max_abs_root,
        backend=root_refine_backend,
        root_refine_lib=root_refine_lib,
    )
    newton_sec = time.perf_counter() - newton_t0
    roots = yam.deduplicate_roots(roots, distance_tol=dedup_tol)
    root_values = [root for root, _rel in roots]
    residuals = [float(rel) for _root, rel in roots]
    real_roots = yam.real_root_centers_from_complex(root_values, imag_tol=real_imag_tol)
    return {
        "ok": bool(real_roots),
        "reason": "static-C action roots extracted" if real_roots else "no static-C action roots passed filters",
        "method": "static_c_action_double",
        "action_weights": weights,
        "seed_count": len(seeds),
        "eig_failures": eig_failures,
        "root_count": len(root_values),
        "real_root_count": len(real_roots),
        "roots": root_values,
        "real_roots": real_roots,
        "root_residuals": residuals,
        "max_relative_residual": max(residuals) if residuals else None,
        "median_relative_residual": float(np.median(residuals)) if residuals else None,
        "eig_sec": eig_sec,
        "root_refine_backend": refine_info.get("backend"),
        "root_refine_library": refine_info.get("library"),
        "root_refine_ok_count": refine_info.get("ok_count"),
        "newton_sec": newton_sec,
        "total_sec": time.perf_counter() - t0,
    }


def quaternions_from_numeric_action_matrices(
    eqs: Sequence[Mapping[Exp3, Any]],
    mats: Mapping[str, np.ndarray],
    *,
    max_roots: int = 80,
    **root_kwargs: Any,
) -> tuple[list[list[float]], dict[str, Any]]:
    roots_info = roots_from_numeric_action_matrices(eqs, mats, **root_kwargs)
    quats: list[list[float]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for root in (roots_info.get("real_roots") or [])[:max_roots]:
        q = yc.normalize_quaternion_from_affine_root(root)
        key = tuple(int(round(v * 1e10)) for v in q)
        neg_key = tuple(int(round(-v * 1e10)) for v in q)
        if key not in seen and neg_key not in seen:
            quats.append(q)
            seen.add(key)
    info = {k: v for k, v in roots_info.items() if k not in {"roots", "real_roots"}}
    info["quaternion_count"] = len(quats)
    return quats, info


def _quaternion_merge_key(q: Sequence[float], *, scale: float = 1e10) -> tuple[int, int, int, int]:
    """Sign-invariant key for merging equivalent quaternion candidates."""

    arr = [float(v) for v in q]
    key = tuple(int(round(v * scale)) for v in arr)
    neg_key = tuple(int(round(-v * scale)) for v in arr)
    return min(key, neg_key)


def quaternions_from_numeric_action_matrices_with_fallback(
    eqs: Sequence[Mapping[Exp3, Any]],
    mats: Mapping[str, np.ndarray],
    *,
    action_weights: str | Sequence[Sequence[float]] | None,
    fallback_action_weights: str | Sequence[Sequence[float]] | None = DEFAULT_ACTION_WEIGHTS,
    min_root_count: int | None = None,
    min_quaternion_count: int | None = None,
    residual_health_tol: float | None = None,
    max_roots: int = 80,
    candidate_profile: str = "full",
    **root_kwargs: Any,
) -> tuple[list[list[float]], dict[str, Any]]:
    """Extract candidates with a fast separator first, then full fallback.

    A single separating linear form represents all quotient roots, so the common
    fast path should not diagonalize six projections.  We first use the selected
    branch weight and ask for the quotient-size root set.  If the numeric root
    stage returns too few quaternion candidates, we rerun one batched C call over
    the full stored separator list.  This keeps the earlier fast behavior while
    still recovering cases where the best FARES pose was dropped.

    The fallback target is qdim times the number of stored weights, which forces
    the C extractor not to stop after the first projection.
    """

    yam = yc._yam()
    primary = yam.parse_action_weights(action_weights or DEFAULT_ACTION_WEIGHTS)
    fallback = yam.parse_action_weights(fallback_action_weights or DEFAULT_ACTION_WEIGHTS)
    ordered: list[list[float]] = []
    seen_weights: set[str] = set()
    for weight in [*primary, *fallback]:
        key = ",".join(f"{float(v):.17g}" for v in weight)
        if key not in seen_weights:
            ordered.append([float(v) for v in weight])
            seen_weights.add(key)

    qdim = int(np.asarray(mats["x"]).shape[0])
    need_roots = int(min_root_count or qdim)
    profile = str(candidate_profile or "full").strip().lower()
    if profile not in {"full", "live_fast"}:
        raise ValueError(f"unknown root candidate profile: {candidate_profile}")
    default_quats = 8 if profile == "live_fast" else max(24, min(48, need_roots // 2))
    need_quats = int(min_quaternion_count or default_quats)
    full_candidate_target = max(need_roots, qdim * len(ordered))
    health_tol = float(residual_health_tol if residual_health_tol is not None else root_kwargs.get("residual_tol", 1e-8))

    backend = str(root_kwargs.get("root_refine_backend") or "auto").lower()
    root_lib = root_kwargs.get("root_refine_lib")
    if backend in {"auto", "c"} and root_lib:
        quick_kwargs = dict(root_kwargs)
        quick_kwargs.pop("target_valid_roots", None)
        quick_kwargs.setdefault("dedup_tol", 1e-12)
        quick_kwargs.setdefault("real_imag_tol", 1e-5)
        quick_quats, quick_info = quaternions_from_numeric_action_matrices(
            eqs,
            mats,
            max_roots=max_roots,
            action_weights=format_action_weights([ordered[0]]),
            target_valid_roots=need_roots,
            **quick_kwargs,
        )
        quick_max_res = quick_info.get("max_relative_residual")
        quick_root_floor = need_quats if profile == "live_fast" else need_roots
        quick_healthy = bool(
            quick_info.get("ok")
            and int(quick_info.get("root_count") or 0) >= quick_root_floor
            and len(quick_quats) >= need_quats
            and (quick_max_res is None or float(quick_max_res) <= health_tol)
        )
        if quick_healthy:
            quick_info.update(
                {
                    "method": "static_c_action_double_lapack_root_preferred_weight",
                    "fallback_used": False,
                    "attempt_count": 1,
                    "stopped_reason": "preferred separator produced a complete candidate set",
                    "required_root_count": need_roots,
                    "required_preferred_root_count": quick_root_floor,
                    "required_quaternion_count": need_quats,
                    "full_candidate_target": need_roots,
                    "quaternion_count": len(quick_quats),
                    "batched_weight_count": 1,
                    "batched_weights": [ordered[0]],
                    "fallback_available_weights": ordered[1:],
                    "candidate_profile": profile,
                }
            )
            return quick_quats, quick_info

        # Robust fallback: the LAPACK C extractor supports a list of weights and
        # stops after target_ok_roots are refined.  Set the target to all
        # possible seeds, not qdim, so every stored projection contributes
        # candidates before FARES scoring chooses the final pose.
        batched_kwargs = dict(root_kwargs)
        batched_kwargs.pop("target_valid_roots", None)
        batched_kwargs.setdefault("dedup_tol", 1e-12)
        batched_kwargs.setdefault("real_imag_tol", 1e-5)
        quats, info = quaternions_from_numeric_action_matrices(
            eqs,
            mats,
            max_roots=max_roots,
            action_weights=format_action_weights(ordered),
            target_valid_roots=full_candidate_target,
            **batched_kwargs,
        )
        max_res = info.get("max_relative_residual")
        healthy = bool(
            info.get("ok")
            and int(info.get("root_count") or 0) >= need_roots
            and (max_res is None or float(max_res) <= health_tol)
        )
        info.update(
            {
                "method": "static_c_action_double_lapack_root_batched_weight_fallback",
                "fallback_used": True,
                "attempt_count": 1,
                "stopped_reason": "healthy candidate set" if healthy else "C extractor exhausted batched fallback weights",
                "required_root_count": need_roots,
                "required_quaternion_count": need_quats,
                "full_candidate_target": full_candidate_target,
                "quaternion_count": len(quats),
                "batched_weight_count": len(ordered),
                "batched_weights": ordered,
                "preferred_attempt": {k: v for k, v in quick_info.items() if k not in {"roots", "real_roots"}},
                "preferred_quaternion_count": len(quick_quats),
                "candidate_profile": profile,
            }
        )
        return quats, info

    merged: list[list[float]] = []
    seen_quats: set[tuple[int, int, int, int]] = set()
    attempts: list[dict[str, Any]] = []
    total_sec = 0.0
    eig_sec = 0.0
    newton_sec = 0.0
    ctypes_overhead_sec = 0.0
    best_info: dict[str, Any] | None = None
    stopped_reason = "exhausted fallback weights"

    for idx, weight in enumerate(ordered):
        weight_text = format_action_weights([weight])
        quats, info = quaternions_from_numeric_action_matrices(
            eqs,
            mats,
            max_roots=max_roots,
            action_weights=weight_text,
            target_valid_roots=need_roots,
            **root_kwargs,
        )
        for q in quats:
            q_key = _quaternion_merge_key(q)
            if q_key not in seen_quats:
                merged.append(q)
                seen_quats.add(q_key)

        max_res = info.get("max_relative_residual")
        root_count = int(info.get("root_count") or 0)
        healthy = bool(
            info.get("ok")
            and root_count >= need_roots
            and (max_res is None or float(max_res) <= health_tol)
        )
        attempt = {k: v for k, v in info.items() if k not in {"roots", "real_roots"}}
        attempt.update(
            {
                "attempt_index": idx,
                "weight": [float(v) for v in weight],
                "weight_key": weight_text,
                "healthy": healthy,
                "merged_quaternion_count_after_attempt": len(merged),
            }
        )
        attempts.append(attempt)
        total_sec += float(info.get("total_sec") or 0.0)
        eig_sec += float(info.get("eig_sec") or 0.0)
        newton_sec += float(info.get("newton_sec") or 0.0)
        ctypes_overhead_sec += float(info.get("ctypes_overhead_sec") or 0.0)

        if best_info is None or root_count > int(best_info.get("root_count") or 0):
            best_info = attempt
        if healthy and merged and stopped_reason == "exhausted fallback weights":
            stopped_reason = "healthy candidate set; continued through all stored weights for full candidate scoring"

    selected_weights = [a["weight"] for a in attempts]
    max_root_count = max((int(a.get("root_count") or 0) for a in attempts), default=0)
    max_real_root_count = max((int(a.get("real_root_count") or 0) for a in attempts), default=0)
    max_residuals = [a.get("max_relative_residual") for a in attempts if a.get("max_relative_residual") is not None]
    med_residuals = [a.get("median_relative_residual") for a in attempts if a.get("median_relative_residual") is not None]
    info = {
        "ok": bool(merged),
        "reason": "fallback weight merge returned quaternion candidates" if merged else "no fallback weight produced quaternion candidates",
        "method": "static_c_action_double_lapack_root_with_weight_fallback",
        "action_weights": selected_weights,
        "fallback_used": len(attempts) > 1,
        "attempt_count": len(attempts),
        "attempts": attempts,
        "stopped_reason": stopped_reason,
        "required_root_count": need_roots,
        "full_candidate_target": full_candidate_target,
        "root_count": max_root_count,
        "real_root_count": max_real_root_count,
        "quaternion_count": len(merged),
        "max_relative_residual": max(max_residuals) if max_residuals else None,
        "median_relative_residual": float(np.median(med_residuals)) if med_residuals else None,
        "eig_sec": eig_sec,
        "newton_sec": newton_sec,
        "total_sec": total_sec,
        "ctypes_overhead_sec": ctypes_overhead_sec,
        "best_attempt": best_info,
    }
    return merged[:max_roots], info


def actions_to_numpy(actions: Mapping[str, Sequence[Sequence[int]]]) -> dict[str, np.ndarray]:
    return {k: np.asarray(actions[k], dtype=np.int64) for k in ("x", "y", "z")}


def matrices_equal(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray], p: int) -> bool:
    return all(np.array_equal(np.asarray(a[k]) % p, np.asarray(b[k]) % p) for k in ("x", "y", "z"))


def matmul_np_mod(A: np.ndarray, B: np.ndarray, p: int) -> np.ndarray:
    return (A.astype(object) @ B.astype(object) % p).astype(np.int64)


def verify_fares_actions(
    eqs: Sequence[Mapping[Exp3, Any]],
    mats: Mapping[str, np.ndarray],
    p: int,
) -> dict[str, Any]:
    names = ("x", "y", "z")
    actions = [np.asarray(mats[n], dtype=np.int64) % p for n in names]
    qdim = actions[0].shape[0]
    ident = np.eye(qdim, dtype=np.int64)

    def power_product(exp: Exp3) -> np.ndarray:
        out = ident.copy()
        for A, power in zip(actions, exp):
            for _ in range(int(power)):
                out = matmul_np_mod(out, A, p)
        return out

    ideal: list[int] = []
    for eq in eqs:
        acc = np.zeros((qdim, qdim), dtype=np.int64)
        for exp, coeff in eq.items():
            c = coeff_mod_fraction(coeff, p)
            if c:
                acc = (acc.astype(object) + c * power_product(tuple(exp)).astype(object)) % p
                acc = acc.astype(np.int64)
        ideal.append(int(np.max(acc % p)) if acc.size else 0)

    comm: dict[str, int] = {}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i >= j:
                continue
            C = (matmul_np_mod(mats[a], mats[b], p) - matmul_np_mod(mats[b], mats[a], p)) % p
            comm[f"{a}{b}"] = int(np.max(C)) if C.size else 0

    return {
        "ok": all(v == 0 for v in ideal) and all(v == 0 for v in comm.values()),
        "ideal": ideal,
        "commutators": comm,
    }


@dataclass
class FaresStaticBranch:
    index: int
    seed: int
    p: int
    tmpl: Any
    coeff_terms: list[tuple[int, Exp3]]
    c_path: Path
    so_path: Path
    learn_info: dict[str, Any]
    preferred_action_weights: str | None = None

    def to_manifest(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "seed": self.seed,
            "p": self.p,
            "template": self.tmpl.to_dict(),
            "coeff_terms": [[i, list(e)] for i, e in self.coeff_terms],
            "c_path": str(self.c_path),
            "so_path": str(self.so_path),
            "learn_info": self.learn_info,
            "preferred_action_weights": self.preferred_action_weights
            or self.learn_info.get("preferred_action_weights"),
        }


def learn_static_branch(
    eqs: Sequence[Mapping[Exp3, Any]],
    *,
    seed: int,
    p: int,
    out_dir: str | Path,
    index: int,
    degree: int = 11,
) -> FaresStaticBranch:
    patch_core_coeff_mod()
    out_dir = Path(out_dir)
    tmpl, info = yc.build_template_from_dehom_eqs(
        eqs,
        degree=degree,
        template_id=index + 1,
        quotient_degree=None,
        discovery_prime=p,
    )
    coeff_terms = coeff_terms_from_template(tmpl, eqs)
    c_path = out_dir / f"fares_static_branch_{index}_seed_{seed}_p_{p}.c"
    so_path = out_dir / f"fares_static_branch_{index}_seed_{seed}_p_{p}.so"
    generate_fares_static_action_kernel(tmpl, coeff_terms, c_path)
    compile_fares_static_action_kernel(c_path, so_path)
    branch = FaresStaticBranch(index, seed, p, tmpl, coeff_terms, c_path, so_path, info)
    (out_dir / f"fares_static_branch_{index}_seed_{seed}_p_{p}.json").write_text(
        json.dumps(branch.to_manifest(), indent=2),
        encoding="utf-8",
    )
    return branch


def ensure_static_branch(
    eqs: Sequence[Mapping[Exp3, Any]],
    *,
    branches: list[FaresStaticBranch],
    p: int,
    seed: int,
    out_dir: str | Path,
    degree: int = 11,
    try_existing: bool = True,
) -> tuple[int, FaresStaticBranch, dict[str, np.ndarray], dict[str, Any], list[str], bool]:
    errors: list[str] = []
    if try_existing:
        for branch in branches:
            try:
                mats, meta = replay_fares_actions_static_c(eqs, branch.tmpl, branch.coeff_terms, branch.so_path, p=p)
                return branch.index, branch, mats, meta, errors, False
            except Exception as exc:
                errors.append(f"branch {branch.index} seed {branch.seed}: {exc}")

    branch = learn_static_branch(eqs, seed=seed, p=p, out_dir=out_dir, index=len(branches), degree=degree)
    fork_info = best_square_block_fork_parent(eqs, branches, p=p)
    if fork_info:
        branch.learn_info.update(
            {
                "learn_method": "fares_square_block_fork_miss_full_branch",
                "fork_from_branch": fork_info.get("branch_index"),
                "parent_seed": fork_info.get("branch_seed"),
                "failed_step": fork_info.get("failed_step"),
                "failed_col": fork_info.get("failed_col"),
                "failed_square_col": fork_info.get("failed_square_col"),
                "reused_prefix_pivots": fork_info.get("reused_prefix_pivots"),
                "suffix_pivots": max(0, len(branch.tmpl.pivot_cols) - int(fork_info.get("reused_prefix_pivots") or 0)),
                "prefix_reuse_note": (
                    "FARES branch currently stores a square action block, not the full RREF trace; "
                    "this records the exact fork point and learns the replacement branch."
                ),
            }
        )
        (Path(out_dir) / f"fares_static_branch_{branch.index}_seed_{branch.seed}_p_{branch.p}.json").write_text(
            json.dumps(branch.to_manifest(), indent=2),
            encoding="utf-8",
        )
    branches.append(branch)
    mats, meta = replay_fares_actions_static_c(eqs, branch.tmpl, branch.coeff_terms, branch.so_path, p=p)
    if fork_info:
        meta["fork_info"] = fork_info
    return branch.index, branch, mats, meta, errors, True


def full_exact_fares_action_matrices(
    eqs: Sequence[Mapping[Exp3, Any]],
    *,
    p: int,
    degree: int = 11,
) -> tuple[Any, dict[str, np.ndarray], dict[str, Any]]:
    patch_core_coeff_mod()
    t0 = time.perf_counter()
    tmpl, info = yc.build_template_from_dehom_eqs(eqs, degree=degree, template_id=0, discovery_prime=p)
    actions, checks = yc.project_action_matrices_mod_prime(tmpl, eqs, prime=p)
    elapsed = time.perf_counter() - t0
    meta = dict(info)
    meta.update(checks)
    meta["full_seconds"] = elapsed
    return tmpl, actions_to_numpy(actions), meta
