#!/usr/bin/env python3
"""msolve-compatible YSolve adapter for the FARES PnP pipeline.

FARES expects ``pnp_poly_solvers.solve_with_msolve(equations)`` to return a
list of quaternion candidates.  This module exposes the same contract while
replacing msolve's algebraic stage by:

    FARES equations
      -> q0=1 dehomogenized FARES quartics
      -> static-C action-matrix replay
      -> root extraction from the action matrices
      -> quaternion candidates

The adapter records the algebraic objects and timings used to produce the
candidates.  The independent proof path can compare the same static-C actions
and Shape/RUR against clean msolve; FARES itself only needs the candidates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import ysolve_template_core as yc
from fares_static_c_replay import (
    DEFAULT_ACTION_WEIGHTS,
    format_action_weights,
    quaternions_from_numeric_action_matrices_with_fallback,
    rationalize_eqs,
    replay_fares_actions_static_c,
    replay_fares_actions_static_c_double,
    verify_fares_actions,
)


Exp3 = tuple[int, int, int]


def add_stage(stages: dict[str, float], name: str, ms: float) -> None:
    stages[name] = float(stages.get(name, 0.0)) + float(ms)


def _weight_key(weight: Sequence[float]) -> str:
    return ",".join(f"{float(v):.17g}" for v in weight)


def select_separating_linear_form(
    mats: Mapping[str, np.ndarray],
    candidates: str | Sequence[Sequence[float]] | None,
    *,
    sep_tol: float = 1e-10,
    cond_max: float = 1e14,
) -> dict[str, Any]:
    """Choose a checked linear form ``t = w_x x + w_y y + w_z z``.

    The action matrices encode multiplication by ``x,y,z`` in the quotient
    algebra.  To extract roots robustly we diagonalize one linear combination
    ``A_t``.  That is valid only when the chosen ``t`` separates the solutions:
    the eigenvalues of ``A_t`` should be distinct.  This routine tests the
    candidate forms and returns the first one whose spectrum is separated and
    whose eigenvector matrix is not catastrophically ill-conditioned.

    This is the runtime guard for the numeric path.  The proof notebook still
    verifies Shape/RUR equivalence against clean msolve over a finite field.
    """

    weights = yc._yam().parse_action_weights(candidates)
    if not weights:
        raise ValueError("no action-weight candidates were supplied")

    actions = [
        np.asarray(mats["x"], dtype=np.float64),
        np.asarray(mats["y"], dtype=np.float64),
        np.asarray(mats["z"], dtype=np.float64),
    ]
    qdim = int(actions[0].shape[0])
    tested: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    accepted_records: list[dict[str, Any]] = []
    for weight in weights:
        w = np.asarray(weight, dtype=np.float64)
        t0 = time.perf_counter()
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
            record = {
                "weight": [float(x) for x in weight],
                "key": _weight_key(weight),
                "accepted": accepted,
                "normalized_min_eigenvalue_separation": min_sep,
                "eigenvector_condition": cond,
                "elapsed_ms": 1000.0 * (time.perf_counter() - t0),
            }
        except Exception as exc:
            record = {
                "weight": [float(x) for x in weight],
                "key": _weight_key(weight),
                "accepted": False,
                "error": repr(exc),
                "normalized_min_eigenvalue_separation": 0.0,
                "eigenvector_condition": float("inf"),
                "elapsed_ms": 1000.0 * (time.perf_counter() - t0),
            }

        tested.append(record)
        if record["accepted"]:
            accepted_records.append(record)
            return {
                "ok": True,
                "selected_weight": record["weight"],
                "selected_key": record["key"],
                "accepted_weights": [record["weight"]],
                "accepted_keys": [record["key"]],
                "sep_tol": sep_tol,
                "cond_max": cond_max,
                "tested": tested,
            }
        if best is None:
            best = record
        else:
            best_score = (
                float(best.get("normalized_min_eigenvalue_separation") or 0.0),
                -float(best.get("eigenvector_condition") or float("inf")),
            )
            record_score = (
                float(record.get("normalized_min_eigenvalue_separation") or 0.0),
                -float(record.get("eigenvector_condition") or float("inf")),
            )
            if record_score > best_score:
                best = record

    if accepted_records:
        selected = accepted_records[0]
        return {
            "ok": True,
            "selected_weight": selected["weight"],
            "selected_key": selected["key"],
            "accepted_weights": [r["weight"] for r in accepted_records],
            "accepted_keys": [r["key"] for r in accepted_records],
            "sep_tol": sep_tol,
            "cond_max": cond_max,
            "tested": tested,
        }

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


@dataclass
class YSolveMsolveAdapter:
    """Drop-in replacement for ``solve_with_msolve``.

    Parameters are deliberately the same objects produced by the offline
    template learner: one static branch, its floating static-C action kernel,
    and an optional C LAPACK root extractor.  The public ``solve`` method has
    the exact call shape FARES expects: ``solve(equations) -> quaternions``.
    """

    branch: Any
    double_so: str | Path
    root_refiner: str | Path | None = None
    equation_mode: str = "first3"
    action_weights: str | Sequence[Sequence[float]] | None = "branch"
    verify_linear_form: bool = True
    linear_form_sep_tol: float = 1e-10
    linear_form_cond_max: float = 1e14
    root_residual_tol: float = 1e-8
    root_refine_backend: str = "auto"
    max_roots: int = 80
    fallback_action_weights: str | Sequence[Sequence[float]] | None = DEFAULT_ACTION_WEIGHTS
    verify_modular: bool = False
    prime: int = 2147483647
    stages: dict[str, float] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def resolved_action_weights(self) -> str | Sequence[Sequence[float]] | None:
        if isinstance(self.action_weights, str) and self.action_weights.strip().lower() == "branch":
            return (
                getattr(self.branch, "preferred_action_weights", None)
                or getattr(self.branch, "learn_info", {}).get("preferred_action_weights")
                or DEFAULT_ACTION_WEIGHTS
            )
        return self.action_weights

    def dehomogenize(self, equations: Sequence[Any]) -> list[dict[Exp3, Any]]:
        t0 = time.perf_counter()
        eqs = rationalize_eqs(yc.dehomogenize_fares_equations(equations, mode=self.equation_mode))
        add_stage(self.stages, "ysolve_dehomogenize_rationalize_ms", 1000.0 * (time.perf_counter() - t0))
        return eqs

    def replay_actions_float(
        self,
        eqs: Sequence[Mapping[Exp3, Any]],
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        mats, meta = replay_fares_actions_static_c_double(
            eqs,
            self.branch.tmpl,
            self.branch.coeff_terms,
            self.double_so,
        )
        add_stage(self.stages, "ysolve_static_action_double_ms", 1000.0 * float(meta["replay_seconds"]))
        return mats, meta

    def replay_actions_modular(
        self,
        eqs: Sequence[Mapping[Exp3, Any]],
        *,
        p: int | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
        prime = int(p or self.prime)
        mats, meta = replay_fares_actions_static_c(
            eqs,
            self.branch.tmpl,
            self.branch.coeff_terms,
            self.branch.so_path,
            p=prime,
        )
        verify = verify_fares_actions(eqs, mats, prime)
        add_stage(self.stages, "ysolve_static_modular_verify_ms", 1000.0 * float(meta["replay_seconds"]))
        return mats, meta, verify

    def roots_to_quaternions(
        self,
        eqs: Sequence[Mapping[Exp3, Any]],
        mats: Mapping[str, np.ndarray],
        *,
        action_weights: str | Sequence[Sequence[float]] | None = None,
    ) -> tuple[list[list[float]], dict[str, Any]]:
        backend = self.root_refine_backend
        if backend == "auto" and self.root_refiner:
            backend = "c"
        quats, roots_info = quaternions_from_numeric_action_matrices_with_fallback(
            eqs,
            mats,
            action_weights=self.action_weights if action_weights is None else action_weights,
            fallback_action_weights=self.fallback_action_weights,
            residual_tol=self.root_residual_tol,
            residual_health_tol=self.root_residual_tol,
            root_refine_backend=backend,
            root_refine_lib=self.root_refiner,
            max_roots=self.max_roots,
        )
        add_stage(self.stages, "ysolve_static_root_total_ms", 1000.0 * float(roots_info.get("total_sec", 0.0) or 0.0))
        add_stage(self.stages, "ysolve_static_root_eig_ms", 1000.0 * float(roots_info.get("eig_sec", 0.0) or 0.0))
        add_stage(self.stages, "ysolve_static_root_newton_ms", 1000.0 * float(roots_info.get("newton_sec", 0.0) or 0.0))
        add_stage(
            self.stages,
            "ysolve_static_root_ctypes_overhead_ms",
            1000.0 * float(roots_info.get("ctypes_overhead_sec", 0.0) or 0.0),
        )
        return quats, roots_info

    def solve(self, equations: Sequence[Any]) -> list[list[float]]:
        """Return quaternion candidates, exactly like FARES' msolve wrapper."""

        call_t0 = time.perf_counter()
        eqs = self.dehomogenize(equations)

        modular_verify = None
        if self.verify_modular:
            _mod_mats, mod_meta, modular_verify = self.replay_actions_modular(eqs, p=self.prime)
            modular_verify["meta"] = mod_meta
            if not modular_verify.get("ok"):
                raise ArithmeticError(f"YSolve modular action verification failed: {modular_verify}")

        mats, action_meta = self.replay_actions_float(eqs)
        linear_form_info = None
        root_weights = self.resolved_action_weights()
        if self.verify_linear_form:
            t_lf = time.perf_counter()
            linear_form_info = select_separating_linear_form(
                mats,
                root_weights,
                sep_tol=self.linear_form_sep_tol,
                cond_max=self.linear_form_cond_max,
            )
            add_stage(self.stages, "ysolve_linear_form_select_ms", 1000.0 * (time.perf_counter() - t_lf))
            if not linear_form_info.get("ok"):
                raise RuntimeError(f"no verified separating linear form: {linear_form_info}")
            selected = linear_form_info.get("selected_weight")
            if selected is not None:
                all_weights = yc._yam().parse_action_weights(root_weights)
                selected_key = _weight_key(selected)
                ordered = [[float(v) for v in selected]]
                ordered.extend([w for w in all_weights if _weight_key(w) != selected_key])
                root_weights = format_action_weights(ordered)

        quats, roots_info = self.roots_to_quaternions(eqs, mats, action_weights=root_weights)

        call_info = {
            "ok": bool(quats),
            "reason": "YSolve static-C quaternions returned" if quats else roots_info.get("reason"),
            "adapter_contract": "solve_with_msolve(equations)->quaternion_candidates",
            "algebraic_boundary": "FARES equations -> dehom quartics -> static-C actions -> root candidates",
            "branch_index": getattr(self.branch, "index", None),
            "branch_seed": getattr(self.branch, "seed", None),
            "branch_prime": getattr(self.branch, "p", None),
            "quotient_dim": len(getattr(self.branch.tmpl, "basis_cols", [])),
            "pivot_count": len(getattr(self.branch.tmpl, "pivot_cols", [])),
            "action_meta": action_meta,
            "roots_info": roots_info,
            "linear_form_selection": linear_form_info,
            "modular_verify": modular_verify,
            "quaternion_count": len(quats),
            "adapter_call_ms": 1000.0 * (time.perf_counter() - call_t0),
        }
        self.calls.append(call_info)
        add_stage(self.stages, "polynomial_solver_total_ms", call_info["adapter_call_ms"])

        if not quats:
            raise RuntimeError(f"YSolve produced no quaternion candidates: {roots_info.get('reason')}")
        return quats

    __call__ = solve
