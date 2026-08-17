#!/usr/bin/env python3
"""Online runner for the PnP symbolic-template tree.

Upload this file to Colab together with ``pnp_symbolic_tree_offline.py`` and
``task3_msolve.zip``.  Run the offline file once, then run this file repeatedly.

The online path loads the saved tree, fills the numeric PnP equations for each
new instance, tries existing symbolic branches, and only creates a new branch
when all existing branches reject.  Successful new branches are saved back to
the same tree file, so a second run should reuse more templates.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from pnp_symbolic_tree_offline import (
    PnPSymbolicTree,
    best_root_by_score,
    generate_k_digit_matrices,
    load_tree,
    msolve_input_text,
    root_match_summary,
    replay_template,
    residual_summary,
    root_relative_residuals,
    run_msolve_baseline,
    save_tree,
    sha256_text,
    setup_msolve_from_zip,
    tree_roots_from_macaulay,
    tree_roots_from_quotient_projection,
    tree_roots_from_template_replay,
    wedge_equations_3eq,
)


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


def finite(values: Iterable[Optional[float]]) -> list[float]:
    return [float(x) for x in values if x is not None]


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    xs = finite(values)
    return None if not xs else float(statistics.mean(xs))


def median_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    xs = finite(values)
    return None if not xs else float(statistics.median(xs))


def mode_set(text: str) -> set[str]:
    text = str(text or "all").strip().lower()
    if text == "all":
        return {"square", "lstsq"}
    return {x.strip() for x in text.split(",") if x.strip()}


def pct(numer: float, denom: float) -> float:
    return 100.0 * numer / denom if denom else 0.0


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def print_rows(rows: list[dict[str, Any]], max_rows: int = 30) -> None:
    if not rows:
        return
    cols = [
        "pass",
        "seed",
        "tree_accepted",
        "template_id",
        "new_branch_created",
        "tree_total_ms",
        "solve_mode",
        "tree_reason",
        "fares_msolve_ms",
        "fares_raw_ms",
        "fares_returncode",
    ]
    if any(row.get("tree_root_extraction_enabled") for row in rows):
        insert_at = cols.index("solve_mode")
        cols[insert_at:insert_at] = ["tree_root_count", "tree_real_root_count"]
    widths = {c: len(c) for c in cols}
    visible = rows[:max_rows]
    for row in visible:
        for c in cols:
            v = row.get(c, "")
            if isinstance(v, float):
                s = f"{v:.3f}"
            else:
                s = str(v)
            widths[c] = max(widths[c], min(len(s), 44))

    print("  ".join(c.ljust(widths[c]) for c in cols))
    for row in visible:
        out = []
        for c in cols:
            v = row.get(c, "")
            if isinstance(v, float):
                s = f"{v:.3f}"
            else:
                s = str(v)
            if len(s) > 44:
                s = s[:41] + "..."
            out.append(s.ljust(widths[c]))
        print("  ".join(out))
    if len(rows) > max_rows:
        print(f"... {len(rows) - max_rows} more rows")


def ms_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Optional[float]]:
    values = [row.get(field) for row in rows]
    return {
        "mean": mean_or_none(values),
        "median": median_or_none(values),
    }


def fmt_ms(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{float(value):.3f} ms"


def fmt_x(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{float(value):.2f}x"


def fmt_pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{float(value):.1f}%"


def write_profile_report(path: str | Path, rows: list[dict[str, Any]], summary: dict[str, Any], args: argparse.Namespace) -> None:
    """Write a compact stage-by-stage profile, close in spirit to Fares traces."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def stage(label: str, field: str) -> str:
        stats = ms_summary(rows, field)
        return f"{label:<34} mean={fmt_ms(stats['mean']):>14}  median={fmt_ms(stats['median']):>14}"

    lines: list[str] = []
    lines.append("PNP SYMBOLIC TEMPLATE PROFILING REPORT")
    lines.append("=" * 44)
    lines.append(f"tree_file: {args.tree}")
    lines.append(f"seeds: {args.seeds}")
    lines.append(f"passes: {args.passes}")
    lines.append(f"k_digits: {args.k_digits}")
    lines.append(f"template_modes: {args.template_modes}")
    lines.append(f"tree_root_method: {args.tree_root_method if args.emit_tree_roots else 'disabled'}")
    lines.append(f"root_refine_backend: {args.root_refine_backend}")
    if args.root_refine_lib:
        lines.append(f"root_refine_lib: {args.root_refine_lib}")
    lines.append(f"action_project_backend: {args.action_project_backend}")
    if args.action_project_lib:
        lines.append(f"action_project_lib: {args.action_project_lib}")
    lines.append(f"msolve: {'disabled' if args.no_msolve else (args.msolve_bin or 'auto')}")
    lines.append("")

    lines.append("INPUT AND CERTIFICATES")
    lines.append(f"rows                              {summary.get('rows')}")
    lines.append(f"same input match rate              {fmt_pct(summary.get('same_input_match_rate'))}")
    lines.append(f"tree accept rate                   {fmt_pct(summary.get('tree_accept_rate'))}")
    lines.append(f"tree replay residual pass rate     {fmt_pct(summary.get('tree_accuracy_pass_rate'))}")
    lines.append(f"tree root residual pass rate       {fmt_pct(summary.get('tree_root_residual_pass_rate'))}")
    lines.append(f"Fares/msolve successful runs       {summary.get('fares_successful_runs')}")
    lines.append(f"Fares root residual pass rate      {fmt_pct(summary.get('fares_root_residual_pass_rate'))}")
    lines.append(f"best-score match rate              {fmt_pct(summary.get('best_score_match_rate'))}")
    if args.emit_tree_roots:
        lines.append(f"action projection backends          {summary.get('action_project_backend_counts')}")
    lines.append("")

    lines.append("TREE ONLINE TIMING")
    lines.append(stage("coefficient fill", "tree_fill_ms"))
    lines.append(stage("template replay", "tree_replay_ms"))
    lines.append(stage("branch resume", "tree_branch_resume_ms"))
    lines.append(stage("algebraic replay total", "tree_total_ms"))
    if args.emit_tree_roots:
        lines.append(stage("root replay reuse", "tree_root_replay_ms"))
        lines.append(stage("quotient/action assemble", "tree_root_assemble_ms"))
        lines.append(stage("quotient projection solve", "tree_root_projection_ms"))
        lines.append(stage("action matrix build", "tree_root_action_ms"))
        lines.append(stage("eigen extraction", "tree_root_eig_ms"))
        lines.append(stage("Newton filter", "tree_root_newton_ms"))
        lines.append(stage("root extraction total", "tree_root_total_ms"))
        lines.append(stage("pose/root scoring", "tree_score_ms"))
        lines.append(stage("tree full solver total", "tree_full_solver_ms"))
    lines.append("")

    lines.append("FARES/MSOLVE BASELINE TIMING")
    lines.append(stage("msolve subprocess", "fares_subprocess_ms"))
    lines.append(stage("msolve output read", "fares_output_read_ms"))
    lines.append(stage("msolve output parse", "fares_parse_ms"))
    lines.append(stage("msolve total", "fares_msolve_ms"))
    lines.append(stage("Fares root scoring", "fares_score_ms"))
    lines.append(stage("Fares full solver total", "fares_full_solver_ms"))
    lines.append("")

    lines.append("SPEEDUPS")
    lines.append(f"replay vs Fares/msolve median      {fmt_x(summary.get('speedup_vs_fares_median'))}")
    lines.append(f"tree roots vs Fares/msolve median  {fmt_x(summary.get('speedup_with_roots_vs_fares_median'))}")
    lines.append(f"full solver vs Fares median        {fmt_x(summary.get('speedup_full_solver_vs_fares_median'))}")
    lines.append("")

    lines.append("INTERPRETATION")
    lines.append("tree replay total is the online algebraic template reuse time.")
    lines.append("tree full solver total includes replay, root extraction, Newton filtering, and scoring.")
    lines.append("Fares/msolve total calls msolve from scratch on the same .ms input; Fares full solver total adds the same local scoring layer.")
    lines.append("Use full solver total for a final solver-speed claim; use replay total only for the algebraic-core claim.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def try_tree(
    tree: PnPSymbolicTree,
    eqs,
    cond_max: float,
    rel_tol: float,
    allowed_modes: set[str],
    check_cond: bool,
    return_rewrite: bool = False,
) -> tuple[bool, Any, Optional[int], dict[str, Any], float]:
    replay_total = 0.0
    last_info: dict[str, Any] = {
        "reason": "no templates available",
        "condition": None,
        "relation_residual": None,
        "replay_sec": 0.0,
    }
    last_template = None

    for tmpl in list(tree.templates):
        if tmpl.solve_mode not in allowed_modes:
            continue
        last_template = tmpl
        ok, info = replay_template(
            tmpl,
            eqs,
            cond_max=cond_max,
            rel_tol=rel_tol,
            check_cond=check_cond,
            return_rewrite=return_rewrite,
        )
        replay_total += float(info.get("replay_sec", 0.0) or 0.0)
        last_info = info
        if ok:
            return True, tmpl, tmpl.template_id, info, replay_total

    last_info = dict(last_info)
    last_info["replay_sec"] = replay_total
    return False, last_template, None, last_info, replay_total


def run_online(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tree_path = Path(args.tree)
    tree = load_tree(tree_path)
    allowed_modes = mode_set(args.template_modes)

    msolve_bin = None
    if not args.no_msolve:
        if args.msolve_bin:
            msolve_bin = args.msolve_bin
            os.environ["MSOLVE_BIN"] = args.msolve_bin
        else:
            msolve_bin = setup_msolve_from_zip(args.msolve_zip, args.extract_dir, verbose=True)
        msolve_bin = msolve_bin or os.getenv("MSOLVE_BIN") or shutil.which("msolve")

    seeds = parse_seeds(args.seeds)
    rows: list[dict[str, Any]] = []

    print("Running online comparison")
    print("Tree file:", tree_path)
    print("Templates available:", len(tree.templates))
    print("Template modes used:", ",".join(sorted(allowed_modes)))
    print("Seeds:", seeds)
    print("Passes:", args.passes)
    print("msolve:", msolve_bin or "disabled/not found")
    print()

    for pass_id in range(1, args.passes + 1):
        for seed in seeds:
            A, B, C = generate_k_digit_matrices(args.k_digits, seed=seed)

            t0 = time.perf_counter()
            eqs = wedge_equations_3eq(A, B, C)
            fill_sec = time.perf_counter() - t0
            system_text = msolve_input_text(eqs)
            system_sha = sha256_text(system_text)

            root_needs_rewrite = bool(args.emit_tree_roots and args.tree_root_method in {"template_action", "monomial_order_action"})
            accepted, used_template, template_id, replay_info, replay_sec = try_tree(
                tree,
                eqs,
                args.cond_max,
                args.rel_tol,
                allowed_modes,
                not args.skip_cond_check,
                return_rewrite=root_needs_rewrite,
            )

            new_branch_created = False
            branch_resume_sec = 0.0
            branch_info: Optional[dict[str, Any]] = None

            if not accepted and args.branch_policy == "learn":
                branch_t0 = time.perf_counter()
                new_tmpl, branch_info = tree.add_template_from_instance(A, B, C)
                ok, new_info = replay_template(
                    new_tmpl,
                    eqs,
                    cond_max=args.cond_max,
                    rel_tol=args.rel_tol,
                    check_cond=not args.skip_cond_check,
                )

                if not ok and args.allow_lstsq_branches:
                    new_tmpl.solve_mode = "lstsq"
                    ok, new_info = replay_template(
                        new_tmpl,
                        eqs,
                        cond_max=args.lstsq_cond_max,
                        rel_tol=args.lstsq_rel_tol,
                        check_cond=not args.skip_cond_check,
                    )

                branch_resume_sec = time.perf_counter() - branch_t0
                new_branch_created = True
                replay_info = new_info
                template_id = new_tmpl.template_id
                used_template = new_tmpl
                accepted = ok

                if not ok and not args.keep_failed_branches:
                    tree.templates = [t for t in tree.templates if t.template_id != new_tmpl.template_id]
            elif not accepted:
                replay_info = dict(replay_info)
                if (
                    args.emit_tree_roots
                    and (
                        args.tree_root_method == "quotient_projection"
                        or (
                            args.tree_root_method == "monomial_order_action"
                            and replay_info.get("_rewrite_table") is not None
                        )
                    )
                    and used_template is not None
                ):
                    replay_info["reason"] = "action root path built; final roots must certify"
                    template_id = used_template.template_id
                else:
                    replay_info["reason"] = "all templates rejected; branch learning skipped"
                    replay_info["solve_mode"] = None

            tree_online_sec = fill_sec + replay_sec + branch_resume_sec
            tree_root_info: dict[str, Any] = {
                "ok": False,
                "reason": "disabled",
                "root_count": 0,
                "real_root_count": 0,
                "root_residuals": [],
                "real_roots": [],
                "max_relative_residual": None,
                "median_relative_residual": None,
                "total_sec": 0.0,
                "root_sample": [],
            }
            root_replay_ready = bool(
                args.emit_tree_roots
                and used_template is not None
                and (
                    accepted
                    or args.tree_root_method == "quotient_projection"
                    or (
                        args.tree_root_method == "monomial_order_action"
                        and replay_info.get("_rewrite_table") is not None
                    )
                )
            )
            if args.emit_tree_roots and root_replay_ready:
                if args.tree_root_method == "quotient_projection":
                    tree_root_info = tree_roots_from_quotient_projection(
                        used_template,
                        eqs,
                        action_weights=args.tree_root_action_weights,
                        residual_tol=args.tree_root_residual_tol,
                        dedup_tol=args.root_match_tol * 0.1,
                        real_imag_tol=args.tree_root_real_imag_tol,
                        max_newton_iter=args.tree_root_newton_iter,
                        max_abs_root=args.tree_root_max_abs,
                        root_refine_backend=args.root_refine_backend,
                        root_refine_lib=args.root_refine_lib or None,
                        action_project_backend=args.action_project_backend,
                        action_project_lib=args.action_project_lib or None,
                    )
                elif args.tree_root_method in {"template_action", "monomial_order_action"}:
                    tree_root_info = tree_roots_from_template_replay(
                        used_template,
                        eqs,
                        rewrite_table=replay_info.get("_rewrite_table"),
                        action_weights=args.tree_root_action_weights,
                        residual_tol=args.tree_root_residual_tol,
                        dedup_tol=args.root_match_tol * 0.1,
                        real_imag_tol=args.tree_root_real_imag_tol,
                        max_newton_iter=args.tree_root_newton_iter,
                        max_abs_root=args.tree_root_max_abs,
                        rel_tol=args.rel_tol,
                        root_refine_backend=args.root_refine_backend,
                        root_refine_lib=args.root_refine_lib or None,
                    )
                    if args.tree_root_method == "monomial_order_action":
                        tree_root_info["method"] = "monomial_order_action"
                else:
                    tree_root_info = tree_roots_from_macaulay(
                        eqs,
                        degree=args.tree_root_degree,
                        random_actions=args.tree_root_random_actions,
                        random_seed=seed,
                        residual_tol=args.tree_root_residual_tol,
                        dedup_tol=args.root_match_tol * 0.1,
                        real_imag_tol=args.tree_root_real_imag_tol,
                        max_newton_iter=args.tree_root_newton_iter,
                        max_abs_root=args.tree_root_max_abs,
                        root_refine_backend=args.root_refine_backend,
                        root_refine_lib=args.root_refine_lib or None,
                    )

            if args.no_msolve:
                fares = {"ran": False, "reason": "disabled"}
            else:
                fares = run_msolve_baseline(
                    eqs,
                    msolve_bin=msolve_bin,
                    timeout=args.msolve_timeout,
                    keep_input=args.keep_msolve_inputs,
                    output_dir=args.msolve_output_dir or None,
                    nvars=3,
                )

            fares_sec = fares.get("elapsed_sec")
            fares_ok = bool(fares.get("ran")) and fares.get("returncode") == 0
            fares_ms = None if fares_sec is None else 1000.0 * float(fares_sec)
            fares_subprocess_ms = None if fares.get("subprocess_sec") is None else 1000.0 * float(fares.get("subprocess_sec") or 0.0)
            fares_output_read_ms = None if fares.get("output_read_sec") is None else 1000.0 * float(fares.get("output_read_sec") or 0.0)
            fares_parse_ms = None if fares.get("parse_sec") is None else 1000.0 * float(fares.get("parse_sec") or 0.0)
            fares_root_residuals = root_relative_residuals(eqs, fares.get("root_centers", []))
            fares_root_summary = residual_summary(fares_root_residuals, args.root_residual_tol)
            tree_root_summary = residual_summary(tree_root_info.get("root_residuals", []), args.tree_root_residual_tol)

            tree_score_t0 = time.perf_counter()
            tree_best = best_root_by_score(A, B, C, tree_root_info.get("real_roots", []))
            tree_score_sec = time.perf_counter() - tree_score_t0
            fares_score_t0 = time.perf_counter()
            fares_best = best_root_by_score(A, B, C, fares.get("root_centers", []))
            fares_score_sec = time.perf_counter() - fares_score_t0
            tree_best_score = tree_best.get("best_score")
            fares_best_score = fares_best.get("best_score")
            if tree_best_score is not None and fares_best_score is not None:
                best_score_abs_gap = abs(float(tree_best_score) - float(fares_best_score))
                best_score_rel_gap = best_score_abs_gap / max(1.0, abs(float(fares_best_score)))
                best_score_match_pass = (
                    best_score_abs_gap <= args.best_score_atol
                    or best_score_rel_gap <= args.best_score_rtol
                )
            else:
                best_score_abs_gap = None
                best_score_rel_gap = None
                best_score_match_pass = False
            root_match = root_match_summary(
                tree_root_info.get("real_roots", []),
                fares.get("root_centers", []),
                args.root_match_tol,
            )
            shared_system_accuracy_pass = bool(
                accepted
                and replay_info.get("relation_residual") is not None
                and float(replay_info.get("relation_residual")) <= args.rel_tol
                and fares_ok
                and fares.get("parse_ok")
                and fares_root_summary["root_residual_pass"]
                and fares.get("input_sha256") == system_sha
            )
            tree_root_total_sec = float(tree_root_info.get("total_sec", 0.0) or 0.0)
            tree_full_solver_ms = (
                1000.0 * (tree_online_sec + tree_root_total_sec + tree_score_sec)
                if args.emit_tree_roots
                else None
            )
            fares_full_solver_ms = (fares_ms + 1000.0 * fares_score_sec) if fares_ms is not None else None

            rows.append(
                {
                    "pass": pass_id,
                    "seed": seed,
                    "system_sha256": system_sha,
                    "equation_terms": json.dumps([len(e) for e in eqs]),
                    "tree_output_kind": "normal_form_certificate",
                    "tree_accepted": bool(accepted),
                    "tree_action_replay_ready": bool(root_replay_ready),
                    "tree_accuracy_pass": bool(accepted and replay_info.get("relation_residual") is not None and float(replay_info.get("relation_residual")) <= args.rel_tol),
                    "tree_accuracy_metric": f"normal_form_relation_residual <= {args.rel_tol:g}",
                    "template_id": template_id,
                    "new_branch_created": bool(new_branch_created),
                    "tree_fill_ms": 1000.0 * fill_sec,
                    "tree_replay_ms": 1000.0 * replay_sec,
                    "tree_branch_resume_ms": 1000.0 * branch_resume_sec,
                    "tree_total_ms": 1000.0 * tree_online_sec,
                    "tree_root_extraction_enabled": bool(args.emit_tree_roots),
                    "tree_root_method": args.tree_root_method if args.emit_tree_roots else None,
                    "tree_root_replay_ms": 1000.0 * float(tree_root_info.get("replay_sec", 0.0) or 0.0),
                    "tree_root_assemble_ms": 1000.0 * float(tree_root_info.get("assemble_sec", 0.0) or 0.0),
                    "tree_root_projection_ms": 1000.0 * float(tree_root_info.get("projection_sec", 0.0) or 0.0),
                    "tree_root_action_ms": 1000.0 * float(tree_root_info.get("action_sec", 0.0) or 0.0),
                    "tree_root_eig_ms": 1000.0 * float(tree_root_info.get("eig_sec", 0.0) or 0.0),
                    "tree_root_newton_ms": 1000.0 * float(tree_root_info.get("newton_sec", 0.0) or 0.0),
                    "tree_root_total_ms": 1000.0 * tree_root_total_sec,
                    "tree_score_ms": 1000.0 * tree_score_sec,
                    "tree_with_roots_total_ms": 1000.0 * (tree_online_sec + tree_root_total_sec),
                    "tree_full_solver_ms": tree_full_solver_ms,
                    "tree_root_ok": bool(tree_root_info.get("ok")),
                    "tree_root_reason": tree_root_info.get("reason"),
                    "tree_root_seed_count": tree_root_info.get("seed_count"),
                    "tree_root_count": tree_root_info.get("root_count"),
                    "tree_real_root_count": tree_root_info.get("real_root_count"),
                    "tree_root_residual_count": tree_root_summary["root_residual_count"],
                    "tree_root_max_relative_residual": tree_root_summary["root_max_relative_residual"],
                    "tree_root_median_relative_residual": tree_root_summary["root_median_relative_residual"],
                    "tree_root_residual_pass": tree_root_summary["root_residual_pass"],
                    "tree_root_sample": json.dumps(tree_root_info.get("root_sample", [])),
                    "tree_root_seed_info": json.dumps(tree_root_info.get("seed_info", {}), default=str),
                    "tree_root_action_info": json.dumps(tree_root_info.get("action_info", {}), default=str),
                    "tree_action_project_backend": (tree_root_info.get("action_info") or {}).get("action_project_backend"),
                    "tree_action_project_library": (tree_root_info.get("action_info") or {}).get("action_project_library"),
                    "tree_action_project_lapack": (tree_root_info.get("action_info") or {}).get("action_project_lapack"),
                    "tree_action_project_solve_mode": (tree_root_info.get("action_info") or {}).get("projection_solve_mode"),
                    "tree_action_project_residual": (tree_root_info.get("action_info") or {}).get("projection_residual"),
                    "tree_root_refine_backend": tree_root_info.get("root_refine_backend"),
                    "tree_root_refine_library": tree_root_info.get("root_refine_library"),
                    "tree_root_refine_ok_count": tree_root_info.get("root_refine_ok_count"),
                    "tree_best_score": tree_best_score,
                    "tree_best_root": json.dumps(tree_best.get("best_root")),
                    "tree_best_scored_count": tree_best.get("scored_count"),
                    "tree_reason": replay_info.get("reason"),
                    "solve_mode": replay_info.get("solve_mode"),
                    "condition": replay_info.get("condition"),
                    "relation_residual": replay_info.get("relation_residual"),
                    "pivot_rank": replay_info.get("pivot_rank"),
                    "branch_rank": None if branch_info is None else branch_info.get("rank"),
                    "branch_basis_size": None if branch_info is None else branch_info.get("basis_size"),
                    "fares_output_kind": "root_boxes" if fares.get("ran") else None,
                    "fares_msolve_ran": fares.get("ran"),
                    "fares_msolve_ok": fares_ok,
                    "fares_accuracy_pass": bool(fares_ok and fares.get("parse_ok")),
                    "fares_accuracy_metric": "returncode == 0 and msolve root output parsed",
                    "fares_root_box_count": fares.get("root_box_count"),
                    "fares_parse_ok": fares.get("parse_ok"),
                    "fares_root_residual_count": fares_root_summary["root_residual_count"],
                    "fares_root_max_relative_residual": fares_root_summary["root_max_relative_residual"],
                    "fares_root_median_relative_residual": fares_root_summary["root_median_relative_residual"],
                    "fares_root_residual_pass": fares_root_summary["root_residual_pass"],
                    "fares_root_residual_tol": args.root_residual_tol,
                    "fares_best_score": fares_best_score,
                    "fares_best_root": json.dumps(fares_best.get("best_root")),
                    "fares_best_scored_count": fares_best.get("scored_count"),
                    "fares_score_ms": (1000.0 * fares_score_sec) if fares.get("parse_ok") else None,
                    "best_score_abs_gap": best_score_abs_gap,
                    "best_score_rel_gap": best_score_rel_gap,
                    "best_score_match_pass": best_score_match_pass,
                    "best_score_metric": f"abs_gap <= {args.best_score_atol:g} or rel_gap <= {args.best_score_rtol:g}",
                    "fares_msolve_ms": fares_ms if fares_ok else None,
                    "fares_raw_ms": fares_ms,
                    "fares_subprocess_ms": fares_subprocess_ms if fares_ok else None,
                    "fares_output_read_ms": fares_output_read_ms if fares_ok else None,
                    "fares_parse_ms": fares_parse_ms if fares_ok else None,
                    "fares_full_solver_ms": fares_full_solver_ms if fares_ok else None,
                    "fares_returncode": fares.get("returncode"),
                    "fares_reason": fares.get("reason"),
                    "fares_stdout_head": fares.get("stdout_head"),
                    "fares_stderr_head": fares.get("stderr_head"),
                    "fares_cmd": fares.get("cmd"),
                    "fares_input_path": fares.get("input_path"),
                    "fares_output_path": fares.get("output_path"),
                    "fares_input_sha256": fares.get("input_sha256"),
                    "same_input_for_tree_and_fares": bool(fares.get("ran") and fares.get("input_sha256") == system_sha),
                    "shared_system_accuracy_pass": shared_system_accuracy_pass,
                    "shared_system_accuracy_metric": "tree replay residual pass and parsed msolve roots satisfy the same equations",
                    "final_root_comparison_available": bool(args.emit_tree_roots and fares.get("parse_ok")),
                    "tree_to_msolve_match_count": root_match["tree_to_msolve_match_count"],
                    "tree_to_msolve_match_rate": root_match["tree_to_msolve_match_rate"],
                    "msolve_to_tree_match_count": root_match["msolve_to_tree_match_count"],
                    "msolve_to_tree_match_rate": root_match["msolve_to_tree_match_rate"],
                    "tree_to_msolve_max_distance": root_match["tree_to_msolve_max_distance"],
                    "msolve_to_tree_max_distance": root_match["msolve_to_tree_max_distance"],
                    "root_match_pass": root_match["root_match_pass"],
                    "root_match_tol": args.root_match_tol,
                    "fares_root_centers_sample": json.dumps(fares.get("root_centers_sample", [])),
                    "comparison_note": "same polynomial input; tree certificate is fast path; optional tree roots are experimental action/Newton roots",
                }
            )

        save_tree(tree, tree_path)

    tree_ms = [r["tree_total_ms"] for r in rows]
    tree_with_roots_ms = [r["tree_with_roots_total_ms"] for r in rows if r.get("tree_root_extraction_enabled")]
    accepted_reuse_ms = [
        r["tree_total_ms"]
        for r in rows
        if r["tree_accepted"] and not r["new_branch_created"]
    ]
    branch_ms = [r["tree_total_ms"] for r in rows if r["new_branch_created"]]
    fares_ok_ms = [r["fares_msolve_ms"] for r in rows if r.get("fares_msolve_ok") and r.get("fares_msolve_ms") is not None]
    fares_full_solver_ms = [r["fares_full_solver_ms"] for r in rows if r.get("fares_msolve_ok") and r.get("fares_full_solver_ms") is not None]
    fares_ran_rows = [r for r in rows if r.get("fares_msolve_ran")]
    both_ok_rows = [r for r in rows if r.get("tree_accuracy_pass") and r.get("fares_msolve_ok") and r.get("same_input_for_tree_and_fares")]
    shared_accuracy_rows = [r for r in rows if r.get("shared_system_accuracy_pass")]
    fares_root_pass_rows = [r for r in rows if r.get("fares_root_residual_pass")]
    tree_root_rows = [r for r in rows if r.get("tree_root_extraction_enabled")]
    tree_root_ok_rows = [r for r in rows if r.get("tree_root_ok")]
    tree_root_pass_rows = [r for r in rows if r.get("tree_root_residual_pass")]
    action_ready_rows = [r for r in rows if r.get("tree_action_replay_ready")]
    root_match_rows = [r for r in rows if r.get("root_match_pass")]
    tree_precision_pass_rows = [
        r for r in tree_root_rows if float(r.get("tree_to_msolve_match_rate") or 0.0) >= 99.999
    ]
    msolve_coverage_pass_rows = [
        r for r in tree_root_rows if float(r.get("msolve_to_tree_match_rate") or 0.0) >= 99.999
    ]
    best_score_rows = [r for r in rows if r.get("best_score_match_pass")]

    summary = {
        "tree_path": str(tree_path),
        "out_csv": args.out_csv,
        "out_json": args.out_json,
        "out_profile": args.out_profile,
        "seeds": seeds,
        "passes": args.passes,
        "k_digits": args.k_digits,
        "degree": tree.D,
        "branch_policy": args.branch_policy,
        "allow_lstsq_branches": bool(args.allow_lstsq_branches),
        "template_modes": sorted(allowed_modes),
        "skip_cond_check": bool(args.skip_cond_check),
        "templates_final": len(tree.templates),
        "rows": len(rows),
        "tree_accept_rate": pct(sum(1 for r in rows if r["tree_accepted"]), len(rows)),
        "tree_action_replay_ready_count": len(action_ready_rows),
        "tree_action_replay_ready_rate": pct(len(action_ready_rows), len(rows)),
        "tree_accuracy_pass_rate": pct(sum(1 for r in rows if r["tree_accuracy_pass"]), len(rows)),
        "new_branch_rate": pct(sum(1 for r in rows if r["new_branch_created"]), len(rows)),
        "fares_attempted_count": len(fares_ran_rows),
        "same_input_match_rate": pct(sum(1 for r in fares_ran_rows if r.get("same_input_for_tree_and_fares")), len(fares_ran_rows)),
        "tree_and_fares_both_success_count": len(both_ok_rows),
        "tree_and_fares_both_success_rate": pct(len(both_ok_rows), len(rows)),
        "fares_root_residual_pass_count": len(fares_root_pass_rows),
        "fares_root_residual_pass_rate": pct(len(fares_root_pass_rows), len(rows)),
        "shared_system_accuracy_pass_count": len(shared_accuracy_rows),
        "shared_system_accuracy_pass_rate": pct(len(shared_accuracy_rows), len(rows)),
        "root_residual_tol": args.root_residual_tol,
        "final_root_comparison_available": False,
        "accuracy_note": "Tree replay is checked by normal-form residual. When tree roots are enabled, template-action roots are Newton-filtered and compared with parsed msolve roots on the same equations.",
        "tree_mean_ms": mean_or_none(tree_ms),
        "tree_median_ms": median_or_none(tree_ms),
        "tree_with_roots_mean_ms": mean_or_none(tree_with_roots_ms),
        "tree_with_roots_median_ms": median_or_none(tree_with_roots_ms),
        "tree_full_solver_mean_ms": mean_or_none([r.get("tree_full_solver_ms") for r in rows]),
        "tree_full_solver_median_ms": median_or_none([r.get("tree_full_solver_ms") for r in rows]),
        "tree_fill_mean_ms": mean_or_none([r.get("tree_fill_ms") for r in rows]),
        "tree_fill_median_ms": median_or_none([r.get("tree_fill_ms") for r in rows]),
        "tree_replay_mean_ms": mean_or_none([r.get("tree_replay_ms") for r in rows]),
        "tree_replay_median_ms": median_or_none([r.get("tree_replay_ms") for r in rows]),
        "tree_root_replay_mean_ms": mean_or_none([r.get("tree_root_replay_ms") for r in rows]),
        "tree_root_replay_median_ms": median_or_none([r.get("tree_root_replay_ms") for r in rows]),
        "tree_root_assemble_mean_ms": mean_or_none([r.get("tree_root_assemble_ms") for r in rows]),
        "tree_root_assemble_median_ms": median_or_none([r.get("tree_root_assemble_ms") for r in rows]),
        "tree_root_projection_mean_ms": mean_or_none([r.get("tree_root_projection_ms") for r in rows]),
        "tree_root_projection_median_ms": median_or_none([r.get("tree_root_projection_ms") for r in rows]),
        "tree_root_action_mean_ms": mean_or_none([r.get("tree_root_action_ms") for r in rows]),
        "tree_root_action_median_ms": median_or_none([r.get("tree_root_action_ms") for r in rows]),
        "tree_root_eig_mean_ms": mean_or_none([r.get("tree_root_eig_ms") for r in rows]),
        "tree_root_eig_median_ms": median_or_none([r.get("tree_root_eig_ms") for r in rows]),
        "tree_root_newton_mean_ms": mean_or_none([r.get("tree_root_newton_ms") for r in rows]),
        "tree_root_newton_median_ms": median_or_none([r.get("tree_root_newton_ms") for r in rows]),
        "tree_score_mean_ms": mean_or_none([r.get("tree_score_ms") for r in rows]),
        "tree_score_median_ms": median_or_none([r.get("tree_score_ms") for r in rows]),
        "accepted_reuse_mean_ms": mean_or_none(accepted_reuse_ms),
        "accepted_reuse_median_ms": median_or_none(accepted_reuse_ms),
        "branch_creation_mean_ms": mean_or_none(branch_ms),
        "branch_creation_median_ms": median_or_none(branch_ms),
        "tree_root_extraction_enabled": bool(args.emit_tree_roots),
        "tree_root_method": args.tree_root_method if args.emit_tree_roots else None,
        "root_refine_backend_requested": args.root_refine_backend,
        "root_refine_lib": args.root_refine_lib,
        "action_project_backend_requested": args.action_project_backend,
        "action_project_lib": args.action_project_lib,
        "action_project_backend_counts": {
            str(k): sum(1 for r in rows if r.get("tree_action_project_backend") == k)
            for k in sorted({r.get("tree_action_project_backend") for r in rows if r.get("tree_action_project_backend") is not None})
        },
        "tree_root_attempted_count": len(tree_root_rows),
        "tree_root_ok_count": len(tree_root_ok_rows),
        "tree_root_ok_rate": pct(len(tree_root_ok_rows), len(tree_root_rows)),
        "tree_root_residual_pass_count": len(tree_root_pass_rows),
        "tree_root_residual_pass_rate": pct(len(tree_root_pass_rows), len(tree_root_rows)),
        "root_match_pass_count": len(root_match_rows),
        "root_match_pass_rate": pct(len(root_match_rows), len(tree_root_rows)),
        "tree_root_precision_pass_count": len(tree_precision_pass_rows),
        "tree_root_precision_pass_rate": pct(len(tree_precision_pass_rows), len(tree_root_rows)),
        "msolve_root_coverage_pass_count": len(msolve_coverage_pass_rows),
        "msolve_root_coverage_pass_rate": pct(len(msolve_coverage_pass_rows), len(tree_root_rows)),
        "best_score_match_count": len(best_score_rows),
        "best_score_match_rate": pct(len(best_score_rows), len(tree_root_rows)),
        "tree_to_msolve_match_rate_mean": mean_or_none([r.get("tree_to_msolve_match_rate") for r in tree_root_rows]),
        "msolve_to_tree_match_rate_mean": mean_or_none([r.get("msolve_to_tree_match_rate") for r in tree_root_rows]),
        "tree_real_root_count_mean": mean_or_none([r.get("tree_real_root_count") for r in tree_root_rows]),
        "msolve_root_count_mean": mean_or_none([r.get("fares_root_box_count") for r in rows if r.get("fares_root_box_count") is not None]),
        "fares_successful_runs": len(fares_ok_ms),
        "fares_mean_ms": mean_or_none(fares_ok_ms),
        "fares_median_ms": median_or_none(fares_ok_ms),
        "fares_subprocess_mean_ms": mean_or_none([r.get("fares_subprocess_ms") for r in rows]),
        "fares_subprocess_median_ms": median_or_none([r.get("fares_subprocess_ms") for r in rows]),
        "fares_output_read_mean_ms": mean_or_none([r.get("fares_output_read_ms") for r in rows]),
        "fares_output_read_median_ms": median_or_none([r.get("fares_output_read_ms") for r in rows]),
        "fares_parse_mean_ms": mean_or_none([r.get("fares_parse_ms") for r in rows]),
        "fares_parse_median_ms": median_or_none([r.get("fares_parse_ms") for r in rows]),
        "fares_score_mean_ms": mean_or_none([r.get("fares_score_ms") for r in rows]),
        "fares_score_median_ms": median_or_none([r.get("fares_score_ms") for r in rows]),
        "fares_full_solver_mean_ms": mean_or_none(fares_full_solver_ms),
        "fares_full_solver_median_ms": median_or_none(fares_full_solver_ms),
        "fares_reference_ms": args.fares_reference_ms,
        "fares_reference_label": args.fares_reference_label,
        "speedup_vs_fares_reference_mean": None,
        "speedup_vs_fares_reference_median": None,
        "faster_than_fares_reference_by_mean": None,
        "faster_than_fares_reference_by_median": None,
        "speedup_vs_fares_mean": None,
        "speedup_vs_fares_median": None,
        "mode_breakdown": {},
    }
    if args.fares_reference_ms is not None and summary["tree_mean_ms"]:
        summary["speedup_vs_fares_reference_mean"] = args.fares_reference_ms / summary["tree_mean_ms"]
        summary["faster_than_fares_reference_by_mean"] = summary["tree_mean_ms"] < args.fares_reference_ms
    if args.fares_reference_ms is not None and summary["tree_median_ms"]:
        summary["speedup_vs_fares_reference_median"] = args.fares_reference_ms / summary["tree_median_ms"]
        summary["faster_than_fares_reference_by_median"] = summary["tree_median_ms"] < args.fares_reference_ms
    for mode in ["square", "lstsq", None]:
        label = "rejected" if mode is None else mode
        mode_rows = [r for r in rows if (r.get("solve_mode") if r["tree_accepted"] else None) == mode]
        mode_ms = [r["tree_total_ms"] for r in mode_rows]
        summary["mode_breakdown"][label] = {
            "count": len(mode_rows),
            "rate": pct(len(mode_rows), len(rows)),
            "mean_ms": mean_or_none(mode_ms),
            "median_ms": median_or_none(mode_ms),
        }
    if summary["fares_mean_ms"] and summary["accepted_reuse_mean_ms"]:
        summary["speedup_vs_fares_mean"] = summary["fares_mean_ms"] / summary["accepted_reuse_mean_ms"]
    if summary["fares_median_ms"] and summary["accepted_reuse_median_ms"]:
        summary["speedup_vs_fares_median"] = summary["fares_median_ms"] / summary["accepted_reuse_median_ms"]
    if summary["fares_mean_ms"] and summary["tree_with_roots_mean_ms"]:
        summary["speedup_with_roots_vs_fares_mean"] = summary["fares_mean_ms"] / summary["tree_with_roots_mean_ms"]
    else:
        summary["speedup_with_roots_vs_fares_mean"] = None
    if summary["fares_median_ms"] and summary["tree_with_roots_median_ms"]:
        summary["speedup_with_roots_vs_fares_median"] = summary["fares_median_ms"] / summary["tree_with_roots_median_ms"]
    else:
        summary["speedup_with_roots_vs_fares_median"] = None
    if summary["fares_full_solver_mean_ms"] and summary["tree_full_solver_mean_ms"]:
        summary["speedup_full_solver_vs_fares_mean"] = summary["fares_full_solver_mean_ms"] / summary["tree_full_solver_mean_ms"]
    else:
        summary["speedup_full_solver_vs_fares_mean"] = None
    if summary["fares_full_solver_median_ms"] and summary["tree_full_solver_median_ms"]:
        summary["speedup_full_solver_vs_fares_median"] = summary["fares_full_solver_median_ms"] / summary["tree_full_solver_median_ms"]
    else:
        summary["speedup_full_solver_vs_fares_median"] = None

    return rows, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default="/content/pnp_symbolic_tree_demo/pnp_tree.json")
    ap.add_argument("--out-csv", default="/content/pnp_symbolic_tree_demo/online_results.csv")
    ap.add_argument("--out-json", default="/content/pnp_symbolic_tree_demo/online_summary.json")
    ap.add_argument("--out-profile", default="/content/pnp_symbolic_tree_demo/profile_report.txt")
    ap.add_argument("--seeds", default="42:52")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--k-digits", type=int, default=2)
    ap.add_argument("--cond-max", type=float, default=1e22)
    ap.add_argument("--rel-tol", type=float, default=1e-4)
    ap.add_argument("--root-residual-tol", type=float, default=1e-4, help="Relative residual tolerance for parsed msolve root centers.")
    ap.add_argument("--emit-tree-roots", action="store_true", help="Build action/Newton tree roots and compare them with msolve roots.")
    ap.add_argument("--tree-root-method", choices=["template_action", "monomial_order_action", "quotient_projection", "macaulay_svd"], default="template_action")
    ap.add_argument("--tree-root-action-weights", default="1,7,11;3,17,5", help="Semicolon-separated action forms, for example '1,7,11;3,17,5'. Use one form, such as '1,7,11', for the fastest quotient/action run.")
    ap.add_argument("--tree-root-degree", type=int, default=11)
    ap.add_argument("--tree-root-random-actions", type=int, default=12)
    ap.add_argument("--tree-root-residual-tol", type=float, default=1e-8)
    ap.add_argument("--tree-root-real-imag-tol", type=float, default=1e-7)
    ap.add_argument("--tree-root-newton-iter", type=int, default=40)
    ap.add_argument("--tree-root-max-abs", type=float, default=500.0)
    ap.add_argument("--root-refine-backend", choices=["auto", "c", "python"], default="auto", help="Use the compiled C Newton filter when available, or force Python/C.")
    ap.add_argument("--root-refine-lib", default="", help="Path to compiled pnp_root_refine shared library.")
    ap.add_argument("--action-project-backend", choices=["auto", "c", "python"], default="python", help="For quotient_projection roots, project quotient actions with NumPy/Python by default; use c/auto only for the experimental C projection path.")
    ap.add_argument("--action-project-lib", default="", help="Path to shared library exporting pnp_project_actions. Defaults to --root-refine-lib when empty.")
    ap.add_argument("--root-match-tol", type=float, default=1e-4)
    ap.add_argument("--best-score-atol", type=float, default=1e-6)
    ap.add_argument("--best-score-rtol", type=float, default=1e-6)
    ap.add_argument("--skip-cond-check", action="store_true", help="Skip online np.linalg.cond SVD check; still verifies residual after solve.")
    ap.add_argument("--msolve-zip", default="/content/task3_msolve.zip")
    ap.add_argument("--extract-dir", default="/content/task3_msolve_unzipped")
    ap.add_argument("--msolve-bin", default="", help="Use this msolve binary instead of the uploaded zip/default PATH.")
    ap.add_argument("--msolve-timeout", type=int, default=60)
    ap.add_argument("--keep-msolve-inputs", action="store_true", help="Keep temporary .ms files for debugging segfaulting msolve runs.")
    ap.add_argument("--msolve-output-dir", default="", help="Directory for kept Fares/msolve .ms input and output files.")
    ap.add_argument("--no-msolve", action="store_true")
    ap.add_argument("--fares-reference-ms", type=float, default=None)
    ap.add_argument("--fares-reference-label", default="external Fares/msolve profiling reference")
    ap.add_argument("--branch-policy", choices=["learn", "skip"], default="learn")
    ap.add_argument("--template-modes", default="all", help='all, square, lstsq, or comma-separated modes such as "square,lstsq"')
    ap.add_argument("--allow-lstsq-branches", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--lstsq-cond-max", type=float, default=1e24)
    ap.add_argument("--lstsq-rel-tol", type=float, default=2e-4)
    ap.add_argument("--keep-failed-branches", action="store_true")
    args = ap.parse_args()

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_profile).parent.mkdir(parents=True, exist_ok=True)

    rows, summary = run_online(args)
    write_csv(args.out_csv, rows)
    Path(args.out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_profile_report(args.out_profile, rows, summary, args)

    print_rows(rows)
    print()
    print("SUMMARY")
    print(f"Tree online mean:            {summary['tree_mean_ms']:.3f} ms")
    print(f"Tree online median:          {summary['tree_median_ms']:.3f} ms")
    if summary["accepted_reuse_median_ms"] is not None:
        print(f"Accepted-template median:    {summary['accepted_reuse_median_ms']:.3f} ms")
    if summary["tree_root_extraction_enabled"]:
        print(f"Tree+roots median:           {summary['tree_with_roots_median_ms']:.3f} ms")
        if summary.get("tree_full_solver_median_ms") is not None:
            print(f"Tree full solver median:     {summary['tree_full_solver_median_ms']:.3f} ms")
        print(f"Tree root ok:                {summary['tree_root_ok_count']}/{summary['tree_root_attempted_count']} ({summary['tree_root_ok_rate']:.1f}%)")
        print(f"Tree root residual pass:     {summary['tree_root_residual_pass_count']}/{summary['tree_root_attempted_count']} ({summary['tree_root_residual_pass_rate']:.1f}%)")
        print(f"Root match pass:             {summary['root_match_pass_count']}/{summary['tree_root_attempted_count']} ({summary['root_match_pass_rate']:.1f}%)")
        print(f"Tree root precision pass:    {summary['tree_root_precision_pass_count']}/{summary['tree_root_attempted_count']} ({summary['tree_root_precision_pass_rate']:.1f}%)")
        print(f"Msolve root coverage pass:   {summary['msolve_root_coverage_pass_count']}/{summary['tree_root_attempted_count']} ({summary['msolve_root_coverage_pass_rate']:.1f}%)")
        print(f"Best-score match:            {summary['best_score_match_count']}/{summary['tree_root_attempted_count']} ({summary['best_score_match_rate']:.1f}%)")
        print(f"Action projection backends:  {summary.get('action_project_backend_counts')}")
        precision_mean = "n/a" if summary["tree_to_msolve_match_rate_mean"] is None else f"{summary['tree_to_msolve_match_rate_mean']:.1f}%"
        coverage_mean = "n/a" if summary["msolve_to_tree_match_rate_mean"] is None else f"{summary['msolve_to_tree_match_rate_mean']:.1f}%"
        print(f"Mean tree-root precision:    {precision_mean}")
        print(f"Mean msolve-root coverage:   {coverage_mean}")
        tree_real_mean = "n/a" if summary["tree_real_root_count_mean"] is None else f"{summary['tree_real_root_count_mean']:.2f}"
        ms_real_mean = "n/a" if summary["msolve_root_count_mean"] is None else f"{summary['msolve_root_count_mean']:.2f}"
        print(f"Mean tree real roots:        {tree_real_mean}")
        print(f"Mean msolve real roots:      {ms_real_mean}")
    if summary["branch_creation_median_ms"] is not None:
        print(f"New-branch median:           {summary['branch_creation_median_ms']:.3f} ms")
    print(f"Tree accept rate:            {summary['tree_accept_rate']:.1f}%")
    print(f"Tree action-ready rate:      {summary['tree_action_replay_ready_rate']:.1f}%")
    print(f"Tree accuracy-pass rate:     {summary['tree_accuracy_pass_rate']:.1f}%")
    print(f"New branch rate:             {summary['new_branch_rate']:.1f}%")
    print(f"Same input match rate:       {summary['same_input_match_rate']:.1f}%")
    print(f"Tree+Fares both success:     {summary['tree_and_fares_both_success_count']}/{summary['rows']} ({summary['tree_and_fares_both_success_rate']:.1f}%)")
    print(f"Fares root residual pass:    {summary['fares_root_residual_pass_count']}/{summary['rows']} ({summary['fares_root_residual_pass_rate']:.1f}%)")
    print(f"Shared-system accuracy pass: {summary['shared_system_accuracy_pass_count']}/{summary['rows']} ({summary['shared_system_accuracy_pass_rate']:.1f}%)")
    print(f"Accuracy note:               {summary['accuracy_note']}")
    print(f"Tree templates now:          {summary['templates_final']}")
    print("Mode breakdown:")
    for mode, stats in summary["mode_breakdown"].items():
        mean_ms = "n/a" if stats["mean_ms"] is None else f"{stats['mean_ms']:.3f}"
        median_ms = "n/a" if stats["median_ms"] is None else f"{stats['median_ms']:.3f}"
        print(
            f"  {mode:8s} count={stats['count']:3d} "
            f"rate={stats['rate']:5.1f}% mean={mean_ms:>8s} ms median={median_ms:>8s} ms"
        )

    if summary["fares_successful_runs"]:
        print(f"Fares msolve mean:           {summary['fares_mean_ms']:.3f} ms")
        print(f"Fares msolve median:         {summary['fares_median_ms']:.3f} ms")
        if summary.get("fares_full_solver_median_ms") is not None:
            print(f"Fares full solver median:    {summary['fares_full_solver_median_ms']:.3f} ms")
        if summary["speedup_vs_fares_median"] is not None:
            print(f"Accepted-template speedup:   {summary['speedup_vs_fares_median']:.2f}x median")
        if summary.get("speedup_with_roots_vs_fares_median") is not None:
            print(f"Tree+roots speedup:          {summary['speedup_with_roots_vs_fares_median']:.2f}x median")
        if summary.get("speedup_full_solver_vs_fares_median") is not None:
            print(f"Full-solver speedup:         {summary['speedup_full_solver_vs_fares_median']:.2f}x median")
    else:
        print("Fares msolve baseline did not produce successful timing.")
        print("Use your previous profiling reference for now: Fares msolve core was about 22-25 ms online.")
    if summary["fares_reference_ms"] is not None:
        print()
        print("REFERENCE BASELINE")
        print(f"Fares reference:             {summary['fares_reference_ms']:.3f} ms")
        print(f"Reference label:             {summary['fares_reference_label']}")
        print(f"Speedup vs reference mean:   {summary['speedup_vs_fares_reference_mean']:.2f}x")
        print(f"Speedup vs reference median: {summary['speedup_vs_fares_reference_median']:.2f}x")
        print(f"Faster by mean:              {summary['faster_than_fares_reference_by_mean']}")
        print(f"Faster by median:            {summary['faster_than_fares_reference_by_median']}")

    print()
    print("FILES WRITTEN")
    print("Updated tree:", args.tree)
    print("Online CSV:", args.out_csv)
    print("Online summary:", args.out_json)
    print("Profiler report:", args.out_profile)
    print()
    print("TREE INTERPRETATION")
    print("- tree_accepted=True means online reused a symbolic branch and did not call msolve.")
    print("- new_branch_created=True means every existing branch rejected, so the tree resumed from the Macaulay checkpoint.")
    print("- Failed new branches are discarded by default; pass --keep-failed-branches only when debugging.")


if __name__ == "__main__":
    main()
