#!/usr/bin/env python3
"""
Reproduce the verifier-impact table (Appendix, tab:verifier_comparison).

For each configuration (agent, model) and each constraint level L0–L3,
compares two variants of the assertion pass rate:
  * A%      – with verifier enforcement: a run that violates any applicable
              static verifier (architecture, database, ORM) has its assertion
              score set to zero before averaging (strict_assertions_perc);
  * Raw A%  – without verifier enforcement: raw behavioral-test scores
              regardless of structural compliance (passed_assertions_perc).

Δ = Raw A% − A% quantifies the impact of verifier false negatives.
The script also prints the summary statistics cited in the appendix text:
maximum Δ across all configurations and levels, average L0→L3 decay with
and without enforcement, and the verifier false-reject rate.

Usage:
  uv run evaluation_verifier_impact.py data/results
  uv run evaluation_verifier_impact.py data/results --latex
"""

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from evaluation import (
    collect_data,
    compute_task_stats,
    compute_verifier_confusion,
)
from evaluation_tables import (
    AGENT_DISPLAY,
    MODEL_DISPLAY,
    TABLE1_ORDER_FULL,
    TABLE1_ORDER_SUBSET,
)


# ── Computation ──────────────────────────────────────────────────────────────

def compute_verifier_impact(all_task_stats: dict) -> dict:
    """
    For each (agent, model) and each constraint level 0–3, compute mean
    strict A% (verifier-enforced), mean raw A% (no enforcement), and Δ.

    Returns: {(agent, model): {level: (strict_pct, raw_pct, delta)}}
    """
    result = {}
    for (agent, model), tasks in all_task_stats.items():
        by_level: dict[int, list[dict]] = defaultdict(list)
        for t in tasks:
            by_level[t["n_constraints"]].append(t)

        level_metrics = {}
        for lvl in range(4):
            if lvl in by_level:
                ts = by_level[lvl]
                strict = statistics.mean(t["strict_assertions_perc"] for t in ts) * 100
                raw = statistics.mean(t["passed_assertions_perc"] for t in ts) * 100
                level_metrics[lvl] = (strict, raw, raw - strict)
        result[(agent, model)] = level_metrics
    return result


def compute_summary(table: dict) -> dict:
    """
    Summary statistics cited in the appendix text:
      * max Δ across all configurations and levels (and where it occurs)
      * average L0→L3 decay with and without verifier enforcement
        (over configurations that have both L0 and L3)
    """
    max_delta = 0.0
    max_delta_where = None
    strict_decays = []
    raw_decays = []

    for (agent, model), metrics in table.items():
        for lvl, (_strict, _raw, delta) in metrics.items():
            if delta > max_delta:
                max_delta = delta
                max_delta_where = (agent, model, lvl)
        if 0 in metrics and 3 in metrics:
            strict_decays.append(metrics[0][0] - metrics[3][0])
            raw_decays.append(metrics[0][1] - metrics[3][1])

    return {
        "max_delta": max_delta,
        "max_delta_where": max_delta_where,
        "avg_strict_decay": statistics.mean(strict_decays) if strict_decays else None,
        "avg_raw_decay": statistics.mean(raw_decays) if raw_decays else None,
        "n_configs_decay": len(strict_decays),
    }


# ── Output ───────────────────────────────────────────────────────────────────

def _row_iter(table: dict):
    """Yield (agent, model, metrics, starred) in paper row order."""
    for agent, model in TABLE1_ORDER_FULL:
        if (agent, model) in table:
            yield agent, model, table[(agent, model)], False
    for agent, model in TABLE1_ORDER_SUBSET:
        if (agent, model) in table:
            yield agent, model, table[(agent, model)], True
    shown = set(TABLE1_ORDER_FULL) | set(TABLE1_ORDER_SUBSET)
    for agent, model in sorted(k for k in table if k not in shown):
        yield agent, model, table[(agent, model)], False


def print_table(table: dict, latex: bool = False):
    print("\n" + "=" * 120)
    print("VERIFIER IMPACT: A% with vs. without verifier enforcement (Δ = Raw − A%)")
    print("=" * 120)

    sub = f"{'':29}" + "".join(f"  {f'─ L{lvl} ─':^19}" for lvl in range(4))
    header = (
        f"{'Agent':<10} {'Model':<18}"
        + "  " + "  ".join(f"{'A%':>5} {'Raw':>5} {'Δ':>5}" for _ in range(4))
    )
    print(sub)
    print(header)
    print("-" * len(header))

    prev_starred = False
    for agent, model, metrics, starred in _row_iter(table):
        if starred and not prev_starred:
            print()  # midrule between full-set and subset rows
        prev_starred = starred
        a_name = AGENT_DISPLAY.get(agent, agent)
        m_name = MODEL_DISPLAY.get(model, model)
        if starred:
            m_name += "*"
        parts = [f"{a_name:<10} {m_name:<18}"]
        for lvl in range(4):
            if lvl in metrics:
                strict, raw, delta = metrics[lvl]
                parts.append(f"{strict:>5.1f} {raw:>5.1f} {delta:>5.1f}")
            else:
                parts.append(f"{'--':>5} {'--':>5} {'--':>5}")
        print("  ".join(parts))

    if latex:
        _print_table_latex(table)


def _print_table_latex(table: dict):
    print("\n% ── LaTeX for verifier-impact table ──")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\footnotesize")
    print(r"\begin{tabular}{@{}ll rr@{\;}c rr@{\;}c rr@{\;}c rr@{\;}c@{}}")
    print(r"\toprule")
    print(
        r"& & \multicolumn{3}{c}{\textbf{L0}} & \multicolumn{3}{c}{\textbf{L1}}"
        r" & \multicolumn{3}{c}{\textbf{L2}} & \multicolumn{3}{c}{\textbf{L3}} \\"
    )
    print(r"\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11} \cmidrule(lr){12-14}")
    print(
        r"\textbf{Agent} & \textbf{Model}"
        + r" & \textbf{A\%} & \textbf{Raw} & $\Delta$" * 4
        + r" \\"
    )
    print(r"\midrule")

    prev_starred = False
    for agent, model, metrics, starred in _row_iter(table):
        if starred and not prev_starred:
            print(r"\midrule")
        prev_starred = starred
        a_name = AGENT_DISPLAY.get(agent, agent)
        m_name = MODEL_DISPLAY.get(model, model)
        if starred:
            m_name += "*"
        cols = []
        for lvl in range(4):
            if lvl in metrics:
                strict, raw, delta = metrics[lvl]
                cols.extend([f"{strict:.1f}", f"{raw:.1f}", f"{delta:.1f}"])
            else:
                cols.extend(["--", "--", "--"])
        print(f"{a_name} & {m_name} & {' & '.join(cols)} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(
        r"\caption{A\% with and without verifier enforcement. ``A\%'' zeros out"
        r" assertion scores for runs that violate structural constraints;"
        r" ``Raw A\%'' uses behavioral-test scores only."
        r" $\Delta$ = Raw A\% $-$ A\%."
        r" Full-set models use all 80 tasks; * = 16-task subset.}"
    )
    print(r"\label{tab:verifier_comparison}")
    print(r"\end{table}")


def print_summary(summary: dict, confusion: dict):
    print("\n" + "=" * 70)
    print("SUMMARY (statistics cited in the appendix text)")
    print("=" * 70)

    if summary["max_delta_where"]:
        agent, model, lvl = summary["max_delta_where"]
        a_name = AGENT_DISPLAY.get(agent, agent)
        m_name = MODEL_DISPLAY.get(model, model)
        print(
            f"Max Δ (Raw − A%) across all configs/levels: "
            f"{summary['max_delta']:.1f} pp ({a_name} + {m_name} at L{lvl})"
        )
    if summary["avg_strict_decay"] is not None:
        print(
            f"Average L0→L3 decay ({summary['n_configs_decay']} configs): "
            f"{summary['avg_raw_decay']:.0f} pp without enforcement → "
            f"{summary['avg_strict_decay']:.0f} pp with enforcement"
        )

    n_constrained = (
        confusion["test_pass_verifier_pass"] + confusion["test_pass_verifier_fail"]
        + confusion["test_fail_verifier_pass"] + confusion["test_fail_verifier_fail"]
    )
    n_violating = confusion["test_pass_verifier_fail"] + confusion["test_fail_verifier_fail"]
    if n_constrained > 0:
        print(
            f"Constrained runs violating structural constraints: "
            f"{n_violating}/{n_constrained} = {n_violating / n_constrained:.1%}"
        )
    print(
        f"  of which test-passing (verifier false rejects): "
        f"{confusion['test_pass_verifier_fail']}/{confusion['constrained_test_pass']} "
        f"= {confusion['false_reject_rate']:.1%} of constrained test-passing runs"
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the verifier-impact table (Appendix, tab:verifier_comparison).",
    )
    parser.add_argument("base_dir", help="Root results directory (e.g. data/results)")
    parser.add_argument("--latex", action="store_true", help="Also print LaTeX source")
    args = parser.parse_args()

    base = Path(args.base_dir)
    if not base.is_dir():
        print(f"Error: '{base}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    all_runs = collect_data(base, agent_filter=None, model_filter=None)
    if not all_runs:
        print("No generation results found.", file=sys.stderr)
        sys.exit(1)

    all_task_stats = {key: compute_task_stats(runs) for key, runs in all_runs.items()}

    table = compute_verifier_impact(all_task_stats)
    print_table(table, latex=args.latex)

    summary = compute_summary(table)
    confusion = compute_verifier_confusion(all_runs)
    print_summary(summary, confusion)


if __name__ == "__main__":
    main()
