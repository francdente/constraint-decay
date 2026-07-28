#!/usr/bin/env python3
"""
Reproduce the token-consumption tables (Appendix, tab:tokens_global and
tab:tokens_per_pair).

  Table A – Global input/output token consumption by constraint level,
            aggregated across all agent–model configurations.
  Table B – Average input/output tokens per run by constraint level and
            agent–model pair (K = ×10³, rounded). * = subset evaluation.

Token extraction reuses tokens.py:
  OpenHands: tokens read directly from base_state.json accumulated usage.
  Mini-SWE:  tokens inferred by simulating cumulative context with tiktoken.

Usage:
  uv run evaluation_token_consumption.py data/results
  uv run evaluation_token_consumption.py data/results --latex
"""

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from evaluation import compute_n_constraints
from evaluation_tables import AGENT_DISPLAY, MODEL_DISPLAY
from tokens import collect_tokens

LEVELS = range(4)

# Row ordering for Table B: agents first (paper order), then models in
# MODEL_DISPLAY order; anything unknown goes after, alphabetically.
AGENT_ORDER = ["mini_swe_sdk", "openhands_sdk"]
MODEL_ORDER = list(MODEL_DISPLAY.keys())


def _pair_sort_key(pair: tuple[str, str]) -> tuple:
    agent, model = pair
    a_idx = AGENT_ORDER.index(agent) if agent in AGENT_ORDER else len(AGENT_ORDER)
    m_idx = MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)
    return (a_idx, agent, m_idx, model)


# ── Aggregation ──────────────────────────────────────────────────────────────

def compute_global_table(data: dict[tuple[str, str], list[dict]]) -> dict:
    """
    Aggregate token usage by constraint level across all configurations.

    Returns: {level: {"n_runs", "avg_in", "avg_out", "total_in", "total_out"}}
    """
    by_level: dict[int, list[dict]] = defaultdict(list)
    for runs in data.values():
        for r in runs:
            by_level[compute_n_constraints(r["task"])].append(r)

    result = {}
    for lvl in LEVELS:
        runs = by_level.get(lvl, [])
        if not runs:
            continue
        total_in = sum(r["total_input"] for r in runs)
        total_out = sum(r["total_output"] for r in runs)
        result[lvl] = {
            "n_runs": len(runs),
            "avg_in": total_in / len(runs),
            "avg_out": total_out / len(runs),
            "total_in": total_in,
            "total_out": total_out,
        }
    return result


def compute_per_pair_table(data: dict[tuple[str, str], list[dict]]) -> dict:
    """
    Per (agent, model): average input/output tokens per run by level,
    plus the number of distinct tasks (to flag subset evaluations).

    Returns: {(agent, model): {"levels": {lvl: (avg_in, avg_out)},
                               "n_tasks": int, "n_runs": int}}
    """
    result = {}
    for (agent, model), runs in data.items():
        by_level: dict[int, list[dict]] = defaultdict(list)
        for r in runs:
            by_level[compute_n_constraints(r["task"])].append(r)

        level_metrics = {}
        for lvl in LEVELS:
            if lvl in by_level:
                rs = by_level[lvl]
                level_metrics[lvl] = (
                    statistics.mean(r["total_input"] for r in rs),
                    statistics.mean(r["total_output"] for r in rs),
                )

        n_tasks = len({(r["runtime"], r["task"]) for r in runs})
        result[(agent, model)] = {
            "levels": level_metrics,
            "n_tasks": n_tasks,
            "n_runs": len(runs),
        }
    return result


def subset_pairs(per_pair: dict) -> set[tuple[str, str]]:
    """
    Flag configurations evaluated on a task subset: any pair covering at most
    half the tasks of the largest (full) evaluation.
    """
    max_tasks = max(v["n_tasks"] for v in per_pair.values())
    return {k for k, v in per_pair.items() if v["n_tasks"] <= max_tasks / 2}


# ── Formatting helpers ───────────────────────────────────────────────────────

def _latex_num(value: float | int) -> str:
    """1234567 → '1{,}234{,}567' (LaTeX thousands separators)."""
    return f"{round(value):,}".replace(",", "{,}")


def _latex_k(value: float) -> str:
    """Tokens → '1{,}676K' (thousands, rounded)."""
    return f"{_latex_num(value / 1000)}K"


def _text_k(value: float) -> str:
    return f"{round(value / 1000):,}K"


# ── Table A: global consumption by level ─────────────────────────────────────

def print_global_table(table: dict, latex: bool = False):
    print("\n" + "=" * 90)
    print("TABLE A: Global token consumption by constraint level (all configurations)")
    print("=" * 90)

    header = (
        f"{'Level':<6} {'Runs':>6} {'Avg Input':>14} {'Avg Output':>12} "
        f"{'Total Input':>16} {'Total Output':>14}"
    )
    print(header)
    print("-" * len(header))

    total_in = 0
    total_out = 0
    total_runs = 0
    for lvl in LEVELS:
        if lvl not in table:
            continue
        row = table[lvl]
        print(
            f"L{lvl:<5} {row['n_runs']:>6} {row['avg_in']:>14,.0f} "
            f"{row['avg_out']:>12,.0f} {row['total_in']:>16,} {row['total_out']:>14,}"
        )
        total_in += row["total_in"]
        total_out += row["total_out"]
        total_runs += row["n_runs"]

    print("-" * len(header))
    print(
        f"{'Total':<6} {total_runs:>6} {'--':>14} {'--':>12} "
        f"{total_in:>16,} {total_out:>14,}"
    )

    if latex:
        _print_global_table_latex(table, total_in, total_out)


def _print_global_table_latex(table: dict, total_in: int, total_out: int):
    print("\n% ── LaTeX for Table A (global token consumption) ──")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{@{}l r r r r@{}}")
    print(r"\toprule")
    print(r"\textbf{Level} & \textbf{Avg Input} & \textbf{Avg Output} & "
          r"\textbf{Total Input} & \textbf{Total Output} \\")
    print(r"\midrule")
    for lvl in LEVELS:
        if lvl not in table:
            continue
        row = table[lvl]
        print(
            f"L{lvl} & {_latex_num(row['avg_in'])} & {_latex_num(row['avg_out'])} & "
            f"{_latex_num(row['total_in'])} & {_latex_num(row['total_out'])} \\\\"
        )
    print(r"\midrule")
    print(f"Total & -- & -- & {_latex_num(total_in)} & {_latex_num(total_out)} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Global token consumption by constraint level, "
          r"aggregated across all agent--model configurations.}")
    print(r"\label{tab:tokens_global}")
    print(r"\end{table}")


# ── Table B: per agent–model pair ────────────────────────────────────────────

def _ordered_pairs(per_pair: dict) -> list[tuple[str, str]]:
    return sorted(per_pair.keys(), key=_pair_sort_key)


def print_per_pair_table(per_pair: dict, latex: bool = False):
    starred = subset_pairs(per_pair)

    print("\n" + "=" * 110)
    print("TABLE B: Average input/output tokens per run by level and agent–model pair"
          " (K = ×10³; * = subset)")
    print("=" * 110)

    header = (
        f"{'Agent':<10} {'Model':<20} {'Tasks':>5} {'Runs':>5}"
        f"  {'L0':>8} {'L1':>8} {'L2':>8} {'L3':>8}"
        f"  {'L0':>6} {'L1':>6} {'L2':>6} {'L3':>6}"
    )
    sub = (f"{'':43}  {'── Avg Input ──':^35}  {'── Avg Output ──':^27}")
    print(sub)
    print(header)
    print("-" * len(header))

    for agent, model in _ordered_pairs(per_pair):
        entry = per_pair[(agent, model)]
        a_name = AGENT_DISPLAY.get(agent, agent)
        m_name = MODEL_DISPLAY.get(model, model)
        if (agent, model) in starred:
            m_name += "*"
        parts = [f"{a_name:<10} {m_name:<20} {entry['n_tasks']:>5} {entry['n_runs']:>5}"]
        for metric_idx, width in ((0, 8), (1, 6)):
            for lvl in LEVELS:
                if lvl in entry["levels"]:
                    parts.append(f"{_text_k(entry['levels'][lvl][metric_idx]):>{width}}")
                else:
                    parts.append(f"{'--':>{width}}")
        print("  ".join(parts[:1]) + "  " + " ".join(parts[1:5]) + "  " + " ".join(parts[5:]))

    if latex:
        _print_per_pair_table_latex(per_pair, starred)


def _print_per_pair_table_latex(per_pair: dict, starred: set):
    print("\n% ── LaTeX for Table B (per agent–model pair) ──")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\footnotesize")
    print(r"\begin{tabular}{@{}ll rrrr rrrr@{}}")
    print(r"\toprule")
    print(r"& & \multicolumn{4}{c}{\textbf{Avg Input Tokens}} & "
          r"\multicolumn{4}{c}{\textbf{Avg Output Tokens}} \\")
    print(r"\cmidrule(lr){3-6} \cmidrule(lr){7-10}")
    print(r"\textbf{Agent} & \textbf{Model} & \textbf{L0} & \textbf{L1} & "
          r"\textbf{L2} & \textbf{L3} & \textbf{L0} & \textbf{L1} & "
          r"\textbf{L2} & \textbf{L3} \\")
    print(r"\midrule")

    prev_agent = None
    for agent, model in _ordered_pairs(per_pair):
        if prev_agent is not None and agent != prev_agent:
            print(r"\midrule")
        prev_agent = agent

        entry = per_pair[(agent, model)]
        a_name = AGENT_DISPLAY.get(agent, agent)
        m_name = MODEL_DISPLAY.get(model, model)
        if (agent, model) in starred:
            m_name += "*"
        cols = []
        for metric_idx in (0, 1):
            for lvl in LEVELS:
                if lvl in entry["levels"]:
                    cols.append(_latex_k(entry["levels"][lvl][metric_idx]))
                else:
                    cols.append("--")
        print(f"{a_name} & {m_name} & {' & '.join(cols)} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Average input and output tokens per run by constraint level "
          r"and agent--model pair (K = $\times 10^3$, rounded). * = subset.}")
    print(r"\label{tab:tokens_per_pair}")
    print(r"\end{table}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the token-consumption tables from API-Bench results.",
    )
    parser.add_argument("base_dir", help="Root results directory (e.g. data/results)")
    parser.add_argument("--latex", action="store_true", help="Also print LaTeX source")
    parser.add_argument("--agent", help="Filter to a specific agent (exact match)")
    parser.add_argument("--llm", help="Filter LLMs containing this substring")
    args = parser.parse_args()

    base = Path(args.base_dir)
    if not base.is_dir():
        print(f"Error: '{base}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    data = collect_tokens(base, agent_filter=args.agent, llm_filter=args.llm)
    if not data:
        print("No token data found.", file=sys.stderr)
        sys.exit(1)

    global_table = compute_global_table(data)
    print_global_table(global_table, latex=args.latex)

    per_pair = compute_per_pair_table(data)
    print_per_pair_table(per_pair, latex=args.latex)


if __name__ == "__main__":
    main()
