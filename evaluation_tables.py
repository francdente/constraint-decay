#!/usr/bin/env python3
"""
Reproduce the three paper tables from API-Bench results.

  Table 1  – Assert% (A%) and pass@1 across constraint levels L0–L3
  Table 2a – Marginal effect of each constraint on A% (matched-pair)
  Table 2b – Feature implementation pass@1

Usage:
  uv run evaluation_tables.py data/results
  uv run evaluation_tables.py data/results --latex
"""

import argparse
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from evaluation import (
    collect_data as collect_generation_data,
    compute_task_stats as compute_generation_task_stats,
    compute_n_constraints,
    compute_framework,
    pass_at_k,
)
from evaluation_feature import (
    collect_data as collect_feature_data,
    compute_task_stats as compute_feature_task_stats,
)

# ── Display-name mappings ────────────────────────────────────────────────────

AGENT_DISPLAY = {
    "mini_swe_sdk": "Mini-SWE",
    "openhands_sdk": "OpenHands",
}

MODEL_DISPLAY = {
    "gpt-5-mini": "GPT-5-mini",
    "gpt-5.2": "GPT-5.2",
    "gpt-5.4": "GPT-5.4",
    "minimax-m2.5": "MiniMax-M2.5",
    "qwen3-coder-next": "Qwen3-Coder",
    "qwen3-235b": "Qwen3-235B-A22B",
    "Kimi-K2.5": "Kimi-K2.5",
    "devstral-small": "Devstral-Small",
}

# Row ordering for Table 1: (agent, model) pairs in paper order.
# "Full" rows come first, then "subset" rows (starred in the paper).
TABLE1_ORDER_FULL = [
    ("mini_swe_sdk", "gpt-5-mini"),
    ("openhands_sdk", "gpt-5-mini"),
    ("mini_swe_sdk", "qwen3-coder-next"),
    ("openhands_sdk", "qwen3-coder-next"),
    ("mini_swe_sdk", "qwen3-235b"),
    ("openhands_sdk", "qwen3-235b"),
    ("mini_swe_sdk", "minimax-m2.5"),
    ("mini_swe_sdk", "gpt-5.4"),
]
TABLE1_ORDER_SUBSET = [
    ("openhands_sdk", "minimax-m2.5"),
    ("mini_swe_sdk", "Kimi-K2.5"),
    ("mini_swe_sdk", "gpt-5.2"),
]

# Models for Table 2b (feature implementation).
TABLE2B_MODELS = ["gpt-5-mini", "gpt-5.2", "minimax-m2.5", "qwen3-coder-next"]
TABLE2B_AGENTS = ["mini_swe_sdk", "openhands_sdk"]

# Constraints for Table 2a.
CONSTRAINT_DISPLAY = {
    "clean_architecture": "Clean architecture",
    "postgres": "PostgreSQL",
    "sqlite": "SQLite",
    "sqlalchemy": "SQLAlchemy",
    "sequelize": "Sequelize",
}
CONSTRAINT_ORDER = ["clean_architecture", "postgres", "sqlite", "sqlalchemy", "sequelize"]

# Table 3: framework leaderboard columns — (agent, model) pairs.
TABLE3_COLUMNS = [
    ("mini_swe_sdk", "gpt-5-mini"),
    ("openhands_sdk", "gpt-5-mini"),
    ("mini_swe_sdk", "qwen3-coder-next"),
    ("openhands_sdk", "qwen3-coder-next"),
    ("mini_swe_sdk", "qwen3-235b"),
    ("openhands_sdk", "qwen3-235b"),
    ("mini_swe_sdk", "minimax-m2.5"),
    ("mini_swe_sdk", "gpt-5.4"),
]
TABLE3_MODEL_GROUPS = [
    ("GPT-5-mini", [("mini_swe_sdk", "gpt-5-mini"), ("openhands_sdk", "gpt-5-mini")]),
    ("Qwen3-Coder", [("mini_swe_sdk", "qwen3-coder-next"), ("openhands_sdk", "qwen3-coder-next")]),
    ("Qwen3-235B", [("mini_swe_sdk", "qwen3-235b"), ("openhands_sdk", "qwen3-235b")]),
    ("MiniMax-M2.5", [("mini_swe_sdk", "minimax-m2.5")]),
    ("GPT-5.4", [("mini_swe_sdk", "gpt-5.4")]),
]

FRAMEWORK_DISPLAY = {
    "flask": "Flask",
    "koa": "Koa",
    "express": "Express",
    "aiohttp": "Aiohttp",
    "fastify": "Fastify",
    "django": "Django",
    "fastapi": "FastAPI",
    "hono": "Hono",
}


# ── Table 1: A% and pass@1 across constraint levels ─────────────────────────

def compute_table1(all_task_stats: dict) -> dict:
    """
    For each (agent, model) and each constraint level 0–3,
    compute mean strict_assertions_perc (A%) and mean pass_at_1.

    Returns: {(agent, model): {level: (a_pct, pass1)}}
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
                a_pct = statistics.mean(t["strict_assertions_perc"] for t in ts) * 100
                p1 = statistics.mean(t["pass_at_1"] for t in ts) * 100
                level_metrics[lvl] = (a_pct, p1)
        result[(agent, model)] = level_metrics
    return result


def compute_table1_raw(all_runs: dict) -> dict:
    """
    Same layout as compute_table1 but WITHOUT verifier enforcement.

    Uses passed_assertions_perc (raw) and pass@1 based on tests_passed only.
    Computed directly from run-level data.

    Returns: {(agent, model): {level: (a_pct, pass1)}}
    """
    result = {}
    for (agent, model), runs in all_runs.items():
        # Group runs by task, then aggregate to task level
        by_task: dict[str, list[dict]] = defaultdict(list)
        for r in runs:
            by_task[r["task"]].append(r)

        task_rows = []
        for task, task_runs in by_task.items():
            n_exec = len(task_runs)
            n_tests_passed = sum(r["tests_passed"] for r in task_runs)
            mean_raw = statistics.mean(r["passed_assertions_perc"] for r in task_runs)
            task_rows.append({
                "n_constraints": compute_n_constraints(task),
                "raw_a_pct": mean_raw,
                "raw_pass1": pass_at_k(1, n_exec, n_tests_passed),
            })

        by_level: dict[int, list[dict]] = defaultdict(list)
        for t in task_rows:
            by_level[t["n_constraints"]].append(t)

        level_metrics = {}
        for lvl in range(4):
            if lvl in by_level:
                ts = by_level[lvl]
                a_pct = statistics.mean(t["raw_a_pct"] for t in ts) * 100
                p1 = statistics.mean(t["raw_pass1"] for t in ts) * 100
                level_metrics[lvl] = (a_pct, p1)
        result[(agent, model)] = level_metrics
    return result


def _compute_delta(metrics: dict) -> str | None:
    """Compute L3 - L0 delta for A% if both levels are present."""
    if 0 in metrics and 3 in metrics:
        return metrics[3][0] - metrics[0][0]
    return None


def print_table1(table1: dict, latex: bool = False, title: str = "TABLE 1: Assert% (A%) and pass@1 across constraint levels", label: str = "tab:main_results"):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)

    header = (
        f"{'Agent':<10} {'Model':<18}"
        f"  {'L0':>5} {'L1':>5} {'L2':>5} {'L3':>5}"
        f"  {'L0':>5} {'L1':>5} {'L2':>5} {'L3':>5}"
        f"  {'ΔA%':>6}"
    )
    sub = f"{'':28}  {'─ A% ─':^23}  {'─ pass@1 ─':^23}  {'L0→L3':>6}"
    print(sub)
    print(header)
    print("-" * len(header))

    def _print_row(agent, model, metrics, starred=False):
        a_name = AGENT_DISPLAY.get(agent, agent)
        m_name = MODEL_DISPLAY.get(model, model)
        if starred:
            m_name += "*"
        parts = [f"{a_name:<10} {m_name:<18}"]
        for metric_idx in (0, 1):  # 0=A%, 1=pass@1
            for lvl in range(4):
                if lvl in metrics:
                    parts.append(f"{metrics[lvl][metric_idx]:>5.1f}")
                else:
                    parts.append(f"{'--':>5}")
        delta = _compute_delta(metrics)
        if delta is not None:
            parts.append(f"{delta:>+5.1f}")
        else:
            parts.append(f"{'--':>5}")
        print("  ".join(parts[:1]) + "  " + " ".join(parts[1:5]) + "  " + " ".join(parts[5:9]) + "  " + parts[9])

    for agent, model in TABLE1_ORDER_FULL:
        if (agent, model) in table1:
            _print_row(agent, model, table1[(agent, model)])

    print()  # midrule

    for agent, model in TABLE1_ORDER_SUBSET:
        if (agent, model) in table1:
            _print_row(agent, model, table1[(agent, model)], starred=True)

    # Any remaining (agent, model) not in the predefined order
    shown = set(TABLE1_ORDER_FULL) | set(TABLE1_ORDER_SUBSET)
    remaining = sorted(k for k in table1 if k not in shown)
    if remaining:
        print()
        for agent, model in remaining:
            _print_row(agent, model, table1[(agent, model)])

    if latex:
        _print_table1_latex(table1, label=label)


def _print_table1_latex(table1: dict, label: str = "tab:main_results"):
    print("\n% ── LaTeX for Table 1 ──")
    print(r"\begin{table}")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{@{}ll cccc cccc r@{}}")
    print(r"\toprule")
    print(r"& & \multicolumn{4}{c}{Assert\% (A\%)} & \multicolumn{4}{c}{\texttt{pass@1}} & \\")
    print(r"\cmidrule(lr){3-6} \cmidrule(lr){7-10}")
    print(r"\textbf{Agent} & \textbf{Model} & \textbf{L0} & \textbf{L1} & \textbf{L2} & \textbf{L3} & \textbf{L0} & \textbf{L1} & \textbf{L2} & \textbf{L3} & \textbf{$\Delta$A\%} \\")
    print(r"\midrule")

    def _latex_row(agent, model, metrics, starred=False):
        a_name = AGENT_DISPLAY.get(agent, agent)
        m_name = MODEL_DISPLAY.get(model, model)
        if starred:
            m_name += "*"
        cols = []
        for metric_idx in (0, 1):
            for lvl in range(4):
                if lvl in metrics:
                    cols.append(f"{metrics[lvl][metric_idx]:.1f}")
                else:
                    cols.append("--")
        delta = _compute_delta(metrics)
        if delta is not None:
            cols.append(f"{delta:+.1f}")
        else:
            cols.append("--")
        print(f"{a_name} & {m_name} & {' & '.join(cols)} \\\\")

    for agent, model in TABLE1_ORDER_FULL:
        if (agent, model) in table1:
            _latex_row(agent, model, table1[(agent, model)])

    print(r"\midrule")

    for agent, model in TABLE1_ORDER_SUBSET:
        if (agent, model) in table1:
            _latex_row(agent, model, table1[(agent, model)], starred=True)

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Assertion pass rate A\% and \texttt{pass@1} across constraint levels. $\Delta$A\% = L3 $-$ L0.}")
    print(f"\\label{{{label}}}")
    print(r"\end{table}")


# ── Table 2a: Marginal constraint effects ────────────────────────────────────

def _parse_task_constraints(task_name: str) -> tuple[str, set[str]]:
    """
    Parse task name into (prefix, constraint_set).

    E.g. 'aiohttp-openapi-clean_architecture-postgres' →
         ('aiohttp-openapi', {'clean_architecture', 'postgres'})
    """
    parts = task_name.split("-")
    prefix = "-".join(parts[:2])
    constraint_parts = parts[2:]
    if len(constraint_parts) == 1 and constraint_parts[0] == "unconstrained":
        return prefix, set()
    return prefix, set(constraint_parts)


def _build_task_name(prefix: str, constraints: set[str]) -> str:
    """Reconstruct task name from prefix and constraint set."""
    if not constraints:
        return f"{prefix}-unconstrained"
    # Maintain canonical order
    order = ["clean_architecture", "postgres", "sqlite", "sqlalchemy", "sequelize"]
    sorted_constraints = sorted(constraints, key=lambda c: order.index(c) if c in order else 999)
    return f"{prefix}-{'-'.join(sorted_constraints)}"


def compute_table2a(all_task_stats: dict) -> list[dict]:
    """
    Compute matched-pair differences for each constraint type.

    For each constraint C, finds task pairs that differ only by the
    presence of C, computes Δ = A%(with C) – A%(without C), and
    reports mean ± std (and SEM) across all (agent, model, framework) pairs.
    """
    # Build lookup: (agent, model, task_name) → strict_assertions_perc
    lookup: dict[tuple[str, str, str], float] = {}
    for (agent, model), tasks in all_task_stats.items():
        for t in tasks:
            lookup[(agent, model, t["task"])] = t["strict_assertions_perc"]

    results = []
    for constraint in CONSTRAINT_ORDER:
        deltas = []
        # Find all tasks that contain this constraint
        for (agent, model, task_name), a_pct in lookup.items():
            prefix, constraints = _parse_task_constraints(task_name)
            if constraint not in constraints:
                continue
            # Construct sibling without this constraint
            sibling_constraints = constraints - {constraint}
            sibling_name = _build_task_name(prefix, sibling_constraints)
            sibling_key = (agent, model, sibling_name)
            if sibling_key in lookup:
                delta = a_pct - lookup[sibling_key]
                deltas.append(delta)

        if deltas:
            std = statistics.stdev(deltas) * 100 if len(deltas) > 1 else 0.0
            sem = std / math.sqrt(len(deltas)) if len(deltas) > 1 else 0.0
            results.append({
                "constraint": constraint,
                "mean_delta": statistics.mean(deltas) * 100,
                "std_delta": std,
                "sem_delta": sem,
                "n_pairs": len(deltas),
            })
    return results


def print_table2a(table2a: list[dict], latex: bool = False):
    print("\n" + "=" * 70)
    print("TABLE 2a: Marginal effect of each constraint on A%")
    print("=" * 70)

    header = f"{'Constraint':<22} {'Avg Δ (pp)':>12} {'± std':>8} {'± SEM':>8} {'N':>5}"
    print(header)
    print("-" * len(header))
    for row in table2a:
        name = CONSTRAINT_DISPLAY.get(row["constraint"], row["constraint"])
        sign = "+" if row["mean_delta"] >= 0 else ""
        print(
            f"{name:<22} {sign}{row['mean_delta']:>10.1f}"
            f" {row['std_delta']:>8.1f}"
            f" {row['sem_delta']:>8.1f}"
            f" {row['n_pairs']:>5}"
        )

    if latex:
        _print_table2a_latex(table2a)


def _print_table2a_latex(table2a: list[dict]):
    print("\n% ── LaTeX for Table 2a ──")
    print(r"\begin{subtable}[t]{0.48\linewidth}")
    print(r"\centering")
    print(r"\begin{tabular}{@{}l r@{$\;\pm\;$}l@{}}")
    print(r"\toprule")
    print(r"\textbf{Constraint} & \multicolumn{2}{c}{\textbf{Avg $\Delta$ (pp)}} \\")
    print(r"\midrule")
    for row in table2a:
        name = CONSTRAINT_DISPLAY.get(row["constraint"], row["constraint"])
        if row["mean_delta"] < 0:
            val_str = f"$-${abs(row['mean_delta']):.1f}"
        else:
            val_str = f"$+${row['mean_delta']:.1f}"
        print(f"{name:<20} & {val_str} & {row['sem_delta']:.1f}  \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Marginal effect of each constraint on A\%, estimated via matched-pair differences (+- std).}")
    print(r"\label{tab:marginal_constraint}")
    print(r"\end{subtable}")


# ── Table 2b: Feature implementation pass@1 ──────────────────────────────────

def compute_table2b(all_feature_task_stats: dict) -> dict:
    """
    Compute overall pass@1 for feature implementation tasks.

    Returns: {(agent, model): pass1_pct}
    """
    result = {}
    for (agent, model), tasks in all_feature_task_stats.items():
        p1 = statistics.mean(t["pass_at_1"] for t in tasks) * 100
        result[(agent, model)] = p1
    return result


def print_table2b(table2b: dict, latex: bool = False):
    print("\n" + "=" * 60)
    print("TABLE 2b: Feature implementation pass@1 (%)")
    print("=" * 60)

    # Pivot: rows = models, columns = agents
    models_present = sorted(
        {m for (_, m) in table2b},
        key=lambda m: TABLE2B_MODELS.index(m) if m in TABLE2B_MODELS else 999,
    )
    agents_present = sorted(
        {a for (a, _) in table2b},
        key=lambda a: TABLE2B_AGENTS.index(a) if a in TABLE2B_AGENTS else 999,
    )

    agent_headers = [AGENT_DISPLAY.get(a, a) for a in agents_present]
    header = f"{'Model':<22}" + "".join(f" {h:>12}" for h in agent_headers)
    print(header)
    print("-" * len(header))

    for model in models_present:
        m_name = MODEL_DISPLAY.get(model, model)
        parts = [f"{m_name:<22}"]
        for agent in agents_present:
            if (agent, model) in table2b:
                parts.append(f"{table2b[(agent, model)]:>12.1f}")
            else:
                parts.append(f"{'--':>12}")
        print("".join(parts))

    if latex:
        _print_table2b_latex(table2b, models_present, agents_present)


def _print_table2b_latex(table2b: dict, models: list[str], agents: list[str]):
    n_agents = len(agents)
    print("\n% ── LaTeX for Table 2b ──")
    print(r"\begin{subtable}[t]{0.48\linewidth}")
    print(r"\centering")
    cols = "l " + "c" * n_agents
    print(f"\\begin{{tabular}}{{@{{}}{cols}@{{}}}}")
    print(r"\toprule")
    print(f"& \\multicolumn{{{n_agents}}}{{c}}{{\\textbf{{pass@1 (\\%)}}}} \\\\")
    print(f"\\cmidrule(lr){{2-{1 + n_agents}}}")
    agent_headers = " & ".join(f"\\textbf{{{AGENT_DISPLAY.get(a, a)}}}" for a in agents)
    print(f"\\textbf{{Model}} & {agent_headers} \\\\")
    print(r"\midrule")
    for model in models:
        m_name = MODEL_DISPLAY.get(model, model)
        cols_str = []
        for agent in agents:
            if (agent, model) in table2b:
                cols_str.append(f"{table2b[(agent, model)]:.1f}")
            else:
                cols_str.append("--")
        print(f"{m_name} & {' & '.join(cols_str)} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{\texttt{pass@1}~(\%) on feature implementation tasks.}")
    print(r"\label{tab:feature_results}")
    print(r"\end{subtable}")


# ── Table 3: Framework leaderboard ────────────────────────────────────────────

def compute_table3(all_task_stats: dict) -> dict:
    """
    For each (agent, model) and each framework, compute mean A% and pass@1
    aggregated across all constraint levels.

    Returns: {(agent, model): {framework: (a_pct, pass1)}}
    """
    result = {}
    for (agent, model), tasks in all_task_stats.items():
        by_fw: dict[str, list[dict]] = defaultdict(list)
        for t in tasks:
            by_fw[t["framework"]].append(t)

        fw_metrics = {}
        for fw, ts in by_fw.items():
            a_pct = statistics.mean(t["strict_assertions_perc"] for t in ts) * 100
            p1 = statistics.mean(t["pass_at_1"] for t in ts) * 100
            fw_metrics[fw] = (a_pct, p1)
        result[(agent, model)] = fw_metrics
    return result


def print_table3(table3: dict, latex: bool = False):
    print("\n" + "=" * 110)
    print("TABLE 3: Framework leaderboard — A% (pass@1 subscript)")
    print("=" * 110)

    # Collect all frameworks present in the TABLE3_COLUMNS configs
    all_fw = set()
    for key in TABLE3_COLUMNS:
        if key in table3:
            all_fw.update(table3[key].keys())

    # Compute average A% per framework (across TABLE3_COLUMNS) for sorting
    fw_avg: dict[str, float] = {}
    for fw in all_fw:
        vals = []
        for key in TABLE3_COLUMNS:
            if key in table3 and fw in table3[key]:
                vals.append(table3[key][fw][0])
        fw_avg[fw] = statistics.mean(vals) if vals else 0.0

    # Sort frameworks by descending average A%
    sorted_fw = sorted(all_fw, key=lambda f: fw_avg[f], reverse=True)

    # Print header
    col_labels = []
    for _, pairs in TABLE3_MODEL_GROUPS:
        for agent, _ in pairs:
            col_labels.append("M" if agent == "mini_swe_sdk" else "O")

    header_models = "".join(
        f"  {name:^15}" for name, _ in TABLE3_MODEL_GROUPS
    )
    header_cols = "".join(f" {l:>7}" for l in col_labels)
    print(f"{'Framework':<12}{header_models}   {'Avg':>5}")
    print(f"{'':12}{header_cols}  {'':>5}")
    print("-" * 75)

    for fw in sorted_fw:
        fw_name = FRAMEWORK_DISPLAY.get(fw, fw)
        parts = [f"{fw_name:<12}"]
        for key in TABLE3_COLUMNS:
            if key in table3 and fw in table3[key]:
                a, p = table3[key][fw]
                parts.append(f"{a:>5.1f}/{p:>2.0f}")
            else:
                parts.append(f"{'--':>7}")
        parts.append(f"{fw_avg[fw]:>5.1f}")
        print(" ".join(parts))

    if latex:
        _print_table3_latex(table3, sorted_fw, fw_avg)


def _print_table3_latex(table3: dict, sorted_fw: list[str], fw_avg: dict[str, float]):
    print("\n% ── LaTeX for Table 3 ──")
    print(r"\begin{table}")
    print(r"\centering")
    print(r"\footnotesize")
    # 6 data columns (M/O per model group) + avg = 7 content columns
    print(r"\begin{tabular}{@{}ll cc cc cc c@{}}")
    print(r"\toprule")

    # Model group headers
    cmidrule_parts = []
    col_idx = 3  # columns start at 3 (1=multirow, 2=framework name)
    header_parts = []
    for name, pairs in TABLE3_MODEL_GROUPS:
        n = len(pairs)
        header_parts.append(f"\\multicolumn{{{n}}}{{c}}{{\\textbf{{{name}}}}}")
        cmidrule_parts.append(f"\\cmidrule(lr){{{col_idx}-{col_idx + n - 1}}}")
        col_idx += n
    print(f"& & {' & '.join(header_parts)} & \\\\")
    print(" ".join(cmidrule_parts))

    # M/O sub-headers
    sub_parts = []
    for _, pairs in TABLE3_MODEL_GROUPS:
        for agent, _ in pairs:
            sub_parts.append("M" if agent == "mini_swe_sdk" else "O")
    print(f"& & {' & '.join(sub_parts)} & \\textbf{{Avg}} \\\\")
    print(r"\midrule")

    n_fw = len(sorted_fw)
    print(f"\\multirow{{{n_fw}}}{{*}}{{\\textit{{Web framework}}}}")
    for fw in sorted_fw:
        fw_name = FRAMEWORK_DISPLAY.get(fw, fw)
        cols = []
        for key in TABLE3_COLUMNS:
            if key in table3 and fw in table3[key]:
                a, p = table3[key][fw]
                cols.append(f"\\cv{{{a:.1f}}}{{{p:.0f}}}")
            else:
                cols.append("--")
        avg_str = f"{fw_avg[fw]:.1f}"
        print(f"& {fw_name:<10} & {' & '.join(cols)} & {avg_str} \\\\")

    print(r"\midrule")
    print(r"\end{tabular}")
    print(r"\caption{Assertion pass rate A\% by framework, aggregated across constraint levels. "
          r"Subscripts denote \texttt{pass@1}~(\%). M\,=\,Mini-SWE; O\,=\,OpenHands.}")
    print(r"\label{tab:framework_leaderboard}")
    print(r"\end{table}")


# ── Table 3-all: Framework leaderboard aggregated across ALL models ────────

def compute_table3_all_models(all_task_stats: dict) -> dict:
    """
    For each framework, compute mean A% and pass@1 averaged across ALL
    (agent, model) pairs.

    First computes per-(agent,model) framework means (same as compute_table3),
    then averages those across all configs so each config has equal weight.

    Returns: {framework: (a_pct, pass1, n_configs)}
    """
    # Step 1: per-(agent,model) framework stats
    per_config = compute_table3(all_task_stats)

    # Step 2: collect per-framework values across all configs
    fw_a_vals: dict[str, list[float]] = defaultdict(list)
    fw_p_vals: dict[str, list[float]] = defaultdict(list)
    for (_agent, _model), fw_metrics in per_config.items():
        for fw, (a_pct, p1) in fw_metrics.items():
            fw_a_vals[fw].append(a_pct)
            fw_p_vals[fw].append(p1)

    result = {}
    for fw in fw_a_vals:
        result[fw] = (
            statistics.mean(fw_a_vals[fw]),
            statistics.mean(fw_p_vals[fw]),
            len(fw_a_vals[fw]),
        )
    return result


def _build_model_groups(per_config_table3: dict) -> list[tuple[str, list[tuple[str, str]]]]:
    """
    Discover all (agent, model) pairs and group them by model,
    sorted by MODEL_DISPLAY order (known models first, then alphabetical).
    """
    model_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for agent, model in sorted(per_config_table3.keys()):
        model_pairs[model].append((agent, model))

    known_order = list(MODEL_DISPLAY.keys())

    def _sort_key(m: str) -> tuple[int, str]:
        return (known_order.index(m) if m in known_order else len(known_order), m)

    return [
        (MODEL_DISPLAY.get(m, m), model_pairs[m])
        for m in sorted(model_pairs, key=_sort_key)
    ]


def print_table3_all_models(table3_all: dict, per_config_table3: dict, latex: bool = False):
    print("\n" + "=" * 110)
    print("TABLE 3-ALL: Framework leaderboard — A% (pass@1) aggregated across ALL models")
    print("=" * 110)

    sorted_fw = sorted(table3_all, key=lambda f: table3_all[f][0], reverse=True)
    model_groups = _build_model_groups(per_config_table3)

    # Flatten to ordered list of (agent, model) keys
    all_configs = [pair for _, pairs in model_groups for pair in pairs]

    # Build column sub-labels (M/O)
    col_labels = []
    for _, pairs in model_groups:
        for agent, _ in pairs:
            col_labels.append("M" if agent == "mini_swe_sdk" else "O")

    # Header: model group names centered over their M/O columns
    model_header = f"{'':12}"
    sub_header = f"{'':12}"
    for name, pairs in model_groups:
        width = 8 * len(pairs)
        model_header += f"{name:^{width}}"
        for agent, _ in pairs:
            label = "M" if agent == "mini_swe_sdk" else "O"
            sub_header += f" {label:>7}"
    model_header += f"  {'Avg':>5}"
    sub_header += f"  {'':>5}"

    print(model_header)
    print(sub_header)
    print("-" * len(model_header))

    for fw in sorted_fw:
        fw_name = FRAMEWORK_DISPLAY.get(fw, fw)
        parts = [f"{fw_name:<12}"]
        for agent, model in all_configs:
            if fw in per_config_table3.get((agent, model), {}):
                a, _p = per_config_table3[(agent, model)][fw]
                parts.append(f"{a:>7.1f}")
            else:
                parts.append(f"{'--':>7}")
        parts.append(f"{table3_all[fw][0]:>5.1f}")
        print(" ".join(parts))

    if latex:
        _print_table3_all_models_latex(table3_all, per_config_table3, sorted_fw)


def _print_table3_all_models_latex(table3_all: dict, per_config_table3: dict, sorted_fw: list[str]):
    model_groups = _build_model_groups(per_config_table3)
    all_configs = [pair for _, pairs in model_groups for pair in pairs]

    n_data_cols = len(all_configs)
    print("\n% ── LaTeX for Table 3-ALL (all models) ──")
    print(r"\begin{table}")
    print(r"\centering")
    print(r"\footnotesize")
    col_spec = "l " + "c " * n_data_cols + "c"
    print(f"\\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}")
    print(r"\toprule")

    # Model group headers
    header_parts = []
    cmidrule_parts = []
    col_idx = 2
    for name, pairs in model_groups:
        n = len(pairs)
        header_parts.append(f"\\multicolumn{{{n}}}{{c}}{{\\textbf{{{name}}}}}")
        cmidrule_parts.append(f"\\cmidrule(lr){{{col_idx}-{col_idx + n - 1}}}")
        col_idx += n
    print(f"& {' & '.join(header_parts)} & \\\\")
    print(" ".join(cmidrule_parts))

    # Agent sub-headers
    sub_parts = []
    for agent, _ in all_configs:
        sub_parts.append("M" if agent == "mini_swe_sdk" else "O")
    print(f"& {' & '.join(sub_parts)} & \\textbf{{Avg}} \\\\")
    print(r"\midrule")

    for fw in sorted_fw:
        fw_name = FRAMEWORK_DISPLAY.get(fw, fw)
        cols = []
        for key in all_configs:
            if key in per_config_table3 and fw in per_config_table3[key]:
                a, p = per_config_table3[key][fw]
                cols.append(f"\\cv{{{a:.1f}}}{{{p:.0f}}}")
            else:
                cols.append("--")
        avg_str = f"{table3_all[fw][0]:.1f}"
        print(f"{fw_name:<10} & {' & '.join(cols)} & {avg_str} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Assertion pass rate A\% by framework, aggregated across all constraint levels and all models. "
          r"Subscripts denote \texttt{pass@1}~(\%). M\,=\,Mini-SWE; O\,=\,OpenHands.}")
    print(r"\label{tab:framework_leaderboard_all}")
    print(r"\end{table}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the three paper tables from API-Bench results.",
    )
    parser.add_argument("base_dir", help="Root results directory (e.g. data/results)")
    parser.add_argument("--latex", action="store_true", help="Also print LaTeX source")
    args = parser.parse_args()

    base = Path(args.base_dir)
    if not base.is_dir():
        print(f"Error: '{base}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    # ── Collect generation task data ─────────────────────────────────
    gen_runs = collect_generation_data(base, agent_filter=None, model_filter=None)
    if not gen_runs:
        print("No generation results found.", file=sys.stderr)
        sys.exit(1)

    all_gen_task_stats = {}
    for key, runs in gen_runs.items():
        all_gen_task_stats[key] = compute_generation_task_stats(runs)

    # ── Table 1 ──────────────────────────────────────────────────────
    table1 = compute_table1(all_gen_task_stats)
    print_table1(table1, latex=args.latex)

    # ── Table 1 (raw — no verifier enforcement) ─────────────────────
    table1_raw = compute_table1_raw(gen_runs)
    print_table1(
        table1_raw,
        latex=args.latex,
        title="TABLE 1-RAW: Assert% and pass@1 WITHOUT verifier enforcement",
        label="tab:main_results_raw",
    )

    # ── Table 2a ─────────────────────────────────────────────────────
    table2a = compute_table2a(all_gen_task_stats)
    print_table2a(table2a, latex=args.latex)

    # ── Table 3 ──────────────────────────────────────────────────────
    table3 = compute_table3(all_gen_task_stats)
    print_table3(table3, latex=args.latex)

    # ── Table 3-ALL (all models) ─────────────────────────────────────
    table3_all = compute_table3_all_models(all_gen_task_stats)
    print_table3_all_models(table3_all, table3, latex=args.latex)

    # ── Collect feature task data ────────────────────────────────────
    feat_runs = collect_feature_data(base, agent_filter=None, model_filter=None)
    if feat_runs:
        all_feat_task_stats = {}
        for key, runs in feat_runs.items():
            all_feat_task_stats[key] = compute_feature_task_stats(runs)

        # ── Table 2b ─────────────────────────────────────────────────
        table2b = compute_table2b(all_feat_task_stats)
        print_table2b(table2b, latex=args.latex)
    else:
        print("\nNo feature implementation results found; skipping Table 2b.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
