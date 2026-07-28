#!/usr/bin/env python3
"""
Reproduce the subset-representativeness table (Appendix, tab:more_results).

For every configuration (agent, model) evaluated on the FULL task set,
compares Assert% (A%) per constraint level L0–L3 on:
  * the full task set, and
  * the subset used for cost-constrained models: the four frameworks
    aiohttp, Express, FastAPI, Fastify (paper methodology: all tasks of
    these frameworks; with --pipeline-only, just the 16 pipeline tasks
    unconstrained → +clean_architecture → +postgres → +postgres+ORM)

and reports per-level Δ (full − subset), per-level MAE, and the
Pearson / Spearman correlations over all paired (full A%, subset A%)
observations (one pair per configuration per level).

A configuration counts as "full" when it covers at least
FULL_COVERAGE_THRESHOLD of the benchmark task set (the union of tasks
seen across all configurations) and includes every subset task.

Usage:
  uv run evaluation_subset_representativeness.py data/results
  uv run evaluation_subset_representativeness.py data/results --latex
"""

import argparse
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from evaluation import collect_data
from evaluation_tables import AGENT_DISPLAY, MODEL_DISPLAY, TABLE1_ORDER_FULL

# ── Subset definition ────────────────────────────────────────────────────────

# framework → ORM used at L3 (uv frameworks use SQLAlchemy, node use Sequelize)
SUBSET_FRAMEWORKS = {
    "aiohttp": "sqlalchemy",
    "fastapi": "sqlalchemy",
    "express": "sequelize",
    "fastify": "sequelize",
}


def build_pipeline_tasks() -> set[str]:
    """The 16 pipeline tasks: 4 frameworks × constraint pipeline L0–L3."""
    tasks = set()
    for fw, orm in SUBSET_FRAMEWORKS.items():
        tasks.add(f"{fw}-openapi-unconstrained")
        tasks.add(f"{fw}-openapi-clean_architecture")
        tasks.add(f"{fw}-openapi-clean_architecture-postgres")
        tasks.add(f"{fw}-openapi-clean_architecture-postgres-{orm}")
    return tasks


# A config is a "full" evaluation if it covers this fraction of the
# benchmark task set (union of tasks across all configs).
FULL_COVERAGE_THRESHOLD = 0.9

# Excluded from the validation set in the paper due to near-zero scores.
EXCLUDED_MODELS = {"devstral-small"}


# ── Correlation statistics (pure stdlib) ─────────────────────────────────────

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    MAXIT, EPS, FPMIN = 300, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_bt = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    bt = math.exp(ln_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _t_pvalue_two_sided(t: float, df: int) -> float:
    """Two-sided p-value for a t statistic with df degrees of freedom."""
    return _betai(df / 2.0, 0.5, df / (df + t * t))


def pearson_r(x: list[float], y: list[float]) -> tuple[float, float]:
    """Pearson correlation and two-sided p-value (t approximation, df = n−2)."""
    n = len(x)
    mx, my = statistics.mean(x), statistics.mean(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return float("nan"), float("nan")
    r = cov / (sx * sy)
    r = max(-1.0, min(1.0, r))
    if abs(r) == 1.0:
        return r, 0.0
    t = r * math.sqrt((n - 2) / (1.0 - r * r))
    return r, _t_pvalue_two_sided(t, n - 2)


def _ranks(values: list[float]) -> list[float]:
    """Fractional ranks (ties get the average rank)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_rho(x: list[float], y: list[float]) -> tuple[float, float]:
    """Spearman correlation (Pearson on ranks) and two-sided p-value."""
    return pearson_r(_ranks(x), _ranks(y))


def format_p(p: float) -> str:
    """Human-readable p-value, e.g. '3.2e-16' or '< 1e-300'."""
    if p <= 0.0:
        return "< 1e-300"
    return f"{p:.1e}"


def latex_p_bound(p: float) -> str:
    """LaTeX upper bound like '$p < 10^{-15}$'."""
    if p <= 0.0:
        return r"$p < 10^{-300}$"
    exponent = math.ceil(math.log10(p))
    return f"$p < 10^{{{exponent}}}$"


# ── Table computation ────────────────────────────────────────────────────────

def compute_level_a_pct(
    tasks: list[dict],
    frameworks: set[str] | None = None,
    task_filter: set[str] | None = None,
) -> dict[int, float]:
    """Mean strict A% per constraint level, optionally restricted to a set of
    frameworks and/or an explicit set of task names."""
    by_level: dict[int, list[float]] = defaultdict(list)
    for t in tasks:
        if frameworks is not None and t["framework"] not in frameworks:
            continue
        if task_filter is not None and t["task"] not in task_filter:
            continue
        by_level[t["n_constraints"]].append(t["strict_assertions_perc"])
    return {lvl: statistics.mean(vals) * 100 for lvl, vals in by_level.items()}


def find_full_configs(all_task_stats: dict, pipeline_tasks: set[str]) -> tuple[list[tuple[str, str]], dict, int]:
    """
    Identify configurations evaluated on the full benchmark.

    Returns (full_configs, coverage_per_config, benchmark_size), where
    benchmark_size is the number of distinct tasks in the union across configs.
    """
    all_tasks: set[str] = set()
    config_tasks: dict[tuple[str, str], set[str]] = {}
    for key, tasks in all_task_stats.items():
        names = {t["task"] for t in tasks}
        config_tasks[key] = names
        all_tasks.update(names)

    benchmark_size = len(all_tasks)
    full_configs = []
    coverage = {}
    for key, names in config_tasks.items():
        coverage[key] = len(names)
        if key[1] in EXCLUDED_MODELS:
            continue
        if not pipeline_tasks <= names:
            continue
        if len(names) / benchmark_size >= FULL_COVERAGE_THRESHOLD:
            full_configs.append(key)

    # Paper row order first, then any remaining configs alphabetically.
    ordered = [k for k in TABLE1_ORDER_FULL if k in full_configs]
    ordered += sorted(k for k in full_configs if k not in ordered)
    return ordered, coverage, benchmark_size


def compute_table(
    all_task_stats: dict,
    full_configs: list[tuple[str, str]],
    pipeline_tasks: set[str],
    pipeline_only: bool = False,
) -> list[dict]:
    """
    One row per full configuration with full-set A%, subset A%, and Δ per level.

    The subset restriction follows the paper: all tasks of the four subset
    frameworks. With pipeline_only=True it is instead the 16 pipeline tasks
    the cost-constrained models were actually evaluated on.
    """
    rows = []
    for key in full_configs:
        tasks = all_task_stats[key]
        full_a = compute_level_a_pct(tasks)
        if pipeline_only:
            sub_a = compute_level_a_pct(tasks, task_filter=pipeline_tasks)
        else:
            sub_a = compute_level_a_pct(tasks, frameworks=set(SUBSET_FRAMEWORKS))
        deltas = {
            lvl: full_a[lvl] - sub_a[lvl]
            for lvl in range(4)
            if lvl in full_a and lvl in sub_a
        }
        rows.append({
            "agent": key[0],
            "model": key[1],
            "full": full_a,
            "subset": sub_a,
            "delta": deltas,
        })
    return rows


def compute_correlations(rows: list[dict]) -> dict:
    """Pearson/Spearman over all paired (full A%, subset A%) observations."""
    full_vals, sub_vals = [], []
    for row in rows:
        for lvl in range(4):
            if lvl in row["full"] and lvl in row["subset"]:
                full_vals.append(row["full"][lvl])
                sub_vals.append(row["subset"][lvl])
    r, r_p = pearson_r(full_vals, sub_vals)
    rho, rho_p = spearman_rho(full_vals, sub_vals)
    return {
        "n_pairs": len(full_vals),
        "pearson_r": r,
        "pearson_p": r_p,
        "spearman_rho": rho,
        "spearman_p": rho_p,
    }


def compute_mae(rows: list[dict]) -> dict[int, float]:
    """Mean absolute Δ per constraint level."""
    mae = {}
    for lvl in range(4):
        vals = [abs(row["delta"][lvl]) for row in rows if lvl in row["delta"]]
        if vals:
            mae[lvl] = statistics.mean(vals)
    return mae


# ── Output ───────────────────────────────────────────────────────────────────

def print_table(rows: list[dict], mae: dict[int, float], stats: dict, latex: bool = False):
    print("\n" + "=" * 118)
    print("SUBSET REPRESENTATIVENESS: A% on the full task set vs. the subset (Δ = full − subset)")
    print("=" * 118)

    header = (
        f"{'Agent':<10} {'Model':<18}"
        f"  {'L0':>6} {'L1':>6} {'L2':>6} {'L3':>6}"
        f"  {'L0':>6} {'L1':>6} {'L2':>6} {'L3':>6}"
        f"  {'L0':>6} {'L1':>6} {'L2':>6} {'L3':>6}"
    )
    sub = f"{'':28}  {'─ Full-set A% ─':^27}  {'─ Subset A% ─':^27}  {'─ Δ (pp) ─':^27}"
    print(sub)
    print(header)
    print("-" * len(header))

    def _cells(metrics: dict, signed: bool = False) -> list[str]:
        cells = []
        for lvl in range(4):
            if lvl in metrics:
                fmt = f"{metrics[lvl]:>+6.1f}" if signed else f"{metrics[lvl]:>6.1f}"
                cells.append(fmt)
            else:
                cells.append(f"{'--':>6}")
        return cells

    for row in rows:
        a_name = AGENT_DISPLAY.get(row["agent"], row["agent"])
        m_name = MODEL_DISPLAY.get(row["model"], row["model"])
        parts = (
            [f"{a_name:<10} {m_name:<18}"]
            + _cells(row["full"]) + _cells(row["subset"]) + _cells(row["delta"], signed=True)
        )
        print("  ".join(parts[:1])
              + "  " + " ".join(parts[1:5])
              + "  " + " ".join(parts[5:9])
              + "  " + " ".join(parts[9:13]))

    print("-" * len(header))
    mae_cells = " ".join(
        f"{mae[lvl]:>6.1f}" if lvl in mae else f"{'--':>6}" for lvl in range(4)
    )
    print(f"{'MAE':<10} {'':<18}  {'':>6} {'':>6} {'':>6} {'':>6}  {'':>6} {'':>6} {'':>6} {'':>6}  {mae_cells}")

    print(f"\nPaired observations: N = {stats['n_pairs']} "
          f"({len(rows)} configurations × 4 constraint levels)")
    print(f"Pearson  r = {stats['pearson_r']:.3f}  (p = {format_p(stats['pearson_p'])})")
    print(f"Spearman ρ = {stats['spearman_rho']:.3f}  (p = {format_p(stats['spearman_p'])})")

    if latex:
        _print_table_latex(rows, mae, stats)


def _latex_delta(d: float) -> str:
    r = round(d, 1)
    if r == 0.0:
        return "0.0"
    sign = "$-$" if r < 0 else "$+$"
    return f"\\llap{{{sign}}}{abs(r):.1f}"


def _print_table_latex(rows: list[dict], mae: dict[int, float], stats: dict):
    print("\n% ── LaTeX for subset-representativeness table ──")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\footnotesize")
    print(r"\setlength{\tabcolsep}{3.5pt}")
    print(r"\begin{tabular}{@{}ll rrrr rrrr rrrr@{}}")
    print(r"\toprule")
    print(r"& & \multicolumn{4}{c}{Full-set A\%} & \multicolumn{4}{c}{Subset A\%} & \multicolumn{4}{c}{$\Delta$ (pp)} \\")
    print(r"\cmidrule(lr){3-6} \cmidrule(lr){7-10} \cmidrule(lr){11-14}")
    print(r"\textbf{Agent} & \textbf{Model} & \textbf{L0} & \textbf{L1} & \textbf{L2} & \textbf{L3} & \textbf{L0} & \textbf{L1} & \textbf{L2} & \textbf{L3} & \textbf{L0} & \textbf{L1} & \textbf{L2} & \textbf{L3} \\")
    print(r"\midrule")

    for row in rows:
        a_name = AGENT_DISPLAY.get(row["agent"], row["agent"])
        m_name = MODEL_DISPLAY.get(row["model"], row["model"])
        cols = []
        for metrics in (row["full"], row["subset"]):
            for lvl in range(4):
                cols.append(f"{metrics[lvl]:.1f}" if lvl in metrics else "--")
        for lvl in range(4):
            cols.append(_latex_delta(row["delta"][lvl]) if lvl in row["delta"] else "--")
        print(f"{a_name} & {m_name} & {' & '.join(cols)} \\\\")

    print(r"\midrule")
    mae_cols = " & ".join(f"{mae[lvl]:.1f}" if lvl in mae else "--" for lvl in range(4))
    print(f"\\multicolumn{{2}}{{@{{}}l}}{{MAE}} & & & & & & & & & {mae_cols} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    p_bound = latex_p_bound(max(stats["pearson_p"], stats["spearman_p"]))
    print(
        r"\caption{Subset representativeness: A\% on the full task set vs.\ the subset "
        f"for each of the {len(rows)} full-evaluation configurations. "
        r"$\Delta$ = full$-$subset. MAE = mean absolute difference. "
        f"Pearson $r = {stats['pearson_r']:.3f}$, "
        f"Spearman $\\rho = {stats['spearman_rho']:.3f}$ "
        f"($N = {stats['n_pairs']}$, {p_bound}).}}"
    )
    print(r"\label{tab:more_results}")
    print(r"\end{table}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the subset-representativeness table from API-Bench results.",
    )
    parser.add_argument("base_dir", help="Root results directory (e.g. data/results)")
    parser.add_argument("--latex", action="store_true", help="Also print LaTeX source")
    parser.add_argument(
        "--pipeline-only",
        action="store_true",
        help="Restrict the subset to the 16 pipeline tasks instead of all tasks "
             "of the four subset frameworks (paper methodology)",
    )
    args = parser.parse_args()

    base = Path(args.base_dir)
    if not base.is_dir():
        print(f"Error: '{base}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    from evaluation import compute_task_stats

    all_runs = collect_data(base, agent_filter=None, model_filter=None)
    if not all_runs:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    all_task_stats = {key: compute_task_stats(runs) for key, runs in all_runs.items()}

    pipeline_tasks = build_pipeline_tasks()
    full_configs, coverage, benchmark_size = find_full_configs(all_task_stats, pipeline_tasks)

    print(f"Benchmark task set: {benchmark_size} tasks "
          f"(full-evaluation threshold: ≥ {FULL_COVERAGE_THRESHOLD:.0%} coverage)")
    if args.pipeline_only:
        print(f"Subset: the {len(pipeline_tasks)} pipeline tasks "
              f"({', '.join(sorted(SUBSET_FRAMEWORKS))} × L0–L3)")
    else:
        print(f"Subset: all tasks of {', '.join(sorted(SUBSET_FRAMEWORKS))} "
              f"(paper methodology)")
    print("\nConfiguration coverage:")
    for key in sorted(coverage, key=lambda k: -coverage[k]):
        tag = "FULL" if key in full_configs else "subset-only/partial"
        print(f"  {AGENT_DISPLAY.get(key[0], key[0]):<10} {MODEL_DISPLAY.get(key[1], key[1]):<18} "
              f"{coverage[key]:>3}/{benchmark_size} tasks  [{tag}]")

    if not full_configs:
        print("No full-evaluation configurations found.", file=sys.stderr)
        sys.exit(1)

    rows = compute_table(all_task_stats, full_configs, pipeline_tasks,
                         pipeline_only=args.pipeline_only)
    mae = compute_mae(rows)
    stats = compute_correlations(rows)
    print_table(rows, mae, stats, latex=args.latex)


if __name__ == "__main__":
    main()
