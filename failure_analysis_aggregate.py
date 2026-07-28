#!/usr/bin/env python3
"""
Aggregate failure analysis and logic-error subcategories across all models.

Reads all failure_analysis_*.csv and logic_subcategories*.csv files,
produces cross-model summary tables for the paper rebuttal.

Usage:
  uv run failure_analysis_aggregate.py
  uv run failure_analysis_aggregate.py --latex
  uv run failure_analysis_aggregate.py --data-dir data
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

# ── Display names (paper order) ─────────────────────────────────────────────

MODEL_DISPLAY = {
    "qwen3-coder-next": "Qwen3-Coder",
    "gpt-5-mini": "GPT-5-mini",
    "minimax-m2.5": "MiniMax-M2.5",
    "gpt-5.2": "GPT-5.2",
    "Kimi-K2.5": "Kimi-K2.5",
    "gpt-5.4": "GPT-5.4"
}

# Paper ordering for rows
MODEL_ORDER = [
    "qwen3-coder-next",
    "gpt-5-mini",
    "minimax-m2.5",
    "Kimi-K2.5",
    "gpt-5.2",
    "gpt-5.4"
]

SUBSET_MODELS = {"Kimi-K2.5", "gpt-5.2"}

# Canonical category ordering (coarse)
COARSE_CATEGORIES = [
    "logic_error",
    "server_startup_failure",
    "incomplete_implementation",
    "schema_format_error",
    "stuck_in_loop",
    "constraint_violation",
    "premature_termination",
    "other",
]

COARSE_DISPLAY = {
    "logic_error": "Logic error",
    "server_startup_failure": "Server startup failure",
    "incomplete_implementation": "Incomplete implementation",
    "schema_format_error": "Schema / format error",
    "stuck_in_loop": "Stuck in loop",
    "constraint_violation": "Constraint violation",
    "premature_termination": "Premature termination",
    "other": "Other",
}

# Canonical subcategory ordering (logic errors)
LOGIC_SUBCATEGORIES = [
    "incorrect_query_logic",
    "database_runtime_error",
    "auth_misconfiguration",
    "framework_idiosyncrasy",
    "business_logic_defect",
    "state_propagation_failure",
]

LOGIC_DISPLAY = {
    "incorrect_query_logic": "Incorrect query logic",
    "database_runtime_error": "DB / ORM runtime error",
    "auth_misconfiguration": "Auth misconfiguration",
    "framework_idiosyncrasy": "Framework idiosyncrasy",
    "business_logic_defect": "Business logic defect",
    "state_propagation_failure": "State propagation failure",
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_failure_csvs(data_dir: Path) -> dict[str, list[dict]]:
    """Load all failure_analysis_*.csv files. Returns {model: [rows]}."""
    by_model: dict[str, list[dict]] = defaultdict(list)
    for csv_path in sorted(data_dir.glob("failure_analysis_*.csv")):
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                by_model[row["model"]].append(row)
    return dict(by_model)


def load_logic_csvs(data_dir: Path) -> dict[str, list[dict]]:
    """Load all logic_subcategories*.csv files. Returns {model: [rows]}.

    A per-model file (logic_subcategories_<model>.csv) supersedes that model's
    rows in the generic logic_subcategories.csv, which still holds an earlier
    subset-scoped analysis for some models. Without this the two overlap and
    the model's logic errors are counted twice.
    """
    per_model: dict[str, list[dict]] = defaultdict(list)
    generic: list[dict] = []
    for csv_path in sorted(data_dir.glob("logic_subcategories*.csv")):
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if csv_path.name == "logic_subcategories.csv":
            generic = rows
        else:
            for row in rows:
                per_model[row["model"]].append(row)

    by_model: dict[str, list[dict]] = defaultdict(list, per_model)
    for row in generic:
        if row["model"] not in per_model:
            by_model[row["model"]].append(row)
    return dict(by_model)


# ── Analysis ─────────────────────────────────────────────────────────────────

def compute_coarse_table(failure_data: dict[str, list[dict]]) -> dict:
    """Compute coarse category % per model and cross-model summary."""
    table = {}  # model -> {category: (count, pct)}
    totals = {}
    for model in MODEL_ORDER:
        if model not in failure_data:
            continue
        rows = failure_data[model]
        total = len(rows)
        totals[model] = total
        counts = defaultdict(int)
        for r in rows:
            counts[r["category"]] += 1
        table[model] = {
            cat: (counts.get(cat, 0), counts.get(cat, 0) / total * 100 if total else 0)
            for cat in COARSE_CATEGORIES
        }
    return table, totals


def compute_logic_table(logic_data: dict[str, list[dict]]) -> dict:
    """Compute logic subcategory % per model and cross-model summary."""
    table = {}
    totals = {}
    for model in MODEL_ORDER:
        if model not in logic_data:
            continue
        rows = logic_data[model]
        total = len(rows)
        totals[model] = total
        counts = defaultdict(int)
        for r in rows:
            counts[r["subcategory"]] += 1
        table[model] = {
            cat: (counts.get(cat, 0), counts.get(cat, 0) / total * 100 if total else 0)
            for cat in LOGIC_SUBCATEGORIES
        }
    return table, totals


def compute_datalayer_aggregate(logic_data: dict[str, list[dict]]) -> dict:
    """Compute % of logic errors that are data-layer defects per model.
    Data-layer = incorrect_query_logic + database_runtime_error."""
    result = {}
    for model in MODEL_ORDER:
        if model not in logic_data:
            continue
        rows = logic_data[model]
        total = len(rows)
        if total == 0:
            continue
        dl = sum(1 for r in rows if r["subcategory"] in
                 {"incorrect_query_logic", "database_runtime_error"})
        result[model] = (dl, total, dl / total * 100)
    return result


# ── Terminal printing ────────────────────────────────────────────────────────

def print_coarse_table(table, totals):
    models = [m for m in MODEL_ORDER if m in table]
    print("\n" + "=" * 100)
    print("COARSE FAILURE CATEGORIES (% of failed runs per model, Mini-SWE-Agent)")
    print("=" * 100)

    # Header
    hdr = f"{'Category':<28}"
    for m in models:
        star = "*" if m in SUBSET_MODELS else ""
        hdr += f"  {MODEL_DISPLAY[m]+star:>14}"
    hdr += f"  {'Avg':>7}"
    print(hdr)
    print("-" * len(hdr))

    for cat in COARSE_CATEGORIES:
        pcts = [table[m][cat][1] for m in models if cat in table[m]]
        if all(p == 0 for p in pcts):
            continue
        line = f"{COARSE_DISPLAY[cat]:<28}"
        for m in models:
            cnt, pct = table[m][cat]
            if cnt > 0:
                line += f"  {pct:>13.1f}"
            else:
                line += f"  {'--':>13}"
        avg = sum(pcts) / len(pcts) if pcts else 0
        line += f"  {avg:>6.1f}"
        print(line)

    print("-" * len(hdr))
    line = f"{'Total failed runs':<28}"
    for m in models:
        line += f"  {totals[m]:>13}"
    print(line)

    # Scope
    line = f"{'Task scope':<28}"
    for m in models:
        scope = "16-task" if m in SUBSET_MODELS else "80-task"
        line += f"  {scope:>13}"
    print(line)


def print_logic_table(table, totals):
    models = [m for m in MODEL_ORDER if m in table]
    print("\n" + "=" * 100)
    print("LOGIC-ERROR SUBCATEGORIES (% of logic errors per model, Mini-SWE-Agent)")
    print("=" * 100)

    hdr = f"{'Subcategory':<28}"
    for m in models:
        star = "*" if m in SUBSET_MODELS else ""
        hdr += f"  {MODEL_DISPLAY[m]+star:>14}"
    hdr += f"  {'Avg':>7}"
    print(hdr)
    print("-" * len(hdr))

    for cat in LOGIC_SUBCATEGORIES:
        pcts = [table[m][cat][1] for m in models if cat in table[m]]
        line = f"{LOGIC_DISPLAY[cat]:<28}"
        for m in models:
            cnt, pct = table[m][cat]
            if cnt > 0:
                line += f"  {pct:>13.1f}"
            else:
                line += f"  {'--':>13}"
        avg = sum(pcts) / len(pcts) if pcts else 0
        line += f"  {avg:>6.1f}"
        print(line)

    print("-" * len(hdr))
    line = f"{'Total logic errors':<28}"
    for m in models:
        line += f"  {totals[m]:>13}"
    print(line)


def print_datalayer_summary(dl_agg):
    print("\n" + "=" * 80)
    print("DATA-LAYER DEFECTS AS % OF LOGIC ERRORS")
    print("(incorrect_query_logic + database_runtime_error)")
    print("=" * 80)
    print(f"{'Model':<20} {'DL errors':>10} {'Total LE':>10} {'DL %':>8}")
    print("-" * 52)
    pcts = []
    for model in MODEL_ORDER:
        if model not in dl_agg:
            continue
        dl, total, pct = dl_agg[model]
        star = "*" if model in SUBSET_MODELS else ""
        print(f"{MODEL_DISPLAY[model]+star:<20} {dl:>10} {total:>10} {pct:>7.1f}%")
        pcts.append(pct)
    print("-" * 52)
    if pcts:
        print(f"{'Cross-model average':<20} {'':>10} {'':>10} {sum(pcts)/len(pcts):>7.1f}%")
        print(f"{'Range':<20} {'':>10} {'':>10} {min(pcts):.1f}--{max(pcts):.1f}%")


# ── LaTeX output ─────────────────────────────────────────────────────────────

def emit_latex_coarse(table, totals):
    models = [m for m in MODEL_ORDER if m in table]
    ncols = len(models)

    print("\n% --- Coarse failure categories (LaTeX) ---")
    print("\\begin{table}")
    print("\\centering\\small")
    cols = "@{}l" + " r" * ncols + "@{}"
    print(f"\\begin{{tabular}}{{{cols}}}")
    print("\\toprule")

    hdr = "\\textbf{Failure category}"
    for m in models:
        star = "$^*$" if m in SUBSET_MODELS else ""
        hdr += f" & \\textbf{{{MODEL_DISPLAY[m]}}}{star}"
    print(hdr + " \\\\")
    print("\\midrule")

    for cat in COARSE_CATEGORIES:
        pcts = [table[m][cat][1] for m in models]
        if all(p == 0 for p in pcts):
            continue
        line = f"\\textit{{{COARSE_DISPLAY[cat]}}}"
        for m in models:
            cnt, pct = table[m][cat]
            if cnt > 0:
                line += f" & {pct:.1f}"
            else:
                line += " & --"
        print(line + " \\\\")

    print("\\midrule")
    line = "\\textbf{Total failed runs}"
    for m in models:
        line += f" & \\textbf{{{totals[m]}}}"
    print(line + " \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Coarse failure categories (\\% of failed runs) across five models with Mini-SWE-Agent. $^*$ = 16-task subset.}")
    print("\\label{tab:failure_all_models}")
    print("\\end{table}")


def emit_latex_logic(table, totals):
    models = [m for m in MODEL_ORDER if m in table]
    ncols = len(models)

    print("\n% --- Logic-error subcategories (LaTeX) ---")
    print("\\begin{table}")
    print("\\centering\\small")
    cols = "@{}l" + " r" * ncols + "@{}"
    print(f"\\begin{{tabular}}{{{cols}}}")
    print("\\toprule")

    hdr = "\\textbf{Logic-error subcat.}"
    for m in models:
        star = "$^*$" if m in SUBSET_MODELS else ""
        hdr += f" & \\textbf{{{MODEL_DISPLAY[m]}}}{star}"
    print(hdr + " \\\\")
    print("\\midrule")

    for cat in LOGIC_SUBCATEGORIES:
        line = f"{LOGIC_DISPLAY[cat]}"
        for m in models:
            cnt, pct = table[m][cat]
            if cnt > 0:
                line += f" & {pct:.1f}"
            else:
                line += " & --"
        print(line + " \\\\")

    print("\\midrule")
    line = "\\textbf{Total logic errors}"
    for m in models:
        line += f" & \\textbf{{{totals[m]}}}"
    print(line + " \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Logic-error root causes (\\% of logic errors) across five models with Mini-SWE-Agent. $^*$ = 16-task subset.}")
    print("\\label{tab:logic_all_models}")
    print("\\end{table}")


def emit_latex_combined(coarse_table, coarse_totals, logic_table, logic_totals):
    """Emit a single combined table matching the paper's Table 5 layout but with all models."""
    models = [m for m in MODEL_ORDER if m in coarse_table]

    print("\n% --- Combined failure taxonomy, all models (LaTeX) ---")
    print("% Replaces Table 5 in rebuttal revision")
    print("\\begin{table}")
    print("\\centering\\small")
    ncols = len(models)
    cols = "@{}l " + "r" * ncols + " c l " + "r" * ncols + "@{}"
    print(f"\\begin{{tabular}}{{{cols}}}")
    print("\\toprule")

    # Header row
    hdr = "\\textbf{Failure category}"
    for m in models:
        star = "$^*$" if m in SUBSET_MODELS else ""
        abbrev = MODEL_DISPLAY[m].split("-")[0] if "-" in MODEL_DISPLAY[m] else MODEL_DISPLAY[m]
        hdr += f" & \\rotatebox{{70}}{{{MODEL_DISPLAY[m]}{star}}}"
    hdr += " & & \\textbf{Logic-error subcat.}"
    for m in models:
        star = "$^*$" if m in SUBSET_MODELS else ""
        hdr += f" & \\rotatebox{{70}}{{{MODEL_DISPLAY[m]}{star}}}"
    print(hdr + " \\\\")
    print("\\midrule")

    # Pair coarse categories with logic subcategories row by row
    coarse_active = [c for c in COARSE_CATEGORIES
                     if any(coarse_table[m][c][0] > 0 for m in models)]
    n_rows = max(len(coarse_active), len(LOGIC_SUBCATEGORIES))

    for i in range(n_rows):
        # Left side: coarse
        if i < len(coarse_active):
            cat = coarse_active[i]
            line = f"\\textit{{{COARSE_DISPLAY[cat]}}}"
            for m in models:
                cnt, pct = coarse_table[m][cat]
                line += f" & {pct:.1f}" if cnt > 0 else " & --"
        else:
            line = " " + " &" * ncols

        line += " & "

        # Right side: logic subcategories
        if i < len(LOGIC_SUBCATEGORIES):
            scat = LOGIC_SUBCATEGORIES[i]
            line += f" {LOGIC_DISPLAY[scat]}"
            for m in models:
                if m in logic_table:
                    cnt, pct = logic_table[m][scat]
                    line += f" & {pct:.1f}" if cnt > 0 else " & --"
                else:
                    line += " & --"
        else:
            line += " " + " &" * ncols

        print(line + " \\\\")

    print("\\midrule")
    # Totals row
    line = "\\textbf{Total failed runs}"
    for m in models:
        line += f" & \\textbf{{{coarse_totals[m]}}}"
    line += " & & \\textbf{Total logic errors}"
    for m in models:
        line += f" & \\textbf{{{logic_totals.get(m, 0)}}}"
    print(line + " \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Failure taxonomy across five models with Mini-SWE-Agent. "
          "Left: coarse categories (\\% of failed runs); right: logic-error subcategories "
          "(\\% of logic errors). $^*$ = 16-task subset.}")
    print("\\label{tab:failure_all_models}")
    print("\\end{table}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate failure analysis across all models for rebuttal."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Directory containing failure_analysis_*.csv and logic_subcategories*.csv")
    parser.add_argument("--latex", action="store_true",
                        help="Also emit LaTeX tables")
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.is_dir():
        print(f"Error: '{data_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    # Load data
    failure_data = load_failure_csvs(data_dir)
    logic_data = load_logic_csvs(data_dir)

    if not failure_data:
        print("No failure_analysis_*.csv files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded failure analysis for: {', '.join(MODEL_DISPLAY.get(m, m) for m in failure_data)}")
    print(f"Loaded logic subcategories for: {', '.join(MODEL_DISPLAY.get(m, m) for m in logic_data)}")

    # Compute tables
    coarse_table, coarse_totals = compute_coarse_table(failure_data)
    logic_table, logic_totals = compute_logic_table(logic_data)
    dl_agg = compute_datalayer_aggregate(logic_data)

    # Print terminal tables
    print_coarse_table(coarse_table, coarse_totals)
    print_logic_table(logic_table, logic_totals)
    print_datalayer_summary(dl_agg)

    # LaTeX
    if args.latex:
        emit_latex_coarse(coarse_table, coarse_totals)
        emit_latex_logic(logic_table, logic_totals)
        emit_latex_combined(coarse_table, coarse_totals, logic_table, logic_totals)


if __name__ == "__main__":
    main()
