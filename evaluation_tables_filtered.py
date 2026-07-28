#!/usr/bin/env python3
"""
Reproduce the paper tables excluding runs where agents hit turn/iteration limits.

Uses trajectory_analysis to identify limit-exceeded runs, filters them out,
then recomputes all tables using the same logic as evaluation_tables.py.

Usage:
  uv run evaluation_tables_filtered.py data/results
  uv run evaluation_tables_filtered.py data/results --latex
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from evaluation import (
    collect_data as collect_generation_data,
    compute_task_stats as compute_generation_task_stats,
)
from evaluation_feature import (
    collect_data as collect_feature_data,
    compute_task_stats as compute_feature_task_stats,
)
from evaluation_tables import (
    compute_table1,
    compute_table1_raw,
    compute_table2a,
    compute_table2b,
    compute_table3,
    compute_table3_all_models,
    print_table1,
    print_table2a,
    print_table2b,
    print_table3,
    print_table3_all_models,
)
from trajectory_analysis import collect_results as collect_trajectory_results


def build_exclusion_set(base_dir: Path) -> set[tuple[str, str, str, str]]:
    """
    Build a set of (agent, model, task, run_id) tuples for runs that
    hit turn/iteration limits.
    """
    traj_results = collect_trajectory_results(base_dir, agent_filter=None, model_filter=None)
    return {
        (r["agent"], r["model"], r["task"], r["run_id"])
        for r in traj_results
        if r["reason"] == "LimitsExceeded"
    }


def filter_runs(
    all_runs: dict[tuple, list[dict]],
    exclusion_set: set[tuple[str, str, str, str]],
) -> tuple[dict[tuple, list[dict]], int]:
    """
    Remove limit-exceeded runs from the collected data.

    Returns: (filtered_runs, n_excluded)
    """
    filtered = {}
    n_excluded = 0
    for (agent, model), runs in all_runs.items():
        kept = []
        for r in runs:
            task = r["task"]
            run_id = r["run_id"]
            if (agent, model, task, run_id) in exclusion_set:
                n_excluded += 1
            else:
                kept.append(r)
        if kept:
            filtered[(agent, model)] = kept
    return filtered, n_excluded


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce paper tables excluding limit-exceeded runs.",
    )
    parser.add_argument("base_dir", help="Root results directory (e.g. data/results)")
    parser.add_argument("--latex", action="store_true", help="Also print LaTeX source")
    args = parser.parse_args()

    base = Path(args.base_dir)
    if not base.is_dir():
        print(f"Error: '{base}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    # ── Build exclusion set from trajectories ────────────────────────
    exclusion_set = build_exclusion_set(base)
    print(f"Identified {len(exclusion_set)} limit-exceeded runs to exclude.\n")

    # ── Collect and filter generation task data ──────────────────────
    gen_runs = collect_generation_data(base, agent_filter=None, model_filter=None)
    if not gen_runs:
        print("No generation results found.", file=sys.stderr)
        sys.exit(1)

    gen_runs_filtered, gen_excluded = filter_runs(gen_runs, exclusion_set)

    # Count totals for the summary
    gen_total = sum(len(runs) for runs in gen_runs.values())
    gen_kept = sum(len(runs) for runs in gen_runs_filtered.values())
    print(f"Generation tasks: {gen_total} runs → {gen_excluded} excluded → {gen_kept} kept")

    # Per agent/model breakdown of exclusions
    for (agent, model) in sorted(gen_runs.keys()):
        orig = len(gen_runs[(agent, model)])
        filt = len(gen_runs_filtered.get((agent, model), []))
        if orig != filt:
            print(f"  {agent}/{model}: {orig} → {filt} (-{orig - filt})")

    all_gen_task_stats = {}
    for key, runs in gen_runs_filtered.items():
        all_gen_task_stats[key] = compute_generation_task_stats(runs)

    # ── Table 1 ──────────────────────────────────────────────────────
    table1 = compute_table1(all_gen_task_stats)
    print_table1(table1, latex=args.latex,
                 title="TABLE 1 (FILTERED): A% and pass@1 — limit-exceeded runs excluded",
                 label="tab:main_results_filtered")

    # ── Table 1-raw ──────────────────────────────────────────────────
    table1_raw = compute_table1_raw(gen_runs_filtered)
    print_table1(table1_raw, latex=args.latex,
                 title="TABLE 1-RAW (FILTERED): A% and pass@1 WITHOUT verifier enforcement",
                 label="tab:main_results_raw_filtered")

    # ── Table 2a ─────────────────────────────────────────────────────
    table2a = compute_table2a(all_gen_task_stats)
    print_table2a(table2a, latex=args.latex)

    # ── Table 3 ──────────────────────────────────────────────────────
    table3 = compute_table3(all_gen_task_stats)
    print_table3(table3, latex=args.latex)

    # ── Table 3-ALL ──────────────────────────────────────────────────
    table3_all = compute_table3_all_models(all_gen_task_stats)
    print_table3_all_models(table3_all, table3, latex=args.latex)

    # ── Collect and filter feature task data ──────────────────────────
    feat_runs = collect_feature_data(base, agent_filter=None, model_filter=None)
    if feat_runs:
        feat_runs_filtered, feat_excluded = filter_runs(feat_runs, exclusion_set)
        feat_total = sum(len(runs) for runs in feat_runs.values())
        feat_kept = sum(len(runs) for runs in feat_runs_filtered.values())
        print(f"\nFeature tasks: {feat_total} runs → {feat_excluded} excluded → {feat_kept} kept")

        for (agent, model) in sorted(feat_runs.keys()):
            orig = len(feat_runs[(agent, model)])
            filt = len(feat_runs_filtered.get((agent, model), []))
            if orig != filt:
                print(f"  {agent}/{model}: {orig} → {filt} (-{orig - filt})")

        if feat_runs_filtered:
            all_feat_task_stats = {}
            for key, runs in feat_runs_filtered.items():
                all_feat_task_stats[key] = compute_feature_task_stats(runs)

            table2b = compute_table2b(all_feat_task_stats)
            print_table2b(table2b, latex=args.latex)
    else:
        print("\nNo feature implementation results found; skipping Table 2b.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
