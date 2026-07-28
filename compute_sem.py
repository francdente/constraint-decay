#!/usr/bin/env python3
"""
Compute per-cell SEM for Table 3 (framework leaderboard).

Uses strict_assertions_perc (A% with verifier enforcement), matching
evaluation_tables.py. For each (agent, model, framework) cell:
  1. Get individual run-level strict A% values
  2. Compute SEM across all runs for that framework

Usage:
  uv run compute_sem.py data/results
"""

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

from evaluation import collect_data, compute_framework


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", help="Root results directory (e.g. data/results)")
    args = parser.parse_args()
    base = Path(args.base_dir)

    all_runs = collect_data(base, agent_filter=None, model_filter=None)

    # (agent, model, framework) -> [run-level strict_assertions_perc]
    cell_values = defaultdict(list)
    for (agent, model), runs in all_runs.items():
        for r in runs:
            fw = compute_framework(r["task"])
            cell_values[(agent, model, fw)].append(r["strict_assertions_perc"])

    configs = [
        ("mini_swe_sdk", "gpt-5-mini", "5m-M"),
        ("openhands_sdk", "gpt-5-mini", "5m-O"),
        ("mini_swe_sdk", "qwen3-coder-next", "QC-M"),
        ("openhands_sdk", "qwen3-coder-next", "QC-O"),
        ("mini_swe_sdk", "qwen3-235b", "Q235-M"),
        ("openhands_sdk", "qwen3-235b", "Q235-O"),
        ("mini_swe_sdk", "minimax-m2.5", "MMx"),
        ("mini_swe_sdk", "gpt-5.4", "5.4"),
    ]
    frameworks = ["express", "koa", "flask", "aiohttp", "fastify", "django", "fastapi", "hono"]
    fw_display = {
        "express": "Express", "koa": "Koa", "flask": "Flask", "aiohttp": "Aiohttp",
        "fastify": "Fastify", "django": "Django", "fastapi": "FastAPI", "hono": "Hono",
    }

    print("=" * 100)
    print("Table 3 cells: strict A% ± SEM  (run-level SEM, n = number of runs)")
    print("=" * 100)
    header = f"{'Framework':<10}" + "".join(f"{label:>16}" for _, _, label in configs)
    print(header)
    print("-" * len(header))

    for fw in frameworks:
        row = f"{fw_display[fw]:<10}"
        for agent, model, label in configs:
            vals = cell_values.get((agent, model, fw), [])
            if vals:
                mean_pct = statistics.mean(vals) * 100
                if len(vals) > 1:
                    sem = (statistics.stdev(vals) / len(vals) ** 0.5) * 100
                else:
                    sem = 0.0
                row += f"  {mean_pct:>5.1f}±{sem:<4.1f} ({len(vals):>2})"
            else:
                row += f"{'--':>16}"
        print(row)


if __name__ == "__main__":
    main()
