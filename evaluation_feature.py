#!/usr/bin/env python3
"""
Compute evaluation statistics for API-Bench feature implementation tasks.

Walks data/results/{aiohttp,express,fastapi,flask,django,fastify,koa,honojs}/
and computes stats following the same logic as evaluation.py (generation tasks).

Only processes directories with the new format: agent/model/task/timestamp/run_N/
(skips old leftovers without timestamp/run_N structure).

Usage:
  python evaluation_feature.py data/results
  python evaluation_feature.py data/results --agent openhands_sdk --model gpt-5-mini
"""

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

FRAMEWORKS = {"aiohttp", "express", "fastapi", "flask", "django", "fastify", "koa", "honojs"}


def _is_timestamp_dir(name: str) -> bool:
    """Check if a directory name looks like a timestamp (YYYY-MM-DDT...)."""
    return len(name) >= 10 and name[4] == "-" and name[7] == "-"


def compute_difficulty(task_name: str) -> str:
    """Extract difficulty from task name like 'hard-focused-task-001'."""
    if task_name.startswith("simple"):
        return "simple"
    elif task_name.startswith("intermediate"):
        return "intermediate"
    elif task_name.startswith("hard"):
        return "hard"
    return "unknown"


def pass_at_k(k: int, n_executed: int, n_succeeded: int) -> float:
    if n_executed < k:
        return float(n_succeeded > 0)
    denom = math.comb(n_executed, k)
    if denom == 0:
        return 0.0
    return 1.0 - (math.comb(n_executed - n_succeeded, k) / denom)


def evaluate_csv(csv_path: Path) -> dict:
    """Parse a Newman CSV report and return assertion counts."""
    n_assertions = 0
    passed = 0
    failed = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                n_assertions += int(row.get("totalAssertions", 0) or 0)
                passed += int(row.get("executedCount", 0) or 0)
                failed += int(row.get("failedCount", 0) or 0)
            except (ValueError, TypeError):
                continue
    return {"n_assertions": n_assertions, "passed_assertions": passed, "failed_assertions": failed}


def collect_data(base_dir: Path, agent_filter: str | None, model_filter: str | None):
    """
    Walk the feature implementation results tree and collect per-run stats.

    Returns: {(agent, model): [run_stat_dict, ...]}
    """
    data: dict[tuple, list[dict]] = defaultdict(list)

    for framework in sorted(FRAMEWORKS):
        fw_dir = base_dir / framework
        if not fw_dir.is_dir():
            continue
        for agent_dir in sorted(fw_dir.iterdir()):
            if not agent_dir.is_dir():
                continue
            agent = agent_dir.name
            if agent_filter and agent != agent_filter:
                continue
            for model_dir in sorted(agent_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                model = model_dir.name
                if model_filter and model != model_filter:
                    continue
                for task_dir in sorted(model_dir.iterdir()):
                    if not task_dir.is_dir():
                        continue
                    task = task_dir.name

                    # Pick most recent timestamp folder (skip old leftovers)
                    date_dirs = sorted(
                        [d for d in task_dir.iterdir() if d.is_dir() and _is_timestamp_dir(d.name)],
                        key=lambda d: d.name,
                        reverse=True,
                    )
                    if not date_dirs:
                        continue
                    date_dir = date_dirs[0]

                    for run_dir in sorted(date_dir.iterdir()):
                        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                            continue
                        run_id = run_dir.name

                        csvs = list(run_dir.glob("newman-run-report-*.csv"))
                        if csvs:
                            result = evaluate_csv(csvs[0])
                        else:
                            result = {"n_assertions": 0, "passed_assertions": 0, "failed_assertions": 0}

                        perc = (
                            result["passed_assertions"] / result["n_assertions"]
                            if result["n_assertions"] > 0
                            else 0.0
                        )
                        tests_passed = (
                            result["failed_assertions"] == 0 and result["n_assertions"] > 0
                        )

                        data[(agent, model)].append(
                            {
                                "framework": framework,
                                "task": task,
                                "difficulty": compute_difficulty(task),
                                "run_id": run_id,
                                "n_assertions": result["n_assertions"],
                                "passed_assertions": result["passed_assertions"],
                                "failed_assertions": result["failed_assertions"],
                                "passed_assertions_perc": perc,
                                "succeeded": int(tests_passed),
                            }
                        )

    return data


def compute_task_stats(runs: list[dict]) -> list[dict]:
    """Aggregate run-level stats into task-level stats."""
    by_task: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        by_task[(r["framework"], r["task"])].append(r)

    task_stats = []
    for (framework, task), task_runs in sorted(by_task.items()):
        n_executed = len(task_runs)
        n_succeeded = sum(r["succeeded"] for r in task_runs)
        percs = [r["passed_assertions_perc"] for r in task_runs]
        mean_perc = statistics.mean(percs)

        task_stats.append(
            {
                "framework": framework,
                "task": task,
                "difficulty": compute_difficulty(task),
                "pass_at_1": pass_at_k(1, n_executed, n_succeeded),
                "passed_assertions_perc": mean_perc,
                "n_succeeded": n_succeeded,
                "n_executed": n_executed,
            }
        )

    return task_stats


def compute_global_stats(task_stats: list[dict]) -> list[dict]:
    """Aggregate task-level stats by framework, difficulty, and overall."""
    results = []

    def aggregate(granularity: str, group: str, tasks: list[dict]):
        if not tasks:
            return
        results.append({
            "granularity": granularity,
            "group": group,
            "pass_at_1": statistics.mean(t["pass_at_1"] for t in tasks),
            "passed_assertions_perc": statistics.mean(t["passed_assertions_perc"] for t in tasks),
            "n_succeeded": sum(t["n_succeeded"] for t in tasks),
            "n_executed": sum(t["n_executed"] for t in tasks),
        })

    # By framework
    by_fw: dict[str, list] = defaultdict(list)
    for t in task_stats:
        by_fw[t["framework"]].append(t)
    for fw in sorted(by_fw):
        aggregate("framework", fw, by_fw[fw])

    # By difficulty
    by_diff: dict[str, list] = defaultdict(list)
    for t in task_stats:
        by_diff[t["difficulty"]].append(t)
    for diff in ["simple", "intermediate", "hard"]:
        if diff in by_diff:
            aggregate("difficulty", diff, by_diff[diff])

    # By framework × difficulty
    by_fw_diff: dict[tuple, list] = defaultdict(list)
    for t in task_stats:
        by_fw_diff[(t["framework"], t["difficulty"])].append(t)
    for (fw, diff) in sorted(by_fw_diff):
        aggregate("framework_difficulty", f"{fw}:{diff}", by_fw_diff[(fw, diff)])

    # Global
    aggregate("global", "all", task_stats)

    return results


def print_task_stats(tasks: list[dict], agent: str, model: str):
    print(f"\n{'=' * 100}")
    print(f"TASK-LEVEL STATS: {agent} / {model}")
    print(f"{'=' * 100}")

    w_task = max(35, max((len(t["task"]) for t in tasks), default=4))
    header = (
        f"{'Framework':<10}  {'Task':<{w_task}}  {'Diff':<14}  "
        f"{'p@1':>6}  {'Assert%':>7}  {'Succ':>4}  {'Exec':>4}"
    )
    print(header)
    print("-" * len(header))
    for t in tasks:
        print(
            f"{t['framework']:<10}  {t['task']:<{w_task}}  {t['difficulty']:<14}  "
            f"{t['pass_at_1']:>6.1%}  {t['passed_assertions_perc']:>6.1%}  "
            f"{t['n_succeeded']:>4}  {t['n_executed']:>4}"
        )


def print_global_stats(global_stats: list[dict], agent: str, model: str):
    print(f"\n{'=' * 90}")
    print(f"GLOBAL STATS: {agent} / {model}")
    print(f"{'=' * 90}")

    header = (
        f"{'Granularity':<22}  {'Group':<20}  "
        f"{'p@1':>6}  {'Assert%':>7}  {'Succ':>4}  {'Exec':>4}"
    )
    print(header)
    print("-" * len(header))
    prev_gran = None
    for g in global_stats:
        if prev_gran and prev_gran != g["granularity"]:
            print()
        prev_gran = g["granularity"]
        print(
            f"{g['granularity']:<22}  {g['group']:<20}  "
            f"{g['pass_at_1']:>6.1%}  {g['passed_assertions_perc']:>6.1%}  "
            f"{g['n_succeeded']:>4}  {g['n_executed']:>4}"
        )


def export_csvs(all_runs, all_task_stats, all_global_stats, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run-level CSV
    run_csv = output_dir / "feature_run_stats.csv"
    with open(run_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "agent", "model", "framework", "task", "difficulty", "run_id",
            "n_assertions", "passed_assertions", "failed_assertions",
            "passed_assertions_perc", "succeeded",
        ])
        for (agent, model), runs in sorted(all_runs.items()):
            for r in runs:
                writer.writerow([
                    agent, model, r["framework"], r["task"], r["difficulty"],
                    r["run_id"], r["n_assertions"], r["passed_assertions"],
                    r["failed_assertions"], f"{r['passed_assertions_perc']:.4f}",
                    r["succeeded"],
                ])
    print(f"\nRun-level CSV:    {run_csv}")

    # Task-level CSV
    task_csv = output_dir / "feature_task_stats.csv"
    with open(task_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "agent", "model", "framework", "task", "difficulty",
            "pass_at_1", "passed_assertions_perc", "n_succeeded", "n_executed",
        ])
        for (agent, model), tasks in sorted(all_task_stats.items()):
            for t in tasks:
                writer.writerow([
                    agent, model, t["framework"], t["task"], t["difficulty"],
                    f"{t['pass_at_1']:.4f}", f"{t['passed_assertions_perc']:.4f}",
                    t["n_succeeded"], t["n_executed"],
                ])
    print(f"Task-level CSV:   {task_csv}")

    # Global CSV
    global_csv = output_dir / "feature_global_stats.csv"
    with open(global_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "agent", "model", "granularity", "group",
            "pass_at_1", "passed_assertions_perc", "n_succeeded", "n_executed",
        ])
        for (agent, model), gstats in sorted(all_global_stats.items()):
            for g in gstats:
                writer.writerow([
                    agent, model, g["granularity"], g["group"],
                    f"{g['pass_at_1']:.4f}", f"{g['passed_assertions_perc']:.4f}",
                    g["n_succeeded"], g["n_executed"],
                ])
    print(f"Global CSV:       {global_csv}")


def main():
    parser = argparse.ArgumentParser(description="Compute evaluation statistics for feature implementation tasks.")
    parser.add_argument("base_dir", help="Root results directory (e.g. data/results)")
    parser.add_argument("--agent", help="Filter to a specific agent")
    parser.add_argument("--model", help="Filter to a specific model")
    parser.add_argument("--output-dir", default=None, help="Directory for CSV output (default: base_dir/../)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print run-level details")
    args = parser.parse_args()

    base = Path(args.base_dir)
    if not base.is_dir():
        print(f"Error: '{base}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else base.parent

    all_runs = collect_data(base, args.agent, args.model)

    if not all_runs:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    all_task_stats = {}
    all_global_stats = {}

    for (agent, model), runs in sorted(all_runs.items()):
        task_stats = compute_task_stats(runs)
        all_task_stats[(agent, model)] = task_stats
        print_task_stats(task_stats, agent, model)

        global_stats = compute_global_stats(task_stats)
        all_global_stats[(agent, model)] = global_stats
        print_global_stats(global_stats, agent, model)

    export_csvs(all_runs, all_task_stats, all_global_stats, output_dir)


if __name__ == "__main__":
    main()
