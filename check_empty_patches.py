#!/usr/bin/env python3
"""
Find runs with empty patches for a given model.

Usage:
  uv run check_empty_patches.py data/results --model gpt-5.4
  uv run check_empty_patches.py data/results --model gpt-5.4 --agent mini_swe_sdk
"""

import argparse
import sys
from pathlib import Path

ALL_RUNTIMES = {"uv", "node", "aiohttp", "express", "fastapi", "flask", "honojs", "gin", "django", "fastify", "koa"}


def main():
    parser = argparse.ArgumentParser(description="Find runs with empty patches.")
    parser.add_argument("base_dir", help="Root results directory (e.g. data/results)")
    parser.add_argument("--model", required=True, help="Model name to check")
    parser.add_argument("--agent", help="Filter to a specific agent")
    args = parser.parse_args()

    base = Path(args.base_dir)
    empty = []
    total = 0

    for runtime in sorted(ALL_RUNTIMES):
        runtime_dir = base / runtime
        if not runtime_dir.is_dir():
            continue
        for agent_dir in sorted(runtime_dir.iterdir()):
            if not agent_dir.is_dir():
                continue
            if args.agent and agent_dir.name != args.agent:
                continue
            model_dir = agent_dir / args.model
            if not model_dir.is_dir():
                continue
            for task_dir in sorted(model_dir.iterdir()):
                if not task_dir.is_dir():
                    continue
                date_dirs = sorted(
                    [d for d in task_dir.iterdir() if d.is_dir()],
                    key=lambda d: d.name, reverse=True,
                )
                if not date_dirs:
                    continue
                for run_dir in sorted(date_dirs[0].iterdir()):
                    if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                        continue
                    total += 1
                    patches = list(run_dir.glob("*.patch"))
                    if not patches or all(p.stat().st_size == 0 for p in patches):
                        empty.append({
                            "runtime": runtime,
                            "agent": agent_dir.name,
                            "task": task_dir.name,
                            "run": run_dir.name,
                            "path": str(run_dir),
                        })

    if not total:
        print(f"No runs found for model '{args.model}'.", file=sys.stderr)
        sys.exit(1)

    print(f"{len(empty)}/{total} runs have empty patches (model={args.model})\n")
    if empty:
        for r in empty:
            print(f"  {r['runtime']:<10} {r['agent']:<16} {r['task']:<50} {r['run']}")
    else:
        print("All patches are non-empty.")


if __name__ == "__main__":
    main()
