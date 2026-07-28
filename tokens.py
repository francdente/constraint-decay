#!/usr/bin/env python3
"""
Compute input/output token counts per agent/LLM from saved trajectories.

OpenHands: tokens read directly from base_state.json accumulated usage.
Mini-SWE:  tokens inferred by simulating cumulative context with tiktoken.

Usage:
  python run/tokens.py data/results
  python run/tokens.py data/results --agent openhands_sdk --llm gpt-5.2
  python run/tokens.py data/results --llm minimax-m2.5
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import tiktoken

RUNTIMES = {"uv", "node"}


def extract_openhands_tokens(base_state_path: Path) -> dict | None:
    """Parse base_state.json and return token usage."""
    try:
        data = json.loads(base_state_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    metrics = data.get("stats", {}).get("usage_to_metrics", {})
    default = metrics.get("default")
    if default is None:
        return None

    acc = default.get("accumulated_token_usage", {})
    prompt = acc.get("prompt_tokens", 0)
    completion = acc.get("completion_tokens", 0)
    per_turn = default.get("token_usages", [])

    return {
        "total_input": prompt,
        "total_output": completion,
        "n_calls": len(per_turn),
    }


def extract_miniswe_tokens(trajectory_path: Path, enc: tiktoken.Encoding) -> dict | None:
    """Parse trajectory.json and simulate cumulative context to count tokens."""
    try:
        data = json.loads(trajectory_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    messages = data.get("trajectories", [])
    if not messages:
        return None

    # Pre-tokenize each message content
    token_counts = []
    for m in messages:
        content = m.get("content", "") or ""
        token_counts.append(len(enc.encode(content)))

    # Simulate cumulative calls: for each assistant message, input = all prior messages
    total_input = 0
    total_output = 0
    n_calls = 0

    for i, m in enumerate(messages):
        if m.get("role") == "assistant":
            # Input = sum of token counts for messages[0..i-1]
            call_input = sum(token_counts[:i])
            call_output = token_counts[i]
            total_input += call_input
            total_output += call_output
            n_calls += 1

    if n_calls == 0:
        return None

    return {
        "total_input": total_input,
        "total_output": total_output,
        "n_calls": n_calls,
    }


def collect_tokens(
    results_dir: Path,
    agent_filter: str | None = None,
    llm_filter: str | None = None,
    task_filter: str | None = None,
) -> dict[tuple[str, str], list[dict]]:
    """Walk directory tree and extract tokens per run.

    Returns: {(agent, llm): [run_dict, ...]}
    """
    data: dict[tuple[str, str], list[dict]] = defaultdict(list)
    enc = None  # lazy-init tiktoken

    for runtime in sorted(RUNTIMES):
        runtime_dir = results_dir / runtime
        if not runtime_dir.is_dir():
            continue
        for agent_dir in sorted(runtime_dir.iterdir()):
            if not agent_dir.is_dir():
                continue
            agent = agent_dir.name
            if agent_filter and agent != agent_filter:
                continue

            for llm_dir in sorted(agent_dir.iterdir()):
                if not llm_dir.is_dir():
                    continue
                llm = llm_dir.name
                if llm_filter and llm_filter not in llm:
                    continue

                for task_dir in sorted(llm_dir.iterdir()):
                    if not task_dir.is_dir():
                        continue
                    task = task_dir.name
                    if task_filter and task_filter not in task:
                        continue

                    if agent == "openhands_sdk":
                        for bs in task_dir.rglob("base_state.json"):
                            result = extract_openhands_tokens(bs)
                            if result is None:
                                continue
                            # run_N is two levels up: .../run_N/{session}/base_state.json
                            run_part = bs.parent.parent.name
                            data[(agent, llm)].append({
                                "runtime": runtime,
                                "task": task,
                                "run": run_part,
                                **result,
                            })

                    elif agent in ("mini_swe_sdk", "oracle_context_interactive"):
                        for traj in task_dir.rglob("mini/trajectory.json"):
                            if enc is None:
                                enc = tiktoken.get_encoding("cl100k_base")
                            result = extract_miniswe_tokens(traj, enc)
                            if result is None:
                                continue
                            run_part = traj.parent.parent.name
                            data[(agent, llm)].append({
                                "runtime": runtime,
                                "task": task,
                                "run": run_part,
                                **result,
                            })

    return dict(data)


def print_report(data: dict[tuple[str, str], list[dict]]) -> None:
    if not data:
        print("No token data found.")
        sys.exit(0)

    print()
    print("=" * 100)
    print(f"{'Agent':<28} {'LLM':<22} {'Runs':>5} "
          f"{'Avg In':>12} {'Avg Out':>12} {'Total In':>14} {'Total Out':>14}")
    print("-" * 100)

    grand_in = 0
    grand_out = 0
    grand_runs = 0

    for (agent, llm) in sorted(data):
        entries = data[(agent, llm)]
        n = len(entries)
        total_in = sum(e["total_input"] for e in entries)
        total_out = sum(e["total_output"] for e in entries)
        avg_in = total_in / n
        avg_out = total_out / n

        print(f"{agent:<28} {llm:<22} {n:>5} "
              f"{avg_in:>12,.0f} {avg_out:>12,.0f} {total_in:>14,} {total_out:>14,}")

        grand_in += total_in
        grand_out += total_out
        grand_runs += n

    print("-" * 100)
    print(f"{'TOTAL':<28} {'':<22} {grand_runs:>5} "
          f"{'':>12} {'':>12} {grand_in:>14,} {grand_out:>14,}")
    print("=" * 100)

    # Per-task breakdown
    for (agent, llm) in sorted(data):
        entries = data[(agent, llm)]
        print(f"\n--- {agent} / {llm} per-task breakdown ---")
        print(f"  {'Rt':<5} {'Task':<50} {'Runs':>4} "
              f"{'Avg In':>11} {'Avg Out':>11} {'Avg Calls':>10}")

        by_task: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for e in entries:
            by_task[(e["runtime"], e["task"])].append(e)

        for (rt, task) in sorted(by_task):
            runs = by_task[(rt, task)]
            n = len(runs)
            avg_in = statistics.mean(r["total_input"] for r in runs)
            avg_out = statistics.mean(r["total_output"] for r in runs)
            avg_calls = statistics.mean(r["n_calls"] for r in runs)
            print(f"  {rt:<5} {task:<50} {n:>4} "
                  f"{avg_in:>11,.0f} {avg_out:>11,.0f} {avg_calls:>10.1f}")


def main():
    parser = argparse.ArgumentParser(description="Compute token usage from trajectories.")
    parser.add_argument("results_dir", type=Path, help="Path to data/results")
    parser.add_argument("--agent", help="Filter to a specific agent (exact match)")
    parser.add_argument("--llm", help="Filter LLMs containing this substring")
    parser.add_argument("--task", help="Filter tasks containing this substring")
    args = parser.parse_args()

    if not args.results_dir.is_dir():
        print(f"Error: {args.results_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    data = collect_tokens(args.results_dir, agent_filter=args.agent,
                          llm_filter=args.llm, task_filter=args.task)
    print_report(data)


if __name__ == "__main__":
    main()
