#!/usr/bin/env python3
"""
Estimate average cost per run for each model from mini/ trajectories.

Uses two methods:
  1. If the trajectory has a nonzero 'cost' field (from litellm), use it directly.
  2. Otherwise, count tokens via tiktoken and apply known pricing.

Usage:
  uv run cost_estimate.py data/results
  uv run cost_estimate.py data/results --agent mini_swe_sdk
  uv run cost_estimate.py data/results --llm Kimi-K2.5
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import tiktoken

RUNTIMES = {"uv", "node"}

# ── Per-model pricing (USD per 1M tokens) ───────────────────────────────────
# Update these as needed. Source: provider pricing pages.

MODEL_PRICING = {
    # model_name: (input_per_1M, output_per_1M)
    "gpt-5-mini":        (1.50,   6.00),
    "gpt-5.2":           (10.00,  30.00),
    "qwen3-coder-next":  (0.50,   2.00),   # together.ai / hosted
    "minimax-m2.5":      (0.50,   2.00),   # together.ai
    "Kimi-K2.5":         (0.50,   2.80),   # together.ai
    "qwen3-235b":        (0.50,   2.00),   # together.ai
    "devstral-small":    (0.25,   1.00),   # mistral / together
}


def estimate_tokens(trajectory_path: Path, enc: tiktoken.Encoding) -> dict | None:
    """Parse mini/trajectory.json and return token counts + embedded cost."""
    try:
        data = json.loads(trajectory_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    messages = data.get("trajectories", [])
    if not messages:
        return None

    embedded_cost = data.get("cost")
    n_calls_embedded = data.get("n_calls")

    # Tokenize each message
    token_counts = []
    for m in messages:
        content = m.get("content", "") or ""
        token_counts.append(len(enc.encode(content)))

    # Simulate cumulative context (each assistant turn sees all prior messages)
    total_input = 0
    total_output = 0
    n_calls = 0

    for i, m in enumerate(messages):
        if m.get("role") == "assistant":
            total_input += sum(token_counts[:i])
            total_output += token_counts[i]
            n_calls += 1

    if n_calls == 0:
        return None

    return {
        "total_input": total_input,
        "total_output": total_output,
        "n_calls": n_calls,
        "embedded_cost": embedded_cost if embedded_cost and embedded_cost > 0 else None,
    }


def collect_data(
    results_dir: Path,
    agent_filter: str | None = None,
    llm_filter: str | None = None,
) -> dict[tuple[str, str], list[dict]]:
    """Walk results tree, extract token/cost data per run."""
    data: dict[tuple[str, str], list[dict]] = defaultdict(list)
    enc = None

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

                for traj_path in llm_dir.rglob("mini/trajectory.json"):
                    if enc is None:
                        enc = tiktoken.get_encoding("cl100k_base")
                    result = estimate_tokens(traj_path, enc)
                    if result is None:
                        continue
                    run_part = traj_path.parent.parent.name
                    task_name = traj_path.parent.parent.parent.parent.name
                    data[(agent, llm)].append({
                        "runtime": runtime,
                        "task": task_name,
                        "run": run_part,
                        **result,
                    })

    return dict(data)


def compute_cost(entry: dict, llm: str) -> float | None:
    """Compute cost for a single run. Prefer embedded cost, else use pricing table."""
    if entry["embedded_cost"] is not None:
        return entry["embedded_cost"]

    pricing = MODEL_PRICING.get(llm)
    if pricing is None:
        return None

    input_per_token = pricing[0] / 1_000_000
    output_per_token = pricing[1] / 1_000_000
    return (entry["total_input"] * input_per_token +
            entry["total_output"] * output_per_token)


def print_report(data: dict[tuple[str, str], list[dict]]) -> None:
    if not data:
        print("No trajectory data found.")
        sys.exit(0)

    print()
    print("=" * 120)
    print(f"{'Agent':<20} {'Model':<22} {'Runs':>5} "
          f"{'Avg In Tok':>12} {'Avg Out Tok':>12} "
          f"{'Avg Cost':>10} {'Total Cost':>12} {'Cost Src':>10}")
    print("-" * 120)

    for (agent, llm) in sorted(data):
        entries = data[(agent, llm)]
        n = len(entries)

        avg_in = statistics.mean(e["total_input"] for e in entries)
        avg_out = statistics.mean(e["total_output"] for e in entries)

        costs = [compute_cost(e, llm) for e in entries]
        valid_costs = [c for c in costs if c is not None]

        if valid_costs:
            avg_cost = statistics.mean(valid_costs)
            total_cost = sum(valid_costs)
            # Determine source
            has_embedded = any(e["embedded_cost"] is not None for e in entries)
            src = "litellm" if has_embedded else "estimate"
        else:
            avg_cost = None
            total_cost = None
            src = "N/A"

        cost_str = f"${avg_cost:.4f}" if avg_cost is not None else "N/A"
        total_str = f"${total_cost:.2f}" if total_cost is not None else "N/A"

        print(f"{agent:<20} {llm:<22} {n:>5} "
              f"{avg_in:>12,.0f} {avg_out:>12,.0f} "
              f"{cost_str:>10} {total_str:>12} {src:>10}")

    # Per-model summary
    print()
    print("=" * 90)
    print("PER-MODEL COST SUMMARY (across all agents)")
    print("=" * 90)
    print(f"{'Model':<22} {'Runs':>5} {'Avg $/run':>10} {'Med $/run':>10} "
          f"{'Std $/run':>10} {'Total $':>10}")
    print("-" * 90)

    by_llm: dict[str, list[float]] = defaultdict(list)
    for (agent, llm), entries in data.items():
        for e in entries:
            c = compute_cost(e, llm)
            if c is not None:
                by_llm[llm].append(c)

    for llm in sorted(by_llm):
        costs = by_llm[llm]
        n = len(costs)
        avg = statistics.mean(costs)
        med = statistics.median(costs)
        std = statistics.stdev(costs) if len(costs) > 1 else 0
        total = sum(costs)
        print(f"{llm:<22} {n:>5} ${avg:>9.4f} ${med:>9.4f} "
              f"${std:>9.4f} ${total:>9.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate average cost per run from mini/ trajectories."
    )
    parser.add_argument("results_dir", type=Path, help="Path to data/results")
    parser.add_argument("--agent", help="Filter to a specific agent")
    parser.add_argument("--llm", help="Filter LLMs containing this substring")
    args = parser.parse_args()

    if not args.results_dir.is_dir():
        print(f"Error: '{args.results_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    data = collect_data(args.results_dir, agent_filter=args.agent, llm_filter=args.llm)
    print_report(data)


if __name__ == "__main__":
    main()
