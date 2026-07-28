#!/usr/bin/env python3
"""
Plot framework sensitivity across model-scaffold configurations.

One line per framework, configurations on X-axis, A% (strict) on Y-axis,
stderr across individual runs as a shaded band. Mid-tier configurations
first, then strong models.

Usage:
  uv run plot_framework_sensitivity.py data/results
  uv run plot_framework_sensitivity.py data/results --output figures/fig_framework_sensitivity.pdf
"""

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from evaluation import collect_data, compute_framework

CONFIGS = [
    ("mini_swe_sdk", "gpt-5-mini", "GPT-5-mini\nMini-SWE"),
    ("openhands_sdk", "gpt-5-mini", "GPT-5-mini\nOpenHands"),
    ("mini_swe_sdk", "qwen3-coder-next", "Qwen3-Coder\nMini-SWE"),
    ("openhands_sdk", "qwen3-coder-next", "Qwen3-Coder\nOpenHands"),
    ("mini_swe_sdk", "minimax-m2.5", "MiniMax-M2.5\nMini-SWE"),
    ("mini_swe_sdk", "gpt-5.4", "GPT-5.4\nMini-SWE"),
]

# Boundary between mid-tier and strong configurations (index of first strong one).
N_MID_TIER = 4

FRAMEWORKS = ["express", "koa", "flask", "django", "fastapi", "hono"]
FW_DISPLAY = {
    "express": "Express", "koa": "Koa", "flask": "Flask", "aiohttp": "Aiohttp",
    "fastify": "Fastify", "django": "Django", "fastapi": "FastAPI", "hono": "Hono",
}

FW_COLORS = {
    "express": "#1f77b4",
    "koa": "#ff7f0e",
    "flask": "#2ca02c",
    "aiohttp": "#d62728",
    "fastify": "#9467bd",
    "django": "#8c564b",
    "fastapi": "#e377c2",
    "hono": "#7f7f7f",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", help="Root results directory")
    parser.add_argument("--output", default="fig_framework_sensitivity.pdf",
                        help="Output file (default: fig_framework_sensitivity.pdf)")
    args = parser.parse_args()

    all_runs = collect_data(Path(args.base_dir), agent_filter=None, model_filter=None)

    # (agent, model, framework) -> [run-level strict_assertions_perc]
    cell_values = defaultdict(list)
    for (agent, model), runs in all_runs.items():
        for r in runs:
            cell_values[(agent, model, compute_framework(r["task"]))].append(
                r["strict_assertions_perc"]
            )

    x_labels = [label for _, _, label in CONFIGS]
    x_pos = list(range(len(CONFIGS)))

    fig, ax = plt.subplots(figsize=(12, 5.5))

    for fw in FRAMEWORKS:
        means, sems = [], []
        for agent, model, _ in CONFIGS:
            vals = cell_values.get((agent, model, fw), [])
            if vals:
                means.append(statistics.mean(vals) * 100)
                sems.append((statistics.stdev(vals) / len(vals) ** 0.5) * 100
                            if len(vals) > 1 else 0.0)
            else:
                means.append(None)
                sems.append(0.0)

        color = FW_COLORS[fw]
        nan = float("nan")
        y_vals = [m if m is not None else nan for m in means]
        y_lo = [m - s if m is not None else nan for m, s in zip(means, sems)]
        y_hi = [m + s if m is not None else nan for m, s in zip(means, sems)]

        ax.plot(x_pos, y_vals, marker="o", markersize=5, label=FW_DISPLAY[fw],
                color=color, linewidth=2.0, zorder=3)
        ax.fill_between(x_pos, y_lo, y_hi, alpha=0.15, color=color, zorder=2)

    boundary = N_MID_TIER - 0.5
    ax.axvline(x=boundary, color="grey", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.text((N_MID_TIER - 1) / 2, 104, "Mid-tier", ha="center", fontsize=11, color="grey")
    ax.text((N_MID_TIER + len(CONFIGS) - 1) / 2, 104, "Strong", ha="center",
            fontsize=11, color="grey")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_xlim(-0.25, len(CONFIGS) - 0.75)
    ax.set_ylim(-2, 110)
    ax.set_ylabel("A\\% (strict, with verifier enforcement)"
                  if plt.rcParams["text.usetex"] else "A% (strict, with verifier enforcement)",
                  fontsize=12)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left", ncol=2, fontsize=11, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved to {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
