#!/usr/bin/env python3
"""
Analyze agent trajectories to determine how runs terminated.

For each run, classifies the termination reason:
  - Submitted:       agent finished normally (called finish/submit)
  - LimitsExceeded:  agent hit step/iteration/turn limits
  - Error:           agent crashed or encountered an unrecoverable error
  - Unknown:         could not determine termination reason

Supports both Mini-SWE-Agent and OpenHands trajectory formats.

Usage:
  uv run trajectory_analysis.py data/results
  uv run trajectory_analysis.py data/results --agent mini_swe_sdk
  uv run trajectory_analysis.py data/results --agent openhands_sdk --model gpt-5.2
  uv run trajectory_analysis.py data/results --csv data/trajectory_analysis.csv
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


# Only generation-task runtimes (consistent with evaluation.py / failure_analysis.py)
RUNTIMES = {"uv", "node"}


def classify_mini_swe(run_dir: Path) -> dict | None:
    """Classify a Mini-SWE-Agent run from its trajectory.json."""
    traj_path = run_dir / "mini" / "trajectory.json"
    if not traj_path.exists():
        return None

    try:
        traj = json.loads(traj_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    status = traj.get("status", "")
    n_calls = traj.get("n_calls", 0)
    messages = traj.get("trajectories", [])
    n_messages = len(messages)

    if status == "Submitted":
        reason = "Submitted"
    elif status == "LimitsExceeded":
        reason = "LimitsExceeded"
    elif status == "in_progress":
        # Agent crashed mid-run (only the interim save exists)
        reason = "Error"
    elif status:
        # Any other TerminatingException subclass
        reason = "Error"
    else:
        reason = "Unknown"

    return {
        "reason": reason,
        "n_calls": n_calls,
        "n_messages": n_messages,
        "raw_status": status,
    }


def classify_openhands(run_dir: Path) -> dict | None:
    """Classify an OpenHands run from its base_state.json and event files."""
    # Find the session directory (UUID-named subdirectory)
    session_dirs = [
        d for d in run_dir.iterdir()
        if d.is_dir() and (d / "base_state.json").exists()
    ]
    if not session_dirs:
        return None

    session_dir = session_dirs[0]
    base_state_path = session_dir / "base_state.json"

    try:
        base_state = json.loads(base_state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    execution_status = base_state.get("execution_status", "")
    max_iterations = base_state.get("max_iterations", 500)

    # Count events and check last event
    events_dir = session_dir / "events"
    has_finish = False
    n_events = 0
    last_event = None

    if events_dir.is_dir():
        event_files = sorted(events_dir.glob("event-*.json"))
        n_events = len(event_files)
        if event_files:
            try:
                last_event = json.loads(event_files[-1].read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

    if last_event:
        # Check for finish tool observation
        tool_name = last_event.get("tool_name", "")
        obs = last_event.get("observation", {})
        obs_kind = obs.get("kind", "") if isinstance(obs, dict) else ""
        if tool_name == "finish" or obs_kind == "FinishObservation":
            has_finish = True

    if has_finish:
        reason = "Submitted"
    elif execution_status in ("running", "stuck", "idle"):
        # running = hit max_iteration_per_run without finishing
        # stuck = agent detected it was looping and gave up
        # idle = agent stopped responding (outer MAX_TURNS loop exhausted)
        reason = "LimitsExceeded"
    elif execution_status == "error":
        reason = "Error"
    elif execution_status == "finished" and not has_finish:
        # Finished but agent never called finish → outer turn loop exhausted
        reason = "LimitsExceeded"
    else:
        reason = "Unknown"

    return {
        "reason": reason,
        "n_events": n_events,
        "max_iterations": max_iterations,
        "raw_status": execution_status,
        "has_finish_event": has_finish,
    }


def classify_run(run_dir: Path, agent: str) -> dict | None:
    """Classify a single run directory."""
    if agent == "mini_swe_sdk":
        return classify_mini_swe(run_dir)
    elif agent == "openhands_sdk":
        return classify_openhands(run_dir)
    # Try both
    result = classify_mini_swe(run_dir)
    if result is not None:
        return result
    return classify_openhands(run_dir)


def collect_results(
    base_dir: Path,
    agent_filter: str | None,
    model_filter: str | None,
) -> list[dict]:
    """Walk the results tree and classify every run."""
    results = []

    for runtime in sorted(RUNTIMES):
        runtime_dir = base_dir / runtime
        if not runtime_dir.is_dir():
            continue

        for agent_dir in sorted(runtime_dir.iterdir()):
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

                    # Pick most recent date folder
                    date_dirs = sorted(
                        [d for d in task_dir.iterdir() if d.is_dir()],
                        key=lambda d: d.name,
                        reverse=True,
                    )
                    if not date_dirs:
                        continue
                    date_dir = date_dirs[0]

                    for run_dir in sorted(date_dir.iterdir()):
                        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                            continue

                        classification = classify_run(run_dir, agent)
                        if classification is None:
                            classification = {"reason": "NoTrajectory", "raw_status": ""}

                        results.append({
                            "runtime": runtime,
                            "agent": agent,
                            "model": model,
                            "task": task,
                            "run_id": run_dir.name,
                            **classification,
                        })

    return results


def print_summary(results: list[dict]):
    """Print termination reason summary tables."""
    if not results:
        print("No trajectories found.", file=sys.stderr)
        sys.exit(1)

    # Global counts
    by_reason = defaultdict(int)
    for r in results:
        by_reason[r["reason"]] += 1
    total = len(results)

    print(f"\n{'=' * 70}")
    print(f"TRAJECTORY TERMINATION ANALYSIS  ({total} runs)")
    print(f"{'=' * 70}")

    print(f"\n{'Reason':<20} {'Count':>6} {'%':>7}")
    print("-" * 35)
    for reason in ["Submitted", "LimitsExceeded", "Error", "NoTrajectory", "Unknown"]:
        count = by_reason.get(reason, 0)
        if count > 0:
            print(f"{reason:<20} {count:>6} {count/total:>6.1%}")
    print("-" * 35)
    print(f"{'Total':<20} {total:>6}")

    # Per agent/model breakdown
    by_agent_model = defaultdict(lambda: defaultdict(int))
    agent_model_total = defaultdict(int)
    for r in results:
        key = (r["agent"], r["model"])
        by_agent_model[key][r["reason"]] += 1
        agent_model_total[key] += 1

    print(f"\n{'=' * 90}")
    print("PER AGENT/MODEL BREAKDOWN")
    print(f"{'=' * 90}")

    reasons = ["Submitted", "LimitsExceeded", "Error", "NoTrajectory", "Unknown"]
    header = f"{'Agent':<16} {'Model':<22} {'Total':>5}"
    for reason in reasons:
        header += f" {reason:>15}"
    print(header)
    print("-" * len(header))

    for (agent, model) in sorted(by_agent_model.keys()):
        counts = by_agent_model[(agent, model)]
        t = agent_model_total[(agent, model)]
        line = f"{agent:<16} {model:<22} {t:>5}"
        for reason in reasons:
            c = counts.get(reason, 0)
            if c > 0:
                line += f" {c:>6} ({c/t:>4.0%})"
            else:
                line += f" {'':>15}"
        print(line)

    # Per runtime breakdown (compact)
    by_runtime = defaultdict(lambda: defaultdict(int))
    runtime_total = defaultdict(int)
    for r in results:
        by_runtime[r["runtime"]][r["reason"]] += 1
        runtime_total[r["runtime"]] += 1

    print(f"\n{'=' * 70}")
    print("PER RUNTIME BREAKDOWN")
    print(f"{'=' * 70}")

    header = f"{'Runtime':<12} {'Total':>5} {'Submitted':>10} {'LimitsExc':>10} {'Error':>10} {'Other':>10}"
    print(header)
    print("-" * len(header))
    for runtime in sorted(by_runtime.keys()):
        counts = by_runtime[runtime]
        t = runtime_total[runtime]
        sub = counts.get("Submitted", 0)
        lim = counts.get("LimitsExceeded", 0)
        err = counts.get("Error", 0)
        other = t - sub - lim - err
        print(
            f"{runtime:<12} {t:>5} {sub:>10} {lim:>10} {err:>10}"
            f" {other:>10}" if other else f"{runtime:<12} {t:>5} {sub:>10} {lim:>10} {err:>10}"
        )


def print_limits_detail(results: list[dict]):
    """Print details for runs that hit limits."""
    limited = [r for r in results if r["reason"] == "LimitsExceeded"]
    if not limited:
        return

    print(f"\n{'=' * 100}")
    print(f"RUNS THAT HIT LIMITS ({len(limited)} total)")
    print(f"{'=' * 100}")

    w_task = max(30, max((len(r["task"]) for r in limited), default=4))
    header = f"{'Runtime':<10} {'Agent':<16} {'Model':<22} {'Task':<{w_task}} {'Run':<6}"
    print(header)
    print("-" * len(header))
    for r in sorted(limited, key=lambda x: (x["agent"], x["model"], x["runtime"], x["task"])):
        print(f"{r['runtime']:<10} {r['agent']:<16} {r['model']:<22} {r['task']:<{w_task}} {r['run_id']:<6}")


def export_csv(results: list[dict], csv_path: Path):
    """Export results to CSV."""
    import csv

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["runtime", "agent", "model", "task", "run_id", "reason", "raw_status"])
        for r in results:
            writer.writerow([
                r["runtime"], r["agent"], r["model"], r["task"],
                r["run_id"], r["reason"], r.get("raw_status", ""),
            ])
    print(f"\nCSV exported to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze trajectory termination reasons.")
    parser.add_argument("base_dir", help="Root results directory (e.g. data/results)")
    parser.add_argument("--agent", help="Filter to a specific agent")
    parser.add_argument("--model", help="Filter to a specific model")
    parser.add_argument("--csv", help="Export results to CSV", metavar="PATH")
    parser.add_argument("--detail", action="store_true", help="Print per-run detail for limit-exceeded runs")
    args = parser.parse_args()

    base = Path(args.base_dir)
    if not base.is_dir():
        print(f"Error: '{base}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    results = collect_results(base, args.agent, args.model)
    print_summary(results)

    if args.detail:
        print_limits_detail(results)

    if args.csv:
        export_csv(results, Path(args.csv))


if __name__ == "__main__":
    main()
