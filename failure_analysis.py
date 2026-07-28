#!/usr/bin/env python3
"""
Trajectory failure analysis using an LLM-as-judge.

For each failed run (tests_passed=False OR any constraint violated), extracts
the last N trajectory turns, Newman test failures, server logs, and verifier
results, then asks an LLM judge to classify the failure into one of 8
categories adapted from SWE-EVO (EMNLP 2025) and SWE-Bench Pro.

Usage:
  python run/failure_analysis.py data/results --llm minimax-m2.5 --agent mini_swe_sdk --output data/failure_analysis.csv
  python run/failure_analysis.py data/results --dry-run
  python run/failure_analysis.py data/results --output data/failure_analysis.csv
"""

import argparse
import asyncio
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation import collect_data, RUNTIMES

# ── Failure taxonomy ─────────────────────────────────────────────────────────

CATEGORIES = {
    "server_startup_failure": (
        "Server never starts — syntax errors, missing imports, wrong entry point, "
        "port/host misconfiguration. Newman reports 0 assertions or all requests "
        "fail with connection errors."
    ),
    "incomplete_implementation": (
        "Server starts but many endpoints are missing or return 404/500. The agent "
        "didn't finish building the API surface."
    ),
    "constraint_violation": (
        "Server is functionally correct (or nearly so) but violates a structural "
        "constraint — wrong DB, no clean architecture layers, wrong/missing ORM."
    ),
    "schema_format_error": (
        "Endpoints exist and return data, but JSON response structure is wrong — "
        "missing fields, wrong nesting, wrong types."
    ),
    "logic_error": (
        "Correct response structure but wrong behavior — broken auth, wrong slugs, "
        "broken filtering/pagination, wrong state mutations."
    ),
    "stuck_in_loop": (
        "Agent stuck retrying the same failing command, re-reading files, or "
        "oscillating between edits without progress."
    ),
    "premature_termination": (
        "Agent gives up early, declares completion prematurely, or exhausts turn "
        "budget before finishing."
    ),
    "other": (
        "The failure does not clearly fit any of the above categories, or the root "
        "cause is ambiguous / involves multiple equally dominant factors."
    ),
}

CATEGORY_LIST_TEXT = "\n".join(
    f"  - `{name}`: {desc}" for name, desc in CATEGORIES.items()
)

SYSTEM_PROMPT = f"""\
You are an expert software engineering analyst. Your task is to categorize \
why an LLM coding agent failed to produce a working backend API.

You will be given:
1. The task description (which framework, which constraints)
2. The last N turns of the agent's trajectory (its actions and reasoning)
3. Test results showing which API tests failed (if available)
4. Server logs (if available)
5. Static verifier results (if applicable)

Classify the failure into exactly ONE primary category:
{CATEGORY_LIST_TEXT}

Respond with a JSON object and nothing else:
{{"category": "<category_name>", "reasoning": "<1-2 sentence explanation>"}}"""


# ── Context extraction helpers ───────────────────────────────────────────────

def _truncate(text: str, max_chars: int = 3000) -> str:
    """Keep first half + last half if text exceeds max_chars."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n[... truncated ...]\n\n" + text[-half:]


def extract_trajectory(run_dir: Path, max_turns: int) -> str | None:
    """Extract last N turns from mini/trajectory.json."""
    traj_path = run_dir / "mini" / "trajectory.json"
    if not traj_path.exists():
        return None
    try:
        obj = json.loads(traj_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    turns = obj.get("trajectories", [])
    if not turns:
        return None
    last_turns = turns[-max_turns:]
    parts = []
    for t in last_turns:
        role = t.get("role", "unknown")
        content = t.get("content", "")
        content = _truncate(content)
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


def extract_newman_failures(run_dir: Path) -> tuple[bool, str]:
    """Parse Newman CSV and return (csv_exists, failure_summary)."""
    csvs = list(run_dir.glob("newman-run-report-*.csv"))
    if not csvs:
        return False, "No Newman CSV found — server likely never started or health-check failed."
    failures = []
    with open(csvs[0], newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                failed_count = int(row.get("failedCount", 0) or 0)
            except (ValueError, TypeError):
                continue
            if failed_count > 0:
                name = row.get("requestName", "?")
                method = row.get("method", "?")
                code = row.get("code", "?")
                failures.append(f"{name} {method} → {code}: {failed_count} failed")
    if not failures:
        return True, "All Newman assertions passed."
    return True, "\n".join(failures)


def extract_server_log(run_dir: Path, last_n: int = 30) -> str | None:
    """Return last N lines of conduit.log."""
    log_path = run_dir / "conduit.log"
    if not log_path.exists():
        return None
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except OSError:
        return None
    return "\n".join(lines[-last_n:]) if lines else None


def describe_task(task_name: str, runtime: str) -> str:
    """Build a human-readable task description from the task name."""
    parts = task_name.split("-")
    framework = parts[0] if parts else task_name
    constraints = parts[2:] if len(parts) > 2 else []
    if constraints == ["unconstrained"]:
        constraints = []
    desc = f"Framework: {framework}, Runtime: {runtime}"
    if constraints:
        desc += f", Constraints: {', '.join(constraints)}"
    else:
        desc += ", Constraints: none (unconstrained)"
    return desc


def build_verifier_summary(run_info: dict) -> str:
    """Summarize verifier compliance from run_info dict."""
    lines = []
    if not run_info["arch_compliant"]:
        lines.append(f"Architecture: FAILED — {run_info['arch_reason']}")
    if not run_info["db_compliant"]:
        lines.append(f"Database: FAILED — {run_info['db_reason']}")
    if not run_info["orm_compliant"]:
        lines.append(f"ORM: FAILED — {run_info['orm_reason']}")
    return "\n".join(lines) if lines else "All verifier checks passed."


# ── Judge call ───────────────────────────────────────────────────────────────

def build_user_prompt(
    task_name: str,
    runtime: str,
    run_info: dict,
    trajectory_text: str | None,
    newman_exists: bool,
    newman_summary: str,
    server_log: str | None,
) -> str:
    """Assemble the user-side prompt for the judge."""
    sections = []

    sections.append(f"## Task\n{describe_task(task_name, runtime)}")

    sections.append(f"## Test Results\n{newman_summary}")

    verifier = build_verifier_summary(run_info)
    if verifier != "All verifier checks passed.":
        sections.append(f"## Verifier Results\n{verifier}")

    if server_log:
        sections.append(f"## Server Log (last 30 lines)\n```\n{server_log}\n```")

    if trajectory_text:
        sections.append(f"## Agent Trajectory (last turns)\n{trajectory_text}")
    else:
        sections.append("## Agent Trajectory\nNo trajectory file found.")

    return "\n\n".join(sections)


async def call_judge(user_prompt: str, model: str, client) -> dict:
    """Call the OpenAI API and return parsed category + reasoning."""
    resp = await client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    text = resp.choices[0].message.content
    result = json.loads(text)
    category = result.get("category", "other")
    if category not in CATEGORIES:
        category = "other"
    return {"category": category, "reasoning": result.get("reasoning", "")}


# ── Main pipeline ────────────────────────────────────────────────────────────

def load_existing(output_path: Path) -> set[tuple]:
    """Load already-categorized (agent, model, runtime, task, run_id) keys."""
    if not output_path.exists():
        return set()
    keys = set()
    with open(output_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys.add((
                row["agent"], row["model"], row["runtime"],
                row["task"], row["run_id"],
            ))
    return keys


def find_run_dir(base_dir: Path, runtime: str, agent: str, model: str,
                 task: str, run_id: str) -> Path | None:
    """Locate the run directory on disk (picks most recent date folder)."""
    task_dir = base_dir / runtime / agent / model / task
    if not task_dir.is_dir():
        return None
    date_dirs = sorted(
        [d for d in task_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    if not date_dirs:
        return None
    run_dir = date_dirs[0] / run_id
    return run_dir if run_dir.is_dir() else None


def run_analysis(
    base_dir: Path,
    agent_filter: str | None,
    model_filter: str | None,
    task_filter: str | None,
    output_path: Path,
    judge_model: str,
    max_turns: int,
    dry_run: bool,
    max_concurrency: int = 10,
):
    # Collect all runs via evaluation.py
    all_runs = collect_data(base_dir, agent_filter, model_filter)
    if not all_runs:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    # Filter to failed runs only
    failed_runs = []
    for (agent, model), runs in sorted(all_runs.items()):
        for r in runs:
            if task_filter and task_filter not in r["task"]:
                continue
            if r["succeeded"] == 1:
                continue
            failed_runs.append({"agent": agent, "model": model, **r})

    if not failed_runs:
        print("No failed runs found matching filters.")
        return

    # Resume support
    done_keys = load_existing(output_path)
    pending = [
        r for r in failed_runs
        if (r["agent"], r["model"], r["runtime"], r["task"], r["run_id"])
        not in done_keys
    ]

    print(f"Failed runs: {len(failed_runs)} total, {len(done_keys)} already categorized, "
          f"{len(pending)} to process.")

    if dry_run:
        if pending:
            sample = pending[0]
            run_dir = find_run_dir(
                base_dir, sample["runtime"], sample["agent"],
                sample["model"], sample["task"], sample["run_id"],
            )
            traj = extract_trajectory(run_dir, max_turns) if run_dir else None
            newman_exists, newman_summary = (
                extract_newman_failures(run_dir) if run_dir else (False, "Run dir not found.")
            )
            server_log = extract_server_log(run_dir) if run_dir else None
            prompt = build_user_prompt(
                sample["task"], sample["runtime"], sample,
                traj, newman_exists, newman_summary, server_log,
            )
            print(f"\n{'=' * 80}")
            print(f"DRY RUN — sample prompt for: {sample['agent']}/{sample['model']} "
                  f"{sample['runtime']}/{sample['task']}/{sample['run_id']}")
            print(f"{'=' * 80}")
            print(f"\n[SYSTEM]\n{SYSTEM_PROMPT}\n")
            print(f"[USER]\n{prompt}")
        return

    # Prepare prompts (I/O-bound file reads, done upfront)
    work_items = []
    for r in pending:
        run_dir = find_run_dir(
            base_dir, r["runtime"], r["agent"], r["model"], r["task"], r["run_id"],
        )
        traj = extract_trajectory(run_dir, max_turns) if run_dir else None
        newman_exists, newman_summary = (
            extract_newman_failures(run_dir) if run_dir else (False, "Run dir not found.")
        )
        server_log = extract_server_log(run_dir) if run_dir else None
        prompt = build_user_prompt(
            r["task"], r["runtime"], r,
            traj, newman_exists, newman_summary, server_log,
        )
        work_items.append((r, prompt))

    # Run concurrent judge calls
    asyncio.run(_process_concurrent(
        work_items, output_path, done_keys, judge_model, max_concurrency,
    ))

    # Print summary
    print_summary(output_path, agent_filter, model_filter)


async def _process_concurrent(
    work_items: list[tuple[dict, str]],
    output_path: Path,
    done_keys: set,
    judge_model: str,
    max_concurrency: int,
):
    import openai
    client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    sem = asyncio.Semaphore(max_concurrency)
    completed = 0
    total = len(work_items)
    lock = asyncio.Lock()

    append_mode = output_path.exists() and len(done_keys) > 0
    f = open(output_path, "a" if append_mode else "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if not append_mode:
        writer.writerow(["agent", "model", "runtime", "task", "run_id", "category", "reasoning"])

    async def process_one(r: dict, prompt: str):
        nonlocal completed
        async with sem:
            try:
                result = await call_judge(prompt, judge_model, client)
            except Exception as e:
                print(f"  ERROR on {r['task']}/{r['run_id']}: {e}", file=sys.stderr)
                result = {"category": "other", "reasoning": f"Judge API error: {e}"}

        async with lock:
            completed += 1
            writer.writerow([
                r["agent"], r["model"], r["runtime"], r["task"], r["run_id"],
                result["category"], result["reasoning"],
            ])
            f.flush()
            print(f"  [{completed}/{total}] {r['runtime']}/{r['task']}/{r['run_id']} "
                  f"→ {result['category']}")

    tasks = [process_one(r, prompt) for r, prompt in work_items]
    await asyncio.gather(*tasks)
    f.close()
    await client.close()


def print_summary(output_path: Path, agent_filter: str | None, model_filter: str | None):
    """Read the full CSV and print a category breakdown."""
    if not output_path.exists():
        return

    counts: dict[str, int] = defaultdict(int)
    with open(output_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if agent_filter and row["agent"] != agent_filter:
                continue
            if model_filter and row["model"] != model_filter:
                continue
            counts[row["category"]] += 1

    total = sum(counts.values())
    if total == 0:
        return

    label = ""
    if agent_filter:
        label += agent_filter
    if model_filter:
        label += f" / {model_filter}" if label else model_filter

    print(f"\n=== FAILURE ANALYSIS{': ' + label if label else ''} ===\n")
    print(f"{'Category':<28} {'Count':>5}  {'%':>6}")
    print("\u2500" * 42)
    for cat in CATEGORIES:
        c = counts.get(cat, 0)
        if c > 0:
            print(f"{cat:<28} {c:>5}  {c/total:>5.1%}")
    print("\u2500" * 42)
    print(f"{'Total':<28} {total:>5} {100.0:>5.1f}%")
    print(f"\nCSV exported to: {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-judge failure analysis for API-Bench runs.",
    )
    parser.add_argument("results_dir", type=Path, help="Root results directory (e.g. data/results)")
    parser.add_argument("--agent", help="Filter to a specific agent")
    parser.add_argument("--llm", help="Filter to a specific LLM (model name)")
    parser.add_argument("--task", help="Filter tasks containing this substring")
    parser.add_argument("--output", type=Path, default=Path("data/failure_analysis.csv"),
                        help="Output CSV path (default: data/failure_analysis.csv)")
    parser.add_argument("--judge-model", default="gpt-5.2",
                        help="OpenAI model for the judge (default: gpt-5.2)")
    parser.add_argument("--max-turns", type=int, default=20,
                        help="Last N trajectory turns to include (default: 20)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print one sample prompt without calling the API")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Max concurrent judge API calls (default: 10)")
    args = parser.parse_args()

    if not args.results_dir.is_dir():
        print(f"Error: '{args.results_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    if not args.dry_run and "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY environment variable is required.", file=sys.stderr)
        sys.exit(1)

    run_analysis(
        base_dir=args.results_dir,
        agent_filter=args.agent,
        model_filter=args.llm,
        task_filter=args.task,
        output_path=args.output,
        judge_model=args.judge_model,
        max_turns=args.max_turns,
        dry_run=args.dry_run,
        max_concurrency=args.concurrency,
    )


if __name__ == "__main__":
    load_dotenv()
    main()
