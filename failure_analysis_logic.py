#!/usr/bin/env python3
"""
Fine-grained sub-categorization of "logic_error" failures.

Reads the CSV(s) produced by failure_analysis.py, filters to logic_error rows,
re-analyzes each using an LLM judge with a 6-category taxonomy targeting the
PRIMARY root cause of the logic failure.

Usage:
  python run/failure_analysis_logic.py data/failure_analysis_minimax-m2.5.csv data/failure_analysis_qwen3-coder-next.csv
  python run/failure_analysis_logic.py data/failure_analysis_minimax-m2.5.csv --dry-run
  python run/failure_analysis_logic.py data/failure_analysis_*.csv -o data/logic_subcategories.csv
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
from failure_analysis import (
    extract_trajectory, extract_newman_failures,
    extract_server_log, describe_task, find_run_dir,
)

# ── Logic-error sub-taxonomy ────────────────────────────────────────────────
#
# Derived from manual inspection of 250 logic_error reasoning strings.
# Each failure gets exactly ONE primary root cause.

SUBCATEGORIES = {
    "auth_misconfiguration": (
        "The primary failure is in authentication or authorization: the agent "
        "parses the wrong header format (e.g. 'Bearer' vs 'Token'), applies "
        "auth middleware to public routes, fails to validate/issue tokens "
        "correctly, or returns 401 on endpoints that should be accessible. "
        "Downstream failures (broken queries, null slugs) are cascading effects "
        "of the auth bug."
    ),
    "incorrect_query_logic": (
        "SQL or ORM queries for filtering, pagination, or aggregation are wrong: "
        "missing or incorrect joins, wrong column references, wrong parameter "
        "types (e.g. passing a string where an integer is expected), broken "
        "tag/author/favorited filtering, or incorrect use of DB-specific "
        "operators (e.g. Postgres @> on SQLite, ARRAY.contains on base type, "
        "Django JSONField contains on SQLite). The endpoint exists and auth "
        "works, but the data retrieval or mutation logic produces wrong results "
        "or 500s."
    ),
    "state_propagation_failure": (
        "Resources are created but their identifiers (slugs, IDs, tokens) are "
        "not properly generated, returned, or propagated to dependent "
        "operations. The test suite then hits paths like /api/articles/null, "
        "causing cascading 422/404 failures. Also covers cases where state "
        "mutations don't persist (e.g. favoriting doesn't increment "
        "favoritesCount, following flag not updated)."
    ),
    "database_runtime_error": (
        "The code crashes at runtime due to incorrect database or ORM API "
        "usage: SQLAlchemy DetachedInstanceError, MissingGreenlet, session "
        "reuse across requests, connection pool exhaustion (QueuePool timeout), "
        "asyncpg 'another operation in progress', aiosqlite thread reuse, "
        "Sequelize EagerLoadingError from wrong associations, or missing "
        "tables/columns because migrations were not run. The logic may be "
        "conceptually correct but the DB/ORM API is used incorrectly."
    ),
    "framework_idiosyncrasy": (
        "The failure is caused by a framework-specific behavior the agent "
        "didn't handle correctly: Fastify rejecting empty JSON bodies "
        "(FST_ERR_CTP_EMPTY_JSON_BODY) on favorite/follow endpoints, Django "
        "APPEND_SLASH causing 301 redirects on API paths, Hono adapter passing "
        "Node request where Fetch API Request is expected, incorrect HTTP "
        "method handling returning 405, or incompatible framework configuration."
    ),
    "business_logic_defect": (
        "The endpoint works, auth is fine, queries execute correctly, but the "
        "specific business logic doesn't match the API specification: wrong "
        "validation rules, incorrect status codes, wrong slug generation "
        "algorithm, broken feed semantics, incorrect comment retrieval, wrong "
        "profile following behavior, or other spec-level behavioral mismatch "
        "that doesn't fit the above categories."
    ),
}

SUBCATEGORY_LIST_TEXT = "\n".join(
    f"  - `{name}`: {desc}" for name, desc in SUBCATEGORIES.items()
)

SYSTEM_PROMPT = f"""\
You are an expert software engineering analyst. You are given a failed backend \
API generation run that has already been classified as a "logic_error" — the \
server starts, routes exist, but the behavior is wrong.

Your task is to identify the PRIMARY root cause of the failure and classify it \
into exactly ONE sub-category. When multiple issues co-occur, pick the one that \
is the upstream root cause (e.g. if broken auth causes queries to fail, the \
root cause is auth_misconfiguration, not incorrect_query_logic).

Sub-categories:
{SUBCATEGORY_LIST_TEXT}

Respond with a JSON object and nothing else:
{{"subcategory": "<subcategory_name>", "reasoning": "<1-2 sentence explanation>"}}"""


# ── Main pipeline ────────────────────────────────────────────────────────────

def load_logic_errors(csv_paths: list[Path]) -> list[dict]:
    """Load all logic_error rows from one or more failure analysis CSVs."""
    rows = []
    for p in csv_paths:
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["category"] == "logic_error":
                    rows.append(row)
    return rows


def load_existing(output_path: Path) -> set[tuple]:
    if not output_path.exists():
        return set()
    keys = set()
    with open(output_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            keys.add((row["agent"], row["model"], row["runtime"],
                       row["task"], row["run_id"]))
    return keys


def build_user_prompt(
    task_name: str,
    runtime: str,
    trajectory_text: str | None,
    newman_summary: str,
    server_log: str | None,
    original_reasoning: str,
) -> str:
    sections = []
    sections.append(f"## Task\n{describe_task(task_name, runtime)}")
    sections.append(f"## Original analysis\n{original_reasoning}")
    sections.append(f"## Test Results\n{newman_summary}")
    if server_log:
        sections.append(f"## Server Log (last 30 lines)\n```\n{server_log}\n```")
    if trajectory_text:
        sections.append(f"## Agent Trajectory (last turns)\n{trajectory_text}")
    else:
        sections.append("## Agent Trajectory\nNo trajectory file found.")
    return "\n\n".join(sections)


async def call_judge(user_prompt: str, model: str, client) -> dict:
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
    subcat = result.get("subcategory", "business_logic_defect")
    if subcat not in SUBCATEGORIES:
        subcat = "business_logic_defect"
    return {"subcategory": subcat, "reasoning": result.get("reasoning", "")}


def run_analysis(
    csv_paths: list[Path],
    results_dir: Path,
    output_path: Path,
    judge_model: str,
    max_turns: int,
    dry_run: bool,
    max_concurrency: int = 10,
):
    rows = load_logic_errors(csv_paths)
    if not rows:
        print("No logic_error rows found.", file=sys.stderr)
        sys.exit(1)

    done_keys = load_existing(output_path)
    pending = [
        r for r in rows
        if (r["agent"], r["model"], r["runtime"], r["task"], r["run_id"])
        not in done_keys
    ]

    print(f"Logic errors: {len(rows)} total, {len(done_keys)} already done, "
          f"{len(pending)} to process.")

    if dry_run:
        if pending:
            s = pending[0]
            run_dir = find_run_dir(results_dir, s["runtime"], s["agent"],
                                   s["model"], s["task"], s["run_id"])
            traj = extract_trajectory(run_dir, max_turns) if run_dir else None
            _, newman_summary = (
                extract_newman_failures(run_dir) if run_dir else (False, "N/A")
            )
            server_log = extract_server_log(run_dir) if run_dir else None
            prompt = build_user_prompt(
                s["task"], s["runtime"], traj, newman_summary,
                server_log, s["reasoning"],
            )
            print(f"\n{'=' * 80}")
            print(f"DRY RUN — {s['model']} {s['runtime']}/{s['task']}/{s['run_id']}")
            print(f"{'=' * 80}")
            print(f"\n[SYSTEM]\n{SYSTEM_PROMPT}\n")
            print(f"[USER]\n{prompt}")
        return

    # Prepare prompts upfront
    work_items = []
    for r in pending:
        run_dir = find_run_dir(results_dir, r["runtime"], r["agent"],
                               r["model"], r["task"], r["run_id"])
        traj = extract_trajectory(run_dir, max_turns) if run_dir else None
        _, newman_summary = (
            extract_newman_failures(run_dir) if run_dir else (False, "N/A")
        )
        server_log = extract_server_log(run_dir) if run_dir else None
        prompt = build_user_prompt(
            r["task"], r["runtime"], traj, newman_summary,
            server_log, r["reasoning"],
        )
        work_items.append((r, prompt))

    asyncio.run(_process_concurrent(
        work_items, output_path, done_keys, judge_model, max_concurrency,
    ))
    print_summary(output_path)


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
        writer.writerow(["agent", "model", "runtime", "task", "run_id",
                          "subcategory", "reasoning"])

    async def process_one(r: dict, prompt: str):
        nonlocal completed
        async with sem:
            try:
                result = await call_judge(prompt, judge_model, client)
            except Exception as e:
                print(f"  ERROR on {r['task']}/{r['run_id']}: {e}", file=sys.stderr)
                result = {"subcategory": "business_logic_defect",
                          "reasoning": f"Judge API error: {e}"}

        async with lock:
            completed += 1
            writer.writerow([
                r["agent"], r["model"], r["runtime"], r["task"], r["run_id"],
                result["subcategory"], result["reasoning"],
            ])
            f.flush()
            print(f"  [{completed}/{total}] {r['model']:<20} "
                  f"{r['runtime']}/{r['task']}/{r['run_id']} → {result['subcategory']}")

    tasks = [process_one(r, prompt) for r, prompt in work_items]
    await asyncio.gather(*tasks)
    f.close()
    await client.close()


def print_summary(output_path: Path):
    if not output_path.exists():
        return
    counts: dict[str, int] = defaultdict(int)
    by_model: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with open(output_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            counts[row["subcategory"]] += 1
            by_model[row["model"]][row["subcategory"]] += 1

    total = sum(counts.values())
    if total == 0:
        return

    print(f"\n=== LOGIC ERROR SUB-CATEGORIES (n={total}) ===\n")
    print(f"{'Subcategory':<30} {'Count':>5}  {'%':>6}")
    print("\u2500" * 44)
    for cat in SUBCATEGORIES:
        c = counts.get(cat, 0)
        if c > 0:
            print(f"{cat:<30} {c:>5}  {c/total:>5.1%}")
    print("\u2500" * 44)
    print(f"{'Total':<30} {total:>5} {100.0:>5.1f}%")

    if len(by_model) > 1:
        for model in sorted(by_model):
            mt = sum(by_model[model].values())
            print(f"\n  {model} (n={mt}):")
            for cat in SUBCATEGORIES:
                c = by_model[model].get(cat, 0)
                if c > 0:
                    print(f"    {cat:<28} {c:>4}  ({c/mt*100:>4.1f}%)")

    print(f"\nCSV: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Sub-categorize logic_error failures from failure analysis CSVs.",
    )
    parser.add_argument("csv_paths", nargs="+", type=Path,
                        help="One or more failure_analysis CSVs")
    parser.add_argument("--results-dir", type=Path, default=Path("data/results"),
                        help="Root results directory for trajectory extraction")
    parser.add_argument("-o", "--output", type=Path,
                        default=Path("data/logic_subcategories.csv"),
                        help="Output CSV (default: data/logic_subcategories.csv)")
    parser.add_argument("--judge-model", default="gpt-5.2",
                        help="OpenAI model for the judge (default: gpt-5.2)")
    parser.add_argument("--max-turns", type=int, default=20,
                        help="Last N trajectory turns (default: 20)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print one sample prompt, no API calls")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Max concurrent judge API calls (default: 10)")
    args = parser.parse_args()

    for p in args.csv_paths:
        if not p.exists():
            print(f"Error: '{p}' not found.", file=sys.stderr)
            sys.exit(1)

    if not args.dry_run and "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY required.", file=sys.stderr)
        sys.exit(1)

    run_analysis(
        csv_paths=args.csv_paths,
        results_dir=args.results_dir,
        output_path=args.output,
        judge_model=args.judge_model,
        max_turns=args.max_turns,
        dry_run=args.dry_run,
        max_concurrency=args.concurrency,
    )


if __name__ == "__main__":
    load_dotenv()
    main()
