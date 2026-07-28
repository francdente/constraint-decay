#!/usr/bin/env python3
"""
Compute evaluation statistics for API-Bench results.

Walks data/results/{uv,node}/ and computes stats at three levels:
  1. Per-run: assertion counts and pass/fail for each individual run
  2. Per-task: pass@k, pass^k, assertion stats aggregated across runs
  3. Global: aggregated by framework, runtime, or overall (optionally grouped by constraint count)

Usage:
  python run/evaluation.py data/results
  python run/evaluation.py data/results --agent mini_swe_sdk --model qwen3-coder-next
"""

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from constraint_decay.verifiers import verify_clean_architecture, verify_database, verify_orm

RUNTIMES = {"uv", "node"}


def compute_n_constraints(task_name: str) -> int:
    constraints = task_name.split("-")[2:]
    if len(constraints) == 1 and constraints[0] == "unconstrained":
        constraints = []
    return len(constraints)


def compute_constraint_types(task_name: str) -> set[str]:
    parts = task_name.split("-")[2:]
    if len(parts) == 1 and parts[0] == "unconstrained":
        return set()
    return set(parts)


def compute_framework(task_name: str) -> str:
    return task_name.split("-")[0]


def pass_at_k(k: int, n_executed: int, n_succeeded: int) -> float:
    if n_executed < k:
        return float(n_succeeded > 0)
    denom = math.comb(n_executed, k)
    if denom == 0:
        return 0.0
    return 1.0 - (math.comb(n_executed - n_succeeded, k) / denom)


def pass_power_k(k: int, n_executed: int, n_succeeded: int) -> float:
    if n_executed == 0:
        return 0.0
    return (n_succeeded / n_executed) ** k


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
    Walk the results tree and collect per-run stats.

    Returns: {(agent, model): [run_stat_dict, ...]}
    Each run_stat_dict contains: runtime, task, n_constraints, run_id,
    n_assertions, passed_assertions, failed_assertions, passed_assertions_perc, succeeded
    """
    data: dict[tuple, list[dict]] = defaultdict(list)

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

                    has_arch_constraint = "clean_architecture" in task
                    # Detect DB and ORM constraints from task name
                    task_parts = task.split("-")
                    expected_db = None
                    if "postgres" in task_parts:
                        expected_db = "postgres"
                    elif "sqlite" in task_parts:
                        expected_db = "sqlite"
                    expected_orm = None
                    if "sqlalchemy" in task_parts:
                        expected_orm = "sqlalchemy"
                    elif "sequelize" in task_parts:
                        expected_orm = "sequelize"

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

                        # Constraint compliance via static verifiers
                        patches = list(run_dir.glob("*.patch"))
                        patch_file = patches[0] if patches else None

                        arch_compliant = True
                        arch_reason = ""
                        if has_arch_constraint:
                            if patch_file:
                                vr = verify_clean_architecture(patch_file)
                                arch_compliant = vr.compliant
                                if not vr.compliant:
                                    arch_reason = vr.reason
                            else:
                                arch_compliant = False
                                arch_reason = "No patch file"

                        db_compliant = True
                        db_reason = ""
                        if expected_db is not None:
                            if patch_file:
                                vr = verify_database(patch_file, expected_db)
                                db_compliant = vr.compliant
                                if not vr.compliant:
                                    db_reason = vr.reason
                            else:
                                db_compliant = False
                                db_reason = "No patch file"

                        orm_compliant = True
                        orm_reason = ""
                        if expected_orm is not None:
                            if patch_file:
                                vr = verify_orm(patch_file, expected_orm)
                                orm_compliant = vr.compliant
                                if not vr.compliant:
                                    orm_reason = vr.reason
                            else:
                                orm_compliant = False
                                orm_reason = "No patch file"

                        all_verifiers_pass = arch_compliant and db_compliant and orm_compliant
                        succeeded = 1 if (tests_passed and all_verifiers_pass) else 0

                        # Strict A%: zero out assertion score when verifiers fail
                        strict_perc = perc if all_verifiers_pass else 0.0

                        data[(agent, model)].append(
                            {
                                "runtime": runtime,
                                "task": task,
                                "n_constraints": compute_n_constraints(task),
                                "run_id": run_id,
                                "n_assertions": result["n_assertions"],
                                "passed_assertions": result["passed_assertions"],
                                "failed_assertions": result["failed_assertions"],
                                "passed_assertions_perc": perc,
                                "strict_assertions_perc": strict_perc,
                                "succeeded": succeeded,
                                "tests_passed": int(tests_passed),
                                "arch_compliant": int(arch_compliant),
                                "arch_reason": arch_reason,
                                "db_compliant": int(db_compliant),
                                "db_reason": db_reason,
                                "orm_compliant": int(orm_compliant),
                                "orm_reason": orm_reason,
                            }
                        )

    return data


def compute_task_stats(runs: list[dict]) -> list[dict]:
    """
    Aggregate run-level stats into task-level stats.

    Groups runs by (runtime, task), computes pass@k, pass^k, assertion stats.
    Returns a list of task-level stat dicts.
    """
    by_task: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        by_task[(r["runtime"], r["task"])].append(r)

    task_stats = []
    for (runtime, task), task_runs in sorted(by_task.items()):
        n_executed = len(task_runs)
        n_succeeded = sum(r["succeeded"] for r in task_runs)
        percs = [r["passed_assertions_perc"] for r in task_runs]
        mean_perc = statistics.mean(percs)
        std_perc = statistics.stdev(percs) if len(percs) > 1 else 0.0

        strict_percs = [r["strict_assertions_perc"] for r in task_runs]
        mean_strict = statistics.mean(strict_percs)
        std_strict = statistics.stdev(strict_percs) if len(strict_percs) > 1 else 0.0

        task_stats.append(
            {
                "runtime": runtime,
                "task": task,
                "framework": compute_framework(task),
                "n_constraints": compute_n_constraints(task),
                "constraint_types": compute_constraint_types(task),
                "pass_at_1": pass_at_k(1, n_executed, n_succeeded),
                "pass_at_3": pass_at_k(3, n_executed, n_succeeded),
                "pass_power_3": pass_power_k(3, n_executed, n_succeeded),
                "passed_assertions_perc": mean_perc,
                "passed_assertions_std": std_perc,
                "strict_assertions_perc": mean_strict,
                "strict_assertions_std": std_strict,
                "n_succeeded": n_succeeded,
                "n_executed": n_executed,
            }
        )

    return task_stats


def compute_global_stats(task_stats: list[dict]) -> list[dict]:
    """
    Aggregate task-level stats at framework, runtime, and global granularities,
    both overall and grouped by constraint count.
    """
    results = []

    def aggregate(granularity: str, group: str, tasks: list[dict], n_constraints=None):
        if not tasks:
            return
        row = {
            "granularity": granularity,
            "group": group,
            "n_constraints": n_constraints if n_constraints is not None else "",
            "pass_at_1": statistics.mean(t["pass_at_1"] for t in tasks),
            "pass_at_3": statistics.mean(t["pass_at_3"] for t in tasks),
            "pass_power_3": statistics.mean(t["pass_power_3"] for t in tasks),
            "passed_assertions_perc": statistics.mean(t["passed_assertions_perc"] for t in tasks),
            "strict_assertions_perc": statistics.mean(t["strict_assertions_perc"] for t in tasks),
            "n_succeeded": sum(t["n_succeeded"] for t in tasks),
            "n_executed": sum(t["n_executed"] for t in tasks),
        }
        results.append(row)

    # 1. Framework overall
    by_fw: dict[str, list] = defaultdict(list)
    for t in task_stats:
        by_fw[t["framework"]].append(t)
    for fw in sorted(by_fw):
        aggregate("framework", fw, by_fw[fw])

    # 2. Framework by constraint count
    by_fw_nc: dict[tuple, list] = defaultdict(list)
    for t in task_stats:
        by_fw_nc[(t["framework"], t["n_constraints"])].append(t)
    for (fw, nc) in sorted(by_fw_nc):
        aggregate("framework", fw, by_fw_nc[(fw, nc)], n_constraints=nc)

    # 3. Runtime overall
    by_rt: dict[str, list] = defaultdict(list)
    for t in task_stats:
        by_rt[t["runtime"]].append(t)
    for rt in sorted(by_rt):
        aggregate("runtime", rt, by_rt[rt])

    # 4. Runtime by constraint count
    by_rt_nc: dict[tuple, list] = defaultdict(list)
    for t in task_stats:
        by_rt_nc[(t["runtime"], t["n_constraints"])].append(t)
    for (rt, nc) in sorted(by_rt_nc):
        aggregate("runtime", rt, by_rt_nc[(rt, nc)], n_constraints=nc)

    # 5. Global
    aggregate("global", "all", task_stats)

    # 6. Global by constraint count
    by_nc: dict[int, list] = defaultdict(list)
    for t in task_stats:
        by_nc[t["n_constraints"]].append(t)
    for nc in sorted(by_nc):
        aggregate("global", "all", by_nc[nc], n_constraints=nc)

    # 7. Global by constraint type
    all_ct: set[str] = set()
    for t in task_stats:
        all_ct.update(t["constraint_types"])
    for ct in sorted(all_ct):
        matching = [t for t in task_stats if ct in t["constraint_types"]]
        aggregate("constraint_type", ct, matching)

    # 8. Framework by constraint type
    by_fw_ct: dict[tuple[str, str], list] = defaultdict(list)
    for t in task_stats:
        for ct in t["constraint_types"]:
            by_fw_ct[(t["framework"], ct)].append(t)
    for (fw, ct) in sorted(by_fw_ct):
        aggregate("framework_constraint", f"{fw}:{ct}", by_fw_ct[(fw, ct)])

    return results


def print_run_stats(runs: list[dict], agent: str, model: str):
    print(f"\n{'=' * 120}")
    print(f"RUN-LEVEL STATS: {agent} / {model}")
    print(f"{'=' * 120}")

    w_task = max(40, max((len(r["task"]) for r in runs), default=4))
    header = (
        f"{'Runtime':<6}  {'Task':<{w_task}}  {'#C':>2}  {'Run':<6}  "
        f"{'Total':>5}  {'Pass':>5}  {'Fail':>5}  {'Pass%':>7}  "
        f"{'Tst':>3}  {'Arc':>3}  {'DB':>3}  {'ORM':>3}  {'OK':>2}"
    )
    print(header)
    print("-" * len(header))
    for r in runs:
        print(
            f"{r['runtime']:<6}  {r['task']:<{w_task}}  {r['n_constraints']:>2}  {r['run_id']:<6}  "
            f"{r['n_assertions']:>5}  {r['passed_assertions']:>5}  {r['failed_assertions']:>5}  "
            f"{r['passed_assertions_perc']:>6.1%}  "
            f"{'Y' if r['tests_passed'] else 'N':>3}  "
            f"{'Y' if r['arch_compliant'] else 'N':>3}  "
            f"{'Y' if r['db_compliant'] else 'N':>3}  "
            f"{'Y' if r['orm_compliant'] else 'N':>3}  {r['succeeded']:>2}"
        )


def print_task_stats(tasks: list[dict], agent: str, model: str):
    print(f"\n{'=' * 130}")
    print(f"TASK-LEVEL STATS: {agent} / {model}")
    print(f"{'=' * 130}")

    w_task = max(40, max((len(t["task"]) for t in tasks), default=4))
    header = (
        f"{'Runtime':<6}  {'Task':<{w_task}}  {'#C':>2}  "
        f"{'p@1':>6}  {'p@3':>6}  {'p^3':>6}  "
        f"{'Assert%':>7}  {'±Std':>6}  {'Succ':>4}  {'Exec':>4}"
    )
    print(header)
    print("-" * len(header))
    for t in tasks:
        print(
            f"{t['runtime']:<6}  {t['task']:<{w_task}}  {t['n_constraints']:>2}  "
            f"{t['pass_at_1']:>6.1%}  {t['pass_at_3']:>6.1%}  {t['pass_power_3']:>6.1%}  "
            f"{t['passed_assertions_perc']:>6.1%}  {t['passed_assertions_std']:>5.1%}  "
            f"{t['n_succeeded']:>4}  {t['n_executed']:>4}"
        )


def print_global_stats(global_stats: list[dict], agent: str, model: str):
    print(f"\n{'=' * 110}")
    print(f"GLOBAL STATS: {agent} / {model}")
    print(f"{'=' * 110}")

    header = (
        f"{'Granularity':<22}  {'Group':<28}  {'#C':>4}  "
        f"{'p@1':>6}  {'p@3':>6}  {'p^3':>6}  "
        f"{'Assert%':>7}  {'Succ':>4}  {'Exec':>4}"
    )
    print(header)
    print("-" * len(header))
    prev_gran = None
    for g in global_stats:
        if prev_gran and prev_gran != g["granularity"]:
            print()
        prev_gran = g["granularity"]
        nc_str = str(g["n_constraints"]) if g["n_constraints"] != "" else ""
        print(
            f"{g['granularity']:<22}  {g['group']:<28}  {nc_str:>4}  "
            f"{g['pass_at_1']:>6.1%}  {g['pass_at_3']:>6.1%}  {g['pass_power_3']:>6.1%}  "
            f"{g['passed_assertions_perc']:>6.1%}  {g['n_succeeded']:>4}  {g['n_executed']:>4}"
        )


def compute_decay_stats(global_stats: list[dict]) -> list[dict]:
    """
    Compute step-wise Assert% decay between constraint levels.

    Uses the global-level rows (granularity=global, group=all) at each
    n_constraints=0..3 to compute Δ(L0→L1), Δ(L1→L2), Δ(L2→L3), and Δ(L0→L3).
    Returns a list of dicts with step, from_level, to_level, from_pct, to_pct, delta.
    """
    level_pct: dict[int, float] = {}
    for g in global_stats:
        if g["granularity"] == "global" and g["group"] == "all" and g["n_constraints"] != "":
            level_pct[int(g["n_constraints"])] = g["passed_assertions_perc"]

    results = []
    for lo, hi in [(0, 1), (1, 2), (2, 3), (0, 3)]:
        if lo in level_pct and hi in level_pct:
            results.append({
                "step": f"L{lo}->L{hi}",
                "from_level": lo,
                "to_level": hi,
                "from_pct": level_pct[lo],
                "to_pct": level_pct[hi],
                "delta": level_pct[hi] - level_pct[lo],
            })
    return results


def print_decay_stats(decay_stats: list[dict], agent: str, model: str):
    print(f"\n{'=' * 70}")
    print(f"ASSERT% DECAY: {agent} / {model}")
    print(f"{'=' * 70}")
    header = f"{'Step':<10}  {'From':>7}  {'To':>7}  {'Delta':>8}"
    print(header)
    print("-" * len(header))
    for d in decay_stats:
        print(
            f"{d['step']:<10}  {d['from_pct']:>6.1%}  {d['to_pct']:>6.1%}  "
            f"{d['delta']:>+7.1%}"
        )


def print_summary(all_global_stats: dict, all_decay_stats: dict):
    """Print two compact cross-model summary tables."""

    # ── Helper: extract per-level metrics from global stats ──────────
    def _level_metrics(gstats: list[dict]) -> dict[int, tuple[float, float, float]]:
        """Return {level: (pass_at_1, assert_pct, strict_assert_pct)} from global rows."""
        out = {}
        for g in gstats:
            if g["granularity"] == "global" and g["group"] == "all" and g["n_constraints"] != "":
                lvl = int(g["n_constraints"])
                out[lvl] = (g["pass_at_1"], g["passed_assertions_perc"], g["strict_assertions_perc"])
        return out

    # Collect all (agent, model) pairs, sorted
    keys = sorted(all_global_stats.keys())

    # ── Table 1: pass@1 & A% per constraint level ───────────────────
    print("\n" + "=" * 130)
    print("SUMMARY TABLE 1: pass@1, Assert%, Strict Assert% per constraint level")
    print("=" * 130)
    header = (
        f"{'Agent':<14} {'Model':<22} "
        f"{'L0 p@1':>7} {'L0 A%':>7} {'L0 sA%':>7}  "
        f"{'L1 p@1':>7} {'L1 A%':>7} {'L1 sA%':>7}  "
        f"{'L2 p@1':>7} {'L2 A%':>7} {'L2 sA%':>7}  "
        f"{'L3 p@1':>7} {'L3 A%':>7} {'L3 sA%':>7}"
    )
    print(header)
    print("-" * len(header))
    for agent, model in keys:
        lm = _level_metrics(all_global_stats[(agent, model)])
        parts = [f"{agent:<14} {model:<22}"]
        for lvl in range(4):
            if lvl in lm:
                p1, ap, sap = lm[lvl]
                parts.append(f"{p1*100:>6.1f}  {ap*100:>6.1f}  {sap*100:>6.1f}")
            else:
                parts.append(f"{'--':>6}  {'--':>6}  {'--':>6}")
        print("  ".join(parts))

    # ── Table 2: step-wise A% deltas ────────────────────────────────
    print("\n" + "=" * 90)
    print("SUMMARY TABLE 2: step-wise Assert% change (pp)")
    print("=" * 90)
    header2 = (
        f"{'Agent':<14} {'Model':<22} "
        f"{'L0->L1':>8} {'L1->L2':>8} {'L2->L3':>8} {'L0->L3':>8}"
    )
    print(header2)
    print("-" * len(header2))
    for agent, model in keys:
        dstats = all_decay_stats.get((agent, model), [])
        deltas = {d["step"]: d["delta"] for d in dstats}
        cols = []
        for step in ["L0->L1", "L1->L2", "L2->L3", "L0->L3"]:
            if step in deltas:
                cols.append(f"{deltas[step]*100:>+7.1f}")
            else:
                cols.append(f"{'--':>7}")
        print(f"{agent:<14} {model:<22} {'  '.join(cols)}")

    # ── Averages ────────────────────────────────────────────────────
    step_names = ["L0->L1", "L1->L2", "L2->L3", "L0->L3"]
    step_vals: dict[str, list[float]] = {s: [] for s in step_names}
    for (agent, model), dstats in all_decay_stats.items():
        deltas = {d["step"]: d["delta"] for d in dstats}
        for s in step_names:
            if s in deltas:
                step_vals[s].append(deltas[s])
    avg_cols = []
    for s in step_names:
        if step_vals[s]:
            avg_cols.append(f"{statistics.mean(step_vals[s])*100:>+7.1f}")
        else:
            avg_cols.append(f"{'--':>7}")
    print("-" * len(header2))
    print(f"{'':14} {'AVERAGE':<22} {'  '.join(avg_cols)}")


def compute_verifier_confusion(all_runs: dict) -> dict:
    """
    Compute a confusion matrix: tests_passed × verifier_compliant across ALL runs.

    Only considers constrained runs (n_constraints > 0) for the verifier side,
    since unconstrained runs have no structural constraints to violate.

    Returns a dict with:
      total_runs, test_pass, test_fail,
      test_pass_verifier_pass, test_pass_verifier_fail,
      test_fail_verifier_pass, test_fail_verifier_fail,
      false_reject_rate (% of test-passing runs rejected by verifiers),
      details (list of individual false-reject runs)
    """
    total = 0
    test_pass = 0
    test_fail = 0
    tp_vp = 0  # tests pass, verifier pass
    tp_vf = 0  # tests pass, verifier fail (false reject)
    tf_vp = 0  # tests fail, verifier pass
    tf_vf = 0  # tests fail, verifier fail
    details = []

    for (agent, model), runs in all_runs.items():
        for r in runs:
            total += 1
            t_pass = bool(r["tests_passed"])
            # Verifier compliant = all applicable verifiers pass
            v_pass = bool(r["arch_compliant"] and r["db_compliant"] and r["orm_compliant"])

            if t_pass:
                test_pass += 1
            else:
                test_fail += 1

            # Only count verifier matrix for constrained runs
            if r["n_constraints"] > 0:
                if t_pass and v_pass:
                    tp_vp += 1
                elif t_pass and not v_pass:
                    tp_vf += 1
                    reasons = []
                    if not r["arch_compliant"]:
                        reasons.append(f"arch: {r['arch_reason']}")
                    if not r["db_compliant"]:
                        reasons.append(f"db: {r['db_reason']}")
                    if not r["orm_compliant"]:
                        reasons.append(f"orm: {r['orm_reason']}")
                    details.append({
                        "agent": agent, "model": model,
                        "task": r["task"], "run_id": r["run_id"],
                        "reasons": "; ".join(reasons),
                    })
                elif not t_pass and v_pass:
                    tf_vp += 1
                else:
                    tf_vf += 1

    constrained_test_pass = tp_vp + tp_vf
    false_reject_rate = tp_vf / constrained_test_pass if constrained_test_pass > 0 else 0.0

    return {
        "total_runs": total,
        "test_pass": test_pass,
        "test_fail": test_fail,
        "constrained_test_pass": constrained_test_pass,
        "test_pass_verifier_pass": tp_vp,
        "test_pass_verifier_fail": tp_vf,
        "test_fail_verifier_pass": tf_vp,
        "test_fail_verifier_fail": tf_vf,
        "false_reject_rate": false_reject_rate,
        "details": details,
    }


def print_verifier_confusion(cm: dict):
    print(f"\n{'=' * 80}")
    print("VERIFIER CONFUSION MATRIX (constrained runs only)")
    print(f"{'=' * 80}")
    print(f"Total runs: {cm['total_runs']}  |  Tests pass: {cm['test_pass']}  |  Tests fail: {cm['test_fail']}")
    print()
    print(f"{'':>30}  {'Verifier PASS':>14}  {'Verifier FAIL':>14}")
    print(f"  {'Tests PASS':>28}  {cm['test_pass_verifier_pass']:>14}  {cm['test_pass_verifier_fail']:>14}")
    print(f"  {'Tests FAIL':>28}  {cm['test_fail_verifier_pass']:>14}  {cm['test_fail_verifier_fail']:>14}")
    print()
    print(f"False-reject rate (test-passing runs rejected by verifier): "
          f"{cm['test_pass_verifier_fail']}/{cm['constrained_test_pass']} "
          f"= {cm['false_reject_rate']:.1%}")

    if cm["details"]:
        print(f"\nFalse-reject details ({len(cm['details'])} runs):")
        for d in cm["details"]:
            print(f"  {d['agent']}/{d['model']}  {d['task']}  {d['run_id']}  — {d['reasons']}")


def print_violations(runs: list[dict], agent: str, model: str):
    arch_violations = [r for r in runs if not r["arch_compliant"]]
    db_violations = [r for r in runs if not r["db_compliant"]]
    orm_violations = [r for r in runs if not r["orm_compliant"]]

    if not arch_violations and not db_violations and not orm_violations:
        return

    print(f"\n{'=' * 120}")
    print(f"CONSTRAINT VIOLATIONS: {agent} / {model}")
    print(f"{'=' * 120}")

    if arch_violations:
        print(f"\n  Architecture ({len(arch_violations)} runs):")
        for r in arch_violations:
            print(f"    {r['runtime']:<5}  {r['task']}  {r['run_id']}  — {r['arch_reason']}")

    if db_violations:
        print(f"\n  Database ({len(db_violations)} runs):")
        for r in db_violations:
            print(f"    {r['runtime']:<5}  {r['task']}  {r['run_id']}  — {r['db_reason']}")

    if orm_violations:
        print(f"\n  ORM ({len(orm_violations)} runs):")
        for r in orm_violations:
            print(f"    {r['runtime']:<5}  {r['task']}  {r['run_id']}  — {r['orm_reason']}")


def export_csvs(
    all_runs: dict,
    all_task_stats: dict,
    all_global_stats: dict,
    all_decay_stats: dict,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run-level CSV
    run_csv = output_dir / "run_stats.csv"
    with open(run_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "agent", "model", "runtime", "task", "n_constraints", "run_id",
                "n_assertions", "passed_assertions", "failed_assertions",
                "passed_assertions_perc", "strict_assertions_perc",
                "tests_passed", "arch_compliant", "db_compliant", "orm_compliant", "succeeded",
            ]
        )
        for (agent, model), runs in sorted(all_runs.items()):
            for r in runs:
                writer.writerow(
                    [
                        agent, model, r["runtime"], r["task"], r["n_constraints"],
                        r["run_id"], r["n_assertions"], r["passed_assertions"],
                        r["failed_assertions"], f"{r['passed_assertions_perc']:.4f}",
                        f"{r['strict_assertions_perc']:.4f}",
                        r["tests_passed"], r["arch_compliant"], r["db_compliant"], r["orm_compliant"], r["succeeded"],
                    ]
                )
    print(f"\nRun-level CSV:  {run_csv}")

    # Task-level CSV
    task_csv = output_dir / "task_stats.csv"
    with open(task_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "agent", "model", "runtime", "task", "framework", "n_constraints",
                "pass_at_1", "pass_at_3", "pass_power_3",
                "passed_assertions_perc", "passed_assertions_std",
                "strict_assertions_perc", "strict_assertions_std",
                "n_succeeded", "n_executed",
            ]
        )
        for (agent, model), tasks in sorted(all_task_stats.items()):
            for t in tasks:
                writer.writerow(
                    [
                        agent, model, t["runtime"], t["task"], t["framework"],
                        t["n_constraints"],
                        f"{t['pass_at_1']:.4f}", f"{t['pass_at_3']:.4f}",
                        f"{t['pass_power_3']:.4f}",
                        f"{t['passed_assertions_perc']:.4f}",
                        f"{t['passed_assertions_std']:.4f}",
                        f"{t['strict_assertions_perc']:.4f}",
                        f"{t['strict_assertions_std']:.4f}",
                        t["n_succeeded"], t["n_executed"],
                    ]
                )
    print(f"Task-level CSV: {task_csv}")

    # Global-level CSV
    global_csv = output_dir / "global_stats.csv"
    with open(global_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "agent", "model", "granularity", "group", "n_constraints",
                "pass_at_1", "pass_at_3", "pass_power_3",
                "passed_assertions_perc", "strict_assertions_perc",
                "n_succeeded", "n_executed",
            ]
        )
        for (agent, model), gstats in sorted(all_global_stats.items()):
            for g in gstats:
                writer.writerow(
                    [
                        agent, model, g["granularity"], g["group"],
                        g["n_constraints"],
                        f"{g['pass_at_1']:.4f}", f"{g['pass_at_3']:.4f}",
                        f"{g['pass_power_3']:.4f}",
                        f"{g['passed_assertions_perc']:.4f}",
                        f"{g['strict_assertions_perc']:.4f}",
                        g["n_succeeded"], g["n_executed"],
                    ]
                )
    print(f"Global CSV:     {global_csv}")

    # Decay stats CSV
    decay_csv = output_dir / "decay_stats.csv"
    with open(decay_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["agent", "model", "step", "from_level", "to_level",
             "from_assert_pct", "to_assert_pct", "delta_assert_pct"]
        )
        for (agent, model), dstats in sorted(all_decay_stats.items()):
            for d in dstats:
                writer.writerow(
                    [
                        agent, model, d["step"], d["from_level"], d["to_level"],
                        f"{d['from_pct']:.4f}", f"{d['to_pct']:.4f}",
                        f"{d['delta']:.4f}",
                    ]
                )
    print(f"Decay CSV:      {decay_csv}")


def main():
    parser = argparse.ArgumentParser(description="Compute evaluation statistics for API-Bench.")
    parser.add_argument("base_dir", help="Root results directory (e.g. data/results)")
    parser.add_argument("--agent", help="Filter to a specific agent")
    parser.add_argument("--model", help="Filter to a specific model")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for CSV output (default: base_dir/../)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print run-level details")
    parser.add_argument("--summary", action="store_true",
                        help="Print compact cross-model summary tables only")
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
    all_decay_stats = {}

    for (agent, model), runs in sorted(all_runs.items()):
        if args.verbose:
            print_run_stats(runs, agent, model)

        task_stats = compute_task_stats(runs)
        all_task_stats[(agent, model)] = task_stats

        global_stats = compute_global_stats(task_stats)
        all_global_stats[(agent, model)] = global_stats

        decay_stats = compute_decay_stats(global_stats)
        all_decay_stats[(agent, model)] = decay_stats

        if not args.summary:
            print_task_stats(task_stats, agent, model)
            print_global_stats(global_stats, agent, model)
            print_decay_stats(decay_stats, agent, model)
            print_violations(runs, agent, model)

    if args.summary:
        print_summary(all_global_stats, all_decay_stats)

    # Verifier confusion matrix (across all runs)
    cm = compute_verifier_confusion(all_runs)
    print_verifier_confusion(cm)

    export_csvs(all_runs, all_task_stats, all_global_stats, all_decay_stats, output_dir)


if __name__ == "__main__":
    main()
