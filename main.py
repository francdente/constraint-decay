import argparse
import datetime
import json
import os
from pathlib import Path
import logging

from dataclass_wizard import fromdict
from dotenv import load_dotenv

from constraint_decay.model import Task
from evaluate import main as evaluate_main
from run_agent import main as run_agent_main

logger = logging.getLogger(__name__)
def _as_string(task_cli: str | list[str]) -> str | None:
    if isinstance(task_cli, str):
        return task_cli
    return task_cli[0] if len(task_cli) == 1 else None


def _get_tasks_absolute_paths(
    task_cli: str | list[str],
    difficulty: str | None = None,
    filters: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[Path]:
    paths: list[Path] = []
    task_absolute_path = Path(os.environ["TASKS_ABSOLUTE_PATH"])

    task_str = _as_string(task_cli)
    if task_str:
        if task_str == "all":
            for runtime in sorted(os.listdir(task_absolute_path)):
                runtime_path = task_absolute_path / runtime
                if not runtime_path.is_dir():
                    continue
                for task in sorted(os.listdir(runtime_path)):
                    if not task.endswith(".json"):
                        continue
                    paths.append(runtime_path / task)
        elif (task_absolute_path / task_str).is_dir():
            runtime_path = task_absolute_path / task_str
            for task in sorted(os.listdir(runtime_path)):
                if not task.endswith(".json"):
                    continue
                paths.append(runtime_path / task)
        else:
            paths.append(task_absolute_path / task_str)
    else:
        for task in task_cli:
            task_name = task if ".json" in task else f"{task}.json"
            paths.append(task_absolute_path / task_name)

    if difficulty:
        paths = [p for p in paths if f"-{difficulty}-" in p.name]
    if filters:
        paths = [p for p in paths if any(f in p.stem for f in filters)]
    if exclude:
        paths = [p for p in paths if not any(e in p.stem for e in exclude)]
    return paths


def main(
    task_cli: str | list[str],
    agent: str,
    run_only: bool,
    evaluate: bool,
    difficulty: str | None = None,
    filters: list[str] | None = None,
    exclude: list[str] | None = None,
    runs: int = 1,
) -> None:
    results_path = Path(os.environ["RESULTS_ABSOLUTE_PATH"])
    llm_api_key = os.environ["LLM_API_KEY"]
    llm_model = os.environ["LLM_MODEL"]
    llm_base_url = os.getenv("LLM_BASE_URL")

    # Bridge LLM_BASE_URL → OPENAI_API_BASE so litellm uses the custom
    # endpoint for openai/ prefixed models (e.g. self-hosted vLLM).
    if llm_base_url:
        os.environ.setdefault("OPENAI_API_BASE", llm_base_url)
        os.environ.setdefault("OPENAI_API_KEY", llm_api_key)

    task_paths = _get_tasks_absolute_paths(task_cli, difficulty, filters, exclude)

    exec_idx = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    for task in task_paths:
        if not evaluate:
            print("Running agent to solve the tasks")
            for run_idx in range(runs):
                print(f"\n{'=' * 60}")
                print(f"Run {run_idx + 1}/{runs}")
                print(f"{'=' * 60}")
                run_agent_main(
                    task_absolute_path=str(task),
                    results_absolute_path=str(results_path),
                    agent_file=f"{agent}.py",
                    llm_api_key=llm_api_key,
                    llm_model=llm_model,
                    llm_base_url=llm_base_url,
                    exec_idx=exec_idx,
                    run_index=run_idx,
                )

        if not run_only:
            for run_idx in range(runs):
                task_instance: Task
                with open(task) as file:
                    json_file = json.load(file)
                    task_instance = fromdict(Task, json_file)

                runtime = task.parts[-2]
                llm_name = llm_model.split("/")[-1]
                patch_name = f"{agent}-{task_instance.name}.patch"
                if evaluate:
                    path = Path(
                        results_path / runtime / agent / llm_name / task_instance.name
                    )
                    if not path.exists():
                        logger.warning("No results for this task, skipping to next task")
                        continue
                    files = sorted(os.listdir(path))
                    if len(files) == 0:
                        logger.warning("No files in task results, skipping to next task")
                        continue
                    exec_idx = files[-1]

                patch_path = (
                    results_path
                    / runtime
                    / agent
                    / llm_name
                    / task_instance.name
                    / exec_idx
                    / f"run_{run_idx}"
                    / patch_name
                )

                if not patch_path.exists():
                    logger.warning("No patch generated for this task !")
                    return

                evaluate_main(
                    task_absolute_path=str(task),
                    agent_patch_absolute_path=str(patch_path),
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--task", required=True, action="append")
    parser.add_argument(
        "--run-only",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--evaluate",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--difficulty",
        choices=["simple", "intermediate", "hard"],
        default=None,
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=None,
        help="Keep only tasks whose name contains this substring (repeatable)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        help="Exclude tasks whose name contains this substring (repeatable)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs per task for pass@k (default: 1)",
    )

    args = parser.parse_args()
    task = args.task
    agent = args.agent
    run_only = args.run_only
    evaluate = args.evaluate
    difficulty = args.difficulty
    filters = args.filter
    exclude = args.exclude
    runs = args.runs

    load_dotenv()

    main(task, agent, run_only, evaluate, difficulty, filters, exclude, runs)
