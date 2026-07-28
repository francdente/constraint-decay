import argparse
import datetime
import json
import os
import uuid
from pathlib import Path

from dataclass_wizard import fromdict

import constraint_decay.dockerized_cmd as cmd
from constraint_decay.model import Task

AGENT_PYTHON_MAP = {
    "mock_agent": "python3",
    "openhands_sdk": "/openhands-env/bin/python",
    "mini_swe_sdk": "/mini-env/bin/python",
    "live_swe_sdk": "/mini-env/bin/python",
}

UNCONSTRAINED_RUNTIMES = {"uv", "node"}

DIFF_EXCLUDE = [
    "node_modules",
    "__pycache__",
    "*.pyc",
    ".venv",
    "venv",
    "*.egg-info",
    ".mypy_cache",
    "dist",
    "build",
    "*.so",
    "*.db",
    "*.sqlite",
    "*.sqlite3",
]

INSTRUCTIONS_PROMPT = """You will work as an expert REST API developer."""


def main(
    task_absolute_path: str,
    agent_file: str,
    results_absolute_path: str,
    llm_api_key: str,
    llm_model: str,
    exec_idx: str,
    run_index: int,
    llm_base_url: str | None,
) -> None:
    task_path = Path(task_absolute_path)
    task: Task

    with open(task_path) as file:
        json_file = json.load(file)
        task = fromdict(Task, json_file)

    docker_folder = os.path.join(os.getcwd(), "runtime", task.docker_folder)
    agents_folder = os.path.join(os.getcwd(), "runtime", "agents")
    project_name = f"{task.docker_folder}-{task.name}-{uuid.uuid4().hex[:8]}"

    llm_name = llm_model.split("/")[-1]

    # Results folder
    agent_name = agent_file.removesuffix(".py")
    results_docker = Path(results_absolute_path) / task.docker_folder
    results_docker.mkdir(exist_ok=True)
    agent_path = results_docker / agent_name
    agent_path.mkdir(exist_ok=True)
    llm_path = agent_path / llm_name
    llm_path.mkdir(exist_ok=True)
    results_path = llm_path / task.name
    results_path.mkdir(exist_ok=True)
    results_path = results_path / exec_idx
    results_path.mkdir(exist_ok=True)
    results_path = results_path / f"run_{run_index}"
    results_path.mkdir(exist_ok=True)

    # Build the images
    cmd.docker_compose_build(docker_folder, "conduit", project_name=project_name)

    # Run the conduit service for agent execution
    volumes = [f"{agents_folder}:/agents", f"{results_path}:/trajectories"]
    container_id = cmd.docker_compose_run(
        docker_folder, "conduit", volumes, project_name=project_name
    )

    # Copy task
    cmd.docker_copy(docker_folder, container_id, str(task_path.parent), "/tasks")

    # Apply patch (if existing)
    if task.patch:
        patch_file = f"/tasks/{task.patch}"
        cmd.docker_git_apply_patch(
            docker_folder, "conduit", patch_file, project_name=project_name
        )

    # Create INSTRUCTIONS.md file
    instructions_cmd = f"echo $'' >> gitkeep"
    cmd.docker_compose_exec(
        docker_folder, "conduit", instructions_cmd, project_name=project_name
    )

    # Reinit git repository
    cmd.docker_git_reinit(docker_folder, "conduit", project_name=project_name)
    base_sha = cmd.docker_git_rev_parse(
        docker_folder, "conduit", "HEAD", project_name=project_name
    )

    # Remove tasks folder
    cmd.docker_compose_exec(
        docker_folder, "conduit", "rm -rf /tasks", project_name=project_name
    )

    # Setup the repository (e.g. run migrations)
    # TODO: move the unconstrained property to task level?
    if task.docker_folder not in UNCONSTRAINED_RUNTIMES:
        setup_cmd = " && ".join(task.repo_metadata.setup_cmds)
        cmd.docker_compose_exec(
            docker_folder, "conduit", setup_cmd, project_name=project_name
        )

    # Run
    run_cmds = " && ".join(task.repo_metadata.run_cmds)

    env = dict()
    env["LLM_API_KEY"] = llm_api_key
    env["LLM_MODEL"] = llm_model
    env["PROMPT"] = f"{task.prompt}\n\n{INSTRUCTIONS_PROMPT}"

    # Ignore cost errors in mini-swe if model is unmapped
    env["MSWEA_COST_TRACKING"] = "ignore_errors"
    # Custom model cost registry (mounted via /agents volume)
    env["LITELLM_MODEL_REGISTRY_PATH"] = "/agents/model_costs.json"
    # Define how to start the server at system level
    env["SYSTEM_GUIDANCE"] = (
        f"## Server Commands\n\nTo start the server, run:\n```bash\n{run_cmds}\n```"
    )

    if llm_base_url:
        docker_base_url = llm_base_url.replace(
            "://localhost",
            "://host.docker.internal",
        )
        env["LLM_BASE_URL"] = docker_base_url

    native_tc = os.getenv("NATIVE_TOOL_CALLING")
    if native_tc:
        env["NATIVE_TOOL_CALLING"] = native_tc

    if agent_name not in AGENT_PYTHON_MAP:
        raise Exception("Unexistent agent python environment!")

    agent_python = AGENT_PYTHON_MAP[agent_name]
    agent_cmd = f"{agent_python} /agents/{agent_file}"
    cmd.docker_compose_exec(
        docker_folder, "conduit", agent_cmd, env=env, project_name=project_name
    )
    agent_patch = cmd.docker_git_diff(
        docker_folder, "conduit", base_sha, DIFF_EXCLUDE, project_name=project_name
    )

    # Save patch
    patch_filename = f"{agent_name}-{task.name}.patch"
    patch_path = results_path / patch_filename
    patch_path.write_text(agent_patch)

    # Cleanup: stop the run container first, then tear down the project
    cmd.docker_stop_rm(container_id)
    cmd.docker_compose_down(docker_folder, project_name=project_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-absolute-path", required=True)
    parser.add_argument("--agent-file", required=True)
    parser.add_argument("--results-absolute-path", required=True)
    parser.add_argument("--llm-api-key", required=True)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--llm-base_url", required=False)
    parser.add_argument("--llm-base_url", required=False)
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs per task for pass@k (default: 1)",
    )

    args = parser.parse_args()
    task_absolute_path = args.task_absolute_path
    agent_file = args.agent_file
    results_absolute_path = args.results_absolute_path
    llm_api_key = args.llm_api_key
    llm_model = args.llm_model
    llm_base_url = args.llm_base_url
    runs = args.runs

    exec_idx = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    main(
        task_absolute_path,
        agent_file,
        results_absolute_path,
        llm_api_key,
        llm_model,
        exec_idx,
        runs,
        llm_base_url,
    )
