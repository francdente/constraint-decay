import argparse
import json
import os
import time
import uuid
from pathlib import Path

from dataclass_wizard import fromdict

import constraint_decay.dockerized_cmd as cmd
from constraint_decay.model import Task


SERVER_LOG_PATH = "/tmp/server.log"


def _save_logs(docker_folder: str, results_dir: Path, project_name: str | None = None) -> None:
    # Server process logs (stdout/stderr from npm run dev / uvicorn / etc.)
    server_logs = cmd.docker_compose_read_file(
        docker_folder,
        "conduit",
        SERVER_LOG_PATH,
        project_name=project_name,
    )
    logs_path = results_dir / "conduit.log"
    logs_path.write_text(server_logs, encoding="utf-8")
    print(f"Conduit logs saved to {logs_path}")


def _dump_server_log(docker_folder: str, results_dir: Path, project_name: str | None = None) -> None:
    """Read the server log from the container, print tail to stdout, and save to disk."""
    server_log = cmd.docker_compose_read_file(docker_folder, "conduit", SERVER_LOG_PATH, project_name=project_name)
    print("── Server log ──")
    print(server_log[-3000:] if len(server_log) > 3000 else server_log)
    print("── End server log ──")
    _save_logs(docker_folder, results_dir, project_name=project_name)


def _wait_conduit(docker_folder: str, task: Task, results_dir: Path, project_name: str | None = None) -> bool:
    """Wait for conduit to become healthy. Returns True if healthy, False otherwise."""
    repo_metadata = task.repo_metadata
    curl_cmd = f"curl -sf --max-time 15 http://conduit:{repo_metadata.port}{repo_metadata.healthcheck_endpoint}"

    for attempt in range(5):
        result = cmd.docker_compose_exec(docker_folder, "conduit", curl_cmd, project_name=project_name)
        if result.returncode == 0:
            return True

        # Check if the server process is still alive
        check = cmd.docker_compose_exec(
            docker_folder,
            "conduit",
            "pgrep -f 'node\\|python\\|uvicorn\\|gunicorn\\|flask'",
            capture_output=True,
            project_name=project_name,
        )
        if check.returncode != 0 and attempt >= 1:
            # Server process died — dump logs and stop waiting
            print(f"Server process crashed (attempt {attempt + 1})")
            _dump_server_log(docker_folder, results_dir, project_name=project_name)
            return False

        time.sleep(5)

    # Exhausted all retries
    print("Server never became healthy after 120s")
    _dump_server_log(docker_folder, results_dir, project_name=project_name)
    return False


def main(task_absolute_path: str, agent_patch_absolute_path: str) -> None:
    task_path = Path(task_absolute_path)
    task: Task

    with open(task_path) as file:
        json_file = json.load(file)
        task = fromdict(Task, json_file)

    docker_folder = os.path.join(os.getcwd(), "runtime", task.docker_folder)
    agent_patch_path = Path(agent_patch_absolute_path)
    results_dir = agent_patch_path.parent
    project_name = f"{task.docker_folder}-eval-{task.name}-{uuid.uuid4().hex[:8]}" 

    # Build the images
    cmd.docker_compose_build(docker_folder, "conduit", project_name=project_name)
    cmd.docker_compose_build(docker_folder, "newman", project_name=project_name)

    volumes = [f"{results_dir}:/results"]
    newman_container_id = cmd.docker_compose_run(docker_folder, "newman", volumes, project_name=project_name)

    if task.patch is not None:
        cmd.docker_compose_copy(
            docker_folder,
            "conduit",
            str(task_path.parent),
            "/tasks",
            project_name=project_name,
        )
        cmd.docker_git_apply_patch(docker_folder, "conduit", f"/tasks/{task.patch}", project_name=project_name)

    cmd.docker_compose_copy(
        docker_folder,
        "conduit",
        str(agent_patch_path.parent),
        "/patch",
        project_name=project_name,
    )

    # Generate a gitkeep file in the repository
    # This prevents git apply patch failures in case the agent deleted the filed in the generation task
    gitkeep_cmd = "echo $'' >> gitkeep"
    cmd.docker_compose_exec(
        docker_folder,
        "conduit",
        gitkeep_cmd,
        project_name=project_name,
    )

    cmd.docker_git_apply_patch(
        docker_folder,
        "conduit",
        f"/patch/{agent_patch_path.name}",
        project_name=project_name,
    )

    setup_cmds = " && ".join(task.repo_metadata.setup_cmds)
    setup_result = cmd.docker_compose_exec(
        docker_folder,
        "conduit",
        setup_cmds,
        capture_output=True,
        project_name=project_name,
    )

    # Get setup logs
    setup_log = setup_result.stdout.decode(
        errors="replace"
    ) + setup_result.stderr.decode(errors="replace")
    (results_dir / "setup.log").write_text(setup_log, encoding="utf-8")
    if setup_result.returncode != 0:
        print(f"WARNING: Setup commands failed (exit {setup_result.returncode})")
        print(setup_log[-2000:] if len(setup_log) > 2000 else setup_log)

    # Start the server in background, redirecting output to a log file
    run_cmds = " && ".join(task.repo_metadata.run_cmds)
    run_cmds_logged = f"({run_cmds}) > {SERVER_LOG_PATH} 2>&1"
    cmd.docker_compose_exec(docker_folder, "conduit", run_cmds_logged, flags=["-d"], project_name=project_name)

    # Wait for server to become healthy
    server_healthy = _wait_conduit(docker_folder, task, results_dir, project_name=project_name)

    # Execute newman (even if server is unhealthy, to get a 0-pass baseline result)
    if not server_healthy:
        print("Running Newman against unhealthy server (expecting failures)...")

    newman_vars = f"--global-var APIURL=http://conduit:{task.repo_metadata.port}{task.repo_metadata.base_url} --global-var USERNAME=user --global-var EMAIL=user@mail.com --global-var PASSWORD=password"
    newman_reporters = "--reporters cli,html,csv --reporter-html-export /results --reporter-csv-export /results"
    collection = task.postman_collection
    newman_cmd = f"newman run --timeout-request 10000 {newman_vars} {newman_reporters} /newman/{collection}"
    cmd.docker_compose_exec(docker_folder, "newman", newman_cmd, project_name=project_name)

    # Always save final conduit logs (captures full server lifecycle)
    _save_logs(docker_folder, results_dir, project_name=project_name)

    # Cleanup
    cmd.docker_stop_rm(newman_container_id)
    cmd.docker_compose_down(docker_folder, project_name=project_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-absolute-path", required=True)
    parser.add_argument("--agent-patch-absolute-path", required=True)

    args = parser.parse_args()
    task_absolute_path = args.task_absolute_path
    agent_patch_absolute_path = args.agent_patch_absolute_path

    main(task_absolute_path, agent_patch_absolute_path)
