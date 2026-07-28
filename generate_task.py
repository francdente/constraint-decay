import argparse
import json
import re
import subprocess
from pathlib import Path

from dataclass_wizard import asdict

from constraint_decay.model import RepoMetadata, Task, TaskMetadata

REPO_METADATA_AIOHTTP: RepoMetadata = RepoMetadata(
    repo_url="https://github.com/stkrizh/realworld-aiohttp.git",
    repo_commit="00cc7902e4c2a88c2f72c8793c20c7713c10faa5",
    setup_cmds=["alembic --config migrations/alembic.ini upgrade head"],
    run_cmds=["python -m conduit"],
    port=8080,
    base_url="/api/v1",
    healthcheck_endpoint="/api/v1/healthcheck",
)

REPO_METADATA_EXPRESS: RepoMetadata = RepoMetadata(
    repo_url="https://github.com/SeuRonao/realworld-express-prisma.git",
    repo_commit="1f5a21ed6eec656410655cab9d5d3c7148b57a93",
    setup_cmds=["npm run migrate:develop"],
    run_cmds=["npm run develop"],
    port=3000,
    base_url="/api",
    healthcheck_endpoint="/",
)

REPO_METADATA_FASTAPI: RepoMetadata = RepoMetadata(
    repo_url="https://github.com/borys25ol/fastapi-realworld-backend.git",
    repo_commit="55111c6b335455734c139a2245d98be8288be528",
    setup_cmds=["alembic upgrade head"],
    run_cmds=["uvicorn conduit.app:app --host 0.0.0.0 --port 8080"],
    port=8080,
    base_url="/api",
    healthcheck_endpoint="/api/health-check",
)

REPO_METADATA_FLASK: RepoMetadata = RepoMetadata(
    repo_url="https://github.com/mhegelheimer/realworld-flask.git",
    repo_commit="c7d865f89464a1770171492b036b56c87cc76c8a",
    setup_cmds=["alembic upgrade head"],
    run_cmds=["poetry run flask run --host=0.0.0.0"],
    port=8080,
    base_url="/api",
    healthcheck_endpoint="/api/ping",
)

REPO_METADATA_GIN: RepoMetadata = RepoMetadata(
    repo_url="https://github.com/gothinkster/golang-gin-realworld-example-app.git",
    repo_commit="626c372d259472148d93303f74aa9b9a1cdcef24",
    setup_cmds=[],
    run_cmds=["go run hello.go"],
    port=8080,
    base_url="/api",
    healthcheck_endpoint="/api/ping",
)

REPO_METADATA_HONO: RepoMetadata = RepoMetadata(
    repo_url="https://github.com/cjfff/realworld-honojs.git",
    repo_commit="298efda2b78f3cb7331ef32495ad2a113b8ea936",
    setup_cmds=["npm run prisma:generate", "npm run prisma:migrate"],
    run_cmds=["npm run dev"],
    port=3000,
    base_url="/api",
    healthcheck_endpoint="/",
)

REPO_METADATA_UV: RepoMetadata = RepoMetadata(
    repo_url=None,
    repo_commit=None,
    setup_cmds=["uv pip install --system -r requirements.txt", "chmod +x run.sh"],
    run_cmds=["./run.sh"],
    port=8080,
    base_url="/api",
    healthcheck_endpoint="/api/health-check",
)

REPO_METADATA_NODE = RepoMetadata(                                                                                                                               
      repo_url="",                                                                                                                                               
      repo_commit="",                                                                                                                                              
      setup_cmds=["npm install", "chmod +x run.sh"],                                                                                                               
      run_cmds=["./run.sh"],                                                                                                                                       
      port=3000,
      base_url="/api",
      healthcheck_endpoint="/api/health-check",
  )

REPO_METADATA_DICT: dict[str, RepoMetadata] = {
    "express": REPO_METADATA_EXPRESS,
    "aiohttp": REPO_METADATA_AIOHTTP,
    "fastapi": REPO_METADATA_FASTAPI,
    "flask": REPO_METADATA_FLASK,
    "gin": REPO_METADATA_GIN,
    "honojs": REPO_METADATA_HONO,
    "uv": REPO_METADATA_UV,
    "node": REPO_METADATA_NODE, 
}

# Matches lines like: diff --git a/path/to/file b/path/to/file
DIFF_GIT_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)\s*$")
# Matches lines like: +++ b/path or --- a/path
ADDITION_RE = re.compile(r"^\+\+\+\s+(.*)\s*$")
DELETION_RE = re.compile(r"^---\s+(.*)\s*$")
TEST_FILE_RE = re.compile(r"test|postman|openapi|scripts", re.IGNORECASE)


def _normalize_path(p: str) -> str | None:
    p = p.strip()
    if p == "/dev/null":
        return None
    if p.startswith("a/") or p.startswith("b/"):
        return p[2:]
    return p


def _compute_task_metadata(task_patch: str) -> TaskMetadata:
    if len(task_patch.strip()) == 0:
        raise Exception("Cannot compute task metadata: patch is empty!")

    files = set[str]()
    added = 0
    removed = 0
    current_file_is_test = False
    lines = task_patch.splitlines()

    for line in lines:
        diff = DIFF_GIT_RE.match(line)
        if diff:
            a_path, b_path = diff.group(1), diff.group(2)
            resolved = _normalize_path(b_path) or _normalize_path(a_path) or b_path
            # If rename, both exist; count the "b/" path as the destination file.
            # If add/delete, one side might be /dev/null later via ---/+++ lines.
            current_file_is_test = bool(TEST_FILE_RE.search(resolved))
            if not current_file_is_test:
                files.add(resolved)
            continue

        addition = ADDITION_RE.match(line)
        deletion = DELETION_RE.match(line)
        re_match = addition if addition is not None else deletion
        if re_match:
            path = _normalize_path(re_match.group(1))
            if path:
                path = path.split("\t", 1)[0].strip()
                current_file_is_test = bool(TEST_FILE_RE.search(path))
                # strip possible prefixes like "b/foo\t(timestamp)" (rare)
                if not current_file_is_test:
                    files.add(path)
                continue
        if not current_file_is_test:
            if line.startswith("+") and not line.startswith("+++"):
                added += 1
            elif line.startswith("-") and not line.startswith("---"):
                removed += 1

    return TaskMetadata(
        lines_added=added,
        lines_removed=removed,
        lines_of_change=added + removed,
        num_changed_files=len(files),
    )


def main(
    docker_folder: str,
    task_name: str,
    task_description: str,
    patched_repo_absolute_path: str | None,
    tasks_folder_absolute_path: str,
    openapi_spec_absolute_path: str,
    prompt: str,
    postman_collection: str = "Conduit.postman_collection.json",
) -> None:
    repo_path = Path(patched_repo_absolute_path) if patched_repo_absolute_path else None

    tasks_path = Path(tasks_folder_absolute_path)
    repo_tasks_path = tasks_path / docker_folder
    repo_tasks_path.mkdir(exist_ok=True)

    task_prompt = prompt
    openapi_path = Path(openapi_spec_absolute_path)
    with open(openapi_path) as file:
        spec = file.read()
        task_prompt = prompt.format(spec)

    patch_name = None
    task_metadata = None

    repo_metadata = None
    repo_metadata = REPO_METADATA_DICT[docker_folder]

    if repo_path is not None:
        git_add = ["git", "add", "-A"]
        subprocess.run(git_add, capture_output=True, cwd=repo_path)

        git_diff = ["git", "diff", "HEAD"]
        task_patch = subprocess.run(
            git_diff,
            capture_output=True,
            text=True,
            cwd=repo_path,
        ).stdout

        patch_name = f"{docker_folder}-{task_name}.patch"
        patch_file = repo_tasks_path / patch_name
        patch_file.write_text(task_patch)

        task_metadata = _compute_task_metadata(task_patch)

    task = Task(
        name=task_name,
        description=task_description,
        docker_folder=docker_folder,
        prompt=task_prompt,
        patch=patch_name,
        repo_metadata=repo_metadata,
        task_metadata=task_metadata,
        postman_collection=postman_collection,
    )

    task_json_name = f"{docker_folder}-{task_name}.json"
    with open(repo_tasks_path / task_json_name, "w") as file:
        json.dump(asdict(task), file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docker-folder", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--task-description", required=True)
    parser.add_argument("--patched-repo-absolute-path", required=False)
    parser.add_argument("--tasks-folder-absolute-path", required=True)
    parser.add_argument("--openapi-spec-absolute-path", required=True)
    parser.add_argument("--prompt", required=True)

    args = parser.parse_args()
    docker_folder: str = args.docker_folder
    task_name: str = args.task_name
    task_description: str = args.task_description
    patched_repo_absolute_path: str | None = args.patched_repo_absolute_path
    tasks_folder_absolute_path: str = args.tasks_folder_absolute_path
    openapi_spec_absolute_path: str = args.openapi_spec_absolute_path
    prompt: str = args.prompt

    if patched_repo_absolute_path and docker_folder not in REPO_METADATA_DICT.keys():
        raise Exception("Unknown docker configuration!")

    main(
        docker_folder,
        task_name,
        task_description,
        patched_repo_absolute_path,
        tasks_folder_absolute_path,
        openapi_spec_absolute_path,
        prompt,
    )
