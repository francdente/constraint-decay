from dataclasses import dataclass


@dataclass
class TaskMetadata:
    lines_added: int
    lines_removed: int
    lines_of_change: int
    num_changed_files: int


@dataclass
class RepoMetadata:
    repo_url: str | None
    repo_commit: str | None
    setup_cmds: list[str]
    run_cmds: list[str]
    port: int
    base_url: str
    healthcheck_endpoint: str


@dataclass
class Task:
    name: str
    description: str
    prompt: str
    docker_folder: str
    patch: str | None
    repo_metadata: RepoMetadata
    task_metadata: TaskMetadata | None
    postman_collection: str = "Conduit.postman_collection.json"
