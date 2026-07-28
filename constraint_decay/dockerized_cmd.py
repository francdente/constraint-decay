import subprocess


def _compose_base(project_name: str | None = None) -> list[str]:
    """Build the base docker compose command, optionally with a project name."""
    cmds = ["docker", "compose"]
    if project_name:
        cmds.extend(["-p", project_name])
    return cmds


def docker_copy(
    docker_folder: str,
    container_id: str,
    src: str,
    dest: str,
) -> None:
    cmds = ["docker", "cp", src, f"{container_id}:{dest}"]
    subprocess.run(cmds, cwd=docker_folder)


def docker_compose_build(docker_folder: str, service: str, project_name: str | None = None) -> None:
    cmds = _compose_base(project_name) + ["build", service]
    subprocess.run(cmds, cwd=docker_folder)


def docker_compose_run(
    docker_folder: str,
    service: str,
    volumes: list[str] | None = None,
    project_name: str | None = None,
) -> str:
    cmds = _compose_base(project_name) + ["run", "-d"]

    if volumes is not None:
        for volume in volumes:
            cmds.append("-v")
            cmds.append(volume)

    cmds.append(service)
    return subprocess.check_output(cmds, cwd=docker_folder).decode().strip()


def docker_stop_rm(container_id: str) -> None:
    """Stop and remove a container by ID. Safe to call if already stopped."""
    subprocess.run(["docker", "rm", "-f", container_id], capture_output=True)


def docker_compose_down(docker_folder: str, project_name: str | None = None) -> None:
    cmds = _compose_base(project_name) + ["down", "--remove-orphans", "-v"]
    subprocess.run(cmds, cwd=docker_folder)


def docker_compose_logs(docker_folder: str, service: str, project_name: str | None = None) -> str:
    cmds = _compose_base(project_name) + ["logs", "--no-color", service]
    result = subprocess.run(cmds, cwd=docker_folder, capture_output=True, text=True)
    return result.stdout


def docker_compose_read_file(docker_folder: str, service: str, path: str, project_name: str | None = None) -> str:
    """Read a file from inside a running container."""
    cmds = _compose_base(project_name) + ["exec", service, "cat", path]
    result = subprocess.run(cmds, cwd=docker_folder, capture_output=True, text=True)
    return result.stdout


def docker_compose_exec(
    docker_folder: str,
    service: str,
    cmd: str,
    capture_output: bool = False,
    flags: list[str] | None = None,
    env: dict[str, str] | None = None,
    project_name: str | None = None,
) -> subprocess.CompletedProcess[bytes]:
    cmds = _compose_base(project_name) + ["exec"]

    if env:
        for key in env.keys():
            cmds.append("-e")
            cmds.append(f"{key}={env[key]}")

    if flags and len(flags) > 0:
        for flag in flags:
            cmds.append(flag)

    cmds.append(service)
    cmds.append("sh")
    cmds.append("-c")
    cmds.append(cmd)

    return subprocess.run(cmds, cwd=docker_folder, capture_output=capture_output)


def docker_compose_copy(
    docker_folder: str,
    container_id: str,
    src: str,
    dest: str,
    project_name: str | None = None,
) -> None:
    cmds = _compose_base(project_name) + ["cp", src, f"{container_id}:{dest}"]
    subprocess.run(cmds, cwd=docker_folder)


def docker_git_apply_patch(
    docker_folder: str,
    service: str,
    file_absolute_path: str,
    project_name: str | None = None,
) -> None:
    git_cmd = f"git apply {file_absolute_path}"
    cmds = _compose_base(project_name) + ["exec", service, "sh", "-c", git_cmd]
    subprocess.run(cmds, cwd=docker_folder)


def docker_git_commit_all(
    docker_folder: str,
    service: str,
    message: str,
    project_name: str | None = None,
) -> None:
    git_cmd = f"git add -A && git commit -m {message}"
    cmds = _compose_base(project_name) + ["exec", service, "sh", "-c", git_cmd]
    subprocess.run(cmds, cwd=docker_folder, capture_output=True)


def docker_git_reinit(docker_folder: str, service: str, project_name: str | None = None) -> None:
    git_cmd = 'rm -rf .git && git init && git add -A && git commit -m "init"'
    cmds = _compose_base(project_name) + ["exec", service, "sh", "-c", git_cmd]
    subprocess.run(cmds, cwd=docker_folder, capture_output=True)


def docker_git_rev_parse(docker_folder: str, service: str, commit: str, project_name: str | None = None) -> str:
    git_cmd = f"git rev-parse {commit}"
    cmds = _compose_base(project_name) + ["exec", service, "sh", "-c", git_cmd]
    return subprocess.run(
        cmds,
        cwd=docker_folder,
        capture_output=True,
        text=True,
    ).stdout.strip()


def docker_git_init_sha(docker_folder: str, service: str, project_name: str | None = None) -> str:
    git_cmd = "git hash-object -t tree /dev/null"
    cmds = _compose_base(project_name) + ["exec", service, "sh", "-c", git_cmd]
    return subprocess.run(
        cmds,
        cwd=docker_folder,
        capture_output=True,
        text=True,
    ).stdout.strip()


def docker_git_diff(docker_folder: str, service: str, commit: str, excludes: list[str] | None = None, project_name: str | None = None) -> str:
    git_cmd = f"git add -A && git diff {commit}"
    if excludes:
        excludes_cmd = " ".join(f"':(exclude){p}'" for p in excludes)
        git_cmd += f" -- . {excludes_cmd}"
    cmds = _compose_base(project_name) + ["exec", service, "sh", "-c", git_cmd]
    return subprocess.run(
        cmds,
        cwd=docker_folder,
        capture_output=True,
        text=True,
    ).stdout
