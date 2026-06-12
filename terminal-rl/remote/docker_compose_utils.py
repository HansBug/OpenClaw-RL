from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from terminal_bench.handlers.trial_handler import TrialHandler
from terminal_bench.terminal.docker_compose_manager import DockerComposeManager
from terminal_bench.terminal.terminal import Terminal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImagePreparationResult:
    mode: Literal["build", "pull"]
    client_image_name: str | None = None


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


_MAX_CONCURRENT_BUILDS = _env_int("WORKER_MAX_CONCURRENT_BUILDS", 4)
_BUILD_SEMAPHORE = threading.BoundedSemaphore(_MAX_CONCURRENT_BUILDS)
_BUILD_LOCKS_GUARD = threading.Lock()
_BUILD_LOCKS: dict[str, threading.Lock] = {}
_BUILD_DONE: set[str] = set()
_BUILD_FAILED: dict[str, tuple[float, str]] = {}
_TASK_IMAGE_BLACKLISTED: dict[str, tuple[float, str]] = {}

_DOCKERFILE_INSTRUCTION_RE = re.compile(
    r"^\s*(?:ADD|ARG|CMD|COPY|ENTRYPOINT|ENV|EXPOSE|FROM|HEALTHCHECK|LABEL|"
    r"MAINTAINER|ONBUILD|RUN|SHELL|STOPSIGNAL|USER|VOLUME|WORKDIR)\b",
    re.IGNORECASE,
)


class DockerImageBuildError(RuntimeError):
    """Deterministic task image build failure cached per task image."""


class TaskImageBlacklistedError(DockerImageBuildError):
    """Task image is known-bad and should not hit Docker again until TTL expires."""


def _safe_project_component(value: str, fallback: str = "task") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    cleaned = cleaned.strip("-.")
    return (cleaned or fallback)[:80]


def _shorten_output(text: str | None, max_chars: int = 4000) -> str:
    if not text:
        return ""
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    return f"{stripped[:max_chars]}...(truncated, total={len(stripped)} chars)"


def _build_docker_pull_error_message(
    *,
    image: str,
    cmd: list[str],
    return_code: int | None = None,
    timeout: float | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
) -> str:
    lines = [
        f"Docker image pull failed for image '{image}'.",
        f"Command: {' '.join(cmd)}",
    ]
    if return_code is not None:
        lines.append(f"Exit code: {return_code}")
    if timeout is not None:
        lines.append(f"Timeout: {timeout:.1f}s")

    out = _shorten_output(stdout)
    err = _shorten_output(stderr)
    if out:
        lines.append(f"STDOUT:\n{out}")
    if err:
        lines.append(f"STDERR:\n{err}")

    lines.append(
        "Hints: verify task_name/task image tag, run docker login for the registry, and ensure image exists."
    )
    return "\n".join(lines)


def _build_compose_up_error_message(
    *,
    cmd: list[str],
    return_code: int | None = None,
    timeout: float | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
    note: str | None = None,
) -> str:
    lines = [
        "Docker compose up --no-build failed.",
        f"Command: {' '.join(cmd)}",
    ]
    if return_code is not None:
        lines.append(f"Exit code: {return_code}")
    if timeout is not None:
        lines.append(f"Timeout: {timeout:.1f}s")

    out = _shorten_output(stdout)
    err = _shorten_output(stderr)
    if out:
        lines.append(f"STDOUT:\n{out}")
    if err:
        lines.append(f"STDERR:\n{err}")
    if note:
        lines.append(f"Note: {note}")

    lines.append(
        "Hints: verify `docker compose version`; if unavailable, install Compose plugin or ensure `docker-compose` is on PATH."
    )
    return "\n".join(lines)


def _compose_plugin_maybe_missing(stderr: str | None) -> bool:
    if not stderr:
        return False
    lowered = stderr.lower()
    return (
        "unknown shorthand flag: 'p' in -p" in lowered
        or "docker: 'compose' is not a docker command" in lowered
        or 'unknown command "compose"' in lowered
    )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _build_failed_ttl() -> float:
    raw = os.getenv("WORKER_DOCKER_BUILD_FAILED_TTL", "3600")
    try:
        value = float(raw)
    except ValueError:
        return 3600.0
    return max(0.0, value)


def _task_image_blacklist_ttl() -> float:
    raw = os.getenv("WORKER_DOCKER_TASK_BLACKLIST_TTL", "86400")
    try:
        value = float(raw)
    except ValueError:
        return 86400.0
    return max(0.0, value)


def _cached_build_failure(build_key: str) -> str | None:
    ttl = _build_failed_ttl()
    if ttl <= 0:
        return None
    cached = _BUILD_FAILED.get(build_key)
    if cached is None:
        return None
    ts, message = cached
    if time.time() - ts <= ttl:
        return message
    _BUILD_FAILED.pop(build_key, None)
    return None


def _cached_task_image_blacklist(build_key: str) -> str | None:
    ttl = _task_image_blacklist_ttl()
    if ttl <= 0:
        return None
    cached = _TASK_IMAGE_BLACKLISTED.get(build_key)
    if cached is None:
        return None
    ts, message = cached
    if time.time() - ts <= ttl:
        return message
    _TASK_IMAGE_BLACKLISTED.pop(build_key, None)
    return None


def _blacklist_task_image(build_key: str, message: str) -> None:
    if _task_image_blacklist_ttl() <= 0:
        return
    _TASK_IMAGE_BLACKLISTED[build_key] = (time.time(), message)


def _is_deterministic_build_failure(message: str) -> bool:
    lowered = message.lower()
    deterministic_markers = (
        "dockerfile parse error",
        "unknown instruction:",
        "failed to read dockerfile",
        "cannot locate specified dockerfile",
        "no such file or directory",
        "yaml: line",
        "services must be a mapping",
    )
    return any(marker in lowered for marker in deterministic_markers)


def _dockerfile_precheck_error(task_path: Path) -> str | None:
    if not _env_bool("WORKER_DOCKERFILE_PRECHECK", True):
        return None
    dockerfile = task_path / "Dockerfile"
    try:
        lines = dockerfile.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return f"{dockerfile} is missing"
    except OSError as exc:
        return f"could not read {dockerfile}: {exc}"

    current_instruction = ""
    skip_heredoc_until: str | None = None
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if skip_heredoc_until is not None:
            if stripped == skip_heredoc_until:
                skip_heredoc_until = None
            continue
        match = _DOCKERFILE_INSTRUCTION_RE.match(line)
        if match:
            current_instruction = match.group(0).strip().split()[0].upper()
        marker = re.search(r"<<\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?", line)
        if marker is not None and current_instruction in {"ADD", "COPY"}:
            skip_heredoc_until = marker.group(1)
            continue
        if "<<" not in line or current_instruction != "RUN":
            if stripped and not line.rstrip().endswith("\\") and match:
                current_instruction = ""
            continue
        run_body = re.sub(r"^\s*RUN\s+", "", line, flags=re.IGNORECASE).strip()
        # Dockerfile heredoc syntax is `RUN <<EOF`; shell redirection heredocs
        # such as `RUN cat > file <<EOF` are parsed as separate Dockerfile
        # instructions on older/fronted-default builders and deterministically fail.
        if run_body.startswith("<<"):
            if marker is not None:
                skip_heredoc_until = marker.group(1)
            continue
        if marker is None:
            continue
        return (
            f"{dockerfile}:{idx} uses a shell heredoc inside RUN "
            f"({marker.group(0)!r}). Use Dockerfile-native `RUN <<EOF` or "
            "rewrite with printf/COPY heredoc; otherwise Docker parses the body "
            "as Dockerfile instructions."
        )
    return None


def _lock_for_build_key(key: str) -> threading.Lock:
    with _BUILD_LOCKS_GUARD:
        lock = _BUILD_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _BUILD_LOCKS[key] = lock
        return lock


def build_docker_image(task: dict[str, Any], timeout: float = 1200.0) -> None:
    dataset_dir = str(os.getenv("DATASET_DIR", "")).strip()
    if not dataset_dir:
        raise ValueError("DATASET_DIR is required")

    task_path = Path(dataset_dir) / str(task.get("task_path", ""))
    task_name = _safe_project_component(str(task.get("task_name") or task_path.name))
    trial_handler = TrialHandler(
        trial_name=f"build_run_{task_name}",
        input_path=task_path,
        output_path=Path("build_outputs"),
    )

    compose_manager = DockerComposeManager(
        client_container_name=trial_handler.client_container_name,
        client_image_name=trial_handler.client_image_name,
        docker_image_name_prefix=trial_handler.docker_image_name_prefix,
        docker_compose_path=trial_handler.task_paths.docker_compose_path,
        no_rebuild=True,
        cleanup=False,
        sessions_logs_path=trial_handler.trial_paths.sessions_path,
        agent_logs_path=trial_handler.trial_paths.agent_logging_dir,
    )
    build_key = trial_handler.client_image_name
    build_dedup = _env_bool("WORKER_DOCKER_BUILD_DEDUP", True)
    skip_existing = _env_bool("WORKER_DOCKER_BUILD_SKIP_EXISTING", True)
    build_lock = _lock_for_build_key(build_key) if build_dedup else threading.Lock()

    with build_lock:
        if build_key in _BUILD_DONE:
            return
        blacklisted_failure = _cached_task_image_blacklist(build_key)
        if blacklisted_failure is not None:
            raise TaskImageBlacklistedError(
                f"TASK_IMAGE_BLACKLISTED image={build_key}: {blacklisted_failure}"
            )
        cached_failure = _cached_build_failure(build_key)
        if cached_failure is not None:
            raise DockerImageBuildError(
                f"TASK_BUILD_FAILED cached for image={build_key}: {cached_failure}"
            )
        if skip_existing and _docker_image_exists_locally(build_key):
            logger.debug("Docker image already exists locally; skip build: %s", build_key)
            _BUILD_DONE.add(build_key)
            _BUILD_FAILED.pop(build_key, None)
            _TASK_IMAGE_BLACKLISTED.pop(build_key, None)
            return
        precheck_error = _dockerfile_precheck_error(task_path)
        if precheck_error is not None:
            message = (
                f"TASK_DOCKERFILE_PRECHECK_FAILED task={task_name}: {precheck_error}"
            )
            _BUILD_FAILED[build_key] = (time.time(), message)
            _blacklist_task_image(build_key, message)
            raise TaskImageBlacklistedError(
                f"TASK_IMAGE_BLACKLISTED image={build_key}: {message}"
            )

        with _BUILD_SEMAPHORE:
            import inspect

            logger.info("Building Docker image for task=%s image=%s", task_name, build_key)
            try:
                if "timeout" in inspect.signature(compose_manager.build).parameters:
                    compose_manager.build(timeout=timeout)
                else:
                    compose_manager.build()
            except Exception as exc:
                message = _shorten_output(str(exc), max_chars=1200) or type(exc).__name__
                _BUILD_FAILED[build_key] = (time.time(), message)
                if _is_deterministic_build_failure(message):
                    _blacklist_task_image(build_key, message)
                raise DockerImageBuildError(
                    f"TASK_BUILD_FAILED image={build_key} task={task_name}: {message}"
                ) from exc
            _BUILD_DONE.add(build_key)
            _BUILD_FAILED.pop(build_key, None)
            _TASK_IMAGE_BLACKLISTED.pop(build_key, None)


def _resolve_pull_image(task: dict[str, Any]) -> str:
    prefix = str(os.getenv("TBENCH_DOCKER_PULL_PREFIX", "")).strip()
    if not prefix:
        raise ValueError("TBENCH_DOCKER_PULL_PREFIX is required in pull mode")

    task_name = str(task.get("task_name", "")).strip()
    if not task_name:
        raise ValueError("task_name is required to resolve pull image")
    if "<" in task_name and ">" in task_name:
        raise ValueError(
            "task_name appears to still be a placeholder "
            f"('{task_name}'). Please provide a concrete task_name."
        )

    return f"{prefix}{task_name}"


def _docker_image_exists_locally(image: str, timeout: float = 30.0) -> bool:
    cmd = ["docker", "image", "inspect", image]
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.returncode == 0


def pull_docker_image(image: str, timeout: float = 1200.0) -> None:
    if _docker_image_exists_locally(image):
        return

    cmd = ["docker", "pull", image]
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            _build_docker_pull_error_message(
                image=image,
                cmd=cmd,
                timeout=timeout,
                stdout=exc.stdout,
                stderr=exc.stderr,
            )
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            _build_docker_pull_error_message(
                image=image,
                cmd=cmd,
                return_code=exc.returncode,
                stdout=exc.stdout,
                stderr=exc.stderr,
            )
        ) from exc


def prepare_task_docker_image(
    task: dict[str, Any],
    timeout: float = 1200.0,
) -> ImagePreparationResult:
    raw_mode = str(os.getenv("TBENCH_DOCKER_IMAGE_SOURCE", "")).strip().lower()
    if not raw_mode:
        raise ValueError("TBENCH_DOCKER_IMAGE_SOURCE is required")

    if raw_mode in {"build", "docker_build"}:
        build_docker_image(task=task, timeout=timeout)
        return ImagePreparationResult(mode="build", client_image_name=None)

    if raw_mode in {"pull", "docker_pull"}:
        image = _resolve_pull_image(task=task)
        pull_docker_image(image=image, timeout=timeout)
        return ImagePreparationResult(mode="pull", client_image_name=image)

    raise ValueError(
        f"Unsupported docker image source '{raw_mode}'. Expected one of: build, pull"
    )


_DEFAULT_CONTAINER_MEMORY_LIMIT = os.getenv("CONTAINER_MEMORY_LIMIT", "16g")
_DEFAULT_CONTAINER_PIDS_LIMIT = os.getenv("CONTAINER_PIDS_LIMIT", "64")


def _apply_container_memory_limit(
    container_name: str,
    memory_limit: str,
    logger: logging.Logger | None = None,
) -> None:
    """Best-effort ``docker update --memory`` on a running container."""
    if not memory_limit:
        return
    cmd = [
        "docker",
        "update",
        f"--memory={memory_limit}",
        f"--memory-swap={memory_limit}",
        container_name,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30.0)
        if logger is not None:
            logger.info(
                "Applied memory limit %s to container %s", memory_limit, container_name
            )
    except Exception as exc:
        if logger is not None:
            logger.warning(
                "Failed to apply memory limit to container %s: %s", container_name, exc
            )


def _apply_container_pids_limit(
    container_name: str,
    pids_limit: str,
    logger: logging.Logger | None = None,
) -> None:
    """Best-effort ``docker update --pids-limit`` on a running container."""
    if not pids_limit:
        return
    try:
        value = int(str(pids_limit).strip())
    except ValueError:
        if logger is not None:
            logger.warning(
                "Invalid CONTAINER_PIDS_LIMIT=%r; skipping pids limit for %s",
                pids_limit,
                container_name,
            )
        return
    if value <= 0:
        return

    cmd = ["docker", "update", f"--pids-limit={value}", container_name]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30.0)
        if logger is not None:
            logger.info("Applied pids limit %s to container %s", value, container_name)
    except Exception as exc:
        if logger is not None:
            logger.warning(
                "Failed to apply pids limit to container %s: %s", container_name, exc
            )


def _apply_container_runtime_limits(
    container_name: str,
    logger: logging.Logger | None = None,
) -> None:
    _apply_container_pids_limit(
        container_name, _DEFAULT_CONTAINER_PIDS_LIMIT, logger=logger
    )
    _apply_container_memory_limit(
        container_name, _DEFAULT_CONTAINER_MEMORY_LIMIT, logger=logger
    )


def compose_up_no_build(
    terminal: Terminal,
    *,
    timeout: float,
    container_name: str,
    logger: logging.Logger | None = None,
) -> None:
    compose_manager = getattr(terminal, "_compose_manager")
    compose_override_path = str(os.getenv("COMPOSE_OVERRIDE_PATH", "")).strip()
    compose_command = ["up", "-d", "--no-build"]
    if compose_override_path:
        compose_command = ["-f", compose_override_path, *compose_command]

    command = compose_manager.get_docker_compose_command(compose_command)
    if logger is not None:
        logger.info("Running docker compose command: %s", " ".join(command))
        if compose_override_path:
            logger.info("Using compose override file: %s", compose_override_path)

    compose_manager.env["http_proxy"] = os.getenv("HTTP_PROXY", "")
    compose_manager.env["https_proxy"] = os.getenv("HTTPS_PROXY", "")

    try:
        subprocess.run(
            command,
            env=compose_manager.env,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        if logger is not None:
            logger.error(
                "Docker compose up --no-build timed out after %.1f sec", timeout
            )
            if exc.stdout:
                logger.error("STDOUT: %s", exc.stdout)
            if exc.stderr:
                logger.error("STDERR: %s", exc.stderr)
        raise RuntimeError(
            _build_compose_up_error_message(
                cmd=command,
                timeout=timeout,
                stdout=exc.stdout,
                stderr=exc.stderr,
            )
        ) from exc
    except subprocess.CalledProcessError as exc:
        if logger is not None:
            logger.error(
                "Docker compose up --no-build failed with code %s", exc.returncode
            )
            if exc.stdout:
                logger.error("STDOUT: %s", exc.stdout)
            if exc.stderr:
                logger.error("STDERR: %s", exc.stderr)
        if _compose_plugin_maybe_missing(exc.stderr):
            fallback_command = ["docker-compose", *command[2:]]
            if shutil.which("docker-compose"):
                if logger is not None:
                    logger.warning(
                        "docker compose plugin may be unavailable; falling back to: %s",
                        " ".join(fallback_command),
                    )
                try:
                    subprocess.run(
                        fallback_command,
                        env=compose_manager.env,
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
                except subprocess.TimeoutExpired as fallback_exc:
                    raise RuntimeError(
                        _build_compose_up_error_message(
                            cmd=fallback_command,
                            timeout=timeout,
                            stdout=fallback_exc.stdout,
                            stderr=fallback_exc.stderr,
                            note=f"Initial command {' '.join(command)} failed with exit code {exc.returncode}.",
                        )
                    ) from fallback_exc
                except subprocess.CalledProcessError as fallback_exc:
                    raise RuntimeError(
                        _build_compose_up_error_message(
                            cmd=fallback_command,
                            return_code=fallback_exc.returncode,
                            stdout=fallback_exc.stdout,
                            stderr=fallback_exc.stderr,
                            note=f"Initial command {' '.join(command)} failed with exit code {exc.returncode}.",
                        )
                    ) from fallback_exc
                container = compose_manager._client.containers.get(container_name)
                terminal.container = container
                compose_manager._client_container = container
                _apply_container_runtime_limits(container_name, logger=logger)
                return

            raise RuntimeError(
                _build_compose_up_error_message(
                    cmd=command,
                    return_code=exc.returncode,
                    stdout=exc.stdout,
                    stderr=exc.stderr,
                    note="Compose plugin appears unavailable and `docker-compose` binary was not found.",
                )
            ) from exc

        raise RuntimeError(
            _build_compose_up_error_message(
                cmd=command,
                return_code=exc.returncode,
                stdout=exc.stdout,
                stderr=exc.stderr,
            )
        ) from exc

    container = compose_manager._client.containers.get(container_name)
    terminal.container = container
    compose_manager._client_container = container
    _apply_container_runtime_limits(container_name, logger=logger)
