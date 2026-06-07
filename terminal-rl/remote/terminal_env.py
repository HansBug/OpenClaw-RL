from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import inspect
from functools import partial
from pathlib import Path
from typing import Any

from camel.toolkits import FunctionTool, TerminalToolkit

from terminal_bench.handlers.trial_handler import TrialHandler
from terminal_bench.parsers.base_parser import UnitTestStatus
from terminal_bench.parsers.parser_factory import ParserFactory
from terminal_bench.terminal.docker_compose_manager import DockerComposeManager
from terminal_bench.terminal.terminal import Terminal

from ..custom_types import RunContext, TaskSpec, TaskTimeouts

from .agentharm_env import AgentHarmEnv
from .agent_safetybench_env import AgentSafetyBenchEnv
from .docker_compose_utils import compose_up_no_build, prepare_task_docker_image

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default


def _docker_name_variants(value: str | None) -> set[str]:
    if not value:
        return set()
    raw = value.strip()
    if not raw:
        return set()
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-_.")
    variants = {
        raw,
        cleaned,
        cleaned.replace(".", "-"),
        cleaned.replace("_", "-"),
        cleaned.replace(".", "_"),
    }
    return {v for v in variants if v and "slime-run" in v}


def _matches_project_name(name: str, project_names: set[str], *, broad: bool) -> bool:
    if not name:
        return False
    for project in project_names:
        if name == project:
            return True
        if broad and (
            name.startswith(f"{project}-")
            or name.startswith(f"{project}_")
            or name.startswith(project)
        ):
            return True
    return False


def _docker_image_prefixes(*values: str | None) -> set[str]:
    prefixes: set[str] = set()
    for value in values:
        if not value:
            continue
        raw = value.strip()
        if raw:
            prefixes.add(raw)
        task_match = re.match(r"^([0-9]+)[-_.]", raw)
        if task_match:
            prefixes.add(f"tb__{task_match.group(1)}__")
    return {prefix for prefix in prefixes if prefix.startswith("tb__")}


def _force_remove_docker_objects(
    *,
    trial_name: str,
    client_container_name: str | None,
    docker_image_name_prefix: str | None = None,
    reason: str,
) -> None:
    if not _env_bool("TERMINAL_ENV_FORCE_DOCKER_CLEANUP", True):
        return

    timeout = float(os.getenv("TERMINAL_ENV_FORCE_DOCKER_CLEANUP_TIMEOUT", "20"))
    broad = _env_bool("TERMINAL_ENV_FORCE_DOCKER_CLEANUP_BROAD", True)
    project_names = _docker_name_variants(trial_name)
    project_names.update(_docker_name_variants(client_container_name))
    image_prefixes = _docker_image_prefixes(
        docker_image_name_prefix,
        trial_name,
        client_container_name,
    )
    if not project_names and not image_prefixes:
        return

    def _run(cmd: list[str], *, check: bool = False) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd,
            check=check,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    try:
        listed = _run(
            ["docker", "ps", "-aq", "--format", "{{.ID}}\t{{.Names}}\t{{.Image}}"]
        )
    except Exception as exc:
        logger.warning(
            "Force cleanup could not list Docker containers for %s (%s): %s",
            trial_name,
            reason,
            exc,
        )
        return

    container_ids: list[str] = []
    for line in listed.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        container_id, name = parts[0], parts[1]
        image = parts[2] if len(parts) > 2 else ""
        if _matches_project_name(name, project_names, broad=broad) or any(
            image.startswith(prefix) for prefix in image_prefixes
        ):
            container_ids.append(container_id)

    if container_ids:
        logger.warning(
            "Force removing %d Docker container(s) for TerminalEnv %s (%s)",
            len(container_ids),
            trial_name,
            reason,
        )
        for start in range(0, len(container_ids), 20):
            chunk = container_ids[start : start + 20]
            try:
                _run(["docker", "rm", "-f", *chunk])
            except Exception as exc:
                logger.warning(
                    "Force docker rm failed for TerminalEnv %s ids=%s: %s",
                    trial_name,
                    ",".join(chunk),
                    exc,
                )

    try:
        networks = _run(["docker", "network", "ls", "--format", "{{.ID}}\t{{.Name}}"])
    except Exception:
        return

    network_ids: list[str] = []
    for line in networks.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        net_id, name = parts[0], parts[1]
        if _matches_project_name(name, project_names, broad=broad):
            network_ids.append(net_id)

    for net_id in network_ids:
        try:
            _run(["docker", "network", "rm", net_id])
        except Exception:
            pass


def _stop_terminal_compat(terminal: Terminal, timeout: float) -> None:
    try:
        supports_timeout = "timeout" in inspect.signature(terminal.stop).parameters
    except (TypeError, ValueError):
        supports_timeout = False

    if supports_timeout:
        terminal.stop(timeout=timeout)
    else:
        if _env_bool("TERMINAL_ENV_FAST_CLOSE", False) or _env_bool(
            "TERMINAL_ENV_SKIP_UNBOUNDED_STOP", False
        ):
            logger.warning(
                "Terminal.stop(timeout=...) is unsupported; skipping unbounded "
                "Terminal.stop() under fast close."
            )
            return
        logger.warning(
            "Terminal.stop(timeout=...) is unsupported; retrying with Terminal.stop()."
        )
        terminal.stop()


def _drain_toolkit_sessions(toolkit: Any) -> None:
    sessions = getattr(toolkit, "shell_sessions", None)
    if not isinstance(sessions, dict):
        return
    lock = getattr(toolkit, "_session_lock", None)
    acquired_lock = False
    try:
        if lock is not None:
            try:
                acquired_lock = bool(lock.acquire(blocking=False))
            except TypeError:
                acquired_lock = bool(lock.acquire(False))
            if not acquired_lock:
                logger.warning(
                    "Skipping TerminalToolkit session drain because session lock "
                    "is currently held."
                )
                return
        for session in sessions.values():
            proc = session.get("process")
            if proc is not None:
                try:
                    if hasattr(proc, "terminate"):
                        proc.terminate()
                    elif hasattr(proc, "close"):
                        proc.close()
                except Exception:
                    pass
            q = session.get("output_stream")
            if q is not None:
                try:
                    while not q.empty():
                        q.get_nowait()
                except Exception:
                    pass
        sessions.clear()
    finally:
        if lock is not None and acquired_lock:
            try:
                lock.release()
            except RuntimeError:
                pass


class TerminalEnv:
    def __init__(self) -> None:
        self._closed = False
        self._task_spec: TaskSpec | None = None
        self._run_ctx: RunContext | None = None
        self._timeouts: TaskTimeouts | None = None

        self._trial_handler: TrialHandler | None = None
        self._terminal: Terminal | None = None
        self._parser = None
        self._terminal_toolkit: TerminalToolkit | None = None
        self._tools: dict[str, Any] = {}
        self._agent_safetybench_env: AgentSafetyBenchEnv | None = None
        self._agentharm_env: AgentHarmEnv | None = None
        self._eval_attempt = 0
        self._last_trial_name: str | None = None
        self._last_client_container_name: str | None = None
        self._last_docker_image_name_prefix: str | None = None

    async def reset(
        self,
        *,
        task_meta: dict[str, Any],
        task_spec: TaskSpec,
        run_ctx: RunContext,
        timeouts: TaskTimeouts,
    ) -> tuple[str, list[dict[str, Any]]]:
        await self.close()

        self._closed = False
        self._task_spec = task_spec
        self._run_ctx = run_ctx
        self._timeouts = timeouts
        self._eval_attempt = 0

        if task_meta.get("data_source") == "agent_safetybench":
            self._agent_safetybench_env = AgentSafetyBenchEnv()
            return await self._agent_safetybench_env.reset(
                task_meta=task_meta,
                task_spec=task_spec,
                run_ctx=run_ctx,
            )
        if task_meta.get("data_source") == "agentharm":
            self._agentharm_env = AgentHarmEnv()
            return await self._agentharm_env.reset(
                task_meta=task_meta,
                task_spec=task_spec,
                run_ctx=run_ctx,
            )

        image_prep = await asyncio.to_thread(
            prepare_task_docker_image,
            task=task_meta,
            timeout=self._timeouts.ensure_image,
        )

        dataset_dir = str(os.getenv("DATASET_DIR", "")).strip()
        if not dataset_dir:
            raise ValueError("DATASET_DIR is required")
        task_path = Path(dataset_dir) / self._task_spec.task_path
        output_path = Path(self._run_ctx.log_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        def _sync_reset() -> tuple[str, list[dict[str, Any]]]:
            self._trial_handler = TrialHandler(
                trial_name=f"{self._task_spec.task_name}.{self._run_ctx.uid}.slime-run",
                input_path=task_path,
                output_path=output_path,
            )
            self._last_trial_name = self._trial_handler.trial_name
            self._last_client_container_name = self._trial_handler.client_container_name
            self._last_docker_image_name_prefix = (
                self._trial_handler.docker_image_name_prefix
            )
            task_config = self._trial_handler.task
            self._parser = ParserFactory.get_parser(task_config.parser_name)
            client_image_name = (
                image_prep.client_image_name or self._trial_handler.client_image_name
            )

            self._terminal = Terminal(
                client_container_name=self._trial_handler.client_container_name,
                client_image_name=client_image_name,
                docker_compose_path=self._trial_handler.task_paths.docker_compose_path,
                docker_image_name_prefix=self._trial_handler.docker_image_name_prefix,
                sessions_logs_path=self._trial_handler.trial_paths.sessions_path,
                agent_logs_path=self._trial_handler.trial_paths.agent_logging_dir,
                no_rebuild=True,
                cleanup=False,
            )
            if image_prep.mode == "pull":
                compose_up_no_build(
                    self._terminal,
                    timeout=self._timeouts.reset_session,
                    container_name=self._trial_handler.client_container_name,
                    logger=logger,
                )
            else:
                import inspect
                if "timeout" in inspect.signature(self._terminal.start).parameters:
                    self._terminal.start(timeout=self._timeouts.reset_session)
                else:
                    self._terminal.start()
                try:
                    from .docker_compose_utils import (
                        _apply_container_runtime_limits,
                    )

                    _apply_container_runtime_limits(
                        self._trial_handler.client_container_name,
                        logger=logger,
                    )
                except Exception:
                    pass

            session_logs_dir = (
                self._trial_handler.trial_paths.sessions_path
                / "terminal_toolkit_session_logs"
            )
            self._terminal_toolkit = TerminalToolkit(
                timeout=20.0,
                working_directory=None,
                use_docker_backend=True,
                docker_container_name=self._trial_handler.client_container_name,
                session_logs_dir=session_logs_dir,
                safe_mode=False,
            )
            self._tools = {
                "shell_exec": self._terminal_toolkit.shell_exec,
                "shell_view": self._terminal_toolkit.shell_view,
                "shell_write_to_process": self._terminal_toolkit.shell_write_to_process,
                "shell_write_content_to_file": self._terminal_toolkit.shell_write_content_to_file,
            }

            user_msg = f"Task name:{self._task_spec.task_name}\nTask instruction: {self._task_spec.instruction}"
            function_tools = [FunctionTool(fn) for fn in self._tools.values()]
            tool_schemas = [
                func_tool.get_openai_tool_schema() for func_tool in function_tools
            ]
            return user_msg, tool_schemas

        return await asyncio.to_thread(_sync_reset)

    async def exec_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if self._agent_safetybench_env is not None:
            return await self._agent_safetybench_env.exec_tool(name, arguments)
        if self._agentharm_env is not None:
            return await self._agentharm_env.exec_tool(name, arguments)

        if not self._tools:
            raise RuntimeError("env is not initialized; call reset first")

        if name not in self._tools:
            return f"[TOOL_ERROR] unknown tool: {name}"

        fn = self._tools[name]

        try:
            if asyncio.iscoroutinefunction(fn):
                result = await fn(**arguments)
            elif hasattr(fn, "async_call") and callable(fn.async_call):
                result = await fn.async_call(**arguments)
            else:
                result = await asyncio.to_thread(partial(fn, **arguments))
        except Exception as exc:
            return f"[TOOL_ERROR] {name}: {type(exc).__name__}: {exc}"

        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False)

    async def evaluate(self, trajectory: dict[str, Any] | None = None) -> float:
        if self._agent_safetybench_env is not None:
            return await self._agent_safetybench_env.evaluate(trajectory)
        if self._agentharm_env is not None:
            return await self._agentharm_env.evaluate(trajectory)

        if (
            self._trial_handler is None
            or self._terminal is None
            or self._parser is None
            or self._timeouts is None
        ):
            raise RuntimeError("env is not initialized; call reset first")

        def _sync_eval() -> float:
            task_name = (
                self._task_spec.task_name if self._task_spec is not None else "unknown"
            )
            paths: list[Path] = [self._trial_handler.task_paths.run_tests_path]
            if self._trial_handler.task_paths.test_dir.exists():
                paths.append(self._trial_handler.task_paths.test_dir)

            self._terminal.copy_to_container(
                paths=paths,
                container_dir=str(DockerComposeManager.CONTAINER_TEST_DIR),
            )

            self._eval_attempt += 1
            run_uid = self._run_ctx.uid if self._run_ctx is not None else "unknown"
            test_session = self._terminal.create_session(
                f"tests-{run_uid}-{self._eval_attempt}",
                is_active_stream=False,
                as_configured_user=False,
            )
            test_script_path = str(
                DockerComposeManager.CONTAINER_TEST_DIR / "run-tests.sh"
            )
            test_timeout_sec = min(
                self._timeouts.eval,
                4 * self._trial_handler.task.max_test_timeout_sec,
            )
            try:
                test_session.send_keys(
                    [f"bash {test_script_path}", "Enter"],
                    block=True,
                    max_timeout_sec=test_timeout_sec,
                )
            except TimeoutError as exc:
                logger.warning(
                    "Evaluation tests timed out for task=%s after %.1fs.",
                    task_name,
                    test_timeout_sec,
                )
                raise RuntimeError(
                    f"Evaluation tests timed out for task={task_name} after {test_timeout_sec:.1f}s"
                ) from exc

            test_output = test_session.capture_pane(capture_entire=True)
            try:
                parser_results = self._parser.parse(test_output)
            except Exception as exc:
                tail = test_output[-2000:] if test_output else ""
                logger.warning(
                    "Failed to parse test output for task=%s with parser=%s: %s. Output tail:\n%s",
                    task_name,
                    type(self._parser).__name__,
                    exc,
                    tail,
                )
                raise RuntimeError(
                    f"Failed to parse test output for task={task_name} with parser={type(self._parser).__name__}: {exc}"
                ) from exc

            if not parser_results:
                return 0.0
            passed = sum(
                1
                for status in parser_results.values()
                if status == UnitTestStatus.PASSED
            )
            reward = (
                float(passed / len(parser_results)) if len(parser_results) > 0 else 0.0
            )
            return reward

        return await asyncio.wait_for(
            asyncio.to_thread(_sync_eval),
            timeout=self._timeouts.eval + 30.0,
        )

    def last_eval_details(self) -> dict[str, Any] | None:
        if self._agent_safetybench_env is not None:
            details = getattr(self._agent_safetybench_env, "_last_eval", None)
            return details if isinstance(details, dict) else None
        if self._agentharm_env is not None:
            details = getattr(self._agentharm_env, "_last_eval", None)
            return details if isinstance(details, dict) else None
        return None

    async def close(self) -> None:
        trial_name = (
            self._trial_handler.trial_name
            if self._trial_handler is not None
            else self._last_trial_name or "unknown"
        )
        client_container_name = (
            self._trial_handler.client_container_name
            if self._trial_handler is not None
            else self._last_client_container_name
        )
        docker_image_name_prefix = (
            self._trial_handler.docker_image_name_prefix
            if self._trial_handler is not None
            else self._last_docker_image_name_prefix
        )
        if self._closed:
            logger.warning("TerminalEnv %s already closed", trial_name)
            return
        self._closed = True

        terminal = self._terminal
        timeouts = self._timeouts
        toolkit = self._terminal_toolkit
        agent_safetybench_env = self._agent_safetybench_env
        agentharm_env = self._agentharm_env

        self._tools = {}
        self._terminal = None
        self._trial_handler = None
        self._parser = None
        self._terminal_toolkit = None
        self._task_spec = None
        self._run_ctx = None
        self._timeouts = None
        self._agent_safetybench_env = None
        self._agentharm_env = None

        cleanup_completed = terminal is None
        cleanup_error = False
        fast_close = _env_bool("TERMINAL_ENV_FAST_CLOSE", False)
        try:
            if agent_safetybench_env is not None:
                try:
                    await agent_safetybench_env.close()
                except Exception:
                    logger.exception(
                        "Failed to cleanup Agent-SafetyBench env for %s", trial_name
                    )

            if agentharm_env is not None:
                try:
                    await agentharm_env.close()
                except Exception:
                    logger.exception(
                        "Failed to cleanup AgentHarm env for %s", trial_name
                    )

            if toolkit is not None:
                if fast_close:
                    logger.warning(
                        "Fast close enabled for %s; skipping TerminalToolkit.cleanup "
                        "and relying on direct Docker cleanup.",
                        trial_name,
                    )
                else:
                    try:
                        await asyncio.to_thread(toolkit.cleanup)
                    except Exception:
                        cleanup_error = True
                        logger.exception(
                            "Failed to cleanup terminal toolkit for %s", trial_name
                        )
                try:
                    await asyncio.to_thread(_drain_toolkit_sessions, toolkit)
                except Exception:
                    cleanup_error = True
                    logger.exception(
                        "Failed to drain toolkit sessions for %s", trial_name
                    )

            if terminal is not None and timeouts is not None:
                try:
                    close_timeout = timeouts.close_session
                    if fast_close:
                        close_timeout = min(
                            close_timeout,
                            _env_float("TERMINAL_ENV_FAST_CLOSE_STOP_TIMEOUT", 5.0),
                        )
                    await asyncio.to_thread(
                        _stop_terminal_compat, terminal, close_timeout
                    )
                    cleanup_completed = True
                    logger.info("TerminalEnv %s closed", trial_name)
                except Exception:
                    cleanup_error = True
                    logger.exception("Failed to stop terminal session during close")
        finally:
            force_always = _env_bool("TERMINAL_ENV_FORCE_DOCKER_CLEANUP_ALWAYS", False)
            force_needed = (
                force_always or fast_close or cleanup_error or not cleanup_completed
            )
            if terminal is not None and force_needed:
                if fast_close:
                    reason = "fast_close"
                elif force_always and cleanup_completed and not cleanup_error:
                    reason = "always"
                else:
                    reason = "close_incomplete"
                try:
                    await asyncio.to_thread(
                        _force_remove_docker_objects,
                        trial_name=trial_name,
                        client_container_name=client_container_name,
                        docker_image_name_prefix=docker_image_name_prefix,
                        reason=reason,
                    )
                except Exception:
                    logger.exception(
                        "Force Docker cleanup failed for TerminalEnv %s", trial_name
                    )

    async def force_cleanup(self, reason: str = "external") -> None:
        await asyncio.to_thread(
            _force_remove_docker_objects,
            trial_name=self._last_trial_name or "unknown",
            client_container_name=self._last_client_container_name,
            docker_image_name_prefix=self._last_docker_image_name_prefix,
            reason=reason,
        )
