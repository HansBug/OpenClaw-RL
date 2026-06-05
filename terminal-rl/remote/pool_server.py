from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import subprocess
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..custom_types import RunContext, TaskSpec, TaskTimeouts
from ..request_utils import json_payload
from .terminal_env import TerminalEnv

logger = logging.getLogger("terminal.env.worker")
app = FastAPI()


def _parse_timeout_overrides(
    base: TaskTimeouts, payload: dict[str, Any] | None
) -> TaskTimeouts:
    if not isinstance(payload, dict):
        return base

    def _pick(key: str, default: float) -> float:
        raw = payload.get(key, default)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return default
        return value if value > 0 else default

    return TaskTimeouts(
        ensure_image=_pick("ensure_image", base.ensure_image),
        reset_session=_pick("reset_session", base.reset_session),
        close_session=_pick("close_session", base.close_session),
        eval=_pick("eval", base.eval),
    )


def _build_task_spec(task_meta: dict[str, Any]) -> TaskSpec:
    return TaskSpec(
        task_name=str(task_meta.get("task_name", "unknown")),
        task_path=str(task_meta.get("task_path", "")),
        instruction=str(task_meta.get("instruction", "")),
    )


def _build_run_ctx(
    run_ctx_payload: dict[str, Any] | None, default_log_dir: Path
) -> RunContext:
    payload = run_ctx_payload if isinstance(run_ctx_payload, dict) else {}
    uid = str(payload.get("uid") or uuid.uuid4().hex[:8])
    try:
        group_index = int(payload.get("group_index") or 0)
    except (TypeError, ValueError):
        group_index = 0
    try:
        sample_index = int(payload.get("sample_index") or 0)
    except (TypeError, ValueError):
        sample_index = 0

    log_dir_raw = payload.get("log_dir")
    if isinstance(log_dir_raw, str) and log_dir_raw:
        log_dir = Path(log_dir_raw).resolve()
    else:
        log_dir = default_log_dir.resolve()

    return RunContext(
        uid=uid,
        group_index=group_index,
        sample_index=sample_index,
        log_dir=log_dir,
    )


class CapacityError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


class ResourcePressureError(Exception):
    def __init__(self, code: str, message: str, details: dict[str, Any]):
        self.code = code
        self.message = message
        self.details = details
        super().__init__(message)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default
    return value


def docker_data_root_stats() -> dict[str, Any]:
    path = os.getenv("DOCKER_DATA_ROOT") or os.getenv("DOCKER_ROOT") or "/data"
    usage = shutil.disk_usage(path)
    st = os.statvfs(path)
    total_inodes = int(st.f_files)
    free_inodes = int(st.f_ffree)
    used_inodes = max(total_inodes - free_inodes, 0)
    used_pct = (usage.used * 100.0 / usage.total) if usage.total else 0.0
    inode_used_pct = (
        (used_inodes * 100.0 / total_inodes) if total_inodes else 0.0
    )
    return {
        "path": path,
        "total_gb": usage.total / 1024**3,
        "used_gb": usage.used / 1024**3,
        "free_gb": usage.free / 1024**3,
        "used_pct": used_pct,
        "total_inodes": total_inodes,
        "used_inodes": used_inodes,
        "free_inodes": free_inodes,
        "inode_used_pct": inode_used_pct,
    }


_PRESSURE_CACHE: tuple[float, dict[str, Any]] | None = None


def _read_proc_pressure_stats() -> dict[str, Any]:
    total_procs = 0
    total_tasks = 0
    zombies = 0
    shim = 0
    runc = 0
    dockerd = 0
    containerd = 0
    docker_cli = 0

    for proc_dir in Path("/proc").glob("[0-9]*"):
        if not proc_dir.is_dir():
            continue
        total_procs += 1
        try:
            name = (proc_dir / "comm").read_text(errors="ignore").strip()
        except OSError:
            name = ""
        try:
            stat = (proc_dir / "stat").read_text(errors="ignore")
            rest = stat.split(") ", 1)[1]
            if rest.split(" ", 1)[0] == "Z":
                zombies += 1
        except (OSError, IndexError):
            pass
        try:
            total_tasks += sum(1 for p in (proc_dir / "task").iterdir() if p.is_dir())
        except OSError:
            pass

        if name == "dockerd":
            dockerd += 1
        elif name == "containerd":
            containerd += 1
        elif name.startswith("containerd-shim"):
            shim += 1
        elif name == "runc":
            runc += 1
        elif name == "docker":
            docker_cli += 1

    pids_max = 0
    try:
        pids_max = int(Path("/proc/sys/kernel/threads-max").read_text().strip())
    except (OSError, ValueError):
        pids_max = 0
    pids_pct = (total_tasks * 100.0 / pids_max) if pids_max > 0 else 0.0
    return {
        "procs": total_procs,
        "tasks": total_tasks,
        "pids_max": pids_max,
        "pids_pct": pids_pct,
        "zombies": zombies,
        "dockerd": dockerd,
        "containerd": containerd,
        "shim": shim,
        "runc": runc,
        "docker_cli_procs": docker_cli,
    }


def _docker_cli_ok(timeout_sec: float) -> bool:
    try:
        result = subprocess.run(
            ["docker", "ps", "-q"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_sec,
        )
        return result.returncode == 0
    except Exception:
        return False


def worker_pressure_stats(*, force: bool = False) -> dict[str, Any]:
    global _PRESSURE_CACHE
    ttl = _env_float("WORKER_PRESSURE_CACHE_TTL", 5.0)
    now = time.time()
    if (
        not force
        and _PRESSURE_CACHE is not None
        and now - _PRESSURE_CACHE[0] <= ttl
    ):
        return dict(_PRESSURE_CACHE[1])

    stats = _read_proc_pressure_stats()
    docker_timeout = _env_float("WORKER_DOCKER_CLI_TIMEOUT", 3.0)
    stats["docker_cli_ok"] = _docker_cli_ok(docker_timeout)
    stats["docker_cli_timeout_sec"] = docker_timeout
    _PRESSURE_CACHE = (now, dict(stats))
    return stats


def assert_worker_has_capacity_for_docker(
    *, phase: str = "health", pending_closes: int = 0
) -> None:
    if os.getenv("WORKER_DISK_GUARD_ENABLED", "1") == "0":
        disk_guard_enabled = False
    else:
        disk_guard_enabled = True

    if disk_guard_enabled:
        min_free_gb = _env_float("WORKER_MIN_DOCKER_FREE_GB", 50.0)
        max_used_pct = _env_float("WORKER_MAX_DOCKER_USED_PCT", 85.0)
        max_inode_pct = _env_float("WORKER_MAX_DOCKER_INODE_PCT", 80.0)

        try:
            stats = docker_data_root_stats()
        except Exception as exc:
            raise ResourcePressureError(
                "WORKER_DISK_STATS_FAILED",
                f"Failed to read Docker data-root stats: {exc}",
                {"error": str(exc), "phase": phase},
            ) from exc

        over_capacity = (
            stats["free_gb"] < min_free_gb
            or stats["used_pct"] > max_used_pct
            or stats["inode_used_pct"] > max_inode_pct
        )
        if over_capacity:
            raise ResourcePressureError(
                "WORKER_DOCKER_DISK_PRESSURE",
                (
                    "Worker Docker data-root is under disk pressure: "
                    f"path={stats['path']} free={stats['free_gb']:.1f}GB "
                    f"used={stats['used_pct']:.1f}% inode={stats['inode_used_pct']:.1f}% "
                    f"thresholds free>={min_free_gb:.1f}GB used<={max_used_pct:.1f}% "
                    f"inode<={max_inode_pct:.1f}%"
                ),
                {
                    **stats,
                    "phase": phase,
                    "min_free_gb": min_free_gb,
                    "max_used_pct": max_used_pct,
                    "max_inode_pct": max_inode_pct,
                },
            )

    if os.getenv("WORKER_PRESSURE_GUARD_ENABLED", "1") == "0":
        return

    pressure = worker_pressure_stats()
    pids_pause_pct = _env_float("WORKER_PIDS_PAUSE_ALLOCATE_PCT", 75.0)
    pids_reject_reset_pct = _env_float("WORKER_PIDS_REJECT_RESET_PCT", 85.0)
    shim_pause = _env_int("WORKER_SHIM_PAUSE_ALLOCATE", 256)
    shim_reject_reset = _env_int("WORKER_SHIM_REJECT_RESET", 384)
    pending_pause = _env_int("WORKER_PENDING_CLOSES_PAUSE_ALLOCATE", 50)
    pending_reject_reset = _env_int("WORKER_PENDING_CLOSES_REJECT_RESET", 100)

    details = {**pressure, "phase": phase, "pending_closes": pending_closes}
    if not bool(pressure.get("docker_cli_ok", False)):
        raise ResourcePressureError(
            "WORKER_DOCKER_CLI_UNHEALTHY",
            "Worker Docker CLI probe failed; refusing new Docker work.",
            details,
        )

    if phase == "reset":
        if pressure["pids_pct"] >= pids_reject_reset_pct:
            raise ResourcePressureError(
                "WORKER_PIDS_PRESSURE",
                (
                    f"Worker pids pressure {pressure['pids_pct']:.1f}% "
                    f">= reset threshold {pids_reject_reset_pct:.1f}%"
                ),
                details,
            )
        if pressure["shim"] >= shim_reject_reset:
            raise ResourcePressureError(
                "WORKER_SHIM_PRESSURE",
                f"Worker shim pressure {pressure['shim']} >= reset threshold {shim_reject_reset}",
                details,
            )
        if pending_closes >= pending_reject_reset:
            raise ResourcePressureError(
                "WORKER_PENDING_CLOSES_PRESSURE",
                f"Worker pending_closes {pending_closes} >= reset threshold {pending_reject_reset}",
                details,
            )
        return

    if phase in {"allocate", "health"}:
        if pressure["pids_pct"] >= pids_pause_pct:
            raise ResourcePressureError(
                "WORKER_PIDS_PRESSURE",
                (
                    f"Worker pids pressure {pressure['pids_pct']:.1f}% "
                    f">= allocate threshold {pids_pause_pct:.1f}%"
                ),
                details,
            )
        if pressure["shim"] >= shim_pause:
            raise ResourcePressureError(
                "WORKER_SHIM_PRESSURE",
                f"Worker shim pressure {pressure['shim']} >= allocate threshold {shim_pause}",
                details,
            )
        if pending_closes >= pending_pause:
            raise ResourcePressureError(
                "WORKER_PENDING_CLOSES_PRESSURE",
                f"Worker pending_closes {pending_closes} >= allocate threshold {pending_pause}",
                details,
            )


@dataclass
class RunSlot:
    run_lease_id: str
    task_key: str
    env: TerminalEnv
    last_used_ts: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    phase: str = "allocated"
    in_flight_ops: int = 0
    active_op: str | None = None
    close_requested: bool = False
    close_reason: str | None = None
    close_requested_ts: float | None = None
    reset_started_ts: float | None = None
    reset_completed_ts: float | None = None
    first_step_ts: float | None = None
    evaluate_completed_ts: float | None = None


@dataclass
class TaskSlot:
    task_key: str
    runs: dict[str, RunSlot] = field(default_factory=dict)
    created_ts: float = field(default_factory=time.time)
    last_used_ts: float = field(default_factory=time.time)


class WorkerPool:
    def __init__(
        self,
        *,
        max_tasks: int,
        max_runs_per_task: int,
        run_idle_ttl: int,
        output_root: str,
        default_timeouts: TaskTimeouts,
        idempotency_ttl: int = 300,
        max_concurrent_closes: int = 8,
    ) -> None:
        self.max_tasks = max_tasks
        self.max_runs_per_task = max_runs_per_task
        self.run_idle_ttl = run_idle_ttl
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.default_timeouts = default_timeouts
        self.idempotency_ttl = idempotency_ttl
        self.close_task_timeout = _env_float(
            "WORKER_CLOSE_TASK_TIMEOUT",
            max(30.0, float(default_timeouts.close_session) + 30.0),
        )

        self._tasks: dict[str, TaskSlot] = {}
        self._run_to_task: dict[str, str] = {}
        self._idempotency: dict[tuple[str, str], tuple[str, float]] = {}
        self._lock = asyncio.Lock()

        self._close_sem = asyncio.Semaphore(max_concurrent_closes)
        self._closing_tasks: set[asyncio.Task] = set()
        self._closing_task_started: dict[asyncio.Task, float] = {}
        self._closing_task_labels: dict[asyncio.Task, str] = {}

    def _new_env(self) -> TerminalEnv:
        return TerminalEnv()

    @staticmethod
    def _run_slot_container_info(run_slot: RunSlot) -> dict[str, Any]:
        env = run_slot.env
        terminal = getattr(env, "_terminal", None)
        container = getattr(terminal, "container", None) if terminal is not None else None
        container_id = getattr(container, "id", None)
        short_id = container_id[:12] if isinstance(container_id, str) else None
        container_name = (
            getattr(container, "name", None)
            or getattr(env, "_last_client_container_name", None)
        )
        container_status = getattr(container, "status", None)
        trial_name = getattr(env, "_last_trial_name", None)
        return {
            "id": container_id,
            "short_id": short_id,
            "name": container_name,
            "status": container_status,
            "trial_name": trial_name,
        }

    @classmethod
    def _run_slot_container_ref(cls, run_slot: RunSlot) -> str:
        info = cls._run_slot_container_info(run_slot)
        return (
            f"container_name={info.get('name') or '?'} "
            f"container_id={info.get('short_id') or '?'} "
            f"container_status={info.get('status') or '?'} "
            f"trial={info.get('trial_name') or '?'}"
        )

    def _pop_run_slot_locked(
        self, run_lease_id: str
    ) -> tuple[str, RunSlot] | None:
        task_key = self._run_to_task.pop(run_lease_id, None)
        if task_key is None:
            return None
        task_slot = self._tasks.get(task_key)
        run_slot = task_slot.runs.pop(run_lease_id, None) if task_slot else None
        if task_slot is not None and not task_slot.runs:
            self._tasks.pop(task_key, None)
            logger.info("Removed empty task slot: %s", task_key)
        if run_slot is None:
            return None
        return task_key, run_slot

    def _phase_for_op(self, op_name: str) -> str:
        return {
            "reset": "resetting",
            "exec_tool": "stepping",
            "evaluate": "evaluating",
            "heartbeat": "heartbeat",
        }.get(op_name, op_name)

    async def _begin_run_op(self, run_lease_id: str, op_name: str) -> RunSlot:
        async with self._lock:
            run_slot = self._get_run_slot(run_lease_id)
            if run_slot.close_requested:
                raise RuntimeError(
                    f"Run {run_lease_id} is closing; rejecting new {op_name} request"
                )
            now = time.time()
            run_slot.in_flight_ops += 1
            run_slot.active_op = op_name
            run_slot.phase = self._phase_for_op(op_name)
            run_slot.last_used_ts = now
            if op_name == "reset":
                run_slot.reset_started_ts = now
            logger.debug(
                "Run op begin: lease=%s task=%s op=%s phase=%s in_flight=%d %s",
                run_lease_id,
                run_slot.task_key,
                op_name,
                run_slot.phase,
                run_slot.in_flight_ops,
                self._run_slot_container_ref(run_slot),
            )
            return run_slot

    async def _finish_run_op(
        self, run_slot: RunSlot, op_name: str, *, success: bool
    ) -> None:
        close_after: tuple[str, str, RunSlot, str] | None = None
        async with self._lock:
            now = time.time()
            run_slot.in_flight_ops = max(0, run_slot.in_flight_ops - 1)
            run_slot.last_used_ts = now
            if success:
                if op_name == "reset":
                    run_slot.reset_completed_ts = now
                    run_slot.phase = "ready"
                elif op_name == "exec_tool":
                    if run_slot.first_step_ts is None:
                        run_slot.first_step_ts = now
                    run_slot.phase = "stepped"
                elif op_name == "evaluate":
                    run_slot.evaluate_completed_ts = now
                    run_slot.phase = "evaluated"
                elif run_slot.in_flight_ops == 0:
                    run_slot.phase = "ready"
            else:
                run_slot.phase = "failed"
            if run_slot.in_flight_ops == 0:
                run_slot.active_op = None

            if run_slot.close_requested and run_slot.in_flight_ops == 0:
                popped = self._pop_run_slot_locked(run_slot.run_lease_id)
                if popped is not None:
                    task_key, popped_slot = popped
                    close_reason = (
                        "Closing run slot after in-flight "
                        f"{op_name}: {popped_slot.close_reason or 'close_requested'}"
                    )
                    close_after = (
                        task_key,
                        popped_slot.run_lease_id,
                        popped_slot,
                        close_reason,
                    )

        if close_after is not None:
            task_key, run_lease_id, slot_to_close, close_reason = close_after
            self._schedule_close(
                task_key,
                run_lease_id,
                slot_to_close,
                reason=close_reason,
            )

    async def _close_run_slot_under_lock(self, run_slot: RunSlot) -> None:
        async with run_slot.lock:
            run_slot.phase = "closing"
            await run_slot.env.close()
            run_slot.phase = "closed"

    def _prune_done_closing_tasks(self) -> int:
        done = {task for task in self._closing_tasks if task.done()}
        self._closing_tasks.difference_update(done)
        for task in done:
            self._closing_task_started.pop(task, None)
            self._closing_task_labels.pop(task, None)
        return len(done)

    async def _close_run_slot_with_semaphore(self, run_slot: RunSlot) -> None:
        async with self._close_sem:
            await self._close_run_slot_under_lock(run_slot)

    async def _force_cleanup_after_close_failure(
        self, run_slot: RunSlot, run_lease_id: str, *, reason: str
    ) -> None:
        timeout = _env_float("WORKER_FORCE_CLEANUP_TIMEOUT", 30.0)
        try:
            logger.warning(
                "Force cleanup starting for run session %s after %s (timeout=%.1fs)",
                run_lease_id,
                reason,
                timeout,
            )
            await asyncio.wait_for(run_slot.env.force_cleanup(reason=reason), timeout=timeout)
            logger.warning(
                "Force cleanup finished for run session %s after %s",
                run_lease_id,
                reason,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Force cleanup timed out for run session %s after %s (timeout=%.1fs)",
                run_lease_id,
                reason,
                timeout,
            )
        except Exception:
            logger.exception(
                "Force cleanup failed after %s for run session %s",
                reason,
                run_lease_id,
            )

    async def _close_run_slot(
        self, task_key: str, run_lease_id: str, run_slot: RunSlot, *, reason: str
    ) -> None:
        logger.warning("%s %s (task=%s)", reason, run_lease_id, task_key)
        try:
            await asyncio.wait_for(
                self._close_run_slot_with_semaphore(run_slot),
                timeout=self.close_task_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Timed out closing run session %s after %.1fs while waiting for "
                "the close semaphore, run lock, and/or env.close(); dropping it "
                "from the pool so the close backlog can drain. Watchdog/preflight "
                "cleanup will remove any orphan Docker objects.",
                run_lease_id,
                self.close_task_timeout,
            )
            await self._force_cleanup_after_close_failure(
                run_slot, run_lease_id, reason="close_timeout"
            )
        except asyncio.CancelledError:
            logger.warning(
                "Close task for run session %s was cancelled; forcing Docker "
                "cleanup before dropping it from the pool.",
                run_lease_id,
            )
            await asyncio.shield(
                self._force_cleanup_after_close_failure(
                    run_slot, run_lease_id, reason="close_cancelled"
                )
            )
            raise
        except Exception:
            logger.exception("Failed to close run session %s", run_lease_id)
            await self._force_cleanup_after_close_failure(
                run_slot, run_lease_id, reason="close_exception"
            )

    def _schedule_close(
        self, task_key: str, run_lease_id: str, run_slot: RunSlot, *, reason: str
    ) -> None:
        task = asyncio.create_task(
            self._close_run_slot(task_key, run_lease_id, run_slot, reason=reason)
        )
        self._closing_tasks.add(task)
        self._closing_task_started[task] = time.time()
        self._closing_task_labels[task] = f"{reason} {run_lease_id} task={task_key}"

        def _on_done(done_task: asyncio.Task) -> None:
            self._closing_tasks.discard(done_task)
            self._closing_task_started.pop(done_task, None)
            self._closing_task_labels.pop(done_task, None)

        task.add_done_callback(_on_done)

    def _reap_idle_locked(self) -> list[tuple[str, str, RunSlot]]:
        now = time.time()
        expired_slots: list[tuple[str, str, RunSlot]] = []

        expired_idem = [
            k
            for k, (_, ts) in self._idempotency.items()
            if now - ts > self.idempotency_ttl
        ]
        for k in expired_idem:
            self._idempotency.pop(k, None)

        for task_key, task_slot in list(self._tasks.items()):
            expired_runs: list[str] = []
            for rid, rslot in task_slot.runs.items():
                if rslot.in_flight_ops > 0 or rslot.lock.locked():
                    continue
                if rslot.close_requested:
                    continue
                if now - rslot.last_used_ts > self.run_idle_ttl:
                    expired_runs.append(rid)

            for rid in expired_runs:
                rslot = task_slot.runs.pop(rid, None)
                self._run_to_task.pop(rid, None)
                if rslot is not None:
                    expired_slots.append((task_key, rid, rslot))

            if task_slot.runs:
                task_slot.last_used_ts = max(
                    r.last_used_ts for r in task_slot.runs.values()
                )
            else:
                logger.info("Reaping empty task slot: %s", task_key)
                self._tasks.pop(task_key, None)

        return expired_slots

    def _get_run_slot(self, run_lease_id: str) -> RunSlot:
        task_key = self._run_to_task.get(run_lease_id)
        if task_key is None:
            raise KeyError(f"Unknown run_lease_id: {run_lease_id}")
        task_slot = self._tasks.get(task_key)
        if task_slot is None:
            raise KeyError(f"Run {run_lease_id} points to missing task slot")
        run_slot = task_slot.runs.get(run_lease_id)
        if run_slot is None:
            raise KeyError(f"Run {run_lease_id} not found in task slot")
        return run_slot

    async def allocate(
        self, task_key: str, request_id: str | None = None
    ) -> dict[str, Any]:
        async with self._lock:
            expired_slots = self._reap_idle_locked()

            if request_id:
                idem_key = (task_key, request_id)
                cached = self._idempotency.get(idem_key)
                if cached is not None:
                    run_lease_id, _ = cached
                    if run_lease_id in self._run_to_task:
                        return {"lease_id": run_lease_id, "reused": True}

            task_slot = self._tasks.get(task_key)
            if task_slot is None:
                if len(self._tasks) >= self.max_tasks:
                    raise CapacityError(
                        "TASK_SLOTS_EXHAUSTED",
                        f"Worker at task capacity: {len(self._tasks)}/{self.max_tasks}",
                    )
                task_slot = TaskSlot(task_key=task_key)
                self._tasks[task_key] = task_slot

            if len(task_slot.runs) >= self.max_runs_per_task:
                raise CapacityError(
                    "RUN_SLOTS_EXHAUSTED",
                    f"Task {task_key} at run capacity: {len(task_slot.runs)}/{self.max_runs_per_task}",
                )

            env = self._new_env()
            run_lease_id = f"run-{uuid.uuid4().hex[:16]}"
            run_slot = RunSlot(run_lease_id=run_lease_id, task_key=task_key, env=env)
            task_slot.runs[run_lease_id] = run_slot
            task_slot.last_used_ts = time.time()
            self._run_to_task[run_lease_id] = task_key

            if request_id:
                self._idempotency[(task_key, request_id)] = (run_lease_id, time.time())

        for tk, rid, rslot in expired_slots:
            self._schedule_close(tk, rid, rslot, reason="Reaping idle run slot")

        return {"lease_id": run_lease_id, "reused": False}

    async def heartbeat(self, run_lease_id: str) -> None:
        run_slot = await self._begin_run_op(run_lease_id, "heartbeat")
        success = False
        try:
            async with run_slot.lock:
                success = True
        finally:
            await self._finish_run_op(run_slot, "heartbeat", success=success)

    async def reset(
        self,
        run_lease_id: str,
        task_meta: dict[str, Any],
        run_ctx_payload: dict[str, Any] | None = None,
        task_timeouts: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(task_meta, dict):
            raise ValueError("task_meta must be a dict")

        run_slot = await self._begin_run_op(run_lease_id, "reset")

        run_ctx = _build_run_ctx(
            run_ctx_payload, default_log_dir=self.output_root / "AgentRunner_Output"
        )
        timeouts = _parse_timeout_overrides(self.default_timeouts, task_timeouts)
        task_spec = _build_task_spec(task_meta)

        success = False
        try:
            async with run_slot.lock:
                user_msg, tool_schemas = await run_slot.env.reset(
                    task_meta=task_meta,
                    task_spec=task_spec,
                    run_ctx=run_ctx,
                    timeouts=timeouts,
                )
                success = True
                return {"user_msg": user_msg, "tool_schemas": tool_schemas}
        finally:
            await self._finish_run_op(run_slot, "reset", success=success)

    async def exec_tool(
        self, run_lease_id: str, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> str:
        run_slot = await self._begin_run_op(run_lease_id, "exec_tool")
        success = False
        try:
            async with run_slot.lock:
                observation = await run_slot.env.exec_tool(tool_name, arguments or {})
                success = True
                return str(observation)
        finally:
            await self._finish_run_op(run_slot, "exec_tool", success=success)

    async def evaluate(
        self, run_lease_id: str, trajectory: dict[str, Any] | None = None
    ) -> tuple[float, dict[str, Any] | None]:
        run_slot = await self._begin_run_op(run_lease_id, "evaluate")
        success = False
        try:
            async with run_slot.lock:
                score = await run_slot.env.evaluate(trajectory)
                details = run_slot.env.last_eval_details()
                success = True
                return float(score), details
        finally:
            await self._finish_run_op(run_slot, "evaluate", success=success)

    async def close_run(self, run_lease_id: str, *, reason: str = "external_close") -> bool:
        close_now: tuple[str, str, RunSlot] | None = None
        async with self._lock:
            task_key = self._run_to_task.get(run_lease_id)
            if task_key is None:
                logger.debug(
                    "close_run: lease %s already gone, nothing to do.", run_lease_id
                )
                return False
            task_slot = self._tasks.get(task_key)
            run_slot = task_slot.runs.get(run_lease_id) if task_slot else None
            if run_slot is None:
                return False

            if run_slot.close_requested:
                logger.info(
                    "close_run: duplicate close ignored lease=%s task=%s phase=%s "
                    "in_flight=%d reason=%s %s",
                    run_lease_id,
                    task_key,
                    run_slot.phase,
                    run_slot.in_flight_ops,
                    run_slot.close_reason,
                    self._run_slot_container_ref(run_slot),
                )
                return True

            run_slot.close_requested = True
            run_slot.close_reason = reason
            run_slot.close_requested_ts = time.time()
            stack = "".join(traceback.format_stack(limit=8))
            logger.warning(
                "close_run requested lease=%s task=%s phase=%s in_flight=%d "
                "first_step=%s evaluate_done=%s reason=%s %s\nClose request stack:\n%s",
                run_lease_id,
                task_key,
                run_slot.phase,
                run_slot.in_flight_ops,
                run_slot.first_step_ts is not None,
                run_slot.evaluate_completed_ts is not None,
                reason,
                self._run_slot_container_ref(run_slot),
                stack,
            )
            if run_slot.in_flight_ops > 0 or run_slot.lock.locked():
                run_slot.phase = "closing_requested"
                return True

            popped = self._pop_run_slot_locked(run_lease_id)
            if popped is not None:
                task_key, run_slot = popped
                close_now = (task_key, run_lease_id, run_slot)

        if close_now is not None:
            task_key, run_lease_id, run_slot = close_now
            self._schedule_close(task_key, run_lease_id, run_slot, reason="Closing run slot")
        return True

    async def status(self) -> dict[str, Any]:
        async with self._lock:
            self._prune_done_closing_tasks()
            now = time.time()
            close_ages = [
                now - started for started in self._closing_task_started.values()
            ]
            pending_close_age_sec = {
                "min": round(min(close_ages), 1) if close_ages else 0.0,
                "max": round(max(close_ages), 1) if close_ages else 0.0,
                "over_close_timeout": sum(
                    1 for age in close_ages if age >= self.close_task_timeout
                ),
            }
            tasks_info: dict[str, Any] = {}
            active_container_ids: set[str] = set()
            active_container_names: set[str] = set()
            active_trial_names: set[str] = set()
            total_runs = 0
            in_flight_runs = 0
            closing_requested_runs = 0
            for tk, ts in self._tasks.items():
                run_details = {}
                for rid, rslot in ts.runs.items():
                    if rslot.in_flight_ops > 0:
                        in_flight_runs += 1
                    if rslot.close_requested:
                        closing_requested_runs += 1
                    container_info = self._run_slot_container_info(rslot)
                    for key in ("id", "short_id"):
                        value = container_info.get(key)
                        if isinstance(value, str) and value:
                            active_container_ids.add(value)
                    container_name = container_info.get("name")
                    if isinstance(container_name, str) and container_name:
                        active_container_names.add(container_name)
                    trial_name = container_info.get("trial_name")
                    if isinstance(trial_name, str) and trial_name:
                        active_trial_names.add(trial_name)
                    run_details[rid] = {
                        "phase": rslot.phase,
                        "in_flight_ops": rslot.in_flight_ops,
                        "active_op": rslot.active_op,
                        "close_requested": rslot.close_requested,
                        "age_sec": round(now - rslot.last_used_ts, 1),
                        "container": container_info,
                    }
                tasks_info[tk] = {"active_runs": len(ts.runs), "runs": run_details}
                total_runs += len(ts.runs)

            return {
                "max_tasks": self.max_tasks,
                "active_tasks": len(self._tasks),
                "max_runs_per_task": self.max_runs_per_task,
                "total_active_runs": total_runs,
                "in_flight_runs": in_flight_runs,
                "closing_requested_runs": closing_requested_runs,
                "pending_closes": len(self._closing_tasks),
                "pending_close_age_sec": pending_close_age_sec,
                "active_container_ids": sorted(active_container_ids),
                "active_container_names": sorted(active_container_names),
                "active_trial_names": sorted(active_trial_names),
                "tasks": tasks_info,
            }

    async def repair_pending_closes(
        self,
        *,
        reason: str,
        max_active_runs: int = 0,
        cancel_timeout: float = 5.0,
        min_age: float | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        if min_age is None:
            min_age = max(0.0, self.close_task_timeout + 5.0)
        async with self._lock:
            pruned_done = self._prune_done_closing_tasks()
            active_runs = sum(len(ts.runs) for ts in self._tasks.values())
            pending_before_cancel = len(self._closing_tasks)
            if active_runs > max_active_runs:
                return {
                    "repaired": False,
                    "reason": "active_runs_above_limit",
                    "active_runs": active_runs,
                    "max_active_runs": max_active_runs,
                    "pending_closes": pending_before_cancel,
                    "pruned_done": pruned_done,
                }
            tasks_to_cancel = [
                task
                for task in self._closing_tasks
                if now - self._closing_task_started.get(task, now) >= min_age
            ]
            skipped_young = pending_before_cancel - len(tasks_to_cancel)

        cancelled = 0
        for task in tasks_to_cancel:
            if not task.done():
                task.cancel()
                cancelled += 1

        if tasks_to_cancel:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                    timeout=max(0.1, cancel_timeout),
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Timed out waiting for pending close task cancellation: "
                    "reason=%s pending=%d timeout=%.1fs",
                    reason,
                    len(tasks_to_cancel),
                    cancel_timeout,
                )

        async with self._lock:
            pruned_after_cancel = self._prune_done_closing_tasks()
            pending_after = len(self._closing_tasks)

        logger.warning(
            "Repaired pending close tasks: reason=%s active_runs=%d "
            "min_age=%.1fs pruned_done=%d cancelled=%d skipped_young=%d "
            "pruned_after_cancel=%d pending_after=%d",
            reason,
            active_runs,
            min_age,
            pruned_done,
            cancelled,
            skipped_young,
            pruned_after_cancel,
            pending_after,
        )
        return {
            "repaired": True,
            "reason": reason,
            "active_runs": active_runs,
            "max_active_runs": max_active_runs,
            "min_age": min_age,
            "pending_before_cancel": pending_before_cancel,
            "pruned_done": pruned_done,
            "cancelled": cancelled,
            "skipped_young": skipped_young,
            "pruned_after_cancel": pruned_after_cancel,
            "pending_after": pending_after,
        }

    async def _force_cleanup_slots(
        self, slots: list[tuple[str, str, RunSlot]], *, reason: str
    ) -> None:
        if not slots:
            return
        timeout = _env_float(
            "WORKER_SHUTDOWN_FORCE_CLEANUP_TIMEOUT",
            max(10.0, min(120.0, len(slots) * 2.0)),
        )
        logger.warning(
            "Shutdown force cleanup starting for %d run slot(s), reason=%s timeout=%.1fs",
            len(slots),
            reason,
            timeout,
        )
        cleanup_tasks = [
            asyncio.create_task(
                self._force_cleanup_after_close_failure(run_slot, run_lease_id, reason=reason)
            )
            for _task_key, run_lease_id, run_slot in slots
        ]
        try:
            await asyncio.wait_for(
                asyncio.gather(*cleanup_tasks, return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Shutdown force cleanup timed out with %d cleanup task(s) still pending",
                sum(1 for task in cleanup_tasks if not task.done()),
            )
            for task in cleanup_tasks:
                task.cancel()
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        logger.warning("Shutdown force cleanup finished for reason=%s", reason)

    async def periodic_reap(self, interval: float = 60.0) -> None:
        while True:
            await asyncio.sleep(interval)
            try:
                async with self._lock:
                    expired_slots = self._reap_idle_locked()
                for tk, rid, rslot in expired_slots:
                    self._schedule_close(
                        tk, rid, rslot, reason="Periodic reaper: idle run slot"
                    )
                if expired_slots:
                    logger.info(
                        "Periodic reaper cleaned up %d idle run slots",
                        len(expired_slots),
                    )
            except Exception:
                logger.exception("Periodic reaper error")

    async def shutdown(self) -> None:
        async with self._lock:
            slots_to_close: list[tuple[str, str, RunSlot]] = []
            for task_key, task_slot in self._tasks.items():
                for run_lease_id, run_slot in task_slot.runs.items():
                    slots_to_close.append((task_key, run_lease_id, run_slot))
            self._tasks.clear()
            self._run_to_task.clear()
            self._idempotency.clear()

        for task_key, run_lease_id, run_slot in slots_to_close:
            self._schedule_close(
                task_key,
                run_lease_id,
                run_slot,
                reason="Closing run slot during shutdown",
            )

        if self._closing_tasks:
            logger.info(
                "Shutdown: waiting for %d pending close tasks...",
                len(self._closing_tasks),
            )
            shutdown_timeout = _env_float(
                "WORKER_SHUTDOWN_CLOSE_TASKS_TIMEOUT",
                max(5.0, self.close_task_timeout + 5.0),
            )
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._closing_tasks, return_exceptions=True),
                    timeout=shutdown_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Shutdown timed out after %.1fs with %d pending close tasks; "
                    "cancelling them and forcing Docker cleanup.",
                    shutdown_timeout,
                    len(self._closing_tasks),
                )
                for task in list(self._closing_tasks):
                    task.cancel()
                try:
                    await asyncio.wait_for(
                        asyncio.gather(
                            *self._closing_tasks, return_exceptions=True
                        ),
                        timeout=5.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Shutdown cancellation wait timed out; exiting with %d "
                        "close task(s) still pending.",
                        len(self._closing_tasks),
                    )
                await self._force_cleanup_slots(
                    slots_to_close,
                    reason="shutdown_close_timeout",
                )
            else:
                await self._force_cleanup_slots(
                    slots_to_close,
                    reason="shutdown_final_sweep",
                )


POOL: WorkerPool | None = None


@app.get("/healthz")
async def healthz() -> JSONResponse:
    try:
        pending_closes = 0
        if POOL is not None:
            pool_status = await POOL.status()
            pending_closes = int(pool_status.get("pending_closes", 0))
        assert_worker_has_capacity_for_docker(
            phase="health", pending_closes=pending_closes
        )
        return JSONResponse({"ok": True})
    except ResourcePressureError as exc:
        return JSONResponse(
            {
                "ok": False,
                "code": exc.code,
                "error": exc.message,
                "details": exc.details,
            },
            status_code=503,
        )


@app.get("/status")
async def status() -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )
    disk: dict[str, Any] | None = None
    pressure: dict[str, Any] | None = None
    disk_ok = True
    disk_error: str | None = None
    pool_status = await POOL.status()
    try:
        disk = docker_data_root_stats()
        pressure = worker_pressure_stats()
        assert_worker_has_capacity_for_docker(
            phase="health",
            pending_closes=int(pool_status.get("pending_closes", 0)),
        )
    except ResourcePressureError as exc:
        disk_ok = False
        disk_error = exc.message
        pressure = exc.details
    except Exception as exc:
        disk_ok = False
        disk_error = str(exc)
    return JSONResponse(
        {
            "ok": True,
            "pool": pool_status,
            "docker_data_root": disk,
            "resource_pressure": pressure,
            "admission_ok": disk_ok,
            "admission_error": disk_error,
        }
    )


@app.post("/repair/pending_closes")
async def repair_pending_closes(request: Request) -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )
    if os.getenv("WORKER_REPAIR_PENDING_CLOSES", "1") != "1":
        return JSONResponse(
            {
                "ok": False,
                "error": "Pending-close repair endpoint is disabled",
                "code": "REPAIR_DISABLED",
            },
            status_code=403,
        )

    data = await json_payload(request)
    reason = str(data.get("reason") or "manual")
    max_active_runs = _env_int("WORKER_REPAIR_PENDING_CLOSES_MAX_ACTIVE_RUNS", 0)
    cancel_timeout = _env_float("WORKER_REPAIR_PENDING_CLOSES_CANCEL_TIMEOUT", 5.0)
    min_age = _env_float(
        "WORKER_REPAIR_PENDING_CLOSES_MIN_AGE",
        max(0.0, POOL.close_task_timeout + 5.0),
    )
    try:
        if "max_active_runs" in data:
            max_active_runs = int(data["max_active_runs"])
    except (TypeError, ValueError):
        pass
    try:
        if "cancel_timeout" in data:
            cancel_timeout = float(data["cancel_timeout"])
    except (TypeError, ValueError):
        pass
    try:
        if "min_age" in data:
            min_age = float(data["min_age"])
    except (TypeError, ValueError):
        pass

    result = await POOL.repair_pending_closes(
        reason=reason,
        max_active_runs=max_active_runs,
        cancel_timeout=cancel_timeout,
        min_age=min_age,
    )
    return JSONResponse({"ok": True, **result})


@app.post("/allocate")
async def allocate(request: Request) -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )

    data = await json_payload(request)
    task_key = data.get("task_key", "")
    request_id = data.get("request_id")

    if not task_key:
        return JSONResponse(
            {"ok": False, "error": "task_key is required"}, status_code=400
        )

    try:
        pool_status = await POOL.status()
        assert_worker_has_capacity_for_docker(
            phase="allocate",
            pending_closes=int(pool_status.get("pending_closes", 0)),
        )
        result = await POOL.allocate(task_key=str(task_key), request_id=request_id)
        return JSONResponse({"ok": True, **result})
    except ResourcePressureError as exc:
        return JSONResponse(
            {
                "ok": False,
                "error": exc.message,
                "code": exc.code,
                "details": exc.details,
            },
            status_code=503,
            headers={"Retry-After": os.getenv("WORKER_PRESSURE_RETRY_AFTER", "10")},
        )
    except CapacityError as exc:
        return JSONResponse(
            {"ok": False, "error": exc.message, "code": exc.code},
            status_code=429,
            headers={"Retry-After": os.getenv("WORKER_CAPACITY_RETRY_AFTER", "5")},
        )
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.post("/heartbeat")
async def heartbeat(request: Request) -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )

    data = await json_payload(request)
    lease_id = data.get("lease_id")
    if not lease_id:
        return JSONResponse(
            {"ok": False, "error": "lease_id is required"}, status_code=400
        )

    try:
        await POOL.heartbeat(str(lease_id))
        return JSONResponse({"ok": True})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.post("/reset")
async def reset(request: Request) -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )

    data = await json_payload(request)
    lease_id = data.get("lease_id")
    task_meta = data.get("task_meta")
    run_ctx_payload = data.get("run_ctx")
    task_timeouts = data.get("task_timeouts")

    if not lease_id:
        return JSONResponse(
            {"ok": False, "error": "lease_id is required"}, status_code=400
        )
    if not isinstance(task_meta, dict):
        return JSONResponse(
            {"ok": False, "error": "task_meta dict is required"}, status_code=400
        )

    try:
        pool_status = await POOL.status()
        assert_worker_has_capacity_for_docker(
            phase="reset",
            pending_closes=int(pool_status.get("pending_closes", 0)),
        )
        out = await POOL.reset(
            run_lease_id=str(lease_id),
            task_meta=task_meta,
            run_ctx_payload=run_ctx_payload,
            task_timeouts=task_timeouts,
        )
        return JSONResponse({"ok": True, **out})
    except ResourcePressureError as exc:
        return JSONResponse(
            {
                "ok": False,
                "error": exc.message,
                "code": exc.code,
                "details": exc.details,
            },
            status_code=503,
        )
    except Exception as exc:
        logger.exception("Reset failed for lease_id=%s", lease_id)
        try:
            await POOL.close_run(str(lease_id), reason="reset_failure")
        except Exception:
            logger.exception("Failed to schedule cleanup after reset failure for %s", lease_id)
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.post("/exec_tool")
async def exec_tool(request: Request) -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )

    data = await json_payload(request)
    lease_id = data.get("lease_id")
    tool_call = data.get("tool_call")

    if not lease_id:
        return JSONResponse(
            {"ok": False, "error": "lease_id is required"}, status_code=400
        )
    if not isinstance(tool_call, dict):
        return JSONResponse(
            {"ok": False, "error": "tool_call dict is required"}, status_code=400
        )

    tool_name = tool_call.get("name")
    arguments = tool_call.get("arguments")

    if not isinstance(tool_name, str) or not tool_name:
        return JSONResponse(
            {"ok": False, "error": "tool_call.name is required"}, status_code=400
        )
    if arguments is not None and not isinstance(arguments, dict):
        return JSONResponse(
            {"ok": False, "error": "tool_call.arguments must be a dict"},
            status_code=400,
        )

    try:
        observation = await POOL.exec_tool(
            str(lease_id), tool_name, arguments=arguments
        )
        return JSONResponse({"ok": True, "observation": observation})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.post("/evaluate")
async def evaluate(request: Request) -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )

    data = await json_payload(request)
    lease_id = data.get("lease_id")
    trajectory = data.get("trajectory")

    if not lease_id:
        return JSONResponse(
            {"ok": False, "error": "lease_id is required"}, status_code=400
        )

    try:
        score, details = await POOL.evaluate(
            str(lease_id), trajectory if isinstance(trajectory, dict) else None
        )
        payload: dict[str, Any] = {"ok": True, "score": score}
        if details is not None:
            payload["details"] = details
        return JSONResponse(payload)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.post("/close")
async def close(request: Request) -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )

    data = await json_payload(request)
    lease_id = data.get("lease_id")
    if not lease_id:
        return JSONResponse(
            {"ok": False, "error": "lease_id is required"}, status_code=400
        )

    try:
        found = await POOL.close_run(str(lease_id), reason="http_close")
        return JSONResponse({"ok": True, "found": found})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


_REAPER_TASK: asyncio.Task | None = None


@app.on_event("startup")
async def _on_startup() -> None:
    global _REAPER_TASK
    if POOL is not None:
        _REAPER_TASK = asyncio.create_task(POOL.periodic_reap(interval=60.0))


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    global POOL, _REAPER_TASK
    if _REAPER_TASK is not None:
        _REAPER_TASK.cancel()
        _REAPER_TASK = None
    if POOL is not None:
        await POOL.shutdown()
        POOL = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C-layer: terminal env worker server")

    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("ENV_SERVER_PORT", "18081"))
    )

    parser.add_argument(
        "--max-tasks", type=int, default=int(os.getenv("WORKER_MAX_TASKS", "16"))
    )
    parser.add_argument(
        "--max-runs-per-task",
        type=int,
        default=int(os.getenv("WORKER_MAX_RUNS_PER_TASK", "8")),
    )
    parser.add_argument(
        "--run-idle-ttl",
        type=int,
        default=int(os.getenv("WORKER_RUN_IDLE_TTL", "600")),
        help="Seconds before an idle RunSlot is reaped",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default=os.getenv("TBENCH_OUTPUT_ROOT", "build_outputs"),
    )

    parser.add_argument(
        "--ensure-image-timeout",
        type=float,
        default=float(os.getenv("ENSURE_IMAGE_TIMEOUT", "300.0")),
    )
    parser.add_argument(
        "--reset-session-timeout",
        type=float,
        default=float(os.getenv("RESET_SESSION_TIMEOUT", "300.0")),
    )
    parser.add_argument(
        "--close-session-timeout",
        type=float,
        default=float(os.getenv("CLOSE_SESSION_TIMEOUT", "60.0")),
    )
    parser.add_argument(
        "--eval-timeout", type=float, default=float(os.getenv("EVAL_TIMEOUT", "600.0"))
    )
    parser.add_argument(
        "--max-concurrent-closes",
        type=int,
        default=int(os.getenv("WORKER_MAX_CONCURRENT_CLOSES", "10")),
        help="Max concurrent Docker stop operations",
    )

    return parser.parse_args()


def main() -> None:
    global POOL
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s %(levelname)s %(name)s] %(message)s"
    )

    POOL = WorkerPool(
        max_tasks=args.max_tasks,
        max_runs_per_task=args.max_runs_per_task,
        run_idle_ttl=args.run_idle_ttl,
        output_root=args.output_root,
        default_timeouts=TaskTimeouts(
            ensure_image=float(args.ensure_image_timeout),
            reset_session=float(args.reset_session_timeout),
            close_session=float(args.close_session_timeout),
            eval=float(args.eval_timeout),
        ),
        max_concurrent_closes=args.max_concurrent_closes,
    )

    logger.info(
        "Starting worker server on %s:%s  max_tasks=%s  max_runs_per_task=%s",
        args.host,
        args.port,
        args.max_tasks,
        args.max_runs_per_task,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
