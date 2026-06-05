from __future__ import annotations

import asyncio
import importlib
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@dataclass(frozen=True)
class _TaskSpec:
    task_name: str
    task_path: str
    instruction: str


@dataclass(frozen=True)
class _RunContext:
    uid: str
    group_index: int
    sample_index: int
    log_dir: Path
    rollout_id: int | None = None
    train_step: int | None = None
    rollout_step: int | None = None


@dataclass
class _TaskTimeouts:
    ensure_image: float = 300.0
    reset_session: float = 300.0
    close_session: float = 60.0
    eval: float = 600.0


class _FastAPI:
    def get(self, *_args, **_kwargs):
        return lambda fn: fn

    def post(self, *_args, **_kwargs):
        return lambda fn: fn

    def on_event(self, *_args, **_kwargs):
        return lambda fn: fn


class _JSONResponse(dict):
    def __init__(self, content=None, status_code=200):
        super().__init__(content or {})
        self.status_code = status_code


class _DummyEnv:
    def __init__(self) -> None:
        self.reset_started = asyncio.Event()
        self.release_reset = asyncio.Event()
        self.close_count = 0

    async def reset(self, **_kwargs):
        self.reset_started.set()
        await self.release_reset.wait()
        return "user", []

    async def exec_tool(self, _tool_name, _arguments):
        return "observation"

    async def evaluate(self, _trajectory=None):
        return 1.0

    def last_eval_details(self):
        return None

    async def close(self):
        self.close_count += 1

    async def force_cleanup(self, reason="external"):
        self.force_cleanup_reason = reason


def _install_import_stubs(monkeypatch):
    fastapi_mod = types.ModuleType("fastapi")
    fastapi_mod.FastAPI = _FastAPI
    fastapi_mod.Request = object
    responses_mod = types.ModuleType("fastapi.responses")
    responses_mod.JSONResponse = _JSONResponse

    custom_types_mod = types.ModuleType("terminal-rl.custom_types")
    custom_types_mod.TaskSpec = _TaskSpec
    custom_types_mod.RunContext = _RunContext
    custom_types_mod.TaskTimeouts = _TaskTimeouts

    request_utils_mod = types.ModuleType("terminal-rl.request_utils")

    async def _json_payload(_request):
        return {}

    request_utils_mod.json_payload = _json_payload

    terminal_env_mod = types.ModuleType("terminal-rl.remote.terminal_env")
    terminal_env_mod.TerminalEnv = _DummyEnv

    monkeypatch.setitem(sys.modules, "uvicorn", types.ModuleType("uvicorn"))
    monkeypatch.setitem(sys.modules, "fastapi", fastapi_mod)
    monkeypatch.setitem(sys.modules, "fastapi.responses", responses_mod)
    monkeypatch.setitem(sys.modules, "terminal-rl.custom_types", custom_types_mod)
    monkeypatch.setitem(sys.modules, "terminal-rl.request_utils", request_utils_mod)
    monkeypatch.setitem(sys.modules, "terminal-rl.remote.terminal_env", terminal_env_mod)
    sys.modules.pop("terminal-rl.remote.pool_server", None)
    return importlib.import_module("terminal-rl.remote.pool_server")


def _new_pool(pool_server_mod, env: _DummyEnv, tmp_path: Path):
    class TestWorkerPool(pool_server_mod.WorkerPool):
        def _new_env(self):
            return env

    return TestWorkerPool(
        max_tasks=4,
        max_runs_per_task=4,
        run_idle_ttl=1,
        output_root=str(tmp_path),
        default_timeouts=_TaskTimeouts(),
        max_concurrent_closes=2,
    )


def test_close_allocated_run_cleans_up_without_unpack_error(monkeypatch, tmp_path):
    async def _case():
        pool_server = _install_import_stubs(monkeypatch)
        env = _DummyEnv()
        pool = _new_pool(pool_server, env, tmp_path)

        lease = await pool.allocate("task")
        lease_id = lease["lease_id"]

        assert await pool.close_run(lease_id, reason="test_close") is True
        if pool._closing_tasks:
            await asyncio.gather(*pool._closing_tasks, return_exceptions=False)

        assert lease_id not in pool._run_to_task
        assert env.close_count == 1

    asyncio.run(_case())


def test_idle_reaper_skips_in_flight_reset(monkeypatch, tmp_path):
    async def _case():
        pool_server = _install_import_stubs(monkeypatch)
        env = _DummyEnv()
        pool = _new_pool(pool_server, env, tmp_path)

        lease = await pool.allocate("task")
        lease_id = lease["lease_id"]
        reset_task = asyncio.create_task(
            pool.reset(
                lease_id,
                {"task_name": "task", "task_path": "task", "instruction": "do it"},
                {"uid": "u1", "log_dir": str(tmp_path)},
            )
        )
        await env.reset_started.wait()

        async with pool._lock:
            run_slot = pool._get_run_slot(lease_id)
            run_slot.last_used_ts = time.time() - 100
            expired = pool._reap_idle_locked()

        assert expired == []
        assert lease_id in pool._run_to_task
        assert env.close_count == 0

        env.release_reset.set()
        await reset_task
        assert lease_id in pool._run_to_task
        assert env.close_count == 0

    asyncio.run(_case())


def test_close_during_reset_is_deferred_until_in_flight_finishes(monkeypatch, tmp_path):
    async def _case():
        pool_server = _install_import_stubs(monkeypatch)
        env = _DummyEnv()
        pool = _new_pool(pool_server, env, tmp_path)

        lease = await pool.allocate("task")
        lease_id = lease["lease_id"]
        reset_task = asyncio.create_task(
            pool.reset(
                lease_id,
                {"task_name": "task", "task_path": "task", "instruction": "do it"},
                {"uid": "u1", "log_dir": str(tmp_path)},
            )
        )
        await env.reset_started.wait()

        assert await pool.close_run(lease_id, reason="test_close") is True
        async with pool._lock:
            run_slot = pool._get_run_slot(lease_id)
            assert run_slot.close_requested is True
            assert run_slot.in_flight_ops == 1
            assert run_slot.phase == "closing_requested"

        env.release_reset.set()
        await reset_task
        if pool._closing_tasks:
            await asyncio.gather(*pool._closing_tasks, return_exceptions=True)

        assert lease_id not in pool._run_to_task
        assert env.close_count == 1

    asyncio.run(_case())
