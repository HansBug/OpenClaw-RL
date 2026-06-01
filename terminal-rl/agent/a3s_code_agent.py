from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from custom_types import Interaction, TurnResult
from inference_client import SGLangTurnClient

logger = logging.getLogger(__name__)


@dataclass
class A3SCodeModelResponse:
    text: str
    tool_calls: List[dict[str, Any]]
    tool_calls_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    raw_result: Any = None


@dataclass
class A3SCodeFinalResponse:
    msg: str
    terminated: bool
    info: dict[str, Any]


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %d", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %.2f", name, raw, default)
        return default


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _bootstrap_a3s_code() -> tuple[Any, Any]:
    try:
        from a3s_code import Agent, SessionOptions

        return Agent, SessionOptions
    except ImportError:
        pass

    repo_root = Path(
        os.getenv(
            "A3S_CODE_REPO_ROOT",
            str(_repo_root().parent / "a3s-lab" / "Code"),
        )
    )
    sdk_python = repo_root / "sdk" / "python"
    version_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        Path(sys.prefix) / "lib" / version_dir / "site-packages",
        Path(os.getenv("CONDA_PREFIX", "")) / "lib" / version_dir / "site-packages",
        sdk_python / ".venv" / "lib" / "python3.13" / "site-packages",
        sdk_python / ".venv" / "lib" / "python3.12" / "site-packages",
    ]
    candidates.extend(
        Path(item).expanduser()
        for item in os.getenv("A3S_CODE_EXTRA_SITE_PACKAGES", "").split(":")
        if item.strip()
    )
    for site in candidates:
        if (site / "a3s_code").exists():
            sys.path.insert(0, str(site))
            from a3s_code import Agent, SessionOptions

            return Agent, SessionOptions

    raise RuntimeError(
        "a3s_code is not importable. Set A3S_CODE_REPO_ROOT or "
        "A3S_CODE_EXTRA_SITE_PACKAGES, or build the SDK before running "
        "HARNESS_OPTION=a3s-code."
    )


def _default_config_path() -> Path:
    return (
        _repo_root()
        / "a3s-code-adapter"
        / "generated_configs"
        / "a3s-code-shared.hcl"
    )


def _text_from_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return "" if content is None else str(content)


def _last_user_text(messages: List[dict[str, Any]]) -> str:
    for message in reversed(messages or []):
        if str(message.get("role", "")).lower() == "user":
            text = _text_from_message_content(message.get("content"))
            if text:
                return text
    return ""


def _tokenize(tokenizer: Any, text: str) -> list[int]:
    if tokenizer is None or not text:
        return []
    try:
        encoded = tokenizer(text, add_special_tokens=False)
        if isinstance(encoded, dict):
            return list(encoded.get("input_ids") or [])
    except Exception:
        pass
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return []


class A3SCodeAgent:
    """A3S Code SDK-backed harness for terminal-rl rollouts.

    The SDK owns code-workspace tool execution inside `session.send()`, so this
    adapter returns no external `tool_call_requests`. The surrounding terminal-rl
    pipeline still evaluates the final response, logs trajectories, and releases
    the terminal lease in the usual `generate.py` finally block.
    """

    def __init__(
        self,
        *,
        model_type: str,
        sglang_client: SGLangTurnClient,
        env_client: Any | None,
        lease_id: str | None,
        run_context: Any | None,
        task_meta: Dict[str, Any],
        max_total_tokens: int,
    ) -> None:
        _ = (model_type, env_client, lease_id, max_total_tokens)
        self._sglang_client = sglang_client
        self._run_context = run_context
        self._task_meta = task_meta or {}
        self._input_message = ""
        self._session = None
        self._agent = None
        self._session_id = ""
        self._workspace = self._resolve_workspace()
        self._max_parse_errors = 3
        self.parse_error_count = 0

    def _resolve_workspace(self) -> Path:
        raw = os.getenv("A3S_CODE_WORKSPACE")
        if raw:
            path = Path(raw).expanduser()
        else:
            uid = getattr(self._run_context, "uid", None) or uuid.uuid4().hex[:8]
            root = Path(
                os.getenv(
                    "A3S_CODE_WORKSPACE_ROOT",
                    str(_repo_root() / "runs" / "a3s_code_workspaces"),
                )
            )
            path = root / f"a3s-code-{uid}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _session_options(self, SessionOptions: Any) -> Any:
        opts = SessionOptions()
        self._session_id = os.getenv("A3S_CODE_SESSION_ID") or (
            f"a3s-code-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        )
        opts.session_id = self._session_id
        opts.builtin_skills = _env_flag("A3S_CODE_BUILTIN_SKILLS", True)
        opts.auto_compact = _env_flag("A3S_CODE_AUTO_COMPACT", True)
        opts.auto_compact_threshold = _env_float("A3S_CODE_AUTO_COMPACT_THRESHOLD", 0.85)
        opts.tool_timeout_ms = _env_int("A3S_CODE_TOOL_TIMEOUT_MS", 240000)
        opts.max_parse_retries = _env_int("A3S_CODE_MAX_PARSE_RETRIES", 4)
        opts.max_tool_rounds = _env_int("A3S_CODE_MAX_TOOL_ROUNDS", 8)
        opts.circuit_breaker_threshold = _env_int("A3S_CODE_CIRCUIT_BREAKER_THRESHOLD", 5)
        thinking_budget = _env_int("A3S_CODE_THINKING_BUDGET", 12000)
        opts.thinking_budget = thinking_budget if thinking_budget > 0 else None
        opts.continuation_enabled = _env_flag("A3S_CODE_CONTINUATION_ENABLED", True)
        opts.max_continuation_turns = _env_int("A3S_CODE_MAX_CONTINUATION_TURNS", 5)
        return opts

    def _ensure_session(self) -> Any:
        if self._session is not None:
            return self._session

        Agent, SessionOptions = _bootstrap_a3s_code()
        config_path = Path(os.getenv("A3S_CODE_CONFIG_PATH", str(_default_config_path())))
        if not config_path.exists():
            raise FileNotFoundError(
                f"A3S_CODE_CONFIG_PATH={config_path} not found. "
                "Set A3S_CODE_CONFIG_PATH or generate a3s-code shared config first."
            )
        self._agent = Agent.create(str(config_path))
        opts = self._session_options(SessionOptions)
        self._session = self._agent.session(
            str(self._workspace),
            opts,
            permissive=_env_flag("A3S_CODE_PERMISSIVE", True),
        )
        return self._session

    def set_max_parse_errors(self, max_parse_errors: int) -> None:
        self._max_parse_errors = max(1, int(max_parse_errors))

    def start_turn_loop(self, input_message: Any) -> None:
        self.parse_error_count = 0
        self._input_message = _text_from_message_content(input_message)

    async def get_turn_context(self) -> tuple[Optional[List[dict[str, Any]]], Optional[Any]]:
        return [{"role": "user", "content": self._input_message}], None

    async def run_model_turn(
        self,
        *,
        context_messages: List[dict[str, Any]],
        sglang_client: SGLangTurnClient,
        tool_schemas: List[Dict[str, Any]],
        turn_idx: int,
    ) -> TurnResult:
        _ = (sglang_client, tool_schemas)
        prompt = _last_user_text(context_messages) or self._input_message
        session = self._ensure_session()
        timeout = _env_float("A3S_CODE_TURN_TIMEOUT_SEC", 240.0)
        try:
            raw_result = await asyncio.wait_for(
                asyncio.to_thread(session.send, prompt),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            cancel = getattr(session, "cancel", None)
            if callable(cancel):
                try:
                    cancel()
                except Exception:
                    pass
            raise TimeoutError(f"a3s-code session.send timed out after {timeout:.0f}s")

        output_text = str(getattr(raw_result, "text", "") or "")
        tool_calls = list(getattr(raw_result, "tool_calls", []) or [])
        tool_calls_count = int(
            getattr(raw_result, "tool_calls_count", len(tool_calls)) or 0
        )
        prompt_tokens = int(getattr(raw_result, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(raw_result, "completion_tokens", 0) or 0)
        total_tokens = int(
            getattr(raw_result, "total_tokens", prompt_tokens + completion_tokens) or 0
        )

        tokenizer = getattr(self._sglang_client, "tokenizer", None)
        input_ids = _tokenize(tokenizer, prompt)
        output_ids = _tokenize(tokenizer, output_text)
        interaction = Interaction(
            turn_idx=turn_idx,
            input_ids=input_ids,
            output_token_ids=output_ids,
            output_token_logprobs=[0.0] * len(output_ids),
            output_text=output_text,
            finish_reason=str(getattr(raw_result, "finish_reason", "stop") or "stop"),
            messages=context_messages,
            latency_ms=float(getattr(raw_result, "latency_ms", 0.0) or 0.0),
        )
        model_response = A3SCodeModelResponse(
            text=output_text,
            tool_calls=tool_calls,
            tool_calls_count=tool_calls_count,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw_result=raw_result,
        )
        return TurnResult(
            interaction=interaction,
            interactions=[interaction],
            model_response=model_response,
            tool_call_requests=[],
            parse_error_recorded=False,
        )

    async def consume_completion(
        self, chat_completion: Any
    ) -> tuple[Optional[Any], List[Any], bool, Optional[Any]]:
        _ = chat_completion
        raise RuntimeError("A3SCodeAgent consumes completions inside run_model_turn().")

    def record_tool_result(self, tool_call_request: Any, raw_result: Any) -> None:
        _ = (tool_call_request, raw_result)

    def finalize_response(self, model_response: Any) -> A3SCodeFinalResponse:
        text = str(getattr(model_response, "text", "") or "")
        return A3SCodeFinalResponse(
            msg=text,
            terminated=False,
            info={
                "termination_reasons": [],
                "harness_option": "a3s-code",
                "session_id": self._session_id,
                "workspace": str(self._workspace),
                "task_path": self._task_meta.get("task_path"),
                "tool_calls_count": getattr(model_response, "tool_calls_count", 0),
                "prompt_tokens": getattr(model_response, "prompt_tokens", 0),
                "completion_tokens": getattr(model_response, "completion_tokens", 0),
                "total_tokens": getattr(model_response, "total_tokens", 0),
            },
        )

    async def close(self) -> None:
        session = self._session
        self._session = None
        if session is None:
            return
        close_fn = getattr(session, "close", None)
        if callable(close_fn):
            result = close_fn()
            if asyncio.iscoroutine(result):
                await result
