from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import math
import os
import random
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def _env_bool(name: str, default: bool = False) -> bool:
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
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _parse_betas(raw: str, k: int) -> list[float]:
    values: list[float] = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            continue
    if not values:
        values = [0.0, 0.003, 0.006, 0.01, 0.015, 0.02, 0.03, 0.04]
    if len(values) < k:
        values.extend([values[-1]] * (k - len(values)))
    return values[:k]


def _default_state_path() -> str:
    explicit = (
        os.getenv("EXPLORE_AGENT57_STATE_PATH")
        or os.getenv("EXPLORE_AGENT57_SQLITE_PATH")
        or ""
    ).strip()
    if explicit:
        return explicit

    run_dir = os.getenv("RUN_DIR", "").strip()
    if run_dir:
        return str(Path(run_dir) / "agent57_lite.sqlite3")

    traj_dir = os.getenv("TERMINAL_SAVE_TRAJ_DIR", "").strip()
    if traj_dir:
        return str(Path(traj_dir).parent / "agent57_lite.sqlite3")

    run_id = os.getenv("RUN_ID", "").strip() or f"pid{os.getpid()}"
    safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id)
    return str(Path("/tmp") / f"openclaw_agent57_lite_{safe_run_id}.sqlite3")


@dataclass(frozen=True)
class Agent57LiteConfig:
    enabled: bool
    k: int
    arm_betas: tuple[float, ...]
    combine_mode: str
    ngu_mod_clip: float
    ngu_episodic_source: str
    max_bonus: float
    controller: str
    ucb_c: float
    ucb_window: int
    ucb_epsilon: float
    ucb_min_per_arm: int
    ucb_value: str
    ucb_dataset_aware: bool
    keep_baseline: bool
    lifelong_enabled: bool
    lifelong_coef: float
    lifelong_clip: float
    lifelong_warmup: int
    lifelong_backend: str
    lifelong_key_version: str
    lifelong_include_dataset: bool
    lifelong_include_task: bool
    lifelong_include_turn: bool
    state_path: str
    success_threshold: float

    @property
    def active(self) -> bool:
        return self.enabled or self.lifelong_enabled or self.controller != "fixed"

    def beta_for_arm(self, arm_id: int | None) -> float:
        if not self.arm_betas:
            return 0.0
        try:
            idx = int(arm_id or 0) % len(self.arm_betas)
        except (TypeError, ValueError):
            idx = 0
        return float(self.arm_betas[idx])


def config_from_env() -> Agent57LiteConfig:
    k = max(1, _env_int("EXPLORE_AGENT57_K", 8))
    enabled = _env_bool(
        "EXPLORE_AGENT57_LITE_ENABLED",
        _env_bool("EXPLORE_AGENT57_LITE", False),
    )
    lifelong_enabled = _env_bool(
        "EXPLORE_AGENT57_LIFELONG_ENABLED",
        _env_bool("EXPLORE_AGENT57_LIFELONG", False),
    )
    controller = os.getenv("EXPLORE_AGENT57_CONTROLLER", "fixed").strip().lower()
    if controller not in {"fixed", "ucb"}:
        controller = "fixed"
    backend = (
        os.getenv("EXPLORE_AGENT57_BACKEND")
        or os.getenv("EXPLORE_AGENT57_STATE_BACKEND")
        or os.getenv("EXPLORE_AGENT57_LIFELONG_BACKEND", "local")
    ).strip().lower()
    if backend not in {"local", "sqlite"}:
        backend = "local"
    betas = _parse_betas(os.getenv("EXPLORE_AGENT57_ARM_BETAS", ""), k)
    combine_mode = os.getenv("EXPLORE_AGENT57_COMBINE_MODE", "add").strip().lower()
    if combine_mode not in {"add", "ngu_lite"}:
        combine_mode = "add"
    ngu_episodic_source = (
        os.getenv("EXPLORE_AGENT57_NGU_EPISODIC_SOURCE", "signature_intrinsic")
        .strip()
        .lower()
    )
    if ngu_episodic_source not in {"signature_intrinsic", "intrinsic"}:
        ngu_episodic_source = "signature_intrinsic"
    ucb_value = os.getenv("EXPLORE_AGENT57_UCB_VALUE", "legacy").strip().lower()
    if ucb_value not in {"legacy", "success", "base", "normalized_base"}:
        ucb_value = "legacy"
    key_version = (
        os.getenv("EXPLORE_AGENT57_LIFELONG_KEY_VERSION", "v1").strip().lower()
    )
    if key_version not in {"v1", "v2"}:
        key_version = "v1"
    return Agent57LiteConfig(
        enabled=enabled,
        k=k,
        arm_betas=tuple(betas),
        combine_mode=combine_mode,
        ngu_mod_clip=max(1.0, _env_float("EXPLORE_AGENT57_NGU_MOD_CLIP", 5.0)),
        ngu_episodic_source=ngu_episodic_source,
        max_bonus=max(0.0, _env_float("EXPLORE_AGENT57_MAX_BONUS", 0.0)),
        controller=controller,
        ucb_c=max(0.0, _env_float("EXPLORE_AGENT57_UCB_C", 0.5)),
        ucb_window=max(1, _env_int("EXPLORE_AGENT57_UCB_WINDOW", 256)),
        ucb_epsilon=min(
            1.0,
            max(0.0, _env_float("EXPLORE_AGENT57_UCB_EPSILON", 0.0)),
        ),
        ucb_min_per_arm=max(0, _env_int("EXPLORE_AGENT57_UCB_MIN_PER_ARM", 0)),
        ucb_value=ucb_value,
        ucb_dataset_aware=_env_bool("EXPLORE_AGENT57_UCB_DATASET_AWARE", False),
        keep_baseline=_env_bool("EXPLORE_AGENT57_KEEP_BASELINE", True),
        lifelong_enabled=lifelong_enabled,
        lifelong_coef=max(0.0, _env_float("EXPLORE_AGENT57_LIFELONG_COEF", 0.01)),
        lifelong_clip=max(0.0, _env_float("EXPLORE_AGENT57_LIFELONG_CLIP", 2.0)),
        lifelong_warmup=max(0, _env_int("EXPLORE_AGENT57_LIFELONG_WARMUP", 64)),
        lifelong_backend=backend,
        lifelong_key_version=key_version,
        lifelong_include_dataset=_env_bool(
            "EXPLORE_AGENT57_LIFELONG_INCLUDE_DATASET", True
        ),
        lifelong_include_task=_env_bool(
            "EXPLORE_AGENT57_LIFELONG_INCLUDE_TASK", False
        ),
        lifelong_include_turn=_env_bool(
            "EXPLORE_AGENT57_LIFELONG_INCLUDE_TURN", False
        ),
        state_path=_default_state_path(),
        success_threshold=_env_float("EXPLORE_AGENT57_SUCCESS_THRESHOLD", 0.0),
    )


_LOCAL_LOCK = threading.Lock()
_LOCAL_COUNTS: dict[str, int] = {}
_LOCAL_TRAJ_SEEN = 0
_LOCAL_ARM_EVENTS: list[dict[str, float]] = []
_SQLITE_SCHEMA_LOCK = threading.Lock()
_SQLITE_SCHEMA_INITIALIZED: set[str] = set()


def _normalize_dataset(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9_.-]+", "_", text).strip("._-")
    if text in {"", "terminal_bench", "seta_env"}:
        return "seta"
    if text in {"agent-safety-bench", "agent_safety_bench", "asb", "safety"}:
        return "agent_safetybench"
    if text in {"agent_harm", "ah"}:
        return "agentharm"
    return text or "unknown"


def _connect(path: str) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=5.0, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=5000")
    path_key = str(db_path)
    if path_key not in _SQLITE_SCHEMA_INITIALIZED:
        with _SQLITE_SCHEMA_LOCK:
            if path_key not in _SQLITE_SCHEMA_INITIALIZED:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS lifelong_counts "
                    "(key TEXT PRIMARY KEY, count INTEGER NOT NULL)"
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS meta "
                    "(name TEXT PRIMARY KEY, value INTEGER NOT NULL)"
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS arm_events "
                    "(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, "
                    "arm_id INTEGER NOT NULL, base_score REAL NOT NULL, "
                    "final_score REAL NOT NULL, success INTEGER NOT NULL, "
                    "parse_error INTEGER NOT NULL, truncated INTEGER NOT NULL, "
                    "bonus REAL NOT NULL)"
                )
                columns = {
                    str(row[1])
                    for row in conn.execute("PRAGMA table_info(arm_events)").fetchall()
                }
                if "dataset" not in columns:
                    conn.execute(
                        "ALTER TABLE arm_events "
                        "ADD COLUMN dataset TEXT NOT NULL DEFAULT ''"
                    )
                if "normalized_base_score" not in columns:
                    conn.execute(
                        "ALTER TABLE arm_events "
                        "ADD COLUMN normalized_base_score REAL NOT NULL DEFAULT 0.0"
                    )
                _SQLITE_SCHEMA_INITIALIZED.add(path_key)
    return conn


def _sqlite_next_counts(config: Agent57LiteConfig, keys: Iterable[str]) -> tuple[int, list[int]]:
    unique_keys = list(dict.fromkeys(keys))
    if not unique_keys:
        return 0, []
    conn = _connect(config.state_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT value FROM meta WHERE name='lifelong_traj_seen'"
        ).fetchone()
        seen = int(row[0]) if row else 0
        conn.execute(
            "INSERT INTO meta(name, value) VALUES('lifelong_traj_seen', 1) "
            "ON CONFLICT(name) DO UPDATE SET value=value+1"
        )
        counts_before: list[int] = []
        for key in unique_keys:
            row = conn.execute(
                "SELECT count FROM lifelong_counts WHERE key=?", (key,)
            ).fetchone()
            before = int(row[0]) if row else 0
            counts_before.append(before)
            if row:
                conn.execute(
                    "UPDATE lifelong_counts SET count=count+1 WHERE key=?", (key,)
                )
            else:
                conn.execute(
                    "INSERT INTO lifelong_counts(key, count) VALUES(?, 1)", (key,)
                )
        conn.execute("COMMIT")
        return seen, counts_before
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        conn.close()


def _local_next_counts(keys: Iterable[str]) -> tuple[int, list[int]]:
    global _LOCAL_TRAJ_SEEN
    unique_keys = list(dict.fromkeys(keys))
    with _LOCAL_LOCK:
        seen = _LOCAL_TRAJ_SEEN
        _LOCAL_TRAJ_SEEN += 1
        counts_before: list[int] = []
        for key in unique_keys:
            before = int(_LOCAL_COUNTS.get(key, 0))
            counts_before.append(before)
            _LOCAL_COUNTS[key] = before + 1
    return seen, counts_before


def _stable_hash(text: str, n: int = 12) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


def _metadata_value(metadata: dict[str, Any] | None, *keys: str) -> Any:
    current: Any = metadata if isinstance(metadata, dict) else {}
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    return num if math.isfinite(num) else default


def _normalized_base_score(base_score: float, dataset: str) -> float:
    dataset_name = _normalize_dataset(dataset)
    base = _finite_float(base_score)
    if dataset_name == "seta":
        return min(1.0, max(0.0, base))
    if dataset_name in {"agent_safetybench", "agentharm"}:
        return (min(1.0, max(-1.0, base)) + 1.0) / 2.0
    return min(1.0, max(0.0, base))


def _clamp_bonus(value: float, max_abs: float) -> tuple[float, bool]:
    if max_abs <= 0.0:
        return value, False
    clipped = min(max(value, -max_abs), max_abs)
    return clipped, clipped != value


def _bucket_len(text: str) -> str:
    size = len(text)
    if size == 0:
        return "len0"
    if size < 80:
        return "lenS"
    if size < 512:
        return "lenM"
    if size < 2048:
        return "lenL"
    return "lenXL"


def _result_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        parts = []
        for key in ("stdout", "stderr", "output", "result", "message", "error"):
            val = value.get(key)
            if val:
                parts.append(str(val))
        if parts:
            return "\n".join(parts)
    return str(value)


def exit_code_bucket(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("exit_code", "returncode", "return_code", "code"):
            if key in value and value[key] is not None:
                try:
                    return f"exit{int(value[key])}"
                except (TypeError, ValueError):
                    return f"exit_{str(value[key])[:16]}"
    text = _result_text(value)
    low = text.lower()
    match = re.search(r"(?:exit|return)\s*(?:code|status)?\s*[:=]\s*(-?\d+)", low)
    if match:
        return f"exit{match.group(1)}"
    if "command not found" in low:
        return "exit127"
    if "permission denied" in low:
        return "exit126"
    return "exit_unknown"


def coarse_observation_fingerprint(value: Any) -> str:
    text = _result_text(value)
    low = text.lower()
    if not low.strip():
        return "empty"
    patterns = (
        ("permission_denied", ("permission denied", "operation not permitted")),
        ("not_found", ("no such file", "not found", "cannot stat")),
        ("cmd_not_found", ("command not found",)),
        ("timeout", ("timed out", "timeout")),
        ("traceback", ("traceback", "exception:", "error:")),
        ("assertion", ("assertionerror", "assertion failed")),
        ("test_fail", ("failed", "failure", "tests failed")),
        ("test_pass", ("all tests passed", "passed", "success")),
        ("install", ("apt-get", "pip install", "npm install")),
        ("build", ("building", "compiling", "make:", "cmake")),
    )
    for label, needles in patterns:
        if any(needle in low for needle in needles):
            return f"{label}:{_bucket_len(text)}"
    normalized = re.sub(r"\s+", " ", text.strip())[:512]
    return f"generic:{_bucket_len(text)}:{_stable_hash(normalized, 8)}"


def _turn_result_fingerprints(turn_records: list[dict[str, Any]]) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    for tr in turn_records or []:
        if tr.get("command"):
            result = tr.get("result") or tr.get("observation") or tr.get("output")
            results.append((coarse_observation_fingerprint(result), exit_code_bucket(result)))
        for call in tr.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            result = call.get("result")
            if result is None:
                result = call.get("observation") or call.get("output")
            results.append((coarse_observation_fingerprint(result), exit_code_bucket(result)))
    return results


class LifelongKeyBuilder(ABC):
    """Build stable count keys for Agent57-lite lifelong novelty."""

    @abstractmethod
    def keys(
        self,
        actions: list[dict[str, Any]],
        turn_records: list[dict[str, Any]],
        metadata: dict[str, Any] | None,
    ) -> list[str]:
        raise NotImplementedError


class V1LifelongKeyBuilder(LifelongKeyBuilder):
    """Original key: action signature + coarse observation + exit-code bucket."""

    def keys(
        self,
        actions: list[dict[str, Any]],
        turn_records: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        del metadata
        result_fps = _turn_result_fingerprints(turn_records)
        keys: list[str] = []
        for idx, action in enumerate(actions or []):
            signature = str(action.get("signature") or action.get("raw") or "unknown")
            if idx < len(result_fps):
                obs_fp, exit_fp = result_fps[idx]
            else:
                obs_fp, exit_fp = "no_result", "exit_unknown"
            keys.append(_stable_hash(f"{signature}\n{obs_fp}\n{exit_fp}", 16))
        return keys


def _command_family(action: dict[str, Any]) -> str:
    tool = str(action.get("tool_name") or "tool").strip().lower() or "tool"
    signature = str(action.get("signature") or action.get("raw") or "")
    parts = [part for part in signature.split("|") if part]
    if len(parts) >= 2:
        return re.sub(r"[^a-z0-9_.-]+", "_", f"{tool}:{parts[1].lower()}")[:80]
    raw = str(action.get("raw") or "").strip().lower()
    match = re.match(r"([a-z0-9_.:/-]+)", raw)
    return re.sub(r"[^a-z0-9_.-]+", "_", f"{tool}:{match.group(1) if match else 'unknown'}")[:80]


def _action_flag_text(action: dict[str, Any]) -> str:
    return str(action.get("danger_text") or action.get("raw") or "").lower()


def _is_test_action(action: dict[str, Any]) -> bool:
    text = _action_flag_text(action)
    patterns = (
        "pytest",
        "python -m pytest",
        "unittest",
        "npm test",
        "pnpm test",
        "yarn test",
        "go test",
        "cargo test",
        "make test",
        "run_tests",
        "test_outputs.py",
    )
    return any(pattern in text for pattern in patterns)


def _is_file_mod_action(action: dict[str, Any]) -> bool:
    text = _action_flag_text(action)
    return bool(
        re.search(
            r"(^|\s)(?:touch|mkdir|rm|mv|cp|chmod|chown|tee|sed\s+-i|install)\b",
            text,
        )
        or ">" in text
        or ">>" in text
        or "apply_patch" in text
    )


def _turn_bucket(action: dict[str, Any]) -> str:
    try:
        idx = int(action.get("turn_idx", -1))
    except (TypeError, ValueError):
        idx = -1
    if idx < 0:
        return "turn_unknown"
    if idx == 0:
        return "turn0"
    if idx <= 2:
        return "turn1_2"
    if idx <= 5:
        return "turn3_5"
    return "turn6p"


def _task_bucket(metadata: dict[str, Any] | None) -> str:
    values = (
        _metadata_value(metadata, "task_path")
        or _metadata_value(metadata, "task_name")
        or _metadata_value(metadata, "task_id")
        or _metadata_value(metadata, "task_meta", "task_path")
        or _metadata_value(metadata, "task_meta", "task_name")
        or _metadata_value(metadata, "task_meta", "task_id")
        or ""
    )
    return _stable_hash(str(values), 12) if values else "task_unknown"


class V2LifelongKeyBuilder(LifelongKeyBuilder):
    """Context-aware key before moving to embedding/k-NN novelty."""

    def __init__(self, config: Agent57LiteConfig) -> None:
        self.config = config

    def keys(
        self,
        actions: list[dict[str, Any]],
        turn_records: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        result_fps = _turn_result_fingerprints(turn_records)
        dataset = _normalize_dataset(
            _metadata_value(metadata, "data_source")
            or _metadata_value(metadata, "task_meta", "data_source")
            or _metadata_value(metadata, "agent57_dataset")
        )
        split = str(
            _metadata_value(metadata, "safety_split")
            or _metadata_value(metadata, "task_meta", "safety_split")
            or _metadata_value(metadata, "task_meta", "agentharm_task_type")
            or ""
        ).strip().lower()
        task_bucket = _task_bucket(metadata)

        keys: list[str] = []
        for idx, action in enumerate(actions or []):
            signature = str(action.get("signature") or action.get("raw") or "unknown")
            if idx < len(result_fps):
                obs_fp, exit_fp = result_fps[idx]
            else:
                obs_fp, exit_fp = "no_result", "exit_unknown"
            parts = ["v2"]
            if self.config.lifelong_include_dataset:
                parts.append(f"dataset:{dataset}")
                if split:
                    parts.append(f"split:{split[:80]}")
            if self.config.lifelong_include_task:
                parts.append(f"task:{task_bucket}")
            if self.config.lifelong_include_turn:
                parts.append(f"turn:{_turn_bucket(action)}")
            parts.extend(
                [
                    f"family:{_command_family(action)}",
                    f"test:{int(_is_test_action(action))}",
                    f"filemod:{int(_is_file_mod_action(action))}",
                    f"sig:{signature}",
                    f"obs:{obs_fp}",
                    f"exit:{exit_fp}",
                ]
            )
            keys.append(_stable_hash("\n".join(parts), 16))
        return keys


def _key_builder(config: Agent57LiteConfig | None) -> LifelongKeyBuilder:
    if config is not None and config.lifelong_key_version == "v2":
        return V2LifelongKeyBuilder(config)
    return V1LifelongKeyBuilder()


def lifelong_keys(
    actions: list[dict[str, Any]],
    turn_records: list[dict[str, Any]],
    *,
    config: Agent57LiteConfig | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[str]:
    return _key_builder(config).keys(actions, turn_records, metadata)


def _status_value(status: Any) -> str:
    value = getattr(status, "value", status)
    return str(value).lower()


def compute_lifelong_bonus(
    *,
    config: Agent57LiteConfig,
    arm_id: int,
    actions: list[dict[str, Any]],
    turn_records: list[dict[str, Any]],
    status: Any,
    parse_error_count: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    beta = config.beta_for_arm(arm_id)
    metrics: dict[str, Any] = {
        "explore_agent57_enabled": bool(config.enabled),
        "explore_agent57_arm_id": int(arm_id),
        "explore_agent57_k": int(config.k),
        "explore_agent57_beta": float(beta),
        "explore_agent57_combine_mode": config.combine_mode,
        "explore_agent57_max_bonus": float(config.max_bonus),
        "explore_agent57_controller": config.controller,
        "explore_agent57_ucb_c": float(config.ucb_c),
        "explore_agent57_ucb_window": int(config.ucb_window),
        "explore_agent57_ucb_epsilon": float(config.ucb_epsilon),
        "explore_agent57_ucb_min_per_arm": int(config.ucb_min_per_arm),
        "explore_agent57_ucb_value": config.ucb_value,
        "explore_agent57_ucb_dataset_aware": bool(config.ucb_dataset_aware),
        "explore_agent57_lifelong_enabled": bool(config.lifelong_enabled),
        "explore_agent57_lifelong_backend": config.lifelong_backend,
        "explore_agent57_lifelong_state_path": config.state_path,
        "explore_agent57_lifelong_coef": float(config.lifelong_coef),
        "explore_agent57_lifelong_clip": float(config.lifelong_clip),
        "explore_agent57_lifelong_warmup": int(config.lifelong_warmup),
        "explore_agent57_lifelong_key_version": config.lifelong_key_version,
        "explore_agent57_lifelong_include_dataset": bool(config.lifelong_include_dataset),
        "explore_agent57_lifelong_include_task": bool(config.lifelong_include_task),
        "explore_agent57_lifelong_include_turn": bool(config.lifelong_include_turn),
        "explore_agent57_lifelong_raw": 0.0,
        "explore_agent57_lifelong_bonus": 0.0,
        "explore_agent57_lifelong_bonus_unclipped": 0.0,
        "explore_agent57_lifelong_unique_keys": 0,
        "explore_agent57_lifelong_seen_before": 0,
        "explore_agent57_lifelong_warmup_remaining": int(config.lifelong_warmup),
        "explore_agent57_lifelong_eligible": 0.0,
        "explore_agent57_lifelong_suppressed_reason": "",
        "explore_agent57_bonus_unclipped": 0.0,
        "explore_agent57_bonus_clipped": 0.0,
    }
    if not config.active or not config.lifelong_enabled:
        return metrics
    keys = lifelong_keys(actions, turn_records, config=config, metadata=metadata)
    metrics["explore_agent57_lifelong_unique_keys"] = len(set(keys))
    if not keys:
        metrics["explore_agent57_lifelong_suppressed_reason"] = "no_actions"
        return metrics

    try:
        if config.lifelong_backend == "sqlite":
            seen_before, counts_before = _sqlite_next_counts(config, keys)
        else:
            seen_before, counts_before = _local_next_counts(keys)
    except Exception as exc:
        metrics["explore_agent57_lifelong_suppressed_reason"] = (
            f"state_error:{type(exc).__name__}"
        )
        return metrics

    metrics["explore_agent57_lifelong_seen_before"] = int(seen_before)
    warmup_remaining = max(0, config.lifelong_warmup - seen_before - 1)
    metrics["explore_agent57_lifelong_warmup_remaining"] = int(warmup_remaining)
    if counts_before:
        raw = sum(1.0 / math.sqrt(c + 1.0) for c in counts_before) / len(counts_before)
    else:
        raw = 0.0
    raw = min(config.lifelong_clip, max(0.0, raw)) if config.lifelong_clip > 0 else raw
    metrics["explore_agent57_lifelong_raw"] = float(raw)

    status_text = _status_value(status)
    bad_status = any(part in status_text for part in ("failed", "aborted", "truncated"))
    if bad_status:
        metrics["explore_agent57_lifelong_suppressed_reason"] = f"status:{status_text}"
        return metrics
    if parse_error_count > 0:
        metrics["explore_agent57_lifelong_suppressed_reason"] = "parse_error"
        return metrics
    if seen_before < config.lifelong_warmup:
        metrics["explore_agent57_lifelong_suppressed_reason"] = "warmup"
        return metrics

    metrics["explore_agent57_lifelong_eligible"] = 1.0
    unclipped = float(beta * config.lifelong_coef * raw)
    bonus, clipped = _clamp_bonus(unclipped, config.max_bonus)
    metrics["explore_agent57_lifelong_bonus_unclipped"] = unclipped
    metrics["explore_agent57_lifelong_bonus"] = float(bonus)
    metrics["explore_agent57_bonus_unclipped"] = unclipped
    metrics["explore_agent57_bonus_clipped"] = 1.0 if clipped else 0.0
    return metrics


def compute_ngu_lite_bonus(
    *,
    config: Agent57LiteConfig,
    arm_id: int,
    episodic_novelty: float,
    lifelong_raw: float,
    lifelong_eligible: bool,
) -> dict[str, Any]:
    """Compute the optional NGU-lite product bonus.

    The function is intentionally pure and side-effect free: lifelong count
    updates remain in `compute_lifelong_bonus`, while this combines the current
    rollout's episode novelty with the already-measured lifelong raw signal.
    """
    beta = config.beta_for_arm(arm_id)
    episodic = max(0.0, _finite_float(episodic_novelty))
    raw_life = max(0.0, _finite_float(lifelong_raw))
    life_mod = min(max(1.0 + raw_life, 1.0), config.ngu_mod_clip)
    metrics: dict[str, Any] = {
        "explore_agent57_ngu_mod_clip": float(config.ngu_mod_clip),
        "explore_agent57_ngu_episodic_source": config.ngu_episodic_source,
        "explore_agent57_ngu_episodic": float(episodic),
        "explore_agent57_ngu_life_mod": float(life_mod),
        "explore_agent57_ngu_bonus": 0.0,
        "explore_agent57_ngu_bonus_unclipped": 0.0,
    }
    if (
        not config.active
        or config.combine_mode != "ngu_lite"
        or not config.lifelong_enabled
        or not lifelong_eligible
    ):
        return metrics

    unclipped = float(beta * config.lifelong_coef * episodic * life_mod)
    bonus, clipped = _clamp_bonus(unclipped, config.max_bonus)
    metrics["explore_agent57_ngu_bonus_unclipped"] = unclipped
    metrics["explore_agent57_ngu_bonus"] = float(bonus)
    metrics["explore_agent57_bonus_unclipped"] = unclipped
    metrics["explore_agent57_bonus_clipped"] = 1.0 if clipped else 0.0
    return metrics


def _local_arm_stats(
    k: int,
    window: int,
    *,
    dataset: str | None = None,
) -> list[dict[str, float]]:
    with _LOCAL_LOCK:
        events = list(_LOCAL_ARM_EVENTS[-window:])
    return _aggregate_arm_stats(k, events, dataset=dataset)


def _sqlite_arm_stats(
    config: Agent57LiteConfig,
    *,
    dataset: str | None = None,
) -> list[dict[str, float]]:
    conn = _connect(config.state_path)
    try:
        if dataset:
            rows = conn.execute(
                "SELECT arm_id, base_score, normalized_base_score, success, "
                "parse_error, truncated, dataset FROM arm_events "
                "WHERE dataset=? ORDER BY id DESC LIMIT ?",
                (_normalize_dataset(dataset), config.ucb_window),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT arm_id, base_score, normalized_base_score, success, "
                "parse_error, truncated, dataset FROM arm_events "
                "ORDER BY id DESC LIMIT ?",
                (config.ucb_window,),
            ).fetchall()
    finally:
        conn.close()
    events = [
        {
            "arm_id": int(row[0]),
            "base_score": float(row[1]),
            "normalized_base_score": float(row[2]),
            "success": float(row[3]),
            "parse_error": float(row[4]),
            "truncated": float(row[5]),
            "dataset": str(row[6] or ""),
        }
        for row in rows
    ]
    return _aggregate_arm_stats(config.k, events, dataset=dataset)


def _aggregate_arm_stats(
    k: int,
    events: list[dict[str, float]],
    *,
    dataset: str | None = None,
) -> list[dict[str, float]]:
    target_dataset = _normalize_dataset(dataset) if dataset else ""
    stats = [
        {
            "n": 0.0,
            "base_sum": 0.0,
            "normalized_base_sum": 0.0,
            "success_sum": 0.0,
            "parse_sum": 0.0,
            "trunc_sum": 0.0,
        }
        for _ in range(k)
    ]
    for event in events:
        if target_dataset and _normalize_dataset(event.get("dataset")) != target_dataset:
            continue
        arm_id = int(event.get("arm_id", 0)) % k
        row = stats[arm_id]
        row["n"] += 1.0
        row["base_sum"] += _finite_float(event.get("base_score", 0.0))
        row["normalized_base_sum"] += _finite_float(
            event.get("normalized_base_score", 0.0)
        )
        row["success_sum"] += _finite_float(event.get("success", 0.0))
        row["parse_sum"] += _finite_float(event.get("parse_error", 0.0))
        row["trunc_sum"] += _finite_float(event.get("truncated", 0.0))
    return stats


def _ucb_scores(
    config: Agent57LiteConfig,
    *,
    dataset: str | None = None,
) -> list[tuple[float, int]]:
    target_dataset = _normalize_dataset(dataset) if config.ucb_dataset_aware and dataset else None
    try:
        stats = (
            _sqlite_arm_stats(config, dataset=target_dataset)
            if config.lifelong_backend == "sqlite"
            else _local_arm_stats(config.k, config.ucb_window, dataset=target_dataset)
        )
    except Exception:
        stats = _aggregate_arm_stats(config.k, [], dataset=target_dataset)
    total = max(1.0, sum(row["n"] for row in stats))
    scored: list[tuple[float, int]] = []
    for arm_id, row in enumerate(stats):
        n = row["n"]
        if n <= 0 or n < config.ucb_min_per_arm:
            score = float("inf")
        else:
            mean_success = row["success_sum"] / n
            mean_base = row["base_sum"] / n
            mean_normalized_base = row["normalized_base_sum"] / n
            parse_rate = row["parse_sum"] / n
            trunc_rate = row["trunc_sum"] / n
            if config.ucb_value == "success":
                value = mean_success - 0.5 * parse_rate - 0.5 * trunc_rate
            elif config.ucb_value == "base":
                value = mean_base - 0.5 * parse_rate - 0.5 * trunc_rate
            elif config.ucb_value == "normalized_base":
                value = mean_normalized_base - 0.5 * parse_rate - 0.5 * trunc_rate
            else:
                value = mean_success + 0.25 * mean_base - 0.5 * parse_rate - 0.5 * trunc_rate
            score = value + config.ucb_c * math.sqrt(math.log(total + 1.0) / n)
        scored.append((score, arm_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored


def assign_group_arms(
    group_size: int,
    *,
    evaluation: bool = False,
    dataset: str | None = None,
) -> list[int]:
    config = config_from_env()
    if evaluation or not config.active:
        return [0 for _ in range(max(0, group_size))]
    group_size = max(0, int(group_size))
    if group_size == 0:
        return []
    if config.controller == "ucb":
        arms: list[int] = []
        if config.keep_baseline:
            arms.append(0)
        ranked = [arm for _, arm in _ucb_scores(config, dataset=dataset)]
        if config.keep_baseline and group_size > 1:
            ranked = [arm for arm in ranked if arm != 0]
            if not ranked:
                ranked = [arm for arm in range(config.k) if arm != 0] or [0]
        cursor = 0
        while len(arms) < group_size:
            if config.ucb_epsilon > 0.0 and random.random() < config.ucb_epsilon:
                if config.keep_baseline and group_size > 1 and config.k > 1:
                    arms.append(random.randrange(1, config.k))
                else:
                    arms.append(random.randrange(0, config.k))
            else:
                arms.append(ranked[cursor % len(ranked)] if ranked else len(arms) % config.k)
            cursor += 1
        return arms[:group_size]
    return [idx % config.k for idx in range(group_size)]


def record_arm_event(
    *,
    config: Agent57LiteConfig,
    arm_id: int,
    base_score: float,
    final_score: float,
    status: Any,
    parse_error_count: int,
    bonus: float,
    dataset: str | None = None,
) -> None:
    if not config.active:
        return
    status_text = _status_value(status)
    truncated = 1 if "truncated" in status_text else 0
    base = _finite_float(base_score)
    final = _finite_float(final_score, base)
    shaped_bonus = _finite_float(bonus)
    dataset_name = _normalize_dataset(dataset)
    normalized_base = _normalized_base_score(base, dataset_name)
    success = 1 if base > config.success_threshold else 0
    event = {
        "arm_id": float(int(arm_id) % max(1, config.k)),
        "base_score": base,
        "normalized_base_score": normalized_base,
        "final_score": final,
        "success": float(success),
        "parse_error": float(1 if parse_error_count > 0 else 0),
        "truncated": float(truncated),
        "bonus": shaped_bonus,
        "dataset": dataset_name,
    }
    if config.lifelong_backend == "sqlite":
        try:
            conn = _connect(config.state_path)
            try:
                conn.execute(
                    "INSERT INTO arm_events"
                    "(ts, arm_id, base_score, normalized_base_score, final_score, "
                    "success, parse_error, truncated, bonus, dataset) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        time.time(),
                        int(event["arm_id"]),
                        event["base_score"],
                        event["normalized_base_score"],
                        event["final_score"],
                        int(event["success"]),
                        int(event["parse_error"]),
                        int(event["truncated"]),
                        event["bonus"],
                        event["dataset"],
                    ),
                )
            finally:
                conn.close()
        except Exception:
            return
        return
    with _LOCAL_LOCK:
        _LOCAL_ARM_EVENTS.append(event)
        if len(_LOCAL_ARM_EVENTS) > 10000:
            del _LOCAL_ARM_EVENTS[: len(_LOCAL_ARM_EVENTS) - 10000]
