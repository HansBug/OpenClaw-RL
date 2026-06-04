from __future__ import annotations

import hashlib
import math
import os
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
    controller: str
    ucb_c: float
    ucb_window: int
    keep_baseline: bool
    lifelong_enabled: bool
    lifelong_coef: float
    lifelong_clip: float
    lifelong_warmup: int
    lifelong_backend: str
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
    return Agent57LiteConfig(
        enabled=enabled,
        k=k,
        arm_betas=tuple(betas),
        controller=controller,
        ucb_c=max(0.0, _env_float("EXPLORE_AGENT57_UCB_C", 0.5)),
        ucb_window=max(1, _env_int("EXPLORE_AGENT57_UCB_WINDOW", 256)),
        keep_baseline=_env_bool("EXPLORE_AGENT57_KEEP_BASELINE", True),
        lifelong_enabled=lifelong_enabled,
        lifelong_coef=max(0.0, _env_float("EXPLORE_AGENT57_LIFELONG_COEF", 0.01)),
        lifelong_clip=max(0.0, _env_float("EXPLORE_AGENT57_LIFELONG_CLIP", 2.0)),
        lifelong_warmup=max(0, _env_int("EXPLORE_AGENT57_LIFELONG_WARMUP", 64)),
        lifelong_backend=backend,
        state_path=_default_state_path(),
        success_threshold=_env_float("EXPLORE_AGENT57_SUCCESS_THRESHOLD", 0.0),
    )


_LOCAL_LOCK = threading.Lock()
_LOCAL_COUNTS: dict[str, int] = {}
_LOCAL_TRAJ_SEEN = 0
_LOCAL_ARM_EVENTS: list[dict[str, float]] = []
_SQLITE_SCHEMA_LOCK = threading.Lock()
_SQLITE_SCHEMA_INITIALIZED: set[str] = set()


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


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    return num if math.isfinite(num) else default


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


def lifelong_keys(
    actions: list[dict[str, Any]],
    turn_records: list[dict[str, Any]],
) -> list[str]:
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
) -> dict[str, Any]:
    beta = config.beta_for_arm(arm_id)
    metrics: dict[str, Any] = {
        "explore_agent57_enabled": bool(config.enabled),
        "explore_agent57_arm_id": int(arm_id),
        "explore_agent57_k": int(config.k),
        "explore_agent57_beta": float(beta),
        "explore_agent57_controller": config.controller,
        "explore_agent57_lifelong_enabled": bool(config.lifelong_enabled),
        "explore_agent57_lifelong_backend": config.lifelong_backend,
        "explore_agent57_lifelong_state_path": config.state_path,
        "explore_agent57_lifelong_coef": float(config.lifelong_coef),
        "explore_agent57_lifelong_clip": float(config.lifelong_clip),
        "explore_agent57_lifelong_warmup": int(config.lifelong_warmup),
        "explore_agent57_lifelong_raw": 0.0,
        "explore_agent57_lifelong_bonus": 0.0,
        "explore_agent57_lifelong_unique_keys": 0,
        "explore_agent57_lifelong_seen_before": 0,
        "explore_agent57_lifelong_warmup_remaining": int(config.lifelong_warmup),
        "explore_agent57_lifelong_eligible": 0.0,
        "explore_agent57_lifelong_suppressed_reason": "",
    }
    if not config.active or not config.lifelong_enabled:
        return metrics
    keys = lifelong_keys(actions, turn_records)
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
    metrics["explore_agent57_lifelong_bonus"] = float(beta * config.lifelong_coef * raw)
    return metrics


def _local_arm_stats(k: int, window: int) -> list[dict[str, float]]:
    with _LOCAL_LOCK:
        events = list(_LOCAL_ARM_EVENTS[-window:])
    return _aggregate_arm_stats(k, events)


def _sqlite_arm_stats(config: Agent57LiteConfig) -> list[dict[str, float]]:
    conn = _connect(config.state_path)
    try:
        rows = conn.execute(
            "SELECT arm_id, base_score, success, parse_error, truncated "
            "FROM arm_events ORDER BY id DESC LIMIT ?",
            (config.ucb_window,),
        ).fetchall()
    finally:
        conn.close()
    events = [
        {
            "arm_id": int(row[0]),
            "base_score": float(row[1]),
            "success": float(row[2]),
            "parse_error": float(row[3]),
            "truncated": float(row[4]),
        }
        for row in rows
    ]
    return _aggregate_arm_stats(config.k, events)


def _aggregate_arm_stats(k: int, events: list[dict[str, float]]) -> list[dict[str, float]]:
    stats = [
        {"n": 0.0, "base_sum": 0.0, "success_sum": 0.0, "parse_sum": 0.0, "trunc_sum": 0.0}
        for _ in range(k)
    ]
    for event in events:
        arm_id = int(event.get("arm_id", 0)) % k
        row = stats[arm_id]
        row["n"] += 1.0
        row["base_sum"] += _finite_float(event.get("base_score", 0.0))
        row["success_sum"] += _finite_float(event.get("success", 0.0))
        row["parse_sum"] += _finite_float(event.get("parse_error", 0.0))
        row["trunc_sum"] += _finite_float(event.get("truncated", 0.0))
    return stats


def _ucb_scores(config: Agent57LiteConfig) -> list[tuple[float, int]]:
    try:
        stats = (
            _sqlite_arm_stats(config)
            if config.lifelong_backend == "sqlite"
            else _local_arm_stats(config.k, config.ucb_window)
        )
    except Exception:
        stats = _aggregate_arm_stats(config.k, [])
    total = max(1.0, sum(row["n"] for row in stats))
    scored: list[tuple[float, int]] = []
    for arm_id, row in enumerate(stats):
        n = row["n"]
        if n <= 0:
            score = float("inf")
        else:
            mean_success = row["success_sum"] / n
            mean_base = row["base_sum"] / n
            parse_rate = row["parse_sum"] / n
            trunc_rate = row["trunc_sum"] / n
            value = mean_success + 0.25 * mean_base - 0.5 * parse_rate - 0.5 * trunc_rate
            score = value + config.ucb_c * math.sqrt(math.log(total + 1.0) / n)
        scored.append((score, arm_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored


def assign_group_arms(group_size: int, *, evaluation: bool = False) -> list[int]:
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
        ranked = [arm for _, arm in _ucb_scores(config)]
        if config.keep_baseline and group_size > 1:
            ranked = [arm for arm in ranked if arm != 0]
            if not ranked:
                ranked = [arm for arm in range(config.k) if arm != 0] or [0]
        cursor = 0
        while len(arms) < group_size:
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
) -> None:
    if not config.active:
        return
    status_text = _status_value(status)
    truncated = 1 if "truncated" in status_text else 0
    base = _finite_float(base_score)
    final = _finite_float(final_score, base)
    shaped_bonus = _finite_float(bonus)
    success = 1 if base > config.success_threshold else 0
    event = {
        "arm_id": float(int(arm_id) % max(1, config.k)),
        "base_score": base,
        "final_score": final,
        "success": float(success),
        "parse_error": float(1 if parse_error_count > 0 else 0),
        "truncated": float(truncated),
        "bonus": shaped_bonus,
    }
    if config.lifelong_backend == "sqlite":
        try:
            conn = _connect(config.state_path)
            try:
                conn.execute(
                    "INSERT INTO arm_events"
                    "(ts, arm_id, base_score, final_score, success, parse_error, truncated, bonus) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        time.time(),
                        int(event["arm_id"]),
                        event["base_score"],
                        event["final_score"],
                        int(event["success"]),
                        int(event["parse_error"]),
                        int(event["truncated"]),
                        event["bonus"],
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
