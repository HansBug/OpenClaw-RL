#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import collections
import csv
import gzip
import hashlib
import json
import math
import re
import shutil
import statistics
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ISSUE25_BASE_ALIGNED = {
    "label": "issue #25：Qwen3-8B-base × TB2.0 mode B aligned",
    "issue_url": "https://github.com/HansBug/OpenClaw-RL/issues/25",
    "n_trials": 267,
    "n_tasks": 89,
    "reward_sum": 3,
    "pass_at_1": 3 / 267,
    "pass_at_3": 2 / 89,
    "n_solved_tasks": 2,
    "solved_tasks": ["hf-model-inference", "modernize-scientific-stack"],
    "status_counts": {"COMPLETED": 119, "TRUNCATED": 147, "FAILED": 0, "ABORTED": 1},
    "mean_agent_steps": 7.52,
    "think_trials": 252,
    "think_blocks": 1978,
    "sglang_tp": 8,
    "sglang_version": "0.5.10.post1",
}

ISSUE27_BASE_TBV21_ALIGNED = {
    "label": "issue #27：Qwen3-8B-base × TBv2.1 aligned",
    "issue_url": "https://github.com/HansBug/OpenClaw-RL/issues/27",
    "n_trials": 267,
    "n_tasks": 89,
    "reward_sum": 5,
    "pass_at_1": 5 / 267,
    "pass_at_3": 3 / 89,
    "n_solved_tasks": 3,
    "solved_tasks": ["filter-js-from-html", "modernize-scientific-stack", "query-optimize"],
    "status_counts": {"COMPLETED": 98, "TRUNCATED": 153, "FAILED": 7, "ABORTED": 9},
    "mean_model_turns": 7.83,
    "think_trials": 255,
    "think_blocks": 2062,
    "sglang_tp": 4,
    "sglang_version": "0.5.9",
}

RUN_NOTES = {
    "requested_checkpoint": "/mnt/shared-storage-user/narmodel/agenticrl/ckpt/terminal-rl_qwen3-8b_8gpu_seta_dapo_nodynamic_exploration_simhash_life_fp_ucb_v0629_3_outcome_gate_dualadv_think_2026-06-30_001206/iter_0000199",
    "actual_checkpoint": "/mnt/shared-storage-user/narmodel/agenticrl/ckpt/terminal-rl_qwen3-8b_8gpu_seta_dapo_nodynamic_exploration_simhash_life_fp_ucb_v0629_3_outcome_gate_dualadv_think_2026-06-30_001206/iter_0000299",
    "train_run_dir": "/mnt/shared-storage-user/narmodel/agenticrl/ckpt/terminal-rl_qwen3-8b_8gpu_seta_dapo_nodynamic_exploration_simhash_life_fp_ucb_v0629_3_outcome_gate_dualadv_think_2026-06-30_001206",
    "latest_checkpointed_iteration": "299",
    "converted_hf_dir": "/mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1/models/qwen3-8b-rl-outcome-gate-iter299-hf",
    "convert_log": "/mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1/logs/convert/convert_outcome_gate_iter299_to_hf_20260706_015441.log",
    "hf_config_sha256": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
    "hf_index_sha256": "b587ab09f421a36714df5cf69a837d2a6388f61d9300a467031f968c6441ac90",
    "manual_verifier_intervention": (
        "最后一个未落盘 trial `query-optimize__XeHWakY` 的 verifier pytest 长时间停在 "
        "`../tests/test_outputs.py ..`；超过该 task `verifier.timeout_sec=1800` 窗口后，"
        "底层 `docker compose exec`/pytest 仍未退出并出现重复 verifier 子进程。为让 Harbor "
        "按正式失败路径收尾，仅对该 trial 对应的 `query-optimize__xehwaky` verifier 子进程发 TERM；"
        "Harbor 随后记录 `RewardFileNotFoundError`，该 trial reward=0。"
        "这是超出 verifier timeout 后的清理干预，不涉及人工判定 reward；不影响此前 266 个已落盘 trial；"
        "`query-optimize` 另外两个 seed 本身也均为 reward=0。"
    ),
    "other_reward_file_missing": (
        "本次 `reward-file-missing` 共 2 个；除上述 `query-optimize__XeHWakY` 外，"
        "另一个是 `configure-git-webserver__hTuXdAT`，后者不是人工干预结果，按普通 verifier/reward 缺失失败处理。"
    ),
}

GIST_GZIP_TRAJECTORY_THRESHOLD_BYTES = 10 * 1024 * 1024


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(errors="replace"))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def task_from_trial_name(name: str) -> str:
    return name.rsplit("__", 1)[0]


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def describe(values: list[float | int | None]) -> dict[str, float | int]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0, "nonzero": 0}
    return {
        "mean": float(sum(vals) / len(vals)),
        "median": float(statistics.median(vals)),
        "p90": float(np.percentile(vals, 90)),
        "p95": float(np.percentile(vals, 95)),
        "max": float(max(vals)),
        "nonzero": int(sum(1 for v in vals if v)),
    }


def read_tail(path: Path, limit: int = 12000) -> str:
    if not path.exists() or path.is_dir():
        return ""
    data = path.read_text(errors="ignore")
    return data[-limit:]


def collect_trial_text(trial_dir: Path, result: dict[str, Any], trajectory: dict[str, Any]) -> str:
    chunks = [
        json.dumps(result.get("exception_info") or {}, ensure_ascii=False),
        json.dumps((result.get("verifier_result") or {}).get("rewards") or {}, ensure_ascii=False),
    ]
    for rel in [
        "verifier/test-stdout.txt",
        "verifier/test-stderr.txt",
        "verifier/stdout.txt",
        "verifier/stderr.txt",
        "trial.log",
    ]:
        chunks.append(read_tail(trial_dir / rel))
    for step in trajectory.get("steps", [])[-4:]:
        chunks.append(str(step.get("message") or "")[-3000:])
        chunks.append(json.dumps(step.get("observation") or {}, ensure_ascii=False)[-3000:])
    return "\n".join(chunks)


def classify_failure(row: dict[str, Any], trial_text: str) -> str:
    if row["reward"] > 0:
        return "pass"
    low = trial_text.lower()
    if row["exception_type"] == "AgentTimeoutError":
        return "agent-timeout"
    if row["exception_type"] == "RewardFileNotFoundError":
        return "reward-file-missing"
    if row["parse_error_count"] >= 3 or (row["status"] == "FAILED" and row["parse_error_count"] > 0):
        return "tool-call-json-parse"
    if "prompt is too long for configured budget" in low:
        return "prompt-budget"
    if "connection refused" in low or "failed to establish a new connection" in low:
        return "service-not-ready"
    if "syntaxerror" in low or "indentationerror" in low:
        return "syntax-error"
    if "no such file or directory" in low or "file not found" in low:
        return "missing-file-or-path"
    if row["status"] == "TRUNCATED" and row["model_turn_count"] >= row["max_iteration"]:
        return "max-iteration-truncated"
    if row["status"] == "TRUNCATED":
        return "token-or-context-truncated"
    if row["total_tool_calls"] == 0 and row["status"] == "COMPLETED":
        return "no-tool-final-response"
    if "assert" in low or "failed" in low or "error" in low:
        return "verifier-or-runtime-failed"
    return "wrong-answer-or-incomplete"


def collect_trials(job_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result_path in sorted(job_dir.glob("*/result.json")):
        trial_dir = result_path.parent
        result = read_json(result_path)
        meta = (result.get("agent_result") or {}).get("metadata") or {}
        rewards = ((result.get("verifier_result") or {}).get("rewards") or {})
        trajectory_path = trial_dir / "agent" / "trajectory.json"
        trajectory = read_json(trajectory_path) if trajectory_path.exists() else {"steps": []}
        steps = trajectory.get("steps", [])
        agent_steps = [s for s in steps if s.get("source") == "agent"]
        agent_messages = [str(s.get("message") or "") for s in agent_steps]
        tool_calls = []
        tool_counter: collections.Counter[str] = collections.Counter()
        for step in agent_steps:
            for call in step.get("tool_calls") or []:
                tool_calls.append(call)
                tool_counter[call.get("function_name") or call.get("name") or "unknown"] += 1
        think_lengths = []
        for message in agent_messages:
            think_lengths.extend(len(m.group(1)) for m in re.finditer(r"<think>(.*?)</think>", message, re.S))
        exception_info = result.get("exception_info") or {}
        row = {
            "trial": trial_dir.name,
            "task": task_from_trial_name(trial_dir.name),
            "harbor_task_name": result.get("task_name") or "",
            "reward": safe_float(rewards.get("reward")),
            "status": meta.get("status") or "UNKNOWN",
            "model_turn_count": safe_int(meta.get("model_turn_count"), len(agent_steps)),
            "parse_error_count": safe_int(meta.get("parse_error_count")),
            "max_iteration": safe_int(meta.get("max_iteration"), 10),
            "elapsed_sec": safe_float(meta.get("elapsed_sec")),
            "n_input_tokens": safe_int((result.get("agent_result") or {}).get("n_input_tokens")),
            "n_output_tokens": safe_int((result.get("agent_result") or {}).get("n_output_tokens")),
            "total_tool_calls": len(tool_calls),
            "tool_functions": "; ".join(f"{k}:{v}" for k, v in sorted(tool_counter.items())),
            "think_blocks": len(think_lengths),
            "think_chars": sum(think_lengths),
            "max_think_chars": max(think_lengths or [0]),
            "agent_message_chars": sum(len(m) for m in agent_messages),
            "max_agent_message_chars": max([len(m) for m in agent_messages] or [0]),
            "raw_tool_call_tags": sum(m.count("<tool_call>") for m in agent_messages),
            "pseudo_json_tool": int(any(re.search(r"""["']name["']\s*:\s*["']shell_""", m) for m in agent_messages)),
            "exception_type": exception_info.get("exception_type") or "",
            "exception_message": exception_info.get("exception_message") or "",
            "started_at": result.get("started_at") or "",
            "finished_at": result.get("finished_at") or "",
            "result_path": str(result_path),
            "trajectory_path": str(trajectory_path) if trajectory_path.exists() else "",
            "trajectory_bytes": trajectory_path.stat().st_size if trajectory_path.exists() else 0,
            "trajectory_sha256": sha256_file(trajectory_path) if trajectory_path.exists() else "",
        }
        row["failure_class"] = classify_failure(row, collect_trial_text(trial_dir, result, trajectory))
        rows.append(row)
    return rows


def summarize_tasks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    task_rows = []
    for task, items in sorted(by_task.items()):
        reward_sum = sum(r["reward"] for r in items)
        statuses = collections.Counter(r["status"] for r in items)
        classes = collections.Counter(r["failure_class"] for r in items if r["reward"] <= 0)
        task_rows.append(
            {
                "task": task,
                "n_trials": len(items),
                "reward_sum": reward_sum,
                "pass_any": int(any(r["reward"] > 0 for r in items)),
                "pass_rate": reward_sum / len(items) if items else 0.0,
                "statuses": "; ".join(f"{k}:{v}" for k, v in sorted(statuses.items())),
                "failure_classes": "; ".join(f"{k}:{v}" for k, v in sorted(classes.items())),
                "pass_trials": ", ".join(r["trial"] for r in items if r["reward"] > 0),
                "fail_trials": ", ".join(r["trial"] for r in items if r["reward"] <= 0),
            }
        )
    return task_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_summary(job_dir: Path, rows: list[dict[str, Any]], task_rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    root_result = read_json(job_dir / "result.json") if (job_dir / "result.json").exists() else {}
    first_trial_started_at = min((r["started_at"] for r in rows if r["started_at"]), default="")
    reward_sum = sum(r["reward"] for r in rows)
    solved_tasks = [r["task"] for r in task_rows if r["pass_any"]]
    pass1_x = int(round(reward_sum))
    pass3_x = len(solved_tasks)
    pass1_ci = wilson_ci(pass1_x, len(rows))
    pass3_ci = wilson_ci(pass3_x, len(task_rows))
    summary = {
        "job": job_dir.name,
        "started_at": first_trial_started_at or root_result.get("started_at") or "",
        "job_result_started_at": root_result.get("started_at") or "",
        "first_trial_started_at": first_trial_started_at,
        "finished_at": root_result.get("finished_at") or "",
        "is_complete": bool(root_result.get("finished_at")) or len(rows) >= safe_int(root_result.get("n_total_trials"), 267),
        "n_total_trials": len(rows),
        "n_tasks": len(task_rows),
        "reward_sum": reward_sum,
        "pass_at_1_trial": reward_sum / len(rows) if rows else 0.0,
        "pass_at_1_wilson_ci": pass1_ci,
        "pass_at_3_empirical": pass3_x / len(task_rows) if task_rows else 0.0,
        "pass_at_3_wilson_ci": pass3_ci,
        "n_solved_tasks": pass3_x,
        "solved_tasks": solved_tasks,
        "status_counts": dict(collections.Counter(r["status"] for r in rows)),
        "failure_class_counts": dict(collections.Counter(r["failure_class"] for r in rows if r["reward"] <= 0)),
        "exception_counts": dict(collections.Counter(r["exception_type"] for r in rows if r["exception_type"])),
        "tool_function_counts": dict(collections.Counter(
            fn.split(":", 1)[0]
            for r in rows
            for fn in (r["tool_functions"].split("; ") if r["tool_functions"] else [])
        )),
        "model_turn_count": describe([r["model_turn_count"] for r in rows]),
        "total_tool_calls": describe([r["total_tool_calls"] for r in rows]),
        "parse_error_count": describe([r["parse_error_count"] for r in rows]),
        "n_input_tokens": describe([r["n_input_tokens"] for r in rows]),
        "n_output_tokens": describe([r["n_output_tokens"] for r in rows]),
        "agent_message_chars": describe([r["agent_message_chars"] for r in rows]),
        "trials_with_tool_calls": sum(1 for r in rows if r["total_tool_calls"] > 0),
        "trials_with_zero_tool_calls": sum(1 for r in rows if r["total_tool_calls"] == 0),
        "trials_with_think_blocks": sum(1 for r in rows if r["think_blocks"] > 0),
        "total_think_blocks": sum(r["think_blocks"] for r in rows),
        "trials_with_pseudo_json_tool": sum(1 for r in rows if r["pseudo_json_tool"]),
        "trials_with_raw_tool_call_tags": sum(1 for r in rows if r["raw_tool_call_tags"] > 0),
        "config_job_name": config.get("job_name"),
        "config_n_attempts": config.get("n_attempts"),
        "config_concurrency": config.get("n_concurrent_trials"),
        "config_retry": ((config.get("retry") or {}).get("max_retries")),
    }
    return summary


def setup_plots() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "legend.frameon": False,
        }
    )


def fig_pass_rates(summary: dict[str, Any], fig_dir: Path) -> None:
    labels = ["#25 TB2.0\nbase aligned", "#27 TBv2.1\nbase aligned", "This TBv2.1\nRL iter299"]
    pass1 = [ISSUE25_BASE_ALIGNED["pass_at_1"], ISSUE27_BASE_TBV21_ALIGNED["pass_at_1"], summary["pass_at_1_trial"]]
    pass3 = [ISSUE25_BASE_ALIGNED["pass_at_3"], ISSUE27_BASE_TBV21_ALIGNED["pass_at_3"], summary["pass_at_3_empirical"]]
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - width / 2, [v * 100 for v in pass1], width, label="trial pass@1", color="#4C78A8")
    ax.bar(x + width / 2, [v * 100 for v in pass3], width, label="empirical pass@3", color="#F58518")
    for i, v in enumerate(pass1):
        ax.text(i - width / 2, v * 100 + 0.08, f"{v*100:.2f}%", ha="center", va="bottom")
    for i, v in enumerate(pass3):
        ax.text(i + width / 2, v * 100 + 0.08, f"{v*100:.2f}%", ha="center", va="bottom")
    ax.set_ylabel("pass rate (%)")
    ax.set_title("Aligned eval reference: #25 TB2.0, #27 TBv2.1 base, this TBv2.1 RL iter299")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(3.0, max(pass3) * 100 + 1.0))
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_A_pass_rates.png")
    plt.close(fig)


def fig_status(summary: dict[str, Any], fig_dir: Path) -> None:
    statuses = ["COMPLETED", "TRUNCATED", "FAILED", "ABORTED", "UNKNOWN"]
    issue25 = [ISSUE25_BASE_ALIGNED["status_counts"].get(s, 0) for s in statuses]
    issue27 = [ISSUE27_BASE_TBV21_ALIGNED["status_counts"].get(s, 0) for s in statuses]
    current = [summary["status_counts"].get(s, 0) for s in statuses]
    x = np.arange(len(statuses))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, issue25, width, label="#25 TB2.0 base aligned", color="#54A24B")
    ax.bar(x, issue27, width, label="#27 TBv2.1 base aligned", color="#4C78A8")
    ax.bar(x + width, current, width, label="This TBv2.1 RL iter299", color="#E45756")
    for i, v in enumerate(issue25):
        if v:
            ax.text(i - width, v + 3, str(v), ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(issue27):
        if v:
            ax.text(i, v + 3, str(v), ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(current):
        if v:
            ax.text(i + width, v + 3, str(v), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("trial count")
    ax.set_title("Exit status distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(statuses)
    ax.set_ylim(0, 285)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_B_status_distribution.png")
    plt.close(fig)


def fig_failure_classes(summary: dict[str, Any], fig_dir: Path) -> None:
    counts = collections.Counter(summary["failure_class_counts"])
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.38 * len(items))))
    y = np.arange(len(items))
    ax.barh(y, values, color="#B279A2")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("failed trials")
    ax.set_title("Failure taxonomy for this TBv2.1 run")
    for i, v in enumerate(values):
        ax.text(v + 0.8, i, str(v), va="center")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_C_failure_taxonomy.png")
    plt.close(fig)


def fig_turns_tools(rows: list[dict[str, Any]], fig_dir: Path) -> None:
    turns = [r["model_turn_count"] for r in rows]
    tools = [r["total_tool_calls"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].hist(turns, bins=np.arange(0.5, max(turns or [1]) + 1.5, 1), color="#4C78A8")
    axes[0].set_title("Model turns per trial")
    axes[0].set_xlabel("model turns")
    axes[0].set_ylabel("trial count")
    axes[1].hist(tools, bins=min(35, max(tools or [1]) + 1), color="#F58518")
    axes[1].set_title("Executed tool calls per trial")
    axes[1].set_xlabel("tool calls")
    axes[1].set_ylabel("trial count")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_D_turns_tools.png")
    plt.close(fig)


def fig_token_scatter(rows: list[dict[str, Any]], fig_dir: Path) -> None:
    colors = ["#54A24B" if r["reward"] > 0 else "#E45756" for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter([r["n_input_tokens"] for r in rows], [r["n_output_tokens"] for r in rows], c=colors, alpha=0.72, edgecolors="none")
    ax.set_title("Token usage by trial")
    ax.set_xlabel("input tokens")
    ax.set_ylabel("output tokens")
    ax.text(0.02, 0.98, "green=pass, red=fail", transform=ax.transAxes, va="top")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_E_token_usage.png")
    plt.close(fig)


def fig_task_rewards(task_rows: list[dict[str, Any]], fig_dir: Path) -> None:
    top = sorted(task_rows, key=lambda r: (-r["reward_sum"], r["task"]))[:25]
    fig, ax = plt.subplots(figsize=(10, max(5, 0.30 * len(top))))
    y = np.arange(len(top))
    ax.barh(y, [r["reward_sum"] for r in top], color="#72B7B2")
    ax.set_yticks(y)
    ax.set_yticklabels([r["task"] for r in top])
    ax.invert_yaxis()
    ax.set_xlabel("reward sum over 3 attempts")
    ax.set_title("Top task rewards")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_F_top_task_rewards.png")
    plt.close(fig)


def make_figures(rows: list[dict[str, Any]], task_rows: list[dict[str, Any]], summary: dict[str, Any], fig_dir: Path) -> None:
    setup_plots()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_pass_rates(summary, fig_dir)
    fig_status(summary, fig_dir)
    fig_failure_classes(summary, fig_dir)
    fig_turns_tools(rows, fig_dir)
    fig_token_scatter(rows, fig_dir)
    fig_task_rewards(task_rows, fig_dir)


def choose_cases(rows: list[dict[str, Any]], max_failures: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    by_trial = {r["trial"]: r for r in rows}
    selected.extend(r for r in rows if r["reward"] > 0)
    reference_slugs = sorted(set(ISSUE25_BASE_ALIGNED["solved_tasks"]) | set(ISSUE27_BASE_TBV21_ALIGNED["solved_tasks"]))
    for slug in reference_slugs:
        selected.extend(r for r in rows if r["task"] == slug)
    by_class: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        if row["reward"] <= 0:
            by_class[row["failure_class"]].append(row)
    per_class = max(1, math.ceil(max_failures / max(1, len(by_class))))
    for _, items in sorted(by_class.items()):
        selected.extend(sorted(items, key=lambda r: (-r["total_tool_calls"], -r["n_output_tokens"], r["trial"]))[:per_class])
    selected.extend(sorted(rows, key=lambda r: (-r["total_tool_calls"], r["trial"]))[:3])
    selected.extend(sorted(rows, key=lambda r: (-r["n_output_tokens"], r["trial"]))[:3])
    result = []
    seen = set()
    for row in selected:
        if row["trial"] in seen:
            continue
        if not row["trajectory_path"]:
            continue
        result.append(row)
        seen.add(row["trial"])
    return result[: max(8, len([r for r in rows if r["reward"] > 0]) + max_failures + 8)]


def copy_gist_payload(rows: list[dict[str, Any]], task_rows: list[dict[str, Any]], summary: dict[str, Any], config_path: Path, out_dir: Path) -> list[dict[str, Any]]:
    payload = out_dir / "gist_payload"
    traj_dir = payload
    if payload.exists():
        shutil.rmtree(payload)
    payload.mkdir(parents=True, exist_ok=True)
    selected = choose_cases(rows, max_failures=14)
    manifest = []
    for row in selected:
        traj_src = Path(row["trajectory_path"])
        res_src = Path(row["result_path"])
        compress_trajectory = traj_src.stat().st_size > GIST_GZIP_TRAJECTORY_THRESHOLD_BYTES
        traj_dst = traj_dir / (
            f"{row['trial']}.trajectory.json.gz.b64" if compress_trajectory else f"{row['trial']}.trajectory.json"
        )
        res_dst = traj_dir / f"{row['trial']}.result.json"
        if compress_trajectory:
            compressed = gzip.compress(traj_src.read_bytes(), compresslevel=6, mtime=0)
            with traj_dst.open("wb") as dst:
                dst.write(base64.encodebytes(compressed))
        else:
            shutil.copy2(traj_src, traj_dst)
        shutil.copy2(res_src, res_dst)
        manifest.append(
            {
                "trial": row["trial"],
                "task": row["task"],
                "reward": row["reward"],
                "status": row["status"],
                "failure_class": row["failure_class"],
                "turns": row["model_turn_count"],
                "tool_calls": row["total_tool_calls"],
                "trajectory_file": traj_dst.name,
                "result_file": res_dst.name,
                "trajectory_compressed": compress_trajectory,
                "trajectory_encoding": "gzip+base64" if compress_trajectory else "plain-json",
                "trajectory_sha256": sha256_file(traj_dst),
                "trajectory_original_sha256": row["trajectory_sha256"],
                "trajectory_original_bytes": row["trajectory_bytes"],
                "trajectory_bytes": traj_dst.stat().st_size,
            }
        )
    shutil.copy2(config_path, payload / "run_config.json")
    shutil.copy2(out_dir / "trial_summary.csv", payload / "trial_summary.csv")
    shutil.copy2(out_dir / "task_summary.csv", payload / "task_summary.csv")
    (payload / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (payload / "selected_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    (payload / "README.md").write_text(make_gist_readme(summary, manifest))
    write_case_notes(out_dir, manifest, rows)
    return manifest


def gist_anchor(filename: str) -> str:
    base = filename.lower()
    base = re.sub(r"[^a-z0-9]+", "-", base).strip("-")
    return f"file-{base}"


def short_excerpt(row: dict[str, Any], limit: int = 1100) -> str:
    path = Path(row["trajectory_path"])
    if not path.exists():
        return ""
    trajectory = read_json(path)
    agent_steps = [s for s in trajectory.get("steps", []) if s.get("source") == "agent"]
    if not agent_steps:
        return ""
    candidates = []
    if row["failure_class"] == "tool-call-json-parse":
        candidates = agent_steps[-2:]
    elif row["reward"] > 0:
        candidates = agent_steps[:2]
    else:
        candidates = [max(agent_steps, key=lambda s: len(str(s.get("message") or "")))]
    text = "\n\n".join(str(s.get("message") or "") for s in candidates).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.replace("\x00", "")
    if len(text) > limit:
        return text[:limit].rstrip() + "\n...[issue 摘录截断，完整内容见 Gist 原始轨迹]..."
    return text


def make_gist_readme(summary: dict[str, Any], manifest: list[dict[str, Any]]) -> str:
    lines = [
        "# TBv2.1 × Qwen3-8B RL outcome_gate iter299 aligned eval 原始轨迹包",
        "",
        f"- job: `{summary['job']}`",
        f"- trials: `{summary['n_total_trials']}`",
        f"- tasks: `{summary['n_tasks']}`",
        f"- reward_sum: `{summary['reward_sum']:.0f}`",
        f"- pass@1: `{pct(summary['pass_at_1_trial'])}`",
        f"- pass@3: `{pct(summary['pass_at_3_empirical'])}`",
        "",
        "## 文件索引",
        "",
        "| 文件 | 内容 |",
        "|---|---|",
        "| `summary.json` | 本次 run 的聚合指标 |",
        "| `trial_summary.csv` | 267 个 trial 的逐条统计 |",
        "| `task_summary.csv` | 89 个 task 的聚合统计 |",
        "| `run_config.json` | Harbor / adapter 运行配置 |",
        "| `case_notes.md` | 中文案例导读 |",
        "| `selected_manifest.json` | 选中原始轨迹 manifest |",
        "| `*.trajectory.json` / `*.trajectory.json.gz.b64` / `*.result.json` | 从 Harbor job 复制的真实原始轨迹样本；超大 JSON 以 gzip 后 base64 文本保存，可用 `base64 -d file.gz.b64 | gunzip > file.json` 还原 |",
        "",
        "## 选中原始轨迹",
        "",
        "| 轨迹 trial | 任务 | reward | status | 失败类型 | 轮数 | 工具调用 | 原始轨迹 | SHA256 |",
        "|---|---|---:|---|---|---:|---:|---|---|",
    ]
    for item in manifest:
        lines.append(
            f"| `{item['trial']}` | `{item['task']}` | {item['reward']:.0f} | {item['status']} | "
            f"{item['failure_class']} | {item['turns']} | {item['tool_calls']} | "
            f"`{item['trajectory_file']}` | `{item['trajectory_sha256'][:12]}...` |"
        )
    return "\n".join(lines) + "\n"


def write_case_notes(out_dir: Path, manifest: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    by_trial = {r["trial"]: r for r in rows}
    reference_slugs = set(ISSUE25_BASE_ALIGNED["solved_tasks"]) | set(ISSUE27_BASE_TBV21_ALIGNED["solved_tasks"])
    lines = ["# TBv2.1 × Qwen3-8B RL outcome_gate iter299 aligned eval 案例导读", ""]
    for item in manifest:
        row = by_trial[item["trial"]]
        why = []
        if row["reward"] > 0:
            why.append("通过样本")
        if row["task"] in reference_slugs:
            why.append("#25/#27 参考结果中曾解出的 task slug")
        if row["failure_class"] != "pass":
            why.append(f"失败类型 `{row['failure_class']}`")
        why_text = "；".join(why) or "代表性样本"
        lines.extend(
            [
                f"## `{row['trial']}`",
                "",
                f"- 选择原因：{why_text}",
                f"- task：`{row['task']}`",
                f"- reward/status/class：`{row['reward']:.0f}` / `{row['status']}` / `{row['failure_class']}`",
                f"- turns/tools/parse_errors：`{row['model_turn_count']}` / `{row['total_tool_calls']}` / `{row['parse_error_count']}`",
                f"- tokens in/out：`{row['n_input_tokens']}` / `{row['n_output_tokens']}`",
                f"- 原始轨迹：`{item['trajectory_file']}`",
                "",
                "短摘录：",
                "",
                "```text",
                short_excerpt(row, 1300),
                "```",
                "",
            ]
        )
    text = "\n".join(lines)
    (out_dir / "case_notes.md").write_text(text)
    payload = out_dir / "gist_payload"
    payload.mkdir(exist_ok=True)
    (payload / "case_notes.md").write_text(text)


def parse_image_markdown(path: Path | None) -> dict[str, str]:
    if not path or not path.exists():
        return {}
    urls: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        m = re.match(r"!\[(?P<name>[^\]]+)\]\((?P<url>[^)]+)\)", line.strip())
        if m:
            urls[m.group("name")] = m.group("url")
    return urls


def figure_markdown(fig_base_url: str, filename: str, alt: str, fig_urls: dict[str, str] | None = None) -> str:
    if fig_urls and fig_urls.get(filename):
        return f"![{alt}]({fig_urls[filename]})"
    if not fig_base_url:
        return f"<!-- {alt}: {filename} 待上传 -->"
    return f"![{alt}]({fig_base_url.rstrip('/')}/{filename})"


def raw_link(gist_url: str, file_path: str) -> str:
    if not gist_url:
        return f"`{file_path}`"
    return f"[`{file_path}`]({gist_url}#{gist_anchor(Path(file_path).name)})"


def issue_body(
    summary: dict[str, Any],
    task_rows: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    config_path: Path,
    sglang_cmd_path: Path | None,
    resume_log_path: Path | None,
    job_dir: Path,
    gist_url: str,
    fig_base_url: str,
    fig_urls: dict[str, str] | None = None,
) -> str:
    agent = (config.get("agents") or [{}])[0]
    kwargs = agent.get("kwargs") or {}
    env = agent.get("env") or {}
    retry = config.get("retry") or {}
    solved = ", ".join(f"`{t}`" for t in summary["solved_tasks"]) or "无"
    pass1_ci = summary["pass_at_1_wilson_ci"]
    pass3_ci = summary["pass_at_3_wilson_ci"]
    reference_slugs = sorted(set(ISSUE25_BASE_ALIGNED["solved_tasks"]) | set(ISSUE27_BASE_TBV21_ALIGNED["solved_tasks"]))
    reference_solved_rows = []
    for slug in reference_slugs:
        task = next((t for t in task_rows if t["task"] == slug), None)
        refs = []
        if slug in ISSUE25_BASE_ALIGNED["solved_tasks"]:
            refs.append("#25 TB2.0 base")
        if slug in ISSUE27_BASE_TBV21_ALIGNED["solved_tasks"]:
            refs.append("#27 TBv2.1 base")
        if task:
            reference_solved_rows.append(
                f"| `{slug}` | {', '.join(refs)} | {task['n_trials']} | {task['reward_sum']:.0f} | {task['pass_any']} | {task['statuses']} | {task['failure_classes']} |"
            )
        else:
            reference_solved_rows.append(f"| `{slug}` | {', '.join(refs)} | 0 | 0 | 0 | 未出现在本次结果 | 未出现在本次结果 |")
    status_rows = []
    for status in ["COMPLETED", "TRUNCATED", "FAILED", "ABORTED", "UNKNOWN"]:
        if (
            summary["status_counts"].get(status, 0)
            or ISSUE25_BASE_ALIGNED["status_counts"].get(status, 0)
            or ISSUE27_BASE_TBV21_ALIGNED["status_counts"].get(status, 0)
        ):
            status_rows.append(
                f"| {status} | {summary['status_counts'].get(status, 0)} | {ISSUE27_BASE_TBV21_ALIGNED['status_counts'].get(status, 0)} | {ISSUE25_BASE_ALIGNED['status_counts'].get(status, 0)} |"
            )
    failure_rows = []
    for key, value in sorted(summary["failure_class_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        failure_rows.append(f"| `{key}` | {value} |")
    gist_table_rows = []
    for item in manifest:
        gist_table_rows.append(
            f"| `{item['trial']}` | `{item['task']}` | {item['reward']:.0f} | {item['status']} | "
            f"`{item['failure_class']}` | {item['turns']} | {item['tool_calls']} | "
            f"{raw_link(gist_url, item['trajectory_file'])} | `{item['trajectory_sha256'][:12]}...` |"
        )
    case_rows = []
    by_trial = {item["trial"]: item for item in manifest}
    case_candidates = []
    for item in manifest:
        if len(case_candidates) >= 14:
            break
        if item["reward"] > 0 or item["task"] in reference_slugs or item["failure_class"] in {
            "tool-call-json-parse",
            "agent-timeout",
            "prompt-budget",
            "max-iteration-truncated",
        }:
            case_candidates.append(item)
    by_trial_row = {r["trial"]: r for r in rows}
    for item in case_candidates:
        row = by_trial_row[item["trial"]]
        excerpt = short_excerpt(row, 900)
        case_rows.append(
            "\n".join(
                [
                    f"### `{item['trial']}`：`{item['failure_class']}`",
                    "",
                    f"- task：`{item['task']}`",
                    f"- reward/status/turns/tools：`{item['reward']:.0f}` / `{item['status']}` / `{item['turns']}` / `{item['tool_calls']}`",
                    f"- 原始轨迹：{raw_link(gist_url, item['trajectory_file'])}",
                    "",
                    "```text",
                    excerpt,
                    "```",
                ]
            )
        )
    sglang_cmd = read_tail(sglang_cmd_path, 2000).strip() if sglang_cmd_path and sglang_cmd_path.exists() else ""
    resume_log_note = ""
    if resume_log_path and resume_log_path.exists():
        resume_log_note = (
            f"- **中断/续跑记录**：首次 Harbor run 在 130 个 trial result 落盘后被外部 `Killed`；"
            f"随后使用同一 job/config 通过 `harbor jobs resume -p {job_dir}` 续跑。"
            f"续跑日志：`{resume_log_path}`。Harbor resume 语义为保留已有 `result.json`，清理无 `result.json` 的中断 trial 目录后继续剩余 trial。"
        )
    issue25_solved_text = "{" + ", ".join(ISSUE25_BASE_ALIGNED["solved_tasks"]) + "}"
    issue27_solved_text = "{" + ", ".join(ISSUE27_BASE_TBV21_ALIGNED["solved_tasks"]) + "}"
    title_line = (
        f"# TBv2.1 × Qwen3-8B RL outcome_gate iter299 对齐全量评测："
        f"pass@1={pct(summary['pass_at_1_trial'])}，pass@3={pct(summary['pass_at_3_empirical'])}"
    )
    body = f"""
{title_line}

> 范围说明：本 issue 记录当前集群上 `Qwen3-8B RL outcome_gate iter299` checkpoint 在 **Terminal-Bench v2.1** 上的一次完整对齐评测。运行配置按 #23/#25 的 Mode B 方向对齐：camel-agent / SGLang `/generate` / Qwen3 chat template / `qwen25` tool parser / `non_think_mode=false` / `max_iteration=10` / sampling temperature 1.0。  
> checkpoint 说明：用户最初指定 `{RUN_NOTES['requested_checkpoint']}`，但本地该目录不存在；同一训练 run 下 `latest_checkpointed_iteration.txt={RUN_NOTES['latest_checkpointed_iteration']}` 且 `iter_0000299` 存在，因此按后续指令改跑 iter299，并先转换为 HF 目录 `{RUN_NOTES['converted_hf_dir']}` 后再由 SGLang serve。  
> 重要边界：这不是与 #25 或 #27 的 bit-identical A/B。#25 是 TB2.0 base aligned；#27 是 TBv2.1 base aligned；本次是 TBv2.1 上的 RL iter299 checkpoint。#25 base cell 使用 TP=8，本节点/本次使用 TP=4；adapter 使用 #23/#25 最新 comment 指向的 upstream gist 版本并在本地持久化。

## TL;DR

- **主结果**：`pass@1 = {pct(summary['pass_at_1_trial'])}`（`{summary['reward_sum']:.0f} / {summary['n_total_trials']}`，Wilson 95% CI `{pct(pass1_ci[0])} - {pct(pass1_ci[1])}`），`pass@3 = {pct(summary['pass_at_3_empirical'])}`（`{summary['n_solved_tasks']} / {summary['n_tasks']}`，Wilson 95% CI `{pct(pass3_ci[0])} - {pct(pass3_ci[1])}`）。
- **解出任务**：{solved}。
- **运行完整性**：`{summary['n_total_trials']} / 267` 个 trial result 已写出；首个 trial 开始 `{summary['started_at']}`，最终 job 结束 `{summary['finished_at']}`。
- **时间口径**：trial `started_at/finished_at` 中带 `Z` 的字段为 UTC；root `result.json` 的 `started_at/finished_at` 在本地环境中无 `Z`，按 UTC+8 解释。本次最终 root `finished_at={summary['finished_at']}` 约对应 `2026-07-05T23:49:15Z`。
{resume_log_note}
- **最后一个 verifier 的边界记录**：{RUN_NOTES['manual_verifier_intervention']}
- **checkpoint 口径**：`iter_0000199` 本地不存在；实际评测对象为 `iter_0000299`，HF 转换后 served name 为 `{kwargs.get('sglang_served_name')}`。
- **参照结果**：#27 的 TBv2.1 base aligned 为 `pass@1=1.87% (5/267)`、`pass@3=3.37% (3/89)`，解出 `{issue27_solved_text}`；#25 的 TB2.0 base aligned 为 `pass@1=1.12% (3/267)`、`pass@3=2.25% (2/89)`，解出 `{issue25_solved_text}`。这些是参照锚点，不是本次 run 重新计算结果。
- **配置结论**：关键 aligned knob 已按 #23/#25 路线设置；残余差异包括 checkpoint、`TB2.0 -> TBv2.1`（相对 #25）、`TP=8 -> TP=4`（相对 #25）、SGLang minor version / 本地 Python shim。本文会把这些差异显式列出，不把本次结果说成 bit-identical 复现。

## 原始数据和附件

| 类型 | 链接 / 路径 | 内容 |
|---|---|---|
| 原始轨迹 Gist | {gist_url or '`<待上传>`'} | `summary.json`、`trial_summary.csv`、`task_summary.csv`、`run_config.json`、`case_notes.md`、选中 trial 的真实 `trajectory.json` / `result.json` |
| 本地 full job | `{job_dir}` | 全量 267 个 trial 的 Harbor 输出 |
| 本地配置 | `{config_path}` | 本次 Harbor config |
| SGLang 命令 | `{sglang_cmd_path or '<未记录>'}` | SGLang launcher 参数 |
| 续跑日志 | `{resume_log_path if resume_log_path and resume_log_path.exists() else '<无续跑日志>'}` | 若发生中断，此处记录 resume 命令输出 |
| 图表来源 | `{fig_base_url or '<待上传>'}` | 本 issue 中所有图表；若使用 Gist raw，则为文本 SVG 包装图 |

### Gist 原始轨迹索引

| 轨迹 trial | 任务 | reward | status | 失败类型 | 轮数 | 工具调用 | 原始轨迹 | SHA256 |
|---|---|---:|---|---|---:|---:|---|---|
{chr(10).join(gist_table_rows)}

> 上表只列出 issue 中引用和失败类型覆盖用的 selected trajectories。全量逐 trial 指标在 Gist 的 `trial_summary.csv`，集群内全量原始轨迹在本地 job 路径下。
> 说明：`.trajectory.json.gz.b64` 是原始 `trajectory.json` 经过 gzip 后再 base64 的文本文件，用于绕开 Gist 二进制/大文件限制；可用 `base64 -d <file>.trajectory.json.gz.b64 | gunzip > <file>.trajectory.json` 还原。
> SHA256 口径：上表 SHA256 对应 Gist 中实际上传文件；`.gz.b64` 行的 SHA256 是编码后文本文件哈希，原始 raw JSON 哈希见 Gist 的 `selected_manifest.json`。

## 1. 核心指标

| 指标 | 本次：Qwen3-8B RL iter299 × TBv2.1 aligned | #27 参考：Qwen3-8B-base × TBv2.1 aligned | #25 参考：Qwen3-8B-base × TB2.0 aligned |
|---|---:|---:|---:|
| Trials | {summary['n_total_trials']} | {ISSUE27_BASE_TBV21_ALIGNED['n_trials']} | {ISSUE25_BASE_ALIGNED['n_trials']} |
| Tasks | {summary['n_tasks']} | {ISSUE27_BASE_TBV21_ALIGNED['n_tasks']} | {ISSUE25_BASE_ALIGNED['n_tasks']} |
| pass@1 trial | **{pct(summary['pass_at_1_trial'])}** (`{summary['reward_sum']:.0f} / {summary['n_total_trials']}`) | **{pct(ISSUE27_BASE_TBV21_ALIGNED['pass_at_1'])}** (`5 / 267`) | **{pct(ISSUE25_BASE_ALIGNED['pass_at_1'])}** (`3 / 267`) |
| pass@3 empirical | **{pct(summary['pass_at_3_empirical'])}** (`{summary['n_solved_tasks']} / {summary['n_tasks']}`) | **{pct(ISSUE27_BASE_TBV21_ALIGNED['pass_at_3'])}** (`3 / 89`) | **{pct(ISSUE25_BASE_ALIGNED['pass_at_3'])}** (`2 / 89`) |
| solved tasks | **{summary['n_solved_tasks']}** | **{ISSUE27_BASE_TBV21_ALIGNED['n_solved_tasks']}** | **{ISSUE25_BASE_ALIGNED['n_solved_tasks']}** |
| reward sum | **{summary['reward_sum']:.0f}** | **{ISSUE27_BASE_TBV21_ALIGNED['reward_sum']}** | **{ISSUE25_BASE_ALIGNED['reward_sum']}** |
| mean turns / steps | `{summary['model_turn_count']['mean']:.2f}` model turns | `{ISSUE27_BASE_TBV21_ALIGNED['mean_model_turns']:.2f}` model turns | `7.52` agent steps |
| trials with think block | `{summary['trials_with_think_blocks']} / {summary['n_total_trials']}` | `{ISSUE27_BASE_TBV21_ALIGNED['think_trials']} / 267` | `252 / 267` |
| total think blocks | `{summary['total_think_blocks']}` | `{ISSUE27_BASE_TBV21_ALIGNED['think_blocks']}` | `1978` |

{figure_markdown(fig_base_url, 'fig_A_pass_rates.png', '图 A：pass rate 对比', fig_urls)}

> #25/#27 列来自对应 issue 正文和本地缓存转录；这些列是公开历史参考，不是本次 run 重新计算结果。由于本次换成 RL iter299 checkpoint，任何 pass@k 差异都不能只归因于 harness 或 TB 版本。  
> 统计解释：本次 `6/267` 与 #27 `5/267` 的 pass@1 差异只有约 `0.37pp`，且 Wilson 95% CI 大幅重叠；这里只把它作为一次完整 run 的点估计，不解读为显著优劣。另，#25 的 `mean turns / steps` 口径来自 agent steps，本次/#27 是 model turns，不应直接做单调趋势比较。

## 2. 配置对齐证据

| 配置项 | 本次取值 | 对齐 / 差异说明 |
|---|---|---|
| OpenClaw-RL | `3573bc3e9a3fb73acdbb53bf83a6f72ca1ba7aea` | 与用户指定 puyuan 分支版本对齐 |
| adapter | `runtime/openclaw_camel_adapter_upstream_gist/openclaw_camel_adapter.py`，SHA256 `70f7976802f1490b8b6ee06fed310c7531d38b7e0bfe70641494fc0ffdefc2da` | 来自 #23/#25 最新 comment 指向的 upstream gist，不是早先本地手写 adapter |
| agent import | `{agent.get('import_path')}` | Mode B camel-agent 路线 |
| model name | `{agent.get('model_name')}` | Harbor routing 名称 |
| 用户原始指定 checkpoint | `{RUN_NOTES['requested_checkpoint']}` | 本地不存在，未作为评测对象 |
| 实际原始 checkpoint | `{RUN_NOTES['actual_checkpoint']}` | 同 run 下可见的 `iter_0000299`；`latest_checkpointed_iteration.txt={RUN_NOTES['latest_checkpointed_iteration']}` |
| HF 转换后 checkpoint | `{kwargs.get('hf_model_dir')}` | SGLang 实际加载目录；config SHA256 `{RUN_NOTES['hf_config_sha256'][:12]}...`，index SHA256 `{RUN_NOTES['hf_index_sha256'][:12]}...` |
| HF 转换日志 | `{RUN_NOTES['convert_log']}` | torch-distributed checkpoint 转 HF 的本地日志 |
| served name | `{kwargs.get('sglang_served_name')}` | SGLang served model name |
| endpoint | `{kwargs.get('sglang_url')}` | adapter 使用 SGLang `/generate` |
| tool parser | `{kwargs.get('tool_call_parser')}` | #23/#25 对齐项 |
| `non_think_mode` | `{kwargs.get('non_think_mode')}` | #23/#25 对齐项 |
| `max_iteration` | `{kwargs.get('max_iteration')}` | #23/#25 对齐项 |
| `max_total_tokens` | `{kwargs.get('max_total_tokens')}` | request budget 对齐项 |
| `max_new_tokens` | `{kwargs.get('max_new_tokens')}` | generation cap |
| sampling | `temperature={kwargs.get('temperature')}`, `top_p={kwargs.get('top_p')}`, `top_k={kwargs.get('top_k')}`, `rollout_seed={kwargs.get('rollout_seed')}`, `skip_special_tokens={kwargs.get('rollout_skip_special_tokens')}` | #23/#25 对齐项 |
| Harbor attempts / concurrency / retry | `{config.get('n_attempts')}` / `{config.get('n_concurrent_trials')}` / `{retry.get('max_retries')}` | 与 #25 的 `k=3, n=4` 运行方式对齐 |
| SGLang TP | `4` | 与 #27 TBv2.1 base aligned 相同；残余差异：#25 base aligned 是 `tp=8` |
| Dataset | `Terminal-Bench v2.1`，89 tasks | 与 #27 相同；残余差异：#25 是 TB2.0 |

SGLang 启动命令记录：

```text
{sglang_cmd or '<未记录到 sglang_30000.cmd>'}
```

## 3. 状态分布

| 状态 | 本次 TBv2.1 RL iter299 | #27 TBv2.1 base-aligned | #25 TB2.0 base-aligned |
|---|---:|---:|---:|
{chr(10).join(status_rows)}

{figure_markdown(fig_base_url, 'fig_B_status_distribution.png', '图 B：状态分布对比', fig_urls)}

## 4. 失败类型和行为诊断

| 失败类型 | 失败 trial 数 |
|---|---:|
{chr(10).join(failure_rows)}

{figure_markdown(fig_base_url, 'fig_C_failure_taxonomy.png', '图 C：失败类型分布', fig_urls)}

| 行为指标 | 本次结果 |
|---|---:|
| mean / median model turns | `{summary['model_turn_count']['mean']:.2f}` / `{summary['model_turn_count']['median']:.0f}` |
| max model turns | `{summary['model_turn_count']['max']:.0f}` |
| mean / median tool calls | `{summary['total_tool_calls']['mean']:.2f}` / `{summary['total_tool_calls']['median']:.0f}` |
| zero-tool trials | `{summary['trials_with_zero_tool_calls']} / {summary['n_total_trials']}` |
| trials with parse errors | `{summary['parse_error_count']['nonzero']} / {summary['n_total_trials']}` |
| trials with `<think>...</think>` | `{summary['trials_with_think_blocks']} / {summary['n_total_trials']}` |
| total think blocks | `{summary['total_think_blocks']}` |
| pseudo JSON tool intent trials | `{summary['trials_with_pseudo_json_tool']} / {summary['n_total_trials']}` |
| raw `<tool_call>` tag trials | `{summary['trials_with_raw_tool_call_tags']} / {summary['n_total_trials']}` |
| input tokens mean / median | `{summary['n_input_tokens']['mean']:.0f}` / `{summary['n_input_tokens']['median']:.0f}` |
| output tokens mean / median | `{summary['n_output_tokens']['mean']:.0f}` / `{summary['n_output_tokens']['median']:.0f}` |

{figure_markdown(fig_base_url, 'fig_D_turns_tools.png', '图 D：turns / tool calls 分布', fig_urls)}

{figure_markdown(fig_base_url, 'fig_E_token_usage.png', '图 E：token 使用散点图', fig_urls)}

## 5. #25/#27 参考 solved task slug 在本次 iter299 中的状况

| 任务 slug | 参考来源 | 本次 trial 数 | reward_sum | 是否至少通过一次 | 状态分布 | 失败类型分布 |
|---|---|---:|---:|---:|---|---|
{chr(10).join(reference_solved_rows)}

这张表只说明参考 issue 中同名 task slug 在本次 RL iter299 配置下的结果。它不能单独证明 TB2.0 与 TBv2.1 同名任务完全等价，也不能排除采样随机性。
本次 solved task 与 #25/#27 参考 solved set 的交集只有 `modernize-scientific-stack`，重合度较低；因此 `4 solved`、`3 solved`、`2 solved` 不应被读成严格排序。

{figure_markdown(fig_base_url, 'fig_F_top_task_rewards.png', '图 F：task reward 排名', fig_urls)}

## 6. 案例分析

以下只放短摘录；完整 system/user/agent/tool observation 原始 JSON 在 Gist 中。

{chr(10).join(case_rows)}

## 7. 解释边界

1. **本次可以确认的事情**：Harbor full eval 完成；SGLang endpoint、Docker task container、adapter import、qwen25 parser、TerminalToolkit、verifier 主链路能支撑 89×3 full run；结果文件完整落盘。
2. **本次可以确认的 checkpoint 事实**：`iter_0000199` 本地缺失；实际跑的是 `iter_0000299`，且转换后的 HF 目录被 SGLang 加载。本文所有指标都应归到 iter299，不应归到 iter199。
3. **最后 verifier 干预边界**：{RUN_NOTES['manual_verifier_intervention']} 因此 `RewardFileNotFoundError` 应读作该 verifier 未能正常产出 `reward.txt/reward.json`，不是模型成功或失败的细分语义证据。{RUN_NOTES['other_reward_file_missing']}
4. **本次不能确认的事情**：不能把本次与 #27 的 pass@k 差异解释成单一 RL 训练收益或损失；也不能把本次与 #25 TB2.0 的差异解释成单一模型能力差异。TP、SGLang minor version、dataset 版本、checkpoint 都是边界条件。
5. **关于“对齐”的措辞**：本文使用“对齐”指关键运行 knob 按 #23/#25 Mode B 方案设置；不表示与 #25 当时 `/tmp` adapter、8-GPU SGLang 拓扑和 TB2.0 数据集 bit-identical，也不表示与 #27 除 checkpoint 外完全 bit-identical。
6. **最保守结论**：这是一次完整的 `Qwen3-8B RL outcome_gate iter299 × TBv2.1 aligned` eval 结果。#27 是最接近的 TBv2.1 base 参照；#25 是 TB2.0 base aligned 历史参照。

## 8. 复现入口

```bash
cd /mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1

TBV21_UPSTREAM_GIST_ALL_TASKS=1 \\
TBV21_UPSTREAM_GIST_ATTEMPTS=3 \\
TBV21_UPSTREAM_GIST_CONCURRENCY=4 \\
TBV21_MODEL_NAME=qwen3-8b-rl-outcome-gate-iter299 \\
TBV21_MODEL_PATH={RUN_NOTES['converted_hf_dir']} \\
TBV21_UPSTREAM_GIST_JOB_NAME={summary['job']} \\
bin/run_upstream_gist_adapter_subset_qwen3_8b_base_tbv21.sh

python3 bin/make_tbv21_qwen3_rl_iter299_full_issue_report.py \\
  --job jobs/{summary['job']} \\
  --config {config_path} \\
  --out reports/tbv21_qwen3_8b_rl_outcome_gate_iter299_aligned_full89_20260706
```

## 9. 后续建议

1. 若要把本次和 #27 做严格 RL-vs-base 归因，优先用同一节点、同一 SGLang/adapter/task images、同一 k/n 复跑 base 与 iter299。
2. 若要把本次和 #25 做严格版本归因，补一个同 adapter / 同 SGLang / 同 TP 的 TB2.0 replay，再看 #25 的两个 solved task 是否可在当前节点复现。
3. 对本次 reward=0 但参考中曾 pass 的 task slug 抽样复跑，区分采样偶然、依赖服务未就绪、路径/文件错误和 verifier/runtime 错误。
"""
    return textwrap.dedent(body).strip() + "\n"


def write_report(args: argparse.Namespace) -> None:
    job_dir = args.job
    config_path = args.config
    out_dir = args.out
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = read_json(config_path)
    rows = collect_trials(job_dir)
    task_rows = summarize_tasks(rows)
    summary = make_summary(job_dir, rows, task_rows, config)
    write_csv(out_dir / "trial_summary.csv", rows)
    write_csv(out_dir / "task_summary.csv", task_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    make_figures(rows, task_rows, summary, fig_dir)
    manifest = copy_gist_payload(rows, task_rows, summary, config_path, out_dir)
    (out_dir / "issue_title.txt").write_text(
        f"TBv2.1 × Qwen3-8B RL outcome_gate iter299 对齐全量评测：pass@1={pct(summary['pass_at_1_trial'])}，pass@3={pct(summary['pass_at_3_empirical'])}\n"
    )
    body = issue_body(
        summary=summary,
        task_rows=task_rows,
        manifest=manifest,
        rows=rows,
        config=config,
        config_path=config_path,
        sglang_cmd_path=args.sglang_cmd,
        resume_log_path=args.resume_log,
        job_dir=job_dir,
        gist_url=args.gist_url,
        fig_base_url=args.fig_base_url,
        fig_urls=parse_image_markdown(args.fig_markdown),
    )
    (out_dir / "issue_body.md").write_text(body)
    archive_base = out_dir / "tbv21_qwen3_8b_rl_outcome_gate_iter299_aligned_report_payload"
    archive_path = shutil.make_archive(str(archive_base), "zip", out_dir / "gist_payload")
    print(json.dumps({
        "out": str(out_dir),
        "trials": len(rows),
        "tasks": len(task_rows),
        "reward_sum": summary["reward_sum"],
        "pass_at_1": summary["pass_at_1_trial"],
        "pass_at_3": summary["pass_at_3_empirical"],
        "solved_tasks": summary["solved_tasks"],
        "figures": [str(p) for p in sorted(fig_dir.glob("*.png"))],
        "payload_zip": archive_path,
    }, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sglang-cmd", type=Path, default=Path("state/sglang_30000.cmd"))
    parser.add_argument("--resume-log", type=Path)
    parser.add_argument("--gist-url", default="")
    parser.add_argument("--fig-base-url", default="")
    parser.add_argument("--fig-markdown", type=Path)
    args = parser.parse_args()
    write_report(args)


if __name__ == "__main__":
    main()
