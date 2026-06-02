#!/usr/bin/env python3
"""Parse <run_dir>/logs/train.log and plot core training curves.

Generates the same figures previously produced by the inline analyzer in
run-specific notebooks:
  overview.png  reward_curve.png  response_length.png
  loss_curve.png  grad_norm.png  kl_entropy.png
  summary_stats.json

Reusable across runs.

Usage:
  python terminal-rl/scripts/plot_training_metrics.py --run-dir runs/<run_id>

Optional:
  --log-file PATH  Override (default <run_dir>/logs/train.log)
  --out-dir DIR    Override output (default <run_dir>/metrics/analysis)
  --no-figs        Skip image generation, only emit summary_stats.json

Exits 0 on success, 1 if log not found, 2 if no parsed rollouts.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROLLOUT_RE = re.compile(r"data\.py:\d+ - rollout (\d+): (\{.+\})")
TRAIN_RE = re.compile(r"model\.py:\d+ - step (\d+): (\{.+\})")
PERF_RE = re.compile(r"rollout\.py:\d+ - perf (\d+): (\{.+\})")
TIMESTAMP_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
TRAJ_RE = re.compile(
    r"\[task=(\S+) uid=(\S+) group_idx=(\d+) sample_idx=(\d+)\] "
    r"Rollout finished: status=(\S+) turns=(\d+) parse_errors=(\d+)"
)
CLAW_RE = re.compile(r"ClawSentry pre_action fail-open.*?'(\d+) ([^']+)'")
RESET500_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*Server error '500 .*?/reset'"
)
STRUCTURED_METRIC_RE = re.compile(r"TERMINAL_RL_METRIC_JSON\s+(\{.+\})")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
RAY_PREFIX_RE = re.compile(r"^\([^)]*\)\s*")
REWARD_BREAKDOWN_RE = re.compile(r"dataset reward breakdown rollout=(\d+) step=(\d+)")


def _clean_log_payload(line: str) -> str:
    text = ANSI_RE.sub("", line).strip()
    text = RAY_PREFIX_RE.sub("", text).strip()
    return text


def _parse_table_float(value: str) -> float | None:
    if value == "-" or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _raw_reward_scale_hint(dataset: str) -> dict[str, Any]:
    name = str(dataset or "").strip().lower()
    if name in {"seta", "terminal_bench", "seta_env"} or name.startswith("seta_"):
        return {
            "raw_reward_scale": "pass_rate_0_1",
            "raw_reward_semantics": "terminal task test pass rate; 1.0 means all trainable samples passed",
            "raw_reward_min": 0.0,
            "raw_reward_max": 1.0,
        }
    if name in {"agent_safetybench", "agentharm", "security"} or name.startswith("agent_"):
        return {
            "raw_reward_scale": "direct_safety_score",
            "raw_reward_semantics": "dataset reward-model score, not a 0/1 pass rate",
            "raw_reward_min": None,
            "raw_reward_max": None,
        }
    return {
        "raw_reward_scale": "unknown",
        "raw_reward_semantics": None,
        "raw_reward_min": None,
        "raw_reward_max": None,
    }


def _parse_log(log_path: Path) -> dict[str, Any]:
    rollout_metrics: dict[int, dict] = {}
    train_metrics: dict[int, dict] = {}
    train_points: list[dict[str, Any]] = []
    perf_metrics: dict[int, dict] = {}
    clawsentry_errs: Counter = Counter()
    status_counts: Counter = Counter()
    turn_counts: list[int] = []
    parse_errs: list[int] = []
    reset500_per_min: Counter = Counter()
    structured_metrics: list[dict[str, Any]] = []
    reward_breakdown_records: list[dict[str, Any]] = []
    reward_table_rollout: int | None = None
    reward_table_step: int | None = None

    print(f"[+] parsing {log_path}")
    with log_path.open(errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            clean_line = _clean_log_payload(line)
            m_table = REWARD_BREAKDOWN_RE.search(clean_line)
            if m_table:
                reward_table_rollout = int(m_table.group(1))
                reward_table_step = int(m_table.group(2))
                continue
            if reward_table_rollout is not None:
                if clean_line.startswith("dataset ") or clean_line.startswith("---"):
                    continue
                parts = clean_line.split()
                if len(parts) >= 9 and parts[1].isdigit() and parts[2].isdigit():
                    dataset = parts[0]
                    record = {
                        "schema": "terminal_rl.dataset_reward_breakdown_table.v1",
                        "phase": "train",
                        "dataset": dataset,
                        "source_datasets": [dataset],
                        "rollout_id": reward_table_rollout,
                        "global_step": reward_table_step,
                        "sample_count": int(parts[1]),
                        "trainable_count": int(parts[2]),
                        "reward/total": _parse_table_float(parts[3]),
                        "total_reward": _parse_table_float(parts[3]),
                        "test_acc": _parse_table_float(parts[4]),
                        "reward/raw": _parse_table_float(parts[5]),
                        "raw_reward": _parse_table_float(parts[5]),
                        "reward/task": _parse_table_float(parts[6]),
                        "task_reward": _parse_table_float(parts[6]),
                        "safety_reward": _parse_table_float(parts[7]),
                        "reward/exploration": _parse_table_float(parts[8]),
                        "exploration_reward": _parse_table_float(parts[8]),
                        "_log_line": line_no,
                    }
                    record.update(_raw_reward_scale_hint(dataset))
                    reward_breakdown_records.append(record)
                    continue
                if clean_line.startswith("[") or "rollout_log.py:" in clean_line:
                    reward_table_rollout = None
                    reward_table_step = None

            m = ROLLOUT_RE.search(line)
            if m:
                try:
                    rollout_metrics[int(m.group(1))] = ast.literal_eval(m.group(2))
                except Exception:
                    pass
                continue
            m = TRAIN_RE.search(line)
            if m:
                try:
                    step_label = int(m.group(1))
                    payload = ast.literal_eval(m.group(2))
                    train_metrics[step_label] = payload
                    point = dict(payload)
                    point["_log_index"] = len(train_points)
                    point["_log_line"] = line_no
                    point["_step_label"] = step_label
                    ts = TIMESTAMP_RE.search(line)
                    if ts:
                        point["_timestamp"] = ts.group(1)
                    train_points.append(point)
                except Exception:
                    pass
                continue
            m = PERF_RE.search(line)
            if m:
                try:
                    perf_metrics[int(m.group(1))] = ast.literal_eval(m.group(2))
                except Exception:
                    pass
                continue
            m = TRAJ_RE.search(line)
            if m:
                st = m.group(5).split(".")[-1]
                status_counts[st] += 1
                turn_counts.append(int(m.group(6)))
                parse_errs.append(int(m.group(7)))
                continue
            m = CLAW_RE.search(line)
            if m:
                clawsentry_errs[f"{m.group(1)} {m.group(2)}"] += 1
                continue
            m = RESET500_RE.search(line)
            if m:
                # bucket by minute
                reset500_per_min[m.group(1)[:16]] += 1
                continue
            m = STRUCTURED_METRIC_RE.search(line)
            if m:
                try:
                    payload = json.loads(m.group(1))
                    if isinstance(payload, dict):
                        payload["_log_line"] = line_no
                        structured_metrics.append(payload)
                except Exception:
                    pass

    return dict(
        rollout_metrics=rollout_metrics,
        train_metrics=train_metrics,
        train_points=train_points,
        perf_metrics=perf_metrics,
        clawsentry_errs=clawsentry_errs,
        status_counts=status_counts,
        turn_counts=turn_counts,
        parse_errs=parse_errs,
        reset500_per_min=reset500_per_min,
        structured_metrics=structured_metrics,
        reward_breakdown_records=reward_breakdown_records,
    )


def _structured_dedupe_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("schema"),
        record.get("phase"),
        record.get("dataset"),
        record.get("rollout_id"),
        record.get("global_step"),
        record.get("sample_count"),
        record.get("trainable_count"),
    )


def _load_structured_metrics_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records
    with path.open(errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except Exception:
                continue
            if isinstance(payload, dict):
                payload["_jsonl_line"] = line_no
                records.append(payload)
    return records


def _merge_structured_metrics(existing: list[dict[str, Any]], extra: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for record in [*existing, *extra]:
        key = _structured_dedupe_key(record)
        if key in seen:
            continue
        seen.add(key)
        merged.append(record)
    return merged


def _stats(arr: list[float], label: str) -> dict[str, float]:
    import math
    nums = [x for x in arr if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not nums:
        return {}
    nums = [float(x) for x in nums]
    n = len(nums)
    head = nums[:10] if n >= 10 else nums
    tail = nums[-10:] if n >= 10 else nums
    return {
        f"{label}_mean": sum(nums) / n,
        f"{label}_first10_mean": sum(head) / len(head),
        f"{label}_last10_mean": sum(tail) / len(tail),
        f"{label}_max": max(nums),
        f"{label}_min": min(nums),
    }


def _detect_collapse(
    r_ids: list[int], resp_len: list[float | None], threshold: float = 5.0
) -> int | None:
    """Return rollout id where mean response length first collapses below threshold."""
    for i, (rid, rl) in enumerate(zip(r_ids, resp_len)):
        if rl is not None and rl < threshold and i > 5:
            return rid
    return None


def _get_series(d: dict, ids: list[int], key: str) -> list[Any]:
    return [d[i].get(key) for i in ids]


def _get_points_series(points: list[dict[str, Any]], key: str) -> list[Any]:
    return [p.get(key) for p in points]


def _has_numeric(values: list[Any]) -> bool:
    return any(_num(value) is not None for value in values)


def _numeric_points(xs: list[int], ys: list[Any]) -> tuple[list[int], list[float]]:
    out_x: list[int] = []
    out_y: list[float] = []
    for x, y in zip(xs, ys):
        value = _num(y)
        if value is None:
            continue
        out_x.append(x)
        out_y.append(value)
    return out_x, out_y


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        import math
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _structured_train_records(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    json_records = []
    for record in parsed.get("structured_metrics") or []:
        if record.get("phase") and record.get("phase") != "train":
            continue
        dataset = str(record.get("dataset") or "").strip()
        if not dataset:
            continue
        json_records.append(record)

    table_records = []
    for record in parsed.get("reward_breakdown_records") or []:
        if record.get("phase") and record.get("phase") != "train":
            continue
        dataset = str(record.get("dataset") or "").strip()
        if not dataset:
            continue
        table_records.append(record)

    table_by_rollout: dict[int, list[dict[str, Any]]] = {}
    for record in table_records:
        try:
            rollout_id = int(record.get("rollout_id"))
        except (TypeError, ValueError):
            continue
        table_by_rollout.setdefault(rollout_id, []).append(record)

    merged: list[dict[str, Any]] = []
    json_datasets_by_rollout: dict[int, set[str]] = {}
    for record in json_records:
        try:
            rollout_id = int(record.get("rollout_id"))
        except (TypeError, ValueError):
            rollout_id = -1
        dataset = str(record.get("dataset"))
        table_names = {str(r.get("dataset")) for r in table_by_rollout.get(rollout_id, [])}
        if dataset == "security" and table_names.intersection({"agent_safetybench", "agentharm"}):
            # Old structured logs collapsed these sources into `security`.
            # The adjacent text table has the recoverable per-source split.
            continue
        merged.append(record)
        json_datasets_by_rollout.setdefault(rollout_id, set()).add(dataset)

    for record in table_records:
        try:
            rollout_id = int(record.get("rollout_id"))
        except (TypeError, ValueError):
            rollout_id = -1
        dataset = str(record.get("dataset"))
        if dataset in json_datasets_by_rollout.get(rollout_id, set()):
            continue
        merged.append(record)

    return merged


def _structured_dataset_names(records: list[dict[str, Any]], include_overall: bool = False) -> list[str]:
    names = sorted(
        {
            str(record.get("dataset"))
            for record in records
            if record.get("dataset") and (include_overall or record.get("dataset") != "mixed-all")
        }
    )
    if include_overall and "mixed-all" in names:
        names.remove("mixed-all")
        names.append("mixed-all")
    return names


def _structured_axis(record: dict[str, Any]) -> int:
    for key in ("rollout_id", "global_step"):
        value = record.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return int(record.get("_log_line") or 0)


def _structured_series(
    records: list[dict[str, Any]],
    dataset: str,
    key: str,
    *,
    break_gaps: bool = True,
) -> tuple[list[int], list[float]]:
    points: list[tuple[int, float]] = []
    for record in records:
        if record.get("dataset") != dataset:
            continue
        value = _num(record.get(key))
        if value is None:
            continue
        points.append((_structured_axis(record), value))
    points.sort(key=lambda item: item[0])
    if not break_gaps or len(points) <= 1:
        return [x for x, _ in points], [y for _, y in points]

    xs: list[int] = []
    ys: list[float] = []
    last_x: int | None = None
    for x, y in points:
        if last_x is not None and x > last_x + 1:
            xs.append(last_x + 1)
            ys.append(float("nan"))
        xs.append(x)
        ys.append(y)
        last_x = x
    return xs, ys


def _plot_structured_lines(
    ax: Any,
    records: list[dict[str, Any]],
    *,
    key: str,
    title: str,
    ylabel: str | None = None,
    datasets: list[str] | None = None,
    include_overall: bool = True,
    fallback: tuple[list[int], list[Any], str] | None = None,
) -> bool:
    plotted = False
    selected = datasets or _structured_dataset_names(records, include_overall=include_overall)
    for dataset in selected:
        xs, ys = _structured_series(records, dataset, key)
        if not ys:
            continue
        kwargs = {"label": dataset}
        if dataset == "mixed-all":
            kwargs.update({"color": "black", "lw": 2.2, "alpha": 0.9})
        ax.plot(xs, ys, ".-", **kwargs)
        plotted = True

    if not plotted and fallback is not None:
        xs, raw_ys, label = fallback
        ys = [_num(y) for y in raw_ys]
        filtered = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if filtered:
            ax.plot([x for x, _ in filtered], [y for _, y in filtered], ".-", label=label)
            plotted = True

    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlabel("rollout")
    ax.grid(alpha=0.3)
    if plotted:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no compatible structured fields", ha="center", va="center", transform=ax.transAxes)
    return plotted


def _structured_reward_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for dataset in _structured_dataset_names(records, include_overall=True):
        dataset_records = [
            record for record in records if record.get("dataset") == dataset
        ]
        item: dict[str, Any] = {
            "n_points": len(dataset_records),
            "first_rollout": min((_structured_axis(r) for r in dataset_records), default=None),
            "last_rollout": max((_structured_axis(r) for r in dataset_records), default=None),
        }
        for record in dataset_records:
            if record.get("raw_reward_scale"):
                item["raw_reward_scale"] = record.get("raw_reward_scale")
                item["raw_reward_semantics"] = record.get("raw_reward_semantics")
                item["raw_reward_min"] = record.get("raw_reward_min")
                item["raw_reward_max"] = record.get("raw_reward_max")
                break
        if "raw_reward_scale" not in item:
            item.update(_raw_reward_scale_hint(dataset))
        for key, label in (
            ("reward/raw", "raw_reward"),
            ("reward/task", "task_reward"),
            ("reward/exploration", "exploration_reward"),
            ("reward/total", "total_reward"),
            ("reward_std", "reward_std"),
            ("sample_count", "sample_count"),
            ("trainable_count", "trainable_count"),
        ):
            _, values = _structured_series(records, dataset, key)
            if values:
                item[label] = _stats(values, label)
        summary[dataset] = item
    return summary


def _train_axis(parsed: dict[str, Any]) -> tuple[list[int], list[dict[str, Any]], str]:
    """Return a stable train metric axis.

    In distributed Ray logs, the printed ``model.py - step N`` label can be
    duplicated, delayed, or non-monotonic. Plot train metrics in log order and
    keep the printed step label only as diagnostic metadata.
    """
    train_points = parsed.get("train_points") or []
    if train_points:
        diag = _train_step_diagnostics(parsed)
        if diag["step_label_axis_reliable"]:
            return [int(p["_step_label"]) for p in train_points], train_points, "train step"
        return [int(p["_log_index"]) for p in train_points], train_points, "train log index"

    train_metrics = parsed["train_metrics"]
    t_ids = sorted(train_metrics)
    return t_ids, [train_metrics[i] for i in t_ids], "train step label"


def _train_step_diagnostics(parsed: dict[str, Any]) -> dict[str, Any]:
    train_points = parsed.get("train_points") or []
    if not train_points:
        t_ids = sorted(parsed["train_metrics"])
        return {
            "n_train_logs": len(t_ids),
            "n_unique_train_step_labels": len(t_ids),
            "max_train_step_label": int(max(t_ids)) if t_ids else None,
            "duplicate_train_step_labels": 0,
            "non_monotonic_step_label_events": 0,
            "step_label_axis_reliable": True,
        }

    labels = [int(p["_step_label"]) for p in train_points]
    counts = Counter(labels)
    duplicate_total = sum(v - 1 for v in counts.values() if v > 1)
    non_monotonic = sum(
        1 for prev, cur in zip(labels, labels[1:]) if cur <= prev
    )
    jump_events = sum(
        1 for prev, cur in zip(labels, labels[1:]) if cur > prev + 1
    )
    top_duplicates = [
        {"step_label": int(step), "count": int(count)}
        for step, count in counts.most_common(10)
        if count > 1
    ]
    high_sparse = {
        "0_1999": sum(1 for s in labels if 0 <= s <= 1999),
        "2000_2499": sum(1 for s in labels if 2000 <= s <= 2499),
        "2500_2999": sum(1 for s in labels if 2500 <= s <= 2999),
        "3000_3499": sum(1 for s in labels if 3000 <= s <= 3499),
        "3500_3999": sum(1 for s in labels if 3500 <= s <= 3999),
    }
    axis_reliable = duplicate_total == 0 and non_monotonic == 0
    return {
        "n_train_logs": len(labels),
        "n_unique_train_step_labels": len(counts),
        "min_train_step_label": int(min(labels)) if labels else None,
        "max_train_step_label": int(max(labels)) if labels else None,
        "duplicate_train_step_labels": int(duplicate_total),
        "non_monotonic_step_label_events": int(non_monotonic),
        "forward_jump_step_label_events": int(jump_events),
        "top_duplicate_step_labels": top_duplicates,
        "step_label_ranges": high_sparse,
        "step_label_axis_reliable": axis_reliable,
        "plot_train_axis": "train_log_index" if not axis_reliable else "step_label",
    }


def _filter_positive(xs: list[int], ys: list[Any]) -> tuple[list[int], list[float]]:
    out_x: list[int] = []
    out_y: list[float] = []
    for x, y in zip(xs, ys):
        try:
            v = float(y)
        except (TypeError, ValueError):
            continue
        if v > 0:
            out_x.append(x)
            out_y.append(v)
    return out_x, out_y


def _select_kl_train_series(train_points: list[dict[str, Any]]) -> tuple[list[Any], str]:
    kl_loss = _get_points_series(train_points, "train/kl_loss")
    if _has_numeric(kl_loss):
        return kl_loss, "kl_loss"
    ppo_kl = _get_points_series(train_points, "train/ppo_kl")
    if _has_numeric(ppo_kl):
        return ppo_kl, "ppo_kl"
    return kl_loss, "kl_loss"


def _plot_entropy_and_kl(
    ax: Any,
    xs: list[int],
    entropy_values: list[Any],
    kl_values: list[Any],
    kl_label: str,
    *,
    xlabel: str,
) -> None:
    ent_x, ent_y = _numeric_points(xs, entropy_values)
    kl_x, kl_y = _numeric_points(xs, kl_values)

    if ent_y:
        ax.plot(ent_x, ent_y, ".-", label="entropy", color="tab:blue")
    ax.set_title("entropy monitor / KL")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("entropy (nats/token)")
    ax.grid(alpha=0.3)

    lines = list(ax.lines)
    labels = [line.get_label() for line in lines]
    if kl_y:
        ax2 = ax.twinx()
        ax2.plot(kl_x, kl_y, ".-", label=kl_label, color="tab:orange", alpha=0.8)
        ax2.axhline(0, color="tab:orange", ls=":", lw=0.8, alpha=0.6, label="_nolegend_")
        ax2.set_ylabel(kl_label)
        lines += list(ax2.lines)
        labels += [line.get_label() for line in ax2.lines]
    elif not ent_y:
        ax.text(0.5, 0.5, "no entropy/KL train metrics", ha="center", va="center", transform=ax.transAxes)

    legend_items = [
        (line, label) for line, label in zip(lines, labels) if label and not label.startswith("_")
    ]
    if legend_items:
        ax.legend(
            [line for line, _ in legend_items],
            [label for _, label in legend_items],
            fontsize=8,
        )


def _plot_all(
    parsed: dict[str, Any],
    out_dir: Path,
    collapse: int | None,
    reset500_total: int,
    clawsentry_total: int,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs_dir = out_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    rollout_metrics = parsed["rollout_metrics"]
    perf_metrics = parsed["perf_metrics"]
    status_counts = parsed["status_counts"]
    turn_counts = parsed["turn_counts"]

    r_ids = sorted(rollout_metrics)
    t_ids, train_points, train_axis_label = _train_axis(parsed)
    p_ids = sorted(perf_metrics)

    raw_rew = _get_series(rollout_metrics, r_ids, "rollout/raw_reward")
    rew = _get_series(rollout_metrics, r_ids, "rollout/rewards")
    trunc = _get_series(rollout_metrics, r_ids, "rollout/truncated")
    resp_len = _get_series(rollout_metrics, r_ids, "rollout/response_lengths")
    structured_records = _structured_train_records(parsed)
    structured_datasets = _structured_dataset_names(structured_records, include_overall=False)

    pg_loss = _get_points_series(train_points, "train/pg_loss")
    kl_loss = _get_points_series(train_points, "train/kl_loss")
    ppo_kl = _get_points_series(train_points, "train/ppo_kl")
    kl_plot_values, kl_plot_label = _select_kl_train_series(train_points)
    ent = _get_points_series(train_points, "train/entropy_loss")
    gnorm = _get_points_series(train_points, "train/grad_norm")

    rl_med = _get_series(perf_metrics, p_ids, "rollout/response_len/median") if p_ids else []
    rl_max = _get_series(perf_metrics, p_ids, "rollout/response_len/max") if p_ids else []

    def fig_save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(figs_dir / name, dpi=120)
        plt.close()

    # reward_curve
    print("[+] plotting reward_curve.png")
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax[0].plot(r_ids, raw_rew, ".-", label="raw_reward (outcome)")
    ax[0].plot(r_ids, rew, ".-", alpha=0.6, label="reward (after norm)")
    ax[0].axhline(0, color="gray", ls=":", lw=0.8)
    if collapse is not None:
        ax[0].axvline(collapse, color="red", ls="--", alpha=0.5, label=f"collapse@{collapse}")
    ax[0].set_ylabel("reward")
    ax[0].legend(loc="upper right")
    ax[0].grid(alpha=0.3)
    ax[0].set_title("Reward curve — raw_reward = 2·acc - 1 (outcome only)")
    ax[1].plot(r_ids, [t for t in trunc], ".-", label="truncated_frac")
    if collapse is not None:
        ax[1].axvline(collapse, color="red", ls="--", alpha=0.5)
    ax[1].set_xlabel("rollout")
    ax[1].set_ylabel("truncated frac")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig_save("reward_curve.png")

    # response_length
    print("[+] plotting response_length.png")
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))
    xs, ys = _filter_positive(r_ids, resp_len)
    if ys:
        ax.semilogy(xs, ys, ".-", label="mean response_length")
    if rl_med:
        xs2, ys2 = _filter_positive(p_ids, rl_med)
        if ys2:
            ax.semilogy(xs2, ys2, ".-", alpha=0.5, label="median (perf)")
    if rl_max:
        xs3, ys3 = _filter_positive(p_ids, rl_max)
        if ys3:
            ax.semilogy(xs3, ys3, ".-", alpha=0.4, label="max (perf)")
    if collapse is not None:
        ax.axvline(collapse, color="red", ls="--", alpha=0.5, label=f"collapse@{collapse}")
    ax.set_xlabel("rollout")
    ax.set_ylabel("response length (tokens, log)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    title = "Response length"
    if collapse is not None:
        title += f" — collapse @ rollout {collapse}"
    ax.set_title(title)
    fig_save("response_length.png")

    # loss_curve
    print("[+] plotting loss_curve.png")
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))
    ax.plot(t_ids, pg_loss, ".-", label="pg_loss")
    ax.plot(t_ids, kl_loss, ".-", alpha=0.7, label="kl_loss")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel(train_axis_label)
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Loss curves")
    fig_save("loss_curve.png")

    # grad_norm
    print("[+] plotting grad_norm.png")
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))
    ax.plot(t_ids, gnorm, ".-", label="grad_norm")
    ax.set_xlabel(train_axis_label)
    ax.set_ylabel("grad_norm")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("grad_norm")
    fig_save("grad_norm.png")

    # kl_entropy
    print("[+] plotting kl_entropy.png")
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))
    _plot_entropy_and_kl(
        ax,
        t_ids,
        ent,
        kl_plot_values,
        kl_plot_label,
        xlabel=train_axis_label,
    )
    fig_save("kl_entropy.png")

    # overview
    print("[+] plotting overview.png")
    fig, axes = plt.subplots(4, 3, figsize=(19, 14))
    axs = axes.flatten()

    overall_dataset = None
    if any(record.get("dataset") == "mixed-all" for record in structured_records):
        overall_dataset = "mixed-all"
    elif len(structured_datasets) == 1:
        overall_dataset = structured_datasets[0]

    plotted_components = False
    if overall_dataset is not None:
        for key, label in (
            ("reward/raw", "raw_reward"),
            ("reward/exploration", "exploration_reward"),
            ("reward/total", "total_reward"),
        ):
            xs_comp, ys_comp = _structured_series(structured_records, overall_dataset, key)
            if ys_comp:
                axs[0].plot(xs_comp, ys_comp, ".-", label=f"{label} ({overall_dataset})")
                plotted_components = True
    if not plotted_components:
        axs[0].plot(r_ids, raw_rew, ".-", label="legacy rollout/raw_reward")
        axs[0].plot(r_ids, rew, ".-", alpha=0.6, label="legacy rollout/rewards")
        plotted_components = bool(r_ids)
    axs[0].axhline(0, color="gray", ls=":")
    axs[0].set_title("overall reward components")
    axs[0].grid(alpha=0.3)
    if plotted_components:
        axs[0].legend(fontsize=8)

    _plot_structured_lines(
        axs[1],
        structured_records,
        key="reward/total",
        title="total_reward by dataset",
        ylabel="mean",
        include_overall=True,
        fallback=(r_ids, rew, "legacy rollout/rewards"),
    )
    axs[1].axhline(0, color="gray", ls=":")

    _plot_structured_lines(
        axs[2],
        structured_records,
        key="reward/raw",
        title="raw_reward by dataset",
        ylabel="mean",
        include_overall=True,
        fallback=(r_ids, raw_rew, "legacy rollout/raw_reward"),
    )
    axs[2].axhline(0, color="gray", ls=":")

    _plot_structured_lines(
        axs[3],
        structured_records,
        key="reward/exploration",
        title="exploration_reward by dataset",
        ylabel="mean",
        include_overall=True,
    )
    axs[3].axhline(0, color="gray", ls=":")

    _plot_structured_lines(
        axs[4],
        structured_records,
        key="reward_std",
        title="reward std by dataset",
        ylabel="std",
        include_overall=True,
    )

    _plot_structured_lines(
        axs[5],
        structured_records,
        key="sample_count",
        title="sample count by dataset",
        ylabel="samples",
        include_overall=True,
    )

    xs, ys = _filter_positive(r_ids, resp_len)
    if ys:
        axs[6].semilogy(xs, ys, ".-", label="legacy/global")
    if structured_records:
        for dataset in _structured_dataset_names(structured_records, include_overall=True):
            xs_resp, ys_resp = _structured_series(structured_records, dataset, "response_length")
            if ys_resp:
                kwargs = {"label": dataset, "alpha": 0.75}
                if dataset == "mixed-all":
                    kwargs.update({"color": "black", "lw": 2.0, "alpha": 0.9})
                axs[6].semilogy(xs_resp, ys_resp, ".-", **kwargs)
    axs[6].set_title("response_length by dataset (log)")
    axs[6].grid(alpha=0.3, which="both")
    if axs[6].lines:
        axs[6].legend(fontsize=8)

    axs[7].plot(r_ids, trunc, ".-", label="legacy/global")
    if structured_records:
        for dataset in _structured_dataset_names(structured_records, include_overall=True):
            xs_trunc, ys_trunc = _structured_series(structured_records, dataset, "truncated")
            if ys_trunc:
                kwargs = {"label": dataset, "alpha": 0.75}
                if dataset == "mixed-all":
                    kwargs.update({"color": "black", "lw": 2.0, "alpha": 0.9})
                axs[7].plot(xs_trunc, ys_trunc, ".-", **kwargs)
    axs[7].set_title("truncated count/fraction")
    axs[7].legend(fontsize=8)
    axs[7].grid(alpha=0.3)

    _plot_structured_lines(
        axs[8],
        structured_records,
        key="trainable_count",
        title="trainable count by dataset",
        ylabel="trainable samples",
        include_overall=True,
    )

    axs[9].plot(t_ids, pg_loss, ".-")
    axs[9].set_title("pg_loss")
    axs[9].grid(alpha=0.3)
    axs[9].set_xlabel(train_axis_label)
    axs[10].plot(t_ids, gnorm, ".-")
    axs[10].set_title("grad_norm")
    axs[10].grid(alpha=0.3)
    axs[10].set_xlabel(train_axis_label)
    _plot_entropy_and_kl(
        axs[11],
        t_ids,
        ent,
        kl_plot_values,
        kl_plot_label,
        xlabel=train_axis_label,
    )
    if collapse is not None:
        for a in axs[:8]:
            a.axvline(collapse, color="red", ls="--", alpha=0.4)
    if status_counts:
        status_text = ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items()))
        fig.text(0.01, 0.01, f"trajectory status: {status_text}", fontsize=9)
    if turn_counts:
        mean_turns = sum(turn_counts) / len(turn_counts)
        fig.text(0.01, 0.03, f"turns/trajectory: n={len(turn_counts)} mean={mean_turns:.1f} max={max(turn_counts)}", fontsize=9)
    suptitle_parts = []
    if collapse is not None:
        suptitle_parts.append(f"collapse @ rollout {collapse}")
    if reset500_total:
        suptitle_parts.append(f"/reset 500: {reset500_total}")
    if clawsentry_total:
        suptitle_parts.append(f"ClawSentry errors: {clawsentry_total}")
    if suptitle_parts:
        fig.suptitle("Run overview — " + " | ".join(suptitle_parts), fontsize=13)
    fig_save("overview.png")


def _build_summary(
    parsed: dict[str, Any], collapse: int | None, run_name: str
) -> dict[str, Any]:
    rollout_metrics = parsed["rollout_metrics"]
    train_metrics = parsed["train_metrics"]
    train_diag = _train_step_diagnostics(parsed)
    clawsentry_errs = parsed["clawsentry_errs"]
    status_counts = parsed["status_counts"]
    turn_counts = parsed["turn_counts"]
    parse_errs = parsed["parse_errs"]
    reset500_per_min = parsed["reset500_per_min"]
    structured_records = _structured_train_records(parsed)

    r_ids = sorted(rollout_metrics)
    t_ids, train_points, train_axis_label = _train_axis(parsed)

    raw_rew = _get_series(rollout_metrics, r_ids, "rollout/raw_reward")
    rew = _get_series(rollout_metrics, r_ids, "rollout/rewards")
    trunc = _get_series(rollout_metrics, r_ids, "rollout/truncated")
    resp_len = _get_series(rollout_metrics, r_ids, "rollout/response_lengths")
    pg_loss = _get_points_series(train_points, "train/pg_loss")
    kl_loss = _get_points_series(train_points, "train/kl_loss")
    ppo_kl = _get_points_series(train_points, "train/ppo_kl")
    _, kl_plot_label = _select_kl_train_series(train_points)
    ent = _get_points_series(train_points, "train/entropy_loss")
    gnorm = _get_points_series(train_points, "train/grad_norm")
    lr = _get_points_series(train_points, "train/lr-pg_0")

    trunc_nums = [t for t in trunc if isinstance(t, (int, float))]
    trunc_mean = sum(trunc_nums) / len(trunc_nums) if trunc_nums else None

    cs_total = sum(clawsentry_errs.values())
    if any("429" in k for k in clawsentry_errs):
        cs_status = "ALIVE_BUT_RATE_LIMITED"
    elif clawsentry_errs:
        cs_status = "OFFLINE"
    else:
        cs_status = "OK"

    summary = {
        "run_name": run_name,
        "n_rollouts_logged": len(r_ids),
        "max_rollout_id": int(max(r_ids)) if r_ids else None,
        "n_train_steps": len(t_ids),
        "max_train_step": int(max(t_ids)) if t_ids else None,
        "train_axis": train_axis_label,
        "max_train_step_label": train_diag["max_train_step_label"],
        "train_step_diagnostics": train_diag,
        "collapse_rollout": collapse,
        "trajectories_logged": sum(status_counts.values()),
        "status_counts": dict(status_counts),
        "raw_reward": _stats(raw_rew, "raw_rew"),
        "rewards_norm": _stats(rew, "rew"),
        "structured_reward_by_dataset": _structured_reward_summary(structured_records),
        "response_lengths": _stats(resp_len, "resp_len"),
        "truncated_frac_mean": trunc_mean,
        "train": {
            "pg_loss": _stats(pg_loss, "pg_loss"),
            "grad_norm": _stats(gnorm, "gnorm"),
            "kl_loss": _stats(kl_loss, "kl"),
            "ppo_kl": _stats(ppo_kl, "ppo_kl"),
            "kl_plot_source": kl_plot_label,
            "entropy_loss": _stats(ent, "ent"),
            "lr_first": float(lr[0]) if lr and lr[0] is not None else None,
            "lr_last": float(lr[-1]) if lr and lr[-1] is not None else None,
        },
        "clawsentry": {
            "total_errors": cs_total,
            "error_breakdown": dict(clawsentry_errs),
            "status": cs_status,
        },
        "reset500": {
            "total": sum(reset500_per_min.values()),
            "max_per_minute": max(reset500_per_min.values()) if reset500_per_min else 0,
        },
        "turn_count_stats": (
            {
                "mean": sum(turn_counts) / len(turn_counts),
                "max": max(turn_counts),
                "median": sorted(turn_counts)[len(turn_counts) // 2],
            }
            if turn_counts
            else None
        ),
        "parse_error_total": int(sum(parse_errs)) if parse_errs else 0,
    }
    return summary


def plot_run(
    run_dir: Path,
    log_file: Path | None = None,
    out_dir: Path | None = None,
    no_figs: bool = False,
) -> dict[str, Any]:
    log_file = log_file or (run_dir / "logs" / "train.log")
    out_dir = out_dir or (run_dir / "metrics" / "analysis")
    if not log_file.is_file():
        raise FileNotFoundError(f"train log not found: {log_file}")
    out_dir.mkdir(parents=True, exist_ok=True)

    parsed = _parse_log(log_file)
    jsonl_records = _load_structured_metrics_jsonl(run_dir / "logs" / "metrics.jsonl")
    if jsonl_records:
        parsed["structured_metrics"] = _merge_structured_metrics(
            parsed.get("structured_metrics") or [],
            jsonl_records,
        )
    rollout_metrics = parsed["rollout_metrics"]
    train_metrics = parsed["train_metrics"]
    train_diag = _train_step_diagnostics(parsed)

    if not rollout_metrics and not train_metrics:
        print("[!] no rollouts or train steps parsed — empty log?")
        return {}

    print(
        f"  rollouts: {len(rollout_metrics)} "
        f"(max id: {max(rollout_metrics) if rollout_metrics else 'n/a'})"
    )
    print(
        f"  train logs: {train_diag['n_train_logs']} "
        f"(unique step labels: {train_diag['n_unique_train_step_labels']}, "
        f"max label: {train_diag['max_train_step_label']})"
    )
    if not train_diag["step_label_axis_reliable"]:
        print(
            "  [!] step labels are non-monotonic/duplicated; "
            "plotting train curves by log order"
        )
    print(f"  trajectories logged: {sum(parsed['status_counts'].values())}")
    print(f"  status: {dict(parsed['status_counts'])}")
    structured_records = _structured_train_records(parsed)
    print(
        f"  structured dataset metrics: {len(structured_records)} "
        f"records ({', '.join(_structured_dataset_names(structured_records, include_overall=True)) or 'none'})"
    )
    print(f"  ClawSentry errors: {sum(parsed['clawsentry_errs'].values())}")
    print(f"  /reset 500 events:  {sum(parsed['reset500_per_min'].values())}")

    r_ids = sorted(rollout_metrics)
    resp_len = _get_series(rollout_metrics, r_ids, "rollout/response_lengths")
    collapse = _detect_collapse(r_ids, resp_len)
    print(f"  collapse rollout: {collapse}")

    summary = _build_summary(parsed, collapse, run_name=run_dir.name)
    json_path = out_dir / "summary_stats.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[+] wrote {json_path}")

    if not no_figs:
        _plot_all(
            parsed,
            out_dir=out_dir,
            collapse=collapse,
            reset500_total=sum(parsed["reset500_per_min"].values()),
            clawsentry_total=sum(parsed["clawsentry_errs"].values()),
        )

    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dir", required=True, type=Path,
                   help="Run root, e.g. runs/<run_id>")
    p.add_argument("--log-file", type=Path, default=None,
                   help="Override train log (default: <run_dir>/logs/train.log)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Override output dir (default: <run_dir>/metrics/analysis)")
    p.add_argument("--no-figs", action="store_true",
                   help="Only emit summary_stats.json, skip image generation")
    args = p.parse_args(argv)

    try:
        s = plot_run(
            run_dir=args.run_dir.resolve(),
            log_file=args.log_file.resolve() if args.log_file else None,
            out_dir=args.out_dir.resolve() if args.out_dir else None,
            no_figs=args.no_figs,
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    if not s:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
