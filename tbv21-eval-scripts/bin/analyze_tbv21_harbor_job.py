#!/usr/bin/env python3
import argparse
import collections
import csv
import hashlib
import json
import math
import shutil
import textwrap
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text())


def task_from_trial_name(name: str) -> str:
    return name.split("__", 1)[0]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def collect_trials(job: Path):
    rows = []
    for result_path in sorted(job.glob("*/result.json")):
        trial_dir = result_path.parent
        data = read_json(result_path)
        meta = (data.get("agent_result") or {}).get("metadata") or {}
        verifier = data.get("verifier_result") or {}
        rewards = verifier.get("rewards") or {}
        task = task_from_trial_name(trial_dir.name)
        trajectory = trial_dir / "agent" / "trajectory.json"
        rows.append(
            {
                "trial": trial_dir.name,
                "task": task,
                "harbor_task_name": data.get("task_name") or "",
                "reward": safe_float(rewards.get("reward")),
                "status": meta.get("status") or "UNKNOWN",
                "model_turn_count": meta.get("model_turn_count"),
                "parse_error_count": meta.get("parse_error_count"),
                "n_input_tokens": (data.get("agent_result") or {}).get("n_input_tokens"),
                "n_output_tokens": (data.get("agent_result") or {}).get("n_output_tokens"),
                "started_at": data.get("started_at") or "",
                "finished_at": data.get("finished_at") or "",
                "elapsed_sec": meta.get("elapsed_sec"),
                "exception_type": ((data.get("exception_info") or {}).get("exception_type") or ""),
                "result_path": str(result_path),
                "trajectory_path": str(trajectory) if trajectory.exists() else "",
                "trajectory_bytes": trajectory.stat().st_size if trajectory.exists() else 0,
                "trajectory_sha256": sha256_file(trajectory) if trajectory.exists() else "",
            }
        )
    return rows


def summarize_tasks(rows):
    tasks = collections.defaultdict(list)
    for row in rows:
        tasks[row["task"]].append(row)
    task_rows = []
    for task, items in sorted(tasks.items()):
        rewards = [r["reward"] for r in items]
        statuses = collections.Counter(r["status"] for r in items)
        task_rows.append(
            {
                "task": task,
                "n_trials": len(items),
                "reward_sum": sum(rewards),
                "pass_any": 1 if any(r > 0 for r in rewards) else 0,
                "pass_rate": (sum(rewards) / len(items)) if items else 0.0,
                "statuses": "; ".join(f"{k}:{v}" for k, v in sorted(statuses.items())),
                "pass_trials": ", ".join(r["trial"] for r in items if r["reward"] > 0),
                "fail_trials": ", ".join(r["trial"] for r in items if r["reward"] <= 0),
            }
        )
    return task_rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def classify_failure(row, trial_dir: Path) -> str:
    if row["reward"] > 0:
        return "pass"
    status = row["status"]
    if row.get("parse_error_count") and int(row["parse_error_count"] or 0) >= 3:
        return "tool-call-json-parse"
    verifier_stdout = trial_dir / "verifier" / "test-stdout.txt"
    text = ""
    if verifier_stdout.exists():
        text += verifier_stdout.read_text(errors="ignore")[-8000:]
    trajectory = trial_dir / "agent" / "trajectory.json"
    if trajectory.exists() and trajectory.stat().st_size < 4_000_000:
        text += trajectory.read_text(errors="ignore")[-12000:]
    low = text.lower()
    if "connection refused" in low or "failed to connect" in low:
        return "service-not-ready"
    if "syntaxerror" in low or "indentationerror" in low:
        return "syntax-error"
    if "no such file" in low or "not found" in low:
        return "missing-file-or-path"
    if "timeout" in low or status == "TRUNCATED":
        return "truncated-or-timeout"
    if "assert" in low or "failed" in low:
        return "verifier-failed"
    return status.lower()


def choose_case_trials(rows, job: Path, max_failures: int):
    pass_rows = [r for r in rows if r["reward"] > 0]
    fail_by_class = collections.defaultdict(list)
    for row in rows:
        if row["reward"] <= 0:
            fail_by_class[classify_failure(row, job / row["trial"])].append(row)
    selected = []
    selected.extend(pass_rows)
    for _, items in sorted(fail_by_class.items()):
        selected.extend(items[: max(1, math.ceil(max_failures / max(1, len(fail_by_class))))])
    dedup = []
    seen = set()
    for row in selected:
        if row["trial"] not in seen:
            dedup.append(row)
            seen.add(row["trial"])
    return dedup[: max(len(pass_rows), len(pass_rows) + max_failures)]


def copy_trajectory_cases(rows, job: Path, out: Path, max_failures: int):
    cases_dir = out / "gist_payload"
    cases_dir.mkdir(parents=True, exist_ok=True)
    selected = choose_case_trials(rows, job, max_failures)
    manifest = []
    for row in selected:
        src = Path(row["trajectory_path"])
        if not src.exists():
            continue
        dst = cases_dir / f"{row['trial']}.trajectory.json"
        shutil.copy2(src, dst)
        manifest.append(
            {
                "trial": row["trial"],
                "task": row["task"],
                "reward": row["reward"],
                "status": row["status"],
                "classification": classify_failure(row, job / row["trial"]),
                "file": dst.name,
                "sha256": sha256_file(dst),
                "bytes": dst.stat().st_size,
            }
        )
    (cases_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    with (cases_dir / "README.md").open("w") as f:
        f.write("# TBv2.1 on Qwen3-8B-base raw trajectory cases\n\n")
        f.write("这些文件是从 Harbor job 原始 `agent/trajectory.json` 直接复制出的真实轨迹样本。\n\n")
        f.write("| trial | task | reward | status | classification | file | sha256 |\n")
        f.write("|---|---:|---:|---|---|---|---|\n")
        for item in manifest:
            f.write(
                f"| {item['trial']} | {item['task']} | {item['reward']:.0f} | {item['status']} | "
                f"{item['classification']} | `{item['file']}` | `{item['sha256'][:12]}...` |\n"
            )
    return manifest


def make_figures(rows, task_rows, out: Path):
    import matplotlib.pyplot as plt
    import pandas as pd

    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    task_df = pd.DataFrame(task_rows)

    status_counts = df["status"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)
    status_counts.plot(kind="bar", ax=ax, color="#4C78A8")
    ax.set_title("Trial status distribution")
    ax.set_xlabel("status")
    ax.set_ylabel("trials")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "status_distribution.png")
    plt.close(fig)

    reward_counts = df["reward"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4), dpi=180)
    reward_counts.plot(kind="bar", ax=ax, color="#59A14F")
    ax.set_title("Reward distribution")
    ax.set_xlabel("reward")
    ax.set_ylabel("trials")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "reward_distribution.png")
    plt.close(fig)

    if not task_df.empty:
        top = task_df.sort_values(["reward_sum", "task"], ascending=[False, True]).head(25)
        fig, ax = plt.subplots(figsize=(10, max(5, len(top) * 0.28)), dpi=180)
        ax.barh(top["task"], top["reward_sum"], color="#F28E2B")
        ax.invert_yaxis()
        ax.set_title("Top task reward sums")
        ax.set_xlabel("reward sum over 3 attempts")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / "top_task_rewards.png")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=180)
    ax.scatter(df["n_input_tokens"], df["n_output_tokens"], c=df["reward"], cmap="viridis", alpha=0.75)
    ax.set_title("Token usage by trial")
    ax.set_xlabel("input tokens")
    ax.set_ylabel("output tokens")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "token_usage_scatter.png")
    plt.close(fig)


def write_markdown(out: Path, rows, task_rows, manifest, config, job_summary):
    total = len(rows)
    reward_sum = sum(r["reward"] for r in rows)
    n_tasks = len(task_rows)
    solved = [r for r in task_rows if r["pass_any"]]
    status = collections.Counter(r["status"] for r in rows)
    title = "TBv2.1 on Qwen3-8B-base aligned eval summary"
    lines = [
        f"# {title}",
        "",
        f"- Trials: {total}",
        f"- Tasks observed: {n_tasks}",
        f"- Reward sum: {reward_sum:.0f}",
        f"- Trial pass@1 proxy: {(reward_sum / total if total else 0):.4f}",
        f"- Empirical pass@3: {(len(solved) / n_tasks if n_tasks else 0):.4f}",
        f"- Solved tasks: {len(solved)}",
        f"- Job started: {job_summary.get('started_at', '')}",
        f"- Job finished: {job_summary.get('finished_at', '')}",
        "",
        "## Config",
        "",
        f"- job_name: `{config.get('job_name')}`",
        f"- n_attempts: `{config.get('n_attempts')}`",
        f"- n_concurrent_trials: `{config.get('n_concurrent_trials')}`",
        f"- agent: `{config.get('agents', [{}])[0].get('import_path')}`",
        f"- model_name: `{config.get('agents', [{}])[0].get('model_name')}`",
        "",
        "## Status",
        "",
        "| status | trials |",
        "|---|---:|",
    ]
    for key, value in sorted(status.items()):
        lines.append(f"| {key} | {value} |")
    lines += [
        "",
        "## Solved Tasks",
        "",
        "| task | reward_sum | pass_trials |",
        "|---|---:|---|",
    ]
    for row in solved:
        lines.append(f"| {row['task']} | {row['reward_sum']:.0f} | {row['pass_trials']} |")
    lines += [
        "",
        "## Raw Trajectory Gist Payload",
        "",
        "| trial | task | reward | status | classification | file |",
        "|---|---|---:|---|---|---|",
    ]
    for item in manifest:
        lines.append(
            f"| {item['trial']} | {item['task']} | {item['reward']:.0f} | {item['status']} | "
            f"{item['classification']} | `{item['file']}` |"
        )
    (out / "analysis_summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job_dir", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-failure-trajectories", type=int, default=12)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows = collect_trials(args.job_dir)
    task_rows = summarize_tasks(rows)
    for row in rows:
        row["failure_class"] = classify_failure(row, args.job_dir / row["trial"])

    write_csv(args.out / "trial_summary.csv", rows)
    write_csv(args.out / "task_summary.csv", task_rows)

    status_rows = [{"status": k, "trials": v} for k, v in sorted(collections.Counter(r["status"] for r in rows).items())]
    write_csv(args.out / "status_summary.csv", status_rows)

    failure_rows = [
        {"failure_class": k, "trials": v}
        for k, v in sorted(collections.Counter(r["failure_class"] for r in rows if r["reward"] <= 0).items())
    ]
    write_csv(args.out / "failure_class_summary.csv", failure_rows)

    manifest = copy_trajectory_cases(rows, args.job_dir, args.out, args.max_failure_trajectories)
    make_figures(rows, task_rows, args.out)

    job_summary = read_json(args.job_dir / "result.json") if (args.job_dir / "result.json").exists() else {}
    config = read_json(args.config)
    write_markdown(args.out, rows, task_rows, manifest, config, job_summary)
    print(json.dumps(
        {
            "trials": len(rows),
            "tasks": len(task_rows),
            "reward_sum": sum(r["reward"] for r in rows),
            "solved_tasks": sum(1 for r in task_rows if r["pass_any"]),
            "out": str(args.out),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
