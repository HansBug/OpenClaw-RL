#!/usr/bin/env python3
import collections
import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: summarize_harbor_job.py JOB_DIR", file=sys.stderr)
        return 2
    job = Path(sys.argv[1])
    root = job / "result.json"
    if not root.exists():
        print(f"[ERROR] missing {root}")
        return 1
    summary = json.loads(root.read_text())
    results = sorted(p for p in job.glob("*/result.json"))
    rewards = []
    statuses = collections.Counter()
    exceptions = collections.Counter()
    solved = collections.defaultdict(float)
    trials_by_task = collections.Counter()
    for p in results:
        data = json.loads(p.read_text())
        task = data.get("task_name") or p.parent.name.split("__", 1)[0]
        trials_by_task[task] += 1
        reward = (((data.get("verifier_result") or {}).get("rewards") or {}).get("reward") or 0.0)
        rewards.append(float(reward))
        solved[task] = max(solved[task], float(reward))
        meta = (data.get("agent_result") or {}).get("metadata") or {}
        statuses[meta.get("status") or "UNKNOWN"] += 1
        exc = data.get("exception_info")
        if exc:
            exceptions[exc.get("exception_type") or "unknown"] += 1
    total = len(results)
    reward_sum = sum(rewards)
    solved_tasks = sorted(k for k, v in solved.items() if v > 0)
    print("job=", job)
    print("started_at=", summary.get("started_at"))
    print("finished_at=", summary.get("finished_at"))
    print("n_total_trials=", summary.get("n_total_trials"), "result_files=", total)
    print("reward_sum=", reward_sum)
    print("pass_at_1_trial=", (reward_sum / total if total else 0.0))
    print("pass_at_3_empirical=", (len(solved_tasks) / len(trials_by_task) if trials_by_task else 0.0))
    print("n_solved_tasks=", len(solved_tasks), "n_tasks=", len(trials_by_task))
    print("solved_tasks=", ",".join(solved_tasks))
    print("status_counts=", dict(statuses))
    print("exception_counts=", dict(exceptions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
