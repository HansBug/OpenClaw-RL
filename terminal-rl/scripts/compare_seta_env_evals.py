#!/usr/bin/env python3
"""Compare SETA-env eval runs, with intervals rather than bare point estimates.

Filling a benchmark table one checkpoint at a time invites reading 21.6% and
23.4% as an improvement. Exact-pass is a binomial count over the dataset, so the
question of whether two checkpoints differ has an answer, and it is usually "not
at this sample size". This prints Wilson 95% intervals next to every rate and
says plainly when a pair's intervals overlap.

    python terminal-rl/scripts/compare_seta_env_evals.py \\
        baseline=runs/<run-a>/final_analysis/summary.json \\
        rl-iter499=runs/<run-b>/final_analysis/summary.json

Overlapping Wilson intervals are not a formal test of no difference, only a cheap
signal that the gap is within sampling noise. Use a two-proportion test when a
claim depends on it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

Z_95 = 1.959963984540054


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval; reproduces the intervals published in issues #27-#29."""
    if trials <= 0:
        return (0.0, 0.0)
    proportion = successes / trials
    denominator = 1 + z * z / trials
    centre = proportion + z * z / (2 * trials)
    margin = z * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials))
    return (centre - margin) / denominator, (centre + margin) / denominator


@dataclass(frozen=True)
class Run:
    label: str
    total: int
    exact_pass: int
    nonzero: int
    raw_score_mean: float
    missing: int

    @property
    def exact_pass_rate(self) -> float:
        return self.exact_pass / self.total if self.total else 0.0

    @property
    def exact_pass_interval(self) -> tuple[float, float]:
        return wilson_interval(self.exact_pass, self.total)


def load_run(label: str, summary_path: Path) -> Run:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return Run(
        label=label,
        total=int(summary["dataset_total"]),
        exact_pass=int(summary["exact_pass_count"]),
        nonzero=int(summary["nonzero_score_count"]),
        # The conservative denominator, matching what the docs report.
        raw_score_mean=float(summary["raw_score_mean_all_dataset_missing_as_zero"]),
        missing=int(summary["missing_count"]),
    )


def overlapping_pairs(runs: Sequence[Run]) -> list[tuple[Run, Run]]:
    pairs = []
    for i, left in enumerate(runs):
        for right in runs[i + 1:]:
            low_l, high_l = left.exact_pass_interval
            low_r, high_r = right.exact_pass_interval
            if low_l <= high_r and low_r <= high_l:
                pairs.append((left, right))
    return pairs


def format_comparison(runs: Sequence[Run]) -> str:
    width = max(len(run.label) for run in runs)
    lines = [
        f"{'run'.ljust(width)}  {'n':>5}  {'miss':>4}  {'raw_score':>9}  "
        f"{'exact_pass':>10}  {'rate':>7}  {'Wilson 95%':>17}",
    ]
    for run in runs:
        low, high = run.exact_pass_interval
        lines.append(
            f"{run.label.ljust(width)}  {run.total:5d}  {run.missing:4d}  "
            f"{run.raw_score_mean * 100:8.2f}%  {run.exact_pass:10d}  "
            f"{run.exact_pass_rate * 100:6.2f}%  {low * 100:6.2f}% - {high * 100:6.2f}%"
        )

    if len(runs) < 2:
        return "\n".join(lines)

    lines.append("")
    overlaps = overlapping_pairs(runs)
    if overlaps:
        lines.append("exact-pass intervals overlap, so these pairs are not separable here:")
        for left, right in overlaps:
            delta = (right.exact_pass_rate - left.exact_pass_rate) * 100
            lines.append(f"  {left.label} vs {right.label}   delta {delta:+.2f} pp")
    else:
        lines.append("no exact-pass intervals overlap; every pair is separable at this sample size.")
    lines.append(
        "Note: raw_score is average partial credit, not a solve rate; exact_pass is the solve rate."
    )
    return "\n".join(lines)


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"expected label=path/to/summary.json, got {value!r}")
    label, _, path = value.partition("=")
    if not label:
        raise argparse.ArgumentTypeError(f"label must be non-empty, got {value!r}")
    return label, Path(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "runs", nargs="+", type=_parse_run, metavar="LABEL=SUMMARY_JSON",
        help="summary.json produced by analyze_seta_env_eval.py",
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="emit JSON")
    args = parser.parse_args(argv)

    runs = [load_run(label, path) for label, path in args.runs]
    if args.as_json:
        print(json.dumps(
            {
                "runs": [
                    {
                        "label": run.label,
                        "dataset_total": run.total,
                        "missing_count": run.missing,
                        "raw_score_mean": run.raw_score_mean,
                        "exact_pass_count": run.exact_pass,
                        "exact_pass_rate": run.exact_pass_rate,
                        "exact_pass_wilson95": list(run.exact_pass_interval),
                    }
                    for run in runs
                ],
                "overlapping_pairs": [[a.label, b.label] for a, b in overlapping_pairs(runs)],
            },
            indent=2,
        ))
    else:
        print(format_comparison(runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
