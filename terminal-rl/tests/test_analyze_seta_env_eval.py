"""Regression guards for the SETA-env eval analyzer.

The fixtures under tests/data are derived from the audit pack published with
issue #33 (seta_qwen3_8b_base_core_audit_20260709_101409.tar.gz, sha256
889f634decddfb681c1cc8b2c52b1c5dbad005313abb218812120893093ce110). The tests
below assert that this analyzer reproduces that run's published aggregates, so
the reported 38.77% / 21.61% baseline stays reproducible from source.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TERMINAL_RL = ROOT / "terminal-rl"
DATA = Path(__file__).resolve().parent / "data"
GOLDEN_PER_SAMPLE = DATA / "seta_env_eval_20260709_per_sample.csv"
GOLDEN_SUMMARY = DATA / "seta_env_eval_20260709_summary.json"

if str(TERMINAL_RL / "scripts") not in sys.path:
    sys.path.insert(0, str(TERMINAL_RL / "scripts"))

import analyze_seta_env_eval as analyzer  # noqa: E402


@pytest.fixture(scope="module")
def golden_rows() -> list[dict[str, object]]:
    with GOLDEN_PER_SAMPLE.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def golden_summary() -> dict[str, object]:
    return json.loads(GOLDEN_SUMMARY.read_text(encoding="utf-8"))


def test_golden_fixture_covers_the_whole_dataset(golden_rows, golden_summary):
    assert len(golden_rows) == golden_summary["dataset_total"] == 1356


@pytest.mark.parametrize(
    "key",
    [
        "dataset_total",
        "result_count",
        "missing_count",
        "exact_pass_count",
        "nonzero_score_count",
    ],
)
def test_summarize_reproduces_published_counts(golden_rows, golden_summary, key):
    assert analyzer.summarize(golden_rows)[key] == golden_summary[key]


@pytest.mark.parametrize(
    "key",
    [
        "raw_score_sum_completed_rows",
        "raw_score_mean_completed_rows",
        "raw_score_mean_all_dataset_missing_as_zero",
        "exact_pass_rate_completed_rows",
        "exact_pass_rate_all_dataset_missing_as_zero",
        "nonzero_score_rate_completed_rows",
        "nonzero_score_rate_all_dataset_missing_as_zero",
    ],
)
def test_summarize_reproduces_published_rates(golden_rows, golden_summary, key):
    # Relative tolerance, not equality: the published numbers were produced by a
    # float summation in a different row order, which differs in the last ulp.
    assert analyzer.summarize(golden_rows)[key] == pytest.approx(golden_summary[key], rel=1e-12)


def test_summarize_reproduces_the_headline_baseline(golden_rows):
    """The two numbers issue #33 reports in its title."""
    summary = analyzer.summarize(golden_rows)
    assert round(summary["raw_score_mean_all_dataset_missing_as_zero"], 4) == 0.3877
    assert round(summary["exact_pass_rate_all_dataset_missing_as_zero"], 4) == 0.2161


def test_summarize_reproduces_published_distributions(golden_rows, golden_summary):
    summary = analyzer.summarize(golden_rows)
    assert summary["status_counts"] == golden_summary["status_counts"]
    assert summary["raw_score_distribution"] == golden_summary["raw_score_distribution"]


def test_exact_pass_is_stricter_than_nonzero(golden_summary):
    """Partial verifier credit must not be counted as solving the task."""
    assert golden_summary["exact_pass_count"] < golden_summary["nonzero_score_count"]


def test_dataset_sample_index_is_the_line_number():
    dataset_path = TERMINAL_RL / "dataset" / "seta_env_convert" / "train.filtered.jsonl"
    if not dataset_path.is_file():
        pytest.skip(f"{dataset_path} is not checked out")
    samples = analyzer.read_dataset(dataset_path)
    assert [s.sample_index for s in samples] == list(range(len(samples)))
    assert samples[0].task_path == f"seta_env/{samples[0].task_name}"


def test_failure_event_regex_matches_the_real_log_format():
    line = (
        "(RolloutManager pid=283525) [2026-07-09 00:14:24] generate.py:4059 - "
        "[task=1080 uid=1b9bfb5a group_idx=-1 sample_idx=17] Generate failed "
        "(HTTPStatusError): Server error '500 Internal Server Error' for url "
        "'http://env-server:18080/reset'"
    )
    match = analyzer.FAILURE_EVENT_RE.search(line)
    assert match is not None
    assert match["task_name"] == "1080"
    assert match["uid"] == "1b9bfb5a"
    assert match["run_sample_index"] == "17"
    assert match["error_type"] == "HTTPStatusError"


def test_failure_events_deduplicate_retries_of_one_rollout(tmp_path):
    """One failing rollout logs a line per retry; the count is per rollout."""
    template = (
        "[task=806 uid=a3954d9a group_idx=-1 sample_idx={idx}] "
        "Generate failed (HTTPStatusError): boom\n"
    )
    log = tmp_path / "train.log"
    log.write_text(template.format(idx=27) * 3 + template.format(idx=28), encoding="utf-8")
    events = analyzer.read_failure_events(log, "main")
    assert [e.run_sample_index for e in events] == [27, 28]


def test_derive_turn_metrics_sums_over_turns():
    trajectory = {
        "turns": [
            {"tool_calls": [{}, {}], "n_input_tokens": 10, "n_output_tokens": 4,
             "parse_error_recorded": False},
            {"tool_calls": [{}], "n_input_tokens": 20, "n_output_tokens": 6,
             "parse_error_recorded": True},
        ]
    }
    assert analyzer.derive_turn_metrics(trajectory) == {
        "num_turns": 2.0,
        "tool_calls": 3,
        "parse_error_turns": 1,
        "input_tokens": 30,
        "output_tokens": 10,
    }


def _sample(index: int) -> analyzer.DatasetSample:
    return analyzer.DatasetSample(
        sample_index=index, task_name=str(index), task_path=f"seta_env/{index}",
        data_source="terminal_bench",
    )


def _index_row(index: int, run_label: str, run_order: int, score: float) -> analyzer.IndexRow:
    return analyzer.IndexRow(
        sample_index=index, sample_index_source="index.sample_index", run_label=run_label,
        run_order=run_order, run_sample_index=index, task_name=str(index),
        task_path=f"seta_env/{index}", uid=f"uid{index}{run_label}", status="COMPLETED",
        raw_score=score, raw_reward=score, task_reward=score, total_reward=score,
        num_turns=1.0, tool_calls=1, parse_error_turns=0, input_tokens=1, output_tokens=1,
        eval_error="", traj_path="",
    )


def test_later_runs_win_so_retries_replace_the_original_attempt():
    merged = analyzer.merge(
        [_sample(0)],
        [_index_row(0, "main", 0, 0.0), _index_row(0, "supp1", 1, 1.0)],
    )
    assert [(r["run_label"], r["raw_score"], r["exact_pass"]) for r in merged] == [("supp1", 1.0, 1)]


def test_samples_with_no_trajectory_are_missing_and_score_zero():
    merged = analyzer.merge([_sample(0), _sample(1)], [_index_row(0, "main", 0, 1.0)])
    missing = [row for row in merged if row["has_result"] == 0]
    assert [row["status"] for row in missing] == [analyzer.MISSING_STATUS]

    summary = analyzer.summarize(merged)
    assert summary["dataset_total"] == 2
    assert summary["missing_count"] == 1
    # Conservative denominator: the missing sample drags the mean to 0.5, and is
    # not silently dropped from the report.
    assert summary["raw_score_mean_all_dataset_missing_as_zero"] == 0.5
    assert summary["raw_score_mean_completed_rows"] == 1.0


def _write_dataset(path: Path, count: int) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "task": [{"role": "user", "content": f"task {i}"}],
                    "metadata": {
                        "task_name": str(i),
                        "task_path": f"seta_env/{i}",
                        "instruction": f"task {i}",
                        "data_source": "terminal_bench",
                    },
                }
            )
            + "\n"
            for i in range(count)
        ),
        encoding="utf-8",
    )


def test_supplement_jsonl_keeps_only_missing_rows_and_records_their_dataset_index(tmp_path):
    dataset = tmp_path / "train.jsonl"
    _write_dataset(dataset, 4)
    per_sample = analyzer.merge(
        [_sample(i) for i in range(4)],
        [_index_row(0, "main", 0, 1.0), _index_row(2, "main", 0, 0.0)],
    )

    out = tmp_path / "supp.jsonl"
    assert analyzer.write_supplement_jsonl(dataset, per_sample, out) == 2

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["metadata"]["supplement_sample_index"] for r in records] == [1, 3]
    assert [r["metadata"]["task_name"] for r in records] == ["1", "3"]


def test_supplement_index_is_what_maps_a_retry_back_to_its_dataset_row(tmp_path):
    """Closes the loop: what write_supplement_jsonl emits is what read_run reads."""
    run_dir = tmp_path / "supp1" / "trajectories" / "seta_task-9_uid"
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                # Local index inside the filtered supplement, not the dataset index.
                "sample_index": 0,
                "task_name": "9",
                "task_path": "seta_env/9",
                "uid": "abc",
                "status": "Status.COMPLETED",
                "raw_score": 1.0,
                "sample_metadata": {"supplement_sample_index": 1192},
            }
        ),
        encoding="utf-8",
    )
    (rows,) = list(analyzer.read_run(tmp_path / "supp1", "supp1", 1))
    assert rows.sample_index == 1192
    assert rows.run_sample_index == 0
    assert rows.sample_index_source == "sample_metadata.supplement_sample_index"


def test_status_enum_prefix_is_stripped(tmp_path):
    run_dir = tmp_path / "main" / "trajectories" / "t0"
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(
        json.dumps({"sample_index": 0, "status": "Status.TRUNCATED", "raw_score": 0.5}),
        encoding="utf-8",
    )
    (row,) = list(analyzer.read_run(tmp_path / "main", "main", 0))
    assert row.status == "TRUNCATED"


def test_analyzer_cli_writes_the_expected_output_files(tmp_path):
    dataset = tmp_path / "train.jsonl"
    _write_dataset(dataset, 2)
    run_dir = tmp_path / "main" / "trajectories" / "t0"
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(
        json.dumps({"sample_index": 0, "task_name": "0", "task_path": "seta_env/0",
                    "status": "Status.COMPLETED", "raw_score": 1.0}),
        encoding="utf-8",
    )
    out = tmp_path / "analysis"
    assert analyzer.main(
        ["--dataset", str(dataset), "--run", f"main={tmp_path / 'main'}", "--out", str(out)]
    ) == 0
    assert {p.name for p in out.iterdir()} == {
        "summary.json", "per_sample.csv", "task_summary.csv",
        "status_counts.csv", "failure_events.csv",
    }
    assert json.loads((out / "summary.json").read_text())["missing_count"] == 1


def test_driver_script_parses_and_is_executable():
    script = TERMINAL_RL / "scripts" / "run_seta_env_eval.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    assert script.stat().st_mode & 0o111, "run_seta_env_eval.sh is not executable"


def test_driver_script_disables_checkpoint_writing_by_default():
    """Nothing is trained, and the default checkpoint dir may not be writable."""
    script = (TERMINAL_RL / "scripts" / "run_seta_env_eval.sh").read_text()
    assert 'MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-0}"' in script
    assert "eval_only.py" in script
